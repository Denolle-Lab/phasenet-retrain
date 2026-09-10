"""
manifest_dataset.py

PyTorch Dataset that loads waveforms on-the-fly from a manifest CSV
produced by build_training_dataset.py.

Each manifest row identifies a trace by (dataset_name, trace_name, chunk).
Waveforms are fetched from the local SeisBench cache — no data is copied.

Chunked datasets (MLAAPDE, CWA) are accessed by opening their per-chunk
HDF5 files directly with h5py.  All other datasets use SeisBench's standard
get_sample() interface at the stored sampling rate.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

from waveform_contract import (CONTRACT_VERSION, METADATA_FIELDS, TraceRecord, canonical_waveform,
                               has_rate, metadata_rate, present, read_hdf5_trace, resample_waveform, text_value, valid_rate)

# ──────────────────────────────────────────────────────────────────────────────
# Chunked HDF5 reader (for MLAAPDE and CWA)
# ──────────────────────────────────────────────────────────────────────────────

class SingleHDF5Reader:
    """Lazy, process-local HDF5 access with a thin per-manifest metadata lookup."""

    def __init__(self, ds_path, hdf5_name="waveforms.hdf5", metadata_name="metadata.csv", wanted=None):
        self.path = Path(ds_path) / hdf5_name
        self.metadata_path = Path(ds_path) / metadata_name
        self.wanted = set(wanted) if wanted is not None else None
        self._h5 = None
        self._pid = None
        self._metadata = None

    def _handle(self):
        if self._pid != os.getpid() or self._h5 is None:
            self.close()
            self._h5 = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._h5

    def _trace_metadata(self, trace_name):
        if self._metadata is None:
            self._metadata = {}
            if self.metadata_path.exists():
                for frame in pd.read_csv(self.metadata_path, chunksize=100_000,
                                         usecols=lambda col: col in METADATA_FIELDS,
                                         dtype={"trace_name": str}):
                    if "trace_name" not in frame:
                        raise ValueError("Source metadata has no trace_name column")
                    if self.wanted is not None:
                        frame = frame[frame.trace_name.isin(self.wanted)]
                    for row in frame.to_dict("records"):
                        name = row["trace_name"]
                        if name in self._metadata:
                            raise ValueError(f"Duplicate trace_name in source metadata: {name}")
                        self._metadata[name] = row
        return self._metadata.get(trace_name, {})

    def get_record(self, trace_name, manifest_row=None):
        metadata = dict(self._trace_metadata(trace_name))
        if manifest_row is not None:
            for key in METADATA_FIELDS - {"trace_name"}:
                value = manifest_row.get(key)
                if not present(value):
                    continue
                if present(metadata.get(key)) and str(metadata[key]) != str(value):
                    # CSV type inference may make the same rate/npts int or float.
                    if key in {"trace_sampling_rate_hz", "sampling_rate", "trace_dt_s", "trace_npts"}:
                        if not np.isclose(float(metadata[key]), float(value), rtol=1e-10):
                            raise ValueError(f"Manifest/source conflict for {key}")
                    else:
                        raise ValueError(f"Manifest/source conflict for {key}")
                metadata[key] = value
        return read_hdf5_trace(self._handle(), trace_name, metadata)

    def get_waveform(self, trace_name):
        return self.get_record(trace_name).waveform

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"], state["_pid"] = None, None
        return state


class ChunkedHDF5Reader:
    """Preserve chunk identity, including leading zeros in chunk tags."""

    def __init__(self, ds_path, hdf5_prefix="waveforms_", wanted=None):
        self.ds_path = Path(ds_path)
        self.hdf5_prefix = hdf5_prefix
        self.wanted = wanted or {}
        self._readers = {}

    def _reader(self, chunk_tag):
        chunk_tag = str(chunk_tag)
        if chunk_tag not in self._readers:
            suffix = self.hdf5_prefix.removeprefix("waveforms") + chunk_tag
            self._readers[chunk_tag] = SingleHDF5Reader(
                self.ds_path, f"waveforms{suffix}.hdf5", f"metadata{suffix}.csv",
                wanted=self.wanted.get(chunk_tag))
        return self._readers[chunk_tag]

    def get_record(self, chunk_tag, trace_name, manifest_row=None):
        return self._reader(chunk_tag).get_record(trace_name, manifest_row)

    def get_waveform(self, chunk_tag, trace_name):
        return self.get_record(chunk_tag, trace_name).waveform

    def close(self):
        for reader in self._readers.values():
            reader.close()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry (mirrors build_training_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

_CHUNKED_DS = {
    "mlaapde":  (Path(SEISBENCH_CACHE) / "datasets" / "mlaapde",  "waveforms_"),
    "cwa":      (Path(SEISBENCH_CACHE) / "datasets" / "cwa",      "waveforms_"),
    "aq2009gm": (Path(SEISBENCH_CACHE) / "datasets" / "aq2009gm", "waveforms"),
    "obs":      (Path(SEISBENCH_CACHE) / "datasets" / "obs",      "waveforms"),
}

# Large datasets accessed directly via HDF5 to avoid loading full metadata.
_SINGLE_HDF5_DS = {
    "pisdl":        Path(SEISBENCH_CACHE) / "datasets" / "pisdl",
    "meier2019jgr": Path(SEISBENCH_CACHE) / "datasets" / "meier2019jgr",
    "ross2018gpd":  Path(SEISBENCH_CACHE) / "datasets" / "ross2018gpd",
}

_NOISE_GLOBAL_PATH   = Path(__file__).parent.parent / "data" / "noise_global"
_NOISE_PREPHASE_PATH = Path(__file__).parent.parent / "data" / "noise_prephase"

_SBD_CLASSES = {
    "stead":          sbd.STEAD,
    "ceed":           sbd.CEED,
    "geofon":         sbd.GEOFON,
    "instancecounts": sbd.InstanceCounts,
    "ethz":           sbd.ETHZ,
    "crew":           sbd.CREW,
    "iquique":        sbd.Iquique,
    "txed":           sbd.TXED,
    "pnw":            sbd.PNW,
    "lendb":          sbd.LenDB,
    "vcseis":         sbd.VCSEIS,
    "obst2024":       sbd.OBST2024,
    "scedc":          sbd.SCEDC,
    "noise_global":   None,  # loaded via NoiseGlobalReader
    "noise_prephase": None,  # loaded via NoisePrephaseReader
}


# ──────────────────────────────────────────────────────────────────────────────
# Noise-global HDF5 reader
# ──────────────────────────────────────────────────────────────────────────────

class NoiseGlobalReader(SingleHDF5Reader):
    def __init__(self, wanted=None):
        super().__init__(_NOISE_GLOBAL_PATH, wanted=wanted)


class NoisePrephaseReader(SingleHDF5Reader):
    def __init__(self, wanted=None):
        super().__init__(_NOISE_PREPHASE_PATH, wanted=wanted)

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SR   = 100          # Hz — resample to this rate
WINDOW_LEN  = 3001         # samples @ 100 Hz = 30 s
LABEL_SIGMA = 10           # samples — Gaussian label width


def _normalise_std(waveform):
    """Per-component demean + unit-std normalisation (norm=std, matches jma_wc training)."""
    waveform = waveform - waveform.mean(axis=-1, keepdims=True)
    std = waveform.std(axis=-1, keepdims=True)
    std[std < 1e-6] = 1.0
    waveform = waveform / std
    return np.clip(waveform, -10.0, 10.0)


def _resample_if_needed(waveform, src_sr, tgt_sr=TARGET_SR):
    return resample_waveform(waveform, src_sr, tgt_sr)


def _ensure_3ch_cw(waveform, components, dimensions):
    return canonical_waveform(waveform, components, dimensions)[0]


def _window(waveform, p_sample, window_len=WINDOW_LEN):
    """
    Cut a window anchored on a P arrival, or S when P is absent.
    Returns (windowed_waveform, p_offset, offset_applied).
    """
    n = waveform.shape[-1]
    # place P pick at ~30 % into the window
    p_int = int(round(p_sample))
    start = max(0, p_int - int(0.30 * window_len))
    start = min(start, max(0, n - window_len))
    end   = start + window_len
    if end > n:
        # zero-pad right side
        chunk = waveform[..., start:n]
        pad   = np.zeros((*waveform.shape[:-1], window_len - chunk.shape[-1]),
                         dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=-1)
    else:
        chunk = waveform[..., start:end]
    p_offset = float(p_sample) - start
    return chunk, p_offset, start


def _gaussian_label(size, centre, sigma=LABEL_SIGMA):
    x = np.arange(size, dtype=np.float32)
    if centre is None or np.isnan(centre):
        return np.zeros(size, dtype=np.float32)
    g = np.exp(-((x - centre) ** 2) / (2 * sigma ** 2))
    return g


def make_labels(p_offset, s_offset, window_len=WINDOW_LEN):
    """
    Build (3, window_len) label tensor in PSN order; label policy changes belong to #41.
    s_offset=None/NaN produces a zero S channel.
    """
    p_lbl = _gaussian_label(window_len, p_offset)
    s_lbl = _gaussian_label(window_len, s_offset)
    noise = np.clip(1.0 - np.maximum(p_lbl, s_lbl), 0.0, 1.0)
    return np.stack([p_lbl, s_lbl, noise]).astype(np.float32)  # PSN — matches jma_wc label convention


# ──────────────────────────────────────────────────────────────────────────────
# ManifestDataset
# ──────────────────────────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """Load verified waveform coordinates, transport picks, and stop on bad rows.

    Arrival indices are relative to the source trace start at the source metadata
    rate unless arrival_sampling_rate_hz is explicitly provided in the manifest.
    Unknown rates/components and missing non-noise labels are rejected. Failure
    records are appended to a per-process JSONL beside the manifest by default.
    """

    def __init__(self, manifest_csv, augment=False, window_len=WINDOW_LEN, rejection_log=None):
        self.manifest_path = Path(manifest_csv)
        self.manifest_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.manifest = pd.read_csv(manifest_csv, low_memory=False,
                                    dtype={"chunk": str, "trace_name": str})
        if "chunk" not in self.manifest:
            self.manifest["chunk"] = ""
        self.manifest["chunk"] = self.manifest["chunk"].fillna("")
        self.augment, self.window_len = augment, int(window_len)
        if self.window_len <= 0:
            raise ValueError("window_len must be positive")
        self.rejection_log = Path(rejection_log) if rejection_log else self.manifest_path.with_suffix(".rejected.jsonl")
        self._sbd_datasets, self._sbd_name_to_idx = {}, {}
        self._sbd_unique_names = {}
        self._chunked, self._single_hdf5 = {}, {}
        self._noise_reader, self._prephase_reader = None, None
        try:
            for ds_name, rows in self.manifest.groupby("dataset_name", sort=False):
                wanted = set(rows.trace_name)
                if ds_name in _CHUNKED_DS:
                    path, prefix = _CHUNKED_DS[ds_name]
                    by_chunk = {str(chunk): set(group.trace_name) for chunk, group in rows.groupby("chunk")}
                    self._chunked[ds_name] = ChunkedHDF5Reader(path, prefix, by_chunk)
                elif ds_name in _SINGLE_HDF5_DS:
                    self._single_hdf5[ds_name] = SingleHDF5Reader(_SINGLE_HDF5_DS[ds_name], wanted=wanted)
                elif ds_name == "noise_global":
                    self._noise_reader = NoiseGlobalReader(wanted)
                elif ds_name == "noise_prephase":
                    self._prephase_reader = NoisePrephaseReader(wanted)
                elif ds_name in _SBD_CLASSES:
                    ds = _SBD_CLASSES[ds_name](sampling_rate=None, component_order="ZNE",
                                               dimension_order="NCW", missing_components="pad")
                    self._sbd_datasets[ds_name] = ds
                    index = self._build_name_index(ds)
                    self._sbd_name_to_idx[ds_name] = index
                    unique = {}
                    for (_, name), position in index.items():
                        unique[name] = None if name in unique else position
                    self._sbd_unique_names[ds_name] = unique
                else:
                    raise ValueError(f"Unknown dataset {ds_name!r}")
        except Exception as exc:
            self._reject(None, {"dataset_name": ds_name}, exc)
            self.close()
            raise

    @staticmethod
    def _build_name_index(ds):
        index = {}
        chunks = ds.metadata.get("trace_chunk", ds.metadata.get("chunk", [""] * len(ds.metadata)))
        for position, (chunk, name) in enumerate(zip(chunks, ds.metadata["trace_name"])):
            key = (str(chunk) if present(chunk) else "", str(name))
            if key in index:
                raise ValueError(f"Ambiguous source identity: {key}")
            index[key] = position
        return index

    def _fetch_sbd(self, ds_name, trace_name, chunk, manifest_row):
        ds = self._sbd_datasets[ds_name]
        index = self._sbd_name_to_idx[ds_name]
        idx = index.get((chunk, trace_name))
        if idx is None and not chunk:
            idx = self._sbd_unique_names[ds_name].get(trace_name)
        if idx is None:
            raise KeyError(f"Missing or ambiguous source identity: {(ds_name, chunk, trace_name)}")
        source = ds.metadata.iloc[idx].to_dict()
        source_rate = metadata_rate(source)
        # Force the original stored rate, even if the dataset was configured to
        # resample. get_sample returns metadata at the rate actually requested.
        wf, effective = ds.get_sample(idx, sampling_rate=source_rate)
        rate = metadata_rate(effective)
        if not np.isclose(rate, source_rate, rtol=1e-10):
            raise ValueError("Reader did not return the requested stored sampling rate")
        if has_rate(manifest_row):
            if not np.isclose(metadata_rate(manifest_row), source_rate, rtol=1e-10, atol=0):
                raise ValueError("Manifest/source sampling-rate conflict")
        # SeisBench already returns ZNE/NCW as explicitly configured above.
        if ds.component_order != "ZNE" or ds.dimension_order != "NCW":
            raise ValueError("Reader component/dimension contract changed")
        source_components = source.get("trace_component_order")
        if not present(source_components):
            source_components = ds.data_format.get("component_order")
        source_components = text_value(source_components) if present(source_components) else ""
        if not source_components or len(set(source_components)) != len(source_components) or set(source_components) - set("ZNE"):
            raise ValueError("Source channel orientation is unknown")
        wave, _ = canonical_waveform(wf, "ZNE", "CW", source.get("trace_npts"))
        return TraceRecord(wave, rate, source_rate, [c in source_components for c in "ZNE"],
                           source.get("trace_start_time"))

    def _fetch(self, row):
        name, trace, chunk = row["dataset_name"], row["trace_name"], row["chunk"]
        if name in self._chunked:
            return self._chunked[name].get_record(chunk, trace, row)
        if name in self._single_hdf5:
            return self._single_hdf5[name].get_record(trace, row)
        if name == "noise_global":
            return self._noise_reader.get_record(trace, row)
        if name == "noise_prephase":
            return self._prephase_reader.get_record(trace, row)
        return self._fetch_sbd(name, trace, chunk, row)

    def _reject(self, idx, row, exc):
        record = dict(time_utc=datetime.now(timezone.utc).isoformat(), contract=CONTRACT_VERSION,
                      manifest_sha256=self.manifest_hash, row_index=idx,
                      dataset_name=row.get("dataset_name"), chunk=row.get("chunk"),
                      trace_name=row.get("trace_name"), error_type=type(exc).__name__, reason=str(exc))
        path = self.rejection_log.with_name(f"{self.rejection_log.stem}.{os.getpid()}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        logger.error("Rejected manifest row %s: %s", idx, exc)

    def get_sample_with_metadata(self, idx):
        row = self.manifest.iloc[idx].to_dict()
        try:
            record = self._fetch(row)
            index_rate = valid_rate(row.get("arrival_sampling_rate_hz"), "arrival sampling rate") \
                if present(row.get("arrival_sampling_rate_hz")) else record.arrival_sampling_rate
            index_rate = valid_rate(index_rate, "arrival sampling rate")
            offsets, rejected_picks = {}, {}
            for phase, column in (("P", "p_arrival_sample"), ("S", "s_arrival_sample")):
                value = row.get(column)
                if not present(value):
                    offsets[phase] = None
                    continue
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"Non-finite {phase} arrival")
                time = value / index_rate
                if time < 0 or time > (record.waveform.shape[-1] - 1) / record.sampling_rate + 1e-9:
                    offsets[phase] = None
                    rejected_picks[phase] = "outside source support"
                else:
                    offsets[phase] = time * TARGET_SR
            is_noise = row["dataset_name"] in {"noise_global", "noise_prephase"}
            if is_noise and any(present(row.get(column)) for column in ("p_arrival_sample", "s_arrival_sample")):
                raise ValueError("Noise row contains arrival labels")
            if not is_noise and all(value is None for value in offsets.values()):
                raise ValueError("Signal row has no valid P or S arrival; not a noise label")
            wf = _resample_if_needed(record.waveform, record.sampling_rate, TARGET_SR)
            if is_noise:
                start = int(np.random.randint(0, max(0, wf.shape[-1] - self.window_len) + 1))
                wf, _, start = _window(wf, start + int(0.30 * self.window_len), self.window_len)
            else:
                anchor = offsets["P"] if offsets["P"] is not None else offsets["S"]
                wf, _, start = _window(wf, anchor, self.window_len)
            valid_samples = min(self.window_len, max(0, int(np.floor(
                (record.waveform.shape[-1] - 1) * TARGET_SR / record.sampling_rate + 1e-9)) + 1 - start))
            for phase, value in offsets.items():
                if value is not None:
                    offset = value - start
                    offsets[phase] = offset if 0 <= offset <= valid_samples - 1 + 1e-9 else None
                    if offsets[phase] is None:
                        rejected_picks[phase] = "outside crop support"
            if not is_noise and all(value is None for value in offsets.values()):
                raise ValueError("Signal row has no arrival inside resampled crop support")
            wf = _normalise_std(wf)
            if self.augment:
                wf *= np.random.uniform(0.5, 2.0)
                if np.random.random() < 0.1:
                    wf = -wf
            labels = make_labels(offsets["P"], offsets["S"], self.window_len)
            info = dict(contract=CONTRACT_VERSION, source_rate_hz=record.sampling_rate,
                        effective_reader_rate_hz=record.sampling_rate,
                        arrival_rate_hz=index_rate, target_rate_hz=TARGET_SR,
                        source_start_time=record.start_time, crop_start_sample=start,
                        crop_start_offset_s=start / TARGET_SR, valid_samples=valid_samples,
                        component_mask=record.component_mask, arrival_offsets=offsets,
                        excluded_arrivals=rejected_picks)
            return torch.from_numpy(wf), torch.from_numpy(labels), info
        except Exception as exc:
            self._reject(int(idx), row, exc)
            raise RuntimeError(f"Rejected manifest row {idx}: {exc}") from exc

    def __getitem__(self, idx):
        waveform, labels, _ = self.get_sample_with_metadata(idx)
        return waveform, labels

    def __len__(self):
        return len(self.manifest)

    def close(self):
        for reader in [*self._chunked.values(), *self._single_hdf5.values(),
                       self._noise_reader, self._prephase_reader]:
            if reader is not None:
                reader.close()
