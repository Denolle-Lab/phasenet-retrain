"""
manifest_dataset.py

PyTorch Dataset that loads waveforms on-the-fly from a manifest CSV
produced by build_training_dataset.py.

Each manifest row identifies a trace by (dataset_name, trace_name, chunk).
Waveforms are fetched from the local SeisBench cache — no data is copied.

Chunked datasets (MLAAPDE, CWA) are accessed by opening their per-chunk
HDF5 files directly with h5py.  All other datasets use SeisBench's standard
get_waveforms() interface.
"""

import logging
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

# ──────────────────────────────────────────────────────────────────────────────
# Chunked HDF5 reader (for MLAAPDE and CWA)
# ──────────────────────────────────────────────────────────────────────────────

class ChunkedHDF5Reader:
    """
    Reads waveforms from chunked SeisBench datasets (MLAAPDE, CWA) by
    trace_name.  Opens each HDF5 chunk lazily and keeps handles cached.
    """

    def __init__(self, ds_path, hdf5_prefix="waveforms_"):
        self.ds_path = Path(ds_path)
        self.hdf5_prefix = hdf5_prefix
        self._handles = {}  # chunk_tag -> h5py.File

    def _get_handle(self, chunk_tag):
        if chunk_tag not in self._handles:
            hdf5 = self.ds_path / f"{self.hdf5_prefix}{chunk_tag}.hdf5"
            if not hdf5.exists():
                raise FileNotFoundError(f"HDF5 chunk not found: {hdf5}")
            self._handles[chunk_tag] = h5py.File(hdf5, "r")
        return self._handles[chunk_tag]

    def get_waveform(self, chunk_tag, trace_name):
        h5 = self._get_handle(chunk_tag)
        # SeisBench stores waveforms under 'data/<trace_name>'
        key = f"data/{trace_name}"
        if key not in h5:
            raise KeyError(f"Trace '{trace_name}' not found in chunk {chunk_tag}")
        return h5[key][()]  # shape: (channels, samples) or (samples, channels)

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()


class SingleHDF5Reader:
    """
    Reads waveforms directly from a single SeisBench-format HDF5 file by
    trace_name.  Avoids loading the full metadata CSV into memory — important
    for large datasets (ross2018gpd: 4.77M rows, meier2019jgr: 1.06M rows).
    """

    def __init__(self, ds_path, hdf5_name="waveforms.hdf5"):
        hdf5 = Path(ds_path) / hdf5_name
        if not hdf5.exists():
            raise FileNotFoundError(f"HDF5 not found: {hdf5}")
        self._h5 = h5py.File(hdf5, "r")

    def get_waveform(self, trace_name):
        key = f"data/{trace_name}"
        if key not in self._h5:
            raise KeyError(f"Trace '{trace_name}' not found in HDF5")
        return self._h5[key][()]

    def close(self):
        self._h5.close()


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

class NoiseGlobalReader:
    """Reads waveforms from data/noise_global/waveforms.hdf5."""

    def __init__(self):
        hdf5 = _NOISE_GLOBAL_PATH / "waveforms.hdf5"
        if not hdf5.exists():
            raise FileNotFoundError(
                f"noise_global dataset not found at {hdf5}. "
                "Run scripts/build_noise_dataset.py first."
            )
        self._h5 = h5py.File(hdf5, "r")
        self._grp = self._h5["data"]

    def get_waveform(self, trace_name):
        if trace_name not in self._grp:
            raise KeyError(f"Trace '{trace_name}' not found in noise_global HDF5")
        return self._grp[trace_name][()]

    def close(self):
        self._h5.close()


class NoisePrephaseReader:
    """Reads waveforms from data/noise_prephase/waveforms.hdf5."""

    def __init__(self):
        hdf5 = _NOISE_PREPHASE_PATH / "waveforms.hdf5"
        if not hdf5.exists():
            raise FileNotFoundError(
                f"noise_prephase dataset not found at {hdf5}. "
                "Run scripts/build_prephase_noise.py first."
            )
        self._h5 = h5py.File(hdf5, "r")
        self._grp = self._h5["data"]

    def get_waveform(self, trace_name):
        if trace_name not in self._grp:
            raise KeyError(f"Trace '{trace_name}' not found in noise_prephase HDF5")
        return self._grp[trace_name][()]

    def close(self):
        self._h5.close()

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
    if src_sr == tgt_sr or src_sr is None:
        return waveform
    from scipy.signal import resample
    n_out = int(waveform.shape[-1] * tgt_sr / src_sr)
    return resample(waveform, n_out, axis=-1)


def _ensure_3ch_cw(waveform):
    """Return (3, samples) array regardless of input shape."""
    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim == 1:
        w = np.stack([w, w, w])
    elif w.shape[0] != 3 and w.shape[-1] == 3:
        w = w.T  # (samples, 3) → (3, samples)
    if w.shape[0] != 3:
        # pad / trim channels
        if w.shape[0] > 3:
            w = w[:3]
        else:
            pad = np.zeros((3 - w.shape[0], w.shape[1]), dtype=np.float32)
            w = np.vstack([w, pad])
    return w


def _window(waveform, p_sample, window_len=WINDOW_LEN):
    """
    Cut a window of `window_len` samples centred roughly on the P arrival.
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
    p_offset = p_int - start
    return chunk, p_offset, start


def _gaussian_label(size, centre, sigma=LABEL_SIGMA):
    x = np.arange(size, dtype=np.float32)
    if centre is None or np.isnan(centre):
        return np.zeros(size, dtype=np.float32)
    g = np.exp(-((x - centre) ** 2) / (2 * sigma ** 2))
    return g


def make_labels(p_offset, s_offset, window_len=WINDOW_LEN):
    """
    Build (3, window_len) label tensor: [noise, P, S].
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
    """
    Loads PhaseNet training samples from a manifest CSV.

    Parameters
    ----------
    manifest_csv : str | Path
        Path to one of train.csv / val.csv / test.csv from build_training_dataset.py
    augment : bool
        Apply amplitude jitter and waveform reversal augmentation during training.
    window_len : int
        Output window length in samples (default 3001 = 30 s @ 100 Hz).
    """

    def __init__(self, manifest_csv, augment=False, window_len=WINDOW_LEN):
        self.manifest  = pd.read_csv(manifest_csv, low_memory=False)
        self.augment   = augment
        self.window_len = window_len

        # ── open SeisBench datasets once ──────────────────────────────────────
        self._sbd_datasets   = {}   # ds_name -> loaded SeisBench dataset
        self._sbd_name_to_idx = {}  # ds_name -> {trace_name: int_idx}
        self._chunked         = {}  # ds_name -> ChunkedHDF5Reader
        self._single_hdf5     = {}  # ds_name -> SingleHDF5Reader

        self._noise_reader    = None
        self._prephase_reader = None
        # Counts waveform-fetch failures (see __getitem__) so they show up in
        # training logs instead of silently becoming fake all-noise samples.
        # Per-worker if num_workers > 0 -- each DataLoader worker gets its own
        # copy of this Dataset, but every worker's stderr still reaches the
        # training log, so nothing goes unseen.
        self._fetch_fail_count = 0

        needed = self.manifest["dataset_name"].unique()
        for ds_name in needed:
            if ds_name in _CHUNKED_DS:
                ds_path, hdf5_prefix = _CHUNKED_DS[ds_name]
                self._chunked[ds_name] = ChunkedHDF5Reader(ds_path, hdf5_prefix=hdf5_prefix)
            elif ds_name in _SINGLE_HDF5_DS:
                self._single_hdf5[ds_name] = SingleHDF5Reader(_SINGLE_HDF5_DS[ds_name])
            elif ds_name == "noise_global":
                self._noise_reader = NoiseGlobalReader()
            elif ds_name == "noise_prephase":
                self._prephase_reader = NoisePrephaseReader()
            elif ds_name in _SBD_CLASSES:
                ds = _SBD_CLASSES[ds_name]()
                self._sbd_datasets[ds_name] = ds
                self._sbd_name_to_idx[ds_name] = self._build_name_index(ds)
            else:
                raise ValueError(f"Unknown dataset '{ds_name}' in manifest")

    @staticmethod
    def _build_name_index(ds):
        meta = ds.metadata
        if "trace_name" not in meta.columns:
            return {}
        return {name: idx for idx, name in enumerate(meta["trace_name"])}

    # ── waveform fetchers ──────────────────────────────────────────────────────

    def _fetch_sbd(self, ds_name, trace_name):
        ds  = self._sbd_datasets[ds_name]
        idx = self._sbd_name_to_idx[ds_name].get(trace_name)
        if idx is None:
            raise KeyError(f"trace_name '{trace_name}' not found in {ds_name}")
        wf = ds.get_waveforms(idx)
        sr = ds.metadata.iloc[idx].get("trace_sampling_rate_hz", None)
        return wf, sr

    def _fetch_chunked(self, ds_name, trace_name, chunk):
        reader = self._chunked[ds_name]
        wf = reader.get_waveform(chunk, trace_name)
        return wf, TARGET_SR  # chunked datasets are stored at 100 Hz

    def _fetch_single_hdf5(self, ds_name, trace_name):
        wf = self._single_hdf5[ds_name].get_waveform(trace_name)
        return wf, TARGET_SR  # SeisBench convention: stored at 100 Hz

    # ── __getitem__ ─────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        ds_name    = row["dataset_name"]
        trace_name = row["trace_name"]
        chunk      = row.get("chunk", "")
        p_sample   = float(row["p_arrival_sample"])
        s_sample   = row["s_arrival_sample"]
        s_sample   = float(s_sample) if pd.notna(s_sample) else None

        # ── fetch raw waveform ─────────────────────────────────────────────────
        is_noise = ds_name in ("noise_global", "noise_prephase")
        try:
            if ds_name in self._chunked:
                wf, sr = self._fetch_chunked(ds_name, trace_name, chunk)
            elif ds_name in self._single_hdf5:
                wf, sr = self._fetch_single_hdf5(ds_name, trace_name)
            elif ds_name == "noise_global":
                wf = self._noise_reader.get_waveform(trace_name)
                sr = TARGET_SR
            elif ds_name == "noise_prephase":
                wf = self._prephase_reader.get_waveform(trace_name)
                sr = TARGET_SR
            else:
                wf, sr = self._fetch_sbd(ds_name, trace_name)
        except Exception as exc:
            # Return a zero sample rather than crashing a training batch --
            # but log it (issue #14): these were previously silent, so a
            # systemic fetch problem (missing chunk file, bad HDF5 handle)
            # could inject an unbounded number of fake all-noise traces into
            # training without any visibility.
            self._fetch_fail_count += 1
            if self._fetch_fail_count <= 20 or self._fetch_fail_count % 500 == 0:
                logger.warning(
                    "waveform fetch failed (#%d so far) dataset=%s trace_name=%s chunk=%r: "
                    "%s -- returning a zero/noise sample instead",
                    self._fetch_fail_count, ds_name, trace_name, chunk, exc,
                )
            wf_zero = torch.zeros(3, self.window_len)
            lbl_zero = torch.zeros(3, self.window_len)
            lbl_zero[2] = 1.0  # N channel (index 2 in PSN ordering)
            return wf_zero, lbl_zero

        # ── preprocess ────────────────────────────────────────────────────────
        wf = _ensure_3ch_cw(wf).astype(np.float32)
        wf = _resample_if_needed(wf, sr, TARGET_SR)

        if is_noise or p_sample is None or (isinstance(p_sample, float) and np.isnan(p_sample)):
            # Pure noise trace — pad/trim to window_len, set no picks
            n = wf.shape[-1]
            if n >= self.window_len:
                # Random start position for variety during training
                max_start = n - self.window_len
                start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
                wf = wf[:, start: start + self.window_len]
            else:
                pad = np.zeros((3, self.window_len - n), dtype=np.float32)
                wf = np.concatenate([wf, pad], axis=-1)
            p_off, s_off = None, None
        else:
            wf, p_off, start = _window(wf, p_sample, self.window_len)
            s_off = (float(s_sample) - start) if s_sample is not None else None
            # clip offsets outside window to None (no label)
            if p_off < 0 or p_off >= self.window_len:
                p_off = None
            if s_off is not None and (s_off < 0 or s_off >= self.window_len):
                s_off = None

        wf = _normalise_std(wf)

        # ── augmentation ──────────────────────────────────────────────────────
        if self.augment:
            # amplitude jitter
            scale = np.random.uniform(0.5, 2.0)
            wf = wf * scale
            # random polarity flip on all components
            if np.random.random() < 0.1:
                wf = -wf

        # ── labels ────────────────────────────────────────────────────────────
        labels = make_labels(p_off, s_off, self.window_len)

        return torch.from_numpy(wf), torch.from_numpy(labels)

    def __len__(self):
        return len(self.manifest)

    def close(self):
        for reader in self._chunked.values():
            reader.close()
        for reader in self._single_hdf5.values():
            reader.close()
        if self._noise_reader is not None:
            self._noise_reader.close()
        if self._prephase_reader is not None:
            self._prephase_reader.close()
