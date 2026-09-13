#!/usr/bin/env python3
"""Read-only #34B replay. Never downloads data, trains, or certifies historical access.

A fresh output directory contains input snapshots, per-occurrence JSONL, phase
summaries and provenance. Missing inputs produce an explicit incomplete report.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEGACY_REF = "3bf98c4"
KEYS = ["dataset_name", "chunk", "trace_name"]
NOISE = {"noise_global", "noise_prephase"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    before = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != before or digest(source) != before:
        raise RuntimeError(f"Input changed while snapshotting: {source}")
    return {"path": str(source.resolve()), "snapshot": str(destination), "sha256": before}


def array_digest(wave):
    wave = np.ascontiguousarray(wave)
    h = hashlib.sha256(str((wave.shape, wave.dtype.str)).encode())
    h.update(wave.tobytes())
    return h.hexdigest()


@contextmanager
def replay_rng(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def load_legacy(source):
    """Execute only definitions from the pinned repository loader, not its setup.

    In particular, do not execute its hardcoded server-cache assignment, warning
    filters, or dataset constructors. Reader paths are supplied by this runner.
    """
    import h5py
    import torch
    from torch.utils.data import Dataset
    wanted = {"ChunkedHDF5Reader", "SingleHDF5Reader", "ManifestDataset",
              "_normalise_std", "_resample_if_needed", "_ensure_3ch_cw", "_window",
              "_gaussian_label", "make_labels"}
    tree = ast.parse(source)
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise ValueError("Pinned legacy loader has an unexpected definition set")
    ns = dict(np=np, pd=pd, torch=torch, h5py=h5py, Path=Path, Dataset=Dataset,
              TARGET_SR=100, WINDOW_LEN=3001, LABEL_SIGMA=10,
              logger=logging.getLogger("legacy_replay"))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "legacy_loader.py", "exec"), ns)
    return SimpleNamespace(**ns), ns


class Sources:
    """Local files only. Generic SeisBench readers need explicit runtime options.

    This reconstructs source access on today's files; subclass filtering, metadata
    ordering and the historical runtime must be verified from run evidence.
    """
    def __init__(self, specs, legacy, wanted=None):
        import manifest_dataset as md
        self.md, self.specs, self.legacy = md, specs, legacy
        self.wanted = wanted or {}
        self.readers, self.errors, self.files = {}, {}, {}
        self.observed = {}

    def note_file(self, path):
        path = Path(path).resolve()
        if str(path) in self.files:
            return
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        self.files[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                     sha256=digest(path) if path.suffix == ".csv" else None)

    def reader(self, name, version, chunk=""):
        spec = self.specs[name]
        route = spec["route"]
        key = (name, version, str(chunk) if route == "chunked" else "")
        if key in self.errors:
            raise RuntimeError(self.errors[key])
        if key not in self.readers:
            path = Path(spec["path"])
            try:
                if route == "seisbench":
                    import seisbench.data as sbd
                    options = dict(spec["legacy_reader_options"])
                    if version == "corrected":
                        options.update(sampling_rate=None, component_order="ZNE",
                                       dimension_order="NCW", missing_components="pad")
                    # WaveformDataset reads local files; BenchmarkDataset subclasses
                    # are deliberately not constructed (they may download data).
                    for file in sorted(path.glob("metadata*.csv")):
                        self.note_file(file)
                    for file in sorted(path.glob("waveforms*.hdf5")):
                        self.note_file(file)
                    reader = sbd.WaveformDataset(path, **options)
                    reader._audit_index = (self.legacy.ManifestDataset._build_name_index(reader)
                                           if version == "legacy" else self.md.ManifestDataset._build_name_index(reader))
                elif route in {"single", "chunked"}:
                    suffix = spec.get("prefix", "waveforms_").removeprefix("waveforms") + str(chunk) if route == "chunked" else ""
                    h5_name, csv_name = f"waveforms{suffix}.hdf5", f"metadata{suffix}.csv"
                    self.note_file(path / h5_name)
                    if (path / csv_name).exists():
                        self.note_file(path / csv_name)
                    if version == "legacy":
                        reader = self.legacy.SingleHDF5Reader(path, h5_name)
                    else:
                        reader = self.md.SingleHDF5Reader(path, h5_name, csv_name, wanted=self.wanted.get(name))
                else:
                    raise ValueError(f"Unknown reader route: {route}")
                self.readers[key] = reader
            except Exception as exc:
                self.errors[key] = f"{type(exc).__name__}: {exc}"
                raise
        return self.readers[key]

    def fetch_legacy(self, row):
        from waveform_contract import metadata_rate
        name, trace, chunk = row["dataset_name"], row["trace_name"], row.get("chunk", "")
        try:
            route = self.specs[name]["route"]
            try:
                reader = self.reader(name, "legacy", chunk)
            except Exception as exc:
                if route != "chunked":
                    self.observed["legacy_initialization_error"] = f"{type(exc).__name__}: {exc}"
                raise
            if route == "seisbench":
                idx = reader._audit_index.get(trace)
                if idx is None:
                    raise KeyError(f"Legacy trace identity not found: {trace}")
                meta = reader.metadata.iloc[idx].to_dict()
                wf = reader.get_waveforms(idx)
                assumed = meta.get("trace_sampling_rate_hz")
                effective = reader.sampling_rate
                if effective is None:
                    try:
                        effective = metadata_rate(meta)
                    except ValueError:
                        effective = None  # instrumentation must not turn a successful read into failure
                identity = [name, str(meta.get("trace_chunk", "")), str(meta["trace_name"])]
                self.observed["legacy_returned_dimension_order"] = reader.dimension_order.replace("N", "")
                self.observed["legacy_returned_component_order"] = reader.component_order
            else:
                wf = reader.get_waveform(trace)
                assumed, effective = 100, None  # physical rate requires source evidence
                identity = [name, str(chunk) if route == "chunked" else "", str(trace)]
            self.observed.update(legacy_fetch_status="ok", legacy_assumed_rate_hz=assumed,
                                 legacy_reader_rate_hz=effective, legacy_source_identity=identity,
                                 legacy_fetched_shape=list(wf.shape), legacy_fetched_sha256=array_digest(wf))
            return wf, assumed
        except Exception as exc:
            self.observed.update(legacy_fetch_status="not_attempted" if "legacy_initialization_error" in self.observed else "failed",
                                 legacy_fetch_error=f"{type(exc).__name__}: {exc}")
            raise

    def fetch_corrected(self, row):
        name, trace, chunk = row["dataset_name"], row["trace_name"], row["chunk"]
        reader = self.reader(name, "corrected", chunk)
        if self.specs[name]["route"] == "seisbench":
            if not hasattr(reader, "_audit_proxy"):
                proxy = object.__new__(self.md.ManifestDataset)
                proxy._sbd_datasets = {name: reader}
                proxy._sbd_name_to_idx = {name: reader._audit_index}
                unique = {}
                for (_, t), i in reader._audit_index.items():
                    unique[t] = None if t in unique else i
                proxy._sbd_unique_names = {name: unique}
                reader._audit_proxy = proxy
            proxy = reader._audit_proxy
            record = proxy._fetch_sbd(name, trace, chunk, row)
            idx = reader._audit_index.get((chunk, trace))
            if idx is None and not chunk:
                idx = proxy._sbd_unique_names[name].get(trace)
            meta = reader.metadata.iloc[idx]
            identity = [name, str(meta.get("trace_chunk", "")), str(meta["trace_name"])]
        else:
            record = reader.get_record(trace, row)
            from waveform_contract import metadata_field
            handle = reader._handle()
            block = trace.split("$", 1)[0]
            defaults = {k: v[()] for k, v in handle["data_format"].items()} if "data_format" in handle else {}
            layers = [row, reader._trace_metadata(trace), dict(handle["data"][block].attrs), defaults]
            from waveform_contract import text_value
            dimensions = metadata_field(layers, "trace_dimension_order", "dimension_order")
            components = metadata_field(layers, "trace_component_order", "component_order")
            self.observed["stored_dimension_order"] = text_value(dimensions)
            self.observed["stored_component_order"] = text_value(components)
            identity = [name, str(chunk) if self.specs[name]["route"] == "chunked" else "", str(trace)]
        self.observed.update(source_rate_hz=record.sampling_rate, source_arrival_rate_hz=record.arrival_sampling_rate,
                             source_npts=record.waveform.shape[-1], corrected_source_identity=identity,
                             corrected_source_sha256=array_digest(record.waveform))
        return record

    def close(self):
        for reader in self.readers.values():
            if hasattr(reader, "close"):
                reader.close()

    def verify_unchanged(self):
        for path, before in self.files.items():
            after = Path(path).stat()
            if (after.st_size, after.st_mtime_ns) != (before["size"], before["mtime_ns"]):
                raise RuntimeError(f"Source changed during audit: {path}")
            if before["sha256"] and digest(path) != before["sha256"]:
                raise RuntimeError(f"Metadata changed during audit: {path}")


class Replay:
    def __init__(self, path, sources, legacy, namespace, window_len):
        self.sources = sources
        self.old = object.__new__(legacy.ManifestDataset)
        self.old.manifest = pd.read_csv(path, low_memory=False)  # preserve historical type inference
        self.old.augment, self.old.window_len, self.old._fetch_fail_count = False, window_len, 0
        self.old._chunked, self.old._single_hdf5 = {}, {}
        self.old._fetch_sbd = self.old._fetch_chunked = self.old._fetch_single_hdf5 = lambda *args: sources.fetch_legacy(self.old_row)
        self.old._noise_reader = self.old._prephase_reader = SimpleNamespace(
            get_waveform=lambda *args: sources.fetch_legacy(self.old_row)[0])
        self.new = object.__new__(sources.md.ManifestDataset)
        self.new.manifest = pd.read_csv(path, low_memory=False, dtype={"chunk": str, "trace_name": str})
        if "chunk" not in self.new.manifest:
            self.new.manifest["chunk"] = ""
        self.new.manifest["chunk"] = self.new.manifest.chunk.fillna("")
        self.new.augment, self.new.window_len = False, window_len
        self.new._fetch = sources.fetch_corrected
        self.new._reject = lambda *args: None  # per-row errors go to the audit output, never beside inputs
        original_window, original_labels, original_resample = namespace["_window"], namespace["make_labels"], namespace["_resample_if_needed"]
        def window(wave, pick, length):
            result = original_window(wave, pick, length)
            sources.observed["legacy_crop_start_sample"] = result[2]
            return result
        def labels(p, s, length):
            sources.observed["legacy_arrival_offsets"] = {"P": p, "S": s}
            return original_labels(p, s, length)
        def resample(wave, rate, target):
            result = original_resample(wave, rate, target)
            sources.observed.update(legacy_resample_input_npts=wave.shape[-1],
                                    legacy_resample_output_npts=result.shape[-1])
            return result
        namespace.update(_window=window, make_labels=labels, _resample_if_needed=resample)
        self.namespace, self.originals = namespace, (original_window, original_labels, original_resample)

    def row(self, idx, split, manifest_hash, exposure=None):
        self.old_row = self.old.manifest.iloc[idx].to_dict()
        row = self.new.manifest.iloc[idx].to_dict()
        sources = self.sources
        sources.observed = {}
        result = dict(split=split, row_index=idx, manifest_sha256=manifest_hash,
                      identity=[row[k] for k in KEYS], dataset_name=row["dataset_name"],
                      distance_bin=row.get("distance_bin", "unknown"), distance_km=row.get("distance_km"),
                      manifest_arrivals={"P": row.get("p_arrival_sample"), "S": row.get("s_arrival_sample")},
                      training_exposures=exposure, historical_fetch_status="unknown",
                      manifest_arrival_rate_hz=row.get("arrival_sampling_rate_hz"),
                      corrected_target_rate_hz=100, window_len=self.new.window_len,
                      crop_rng_basis="deterministic replay; historical cached random crop unavailable")
        seed = int(hashlib.sha256(f"{manifest_hash}:{idx}".encode()).hexdigest()[:8], 16)
        try:
            with replay_rng(seed):
                x, y = self.old[idx]
            result["legacy_status"] = ("initialization_error" if sources.observed.get("legacy_initialization_error")
                                       else "zero_noise_substitution" if sources.observed.get("legacy_fetch_status") == "failed" else "ok")
            result["legacy_finite_output"] = bool(np.isfinite(x.numpy()).all() and np.isfinite(y.numpy()).all())
            result["legacy_output_sha256"] = array_digest(x.numpy())
            if result["legacy_status"] == "initialization_error":
                result["legacy_finite_output"] = result["legacy_output_sha256"] = None
        except Exception as exc:
            result.update(legacy_status="preprocessing_error", legacy_error=f"{type(exc).__name__}: {exc}")
        try:
            with replay_rng(seed):
                x, y, info = self.new.get_sample_with_metadata(idx)
            result.update(corrected_status="ok", corrected=info, corrected_output_sha256=array_digest(x.numpy()))
        except Exception as exc:
            result.update(corrected_status="rejected", corrected_error=f"{type(exc).__name__}: {exc}")
        result.update(sources.observed)
        # Direct legacy reads use the stored waveform grid, regardless of the
        # hardcoded rate supplied to the old resampler. Do not infer a rate when
        # identity differs (e.g., numeric chunk inference or duplicate names).
        same = (result["legacy_source_identity"] == result["corrected_source_identity"]
                if "legacy_source_identity" in result and "corrected_source_identity" in result else None)
        result["same_source_identity"] = same
        if same and result.get("legacy_reader_rate_hz") is None:
            result["legacy_reader_rate_hz"] = result.get("source_rate_hz")
        shape = result.get("legacy_fetched_shape", [])
        dimensions = result.get("legacy_returned_dimension_order") or (result.get("stored_dimension_order") if same else None)
        inferred_axis = 0 if len(shape) == 1 or (len(shape) == 2 and shape[0] != 3 and shape[1] == 3) else 1
        time_axis = {"CW": 1, "WC": 0, "W": 0}.get(dimensions)
        result["legacy_time_axis_verified"] = inferred_axis == time_axis if time_axis is not None else None
        n_in, n_out = result.get("legacy_resample_input_npts"), result.get("legacy_resample_output_npts")
        physical_rate = result.get("legacy_reader_rate_hz")
        if n_in and n_out and physical_rate and result["legacy_time_axis_verified"]:
            result["legacy_physical_output_rate_hz"] = physical_rate * n_out / n_in
        for phase in ("P", "S"):
            old_off = result.get("legacy_arrival_offsets", {}).get(phase)
            new_off = result.get("corrected", {}).get("arrival_offsets", {}).get(phase)
            details = dict(legacy_offset=old_off, corrected_offset=new_off,
                           legacy_effective_label=(old_off is not None) if result["legacy_status"] in {"ok", "zero_noise_substitution"} else None,
                           corrected_effective_label=new_off is not None if result["corrected_status"] == "ok" else None)
            if same and old_off is not None and result.get("corrected"):
                reference = row.get(f"{phase.lower()}_arrival_sample")
                rate = result["corrected"]["arrival_rate_hz"]
                output_rate = result.get("legacy_physical_output_rate_hz")
                if output_rate and pd.notna(reference):
                    legacy_time = (old_off + result["legacy_crop_start_sample"]) / output_rate
                    source_end = (result["source_npts"] - 1) / result["source_rate_hz"]
                    reference_time = float(reference) / rate
                    details.update(legacy_label_time_from_source_s=legacy_time,
                                   source_arrival_time_s=reference_time,
                                   source_arrival_within_support=0 <= reference_time <= source_end,
                                   legacy_label_within_source_support=0 <= legacy_time <= source_end)
                    if details["source_arrival_within_support"]:
                        details["legacy_label_displacement_s"] = legacy_time - reference_time
                    result["legacy_crop_start_offset_s"] = result["legacy_crop_start_sample"] / output_rate
            result[phase] = details
        return clean(result)

    def close(self):
        self.namespace.update(zip(("_window", "make_labels", "_resample_if_needed"), self.originals))


def summarize(rows):
    groups = defaultdict(lambda: dict(occurrences=0, identities=set(), declared=0, legacy=0,
        corrected=0, corrected_rejected=0, legacy_error=0, zero_noise=0, comparable=0, displaced=0,
        exposures=0, exposure_unknown=0, rate_mismatch=0, identity_mismatch=0, axis_mismatch=0))
    for row in rows:
        for phase in ("P", "S"):
            key = (row["split"], row["dataset_name"], row.get("distance_bin"), row.get("source_rate_hz"), phase)
            g = groups[key]
            g["occurrences"] += 1
            g["identities"].add(tuple(row["identity"]))
            value = row["manifest_arrivals"][phase]
            try:
                declared = value is not None and np.isfinite(float(value)) and float(value) >= 0
            except (TypeError, ValueError):
                declared = False
            g["declared"] += declared
            g["legacy"] += row[phase]["legacy_effective_label"] is True
            g["legacy_error"] += row["legacy_status"] not in {"ok", "zero_noise_substitution"}
            g["corrected"] += row[phase]["corrected_effective_label"] is True
            g["corrected_rejected"] += row["corrected_status"] != "ok"
            g["zero_noise"] += row["legacy_status"] == "zero_noise_substitution"
            assumed, effective = row.get("legacy_assumed_rate_hz"), row.get("legacy_reader_rate_hz")
            g["rate_mismatch"] += assumed is not None and effective is not None and not np.isclose(assumed, effective, rtol=1e-6, atol=0)
            g["identity_mismatch"] += row.get("same_source_identity") is False
            g["axis_mismatch"] += row.get("legacy_time_axis_verified") is False
            delta = row[phase].get("legacy_label_displacement_s")
            g["comparable"] += delta is not None
            g["displaced"] += delta is not None and abs(delta) > .01
            g["exposure_unknown"] += row["training_exposures"] is None
            g["exposures"] += row["training_exposures"] or 0
    return [dict(zip(("split", "dataset_name", "distance_bin", "source_rate_hz", "phase"), key),
                 manifest_occurrences=g["occurrences"], unique_trace_identities=len(g["identities"]),
                 declared_labels=g["declared"], legacy_effective_labels=g["legacy"],
                 corrected_effective_labels=g["corrected"], corrected_rejected_rows=g["corrected_rejected"],
                 legacy_zero_noise_rows=g["zero_noise"], legacy_error_rows=g["legacy_error"], comparable_label_rows=g["comparable"],
                 displaced_label_rows=g["displaced"], legacy_rate_mismatch_rows=g["rate_mismatch"],
                 source_identity_mismatch_rows=g["identity_mismatch"], legacy_time_axis_mismatch_rows=g["axis_mismatch"],
                 training_exposures=g["exposures"] if not g["exposure_unknown"] else None,
                 rows_without_exposure_evidence=g["exposure_unknown"])
            for key, g in groups.items()]


def jsonl_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def default_specs(names, cache_root, repo_root):
    if not names:
        return {}
    import manifest_dataset as md
    specs = {}
    for name in names:
        if name in md._CHUNKED_DS:
            specs[name] = dict(route="chunked", path=str(cache_root / "datasets" / name), prefix=md._CHUNKED_DS[name][1])
        elif name in md._SINGLE_HDF5_DS or name in NOISE:
            path = repo_root / "data" / name if name in NOISE else cache_root / "datasets" / name
            specs[name] = dict(route="single", path=str(path))
        elif name in md._SBD_CLASSES:
            # Do not silently assert historical SeisBench options.
            specs[name] = dict(route="seisbench", path=str(cache_root / "datasets" / name))
        else:
            specs[name] = dict(route="unknown", path=str(cache_root / "datasets" / name))
    return specs


def run(args):
    import yaml
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(schema="34b-v1", status="incomplete", historical_attribution="unverified",
        git_head=git("rev-parse", "HEAD").strip(), host=platform.node(), python=platform.python_version(),
        inputs={}, missing=[], limitations=[
            "Replay fetch failures are not historical training failures without contemporaneous logs.",
            "A sorted-key checksum does not certify arrival columns, row order, runtime or run binding.",
            "Cached random crop positions and training exposures require historical run evidence.",
            "Generic local SeisBench readers do not certify historical subclass filtering/order.",
            "Row paths replay independently; any legacy initialization_error would have aborted the original whole-loader setup.",
            "HDF5 file bytes are not fully hashed: file stats and consumed array hashes are recorded."])
    try:
        for filename in ("audit_v7_rows.py", "manifest_dataset.py", "waveform_contract.py"):
            provenance["inputs"][filename] = archive(ROOT / "scripts" / filename, out / "code" / filename)
        config_path = Path(args.config).resolve()
        provenance["inputs"]["config"] = archive(config_path, out / "inputs/config.yaml")
        cfg = yaml.safe_load((out / "inputs/config.yaml").read_text())
        snapshots = {}
        for split in ("train", "val", "test"):
            source = (Path(args.manifest_dir) / f"{split}.csv") if args.manifest_dir else ROOT / cfg["data"][f"{split}_manifest"]
            if not source.is_file():
                provenance["missing"].append(str(source))
                continue
            snapshots[split] = out / f"inputs/{split}.csv"
            provenance["inputs"][split] = archive(source, snapshots[split])
        checksum_path = ROOT / "data/manifest_checksums.csv"
        provenance["inputs"]["reference_checksums"] = archive(checksum_path, out / "inputs/reference_checksums.csv")
        expected = pd.read_csv(checksum_path).set_index("manifest")
        from hash_manifests import hash_manifest
        names, wanted = set(), defaultdict(set)
        for split, path in snapshots.items():
            frame = pd.read_csv(path, dtype={"chunk": str, "trace_name": str})
            missing_columns = set(KEYS + ["p_arrival_sample", "s_arrival_sample"]) - set(frame.columns)
            if missing_columns:
                raise ValueError(f"{split}: missing manifest columns {sorted(missing_columns)}")
            names.update(frame.dataset_name.dropna().unique())
            for name, group in frame.groupby("dataset_name"):
                wanted[name].update(group.trace_name)
            actual_hash, n_rows = hash_manifest(path)
            reference = cfg["data"][f"{split}_manifest"]
            match = reference in expected.index and actual_hash == expected.loc[reference, "sha256_of_sorted_keys"] and n_rows == expected.loc[reference, "n_rows"]
            provenance["inputs"][split].update(rows=n_rows, sorted_key_sha256=actual_hash,
                reference_manifest=reference, reference_keys_match=bool(match))
        specs = default_specs(names, Path(args.cache_root).expanduser().resolve(), ROOT)
        if args.sources:
            provenance["inputs"]["sources"] = archive(args.sources, out / "inputs/sources.json")
            specs.update(json.loads((out / "inputs/sources.json").read_text()))
        provenance["sources"] = specs
        for name in names:
            if not Path(specs[name]["path"]).is_dir():
                provenance["missing"].append(f"source directory: {specs[name]['path']}")
            if specs[name]["route"] == "seisbench":
                required = {"sampling_rate", "component_order", "dimension_order", "missing_components"}
                absent = required - set(specs[name].get("legacy_reader_options", {}))
                if absent:
                    provenance["missing"].append(f"explicit legacy_reader_options for {name}: {sorted(absent)}")
        provenance["secondary_tables"] = dict(H2="pending row replay", noise_ordering="not computed: requires independent negatives, provenance, and frozen calibration thresholds")
        if args.inventory_only or provenance["missing"]:
            return 2
        source = git("show", f"{LEGACY_REF}:scripts/manifest_dataset.py")
        (out / "legacy_loader.py").write_text(source)
        provenance["legacy_loader"] = dict(reference=git("rev-parse", LEGACY_REF).strip(), sha256=digest(out / "legacy_loader.py"), run_binding="unverified")
        legacy, namespace = load_legacy(source)
        sources = Sources(specs, legacy, wanted)
        provenance["sources_used"] = sources.files
        import torch, seisbench, scipy, h5py
        provenance["versions"] = {"torch": torch.__version__, "seisbench": seisbench.__version__, "scipy": scipy.__version__, "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__}
        exposures = {}
        if args.exposures:
            provenance["inputs"]["exposures"] = archive(args.exposures, out / "inputs/exposures.csv")
            for item in pd.read_csv(out / "inputs/exposures.csv").to_dict("records"):
                index = float(item["row_index"])
                if not np.isfinite(index) or not index.is_integer():
                    raise ValueError("Training exposure row_index must be an integer")
                key = (item["split"], int(index))
                count = item["training_exposures"]
                if key[0] != "train" or key[1] < 0 or key[1] >= provenance["inputs"]["train"]["rows"] or key in exposures:
                    raise ValueError("Invalid or duplicate training exposure identity")
                if item["manifest_sha256"] != provenance["inputs"]["train"]["sha256"] or not np.isfinite(count) or count < 0 or int(count) != count:
                    raise ValueError("Invalid training exposure count or manifest hash")
                exposures[key] = int(count)
        try:
            with (out / "rows.jsonl").open("x") as stream:
                for split, path in snapshots.items():
                    replay = Replay(path, sources, legacy, namespace, cfg["data"].get("window_length", 3001))
                    try:
                        n = len(replay.new.manifest)
                        stop = min(n, args.max_rows) if args.max_rows is not None else n
                        for idx in range(stop):
                            row = replay.row(idx, split, provenance["inputs"][split]["sha256"], exposures.get((split, idx)))
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            if (idx + 1) % 10000 == 0:
                                print(f"{split}: {idx + 1}/{stop}", flush=True)
                    finally:
                        replay.close()
            sources.verify_unchanged()
        finally:
            sources.close()
        pd.DataFrame(summarize(jsonl_rows(out / "rows.jsonl"))).to_csv(out / "phase_summary.csv", index=False)
        provenance.update(status="replay_complete" if args.max_rows is None else "partial_replay",
                          max_rows_per_split=args.max_rows)
        provenance["secondary_tables"]["H2"] = "phase_summary.csv: declared versus effective P/S counts with explicit rejected/comparable denominators"
        provenance["outputs"] = {file.name: digest(file) for file in (out / "rows.jsonl", out / "phase_summary.csv", out / "legacy_loader.py")}
        return 0
    except Exception as exc:
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out / "provenance.json", provenance)
        (out / "README.md").write_text(
            f"# #34B audit artifact\n\nStatus: **{provenance['status']}**. Historical attribution: **unverified**.\n\n"
            "See provenance.json for exact inputs, missing evidence, runtime and output hashes. "
            "Replay completion does not close #34B. Training exposures are unknown unless supplied "
            "by a manifest-hash-bound ledger. Independent noise ordering is still outstanding.\n")
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256")
        (out / "CHECKSUMS.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(out)}\n" for p in files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory; an existing directory is never overwritten")
    parser.add_argument("--config", default=str(ROOT / "configs/finetune_jma_wc_global_v7.yaml"))
    parser.add_argument("--manifest-dir", help="Directory of preserved train/val/test.csv; never regenerate for this audit")
    parser.add_argument("--cache-root", default=os.environ.get("SEISBENCH_CACHE_ROOT", "~/.seisbench"))
    parser.add_argument("--sources", help="JSON source overrides; explicit legacy SeisBench options required")
    parser.add_argument("--exposures", help="CSV: split,row_index,manifest_sha256,training_exposures")
    parser.add_argument("--max-rows", type=int, help="Diagnostic prefix per split, never a corpus estimate")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    code = run(args)
    provenance = json.loads((Path(args.output) / "provenance.json").read_text())
    print(f"{provenance['status']}: {Path(args.output).resolve()}")
    for missing in provenance["missing"]:
        print(f"Missing: {missing}")
    return code


if __name__ == "__main__":
    sys.exit(main())
