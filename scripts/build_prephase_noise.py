#!/usr/bin/env python3
"""
build_prephase_noise.py

Extracts the 30-second window immediately before the P arrival
(P−30 s, P) from existing seismic traces in the SeisBench cache as
additional globally-distributed ambient noise for PhaseNet fine-tuning.

All data comes from the local SeisBench cache — no downloads required.

Sources
-------
  Every seismic trace in data/manifests/{train,val}.csv whose P pick is
  late enough to fit the full 30-second window.  Capped at --cap-per-ds
  traces per dataset for cross-region balance.

Output
------
  data/noise_prephase/waveforms.hdf5  — HDF5 group 'data/'
  data/noise_prephase/metadata.csv

Run from repo root:
    conda activate surface
    python scripts/build_prephase_noise.py [--seed 42] [--cap-per-ds 5000] [--dry-run]

Resume: re-running skips traces already in metadata.csv.
Next:   python scripts/add_prephase_to_manifests.py
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample

warnings.filterwarnings("ignore")

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

REPO_ROOT = Path(__file__).parent.parent.resolve()
OUT_DIR   = REPO_ROOT / "data" / "noise_prephase"
HDF5_PATH = OUT_DIR / "waveforms.hdf5"
META_PATH = OUT_DIR / "metadata.csv"

TARGET_SR   = 100
PRE_START_S = 30     # seconds before P — window start
PRE_END_S   = 0      # seconds before P — window end (right up to P arrival)
WIN_SAMPLES = int((PRE_START_S - PRE_END_S) * TARGET_SR)   # 3000 samples = 30 s
MIN_SAMPLES = 300    # reject resampled chunks shorter than 3 s

META_FIELDS = [
    "trace_name", "source_dataset",
    "network", "station",
    "latitude", "longitude",
    "tectonic_setting", "region",
    "distance_bin", "sampling_rate",
]

CHUNKED_PATHS = {
    "mlaapde": Path(SEISBENCH_CACHE) / "datasets" / "mlaapde",
    "cwa":     Path(SEISBENCH_CACHE) / "datasets" / "cwa",
}

PISDL_PATH = Path(SEISBENCH_CACHE) / "datasets" / "pisdl"

SBD_CLASSES = {
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
}

SKIP_DATASETS = {"noise_global", "noise_prephase"}


# ── geographic helpers ─────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def lat_lon_to_region(lat, lon):
    if lat is None or lon is None:
        return "unknown"
    if lat > 67:   return "Arctic"
    if lat < -60:  return "Antarctica"
    if  10 < lat < 80  and -170 < lon < -50:  return "North_America"
    if -60 < lat < 15  and  -90 < lon < -30:  return "South_America"
    if  35 < lat < 75  and  -15 < lon <  40:  return "Europe"
    if -40 < lat < 35  and  -20 < lon <  55:  return "Africa"
    if   5 < lat < 80  and   40 < lon < 100:  return "Central_Asia"
    if -15 < lat < 50  and  100 < lon < 150:  return "East_Asia"
    if -45 < lat < -10 and  110 < lon < 155:  return "Australia"
    return "Pacific_or_other"


def lat_lon_to_tectonic(lat, lon, default):
    if lat is None or lon is None:
        return default
    if (30 < lat < 50 and 130 < lon < 150) or \
       (-10 < lat < 10 and 115 < lon < 130) or \
       (-40 < lat <  0 and -80 < lon < -60) or \
       (50  < lat < 65 and -170 < lon < -140):
        return "subduction"
    if (-30 < lat < 20 and 25 < lon < 45) or \
       (60  < lat < 68 and -30 < lon < -10):
        return "rift"
    return default


# ── waveform helpers ───────────────────────────────────────────────────────────

def ensure_3ch(wf):
    w = np.asarray(wf, dtype=np.float32)
    if w.ndim == 1:
        w = np.stack([w, w, w])
    elif w.ndim == 2 and w.shape[0] != 3 and w.shape[-1] == 3:
        w = w.T
    if w.shape[0] > 3:
        w = w[:3]
    elif w.shape[0] < 3:
        pad = np.zeros((3 - w.shape[0], w.shape[-1]), dtype=np.float32)
        w = np.vstack([w, pad])
    return w


def extract_prephase(wf, sr, p_sample):
    """
    Cut the (P−35 s, P−5 s) window, resample to 100 Hz, return (3, 3000) or None.
    """
    sr = float(sr) if sr else TARGET_SR
    pre_start = int(p_sample - PRE_START_S * sr)
    pre_end   = int(p_sample - PRE_END_S   * sr)
    n = wf.shape[-1]
    if pre_start < 0 or pre_end > n or pre_end <= pre_start:
        return None

    chunk = wf[:, pre_start:pre_end].astype(np.float32)

    if sr != TARGET_SR:
        n_out = max(1, int(chunk.shape[-1] * TARGET_SR / sr))
        chunk = scipy_resample(chunk, n_out, axis=-1).astype(np.float32)

    if chunk.shape[-1] < MIN_SAMPLES:
        return None

    # pad or trim to exact WIN_SAMPLES
    if chunk.shape[-1] < WIN_SAMPLES:
        pad = np.zeros((3, WIN_SAMPLES - chunk.shape[-1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=-1)
    elif chunk.shape[-1] > WIN_SAMPLES:
        chunk = chunk[:, :WIN_SAMPLES]

    if chunk.std(axis=-1).min() < 1e-8:
        return None

    return chunk


def _safe_name(ds_name, trace_name):
    """HDF5 dataset key — replace '/' so h5py doesn't interpret it as a group."""
    return f"prephase_{ds_name}_{trace_name}".replace("/", "_")


# ── chunked HDF5 reader ────────────────────────────────────────────────────────

class ChunkedReader:
    def __init__(self, ds_path):
        self._path = Path(ds_path)
        self._handles = {}

    def get(self, chunk_tag, trace_name):
        if chunk_tag not in self._handles:
            hdf5 = self._path / f"waveforms_{chunk_tag}.hdf5"
            if not hdf5.exists():
                raise FileNotFoundError(f"Chunk HDF5 not found: {hdf5}")
            self._handles[chunk_tag] = h5py.File(hdf5, "r")
        h5 = self._handles[chunk_tag]
        key = f"data/{trace_name}"
        if key not in h5:
            raise KeyError(f"'{trace_name}' missing in chunk {chunk_tag}")
        return h5[key][()]

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()


# ── per-dataset processing ─────────────────────────────────────────────────────

def process_dataset(ds_name, rows, existing_names, data_grp, meta_writer,
                    dry_run, rng, cap):
    print(f"\n[{ds_name}]  {len(rows):,} candidate manifest rows", flush=True)

    if ds_name in SKIP_DATASETS:
        print("  SKIP — noise source")
        return 0, 0

    # ── load dataset ───────────────────────────────────────────────────────────
    is_chunked = ds_name in CHUNKED_PATHS
    try:
        if is_chunked:
            meta = _load_chunked_meta(CHUNKED_PATHS[ds_name])
            ds = None
            idx_map = None
            sr_default = float(TARGET_SR)   # chunked waveforms stored at 100 Hz
        elif ds_name == "pisdl":
            ds = sbd.WaveformDataset(str(PISDL_PATH))
            meta = ds.metadata.copy()
            idx_map = {n: i for i, n in enumerate(meta.get("trace_name", []))}
            sr_default = None
        elif ds_name in SBD_CLASSES:
            ds = SBD_CLASSES[ds_name]()
            meta = ds.metadata.copy()
            idx_map = {n: i for i, n in enumerate(meta.get("trace_name", []))}
            sr_default = None
        else:
            print(f"  SKIP — unknown dataset")
            return 0, 0
    except Exception as e:
        print(f"  SKIP — load failed: {e}")
        return 0, 0

    # ── metadata column discovery ──────────────────────────────────────────────
    sr_col  = next((c for c in ["trace_sampling_rate_hz"] if c in meta.columns), None)
    lat_col = next((c for c in ["station_latitude_deg", "source_latitude_deg"]
                    if c in meta.columns), None)
    lon_col = next((c for c in ["station_longitude_deg", "source_longitude_deg"]
                    if c in meta.columns), None)
    net_col = next((c for c in ["station_network_code", "network"] if c in meta.columns), None)
    sta_col = next((c for c in ["station_code", "station"] if c in meta.columns), None)

    # For chunked datasets build (chunk, trace_name) → metadata row lookup
    if is_chunked and "trace_name" in meta.columns and "chunk" in meta.columns:
        chunked_meta = {}
        for _, row in meta.iterrows():
            chunked_meta[(str(row["chunk"]), str(row["trace_name"]))] = row
        chunked_reader = ChunkedReader(CHUNKED_PATHS[ds_name])
    else:
        chunked_meta = {}
        chunked_reader = None

    # ── shuffle and cap ────────────────────────────────────────────────────────
    perm = rng.permutation(len(rows))
    rows = [rows[i] for i in perm]
    if cap and len(rows) > cap:
        rows = rows[:cap]

    n_written = n_skipped = 0

    for row in rows:
        trace_name = str(row["trace_name"])
        chunk      = str(row.get("chunk") or "")
        p_sample   = float(row["p_arrival_sample"])
        tname      = _safe_name(ds_name, trace_name)

        if tname in existing_names:
            continue

        if dry_run:
            existing_names.add(tname)
            n_written += 1
            continue

        # ── fetch waveform ─────────────────────────────────────────────────────
        try:
            if is_chunked:
                wf = chunked_reader.get(chunk, trace_name)
                sr = sr_default
                meta_row = chunked_meta.get((chunk, trace_name), {})
            else:
                idx = idx_map.get(trace_name)
                if idx is None:
                    n_skipped += 1
                    continue
                wf = ds.get_waveforms(idx)
                meta_row = meta.iloc[idx]
                if sr_col and sr_col in meta_row.index and not pd.isna(meta_row[sr_col]):
                    sr = float(meta_row[sr_col])
                else:
                    sr = float(TARGET_SR)
        except Exception:
            n_skipped += 1
            continue

        wf = ensure_3ch(wf)

        # ── extract pre-phase window ───────────────────────────────────────────
        chunk_wf = extract_prephase(wf, sr, p_sample)
        if chunk_wf is None:
            n_skipped += 1
            continue

        # ── geographic metadata ────────────────────────────────────────────────
        def _get(col):
            if col is None:
                return None
            if isinstance(meta_row, dict):
                return meta_row.get(col)
            return meta_row[col] if col in meta_row.index else None

        lat = _safe_float(_get(lat_col))
        lon = _safe_float(_get(lon_col))
        net = str(_get(net_col) or "")
        sta = str(_get(sta_col) or "")
        net = "" if net in ("nan", "None", "none") else net
        sta = "" if sta in ("nan", "None", "none") else sta

        region   = lat_lon_to_region(lat, lon)
        tectonic = lat_lon_to_tectonic(lat, lon, "unknown")

        # ── write ──────────────────────────────────────────────────────────────
        data_grp.create_dataset(tname, data=chunk_wf, compression="gzip", compression_opts=4)
        meta_writer.writerow({
            "trace_name":       tname,
            "source_dataset":   ds_name,
            "network":          net,
            "station":          sta,
            "latitude":         f"{lat:.4f}" if lat is not None else "",
            "longitude":        f"{lon:.4f}" if lon is not None else "",
            "tectonic_setting": tectonic,
            "region":           region,
            "distance_bin":     str(row.get("distance_bin", "") or ""),
            "sampling_rate":    TARGET_SR,
        })

        existing_names.add(tname)
        n_written += 1

        if n_written % 1000 == 0:
            print(f"  {n_written:,} written  {n_skipped} skipped", flush=True)

    if chunked_reader:
        chunked_reader.close()

    print(f"  Done — {n_written:,} written, {n_skipped} skipped")
    return n_written, n_skipped


def _load_chunked_meta(ds_path):
    path = Path(ds_path)
    csvs = sorted(f for f in path.iterdir()
                  if f.name.startswith("metadata_") and f.suffix == ".csv")
    if not csvs:
        raise FileNotFoundError(f"No metadata CSVs in {path}")
    frames = []
    for csv_f in csvs:
        chunk_tag = csv_f.stem.replace("metadata_", "")
        df = pd.read_csv(csv_f, low_memory=False)
        df["chunk"] = chunk_tag
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load manifests ─────────────────────────────────────────────────────────
    train_csv = REPO_ROOT / "data" / "manifests" / "train.csv"
    val_csv   = REPO_ROOT / "data" / "manifests" / "val.csv"

    dfs = []
    for p in [train_csv, val_csv]:
        if p.exists():
            dfs.append(pd.read_csv(p, low_memory=False))
    if not dfs:
        raise FileNotFoundError("No manifest CSVs found — run build_training_dataset.py first")

    manifest = pd.concat(dfs, ignore_index=True)
    manifest = manifest[~manifest["dataset_name"].isin(SKIP_DATASETS)]
    manifest["p_arrival_sample"] = pd.to_numeric(manifest["p_arrival_sample"], errors="coerce")
    manifest = manifest[manifest["p_arrival_sample"].notna()]
    # Quick pre-filter: at 100 Hz, need P >= PRE_START_S * 100 to fit the window.
    # Datasets at other rates are re-checked precisely in extract_prephase().
    manifest = manifest[manifest["p_arrival_sample"] >= PRE_START_S * TARGET_SR]

    print(f"Building pre-phase noise dataset")
    print(f"  Window       : P−{PRE_START_S}s → P−{PRE_END_S}s  ({WIN_SAMPLES} samples @ {TARGET_SR} Hz)")
    print(f"  Cap per ds   : {args.cap_per_ds}")
    print(f"  Seed         : {args.seed}")
    print(f"  Candidates   : {len(manifest):,} manifest rows (after P-sample pre-filter)")
    if args.dry_run:
        print("  DRY RUN      : no data written")
    print()

    # ── load existing trace names for resume ───────────────────────────────────
    existing_names = set()
    if META_PATH.exists():
        with open(META_PATH) as f:
            for row in csv.DictReader(f):
                existing_names.add(row["trace_name"])
        if existing_names:
            print(f"Resuming — {len(existing_names):,} traces already extracted\n")

    # ── open output files ──────────────────────────────────────────────────────
    if not args.dry_run:
        h5        = h5py.File(HDF5_PATH, "a")
        if "data" not in h5:
            h5.create_group("data")
        data_grp = h5["data"]

        meta_file   = open(META_PATH, "a", newline="")
        meta_writer = csv.DictWriter(meta_file, fieldnames=META_FIELDS)
        if META_PATH.stat().st_size == 0:
            meta_writer.writeheader()
    else:
        data_grp = h5 = meta_writer = meta_file = None

    # ── process each dataset ───────────────────────────────────────────────────
    total_written = len(existing_names)
    total_skipped = 0

    for ds_name, group in manifest.groupby("dataset_name"):
        rows = group.to_dict("records")
        nw, ns = process_dataset(
            ds_name, rows, existing_names,
            data_grp, meta_writer,
            dry_run=args.dry_run, rng=rng, cap=args.cap_per_ds,
        )
        total_written += nw
        total_skipped += ns
        if not args.dry_run and h5 is not None:
            h5.flush()
            meta_file.flush()

    if not args.dry_run and h5 is not None:
        h5.close()
        meta_file.close()

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pre-phase noise traces written : {total_written:,}")
    print(f"Skipped (window too short/bad) : {total_skipped}")
    print(f"Output : {OUT_DIR}")

    if META_PATH.exists():
        df = pd.read_csv(META_PATH)
        print(f"\nSource dataset breakdown:")
        for ds, n in df["source_dataset"].value_counts().items():
            print(f"  {ds:<22s} {n:>7,}")
        print(f"\nTectonic breakdown:")
        for t, n in df["tectonic_setting"].value_counts().items():
            print(f"  {t:<25s} {n:>7,}")
        print(f"\nRegion breakdown:")
        for r, n in df["region"].value_counts().items():
            print(f"  {r:<25s} {n:>7,}")
    print(f"{'='*60}")
    print("\nNext step: python scripts/add_prephase_to_manifests.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pre-phase noise windows from SeisBench cache"
    )
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--cap-per-ds", type=int, default=5000,
                        help="Max traces to extract per source dataset (default: 5000)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Count eligible traces without writing data")
    main(parser.parse_args())
