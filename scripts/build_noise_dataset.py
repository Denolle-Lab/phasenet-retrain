#!/usr/bin/env python3
"""
build_noise_dataset.py

Extracts noise waveforms from existing SeisBench datasets in the local cache
and saves them to data/noise_global/ for noise-aware fine-tuning of PhaseNet.

Sources (all from cache — no downloads required):
  STEAD     – 20,000 noise traces  (North America, mixed tectonic)
  LenDB     – 20,000 noise traces  (global, lat-stratified for diversity)
  TXED      –  5,000 noise traces  (Texas, induced seismicity)
  VCSEIS    –   all  noise traces  (~12k, volcanic settings)
  OBST2024  –   all  noise traces  (~25k, ocean-bottom, globally distributed)

Total: ~82,000 noise traces covering diverse tectonic settings and regions.

Output
------
  data/noise_global/waveforms.hdf5  — HDF5, group 'data/', one dataset per trace
  data/noise_global/metadata.csv    — trace metadata (same format as FDSN version)

Run from repo root:
    conda activate surface
    python scripts/build_noise_dataset.py [--seed 42] [--dry-run]

Resume: re-running the script skips traces already in metadata.csv.
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample

warnings.filterwarnings("ignore")

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent.resolve()
OUT_DIR    = REPO_ROOT / "data" / "noise_global"
HDF5_PATH  = OUT_DIR / "waveforms.hdf5"
META_PATH  = OUT_DIR / "metadata.csv"

TARGET_SR   = 100    # Hz — resample all waveforms to this
MAX_SAMPLES = 6000   # cap stored waveform at 60 s to bound disk usage
MIN_SAMPLES = 300    # skip traces shorter than 3 s (too short for noise training)

META_FIELDS = [
    "trace_name", "network", "station", "location", "channel",
    "latitude", "longitude", "tectonic_setting", "region",
    "starttime", "sampling_rate",
    "trace_P_arrival_sample", "trace_S_arrival_sample",
]

# ── source configurations ──────────────────────────────────────────────────────
# cap=None means keep all noise traces from that source.
# lat_stratify=True samples equally from 6 lat bands to ensure global diversity.
NOISE_CONFIGS = [
    dict(
        name="stead",
        cls=sbd.STEAD,
        cat_col="trace_category", cat_val="noise",
        lat_col="station_latitude_deg", lon_col="station_longitude_deg",
        net_col="station_network_code", sta_col="station_code", chan_col=None,
        sr_col="trace_sampling_rate_hz",
        default_tectonic="mixed", default_region="North_America",
        cap=20_000, lat_stratify=False,
    ),
    dict(
        name="lendb",
        cls=sbd.LenDB,
        cat_col="trace_category", cat_val="noise",
        lat_col="station_latitude_deg", lon_col="station_longitude_deg",
        net_col="station_network_code", sta_col="station_code", chan_col=None,
        sr_col="trace_sampling_rate_hz",
        default_tectonic="mixed", default_region="unknown",
        cap=20_000, lat_stratify=True,  # 80% of LenDB noise is N temperate; force diversity
    ),
    dict(
        name="txed",
        cls=sbd.TXED,
        cat_col="trace_category", cat_val="noise",
        lat_col=None, lon_col=None,              # station lat/lon all NaN for TXED noise
        net_col=None, sta_col="station_code", chan_col=None,
        sr_col="trace_sampling_rate_hz",
        default_tectonic="induced_seismicity", default_region="Texas_USA",
        cap=5_000, lat_stratify=False,
    ),
    dict(
        name="vcseis",
        cls=sbd.VCSEIS,
        cat_col="source_type", cat_val="noise",
        lat_col="station_latitude_deg", lon_col="station_longitude_deg",
        net_col="station_network_code", sta_col="station_code", chan_col=None,
        sr_col="trace_sampling_rate_hz",
        default_tectonic="volcanic", default_region="volcanic",
        cap=None, lat_stratify=False,   # take all ~12k
    ),
    dict(
        name="obst2024",
        cls=sbd.OBST2024,
        cat_col="trace_category", cat_val="noise",
        lat_col="station_latitude_deg", lon_col="station_longitude_deg",
        net_col="station_network_code", sta_col="station_code", chan_col="trace_channel",
        sr_col=None,                             # assume 100 Hz
        default_tectonic="ocean_bottom", default_region="ocean_bottom",
        cap=None, lat_stratify=False,   # take all ~25k
    ),
]


# ── geographic helpers ─────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def lat_lon_to_region(lat, lon):
    """Approximate geographic region from station coordinates."""
    if lat is None or lon is None:
        return "unknown"
    if lat > 67:
        return "Arctic"
    if lat < -60:
        return "Antarctica"
    if 10 < lat < 80 and -170 < lon < -50:
        return "North_America"
    if -60 < lat < 15 and -90 < lon < -30:
        return "South_America"
    if 35 < lat < 75 and -15 < lon < 40:
        return "Europe"
    if -40 < lat < 35 and -20 < lon < 55:
        return "Africa"
    if 5 < lat < 80 and 40 < lon < 100:
        return "Central_Asia"
    if -15 < lat < 50 and 100 < lon < 150:
        return "East_Asia"
    if -45 < lat < -10 and 110 < lon < 155:
        return "Australia"
    return "Pacific_or_other"


def lat_lon_to_tectonic(lat, lon, default):
    """Rough tectonic classification from station coordinates."""
    if lat is None or lon is None:
        return default
    # Subduction margins (very approximate)
    if (30 < lat < 50 and 130 < lon < 150) or \
       (-10 < lat < 10 and 115 < lon < 130) or \
       (-40 < lat < 0 and -80 < lon < -60) or \
       (50 < lat < 65 and -170 < lon < -140):
        return "subduction"
    # Rift systems
    if (-30 < lat < 20 and 25 < lon < 45) or \
       (60 < lat < 68 and -30 < lon < -10):
        return "rift"
    return default


# ── waveform helpers ───────────────────────────────────────────────────────────

def ensure_3ch_100hz(wf, sr):
    """
    Convert waveform to (3, N) float32 at 100 Hz.
    Returns None if the trace is flat, too short, or otherwise invalid.
    """
    if wf is None:
        return None
    wf = np.asarray(wf, dtype=np.float32)

    # (N, 3) → (3, N)
    if wf.ndim == 1:
        wf = np.stack([wf, wf, wf])
    elif wf.ndim == 2 and wf.shape[0] != 3 and wf.shape[-1] == 3:
        wf = wf.T
    # Channel count normalisation
    if wf.shape[0] > 3:
        wf = wf[:3]
    elif wf.shape[0] < 3:
        pad = np.zeros((3 - wf.shape[0], wf.shape[-1]), dtype=np.float32)
        wf = np.vstack([wf, pad])

    # Resample
    sr = float(sr) if sr else TARGET_SR
    if sr != TARGET_SR:
        n_out = max(1, int(wf.shape[-1] * TARGET_SR / sr))
        wf = resample(wf, n_out, axis=-1).astype(np.float32)

    # Cap length
    if wf.shape[-1] > MAX_SAMPLES:
        wf = wf[:, :MAX_SAMPLES]

    # Reject too-short or flat traces
    if wf.shape[-1] < MIN_SAMPLES:
        return None
    if wf.std(axis=-1).min() < 1e-8:
        return None

    return wf


# ── stratification ─────────────────────────────────────────────────────────────

def stratify_by_lat(df, lat_col, cap, rng):
    """
    Sub-sample `df` to at most `cap` rows with roughly equal lat-band coverage.
    The 6 bands are: <-60, -60:-30, -30:0, 0:30, 30:60, >60.
    """
    lats = pd.to_numeric(df[lat_col], errors="coerce")
    band_edges = [-90, -60, -30, 0, 30, 60, 90]
    per_band = max(1, cap // (len(band_edges) - 1))

    sampled = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band_idx = df.index[(lats >= lo) & (lats < hi)]
        n = min(len(band_idx), per_band)
        if n > 0:
            sampled.extend(rng.choice(band_idx, size=n, replace=False).tolist())

    # Fill remaining slots from any band
    remaining = cap - len(sampled)
    if remaining > 0:
        pool = list(set(df.index) - set(sampled))
        extra = min(remaining, len(pool))
        if extra > 0:
            sampled.extend(rng.choice(pool, size=extra, replace=False).tolist())

    return df.loc[sampled]


# ── per-source extraction ──────────────────────────────────────────────────────

def process_source(cfg, rng, existing_names, data_grp, meta_writer, dry_run=False):
    """
    Extract noise waveforms from one SeisBench source.
    Returns (n_written, n_skipped).
    """
    name = cfg["name"]
    print(f"\n[{name}]", flush=True)

    try:
        ds = cfg["cls"]()
    except Exception as e:
        print(f"  SKIP — failed to load: {e}")
        return 0, 0

    meta = ds.metadata.copy().reset_index(drop=True)  # 0-based pos index

    # ── filter to noise traces ─────────────────────────────────────────────────
    cat_col = cfg.get("cat_col")
    if not cat_col or cat_col not in meta.columns:
        print(f"  SKIP — column '{cat_col}' not found")
        return 0, 0

    noise_mask = meta[cat_col].str.lower().str.contains(cfg["cat_val"], na=False)
    noise_meta = meta[noise_mask]
    print(f"  {len(noise_meta):,} noise traces available")

    # ── sub-sample / stratify ─────────────────────────────────────────────────
    cap = cfg.get("cap")
    if cfg.get("lat_stratify") and cfg.get("lat_col") and cfg["lat_col"] in noise_meta.columns:
        noise_meta = stratify_by_lat(noise_meta, cfg["lat_col"], cap, rng)
        print(f"  {len(noise_meta):,} after lat-stratified sampling (cap={cap:,})")
    elif cap and len(noise_meta) > cap:
        sel = rng.choice(noise_meta.index, size=cap, replace=False)
        noise_meta = noise_meta.loc[sel]
        print(f"  {len(noise_meta):,} after random cap={cap:,}")

    # Sort positional indices for sequential HDF5 reads (same bucket = adjacent)
    pos_indices = sorted(noise_meta.index.tolist())
    target = len(pos_indices)

    n_written = n_skipped = 0
    for pos_idx in pos_indices:
        row = meta.iloc[pos_idx]

        # Trace name: deterministic — source + positional index in source metadata
        tname = f"{name}_{pos_idx:07d}"
        if tname in existing_names:
            continue

        if dry_run:
            existing_names.add(tname)
            n_written += 1
            continue

        # ── fetch waveform ─────────────────────────────────────────────────────
        try:
            wf = ds.get_waveforms(pos_idx)
            sr_col = cfg.get("sr_col")
            sr = float(row[sr_col]) if sr_col and sr_col in row.index and not pd.isna(row[sr_col]) else TARGET_SR
        except Exception:
            n_skipped += 1
            continue

        wf = ensure_3ch_100hz(wf, sr)
        if wf is None:
            n_skipped += 1
            continue

        # ── geographic metadata ────────────────────────────────────────────────
        lat_col = cfg.get("lat_col")
        lon_col = cfg.get("lon_col")
        lat = _safe_float(row[lat_col]) if lat_col and lat_col in row.index else None
        lon = _safe_float(row[lon_col]) if lon_col and lon_col in row.index else None

        region   = lat_lon_to_region(lat, lon) if lat is not None else cfg["default_region"]
        tectonic = lat_lon_to_tectonic(lat, lon, cfg["default_tectonic"]) if lat is not None else cfg["default_tectonic"]

        net_col  = cfg.get("net_col")
        sta_col  = cfg.get("sta_col")
        chan_col = cfg.get("chan_col")
        net  = str(row[net_col])  if net_col  and net_col  in row.index else ""
        sta  = str(row[sta_col])  if sta_col  and sta_col  in row.index else ""
        chan = str(row[chan_col]) if chan_col and chan_col in row.index else ""

        # ── write ──────────────────────────────────────────────────────────────
        data_grp.create_dataset(tname, data=wf, compression="gzip", compression_opts=4)
        meta_writer.writerow({
            "trace_name":             tname,
            "network":                net if net != "nan" else "",
            "station":                sta if sta != "nan" else "",
            "location":               "",
            "channel":                chan if chan != "nan" else "",
            "latitude":               f"{lat:.4f}" if lat is not None else "",
            "longitude":              f"{lon:.4f}" if lon is not None else "",
            "tectonic_setting":       tectonic,
            "region":                 region,
            "starttime":              "",
            "sampling_rate":          TARGET_SR,
            "trace_P_arrival_sample": "",
            "trace_S_arrival_sample": "",
        })

        existing_names.add(tname)
        n_written += 1

        if n_written % 2000 == 0:
            pct = 100 * (n_written + n_skipped) / target
            print(f"  {n_written:,} written  {n_skipped} skipped  ({pct:.0f}%)", flush=True)

    print(f"  Done — {n_written:,} written, {n_skipped} skipped")
    return n_written, n_skipped


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expected = sum(
        cfg["cap"] if cfg["cap"] else
        (25_000 if cfg["name"] == "obst2024" else 12_415)
        for cfg in NOISE_CONFIGS
    )
    print(f"Building noise dataset from SeisBench cache")
    print(f"  Output   : {OUT_DIR}")
    print(f"  Expected : ~{expected:,} noise traces")
    print(f"  Seed     : {args.seed}")
    if args.dry_run:
        print("  DRY RUN  : no data written")
    print()

    # ── load already-written trace names (for resume) ──────────────────────────
    existing_names = set()
    if META_PATH.exists():
        with open(META_PATH) as f:
            for row in csv.DictReader(f):
                existing_names.add(row["trace_name"])
        if existing_names:
            print(f"Resuming — {len(existing_names):,} traces already in metadata.csv\n")

    # ── open output files ──────────────────────────────────────────────────────
    if not args.dry_run:
        hdf5_file = h5py.File(HDF5_PATH, "a")
        if "data" not in hdf5_file:
            hdf5_file.create_group("data")
        data_grp = hdf5_file["data"]

        meta_file   = open(META_PATH, "a", newline="")
        meta_writer = csv.DictWriter(meta_file, fieldnames=META_FIELDS)
        if META_PATH.stat().st_size == 0 or not META_PATH.exists():
            meta_writer.writeheader()
    else:
        data_grp = hdf5_file = meta_writer = meta_file = None

    # ── process each source ────────────────────────────────────────────────────
    total_written = len(existing_names)
    total_skipped = 0

    for cfg in NOISE_CONFIGS:
        nw, ns = process_source(
            cfg, rng, existing_names, data_grp, meta_writer,
            dry_run=args.dry_run,
        )
        total_written += nw
        total_skipped += ns
        if not args.dry_run and data_grp is not None:
            hdf5_file.flush()
            meta_file.flush()

    # ── finish ─────────────────────────────────────────────────────────────────
    if not args.dry_run and hdf5_file is not None:
        hdf5_file.close()
        meta_file.close()

    print(f"\n{'='*60}")
    print(f"Total noise traces written : {total_written:,}")
    print(f"Total skipped (bad wf)     : {total_skipped}")
    print(f"Output : {OUT_DIR}")

    if META_PATH.exists():
        df = pd.read_csv(META_PATH)
        print(f"\nTectonic breakdown:")
        for t, n in df["tectonic_setting"].value_counts().items():
            print(f"  {t:<25s} {n:>7,}")
        print(f"\nRegion breakdown:")
        for r, n in df["region"].value_counts().items():
            print(f"  {r:<25s} {n:>7,}")
    print(f"{'='*60}")
    print("\nNext step: python scripts/add_noise_to_manifests.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build noise dataset from SeisBench cache"
    )
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Count available traces without writing anything")
    main(parser.parse_args())
