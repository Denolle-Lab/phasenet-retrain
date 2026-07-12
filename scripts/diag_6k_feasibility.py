#!/usr/bin/env python3
"""
diag_6k_feasibility.py

For each trace in benchmark_manifest.csv, check whether a 6000-sample window
with P at offset P_OFFSET can be extracted cleanly from the raw waveform.

Reports per-dataset breakdown of:
  - clean     : no padding needed
  - pad_left  : not enough pre-P samples (raw waveform starts too late)
  - pad_right : not enough post-P samples (raw waveform ends too early)
  - both      : needs padding on both sides
  - skip      : load error or no P pick

Run from repo root:
    conda activate surface
    python scripts/diag_6k_feasibility.py
"""

import sys, re, os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

REPO_ROOT       = Path(__file__).parent.parent.resolve()
NB_DIR          = REPO_ROOT / "notebooks"
SEISBENCH_CACHE = Path(os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))) / "datasets"

MANIFEST_PATH   = NB_DIR / "benchmark_manifest.csv"
INDEX_PATH      = NB_DIR / "benchmark_waveforms_index.csv"

TARGET_SR       = 100
WINDOW_SAMPLES  = 6000   # 60 s at 100 Hz
P_OFFSET        = 1500   # P placed at sample 1500 (15 s in)

LOCAL_HDF5_DATASETS     = {"stead", "instancecounts", "pnw", "txed", "mlaapde", "ethz", "pisdl"}
SEISBENCH_API_DATASETS  = {"ceed", "vcseis", "aq2009gm", "cwa"}
SKIP_DATASETS           = {"obst2024"}
NATIVE_SR = {
    "stead": 100.0, "instancecounts": 100.0, "pnw": 100.0,
    "txed": 100.0, "mlaapde": 40.0, "ethz": 100.0, "pisdl": 100.0,
    "ceed": 100.0, "vcseis": 100.0, "aq2009gm": 100.0, "cwa": 100.0,
}


# ── Helpers (copied from notebook 05) ─────────────────────────────────────────

def parse_trace_name(trace_name):
    m = re.match(r"bucket(\d+)\$(\d+)", trace_name)
    if not m:
        raise ValueError(f"Unrecognised trace_name: {trace_name!r}")
    return int(m.group(1)), int(m.group(2))


def raw_length_local(ds_name, trace_name, source_month=None):
    """Return (n_samples,) of the raw waveform without loading the full array."""
    bucket_id, row_id = parse_trace_name(trace_name)
    if ds_name == "mlaapde":
        month_str = str(int(float(source_month)))
        hdf5_path = SEISBENCH_CACHE / "mlaapde" / f"waveforms_{month_str}.hdf5"
    else:
        hdf5_path = SEISBENCH_CACHE / ds_name / "waveforms.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        shape = f["data"][f"bucket{bucket_id}"].shape   # (n_rows, 3, n_samples)
    return shape[-1]


def build_sb_index():
    idx = {}
    for ds_name in SEISBENCH_API_DATASETS:
        ds_path = SEISBENCH_CACHE / ds_name
        index = {}
        for csv_path in sorted(ds_path.glob("metadata*.csv")):
            suffix    = csv_path.stem[len("metadata"):]
            hdf5_path = ds_path / f"waveforms{suffix}.hdf5"
            if not hdf5_path.exists():
                continue
            df = pd.read_csv(csv_path, usecols=["trace_name"])
            for tname in df["trace_name"]:
                index[tname] = hdf5_path
        idx[ds_name] = index
    return idx


def raw_length_seisbench(ds_name, trace_name, sb_index):
    hdf5_path = sb_index[ds_name].get(trace_name)
    if hdf5_path is None:
        raise KeyError(f"{trace_name!r} not in {ds_name} index")
    with h5py.File(hdf5_path, "r") as f:
        if ds_name == "ceed":
            shape = f[trace_name].shape
        elif ds_name in {"vcseis", "aq2009gm"}:
            bucket_id, row_id = parse_trace_name(trace_name)
            shape = f["data"][f"bucket{bucket_id}"].shape
        elif ds_name == "cwa":
            shape = f["data"][trace_name].shape
        else:
            raise ValueError(ds_name)
    return shape[-1]


# ── Load manifest + benchmark index (ok traces only) ──────────────────────────

manifest = pd.read_csv(MANIFEST_PATH)
ok_index = pd.read_csv(INDEX_PATH)
ok_index = ok_index[ok_index["status"] == "ok"]
ok_traces = set(ok_index["trace_name"])

# Filter manifest to ok benchmark traces with P picks
mf = manifest[
    manifest["trace_name"].isin(ok_traces) &
    manifest["p_arrival_sample"].notna()
].copy().reset_index(drop=True)

print(f"Benchmark traces with P pick: {len(mf):,}")
print(f"Datasets: {sorted(mf['dataset'].unique())}\n")

print("Building SeisBench index...")
sb_index = build_sb_index()
print("Done.\n")


# ── Feasibility check ─────────────────────────────────────────────────────────

records = []

for _, row in tqdm(mf.iterrows(), total=len(mf), desc="Checking"):
    tname   = row["trace_name"]
    ds_name = row["dataset"]
    p_raw   = float(row["p_arrival_sample"])
    native  = NATIVE_SR.get(ds_name, 100.0)

    # rescale p_arrival_sample to 100 Hz
    p_100 = int(round(p_raw * TARGET_SR / native))

    try:
        if ds_name in LOCAL_HDF5_DATASETS:
            n_raw = raw_length_local(ds_name, tname, row.get("source_month"))
        elif ds_name in SEISBENCH_API_DATASETS:
            n_raw = raw_length_seisbench(ds_name, tname, sb_index)
        else:
            records.append({"trace_name": tname, "dataset": ds_name,
                            "status": "skip", "pad_left": 0, "pad_right": 0})
            continue
    except Exception as e:
        records.append({"trace_name": tname, "dataset": ds_name,
                        "status": "load_error", "pad_left": 0, "pad_right": 0,
                        "note": str(e)[:60]})
        continue

    # resample raw length to 100 Hz
    n_100 = int(round(n_raw * TARGET_SR / native))

    win_start = p_100 - P_OFFSET
    win_end   = win_start + WINDOW_SAMPLES

    pad_left  = max(0, -win_start)
    pad_right = max(0, win_end - n_100)

    if pad_left == 0 and pad_right == 0:
        status = "clean"
    elif pad_left > 0 and pad_right > 0:
        status = "pad_both"
    elif pad_left > 0:
        status = "pad_left"
    else:
        status = "pad_right"

    records.append({
        "trace_name": tname, "dataset": ds_name, "status": status,
        "pad_left": pad_left, "pad_right": pad_right,
        "n_raw_100hz": n_100, "p_100hz": p_100,
    })

df = pd.DataFrame(records)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(f"{'STATUS':<12} {'COUNT':>7}  {'%':>6}")
print("-" * 70)
for status, grp in df.groupby("status"):
    pct = 100 * len(grp) / len(df)
    print(f"  {status:<10} {len(grp):>7,}  {pct:>5.1f}%")
print(f"  {'TOTAL':<10} {len(df):>7,}")

print("\n── Per-dataset breakdown ──────────────────────────────────────────────")
pivot = df.groupby(["dataset", "status"]).size().unstack(fill_value=0)
print(pivot.to_string())

needs_pad = df[df["status"].isin(["pad_left", "pad_right", "pad_both"])]
if len(needs_pad) > 0:
    print(f"\n── Padding amounts (samples at 100 Hz) ────────────────────────────────")
    print(f"  pad_left  — max: {df['pad_left'].max():.0f}  mean: {df[df['pad_left']>0]['pad_left'].mean():.0f}")
    print(f"  pad_right — max: {df['pad_right'].max():.0f}  mean: {df[df['pad_right']>0]['pad_right'].mean():.0f}")

print("\nDone.")
