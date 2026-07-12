#!/usr/bin/env python3
"""
build_benchmark_6k.py

Build a 6000-sample (60 s at 100 Hz) version of the benchmark dataset for
EQTransformer evaluation.  Same set of traces as benchmark_waveforms.hdf5
but with a wider window so EQT sees real seismic context instead of zero-padding.

Window placement
----------------
P is placed at sample P_OFFSET (1500 = 15 s into the window).  When the raw
waveform doesn't provide enough context:

  pad_left  (mainly STEAD / TXED — P arrives early in the raw window):
      Tile the available pre-P noise segment to fill missing left samples.

  pad_right (mainly PISDL — short raw recordings):
      Tile the post-event tail to fill missing right samples.

Outputs
-------
  notebooks/benchmark_waveforms_6k.hdf5      — (3, 6000) float32 per trace
  notebooks/benchmark_waveforms_6k_index.csv — same columns as the 3k index,
                                               updated p_in_window / s_in_window

Run from repo root:
    conda activate surface
    python scripts/build_benchmark_6k.py
"""

import sys, re, os
import numpy as np
import pandas as pd
import h5py
import scipy.signal
from math import gcd
from pathlib import Path
from tqdm import tqdm

REPO_ROOT       = Path(__file__).parent.parent.resolve()
NB_DIR          = REPO_ROOT / "notebooks"
SEISBENCH_CACHE = Path(os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))) / "datasets"

MANIFEST_PATH   = NB_DIR / "benchmark_manifest.csv"
INDEX_PATH      = NB_DIR / "benchmark_waveforms_index.csv"
OUTPUT_HDF5     = NB_DIR / "benchmark_waveforms_6k.hdf5"
OUTPUT_INDEX    = NB_DIR / "benchmark_waveforms_6k_index.csv"

TARGET_SR      = 100
WINDOW_SAMPLES = 6000   # 60 s at 100 Hz
P_OFFSET       = 1500   # target: P at sample 1500 (15 s in)

LOCAL_HDF5_DATASETS    = {"stead", "instancecounts", "pnw", "txed", "mlaapde", "ethz", "pisdl"}
SEISBENCH_API_DATASETS = {"ceed", "vcseis", "aq2009gm", "cwa"}
SKIP_DATASETS          = {"obst2024"}
NATIVE_SR = {
    "stead": 100.0, "instancecounts": 100.0, "pnw": 100.0,
    "txed":  100.0, "mlaapde":  40.0, "ethz": 100.0, "pisdl": 100.0,
    "ceed":  100.0, "vcseis":  100.0, "aq2009gm": 100.0, "cwa": 100.0,
}


# ── Waveform loading (verbatim from notebook 05) ───────────────────────────────

def parse_trace_name(trace_name):
    m = re.match(r"bucket(\d+)\$(\d+)", trace_name)
    if not m:
        raise ValueError(f"Unrecognised trace_name: {trace_name!r}")
    return int(m.group(1)), int(m.group(2))


def load_waveform_local(ds_name, trace_name, source_month=None):
    bucket_id, row_id = parse_trace_name(trace_name)
    if ds_name == "mlaapde":
        month_str = str(int(float(source_month)))
        hdf5_path = SEISBENCH_CACHE / "mlaapde" / f"waveforms_{month_str}.hdf5"
    else:
        hdf5_path = SEISBENCH_CACHE / ds_name / "waveforms.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        waveform = f["data"][f"bucket{bucket_id}"][row_id]
    return np.array(waveform, dtype=np.float32)


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


def load_waveform_seisbench(ds_name, trace_name, sb_index):
    hdf5_path = sb_index[ds_name].get(trace_name)
    if hdf5_path is None:
        raise KeyError(f"{trace_name!r} not in {ds_name} index")
    with h5py.File(hdf5_path, "r") as f:
        if ds_name == "ceed":
            waveform = f[trace_name][:]
        elif ds_name in {"vcseis", "aq2009gm"}:
            bucket_id, row_id = parse_trace_name(trace_name)
            waveform = f["data"][f"bucket{bucket_id}"][row_id]
        elif ds_name == "cwa":
            waveform = f["data"][trace_name][:]
        else:
            raise ValueError(ds_name)
    waveform = np.array(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    if waveform.shape[0] < 3:
        waveform = np.pad(waveform, ((0, 3 - waveform.shape[0]), (0, 0)))
    return waveform


def resample_waveform(data, orig_sr, target_sr=TARGET_SR):
    if abs(orig_sr - target_sr) < 0.5:
        return data.astype(np.float32)
    up   = int(round(target_sr))
    down = int(round(orig_sr))
    g    = gcd(up, down); up //= g; down //= g
    return scipy.signal.resample_poly(data, up, down, axis=-1,
                                      padtype="line").astype(np.float32)


def rescale_sample(sample, orig_sr, target_sr=TARGET_SR):
    return int(round(sample * target_sr / orig_sr))


# ── Window extraction with noise-tiling padding ────────────────────────────────

def tile_to_length(segment, target_len):
    """Tile `segment` (n_comp, k) along time axis to reach exactly target_len."""
    if segment.shape[-1] == 0:
        return np.zeros((segment.shape[0], target_len), dtype=np.float32)
    reps = int(np.ceil(target_len / segment.shape[-1]))
    tiled = np.tile(segment, (1, reps))
    return tiled[:, :target_len]


def extract_window_6k(waveform, p_100, s_100=None):
    """
    Extract a WINDOW_SAMPLES window from `waveform` with P at P_OFFSET.
    If the raw waveform is too short on either side, pad with tiled noise.

    Returns (window, p_in_window, s_in_window) where s_in_window = -1 if absent.
    """
    n = waveform.shape[-1]

    win_start = p_100 - P_OFFSET
    win_end   = win_start + WINDOW_SAMPLES

    pad_left  = max(0, -win_start)
    pad_right = max(0, win_end - n)

    if pad_left > 0 or pad_right > 0:
        # Build padded waveform with real noise tiles
        parts = []

        if pad_left > 0:
            # Pre-noise: tile whatever is available before the window
            avail_start = max(0, win_start)   # = 0 when pad_left > 0
            pre_noise   = waveform[:, avail_start:p_100]
            parts.append(tile_to_length(pre_noise, pad_left))

        # Middle: the available overlap between [win_start, win_end] and [0, n]
        seg_start = max(0, win_start)
        seg_end   = min(n, win_end)
        parts.append(waveform[:, seg_start:seg_end])

        if pad_right > 0:
            # Post-noise: tile whatever is available after the window
            avail_end  = min(n, win_end)   # = n when pad_right > 0
            post_noise = waveform[:, p_100:avail_end]
            parts.append(tile_to_length(post_noise, pad_right))

        padded    = np.concatenate(parts, axis=-1)
        window    = padded[:, :WINDOW_SAMPLES].astype(np.float32)
        p_in_win  = p_100 - win_start   # = P_OFFSET when pad_left == 0; else < P_OFFSET
    else:
        window   = waveform[:, win_start:win_end].astype(np.float32)
        p_in_win = P_OFFSET

    # Compute S position in the window
    s_in_win = -1
    if s_100 is not None and np.isfinite(s_100):
        s_candidate = int(round(s_100)) - win_start
        if 0 <= s_candidate < WINDOW_SAMPLES:
            s_in_win = s_candidate

    assert window.shape == (waveform.shape[0], WINDOW_SAMPLES), \
        f"Window shape mismatch: {window.shape}"

    return window, p_in_win, s_in_win


# ── Main ───────────────────────────────────────────────────────────────────────

print("Loading benchmark manifest and index...")
manifest = pd.read_csv(MANIFEST_PATH)
ok_index = pd.read_csv(INDEX_PATH)
ok_index = ok_index[ok_index["status"] == "ok"]
ok_traces = set(ok_index["trace_name"])

# Filter to ok traces only (same as original benchmark)
mf = manifest[manifest["trace_name"].isin(ok_traces)].copy().reset_index(drop=True)
print(f"  {len(mf):,} benchmark traces")

print("Building SeisBench index...")
sb_index = build_sb_index()
print("  Done.\n")

skip_log   = []
success_log = []

hdf5_mode = "a" if OUTPUT_HDF5.exists() else "w"
print(f"Opening {OUTPUT_HDF5} in '{hdf5_mode}' mode")

with h5py.File(OUTPUT_HDF5, hdf5_mode) as hf:
    wf_grp = hf.require_group("waveforms")
    already = len(wf_grp)
    print(f"Traces already written: {already:,}\n")

    for _, row in tqdm(mf.iterrows(), total=len(mf), desc="Building 6k benchmark"):
        tname   = row["trace_name"]
        ds_name = row["dataset"]

        if ds_name in SKIP_DATASETS:
            skip_log.append({"trace_name": tname, "dataset": ds_name,
                             "status": "skip", "skip_reason": "dataset_skipped"})
            continue

        if tname in wf_grp:
            continue  # already written — safe to re-run

        has_p = bool(row.get("has_p_pick", pd.notna(row["p_arrival_sample"])))
        has_s = bool(row.get("has_s_pick", False))

        if not has_p:
            skip_log.append({"trace_name": tname, "dataset": ds_name,
                             "status": "skip", "skip_reason": "no_p_pick"})
            continue

        # ── Load raw waveform ──────────────────────────────────────────────
        try:
            if ds_name in LOCAL_HDF5_DATASETS:
                waveform = load_waveform_local(ds_name, tname, row.get("source_month"))
            elif ds_name in SEISBENCH_API_DATASETS:
                waveform = load_waveform_seisbench(ds_name, tname, sb_index)
            else:
                skip_log.append({"trace_name": tname, "dataset": ds_name,
                                 "status": "skip", "skip_reason": "no_loader"})
                continue
        except Exception as e:
            skip_log.append({"trace_name": tname, "dataset": ds_name,
                             "status": "error", "skip_reason": f"load_error:{str(e)[:60]}"})
            continue

        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        if waveform.shape[0] < 3:
            waveform = np.pad(waveform, ((0, 3 - waveform.shape[0]), (0, 0)))

        # ── Resample to 100 Hz ─────────────────────────────────────────────
        native_sr = NATIVE_SR.get(ds_name, 100.0)
        try:
            waveform = resample_waveform(waveform, native_sr, TARGET_SR)
        except Exception as e:
            skip_log.append({"trace_name": tname, "dataset": ds_name,
                             "status": "error", "skip_reason": f"resample_error:{str(e)[:40]}"})
            continue

        # ── Rescale pick samples to 100 Hz ────────────────────────────────
        p_raw = float(row["p_arrival_sample"])
        p_100 = rescale_sample(p_raw, native_sr, TARGET_SR)

        s_100 = None
        if has_s and pd.notna(row.get("s_arrival_sample")):
            s_100 = rescale_sample(float(row["s_arrival_sample"]), native_sr, TARGET_SR)

        # ── Extract 6000-sample window ─────────────────────────────────────
        try:
            window, p_in_win, s_in_win = extract_window_6k(waveform, p_100, s_100)
        except Exception as e:
            skip_log.append({"trace_name": tname, "dataset": ds_name,
                             "status": "error", "skip_reason": f"window_error:{str(e)[:60]}"})
            continue

        # ── Write to HDF5 ──────────────────────────────────────────────────
        wf_grp.create_dataset(tname, data=window, compression="gzip",
                              compression_opts=4)

        # Copy metadata from the original index row
        orig = ok_index[ok_index["trace_name"] == tname].iloc[0]
        success_log.append({
            **{c: orig[c] for c in ok_index.columns
               if c not in ("p_in_window", "s_in_window", "status", "skip_reason")},
            "p_in_window": p_in_win,
            "s_in_window": s_in_win,
            "status": "ok",
            "skip_reason": "",
        })

print(f"\nDone. Written: {len(success_log):,}  Skipped/error: {len(skip_log):,}")

# ── Save index ─────────────────────────────────────────────────────────────────

skip_df   = pd.DataFrame(skip_log)
success_df = pd.DataFrame(success_log)
index_df  = pd.concat([success_df, skip_df], ignore_index=True)
index_df.to_csv(OUTPUT_INDEX, index=False)
print(f"Index saved → {OUTPUT_INDEX}")

# ── Quick summary ──────────────────────────────────────────────────────────────

print(f"\n{'Dataset':<20} {'Written':>8}")
print("-" * 30)
for ds, grp in success_df.groupby("dataset"):
    print(f"  {ds:<18} {len(grp):>8,}")
print(f"\n  {'TOTAL':<18} {len(success_df):>8,}")
