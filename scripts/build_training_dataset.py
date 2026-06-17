#!/usr/bin/env python3
"""
build_training_dataset.py

Creates stratified train/val/test manifest CSVs for fine-tuning PhaseNet (jma_wc)
as a generalised global phase picker.  P-only for teleseismic traces.

All data loaded from local SeisBench cache — no downloading required.

Output manifests (written to --output-dir, default: data/manifests/):
  train.csv, val.csv, test.csv   — one row per training trace
  composition_summary.csv        — per-dataset / per-distance-bin counts

Manifest columns:
  dataset_name        source dataset key (matches DATASET_CONFIGS)
  trace_name          SeisBench trace_name (primary lookup key)
  chunk               chunk tag for chunked datasets (MLAAPDE / CWA), else ""
  p_arrival_sample    P-wave pick sample index
  s_arrival_sample    S-wave pick (NaN for teleseismic — P-only policy)
  distance_km         epicentral distance
  distance_bin        local | regional | teleseismic | unknown
  p_col               source column used for P pick
  s_col               source column used for S pick (empty if none)

Usage:
  python scripts/build_training_dataset.py
  python scripts/build_training_dataset.py --output-dir data/manifests --seed 42
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEISBENCH_CACHE = "/data/wsd04/ak287/.seisbench"
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

# ──────────────────────────────────────────────────────────────────────────────
# Distance helpers
# ──────────────────────────────────────────────────────────────────────────────

LOCAL_KM      = 150
REGIONAL_KM   = 1500


def distance_bin(dist_km, default="unknown"):
    if pd.isna(dist_km) or dist_km < 0:
        return default
    if dist_km < LOCAL_KM:
        return "local"
    if dist_km < REGIONAL_KM:
        return "regional"
    return "teleseismic"


def to_km(series, unit):
    s = pd.to_numeric(series, errors="coerce")
    if unit == "deg":
        return s * 111.2
    if unit == "m":
        return s / 1000.0
    return s  # already km


# ──────────────────────────────────────────────────────────────────────────────
# Pick-column priority lists
# ──────────────────────────────────────────────────────────────────────────────

P_PRIORITY = [
    "trace_p_arrival_sample",   # STEAD / CEED / VCSEIS / LenDB / TXED / CWA
    "trace_P_arrival_sample",   # InstanceCounts / GEOFON / PNW / CREW / PiSDL
    "trace_Pg_arrival_sample",  # ETHZ (primary) / MLAAPDE regional Pg
    "trace_Pn_arrival_sample",  # MLAAPDE mantle Pn
    "trace_P1_arrival_sample",  # ETHZ fallback
    "trace_PmP_arrival_sample", # ETHZ reflected P
]

S_PRIORITY = [
    "trace_s_arrival_sample",
    "trace_S_arrival_sample",
    "trace_Sg_arrival_sample",
    "trace_Sn_arrival_sample",
    "trace_S1_arrival_sample",
]


def best_col(columns, priority):
    """Return the first column from priority that exists — used for s_col label only."""
    for c in priority:
        if c in columns:
            return c
    return None


def coalesce_picks(meta, priority):
    """
    Return a Series of pick sample values by taking the first non-null value
    across all columns in priority that exist in meta.  This handles datasets
    like ETHZ that spread picks across trace_Pg / trace_P1 / trace_P columns.
    """
    present = [c for c in priority if c in meta.columns]
    if not present:
        return pd.Series(np.nan, index=meta.index), None
    # combine: for each row take the first non-NaN value in priority order
    result = pd.to_numeric(meta[present[0]], errors="coerce")
    for c in present[1:]:
        result = result.combine_first(pd.to_numeric(meta[c], errors="coerce"))
    label = present[0]  # report the highest-priority col as the label
    return result, label


# ──────────────────────────────────────────────────────────────────────────────
# Chunked-dataset manual loaders (avoid SeisBench "partial instance" errors)
# ──────────────────────────────────────────────────────────────────────────────

def _load_chunked_meta(ds_path, prefix="metadata_"):
    """
    Combine all complete (non-.partial) metadata CSVs in a chunked SeisBench
    dataset directory.  Returns a DataFrame with an added 'chunk' column.
    prefix : filename prefix before the chunk tag (e.g. "metadata_" for MLAAPDE,
             "metadata" for AQ2009GM).
    """
    path = Path(ds_path)
    csvs = sorted(
        f for f in path.iterdir()
        if f.name.startswith(prefix) and f.suffix == ".csv" and not f.name.endswith(".partial")
    )
    if not csvs:
        raise FileNotFoundError(f"No complete metadata CSVs in {path} (prefix='{prefix}')")

    frames = []
    for csv in csvs:
        chunk_tag = csv.stem.replace(prefix, "")
        df = pd.read_csv(csv, low_memory=False)
        df["chunk"] = chunk_tag
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


MLAAPDE_PATH      = Path(SEISBENCH_CACHE) / "datasets" / "mlaapde"
CWA_PATH          = Path(SEISBENCH_CACHE) / "datasets" / "cwa"
PISDL_PATH        = Path(SEISBENCH_CACHE) / "datasets" / "pisdl"
AQ2009GM_PATH     = Path(SEISBENCH_CACHE) / "datasets" / "aq2009gm"
MEIER2019JGR_PATH = Path(SEISBENCH_CACHE) / "datasets" / "meier2019jgr"
ROSS2018GPD_PATH  = Path(SEISBENCH_CACHE) / "datasets" / "ross2018gpd"
OBS_PATH          = Path(SEISBENCH_CACHE) / "datasets" / "obs"


def _load_mlaapde():
    return _load_chunked_meta(MLAAPDE_PATH)


def _load_cwa():
    return _load_chunked_meta(CWA_PATH)


def _load_aq2009gm():
    return _load_chunked_meta(AQ2009GM_PATH, prefix="metadata")


def _load_pisdl():
    ds = sbd.WaveformDataset(str(PISDL_PATH))
    meta = ds.metadata.copy()
    meta["chunk"] = ""
    return meta


def _load_meier2019jgr():
    ds = sbd.WaveformDataset(str(MEIER2019JGR_PATH))
    meta = ds.metadata.copy()
    meta["chunk"] = ""
    return meta


def _load_ross2018gpd():
    ds = sbd.WaveformDataset(str(ROSS2018GPD_PATH))
    meta = ds.metadata.copy()
    meta["chunk"] = ""
    return meta


def _load_obs():
    return _load_chunked_meta(OBS_PATH, prefix="metadata")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────
# Each entry:
#   cls          : SeisBench class (None for manually-loaded datasets)
#   meta_fn      : callable() -> DataFrame  (used when cls is None)
#   dist_col     : column name for distance, or None
#   dist_unit    : "km" | "deg" | "m"
#   cap          : max traces to sample (combined across all splits)
#   default_bin  : fallback distance bin when dist_col is absent / NaN
#   use_s        : keep S picks?  False for teleseismic-only datasets

DATASET_CONFIGS = [
    # ── High-priority teleseismic source ──────────────────────────────────────
    dict(name="geofon",
         cls=sbd.GEOFON,          meta_fn=None,
         dist_col=None,           dist_unit="km",
         cap=150_000,             default_bin="teleseismic",  use_s=False),

    # ── Large generalist sources ──────────────────────────────────────────────
    dict(name="stead",
         cls=sbd.STEAD,           meta_fn=None,
         dist_col="source_distance_km", dist_unit="km",
         cap=100_000,             default_bin=None,           use_s=True),

    dict(name="ceed",
         cls=sbd.CEED,            meta_fn=None,
         dist_col="path_ep_distance_km", dist_unit="km",
         cap=100_000,             default_bin=None,           use_s=True),

    dict(name="instancecounts",
         cls=sbd.InstanceCounts,  meta_fn=None,
         dist_col="path_ep_distance_km", dist_unit="km",
         cap=100_000,             default_bin=None,           use_s=True),

    # ── Regional diversity ────────────────────────────────────────────────────
    dict(name="mlaapde",
         cls=None,                meta_fn=_load_mlaapde,
         dist_col="path_ep_distance_km", dist_unit="km",
         cap=80_000,              default_bin=None,           use_s=True),

    dict(name="ethz",
         cls=sbd.ETHZ,            meta_fn=None,
         dist_col=None,           dist_unit="km",
         cap=60_000,              default_bin="local",        use_s=True),

    dict(name="crew",
         cls=sbd.CREW,            meta_fn=None,
         dist_col="path_epicentral_distance_deg", dist_unit="deg",
         cap=30_000,              default_bin=None,           use_s=True),

    dict(name="cwa",
         cls=None,                meta_fn=_load_cwa,
         dist_col="path_ep_distance_km", dist_unit="km",
         cap=30_000,              default_bin=None,           use_s=True),

    dict(name="iquique",
         cls=sbd.Iquique,         meta_fn=None,
         dist_col=None,           dist_unit="km",
         cap=13_400,              default_bin="regional",     use_s=True),

    # ── Local / induced sources ───────────────────────────────────────────────
    dict(name="txed",
         cls=sbd.TXED,            meta_fn=None,
         dist_col=None,           dist_unit="km",
         cap=40_000,              default_bin="local",        use_s=True),

    dict(name="pnw",
         cls=sbd.PNW,             meta_fn=None,
         dist_col=None,           dist_unit="km",
         cap=40_000,              default_bin="regional",     use_s=True),

    dict(name="lendb",
         cls=sbd.LenDB,           meta_fn=None,
         dist_col="path_ep_distance_km", dist_unit="km",
         cap=40_000,              default_bin=None,           use_s=False),

    dict(name="pisdl",
         cls=None,                meta_fn=_load_pisdl,
         dist_col=None,           dist_unit="km",
         cap=10_000,              default_bin="local",        use_s=True),

    # ── Volcanic / exotic ─────────────────────────────────────────────────────
    dict(name="vcseis",
         cls=sbd.VCSEIS,          meta_fn=None,
         dist_col="station_epicentral_distance_m", dist_unit="m",
         cap=30_000,              default_bin="local",        use_s=True),

    # ── Global additions (benchmark coverage + OBS diversity) ─────────────────
    dict(name="aq2009gm",
         cls=None,                meta_fn=_load_aq2009gm,
         dist_col="path_ep_distance_km", dist_unit="km",
         cap=60_000,              default_bin="local",        use_s=True),

    dict(name="obst2024",
         cls=sbd.OBST2024,        meta_fn=None,
         dist_col="source_distance_deg", dist_unit="deg",
         cap=60_000,              default_bin="regional",     use_s=True),

    dict(name="scedc",
         cls=sbd.SCEDC,           meta_fn=None,
         dist_col="station_epicentral_distance", dist_unit="km",
         cap=60_000,              default_bin="local",        use_s=True),

    # ── New diversity datasets ────────────────────────────────────────────────
    # meier2019jgr: global catalog, P-only, hypocentral distances 4-12000 km
    dict(name="meier2019jgr",
         cls=None,                meta_fn=_load_meier2019jgr,
         dist_col="path_hyp_distance_km", dist_unit="km",
         cap=150_000,             default_bin="regional",     use_s=False),

    # ross2018gpd: Southern California, 4.77M traces, P+S, local seismicity
    dict(name="ross2018gpd",
         cls=None,                meta_fn=_load_ross2018gpd,
         dist_col=None,           dist_unit="km",
         cap=200_000,             default_bin="local",        use_s=True),

    # obs: ocean-bottom seismometers, Pg/Sg phases, unique sensor environment
    dict(name="obs",
         cls=None,                meta_fn=_load_obs,
         dist_col=None,           dist_unit="km",
         cap=100_000,             default_bin="local",        use_s=True),
]

# Target distance fractions for the TRAINING split.
# Teleseismic raised to 0.25 (from 0.20) to address the biggest weakness
# in the global fine-tune.  Local reduced to 0.40 because ross2018gpd adds
# abundant local data — no need to oversample that bin further.
TARGET_FRACTIONS = {
    "local":       0.40,
    "regional":    0.25,
    "teleseismic": 0.25,
    "unknown":     0.10,
}

# ──────────────────────────────────────────────────────────────────────────────
# SeisBench split-column normalisation
# ──────────────────────────────────────────────────────────────────────────────

SEISBENCH_TRAIN = {"train"}
SEISBENCH_VAL   = {"dev", "val", "valid", "development", "eval"}
SEISBENCH_TEST  = {"test"}


def normalise_split(s):
    if not isinstance(s, str):
        return ""
    v = s.strip().lower()
    if v in SEISBENCH_TRAIN:
        return "train"
    if v in SEISBENCH_VAL:
        return "val"
    if v in SEISBENCH_TEST:
        return "test"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset processing
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset(cfg, rng, benchmark_exclude=None, s_balanced=False):
    """
    Load one dataset, filter for valid P picks, compute distances, apply cap.
    benchmark_exclude : set of trace_name strings to exclude (benchmark traces).
    s_balanced        : if True and use_s=True, additionally require a valid S pick.
    Returns a standardised DataFrame or None on failure.
    """
    name = cfg["name"]
    print(f"\n  [{name}]")

    # ── load metadata ──────────────────────────────────────────────────────────
    try:
        if cfg["meta_fn"] is not None:
            meta = cfg["meta_fn"]()
        else:
            ds = cfg["cls"]()
            meta = ds.metadata.copy()
            if "chunk" not in meta.columns:
                meta["chunk"] = ""
    except Exception as exc:
        print(f"    SKIP — failed to load: {exc}")
        return None

    print(f"    loaded {len(meta):,} total rows")

    # ── pick columns ──────────────────────────────────────────────────────────
    cols = meta.columns.tolist()
    p_vals, p_col = coalesce_picks(meta, P_PRIORITY)
    s_col = best_col(cols, S_PRIORITY) if cfg["use_s"] else None

    if p_col is None:
        print(f"    SKIP — no recognisable P-pick column (have: {[c for c in cols if 'arrival' in c.lower()][:6]})")
        return None

    # ── filter to valid P picks ───────────────────────────────────────────────
    keep = p_vals.notna() & (p_vals >= 0)
    meta = meta.loc[keep].copy()
    p_vals = p_vals.loc[keep]

    if len(meta) == 0:
        print(f"    SKIP — 0 valid P picks across P-type columns")
        return None

    print(f"    {len(meta):,} with valid P pick  (p_col=coalesced/{p_col}, s_col={s_col})")

    # ── S-balanced mode: require S pick for use_s=True datasets ──────────────
    if s_balanced and cfg["use_s"] and s_col and s_col in meta.columns:
        s_vals_pre = pd.to_numeric(meta[s_col], errors="coerce")
        s_mask = s_vals_pre.notna() & (s_vals_pre >= 0)
        meta   = meta.loc[s_mask].copy()
        p_vals = p_vals.loc[s_mask]
        print(f"    {len(meta):,} after S-pick filter (s_balanced=True)")

    # ── exclude benchmark traces ──────────────────────────────────────────────
    if benchmark_exclude and "trace_name" in meta.columns:
        before = len(meta)
        keep_bm = ~meta["trace_name"].isin(benchmark_exclude)
        meta   = meta.loc[keep_bm].copy()
        p_vals = p_vals.loc[keep_bm]
        n_removed = before - len(meta)
        if n_removed:
            print(f"    excluded {n_removed:,} benchmark traces → {len(meta):,} remaining")

    # ── distance ──────────────────────────────────────────────────────────────
    if cfg["dist_col"] and cfg["dist_col"] in meta.columns:
        dist_km = to_km(meta[cfg["dist_col"]], cfg["dist_unit"])
    else:
        dist_km = pd.Series(np.nan, index=meta.index)

    def _bin(d):
        return distance_bin(d, default=cfg["default_bin"] or "unknown")

    dist_bin = dist_km.apply(_bin)

    # ── per-dataset cap (stratified: sample proportionally across bins) ───────
    cap = cfg["cap"]
    if len(meta) > cap:
        # build per-bin sample counts proportional to natural bin frequencies
        bin_counts = dist_bin.value_counts()
        sampled_idx = []
        for b, count in bin_counts.items():
            b_idx = dist_bin[dist_bin == b].index
            n = max(1, int(round(cap * count / len(meta))))
            n = min(n, len(b_idx))
            sampled_idx.extend(rng.choice(b_idx, size=n, replace=False).tolist())
        # if rounding left us short, top up from the largest bin
        if len(sampled_idx) < cap:
            remaining = list(set(meta.index) - set(sampled_idx))
            extra = min(cap - len(sampled_idx), len(remaining))
            sampled_idx.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        meta     = meta.loc[sampled_idx]
        p_vals   = p_vals.loc[sampled_idx]
        dist_km  = dist_km.loc[sampled_idx]
        dist_bin = dist_bin.loc[sampled_idx]

    print(f"    {len(meta):,} after cap={cap:,} | bins: {dist_bin.value_counts().to_dict()}")

    # ── assemble output ───────────────────────────────────────────────────────
    s_vals = (
        pd.to_numeric(meta[s_col], errors="coerce")
        if s_col and s_col in meta.columns
        else pd.Series(np.nan, index=meta.index)
    )

    # P-only policy: null S for teleseismic rows
    tele_mask = dist_bin == "teleseismic"
    s_vals = s_vals.copy()
    s_vals[tele_mask] = np.nan

    out = pd.DataFrame({
        "dataset_name":      name,
        "trace_name":        meta["trace_name"].values if "trace_name" in meta.columns
                             else [f"{name}_{i}" for i in range(len(meta))],
        "chunk":             meta["chunk"].values if "chunk" in meta.columns else "",
        "p_arrival_sample":  p_vals.values,
        "s_arrival_sample":  s_vals.values,
        "distance_km":       dist_km.values,
        "distance_bin":      dist_bin.values,
        "p_col":             p_col,
        "s_col":             s_col or "",
        "orig_split":        (
            meta["split"].map(normalise_split).values
            if "split" in meta.columns
            else np.full(len(meta), "")
        ),
    })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Distance stratification (training set only)
# ──────────────────────────────────────────────────────────────────────────────

def stratify_training(train_df, rng):
    """
    Resample train_df so the distance-bin distribution matches TARGET_FRACTIONS.
    Bins below their target fraction are kept whole; over-represented bins are
    downsampled.  The total size is determined by the smallest-ratio bin.
    """
    bin_counts = train_df["distance_bin"].value_counts()
    total_available = len(train_df)

    # compute how many traces each bin *could* support given its target fraction
    max_total_per_bin = {}
    for b, frac in TARGET_FRACTIONS.items():
        if b not in bin_counts or frac == 0:
            max_total_per_bin[b] = 0
            continue
        # if this bin has `n` traces and its target is `frac`, the implied total is n/frac
        max_total_per_bin[b] = int(bin_counts[b] / frac)

    target_total = min(v for v in max_total_per_bin.values() if v > 0)
    target_total = min(target_total, total_available)

    sampled = []
    for b, frac in TARGET_FRACTIONS.items():
        target_n = int(round(target_total * frac))
        available = train_df[train_df["distance_bin"] == b]
        if len(available) == 0:
            continue
        if len(available) <= target_n:
            sampled.append(available)  # keep all (bin is under-represented)
        else:
            sampled.append(available.sample(n=target_n, random_state=rng))

    result = pd.concat(sampled).sample(frac=1.0, random_state=rng)  # shuffle
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Split assignment
# ──────────────────────────────────────────────────────────────────────────────

def assign_splits(df, rng, val_frac=0.10, test_frac=0.10):
    """
    Use existing SeisBench splits where available; apply random 80/10/10 to
    rows that have no predefined split.
    """
    split = pd.Series("", index=df.index, dtype=str)

    has_orig = df["orig_split"].str.len() > 0
    split[has_orig & (df["orig_split"] == "train")] = "train"
    split[has_orig & (df["orig_split"] == "val")]   = "val"
    split[has_orig & (df["orig_split"] == "test")]  = "test"

    no_split = df.index[split == ""]
    if len(no_split):
        n = len(no_split)
        perm = rng.permutation(n)
        n_val  = int(val_frac  * n)
        n_test = int(test_frac * n)
        val_pos  = no_split[perm[:n_val]]
        test_pos = no_split[perm[n_val:n_val + n_test]]
        train_pos = no_split[perm[n_val + n_test:]]
        split[train_pos] = "train"
        split[val_pos]   = "val"
        split[test_pos]  = "test"

    return split


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(train_df, val_df, test_df):
    def _tbl(df, label):
        print(f"\n  {label} ({len(df):,} traces):")
        print(f"  {'dataset':<20} {'local':>8} {'regional':>9} {'teleseismic':>12} {'unknown':>8} {'total':>7}")
        print(f"  {'-'*20} {'-'*8} {'-'*9} {'-'*12} {'-'*8} {'-'*7}")
        for ds in sorted(df["dataset_name"].unique()):
            sub = df[df["dataset_name"] == ds]
            bc  = sub["distance_bin"].value_counts()
            print(f"  {ds:<20} {bc.get('local',0):>8,} {bc.get('regional',0):>9,} "
                  f"{bc.get('teleseismic',0):>12,} {bc.get('unknown',0):>8,} {len(sub):>7,}")
        bc = df["distance_bin"].value_counts()
        print(f"  {'TOTAL':<20} {bc.get('local',0):>8,} {bc.get('regional',0):>9,} "
              f"{bc.get('teleseismic',0):>12,} {bc.get('unknown',0):>8,} {len(df):>7,}")
        s_frac = df["s_arrival_sample"].notna().mean()
        print(f"  S-pick coverage: {s_frac:.1%}")

    _tbl(train_df, "TRAIN")
    _tbl(val_df,   "VAL")
    _tbl(test_df,  "TEST")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_CSV = Path(__file__).parent.parent / "notebooks" / "benchmark_manifest.csv"


def load_benchmark_exclusions():
    """
    Load (dataset_name, trace_name) pairs from the benchmark manifest so they
    can be excluded from training/validation data.
    Returns a dict: {dataset_name: set_of_trace_names}.
    """
    if not BENCHMARK_CSV.exists():
        print(f"  WARNING: benchmark manifest not found at {BENCHMARK_CSV} — no exclusions applied")
        return {}
    bm = pd.read_csv(BENCHMARK_CSV, usecols=["dataset", "trace_name"])
    exclusions = {}
    for ds, group in bm.groupby("dataset"):
        exclusions[ds] = set(group["trace_name"])
    total = sum(len(v) for v in exclusions.values())
    print(f"  Loaded {total:,} benchmark traces to exclude across {len(exclusions)} datasets")
    return exclusions


def main(output_dir, seed, s_balanced=False):
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Building PhaseNet training manifests")
    print(f"  SeisBench cache : {SEISBENCH_CACHE}")
    print(f"  Output dir      : {out_path.resolve()}")
    print(f"  Random seed     : {seed}")
    print(f"  S-balanced mode : {s_balanced}")
    print("=" * 70)

    # ── load benchmark exclusions ────────────────────────────────────────────
    benchmark_exclusions = load_benchmark_exclusions()

    # ── process all datasets ─────────────────────────────────────────────────
    frames = []
    for cfg in DATASET_CONFIGS:
        exclude = benchmark_exclusions.get(cfg["name"], set())
        df = process_dataset(cfg, rng, benchmark_exclude=exclude, s_balanced=s_balanced)
        if df is not None:
            frames.append(df)

    if not frames:
        sys.exit("ERROR: no datasets loaded — check SeisBench cache path")

    all_data = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined pool : {len(all_data):,} traces before split assignment")

    # ── assign train / val / test ────────────────────────────────────────────
    all_data["split"] = assign_splits(all_data, rng)

    train_df = all_data[all_data["split"] == "train"].copy()
    val_df   = all_data[all_data["split"] == "val"].copy()
    test_df  = all_data[all_data["split"] == "test"].copy()

    # ── distance-stratify training set ───────────────────────────────────────
    print(f"\n  Training set before stratification : {len(train_df):,}")
    train_df = stratify_training(train_df, rng)
    print(f"  Training set after stratification  : {len(train_df):,}")

    # ── drop working column ──────────────────────────────────────────────────
    for df in (train_df, val_df, test_df):
        df.drop(columns=["orig_split"], errors="ignore", inplace=True)

    # ── write manifests ──────────────────────────────────────────────────────
    KEEP_COLS = [
        "dataset_name", "trace_name", "chunk",
        "p_arrival_sample", "s_arrival_sample",
        "distance_km", "distance_bin",
        "p_col", "s_col",
    ]
    train_df[KEEP_COLS].to_csv(out_path / "train.csv", index=False)
    val_df[KEEP_COLS].to_csv(out_path / "val.csv",   index=False)
    test_df[KEEP_COLS].to_csv(out_path / "test.csv", index=False)

    # ── composition summary ──────────────────────────────────────────────────
    rows = []
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for ds in df["dataset_name"].unique():
            sub = df[df["dataset_name"] == ds]
            for b in ["local", "regional", "teleseismic", "unknown"]:
                rows.append({
                    "split": split_name, "dataset": ds, "distance_bin": b,
                    "n_traces": int((sub["distance_bin"] == b).sum()),
                    "n_with_s":  int((sub[sub["distance_bin"] == b]["s_arrival_sample"].notna()).sum()),
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path / "composition_summary.csv", index=False)

    print_summary(train_df, val_df, test_df)

    print(f"\n  Manifests written to {out_path.resolve()}/")
    print("  train.csv | val.csv | test.csv | composition_summary.csv")
    print("=" * 70)
    print("\nNote: ManifestDataset (scripts/manifest_dataset.py) loads waveforms")
    print("      from these manifests during training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build stratified PhaseNet training manifests from SeisBench cache"
    )
    parser.add_argument("--output-dir", default="data/manifests",
                        help="Directory to write manifest CSVs (default: data/manifests)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--s-balanced", action="store_true",
                        help="Require valid S pick for datasets with use_s=True (boosts S-recall training signal)")
    args = parser.parse_args()
    main(args.output_dir, args.seed, s_balanced=args.s_balanced)
