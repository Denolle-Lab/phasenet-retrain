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
  source_origin_time, source_latitude_deg, source_longitude_deg
                      event fingerprint (NaN when the source has none), so the
                      held-out-sequence and 2016/2021 year hold-out can be
                      re-verified on the manifest itself (2026-09-07)
  independence_unverified
                      True when the row has no testable origin and was kept
                      under the bundle's allow_unknown quarantine policy
                      (2026-09-11, #33A); always False under the default policy

Exclusions applied, in order: benchmark traces, the exclusion bundle
(data/exclusions/bundle.json, scripts/exclusion_bundle.py, #33A: the listed
(dataset, chunk, trace_name) rows of data/exclusions/heldout_sequences.csv,
origins inside a held-out window, 2016/2021 origins, and quarantine of rows
whose origin time or location is missing; the build refuses to run without a
valid bundle), label-error flagged traces, benchmark events under another
trace_name. provenance.json beside the manifests records the bundle hash, git
commit, per-source counts and the manifest key hashes.

Corpus profiles (#40A): --profile <name> selects a profile from
configs/corpus_profiles.yaml (which sources, their cap and use_s, an optional
manual-status filter per source, the training distance fractions, a maximum
epicentral distance, whether P-only sources are allowed, which sources are
skipped). Without --profile the module defaults below apply, which the
`legacy_v2` profile reproduces exactly. The profile name and hash are written
to provenance.json and heldout_removal_report.csv.

Usage:
  python scripts/build_training_dataset.py
  python scripts/build_training_dataset.py --output-dir data/manifests --seed 42
  python scripts/build_training_dataset.py --profile t0_pilot --output-dir data/manifests_t0
"""

import argparse
import hashlib
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waveform_contract import METADATA_FIELDS

import heldout_sequences as hs  # held-out external sequences + 2016/2021 year hold-out (2026-09-07)
import exclusion_bundle as eb   # versioned exclusion bundle, (dataset, chunk, trace_name) identity (#33A)

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
        df = pd.read_csv(csv, low_memory=False, dtype={"trace_name": str, "trace_chunk": str})
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
# Corpus profiles (#40A): configs/corpus_profiles.yaml
# ──────────────────────────────────────────────────────────────────────────────

PROFILES_PATH = Path(__file__).resolve().parent.parent / "configs" / "corpus_profiles.yaml"
DISTANCE_BINS = ("local", "regional", "teleseismic", "unknown")
MANUAL_STATUS_VALUES = ("manual",)
PROFILE_SOURCE_KEYS = ("cap", "use_s", "require_status_columns", "manual_values", "default_bin")


def load_profiles(path=PROFILES_PATH):
    """The `profiles` mapping of the YAML file (version 1) and the file's sha256."""
    path = Path(path)
    raw = path.read_bytes()
    doc = yaml.safe_load(raw) or {}
    if doc.get("version") != 1 or not isinstance(doc.get("profiles"), dict) or not doc["profiles"]:
        raise ValueError(f"{path}: expected version 1 and a non-empty `profiles` mapping")
    return doc["profiles"], hashlib.sha256(raw).hexdigest()


def normalise_fractions(fractions):
    """Every bin of DISTANCE_BINS, missing bins 0, negatives refused, renormalised
    to sum to one (left untouched when the sum is already 1 within 1e-9, so the
    legacy fractions pass through bit for bit)."""
    if not isinstance(fractions, dict) or not fractions:
        raise ValueError("target_fractions must be a non-empty mapping of distance bin -> fraction")
    unknown = set(fractions) - set(DISTANCE_BINS)
    if unknown:
        raise ValueError(f"target_fractions has unknown bins {sorted(unknown)}; expected {DISTANCE_BINS}")
    out = {}
    for b in DISTANCE_BINS:
        v = float(fractions.get(b, 0.0))
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"target_fractions[{b!r}] must be a finite, non-negative number, got {v}")
        out[b] = v
    total = sum(out.values())
    if total <= 0:
        raise ValueError("target_fractions sum to zero")
    if abs(total - 1.0) > 1e-9:
        out = {b: v / total for b, v in out.items()}
    return out


def profile_sha256(profile):
    """sha256 of the profile mapping as sorted, whitespace-free JSON (layout-independent)."""
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _status_phase(column):
    """Phase a status column constrains, from its name: trace_P*/trace_p* -> P,
    trace_S*/trace_s* -> S, anything else -> None (both picks)."""
    m = re.match(r"^trace_([PpSs])", str(column))
    return m.group(1).upper() if m else None


def resolve_profile(name, path=PROFILES_PATH, dataset_configs=None):
    """Apply a profile to DATASET_CONFIGS.

    Returns a dict with `configs` (the DATASET_CONFIGS entries the profile
    lists, in DATASET_CONFIGS order, with cap/use_s and the optional keys
    overridden), `target_fractions` (normalised), `skip_sources` (frozenset),
    `max_distance_km`, `allow_p_only`, `name`, `description`, `sha256`
    (profile mapping) and `file_sha256`. Refuses an unknown profile, a source
    DATASET_CONFIGS does not know, a source key outside PROFILE_SOURCE_KEYS,
    and, when allow_p_only is false, any source with use_s false.
    """
    profiles, file_sha = load_profiles(path)
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; {Path(path).name} has {sorted(profiles)}")
    prof = profiles[name] or {}
    base = DATASET_CONFIGS if dataset_configs is None else dataset_configs
    known = {cfg["name"]: cfg for cfg in base}
    sources = prof.get("sources") or {}
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"profile {name!r} lists no sources")
    missing = sorted(set(sources) - set(known))
    if missing:
        raise ValueError(f"profile {name!r} names sources DATASET_CONFIGS does not know: {missing}")
    allow_p_only = bool(prof.get("allow_p_only", True))
    configs = []
    for cfg in base:                                   # DATASET_CONFIGS order, not the profile's
        if cfg["name"] not in sources:
            continue
        spec = sources[cfg["name"]] or {}
        extra = sorted(set(spec) - set(PROFILE_SOURCE_KEYS))
        if extra:
            raise ValueError(f"profile {name!r}, source {cfg['name']!r}: unknown keys {extra}; "
                             f"allowed {PROFILE_SOURCE_KEYS}")
        if "cap" not in spec or "use_s" not in spec:
            raise ValueError(f"profile {name!r}, source {cfg['name']!r}: cap and use_s are required")
        new = dict(cfg)
        new["cap"] = int(spec["cap"])
        new["use_s"] = bool(spec["use_s"])
        if "default_bin" in spec:
            new["default_bin"] = spec["default_bin"]
        if spec.get("require_status_columns"):
            cols = [str(c) for c in spec["require_status_columns"]]
            new["require_status_columns"] = cols
            new["manual_values"] = [str(v) for v in (spec.get("manual_values") or MANUAL_STATUS_VALUES)]
        if not allow_p_only and not new["use_s"]:
            raise ValueError(f"profile {name!r} has allow_p_only: false but source {cfg['name']!r} "
                             "is P-only (use_s: false)")
        configs.append(new)
    max_dist = prof.get("max_distance_km")
    if max_dist is not None:
        max_dist = float(max_dist)
        if not (np.isfinite(max_dist) and max_dist > 0):
            raise ValueError(f"profile {name!r}: max_distance_km must be positive, got {max_dist}")
    return {
        "name": name,
        "description": " ".join(str(prof.get("description", "")).split()),
        "configs": configs,
        "target_fractions": normalise_fractions(prof.get("target_fractions") or TARGET_FRACTIONS),
        "skip_sources": frozenset(prof.get("skip_sources") or ()),
        "max_distance_km": max_dist,
        "allow_p_only": allow_p_only,
        "sha256": profile_sha256(prof),
        "file_sha256": file_sha,
        "path": str(path),
    }


def apply_status_filter(meta, p_vals, s_vals, columns, manual_values=MANUAL_STATUS_VALUES):
    """Null every pick whose status column (when present) is not a manual value.

    columns : status column names; the phase each constrains comes from
              _status_phase (None constrains both picks). Absent columns are
              no-ops and are reported.
    Returns (p_vals, s_vals, report) with report keys status_columns_listed,
    status_columns_present, n_p_dropped_by_status, n_s_dropped_by_status,
    n_rows_removed_by_status (rows left with no pick; the caller drops them).
    """
    manual = {str(v).strip().lower() for v in manual_values}
    present = [c for c in columns if c in meta.columns]
    p_vals, s_vals = p_vals.copy(), s_vals.copy()
    had_pick = p_vals.notna() | s_vals.notna()
    n_p0, n_s0 = int(p_vals.notna().sum()), int(s_vals.notna().sum())
    for col in present:
        ok = meta[col].astype(str).str.strip().str.lower().isin(manual) & meta[col].notna()
        phase = _status_phase(col)
        if phase in (None, "P"):
            p_vals = p_vals.where(ok)
        if phase in (None, "S"):
            s_vals = s_vals.where(ok)
    report = {
        "status_columns_listed": ";".join(columns),
        "status_columns_present": ";".join(present),
        "n_p_dropped_by_status": n_p0 - int(p_vals.notna().sum()),
        "n_s_dropped_by_status": n_s0 - int(s_vals.notna().sum()),
        "n_rows_removed_by_status": int((had_pick & ~(p_vals.notna() | s_vals.notna())).sum()),
    }
    return p_vals, s_vals, report

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

def process_dataset(cfg, rng, benchmark_exclude=None, event_exclude=None, s_balanced=False,
                     label_error_exclude=None, label_error_report=None,
                     bundle=None, holdout_report=None, max_distance_km=None):
    """
    Load one dataset, retain valid P or permitted S picks, compute distances, apply cap.
    cfg may carry `require_status_columns` (and `manual_values`) from a corpus
    profile (#40A): picks whose status is not manual are nulled before anything
    else (apply_status_filter). max_distance_km drops rows whose known distance
    exceeds it, before the cap; rows without a distance are kept.
    benchmark_exclude : set of trace_name strings to exclude (benchmark traces).
    bundle            : the exclusion bundle (scripts/exclusion_bundle.py, #33A); required.
                        exclusion_bundle.apply_exclusions(kind="signal") removes the listed
                        (dataset, chunk, trace_name) rows, rows whose source origin lies in a
                        held-out window and rows with a 2016/2021 origin; rows whose origin
                        time or location is missing are quarantined per the bundle's policy
                        (dropped, or kept with `independence_unverified` set).
    holdout_report    : optional list to append the per-dataset removal counts to (the
                        apply_exclusions counts plus the status-filter and distance counts).
    event_exclude     : frozenset of event_keys.py fingerprints to exclude — catches
                        the same earthquake landing in the benchmark under a
                        DIFFERENT trace_name (issue #32), which benchmark_exclude
                        alone cannot.
    s_balanced        : if True and use_s=True, additionally require a valid S pick.
    label_error_exclude : set of trace_name strings Aguilar's confident-learning
                        analysis flags as bad labels (GitHub #10), or None.
    label_error_report : optional list to append a per-dataset removal-fraction
                        row to (dataset, n_before, n_flagged_present, pct_removed).
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
    except Exception as exc:
        print(f"    SKIP — failed to load: {exc}")
        return None

    if "chunk" not in meta.columns:
        meta["chunk"] = meta.get("trace_chunk", "")
    print(f"    loaded {len(meta):,} total rows")

    # ── pick columns ──────────────────────────────────────────────────────────
    p_vals, p_col = coalesce_picks(meta, P_PRIORITY)
    s_vals, s_col = coalesce_picks(meta, S_PRIORITY) if cfg["use_s"] else (
        pd.Series(np.nan, index=meta.index), None)
    p_vals = p_vals.where(np.isfinite(p_vals) & (p_vals >= 0))
    s_vals = s_vals.where(np.isfinite(s_vals) & (s_vals >= 0))
    source_report = {"dataset": name}

    # ── manual-status filter (corpus profile, #40A) ───────────────────────────
    status_cols = cfg.get("require_status_columns")
    if status_cols:
        p_vals, s_vals, st = apply_status_filter(meta, p_vals, s_vals, status_cols,
                                                 cfg.get("manual_values") or MANUAL_STATUS_VALUES)
        source_report.update(st)
        if not st["status_columns_present"]:
            print(f"    WARNING: none of the status columns {status_cols} exists in {name}; "
                  "the manual-status filter is a no-op for this source (39A census decides)")
        else:
            print(f"    status filter on {st['status_columns_present']}: dropped "
                  f"{st['n_p_dropped_by_status']:,} P and {st['n_s_dropped_by_status']:,} S picks, "
                  f"{st['n_rows_removed_by_status']:,} rows left without a pick")

    keep = p_vals.notna() | s_vals.notna()
    if s_balanced and cfg["use_s"]:
        keep &= s_vals.notna()
    meta = meta.loc[keep].copy()
    p_vals = p_vals.loc[keep]
    if len(meta) == 0:
        print("    SKIP — 0 valid permitted P/S picks")
        return None
    print(f"    {len(meta):,} with valid P or S pick (p_col={p_col}, s_col={s_col})")

    # ── exclude benchmark traces ──────────────────────────────────────────────
    if benchmark_exclude and "trace_name" in meta.columns:
        before = len(meta)
        keep_bm = ~meta["trace_name"].isin(benchmark_exclude)
        meta   = meta.loc[keep_bm].copy()
        p_vals = p_vals.loc[keep_bm]
        n_removed = before - len(meta)
        if n_removed:
            print(f"    excluded {n_removed:,} benchmark traces → {len(meta):,} remaining")

    # ── exclusion bundle (#33A): listed (dataset, chunk, trace_name) rows, ──
    #    held-out windows, 2016/2021 origins, quarantine of unknown origins
    if bundle is None:
        raise ValueError("process_dataset needs the exclusion bundle (scripts/exclusion_bundle.py); "
                         "nothing is built without it")
    before = len(meta)
    meta, ex_report = eb.apply_exclusions(meta, bundle, kind="signal", dataset=name)
    p_vals = p_vals.loc[meta.index]
    if before - len(meta):
        print(f"    excluded {before - len(meta):,} rows by the exclusion bundle "
              f"(listed {ex_report['n_trace_listed']:,}, in-window {ex_report['n_in_window']:,}, "
              f"{sorted(hs.HOLDOUT_YEARS)} origin {ex_report['n_year_holdout']:,}, "
              f"quarantined unknown {ex_report['n_quarantined_unknown']:,}) → {len(meta):,} remaining")
    if ex_report["n_unknown_kept_flagged"]:
        print(f"    WARNING: {ex_report['n_unknown_kept_flagged']:,} rows have no origin time or location; "
              f"kept with {eb.FLAG_COL}=True (bundle policy allow_unknown) and outside any independence claim")
    source_report.update({k: v for k, v in ex_report.items()
                          if k.startswith("n_") or k in ("trace_list_checked", "bundle_sha256")})

    # ── exclude Aguilar-flagged bad-label traces (issue #10) ──────────────────
    if label_error_exclude and "trace_name" in meta.columns:
        before = len(meta)
        keep_le = ~meta["trace_name"].isin(label_error_exclude)
        meta   = meta.loc[keep_le].copy()
        p_vals = p_vals.loc[keep_le]
        n_removed = before - len(meta)
        if label_error_report is not None:
            label_error_report.append({
                "dataset": name,
                "n_before_label_error_filter": before,
                "n_removed": n_removed,
                "pct_removed": round(100 * n_removed / before, 3) if before else 0.0,
            })
        if n_removed:
            print(f"    excluded {n_removed:,} Aguilar bad-label traces → {len(meta):,} remaining")

    # ── exclude benchmark EVENTS under a different trace_name (issue #32) ────
    if event_exclude:
        import event_keys as ek
        row_keys = ek.derive_event_keys(name, meta)
        keep_ev = ~row_keys.map(lambda ks: bool(ks & event_exclude))
        before = len(meta)
        meta   = meta.loc[keep_ev].copy()
        p_vals = p_vals.loc[keep_ev]
        n_removed = before - len(meta)
        if n_removed:
            print(f"    excluded {n_removed:,} benchmark EVENTS (different trace_name) → {len(meta):,} remaining")

    # ── distance ──────────────────────────────────────────────────────────────
    if cfg["dist_col"] and cfg["dist_col"] in meta.columns:
        dist_km = to_km(meta[cfg["dist_col"]], cfg["dist_unit"])
    else:
        dist_km = pd.Series(np.nan, index=meta.index)

    # ── maximum distance (corpus profile, #40A): known distances only ────────
    if max_distance_km is not None:
        far = dist_km.notna() & (dist_km > max_distance_km)
        source_report["n_beyond_max_distance"] = int(far.sum())
        if far.any():
            meta    = meta.loc[~far].copy()
            p_vals  = p_vals.loc[~far]
            dist_km = dist_km.loc[~far]
            print(f"    excluded {int(far.sum()):,} rows beyond {max_distance_km:g} km → {len(meta):,} remaining")

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
    s_vals = s_vals.loc[meta.index].copy()
    source_report["n_after_cap"] = int(len(meta))

    # P-only policy: null S for teleseismic rows
    tele_mask = dist_bin == "teleseismic"
    s_vals = s_vals.copy()
    source_report["n_s_nulled_teleseismic"] = int((tele_mask & s_vals.notna()).sum())
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
        "p_col":             p_col or "",
        "s_col":             s_col or "",
        # fingerprint columns (NaN when the source has none) so that
        # scripts/audit_heldout_sequences.py --check-manifest can verify the
        # sequence and year hold-outs without re-loading the metadata
        "source_origin_time":   meta[hs.TIME_COL].values if hs.TIME_COL in meta.columns else np.nan,
        "source_latitude_deg":  meta[hs.LAT_COL].values  if hs.LAT_COL  in meta.columns else np.nan,
        "source_longitude_deg": meta[hs.LON_COL].values  if hs.LON_COL  in meta.columns else np.nan,
        eb.FLAG_COL:         meta[eb.FLAG_COL].values,
        "orig_split":        (
            meta["split"].map(normalise_split).values
            if "split" in meta.columns
            else np.full(len(meta), "")
        ),
    })
    # Preserve known source coordinates; unknown fields remain explicit NaNs and
    # must be resolved from verified source metadata by the loader.
    for column in sorted(METADATA_FIELDS - {"trace_name"} | {"arrival_sampling_rate_hz"}):
        out[column] = meta[column].values if column in meta else np.nan
    # The existing teleseismic P-only policy can remove an S-only row's last label.
    out = out.loc[out.p_arrival_sample.notna() | out.s_arrival_sample.notna()].reset_index(drop=True)
    # counts of what this source contributes to the pool, after the P-only
    # policy and the last-label drop, so the report matches the written rows
    source_report["n_written"] = int(len(out))
    source_report["n_with_s_written"] = int(out["s_arrival_sample"].notna().sum())
    if holdout_report is not None:
        holdout_report.append(source_report)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Distance stratification (training set only)
# ──────────────────────────────────────────────────────────────────────────────

def stratify_training(train_df, rng, fractions=None):
    """
    Resample train_df so the distance-bin distribution matches `fractions`
    (default TARGET_FRACTIONS; a corpus profile passes its own).
    Bins below their target fraction are kept whole; over-represented bins are
    downsampled.  The total size is determined by the smallest-ratio bin. A bin
    with fraction 0 is removed from the training split.
    """
    fractions = TARGET_FRACTIONS if fractions is None else fractions
    bin_counts = train_df["distance_bin"].value_counts()
    total_available = len(train_df)

    # compute how many traces each bin *could* support given its target fraction
    max_total_per_bin = {}
    for b, frac in fractions.items():
        if b not in bin_counts or frac == 0:
            max_total_per_bin[b] = 0
            continue
        # if this bin has `n` traces and its target is `frac`, the implied total is n/frac
        max_total_per_bin[b] = int(bin_counts[b] / frac)

    target_total = min(v for v in max_total_per_bin.values() if v > 0)
    target_total = min(target_total, total_available)

    sampled = []
    for b, frac in fractions.items():
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

def _event_group_ids(df):
    """
    Positional group id per row of df (0..len(df)-1 order), such that rows
    sharing any event_keys.py fingerprint end up in the same group — so an
    earthquake's traces move together across train/val/test instead of being
    split independently. Rows whose source origins coincide within
    exclusion_bundle.origin_unions' tolerances (2 s, 0.1 deg) are unioned too,
    so an equivalent event present in two datasets (or in a dataset with no
    id column, e.g. geofon) stays in one split (#33A). Rows with neither a
    key nor an origin fingerprint are singleton groups, as before.
    """
    import event_keys as ek

    dnames = df["dataset_name"].values
    tnames = df["trace_name"].values
    chunks = df["chunk"].fillna("").values if "chunk" in df.columns else np.full(len(df), "")
    n = len(df)

    maps = {}
    for dname in pd.unique(dnames):
        try:
            maps[dname] = ek.trace_key_map(dname)
        except Exception as exc:
            print(f"    event-key lookup unavailable for {dname!r} ({exc}) "
                  f"— falling back to trace-level assignment for it")
            maps[dname] = {}

    uf = eb.UnionFind(n)

    key_to_first_row = {}
    for pos in range(n):
        # (trace_name, chunk) — NOT trace_name alone: mlaapde/aq2009gm/cwa
        # reuse trace_name as a positional slot index across chunks, so a
        # plain trace_name lookup would silently group unrelated rows from
        # different chunks together (see event_keys.trace_key_map docstring).
        keys = maps.get(dnames[pos], {}).get((tnames[pos], chunks[pos]), frozenset())
        for k in keys:
            if k in key_to_first_row:
                uf.union(pos, key_to_first_row[k])
            else:
                key_to_first_row[k] = pos

    # #33A: equivalent events across datasets (same origin within tolerance)
    if all(c in df.columns for c in (hs.TIME_COL, hs.LAT_COL, hs.LON_COL)):
        for i, j in eb.origin_unions(df[hs.TIME_COL], df[hs.LAT_COL], df[hs.LON_COL]):
            uf.union(int(i), int(j))

    return uf.labels()


def assign_splits(df, rng, val_frac=0.10, test_frac=0.10):
    """
    Group by event identity (scripts/event_keys.py) wherever an event key is
    derivable, so the same earthquake can't land on both sides of a split —
    regardless of whether the source dataset ships its own train/dev/test
    partition. Vendor splits (orig_split) are NOT trusted as event-clean:
    the 2026-07 leakage audit found several SeisBench-provided splits
    (mlaapde, aq2009gm, cwa, stead) still put the same earthquake, recorded
    at a different station, on both sides. orig_split is used only as a
    fallback for rows with NO derivable event key at all (event_keys.py's
    UNVERIFIABLE_DATASETS, e.g. obst2024, or a partial-coverage dataset's
    unkeyed rows, e.g. TXED's ~40% noise-only rows) — better than a coin
    flip when there's no other signal. Keyless rows with no orig_split
    either fall back to random per-trace assignment (prior behavior).
    """
    import event_keys as ek

    dnames = df["dataset_name"].values
    tnames = df["trace_name"].values
    chunks = df["chunk"].fillna("").values if "chunk" in df.columns else np.full(len(df), "")
    n = len(df)

    maps = {}
    for dname in pd.unique(dnames):
        try:
            maps[dname] = ek.trace_key_map(dname)
        except Exception as exc:
            print(f"    event-key lookup unavailable for {dname!r} ({exc}) "
                  f"— falling back to orig_split/random for it")
            maps[dname] = {}

    # (trace_name, chunk) — see _event_group_ids for why chunk can't be dropped.
    has_key = np.fromiter(
        (bool(maps.get(dnames[i], {}).get((tnames[i], chunks[i]), frozenset())) for i in range(n)),
        dtype=bool, count=n,
    )
    # #33A: a row with an origin fingerprint has an identity too (origin
    # coincidence groups it with equivalent events of other datasets), so it
    # is grouped rather than handed to the vendor split. hs._to_utc, not a
    # bare pd.to_datetime: pandas >= 2 infers one format from the first value
    # and coerces the rest to NaT, which would hand a row with fractional
    # seconds to the vendor split while its twin is grouped (the same parser
    # eb.origin_unions uses, so "has a fingerprint" and "is grouped" agree).
    if all(c in df.columns for c in (hs.TIME_COL, hs.LAT_COL, hs.LON_COL)):
        has_key |= (hs._to_utc(df[hs.TIME_COL]).notna()
                    & pd.to_numeric(df[hs.LAT_COL], errors="coerce").notna()
                    & pd.to_numeric(df[hs.LON_COL], errors="coerce").notna()).to_numpy()

    split = pd.Series("", index=df.index, dtype=str)

    no_key_idx = df.index[~has_key]
    orig = df.loc[no_key_idx, "orig_split"]
    split.loc[no_key_idx[(orig == "train").values]] = "train"
    split.loc[no_key_idx[(orig == "val").values]]   = "val"
    split.loc[no_key_idx[(orig == "test").values]]  = "test"

    ungrouped_idx = df.index[split == ""]
    if len(ungrouped_idx):
        sub = df.loc[ungrouped_idx]
        m = len(sub)
        group_pos = _event_group_ids(sub)

        # whole groups into val, then test, the rest train (exclusion_bundle
        # keeps the rule so the fixture tests exercise the same code)
        assign = eb.fill_splits_by_group(group_pos, rng, val_frac=val_frac, test_frac=test_frac)

        _, sizes = np.unique(group_pos, return_counts=True)
        n_multi = int((sizes > 1).sum())
        print(f"    event-aware split: {m:,} rows -> {len(sizes):,} groups "
              f"({n_multi:,} multi-trace events, "
              f"{len(sizes) - n_multi:,} singleton events / no-key rows without orig_split)")

        split.loc[ungrouped_idx] = assign

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


def load_label_error_exclusions():
    """
    Load Aguilar's confident-learning "bad label" trace_names (GitHub #10) for
    every dataset with a published multiplet report, so they're excluded from
    the training pool the same way the benchmark pool already excludes them
    (notebooks/04_creating_benchmark_dataset.ipynb §1.4b).

    Returns {dataset_name: frozenset of trace_name}.
    """
    import label_error_filter as lef
    exclusions = {}
    for dataset_name in sorted(set(lef.REPORT_STEM_TO_DATASET.values())):
        bad = lef.load_bad_trace_names(dataset_name, extra_cache_dirs=["data/labelerrors"])
        if bad:
            exclusions[dataset_name] = frozenset(bad)
    total = sum(len(v) for v in exclusions.values())
    print(f"  Loaded {total:,} Aguilar-flagged bad-label traces across {len(exclusions)} datasets")
    return exclusions


def load_benchmark_exclusions():
    """
    Load (dataset_name, trace_name) pairs from the benchmark manifest so they
    can be excluded from training/validation data, PLUS the event-level
    fingerprint (scripts/event_keys.py) of each benchmark trace so the same
    earthquake recorded under a different trace_name is caught too (issue #32
    — exact trace_name exclusion alone left ~75-95% event-level leakage for
    mlaapde/aq2009gm/cwa).

    Uses trace_key_map_any_chunk(), NOT trace_key_map(): notebooks/benchmark_
    manifest.csv doesn't retain which chunk/shard a benchmark row came from
    (mlaapde/aq2009gm/cwa reuse trace_name as a positional slot index across
    monthly chunks), so a benchmark trace_name is resolved against the UNION
    of every chunk's event for that slot — deliberately conservative, since
    this is a one-directional exclusion (over-excluding a few unrelated
    events from training is harmless; missing a real leak is not).

    Returns (trace_exclusions, event_exclusions), both {dataset_name: set/frozenset}.
    """
    if not BENCHMARK_CSV.exists():
        print(f"  WARNING: benchmark manifest not found at {BENCHMARK_CSV} — no exclusions applied")
        return {}, {}
    import event_keys as ek

    bm = pd.read_csv(BENCHMARK_CSV, usecols=["dataset", "trace_name"])
    trace_exclusions = {}
    event_exclusions = {}
    for ds, group in bm.groupby("dataset"):
        trace_exclusions[ds] = set(group["trace_name"])
        try:
            trace_map = ek.trace_key_map_any_chunk(ds)
        except Exception as exc:
            print(f"    event-key lookup unavailable for benchmark dataset {ds!r} ({exc}) "
                  f"— falling back to trace_name-only exclusion for it")
            event_exclusions[ds] = frozenset()
            continue
        keys = (trace_map.get(t, frozenset()) for t in group["trace_name"])
        event_exclusions[ds] = frozenset().union(*keys)

    total_traces = sum(len(v) for v in trace_exclusions.values())
    total_events = sum(len(v) for v in event_exclusions.values())
    print(f"  Loaded {total_traces:,} benchmark traces ({total_events:,} distinct events) "
          f"to exclude across {len(trace_exclusions)} datasets")
    return trace_exclusions, event_exclusions


# 2026-09-08 scope decision: no ocean-bottom observations this round.
# Both OBS sources are skipped unless --include-obs is passed.
SKIP_SOURCES_THIS_ROUND = frozenset({"obst2024", "obs"})


def main(output_dir, seed, s_balanced=False, label_error_filter=True, include_obs=False,
         allow_uncertified_bundle=False, profile=None, profiles_file=None):
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── corpus profile (#40A) or the module defaults ─────────────────────────
    if profile is not None:
        resolved = resolve_profile(profile, path=profiles_file or PROFILES_PATH)
        dataset_configs = resolved["configs"]
        fractions = resolved["target_fractions"]
        skip_sources = resolved["skip_sources"]
        max_distance_km = resolved["max_distance_km"]
        profile_record = {k: resolved[k] for k in ("name", "description", "sha256", "file_sha256", "path",
                                                   "allow_p_only", "max_distance_km", "target_fractions")}
        profile_record["skip_sources"] = sorted(skip_sources)
    else:
        dataset_configs = DATASET_CONFIGS
        fractions = TARGET_FRACTIONS
        skip_sources = SKIP_SOURCES_THIS_ROUND
        max_distance_km = None
        profile_record = {"name": None, "sha256": None,
                          "reason": "no --profile; module DATASET_CONFIGS, TARGET_FRACTIONS and "
                                    "SKIP_SOURCES_THIS_ROUND (the legacy_v2 profile)"}

    print("=" * 70)
    print("Building PhaseNet training manifests")
    print(f"  SeisBench cache      : {SEISBENCH_CACHE}")
    print(f"  Output dir           : {out_path.resolve()}")
    print(f"  Random seed          : {seed}")
    print(f"  S-balanced mode      : {s_balanced}")
    print(f"  Label-error filter   : {label_error_filter}")
    print(f"  Year hold-out        : {sorted(hs.HOLDOUT_YEARS)} (unknown origins: bundle quarantine policy)")
    if profile is not None:
        print(f"  Corpus profile       : {profile} ({profile_record['sha256'][:12]}, "
              f"{len(dataset_configs)} sources, max distance {max_distance_km}, "
              f"P-only allowed {profile_record['allow_p_only']})")
    else:
        print("  Corpus profile       : none (module defaults = legacy_v2)")
    print(f"  Target fractions     : {fractions}")
    print("=" * 70)

    # ── load benchmark exclusions ────────────────────────────────────────────
    benchmark_exclusions, benchmark_event_exclusions = load_benchmark_exclusions()
    label_error_exclusions = load_label_error_exclusions() if label_error_filter else {}
    # Exclusion bundle (#33A): fails closed when absent, stale or, unless
    # --allow-uncertified-bundle, not certifying every source snapshot. The
    # sequence list itself is always required.
    bundle_check = {}
    bundle = eb.load_bundle(require_certified=not allow_uncertified_bundle, report=bundle_check)
    if not bundle["sequence_list_present"]:
        sys.exit(f"ERROR: the exclusion bundle records {hs.EXCLUSION_CSV.name} as absent; run "
                 "audit_heldout_sequences.py on the server, commit the list and rebuild the bundle")
    sequence_exclusions = eb.trace_exclusions(bundle)
    print(f"  Exclusion bundle     : {bundle['sha256'][:16]}... certified={bundle['certified']} "
          f"allow_unknown={bundle['quarantine_policy']['allow_unknown']} "
          f"({len(bundle_check['checked'])} inputs verified, {len(bundle_check['unchecked'])} unchecked)")
    if not bundle["certified"]:
        print(f"  WARNING: bundle not certified (unhashed sources: {', '.join(bundle['uncertified_sources'])}); "
              "these manifests carry no source-snapshot certificate")
    print(f"  Loaded {sum(len(v) for v in sequence_exclusions.values()):,} held-out-sequence trace names "
          f"across {len(sequence_exclusions)} datasets from {hs.EXCLUSION_CSV.relative_to(Path(__file__).parent.parent)}")

    # ── process all datasets ─────────────────────────────────────────────────
    frames = []
    label_error_report = []
    holdout_report = []
    for cfg in dataset_configs:
        if cfg["name"] in skip_sources and not include_obs:
            print(f"\n  [{cfg['name']}]\n    SKIP — ocean-bottom data are out of scope this round (2026-09-08); pass --include-obs to override")
            continue
        exclude = benchmark_exclusions.get(cfg["name"], set())
        event_exclude = benchmark_event_exclusions.get(cfg["name"], frozenset())
        le_exclude = label_error_exclusions.get(cfg["name"], frozenset())
        df = process_dataset(cfg, rng, benchmark_exclude=exclude,
                              event_exclude=event_exclude, s_balanced=s_balanced,
                              label_error_exclude=le_exclude,
                              label_error_report=label_error_report,
                              bundle=bundle,
                              holdout_report=holdout_report,
                              max_distance_km=max_distance_km)
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
    train_df = stratify_training(train_df, rng, fractions=fractions)
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
        "source_origin_time", "source_latitude_deg", "source_longitude_deg",
        eb.FLAG_COL,
    ]
    KEEP_COLS += sorted(METADATA_FIELDS - {"trace_name"} | {"arrival_sampling_rate_hz"})
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

    ho_df = pd.DataFrame(holdout_report)
    ho_df["profile"] = profile_record["name"] or ""
    ho_df["profile_sha256"] = profile_record["sha256"] or ""
    ho_df.to_csv(out_path / "heldout_removal_report.csv", index=False)
    print("\n  Exclusion bundle removal per source (#33A):")
    for _, r in ho_df.iterrows():
        print(f"    {r['dataset']:16s}  listed {r['n_trace_listed']:>7,}  in-window {r['n_in_window']:>7,}  "
              f"years {r['n_year_holdout']:>7,}  quarantined {r['n_quarantined_unknown']:>7,}  "
              f"kept-flagged {r['n_unknown_kept_flagged']:>7,}")
    # final gate: nothing written may be listed, sit in a window or in a
    # held-out year (chunk-aware check_manifest, then the bundle itself)
    gate = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rep = hs.check_manifest(df, sequence_exclusions)
        if rep["n_excluded_present"] or rep["n_in_window"] or rep["n_year_holdout"]:
            sys.exit(f"ERROR: {split_name} manifest violates the hold-out: {rep}")
        _, brep = eb.apply_exclusions(df, bundle, kind="signal")
        if brep["n_removed"]:
            sys.exit(f"ERROR: {split_name} manifest violates the exclusion bundle: {brep}")
        gate[split_name] = {**rep, "bundle": brep}
    print("  Hold-out gate: train/val/test contain no listed trace, no in-window origin and no 2016/2021 origin")

    # ── provenance beside the manifests (#33A) ───────────────────────────────
    from hash_manifests import hash_manifest
    manifest_hashes = {}
    for fn in ("train.csv", "val.csv", "test.csv"):
        digest, n_rows = hash_manifest(out_path / fn)
        manifest_hashes[fn] = {"sha256_of_sorted_keys": digest, "n_rows": int(n_rows)}
    provenance = {
        "bundle_sha256": bundle["sha256"],
        "bundle_path": str(eb.BUNDLE_PATH.relative_to(eb.REPO_ROOT)),
        "bundle_certified": bundle["certified"],
        "bundle_uncertified_sources": bundle["uncertified_sources"],
        "bundle_rules_sha256": bundle["rules"]["sha256"],
        "bundle_policy_sha256": bundle["inputs"]["evaluation_suites"]["policy_sha256"],
        "quarantine_policy": bundle["quarantine_policy"],
        "git_commit": eb._git_commit(eb.REPO_ROOT),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/build_training_dataset.py",
        "options": {"seed": seed, "s_balanced": s_balanced, "label_error_filter": label_error_filter,
                    "include_obs": include_obs, "allow_uncertified_bundle": allow_uncertified_bundle,
                    "profile": profile},
        "profile": profile_record,
        "target_fractions": fractions,
        "sources": [{"name": c["name"], "cap": c["cap"], "use_s": c["use_s"],
                     "require_status_columns": c.get("require_status_columns")} for c in dataset_configs
                    if not (c["name"] in skip_sources and not include_obs)],
        "per_source": holdout_report,
        "gate": gate,
        "manifests": manifest_hashes,
    }
    (out_path / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=eb._json_default) + "\n")
    print(f"  Provenance written to {out_path / 'provenance.json'} (bundle {bundle['sha256'][:12]})")

    if label_error_report:
        le_df = pd.DataFrame(label_error_report)
        le_df.to_csv(out_path / "label_error_removal_report.csv", index=False)
        print("\n  Aguilar bad-label removal (GitHub #10):")
        for _, r in le_df.iterrows():
            print(f"    {r['dataset']:16s}  removed {r['n_removed']:>7,} / "
                  f"{r['n_before_label_error_filter']:>7,}  ({r['pct_removed']:.2f}%)")

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
    parser.add_argument("--no-label-error-filter", action="store_true",
                        help="Skip excluding Aguilar-flagged bad-label traces (GitHub #10; on by default)")
    parser.add_argument("--strict-year-holdout", action="store_true",
                        help="Deprecated no-op: the exclusion bundle's quarantine policy decides what happens "
                             "to rows without an origin time or location (default: dropped)")
    parser.add_argument("--allow-uncertified-bundle", action="store_true",
                        help="Build even if the exclusion bundle does not certify every source snapshot "
                             "(the sequence list is still required); recorded in provenance.json")
    parser.add_argument("--include-obs", action="store_true",
                        help="Include obst2024 and obs (ocean-bottom) sources, which are skipped this round (2026-09-08)")
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help="Corpus profile from --profiles-file (t0_pilot, legacy_v2, ...); default: the module "
                             "defaults, which legacy_v2 reproduces")
    parser.add_argument("--profiles-file", default=str(PROFILES_PATH),
                        help=f"Profile YAML (default: {PROFILES_PATH.relative_to(PROFILES_PATH.parents[1])})")
    parser.add_argument("--list-profiles", action="store_true", help="Print the profiles of --profiles-file and exit")
    args = parser.parse_args()
    if args.list_profiles:
        profiles, file_sha = load_profiles(args.profiles_file)
        print(f"{args.profiles_file} (sha256 {file_sha[:12]})")
        for pname, prof in profiles.items():
            srcs = prof.get("sources") or {}
            print(f"  {pname:12s} {profile_sha256(prof)[:12]}  {len(srcs)} sources, "
                  f"{sum(int(v['cap']) for v in srcs.values()):,} rows at cap, "
                  f"max distance {prof.get('max_distance_km')}, P-only allowed {prof.get('allow_p_only', True)}")
            print(f"    {' '.join(str(prof.get('description', '')).split())}")
        sys.exit(0)
    if args.strict_year_holdout:
        print("NOTE: --strict-year-holdout is a no-op; the exclusion bundle's quarantine policy applies")
    main(args.output_dir, args.seed, s_balanced=args.s_balanced,
         label_error_filter=not args.no_label_error_filter,
         include_obs=args.include_obs,
         allow_uncertified_bundle=args.allow_uncertified_bundle,
         profile=args.profile, profiles_file=args.profiles_file)
