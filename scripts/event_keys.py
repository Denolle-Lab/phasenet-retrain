#!/usr/bin/env python3
"""
event_keys.py

Derives an event-level fingerprint per trace, for datasets where trace-name
exclusion (scripts/build_training_dataset.py's load_benchmark_exclusions())
isn't enough to prove independence: the same earthquake, recorded at a
different station, has a different trace_name but is still the same event.

Coverage (confirmed empirically against the local SeisBench cache):
  - stead, instancecounts, ethz, mlaapde, aq2009gm, pisdl, vcseis: scalar
    `source_id` column, one event id per row.
  - pnw: `event_id`. cwa: `source_event_id`.
  - ceed: `source_id_list` can hold MULTIPLE distinct ids in a single row
    (~12% of rows, from template-matched detections sharing a trace window)
    — handled as a set, not a scalar.
  - txed: no id column, but `source_origin_time` + lat/lon are present for
    ~60% of rows (event traces; the rest are unpicked noise traces) — used
    as a synthetic fingerprint.
  - obst2024: every `source_*` column is 100% NaN in the cached copy — event
    identity is fundamentally unverifiable with current data. Returns empty
    keys for every row rather than a false negative.

All keys are namespaced as f"{dataset_name}:{raw_id}" since ids are only
guaranteed unique within a dataset, not globally. Every function returns a
`frozenset` per row (possibly empty) so callers can use uniform set-overlap
logic regardless of whether a dataset yields 0, 1, or multiple ids per row.
"""

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

SEISBENCH_CACHE = "/data/wsd04/ak287/.seisbench"
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)
import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

from build_training_dataset import (
    _load_chunked_meta, MLAAPDE_PATH, CWA_PATH, AQ2009GM_PATH, PISDL_PATH,
)

# ── Metadata loaders, one per benchmark dataset ──────────────────────────────
_SBD_CLASS = {
    "stead":          sbd.STEAD,
    "ceed":           sbd.CEED,
    "instancecounts": sbd.InstanceCounts,
    "ethz":           sbd.ETHZ,
    "pnw":            sbd.PNW,
    "txed":           sbd.TXED,
    "vcseis":         sbd.VCSEIS,
    "obst2024":       sbd.OBST2024,
}
_CHUNKED = {
    "mlaapde":  (MLAAPDE_PATH,  "metadata_"),
    "cwa":      (CWA_PATH,      "metadata_"),
    "aq2009gm": (AQ2009GM_PATH, "metadata"),
}
_WAVEFORM_DATASET = {"pisdl": PISDL_PATH}


@lru_cache(maxsize=None)
def load_metadata(dataset_name: str) -> pd.DataFrame:
    if dataset_name in _SBD_CLASS:
        return _SBD_CLASS[dataset_name](sampling_rate=100).metadata
    if dataset_name in _CHUNKED:
        path, prefix = _CHUNKED[dataset_name]
        return _load_chunked_meta(path, prefix=prefix)
    if dataset_name in _WAVEFORM_DATASET:
        return sbd.WaveformDataset(str(_WAVEFORM_DATASET[dataset_name])).metadata
    raise KeyError(f"No metadata loader registered for dataset {dataset_name!r}")


# ── Event-id column registry ─────────────────────────────────────────────────
SCALAR_ID_COL = {
    "stead":          "source_id",
    "instancecounts": "source_id",
    "ethz":           "source_id",
    "mlaapde":        "source_id",
    "aq2009gm":       "source_id",
    "pisdl":          "source_id",
    "vcseis":         "source_id",
    "pnw":            "event_id",
    "cwa":            "source_event_id",
}
MULTI_ID_COL = {"ceed": "source_id_list"}

FALLBACK_TIME_COL = "source_origin_time"
FALLBACK_LAT_COL  = "source_latitude_deg"
FALLBACK_LON_COL  = "source_longitude_deg"

# 100% NaN on every source_* column in the cached copy — no fingerprint
# possible. Named explicitly so it never silently reports 0% leakage.
UNVERIFIABLE_DATASETS = {"obst2024"}

_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:-]+")


def key_method(dataset_name: str, meta_df: pd.DataFrame) -> str:
    """Which strategy derive_event_keys() will use for this dataset."""
    if dataset_name in UNVERIFIABLE_DATASETS:
        return "unverifiable"
    if dataset_name in MULTI_ID_COL:
        return "multi_id"
    if dataset_name in SCALAR_ID_COL and SCALAR_ID_COL[dataset_name] in meta_df.columns:
        return "scalar_id"
    if all(c in meta_df.columns for c in (FALLBACK_TIME_COL, FALLBACK_LAT_COL, FALLBACK_LON_COL)):
        return "synthetic"
    return "unverifiable"


def derive_event_keys(dataset_name: str, meta_df: pd.DataFrame) -> pd.Series:
    """Per-row frozenset of namespaced event-id strings (empty if unknown)."""
    empty = pd.Series([frozenset()] * len(meta_df), index=meta_df.index)

    if dataset_name in UNVERIFIABLE_DATASETS:
        return empty

    if dataset_name in MULTI_ID_COL:
        col = MULTI_ID_COL[dataset_name]
        if col not in meta_df.columns:
            return empty

        def _parse_multi(v):
            toks = _ID_TOKEN_RE.findall(str(v))
            return frozenset(f"{dataset_name}:{t}" for t in toks) if toks else frozenset()

        return meta_df[col].map(_parse_multi)

    if dataset_name in SCALAR_ID_COL and SCALAR_ID_COL[dataset_name] in meta_df.columns:
        col = SCALAR_ID_COL[dataset_name]

        def _scalar(v):
            return frozenset() if pd.isna(v) else frozenset({f"{dataset_name}:{v}"})

        return meta_df[col].map(_scalar)

    if all(c in meta_df.columns for c in (FALLBACK_TIME_COL, FALLBACK_LAT_COL, FALLBACK_LON_COL)):
        t   = pd.to_datetime(meta_df[FALLBACK_TIME_COL], errors="coerce", utc=True)
        lat = pd.to_numeric(meta_df[FALLBACK_LAT_COL], errors="coerce")
        lon = pd.to_numeric(meta_df[FALLBACK_LON_COL], errors="coerce")

        def _synthetic(ts, la, lo):
            if pd.isna(ts) or pd.isna(la) or pd.isna(lo):
                return frozenset()
            key = f"{dataset_name}:{ts.floor('s').isoformat()}|{round(la, 2)}|{round(lo, 2)}"
            return frozenset({key})

        return pd.Series(
            [_synthetic(a, b, c) for a, b, c in zip(t, lat, lon)], index=meta_df.index
        )

    return empty


def trace_key_map(dataset_name: str) -> dict:
    """trace_name -> frozenset(event keys) for every row of a dataset."""
    meta = load_metadata(dataset_name)
    keys = derive_event_keys(dataset_name, meta)
    return dict(zip(meta["trace_name"], keys))


if __name__ == "__main__":
    for name in sorted(set(SCALAR_ID_COL) | set(MULTI_ID_COL) | UNVERIFIABLE_DATASETS | {"txed"}):
        meta = load_metadata(name)
        keys = derive_event_keys(name, meta)
        coverage = (keys.map(len) > 0).mean()
        print(f"{name:16s} method={key_method(name, meta):12s} "
              f"n={len(meta):8,d} coverage={coverage:6.1%}")
