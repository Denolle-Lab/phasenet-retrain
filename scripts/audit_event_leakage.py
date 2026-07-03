#!/usr/bin/env python3
"""
audit_event_leakage.py

Read-only proof of train/benchmark independence at the EVENT level, not
just the trace_name level. scripts/build_training_dataset.py's
load_benchmark_exclusions() already guarantees 0 exact trace_name overlap
between data/manifests/{train,val,test}.csv and notebooks/benchmark_manifest.csv
(verified separately). What that mechanism CANNOT catch: the same
earthquake, recorded at a different station, landing in both the training
manifest and the benchmark under a different trace_name.

This script does not regenerate any manifest — it only reads
data/manifests/{train,val}.csv and notebooks/benchmark_manifest.csv as they
exist today and cross-references event identities (scripts/event_keys.py).

`obst2024` has no derivable event identity at all in the cached SeisBench
copy (every source_* column is 100% NaN) — it is reported as
`method_used=unverifiable` with leaked_event_frac=NaN rather than a
misleadingly clean 0%.

Run from repo root:
    conda activate surface
    python scripts/audit_event_leakage.py

Outputs
-------
  results/event_leakage_audit.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import event_keys as ek

TRAIN_PATH     = REPO_ROOT / "data" / "manifests" / "train.csv"
VAL_PATH       = REPO_ROOT / "data" / "manifests" / "val.csv"
BENCHMARK_PATH = REPO_ROOT / "notebooks" / "benchmark_manifest.csv"
OUT_CSV        = REPO_ROOT / "results" / "event_leakage_audit.csv"

BENCHMARK_DATASETS = [
    "instancecounts", "stead", "ceed", "pnw", "txed", "ethz",
    "mlaapde", "aq2009gm", "pisdl", "obst2024", "vcseis", "cwa",
]


def _union(keys_series: pd.Series) -> set:
    out = set()
    for ks in keys_series:
        out |= ks
    return out


def audit_dataset(name, bm_df, train_df, val_df):
    bm_traces    = bm_df.loc[bm_df["dataset"] == name, "trace_name"]
    train_traces = train_df.loc[train_df["dataset_name"] == name, "trace_name"]
    val_traces   = val_df.loc[val_df["dataset_name"] == name, "trace_name"]

    trace_map = ek.trace_key_map(name)
    method    = ek.key_method(name, ek.load_metadata(name))

    bm_keys    = bm_traces.map(lambda t: trace_map.get(t, frozenset()))
    train_keys = train_traces.map(lambda t: trace_map.get(t, frozenset()))
    val_keys   = val_traces.map(lambda t: trace_map.get(t, frozenset()))

    train_event_set = _union(train_keys)
    val_event_set   = _union(val_keys)

    n_bm_rows         = len(bm_traces)
    n_bm_unverifiable = int((bm_keys.map(len) == 0).sum())

    bm_verifiable = bm_keys[bm_keys.map(len) > 0]
    bm_event_set  = _union(bm_verifiable)
    n_bm_events   = len(bm_event_set)

    n_leaked = sum(1 for k in bm_event_set if k in train_event_set)

    # Any-id-overlap leak per verifiable benchmark row — handles CEED rows
    # that carry multiple distinct event ids (a row leaks if ANY id matches).
    n_rows_leaked = int(bm_verifiable.map(lambda ks: bool(ks & train_event_set)).sum())

    leaked_frac = (n_leaked / n_bm_events) if n_bm_events else np.nan

    return dict(
        dataset                        = name,
        method_used                    = method,
        n_benchmark_rows               = n_bm_rows,
        n_benchmark_rows_unverifiable  = n_bm_unverifiable,
        unverifiable_row_frac          = round(n_bm_unverifiable / n_bm_rows, 4) if n_bm_rows else np.nan,
        n_benchmark_events             = n_bm_events,
        n_events_leaked_into_train     = n_leaked,
        leaked_event_frac              = round(leaked_frac, 4) if not np.isnan(leaked_frac) else np.nan,
        fully_clean_event_frac         = round(1 - leaked_frac, 4) if not np.isnan(leaked_frac) else np.nan,
        n_benchmark_rows_leaked        = n_rows_leaked,
        n_train_events                 = len(train_event_set),
        n_val_events                   = len(val_event_set),
        n_train_val_event_overlap      = len(train_event_set & val_event_set),
    )


def main():
    print(f"Loading {BENCHMARK_PATH} …")
    bm_df = pd.read_csv(BENCHMARK_PATH, usecols=["dataset", "trace_name"])
    print(f"Loading {TRAIN_PATH} …")
    train_df = pd.read_csv(TRAIN_PATH, usecols=["dataset_name", "trace_name"], low_memory=False)
    print(f"Loading {VAL_PATH} …")
    val_df = pd.read_csv(VAL_PATH, usecols=["dataset_name", "trace_name"], low_memory=False)

    rows = []
    for name in BENCHMARK_DATASETS:
        print(f"\n[{name}]")
        try:
            row = audit_dataset(name, bm_df, train_df, val_df)
            for k, v in row.items():
                print(f"    {k}: {v}")
        except Exception as exc:
            print(f"    FAILED: {exc}")
            row = dict(dataset=name, method_used=f"error: {exc}")
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    print("\n" + "=" * 100)
    print("Event-level train/benchmark independence audit (per dataset)")
    print("=" * 100)
    cols = ["dataset", "method_used", "n_benchmark_events",
            "n_events_leaked_into_train", "leaked_event_frac", "unverifiable_row_frac"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
