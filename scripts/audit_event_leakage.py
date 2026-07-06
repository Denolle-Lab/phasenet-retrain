#!/usr/bin/env python3
"""
audit_event_leakage.py

Read-only proof of train/benchmark independence at the EVENT level, not
just the trace_name level. scripts/build_training_dataset.py's
load_benchmark_exclusions() already guarantees 0 exact trace_name overlap
between our training manifests and notebooks/benchmark_manifest.csv
(verified separately). What that mechanism CANNOT catch: the same
earthquake, recorded at a different station, landing in both the training
manifest and the benchmark under a different trace_name.

This script does not regenerate any manifest — it only reads each manifest
family's existing train/val CSVs and notebooks/benchmark_manifest.csv as
they exist today, and cross-references event identities (scripts/event_keys.py).

Runs once PER MANIFEST FAMILY, not just data/manifests/. Different
jma_wc_ft_* fine-tunes trained on materially different manifests (see
scripts/domain_registry.py's WEIGHT_MANIFESTS, parsed from
configs/finetune_jma_wc*.yaml) — e.g. jma_wc_ft_global_v7 trained on
data/manifests_v2/train.csv (527k rows, includes obs/meier2019jgr/ross2018gpd)
while data/manifests/train.csv (371k rows) does not. Checking every weight's
clean_holdout against a single hardcoded manifest silently mis-scores every
weight trained on a different one. Each unique (train, val) pair actually
referenced by a finetune config gets its own row-mask output, keyed by
domain_registry.family_key_for_manifest() so scripts/domain_registry.py's
clean_holdout_mask() can look up the right one per weight_name.

`obst2024` has no derivable event identity at all in the cached SeisBench
copy (every source_* column is 100% NaN) — it is reported as
`method_used=unverifiable` with leaked_event_frac=NaN rather than a
misleadingly clean 0%.

Run from repo root:
    conda activate surface
    python scripts/audit_event_leakage.py

Outputs (one row-mask CSV per manifest family; summary covers all families)
-------
  results/event_leakage_audit.csv
  results/event_leakage_row_mask__<family_key>.csv   (e.g. ..._v2__train.csv)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import event_keys as ek
import domain_registry as dr

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


def _manifest_families():
    """Every distinct (train_path, val_path) pair actually used by a
    finetune config, plus which weight names use it. Reading straight from
    domain_registry.WEIGHT_MANIFESTS means this can't drift out of sync with
    what each training run really pointed at."""
    families = {}
    for weight, paths in dr.WEIGHT_MANIFESTS.items():
        key = (paths["train"], paths["val"])
        families.setdefault(key, []).append(weight)
    return families


def benchmark_keys_by_dataset(bm_df):
    """Benchmark-side event keys/method don't depend on the manifest family
    being checked — compute once and reuse across every family instead of
    recomputing event_keys.derive_event_keys() per family per dataset."""
    cache = {}
    for name in BENCHMARK_DATASETS:
        bm_traces = bm_df.loc[bm_df["dataset"] == name, "trace_name"]
        trace_map = ek.trace_key_map(name)
        method    = ek.key_method(name, ek.load_metadata(name))
        bm_keys   = bm_traces.map(lambda t: trace_map.get(t, frozenset()))
        cache[name] = dict(bm_traces=bm_traces, trace_map=trace_map,
                            method=method, bm_keys=bm_keys)
    return cache


def audit_dataset(name, bm_cache, train_df, val_df):
    bm_traces = bm_cache["bm_traces"]
    trace_map = bm_cache["trace_map"]
    method    = bm_cache["method"]
    bm_keys   = bm_cache["bm_keys"]

    train_traces = train_df.loc[train_df["dataset_name"] == name, "trace_name"]
    val_traces   = val_df.loc[val_df["dataset_name"] == name, "trace_name"]

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
    row_leaked    = bm_keys.map(lambda ks: bool(ks & train_event_set))
    n_rows_leaked = int(bm_verifiable.map(lambda ks: bool(ks & train_event_set)).sum())

    row_df = pd.DataFrame({
        "dataset":    name,
        "trace_name": bm_traces.values,
        "verifiable": (bm_keys.map(len) > 0).values,
        "leaked":     row_leaked.values,
    })

    leaked_frac = (n_leaked / n_bm_events) if n_bm_events else np.nan

    summary = dict(
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

    return summary, row_df


def main():
    print(f"Loading {BENCHMARK_PATH} …")
    bm_df = pd.read_csv(BENCHMARK_PATH, usecols=["dataset", "trace_name"])
    bm_cache_by_dataset = benchmark_keys_by_dataset(bm_df)

    families = _manifest_families()
    print(f"\n{len(families)} distinct manifest families in use:")
    for (train_path, val_path), weights in families.items():
        print(f"  {train_path.relative_to(REPO_ROOT)}  <- {', '.join(sorted(weights))}")

    all_rows = []
    for (train_path, val_path), weights in families.items():
        family_key = dr.family_key_for_manifest(train_path)
        row_mask_csv = REPO_ROOT / "results" / f"event_leakage_row_mask__{family_key}.csv"
        print(f"\n{'='*100}\nFamily: {family_key}  (weights: {', '.join(sorted(weights))})")
        print(f"Loading {train_path} …")
        train_df = pd.read_csv(train_path, usecols=["dataset_name", "trace_name"], low_memory=False)
        print(f"Loading {val_path} …")
        val_df = pd.read_csv(val_path, usecols=["dataset_name", "trace_name"], low_memory=False)

        row_dfs = []
        for name in BENCHMARK_DATASETS:
            print(f"\n[{name}]")
            try:
                row, row_df = audit_dataset(name, bm_cache_by_dataset[name], train_df, val_df)
                row["manifest_family"] = family_key
                row["weights"] = ",".join(sorted(weights))
                for k, v in row.items():
                    print(f"    {k}: {v}")
                row_dfs.append(row_df)
            except Exception as exc:
                print(f"    FAILED: {exc}")
                row = dict(dataset=name, method_used=f"error: {exc}",
                           manifest_family=family_key, weights=",".join(sorted(weights)))
            all_rows.append(row)

        row_mask_df = pd.concat(row_dfs, ignore_index=True)
        row_mask_csv.parent.mkdir(parents=True, exist_ok=True)
        row_mask_df.to_csv(row_mask_csv, index=False)
        print(f"\nSaved row-level clean/leaked mask ({len(row_mask_df):,} rows) → {row_mask_csv}")

    out = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    print("\n" + "=" * 100)
    print("Event-level train/benchmark independence audit (per dataset x manifest family)")
    print("=" * 100)
    cols = ["manifest_family", "dataset", "method_used", "n_benchmark_events",
            "n_events_leaked_into_train", "leaked_event_frac", "unverifiable_row_frac"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
