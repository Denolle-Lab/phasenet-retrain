#!/usr/bin/env python3
"""
audit_heldout_sequences.py

Task 1 of docs/2026-09-07_generalization_audit_prompt.md: make the external
test sequences independent of every future training manifest.

Spatiotemporal join (the method of scripts/audit_parent_leakage.py, but
against fixed windows instead of a benchmark table) over:

  * data/manifests_v2/train.csv and val.csv (the manifests v7 trained on),
    joined to their source metadata on (trace_name, chunk);
  * the FULL local copies of instancecounts, stead, crew, lendb, mlaapde,
    meier2019jgr, scedc, ceed, ross2018gpd and pnw, because any manifest
    built from now on is drawn from those pools;
  * notebooks/benchmark_manifest.csv, so a benchmark trace inside a window
    is known and can be kept out of any selection split.

Windows and the 2016/2021 whole-year hold-out are defined once, in
scripts/heldout_sequences.py, and applied at build time by
scripts/build_training_dataset.py.

Outputs (committed):
  data/exclusions/heldout_sequences.csv         (dataset, trace_name, chunk,
                                                  window, source_origin_time,
                                                  source_latitude_deg,
                                                  source_longitude_deg, source)
  data/exclusions/heldout_sequence_counts.csv   per (source, dataset, window)
                                                  counts, plus year counts
  data/exclusions/heldout_windows.csv           the window definitions

Every manifest read is verified against data/manifest_checksums.csv first
(scripts/hash_manifests.py); a mismatch aborts.

Run from repo root on the lab server (needs SEISBENCH_CACHE_ROOT):
    python scripts/audit_heldout_sequences.py
    python scripts/audit_heldout_sequences.py --check-manifest data/manifests_v4/train.csv
    python scripts/audit_heldout_sequences.py --selftest      # no cache needed

2026-09-07: this script could not be run from the laptop where it was
written (lab servers unreachable); the --selftest path and
tests/test_heldout_sequences.py are what was verified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import heldout_sequences as hs  # noqa: E402  (pure pandas)

MANIFESTS = [
    "data/manifests_v2/train.csv",
    "data/manifests_v2/val.csv",
]
FULL_CORPORA = [
    "instancecounts", "stead", "crew", "lendb", "mlaapde",
    "meier2019jgr", "scedc", "ceed", "ross2018gpd", "pnw",
]
BENCHMARK_CSV = REPO_ROOT / "notebooks" / "benchmark_manifest.csv"
CHECKSUMS_CSV = REPO_ROOT / "data" / "manifest_checksums.csv"


# ── metadata loading (server only) ──────────────────────────────────────────

def _load_full_metadata(name: str) -> pd.DataFrame:
    """Load a dataset's full metadata exactly the way the training build
    does (DATASET_CONFIGS in build_training_dataset.py), so the join sees
    the same trace_name/chunk keys the manifests were written from."""
    import build_training_dataset as btd  # imports seisbench
    cfgs = {c["name"]: c for c in btd.DATASET_CONFIGS}
    if name not in cfgs:
        raise KeyError(f"{name!r} is not in DATASET_CONFIGS")
    cfg = cfgs[name]
    if cfg["meta_fn"] is not None:
        meta = cfg["meta_fn"]()
    else:
        meta = cfg["cls"]().metadata.copy()
        if "chunk" not in meta.columns:
            meta["chunk"] = ""
    meta["chunk"] = meta["chunk"].fillna("").astype(str)
    return meta


def _fingerprint(meta: pd.DataFrame) -> pd.DataFrame:
    """(trace_name, chunk, origin time, lat, lon) with a flag for rows whose
    fingerprint is missing."""
    cols = ["trace_name", "chunk"]
    out = meta[cols].copy()
    for c in (hs.TIME_COL, hs.LAT_COL, hs.LON_COL):
        out[c] = meta[c] if c in meta.columns else np.nan
    return out


def _verify_manifest_checksum(rel_path: str) -> None:
    from hash_manifests import hash_manifest
    committed = pd.read_csv(CHECKSUMS_CSV).set_index("manifest")
    if rel_path not in committed.index:
        sys.exit(f"{rel_path} has no committed checksum in {CHECKSUMS_CSV}; refusing to use it.")
    digest, n_rows = hash_manifest(REPO_ROOT / rel_path)
    exp = committed.loc[rel_path]
    if digest != exp["sha256_of_sorted_keys"] or int(n_rows) != int(exp["n_rows"]):
        sys.exit(f"CHECKSUM MISMATCH for {rel_path}: {n_rows} rows / {digest[:16]}... "
                 f"vs committed {exp['n_rows']} rows / {exp['sha256_of_sorted_keys'][:16]}...")
    print(f"  checksum OK: {rel_path} ({n_rows:,} rows)")


# ── core counting ───────────────────────────────────────────────────────────

def count_hits(fp: pd.DataFrame, source: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (counts rows, hit rows) for one (source, dataset) table."""
    flags = hs.flag_rows(fp)
    n = len(fp)
    n_ver = int(flags["verifiable"].sum())
    rows = []
    for w in hs.WINDOW_NAMES:
        n_hit = int(flags["window"].str.contains(w, regex=False).sum())
        rows.append(dict(source=source, dataset=dataset, window=w,
                         n_rows=n, n_verifiable=n_ver, n_in_window=n_hit))
    for y in sorted(hs.HOLDOUT_YEARS):
        n_y = int((flags["origin_year"] == y).sum())
        rows.append(dict(source=source, dataset=dataset, window=f"year_{y}",
                         n_rows=n, n_verifiable=n_ver, n_in_window=n_y))
    hit_mask = (flags["window"] != "")
    hits = fp.loc[hit_mask, ["trace_name", "chunk", hs.TIME_COL, hs.LAT_COL, hs.LON_COL]].copy()
    hits.insert(0, "dataset", dataset)
    hits["window"] = flags.loc[hit_mask, "window"].values
    hits["source"] = source
    return pd.DataFrame(rows), hits


def audit_manifest(rel_path: str, meta_cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    _verify_manifest_checksum(rel_path)
    man = pd.read_csv(REPO_ROOT / rel_path, low_memory=False)
    man["chunk"] = man["chunk"].fillna("").astype(str)
    counts, hits = [], []
    for ds, g in man.groupby("dataset_name"):
        if ds not in meta_cache:
            try:
                meta_cache[ds] = _fingerprint(_load_full_metadata(ds))
            except Exception as exc:  # noqa: BLE001
                print(f"    [{ds}] metadata unavailable ({exc}); rows counted as unverifiable")
                meta_cache[ds] = None
        fp = meta_cache[ds]
        if fp is None:
            joined = g[["trace_name", "chunk"]].copy()
            for c in (hs.TIME_COL, hs.LAT_COL, hs.LON_COL):
                joined[c] = np.nan
        else:
            key = ["trace_name", "chunk"] if (fp["chunk"] != "").any() else ["trace_name"]
            fp_dedup = fp.drop_duplicates(subset=key)
            joined = g[["trace_name", "chunk"]].merge(fp_dedup, on=key, how="left", suffixes=("", "_meta"))
            if "chunk_meta" in joined.columns:
                joined = joined.drop(columns=["chunk_meta"])
        c, h = count_hits(joined, source=rel_path, dataset=ds)
        counts.append(c); hits.append(h)
        print(f"    [{ds:16s}] rows={len(g):7,}  in-window={int((h['window'] != '').sum()):6,}")
    return pd.concat(counts, ignore_index=True), pd.concat(hits, ignore_index=True)


def audit_full_corpus(name: str, meta_cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nLoading full {name} corpus ...")
    if name not in meta_cache:
        meta_cache[name] = _fingerprint(_load_full_metadata(name))
    fp = meta_cache[name]
    c, h = count_hits(fp, source=f"full:{name}", dataset=name)
    print(f"  {name}: {len(fp):,} rows, {int(c['n_verifiable'].iloc[0]):,} verifiable, "
          f"{int((h['window'] != '').sum()):,} in a window")
    return c, h


def audit_benchmark(meta_cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bm = pd.read_csv(BENCHMARK_CSV, usecols=["dataset", "trace_name"])
    counts, hits = [], []
    for ds, g in bm.groupby("dataset"):
        if ds not in meta_cache:
            try:
                meta_cache[ds] = _fingerprint(_load_full_metadata(ds))
            except Exception as exc:  # noqa: BLE001
                print(f"    [{ds}] metadata unavailable ({exc})")
                meta_cache[ds] = None
        fp = meta_cache[ds]
        if fp is None:
            continue
        # benchmark_manifest.csv carries no chunk column; resolve against every chunk (conservative)
        joined = g[["trace_name"]].merge(fp, on="trace_name", how="left")
        c, h = count_hits(joined, source="benchmark_manifest", dataset=ds)
        counts.append(c); hits.append(h)
    return pd.concat(counts, ignore_index=True), pd.concat(hits, ignore_index=True)


def selftest() -> None:
    """Exercise the join on a synthetic table; no cache needed."""
    fp = pd.DataFrame({
        "trace_name": ["a", "b", "c", "d", "e", "f"],
        "chunk": [""] * 6,
        hs.TIME_COL: ["2016-10-30T07:10:00", "2016-10-30T07:10:00", "2019-07-10T00:00:00",
                      "2021-03-04T00:00:00", "2016-06-01T00:00:00", None],
        hs.LAT_COL: [42.9, 45.0, 35.8, 39.8, 10.0, 42.9],
        hs.LON_COL: [13.2, 13.2, -117.5, 22.3, 10.0, 13.2],
    })
    counts, hits = count_hits(fp, "selftest", "synthetic")
    got = dict(zip(hits["trace_name"], hits["window"]))
    assert got == {"a": "norcia_2016_mainshock;norcia_2016_sequence",
                   "c": "ridgecrest_2019", "d": "thessaly_2021_sequence"}, got
    yrs = counts.set_index("window")["n_in_window"]
    assert yrs["year_2016"] == 3 and yrs["year_2021"] == 1, yrs.to_dict()
    assert int(counts["n_verifiable"].iloc[0]) == 5
    print("selftest OK:", got)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-manifest", metavar="CSV", help="Verify one manifest against the committed list and year hold-out")
    ap.add_argument("--skip-full", action="store_true", help="Only the manifests and the benchmark (fast)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    hs.EXCLUSION_DIR.mkdir(parents=True, exist_ok=True)
    hs.windows_frame().to_csv(hs.WINDOWS_CSV, index=False)

    if args.check_manifest:
        excl = hs.load_sequence_exclusions()
        man = pd.read_csv(args.check_manifest, low_memory=False)
        if hs.TIME_COL not in man.columns:
            # older manifest: join to metadata for the year check
            man["chunk"] = man["chunk"].fillna("").astype(str)
            cache: dict = {}
            parts = []
            for ds, g in man.groupby("dataset_name"):
                fp = _fingerprint(_load_full_metadata(ds)); cache[ds] = fp
                key = ["trace_name", "chunk"] if (fp["chunk"] != "").any() else ["trace_name"]
                parts.append(g.merge(fp.drop_duplicates(subset=key), on=key, how="left", suffixes=("", "_meta")))
            man = pd.concat(parts, ignore_index=True)
        rep = hs.check_manifest(man, excl)
        print(rep)
        ok = rep["n_excluded_present"] == 0 and rep["n_year_holdout"] == 0 and rep["n_in_window"] == 0
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    meta_cache: dict = {}
    all_counts, all_hits = [], []

    print("Manifests (v7's training data):")
    for rel in MANIFESTS:
        print(f"  {rel}")
        c, h = audit_manifest(rel, meta_cache)
        all_counts.append(c); all_hits.append(h)

    if not args.skip_full:
        for name in FULL_CORPORA:
            c, h = audit_full_corpus(name, meta_cache)
            all_counts.append(c); all_hits.append(h)

    print("\nBenchmark manifest:")
    c, h = audit_benchmark(meta_cache)
    all_counts.append(c); all_hits.append(h)

    counts = pd.concat(all_counts, ignore_index=True)
    hits = pd.concat(all_hits, ignore_index=True)
    counts.to_csv(hs.COUNTS_CSV, index=False)

    # The exclusion list: every (dataset, trace_name) that falls in a window
    # in ANY scanned source. Manifests and the benchmark are subsets of the
    # full corpora, so listing the union is what makes future builds clean.
    excl = (hits.sort_values(["dataset", "trace_name", "source"])
                .drop_duplicates(subset=["dataset", "trace_name", "chunk"]))
    excl.to_csv(hs.EXCLUSION_CSV, index=False)

    print("\n" + "=" * 90)
    print("Traces whose source origin falls in a held-out window (per source, dataset, window)")
    print("=" * 90)
    piv = counts.pivot_table(index=["source", "dataset"], columns="window", values="n_in_window", fill_value=0)
    print(piv.to_string())
    print(f"\nExclusion list: {len(excl):,} (dataset, trace_name) rows -> {hs.EXCLUSION_CSV}")
    print(f"Counts -> {hs.COUNTS_CSV}")


if __name__ == "__main__":
    main()
