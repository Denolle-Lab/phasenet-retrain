#!/usr/bin/env python3
"""
audit_parent_leakage.py

Cross-dataset, event-level contamination check for PUBLIC pretrained
weights (stead, instance, ethz, scedc, iquique, pisdl, ...) — a companion to
scripts/audit_event_leakage.py, which only covers our OWN fine-tunes.

scripts/domain_registry.py's split_masks() already excludes benchmark rows
whose *dataset* is nominally the same public corpus a weight was trained on
(via the benchmark's `trained_models` label column, e.g. instancecounts ->
"instance", mlaapde -> "neic"). What that CANNOT catch: the same earthquake,
recorded at a different station, showing up in BOTH that public training
corpus and a DIFFERENT benchmark dataset that isn't nominally the same
corpus — e.g. an event in the "cwa" benchmark slice (trained_models=NaN,
nominally independent of every public corpus) that also happens to be in
STEAD's ~1.27M-trace corpus.

This can't be checked by exact event-id matching (scripts/event_keys.py's
approach for the own-model audit): ids are only unique WITHIN a dataset (a
STEAD `source_id` and a CWA `source_event_id` are different, incomparable
numbering schemes — matching them by string would be either always-false,
namespaced as event_keys.py does, or a false-positive minefield if not
namespaced). The only reliable cross-provider identity is the event's
physical origin time + location, so this script does a spatiotemporal
nearest-neighbor join instead: same earthquake if origin times are within
TIME_TOL_S seconds AND locations within DIST_TOL_KM km. Tolerance is
intentionally tight — loose tolerances risk conflating distinct-but-nearby
aftershocks (which can occur seconds apart, kilometers apart, in a genuine
sequence) with true duplicates.

For each verifiable public domain (PARENT_DOMAINS below), this script loads
that domain's FULL local SeisBench copy (not just its benchmark subset,
since the pretrained model was trained on the whole public corpus) and
checks every one of the 12 BENCHMARK_DATASETS rows for a spatiotemporal
match against it.

Domains NOT covered (not silently reported as 0% leaked):
  - obst2024 ("obs" weight): every source_* column is 100% NaN locally —
    no origin time/location at all, so no fingerprint is possible.
  - neic ("neic"/"eqt_neic" weights, and mlaapde's `trained_models="neic"`
    label): local SeisBench copy is a partial/failed download — this script
    does not trigger a fresh download (see scripts/event_keys.py).

Outputs
-------
  results/parent_event_leakage_audit.csv
  results/parent_event_leakage_row_mask__<trained_on>.csv   (one per verifiable domain)

Run from repo root:
    conda activate surface
    python scripts/audit_parent_leakage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import event_keys as ek

BENCHMARK_PATH = REPO_ROOT / "notebooks" / "benchmark_manifest.csv"
OUT_CSV        = REPO_ROOT / "results" / "parent_event_leakage_audit.csv"

BENCHMARK_DATASETS = [
    "instancecounts", "stead", "ceed", "pnw", "txed", "ethz",
    "mlaapde", "aq2009gm", "pisdl", "obst2024", "vcseis", "cwa",
]

# trained_on value (scripts/domain_registry.py BASE_TRAINED_ON) -> the
# event_keys.py dataset_name whose FULL local metadata approximates that
# weight's public training corpus.
PARENT_DOMAINS = {
    "stead":    "stead",
    "instance": "instancecounts",
    "ethz":     "ethz",
    "scedc":    "scedc",
    "iquique":  "iquique",
    "pisdl":    "pisdl",
    # volpick (Zhong & Tan 2024): VCSEIS is "a subset of the dataset in
    # Zhong and Tan (2024), with the data from Japan excluded" (per
    # github.com/zhong-yy/volpick's README) -- Alaska/Hawaii/N.California/
    # Cascades regions, already downloaded locally as the "vcseis" benchmark
    # dataset's full corpus. This catches cross-dataset leakage for the
    # non-Japan portion of volpick's training data; any overlap specific to
    # its Japan-region training slice (not available locally) is NOT
    # covered and remains unverified.
    "volpick":  "vcseis",
}
UNVERIFIABLE_DOMAINS = ["obst2024", "neic"]

TIME_TOL_S  = 2.0
DIST_TOL_KM = 10.0
EARTH_R_KM  = 6371.0


def _load_latlon_time(dataset_name: str) -> pd.DataFrame:
    """trace_name, epoch_s, lat, lon for every row with a usable fingerprint
    (drops rows missing any of origin time / lat / lon)."""
    meta = ek.load_metadata(dataset_name)
    cols = [ek.FALLBACK_TIME_COL, ek.FALLBACK_LAT_COL, ek.FALLBACK_LON_COL]
    if not all(c in meta.columns for c in cols):
        return pd.DataFrame(columns=["trace_name", "epoch_s", "lat", "lon"])
    t   = pd.to_datetime(meta[ek.FALLBACK_TIME_COL], errors="coerce", utc=True)
    lat = pd.to_numeric(meta[ek.FALLBACK_LAT_COL], errors="coerce")
    lon = pd.to_numeric(meta[ek.FALLBACK_LON_COL], errors="coerce")
    out = pd.DataFrame({
        "trace_name": meta["trace_name"].values,
        "epoch_s":    t.astype("int64").values / 1e9,
        "lat":        lat.values,
        "lon":        lon.values,
    })
    out.loc[t.isna().values | lat.isna().values | lon.isna().values,
            ["epoch_s", "lat", "lon"]] = np.nan
    return out.dropna(subset=["epoch_s", "lat", "lon"])


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def _build_corpus_index(corpus_df: pd.DataFrame):
    order = np.argsort(corpus_df["epoch_s"].values)
    return dict(
        times = corpus_df["epoch_s"].values[order],
        lats  = corpus_df["lat"].values[order],
        lons  = corpus_df["lon"].values[order],
    )


def _any_match(t, lat, lon, idx):
    lo = np.searchsorted(idx["times"], t - TIME_TOL_S, side="left")
    hi = np.searchsorted(idx["times"], t + TIME_TOL_S, side="right")
    if hi <= lo:
        return False
    d = _haversine_km(lat, lon, idx["lats"][lo:hi], idx["lons"][lo:hi])
    return bool(np.any(d <= DIST_TOL_KM))


def audit_domain(trained_on, corpus_name, bm_df):
    print(f"\nLoading full {corpus_name} corpus for trained_on={trained_on!r} …")
    corpus_df = _load_latlon_time(corpus_name)
    print(f"  {corpus_name}: {len(corpus_df):,} rows with usable time+location")
    idx = _build_corpus_index(corpus_df)

    summaries, row_dfs = [], []
    for name in BENCHMARK_DATASETS:
        bm_traces = bm_df.loc[bm_df["dataset"] == name, "trace_name"]
        n_rows = len(bm_traces)
        if n_rows == 0:
            continue
        fp = _load_latlon_time(name)
        fp = fp[fp["trace_name"].isin(set(bm_traces))]
        fp_map = {r.trace_name: (r.epoch_s, r.lat, r.lon) for r in fp.itertuples()}

        verifiable, leaked = [], []
        for t in bm_traces:
            rec = fp_map.get(t)
            if rec is None:
                verifiable.append(False)
                leaked.append(False)
                continue
            verifiable.append(True)
            leaked.append(_any_match(rec[0], rec[1], rec[2], idx))

        row_dfs.append(pd.DataFrame({
            "dataset":    name,
            "trace_name": bm_traces.values,
            "verifiable": verifiable,
            "leaked":     leaked,
        }))
        n_verifiable = int(sum(verifiable))
        n_leaked     = int(sum(l for v, l in zip(verifiable, leaked) if v))
        summaries.append(dict(
            trained_on            = trained_on,
            benchmark_dataset     = name,
            n_benchmark_rows      = n_rows,
            n_verifiable          = n_verifiable,
            unverifiable_row_frac = round(1 - n_verifiable / n_rows, 4) if n_rows else np.nan,
            n_leaked              = n_leaked,
            leaked_row_frac       = round(n_leaked / n_verifiable, 4) if n_verifiable else np.nan,
        ))
        print(f"    [{name:16s}] n={n_rows:6d} verifiable={n_verifiable:6d} leaked={n_leaked:6d}")

    return pd.DataFrame(summaries), pd.concat(row_dfs, ignore_index=True)


def main():
    print(f"Loading {BENCHMARK_PATH} …")
    bm_df = pd.read_csv(BENCHMARK_PATH, usecols=["dataset", "trace_name"])

    all_summaries = []
    for trained_on, corpus_name in PARENT_DOMAINS.items():
        summary_df, row_mask_df = audit_domain(trained_on, corpus_name, bm_df)
        all_summaries.append(summary_df)
        out_path = REPO_ROOT / "results" / f"parent_event_leakage_row_mask__{trained_on}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row_mask_df.to_csv(out_path, index=False)
        print(f"  Saved row mask ({len(row_mask_df):,} rows) → {out_path}")

    for trained_on in UNVERIFIABLE_DOMAINS:
        print(f"\n{trained_on}: UNVERIFIABLE locally (see module docstring) — no row mask written.")
        all_summaries.append(pd.DataFrame([dict(
            trained_on=trained_on, benchmark_dataset=None, n_benchmark_rows=np.nan,
            n_verifiable=np.nan, unverifiable_row_frac=np.nan, n_leaked=np.nan,
            leaked_row_frac=np.nan,
        )]))

    out = pd.concat(all_summaries, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    print("\n" + "=" * 100)
    print(f"Parent (public pretrained) domain vs benchmark-dataset event overlap "
          f"(tol: ±{TIME_TOL_S:.0f}s, ±{DIST_TOL_KM:.0f}km)")
    print("=" * 100)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
