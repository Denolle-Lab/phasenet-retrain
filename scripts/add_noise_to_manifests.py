#!/usr/bin/env python3
"""
add_noise_to_manifests.py

Appends clean noise_global traces to train.csv and val.csv.
Traces are quality-filtered using the jma_wc audit CSV so that only windows
with no detectable seismic signal are added.

noise_prephase is intentionally excluded — the audit shows 26% of those
traces have P-prob > 0.3, which confuses the model (it learns to suppress
legitimate pre-phase arrivals).

2026-09-11 (#33A): the exclusion bundle (scripts/exclusion_bundle.py) is
applied with kind="noise" before appending: held-out windows on the STATION
location and the trace start time, the 2016/2021 hold-out on the start-time
year, and quarantine of rows without a station location or start time
(data/noise_global/metadata.csv written before 2026-09-11 has no start time,
so every such row is quarantined under the default policy). Historical
manifests listed in data/manifest_checksums.csv are refused. The append is
recorded in <manifests-dir>/provenance.json.

Usage:
    python scripts/add_noise_to_manifests.py
    python scripts/add_noise_to_manifests.py --manifests-dir data/manifests_v2
    python scripts/add_noise_to_manifests.py --p-thresh 0.1 --s-thresh 0.1
"""

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exclusion_bundle as eb  # noqa: E402  (#33A)

REPO_ROOT  = Path(__file__).parent.parent
NOISE_META  = REPO_ROOT / "data" / "noise_global" / "metadata.csv"
NOISE_AUDIT = REPO_ROOT / "data" / "noise_audit" / "noise_global_audit.csv"

TRAIN_FRAC = 0.90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-dir", default="data/manifests",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--p-thresh", type=float, default=0.1,
                        help="Max jma_wc P-probability to accept as clean noise (default 0.1)")
    parser.add_argument("--s-thresh", type=float, default=0.1,
                        help="Max jma_wc S-probability to accept as clean noise (default 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-uncertified-bundle", action="store_true",
                        help="Proceed even if the exclusion bundle does not certify every source snapshot")
    args = parser.parse_args()

    random.seed(args.seed)

    manifests_dir = REPO_ROOT / args.manifests_dir
    train_csv = manifests_dir / "train.csv"
    val_csv   = manifests_dir / "val.csv"

    if not NOISE_META.exists():
        raise FileNotFoundError(f"Noise metadata not found: {NOISE_META}")
    if not NOISE_AUDIT.exists():
        raise FileNotFoundError(f"Noise audit not found: {NOISE_AUDIT}\n"
                                "Run scripts/audit_noise_picks.py first.")

    noise_meta  = pd.read_csv(NOISE_META)
    noise_audit = pd.read_csv(NOISE_AUDIT)

    # Quality filter: keep only genuinely clean windows
    clean = noise_audit[
        (noise_audit["max_p_prob"] < args.p_thresh) &
        (noise_audit["max_s_prob"] < args.s_thresh)
    ]
    noise_meta = noise_meta[noise_meta["trace_name"].isin(clean["trace_name"])]

    print(f"noise_global total     : {pd.read_csv(NOISE_META).shape[0]:,}")
    print(f"After quality filter   : {len(noise_meta):,}  "
          f"(P<{args.p_thresh}, S<{args.s_thresh})")

    # ── exclusion bundle (#33A): windows on station location + start time ────
    bundle = eb.load_bundle(require_certified=not args.allow_uncertified_bundle)
    eb.assert_not_historical(train_csv)
    eb.assert_not_historical(val_csv)
    noise_meta, ex_report = eb.apply_exclusions(
        noise_meta, bundle, kind="noise", dataset="noise_global",
        station_lat_col="latitude", station_lon_col="longitude", start_col="starttime")
    print(f"After exclusion bundle : {len(noise_meta):,}  "
          f"(in-window {ex_report['n_in_window']:,}, {sorted(eb.hs.HOLDOUT_YEARS)} start {ex_report['n_year_holdout']:,}, "
          f"quarantined unknown {ex_report['n_quarantined_unknown']:,}, "
          f"kept-flagged {ex_report['n_unknown_kept_flagged']:,}; bundle {bundle['sha256'][:12]})")
    if len(noise_meta) == 0:
        print("Nothing to add after the exclusion bundle.")
        return
    print(f"Tectonic settings      : {noise_meta['tectonic_setting'].value_counts().to_dict()}")
    print(f"Regions                : {noise_meta['region'].nunique()} unique")

    rows = []
    for _, row in noise_meta.iterrows():
        rows.append({
            "dataset_name":     "noise_global",
            "trace_name":       row["trace_name"],
            "chunk":            "",
            "p_arrival_sample": "",
            "s_arrival_sample": "",
            "distance_km":      "",
            "distance_bin":     "noise",
            "p_col":            "",
            "s_col":            "",
            eb.FLAG_COL:        bool(row[eb.FLAG_COL]),
        })

    random.shuffle(rows)
    n_train   = int(len(rows) * TRAIN_FRAC)
    train_rows = rows[:n_train]
    val_rows   = rows[n_train:]

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    existing_train = set(
        train_df[train_df["dataset_name"] == "noise_global"]["trace_name"]
    ) if "noise_global" in train_df["dataset_name"].values else set()
    existing_val = set(
        val_df[val_df["dataset_name"] == "noise_global"]["trace_name"]
    ) if "noise_global" in val_df["dataset_name"].values else set()

    new_train = [r for r in train_rows if r["trace_name"] not in existing_train]
    new_val   = [r for r in val_rows   if r["trace_name"] not in existing_val]

    if not new_train and not new_val:
        print("All noise traces already in manifests — nothing to add.")
        return

    cols = list(train_df.columns)
    pd.DataFrame(new_train, columns=cols).to_csv(train_csv, mode="a", header=False, index=False)
    pd.DataFrame(new_val,   columns=cols).to_csv(val_csv,   mode="a", header=False, index=False)

    eb.append_provenance(manifests_dir, "noise_appends", {
        "script": "scripts/add_noise_to_manifests.py", "dataset": "noise_global",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle_sha256": bundle["sha256"], "bundle_certified": bundle["certified"],
        "git_commit": eb._git_commit(eb.REPO_ROOT),
        "options": {"p_thresh": args.p_thresh, "s_thresh": args.s_thresh, "seed": args.seed},
        "exclusions": ex_report, "n_added_train": len(new_train), "n_added_val": len(new_val),
    })

    print(f"\nAdded to train.csv : {len(new_train):,} noise traces")
    print(f"Added to val.csv   : {len(new_val):,} noise traces")
    print(f"New train size     : {len(train_df) + len(new_train):,}")
    print(f"New val size       : {len(val_df)   + len(new_val):,}")


if __name__ == "__main__":
    main()
