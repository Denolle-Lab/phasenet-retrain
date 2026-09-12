#!/usr/bin/env python3
"""
add_prephase_to_manifests.py

Reads data/noise_prephase/metadata.csv and appends pre-phase noise traces
to <manifests-dir>/train.csv (90%) and val.csv (10%).

2026-09-11 (#33A): the exclusion bundle (scripts/exclusion_bundle.py) is
applied with kind="noise" before appending (held-out windows on the station
location and the trace start time, 2016/2021 start years, quarantine of rows
without them; metadata written before 2026-09-11 has no starttime column, so
every such row is quarantined under the default policy). Historical
manifests listed in data/manifest_checksums.csv are refused; the append is
recorded in <manifests-dir>/provenance.json.

Schema: the new rows are written in each manifest's own column order, and a
manifest whose header lacks independence_unverified (built before #33A) is
refused (exclusion_bundle.ManifestSchemaError) rather than rewritten with
the column added, because an append never touches an existing row: the rows
a manifest was built with stay byte-identical whether or not it is listed in
manifest_checksums.csv, and the flag of an --allow-unknown bundle is never
dropped silently. Rebuild the manifest with build_training_dataset.py or copy
it to a new directory with the column added (False for every existing row).

Run from repo root after build_prephase_noise.py:
    python scripts/add_prephase_to_manifests.py [--manifests-dir data/manifests]
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
NOISE_META = REPO_ROOT / "data" / "noise_prephase" / "metadata.csv"

TRAIN_FRAC = 0.90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-dir", default="data/manifests",
                        help="Directory containing train.csv / val.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-uncertified-bundle", action="store_true",
                        help="Proceed even if the exclusion bundle does not certify every source snapshot")
    args = parser.parse_args()
    random.seed(args.seed)

    manifests_dir = REPO_ROOT / args.manifests_dir
    TRAIN_CSV = manifests_dir / "train.csv"
    VAL_CSV   = manifests_dir / "val.csv"

    if not NOISE_META.exists():
        raise FileNotFoundError(
            f"Pre-phase noise metadata not found: {NOISE_META}\n"
            "Run scripts/build_prephase_noise.py first."
        )

    noise_df = pd.read_csv(NOISE_META)
    print(f"Pre-phase noise traces available : {len(noise_df):,}")
    print(f"Source datasets : {noise_df['source_dataset'].value_counts().to_dict()}")
    print(f"Tectonic        : {noise_df['tectonic_setting'].value_counts().to_dict()}")
    print(f"Regions         : {noise_df['region'].nunique()} unique")

    # ── exclusion bundle (#33A): windows on station location + start time ────
    bundle = eb.load_bundle(require_certified=not args.allow_uncertified_bundle)
    eb.assert_not_historical(TRAIN_CSV)
    eb.assert_not_historical(VAL_CSV)
    noise_df, ex_report = eb.apply_exclusions(
        noise_df, bundle, kind="noise", dataset="noise_prephase",
        station_lat_col="latitude", station_lon_col="longitude",
        start_col="starttime" if "starttime" in noise_df.columns else None)
    print(f"After exclusion bundle : {len(noise_df):,}  "
          f"(in-window {ex_report['n_in_window']:,}, {sorted(eb.hs.HOLDOUT_YEARS)} start {ex_report['n_year_holdout']:,}, "
          f"quarantined unknown {ex_report['n_quarantined_unknown']:,}, "
          f"kept-flagged {ex_report['n_unknown_kept_flagged']:,}; bundle {bundle['sha256'][:12]})")
    if len(noise_df) == 0:
        print("Nothing to add after the exclusion bundle.")
        return

    rows = []
    for _, row in noise_df.iterrows():
        rows.append({
            "dataset_name":      "noise_prephase",
            "trace_name":        row["trace_name"],
            "chunk":             "",
            "p_arrival_sample":  "",
            "s_arrival_sample":  "",
            "distance_km":       "",
            "distance_bin":      "noise",
            "p_col":             "",
            "s_col":             "",
            eb.FLAG_COL:         bool(row[eb.FLAG_COL]),
        })

    random.shuffle(rows)
    n_train = int(len(rows) * TRAIN_FRAC)
    train_rows = rows[:n_train]
    val_rows   = rows[n_train:]

    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    val_df   = pd.read_csv(VAL_CSV,   low_memory=False)
    # refuse before any write: a column the header lacks would be dropped
    row_cols = list(rows[0].keys())
    eb.assert_append_columns(train_df.columns, row_cols, TRAIN_CSV)
    eb.assert_append_columns(val_df.columns, row_cols, VAL_CSV)

    existing_train = set(
        train_df[train_df["dataset_name"] == "noise_prephase"]["trace_name"]
    ) if "noise_prephase" in train_df["dataset_name"].values else set()
    existing_val = set(
        val_df[val_df["dataset_name"] == "noise_prephase"]["trace_name"]
    ) if "noise_prephase" in val_df["dataset_name"].values else set()

    new_train = [r for r in train_rows if r["trace_name"] not in existing_train]
    new_val   = [r for r in val_rows   if r["trace_name"] not in existing_val]

    if not new_train and not new_val:
        print("All pre-phase noise traces already in manifests — nothing to add.")
        return

    pd.DataFrame(new_train, columns=list(train_df.columns)).to_csv(
        TRAIN_CSV, mode="a", header=False, index=False
    )
    pd.DataFrame(new_val, columns=list(val_df.columns)).to_csv(
        VAL_CSV, mode="a", header=False, index=False
    )

    eb.append_provenance(manifests_dir, "noise_appends", {
        "script": "scripts/add_prephase_to_manifests.py", "dataset": "noise_prephase",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle_sha256": bundle["sha256"], "bundle_certified": bundle["certified"],
        "git_commit": eb._git_commit(eb.REPO_ROOT), "options": {"seed": args.seed},
        "exclusions": ex_report, "n_added_train": len(new_train), "n_added_val": len(new_val),
    })

    print(f"\nAdded to train.csv : {len(new_train):,} pre-phase noise traces")
    print(f"Added to val.csv   : {len(new_val):,} pre-phase noise traces")
    print(f"New train.csv size : {len(train_df) + len(new_train):,}")
    print(f"New val.csv size   : {len(val_df)   + len(new_val):,}")


if __name__ == "__main__":
    main()
