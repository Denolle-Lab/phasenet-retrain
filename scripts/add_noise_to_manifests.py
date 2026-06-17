#!/usr/bin/env python3
"""
add_noise_to_manifests.py

Appends clean noise_global traces to train.csv and val.csv.
Traces are quality-filtered using the jma_wc audit CSV so that only windows
with no detectable seismic signal are added.

noise_prephase is intentionally excluded — the audit shows 26% of those
traces have P-prob > 0.3, which confuses the model (it learns to suppress
legitimate pre-phase arrivals).

Usage:
    python scripts/add_noise_to_manifests.py
    python scripts/add_noise_to_manifests.py --manifests-dir data/manifests_v2
    python scripts/add_noise_to_manifests.py --p-thresh 0.1 --s-thresh 0.1
"""

import argparse
import random
from pathlib import Path

import pandas as pd

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

    print(f"\nAdded to train.csv : {len(new_train):,} noise traces")
    print(f"Added to val.csv   : {len(new_val):,} noise traces")
    print(f"New train size     : {len(train_df) + len(new_train):,}")
    print(f"New val size       : {len(val_df)   + len(new_val):,}")


if __name__ == "__main__":
    main()
