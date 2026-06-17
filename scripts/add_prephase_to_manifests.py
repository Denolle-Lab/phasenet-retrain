#!/usr/bin/env python3
"""
add_prephase_to_manifests.py

Reads data/noise_prephase/metadata.csv and appends pre-phase noise traces
to data/manifests/train.csv (90%) and val.csv (10%).

Run from repo root after build_prephase_noise.py:
    python scripts/add_prephase_to_manifests.py
"""

import random
from pathlib import Path

import pandas as pd

random.seed(42)

REPO_ROOT  = Path(__file__).parent.parent
NOISE_META = REPO_ROOT / "data" / "noise_prephase" / "metadata.csv"
TRAIN_CSV  = REPO_ROOT / "data" / "manifests" / "train.csv"
VAL_CSV    = REPO_ROOT / "data" / "manifests" / "val.csv"

TRAIN_FRAC = 0.90


def main():
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
        })

    random.shuffle(rows)
    n_train = int(len(rows) * TRAIN_FRAC)
    train_rows = rows[:n_train]
    val_rows   = rows[n_train:]

    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    val_df   = pd.read_csv(VAL_CSV,   low_memory=False)

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

    cols = list(train_df.columns)
    pd.DataFrame(new_train, columns=cols).to_csv(
        TRAIN_CSV, mode="a", header=False, index=False
    )
    pd.DataFrame(new_val, columns=cols).to_csv(
        VAL_CSV, mode="a", header=False, index=False
    )

    print(f"\nAdded to train.csv : {len(new_train):,} pre-phase noise traces")
    print(f"Added to val.csv   : {len(new_val):,} pre-phase noise traces")
    print(f"New train.csv size : {len(train_df) + len(new_train):,}")
    print(f"New val.csv size   : {len(val_df)   + len(new_val):,}")


if __name__ == "__main__":
    main()
