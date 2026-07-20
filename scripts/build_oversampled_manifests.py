#!/usr/bin/env python3
"""
build_oversampled_manifests.py

Derives the oversampled/filtered training-manifest variants referenced by
configs/finetune_jma_wc_global_v9/v10/v16/v17 (train_tele2x.csv), v18
(train_v18.csv), and v19 (train_p_focused.csv / val_p_focused.csv). Each is a
deterministic row-level transform of an already-built base manifest (from
build_training_dataset.py) -- no SeisBench access needed, just pandas.

  train_tele2x.csv     = base train manifest + a full resample of its
                         teleseismic rows (2x teleseismic oversampling)
  train_v18.csv        = manifests_v3/train.csv (S-balanced) + a half
                         resample of its teleseismic rows (1.5x oversampling)
  train_p_focused.csv  = base train manifest filtered to
                         distance_bin in {local, regional} (no tele, no noise)
  val_p_focused.csv    = base val manifest filtered the same way

Reverse-engineered from the manifests actually on disk (2026-07-20): the
oversample resample uses pandas' `sample(frac=..., random_state=42)`, which
exactly reproduces the existing train_v18.csv (mismatches limited to ~194
rows differing only in float32 string-formatting noise from repeated CSV
round-trips) and exactly reproduces train_tele2x.csv / train_p_focused.csv /
val_p_focused.csv byte-for-byte at the row-set level.

Usage:
  # Regenerate all four variants using the paths actually referenced by
  # the finetune configs:
  python scripts/build_oversampled_manifests.py --all

  # Or one at a time:
  python scripts/build_oversampled_manifests.py --oversample-tele \\
      --base data/manifests_v2/train.csv --factor 2.0 \\
      --out data/manifests_v2/train_tele2x.csv
  python scripts/build_oversampled_manifests.py --oversample-tele \\
      --base data/manifests_v3/train.csv --factor 1.5 \\
      --out data/manifests_v2/train_v18.csv
  python scripts/build_oversampled_manifests.py --p-focused \\
      --base data/manifests_v2/train.csv --out data/manifests_v2/train_p_focused.csv
  python scripts/build_oversampled_manifests.py --p-focused \\
      --base data/manifests_v2/val.csv --out data/manifests_v2/val_p_focused.csv
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SEED = 42

# (base manifest, factor, output path) -- the exact recipe used for every
# committed *_tele2x.csv / *_v18.csv variant referenced by configs/.
ALL_OVERSAMPLE_JOBS = [
    ("data/manifests_v2/train.csv", 2.0, "data/manifests_v2/train_tele2x.csv"),
    ("data/manifests_v3/train.csv", 1.5, "data/manifests_v2/train_v18.csv"),
]
ALL_P_FOCUSED_JOBS = [
    ("data/manifests_v2/train.csv", "data/manifests_v2/train_p_focused.csv"),
    ("data/manifests_v2/val.csv", "data/manifests_v2/val_p_focused.csv"),
]


def oversample_teleseismic(df, factor, seed=DEFAULT_SEED):
    """Return df with its teleseismic rows oversampled by `factor` (e.g. 1.5, 2.0)."""
    tele = df[df["distance_bin"] == "teleseismic"]
    extra_frac = factor - 1.0
    if extra_frac <= 0:
        raise ValueError(f"factor must be > 1.0, got {factor}")
    extra = tele.sample(frac=extra_frac, random_state=seed)
    return pd.concat([df, extra], ignore_index=True)


def filter_p_focused(df):
    """Return df restricted to local + regional rows (drop teleseismic + noise)."""
    return df[df["distance_bin"].isin(["local", "regional"])].reset_index(drop=True)


def run_oversample(base_path, factor, out_path, seed):
    df = pd.read_csv(base_path, low_memory=False)
    out = oversample_teleseismic(df, factor, seed=seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    n_tele_before = int((df["distance_bin"] == "teleseismic").sum())
    n_tele_after = int((out["distance_bin"] == "teleseismic").sum())
    print(f"{base_path} ({len(df):,} rows, {n_tele_before:,} tele) "
          f"-> {out_path} ({len(out):,} rows, {n_tele_after:,} tele, "
          f"factor {factor}x)")


def run_p_focused(base_path, out_path):
    df = pd.read_csv(base_path, low_memory=False)
    out = filter_p_focused(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{base_path} ({len(df):,} rows) -> {out_path} "
          f"({len(out):,} rows, local+regional only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
        help="Regenerate all four committed variants at their default paths")
    parser.add_argument("--oversample-tele", action="store_true",
        help="Run a single teleseismic-oversample job (needs --base/--factor/--out)")
    parser.add_argument("--p-focused", action="store_true",
        help="Run a single local+regional filter job (needs --base/--out)")
    parser.add_argument("--base", help="Input manifest CSV")
    parser.add_argument("--factor", type=float, help="Oversample factor, e.g. 1.5 or 2.0")
    parser.add_argument("--out", help="Output manifest CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.all:
        for base, factor, out in ALL_OVERSAMPLE_JOBS:
            run_oversample(base, factor, out, args.seed)
        for base, out in ALL_P_FOCUSED_JOBS:
            run_p_focused(base, out)
    elif args.oversample_tele:
        if not (args.base and args.factor and args.out):
            parser.error("--oversample-tele requires --base, --factor, and --out")
        run_oversample(args.base, args.factor, args.out, args.seed)
    elif args.p_focused:
        if not (args.base and args.out):
            parser.error("--p-focused requires --base and --out")
        run_p_focused(args.base, args.out)
    else:
        parser.error("specify --all, --oversample-tele, or --p-focused")
