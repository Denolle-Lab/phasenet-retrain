#!/usr/bin/env python3
"""
recompute_eqt_metrics.py

Regenerates results/eval_eqtransformer.csv from the already-inferred
notebooks/step3_results.parquet cache, using the corrected split logic in
scripts/domain_registry.py — no EQTransformer model loading, no
re-inference (that's the expensive part; this script skips it entirely).

Mirrors the metrics section of scripts/eval_eqtransformer.py exactly (same
thresholds, same MCC/outlier definitions, same COMPARE_WEIGHTS list feeding
results/model_evaluation_results.pptx via scripts/make_results_slides.py).

Run from repo root:
    conda activate surface
    python scripts/recompute_eqt_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from domain_registry import (
    split_masks, clean_holdout_mask, is_own_model, parent_clean_cross_domain_mask,
)
from metrics import compute_metrics as _compute_metrics

RESULTS_PATH = REPO_ROOT / "notebooks" / "step3_results.parquet"
OUT_CSV      = REPO_ROOT / "results" / "eval_eqtransformer.csv"

THRESHOLD   = 0.3
DIST_BINS   = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]

# Same weight names as scripts/eval_eqtransformer.py's EQT_WEIGHTS.
EQT_WEIGHT_NAMES = [
    "original", "original_nonconservative", "stead", "instance", "neic",
    "ethz", "scedc", "iquique", "pnw", "geofon", "lendb", "obs", "volpick",
]


def compute_metrics(df, weight_name, split, dist_label="all"):
    return _compute_metrics(
        df, weight_name, split, dist_label,
        p_threshold=THRESHOLD, s_threshold=THRESHOLD, tier="EQT",
    )


def main():
    old = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else None

    print(f"Loading {RESULTS_PATH} …")
    results_df = pd.read_parquet(RESULTS_PATH)

    compare_weights = (
        [f"eqt_{w}" for w in EQT_WEIGHT_NAMES if f"eqt_{w}" in results_df["weight"].unique()]
        + ["jma_wc", "jma_wc_ft_global_v7", "instance", "stead", "neic"]
    )

    metrics_rows = []
    for weight_name in compare_weights:
        wdf = results_df[results_df["weight"] == weight_name]
        if len(wdf) == 0:
            print(f"  WARNING: no rows for {weight_name}")
            continue
        _, cross_mask = split_masks(wdf, weight_name)
        own_model = is_own_model(weight_name)
        clean_mask = clean_holdout_mask(wdf, weight_name) if own_model else None
        parent_clean_mask = None if own_model else parent_clean_cross_domain_mask(wdf, weight_name)

        for dist in DIST_BINS:
            sub   = wdf if dist == "all" else wdf[wdf["dist_bin"] == dist]
            sub_x = sub[cross_mask.reindex(sub.index, fill_value=True)]
            splits = [(sub, "all"), (sub_x, "cross_domain")]
            if own_model:
                sub_clean = sub[clean_mask.reindex(sub.index, fill_value=False)]
                splits.append((sub_clean, "clean_holdout"))
            if parent_clean_mask is not None:
                sub_parent_clean = sub[parent_clean_mask.reindex(sub.index, fill_value=False)]
                splits.append((sub_parent_clean, "cross_domain_clean"))
            for df_s, split in splits:
                row = compute_metrics(df_s, weight_name, split, dist)
                if row:
                    metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_CSV, index=False)
    print(f"Saved metrics → {OUT_CSV}")

    cross = metrics_df[(metrics_df["split"] == "cross_domain") &
                        (metrics_df["dist_bin"] == "all")].sort_values("p_mae_s")
    print("\nEQTransformer vs PhaseNet baselines — cross_domain, all distances (corrected split)")
    print(cross[["weight", "n_traces", "p_mae_s", "s_mae_s", "p_recall", "mcc",
                 "p_outlier"]].to_string(index=False))

    if old is not None:
        print("\nBefore vs after n_traces — cross_domain, all distances, own fine-tunes:")
        for w in ["jma_wc", "jma_wc_ft_global_v7"]:
            old_row = old[(old["weight"] == w) & (old["split"] == "cross_domain") & (old["dist_bin"] == "all")]
            new_row = metrics_df[(metrics_df["weight"] == w) & (metrics_df["split"] == "cross_domain") & (metrics_df["dist_bin"] == "all")]
            old_n = int(old_row["n_traces"].iloc[0]) if len(old_row) else None
            new_n = int(new_row["n_traces"].iloc[0]) if len(new_row) else 0
            print(f"  {w:28s}  n_traces: {old_n} -> {new_n}")


if __name__ == "__main__":
    main()
