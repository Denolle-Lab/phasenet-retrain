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
from sklearn.metrics import matthews_corrcoef

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from domain_registry import split_masks

RESULTS_PATH = REPO_ROOT / "notebooks" / "step3_results.parquet"
OUT_CSV      = REPO_ROOT / "results" / "eval_eqtransformer.csv"

OUTLIER_THR = 1.5
THRESHOLD   = 0.3
DIST_BINS   = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]

# Same weight names as scripts/eval_eqtransformer.py's EQT_WEIGHTS.
EQT_WEIGHT_NAMES = [
    "original", "original_nonconservative", "stead", "instance", "neic",
    "ethz", "scedc", "iquique", "pnw", "geofon", "lendb", "obs", "volpick",
]


def compute_metrics(df, weight_name, split, dist_label="all"):
    if len(df) == 0:
        return None
    p_tr = df[df["p_in_window"] >= 0]
    s_tr = df[df["s_in_window"] >= 0]

    p_recall = (p_tr["p_prob"] >= THRESHOLD).mean() if len(p_tr) > 0 else np.nan
    s_recall = (s_tr["s_prob"] >= THRESHOLD).mean() if len(s_tr) > 0 else np.nan

    both = df[(df["p_in_window"] >= 0) & (df["s_in_window"] >= 0)]
    mcc = np.nan
    if len(both) >= 5:
        y_true = np.concatenate([np.ones(len(both)), np.zeros(len(both))])
        y_pred = np.concatenate([
            (both["p_prob"] > both["s_prob"]).astype(int).values,
            (both["s_prob"] > both["p_prob"]).astype(int).values,
        ])
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = np.nan

    p_res = p_tr["p_residual_s"].dropna()
    s_res = s_tr["s_residual_s"].dropna()

    return dict(
        weight     = weight_name,
        tier       = "EQT",
        split      = split,
        dist_bin   = dist_label,
        n_traces   = len(df),
        p_recall   = round(p_recall, 4)                             if not np.isnan(p_recall)  else np.nan,
        s_recall   = round(s_recall, 4)                             if not np.isnan(s_recall)  else np.nan,
        p_mae_s    = round(np.abs(p_res).mean(), 4)                 if len(p_res) > 0 else np.nan,
        s_mae_s    = round(np.abs(s_res).mean(), 4)                 if len(s_res) > 0 else np.nan,
        p_outlier  = round((np.abs(p_res) > OUTLIER_THR).mean(), 4) if len(p_res) > 0 else np.nan,
        s_outlier  = round((np.abs(s_res) > OUTLIER_THR).mean(), 4) if len(s_res) > 0 else np.nan,
        mcc        = round(mcc, 4)                                  if not np.isnan(mcc)       else np.nan,
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

        for dist in DIST_BINS:
            sub   = wdf if dist == "all" else wdf[wdf["dist_bin"] == dist]
            sub_x = sub[cross_mask.reindex(sub.index, fill_value=True)]
            for df_s, split in [(sub, "all"), (sub_x, "cross_domain")]:
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
