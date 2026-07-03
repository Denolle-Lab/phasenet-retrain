#!/usr/bin/env python3
"""
recompute_step3_metrics.py

Regenerates notebooks/step3_metrics.csv from the already-inferred
notebooks/step3_results.parquet cache, using the corrected split logic in
scripts/domain_registry.py — no model loading, no re-inference.

Mirrors the metrics section of scripts/eval_finetuned.py exactly (same
thresholds, same recall-curve/outlier/MCC definitions), so this is a
metrics-only re-run of that script's tail half, not a new metric definition.

Run from repo root:
    conda activate surface
    python scripts/recompute_step3_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from domain_registry import split_masks

NB_DIR       = REPO_ROOT / "notebooks"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
METRICS_PATH = NB_DIR / "step3_metrics.csv"

THRESHOLD_P   = 0.3
THRESHOLD_S   = 0.3
OUTLIER_THR_S = 1.50

# Same tier labels as eval_finetuned.py's PHASENET_WEIGHTS (display only —
# trained_on is resolved by domain_registry now, not tracked here).
TIERS = {
    "stead": "A", "instance": "A", "neic": "A",
    "diting": "B", "obs": "B", "volpick": "B", "pisdl": "B", "phasenet_sn": "B",
    "jma": "B", "jma_wc": "B", "jma_wc_ft": "B", "jma_wc_ft_frozen": "B",
    "jma_wc_ft_noise": "B",
    "scedc": "C", "ethz": "C", "iquique": "C", "lendb": "C", "original": "C",
    "geofon": "D",
}


def tier_for(weight_name: str) -> str:
    if weight_name in TIERS:
        return TIERS[weight_name]
    if weight_name.startswith("jma_wc_ft_global_v"):
        return "B"
    return "?"


def compute_metrics(df, weight_name, split_name, dist_label, degenerate_models):
    if len(df) == 0:
        return None
    p_traces = df[df["p_in_window"] >= 0]
    s_traces = df[df["s_in_window"] >= 0]

    p_recalls, s_recalls = {}, {}
    for t in [0.1, 0.2, 0.3, 0.5, 0.7]:
        p_recalls[f"p_recall_t{int(t*10):02d}"] = (
            (p_traces["p_prob"] >= t).mean() if len(p_traces) > 0 else np.nan)
        s_recalls[f"s_recall_t{int(t*10):02d}"] = (
            (s_traces["s_prob"] >= t).mean() if len(s_traces) > 0 else np.nan)

    p_recall   = p_recalls["p_recall_t03"]
    s_recall   = s_recalls["s_recall_t03"]
    p_med_prob = p_traces["p_prob"].median() if len(p_traces) > 0 else np.nan
    s_med_prob = s_traces["s_prob"].median() if len(s_traces) > 0 else np.nan

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

    p_res = df.loc[df["p_in_window"] >= 0, "p_residual_s"].dropna()
    s_res = df.loc[df["s_in_window"] >= 0, "s_residual_s"].dropna()
    p_mae     = np.abs(p_res).mean()       if len(p_res) > 0 else np.nan
    p_rmse    = np.sqrt((p_res**2).mean()) if len(p_res) > 0 else np.nan
    s_mae     = np.abs(s_res).mean()       if len(s_res) > 0 else np.nan
    s_rmse    = np.sqrt((s_res**2).mean()) if len(s_res) > 0 else np.nan
    p_outlier = (np.abs(p_res) > OUTLIER_THR_S).mean() if len(p_res) > 0 else np.nan
    s_outlier = (np.abs(s_res) > OUTLIER_THR_S).mean() if len(s_res) > 0 else np.nan

    row = {
        "weight":        weight_name,
        "tier":          tier_for(weight_name),
        "split":         split_name,
        "dist_bin":      dist_label,
        "n_traces":      len(df),
        "degenerate":    weight_name in degenerate_models,
        "p_recall":      round(p_recall, 4)   if not np.isnan(p_recall)   else np.nan,
        "s_recall":      round(s_recall, 4)   if not np.isnan(s_recall)   else np.nan,
        "p_med_prob":    round(p_med_prob, 4) if not np.isnan(p_med_prob) else np.nan,
        "s_med_prob":    round(s_med_prob, 4) if not np.isnan(s_med_prob) else np.nan,
        "mcc":           round(mcc, 4)        if not np.isnan(mcc)        else np.nan,
        "p_mae_s":       round(p_mae, 4)      if not np.isnan(p_mae)      else np.nan,
        "p_rmse_s":      round(p_rmse, 4)     if not np.isnan(p_rmse)     else np.nan,
        "s_mae_s":       round(s_mae, 4)      if not np.isnan(s_mae)      else np.nan,
        "s_rmse_s":      round(s_rmse, 4)     if not np.isnan(s_rmse)     else np.nan,
        "p_outlier":     round(p_outlier, 4)  if not np.isnan(p_outlier)  else np.nan,
        "s_outlier":     round(s_outlier, 4)  if not np.isnan(s_outlier)  else np.nan,
        "outlier_thr_s": OUTLIER_THR_S,
    }
    for k, v in {**p_recalls, **s_recalls}.items():
        row[k] = round(v, 4) if not np.isnan(v) else np.nan
    return row


def main():
    old_metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else None

    print(f"Loading {RESULTS_PATH} …")
    results_df = pd.read_parquet(RESULTS_PATH)

    degenerate_models = set()
    for wname in results_df["weight"].unique():
        wdf = results_df[results_df["weight"] == wname]
        rec = (wdf["p_prob"] >= THRESHOLD_P).mean()
        mae = wdf["p_residual_s"].abs().mean()
        if rec > 0.99 and mae > 2.0:
            degenerate_models.add(wname)
    if degenerate_models:
        print(f"Degenerate models: {sorted(degenerate_models)}")

    print("\nComputing metrics with corrected domain_registry split logic …")
    metrics_rows = []
    dist_bins = results_df["dist_bin"].dropna().unique().tolist() + ["all"]

    for weight_name in tqdm(results_df["weight"].unique(), desc="Metrics"):
        wdf = results_df[results_df["weight"] == weight_name]
        in_mask, cross_mask = split_masks(wdf, weight_name)

        for dist_label in dist_bins:
            if dist_label == "all":
                sub_all, sub_cross, sub_in = wdf, wdf[cross_mask], wdf[in_mask]
            else:
                d_mask    = wdf["dist_bin"] == dist_label
                sub_all   = wdf[d_mask]
                sub_cross = wdf[cross_mask & d_mask]
                sub_in    = wdf[in_mask & d_mask]
            for sub, split in [(sub_all, "all"), (sub_cross, "cross_domain"), (sub_in, "in_domain")]:
                row = compute_metrics(sub, weight_name, split, dist_label, degenerate_models)
                if row:
                    metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Saved {len(metrics_df):,} metric rows → {METRICS_PATH}")

    clean = metrics_df[~metrics_df["degenerate"].fillna(False)]
    cross_all = (clean[(clean["split"] == "cross_domain") & (clean["dist_bin"] == "all")]
                 .sort_values("p_mae_s"))
    print("\nCross-domain P-MAE ranking (all distances) — corrected split:")
    print(cross_all[["weight", "tier", "n_traces", "p_mae_s", "s_mae_s", "p_recall", "mcc",
                      "p_outlier"]].to_string(index=False))

    if old_metrics is not None:
        print("\n" + "=" * 90)
        print("Before vs after — cross_domain, all distances, own fine-tunes")
        print("=" * 90)
        own_weights = [w for w in metrics_df["weight"].unique()
                       if w == "jma_wc" or w.startswith("jma_wc_ft")]
        for w in sorted(own_weights):
            old_row = old_metrics[(old_metrics["weight"] == w) &
                                   (old_metrics["split"] == "cross_domain") &
                                   (old_metrics["dist_bin"] == "all")]
            new_row = metrics_df[(metrics_df["weight"] == w) &
                                  (metrics_df["split"] == "cross_domain") &
                                  (metrics_df["dist_bin"] == "all")]
            old_n = int(old_row["n_traces"].iloc[0]) if len(old_row) else None
            new_n = int(new_row["n_traces"].iloc[0]) if len(new_row) else 0
            print(f"  {w:28s}  n_traces: {old_n} -> {new_n}")


if __name__ == "__main__":
    main()
