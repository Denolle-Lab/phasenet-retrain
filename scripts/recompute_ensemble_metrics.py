#!/usr/bin/env python3
"""
Recompute ensemble metrics with corrected cross_domain exclusion.
No inference — reads existing parquet, resolves trained_on via
scripts/domain_registry.py, overwrites eval_ensemble_eqt.csv.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT      = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from domain_registry import (
    split_masks, clean_holdout_mask, is_own_model, parent_clean_cross_domain_mask,
)
from metrics import compute_metrics as _compute_metrics

RESULTS_PATH   = REPO_ROOT / "notebooks" / "step3_results.parquet"
OUT_CSV        = REPO_ROOT / "results" / "eval_ensemble_eqt.csv"
ENSEMBLE_LABEL = "eqt_ensemble_volpick_nc"
THRESHOLD      = 0.3


def compute_metrics(df, weight_name, split, dist_label="all"):
    return _compute_metrics(df, weight_name, split, dist_label,
                             p_threshold=THRESHOLD, s_threshold=THRESHOLD)


results_df = pd.read_parquet(RESULTS_PATH)
DIST_BINS  = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]

# trained_on resolution now lives in scripts/domain_registry.py (single
# source of truth — ensemble trained_on="stead" via COMPOSITE_TRAINED_ON,
# same STEAD exclusion as its original_nc component).
COMPARE_WEIGHTS = [
    ENSEMBLE_LABEL,
    "eqt_volpick",
    "eqt_original_nonconservative",
    "jma_wc_ft_global_v7",
    "jma_wc",
    "eqt_scedc",
    "eqt_instance",
]

metrics_rows = []
for weight_name in COMPARE_WEIGHTS:
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
print(f"Saved → {OUT_CSV}")

cross = metrics_df[(metrics_df["split"] == "cross_domain") &
                   (metrics_df["dist_bin"] == "all")].copy()
cross = cross.sort_values("p_mae_s")

print("\n" + "=" * 100)
print("Corrected cross_domain evaluation (ensemble excludes STEAD same as original_nc)")
print("%-38s %7s %7s %7s %7s %6s %7s %6s" % (
    "weight", "P-MAE", "S-MAE", "P-rec", "S-rec", "MCC", "P-out%", "n"))
print("-" * 100)
for _, r in cross.iterrows():
    marker = " [ENSEMBLE]" if r.weight == ENSEMBLE_LABEL else \
             " *" if r.weight == "jma_wc_ft_global_v7" else ""
    pm = "%.4f" % r.p_mae_s   if not np.isnan(r.p_mae_s)   else "  --  "
    sm = "%.4f" % r.s_mae_s   if not np.isnan(r.s_mae_s)   else "  --  "
    pr = "%.4f" % r.p_recall  if not np.isnan(r.p_recall)  else "  --  "
    sr = "%.4f" % r.s_recall  if not np.isnan(r.s_recall)  else "  --  "
    mc = "%.4f" % r.mcc       if not np.isnan(r.mcc)       else "  --  "
    po = "%.2f" % (r.p_outlier * 100) if not np.isnan(r.p_outlier) else "  -- "
    print("  %-36s %7s %7s %7s %7s %6s %7s %6d" % (
        r.weight + marker, pm, sm, pr, sr, mc, po, int(r.n_traces)))

print("\n── Per-distance bin (cross_domain) ───────────────────────────────────────────")
KEY_MODELS = [ENSEMBLE_LABEL, "eqt_volpick", "eqt_original_nonconservative", "jma_wc_ft_global_v7"]
for metric, label in [("p_mae_s", "P-MAE (s)"), ("p_recall", "P-Recall"), ("s_recall", "S-Recall")]:
    print(f"\n  {label}")
    print("  %-38s %12s %14s %16s %8s" % ("", "local", "regional", "teleseismic", "all"))
    for m in KEY_MODELS:
        vals = []
        for b in DIST_BINS:
            r = metrics_df[(metrics_df["weight"] == m) &
                           (metrics_df["split"] == "cross_domain") &
                           (metrics_df["dist_bin"] == b)]
            if r.empty or np.isnan(r.iloc[0][metric]):
                vals.append("  --  ")
            else:
                vals.append("%.3f" % r.iloc[0][metric])
        marker = " [ENS]" if m == ENSEMBLE_LABEL else \
                 " *" if m == "jma_wc_ft_global_v7" else ""
        print("  %-38s %12s %14s %16s %8s" % (
            (m + marker)[:38], vals[0], vals[1], vals[2], vals[3]))
