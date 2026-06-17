#!/usr/bin/env python3
"""
threshold_sweep_v7.py

Reads step3_results.parquet and computes, for v7 and jma_wc, the recall vs.
conditional P-MAE tradeoff across detection thresholds (0.05 – 0.70).

Conditional P-MAE = mean |residual| for traces that are BOTH:
  (a) expected to have a P pick (p_in_window >= 0)
  (b) detected at that threshold (p_prob >= threshold)

This is the practically meaningful metric: what is the timing accuracy
of the picks you actually USE at a given confidence cutoff?

Outputs
-------
  results/threshold_sweep_v7.csv   — table of recall × cond-MAE per threshold
  results/threshold_sweep_v7.png   — recall-MAE tradeoff plot
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT    = Path(__file__).parent.parent.resolve()
RESULTS_PATH = REPO_ROOT / "notebooks" / "step3_results.parquet"
OUT_CSV      = REPO_ROOT / "results" / "threshold_sweep_v7.csv"
OUT_PNG      = REPO_ROOT / "results" / "threshold_sweep_v7.png"
OUTLIER_THR  = 1.5   # seconds

MODELS = [
    ("jma_wc_ft_global_v7", "#E6821A", "v7 (best ft)", "--"),
    ("jma_wc_ft_global_v11", "#16A085", "v11 (lower T)", ":"),
    ("jma_wc_ft_global_v3",  "#FF8C42", "v3 (stable baseline)", "-."),
    ("jma_wc",               "#D25F10", "jma_wc (parent)", "-"),
]

THRESHOLDS = np.round(np.arange(0.05, 0.75, 0.05), 2)

df = pd.read_parquet(RESULTS_PATH)
# use cross-domain, all distances
bench_traces = set(df[(df["dist_bin"].notna())]["trace_name"].unique())

rows = []
for weight, color, label, ls in MODELS:
    wdf = df[df["weight"] == weight].copy()
    # only traces with a true P pick
    p_df = wdf[wdf["p_in_window"] >= 0].copy()
    n_total = len(p_df)
    if n_total == 0:
        print(f"WARNING: no traces for {weight}")
        continue

    for t in THRESHOLDS:
        det = p_df[p_df["p_prob"] >= t]
        n_det = len(det)
        recall = n_det / n_total
        if n_det > 0:
            res = det["p_residual_s"].dropna()
            cond_mae     = res.abs().mean()
            cond_outlier = (res.abs() > OUTLIER_THR).mean()
            cond_med     = res.abs().median()
        else:
            cond_mae = cond_outlier = cond_med = np.nan

        rows.append(dict(
            weight=weight, label=label,
            threshold=t, n_total=n_total, n_detected=n_det,
            recall=round(recall, 4),
            cond_p_mae_s=round(cond_mae, 4)     if not np.isnan(cond_mae)     else np.nan,
            cond_p_outlier=round(cond_outlier,4) if not np.isnan(cond_outlier) else np.nan,
            cond_p_median_s=round(cond_med, 4)  if not np.isnan(cond_med)     else np.nan,
        ))

sweep = pd.DataFrame(rows)
sweep.to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}")

# ── Print table ──────────────────────────────────────────────────────────────
print("\nThreshold sweep — cross-domain, all distances")
print(f"{'Model':<26} {'thr':>5}  {'recall':>7}  {'cond-P-MAE':>10}  {'cond-outlier':>12}")
print("-" * 70)
for _, g in sweep.groupby("weight", sort=False):
    for _, r in g.iterrows():
        star = " ←" if abs(r.threshold - 0.3) < 0.01 else ""
        print(f"{r.label:<26} {r.threshold:>5.2f}  {r.recall:>7.4f}  "
              f"{r.cond_p_mae_s:>10.4f}  {r.cond_p_outlier:>12.4f}{star}")
    print()

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Threshold sweep: recall vs. conditional P-MAE (cross-domain, all distances)",
             fontsize=12, fontweight="bold")

ax_l, ax_r = axes

for weight, color, label, ls in MODELS:
    sub = sweep[sweep["weight"] == weight].sort_values("threshold")
    ax_l.plot(sub["threshold"], sub["recall"],       color=color, ls=ls, lw=2, label=label, marker="o", ms=4)
    ax_r.plot(sub["threshold"], sub["cond_p_mae_s"], color=color, ls=ls, lw=2, label=label, marker="o", ms=4)

for ax in axes:
    ax.axvline(0.3, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Detection threshold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

ax_l.set_ylabel("P-Recall")
ax_l.set_title("Recall vs threshold")
ax_r.set_ylabel("Conditional P-MAE (s)")
ax_r.set_title("Conditional P-MAE vs threshold\n(MAE for detected picks only)")

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
