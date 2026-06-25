#!/usr/bin/env python3
"""
threshold_sweep_v7_teleseismic.py

Post-hoc threshold sweep for jma_wc_ft_global_v7 using pre-computed per-trace
peak probabilities and residuals stored in step3_results.parquet.

Rationale: model.annotate(P_threshold=t) only controls which probability peaks
are reported as picks — it does not change the underlying probability curve or
the peak location (and thus the residual).  The parquet already stores, for each
trace, the peak P probability found within ±5 s of the known arrival and the
residual at that peak.  Applying a threshold is therefore equivalent to filtering
those stored values, with no need to rerun inference.

Sweep: P/S threshold ∈ [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
Metrics (conditional on detection, i.e. p_prob ≥ threshold):
  • P-recall    — fraction of P-in-window traces detected
  • P-MAE (s)   — mean |residual| for detected picks only
  • P-outlier % — fraction of detected picks with |residual| > 1.5 s
  • n_picks     — raw count of detected picks

Run from repo root:
    conda activate surface
    python scripts/threshold_sweep_v7_teleseismic.py

Outputs (written to results/):
  threshold_sweep_v7_teleseismic.csv  — teleseismic sweep table (v7 + jma_wc ref)
  threshold_sweep_v7_allbins.csv      — same sweep across all three distance bins
  threshold_sweep_v7_teleseismic.png  — table + 3-panel line plot, teleseismic only
  threshold_sweep_v7_allbins.png      — per-bin breakdown (shows local/regional impact)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.resolve()
RESULTS_PATH = REPO_ROOT / "notebooks" / "step3_results.parquet"
OUT_DIR      = REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV_TELE = OUT_DIR / "threshold_sweep_v7_teleseismic.csv"
OUT_CSV_ALL  = OUT_DIR / "threshold_sweep_v7_allbins.csv"
OUT_PNG_TELE = OUT_DIR / "threshold_sweep_v7_teleseismic.png"
OUT_PNG_ALL  = OUT_DIR / "threshold_sweep_v7_allbins.png"

# ── Constants ─────────────────────────────────────────────────────────────────
THRESHOLDS   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
OUTLIER_THR  = 1.5   # seconds

DIST_TELESEISMIC = "teleseismic (>1500km)"
DIST_LOCAL       = "local (<150km)"
DIST_REGIONAL    = "regional (150-1500km)"
DIST_BINS        = [DIST_LOCAL, DIST_REGIONAL, DIST_TELESEISMIC]
DIST_LABELS      = {
    DIST_LOCAL:       "Local\n(<150 km)",
    DIST_REGIONAL:    "Regional\n(150–1500 km)",
    DIST_TELESEISMIC: "Teleseismic\n(>1500 km)",
}

MODELS = [
    ("jma_wc_ft_global_v7", "#E6821A", "v7 (fine-tuned)", "-"),
    ("jma_wc",              "#2C7BB6", "jma_wc (parent)",  "--"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sweep_metrics(df_p: pd.DataFrame, thresholds=THRESHOLDS) -> list[dict]:
    """
    Given a DataFrame of traces with P in window (p_in_window >= 0),
    compute metrics at each threshold.

    df_p must have columns: p_prob, p_residual_s
    """
    n_total = len(df_p)
    rows = []
    for t in thresholds:
        det = df_p[df_p["p_prob"] >= t]
        n_det = len(det)
        recall = n_det / n_total if n_total > 0 else np.nan

        if n_det > 0:
            res = det["p_residual_s"].dropna()
            p_mae     = res.abs().mean()
            p_outlier = (res.abs() > OUTLIER_THR).mean()
        else:
            p_mae = p_outlier = np.nan

        rows.append(dict(
            threshold=t,
            n_total=n_total,
            n_picks=n_det,
            p_recall=round(recall, 4),
            p_mae_s=round(p_mae, 4)     if not np.isnan(p_mae)     else np.nan,
            p_outlier_pct=round(p_outlier * 100, 2) if not np.isnan(p_outlier) else np.nan,
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

print(f"Loading {RESULTS_PATH} …")
df = pd.read_parquet(RESULTS_PATH)
print(f"  {len(df):,} rows, {df['weight'].nunique()} models")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Teleseismic-only sweep
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Teleseismic-only sweep ──")
tele_rows = []
for weight, color, label, ls in MODELS:
    wdf = df[(df["weight"] == weight) &
             (df["dist_bin"] == DIST_TELESEISMIC) &
             (df["p_in_window"] >= 0)].copy()
    print(f"  {label}: {len(wdf):,} teleseismic P traces")
    for r in sweep_metrics(wdf):
        tele_rows.append({"weight": weight, "label": label, **r})

tele_df = pd.DataFrame(tele_rows)
tele_df.to_csv(OUT_CSV_TELE, index=False)
print(f"Saved → {OUT_CSV_TELE}")

# Print table
print()
print(f"{'Model':<28} {'thr':>5}  {'P-recall':>8}  {'P-MAE (s)':>9}  "
      f"{'P-outlier %':>11}  {'n_picks':>7}  {'n_total':>7}")
print("-" * 82)
for weight, _, label, _ in MODELS:
    sub = tele_df[tele_df["weight"] == weight]
    for _, r in sub.iterrows():
        star = " ←" if abs(r.threshold - 0.30) < 0.001 else "  "
        print(f"{label:<28} {r.threshold:>5.2f}  {r.p_recall:>8.4f}  "
              f"{r.p_mae_s:>9.4f}  {r.p_outlier_pct:>11.2f}  "
              f"{int(r.n_picks):>7}  {int(r.n_total):>7}{star}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  All-bins sweep
# ══════════════════════════════════════════════════════════════════════════════

print("── All-bins sweep ──")
allbins_rows = []
for weight, color, label, ls in MODELS:
    wdf_all = df[(df["weight"] == weight) & (df["p_in_window"] >= 0)].copy()
    for dist in DIST_BINS + ["all"]:
        subset = wdf_all if dist == "all" else wdf_all[wdf_all["dist_bin"] == dist]
        for r in sweep_metrics(subset):
            allbins_rows.append({"weight": weight, "label": label, "dist_bin": dist, **r})

allbins_df = pd.DataFrame(allbins_rows)
allbins_df.to_csv(OUT_CSV_ALL, index=False)
print(f"Saved → {OUT_CSV_ALL}")

# Print per-bin table
for dist in DIST_BINS + ["all"]:
    print(f"\n  dist_bin = {dist}")
    print(f"  {'Model':<24} {'thr':>5}  {'recall':>8}  {'MAE (s)':>8}  {'outlier%':>9}  {'n_picks':>7}")
    print("  " + "-" * 65)
    for weight, _, label, _ in MODELS:
        sub = allbins_df[(allbins_df["weight"] == weight) & (allbins_df["dist_bin"] == dist)]
        for _, r in sub.iterrows():
            star = " ←" if abs(r.threshold - 0.30) < 0.001 else "  "
            print(f"  {label:<24} {r.threshold:>5.2f}  {r.p_recall:>8.4f}  "
                  f"{r.p_mae_s:>8.4f}  {r.p_outlier_pct:>9.2f}  {int(r.n_picks):>7}{star}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Figure 1 — Teleseismic: matplotlib table + 3-panel line plot
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating Figure 1: teleseismic sweep …")

fig = plt.figure(figsize=(14, 8))
fig.suptitle(
    "v7 Threshold Sweep — Teleseismic Only (>1500 km)\n"
    "Metrics are conditional on detection (p_prob ≥ threshold)",
    fontsize=12, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[1.15, 1],
                       hspace=0.55, wspace=0.38)

# ── Top row: matplotlib table ─────────────────────────────────────────────────
ax_table = fig.add_subplot(gs[0, :])
ax_table.axis("off")

col_labels = ["Threshold", "P-Recall", "P-MAE (s)", "P-Outlier %", "n_picks", "n_total"]
header_bg  = "#2C3E50"
v7_bg      = "#FEF3E2"
ref_bg     = "#EBF5FB"
alt_v7     = "#FDEBD0"
alt_ref    = "#D6EAF8"

table_data  = []
cell_colors = []

for weight, _, label, _ in MODELS:
    sub = tele_df[tele_df["weight"] == weight].copy()
    base_bg = v7_bg if "v7" in weight else ref_bg
    alt_bg  = alt_v7 if "v7" in weight else alt_ref
    for i, (_, r) in enumerate(sub.iterrows()):
        star = " ◄" if abs(r.threshold - 0.30) < 0.001 else ""
        row  = [
            f"{r.threshold:.2f}{star}",
            f"{r.p_recall:.4f}",
            f"{r.p_mae_s:.4f}",
            f"{r.p_outlier_pct:.2f} %",
            f"{int(r.n_picks):,}",
            f"{int(r.n_total):,}",
        ]
        table_data.append(row)
        bg = base_bg if i % 2 == 0 else alt_bg
        cell_colors.append([bg] * len(col_labels))

# Add model-label rows as section headers
final_data, final_colors = [], []
v7_sub  = tele_df[tele_df["weight"] == "jma_wc_ft_global_v7"]
ref_sub = tele_df[tele_df["weight"] == "jma_wc"]

for sub, label, base_bg, alt_bg in [
    (v7_sub,  "v7 (fine-tuned)", v7_bg, alt_v7),
    (ref_sub, "jma_wc (parent)", ref_bg, alt_ref),
]:
    final_data.append([f"── {label} ──", "", "", "", "", ""])
    final_colors.append(["#BDC3C7"] * 6)
    for i, (_, r) in enumerate(sub.iterrows()):
        star = " ◄" if abs(r.threshold - 0.30) < 0.001 else ""
        row = [
            f"  {r.threshold:.2f}{star}",
            f"{r.p_recall:.4f}",
            f"{r.p_mae_s:.4f}",
            f"{r.p_outlier_pct:.2f} %",
            f"{int(r.n_picks):,}",
            f"{int(r.n_total):,}",
        ]
        final_data.append(row)
        bg = base_bg if i % 2 == 0 else alt_bg
        final_colors.append([bg] * 6)

tbl = ax_table.table(
    cellText=final_data,
    colLabels=col_labels,
    cellColours=final_colors,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.35)

# Style header row
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor(header_bg)
    cell.set_text_props(color="white", fontweight="bold")

# ── Bottom row: 3-panel line plots ────────────────────────────────────────────
metrics_cfg = [
    ("p_recall",      "P-Recall",      "P-Recall",      True),
    ("p_mae_s",       "P-MAE (s)",     "P-MAE (s)",     False),
    ("p_outlier_pct", "P-Outlier (%)", "P-Outlier (%)", False),
]

for col_i, (metric, ylabel, title, higher_better) in enumerate(metrics_cfg):
    ax = fig.add_subplot(gs[1, col_i])
    for weight, color, label, ls in MODELS:
        sub = tele_df[tele_df["weight"] == weight].sort_values("threshold")
        lw = 2.5 if "v7" in weight else 1.8
        ax.plot(sub["threshold"], sub[metric],
                color=color, ls=ls, lw=lw, marker="o", ms=5, label=label)

    ax.axvline(0.30, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Detection threshold", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks(THRESHOLDS)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "p_recall":
        ax.set_ylim(0, None)
    ax.text(0.30, ax.get_ylim()[0], "thr=0.30", fontsize=6.5, color="gray",
            ha="center", va="bottom", transform=ax.get_xaxis_transform())

fig.savefig(OUT_PNG_TELE, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PNG_TELE}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Figure 2 — Per-bin breakdown: does lowering threshold hurt local/regional?
# ══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 2: all-bins breakdown …")

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 10))
fig2.suptitle(
    "v7 Threshold Sweep — All Distance Bins\n"
    "Columns: P-Recall | P-MAE (s) | P-Outlier (%)     "
    "Rows: Local | Regional | Teleseismic\n"
    "Dashed vertical line = current default threshold (0.30)",
    fontsize=10, fontweight="bold", y=1.01,
)

row_order = [DIST_LOCAL, DIST_REGIONAL, DIST_TELESEISMIC]
row_titles = ["Local (<150 km)", "Regional (150–1500 km)", "Teleseismic (>1500 km)"]

metrics_cols = [
    ("p_recall",      "P-Recall",      True),
    ("p_mae_s",       "P-MAE (s)",     False),
    ("p_outlier_pct", "P-Outlier (%)", False),
]

for row_i, (dist, dist_title) in enumerate(zip(row_order, row_titles)):
    for col_i, (metric, ylabel, higher_better) in enumerate(metrics_cols):
        ax = axes2[row_i, col_i]
        for weight, color, label, ls in MODELS:
            sub = allbins_df[
                (allbins_df["weight"] == weight) &
                (allbins_df["dist_bin"] == dist)
            ].sort_values("threshold")
            lw = 2.5 if "v7" in weight else 1.8
            ax.plot(sub["threshold"], sub[metric],
                    color=color, ls=ls, lw=lw, marker="o", ms=4, label=label)

        ax.axvline(0.30, color="gray", lw=1, ls=":", alpha=0.7)
        ax.set_xticks(THRESHOLDS)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        if row_i == 0:
            ax.set_title(ylabel, fontsize=9, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(dist_title + "\n" + ylabel, fontsize=8)
        else:
            ax.set_ylabel(ylabel, fontsize=8)
        if row_i == 2:
            ax.set_xlabel("Detection threshold", fontsize=8)
        if col_i == 0 and row_i == 0:
            ax.legend(fontsize=7.5, framealpha=0.8)
        if metric == "p_recall":
            ax.set_ylim(0, None)

plt.tight_layout()
fig2.savefig(OUT_PNG_ALL, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved → {OUT_PNG_ALL}")

print("\nDone.")
