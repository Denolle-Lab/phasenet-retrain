#!/usr/bin/env python3
"""
compare_v7_thresholds.py

Complete benchmark evaluation producing a three-way side-by-side comparison:
  col A — jma_wc        @ thr P=0.30 / S=0.30  (parent model, current default)
  col B — v7            @ thr P=0.30 / S=0.30  (fine-tuned, current default)
  col C — v7            @ thr P=0.10 / S=0.10  (fine-tuned, low threshold)

Metrics match the Münchmeyer et al. (2022) benchmark definition exactly:
  • P/S-recall   — fraction of in-window arrivals detected at the given threshold
  • P/S-MAE (s)  — unconditional mean |residual| (all in-window traces, incl. misses)
  • P/S-outlier% — unconditional fraction with |residual| > 1.5 s
  • MCC          — Matthews Correlation Coefficient for P vs S phase ID
  • n_traces     — number of benchmark traces in each bin

All metrics are computed from step3_results.parquet for cross_domain split.
In-domain rows are also included for completeness.

Run from repo root:
    conda activate surface
    python scripts/compare_v7_thresholds.py

Outputs
-------
  results/compare_v7_thresholds.csv   — full numeric table
  results/compare_v7_thresholds.png   — formatted matplotlib table figure
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.resolve()
RESULTS_PATH = REPO_ROOT / "notebooks" / "step3_results.parquet"
INDEX_PATH   = REPO_ROOT / "notebooks" / "benchmark_waveforms_index.csv"
OUT_CSV      = REPO_ROOT / "results" / "compare_v7_thresholds.csv"
OUT_PNG      = REPO_ROOT / "results" / "compare_v7_thresholds.png"
OUTLIER_THR  = 1.50   # seconds — Münchmeyer et al. 2022

# ── Distance bins (display order) ─────────────────────────────────────────────
DIST_BINS   = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]
DIST_SHORT  = {
    "local (<150km)":            "Local\n(<150 km)",
    "regional (150-1500km)":     "Regional\n(150–1500 km)",
    "teleseismic (>1500km)":     "Teleseismic\n(>1500 km)",
    "all":                       "All\ndistances",
}

# ── Three configurations to compare ───────────────────────────────────────────
CONFIGS = [
    dict(weight="jma_wc",              thr_p=0.30, thr_s=0.30,
         label="jma_wc\nthr=0.30",     short="jma_wc\n@0.30",  color="#2C7BB6"),
    dict(weight="jma_wc_ft_global_v7", thr_p=0.30, thr_s=0.30,
         label="v7\nthr=0.30",         short="v7\n@0.30",       color="#E6821A"),
    dict(weight="jma_wc_ft_global_v7", thr_p=0.10, thr_s=0.10,
         label="v7\nthr=0.10",         short="v7\n@0.10",       color="#27AE60"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame, thr_p: float, thr_s: float) -> dict:
    """
    Compute the full Münchmeyer et al. (2022) metric set for a single-model
    DataFrame slice, at given detection thresholds.

    MAE and outlier% are UNCONDITIONAL — computed over all traces that have
    a true arrival in window, regardless of whether that arrival is detected.
    This matches the benchmark definition exactly and means these values are
    identical across threshold choices for the same model.

    Recall is threshold-dependent: the fraction of in-window arrivals whose
    stored peak probability meets or exceeds thr_p / thr_s.
    """
    p_traces = df[df["p_in_window"] >= 0].copy()
    s_traces = df[df["s_in_window"] >= 0].copy()

    n_p = len(p_traces)
    n_s = len(s_traces)

    # ── Recall (threshold-dependent) ──────────────────────────────────────────
    p_recall = (p_traces["p_prob"] >= thr_p).mean() if n_p > 0 else np.nan
    s_recall = (s_traces["s_prob"] >= thr_s).mean() if n_s > 0 else np.nan

    # ── MCC (threshold-independent — compares p_prob vs s_prob) ───────────────
    both = df[(df["p_in_window"] >= 0) & (df["s_in_window"] >= 0)].copy()
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

    # ── MAE / RMSE / outlier (unconditional) ──────────────────────────────────
    p_res = p_traces["p_residual_s"].dropna()
    s_res = s_traces["s_residual_s"].dropna()

    p_mae     = np.abs(p_res).mean()       if len(p_res) > 0 else np.nan
    p_rmse    = np.sqrt((p_res**2).mean()) if len(p_res) > 0 else np.nan
    p_outlier = (np.abs(p_res) > OUTLIER_THR).mean() if len(p_res) > 0 else np.nan

    s_mae     = np.abs(s_res).mean()       if len(s_res) > 0 else np.nan
    s_rmse    = np.sqrt((s_res**2).mean()) if len(s_res) > 0 else np.nan
    s_outlier = (np.abs(s_res) > OUTLIER_THR).mean() if len(s_res) > 0 else np.nan

    return dict(
        n_traces   = len(df),
        p_recall   = round(p_recall, 4)   if not np.isnan(p_recall)   else np.nan,
        s_recall   = round(s_recall, 4)   if not np.isnan(s_recall)   else np.nan,
        p_mae_s    = round(p_mae, 4)      if not np.isnan(p_mae)      else np.nan,
        s_mae_s    = round(s_mae, 4)      if not np.isnan(s_mae)      else np.nan,
        p_rmse_s   = round(p_rmse, 4)     if not np.isnan(p_rmse)     else np.nan,
        s_rmse_s   = round(s_rmse, 4)     if not np.isnan(s_rmse)     else np.nan,
        p_outlier  = round(p_outlier, 4)  if not np.isnan(p_outlier)  else np.nan,
        s_outlier  = round(s_outlier, 4)  if not np.isnan(s_outlier)  else np.nan,
        mcc        = round(mcc, 4)        if not np.isnan(mcc)        else np.nan,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Load + filter to cross_domain split
# ══════════════════════════════════════════════════════════════════════════════

print(f"Loading {RESULTS_PATH} …")
df_all = pd.read_parquet(RESULTS_PATH)

# Build in_domain / cross_domain masks for models trained on a known dataset.
# jma and jma_wc have no known trained_on dataset → all traces are cross_domain.
TRAINED_ON = {
    "stead":     "stead",
    "instance":  "instance",
    "neic":      "neic",
    "scedc":     "scedc",
    "ethz":      "ethz",
    "iquique":   "iquique",
    "obs":       "obst2024",
    "pisdl":     "pisdl",
}

def get_split_mask(wdf: pd.DataFrame, weight: str) -> pd.Series:
    """True = cross_domain row for this model."""
    trained_on = TRAINED_ON.get(weight)
    if trained_on is None:
        return pd.Series(True, index=wdf.index)  # all cross_domain
    return ~wdf["trained_models"].str.contains(trained_on, na=False, regex=False)

print(f"  {len(df_all):,} rows, {df_all['weight'].nunique()} models\n")


# ══════════════════════════════════════════════════════════════════════════════
# Run metrics for every config × dist_bin × split
# ══════════════════════════════════════════════════════════════════════════════

rows = []
for cfg in CONFIGS:
    wdf     = df_all[df_all["weight"] == cfg["weight"]].copy()
    cd_mask = get_split_mask(wdf, cfg["weight"])   # cross_domain = True

    for split_name, mask in [("cross_domain", cd_mask), ("all", pd.Series(True, index=wdf.index))]:
        sdf = wdf[mask]
        for dist in DIST_BINS:
            sub = sdf if dist == "all" else sdf[sdf["dist_bin"] == dist]
            m   = compute_metrics(sub, cfg["thr_p"], cfg["thr_s"])
            rows.append(dict(
                weight   = cfg["weight"],
                label    = cfg["label"],
                thr_p    = cfg["thr_p"],
                thr_s    = cfg["thr_s"],
                split    = split_name,
                dist_bin = dist,
                **m,
            ))

results = pd.DataFrame(rows)
results.to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Print numeric table to console (cross_domain split)
# ══════════════════════════════════════════════════════════════════════════════

METRIC_COLS = ["p_recall", "s_recall", "p_mae_s", "s_mae_s", "mcc", "p_outlier", "s_outlier"]

cross = results[results["split"] == "cross_domain"].copy()

print("=" * 110)
print("Three-way benchmark comparison — cross_domain split")
print(f"  (A) jma_wc  @thr=0.30     (B) v7 @thr=0.30     (C) v7 @thr=0.10")
print(f"  Unconditional MAE/outlier/MCC (Münchmeyer et al. 2022 definition)")
print("=" * 110)

hdr = f"{'Dist bin':<26} {'':>9} {'P-recall':>8} {'S-recall':>8} {'P-MAE':>7} {'S-MAE':>7} {'MCC':>6} {'P-out%':>7} {'S-out%':>7}"
print(hdr)
print("-" * 110)

for dist in DIST_BINS:
    print(f"\n  {dist}")
    for cfg in CONFIGS:
        row = cross[(cross["weight"] == cfg["weight"]) &
                    (cross["thr_p"]  == cfg["thr_p"])  &
                    (cross["dist_bin"] == dist)]
        if row.empty:
            continue
        r  = row.iloc[0]
        pr = f"{r.p_recall:.4f}" if not np.isnan(r.p_recall) else "    —  "
        sr = f"{r.s_recall:.4f}" if not np.isnan(r.s_recall) else "    —  "
        pm = f"{r.p_mae_s:.4f}"  if not np.isnan(r.p_mae_s)  else "    —  "
        sm = f"{r.s_mae_s:.4f}"  if not np.isnan(r.s_mae_s)  else "    —  "
        mc = f"{r.mcc:.4f}"      if not np.isnan(r.mcc)       else "    —  "
        po = f"{r.p_outlier*100:.2f}"  if not np.isnan(r.p_outlier)  else "  —  "
        so = f"{r.s_outlier*100:.2f}"  if not np.isnan(r.s_outlier)  else "  —  "
        tag = cfg["short"].replace("\n", " ")
        print(f"    {tag:<22} n={int(r.n_traces):>5}   {pr:>8} {sr:>8} {pm:>7} {sm:>7} {mc:>6} {po:>7} {so:>7}")


# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib table figure — identically styled to existing benchmark output
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating figure …")

# ── Layout: 4 rows (one per dist_bin), 1 matplotlib table per row ─────────────
# Each sub-table: 3 data rows (one per config) × 8 columns (metrics)
# We use one tall figure with gridspec.

N_BINS   = len(DIST_BINS)   # 4
N_CFGS   = len(CONFIGS)     # 3
COL_HDRS = ["n traces", "P-recall", "S-recall", "P-MAE (s)", "S-MAE (s)",
            "MCC", "P-outlier %", "S-outlier %"]

# Colour palette
HEADER_BG  = "#1A252F"
SECTION_BG = "#2C3E50"
SECTION_FG = "white"
CFG_COLORS = {
    (cfg["weight"], cfg["thr_p"]): cfg["color"]
    for cfg in CONFIGS
}

# Per-metric: is higher better?
HIGHER_BETTER = {
    "P-recall": True, "S-recall": True, "MCC": True,
    "P-MAE (s)": False, "S-MAE (s)": False,
    "P-outlier %": False, "S-outlier %": False,
    "n traces": None,
}

def lighten(hex_color, amount=0.6):
    import matplotlib.colors as mc
    c = mc.to_rgb(hex_color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)

fig = plt.figure(figsize=(15, 11))
fig.suptitle(
    "Complete Benchmark Evaluation — Three-Way Comparison (cross_domain split)\n"
    "Unconditional MAE / Outlier / MCC   |   Outlier threshold = ±1.5 s   |   Münchmeyer et al. (2022) definition",
    fontsize=11, fontweight="bold", y=0.995,
)

# Add a colour-coded legend strip at the top
legend_ax = fig.add_axes([0.10, 0.945, 0.80, 0.032])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.axis("off")
x_positions = [0.12, 0.45, 0.78]
for xi, cfg in zip(x_positions, CONFIGS):
    legend_ax.add_patch(plt.Rectangle((xi - 0.08, 0.0), 0.30, 1.0,
                                       color=cfg["color"], alpha=0.85,
                                       transform=legend_ax.transAxes, clip_on=False))
    lbl = cfg["short"].replace("\n", "  ")
    legend_ax.text(xi + 0.07, 0.5, lbl, transform=legend_ax.transAxes,
                   ha="center", va="center", fontsize=9, fontweight="bold",
                   color="white")

gs = gridspec.GridSpec(N_BINS, 1, figure=fig,
                       top=0.935, bottom=0.04,
                       hspace=0.08)

for row_i, dist in enumerate(DIST_BINS):
    ax = fig.add_subplot(gs[row_i])
    ax.axis("off")

    # ── Section header ────────────────────────────────────────────────────────
    sect_label = DIST_SHORT[dist].replace("\n", "  ")
    ax.text(-0.01, 1.08, sect_label, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="bottom", ha="left",
            color=SECTION_BG)

    # ── Build cell data ───────────────────────────────────────────────────────
    sub = cross[cross["dist_bin"] == dist]

    cell_text   = []
    cell_colour = []
    row_labels  = []

    # Collect numeric values for best-highlighting
    numeric = {h: [] for h in COL_HDRS}

    for cfg in CONFIGS:
        row_data = sub[(sub["weight"] == cfg["weight"]) &
                       (sub["thr_p"]  == cfg["thr_p"])]
        if row_data.empty:
            vals = ["—"] * len(COL_HDRS)
            cell_text.append(vals)
            cell_colour.append([lighten(cfg["color"], 0.85)] * len(COL_HDRS))
            row_labels.append(cfg["short"].replace("\n", " "))
            for h in COL_HDRS:
                numeric[h].append(np.nan)
            continue

        r = row_data.iloc[0]
        vals = [
            f"{int(r.n_traces):,}",
            f"{r.p_recall:.4f}"      if not np.isnan(r.p_recall)   else "—",
            f"{r.s_recall:.4f}"      if not np.isnan(r.s_recall)   else "—",
            f"{r.p_mae_s:.4f}"       if not np.isnan(r.p_mae_s)    else "—",
            f"{r.s_mae_s:.4f}"       if not np.isnan(r.s_mae_s)    else "—",
            f"{r.mcc:.4f}"           if not np.isnan(r.mcc)         else "—",
            f"{r.p_outlier*100:.2f}" if not np.isnan(r.p_outlier)  else "—",
            f"{r.s_outlier*100:.2f}" if not np.isnan(r.s_outlier)  else "—",
        ]
        cell_text.append(vals)
        cell_colour.append([lighten(cfg["color"], 0.78)] * len(COL_HDRS))
        row_labels.append(cfg["short"].replace("\n", " "))

        num_vals = [
            r.n_traces,
            r.p_recall   if not np.isnan(r.p_recall)  else np.nan,
            r.s_recall   if not np.isnan(r.s_recall)  else np.nan,
            r.p_mae_s    if not np.isnan(r.p_mae_s)   else np.nan,
            r.s_mae_s    if not np.isnan(r.s_mae_s)   else np.nan,
            r.mcc        if not np.isnan(r.mcc)        else np.nan,
            r.p_outlier  if not np.isnan(r.p_outlier) else np.nan,
            r.s_outlier  if not np.isnan(r.s_outlier) else np.nan,
        ]
        for h, v in zip(COL_HDRS, num_vals):
            numeric[h].append(v)

    # Highlight best cell in each metric column (bold + slightly darker bg)
    for col_i, hdr in enumerate(COL_HDRS):
        higher = HIGHER_BETTER.get(hdr)
        if higher is None:
            continue
        vals_num = numeric[hdr]
        valid    = [(v, i) for i, v in enumerate(vals_num) if not np.isnan(v)]
        if not valid:
            continue
        best_val, best_row = (max if higher else min)(valid, key=lambda x: x[0])
        cfg_best = CONFIGS[best_row]
        cell_colour[best_row][col_i] = lighten(cfg_best["color"], 0.45)

    # ── Draw table ────────────────────────────────────────────────────────────
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=COL_HDRS,
        cellColours=cell_colour,
        rowColours=[lighten(cfg["color"], 0.55) for cfg in CONFIGS],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.8)

    # Style header row (col indices 0..N-1; -1 is the row-label header and may not exist)
    for col_i in range(len(COL_HDRS)):
        cell = tbl[0, col_i]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Bold text in best cells
    for col_i, hdr in enumerate(COL_HDRS):
        higher = HIGHER_BETTER.get(hdr)
        if higher is None:
            continue
        vals_num = numeric[hdr]
        valid    = [(v, i) for i, v in enumerate(vals_num) if not np.isnan(v)]
        if not valid:
            continue
        _, best_row = (max if higher else min)(valid, key=lambda x: x[0])
        tbl[best_row + 1, col_i].set_text_props(fontweight="bold")

    # Bold row labels
    for ri in range(N_CFGS):
        tbl[ri + 1, -1].set_text_props(fontweight="bold", fontsize=8)

fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PNG}")
print("\nNote: P/S-MAE and P/S-outlier are threshold-independent (unconditional Münchmeyer definition).")
print("      The only columns that differ between v7@0.30 and v7@0.10 are P-recall and S-recall.")
print("      This means recall gains at thr=0.10 come at zero MAE cost by the benchmark metric.")
print("\nDone.")
