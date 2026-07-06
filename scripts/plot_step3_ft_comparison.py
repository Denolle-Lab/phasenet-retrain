#!/usr/bin/env python3
"""
plot_step3_ft_comparison.py

Regenerates the step3_ft_* comparison PNGs from the current
notebooks/step3_metrics.csv / step3_results.parquet WITHOUT re-running
inference (that only happens in scripts/eval_finetuned.py, which also
overwrites step3_metrics.csv with a version lacking the clean_holdout /
cross_domain_clean rows added by scripts/recompute_step3_metrics.py — do
not run that script just to refresh plots).

Figures 1-4 reproduce the original fine-tune-vs-baseline dashboard
(cross_domain split), updated to highlight the current best model
(jma_wc_ft_global_v7) instead of the stale v8 default.

Figure 5 is new: for public pretrained weights with a verified
cross-dataset event-leakage audit (scripts/audit_parent_leakage.py),
compares cross_domain (uncorrected) vs cross_domain_clean
(spatiotemporal-leakage-excluded) P-MAE / P-Recall / MCC, to visualize how
much the leakage found on 2026-07-06 (e.g. stead<->pnw 38.5%) actually
moves the numbers.

Run from repo root:
    conda activate surface
    python scripts/plot_step3_ft_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from domain_registry import split_masks, parent_clean_cross_domain_mask

REPO_ROOT    = Path(__file__).parent.parent.resolve()
NB_DIR       = REPO_ROOT / "notebooks"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
METRICS_PATH = NB_DIR / "step3_metrics.csv"

FT_WEIGHT = "jma_wc_ft_global_v7"

results_df = pd.read_parquet(RESULTS_PATH)
metrics_df = pd.read_csv(METRICS_PATH)

# ── Palette ───────────────────────────────────────────────────────────────────
HIGHLIGHT  = "#E6A817"   # gold  — jma_wc_ft
PARENT     = "#D25F10"   # burnt orange — jma_wc
TOP_MODELS = ["stead", "instance", "neic", "diting", FT_WEIGHT,
              "jma_wc_ft_global_v6", "jma_wc_ft_global_v5", "jma_wc_ft_global_v4", "jma_wc_ft_global_v3", "jma_wc_ft_frozen", "jma_wc_ft_noise", "jma_wc_ft", "jma_wc"]
COLORS = {
    "stead":                  "#1f77b4",
    "instance":               "#2ca02c",
    "neic":                   "#9467bd",
    "diting":                 "#8c564b",
    FT_WEIGHT:                "#16A085",
    "jma_wc_ft_global_v6":    "#27AE60",
    "jma_wc_ft_global_v5":    "#C0392B",
    "jma_wc_ft_global_v4":    "#E65C00",
    "jma_wc_ft_global_v3":    "#FF8C42",
    "jma_wc_ft_frozen":       HIGHLIGHT,
    "jma_wc_ft_noise":        "#F4C542",
    "jma_wc_ft":              "#D4B442",
    "jma_wc":                 PARENT,
}
LABELS = {
    "stead":                  "stead",
    "instance":               "instance",
    "neic":                   "neic",
    "diting":                 "diting",
    FT_WEIGHT:                "jma_wc_ft_global_v7 ★",
    "jma_wc_ft_global_v6":    "jma_wc_ft_global_v6",
    "jma_wc_ft_global_v5":    "jma_wc_ft_global_v5",
    "jma_wc_ft_global_v4":    "jma_wc_ft_global_v4",
    "jma_wc_ft_global_v3":    "jma_wc_ft_global_v3",
    "jma_wc_ft_frozen":       "jma_wc_ft_frozen",
    "jma_wc_ft_noise":        "jma_wc_ft_noise",
    "jma_wc_ft":              "jma_wc_ft",
    "jma_wc":                 "jma_wc",
}

dist_order = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)"]

cross = metrics_df[(metrics_df["split"] == "cross_domain") &
                   (~metrics_df["degenerate"].fillna(False))]
cross_all_df = cross[cross["dist_bin"] == "all"].set_index("weight")
cross_dist   = cross[cross["dist_bin"].isin(dist_order)]


# ── Figure 1 — Dashboard (P-MAE, S-MAE, Recall, MCC, Outlier) ────────────────
print("Generating Figure 1: dashboard …")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Fine-tuned jma_wc vs Pretrained Baselines — Cross-Domain Benchmark",
             fontsize=13, fontweight="bold", y=0.98)

metrics_to_plot = [
    ("p_mae_s",   "P-MAE (s)",            False),
    ("s_mae_s",   "S-MAE (s)",            False),
    ("p_recall",  "P-Recall @ thr=0.3",   True),
    ("s_recall",  "S-Recall @ thr=0.3",   True),
    ("mcc",       "MCC (Phase ID)",        True),
    ("p_outlier", "P-Outlier fraction",    False),
]

all_models_sorted = (cross_all_df["p_mae_s"].dropna()
                     .sort_values().index.tolist())

for ax, (col, ylabel, higher_better) in zip(axes.flat, metrics_to_plot):
    vals, colors, labels_list = [], [], []
    for w in all_models_sorted:
        if w not in cross_all_df.index or np.isnan(cross_all_df.loc[w, col]):
            continue
        vals.append(cross_all_df.loc[w, col])
        colors.append(HIGHLIGHT if w == FT_WEIGHT else
                      PARENT    if w == "jma_wc" else "#aaaaaa")
        labels_list.append(LABELS.get(w, w))

    y_pos = range(len(vals))
    bars = ax.barh(list(y_pos), vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_list, fontsize=7.5)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.invert_yaxis()
    if not higher_better:
        best_idx = vals.index(min(vals))
    else:
        best_idx = vals.index(max(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.5)
    ax.axvline(vals[labels_list.index(LABELS[FT_WEIGHT])] if LABELS[FT_WEIGHT] in labels_list else 0,
               color=HIGHLIGHT, lw=1.2, ls="--", alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

legend_handles = [
    Patch(color=COLORS[FT_WEIGHT], label="jma_wc_ft_global_v7 (this work)"),
    Patch(color=PARENT,            label="jma_wc (base model)"),
    Patch(color="#aaaaaa",         label="other pretrained"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
out1 = NB_DIR / "step3_ft_dashboard.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out1}")


# ── Figure 2 — P-MAE and S-MAE by distance bin (selected models) ─────────────
print("Generating Figure 2: distance-bin breakdown …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Timing Error by Distance Bin — Cross-Domain",
             fontsize=12, fontweight="bold")

for ax, (phase, col) in zip(axes, [("P", "p_mae_s"), ("S", "s_mae_s")]):
    x = np.arange(len(dist_order))
    w = 0.13
    for i, model in enumerate(TOP_MODELS):
        vals = []
        for d in dist_order:
            row = cross_dist[(cross_dist["weight"] == model) &
                             (cross_dist["dist_bin"] == d)]
            vals.append(row[col].values[0] if len(row) > 0 and not row[col].isna().all()
                        else np.nan)
        offset = (i - len(TOP_MODELS) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, width=w, label=LABELS.get(model, model),
                      color=COLORS.get(model, "#aaaaaa"),
                      edgecolor="white", zorder=3)
        if model == FT_WEIGHT:
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Local\n(<150 km)", "Regional\n(150–1500 km)",
                        "Teleseismic\n(>1500 km)"], fontsize=9)
    ax.set_ylabel(f"{phase}-MAE (s)", fontsize=10)
    ax.set_title(f"{phase}-wave Timing Error", fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out2 = NB_DIR / "step3_ft_distance_bins.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out2}")


# ── Figure 3 — P and S residual histograms by distance bin ───────────────────
print("Generating Figure 3: residual histograms …")

COMPARE_MODELS = [FT_WEIGHT, "jma_wc", "stead", "instance"]
colors_hist = [HIGHLIGHT, PARENT, COLORS["stead"], COLORS["instance"]]

fig, axes = plt.subplots(len(dist_order), 2, figsize=(13, 10))
fig.suptitle("P and S Pick Residual Distributions — Cross-Domain",
             fontsize=12, fontweight="bold")

bins = np.linspace(-2.0, 2.0, 60)
for row_i, dist in enumerate(dist_order):
    for col_i, (phase, res_col, in_col) in enumerate(
            [("P", "p_residual_s", "p_in_window"),
             ("S", "s_residual_s", "s_in_window")]):
        ax = axes[row_i, col_i]
        for model, color in zip(COMPARE_MODELS, colors_hist):
            mdf = results_df[(results_df["weight"] == model) &
                             (results_df["dist_bin"] == dist) &
                             (results_df[in_col] >= 0)]
            res = mdf[res_col].dropna()
            if len(res) < 10:
                continue
            ax.hist(res.clip(-2, 2), bins=bins, alpha=0.55, color=color,
                    label=LABELS.get(model, model), density=True)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlim(-2, 2)
        ax.set_xlabel(f"{phase} residual (s)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{dist}\n{phase}-wave", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out3 = NB_DIR / "step3_ft_residuals.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out3}")


# ── Figure 4 — Recall curves (P and S) ───────────────────────────────────────
print("Generating Figure 4: recall curves …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Detection Recall vs Probability Threshold — Cross-Domain (all distances)",
             fontsize=11, fontweight="bold")

thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
t_keys_p   = [f"p_recall_t{int(t*10):02d}" for t in thresholds]
t_keys_s   = [f"s_recall_t{int(t*10):02d}" for t in thresholds]

for ax, (phase, t_keys) in zip(axes, [("P", t_keys_p), ("S", t_keys_s)]):
    for model in TOP_MODELS:
        row = cross_all_df.loc[model] if model in cross_all_df.index else None
        if row is None:
            continue
        vals = [row[k] if k in row.index and not np.isnan(row[k]) else np.nan
                for k in t_keys]
        lw  = 2.5 if model == FT_WEIGHT else 1.5
        ls  = "-"  if model == FT_WEIGHT else "--" if model == "jma_wc" else "-"
        ax.plot(thresholds, vals, color=COLORS.get(model, "#aaaaaa"),
                lw=lw, ls=ls, marker="o", ms=4,
                label=LABELS.get(model, model))

    ax.set_xlabel("Probability threshold", fontsize=10)
    ax.set_ylabel(f"{phase}-Recall", fontsize=10)
    ax.set_title(f"{phase}-wave Detection Recall", fontsize=10)
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out4 = NB_DIR / "step3_ft_recall_curves.png"
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out4}")


# ── Figure 5 — Cross-dataset leakage correction (NEW, 2026-07-06 audit) ─────
print("Generating Figure 5: leakage correction …")

# Weights with a verified cross-dataset event-leakage audit, i.e. rows
# present under split=="cross_domain_clean" in metrics_df, restricted to
# the plain (non-eqt) PhaseNet pretrained weights this dashboard covers.
LEAK_MODELS = ["stead", "instance", "ethz", "pisdl", "scedc", "iquique"]
clean_all_df = metrics_df[(metrics_df["split"] == "cross_domain_clean") &
                          (metrics_df["dist_bin"] == "all")].set_index("weight")
leak_models = [m for m in LEAK_MODELS if m in clean_all_df.index and m in cross_all_df.index]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Cross-Dataset Event Leakage Correction — Public Pretrained Weights\n"
             "(spatiotemporal match ±2s/±10km across differently-named benchmark datasets, see scripts/audit_parent_leakage.py)",
             fontsize=11, fontweight="bold")

leak_metrics = [("p_mae_s", "P-MAE (s)", False), ("p_recall", "P-Recall", True), ("mcc", "MCC", True)]
x = np.arange(len(leak_models))
bw = 0.35
for ax, (col, ylabel, higher_better) in zip(axes, leak_metrics):
    raw_vals   = [cross_all_df.loc[m, col] for m in leak_models]
    clean_vals = [clean_all_df.loc[m, col] for m in leak_models]
    ax.bar(x - bw/2, raw_vals,   width=bw, label="cross_domain (uncorrected)",
           color="#aaaaaa", edgecolor="white")
    ax.bar(x + bw/2, clean_vals, width=bw, label="cross_domain_clean (leakage-excluded)",
           color="#E74C3C", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(leak_models, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for xi, (rv, cv) in enumerate(zip(raw_vals, clean_vals)):
        if np.isnan(rv) or np.isnan(cv) or rv == 0:
            continue
        pct = (cv - rv) / rv * 100
        ax.annotate(f"{pct:+.0f}%", (xi, max(rv, cv)), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7.5, color="#333333")

axes[0].legend(fontsize=8, framealpha=0.8, loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.90])
out5 = NB_DIR / "step3_leakage_correction.png"
fig.savefig(out5, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out5}")


# ── Figure 6 — worst-offender (trained_on, benchmark_dataset) pairs ─────────
# The aggregate view above dilutes the effect since leaked rows are a
# minority of each weight's full cross_domain pool. This isolates the
# specific dataset pairs the audit flagged as most contaminated.
print("Generating Figure 6: worst-offender pair breakdown …")

from sklearn.metrics import matthews_corrcoef


def _simple_metrics(df):
    p_traces = df[df["p_in_window"] >= 0]
    p_recall = (p_traces["p_prob"] >= 0.3).mean() if len(p_traces) > 0 else np.nan
    p_res = df.loc[df["p_in_window"] >= 0, "p_residual_s"].dropna()
    p_mae = np.abs(p_res).mean() if len(p_res) > 0 else np.nan
    both = df[(df["p_in_window"] >= 0) & (df["s_in_window"] >= 0)]
    mcc = np.nan
    if len(both) >= 5:
        y_true = np.concatenate([np.ones(len(both)), np.zeros(len(both))])
        y_pred = np.concatenate(
            [(both["p_prob"] > both["s_prob"]).astype(int).values,
             (both["s_prob"] > both["p_prob"]).astype(int).values])
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = np.nan
    return p_mae, p_recall, mcc, len(df)


# (weight, benchmark_dataset) — weight, not trained_on, since split_masks()/
# parent_clean_cross_domain_mask() key off the actual result rows' "weight"
# column (e.g. eqt_original_nonconservative's rows, not a bare "stead" row).
WORST_PAIRS = [("stead", "pnw"), ("instance", "ethz"), ("pisdl", "ethz"),
               ("eqt_original_nonconservative", "pnw")]
PAIR_LABELS = {"eqt_original_nonconservative": "eqt_orig_nc"}

pair_rows = []
for weight_name, bench_ds in WORST_PAIRS:
    wdf = results_df[results_df["weight"] == weight_name]
    if wdf.empty:
        continue
    _, cross_mask = split_masks(wdf, weight_name)
    clean_mask = parent_clean_cross_domain_mask(wdf, weight_name)
    ds_mask = wdf["dataset"] == bench_ds
    raw_df = wdf[cross_mask & ds_mask]
    if clean_mask is None or raw_df.empty:
        continue
    clean_df = wdf[clean_mask & ds_mask]
    label = PAIR_LABELS.get(weight_name, weight_name)
    pair_rows.append({
        "label": f"{label}\n→{bench_ds}",
        "raw": _simple_metrics(raw_df),
        "clean": _simple_metrics(clean_df),
    })

if pair_rows:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.suptitle("Worst-Offender Pairs — Same Earthquake, Different Benchmark Dataset Name",
                 fontsize=11, fontweight="bold")
    n_caption = "  ·  ".join(
        f"{r['label'].replace(chr(10), ' ')}: n={r['raw'][3]}→{r['clean'][3]}" for r in pair_rows)
    fig.text(0.5, 0.90, n_caption, ha="center", fontsize=8, color="#666666")
    pair_metrics = [(0, "P-MAE (s)"), (1, "P-Recall"), (2, "MCC")]
    x = np.arange(len(pair_rows))
    bw = 0.35
    for ax, (mi, ylabel) in zip(axes, pair_metrics):
        raw_vals   = [r["raw"][mi]   for r in pair_rows]
        clean_vals = [r["clean"][mi] for r in pair_rows]
        ax.bar(x - bw/2, raw_vals,   width=bw, label="cross_domain (uncorrected)",
               color="#aaaaaa", edgecolor="white")
        ax.bar(x + bw/2, clean_vals, width=bw, label="cross_domain_clean (leakage-excluded)",
               color="#E74C3C", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in pair_rows], fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for xi, r in enumerate(pair_rows):
            rv, cv = r["raw"][mi], r["clean"][mi]
            if np.isnan(rv) or np.isnan(cv) or rv == 0:
                continue
            pct = (cv - rv) / rv * 100
            ax.annotate(f"{pct:+.0f}%", (xi, max(rv, cv)), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color="#333333")

    axes[0].legend(fontsize=8, framealpha=0.8, loc="upper left")
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    out6 = NB_DIR / "step3_leakage_worst_pairs.png"
    fig.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out6}")
else:
    print("  (skipped — no worst-offender pairs resolvable, check results/parent_event_leakage_row_mask__*.csv)")


# ── Summary print ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print(f"SUMMARY — {FT_WEIGHT} vs key baselines (cross-domain, all distances)")
print("═" * 70)
cols = ["weight", "p_mae_s", "s_mae_s", "p_recall", "s_recall", "mcc", "p_outlier"]
summary = (cross_all_df.reset_index()
           .loc[cross_all_df.reset_index()["weight"].isin(TOP_MODELS), cols]
           .sort_values("p_mae_s"))
print(summary.to_string(index=False))
print(f"\nPlots saved to {NB_DIR}/")
