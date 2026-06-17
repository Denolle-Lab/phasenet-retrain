#!/usr/bin/env python3
"""
plot_noise_audit.py

Six-panel summary of the noise dataset audit:
  1. World map — station locations coloured by tectonic setting (after filter)
  2. P-probability histogram — noise_global
  3. P-probability histogram — noise_prephase
  4. Contamination rate by source dataset (noise_prephase)
  5. Tectonic composition before vs after filter
  6. Region breakdown (after filter)
  7. Training manifest composition

Output: notebooks/noise_audit_full_summary.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── load ───────────────────────────────────────────────────────────────────────
ng_meta  = pd.read_csv(ROOT / "data/noise_global/metadata.csv")
pp_meta  = pd.read_csv(ROOT / "data/noise_prephase/metadata.csv")
ng_audit = pd.read_csv(ROOT / "data/noise_audit/noise_global_audit.csv")
pp_audit = pd.read_csv(ROOT / "data/noise_audit/noise_prephase_audit.csv")
train    = pd.read_csv(ROOT / "data/manifests/train.csv", low_memory=False)
val      = pd.read_csv(ROOT / "data/manifests/val.csv",   low_memory=False)

THRESHOLD = 0.5

TECT_COLORS = {
    "mixed":              "#4e79a7",
    "unknown":            "#aaaaaa",
    "ocean_bottom":       "#59a14f",
    "subduction":         "#e15759",
    "volcanic":           "#f28e2b",
    "rift":               "#9c755f",
    "induced_seismicity": "#b07aa1",
}

# ── derived ────────────────────────────────────────────────────────────────────
ng_audit["flagged"] = ng_audit["max_p_prob"] > THRESHOLD
pp_audit["flagged"] = pp_audit["max_p_prob"] > THRESHOLD

# recover source_dataset from trace name (format: prephase_{dataset}_{trace})
# — pp_meta has already been filtered so a join would lose removed traces
pp_audit["source_dataset"] = (
    pp_audit["trace_name"]
    .str.replace("prephase_", "", n=1, regex=False)
    .str.split("_").str[0]
)
pp_joined = pp_audit

# combined after-filter metadata (both sources)
combined = pd.concat([
    ng_meta[["trace_name","tectonic_setting","region","latitude","longitude"]]
           .assign(source="noise_global"),
    pp_meta[["trace_name","tectonic_setting","region","latitude","longitude"]]
           .assign(source="noise_prephase"),
], ignore_index=True)
combined["latitude"]  = pd.to_numeric(combined["latitude"],  errors="coerce")
combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")
with_coords = combined.dropna(subset=["latitude", "longitude"])
with_coords = with_coords.copy()
with_coords["color"] = with_coords["tectonic_setting"].map(TECT_COLORS).fillna("#cccccc")

# tectonic before/after (both sources)
ng_all = ng_audit.merge(ng_meta[["trace_name","tectonic_setting"]], on="trace_name", how="left")
pp_all = pp_audit.merge(pp_meta[["trace_name","tectonic_setting"]], on="trace_name", how="left")
before_all = pd.concat([ng_all[["tectonic_setting","flagged"]],
                        pp_all[["tectonic_setting","flagged"]]], ignore_index=True)
tect_before = before_all["tectonic_setting"].value_counts()
tect_after  = before_all[~before_all["flagged"]]["tectonic_setting"].value_counts()

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 23))
gs  = gridspec.GridSpec(
    4, 2, figure=fig,
    hspace=0.44, wspace=0.30,
    height_ratios=[1.35, 1, 1, 1],
)

# ── [0,:] world map ────────────────────────────────────────────────────────────
ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
ax_map.set_global()
ax_map.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
ax_map.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
ax_map.add_feature(cfeature.BORDERS,   linewidth=0.2, linestyle=":", zorder=1)
ax_map.gridlines(linewidth=0.3, color="gray", alpha=0.4)

for src, marker, ms, alpha in [
    ("noise_global",   "o", 3,  0.28),
    ("noise_prephase", "^", 4,  0.50),
]:
    sub = with_coords[with_coords["source"] == src]
    ax_map.scatter(
        sub["longitude"].values, sub["latitude"].values,
        c=sub["color"].values, s=ms**2, marker=marker,
        alpha=alpha, linewidths=0, zorder=2,
        transform=ccrs.PlateCarree(),
    )

tect_present = [t for t in TECT_COLORS if t in combined["tectonic_setting"].values]
leg1 = ax_map.legend(
    handles=[mpatches.Patch(color=TECT_COLORS[t], label=t) for t in tect_present],
    title="Tectonic setting", loc="lower left",
    fontsize=7, title_fontsize=8, framealpha=0.88,
)
ax_map.add_artist(leg1)
ax_map.legend(
    handles=[
        plt.scatter([], [], marker=m, s=sz**2, c="gray", alpha=0.7, label=lbl)
        for lbl, m, sz in [
            (f"noise_global  ({len(ng_meta):,} traces)",   "o", 3),
            (f"noise_prephase  ({len(pp_meta):,} traces)", "^", 4),
        ]
    ],
    title="Source (after filter)", loc="lower right",
    fontsize=7.5, title_fontsize=8.5, framealpha=0.88,
)
ax_map.set_title(
    f"Geographical distribution of noise traces after filtering  "
    f"(n = {len(with_coords):,} with coordinates)",
    fontsize=12, pad=8,
)

# ── [1,0] P-prob histogram — noise_global ─────────────────────────────────────
def prob_hist(ax, audit_df, title):
    vals = audit_df["max_p_prob"].dropna().values
    kept    = vals[vals <= THRESHOLD]
    removed = vals[vals >  THRESHOLD]
    ax.hist(kept,    bins=60, range=(0, THRESHOLD),  color="steelblue", alpha=0.85,
            label=f"kept  ({len(kept):,})")
    ax.hist(removed, bins=40, range=(THRESHOLD, 1.0), color="#e15759",   alpha=0.85,
            label=f"removed  ({len(removed):,})")
    ax.axvline(THRESHOLD, color="black", linewidth=1.6, linestyle="--",
               label=f"threshold = {THRESHOLD}")
    ax.set_xlabel("Max P probability (over full 30-s window)", fontsize=9.5)
    ax.set_ylabel("Number of traces", fontsize=9.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5)
    ax.set_yscale("log")

prob_hist(fig.add_subplot(gs[1, 0]), ng_audit,
          f"P-probability distribution — noise_global  (n = {len(ng_audit):,})")
prob_hist(fig.add_subplot(gs[1, 1]), pp_audit,
          f"P-probability distribution — noise_prephase  (n = {len(pp_audit):,})")

# ── [2,0] contamination by source dataset (prephase) ─────────────────────────
ax_cont = fig.add_subplot(gs[2, 0])
src_stats = (
    pp_joined.groupby("source_dataset")["flagged"]
    .agg(total="count", flagged="sum")
    .reset_index()
)
src_stats["pct"] = 100.0 * src_stats["flagged"] / src_stats["total"]
src_stats = src_stats.sort_values("pct", ascending=True)

bars = ax_cont.barh(
    src_stats["source_dataset"], src_stats["pct"],
    color="#f28e2b", alpha=0.85, edgecolor="white",
)
for bar, row in zip(bars, src_stats.itertuples()):
    ax_cont.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{row.flagged:,} / {row.total:,}",
        va="center", fontsize=8.5,
    )
ax_cont.axvline(src_stats["pct"].mean(), color="gray", linewidth=1.2,
                linestyle="--", label=f"mean {src_stats['pct'].mean():.1f}%")
ax_cont.set_xlabel("Contamination rate (%)", fontsize=10)
ax_cont.set_title("noise_prephase: contamination rate by source dataset\n(P > 0.5)", fontsize=11)
ax_cont.set_xlim(0, src_stats["pct"].max() * 1.38)
ax_cont.legend(fontsize=8.5)

# ── [2,1] tectonic before vs after ────────────────────────────────────────────
ax_tect = fig.add_subplot(gs[2, 1])
tects = [t for t in TECT_COLORS if tect_before.get(t, 0) > 0]
x = np.arange(len(tects))
w = 0.38
ax_tect.bar(x - w/2, [tect_before.get(t, 0) for t in tects], width=w,
            color=[TECT_COLORS[t] for t in tects], alpha=0.40, label="before filter",
            edgecolor="gray", linewidth=0.5)
ax_tect.bar(x + w/2, [tect_after.get(t, 0)  for t in tects], width=w,
            color=[TECT_COLORS[t] for t in tects], alpha=0.95, label="after filter",
            edgecolor="none")
ax_tect.set_xticks(x)
ax_tect.set_xticklabels(tects, rotation=32, ha="right", fontsize=8.5)
ax_tect.set_ylabel("Traces", fontsize=10)
ax_tect.set_title("Tectonic composition before vs after filter\n(both noise datasets combined)", fontsize=11)
ax_tect.legend(fontsize=9)
ax_tect.set_xlim(-0.6, len(tects) - 0.4)

# ── [3,0] region breakdown ─────────────────────────────────────────────────────
ax_reg = fig.add_subplot(gs[3, 0])
region_counts = combined["region"].value_counts()
cmap = plt.cm.tab20(np.linspace(0, 1, len(region_counts)))
bars2 = ax_reg.barh(region_counts.index, region_counts.values,
                    color=cmap, alpha=0.88, edgecolor="white")
for bar, n in zip(bars2, region_counts.values):
    ax_reg.text(bar.get_width() + 150, bar.get_y() + bar.get_height() / 2,
                f"{n:,}", va="center", fontsize=8.5)
ax_reg.set_xlabel("Traces (after filter)", fontsize=10)
ax_reg.set_title("Noise traces by geographic region\n(after filter, both datasets)", fontsize=11)
ax_reg.set_xlim(0, region_counts.max() * 1.22)

# ── [3,1] training manifest composition ──────────────────────────────────────
ax_comp = fig.add_subplot(gs[3, 1])
all_manifest = pd.concat([train, val], ignore_index=True)
ds_counts = all_manifest["dataset_name"].value_counts()
noise_ds   = {"noise_global", "noise_prephase"}
seismic_ds = ds_counts[~ds_counts.index.isin(noise_ds)]
noise_ds_  = ds_counts[ds_counts.index.isin(noise_ds)]
labels     = list(seismic_ds.index) + list(noise_ds_.index)
values     = list(seismic_ds.values) + list(noise_ds_.values)
bar_colors = (["#4e79a7"] * len(seismic_ds)) + ["#e15759", "#f28e2b"]

bars3 = ax_comp.barh(labels, values, color=bar_colors, alpha=0.85, edgecolor="white")
for bar, n in zip(bars3, values):
    ax_comp.text(bar.get_width() + 150, bar.get_y() + bar.get_height() / 2,
                 f"{n:,}", va="center", fontsize=8.5)

total = sum(values)
noise_total = sum(noise_ds_.values)
ax_comp.set_xlabel("Traces (train + val combined)", fontsize=10)
ax_comp.set_title(
    f"Training manifest composition\n"
    f"(noise = red/orange,  {noise_total:,} / {total:,} = {100*noise_total/total:.1f}%)",
    fontsize=11,
)
ax_comp.set_xlim(0, max(values) * 1.22)

# ── save ───────────────────────────────────────────────────────────────────────
fig.suptitle("Noise Dataset Audit — Full Summary", fontsize=15, fontweight="bold", y=1.003)
out = ROOT / "notebooks/noise_audit_full_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
