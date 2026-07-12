#!/usr/bin/env python3
"""
visualize_training_dataset.py

Produces diagnostic figures for the training dataset built by
build_training_dataset.py.  Reads data/manifests/train.csv (and val.csv),
joins back to source-dataset metadata for lat/lon/depth/magnitude, then plots:

  Fig 1  — Global spatial distribution of events, coloured by dataset
  Fig 2  — Physics distributions: magnitude, depth, epicentral distance
  Fig 3  — S-pick coverage breakdown (why S generalisation is limited)
  Fig 4  — Dataset composition by distance bin (stacked bar)

Figures saved to notebooks/.
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)

import seisbench
seisbench.cache_root = SEISBENCH_CACHE
import seisbench.data as sbd

MANIFEST_DIR = Path("data/manifests")
FIG_DIR      = Path("notebooks")
FIG_DIR.mkdir(exist_ok=True)

MLAAPDE_PATH = Path(SEISBENCH_CACHE) / "datasets" / "mlaapde"
CWA_PATH     = Path(SEISBENCH_CACHE) / "datasets" / "cwa"
PISDL_PATH   = Path(SEISBENCH_CACHE) / "datasets" / "pisdl"

# ── per-dataset colour palette ──────────────────────────────────────────────
PALETTE = {
    "stead":          "#e41a1c",   # red
    "ceed":           "#ff7f00",   # orange
    "instancecounts": "#377eb8",   # blue
    "geofon":         "#984ea3",   # purple
    "mlaapde":        "#4daf4a",   # green
    "ethz":           "#a65628",   # brown
    "crew":           "#f781bf",   # pink
    "cwa":            "#999999",   # grey
    "iquique":        "#17becf",   # cyan
    "txed":           "#bcbd22",   # olive
    "pnw":            "#1f77b4",   # steel-blue
    "lendb":          "#ff9896",   # light-red
    "pisdl":          "#aec7e8",   # light-blue
    "vcseis":         "#c49c94",   # tan
}

# ──────────────────────────────────────────────────────────────────────────────
# Metadata join helpers
# ──────────────────────────────────────────────────────────────────────────────

META_COLS = [
    "trace_name",
    "source_latitude_deg", "source_longitude_deg",
    "source_depth_km", "source_magnitude",
]


def _sbd_meta(cls):
    ds = cls()
    m = ds.metadata
    have = [c for c in META_COLS if c in m.columns]
    return m[have].copy()


def _chunked_meta(path):
    csvs = sorted(f for f in Path(path).iterdir()
                  if f.name.startswith("metadata_") and f.suffix == ".csv")
    frames = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        have = [c for c in META_COLS if c in df.columns]
        frames.append(df[have])
    return pd.concat(frames, ignore_index=True)


def load_all_metadata():
    """Load source_lat/lon/depth/magnitude for every dataset, keyed by trace_name."""
    loaders = {
        "stead":          lambda: _sbd_meta(sbd.STEAD),
        "ceed":           lambda: _sbd_meta(sbd.CEED),
        "geofon":         lambda: _sbd_meta(sbd.GEOFON),
        "instancecounts": lambda: _sbd_meta(sbd.InstanceCounts),
        "ethz":           lambda: _sbd_meta(sbd.ETHZ),
        "crew":           lambda: _sbd_meta(sbd.CREW),
        "iquique":        lambda: _sbd_meta(sbd.Iquique),
        "txed":           lambda: _sbd_meta(sbd.TXED),
        "pnw":            lambda: _sbd_meta(sbd.PNW),
        "lendb":          lambda: _sbd_meta(sbd.LenDB),
        "vcseis":         lambda: _sbd_meta(sbd.VCSEIS),
        "mlaapde":        lambda: _chunked_meta(MLAAPDE_PATH),
        "cwa":            lambda: _chunked_meta(CWA_PATH),
        "pisdl":          lambda: _sbd_meta(lambda: sbd.WaveformDataset(str(PISDL_PATH))),
    }
    all_meta = {}
    for name, fn in loaders.items():
        print(f"  loading metadata: {name}")
        try:
            all_meta[name] = fn()
        except Exception as e:
            print(f"    WARNING: {e}")
    return all_meta


def enrich_manifest(manifest_df, all_meta):
    """Left-join source metadata onto the manifest."""
    frames = []
    for ds_name, group in manifest_df.groupby("dataset_name"):
        meta = all_meta.get(ds_name)
        if meta is None or "trace_name" not in meta.columns:
            frames.append(group)
            continue
        merged = group.merge(
            meta.drop_duplicates("trace_name"),
            on="trace_name", how="left"
        )
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Global spatial distribution
# ──────────────────────────────────────────────────────────────────────────────

def fig_spatial(df):
    has_loc = df["source_latitude_deg"].notna() & df["source_longitude_deg"].notna()
    sub = df[has_loc].copy()

    # try cartopy; fall back to plain matplotlib
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False

    datasets_present = sorted(sub["dataset_name"].unique())

    if USE_CARTOPY:
        proj = ccrs.Robinson()
        fig, ax = plt.subplots(
            1, 1, figsize=(16, 8),
            subplot_kw={"projection": proj}
        )
        ax.set_global()
        ax.add_feature(cfeature.LAND,  facecolor="#f0ede8", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#d8eaf5", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888888", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#aaaaaa", zorder=1)
        transform = ccrs.PlateCarree()
        for ds in datasets_present:
            rows = sub[sub["dataset_name"] == ds]
            ax.scatter(
                rows["source_longitude_deg"], rows["source_latitude_deg"],
                s=2, alpha=0.25, color=PALETTE.get(ds, "#333333"),
                transform=transform, zorder=2, rasterized=True,
            )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.set_facecolor("#d8eaf5")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.axhline(0, color="#aaaaaa", lw=0.4)
        ax.set_xlabel("Longitude (°)", fontsize=11)
        ax.set_ylabel("Latitude (°)",  fontsize=11)
        for ds in datasets_present:
            rows = sub[sub["dataset_name"] == ds]
            ax.scatter(
                rows["source_longitude_deg"], rows["source_latitude_deg"],
                s=2, alpha=0.25, color=PALETTE.get(ds, "#333333"),
                rasterized=True,
            )

    # legend
    handles = [
        mpatches.Patch(color=PALETTE.get(ds, "#333333"),
                       label=f"{ds}  (n={len(sub[sub['dataset_name']==ds]):,})")
        for ds in datasets_present
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5,
              framealpha=0.85, ncol=2, markerscale=2)

    ax.set_title("Training dataset — global event distribution (coloured by source dataset)",
                 fontsize=13, pad=10)
    plt.tight_layout()
    out = FIG_DIR / "training_spatial_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Physics distributions: magnitude, depth, distance
# ──────────────────────────────────────────────────────────────────────────────

def fig_physics(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── magnitude ────────────────────────────────────────────────────────────
    ax = axes[0]
    bins_mag = np.arange(-1, 10, 0.25)
    for ds in sorted(df["dataset_name"].unique()):
        sub = df[(df["dataset_name"] == ds) & df["source_magnitude"].notna()]
        if len(sub) == 0:
            continue
        ax.hist(sub["source_magnitude"], bins=bins_mag,
                color=PALETTE.get(ds, "#333333"), alpha=0.55,
                histtype="stepfilled", label=ds, linewidth=0.5)
    ax.set_xlabel("Magnitude", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Magnitude distribution", fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    med = df["source_magnitude"].dropna()
    if len(med):
        ax.axvline(med.median(), color="black", lw=1.2, ls="--",
                   label=f"median={med.median():.1f}")
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.8)

    # ── depth ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    bins_depth = np.arange(0, 700, 10)
    BIN_COLORS = {"local": "#2196f3", "regional": "#ff9800", "teleseismic": "#9c27b0"}
    for b, color in BIN_COLORS.items():
        sub = df[(df["distance_bin"] == b) & df["source_depth_km"].notna()]
        if len(sub) == 0:
            continue
        ax.hist(sub["source_depth_km"], bins=bins_depth,
                color=color, alpha=0.6, histtype="stepfilled",
                label=f"{b} (n={len(sub):,})", linewidth=0.5)
    ax.set_xlabel("Depth (km)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Depth distribution by distance bin", fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.8)

    # ── epicentral distance (log x) ───────────────────────────────────────────
    ax = axes[2]
    has_dist = df["distance_km"].notna() & (df["distance_km"] > 0)
    sub = df[has_dist]
    bins_dist = np.logspace(np.log10(1), np.log10(20000), 80)
    for ds in sorted(sub["dataset_name"].unique()):
        rows = sub[sub["dataset_name"] == ds]
        ax.hist(rows["distance_km"], bins=bins_dist,
                color=PALETTE.get(ds, "#333333"), alpha=0.55,
                histtype="stepfilled", label=ds, linewidth=0.5)
    ax.set_xscale("log")
    ax.axvline(150,  color="grey", lw=1, ls=":", alpha=0.7)
    ax.axvline(1500, color="grey", lw=1, ls=":", alpha=0.7)
    ax.text(80,   ax.get_ylim()[1] * 0.6, "local",        fontsize=7.5, color="grey", ha="center")
    ax.text(500,  ax.get_ylim()[1] * 0.6, "regional",     fontsize=7.5, color="grey", ha="center")
    ax.text(5000, ax.get_ylim()[1] * 0.6, "teleseismic",  fontsize=7.5, color="grey", ha="center")
    ax.set_xlabel("Epicentral distance (km, log scale)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Epicentral distance distribution", fontsize=13)
    ax.grid(True, alpha=0.3, lw=0.5, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6.5, ncol=2, framealpha=0.8)

    fig.suptitle("Training dataset physics distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "training_physics_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — S-pick coverage + commentary
# ──────────────────────────────────────────────────────────────────────────────

def fig_s_coverage(df):
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)

    # ── panel A: S coverage by distance bin ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    bins_order = ["local", "regional", "teleseismic", "unknown"]
    bin_total = df.groupby("distance_bin").size()
    bin_with_s = df[df["s_arrival_sample"].notna()].groupby("distance_bin").size()
    bin_frac  = (bin_with_s / bin_total).reindex(bins_order, fill_value=0)
    colors_bin = ["#2196f3", "#ff9800", "#9c27b0", "#9e9e9e"]
    bars = ax.bar(bins_order, bin_frac * 100, color=colors_bin, edgecolor="white", linewidth=1.2)
    for bar, frac in zip(bars, bin_frac):
        n = int(bin_total.get(bar.get_x() + bar.get_width()/2, 0))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{frac*100:.0f}%\n(n={bin_total.get(bar.get_label(), 0):,})",
                ha="center", va="bottom", fontsize=8)
    # fix label positioning
    for i, (b, bar) in enumerate(zip(bins_order, bars)):
        n = int(bin_total.get(b, 0))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{bin_frac[b]*100:.0f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Traces with S pick (%)", fontsize=11)
    ax.set_title("(A) S-pick coverage by distance bin", fontsize=12)
    ax.axhline(100, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── panel B: S coverage per dataset ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ds_order = sorted(df["dataset_name"].unique())
    ds_total  = df.groupby("dataset_name").size()
    ds_with_s = df[df["s_arrival_sample"].notna()].groupby("dataset_name").size()
    ds_frac   = (ds_with_s / ds_total).reindex(ds_order, fill_value=0)
    x = np.arange(len(ds_order))
    c = [PALETTE.get(ds, "#333333") for ds in ds_order]
    ax.bar(x, ds_frac * 100, color=c, edgecolor="white", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(ds_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Traces with S pick (%)", fontsize=11)
    ax.set_title("(B) S-pick coverage per dataset", fontsize=12)
    ax.axhline(100, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── panel C: P-S time distribution (where both exist) ────────────────────
    ax = fig.add_subplot(gs[1, 0])
    has_both = df["s_arrival_sample"].notna() & df["p_arrival_sample"].notna()
    ps_time  = (df.loc[has_both, "s_arrival_sample"] -
                df.loc[has_both, "p_arrival_sample"]) / 100.0  # assuming 100 Hz
    ps_time  = ps_time[ps_time > 0]
    BIN_COLORS = {"local": "#2196f3", "regional": "#ff9800"}
    for b, color in BIN_COLORS.items():
        idx = df[has_both & (df["distance_bin"] == b)].index
        t   = ((df.loc[idx, "s_arrival_sample"] - df.loc[idx, "p_arrival_sample"]) / 100.0)
        t   = t[t > 0]
        if len(t):
            ax.hist(t, bins=80, color=color, alpha=0.6, histtype="stepfilled",
                    label=f"{b} (n={len(t):,})", density=True)
    ax.set_xlabel("P-S time (s)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(C) P-S time distribution\n(30 s window → S picks cut off at ~28 s)",
                 fontsize=11)
    ax.axvline(28, color="red", lw=1.2, ls="--", alpha=0.7, label="window limit (~28 s)")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── panel D: annotation / commentary ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    # compute summary numbers for the text
    total_train = len(df)
    s_frac_local = (
        df[df["distance_bin"] == "local"]["s_arrival_sample"].notna().mean() * 100
    )
    s_frac_reg = (
        df[df["distance_bin"] == "regional"]["s_arrival_sample"].notna().mean() * 100
    )
    n_tele = int((df["distance_bin"] == "teleseismic").sum())
    n_local = int((df["distance_bin"] == "local").sum())
    n_reg   = int((df["distance_bin"] == "regional").sum())
    overall_s = df["s_arrival_sample"].notna().mean() * 100

    # typical regional P-S time at 150-1500 km
    # Vp~6, Vs~3.5 km/s -> at 500 km, P arrives ~83s, S ~143s -> PS~60s > 30s window
    commentary = (
        "Expected S-pick model performance\n"
        "──────────────────────────────────────\n\n"
        f"Overall S-pick coverage (train): {overall_s:.0f}%\n\n"
        f"Local traces   ({n_local:,}):  {s_frac_local:.0f}% have S picks\n"
        f"  → Model will learn S well at local distances.\n"
        f"  → P-S times are short (< 15 s), comfortably\n"
        f"    within the 30 s window.\n\n"
        f"Regional traces ({n_reg:,}):  {s_frac_reg:.0f}% have S picks\n"
        f"  → Coverage is partial.  More importantly,\n"
        f"    at 300-1500 km, P-S time exceeds 30-120 s —\n"
        f"    S often falls OUTSIDE the 30 s window.\n"
        f"  → Expect poor S generalisation at regional\n"
        f"    distances even where labels exist.\n\n"
        f"Teleseismic traces ({n_tele:,}): 0% S picks\n"
        f"  → P-only by design.  S arrives 3-15 min\n"
        f"    after P — impossible in a 30 s window.\n"
        f"  → Model will output near-zero S probability\n"
        f"    for teleseismic input, which is correct.\n\n"
        "Bottom line:  Fine-tune for S only at local\n"
        "distances.  Regional/teleseismic S requires\n"
        "longer windows (60-120 s) and separate training."
    )
    ax.text(0.03, 0.97, commentary, transform=ax.transAxes,
            fontsize=9, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#fff9e6", alpha=0.9, edgecolor="#ccaa00"))
    ax.set_title("(D) Implications for S-pick generalisation", fontsize=12, pad=8)

    fig.suptitle("S-pick coverage analysis — training dataset", fontsize=14, y=1.005)
    out = FIG_DIR / "training_s_pick_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Dataset composition (stacked bar by distance bin)
# ──────────────────────────────────────────────────────────────────────────────

def fig_composition(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ds_order = sorted(df["dataset_name"].unique(),
                      key=lambda d: -len(df[df["dataset_name"] == d]))
    bins = ["local", "regional", "teleseismic", "unknown"]
    bin_colors = {"local": "#2196f3", "regional": "#ff9800",
                  "teleseismic": "#9c27b0", "unknown": "#9e9e9e"}

    # ── stacked bar: counts per dataset per bin ───────────────────────────────
    ax = axes[0]
    counts = {b: [] for b in bins}
    for ds in ds_order:
        sub = df[df["dataset_name"] == ds]
        vc = sub["distance_bin"].value_counts()
        for b in bins:
            counts[b].append(vc.get(b, 0))

    x = np.arange(len(ds_order))
    bottom = np.zeros(len(ds_order))
    for b in bins:
        ax.bar(x, counts[b], bottom=bottom, color=bin_colors[b],
               label=b, edgecolor="white", linewidth=0.5)
        bottom += np.array(counts[b])

    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Number of training traces", fontsize=11)
    ax.set_title("(A) Training traces per dataset\n(coloured by distance bin)", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── stacked bar: dataset share per distance bin ───────────────────────────
    ax = axes[1]
    bin_ds_counts = {}
    for b in bins:
        sub = df[df["distance_bin"] == b]
        if len(sub) == 0:
            continue
        vc = sub["dataset_name"].value_counts()
        bin_ds_counts[b] = vc

    all_ds = sorted(df["dataset_name"].unique())
    x = np.arange(len(bins))
    bottoms = np.zeros(len(bins))
    for ds in all_ds:
        vals = [bin_ds_counts.get(b, pd.Series(dtype=int)).get(ds, 0) for b in bins]
        total_per_bin = np.array([len(df[df["distance_bin"] == b]) for b in bins], dtype=float)
        total_per_bin[total_per_bin == 0] = 1
        fracs = np.array(vals) / total_per_bin * 100
        ax.bar(x, fracs, bottom=bottoms, color=PALETTE.get(ds, "#333333"),
               label=ds, edgecolor="white", linewidth=0.5)
        bottoms += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=10)
    ax.set_ylabel("Share of distance bin (%)", fontsize=11)
    ax.set_title("(B) Dataset composition within each\ndistance bin", fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7, framealpha=0.85, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # add total-n labels on x-axis
    for i, b in enumerate(bins):
        n = int((df["distance_bin"] == b).sum())
        ax.text(i, 102, f"n={n:,}", ha="center", fontsize=7.5, color="dimgrey")

    fig.suptitle("Training dataset composition", fontsize=14, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "training_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    train_csv = MANIFEST_DIR / "train.csv"
    if not train_csv.exists():
        sys.exit(f"ERROR: {train_csv} not found — run build_training_dataset.py first")

    print("Loading manifests …")
    train_df = pd.read_csv(train_csv)
    print(f"  Train: {len(train_df):,} traces")

    print("\nLoading source-dataset metadata for enrichment …")
    all_meta = load_all_metadata()

    print("\nEnriching manifest with lat/lon/depth/magnitude …")
    train_df = enrich_manifest(train_df, all_meta)

    has_loc = train_df["source_latitude_deg"].notna().sum()
    print(f"  {has_loc:,} / {len(train_df):,} traces have source location")

    print("\nGenerating figures …")
    fig_spatial(train_df)
    fig_physics(train_df)
    fig_s_coverage(train_df)
    fig_composition(train_df)

    print("\nDone.  Figures written to notebooks/:")
    for f in ["training_spatial_distribution.png",
              "training_physics_distributions.png",
              "training_s_pick_analysis.png",
              "training_composition.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
