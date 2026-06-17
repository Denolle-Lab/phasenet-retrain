"""
SeisBench Dataset Metadata Audit
=================================
Phase 1: Dataset Audit & Curation

Extracts and compares metadata ONLY (no waveforms) across all available
SeisBench benchmark datasets along the following axes:
  - Distance range (local / regional / teleseismic coverage)
  - Magnitude distribution
  - Noise floor & instrument types
  - Pick quality (analyst vs autopicker)
  - Geographic clustering

Usage:
    python audit_metadata.py [--datasets all] [--out_dir ./audit_results]

Author: Generated for PhD seismology project
"""

from __future__ import annotations

import os
import sys
import warnings
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── SeisBench imports ────────────────────────────────────────────────────────
try:
    import seisbench.data as sbd
except ImportError:
    sys.exit(
        "SeisBench not found.  Install with:  pip install seisbench"
    )

# ── Dataset registry ─────────────────────────────────────────────────────────
#   key  : short name used throughout the script
#   value: (SeisBench class, extra kwargs for constructor)
DATASET_REGISTRY = {
    "STEAD":        (sbd.STEAD,           {}),
    "INSTANCE":     (sbd.InstanceCounts,  {}),   # InstanceCounts = main INSTANCE dataset
    "PNW":          (sbd.PNW,             {}),
    "IQUIQUE":      (sbd.Iquique,         {}),
    "LENDB":        (sbd.LenDB,           {}),
    "SCEDC":        (sbd.SCEDC,           {}),
    "ETHZ":         (sbd.ETHZ,            {}),
    "GEOFON":       (sbd.GEOFON,          {}),
    "NEIC":         (sbd.NEIC,            {}),
    "OBST2024":     (sbd.OBST2024,        {}),
    "TXED":         (sbd.TXED,            {}),
    "MEIER2019":    (sbd.Meier2019JGR,    {}),
    "ROSS2018GPD":  (sbd.Ross2018GPD,     {}),
    "VCSEIS":       (sbd.VCSEIS,          {}),
    "CEED":         (sbd.CEED,            {}),
    "MLAAPDE":      (sbd.MLAAPDE,         {}),
    "PNW_ACCEL":    (sbd.PNWAccelerometers, {}),
    "CREW":         (sbd.CREW,            {}),
}

# ── Distance bin edges (km) ──────────────────────────────────────────────────
DIST_BINS   = [0, 50, 200, 800, 2000, 20000]
DIST_LABELS = ["Local\n<50 km", "Near\n50–200 km",
                "Regional\n200–800 km", "Far-regional\n800–2000 km",
                "Teleseismic\n>2000 km"]

# ── Magnitude bins ───────────────────────────────────────────────────────────
MAG_BINS   = [-2, 0, 1, 2, 3, 4, 5, 6, 7, 10]
MAG_LABELS = ["<0", "0–1", "1–2", "2–3", "3–4", "4–5", "5–6", "6–7", ">7"]

# ── Colour palette (one per dataset) ─────────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#A8DADC", "#8338EC", "#FB5607", "#06D6A0",
    "#118AB2", "#FFD166",
]


# ═══════════════════════════════════════════════════════════════════════════
#  METADATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_metadata_only(cls) -> list:
    """
    Download ONLY the metadata CSV(s) for a SeisBench benchmark dataset.

    Strategy:
      1. Fetch the `chunks` index file from the remote repo (a tiny text
         file listing chunk suffixes, e.g. "0\n1\n2").
      2. Download each `metadata<chunk>.csv` that is not already present.
      3. Never touch waveforms*.hdf5.

    Returns a list of local Path objects for the metadata CSVs, or raises
    on unrecoverable failure.
    """
    import requests, seisbench

    remote_base = cls._remote_path()          # e.g. https://.../stead
    local_dir   = cls._path_internal()        # e.g. ~/.seisbench/datasets/stead
    local_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: resolve chunk list ───────────────────────────────────────
    chunks_local  = local_dir / "chunks"
    chunks_remote = f"{remote_base}/chunks"

    chunk_suffixes = [""]          # default: unchunked → metadata.csv
    if chunks_local.is_file():
        raw = chunks_local.read_text().strip()
        if raw:
            chunk_suffixes = [x for x in raw.split("\n") if x.strip()]
    else:
        try:
            r = requests.get(chunks_remote, timeout=15,
                             headers={"User-Agent": "SeisBench"})
            if r.status_code == 200 and r.text.strip():
                chunks_local.write_text(r.text)
                chunk_suffixes = [x for x in r.text.strip().split("\n") if x.strip()]
        except Exception:
            pass   # No chunks file → treat as unchunked

    # ── Step 2: download missing metadata CSVs ───────────────────────────
    fetched = []
    for suffix in chunk_suffixes:
        local_csv  = local_dir / f"metadata{suffix}.csv"
        remote_csv = f"{remote_base}/metadata{suffix}.csv"

        if local_csv.is_file():
            fetched.append(local_csv)
            continue

        try:
            r = requests.get(remote_csv, timeout=120, stream=True,
                             headers={"User-Agent": "SeisBench"})
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            partial = local_csv.with_suffix(".csv.partial")
            try:
                from tqdm import tqdm
                bar = tqdm(total=total, unit="B", unit_scale=True,
                           desc=f"    {local_csv.name}", leave=False)
            except ImportError:
                bar = None
            with open(partial, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    if bar:
                        bar.update(len(chunk))
            if bar:
                bar.close()
            partial.rename(local_csv)
            fetched.append(local_csv)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {remote_csv}: {exc}"
            ) from exc

    return fetched


def load_metadata(name: str, cls, kwargs: dict) -> pd.DataFrame | None:
    """
    Load ONLY the metadata CSVs for a SeisBench dataset.

    Behaviour:
      • If metadata*.csv files already exist in the local cache they are
        read directly — zero network activity.
      • If they are missing, only the metadata CSVs (and the tiny `chunks`
        manifest) are downloaded from the SeisBench repository.
        Waveform HDF5 files are never touched.

    The dataset constructor (cls.__init__) is never called, which is the
    only reliable way to avoid the chunk-manifest + waveform download that
    SeisBench triggers at __init__ time.
    """
    dataset_dir = cls._path_internal()
    print(f"  Loading {name} ...", end=" ", flush=True)

    # ── Check local cache first ──────────────────────────────────────────
    csv_files = sorted(dataset_dir.glob("metadata*.csv")) if dataset_dir.exists() else []

    # ── Download only metadata CSVs if not cached ────────────────────────
    if not csv_files:
        print(f"downloading metadata CSV(s)...", end=" ", flush=True)
        try:
            csv_files = _fetch_metadata_only(cls)
        except Exception as exc:
            print(f"FAILED\n    {exc}")
            return None

    if not csv_files:
        print("FAILED  (no metadata files found or downloaded)")
        return None

    # ── Read CSVs ────────────────────────────────────────────────────────
    frames = []
    for csv_path in csv_files:
        try:
            frames.append(pd.read_csv(csv_path, low_memory=False))
        except Exception as exc:
            print(f"\n    WARNING: could not read {csv_path.name}: {exc}")

    if not frames:
        print("FAILED  (all CSV reads failed)")
        return None

    meta = pd.concat(frames, ignore_index=True)
    meta["_dataset"] = name
    print(f"OK  ({len(meta):,} traces, {len(csv_files)} chunk(s))")
    return meta


def extract_columns(df: pd.DataFrame) -> dict:
    """
    Robustly extract the relevant columns regardless of dataset-specific
    naming conventions used across SeisBench datasets.
    """
    col = {}

    # ── helper: try a list of candidate column names ─────────────────────
    def pick(candidates, default=None):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([default] * len(df), index=df.index)

    col["distance_km"]   = pd.to_numeric(
        pick(["path_ep_distance_km", "source_distance_km",
              "ep_distance_km", "distance_km"]), errors="coerce")

    col["magnitude"]     = pd.to_numeric(
        pick(["source_magnitude", "magnitude",
              "event_magnitude", "mag"]), errors="coerce")

    col["latitude"]      = pd.to_numeric(
        pick(["source_latitude_deg", "source_latitude",
              "event_latitude", "latitude"]), errors="coerce")

    col["longitude"]     = pd.to_numeric(
        pick(["source_longitude_deg", "source_longitude",
              "event_longitude", "longitude"]), errors="coerce")

    col["depth_km"]      = pd.to_numeric(
        pick(["source_depth_km", "source_depth",
              "event_depth_km", "depth"]), errors="coerce")

    col["station_lat"]   = pd.to_numeric(
        pick(["station_latitude_deg", "station_latitude",
              "receiver_latitude"]), errors="coerce")

    col["station_lon"]   = pd.to_numeric(
        pick(["station_longitude_deg", "station_longitude",
              "receiver_longitude"]), errors="coerce")

    col["sampling_rate"] = pd.to_numeric(
        pick(["trace_sampling_rate_hz", "sampling_rate",
              "sample_rate"]), errors="coerce")

    # Instrument type — infer from channel codes where possible
    col["channel"]       = pick(["trace_channel", "channel",
                                  "channels", "component"], "UNK")
    col["instrument"]    = col["channel"].astype(str).apply(infer_instrument)

    # Pick quality / review status
    col["pick_quality"]  = pick(["trace_pick_quality", "pick_quality",
                                  "reviewer", "review_level"], "unknown")
    col["p_pick"]        = pick(["trace_p_arrival_sample",
                                  "p_arrival_sample", "p_pick_sample"])
    col["s_pick"]        = pick(["trace_s_arrival_sample",
                                  "s_arrival_sample", "s_pick_sample"])
    col["snr_db"]        = pd.to_numeric(
        pick(["trace_snr_db", "snr_db", "snr"]), errors="coerce")

    col["mag_type"]      = pick(["source_magnitude_type", "magnitude_type",
                                  "mag_type"], "unknown")

    return col


def infer_instrument(channel_str: str) -> str:
    """Map SEED channel code prefix to instrument class."""
    c = str(channel_str).upper().strip()
    if not c or c in ("NAN", "UNK", "NONE"):
        return "Unknown"
    band = c[0] if len(c) >= 1 else "?"
    inst = c[1] if len(c) >= 2 else "?"
    # Band code → approximate instrument
    if band in ("B", "H"):
        return "Broadband"
    if band in ("S", "E", "s"):
        return "Short-period"
    if band in ("N", "L"):
        return "Accelerometer"
    if band in ("G", "M"):
        return "Geophone"
    if band in ("V", "U"):
        return "VLBI/ULF"
    # Instrument code fallback
    if inst in ("H", "G"):
        return "Broadband"
    if inst in ("P", "L"):
        return "Short-period"
    if inst in ("N",):
        return "Accelerometer"
    return f"Other ({c[:2]})"


def compute_distance_from_coords(row) -> float:
    """
    Approximate epicentral distance (km) from lat/lon pairs using
    the haversine formula, used when path_ep_distance_km is absent.
    """
    R = 6371.0
    lat1, lon1 = np.radians(row["latitude"]), np.radians(row["longitude"])
    lat2, lon2 = np.radians(row["station_lat"]), np.radians(row["station_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def build_summary(name: str, raw: pd.DataFrame, col: dict) -> dict:
    dist    = col["distance_km"].dropna()
    mag     = col["magnitude"].dropna()
    snr     = col["snr_db"].dropna()
    sr      = col["sampling_rate"].dropna()

    n_total = len(raw)
    n_p     = col["p_pick"].notna().sum()
    n_s     = col["s_pick"].notna().sum()

    # Distance coverage flags
    has_dist  = len(dist) > 0
    dist_bins = pd.cut(dist, DIST_BINS, labels=DIST_LABELS) if has_dist else None

    # Instrument mix
    inst_counts = col["instrument"].value_counts()

    # Pick quality distribution
    pq = col["pick_quality"].astype(str).str.lower()
    analyst_keywords = ["analyst", "manual", "reviewed", "confirmed", "1", "a"]
    n_analyst = pq.apply(lambda x: any(k in x for k in analyst_keywords)).sum()

    return {
        "dataset":          name,
        "n_traces":         n_total,
        "n_events":         raw.get("source_id", raw.index).nunique()
                            if "source_id" in raw.columns else np.nan,
        # Distance
        "dist_available":   has_dist,
        "dist_min":         float(dist.min())  if has_dist else np.nan,
        "dist_p10":         float(dist.quantile(.10)) if has_dist else np.nan,
        "dist_median":      float(dist.median()) if has_dist else np.nan,
        "dist_p90":         float(dist.quantile(.90)) if has_dist else np.nan,
        "dist_max":         float(dist.max())  if has_dist else np.nan,
        "dist_bins":        dist_bins,
        # Magnitude
        "mag_min":          float(mag.min())    if len(mag) else np.nan,
        "mag_median":       float(mag.median()) if len(mag) else np.nan,
        "mag_max":          float(mag.max())    if len(mag) else np.nan,
        "mag_std":          float(mag.std())    if len(mag) else np.nan,
        # SNR
        "snr_median":       float(snr.median()) if len(snr) else np.nan,
        "snr_available":    len(snr) > 0,
        # Sampling rate
        "sampling_rates":   sorted(sr.unique().tolist()) if len(sr) else [],
        "dominant_sr":      float(sr.mode()[0]) if len(sr) else np.nan,
        # Instruments
        "instrument_mix":   inst_counts.to_dict(),
        "dominant_instrument": inst_counts.idxmax() if len(inst_counts) else "Unknown",
        # Picks
        "n_p_picks":        int(n_p),
        "n_s_picks":        int(n_s),
        "p_pick_frac":      n_p / n_total if n_total else np.nan,
        "s_pick_frac":      n_s / n_total if n_total else np.nan,
        "n_analyst_picks":  int(n_analyst),
        "analyst_frac":     n_analyst / n_total if n_total else np.nan,
        # Geography
        "lat_min":          float(col["latitude"].dropna().min())  if col["latitude"].notna().any() else np.nan,
        "lat_max":          float(col["latitude"].dropna().max())  if col["latitude"].notna().any() else np.nan,
        "lon_min":          float(col["longitude"].dropna().min()) if col["longitude"].notna().any() else np.nan,
        "lon_max":          float(col["longitude"].dropna().max()) if col["longitude"].notna().any() else np.nan,
        # Raw columns for plotting
        "_distance":        dist,
        "_magnitude":       mag,
        "_latitude":        col["latitude"].dropna(),
        "_longitude":       col["longitude"].dropna(),
        "_instrument":      col["instrument"],
        "_snr":             snr,
        "_col":             col,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def style_ax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_overview_table(summaries: list[dict], out_dir: Path):
    """A clean summary table saved as a figure."""
    rows = []
    for s in summaries:
        inst = s["instrument_mix"]
        dom_inst = max(inst, key=inst.get) if inst else "?"
        rows.append({
            "Dataset":       s["dataset"],
            "Traces":        f"{s['n_traces']:,}",
            "Dist median\n(km)": f"{s['dist_median']:.0f}" if not np.isnan(s['dist_median']) else "—",
            "Dist range\n(km)": (f"{s['dist_min']:.0f}–{s['dist_max']:.0f}"
                                  if not np.isnan(s['dist_min']) else "—"),
            "Mag\nmedian":   f"{s['mag_median']:.1f}" if not np.isnan(s['mag_median']) else "—",
            "Mag range":     f"{s['mag_min']:.1f}–{s['mag_max']:.1f}" if not np.isnan(s['mag_min']) else "—",
            "Dom.\nInstrument": dom_inst,
            "Analyst\npicks %": f"{s['analyst_frac']*100:.0f}%" if not np.isnan(s['analyst_frac']) else "—",
            "P-pick\n%":     f"{s['p_pick_frac']*100:.0f}%" if not np.isnan(s['p_pick_frac']) else "—",
            "S-pick\n%":     f"{s['s_pick_frac']*100:.0f}%" if not np.isnan(s['s_pick_frac']) else "—",
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    # Header style
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1A1A2E")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
        cell.set_edgecolor("#CCCCCC")

    fig.suptitle("SeisBench Dataset Overview — Metadata Audit", 
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "00_overview_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved overview table")


def plot_distance_distributions(summaries: list[dict], out_dir: Path):
    """KDE + histogram of epicentral distance per dataset."""
    datasets_with_dist = [s for s in summaries if s["dist_available"] and len(s["_distance"]) > 10]
    if not datasets_with_dist:
        print("  ⚠ No distance data available — skipping distance plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: overlapping KDEs (log scale)
    ax = axes[0]
    for i, s in enumerate(datasets_with_dist):
        d = s["_distance"].clip(lower=0.1)
        log_d = np.log10(d)
        x = np.linspace(log_d.min() - 0.5, log_d.max() + 0.5, 300)
        try:
            kde = stats.gaussian_kde(log_d, bw_method=0.15)
            ax.plot(10 ** x, kde(x), label=s["dataset"],
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8)
            ax.fill_between(10 ** x, kde(x), alpha=0.08,
                            color=PALETTE[i % len(PALETTE)])
        except Exception:
            pass
    ax.set_xscale("log")
    for xv, lbl in zip(DIST_BINS[1:-1], ["Local|Near", "Near|Regional", "Regional|Far", "Far|Tele"]):
        ax.axvline(xv, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    style_ax(ax, "Epicentral Distance Distribution (KDE)", "Distance (km)", "Density")
    ax.legend(fontsize=7, framealpha=0.7)

    # Right: stacked bar — fraction in each distance bin
    ax2 = axes[1]
    bin_fracs = []
    names = []
    for s in datasets_with_dist:
        d = s["_distance"]
        counts = pd.cut(d, DIST_BINS, labels=DIST_LABELS).value_counts(sort=False)
        bin_fracs.append((counts / counts.sum()).values)
        names.append(s["dataset"])

    arr = np.array(bin_fracs)
    bottom = np.zeros(len(names))
    colors_dist = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(DIST_LABELS)))
    for j, lbl in enumerate(DIST_LABELS):
        ax2.barh(names, arr[:, j], left=bottom, label=lbl.replace("\n", " "),
                 color=colors_dist[j], edgecolor="white", linewidth=0.5)
        bottom += arr[:, j]
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    style_ax(ax2, "Distance Bin Composition per Dataset", "Fraction of traces", "")
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.8)

    fig.suptitle("Epicentral Distance Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "01_distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved distance distributions")


def plot_magnitude_distributions(summaries: list[dict], out_dir: Path):
    """Violin + CDF of magnitudes."""
    datasets_with_mag = [s for s in summaries if len(s["_magnitude"]) > 10]
    if not datasets_with_mag:
        print("  ⚠ No magnitude data — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: violin
    ax = axes[0]
    data_to_plot = [s["_magnitude"].clip(-3, 10).values for s in datasets_with_mag]
    parts = ax.violinplot(data_to_plot, positions=range(len(datasets_with_mag)),
                           showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(datasets_with_mag)))
    ax.set_xticklabels([s["dataset"] for s in datasets_with_mag],
                        rotation=35, ha="right", fontsize=8)
    ax.axhline(3, color="red", linestyle="--", linewidth=0.9, alpha=0.6, label="M3")
    ax.axhline(5, color="darkred", linestyle="--", linewidth=0.9, alpha=0.6, label="M5")
    style_ax(ax, "Magnitude Distribution (Violin)", "", "Magnitude")
    ax.legend(fontsize=8)

    # Right: CDFs
    ax2 = axes[1]
    for i, s in enumerate(datasets_with_mag):
        m = np.sort(s["_magnitude"].values)
        cdf = np.arange(1, len(m) + 1) / len(m)
        ax2.plot(m, cdf, label=s["dataset"], color=PALETTE[i % len(PALETTE)], linewidth=1.6)
    style_ax(ax2, "Cumulative Magnitude Distribution", "Magnitude", "Cumulative Fraction")
    ax2.legend(fontsize=7, framealpha=0.7)

    fig.suptitle("Magnitude Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "02_magnitude_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved magnitude distributions")


def plot_instrument_types(summaries: list[dict], out_dir: Path):
    """Grouped bar chart of instrument type fractions per dataset."""
    all_insts = set()
    for s in summaries:
        all_insts |= set(s["instrument_mix"].keys())
    all_insts = sorted(all_insts)

    names    = [s["dataset"] for s in summaries]
    n_ds     = len(names)
    n_inst   = len(all_insts)

    fracs = np.zeros((n_ds, n_inst))
    for i, s in enumerate(summaries):
        total = sum(s["instrument_mix"].values()) or 1
        for j, inst in enumerate(all_insts):
            fracs[i, j] = s["instrument_mix"].get(inst, 0) / total

    fig, ax = plt.subplots(figsize=(max(12, n_ds * 1.2), 5))
    x = np.arange(n_ds)
    width = 0.8 / n_inst
    inst_colors = plt.cm.tab10(np.linspace(0, 0.9, n_inst))

    for j, (inst, col) in enumerate(zip(all_insts, inst_colors)):
        ax.bar(x + j * width - 0.4 + width / 2, fracs[:, j],
               width=width, label=inst, color=col, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    style_ax(ax, "Instrument Type Composition per Dataset", "", "Fraction of traces")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    fig.suptitle("Instrument & Sensor Type Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "03_instrument_types.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved instrument types")


def plot_pick_quality(summaries: list[dict], out_dir: Path):
    """Stacked bar showing analyst vs other pick fractions + P/S availability."""
    names = [s["dataset"] for s in summaries]
    analyst_f = [s["analyst_frac"] if not np.isnan(s["analyst_frac"]) else 0 for s in summaries]
    other_f   = [1 - a for a in analyst_f]
    p_frac    = [s["p_pick_frac"] if not np.isnan(s["p_pick_frac"]) else 0 for s in summaries]
    s_frac    = [s["s_pick_frac"] if not np.isnan(s["s_pick_frac"]) else 0 for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: analyst vs auto
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, analyst_f, label="Analyst/Manual", color="#2A9D8F", edgecolor="white")
    ax.bar(x, other_f, bottom=analyst_f, label="Auto/Unknown", color="#E9C46A", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    style_ax(ax, "Pick Quality: Analyst vs Automatic", "", "Fraction of traces")
    ax.legend(fontsize=9)

    # Right: P and S pick availability
    ax2 = axes[1]
    w = 0.35
    ax2.bar(x - w / 2, p_frac, width=w, label="P-pick available", color="#457B9D", edgecolor="white")
    ax2.bar(x + w / 2, s_frac, width=w, label="S-pick available", color="#E63946", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    style_ax(ax2, "P & S Pick Availability", "", "Fraction of traces")
    ax2.legend(fontsize=9)

    fig.suptitle("Pick Quality & Availability Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "04_pick_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved pick quality")


def plot_geographic_coverage(summaries: list[dict], out_dir: Path):
    """Global map of source epicenters, colored by dataset."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    datasets_with_geo = [s for s in summaries
                         if s["_latitude"].notna().sum() > 10]
    if not datasets_with_geo:
        print("  ⚠ No coordinate data — skipping geographic map")
        return

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.LAND,  facecolor="#E8E8E8", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#C8D8E8", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, zorder=1)
        ax.gridlines(color="white", linewidth=0.4, alpha=0.5)
        for i, s in enumerate(datasets_with_geo):
            # Subsample for readability
            lat = s["_latitude"].values
            lon = s["_longitude"].values
            n   = min(len(lat), 5000)
            idx = np.random.choice(len(lat), n, replace=False)
            ax.scatter(lon[idx], lat[idx], s=2, alpha=0.4,
                       color=PALETTE[i % len(PALETTE)],
                       label=s["dataset"], transform=ccrs.PlateCarree(), zorder=2)
        ax.legend(loc="lower left", fontsize=8, markerscale=4,
                  framealpha=0.85, ncol=2)
    else:
        # Fallback: plain scatter
        fig, ax = plt.subplots(figsize=(18, 9), facecolor="#C8D8E8")
        ax.set_facecolor("#C8D8E8")
        for i, s in enumerate(datasets_with_geo):
            lat = s["_latitude"].values
            lon = s["_longitude"].values
            n   = min(len(lat), 5000)
            idx = np.random.choice(len(lat), n, replace=False)
            ax.scatter(lon[idx], lat[idx], s=3, alpha=0.4,
                       color=PALETTE[i % len(PALETTE)], label=s["dataset"])
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.legend(loc="lower left", fontsize=8, markerscale=3,
                  framealpha=0.85, ncol=2)
        style_ax(ax, "")

    fig.suptitle("Geographic Coverage of Source Epicenters  (up to 5 000 pts/dataset)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "05_geographic_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved geographic coverage")


def plot_depth_distribution(summaries: list[dict], out_dir: Path):
    """Depth distribution — important for regional vs teleseismic."""
    datasets_with_depth = []
    for s in summaries:
        col = s["_col"]
        dep = col.get("depth_km")
        if dep is not None:
            dep = pd.to_numeric(dep, errors="coerce").dropna()
            if len(dep) > 10:
                datasets_with_depth.append((s["dataset"], dep))

    if not datasets_with_depth:
        print("  ⚠ No depth data — skipping depth plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, dep) in enumerate(datasets_with_depth):
        d = dep.clip(0, 700)
        x = np.linspace(0, 700, 300)
        try:
            kde = stats.gaussian_kde(d, bw_method=0.2)
            ax.plot(x, kde(x), label=name,
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8)
            ax.fill_between(x, kde(x), alpha=0.08,
                            color=PALETTE[i % len(PALETTE)])
        except Exception:
            pass
    ax.axvline(70,  color="steelblue", linestyle=":", linewidth=0.9, label="Crustal/mantle ~70 km")
    ax.axvline(300, color="orange",    linestyle=":", linewidth=0.9, label="Deep slab ~300 km")
    style_ax(ax, "Source Depth Distribution", "Depth (km)", "Density")
    ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("Focal Depth Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "06_depth_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved depth distribution")


def plot_mag_distance_scatter(summaries: list[dict], out_dir: Path):
    """
    M–D scatter per dataset — reveals dataset biases at a glance.
    """
    datasets_ok = [s for s in summaries
                   if s["dist_available"] and len(s["_magnitude"]) > 10]
    if not datasets_ok:
        return

    n = len(datasets_ok)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4 * nrows),
                              sharex=False, sharey=True)
    axes = np.array(axes).flatten()

    for i, s in enumerate(datasets_ok):
        ax = axes[i]
        dist = s["_distance"].values
        mag  = s["_magnitude"].values
        # align
        mask = np.isfinite(dist) & np.isfinite(mag)
        dist, mag = dist[mask], mag[mask]
        n_pts = min(len(dist), 10000)
        idx = np.random.choice(len(dist), n_pts, replace=False)
        h = ax.hexbin(np.log10(dist[idx] + 0.1), mag[idx],
                      gridsize=40, cmap="YlOrRd", mincnt=1, linewidths=0.2)
        plt.colorbar(h, ax=ax, label="count", pad=0.02)
        ax.set_xlabel("log₁₀ Distance (km)", fontsize=8)
        ax.set_ylabel("Magnitude", fontsize=8)
        style_ax(ax, s["dataset"], grid=False)

    for ax in axes[len(datasets_ok):]:
        ax.set_visible(False)

    fig.suptitle("Magnitude–Distance Space per Dataset  (hexbin density)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "07_mag_distance_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved M–D scatter")


def plot_snr_distributions(summaries: list[dict], out_dir: Path):
    """SNR distributions where available."""
    datasets_snr = [s for s in summaries
                    if s["snr_available"] and len(s["_snr"]) > 10]
    if not datasets_snr:
        print("  ⚠ No SNR data in any dataset — skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, s in enumerate(datasets_snr):
        snr = s["_snr"].clip(-20, 80)
        x = np.linspace(-20, 80, 300)
        try:
            kde = stats.gaussian_kde(snr, bw_method=0.2)
            ax.plot(x, kde(x), label=f"{s['dataset']} (med={s['snr_median']:.1f} dB)",
                    color=PALETTE[i % len(PALETTE)], linewidth=1.8)
            ax.fill_between(x, kde(x), alpha=0.1, color=PALETTE[i % len(PALETTE)])
        except Exception:
            pass
    ax.axvline(10, color="gray", linestyle="--", linewidth=0.9, label="SNR=10 dB threshold")
    style_ax(ax, "SNR Distribution (where available)", "SNR (dB)", "Density")
    ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("Signal-to-Noise Ratio Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "08_snr_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved SNR distributions")


def plot_dataset_size_comparison(summaries: list[dict], out_dir: Path):
    """Simple trace count bar chart with P/S availability overlay."""
    fig, ax = plt.subplots(figsize=(12, 5))
    names = [s["dataset"] for s in summaries]
    counts = [s["n_traces"] for s in summaries]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(summaries))]

    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.1,
                f"{cnt:,}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    style_ax(ax, "Dataset Size Comparison (trace count, log scale)", "", "# Traces (log)")

    fig.suptitle("Dataset Size Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "09_dataset_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved dataset sizes")


def plot_sampling_rates(summaries: list[dict], out_dir: Path):
    """Heatmap: dataset × sampling rate → fraction of traces."""
    all_srs = set()
    for s in summaries:
        all_srs |= set(s["sampling_rates"])
    all_srs = sorted(all_srs)
    if not all_srs:
        return

    names = [s["dataset"] for s in summaries]
    mat = np.zeros((len(names), len(all_srs)))
    for i, s in enumerate(summaries):
        col = s["_col"]
        sr_series = pd.to_numeric(col["sampling_rate"], errors="coerce").dropna()
        if len(sr_series) == 0:
            continue
        for j, sr in enumerate(all_srs):
            mat[i, j] = (sr_series == sr).sum() / len(sr_series)

    fig, ax = plt.subplots(figsize=(max(8, len(all_srs) * 0.9), max(5, len(names) * 0.5 + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_srs)))
    ax.set_xticklabels([f"{sr:.0f} Hz" for sr in all_srs], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    plt.colorbar(im, ax=ax, label="Fraction of traces")
    for i in range(len(names)):
        for j in range(len(all_srs)):
            if mat[i, j] > 0.05:
                ax.text(j, i, f"{mat[i,j]:.0%}", ha="center", va="center",
                        fontsize=7, color="navy" if mat[i, j] < 0.7 else "white")

    style_ax(ax, "Sampling Rate Distribution Across Datasets", grid=False)
    fig.suptitle("Sampling Rate Audit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "10_sampling_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → Saved sampling rates")


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════

def write_text_report(summaries: list[dict], out_dir: Path):
    lines = [
        "=" * 80,
        "  SEISBENCH DATASET METADATA AUDIT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80, "",
    ]
    for s in summaries:
        lines += [
            f"── {s['dataset']} {'─' * (60 - len(s['dataset']))}",
            f"  Traces (total):     {s['n_traces']:,}",
            f"  Unique events:      {s['n_events']:,.0f}" if not np.isnan(s['n_events']) else "  Unique events:      N/A",
            "",
            "  DISTANCE",
            f"    Available:        {'Yes' if s['dist_available'] else 'No'}",
        ]
        if s["dist_available"]:
            lines += [
                f"    Min / p10:        {s['dist_min']:.1f} km / {s['dist_p10']:.1f} km",
                f"    Median:           {s['dist_median']:.1f} km",
                f"    p90 / Max:        {s['dist_p90']:.1f} km / {s['dist_max']:.1f} km",
            ]
        lines += [
            "",
            "  MAGNITUDE",
            f"    Min / Median / Max: {s['mag_min']:.1f} / {s['mag_median']:.1f} / {s['mag_max']:.1f}",
            f"    Std dev:            {s['mag_std']:.2f}",
            "",
            "  INSTRUMENT MIX",
        ]
        for inst, cnt in sorted(s["instrument_mix"].items(), key=lambda x: -x[1]):
            total = sum(s["instrument_mix"].values()) or 1
            lines.append(f"    {inst:<25} {cnt:>7,}  ({cnt/total*100:.1f}%)")
        lines += [
            "",
            "  PICKS",
            f"    P-pick available:  {s['n_p_picks']:,}  ({s['p_pick_frac']*100:.1f}%)",
            f"    S-pick available:  {s['n_s_picks']:,}  ({s['s_pick_frac']*100:.1f}%)",
            f"    Analyst-quality:   {s['n_analyst_picks']:,}  ({s['analyst_frac']*100:.1f}%)",
            "",
            "  GEOGRAPHY",
            f"    Lat range:  {s['lat_min']:.1f}° – {s['lat_max']:.1f}°" if not np.isnan(s['lat_min']) else "    Lat range:  N/A",
            f"    Lon range:  {s['lon_min']:.1f}° – {s['lon_max']:.1f}°" if not np.isnan(s['lon_min']) else "    Lon range:  N/A",
            "",
            "  SNR",
            f"    Median SNR: {s['snr_median']:.1f} dB" if s['snr_available'] else "    Median SNR: N/A",
            "",
        ]

    lines += [
        "=" * 80,
        "  CURATION RECOMMENDATIONS",
        "=" * 80,
        "",
        "  1. STRATIFIED SAMPLING — distance bins:",
        "     Build a sampler that targets equal representation across:",
        "     Local (<50 km), Near-regional (50–200 km), Regional (200–800 km),",
        "     Far-regional (800–2000 km), Teleseismic (>2000 km).",
        "",
        "  2. INSTRUMENT RESPONSE NORMALISATION:",
        "     Deconvolve all waveforms to ground velocity before training.",
        "     Mixing raw counts across broadband / short-period / accelerometer",
        "     is a known silent killer of cross-dataset generalisation.",
        "",
        "  3. PICK QUALITY WEIGHTING:",
        "     Assign higher training loss weight to analyst-reviewed picks.",
        "     Consider downweighting or excluding autopick-only subsets for",
        "     phase-onset precision tasks.",
        "",
        "  4. GEOGRAPHIC HOLDOUT SETS:",
        "     STEAD → heavily western US → hold out Pacific Northwest subregion",
        "     IQUIQUE → Tarapacá / N. Chile subduction → hold out entirely",
        "     for generalization evaluation.",
        "",
        "  5. SAMPLING RATE HOMOGENISATION:",
        "     Resample everything to a single target rate (recommend 100 Hz).",
        "     Document datasets requiring up-/down-sampling.",
        "",
        "  6. MAGNITUDE BALANCE:",
        "     Apply magnitude-stratified sampling to prevent micro-seismic",
        "     (M<2) events from dominating loss — aim for roughly log-flat",
        "     magnitude distribution per training batch.",
        "",
    ]

    report_path = out_dir / "audit_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"  → Saved text report → {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SeisBench Metadata Audit")
    p.add_argument("--datasets", nargs="+",
                   default=["all"],
                   help="Datasets to audit (names from registry, or 'all')")
    p.add_argument("--out_dir", default="./audit_results",
                   help="Directory for output figures and report")
    p.add_argument("--skip-download", action="store_true",
                   help="Only use already-cached datasets; never attempt a download")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets == ["all"]:
        to_audit = list(DATASET_REGISTRY.items())
    else:
        to_audit = [(k, DATASET_REGISTRY[k]) for k in args.datasets
                    if k in DATASET_REGISTRY]

    print(f"\n{'='*60}")
    print(f"  SeisBench Metadata Audit — {len(to_audit)} datasets")
    print(f"  Output → {out_dir.resolve()}")
    print(f"{'='*60}\n")

    # ── Step 1: Load metadata ────────────────────────────────────────────
    if args.skip_download:
        print("[ 1/4 ]  Loading metadata (cached datasets only — download disabled)...")
    else:
        print("[ 1/4 ]  Loading metadata (will auto-download metadata CSVs if missing)...")

    raw_metas = {}
    for name, (cls, kwargs) in to_audit:
        if args.skip_download:
            # Read from cache only; skip anything not already present
            d = cls._path_internal()
            csvs = sorted(d.glob("metadata*.csv")) if d.exists() else []
            if not csvs:
                print(f"  Loading {name} ... SKIPPED (not cached)")
                continue
        result = load_metadata(name, cls, kwargs)
        if result is not None:
            raw_metas[name] = result

    if not raw_metas:
        sys.exit("No datasets loaded successfully. Check your SeisBench installation.")

    # ── Step 2: Extract & summarise ──────────────────────────────────────
    print(f"\n[ 2/4 ]  Extracting and summarising metadata for {len(raw_metas)} datasets...")
    summaries = []
    for name, df in raw_metas.items():
        print(f"  Processing {name}...")
        col = extract_columns(df)

        # Fill in missing distance from lat/lon if possible
        if col["distance_km"].isna().all():
            if col["latitude"].notna().any() and col["station_lat"].notna().any():
                print(f"    Computing distance from coordinates for {name}...")
                tmp = pd.DataFrame({
                    "latitude":    col["latitude"],
                    "longitude":   col["longitude"],
                    "station_lat": col["station_lat"],
                    "station_lon": col["station_lon"],
                })
                mask = tmp.notna().all(axis=1)
                dist_computed = tmp[mask].apply(compute_distance_from_coords, axis=1)
                col["distance_km"].loc[mask] = dist_computed

        summaries.append(build_summary(name, df, col))

    # ── Step 3: Save CSV summary ─────────────────────────────────────────
    csv_cols = [k for k in summaries[0].keys() if not k.startswith("_")]
    csv_df = pd.DataFrame([{k: s[k] for k in csv_cols} for s in summaries])
    csv_df.to_csv(out_dir / "summary_statistics.csv", index=False)
    print(f"\n  → Saved summary_statistics.csv")

    # ── Step 4: Generate figures ─────────────────────────────────────────
    print(f"\n[ 3/4 ]  Generating audit figures...")
    plot_overview_table(summaries, out_dir)
    plot_distance_distributions(summaries, out_dir)
    plot_magnitude_distributions(summaries, out_dir)
    plot_instrument_types(summaries, out_dir)
    plot_pick_quality(summaries, out_dir)
    plot_geographic_coverage(summaries, out_dir)
    plot_depth_distribution(summaries, out_dir)
    plot_mag_distance_scatter(summaries, out_dir)
    plot_snr_distributions(summaries, out_dir)
    plot_dataset_size_comparison(summaries, out_dir)
    plot_sampling_rates(summaries, out_dir)

    # ── Step 5: Text report ───────────────────────────────────────────────
    print(f"\n[ 4/4 ]  Writing text report...")
    write_text_report(summaries, out_dir)

    print(f"\n{'='*60}")
    print(f"  Audit complete.  All outputs → {out_dir.resolve()}")
    print(f"  Datasets successfully audited: {', '.join(raw_metas.keys())}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()