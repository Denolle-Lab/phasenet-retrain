#!/usr/bin/env python3
"""
eval_finetuned.py

Evaluates the fine-tuned jma_wc checkpoint on the benchmark dataset and
compares it against every existing pretrained weight in step3_results.parquet.

Steps
-----
1. Load fine-tuned checkpoint → run inference on benchmark_waveforms.hdf5
2. Append results to step3_results.parquet (keyed as "jma_wc_ft")
3. Compute all metrics (same logic as notebook cell 10)
4. Append metric rows to step3_metrics.csv
5. Generate comparison plots → notebooks/step3_ft_comparison*.png

Run from repo root:
    conda activate surface
    python scripts/eval_finetuned.py
"""

import sys
import os
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/data/wsd04/ak287/.seisbench")
import seisbench
seisbench.cache_root = "/data/wsd04/ak287/.seisbench"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import h5py
import torch
import seisbench.models as sbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# ── CLI args ──────────────────────────────────────────────────────────────────
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--ckpt",       default=None, help="Path to best.pt checkpoint")
_parser.add_argument("--ft-weight",  default=None, help="Weight label (e.g. jma_wc_ft_global_v9)")
_args, _ = _parser.parse_known_args()

# ── constants ─────────────────────────────────────────────────────────────────
NB_DIR       = REPO_ROOT / "notebooks"
CKPT_PATH    = Path(_args.ckpt) if _args.ckpt else REPO_ROOT / "checkpoints" / "finetune_jma_wc_global_v8" / "best.pt"
HDF5_PATH    = NB_DIR / "benchmark_waveforms.hdf5"
INDEX_PATH   = NB_DIR / "benchmark_waveforms_index.csv"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
METRICS_PATH = NB_DIR / "step3_metrics.csv"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 128
TARGET_SR     = 100
SEARCH_WIN_S  = 5.0
OUTLIER_THR_S = 1.50
THRESHOLD_P   = 0.3
THRESHOLD_S   = 0.3
FT_WEIGHT     = _args.ft_weight if _args.ft_weight else "jma_wc_ft_global_v8"

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load fine-tuned model
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading fine-tuned checkpoint: {CKPT_PATH}")
assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
# Strip "model." prefix shared by both PhaseNetFinetune and PhaseNetScratch checkpoints.
raw_sd   = ckpt["model"]
inner_sd = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}

# Scratch checkpoints use the default (smaller) PhaseNet architecture;
# fine-tune checkpoints were trained from jma_wc (4x larger).
if "scratch" in FT_WEIGHT.lower():
    ft_model = sbm.PhaseNet(in_channels=3, classes=3, phases="PSN", sampling_rate=100)
else:
    ft_model = sbm.PhaseNet.from_pretrained("jma_wc", update=False)
ft_model.load_state_dict(inner_sd)
ft_model.eval()
ft_model.to(DEVICE)
print(f"  Loaded epoch={ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', float('nan')):.6f}")
print(f"  Norm type : {getattr(ft_model, 'norm', 'std')}  (jma_wc inherits 'std')")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Run inference on benchmark
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_std(batch):
    """Per-component demean + unit-std (matches jma_wc training norm)."""
    b   = batch - batch.mean(axis=-1, keepdims=True)
    std = b.std(axis=-1, keepdims=True)
    std[std < 1e-10] = 1.0
    return (b / std).astype(np.float32)


print(f"\nLoading benchmark index …")
ok = pd.read_csv(INDEX_PATH)
ok = ok[ok["status"] == "ok"].copy().reset_index(drop=True)
print(f"  {len(ok):,} benchmark traces")

SEARCH = int(SEARCH_WIN_S * TARGET_SR)
n_in   = int(getattr(ft_model, "in_samples", 3001))

results = {}
rows_all = [(i, row) for i, row in ok.iterrows()]

print(f"Running inference (batch={BATCH_SIZE}, device={DEVICE}) …")
with h5py.File(HDF5_PATH, "r") as hf:
    for start in tqdm(range(0, len(rows_all), BATCH_SIZE), desc="Batches"):
        batch_rows = rows_all[start: start + BATCH_SIZE]
        waves, meta = [], []
        for idx, row in batch_rows:
            tname = row["trace_name"]
            if tname not in hf["waveforms"]:
                continue
            waves.append(hf["waveforms"][tname][:])
            meta.append((idx, row))
        if not waves:
            continue

        batch_np  = _normalize_std(np.stack(waves))
        wave_len  = batch_np.shape[-1]
        if wave_len < n_in:
            batch_np = np.pad(batch_np, ((0, 0), (0, 0), (0, n_in - wave_len)))

        batch_t = torch.tensor(batch_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = ft_model(batch_t)

        if isinstance(out, (tuple, list)):
            p_full = out[0].cpu().numpy()[:, :wave_len]
            s_full = out[1].cpu().numpy()[:, :wave_len]
        else:
            out_np = out.cpu().numpy()
            p_full = out_np[:, 0, :wave_len]  # ch0=P (PSN ordering, matches jma_wc)
            s_full = out_np[:, 1, :wave_len]  # ch1=S

        for i, (idx, row) in enumerate(meta):
            p_in = int(row["p_in_window"])
            s_in = int(row["s_in_window"])

            p_prob, p_res = 0.0, np.nan
            if p_in >= 0:
                ps = max(0, p_in - SEARCH); pe = min(wave_len, p_in + SEARCH)
                pk = int(np.argmax(p_full[i, ps:pe])) + ps
                p_prob = float(p_full[i, pk])
                p_res  = (pk - p_in) / TARGET_SR

            s_prob, s_res = 0.0, np.nan
            if s_in >= 0:
                ss = max(0, s_in - SEARCH); se = min(wave_len, s_in + SEARCH)
                sk = int(np.argmax(s_full[i, ss:se])) + ss
                s_prob = float(s_full[i, sk])
                s_res  = (sk - s_in) / TARGET_SR

            results[idx] = dict(
                p_prob=round(p_prob, 4),
                p_res_s=round(float(p_res), 4) if not np.isnan(p_res) else np.nan,
                s_prob=round(s_prob, 4),
                s_res_s=round(float(s_res), 4) if not np.isnan(s_res) else np.nan,
            )

ft_model.cpu()
torch.cuda.empty_cache()
print(f"  Inference done — {len(results):,} traces")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build results rows and append to parquet
# ══════════════════════════════════════════════════════════════════════════════

ft_rows = []
for idx, row in ok.iterrows():
    pred = results.get(idx)
    if not pred:
        continue
    ft_rows.append({
        "weight":         FT_WEIGHT,
        "tier":           "B",
        "trace_name":     row["trace_name"],
        "dataset":        row["dataset"],
        "dist_bin":       row.get("dist_bin",  np.nan),
        "depth_bin":      row.get("depth_bin", np.nan),
        "mag_bin":        row.get("mag_bin",   np.nan),
        "trained_models": str(row.get("trained_models", "")),
        "snr_db":         row.get("snr_db",    np.nan),
        "p_in_window":    int(row["p_in_window"]),
        "s_in_window":    int(row["s_in_window"]),
        "p_prob":         pred["p_prob"],
        "s_prob":         pred["s_prob"],
        "p_residual_s":   pred["p_res_s"],
        "s_residual_s":   pred["s_res_s"],
    })

ft_df = pd.DataFrame(ft_rows)

# Load existing results, drop any prior jma_wc_ft rows, append new
existing = pd.read_parquet(RESULTS_PATH)
existing = existing[existing["weight"] != FT_WEIGHT]
combined = pd.concat([existing, ft_df], ignore_index=True)
combined.to_parquet(RESULTS_PATH, index=False)
print(f"Saved {len(combined):,} rows → {RESULTS_PATH}  (+{len(ft_df):,} ft rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Compute metrics (same logic as notebook cell 10)
# ══════════════════════════════════════════════════════════════════════════════

results_df = pd.read_parquet(RESULTS_PATH)

PHASENET_WEIGHTS = {
    "stead": {"tier":"A","trained_on":"stead"},
    "instance": {"tier":"A","trained_on":"instance"},
    "neic": {"tier":"A","trained_on":"neic"},
    "diting": {"tier":"B","trained_on":None},
    "obs": {"tier":"B","trained_on":"obst2024"},
    "volpick": {"tier":"B","trained_on":None},
    "pisdl": {"tier":"B","trained_on":"pisdl"},
    "phasenet_sn": {"tier":"B","trained_on":None},
    "jma": {"tier":"B","trained_on":None},
    "jma_wc": {"tier":"B","trained_on":None},
    "jma_wc_ft": {"tier":"B","trained_on":None},
    "jma_wc_ft_frozen": {"tier":"B","trained_on":None},
    "jma_wc_ft_noise": {"tier":"B","trained_on":None},
    "jma_wc_ft_global_v3": {"tier":"B","trained_on":None},
    "jma_wc_ft_global_v4": {"tier":"B","trained_on":None},
    "jma_wc_ft_global_v5": {"tier":"B","trained_on":None},
    "jma_wc_ft_global_v6": {"tier":"B","trained_on":None},
    "jma_wc_ft_global_v7": {"tier":"B","trained_on":None},
    FT_WEIGHT: {"tier":"B","trained_on":None},
    "scedc": {"tier":"C","trained_on":"scedc"},
    "ethz": {"tier":"C","trained_on":"ethz"},
    "iquique": {"tier":"C","trained_on":"iquique"},
    "lendb": {"tier":"C","trained_on":None},
    "original": {"tier":"C","trained_on":"stead"},
    "geofon": {"tier":"D","trained_on":None},
}

# Degenerate model detection
DEGENERATE_MODELS = set()
for wname in results_df["weight"].unique():
    wdf = results_df[results_df["weight"] == wname]
    rec = (wdf["p_prob"] >= THRESHOLD_P).mean()
    mae = wdf["p_residual_s"].abs().mean()
    if rec > 0.99 and mae > 2.0:
        DEGENERATE_MODELS.add(wname)
if DEGENERATE_MODELS:
    print(f"Degenerate models: {sorted(DEGENERATE_MODELS)}")


def compute_metrics(df, weight_name, split_name, dist_label=None):
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

    both = df[(df["p_in_window"] >= 0) & (df["s_in_window"] >= 0)].copy()
    mcc  = np.nan
    if len(both) >= 5:
        y_true = np.concatenate([np.ones(len(both)),  np.zeros(len(both))])
        y_pred = np.concatenate(
            [(both["p_prob"] > both["s_prob"]).astype(int).values,
             (both["s_prob"] > both["p_prob"]).astype(int).values])
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
        "tier":          PHASENET_WEIGHTS.get(weight_name, {}).get("tier", "?"),
        "split":         split_name,
        "dist_bin":      dist_label or "all",
        "n_traces":      len(df),
        "degenerate":    weight_name in DEGENERATE_MODELS,
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


print("\nComputing metrics …")
metrics_rows = []
dist_bins = results_df["dist_bin"].dropna().unique().tolist() + ["all"]

for weight_name in tqdm(results_df["weight"].unique(), desc="Metrics"):
    wdf        = results_df[results_df["weight"] == weight_name]
    trained_on = PHASENET_WEIGHTS.get(weight_name, {}).get("trained_on", None)
    if trained_on:
        in_mask    = wdf["trained_models"].str.contains(trained_on, na=False, regex=False)
        cross_mask = ~in_mask
    else:
        in_mask    = pd.Series(False, index=wdf.index)
        cross_mask = pd.Series(True,  index=wdf.index)

    for dist_label in dist_bins:
        if dist_label == "all":
            sub_all, sub_cross, sub_in = wdf, wdf[cross_mask], wdf[in_mask]
        else:
            d_mask     = wdf["dist_bin"] == dist_label
            sub_all    = wdf[d_mask]
            sub_cross  = wdf[cross_mask & d_mask]
            sub_in     = wdf[in_mask    & d_mask]
        for sub, split in [(sub_all,"all"), (sub_cross,"cross_domain"), (sub_in,"in_domain")]:
            row = compute_metrics(sub, weight_name, split, dist_label)
            if row:
                metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_PATH, index=False)
print(f"Saved {len(metrics_df):,} metric rows → {METRICS_PATH}")

# Quick summary
clean = metrics_df[~metrics_df["degenerate"].fillna(False)]
cross_all = (clean[(clean["split"] == "cross_domain") & (clean["dist_bin"] == "all")]
             .sort_values("p_mae_s"))
print("\nCross-domain P-MAE ranking (all distances):")
print(cross_all[["weight","tier","p_mae_s","s_mae_s","p_recall","mcc",
                  "p_outlier"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Visualisations
# ══════════════════════════════════════════════════════════════════════════════

# ── Palette ───────────────────────────────────────────────────────────────────
# Fine-tuned model gets a distinct gold colour; jma_wc parent is darker orange;
# top baselines are various blues/greens.
HIGHLIGHT  = "#E6A817"   # gold  — jma_wc_ft
PARENT     = "#D25F10"   # burnt orange — jma_wc
TOP_MODELS = ["stead", "instance", "neic", "diting", FT_WEIGHT,
              "jma_wc_ft_global_v6", "jma_wc_ft_global_v5", "jma_wc_ft_global_v4", "jma_wc_ft_global_v3", "jma_wc_ft_frozen", "jma_wc_ft_noise", "jma_wc_ft", "jma_wc"]
COLORS     = {
    "stead":                  "#1f77b4",
    "instance":               "#2ca02c",
    "neic":                   "#9467bd",
    "diting":                 "#8c564b",
    FT_WEIGHT:                "#16A085",   # teal — v7 global ft
    "jma_wc_ft_global_v6":    "#27AE60",   # green — v6 global ft
    "jma_wc_ft_global_v5":    "#C0392B",   # deep red — v5 global ft
    "jma_wc_ft_global_v4":    "#E65C00",   # deep orange — v4 global ft
    "jma_wc_ft_global_v3":    "#FF8C42",   # lighter orange — v3 global ft
    "jma_wc_ft_frozen":       HIGHLIGHT,   # gold — frozen ft
    "jma_wc_ft_noise":        "#F4C542",   # lighter gold
    "jma_wc_ft":              "#D4B442",   # muted gold
    "jma_wc":                 PARENT,
}
LABELS = {
    "stead":                  "stead",
    "instance":               "instance",
    "neic":                   "neic",
    "diting":                 "diting",
    FT_WEIGHT:                "jma_wc_ft_global_v8 ★",
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
print("\nGenerating Figure 1: dashboard …")

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

from matplotlib.patches import Patch
legend_handles = [
    Patch(color=COLORS[FT_WEIGHT], label="jma_wc_ft_global_v8 (this work)"),
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
        # bold edge on fine-tuned bars
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


# ── Summary print ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("SUMMARY — jma_wc_ft_global_v8 vs key baselines (cross-domain, all distances)")
print("═" * 70)
cols = ["weight", "p_mae_s", "s_mae_s", "p_recall", "s_recall", "mcc", "p_outlier"]
summary = (cross_all_df.reset_index()
           .loc[cross_all_df.reset_index()["weight"].isin(TOP_MODELS), cols]
           .sort_values("p_mae_s"))
print(summary.to_string(index=False))
print(f"\nPlots saved to {NB_DIR}/")
