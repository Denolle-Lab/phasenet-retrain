#!/usr/bin/env python3
"""
eval_ensemble.py

Ensemble evaluation: averages raw P/S probability curves from multiple
fine-tuned checkpoints before peak-picking.  Proper early fusion — averaging
peak probs after argmax would give wrong residuals when the two models pick
slightly different sample locations.

Ensemble members (edit ENSEMBLE below to change):
  v3  — best recall, worst timing
  v7  — best timing, lower recall

Steps
-----
1. Load all ensemble checkpoints
2. Run inference with each model → average raw probability arrays
3. Peak-pick on averaged arrays → p_prob, p_residual_s, s_prob, s_residual_s
4. Append to step3_results.parquet keyed as ENSEMBLE_KEY
5. Recompute all metrics → update step3_metrics.csv
6. Generate comparison plots

Run from repo root:
    conda activate surface
    python scripts/eval_ensemble.py
"""

import sys
import os
from pathlib import Path

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
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from matplotlib.patches import Patch

# ── config ────────────────────────────────────────────────────────────────────
NB_DIR       = REPO_ROOT / "notebooks"
HDF5_PATH    = NB_DIR / "benchmark_waveforms.hdf5"
INDEX_PATH   = NB_DIR / "benchmark_waveforms_index.csv"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
METRICS_PATH = NB_DIR / "step3_metrics.csv"

ENSEMBLE = [
    ("jma_wc_ft_global_v3", REPO_ROOT / "checkpoints" / "finetune_jma_wc_global_v3" / "best.pt"),
    ("jma_wc_ft_global_v7", REPO_ROOT / "checkpoints" / "finetune_jma_wc_global_v7" / "best.pt"),
]
ENSEMBLE_KEY  = "jma_wc_ft_ensemble_v3v7"
ENSEMBLE_WEIGHTS = [1.0, 1.0]   # uniform average; adjust to bias toward one model

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 128
TARGET_SR     = 100
SEARCH_WIN_S  = 5.0
OUTLIER_THR_S = 1.50
THRESHOLD_P   = 0.3
THRESHOLD_S   = 0.3

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"Ensemble members: {[name for name, _ in ENSEMBLE]}")
print(f"Ensemble key    : {ENSEMBLE_KEY}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load all ensemble models
# ══════════════════════════════════════════════════════════════════════════════

def load_finetuned(ckpt_path):
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    ckpt   = torch.load(ckpt_path, map_location="cpu")
    model  = sbm.PhaseNet.from_pretrained("jma_wc", update=False)
    raw_sd = ckpt["model"]
    inner  = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
    model.load_state_dict(inner)
    model.eval()
    epoch  = ckpt.get("epoch", "?")
    vloss  = ckpt.get("val_loss", float("nan"))
    print(f"  epoch={epoch}  val_loss={vloss:.6f}  ← {ckpt_path.parent.name}")
    return model

print("\nLoading ensemble checkpoints …")
models = []
for name, path in ENSEMBLE:
    m = load_finetuned(path)
    m.to(DEVICE)
    models.append(m)

weight_sum = sum(ENSEMBLE_WEIGHTS)
norm_weights = [w / weight_sum for w in ENSEMBLE_WEIGHTS]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Run ensemble inference on benchmark
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_std(batch):
    b   = batch - batch.mean(axis=-1, keepdims=True)
    std = b.std(axis=-1, keepdims=True)
    std[std < 1e-10] = 1.0
    return (b / std).astype(np.float32)


print(f"\nLoading benchmark index …")
ok = pd.read_csv(INDEX_PATH)
ok = ok[ok["status"] == "ok"].copy().reset_index(drop=True)
print(f"  {len(ok):,} benchmark traces")

SEARCH = int(SEARCH_WIN_S * TARGET_SR)
n_in   = int(getattr(models[0], "in_samples", 3001))

results    = {}
rows_all   = [(i, row) for i, row in ok.iterrows()]

print(f"Running ensemble inference (batch={BATCH_SIZE}, device={DEVICE}) …")
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

        batch_np = _normalize_std(np.stack(waves))
        wave_len = batch_np.shape[-1]
        if wave_len < n_in:
            batch_np = np.pad(batch_np, ((0, 0), (0, 0), (0, n_in - wave_len)))

        batch_t = torch.tensor(batch_np, dtype=torch.float32).to(DEVICE)

        # Accumulate weighted sum of probability curves across ensemble members
        p_avg = np.zeros((len(meta), wave_len), dtype=np.float32)
        s_avg = np.zeros((len(meta), wave_len), dtype=np.float32)

        with torch.no_grad():
            for model, w in zip(models, norm_weights):
                out = model(batch_t)
                if isinstance(out, (tuple, list)):
                    p_full = out[0].cpu().numpy()[:len(meta), :wave_len]
                    s_full = out[1].cpu().numpy()[:len(meta), :wave_len]
                else:
                    out_np = out.cpu().numpy()
                    p_full = out_np[:len(meta), 0, :wave_len]
                    s_full = out_np[:len(meta), 1, :wave_len]
                p_avg += w * p_full
                s_avg += w * s_full

        # Peak-pick on the averaged probability curve
        for i, (idx, row) in enumerate(meta):
            p_in = int(row["p_in_window"])
            s_in = int(row["s_in_window"])

            p_prob, p_res = 0.0, np.nan
            if p_in >= 0:
                ps = max(0, p_in - SEARCH); pe = min(wave_len, p_in + SEARCH)
                pk = int(np.argmax(p_avg[i, ps:pe])) + ps
                p_prob = float(p_avg[i, pk])
                p_res  = (pk - p_in) / TARGET_SR

            s_prob, s_res = 0.0, np.nan
            if s_in >= 0:
                ss = max(0, s_in - SEARCH); se = min(wave_len, s_in + SEARCH)
                sk = int(np.argmax(s_avg[i, ss:se])) + ss
                s_prob = float(s_avg[i, sk])
                s_res  = (sk - s_in) / TARGET_SR

            results[idx] = dict(
                p_prob=round(p_prob, 4),
                p_res_s=round(float(p_res), 4) if not np.isnan(p_res) else np.nan,
                s_prob=round(s_prob, 4),
                s_res_s=round(float(s_res), 4) if not np.isnan(s_res) else np.nan,
            )

for m in models:
    m.cpu()
torch.cuda.empty_cache()
print(f"  Inference done — {len(results):,} traces")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build result rows and append to parquet
# ══════════════════════════════════════════════════════════════════════════════

ens_rows = []
for idx, row in ok.iterrows():
    pred = results.get(idx)
    if not pred:
        continue
    ens_rows.append({
        "weight":         ENSEMBLE_KEY,
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

ens_df = pd.DataFrame(ens_rows)

existing = pd.read_parquet(RESULTS_PATH)
existing = existing[existing["weight"] != ENSEMBLE_KEY]
combined = pd.concat([existing, ens_df], ignore_index=True)
combined.to_parquet(RESULTS_PATH, index=False)
print(f"Saved {len(combined):,} rows → {RESULTS_PATH}  (+{len(ens_df):,} ensemble rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Compute metrics
# ══════════════════════════════════════════════════════════════════════════════

results_df = pd.read_parquet(RESULTS_PATH)

PHASENET_WEIGHTS = {
    "stead": {"tier":"A","trained_on":"stead"},
    "instance": {"tier":"A","trained_on":"instance"},
    "neic": {"tier":"A","trained_on":"neic"},
    "diting": {"tier":"B","trained_on":None},
    "obs": {"tier":"B","trained_on":"obst2024"},
    "volpick": {"tier":"B","trained_on":None},
    "pisdl": {"tier":"B","trained_on":None},
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
    "jma_wc_ft_global_v8": {"tier":"B","trained_on":None},
    ENSEMBLE_KEY: {"tier":"B","trained_on":None},
    "scedc": {"tier":"C","trained_on":"scedc"},
    "ethz": {"tier":"C","trained_on":"ethz"},
    "iquique": {"tier":"C","trained_on":"iquique"},
    "lendb": {"tier":"C","trained_on":None},
    "original": {"tier":"C","trained_on":"stead"},
    "geofon": {"tier":"D","trained_on":None},
}

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
        y_true = np.concatenate([np.ones(len(both)), np.zeros(len(both))])
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
            d_mask    = wdf["dist_bin"] == dist_label
            sub_all   = wdf[d_mask]
            sub_cross = wdf[cross_mask & d_mask]
            sub_in    = wdf[in_mask    & d_mask]
        for sub, split in [(sub_all,"all"), (sub_cross,"cross_domain"), (sub_in,"in_domain")]:
            row = compute_metrics(sub, weight_name, split, dist_label)
            if row:
                metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_PATH, index=False)
print(f"Saved {len(metrics_df):,} metric rows → {METRICS_PATH}")

clean = metrics_df[~metrics_df["degenerate"].fillna(False)]
cross_all_df_full = (clean[(clean["split"] == "cross_domain") & (clean["dist_bin"] == "all")]
                     .sort_values("p_mae_s"))
print("\nCross-domain P-MAE ranking (all distances):")
print(cross_all_df_full[["weight","tier","p_mae_s","s_mae_s","p_recall","mcc",
                          "p_outlier"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Visualisations
# ══════════════════════════════════════════════════════════════════════════════

HIGHLIGHT  = "#E74C3C"   # red — ensemble
V3_COLOR   = "#FF8C42"
V7_COLOR   = "#16A085"
PARENT     = "#D25F10"

FOCUS_MODELS = [ENSEMBLE_KEY, "jma_wc_ft_global_v7", "jma_wc_ft_global_v3", "jma_wc", "instance", "stead"]
COLORS = {
    ENSEMBLE_KEY:           HIGHLIGHT,
    "jma_wc_ft_global_v7":  V7_COLOR,
    "jma_wc_ft_global_v3":  V3_COLOR,
    "jma_wc":               PARENT,
    "instance":             "#2ca02c",
    "stead":                "#1f77b4",
}
LABELS = {
    ENSEMBLE_KEY:           "ensemble (v3+v7) ★",
    "jma_wc_ft_global_v7":  "v7",
    "jma_wc_ft_global_v3":  "v3",
    "jma_wc":               "jma_wc (base)",
    "instance":             "instance",
    "stead":                "stead",
}

cross     = metrics_df[(metrics_df["split"] == "cross_domain") &
                       (~metrics_df["degenerate"].fillna(False))]
cross_all = cross[cross["dist_bin"] == "all"].set_index("weight")
dist_order = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)"]
cross_dist = cross[cross["dist_bin"].isin(dist_order)]

all_models_sorted = cross_all["p_mae_s"].dropna().sort_values().index.tolist()


# ── Figure 1 — Dashboard ──────────────────────────────────────────────────────
print("\nGenerating Figure 1: dashboard …")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Ensemble ({', '.join(n for n, _ in ENSEMBLE)}) vs Baselines — Cross-Domain",
             fontsize=13, fontweight="bold", y=0.98)

metrics_to_plot = [
    ("p_mae_s",   "P-MAE (s)",           False),
    ("s_mae_s",   "S-MAE (s)",           False),
    ("p_recall",  "P-Recall @ thr=0.3",  True),
    ("s_recall",  "S-Recall @ thr=0.3",  True),
    ("mcc",       "MCC (Phase ID)",       True),
    ("p_outlier", "P-Outlier fraction",   False),
]

for ax, (col, ylabel, higher_better) in zip(axes.flat, metrics_to_plot):
    vals, bar_colors, labels_list = [], [], []
    for w in all_models_sorted:
        if w not in cross_all.index or np.isnan(cross_all.loc[w, col]):
            continue
        vals.append(cross_all.loc[w, col])
        if w == ENSEMBLE_KEY:
            bar_colors.append(HIGHLIGHT)
        elif w == "jma_wc":
            bar_colors.append(PARENT)
        elif w in ("jma_wc_ft_global_v7", "jma_wc_ft_global_v3"):
            bar_colors.append(COLORS[w])
        else:
            bar_colors.append("#aaaaaa")
        labels_list.append(LABELS.get(w, w))

    y_pos = range(len(vals))
    bars = ax.barh(list(y_pos), vals, color=bar_colors, edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_list, fontsize=7.5)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.invert_yaxis()
    best_idx = vals.index(min(vals)) if not higher_better else vals.index(max(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.5)
    ens_label = LABELS[ENSEMBLE_KEY]
    if ens_label in labels_list:
        ax.axvline(vals[labels_list.index(ens_label)],
                   color=HIGHLIGHT, lw=1.2, ls="--", alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Patch(color=HIGHLIGHT, label=f"ensemble ({'+'.join(n.split('_')[-1] for n, _ in ENSEMBLE)})"),
    Patch(color=V7_COLOR,  label="v7"),
    Patch(color=V3_COLOR,  label="v3"),
    Patch(color=PARENT,    label="jma_wc (base)"),
    Patch(color="#aaaaaa", label="other pretrained"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
out1 = NB_DIR / "step3_ensemble_dashboard.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out1}")


# ── Figure 2 — P-MAE by distance bin ─────────────────────────────────────────
print("Generating Figure 2: distance-bin breakdown …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Timing Error by Distance Bin — Cross-Domain (ensemble vs members vs baselines)",
             fontsize=11, fontweight="bold")

for ax, (phase, col) in zip(axes, [("P", "p_mae_s"), ("S", "s_mae_s")]):
    x = np.arange(len(dist_order))
    w = 0.13
    for i, model in enumerate(FOCUS_MODELS):
        vals = []
        for d in dist_order:
            row = cross_dist[(cross_dist["weight"] == model) &
                             (cross_dist["dist_bin"] == d)]
            vals.append(row[col].values[0]
                        if len(row) > 0 and not row[col].isna().all() else np.nan)
        offset = (i - len(FOCUS_MODELS) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, width=w,
                      label=LABELS.get(model, model),
                      color=COLORS.get(model, "#aaaaaa"),
                      edgecolor="white", zorder=3)
        if model == ENSEMBLE_KEY:
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Local\n(<150 km)", "Regional\n(150–1500 km)",
                        "Teleseismic\n(>1500 km)"], fontsize=9)
    ax.set_ylabel(f"{phase}-MAE (s)", fontsize=10)
    ax.set_title(f"{phase}-wave Timing Error", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = NB_DIR / "step3_ensemble_distance_bins.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out2}")


# ── Figure 3 — P residual histograms comparing ensemble vs members ────────────
print("Generating Figure 3: residual histograms …")

compare = [ENSEMBLE_KEY, "jma_wc_ft_global_v7", "jma_wc_ft_global_v3", "jma_wc"]
hist_colors = [HIGHLIGHT, V7_COLOR, V3_COLOR, PARENT]

fig, axes = plt.subplots(len(dist_order), 2, figsize=(13, 10))
fig.suptitle("P and S Pick Residual Distributions — Cross-Domain",
             fontsize=12, fontweight="bold")
bins = np.linspace(-2.0, 2.0, 60)

for row_i, dist in enumerate(dist_order):
    for col_i, (phase, res_col, in_col) in enumerate(
            [("P", "p_residual_s", "p_in_window"),
             ("S", "s_residual_s", "s_in_window")]):
        ax = axes[row_i, col_i]
        for model, color in zip(compare, hist_colors):
            mdf = results_df[(results_df["weight"] == model) &
                             (results_df["dist_bin"] == dist) &
                             (results_df[in_col] >= 0)]
            res = mdf[res_col].dropna()
            if len(res) < 10:
                continue
            lw = 2.0 if model == ENSEMBLE_KEY else 1.0
            ax.hist(res.clip(-2, 2), bins=bins, alpha=0.55, color=color,
                    label=LABELS.get(model, model), density=True,
                    linewidth=lw, edgecolor=color if model == ENSEMBLE_KEY else "none")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlim(-2, 2)
        ax.set_xlabel(f"{phase} residual (s)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(f"{dist} — {phase}-wave", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out3 = NB_DIR / "step3_ensemble_residuals.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out3}")


# ── Figure 4 — Recall curves ──────────────────────────────────────────────────
print("Generating Figure 4: recall curves …")

thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
t_keys_p   = [f"p_recall_t{int(t*10):02d}" for t in thresholds]
t_keys_s   = [f"s_recall_t{int(t*10):02d}" for t in thresholds]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Detection Recall vs Probability Threshold — Cross-Domain",
             fontsize=11, fontweight="bold")

for ax, (phase, t_keys) in zip(axes, [("P", t_keys_p), ("S", t_keys_s)]):
    for model in FOCUS_MODELS:
        if model not in cross_all.index:
            continue
        row = cross_all.loc[model]
        vals = [row[k] if k in row.index and not np.isnan(row[k]) else np.nan
                for k in t_keys]
        lw = 2.5 if model == ENSEMBLE_KEY else 1.5
        ls = "-" if model in (ENSEMBLE_KEY, "jma_wc_ft_global_v7", "jma_wc_ft_global_v3") else "--"
        ax.plot(thresholds, vals,
                color=COLORS.get(model, "#aaaaaa"),
                lw=lw, ls=ls, marker="o", ms=4,
                label=LABELS.get(model, model))
    ax.set_xlabel("Probability threshold", fontsize=10)
    ax.set_ylabel(f"{phase}-Recall", fontsize=10)
    ax.set_title(f"{phase}-wave Detection Recall", fontsize=10)
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out4 = NB_DIR / "step3_ensemble_recall_curves.png"
fig.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out4}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print(f"SUMMARY — {ENSEMBLE_KEY} vs members and baselines (cross-domain, all distances)")
print("═" * 70)
cols = ["weight", "p_mae_s", "s_mae_s", "p_recall", "s_recall", "mcc", "p_outlier"]
focus_rows = cross_all_df_full[cross_all_df_full["weight"].isin(
    [ENSEMBLE_KEY, "jma_wc_ft_global_v7", "jma_wc_ft_global_v3",
     "jma_wc", "instance", "stead"])][cols]
print(focus_rows.to_string(index=False))
print(f"\nPlots saved to {NB_DIR}/")
