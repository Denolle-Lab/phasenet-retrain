#!/usr/bin/env python3
"""
eval_ensemble_eqt.py

Ensemble of eqt_volpick + eqt_original_nonconservative evaluated on the
6k benchmark.  Both models share the EQTransformer architecture but differ
in training data and normalisation:

  eqt_volpick               norm=peak  (volcano-tectonic, Zhong & Tan 2024)
  eqt_original_nonconservative  norm=std   (STEAD, original weights)

Strategy: normalise each model's input batch separately, run inference,
then average the P and S probability curves before picking.

Run from repo root:
    conda activate surface
    python scripts/eval_ensemble_eqt.py

Outputs
-------
  notebooks/step3_results.parquet      — updated with eqt_ensemble row
  results/eval_ensemble_eqt.csv        — metrics table
  results/eval_ensemble_eqt.png        — comparison figure
"""

import sys, os
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
from matplotlib.patches import Patch
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
NB_DIR       = REPO_ROOT / "notebooks"
HDF5_PATH    = NB_DIR / "benchmark_waveforms_6k.hdf5"
INDEX_PATH   = NB_DIR / "benchmark_waveforms_6k_index.csv"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
OUT_CSV      = REPO_ROOT / "results" / "eval_ensemble_eqt.csv"
OUT_PNG      = REPO_ROOT / "results" / "eval_ensemble_eqt.png"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 32       # two forward passes per batch
TARGET_SR     = 100
SEARCH_WIN_S  = 5.0
OUTLIER_THR   = 1.5
THRESHOLD     = 0.3
WAVE_LEN      = 6000
ENSEMBLE_LABEL = "eqt_ensemble_volpick_nc"

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

SEARCH = int(SEARCH_WIN_S * TARGET_SR)


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalize(batch: np.ndarray, norm: str) -> np.ndarray:
    b = batch - batch.mean(axis=-1, keepdims=True)
    if norm == "peak":
        scale = np.abs(b).max(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
    else:
        scale = b.std(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
    return (b / scale).astype(np.float32)


# ── Load benchmark index ───────────────────────────────────────────────────────

print("\nLoading benchmark index …")
ok = pd.read_csv(INDEX_PATH)
ok = ok[ok["status"] == "ok"].copy().reset_index(drop=True)
print(f"  {len(ok):,} benchmark traces")


# ── Load both models ───────────────────────────────────────────────────────────

print("\nLoading models …")
model_vp = sbm.EQTransformer.from_pretrained("volpick", update=False)
model_nc = sbm.EQTransformer.from_pretrained("original_nonconservative", update=False)

norm_vp = getattr(model_vp, "norm", "std")   # peak
norm_nc = getattr(model_nc, "norm", "std")   # std
print(f"  volpick              norm={norm_vp}")
print(f"  original_nc          norm={norm_nc}")

model_vp.eval().to(DEVICE)
model_nc.eval().to(DEVICE)


# ── Ensemble inference ─────────────────────────────────────────────────────────

print("\nRunning ensemble inference …")
rows_all = list(ok.iterrows())
results  = {}

with h5py.File(HDF5_PATH, "r") as hf:
    for start in tqdm(range(0, len(rows_all), BATCH_SIZE), desc="  ensemble"):
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

        raw = np.stack(waves)   # (B, 3, 6000)

        # Two separate normalised batches
        batch_vp = torch.tensor(normalize(raw, norm_vp), dtype=torch.float32).to(DEVICE)
        batch_nc = torch.tensor(normalize(raw, norm_nc), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            out_vp = model_vp(batch_vp)   # (detector, P, S)
            out_nc = model_nc(batch_nc)

        # Average probability curves
        p_full = (out_vp[1].cpu().numpy() + out_nc[1].cpu().numpy()) / 2.0
        s_full = (out_vp[2].cpu().numpy() + out_nc[2].cpu().numpy()) / 2.0
        p_full = p_full[:, :WAVE_LEN]
        s_full = s_full[:, :WAVE_LEN]

        for i, (idx, row) in enumerate(meta):
            p_in = int(row["p_in_window"])
            s_in = int(row["s_in_window"])

            p_prob, p_res = 0.0, np.nan
            if p_in >= 0:
                ps = max(0, p_in - SEARCH)
                pe = min(WAVE_LEN, p_in + SEARCH)
                pk = int(np.argmax(p_full[i, ps:pe])) + ps
                p_prob = float(p_full[i, pk])
                p_res  = (pk - p_in) / TARGET_SR

            s_prob, s_res = 0.0, np.nan
            if s_in >= 0:
                ss = max(0, s_in - SEARCH)
                se = min(WAVE_LEN, s_in + SEARCH)
                sk = int(np.argmax(s_full[i, ss:se])) + ss
                s_prob = float(s_full[i, sk])
                s_res  = (sk - s_in) / TARGET_SR

            results[idx] = dict(
                p_prob  = round(p_prob, 4),
                p_res_s = round(float(p_res), 4) if not np.isnan(p_res) else np.nan,
                s_prob  = round(s_prob, 4),
                s_res_s = round(float(s_res), 4) if not np.isnan(s_res) else np.nan,
            )

model_vp.cpu(); model_nc.cpu()
torch.cuda.empty_cache()
print(f"  {len(results):,} traces processed")


# ── Build rows ─────────────────────────────────────────────────────────────────

new_rows = []
for idx, row in ok.iterrows():
    pred = results.get(idx)
    if not pred:
        continue
    new_rows.append({
        "weight":         ENSEMBLE_LABEL,
        "tier":           "EQT",
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

# ── Update parquet ─────────────────────────────────────────────────────────────

existing = pd.read_parquet(RESULTS_PATH)
combined = pd.concat(
    [existing[existing["weight"] != ENSEMBLE_LABEL],
     pd.DataFrame(new_rows)],
    ignore_index=True,
)
combined.to_parquet(RESULTS_PATH, index=False)
print(f"Saved {len(combined):,} rows → {RESULTS_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df, weight_name, split, dist_label="all"):
    if len(df) == 0:
        return None
    p_tr = df[df["p_in_window"] >= 0]
    s_tr = df[df["s_in_window"] >= 0]

    p_recall = (p_tr["p_prob"] >= THRESHOLD).mean() if len(p_tr) > 0 else np.nan
    s_recall = (s_tr["s_prob"] >= THRESHOLD).mean() if len(s_tr) > 0 else np.nan

    both = df[(df["p_in_window"] >= 0) & (df["s_in_window"] >= 0)]
    mcc  = np.nan
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

    p_res = p_tr["p_residual_s"].dropna()
    s_res = s_tr["s_residual_s"].dropna()

    return dict(
        weight    = weight_name,
        split     = split,
        dist_bin  = dist_label,
        n_traces  = len(df),
        p_recall  = round(p_recall, 4)                             if not np.isnan(p_recall) else np.nan,
        s_recall  = round(s_recall, 4)                             if not np.isnan(s_recall) else np.nan,
        p_mae_s   = round(np.abs(p_res).mean(), 4)                 if len(p_res) > 0 else np.nan,
        s_mae_s   = round(np.abs(s_res).mean(), 4)                 if len(s_res) > 0 else np.nan,
        p_outlier = round((np.abs(p_res) > OUTLIER_THR).mean(), 4) if len(p_res) > 0 else np.nan,
        s_outlier = round((np.abs(s_res) > OUTLIER_THR).mean(), 4) if len(s_res) > 0 else np.nan,
        mcc       = round(mcc, 4)                                  if not np.isnan(mcc)      else np.nan,
    )


results_df = pd.read_parquet(RESULTS_PATH)
DIST_BINS  = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]

COMPARE_WEIGHTS = [
    (ENSEMBLE_LABEL,                  "stead"),   # original_nc component was trained on STEAD
    ("eqt_volpick",                   None),
    ("eqt_original_nonconservative",  "stead"),
    ("jma_wc_ft_global_v7",           None),
    ("jma_wc",                        None),
    ("eqt_scedc",                     None),
    ("eqt_instance",                  "instance"),
]

metrics_rows = []
for weight_name, trained_on in COMPARE_WEIGHTS:
    wdf = results_df[results_df["weight"] == weight_name]
    if len(wdf) == 0:
        continue
    if trained_on:
        cross_mask = ~wdf["trained_models"].str.contains(trained_on, na=False, regex=False)
    else:
        cross_mask = pd.Series(True, index=wdf.index)

    for dist in DIST_BINS:
        sub   = wdf if dist == "all" else wdf[wdf["dist_bin"] == dist]
        sub_x = sub[cross_mask.reindex(sub.index, fill_value=True)]
        for df_s, split in [(sub, "all"), (sub_x, "cross_domain")]:
            row = compute_metrics(df_s, weight_name, split, dist)
            if row:
                metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_CSV, index=False)
print(f"Saved metrics → {OUT_CSV}")


# ── Print summary table ────────────────────────────────────────────────────────

cross = metrics_df[(metrics_df["split"] == "cross_domain") &
                   (metrics_df["dist_bin"] == "all")].copy()
cross = cross.sort_values("p_mae_s")

print("\n" + "=" * 100)
print("Ensemble vs components vs baselines — cross_domain, all distances")
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

# ── Per-distance-bin breakdown ─────────────────────────────────────────────────

print("\n── Per-distance bin (cross_domain) ───────────────────────────────────────────")
KEY_MODELS = [ENSEMBLE_LABEL, "eqt_volpick", "eqt_original_nonconservative",
              "jma_wc_ft_global_v7"]
for metric, label in [("p_mae_s", "P-MAE (s)"), ("p_recall", "P-Recall"),
                       ("s_recall", "S-Recall")]:
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


# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating figure …")

PANEL_METRICS = [
    ("p_mae_s",  "P-MAE (s)",       False),
    ("p_recall", "P-Recall @ 0.30", True),
    ("s_recall", "S-Recall @ 0.30", True),
    ("mcc",      "MCC",             True),
]

COLORS = {
    ENSEMBLE_LABEL:                  "#E74C3C",
    "eqt_volpick":                   "#9B59B6",
    "eqt_original_nonconservative":  "#3498DB",
    "jma_wc_ft_global_v7":           "#E6821A",
    "jma_wc":                        "#2C7BB6",
    "eqt_scedc":                     "#1ABC9C",
    "eqt_instance":                  "#95A5A6",
}

display_names = {
    ENSEMBLE_LABEL:                  "eqt ensemble (volpick + nc)",
    "eqt_volpick":                   "eqt_volpick",
    "eqt_original_nonconservative":  "eqt_original_nc",
    "jma_wc_ft_global_v7":           "jma_wc_ft_v7 (best fine-tune)",
    "jma_wc":                        "jma_wc (parent)",
    "eqt_scedc":                     "eqt_scedc",
    "eqt_instance":                  "eqt_instance",
}

order = list(COLORS.keys())

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    "EQT ensemble (volpick + original_nc) vs components and PhaseNet baselines\n"
    "Cross-domain benchmark — all distances — thr=0.30",
    fontsize=11, fontweight="bold", y=1.02,
)

for ax, (col, xlabel, higher) in zip(axes, PANEL_METRICS):
    vals, colors, labels_ = [], [], []
    for w in order:
        r = cross[cross["weight"] == w]
        if r.empty or np.isnan(r.iloc[0][col]):
            continue
        vals.append(r.iloc[0][col])
        colors.append(COLORS[w])
        labels_.append(display_names[w])

    y_pos = list(range(len(vals)))
    bars  = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    if vals:
        best_i = vals.index(min(vals) if not higher else max(vals))
        bars[best_i].set_edgecolor("black")
        bars[best_i].set_linewidth(2.0)

legend_elems = [
    Patch(color=COLORS[ENSEMBLE_LABEL], label="Ensemble (volpick + nc)"),
    Patch(color=COLORS["eqt_volpick"],  label="eqt_volpick"),
    Patch(color=COLORS["eqt_original_nonconservative"], label="eqt_original_nc"),
    Patch(color=COLORS["jma_wc_ft_global_v7"], label="v7 (best fine-tune)"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PNG}")
print("\nDone.")
