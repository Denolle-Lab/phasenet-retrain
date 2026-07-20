#!/usr/bin/env python3
"""
eval_ensemble_v7v11.py

Ensemble evaluation: average the P/S probability curves from v7 and v11
BEFORE peak detection, then compute metrics on the merged curves.

Why average curves (not probabilities):
  The parquet only stores the peak p_prob and residual, not the full curve.
  Averaging peak-probabilities would not account for cases where the two
  models disagree on WHICH sample is the peak — averaging the full arrays
  finds a consensus peak that is better localised than either model alone.

Outputs
-------
  Appends "jma_wc_ft_ensemble_v7v11" rows to notebooks/step3_results.parquet
  Appends metric rows to notebooks/step3_metrics.csv
  Prints cross-domain ranking table
"""

import sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
import seisbench
seisbench.cache_root = os.environ["SEISBENCH_CACHE_ROOT"]

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import h5py
import torch
import seisbench.models as sbm
from metrics import compute_metrics as _compute_metrics
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────
NB_DIR       = REPO_ROOT / "notebooks"
CKPT_V7      = REPO_ROOT / "checkpoints" / "finetune_jma_wc_global_v7" / "best.pt"
CKPT_V11     = REPO_ROOT / "checkpoints" / "finetune_jma_wc_global_v11" / "best.pt"
HDF5_PATH    = NB_DIR / "benchmark_waveforms.hdf5"
INDEX_PATH   = NB_DIR / "benchmark_waveforms_index.csv"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
METRICS_PATH = NB_DIR / "step3_metrics.csv"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 128
TARGET_SR     = 100
SEARCH_WIN_S  = 5.0
THRESHOLD_P   = 0.3
THRESHOLD_S   = 0.3
ENS_WEIGHT    = "jma_wc_ft_ensemble_v7v11"

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def _load_ft_model(ckpt_path: Path) -> sbm.PhaseNet:
    ckpt     = torch.load(ckpt_path, map_location="cpu")
    raw_sd   = ckpt["model"]
    inner_sd = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
    model    = sbm.PhaseNet.from_pretrained("jma_wc", update=False)
    model.load_state_dict(inner_sd)
    model.eval()
    return model


def _normalize_std(batch: np.ndarray) -> np.ndarray:
    b   = batch - batch.mean(axis=-1, keepdims=True)
    std = b.std(axis=-1, keepdims=True)
    std[std < 1e-10] = 1.0
    return (b / std).astype(np.float32)


# ── Load both models ──────────────────────────────────────────────────────────
print(f"\nLoading v7  : {CKPT_V7}")
model_v7 = _load_ft_model(CKPT_V7)
model_v7.to(DEVICE)
print(f"Loading v11 : {CKPT_V11}")
model_v11 = _load_ft_model(CKPT_V11)
model_v11.to(DEVICE)

# ── Benchmark traces ──────────────────────────────────────────────────────────
print("\nLoading benchmark index …")
ok = pd.read_csv(INDEX_PATH)
ok = ok[ok["status"] == "ok"].copy().reset_index(drop=True)
print(f"  {len(ok):,} benchmark traces")

SEARCH = int(SEARCH_WIN_S * TARGET_SR)
n_in   = int(getattr(model_v7, "in_samples", 3001))

results   = {}
rows_all  = [(i, row) for i, row in ok.iterrows()]

# ── Inference: average full probability curves ────────────────────────────────
print(f"\nRunning ensemble inference (batch={BATCH_SIZE}, device={DEVICE}) …")
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
        with torch.no_grad():
            out_v7  = model_v7(batch_t).cpu().numpy()
            out_v11 = model_v11(batch_t).cpu().numpy()

        # PSN ordering: channel 0=P, 1=S
        p_full = 0.5 * (out_v7[:, 0, :wave_len] + out_v11[:, 0, :wave_len])
        s_full = 0.5 * (out_v7[:, 1, :wave_len] + out_v11[:, 1, :wave_len])

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

model_v7.cpu(); model_v11.cpu()
torch.cuda.empty_cache()
print(f"  Inference done — {len(results):,} traces")


# ── Build result rows ─────────────────────────────────────────────────────────
print("\nBuilding result rows …")
ens_rows = []
for i, row in ok.iterrows():
    pred = results.get(i)
    if pred is None:
        continue
    ens_rows.append({
        "weight":        ENS_WEIGHT,
        "tier":          "B",
        "trace_name":    row["trace_name"],
        "dataset":       row.get("dataset",   ""),
        "dist_bin":      row.get("dist_bin",  np.nan),
        "depth_bin":     row.get("depth_bin", np.nan),
        "mag_bin":       row.get("mag_bin",   np.nan),
        "trained_models":str(row.get("trained_models", "")),
        "snr_db":        row.get("snr_db",    np.nan),
        "p_in_window":   int(row["p_in_window"]),
        "s_in_window":   int(row["s_in_window"]),
        "p_prob":        pred["p_prob"],
        "s_prob":        pred["s_prob"],
        "p_residual_s":  pred["p_res_s"],
        "s_residual_s":  pred["s_res_s"],
    })

ens_df = pd.DataFrame(ens_rows)
existing = pd.read_parquet(RESULTS_PATH)
existing = existing[existing["weight"] != ENS_WEIGHT]
combined = pd.concat([existing, ens_df], ignore_index=True)
combined.to_parquet(RESULTS_PATH, index=False)
print(f"  Saved {len(combined):,} rows → {RESULTS_PATH}  (+{len(ens_df):,} ensemble rows)")


# ── Metric computation (mirrors eval_finetuned.py) ───────────────────────────
results_df = combined

KNOWN_WEIGHTS = {
    "jma_wc":                   {"tier": "B"},
    "jma_wc_ft_global_v3":      {"tier": "B"},
    "jma_wc_ft_global_v7":      {"tier": "B"},
    "jma_wc_ft_global_v11":     {"tier": "B"},
    ENS_WEIGHT:                 {"tier": "B"},
}

DEGENERATE_MODELS = set()
for wname in results_df["weight"].unique():
    wdf = results_df[results_df["weight"] == wname]
    rec = (wdf["p_prob"] >= THRESHOLD_P).mean()
    mae = wdf["p_residual_s"].abs().mean()
    if rec > 0.99 and mae > 2.0:
        DEGENERATE_MODELS.add(wname)


def compute_metrics(df, weight_name, split_name, dist_label=None):
    return _compute_metrics(
        df, weight_name, split_name, dist_label or "all",
        p_threshold=THRESHOLD_P, s_threshold=THRESHOLD_S,
        tier=KNOWN_WEIGHTS.get(weight_name, {}).get("tier", "B"),
        degenerate=weight_name in DEGENERATE_MODELS,
    )


dist_bins = ["all", "local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)"]
report_weights = [ENS_WEIGHT, "jma_wc_ft_global_v7", "jma_wc_ft_global_v11",
                  "jma_wc_ft_global_v3", "jma_wc"]

metrics_rows = []
for weight_name in report_weights:
    info = KNOWN_WEIGHTS.get(weight_name, {"tier": "?"})
    trained_on = info.get("trained_on", None)
    wdf = results_df[results_df["weight"] == weight_name]
    if len(wdf) == 0:
        print(f"  WARNING: no rows for {weight_name}")
        continue

    if trained_on:
        in_mask    = wdf["trained_models"].str.contains(trained_on, na=False, regex=False)
        cross_mask = ~in_mask
    else:
        in_mask    = pd.Series(False, index=wdf.index)
        cross_mask = pd.Series(True,  index=wdf.index)

    for dist_label in dist_bins:
        if dist_label == "all":
            sub_cross = wdf[cross_mask]
        else:
            d_mask    = wdf["dist_bin"] == dist_label
            sub_cross = wdf[cross_mask & d_mask]
        row = compute_metrics(sub_cross, weight_name, "cross_domain", dist_label)
        if row:
            metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)

# Read existing metrics, drop old ensemble rows, append new
if METRICS_PATH.exists():
    existing_m = pd.read_csv(METRICS_PATH)
    existing_m = existing_m[existing_m["weight"] != ENS_WEIGHT]
    metrics_df = pd.concat([existing_m, metrics_df], ignore_index=True)

metrics_df.to_csv(METRICS_PATH, index=False)
print(f"Saved {len(metrics_df):,} metric rows → {METRICS_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
clean = metrics_df[~metrics_df["degenerate"].fillna(False)]
cross_all = (clean[(clean["split"] == "cross_domain") & (clean["dist_bin"] == "all")]
             [clean["weight"].isin(report_weights)]
             .sort_values("p_mae_s"))

print("\n" + "="*72)
print(f"SUMMARY — {ENS_WEIGHT} vs components (cross-domain, all distances)")
print("="*72)
cols = ["weight", "p_mae_s", "s_mae_s", "p_recall", "s_recall", "mcc", "p_outlier",
        "p_recall_t02", "p_med_prob"]
print(cross_all[cols].to_string(index=False))
