#!/usr/bin/env python3
"""
eval_eqtransformer.py

Evaluate every available EQTransformer pretrained weight on the benchmark
dataset and compare against PhaseNet baselines (jma_wc, v7, instance).

EQTransformer specifics handled here:
  - in_samples = 6000  → uses benchmark_waveforms_6k.hdf5 (real 60s windows)
                          instead of zero-padding the 3k benchmark
  - output tuple = (detector, P_prob, S_prob)  → P at index 1, S at index 2
  - norm = "std" (same as jma_wc)

Weight labels stored in parquet as "eqt_{weight}" to avoid collision with
PhaseNet weights of the same name (e.g. "stead", "instance").

Run from repo root:
    conda activate surface
    python scripts/eval_eqtransformer.py

Outputs
-------
  notebooks/step3_results.parquet  — updated with eqt_* rows
  results/eval_eqtransformer.log   — console-level output captured here too
  results/eval_eqtransformer.csv   — metrics table (cross_domain, all distances)
  results/eval_eqtransformer.png   — comparison figure
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
from domain_registry import split_masks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
NB_DIR       = REPO_ROOT / "notebooks"
HDF5_PATH    = NB_DIR / "benchmark_waveforms_6k.hdf5"   # 60s windows, no zero-pad
INDEX_PATH   = NB_DIR / "benchmark_waveforms_6k_index.csv"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
OUT_CSV      = REPO_ROOT / "results" / "eval_eqtransformer.csv"
OUT_PNG      = REPO_ROOT / "results" / "eval_eqtransformer.png"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 64      # smaller than PhaseNet — EQT is heavier (6k input)
TARGET_SR    = 100
SEARCH_WIN_S = 5.0
OUTLIER_THR  = 1.5
THRESHOLD    = 0.3

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── EQTransformer weights to evaluate ─────────────────────────────────────────
# trained_on: dataset name as it appears in benchmark_waveforms_index.csv
# "trained_models" column — used to split cross_domain vs in_domain.
EQT_WEIGHTS = [
    ("original",               "stead"),
    ("original_nonconservative","stead"),
    ("stead",                  "stead"),
    ("instance",               "instance"),
    ("neic",                   "neic"),
    ("ethz",                   "ethz"),
    ("scedc",                  "scedc"),
    ("iquique",                "iquique"),
    ("pnw",                    None),
    ("geofon",                 None),
    ("lendb",                  None),
    ("obs",                    "obst2024"),
    ("volpick",                None),
]

# ── Normalisation ──────────────────────────────────────────────────────────────

def normalize(batch: np.ndarray, norm: str) -> np.ndarray:
    """Demean + normalize per component. norm='std' or 'peak'."""
    b = batch - batch.mean(axis=-1, keepdims=True)
    if norm == "peak":
        scale = np.abs(b).max(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
    else:  # std (default for jma_wc and EQT original)
        scale = b.std(axis=-1, keepdims=True)
        scale[scale < 1e-10] = 1.0
    return (b / scale).astype(np.float32)


# ── Load benchmark index ───────────────────────────────────────────────────────

print(f"\nLoading benchmark index …")
ok = pd.read_csv(INDEX_PATH)
ok = ok[ok["status"] == "ok"].copy().reset_index(drop=True)
print(f"  {len(ok):,} benchmark traces")

SEARCH   = int(SEARCH_WIN_S * TARGET_SR)
WAVE_LEN = 6000   # actual stored length in HDF5 (6k benchmark)


# ══════════════════════════════════════════════════════════════════════════════
# Inference loop — one pass per EQTransformer weight
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, weight_label: str) -> list[dict]:
    """Run model over entire benchmark; return list of per-trace result dicts."""
    model.eval()
    model.to(DEVICE)
    norm_type = getattr(model, "norm", "std")
    n_in      = int(getattr(model, "in_samples", 6000))

    rows_all = list(ok.iterrows())
    results  = {}

    with h5py.File(HDF5_PATH, "r") as hf:
        for start in tqdm(range(0, len(rows_all), BATCH_SIZE),
                          desc=f"  {weight_label}", leave=False):
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

            batch_np = normalize(np.stack(waves), norm_type)
            # 6k benchmark waveforms match EQT's native input length — no padding needed
            assert batch_np.shape[-1] == n_in, \
                f"Expected {n_in} samples, got {batch_np.shape[-1]}"

            batch_t = torch.tensor(batch_np, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                out = model(batch_t)

            # EQTransformer: out = (detector, P_prob, S_prob)
            p_full = out[1].cpu().numpy()[:, :WAVE_LEN]
            s_full = out[2].cpu().numpy()[:, :WAVE_LEN]

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

    model.cpu()
    torch.cuda.empty_cache()

    ft_rows = []
    for idx, row in ok.iterrows():
        pred = results.get(idx)
        if not pred:
            continue
        ft_rows.append({
            "weight":        weight_label,
            "tier":          "EQT",
            "trace_name":    row["trace_name"],
            "dataset":       row["dataset"],
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
    return ft_rows


# ── Main loop ──────────────────────────────────────────────────────────────────

existing   = pd.read_parquet(RESULTS_PATH)
all_new    = []
succeeded  = []
failed     = []

for weight, trained_on in EQT_WEIGHTS:
    label = f"eqt_{weight}"
    print(f"\n[{label}]  trained_on={trained_on}")
    try:
        model = sbm.EQTransformer.from_pretrained(weight, update=False)
        rows  = run_inference(model, label)
        all_new.extend(rows)
        succeeded.append((label, trained_on))
        print(f"  ✓  {len(rows):,} rows")
    except Exception as e:
        print(f"  ✗  FAILED: {e}")
        failed.append((label, str(e)))

print(f"\n{len(succeeded)}/{len(EQT_WEIGHTS)} weights succeeded")
if failed:
    print("Failed:", [f for f, _ in failed])

# ── Update parquet ─────────────────────────────────────────────────────────────

eqt_labels = [f"eqt_{w}" for w, _ in EQT_WEIGHTS]
combined   = pd.concat(
    [existing[~existing["weight"].isin(eqt_labels)],
     pd.DataFrame(all_new)],
    ignore_index=True,
)
combined.to_parquet(RESULTS_PATH, index=False)
print(f"\nSaved {len(combined):,} rows → {RESULTS_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame, weight_name: str,
                    split: str, dist_label: str = "all"):
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
        weight     = weight_name,
        split      = split,
        dist_bin   = dist_label,
        n_traces   = len(df),
        p_recall   = round(p_recall, 4)                            if not np.isnan(p_recall)  else np.nan,
        s_recall   = round(s_recall, 4)                            if not np.isnan(s_recall)  else np.nan,
        p_mae_s    = round(np.abs(p_res).mean(), 4)                if len(p_res) > 0 else np.nan,
        s_mae_s    = round(np.abs(s_res).mean(), 4)                if len(s_res) > 0 else np.nan,
        p_outlier  = round((np.abs(p_res) > OUTLIER_THR).mean(),4) if len(p_res) > 0 else np.nan,
        s_outlier  = round((np.abs(s_res) > OUTLIER_THR).mean(),4) if len(s_res) > 0 else np.nan,
        mcc        = round(mcc, 4)                                  if not np.isnan(mcc)       else np.nan,
    )


results_df  = pd.read_parquet(RESULTS_PATH)
DIST_BINS   = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]
metrics_rows = []

# Include key PhaseNet baselines for side-by-side comparison
COMPARE_WEIGHTS = (
    [f"eqt_{w}" for w, _ in EQT_WEIGHTS if f"eqt_{w}" in results_df["weight"].unique()]
    + ["jma_wc", "jma_wc_ft_global_v7", "instance", "stead", "neic"]
)

for weight_name in COMPARE_WEIGHTS:
    wdf = results_df[results_df["weight"] == weight_name]
    if len(wdf) == 0:
        continue
    _, cross_mask = split_masks(wdf, weight_name)

    for dist in DIST_BINS:
        sub = wdf if dist == "all" else wdf[wdf["dist_bin"] == dist]
        sub_x = sub[cross_mask.reindex(sub.index, fill_value=True)]
        for df_s, split in [(sub, "all"), (sub_x, "cross_domain")]:
            row = compute_metrics(df_s, weight_name, split, dist)
            if row:
                metrics_rows.append(row)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_CSV, index=False)
print(f"Saved metrics → {OUT_CSV}")

# ── Print cross-domain summary table ──────────────────────────────────────────

cross = metrics_df[(metrics_df["split"] == "cross_domain") &
                   (metrics_df["dist_bin"] == "all")].copy()
cross = cross.sort_values("p_mae_s")

print("\n" + "=" * 90)
print("EQTransformer vs PhaseNet baselines — cross_domain, all distances")
print(f"{'weight':<35} {'P-MAE':>7} {'S-MAE':>7} {'P-rec':>7} {'S-rec':>7} {'MCC':>6} {'P-out%':>7} {'n':>6}")
print("-" * 90)
for _, r in cross.iterrows():
    marker = " ★" if r.weight == "jma_wc_ft_global_v7" else \
             " (parent)" if r.weight == "jma_wc" else ""
    pm = f"{r.p_mae_s:.4f}" if not np.isnan(r.p_mae_s) else "   —  "
    sm = f"{r.s_mae_s:.4f}" if not np.isnan(r.s_mae_s) else "   —  "
    pr = f"{r.p_recall:.4f}" if not np.isnan(r.p_recall) else "   —  "
    sr = f"{r.s_recall:.4f}" if not np.isnan(r.s_recall) else "   —  "
    mc = f"{r.mcc:.4f}"      if not np.isnan(r.mcc)      else "   —  "
    po = f"{r.p_outlier*100:.2f}" if not np.isnan(r.p_outlier) else "  —  "
    print(f"  {r.weight + marker:<33} {pm:>7} {sm:>7} {pr:>7} {sr:>7} {mc:>6} {po:>7} {int(r.n_traces):>6}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure — 4-panel comparison: P-MAE, P-Recall, S-Recall, MCC
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating figure …")

eqt_rows = cross[cross["weight"].str.startswith("eqt_")].sort_values("p_mae_s")
ref_rows  = cross[~cross["weight"].str.startswith("eqt_")]

PANEL_METRICS = [
    ("p_mae_s",   "P-MAE (s)",         False),
    ("p_recall",  "P-Recall @ 0.30",   True),
    ("s_recall",  "S-Recall @ 0.30",   True),
    ("mcc",       "MCC (phase ID)",     True),
]

COLOR_EQT  = "#3498DB"
COLOR_V7   = "#E6821A"
COLOR_JMA  = "#2C7BB6"
COLOR_REF  = "#95A5A6"

fig, axes = plt.subplots(1, 4, figsize=(18, max(6, len(eqt_rows) * 0.35 + 2)))
fig.suptitle(
    "EQTransformer (all weights) vs PhaseNet baselines\n"
    "Cross-domain benchmark — all distances — thr=0.30",
    fontsize=11, fontweight="bold", y=1.01,
)

all_weights = eqt_rows["weight"].tolist() + ref_rows["weight"].tolist()

for ax, (col, xlabel, higher) in zip(axes, PANEL_METRICS):
    vals, colors, labels_ = [], [], []
    for w in all_weights:
        row = cross[cross["weight"] == w]
        if row.empty or np.isnan(row.iloc[0][col]):
            continue
        v = row.iloc[0][col]
        vals.append(v)
        labels_.append(w.replace("eqt_", "eqt/"))
        if w == "jma_wc_ft_global_v7":
            colors.append(COLOR_V7)
        elif w == "jma_wc":
            colors.append(COLOR_JMA)
        elif w.startswith("eqt_"):
            colors.append(COLOR_EQT)
        else:
            colors.append(COLOR_REF)

    y_pos = list(range(len(vals)))
    bars  = ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Bold border on best bar
    if vals:
        best_i = vals.index(min(vals) if not higher else max(vals))
        bars[best_i].set_edgecolor("black")
        bars[best_i].set_linewidth(1.5)

    # Reference line at v7
    v7_row = cross[cross["weight"] == "jma_wc_ft_global_v7"]
    if not v7_row.empty and not np.isnan(v7_row.iloc[0][col]):
        ax.axvline(v7_row.iloc[0][col], color=COLOR_V7,
                   lw=1.2, ls="--", alpha=0.7, label="v7")

from matplotlib.patches import Patch
legend_elems = [
    Patch(color=COLOR_EQT, label="EQTransformer"),
    Patch(color=COLOR_V7,  label="v7 (best fine-tune)"),
    Patch(color=COLOR_JMA, label="jma_wc (parent)"),
    Patch(color=COLOR_REF, label="other PhaseNet"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=4,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PNG}")
print("\nDone.")
