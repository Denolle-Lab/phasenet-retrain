#!/usr/bin/env python3
"""
audit_noise_fp_leaderboard.py

Completes the missing precision/false-positive axis of the benchmark. Every
existing recall/MAE number in this project (notebooks/step3_metrics.csv,
results/eval_eqtransformer.csv, results/eval_ensemble_eqt.csv) is measured on
traces that ALWAYS contain a true arrival (oracle +-5s search window), so none
of them can distinguish a well-calibrated model from one that fires
constantly. This script runs the headline-leaderboard models
(paper_results_final.md SS4a/4b) over the existing pure-noise pool
(data/noise_global + data/noise_prephase, ~94k traces with NO true arrival)
and records the max P/S probability each model assigns anywhere in each
noise window -- the false-positive analogue of recall.

Run from repo root:
    conda activate surface
    python scripts/audit_noise_fp_leaderboard.py

Outputs
-------
  results/noise_fp_audit.csv        -- per-trace, per-model max_p_prob/max_s_prob
  results/noise_fp_leaderboard.csv  -- per-model FP-rate summary at several
                                        thresholds + threshold-free mean-prob
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
import seisbench
seisbench.cache_root = os.environ["SEISBENCH_CACHE_ROOT"]

import argparse

import h5py
import numpy as np
import pandas as pd
import torch
import seisbench.models as sbm
from tqdm import tqdm

_parser = argparse.ArgumentParser()
_parser.add_argument("--limit", type=int, default=None,
                      help="Subsample noise pool to N traces per source (smoke test)")
_args, _ = _parser.parse_known_args()

# ── paths ────────────────────────────────────────────────────────────────────
NOISE_SOURCES = [
    dict(name="noise_global",   hdf5=REPO_ROOT / "data/noise_global/waveforms.hdf5",
         meta=REPO_ROOT / "data/noise_global/metadata.csv"),
    dict(name="noise_prephase", hdf5=REPO_ROOT / "data/noise_prephase/waveforms.hdf5",
         meta=REPO_ROOT / "data/noise_prephase/metadata.csv"),
]
CKPT_DIR   = REPO_ROOT / "checkpoints"
OUT_AUDIT  = REPO_ROOT / "results" / "noise_fp_audit.csv"
OUT_LEADER = REPO_ROOT / "results" / "noise_fp_leaderboard.csv"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
THRESHOLDS  = [0.1, 0.2, 0.3, 0.5, 0.7]   # same grid used for recall_t01..t07 elsewhere

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load noise pool once (shared across all models)
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading noise pool …")
noise_meta = []
for cfg in NOISE_SOURCES:
    m = pd.read_csv(cfg["meta"])
    m["source"] = cfg["name"]
    m["hdf5_path"] = str(cfg["hdf5"])
    noise_meta.append(m[["trace_name", "source", "hdf5_path"]])
noise_meta = pd.concat(noise_meta, ignore_index=True)
if _args.limit:
    noise_meta = noise_meta.groupby("source", group_keys=False).apply(
        lambda g: g.head(_args.limit))
noise_meta["dataset"] = noise_meta["trace_name"].str.rsplit("_", n=1).str[0]
print(f"  {len(noise_meta):,} noise traces "
      f"({noise_meta['source'].value_counts().to_dict()})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model registry — headline leaderboard from paper_results_final.md SS4a/4b
# ══════════════════════════════════════════════════════════════════════════════

def load_finetune(ckpt_name):
    """Own PhaseNet fine-tune checkpoint (jma_wc architecture)."""
    ckpt = torch.load(CKPT_DIR / f"finetune_jma_wc_global_{ckpt_name}" / "best.pt",
                       map_location="cpu")
    inner_sd = {k[len("model."):]: v for k, v in ckpt["model"].items()
                if k.startswith("model.")}
    model = sbm.PhaseNet.from_pretrained("jma_wc", update=False)
    model.load_state_dict(inner_sd)
    return model


# Single-model entries: (label, kind, loader)
#   kind "phasenet" -> model(x) returns (B, 3, N) PSN order, P=ch0 S=ch1
#   kind "eqt"      -> model(x) returns (detector, P, S)
SINGLE_MODELS = [
    ("jma_wc_ft_global_v7",           "phasenet", lambda: load_finetune("v7")),
    ("jma_wc_ft_global_v11",          "phasenet", lambda: load_finetune("v11")),
    ("jma_wc_ft_global_v3",           "phasenet", lambda: load_finetune("v3")),
    ("jma_wc_ft_global_v18",          "phasenet", lambda: load_finetune("v18")),
    ("jma_wc_ft_global_v20",          "phasenet", lambda: load_finetune("v20")),
    ("jma_wc_ft_global_v7_eventclean","phasenet", lambda: load_finetune("v7_eventclean")),
    ("jma_wc",                        "phasenet", lambda: sbm.PhaseNet.from_pretrained("jma_wc", update=False)),
    ("volpick",                       "phasenet", lambda: sbm.PhaseNet.from_pretrained("volpick", update=False)),
    ("eqt_original_nonconservative",  "eqt",      lambda: sbm.EQTransformer.from_pretrained("original_nonconservative", update=False)),
    ("eqt_volpick",                   "eqt",      lambda: sbm.EQTransformer.from_pretrained("volpick", update=False)),
    ("eqt_scedc",                     "eqt",      lambda: sbm.EQTransformer.from_pretrained("scedc", update=False)),
    ("eqt_instance",                  "eqt",      lambda: sbm.EQTransformer.from_pretrained("instance", update=False)),
]

# Ensemble entries: (label, kind, [loaders]) — probability curves averaged
ENSEMBLE_MODELS = [
    ("jma_wc_ft_ensemble_v3v7",  "phasenet", [lambda: load_finetune("v3"), lambda: load_finetune("v7")]),
    ("jma_wc_ft_ensemble_v7v11", "phasenet", [lambda: load_finetune("v7"), lambda: load_finetune("v11")]),
    ("eqt_ensemble_volpick_nc",  "eqt",      [lambda: sbm.EQTransformer.from_pretrained("volpick", update=False),
                                               lambda: sbm.EQTransformer.from_pretrained("original_nonconservative", update=False)]),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize(batch: np.ndarray, norm: str) -> np.ndarray:
    b = batch - batch.mean(axis=-1, keepdims=True)
    if norm == "peak":
        scale = np.abs(b).max(axis=-1, keepdims=True)
    else:
        scale = b.std(axis=-1, keepdims=True)
    scale[scale < 1e-10] = 1.0
    return np.clip(b / scale, -10.0, 10.0).astype(np.float32)


def to_fixed_length(wf: np.ndarray, n_in: int) -> np.ndarray:
    """(3, N) -> (3, n_in): truncate or zero-pad."""
    n = wf.shape[-1]
    if n < n_in:
        wf = np.concatenate([wf, np.zeros((wf.shape[0], n_in - n), dtype=np.float32)], axis=-1)
    else:
        wf = wf[:, :n_in]
    return wf


def run_single(model, kind, label):
    """Run one model over the full noise pool; return DataFrame of max probs."""
    model.eval().to(DEVICE)
    norm = getattr(model, "norm", "std")
    n_in = int(getattr(model, "in_samples", 3001 if kind == "phasenet" else 6000))

    results = []
    by_hdf5 = noise_meta.groupby("hdf5_path")
    for hdf5_path, grp in by_hdf5:
        with h5py.File(hdf5_path, "r") as hf:
            data_grp = hf["data"]
            names = grp["trace_name"].tolist()
            for start in tqdm(range(0, len(names), BATCH_SIZE), desc=f"  {label}", leave=False):
                batch_names = names[start:start + BATCH_SIZE]
                waves, valid = [], []
                for tname in batch_names:
                    if tname not in data_grp:
                        continue
                    wf = np.asarray(data_grp[tname][()], dtype=np.float32)
                    waves.append(to_fixed_length(wf, n_in))
                    valid.append(tname)
                if not waves:
                    continue
                batch_np = normalize(np.stack(waves), norm)
                batch_t = torch.from_numpy(batch_np).to(DEVICE)
                with torch.no_grad():
                    out = model(batch_t)
                if kind == "eqt":
                    p_full = out[1].cpu().numpy()
                    s_full = out[2].cpu().numpy()
                else:
                    probs = out.cpu().numpy()
                    p_full, s_full = probs[:, 0, :], probs[:, 1, :]
                for i, tname in enumerate(valid):
                    results.append((tname, float(p_full[i].max()), float(s_full[i].max())))

    model.cpu()
    torch.cuda.empty_cache()
    return pd.DataFrame(results, columns=["trace_name", "max_p_prob", "max_s_prob"])


def run_ensemble(loaders, kind, label):
    models = [ld() for ld in loaders]
    for m in models:
        m.eval().to(DEVICE)
    norms = [getattr(m, "norm", "std") for m in models]
    n_in = int(getattr(models[0], "in_samples", 3001 if kind == "phasenet" else 6000))

    results = []
    by_hdf5 = noise_meta.groupby("hdf5_path")
    for hdf5_path, grp in by_hdf5:
        with h5py.File(hdf5_path, "r") as hf:
            data_grp = hf["data"]
            names = grp["trace_name"].tolist()
            for start in tqdm(range(0, len(names), BATCH_SIZE), desc=f"  {label}", leave=False):
                batch_names = names[start:start + BATCH_SIZE]
                waves, valid = [], []
                for tname in batch_names:
                    if tname not in data_grp:
                        continue
                    wf = np.asarray(data_grp[tname][()], dtype=np.float32)
                    waves.append(to_fixed_length(wf, n_in))
                    valid.append(tname)
                if not waves:
                    continue
                raw = np.stack(waves)
                p_sum = s_sum = None
                for m, norm in zip(models, norms):
                    batch_t = torch.from_numpy(normalize(raw, norm)).to(DEVICE)
                    with torch.no_grad():
                        out = m(batch_t)
                    if kind == "eqt":
                        p, s = out[1].cpu().numpy(), out[2].cpu().numpy()
                    else:
                        probs = out.cpu().numpy()
                        p, s = probs[:, 0, :], probs[:, 1, :]
                    p_sum = p if p_sum is None else p_sum + p
                    s_sum = s if s_sum is None else s_sum + s
                p_full, s_full = p_sum / len(models), s_sum / len(models)
                for i, tname in enumerate(valid):
                    results.append((tname, float(p_full[i].max()), float(s_full[i].max())))

    for m in models:
        m.cpu()
    torch.cuda.empty_cache()
    return pd.DataFrame(results, columns=["trace_name", "max_p_prob", "max_s_prob"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Run every model, build long audit table
# ══════════════════════════════════════════════════════════════════════════════

all_rows = []
n_total = len(SINGLE_MODELS) + len(ENSEMBLE_MODELS)
done = 0

for label, kind, loader in SINGLE_MODELS:
    done += 1
    print(f"\n[{done}/{n_total}] {label}")
    try:
        model = loader()
        df = run_single(model, kind, label)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue
    df["weight"] = label
    all_rows.append(df)
    print(f"  {len(df):,} traces scored")

for label, kind, loaders in ENSEMBLE_MODELS:
    done += 1
    print(f"\n[{done}/{n_total}] {label}")
    try:
        df = run_ensemble(loaders, kind, label)
    except Exception as e:
        print(f"  FAILED: {e}")
        continue
    df["weight"] = label
    all_rows.append(df)
    print(f"  {len(df):,} traces scored")

audit_df = pd.concat(all_rows, ignore_index=True)
audit_df = audit_df.merge(noise_meta[["trace_name", "source", "dataset"]],
                           on="trace_name", how="left")
audit_df.to_csv(OUT_AUDIT, index=False)
print(f"\nSaved per-trace audit → {OUT_AUDIT}  ({len(audit_df):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Leaderboard — FP rate at each threshold + threshold-free mean-prob
# ══════════════════════════════════════════════════════════════════════════════

leaderboard = []
for label, sub in audit_df.groupby("weight"):
    n = len(sub)
    row = {"weight": label, "n_noise_traces": n}
    for t in THRESHOLDS:
        key = f"{int(round(t * 10)):02d}"
        row[f"p_fp_rate_t{key}"] = float((sub["max_p_prob"] >= t).mean())
        row[f"s_fp_rate_t{key}"] = float((sub["max_s_prob"] >= t).mean())
    # AUC-style, threshold-free summary — mean max-prob over noise windows.
    # Symmetric to AUC_recall = mean prob at the true-arrival location (used
    # for recall in paper_results_final.md SS4e); here there is no true
    # arrival, so a lower value is better (less spurious confidence).
    row["p_mean_max_prob"] = float(sub["max_p_prob"].mean())
    row["s_mean_max_prob"] = float(sub["max_s_prob"].mean())
    leaderboard.append(row)

leaderboard_df = pd.DataFrame(leaderboard).sort_values("p_fp_rate_t03")
leaderboard_df.to_csv(OUT_LEADER, index=False)
print(f"Saved leaderboard → {OUT_LEADER}")

print("\n" + "=" * 100)
print("False-positive rate on pure-noise pool (no true arrival present), threshold=0.30")
print(f"{'weight':<38} {'P-FP@.3':>8} {'S-FP@.3':>8} {'P-meanmax':>10} {'n':>7}")
print("-" * 100)
for _, r in leaderboard_df.iterrows():
    print(f"  {r['weight']:<36} {r['p_fp_rate_t03']:>8.4f} {r['s_fp_rate_t03']:>8.4f} "
          f"{r['p_mean_max_prob']:>10.4f} {int(r['n_noise_traces']):>7,}")
print("=" * 100)
print("\nDone.")
