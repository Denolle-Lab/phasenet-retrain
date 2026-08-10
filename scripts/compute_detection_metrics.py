#!/usr/bin/env python3
"""
compute_detection_metrics.py

Closes the remaining gap in GitHub #8: a real detection precision/recall/MCC,
computed from an actual confusion matrix (TP/FP/TN/FN), not just the P-vs-S
discriminability proxy MCC in scripts/metrics.py (see MCC_CAVEAT there). This
was blocked before because the benchmark had no negative/noise windows;
scripts/audit_noise_fp_leaderboard.py built that missing negative population
(results/noise_fp_audit.csv, ~94k pure-noise traces, no true arrival).

A single shared threshold (originally fixed at 0.3 for every model) is not a
fair comparison -- different models' probability outputs are calibrated
differently, so 0.3 is each model's own natural operating point only by
coincidence. Instead this sweeps a threshold grid per model and reports each
model at ITS OWN best-detection-MCC threshold (results/detection_metrics.csv),
plus the full per-threshold curve (results/detection_metrics_sweep.csv) so the
choice of best point is inspectable, not just asserted.

Positive population: benchmark traces with a real P arrival, leak-corrected
(clean_holdout for own fine-tunes, cross_domain_clean for parent weights) --
same population + same p_prob column already cached in
notebooks/step3_results.parquet, no re-inference.

Negative population: results/noise_fp_audit.csv's max_p_prob per weight.

Caveat (stated once here, must be repeated wherever this is cited): the
positive:negative ratio here (~21-32k : ~94k) reflects benchmark/noise-pool
construction, NOT the true earthquake:noise ratio of real continuous data
(which is far more noise-dominated) -- so precision/MCC below are a real,
computable detection score on this pooled test set, not a deployment-accurate
false-alarm rate.

Run from repo root:
    conda activate surface
    python scripts/compute_detection_metrics.py

Output: results/detection_metrics.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from domain_registry import is_own_model, clean_holdout_mask, parent_clean_cross_domain_mask
from metrics import bootstrap_ci

NB_DIR       = REPO_ROOT / "notebooks"
RESULTS_PQ   = NB_DIR / "step3_results.parquet"
NOISE_AUDIT  = REPO_ROOT / "results" / "noise_fp_audit.csv"
OUT_CSV      = REPO_ROOT / "results" / "detection_metrics.csv"
OUT_SWEEP    = REPO_ROOT / "results" / "detection_metrics_sweep.csv"

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

HEADLINE_WEIGHTS = [
    "jma_wc_ft_global_v7", "jma_wc_ft_global_v11", "jma_wc_ft_global_v3",
    "jma_wc_ft_global_v18", "jma_wc_ft_global_v20", "jma_wc_ft_global_v7_eventclean",
    "jma_wc_ft_ensemble_v3v7", "jma_wc_ft_ensemble_v7v11",
    "jma_wc", "volpick",
    "eqt_original_nonconservative", "eqt_volpick", "eqt_scedc", "eqt_instance",
    "eqt_ensemble_volpick_nc",
]


def detection_mcc(tp, fp, tn, fn):
    tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else np.nan


def confusion_at(pos_prob, neg_prob, threshold):
    pos_hit = pos_prob >= threshold
    neg_hit = neg_prob >= threshold
    tp = int(pos_hit.sum())
    fn = int((~pos_hit).sum())
    fp = int(neg_hit.sum())
    tn = int((~neg_hit).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan
    mcc = detection_mcc(tp, fp, tn, fn)
    return dict(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1, detection_mcc=mcc)


def bootstrap_ci_at(pos_hit, neg_hit, n_boot=1000, seed=42):
    """Bootstrap CI on precision/detection-MCC at a fixed threshold, resampling
    the pooled (positive + negative) hit/label arrays together."""
    labels = np.concatenate([np.ones(len(pos_hit)), np.zeros(len(neg_hit))])
    hits = np.concatenate([pos_hit.astype(float), neg_hit.astype(float)])
    n = len(labels)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_labels, boot_hits = labels[idx], hits[idx]
    boot_tp = ((boot_labels == 1) & (boot_hits == 1)).sum(axis=1)
    boot_fp = ((boot_labels == 0) & (boot_hits == 1)).sum(axis=1)
    boot_tn = ((boot_labels == 0) & (boot_hits == 0)).sum(axis=1)
    boot_fn = ((boot_labels == 1) & (boot_hits == 0)).sum(axis=1)
    boot_prec = np.divide(boot_tp, boot_tp + boot_fp,
                           out=np.full(n_boot, np.nan), where=(boot_tp + boot_fp) > 0)
    boot_mcc = np.array([detection_mcc(*t) for t in
                         zip(boot_tp, boot_fp, boot_tn, boot_fn)])
    prec_ci = tuple(np.nanpercentile(boot_prec, [2.5, 97.5]))
    mcc_ci = tuple(np.nanpercentile(boot_mcc, [2.5, 97.5]))
    return prec_ci, mcc_ci


def main():
    print(f"Loading {RESULTS_PQ} …")
    results_df = pd.read_parquet(RESULTS_PQ)
    print(f"Loading {NOISE_AUDIT} …")
    noise_df = pd.read_csv(NOISE_AUDIT)

    sweep_rows = []
    best_rows = []
    for weight in HEADLINE_WEIGHTS:
        wdf = results_df[results_df["weight"] == weight]
        if wdf.empty:
            print(f"  {weight}: SKIP — no positive-side rows in parquet")
            continue

        if is_own_model(weight):
            pos_mask = clean_holdout_mask(wdf, weight)
            split_used = "clean_holdout"
        else:
            pos_mask = parent_clean_cross_domain_mask(wdf, weight)
            split_used = "cross_domain_clean"
        if pos_mask is None:
            print(f"  {weight}: SKIP — no verifiable leak-corrected positive population")
            continue

        pos = wdf[pos_mask]
        pos = pos[pos["p_in_window"] >= 0]   # only rows with a real P arrival

        neg = noise_df[noise_df["weight"] == weight]
        if len(pos) == 0 or len(neg) == 0:
            print(f"  {weight}: SKIP — empty pos ({len(pos)}) or neg ({len(neg)}) population")
            continue

        pos_prob = pos["p_prob"].to_numpy()
        neg_prob = neg["max_p_prob"].to_numpy()

        # Sweep the full threshold grid — every model gets the SAME grid, but
        # each is then judged at ITS OWN best point on that grid, not a
        # shared fixed threshold (see module docstring for why).
        sweep = [confusion_at(pos_prob, neg_prob, t) for t in THRESHOLD_GRID]
        for s in sweep:
            sweep_rows.append(dict(weight=weight, pos_split=split_used, **s))

        best = max(sweep, key=lambda s: (s["detection_mcc"] if not np.isnan(s["detection_mcc"]) else -np.inf))
        best_thr = best["threshold"]
        pos_hit = pos_prob >= best_thr
        neg_hit = neg_prob >= best_thr
        prec_ci, mcc_ci = bootstrap_ci_at(pos_hit, neg_hit)

        best_rows.append(dict(
            weight=weight, pos_split=split_used,
            n_pos=len(pos), n_neg=len(neg),
            best_threshold=best_thr,
            tp=best["tp"], fp=best["fp"], tn=best["tn"], fn=best["fn"],
            precision=round(best["precision"], 4), recall=round(best["recall"], 4),
            f1=round(best["f1"], 4) if not np.isnan(best["f1"]) else np.nan,
            detection_mcc=round(best["detection_mcc"], 4),
            precision_ci_lo=round(prec_ci[0], 4), precision_ci_hi=round(prec_ci[1], 4),
            detection_mcc_ci_lo=round(mcc_ci[0], 4), detection_mcc_ci_hi=round(mcc_ci[1], 4),
        ))
        print(f"  {weight}: n_pos={len(pos)} n_neg={len(neg)} best_thr={best_thr:.2f} "
              f"precision={best['precision']:.4f} recall={best['recall']:.4f} "
              f"detection_mcc={best['detection_mcc']:.4f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_SWEEP, index=False)
    print(f"\nSaved full threshold sweep → {OUT_SWEEP}  ({len(sweep_df):,} rows)")

    out_df = pd.DataFrame(best_rows).sort_values("detection_mcc", ascending=False)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved best-threshold summary → {OUT_CSV}")

    print("\n" + "=" * 100)
    print("Each model at its OWN best-detection-MCC threshold (grid: "
          f"{THRESHOLD_GRID[0]:.2f}-{THRESHOLD_GRID[-1]:.2f} step 0.05)")
    print(f"{'weight':<32} {'best_thr':>8} {'precision':>10} {'recall':>8} {'F1':>7} "
          f"{'detMCC':>8} {'[95% CI]':>18}")
    print("-" * 100)
    for _, r in out_df.iterrows():
        ci = f"[{r.detection_mcc_ci_lo:.3f},{r.detection_mcc_ci_hi:.3f}]"
        print(f"  {r.weight:<30} {r.best_threshold:>8.2f} {r.precision:>10.4f} {r.recall:>8.4f} "
              f"{r.f1:>7.4f} {r.detection_mcc:>8.4f} {ci:>18}")
    print("=" * 100)


if __name__ == "__main__":
    main()
