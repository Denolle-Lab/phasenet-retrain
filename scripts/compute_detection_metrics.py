#!/usr/bin/env python3
"""
compute_detection_metrics.py

Closes the remaining gap in GitHub #8: a real detection precision/recall/MCC,
computed from an actual confusion matrix (TP/FP/TN/FN), not just the P-vs-S
discriminability proxy MCC in scripts/metrics.py (see MCC_CAVEAT there). This
was blocked before because the benchmark had no negative/noise windows;
scripts/audit_noise_fp_leaderboard.py built that missing negative population
(results/noise_fp_audit.csv, ~94k pure-noise traces, no true arrival).

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

NB_DIR      = REPO_ROOT / "notebooks"
RESULTS_PQ  = NB_DIR / "step3_results.parquet"
NOISE_AUDIT = REPO_ROOT / "results" / "noise_fp_audit.csv"
OUT_CSV     = REPO_ROOT / "results" / "detection_metrics.csv"

THRESHOLD = 0.3

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


def main():
    print(f"Loading {RESULTS_PQ} …")
    results_df = pd.read_parquet(RESULTS_PQ)
    print(f"Loading {NOISE_AUDIT} …")
    noise_df = pd.read_csv(NOISE_AUDIT)

    rows = []
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

        pos_hit = (pos["p_prob"] >= THRESHOLD).to_numpy()
        neg_hit = (neg["max_p_prob"] >= THRESHOLD).to_numpy()

        tp = int(pos_hit.sum())
        fn = int((~pos_hit).sum())
        fp = int(neg_hit.sum())
        tn = int((~neg_hit).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan
        mcc = detection_mcc(tp, fp, tn, fn)

        # Bootstrap CI on precision/detection-MCC via resampling the pooled
        # (positive + negative) hit/label arrays together.
        labels = np.concatenate([np.ones(len(pos_hit)), np.zeros(len(neg_hit))])
        hits = np.concatenate([pos_hit.astype(float), neg_hit.astype(float)])
        n = len(labels)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, n, size=(1000, n))
        boot_labels, boot_hits = labels[idx], hits[idx]
        boot_tp = ((boot_labels == 1) & (boot_hits == 1)).sum(axis=1)
        boot_fp = ((boot_labels == 0) & (boot_hits == 1)).sum(axis=1)
        boot_tn = ((boot_labels == 0) & (boot_hits == 0)).sum(axis=1)
        boot_fn = ((boot_labels == 1) & (boot_hits == 0)).sum(axis=1)
        boot_prec = np.divide(boot_tp, boot_tp + boot_fp,
                               out=np.full(1000, np.nan), where=(boot_tp + boot_fp) > 0)
        boot_mcc = np.array([detection_mcc(*t) for t in
                             zip(boot_tp, boot_fp, boot_tn, boot_fn)])
        prec_ci = tuple(np.nanpercentile(boot_prec, [2.5, 97.5]))
        mcc_ci = tuple(np.nanpercentile(boot_mcc, [2.5, 97.5]))

        rows.append(dict(
            weight=weight, pos_split=split_used,
            n_pos=len(pos), n_neg=len(neg),
            tp=tp, fp=fp, tn=tn, fn=fn,
            precision=round(precision, 4), recall=round(recall, 4),
            f1=round(f1, 4) if not np.isnan(f1) else np.nan,
            detection_mcc=round(mcc, 4),
            precision_ci_lo=round(prec_ci[0], 4), precision_ci_hi=round(prec_ci[1], 4),
            detection_mcc_ci_lo=round(mcc_ci[0], 4), detection_mcc_ci_hi=round(mcc_ci[1], 4),
        ))
        print(f"  {weight}: n_pos={len(pos)} n_neg={len(neg)} "
              f"precision={precision:.4f} recall={recall:.4f} detection_mcc={mcc:.4f}")

    out_df = pd.DataFrame(rows).sort_values("detection_mcc", ascending=False)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")

    print("\n" + "=" * 100)
    print(f"{'weight':<32} {'precision':>10} {'recall':>8} {'F1':>7} "
          f"{'detMCC':>8} {'[95% CI]':>18}")
    print("-" * 100)
    for _, r in out_df.iterrows():
        ci = f"[{r.detection_mcc_ci_lo:.3f},{r.detection_mcc_ci_hi:.3f}]"
        print(f"  {r.weight:<30} {r.precision:>10.4f} {r.recall:>8.4f} {r.f1:>7.4f} "
              f"{r.detection_mcc:>8.4f} {ci:>18}")
    print("=" * 100)


if __name__ == "__main__":
    main()
