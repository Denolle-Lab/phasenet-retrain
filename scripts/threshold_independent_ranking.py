#!/usr/bin/env python3
"""
threshold_independent_ranking.py

A fair, threshold-independent re-ranking of every PhaseNet weight (our
fine-tunes + all pretrained baselines), on the leakage-corrected populations
already established (clean_holdout for own fine-tunes, cross_domain_clean for
public pretrained weights where locally verifiable). EQTransformer weights
are intentionally excluded (out of deployment scope per project decision).

Key fact this script relies on (see scripts/metrics.py's own docstring):
p_mae_s / s_mae_s / p_rmse_s / s_rmse_s / p_outlier / s_outlier / mcc are
ALREADY threshold-independent as defined in this codebase -- they are
computed over ALL in-window traces (or, for MCC, purely from the relative
ranking of p_prob vs s_prob), never gated on p_prob/s_prob >= threshold. The
ONLY threshold-dependent quantity in the existing metric suite is recall
(p_recall/s_recall = fraction with prob >= threshold).

For recall, this script reports the FULL curve (not a single point) plus a
mathematically exact area-under-the-recall-curve summary:

    AUC_recall = integral_0^1 recall(t) dt = mean(prob at the picked sample)

(This equality holds because recall(t) = P(prob >= t) for prob in [0,1], and
integrating that indicator over t in [0,1] and swapping the order of
integration/expectation gives E[prob].) This is an exact, not sampled,
threshold-independent scalar -- no arbitrary threshold or curve-discretization
choice is involved.

No re-inference: reads notebooks/step3_results.parquet only.

Run from repo root:
    conda activate surface
    python scripts/threshold_independent_ranking.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from domain_registry import (
    is_own_model, clean_holdout_mask, parent_clean_cross_domain_mask,
)
from metrics import bootstrap_ci, OUTLIER_THR_MUNCHMEYER_S, OUTLIER_THR_LEGACY_S

NB_DIR = REPO_ROOT / "notebooks"
RESULTS_PATH = NB_DIR / "step3_results.parquet"
OUT_CSV = REPO_ROOT / "results" / "threshold_independent_ranking.csv"
OUT_CURVES_CSV = REPO_ROOT / "results" / "threshold_independent_recall_curves.csv"

EXCLUDE_PREFIX = ("eqt_",)  # EQTransformer out of scope for this ranking
CURVE_THRESHOLDS = np.round(np.arange(0.0, 1.001, 0.01), 2)  # fine sweep for the plotted curve


def degenerate_weights(results_df):
    bad = set()
    for wname in results_df["weight"].unique():
        wdf = results_df[results_df["weight"] == wname]
        rec = (wdf["p_prob"] >= 0.3).mean()
        mae = wdf["p_residual_s"].abs().mean()
        if rec > 0.99 and mae > 2.0:
            bad.add(wname)
    return bad


def auc_recall(probs):
    """Exact area under the recall-vs-threshold curve = mean(prob)."""
    probs = probs.dropna()
    return float(probs.mean()) if len(probs) else np.nan


def recall_curve(probs, thresholds):
    probs = probs.dropna().values
    if len(probs) == 0:
        return np.full(len(thresholds), np.nan)
    return np.array([(probs >= t).mean() for t in thresholds])


def phase_metrics(sub, phase):
    """Threshold-independent metrics for one phase ('p' or 's') on one
    (weight, population) slice. Mirrors metrics.py's unconditional
    definitions exactly, plus the AUC-recall addition."""
    in_win = sub[sub[f"{phase}_in_window"] >= 0]
    n = len(in_win)
    if n == 0:
        return None
    prob = in_win[f"{phase}_prob"]
    res = in_win[f"{phase}_residual_s"].dropna()
    mae = np.abs(res).mean() if len(res) else np.nan
    rmse = np.sqrt((res ** 2).mean()) if len(res) else np.nan
    outlier_m = (np.abs(res) > OUTLIER_THR_MUNCHMEYER_S).mean() if len(res) else np.nan
    outlier_legacy = (np.abs(res) > OUTLIER_THR_LEGACY_S).mean() if len(res) else np.nan
    mae_lo, mae_hi = bootstrap_ci(np.abs(res).values) if len(res) else (np.nan, np.nan)
    auc = auc_recall(prob)
    auc_lo, auc_hi = bootstrap_ci(prob.dropna().values) if len(prob.dropna()) else (np.nan, np.nan)
    return {
        f"n_{phase}": n,
        f"{phase}_mae_s": mae, f"{phase}_mae_s_ci_lo": mae_lo, f"{phase}_mae_s_ci_hi": mae_hi,
        f"{phase}_rmse_s": rmse,
        f"{phase}_outlier_1s": outlier_m, f"{phase}_outlier_1_5s": outlier_legacy,
        f"{phase}_auc_recall": auc, f"{phase}_auc_recall_ci_lo": auc_lo, f"{phase}_auc_recall_ci_hi": auc_hi,
    }


def mcc_p_vs_s(sub):
    from sklearn.metrics import matthews_corrcoef
    both = sub[(sub["p_in_window"] >= 0) & (sub["s_in_window"] >= 0)]
    if len(both) < 5:
        return np.nan
    y_true = np.concatenate([np.ones(len(both)), np.zeros(len(both))])
    y_pred = np.concatenate([
        (both["p_prob"] > both["s_prob"]).astype(int).values,
        (both["s_prob"] > both["p_prob"]).astype(int).values,
    ])
    try:
        return matthews_corrcoef(y_true, y_pred)
    except Exception:
        return np.nan


def main():
    print(f"Loading {RESULTS_PATH} ...")
    results_df = pd.read_parquet(RESULTS_PATH)
    bad = degenerate_weights(results_df)
    print(f"Excluding degenerate weights: {sorted(bad)}")

    weights = [w for w in results_df["weight"].unique()
               if w not in bad and not any(w.startswith(p) for p in EXCLUDE_PREFIX)]
    print(f"PhaseNet-family weights included ({len(weights)}): {sorted(weights)}")

    rows = []
    curve_rows = []
    for wname in weights:
        wdf = results_df[results_df["weight"] == wname]
        own = is_own_model(wname)
        if own:
            mask = clean_holdout_mask(wdf, wname)
            population = "clean_holdout"
        else:
            mask = parent_clean_cross_domain_mask(wdf, wname)
            population = "cross_domain_clean" if mask is not None else "all (unverifiable corpus)"
            if mask is None:
                mask = pd.Series(True, index=wdf.index)
        sub = wdf[mask]
        if len(sub) == 0:
            print(f"  SKIP {wname}: 0 rows in leakage-corrected population")
            continue

        row = {"weight": wname, "own_model": own, "population": population, "n_traces": len(sub)}
        pm = phase_metrics(sub, "p")
        sm = phase_metrics(sub, "s")
        if pm is None:
            print(f"  SKIP {wname}: no P-phase rows")
            continue
        row.update(pm)
        if sm is not None:
            row.update(sm)
        row["mcc_p_vs_s"] = mcc_p_vs_s(sub)
        rows.append(row)

        p_in_win = sub[sub["p_in_window"] >= 0]
        s_in_win = sub[sub["s_in_window"] >= 0]
        p_curve = recall_curve(p_in_win["p_prob"], CURVE_THRESHOLDS)
        s_curve = recall_curve(s_in_win["s_prob"], CURVE_THRESHOLDS)
        for t, pr, sr in zip(CURVE_THRESHOLDS, p_curve, s_curve):
            curve_rows.append({"weight": wname, "threshold": t, "p_recall": pr, "s_recall": sr})

    ranking_df = pd.DataFrame(rows).sort_values("p_mae_s")
    ranking_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(ranking_df)} rows -> {OUT_CSV}")

    curves_df = pd.DataFrame(curve_rows)
    curves_df.to_csv(OUT_CURVES_CSV, index=False)
    print(f"Saved recall curves -> {OUT_CURVES_CSV}")

    print("\n" + "=" * 100)
    print("THRESHOLD-INDEPENDENT RANKING -- by P-MAE (unconditional, exact, no threshold involved)")
    print("=" * 100)
    cols = ["weight", "population", "n_traces", "p_mae_s", "p_auc_recall", "s_mae_s", "s_auc_recall",
            "mcc_p_vs_s", "p_outlier_1s"]
    print(ranking_df[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("THRESHOLD-INDEPENDENT RANKING -- by P-AUC-recall (detection confidence, exact, no threshold involved)")
    print("=" * 100)
    print(ranking_df.sort_values("p_auc_recall", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
