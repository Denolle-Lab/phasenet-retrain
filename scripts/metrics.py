"""
metrics.py

Single source of truth for pick-evaluation metrics (GitHub #8, #12) -- was
duplicated with slight drift across eval_finetuned.py, eval_eqtransformer.py,
eval_ensemble.py, eval_ensemble_v7v11.py, eval_ensemble_eqt.py,
recompute_step3_metrics.py, recompute_eqt_metrics.py,
recompute_ensemble_metrics.py, compare_v7_thresholds.py.

Every one of those computed `p_residual_s`/`s_residual_s` by taking the
argmax of the model's probability curve inside an oracle +-5s search window
around the true pick -- a peak is found *regardless of whether the model's
probability at that peak ever crossed a detection threshold*. That made the
headline MAE/outlier numbers threshold-independent by construction (GitHub #8
item 1: "recall gains at zero MAE cost" is circular under this definition).

This module keeps those original (unconditional) numbers for backward
compatibility with existing plots/slides/tables, and ADDS the conditional
(detected-only) versions alongside them -- see `compute_metrics()`'s
docstring for the exact column contract.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

# Munchmeyer et al. (2022) use a 1.0s outlier threshold. This project's
# original code used 1.5s while citing Munchmeyer for it -- both are now
# reported, correctly labeled, rather than silently picking one.
OUTLIER_THR_MUNCHMEYER_S = 1.0
OUTLIER_THR_LEGACY_S = 1.5

DEFAULT_RECALL_THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)

# This MCC is NOT Munchmeyer's detection MCC (true/false positive/negative
# against a threshold over event + noise windows). This benchmark has no
# noise/negative windows (GitHub #8 item 3), so there is no TN/FP available.
# What's computed instead: for traces with both a P and an S arrival, whether
# p_prob > s_prob correctly identifies "this location is P" vs "this
# location is S" -- a P-vs-S discriminability score, not a detection score.
MCC_CAVEAT = ("not Munchmeyer's detection MCC -- P-vs-S discriminability "
              "only; this benchmark has no noise/negative windows")


def bootstrap_ci(values, statistic_fn=np.mean, n_boot=1000, ci=0.95, seed=42):
    """
    Percentile bootstrap CI for a statistic over a 1D array of per-trace
    values (e.g. absolute residuals, or a 0/1 detection indicator).
    Returns (lo, hi); (nan, nan) if there are too few values (<5) to bootstrap.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = statistic_fn(values[idx], axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return (float(lo), float(hi))


def _mcc_p_vs_s(both):
    """P-vs-S discriminability MCC on dual-phase traces. See MCC_CAVEAT."""
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


def compute_metrics(df, weight_name, split_name, dist_label="all",
                     p_threshold=0.3, s_threshold=0.3,
                     tier=None, degenerate=False,
                     recall_thresholds=DEFAULT_RECALL_THRESHOLDS,
                     bootstrap=True, n_boot=1000, seed=42):
    """
    Canonical pick-evaluation metrics for one (weight, split, dist_bin) slice.

    Required df columns: p_in_window, s_in_window, p_prob, s_prob,
    p_residual_s, s_residual_s (p_in_window/s_in_window are sample indices,
    -1/negative meaning "no true pick of that phase in this window").

    Returned columns (superset of every caller's previous schema --
    pre-existing column names/values are UNCHANGED for backward
    compatibility; the `_cond` / `_1s` / `_ci_*` columns are new, #8's fix):

    Unconditional (existing, oracle-window, threshold-independent):
      p_recall, s_recall, p_recall_t{01,02,03,05,07}, s_recall_t{...},
      p_mae_s, p_rmse_s, s_mae_s, s_rmse_s, p_outlier, s_outlier,
      outlier_thr_s (=1.5, legacy), mcc, p_med_prob, s_med_prob

    Conditional (NEW -- detected-only, i.e. computed on traces where the
    model's probability at the picked sample >= threshold):
      p_mae_s_cond, p_rmse_s_cond, s_mae_s_cond, s_rmse_s_cond,
      p_outlier_cond, s_outlier_cond, n_p_detected, n_s_detected

    Munchmeyer-threshold (NEW -- 1.0s, both unconditional and conditional):
      p_outlier_1s, s_outlier_1s, p_outlier_1s_cond, s_outlier_1s_cond

    Bootstrap 95% CIs (NEW, only if bootstrap=True):
      p_mae_s_ci_lo/hi, s_mae_s_ci_lo/hi, p_recall_ci_lo/hi,
      s_recall_ci_lo/hi, mcc_ci_lo/hi

    mcc_caveat: constant string, see MCC_CAVEAT above.
    """
    if len(df) == 0:
        return None

    p_all = df[df["p_in_window"] >= 0]
    s_all = df[df["s_in_window"] >= 0]

    recall_cols = {}
    for t in recall_thresholds:
        key = f"t{int(round(t * 10)):02d}"
        recall_cols[f"p_recall_{key}"] = (p_all["p_prob"] >= t).mean() if len(p_all) else np.nan
        recall_cols[f"s_recall_{key}"] = (s_all["s_prob"] >= t).mean() if len(s_all) else np.nan

    p_recall = (p_all["p_prob"] >= p_threshold).mean() if len(p_all) else np.nan
    s_recall = (s_all["s_prob"] >= s_threshold).mean() if len(s_all) else np.nan
    p_med_prob = p_all["p_prob"].median() if len(p_all) else np.nan
    s_med_prob = s_all["s_prob"].median() if len(s_all) else np.nan

    both = df[(df["p_in_window"] >= 0) & (df["s_in_window"] >= 0)]
    mcc = _mcc_p_vs_s(both)

    # ── unconditional residuals (existing headline number) ─────────────────
    p_res_all = p_all["p_residual_s"].dropna()
    s_res_all = s_all["s_residual_s"].dropna()
    p_mae = np.abs(p_res_all).mean() if len(p_res_all) else np.nan
    p_rmse = np.sqrt((p_res_all ** 2).mean()) if len(p_res_all) else np.nan
    s_mae = np.abs(s_res_all).mean() if len(s_res_all) else np.nan
    s_rmse = np.sqrt((s_res_all ** 2).mean()) if len(s_res_all) else np.nan
    p_outlier = (np.abs(p_res_all) > OUTLIER_THR_LEGACY_S).mean() if len(p_res_all) else np.nan
    s_outlier = (np.abs(s_res_all) > OUTLIER_THR_LEGACY_S).mean() if len(s_res_all) else np.nan
    p_outlier_1s = (np.abs(p_res_all) > OUTLIER_THR_MUNCHMEYER_S).mean() if len(p_res_all) else np.nan
    s_outlier_1s = (np.abs(s_res_all) > OUTLIER_THR_MUNCHMEYER_S).mean() if len(s_res_all) else np.nan

    # ── conditional (detected-only) residuals -- issue #8 fix ──────────────
    p_detected = p_all[p_all["p_prob"] >= p_threshold]
    s_detected = s_all[s_all["s_prob"] >= s_threshold]
    p_res_cond = p_detected["p_residual_s"].dropna()
    s_res_cond = s_detected["s_residual_s"].dropna()
    p_mae_cond = np.abs(p_res_cond).mean() if len(p_res_cond) else np.nan
    p_rmse_cond = np.sqrt((p_res_cond ** 2).mean()) if len(p_res_cond) else np.nan
    s_mae_cond = np.abs(s_res_cond).mean() if len(s_res_cond) else np.nan
    s_rmse_cond = np.sqrt((s_res_cond ** 2).mean()) if len(s_res_cond) else np.nan
    p_outlier_cond = (np.abs(p_res_cond) > OUTLIER_THR_LEGACY_S).mean() if len(p_res_cond) else np.nan
    s_outlier_cond = (np.abs(s_res_cond) > OUTLIER_THR_LEGACY_S).mean() if len(s_res_cond) else np.nan
    p_outlier_1s_cond = (np.abs(p_res_cond) > OUTLIER_THR_MUNCHMEYER_S).mean() if len(p_res_cond) else np.nan
    s_outlier_1s_cond = (np.abs(s_res_cond) > OUTLIER_THR_MUNCHMEYER_S).mean() if len(s_res_cond) else np.nan

    def r(v):
        return round(v, 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan

    row = {
        "weight": weight_name,
        "split": split_name,
        "dist_bin": dist_label,
        "n_traces": len(df),
        "n_p_traces": len(p_all),
        "n_s_traces": len(s_all),
        "n_p_detected": len(p_detected),
        "n_s_detected": len(s_detected),
        "p_recall": r(p_recall),
        "s_recall": r(s_recall),
        "p_med_prob": r(p_med_prob),
        "s_med_prob": r(s_med_prob),
        "mcc": r(mcc),
        "mcc_caveat": MCC_CAVEAT,
        "p_mae_s": r(p_mae),
        "p_rmse_s": r(p_rmse),
        "s_mae_s": r(s_mae),
        "s_rmse_s": r(s_rmse),
        "p_outlier": r(p_outlier),
        "s_outlier": r(s_outlier),
        "outlier_thr_s": OUTLIER_THR_LEGACY_S,
        "p_outlier_1s": r(p_outlier_1s),
        "s_outlier_1s": r(s_outlier_1s),
        "p_mae_s_cond": r(p_mae_cond),
        "p_rmse_s_cond": r(p_rmse_cond),
        "s_mae_s_cond": r(s_mae_cond),
        "s_rmse_s_cond": r(s_rmse_cond),
        "p_outlier_cond": r(p_outlier_cond),
        "s_outlier_cond": r(s_outlier_cond),
        "p_outlier_1s_cond": r(p_outlier_1s_cond),
        "s_outlier_1s_cond": r(s_outlier_1s_cond),
        **{k: r(v) for k, v in recall_cols.items()},
    }
    if tier is not None:
        row["tier"] = tier
    if degenerate:
        row["degenerate"] = degenerate

    if bootstrap:
        p_mae_ci = bootstrap_ci(np.abs(p_res_all.values), np.mean, n_boot=n_boot, seed=seed)
        s_mae_ci = bootstrap_ci(np.abs(s_res_all.values), np.mean, n_boot=n_boot, seed=seed)
        p_recall_ci = bootstrap_ci(
            (p_all["p_prob"] >= p_threshold).astype(float).values, np.mean, n_boot=n_boot, seed=seed
        ) if len(p_all) else (np.nan, np.nan)
        s_recall_ci = bootstrap_ci(
            (s_all["s_prob"] >= s_threshold).astype(float).values, np.mean, n_boot=n_boot, seed=seed
        ) if len(s_all) else (np.nan, np.nan)
        row.update({
            "p_mae_s_ci_lo": r(p_mae_ci[0]), "p_mae_s_ci_hi": r(p_mae_ci[1]),
            "s_mae_s_ci_lo": r(s_mae_ci[0]), "s_mae_s_ci_hi": r(s_mae_ci[1]),
            "p_recall_ci_lo": r(p_recall_ci[0]), "p_recall_ci_hi": r(p_recall_ci[1]),
            "s_recall_ci_lo": r(s_recall_ci[0]), "s_recall_ci_hi": r(s_recall_ci[1]),
        })

    return row
