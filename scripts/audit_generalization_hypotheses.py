#!/usr/bin/env python3
"""
audit_generalization_hypotheses.py

Tasks 2 and 3 of docs/2026-09-07_generalization_audit_prompt.md, run on the
committed per-trace benchmark results (notebooks/step3_results.parquet,
notebooks/step3_metrics.csv) so that they can be reproduced without the lab
server. Every number carries a 95% percentile-bootstrap interval from
scripts/metrics.py::bootstrap_ci (1000 resamples, seed 42); paired
differences between v7 and its parent resample the same traces for both.

What runs here and what cannot
------------------------------
* clean_holdout aggregates (task 2a) are read from step3_metrics.csv, where
  they were computed on the server with the event-leakage row masks
  (results/event_leakage_row_mask__*.csv, git-ignored). The per-trace
  clean_holdout population cannot be rebuilt locally, so the paired
  per-trace analyses (task 2b, H1, H4) use the full benchmark, which for
  v7 IS its in_domain split (every benchmark dataset is in manifests_v2;
  step3_metrics.csv rows in_domain == all for v7).
* "Recall at matched pick budget" needs picks that are not at true
  arrivals. The benchmark stores one probability per trace, at the argmax
  inside +-5 s of the true pick (scripts/eval_finetuned.py:157-171), so a
  pick budget on it is the recall itself. The threshold-independent
  summary this repository uses instead is AUC-recall = mean probability at
  the pick (scripts/threshold_independent_ranking.py:20-28); it is reported
  here with the full recall curves, and the matched-budget numbers come
  from the QuakeScope notebooks (docs/audit_2026-09-07/external_results_2026-09-07.csv).
* The noise-pool detection MCC (task 2c, H00) needs results/noise_fp_audit.csv
  and results/detection_metrics.csv, which exist only on the server. This
  script reads results/detection_metrics.csv when present and otherwise
  the transcription of the 2026-08-10 table in paper_draft.qmd
  (docs/audit_2026-09-07/detection_metrics_2026-08-10_snapshot.csv).
* H2 needs data/manifests_v2/train.csv (server). When it is absent the
  script reports the cap-based bound parsed from DATASET_CONFIGS and the
  benchmark-side consequence only.

Usage (from repo root):
    python scripts/audit_generalization_hypotheses.py all
    python scripts/audit_generalization_hypotheses.py task2 h1 h2 h3 h4 h00

Outputs: docs/audit_2026-09-07/<table>.csv and .md, one per table.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from metrics import bootstrap_ci  # noqa: E402
from threshold_independent_ranking import auc_recall, recall_curve  # noqa: E402
import heldout_sequences as hs  # noqa: E402

NB = REPO_ROOT / "notebooks"
OUT = REPO_ROOT / "docs" / "audit_2026-09-07"
PARQUET = NB / "step3_results.parquet"
METRICS = NB / "step3_metrics.csv"
BENCH_MANIFEST = NB / "benchmark_manifest.csv"
DET_SERVER = REPO_ROOT / "results" / "detection_metrics.csv"
DET_SNAPSHOT = OUT / "detection_metrics_2026-08-10_snapshot.csv"
MANIFEST_V2 = REPO_ROOT / "data" / "manifests_v2" / "train.csv"

PARENT = "jma_wc"
V7 = "jma_wc_ft_global_v7"
THR = 0.3
N_BOOT = 1000
SEED = 42
DIST_ORDER = ["local (<150km)", "regional (150-1500km)", "teleseismic (>1500km)", "all"]
SNR_EDGES = [-np.inf, 0, 5, 10, 20, np.inf]
SNR_LABELS = ["<0 dB", "0-5 dB", "5-10 dB", "10-20 dB", ">20 dB"]
CURVE_THR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── helpers ──────────────────────────────────────────────────────────────────

def ci(values, stat=np.mean):
    return bootstrap_ci(values, stat, n_boot=N_BOOT, seed=SEED)


def paired_diff_ci(a, b, stat=np.mean, n_boot=N_BOOT, seed=SEED, chunk=200):
    """stat(a) - stat(b) with a paired percentile bootstrap (same resampled
    rows for both arrays). Returns (diff, lo, hi)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        boots.append(stat(a[idx], axis=1) - stat(b[idx], axis=1))
    boots = np.concatenate(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(stat(a) - stat(b)), float(lo), float(hi))


def fmt(v, lo=None, hi=None, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    s = f"{v:.{nd}f}"
    if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi)):
        s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
    return s


def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("n/a" if np.isnan(v) else (f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(name: str, df: pd.DataFrame, title: str, notes: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    md = f"### {title}\n\n{to_md(df)}\n"
    if notes:
        md += f"\n{notes}\n"
    (OUT / f"{name}.md").write_text(md)
    print(f"\n{md}")


def load_pairs(weights=(PARENT, V7)) -> pd.DataFrame:
    """One row per benchmark trace present for BOTH weights, columns suffixed
    _par and _v7. Duplicate (dataset, trace_name) rows (227 in the parent's
    slice; sharded datasets reuse trace_name) keep the first occurrence."""
    cols = ["weight", "trace_name", "dataset", "dist_bin", "snr_db",
            "p_in_window", "s_in_window", "p_prob", "s_prob", "p_residual_s", "s_residual_s"]
    df = pd.read_parquet(PARQUET, columns=cols, filters=[("weight", "in", list(weights))])
    a = df[df.weight == weights[0]].drop_duplicates(["dataset", "trace_name"])
    b = df[df.weight == weights[1]].drop_duplicates(["dataset", "trace_name"])
    keep = ["dataset", "trace_name", "dist_bin", "snr_db", "p_in_window", "s_in_window",
            "p_prob", "s_prob", "p_residual_s", "s_residual_s"]
    m = a[keep].merge(b[["dataset", "trace_name", "p_prob", "s_prob", "p_residual_s", "s_residual_s"]],
                      on=["dataset", "trace_name"], suffixes=("_par", "_v7"))
    m["snr_bin"] = pd.cut(m["snr_db"], SNR_EDGES, labels=SNR_LABELS)
    return m


def _slices(m: pd.DataFrame):
    for d in DIST_ORDER:
        yield d, (m if d == "all" else m[m.dist_bin == d])


# ── task 2 ───────────────────────────────────────────────────────────────────

def task2():
    met = pd.read_csv(METRICS)
    ch = met[(met.split == "clean_holdout") & met.weight.isin([PARENT, V7])].copy()
    rows = []
    for _, r in ch.sort_values(["dist_bin", "weight"]).iterrows():
        rows.append(dict(
            weight=r.weight, dist_bin=r.dist_bin, n_p=int(r.n_p_traces), n_s=int(r.n_s_traces),
            p_recall_0p3=fmt(r.p_recall, r.p_recall_ci_lo, r.p_recall_ci_hi),
            s_recall_0p3=fmt(r.s_recall, r.s_recall_ci_lo, r.s_recall_ci_hi),
            p_mae_cond_s=fmt(r.p_mae_s_cond), s_mae_cond_s=fmt(r.s_mae_s_cond),
            p_med_prob=fmt(r.p_med_prob), s_med_prob=fmt(r.s_med_prob),
        ))
    write_table("task2a_clean_holdout_aggregates", pd.DataFrame(rows),
                "Task 2a. clean_holdout, as committed in notebooks/step3_metrics.csv (server-computed, 2026-08-14)",
                "Recall intervals are the committed bootstrap CIs (scripts/metrics.py::compute_metrics). "
                "Conditional MAE has no committed interval. No teleseismic clean_holdout rows exist: the "
                "row masks cover only datasets with a verifiable event id. The parent's clean_holdout "
                "population (21,864) is a proxy built from data/manifests, which jma_wc never trained on "
                "(scripts/domain_registry.py:154-162), so the two populations differ by ~1,000 traces and "
                "the comparison is not paired.")

    m = load_pairs()
    rows = []
    for phase in ("p", "s"):
        for d, sub in _slices(m):
            sub = sub[sub[f"{phase}_in_window"] >= 0]
            if len(sub) < 5:
                continue
            pa, pv = sub[f"{phase}_prob_par"].to_numpy(), sub[f"{phase}_prob_v7"].to_numpy()
            ra, rv = (pa >= THR).astype(float), (pv >= THR).astype(float)
            d_rec = paired_diff_ci(rv, ra)
            d_auc = paired_diff_ci(pv, pa)
            both = sub[(pa >= THR) & (pv >= THR)]
            ea, ev = np.abs(both[f"{phase}_residual_s_par"].to_numpy()), np.abs(both[f"{phase}_residual_s_v7"].to_numpy())
            d_mae = paired_diff_ci(ev, ea)
            rows.append(dict(
                phase=phase.upper(), dist_bin=d, n=len(sub),
                recall_parent=fmt(ra.mean(), *ci(ra)), recall_v7=fmt(rv.mean(), *ci(rv)),
                recall_diff_v7_minus_parent=fmt(*d_rec),
                auc_recall_parent=fmt(auc_recall(pd.Series(pa)), *ci(pa)),
                auc_recall_v7=fmt(auc_recall(pd.Series(pv)), *ci(pv)),
                auc_diff=fmt(*d_auc),
                n_both_detected=len(both),
                mae_cond_parent_s=fmt(ea.mean(), *ci(ea)) if len(ea) >= 5 else "n/a",
                mae_cond_v7_s=fmt(ev.mean(), *ci(ev)) if len(ev) >= 5 else "n/a",
                mae_diff_s=fmt(*d_mae),
            ))
    write_table("task2b_paired_full_benchmark", pd.DataFrame(rows),
                "Task 2b. Paired re-score on the full benchmark (v7's in_domain split), threshold 0.3",
                "Paired bootstrap over the same traces (1000 resamples). AUC-recall = mean probability at the "
                "true pick, the exact threshold-free recall summary of scripts/threshold_independent_ranking.py. "
                "Conditional MAE is computed on traces BOTH models detect at 0.3, so the timing comparison is "
                "paired and not confounded by which traces each model detects. A matched pick budget cannot be "
                "formed on this benchmark (positives only; see the module docstring).")

    rows = []
    for phase in ("p", "s"):
        sub = m[m[f"{phase}_in_window"] >= 0]
        pa, pv = sub[f"{phase}_prob_par"], sub[f"{phase}_prob_v7"]
        ca, cv = recall_curve(pa, CURVE_THR), recall_curve(pv, CURVE_THR)
        for t, a_, v_ in zip(CURVE_THR, ca, cv):
            rows.append(dict(phase=phase.upper(), threshold=t, n=len(sub),
                             recall_parent=round(float(a_), 4), picks_parent=int(round(a_ * len(sub))),
                             recall_v7=round(float(v_), 4), picks_v7=int(round(v_ * len(sub))),
                             threshold_v7_for_parent_budget=round(float(np.quantile(pv, 1 - a_)), 3) if 0 < a_ < 1 else np.nan))
    write_table("task2c_recall_curves", pd.DataFrame(rows),
                "Task 2c. Recall curves, all distances (the budget on positives equals recall x n)",
                "`threshold_v7_for_parent_budget` is the v7 threshold that emits the parent's number of picks on "
                "these positives; by construction it gives v7 the parent's recall, which is why matched-budget "
                "recall is uninformative here and must come from continuous data with false alarms.")

    src = DET_SERVER if DET_SERVER.exists() else DET_SNAPSHOT
    det = pd.read_csv(src)
    det = det[det.weight.isin([PARENT, V7, "jma_wc_ft_global_v7_eventclean"])]
    cols = [c for c in ["weight", "best_threshold", "precision", "recall", "detection_mcc", "mcc_ci_lo", "mcc_ci_hi"] if c in det.columns]
    det = det[cols].copy()
    det["source"] = "results/detection_metrics.csv (server)" if src == DET_SERVER else "paper_draft.qmd sec-detection table of 2026-08-10, transcribed"
    write_table("task2d_noise_pool_detection_mcc", det,
                "Task 2d. Noise-pool detection MCC at each model's own best threshold",
                "Positive population: clean_holdout P arrivals; negative population: ~94k pure-noise traces "
                "(scripts/compute_detection_metrics.py). Not recomputed here: results/noise_fp_audit.csv is on the server.")


# ── H1: calibration / peak probability ───────────────────────────────────────

def h1():
    m = load_pairs()
    rows = []
    for phase in ("p", "s"):
        for d, sub in _slices(m):
            sub = sub[sub[f"{phase}_in_window"] >= 0]
            if len(sub) < 5:
                continue
            pa, pv = sub[f"{phase}_prob_par"].to_numpy(), sub[f"{phase}_prob_v7"].to_numpy()
            qa, qv = np.quantile(pa, [.1, .25, .5, .75, .9]), np.quantile(pv, [.1, .25, .5, .75, .9])
            d_med = paired_diff_ci(pv, pa, stat=np.median)
            from scipy.stats import spearmanr
            rho = spearmanr(pa, pv).correlation
            rows.append(dict(
                phase=phase.upper(), dist_bin=d, n=len(sub),
                parent_q10=round(qa[0], 3), parent_q25=round(qa[1], 3),
                parent_median=fmt(qa[2], *ci(pa, np.median)), parent_q75=round(qa[3], 3), parent_q90=round(qa[4], 3),
                v7_q10=round(qv[0], 3), v7_q25=round(qv[1], 3),
                v7_median=fmt(qv[2], *ci(pv, np.median)), v7_q75=round(qv[3], 3), v7_q90=round(qv[4], 3),
                median_diff_v7_minus_parent=fmt(*d_med), spearman_rho=round(float(rho), 3),
                parent_only_at_0p3=int(((pa >= THR) & (pv < THR)).sum()),
                v7_only_at_0p3=int(((pv >= THR) & (pa < THR)).sum()),
            ))
    write_table("h1a_peak_probability_distributions", pd.DataFrame(rows),
                "H1a. Peak probability at the true arrival, v7 vs parent (benchmark, paired traces)",
                "Quantiles of the probability at the argmax inside +-5 s of the analyst pick. "
                "`parent_only`/`v7_only`: traces one model detects at 0.3 and the other does not.")

    # Quantile-mapping test: remove the global calibration offset per phase
    # (map v7's probabilities onto the parent's marginal by rank, pooled over
    # all distances), then ask whether the deficit persists in any bin.
    rows = []
    for phase in ("p", "s"):
        pool = m[m[f"{phase}_in_window"] >= 0].copy()
        pa_all = np.sort(pool[f"{phase}_prob_par"].to_numpy())
        ranks = pool[f"{phase}_prob_v7"].rank(method="average", pct=True).to_numpy()
        pool["v7_cal"] = np.quantile(pa_all, np.clip(ranks, 0, 1))
        for kind, key, order in (("distance", "dist_bin", DIST_ORDER[:-1]), ("snr", "snr_bin", SNR_LABELS)):
            for b in order:
                sub = pool[pool[key] == b]
                if len(sub) < 5:
                    continue
                ra = (sub[f"{phase}_prob_par"].to_numpy() >= THR).astype(float)
                rv = (sub[f"{phase}_prob_v7"].to_numpy() >= THR).astype(float)
                rc = (sub["v7_cal"].to_numpy() >= THR).astype(float)
                rows.append(dict(
                    phase=phase.upper(), bin_type=kind, bin=b, n=len(sub),
                    recall_parent=fmt(ra.mean(), *ci(ra)),
                    recall_v7_raw=fmt(rv.mean(), *ci(rv)),
                    recall_v7_recalibrated=fmt(rc.mean(), *ci(rc)),
                    raw_diff=fmt(*paired_diff_ci(rv, ra)),
                    residual_diff_after_recalibration=fmt(*paired_diff_ci(rc, ra)),
                ))
    write_table("h1b_recalibration_residual", pd.DataFrame(rows),
                "H1b. Does the deficit survive a global recalibration of v7 onto the parent's probability scale?",
                "v7's probabilities are rank-mapped onto the parent's marginal distribution per phase (pooled over "
                "all distances), which forces equal recall at every threshold on the pooled set. A non-zero "
                "residual inside a bin is structure a threshold cannot remove: capability, not calibration.")


# ── H2: S supervision by distance ────────────────────────────────────────────

def _parse_dataset_configs() -> pd.DataFrame:
    src = (REPO_ROOT / "scripts" / "build_training_dataset.py").read_text()
    block = src[src.index("DATASET_CONFIGS = ["):src.index("TARGET_FRACTIONS = {")]
    rows = []
    for mm in re.finditer(r'dict\(name="(\w+)".*?cap=([\d_]+),\s*default_bin=("?\w*"?|None),\s*use_s=(True|False)\)', block, flags=re.S):
        rows.append(dict(dataset=mm.group(1), cap=int(mm.group(2).replace("_", "")),
                         default_bin=mm.group(3).strip('"'), use_s=mm.group(4) == "True"))
    return pd.DataFrame(rows)


def h2():
    if MANIFEST_V2.exists():
        man = pd.read_csv(MANIFEST_V2, low_memory=False)
        rows = []
        for b, sub in man.groupby("distance_bin"):
            has_s = sub["s_arrival_sample"].notna().astype(float).to_numpy()
            rows.append(dict(distance_bin=b, n_windows=len(sub), s_fraction=fmt(has_s.mean(), *ci(has_s)),
                             n_datasets=sub.dataset_name.nunique(), source="data/manifests_v2/train.csv"))
        write_table("h2a_s_fraction_by_distance_manifest", pd.DataFrame(rows),
                    "H2a. S-label fraction of manifests_v2/train.csv by distance bin")
    else:
        cfg = _parse_dataset_configs()
        cfg["p_only_by_policy"] = ~cfg.use_s
        total = cfg.cap.sum()
        rows = [dict(quantity="cap budget, all sources", value=int(total), basis="DATASET_CONFIGS caps (scripts/build_training_dataset.py:209-319)"),
                dict(quantity="cap budget, P-only sources (geofon, lendb, meier2019jgr)", value=int(cfg[~cfg.use_s].cap.sum()), basis="use_s=False"),
                dict(quantity="share of cap budget that can never carry S", value=round(float(cfg[~cfg.use_s].cap.sum() / total), 3), basis="upper bound; the manifest is drawn from these pools"),
                dict(quantity="teleseismic target fraction of the training split", value=0.25, basis="TARGET_FRACTIONS (build_training_dataset.py:325-330); S nulled for every teleseismic window (P-only policy, ~line 570)"),
                dict(quantity="S fraction of manifests_v2/train.csv, all bins", value=0.38, basis="data/README.md ('~527k train, ~38% S'); by-bin split needs the manifest (server)"),
                dict(quantity="S fraction by distance bin", value="not computed", basis="data/manifests_v2/train.csv is on the lab server (unreachable 2026-09-07)")]
        write_table("h2a_s_fraction_by_distance_bound", pd.DataFrame(rows),
                    "H2a. S supervision in the training corpus: what can be bounded without the manifest",
                    "The per-bin S fraction is the number H2 needs; it must be computed on the server with this "
                    "script once data/manifests_v2/train.csv is reachable (verify with scripts/hash_manifests.py --check).")
        write_table("h2a_dataset_configs_parsed", cfg, "H2a (appendix). DATASET_CONFIGS as parsed: caps, default bin, S policy")

    m = load_pairs()
    rows = []
    for d, sub in _slices(m):
        for phase in ("s", "p"):
            sub2 = sub[sub[f"{phase}_in_window"] >= 0]
            if len(sub2) < 5:
                continue
            ra = (sub2[f"{phase}_prob_par"].to_numpy() >= THR).astype(float)
            rv = (sub2[f"{phase}_prob_v7"].to_numpy() >= THR).astype(float)
            rows.append(dict(phase=phase.upper(), dist_bin=d, n=len(sub2),
                             recall_parent=fmt(ra.mean(), *ci(ra)), recall_v7=fmt(rv.mean(), *ci(rv)),
                             diff_v7_minus_parent=fmt(*paired_diff_ci(rv, ra))))
    write_table("h2b_recall_deficit_by_distance", pd.DataFrame(rows),
                "H2b. The observable H2 predicts: v7's recall deficit by phase and distance (paired, threshold 0.3)",
                "H2 predicts a larger S deficit than P deficit, growing with distance. Teleseismic S is absent from "
                "the benchmark (no S labels at >1500 km), as it is from the training policy.")


# ── H3: contamination / in-domain ────────────────────────────────────────────

def h3():
    met = pd.read_csv(METRICS)
    sub = met[met.weight.isin([PARENT, V7]) & met.split.isin(["all", "in_domain", "cross_domain", "clean_holdout"]) & (met.dist_bin == "all")]
    rows = [dict(weight=r.weight, split=r.split, n=int(r.n_traces),
                 p_recall=fmt(r.p_recall, r.p_recall_ci_lo, r.p_recall_ci_hi),
                 s_recall=fmt(r.s_recall, r.s_recall_ci_lo, r.s_recall_ci_hi),
                 p_mae_cond_s=fmt(r.p_mae_s_cond)) for _, r in sub.iterrows()]
    write_table("h3a_in_domain_vs_all", pd.DataFrame(rows),
                "H3a. v7 is scored in-domain on the whole benchmark and still loses",
                "For v7, in_domain == all: every one of the 12 benchmark datasets is in manifests_v2 "
                "(scripts/domain_registry.py:188-195 splits own models on the manifest's dataset set), so no "
                "cross_domain row exists for v7. The parent has no verifiable training split here "
                "(BASE_TRAINED_ON has no jma_wc entry), so its in_domain row is the fallback all-True mask. "
                "Contamination of the external sequences can only make v7 look better than it is; it cannot "
                "explain a loss.")

    bm = pd.read_csv(BENCH_MANIFEST, usecols=["dataset", "trace_name", "source_month", "source_latitude_deg", "source_longitude_deg"])
    hits = hs.spatial_hits(bm.source_latitude_deg, bm.source_longitude_deg)
    hits.index = bm.index
    rows = []
    for ds, g in bm.groupby("dataset"):
        r = dict(dataset=ds, n_benchmark=len(g), n_with_location=int(g.source_latitude_deg.notna().sum()),
                 months_known=f"{int(g.source_month.min())}-{int(g.source_month.max())}" if g.source_month.notna().any() else "unknown")
        for w in hs.WINDOW_NAMES:
            if w.endswith("_mainshock"):
                continue
            r[w.replace("_sequence", "") + "_radius"] = int(hits.loc[g.index, w].sum())
        rows.append(r)
    write_table("h3b_benchmark_spatial_prescreen", pd.DataFrame(rows),
                "H3b. Benchmark traces inside a held-out window's RADIUS (space only; origin times are on the server)",
                "notebooks/benchmark_manifest.csv carries source lat/lon but an origin month only for mlaapde "
                "(2013-07 to 2014-10). Counts are an upper bound on benchmark traces that could sit in a window; "
                "the spatiotemporal join (scripts/audit_heldout_sequences.py, benchmark_manifest source) resolves them.")

    rows = [
        dict(sequence="Norcia 2016 (INGV)", corpus_datasets_covering_region_and_dates="instancecounts (Italy, 2005-01 to 2020-01); stead (global, 2005-2018)", status="likely in v7's corpus", basis="docs/2026-09-07_training_history_audit.md section 5 (date coverage only, no join yet)"),
        dict(sequence="Kaikoura 2016 (GeoNet)", corpus_datasets_covering_region_and_dates="stead (global, 2005-2018); crew (global regional)", status="possible", basis="same"),
        dict(sequence="Thessaly 2021 (NOA)", corpus_datasets_covering_region_and_dates="crew (global regional)", status="possible; instancecounts ends 2020-01, stead 2018", basis="same"),
        dict(sequence="Ridgecrest 2019 (SCEDC/NCEDC)", corpus_datasets_covering_region_and_dates="scedc, ceed, ross2018gpd (S. California)", status="likely in v7's corpus", basis="same"),
        dict(sequence="Monroe 2019 (UW)", corpus_datasets_covering_region_and_dates="pnw (Cascadia)", status="likely in v7's corpus", basis="same"),
        dict(sequence="all five", corpus_datasets_covering_region_and_dates="manifests_v2 train/val + 10 full corpora", status="counts NOT computed: lab server unreachable 2026-09-07", basis="scripts/audit_heldout_sequences.py (task 1) must run on the server"),
    ]
    write_table("h3c_sequence_corpus_coverage", pd.DataFrame(rows),
                "H3c. Which corpus datasets could hold each external sequence (documentation, not a join)")


# ── H4: low SNR ──────────────────────────────────────────────────────────────

def h4():
    m = load_pairs()
    rows = []
    for phase in ("p", "s"):
        pool = m[m[f"{phase}_in_window"] >= 0]
        for b in SNR_LABELS:
            sub = pool[pool.snr_bin == b]
            if len(sub) < 5:
                continue
            pa, pv = sub[f"{phase}_prob_par"].to_numpy(), sub[f"{phase}_prob_v7"].to_numpy()
            ra, rv = (pa >= THR).astype(float), (pv >= THR).astype(float)
            rows.append(dict(phase=phase.upper(), snr_bin=b, n=len(sub), share_of_traces=round(len(sub) / len(pool), 3),
                             recall_parent=fmt(ra.mean(), *ci(ra)), recall_v7=fmt(rv.mean(), *ci(rv)),
                             recall_diff=fmt(*paired_diff_ci(rv, ra)),
                             median_prob_parent=fmt(np.median(pa), *ci(pa, np.median)),
                             median_prob_v7=fmt(np.median(pv), *ci(pv, np.median)),
                             median_prob_diff=fmt(*paired_diff_ci(pv, pa, stat=np.median))))
    write_table("h4a_recall_by_snr", pd.DataFrame(rows),
                "H4a. Paired recall and peak probability by SNR bin (benchmark snr_db column, threshold 0.3)")

    rows = []
    for phase in ("p", "s"):
        pool = m[m[f"{phase}_in_window"] >= 0]
        pa, pv = pool[f"{phase}_prob_par"].to_numpy(), pool[f"{phase}_prob_v7"].to_numpy()
        snr = pool.snr_db.to_numpy()
        miss_v7 = pv < THR
        miss_par = pa < THR
        par_only = (pa >= THR) & (pv < THR)
        v7_only = (pv >= THR) & (pa < THR)
        low = (snr < 5).astype(float)
        rows.append(dict(phase=phase.upper(), n=len(pool),
                         share_below_5dB_all=fmt(low.mean(), *ci(low)),
                         share_below_5dB_of_v7_misses=fmt(low[miss_v7].mean(), *ci(low[miss_v7])),
                         share_below_5dB_of_parent_misses=fmt(low[miss_par].mean(), *ci(low[miss_par])),
                         n_parent_only=int(par_only.sum()),
                         share_below_5dB_of_parent_only=fmt(low[par_only].mean(), *ci(low[par_only])),
                         n_v7_only=int(v7_only.sum()),
                         share_below_5dB_of_v7_only=fmt(low[v7_only].mean(), *ci(low[v7_only])),
                         median_snr_parent_only_dB=round(float(np.median(snr[par_only])), 2) if par_only.any() else np.nan,
                         median_snr_v7_only_dB=round(float(np.median(snr[v7_only])), 2) if v7_only.any() else np.nan))
    write_table("h4b_misses_by_snr", pd.DataFrame(rows),
                "H4b. Where the misses live: SNR of the traces each model misses, and of the traces only one detects",
                "The v13 config header reports '82% of v7's misses below 5 dB'; this table checks it on the committed results.")


# ── H00: selection metric vs detection MCC ordering ──────────────────────────

def h00():
    src = DET_SERVER if DET_SERVER.exists() else DET_SNAPSHOT
    det = pd.read_csv(src)
    own = det[det.weight.str.startswith("jma_wc")].copy()
    met = pd.read_csv(METRICS)
    sel = met[(met.split == "all") & (met.dist_bin == "all")].set_index("weight")
    chd = met[(met.split == "clean_holdout") & (met.dist_bin == "all")].set_index("weight")
    df = pd.read_parquet(PARQUET, columns=["weight", "p_in_window", "p_prob"], filters=[("weight", "in", own.weight.tolist())])
    auc = df[df.p_in_window >= 0].groupby("weight").p_prob.mean()
    rows = []
    for _, r in own.iterrows():
        w = r.weight
        rows.append(dict(weight=w,
                         detection_mcc=fmt(r.detection_mcc, r.get("mcc_ci_lo", np.nan), r.get("mcc_ci_hi", np.nan)),
                         p_mae_unconditional_all=float(sel.loc[w, "p_mae_s"]) if w in sel.index else np.nan,
                         p_recall_all=float(sel.loc[w, "p_recall"]) if w in sel.index else np.nan,
                         p_recall_clean_holdout=float(chd.loc[w, "p_recall"]) if w in chd.index else np.nan,
                         p_auc_recall_all=round(float(auc.get(w, np.nan)), 4),
                         _mcc=float(r.detection_mcc)))
    t = pd.DataFrame(rows)
    t["rank_detection_mcc"] = t["_mcc"].rank(ascending=False).astype(int)
    t["rank_p_mae_selection_metric"] = t["p_mae_unconditional_all"].rank(ascending=True).astype("Int64")
    t["rank_p_recall_all"] = t["p_recall_all"].rank(ascending=False).astype("Int64")
    t["rank_p_auc_recall"] = t["p_auc_recall_all"].rank(ascending=False).astype("Int64")
    from scipy.stats import spearmanr
    notes = []
    for c in ("rank_p_mae_selection_metric", "rank_p_recall_all", "rank_p_auc_recall"):
        ok = t[c].notna()
        rho = spearmanr(t.loc[ok, "rank_detection_mcc"], t.loc[ok, c].astype(int)).correlation
        notes.append(f"Spearman(rank detection MCC, {c}) = {rho:.2f} over {int(ok.sum())} weights")
    t = t.drop(columns=["_mcc"]).sort_values("rank_detection_mcc")
    scored = set(own.weight)
    all_versions = [f"jma_wc_ft_global_v{i}" for i in range(2, 21)] + ["jma_wc_ft", "jma_wc_ft_frozen", "jma_wc_ft_global", "jma_wc_ft_noise"]
    missing = [v for v in all_versions if v not in scored]
    write_table("h00_detection_mcc_ordering", t,
                "H00. Ordering under the noise-pool detection MCC versus the metrics that drove selection",
                "; ".join(notes) + f".\n\nSource: {'results/detection_metrics.csv (server)' if src == DET_SERVER else 'the 2026-08-10 table transcribed from paper_draft.qmd'}. "
                f"Only {len(scored)} own weights were ever run over the noise pool. Never scored: {', '.join(missing)} "
                "(their checkpoints, if kept, are under checkpoints/ on the server; scripts/audit_noise_fp_leaderboard.py "
                "SINGLE_MODELS lists what was run). The full v1-v20 ordering H00 asks for cannot be produced without them.")


# ── external results: intervals from the published counts ───────────────────

def external():
    """Bootstrap intervals for the QuakeScope recall numbers from their
    (matched, analyst) counts. The per-pick data behind the matched-budget
    curves is not stored (the notebooks cache only the analyst harvest), so
    the matched-budget differences carry no interval here; the shared-
    threshold difference gets an unpaired one (two independent 0/1 vectors)."""
    ext = pd.read_csv(OUT / "external_results_2026-09-07.csv")
    rows = []
    for (seq, phase), g in ext.groupby(["sequence", "phase"], sort=False):
        q = g[g.weights == "quakescope2026"].iloc[0]
        j = g[g.weights == "jma_wc"].iloc[0]
        n = int(q.analyst)
        vq = np.r_[np.ones(int(q.matched)), np.zeros(n - int(q.matched))]
        vj = np.r_[np.ones(int(j.matched)), np.zeros(n - int(j.matched))]
        rng = np.random.default_rng(SEED)
        bq = vq[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
        bj = vj[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
        d = bq - bj
        rows.append(dict(sequence=seq, region=q.region, phase=phase, analyst=n,
                         recall_0p3_v7=fmt(vq.mean(), *ci(vq)), recall_0p3_parent=fmt(vj.mean(), *ci(vj)),
                         diff_0p3_v7_minus_parent=fmt(vq.mean() - vj.mean(), *np.percentile(d, [2.5, 97.5])),
                         budget_mid=q.budget_mid if pd.notna(q.budget_mid) else "n/a",
                         diff_at_matched_budget=q.diff_at_budget if pd.notna(q.diff_at_budget) else "n/a",
                         MAE_v7_s=q.MAE_s, MAE_parent_s=j.MAE_s))
    t = pd.DataFrame(rows)
    non_us = t[t.region == "non-US"]
    signs = (non_us.diff_at_matched_budget.astype(float) < 0).sum()
    write_table("external_recall_with_intervals", t,
                "External results (QuakeScope notebooks) with bootstrap intervals on recall",
                f"Unpaired difference intervals (independent resampling of the two 0/1 vectors; the true paired "
                f"interval would be narrower). At matched budget the fine-tune is below the parent in {signs} of "
                f"{len(non_us)} non-US sequence x phase pairs; a sign test gives p = {0.5 ** len(non_us):.3f} one-sided. "
                "Matched-budget differences have no interval: the per-pick data is not stored by the notebooks.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["all", "task2", "h1", "h2", "h3", "h4", "h00", "external"])
    args = ap.parse_args()
    todo = ["task2", "h1", "h2", "h3", "h4", "h00", "external"] if "all" in args.what else args.what
    for w in todo:
        print("\n" + "=" * 100 + f"\n{w}\n" + "=" * 100)
        globals()[w]()


if __name__ == "__main__":
    main()
