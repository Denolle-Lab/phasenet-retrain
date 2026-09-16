#!/usr/bin/env python3
"""
audit_benchmark_labels.py

Laptop label audit of the v7 benchmark test set (issue #41, checkpoint 41B,
benchmark part). No waveform and no model: the inputs are the committed
`notebooks/benchmark_manifest.csv` (labels, distances, the flags the
notebook wrote) and `notebooks/step3_results.parquet` (per weight x trace
the probability at the argmax within +-5 s of the label and the residual of
that argmax, `scripts/eval_finetuned.py` lines 157-171). Every residual is
measured in the 30 s evaluation window of notebook 05 (3000 samples at a
nominal 100 Hz, the P placed at a random position 3-27 s into the window,
the window zero-padded where it reaches past the source trace). A consensus
within EDGE_S of the first or last real sample of the source trace (its
length from the `:N` of the bucket-style trace name, in the nominal seconds
of notebook 05: 40 Hz for mlaapde, 100 Hz otherwise) or of the evaluation
window is a picker edge artefact (the step from the zero padding into the
data), not a label error, and is reported separately.

Screens (constants below; per-trace columns in per_trace_flags.parquet):

  S1  consensus offset   for the P and the S label separately: at least
                         S1_MIN_DETECT public PhaseNet weights that were not
                         trained on the trace's own dataset detect the phase
                         (prob >= DETECT_THRESHOLD) and agree among themselves
                         (MAD of their residuals <= S1_MAX_MAD_S) on a pick
                         more than S1_MIN_OFFSET_S from the label; a consensus
                         within EDGE_S of the source trace's or the window's
                         start or end is `consensus_at_edge` and never flags
                         (the source end is unknown for ceed and cwa, whose
                         names carry no length: only the start applies); one at
                         the +-5 s search boundary is `consensus_saturated`
                         (flagged, counted separately).
  S2  consensus miss     SNR > S2_MIN_SNR_DB and at most S2_MAX_DETECT
                         independent weights detect the phase.
  C1  S-P vs distance    audit_source_labels.sp_consistency on the manifest's
                         ts_tp_s per dataset, once against distance_km (the
                         epicentral slope and Vp/Vs reported) and once against
                         the hypocentral distance sqrt(D^2 + z^2) from depth_km;
                         the hypocentral residual is the flag (`c1_flag`; the
                         epicentral one, `c1_epi_flag`, marks every deep
                         INSTANCE event at short epicentral distance).
  C4  placement          modal share of p_arrival_sample and s_arrival_sample,
                         edge share in the source trace when its length is in
                         the bucket name, the manifest's own p_in_s_window and
                         multi_arrival flags cross-tabulated with S1.
  Aguilar cross-check    benchmark rows still named in a cached multiplet
                         report (data/labelerrors/ or
                         ~/.cache/phasenet_retrain/label_errors/; downloaded
                         once into the first directory when absent, sha256
                         recorded); S1/S2 rates by report flag.
  Sensitivity            P and S recall at 0.3 and the conditional MAE of
                         SENSITIVITY_WEIGHTS on all rows and on rows with no
                         S1/S2/C1 flag (`clean_rule` s1_s2_c1) or no S1/C1 flag
                         (s1_c1: S2 rests on the P-window SNR of notebook 05 and
                         on the pickers' S sensitivity), with a paired
                         bootstrap on the difference, and the paired
                         v7 - parent gap on both.

Independence: a weight is independent of a trace when the trace's
`trained_models` label does not contain the weight's public training corpus
(`domain_registry.BASE_TRAINED_ON`, `original` -> stead). `jma_wc` is the
parent trained on Japanese data (no benchmark dataset) and counts as
independent everywhere.

    python scripts/audit_benchmark_labels.py --benchmark notebooks/benchmark_manifest.csv \\
        --results notebooks/step3_results.parquet --out-dir docs/audit_2026-09-16_benchmark_labels
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import domain_registry as dr  # noqa: E402
import label_error_filter as lef  # noqa: E402
from audit_source_labels import (  # noqa: E402
    DEFAULT_REPORT_DIRS, VP_KM_S, _frac_ci, _num, _pct, _sha256, modal_share, sp_consistency)
from metrics import bootstrap_ci  # noqa: E402

AUDIT_VERSION = "41b-benchmark-v1"
N_BOOT = 1000
BOOT_SEED = 0

PUBLIC_WEIGHTS = ("stead", "jma", "original", "lendb", "iquique", "ethz", "scedc", "jma_wc", "geofon", "phasenet_sn",
                  "diting", "pisdl", "neic", "instance", "obs", "volpick")
SENSITIVITY_WEIGHTS = ("jma_wc", "instance", "jma_wc_ft_global_v7")
GAP_PAIRS = (("jma_wc_ft_global_v7", "jma_wc"), ("jma_wc_ft_global_v7", "instance"))

# notebooks/05_benchmark_waveform_processing.ipynb: WINDOW_SAMPLES 3000, TARGET_SR 100, P_JITTER 300-2700
WINDOW_S = 30.0
RATE_HZ = 100.0
SEARCH_S = 5.0                 # scripts/eval_finetuned.py SEARCH_WIN_S
DETECT_THRESHOLD = 0.3
S1_MIN_DETECT = 6
S1_MAX_MAD_S = 0.15
S1_MIN_OFFSET_S = 0.3
EDGE_S = 1.0
SATURATION_TOL_S = 0.011       # residuals are rounded to 0.01 s
S2_MIN_SNR_DB = 10.0
S2_MAX_DETECT = 1
REVIEW_ROWS_PER_DATASET = 30
SOURCE_EDGE_S = 1.0
# notebook 05 NATIVE_SR: the rate the evaluation window and its label indices were cut at (nominal for ethz/aq2009gm)
NOTEBOOK05_NATIVE_SR = {"mlaapde": 40.0}
NOTEBOOK05_DEFAULT_SR = 100.0
# datasets whose trace_name is a positional slot reused across chunks: a name-level match is ambiguous
CHUNKED_NAME_DATASETS = ("mlaapde", "cwa", "aq2009gm")

MANIFEST_COLUMNS = ["dataset", "trace_name", "trained_models", "distance_km", "depth_km", "ts_tp_s", "p_arrival_sample",
                    "s_arrival_sample", "has_p_pick", "has_s_pick", "evaluate_s", "p_in_s_window", "multi_arrival",
                    "clipped_flag", "dist_bin", "magnitude"]
RESULT_COLUMNS = ["weight", "trace_name", "dataset", "dist_bin", "trained_models", "snr_db", "p_in_window", "s_in_window",
                  "p_prob", "s_prob", "p_residual_s", "s_residual_s"]
KEY = ["dataset", "trace_name"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── inputs ───────────────────────────────────────────────────────────────────

def load_manifest(path) -> pd.DataFrame:
    """The benchmark manifest, one row per (dataset, trace_name) (the chunked
    sources repeat a few names; the first occurrence is kept, as the waveform
    HDF5 of notebook 05 did)."""
    m = pd.read_csv(path, low_memory=False)
    missing = [c for c in MANIFEST_COLUMNS if c not in m.columns]
    if missing:
        raise ValueError(f"benchmark manifest lacks columns {missing}")
    m = m.drop_duplicates(KEY, keep="first").reset_index(drop=True)
    m["source_trace_samples"] = bucket_length(m["trace_name"])
    nb_rate = m["dataset"].map(NOTEBOOK05_NATIVE_SR).fillna(NOTEBOOK05_DEFAULT_SR)
    m["source_length_s"] = m["source_trace_samples"] / nb_rate
    m["p_source_s"] = m["p_arrival_sample"] / nb_rate
    m["s_source_s"] = m["s_arrival_sample"] / nb_rate
    m["notebook_rate_hz"] = (m["s_arrival_sample"] - m["p_arrival_sample"]) / m["ts_tp_s"]
    m.loc[~np.isfinite(m["notebook_rate_hz"]) | (m["notebook_rate_hz"] <= 0), "notebook_rate_hz"] = np.nan
    return m


def bucket_length(names: pd.Series) -> pd.Series:
    """Trace length in samples from the `bucket{N}${row},:3,:{samples}` name; NaN otherwise."""
    return pd.to_numeric(names.astype(str).str.extract(r":(\d+)$")[0], errors="coerce")


def load_results(path, weights) -> pd.DataFrame:
    """Rows of the listed weights, one per (weight, dataset, trace_name)."""
    r = pd.read_parquet(path, columns=RESULT_COLUMNS)
    r = r[r["weight"].isin(list(weights))]
    return r.drop_duplicates(["weight"] + KEY, keep="first").reset_index(drop=True)


def trace_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, trace_name): the label positions in the evaluation
    window, SNR and the trained_models label, indexed by the key."""
    t = results.drop_duplicates(KEY).set_index(KEY)[["p_in_window", "s_in_window", "snr_db", "trained_models", "dist_bin"]]
    return t.sort_index()


def weight_trained_on(weight: str):
    """Public training corpus a weight must be independent of; None when it
    trained on no benchmark dataset. `jma_wc` is the parent (Japan) and is
    independent of every benchmark dataset."""
    if weight == "jma_wc":
        return None
    return dr._resolve_public_trained_on(weight)


def independence(traces: pd.DataFrame, weights) -> pd.DataFrame:
    """(n_traces x n_weights) boolean frame: True where the weight was not
    trained on the trace's dataset (`trained_models` does not contain the
    weight's corpus)."""
    tm = traces["trained_models"].astype(str)
    cols = {}
    for w in weights:
        corpus = weight_trained_on(w)
        cols[w] = np.ones(len(traces), dtype=bool) if corpus is None else ~tm.str.contains(corpus, regex=False).to_numpy()
    return pd.DataFrame(cols, index=traces.index)


def pivot(results: pd.DataFrame, column: str, weights, index) -> pd.DataFrame:
    p = results.pivot(index=KEY, columns="weight", values=column)
    return p.reindex(index=index, columns=list(weights))


# ── S1 / S2 ──────────────────────────────────────────────────────────────────

def consensus(prob, residual, independent, label_s, label_source_s=None, source_length_s=None, window_s=WINDOW_S, *,
              threshold=DETECT_THRESHOLD, min_detect=S1_MIN_DETECT, max_mad_s=S1_MAX_MAD_S,
              min_offset_s=S1_MIN_OFFSET_S, edge_s=EDGE_S, search_s=SEARCH_S) -> pd.DataFrame:
    """S1 rule on arrays of shape (n_traces, n_weights): prob and residual of
    every weight, `independent` the boolean matrix, `label_s` the label
    position in the evaluation window, `label_source_s` the same label in
    the source trace and `source_length_s` the source length (NaN when
    unknown; both default to the window). A weight detects when it is
    independent, has a finite residual and prob >= threshold. Over the
    detecting weights the consensus residual is the median and the MAD the
    median absolute deviation from it; a consensus exists when >= min_detect
    weights detect and MAD <= max_mad_s. consensus_at_edge: the consensus
    position (label + residual) lies within edge_s of the start or end of
    the window or of the source trace (before the first real sample when the
    window was zero-padded). consensus_saturated: |residual| at the search
    boundary. s1_flag: consensus, |residual| > min_offset_s and not at an edge."""
    prob = np.asarray(prob, dtype=float)
    residual = np.asarray(residual, dtype=float)
    independent = np.asarray(independent, dtype=bool)
    label_s = np.asarray(label_s, dtype=float)
    window_s = np.broadcast_to(np.asarray(window_s, dtype=float), label_s.shape)
    label_source_s = label_s if label_source_s is None else np.asarray(label_source_s, dtype=float)
    source_length_s = window_s if source_length_s is None else np.asarray(source_length_s, dtype=float)
    has_row = np.isfinite(prob)
    detect = independent & has_row & np.isfinite(residual) & (prob >= threshold)
    n_independent = (independent & has_row).sum(axis=1)
    n_detect = detect.sum(axis=1)
    r = np.where(detect, residual, np.nan)
    with np.errstate(all="ignore"):
        med = _row_nanmedian(r)
        mad = _row_nanmedian(np.abs(r - med[:, None]))
    has_consensus = (n_detect >= min_detect) & (mad <= max_mad_s)
    pos = label_s + med
    src = label_source_s + med
    with np.errstate(invalid="ignore"):
        near_end = np.isfinite(source_length_s) & (src > source_length_s - edge_s)
        at_edge = has_consensus & ((pos < edge_s) | (pos > window_s - edge_s) | (src < edge_s) | near_end)
    saturated = has_consensus & (np.abs(med) >= search_s - SATURATION_TOL_S)
    offset = has_consensus & (np.abs(med) > min_offset_s)
    flag = offset & ~at_edge
    return pd.DataFrame(dict(n_independent=n_independent, n_detect=n_detect, consensus_residual_s=med,
                             consensus_mad_s=mad, consensus_position_s=pos, consensus_source_position_s=src,
                             has_consensus=has_consensus, consensus_at_edge=at_edge, consensus_saturated=saturated,
                             s1_flag=flag))


def _row_nanmedian(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    ok = np.isfinite(a).any(axis=1)
    if ok.any():
        out[ok] = np.nanmedian(a[ok], axis=1)
    return out


def consensus_miss(n_detect, snr_db, has_label, *, min_snr_db=S2_MIN_SNR_DB, max_detect=S2_MAX_DETECT) -> np.ndarray:
    """S2 rule: the phase is labelled, SNR > min_snr_db and at most max_detect independent weights detect it."""
    n_detect = np.asarray(n_detect, dtype=float)
    snr_db = np.asarray(snr_db, dtype=float)
    has_label = np.asarray(has_label, dtype=bool)
    return has_label & np.isfinite(snr_db) & (snr_db > min_snr_db) & (n_detect <= max_detect)


def screen_phase(results, traces, indep, phase: str) -> pd.DataFrame:
    """S1 and S2 for one phase ("p" or "s") over the trace table."""
    prob = pivot(results, f"{phase}_prob", indep.columns, traces.index)
    resid = pivot(results, f"{phase}_residual_s", indep.columns, traces.index)
    label = traces[f"{phase}_in_window"].to_numpy(dtype=float)
    has_label = label >= 0
    label_s = np.where(has_label, label / RATE_HZ, np.nan)
    src_col, len_col = f"{phase}_source_s", "source_length_s"
    label_source_s = traces[src_col].to_numpy(dtype=float) if src_col in traces else None
    source_length_s = traces[len_col].to_numpy(dtype=float) if len_col in traces else None
    out = consensus(prob.to_numpy(), resid.to_numpy(), indep.to_numpy(), label_s, label_source_s, source_length_s)
    out.index = traces.index
    out["s2_flag"] = consensus_miss(out["n_detect"], traces["snr_db"], has_label)
    out.loc[~has_label, ["consensus_residual_s", "consensus_mad_s", "consensus_position_s",
                         "consensus_source_position_s"]] = np.nan
    out.loc[~has_label, ["has_consensus", "consensus_at_edge", "consensus_saturated", "s1_flag"]] = False
    out["has_label"] = has_label
    out["label_s"] = label_s
    if phase == "s":
        out = out.rename(columns={c: f"{c}_s" for c in out.columns})
    return out


# ── C1 and C4 on the manifest ───────────────────────────────────────────────

def hypocentral_km(dist_km, depth_km) -> np.ndarray:
    """sqrt(D^2 + z^2); a missing or negative depth counts as 0."""
    d = np.asarray(dist_km, dtype=float)
    z = np.nan_to_num(np.asarray(depth_km, dtype=float), nan=0.0).clip(min=0.0)
    return np.sqrt(d ** 2 + z ** 2)


def c1_manifest(m: pd.DataFrame) -> tuple:
    """C1 per dataset (sp_consistency with tp = 0, ts = ts_tp_s) against the
    epicentral distance_km (slope, intercept, Vp/Vs and `c1_epi_flag`) and
    against the hypocentral distance (`c1_residual_s`, `c1_flag`: the flag
    the clean subset uses, since a single line in epicentral distance marks
    every deep event at short distance). Returns (per-row frame, per-dataset dict)."""
    rows = pd.DataFrame(index=m.index, data=dict(c1_residual_s=np.nan, c1_flag=False, c1_epi_residual_s=np.nan,
                                                 c1_epi_flag=False))
    per = {}
    for ds, g in m.groupby("dataset", sort=True):
        ts = g["ts_tp_s"].to_numpy(dtype=float)
        tp = np.where(np.isfinite(ts), 0.0, np.nan)
        d = g["distance_km"].to_numpy(dtype=float)
        epi = sp_consistency(tp.tolist(), ts.tolist(), d.tolist())
        hyp = sp_consistency(tp.tolist(), ts.tolist(), hypocentral_km(d, g["depth_km"]).tolist())
        rows.loc[g.index, "c1_epi_residual_s"] = epi["residual_s"]
        rows.loc[g.index, "c1_epi_flag"] = epi["flag"]
        rows.loc[g.index, "c1_residual_s"] = hyp["residual_s"]
        rows.loc[g.index, "c1_flag"] = hyp["flag"]
        both = np.isfinite(ts)
        frac, lo, hi = _frac_ci(hyp["flag"][both]) if both.any() else (np.nan, np.nan, np.nan)
        e_frac, e_lo, e_hi = _frac_ci(epi["flag"][both]) if both.any() else (np.nan, np.nan, np.nan)
        per[ds] = dict(c1_n_fit=int(epi["n_fit"]), c1_slope_s_per_km=epi["slope_s_per_km"], c1_intercept_s=epi["intercept_s"],
                       c1_implied_vp_vs=epi["implied_vp_vs"], c1_epi_residual_mad_s=epi["mad_s"],
                       c1_epi_threshold_s=epi["threshold_s"], c1_epi_n_flag=int(epi["flag"].sum()), c1_epi_flag_frac=e_frac,
                       c1_epi_flag_lo=e_lo, c1_epi_flag_hi=e_hi,
                       c1_hyp_slope_s_per_km=hyp["slope_s_per_km"], c1_hyp_intercept_s=hyp["intercept_s"],
                       c1_hyp_implied_vp_vs=hyp["implied_vp_vs"], c1_residual_mad_s=hyp["mad_s"],
                       c1_threshold_s=hyp["threshold_s"], c1_n_flag=int(hyp["flag"].sum()), c1_flag_frac=frac,
                       c1_flag_lo=lo, c1_flag_hi=hi)
    return rows, per


def c4_manifest(g: pd.DataFrame) -> dict:
    """C4 for one dataset's manifest rows: modal P and S samples and their
    shares, the edge share in the source trace (needs the length in the name
    and the rate the notebook used), the source length and rate modes."""
    p = g["p_arrival_sample"].astype(float)
    s = g["s_arrival_sample"].astype(float)
    p_mode, p_share = modal_share(p)
    s_mode, s_share = modal_share(s)
    n_mode, _ = modal_share(g["source_trace_samples"])
    rate_mode, _ = modal_share(g["notebook_rate_hz"])
    rate = g["notebook_rate_hz"].fillna(rate_mode if np.isfinite(rate_mode) else RATE_HZ)
    length_s = g["source_trace_samples"] / rate
    p_s = p / rate
    known = p.notna() & length_s.notna()
    edge = known & ((p_s < SOURCE_EDGE_S) | (p_s > length_s - SOURCE_EDGE_S))
    return dict(c4_p_sample_mode=p_mode, c4_p_sample_mode_share=p_share, c4_s_sample_mode=s_mode,
                c4_s_sample_mode_share=s_share, c4_source_samples_mode=n_mode, c4_notebook_rate_hz_mode=rate_mode,
                c4_n_length_known=int(known.sum()),
                c4_source_edge_frac=float(edge[known].mean()) if known.any() else np.nan,
                c4_n_source_edge=int(edge.sum()),
                p_in_s_window_frac=float(g["p_in_s_window"].astype(bool).mean()),
                multi_arrival_frac=float(g["multi_arrival"].astype(bool).mean()),
                n_p_only=int((g["has_p_pick"].astype(bool) & ~g["has_s_pick"].astype(bool)).sum()),
                n_s_only=int((~g["has_p_pick"].astype(bool) & g["has_s_pick"].astype(bool)).sum()))


# ── Aguilar reports ──────────────────────────────────────────────────────────

def multiplet_names(dataset: str, report_dirs=DEFAULT_REPORT_DIRS, download: bool = True) -> tuple:
    """(set of flagged trace names, info) for a dataset with a multiplet
    report; the cached file is used when present in any report dir, else
    downloaded once into the first dir (never when download is False)."""
    stem = next((s for s, d in lef.REPORT_STEM_TO_DATASET.items() if d == dataset), None)
    if stem is None:
        return set(), dict(status="no_report_for_dataset")
    path = next((lef._cache_path(d, stem) for d in report_dirs if lef._cache_path(d, stem).exists()), None)
    status = "cached"
    if path is None:
        if not download:
            return set(), dict(status="report_not_cached", stem=stem, searched=[str(d) for d in report_dirs])
        path = lef.download_multiplet_report(stem, cache_dir=str(report_dirs[0]))
        status = "downloaded"
        if path is None or not Path(path).exists():
            return set(), dict(status="download_failed", stem=stem)
    names = set(pd.read_csv(path, usecols=["trace_name"])["trace_name"].astype(str))
    return names, dict(status=status, path=str(path), sha256=_sha256(path), n_report_rows=len(names),
                       name_key_ambiguous=dataset in CHUNKED_NAME_DATASETS)


# ── sensitivity ──────────────────────────────────────────────────────────────

def paired_difference(values, clean, n_boot: int = N_BOOT, seed: int = BOOT_SEED, statistic=np.mean) -> dict:
    """statistic on all rows and on the clean subset, with a paired
    percentile bootstrap (the same resampled rows evaluate both) of
    all - clean. NaN intervals below five rows."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clean, dtype=bool)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    n, n_clean = len(v), int(c.sum())
    all_stat = float(statistic(v)) if n else np.nan
    clean_stat = float(statistic(v[c])) if n_clean else np.nan
    out = dict(n_all=n, n_clean=n_clean, value_all=all_stat, value_clean=clean_stat, diff=all_stat - clean_stat,
               diff_lo=np.nan, diff_hi=np.nan)
    if n < 5 or n_clean < 5:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vb, cb = v[idx], c[idx]
    if statistic is np.mean:
        a = vb.mean(axis=1)
        k = cb.sum(axis=1)
        b = np.where(k > 0, (vb * cb).sum(axis=1) / np.maximum(k, 1), np.nan)
    else:
        a = np.array([statistic(row) for row in vb])
        b = np.array([statistic(row[m]) if m.any() else np.nan for row, m in zip(vb, cb)])
    d = a - b
    d = d[np.isfinite(d)]
    if len(d):
        out["diff_lo"], out["diff_hi"] = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return out


CLEAN_RULES = {"s1_s2_c1": ("s1", "s2", "c1"), "s1_c1": ("s1", "c1")}


def clean_masks(flags: pd.DataFrame, rule: str) -> tuple:
    """(clean_p, clean_s) boolean Series under a CLEAN_RULES entry: the phase is
    labelled and none of the rule's screens flags it (C1 is per trace)."""
    screens = CLEAN_RULES[rule]
    out = []
    for phase, sfx in (("p", ""), ("s", "_s")):
        bad = pd.Series(False, index=flags.index)
        for scr in screens:
            bad |= flags[f"{scr}_flag{sfx}" if scr != "c1" else "c1_flag"].astype(bool)
        out.append(flags[f"has_label{sfx}"].astype(bool) & ~bad)
    return out[0], out[1]


def sensitivity_table(results: pd.DataFrame, flags: pd.DataFrame, weights=SENSITIVITY_WEIGHTS,
                      rules=tuple(CLEAN_RULES)) -> pd.DataFrame:
    """Per weight, clean rule, dataset (and `all`) and phase: recall at
    DETECT_THRESHOLD and the conditional MAE on all rows versus the clean
    rows, with the paired bootstrap on the difference."""
    rows = []
    for rule in rules:
        cp, cs = clean_masks(flags, rule)
        f = pd.DataFrame({"clean_p": cp, "clean_s": cs, "dataset": flags["dataset"], "trace_name": flags["trace_name"]})
        f = f.set_index(KEY)
        for w in weights:
            r = results[results["weight"] == w].merge(f, left_on=KEY, right_index=True, how="inner")
            for phase in ("p", "s"):
                has = r[r[f"{phase}_in_window"] >= 0]
                for ds, g in [("all", has)] + list(has.groupby("dataset", sort=True)):
                    detect = (g[f"{phase}_prob"] >= DETECT_THRESHOLD).to_numpy(dtype=float)
                    clean = g[f"clean_{phase}"].to_numpy(dtype=bool)
                    rec = paired_difference(detect, clean)
                    d = g[g[f"{phase}_prob"] >= DETECT_THRESHOLD]
                    mae = paired_difference(np.abs(d[f"{phase}_residual_s"].to_numpy(dtype=float)),
                                            d[f"clean_{phase}"].to_numpy(dtype=bool))
                    rows.append(dict(weight=w, clean_rule=rule, dataset=ds, phase=phase.upper(), metric="recall_t03", **rec))
                    rows.append(dict(weight=w, clean_rule=rule, dataset=ds, phase=phase.upper(), metric="mae_cond_s", **mae))
    return pd.DataFrame(rows)


def gap_table(results: pd.DataFrame, flags: pd.DataFrame, pairs=GAP_PAIRS, rules=tuple(CLEAN_RULES)) -> pd.DataFrame:
    """Paired (same traces) recall and conditional-MAE gap `a - b` of each
    weight pair on all rows and on the clean rows of each rule, per phase
    and distance bin, with percentile-bootstrap intervals over traces."""
    fl_index = flags.set_index(KEY).index
    cleans = {rule: [c.to_numpy() for c in clean_masks(flags, rule)] for rule in rules}
    rows = []
    for a, b in pairs:
        ra = results[results["weight"] == a].set_index(KEY)
        rb = results[results["weight"] == b].set_index(KEY)
        common = ra.index.intersection(rb.index)
        ra, rb = ra.loc[common], rb.loc[common]
        pos = fl_index.get_indexer(common)
        for phase, k in (("p", 0), ("s", 1)):
            has = (ra[f"{phase}_in_window"] >= 0).to_numpy()
            det_a = (ra[f"{phase}_prob"] >= DETECT_THRESHOLD).to_numpy(dtype=float)
            det_b = (rb[f"{phase}_prob"] >= DETECT_THRESHOLD).to_numpy(dtype=float)
            res_a = np.abs(ra[f"{phase}_residual_s"].to_numpy(dtype=float))
            res_b = np.abs(rb[f"{phase}_residual_s"].to_numpy(dtype=float))
            bins = ra["dist_bin"].fillna("unknown").astype(str).replace("nan", "unknown").to_numpy()
            subsets = [("all", np.ones(len(common), dtype=bool))]
            for rule in rules:
                c = np.where(pos >= 0, cleans[rule][k][np.maximum(pos, 0)], False)
                subsets.append((rule, c))
            for bin_name in ["all"] + sorted(set(bins[has])):
                sel_bin = has if bin_name == "all" else has & (bins == bin_name)
                for subset, cmask in subsets:
                    sel = sel_bin & cmask
                    if sel.sum() == 0:
                        continue
                    d_rec = det_a[sel] - det_b[sel]
                    lo, hi = bootstrap_ci(d_rec, n_boot=N_BOOT, seed=BOOT_SEED)
                    both = sel & (det_a > 0) & (det_b > 0) & np.isfinite(res_a) & np.isfinite(res_b)
                    d_mae = res_a[both] - res_b[both]
                    mlo, mhi = bootstrap_ci(d_mae, n_boot=N_BOOT, seed=BOOT_SEED) if len(d_mae) else (np.nan, np.nan)
                    rows.append(dict(pair=f"{a} - {b}", phase=phase.upper(), dist_bin=bin_name, subset=subset,
                                     n=int(sel.sum()), recall_gap=float(d_rec.mean()), recall_gap_lo=lo, recall_gap_hi=hi,
                                     n_both_detect=int(both.sum()),
                                     mae_cond_gap_s=float(d_mae.mean()) if len(d_mae) else np.nan,
                                     mae_cond_gap_lo=mlo, mae_cond_gap_hi=mhi))
    return pd.DataFrame(rows)


# ── review sheet ─────────────────────────────────────────────────────────────

def review_sheet(flags: pd.DataFrame, n: int = REVIEW_ROWS_PER_DATASET) -> pd.DataFrame:
    """Up to n rows per dataset: the S1 rows with the strongest consensus
    disagreement (score = |consensus residual| / (MAD + 0.05 s) x detecting
    share), P and S rows together, then S2 rows by SNR when fewer than n S1
    rows exist. Columns let a reviewer pull the waveform on the server."""
    cols = ["dataset", "trace_name", "phase", "reason", "label_sample_source", "label_s_window", "consensus_residual_s",
            "consensus_position_s", "consensus_source_position_s", "n_detect", "n_independent", "consensus_mad_s", "consensus_saturated", "snr_db",
            "s1_flag_p", "s1_flag_s", "s2_flag_p", "s2_flag_s", "c1_flag", "p_in_s_window", "multi_arrival",
            "aguilar_flagged", "score"]
    parts = []
    for phase, sfx in (("P", ""), ("S", "_s")):
        col = lambda c: f"{c}{sfx}"  # noqa: E731
        base = dict(dataset=flags["dataset"], trace_name=flags["trace_name"], phase=phase,
                    label_sample_source=flags["p_arrival_sample" if phase == "P" else "s_arrival_sample"],
                    label_s_window=flags[col("label_s")], consensus_residual_s=flags[col("consensus_residual_s")],
                    consensus_position_s=flags[col("consensus_position_s")],
                    consensus_source_position_s=flags[col("consensus_source_position_s")], n_detect=flags[col("n_detect")],
                    n_independent=flags[col("n_independent")], consensus_mad_s=flags[col("consensus_mad_s")],
                    consensus_saturated=flags[col("consensus_saturated")], snr_db=flags["snr_db"],
                    s1_flag_p=flags["s1_flag"], s1_flag_s=flags["s1_flag_s"], s2_flag_p=flags["s2_flag"],
                    s2_flag_s=flags["s2_flag_s"], c1_flag=flags["c1_flag"], p_in_s_window=flags["p_in_s_window"],
                    multi_arrival=flags["multi_arrival"], aguilar_flagged=flags["aguilar_flagged"])
        d = pd.DataFrame(base)
        s1 = flags[col("s1_flag")].astype(bool).to_numpy()
        s2 = flags[col("s2_flag")].astype(bool).to_numpy()
        share = (flags[col("n_detect")] / flags[col("n_independent")].replace(0, np.nan)).fillna(0)
        d["score"] = np.where(s1, flags[col("consensus_residual_s")].abs() / (flags[col("consensus_mad_s")] + 0.05) * share,
                              np.where(s2, flags["snr_db"] / 100.0, np.nan))
        d["reason"] = np.where(s1, "s1_consensus_offset", np.where(s2, "s2_consensus_miss", ""))
        parts.append(d[s1 | s2])
    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    if cand.empty:
        return pd.DataFrame(columns=cols)
    cand["_rank"] = (cand["reason"] != "s1_consensus_offset").astype(int)
    cand = cand.sort_values(["dataset", "_rank", "score"], ascending=[True, True, False])
    out = cand.groupby("dataset", sort=True).head(n).drop(columns="_rank").reset_index(drop=True)
    return out[cols]


# ── summary ──────────────────────────────────────────────────────────────────

def summarise_dataset(ds: str, m: pd.DataFrame, fl: pd.DataFrame, c1: dict, report_info: dict) -> dict:
    """One row per dataset: manifest counts and C4, S1/S2 rates with
    bootstrap intervals (P and S), edge and saturation shares, C1, the
    p_in_s_window cross-tabulation and the Aguilar cross-check."""
    s = dict(dataset=ds, n_manifest=int(len(m)), n_results=int(len(fl)))
    s.update(c4_manifest(m))
    s.update(c1)
    for phase, sfx in (("p", ""), ("s", "_s")):
        col = lambda c: f"{c}{sfx}"  # noqa: E731
        has = fl[col("has_label")].astype(bool)
        g = fl[has]
        n = int(len(g))
        frac, lo, hi = _frac_ci(g[col("s1_flag")]) if n else (np.nan, np.nan, np.nan)
        e_frac, e_lo, e_hi = _frac_ci(g[col("consensus_at_edge")]) if n else (np.nan, np.nan, np.nan)
        m_frac, m_lo, m_hi = _frac_ci(g[col("s2_flag")]) if n else (np.nan, np.nan, np.nan)
        raw = g[col("has_consensus")].astype(bool) & (g[col("consensus_residual_s")].abs() > S1_MIN_OFFSET_S)
        s.update({f"n_{phase}_labels": n,
                  f"{phase}_n_independent_median": float(g[col("n_independent")].median()) if n else np.nan,
                  f"{phase}_n_detect_median": float(g[col("n_detect")].median()) if n else np.nan,
                  f"{phase}_consensus_frac": float(g[col("has_consensus")].mean()) if n else np.nan,
                  f"s1_{phase}_raw_offset_frac": float(raw.mean()) if n else np.nan,
                  f"s1_{phase}_n_flag": int(g[col("s1_flag")].sum()), f"s1_{phase}_flag_frac": frac,
                  f"s1_{phase}_flag_lo": lo, f"s1_{phase}_flag_hi": hi,
                  f"s1_{phase}_n_edge": int(g[col("consensus_at_edge")].sum()), f"s1_{phase}_edge_frac": e_frac,
                  f"s1_{phase}_edge_lo": e_lo, f"s1_{phase}_edge_hi": e_hi,
                  f"s1_{phase}_n_saturated_flag": int((g[col("consensus_saturated")] & g[col("s1_flag")]).sum()),
                  f"s1_{phase}_flag_residual_median_s": float(g.loc[g[col("s1_flag")].astype(bool), col("consensus_residual_s")].median())
                  if g[col("s1_flag")].any() else np.nan,
                  f"s1_{phase}_flag_early_frac": float((g.loc[g[col("s1_flag")].astype(bool), col("consensus_residual_s")] < 0).mean())
                  if g[col("s1_flag")].any() else np.nan,
                  f"s2_{phase}_n_flag": int(g[col("s2_flag")].sum()), f"s2_{phase}_flag_frac": m_frac,
                  f"s2_{phase}_flag_lo": m_lo, f"s2_{phase}_flag_hi": m_hi,
                  f"s2_{phase}_n_eligible": int((g["snr_db"] > S2_MIN_SNR_DB).sum())})
        for name, sel in (("p_in_s_window", g["p_in_s_window"].astype(bool)), ("multi_arrival", g["multi_arrival"].astype(bool))):
            s[f"s1_{phase}_flag_frac_{name}"] = float(g.loc[sel, col("s1_flag")].mean()) if sel.any() else np.nan
            s[f"s1_{phase}_flag_frac_not_{name}"] = float(g.loc[~sel, col("s1_flag")].mean()) if (~sel).any() else np.nan
    any_flag = fl["s1_flag"] | fl["s1_flag_s"] | fl["s2_flag"] | fl["s2_flag_s"] | fl["c1_flag"]
    s.update(n_any_flag=int(any_flag.sum()), any_flag_frac=float(any_flag.mean()) if len(fl) else np.nan,
             n_clean_p=int(fl["clean_p"].sum()), n_clean_s=int((fl["clean_s"] & fl["has_label_s"]).sum()))
    s.update(aguilar_status=report_info.get("status", "no_report_for_dataset"),
             aguilar_name_key_ambiguous=bool(report_info.get("name_key_ambiguous", False)),
             aguilar_n_report_rows=int(report_info.get("n_report_rows", 0)),
             aguilar_n_manifest_in_report=int(m["aguilar_flagged"].sum()),
             aguilar_manifest_in_report_frac=float(m["aguilar_flagged"].mean()) if len(m) else np.nan)
    flagged = fl["aguilar_flagged"].astype(bool)
    for tag, sel in (("flagged", flagged), ("unflagged", ~flagged)):
        g = fl[sel]
        s[f"aguilar_{tag}_n"] = int(len(g))
        for phase, sfx in (("p", ""), ("s", "_s")):
            hp = g[g[f"has_label{sfx}"].astype(bool)]
            s[f"aguilar_{tag}_s1_{phase}_frac"] = float(hp[f"s1_flag{sfx}"].mean()) if len(hp) else np.nan
            s[f"aguilar_{tag}_s2_{phase}_frac"] = float(hp[f"s2_flag{sfx}"].mean()) if len(hp) else np.nan
    return s


def render_tables(summary: pd.DataFrame, sens: pd.DataFrame, gaps: pd.DataFrame, prov: dict) -> str:
    lines = ["# Benchmark label audit (41B): tables", "",
             f"*commit `{prov.get('git_commit', '')[:8]}`, created {prov.get('created_utc', '')}, "
             f"bootstrap n={N_BOOT} seed={BOOT_SEED}; S1: >= {S1_MIN_DETECT} independent weights, prob >= {DETECT_THRESHOLD}, "
             f"MAD <= {S1_MAX_MAD_S} s, offset > {S1_MIN_OFFSET_S} s, edge {EDGE_S} s of the source trace or the {WINDOW_S:.0f} s window; "
             f"S2: SNR > {S2_MIN_SNR_DB:.0f} dB and <= {S2_MAX_DETECT} detection. Percentages with 95 % intervals.*", ""]
    head = ["dataset", "rows (manifest / results)", "P labels", "S1 P raw %", "S1 P edge %", "S1 P flag %",
            "S1 P flag early %", "S2 P %", "S labels", "S1 S edge %", "S1 S flag %", "S1 S flag early %", "S2 S %",
            "C1 n fit", "C1 slope s/km", "Vp/Vs", "C1 hyp. slope", "Vp/Vs hyp.", "C1 flag % (hyp.)", "C1 flag % (epi.)",
            "C4 P mode share", "C4 source edge %", "p_in_s_window %", "S1 P in / not in S window %",
            "S1 S in / not in S window %", "Aguilar in report", "any flag %"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    g = lambda r, k: r.get(k, np.nan)  # noqa: E731
    for _, r in summary.iterrows():
        cells = [str(r["dataset"]), f"{int(r['n_manifest'])} / {int(r['n_results'])}", str(int(r["n_p_labels"])),
                 _pct(g(r, "s1_p_raw_offset_frac")),
                 _pct(g(r, "s1_p_edge_frac"), g(r, "s1_p_edge_lo"), g(r, "s1_p_edge_hi")),
                 _pct(g(r, "s1_p_flag_frac"), g(r, "s1_p_flag_lo"), g(r, "s1_p_flag_hi")),
                 _pct(g(r, "s1_p_flag_early_frac")),
                 _pct(g(r, "s2_p_flag_frac"), g(r, "s2_p_flag_lo"), g(r, "s2_p_flag_hi")),
                 str(int(r["n_s_labels"])),
                 _pct(g(r, "s1_s_edge_frac"), g(r, "s1_s_edge_lo"), g(r, "s1_s_edge_hi")),
                 _pct(g(r, "s1_s_flag_frac"), g(r, "s1_s_flag_lo"), g(r, "s1_s_flag_hi")),
                 _pct(g(r, "s1_s_flag_early_frac")),
                 _pct(g(r, "s2_s_flag_frac"), g(r, "s2_s_flag_lo"), g(r, "s2_s_flag_hi")),
                 str(int(g(r, "c1_n_fit")) if np.isfinite(g(r, "c1_n_fit")) else 0),
                 _num(g(r, "c1_slope_s_per_km"), "{:.4f}"), _num(g(r, "c1_implied_vp_vs")),
                 _num(g(r, "c1_hyp_slope_s_per_km"), "{:.4f}"), _num(g(r, "c1_hyp_implied_vp_vs")),
                 _pct(g(r, "c1_flag_frac"), g(r, "c1_flag_lo"), g(r, "c1_flag_hi")),
                 _pct(g(r, "c1_epi_flag_frac"), g(r, "c1_epi_flag_lo"), g(r, "c1_epi_flag_hi")),
                 _pct(g(r, "c4_p_sample_mode_share")), _pct(g(r, "c4_source_edge_frac")),
                 _pct(g(r, "p_in_s_window_frac")),
                 f"{_pct(g(r, 's1_p_flag_frac_p_in_s_window'))} / {_pct(g(r, 's1_p_flag_frac_not_p_in_s_window'))}",
                 f"{_pct(g(r, 's1_s_flag_frac_p_in_s_window'))} / {_pct(g(r, 's1_s_flag_frac_not_p_in_s_window'))}",
                 f"{int(g(r, 'aguilar_n_manifest_in_report'))} ({r['aguilar_status']}"
                 f"{', name-level only' if r['aguilar_name_key_ambiguous'] else ''})",
                 _pct(g(r, "any_flag_frac"))]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Sensitivity: all rows versus the clean rows (dataset `all`; per dataset in sensitivity.csv)", "",
              "| weight | clean rule | phase | metric | n all | n clean | all | clean | all - clean [95 %] |",
              "|---|---|---|---|--:|--:|--:|--:|--:|"]
    for _, r in sens[sens["dataset"] == "all"].iterrows():
        fmt = "{:.3f}"
        lines.append(f"| {r['weight']} | {r['clean_rule']} | {r['phase']} | {r['metric']} | {int(r['n_all'])} | {int(r['n_clean'])} | "
                     f"{_num(r['value_all'], fmt)} | {_num(r['value_clean'], fmt)} | {_num(r['diff'], '{:+.3f}')} "
                     f"[{_num(r['diff_lo'], '{:+.3f}')}, {_num(r['diff_hi'], '{:+.3f}')}] |")
    lines += ["", "## Paired gaps on the same traces, all rows versus clean rows", "",
              "| pair | phase | dist bin | subset | n | recall gap [95 %] | n both detect | cond. MAE gap s [95 %] |",
              "|---|---|---|---|--:|--:|--:|--:|"]
    for _, r in gaps.iterrows():
        lines.append(f"| {r['pair']} | {r['phase']} | {r['dist_bin']} | {r['subset']} | {int(r['n'])} | "
                     f"{_num(r['recall_gap'], '{:+.3f}')} [{_num(r['recall_gap_lo'], '{:+.3f}')}, {_num(r['recall_gap_hi'], '{:+.3f}')}] | "
                     f"{int(r['n_both_detect'])} | {_num(r['mae_cond_gap_s'], '{:+.3f}')} "
                     f"[{_num(r['mae_cond_gap_lo'], '{:+.3f}')}, {_num(r['mae_cond_gap_hi'], '{:+.3f}')}] |")
    return "\n".join(lines) + "\n"


# ── driver ───────────────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def constants() -> dict:
    return dict(WINDOW_S=WINDOW_S, RATE_HZ=RATE_HZ, SEARCH_S=SEARCH_S, DETECT_THRESHOLD=DETECT_THRESHOLD,
                S1_MIN_DETECT=S1_MIN_DETECT, S1_MAX_MAD_S=S1_MAX_MAD_S, S1_MIN_OFFSET_S=S1_MIN_OFFSET_S, EDGE_S=EDGE_S,
                SATURATION_TOL_S=SATURATION_TOL_S, S2_MIN_SNR_DB=S2_MIN_SNR_DB, S2_MAX_DETECT=S2_MAX_DETECT,
                REVIEW_ROWS_PER_DATASET=REVIEW_ROWS_PER_DATASET, SOURCE_EDGE_S=SOURCE_EDGE_S, N_BOOT=N_BOOT,
                BOOT_SEED=BOOT_SEED, VP_KM_S=VP_KM_S)


def run(benchmark, results_path, out_dir, report_dirs=DEFAULT_REPORT_DIRS, download: bool = True, argv=None,
        public_weights=PUBLIC_WEIGHTS, sensitivity_weights=SENSITIVITY_WEIGHTS) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = dict(audit_version=AUDIT_VERSION, created_utc=datetime.now(timezone.utc).isoformat(), git_commit=git_commit(),
                command=" ".join(argv) if argv else "", python=sys.version.split()[0], numpy=np.__version__,
                pandas=pd.__version__, constants=constants(), public_weights=list(public_weights),
                sensitivity_weights=list(sensitivity_weights),
                weight_trained_on={w: weight_trained_on(w) for w in public_weights},
                inputs={str(p): _sha256(p) for p in (benchmark, results_path)})
    log("reading inputs")
    m = load_manifest(benchmark)
    all_weights = list(dict.fromkeys(list(public_weights) + list(sensitivity_weights)))
    results = load_results(results_path, all_weights)
    present = sorted(results["weight"].unique())
    missing = [w for w in all_weights if w not in present]
    if missing:
        log(f"  weights absent from the results: {missing}")
    prov["weights_missing"] = missing
    prov["n_result_rows"] = int(len(results))
    public = [w for w in public_weights if w in present]
    traces = trace_table(results[results["weight"].isin(public)])
    indep = independence(traces, public)
    prov["n_traces"] = int(len(traces))
    log(f"  {len(traces)} traces, {len(public)} public weights")

    log("C1/C4 on the manifest")
    c1_rows, c1_per = c1_manifest(m)
    m = pd.concat([m, c1_rows], axis=1)
    coords = m.set_index(KEY)[["p_source_s", "s_source_s", "source_length_s"]].reindex(traces.index)
    traces = pd.concat([traces, coords], axis=1)

    log("S1/S2")
    pub = results[results["weight"].isin(public)]
    sp = screen_phase(pub, traces, indep, "p")
    ss = screen_phase(pub, traces, indep, "s")
    flags = pd.concat([traces, sp, ss], axis=1).reset_index()
    reports = {}
    m["aguilar_flagged"] = False
    for ds in sorted(m["dataset"].unique()):
        names, info = multiplet_names(ds, report_dirs, download=download)
        reports[ds] = info
        if names:
            sel = m["dataset"] == ds
            m.loc[sel, "aguilar_flagged"] = m.loc[sel, "trace_name"].astype(str).isin(names).to_numpy()
    prov["label_error_reports"] = reports
    keep = ["p_arrival_sample", "s_arrival_sample", "ts_tp_s", "distance_km", "depth_km", "magnitude", "has_p_pick",
            "has_s_pick", "evaluate_s", "p_in_s_window", "multi_arrival", "clipped_flag", "source_trace_samples",
            "notebook_rate_hz", "c1_residual_s", "c1_flag", "c1_epi_residual_s", "c1_epi_flag", "aguilar_flagged"]
    flags = flags.drop(columns=["p_source_s", "s_source_s", "source_length_s"])
    flags = flags.merge(m[KEY + keep], on=KEY, how="left")
    for c in ("p_in_s_window", "multi_arrival", "clipped_flag", "aguilar_flagged", "c1_flag", "has_p_pick", "has_s_pick"):
        flags[c] = flags[c].fillna(False).astype(bool)
    for c in ("c1_epi_flag",):
        flags[c] = flags[c].fillna(False).astype(bool)
    flags["clean_p"], flags["clean_s"] = clean_masks(flags, "s1_s2_c1")
    flags["clean_p_s1_c1"], flags["clean_s_s1_c1"] = clean_masks(flags, "s1_c1")

    log("summary")
    summaries = []
    for ds in sorted(m["dataset"].unique()):
        summaries.append(summarise_dataset(ds, m[m["dataset"] == ds], flags[flags["dataset"] == ds], c1_per.get(ds, {}),
                                           reports.get(ds, {})))
    summary = pd.DataFrame(summaries)

    log("sensitivity")
    sens_w = [w for w in sensitivity_weights if w in present]
    sens = sensitivity_table(results, flags, sens_w)
    pairs = [(a, b) for a, b in GAP_PAIRS if a in present and b in present]
    gaps = gap_table(results, flags, pairs)

    log("writing")
    flags.to_parquet(out_dir / "per_trace_flags.parquet", index=False)
    summary.to_csv(out_dir / "per_dataset_summary.csv", index=False)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    gaps.to_csv(out_dir / "sensitivity_gap.csv", index=False)
    sheet = review_sheet(flags)
    sheet.to_csv(out_dir / "review_sheet.csv", index=False)
    prov["outputs"] = dict(n_flag_rows=int(len(flags)), n_review_rows=int(len(sheet)),
                           totals={k: int(flags[k].sum()) for k in ("s1_flag", "s1_flag_s", "s2_flag", "s2_flag_s", "c1_flag",
                                                                   "consensus_at_edge", "consensus_at_edge_s", "aguilar_flagged")})
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    (out_dir / "tables.md").write_text(render_tables(summary, sens, gaps, prov))
    return dict(flags=flags, summary=summary, sensitivity=sens, gaps=gaps, review=sheet, provenance=prov)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=str(REPO_ROOT / "notebooks" / "benchmark_manifest.csv"))
    ap.add_argument("--results", default=str(REPO_ROOT / "notebooks" / "step3_results.parquet"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report-dirs", nargs="*", default=None,
                    help="Aguilar report caches; default data/labelerrors, ~/.cache/phasenet_retrain/label_errors")
    ap.add_argument("--no-download", action="store_true", help="never fetch a missing Aguilar report")
    a = ap.parse_args(argv)
    report_dirs = [Path(d) for d in a.report_dirs] if a.report_dirs else list(DEFAULT_REPORT_DIRS)
    out = run(a.benchmark, a.results, a.out_dir, report_dirs=report_dirs, download=not a.no_download, argv=argv)
    s = out["summary"]
    cols = ["dataset", "n_results", "s1_p_flag_frac", "s1_p_edge_frac", "s2_p_flag_frac", "s1_s_flag_frac", "c1_flag_frac"]
    print(s[[c for c in cols if c in s]].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
