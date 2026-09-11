#!/usr/bin/env python3
"""
continuous_scoring.py

Pick-level scoring engine for continuous waveform windows (issue #35,
checkpoint 35A). Replaces the notebook logic that heldout_testset_score.py
carried until 44A: one classify() at 0.02 followed by score filtering, a
proximity-based reference collapse, and a greedy nearest-first matcher.

What it does instead:

  * AnnotationStore keeps the raw P/S probability traces of every
    (sequence, window, station, model) once, with the time grid and a valid
    mask. Nothing is thresholded at storage time.
  * extract_picks applies the SeisBench production trigger rule at each
    threshold independently (obspy trigger_onset, off = on / 2, one pick per
    trigger at the argmax).
  * reference_picks deduplicates bulletin rows by (event, station, phase)
    identity; two events 0.3 s apart on one station stay two references.
  * match_picks is a maximum-cardinality one-to-one assignment with minimum
    total |residual| as the secondary objective.
  * score_window, aggregate, operating_points and matched_budget report
    attained operating points only; counts are summed over explicitly listed
    windows of one sequence and never interpolated.
  * common_support and failure_row make the coverage and failure policy
    explicit: a model failure removes that (window, station) from every
    compared model and is listed, never silently dropped.

Pure numpy/pandas/scipy/obspy. No torch or seisbench import: inference and
the model adapter live in the caller (scripts/heldout_testset_score.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset
from scipy.optimize import linear_sum_assignment

PHASES = ("P", "S")
COUNT_COLUMNS = ["n_reference", "n_reference_uncovered", "matched", "emitted", "unmatched_candidates"]
ROW_COLUMNS = ["model_id", "window_id", "phase", "threshold"] + COUNT_COLUMNS + [
    "recall", "residual_mae", "residual_median", "residual_p90"]
PICK_COLUMNS = ["pick_id", "station", "phase", "time", "score", "threshold", "i_on", "i_off", "i_peak"]
REFERENCE_COLUMNS = ["ref_id", "event", "station", "phase", "time", "tier", "time_weight", "n_rows"]
MATCH_COLUMNS = ["model_id", "window_id", "station", "phase", "threshold", "pick_id", "ref_id", "event", "residual"]
FAILURE_COLUMNS = ["model_id", "window_id", "station", "stage", "exception", "message"]
EXCLUDED_COLUMNS = ["window_id", "station", "missing_models"]
BUDGET_COLUMNS = ["phase", "model_id", "target_emitted", "threshold", "emitted", "matched", "n_reference",
                  "recall", "within_tolerance", "reason"]


# ── time helpers ─────────────────────────────────────────────────────────────

def to_timestamp(t) -> pd.Timestamp:
    """UTC pandas Timestamp from str, UTCDateTime, datetime, np.datetime64 or Timestamp."""
    if isinstance(t, UTCDateTime):
        return pd.Timestamp(t.ns, unit="ns", tz="UTC")
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_ns(values) -> np.ndarray:
    """Integer nanoseconds. Numbers are seconds; anything else is a UTC instant.

    Integer arithmetic keeps the tolerance boundary exact: |d| == tol is feasible
    without a float slack.
    """
    values = list(values) if not isinstance(values, (np.ndarray, pd.Series, pd.Index)) else values
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64)
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return np.round(arr.astype(float) * 1e9).astype(np.int64)
    if isinstance(values, (pd.Series, pd.Index)) and pd.api.types.is_datetime64_any_dtype(values):
        idx = pd.DatetimeIndex(values)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return idx.asi8.copy()
    return pd.DatetimeIndex([to_timestamp(v) for v in values]).asi8.copy()


def dedup_thresholds(thresholds) -> list:
    """Sorted unique float thresholds; the 0.3-twice defect of the old sweep cannot recur."""
    out = sorted({float(t) for t in thresholds})
    if any(t <= 0 for t in out):
        raise ValueError(f"Thresholds must be positive: {out}")
    return out


# ── annotations ──────────────────────────────────────────────────────────────

@dataclass
class Annotation:
    """One station's P and S probability traces on a common time grid."""
    start_time: pd.Timestamp
    rate: float
    P: np.ndarray
    S: np.ndarray
    valid: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.start_time = to_timestamp(self.start_time)
        self.rate = float(self.rate)
        self.P = np.asarray(self.P, dtype=np.float32)
        self.S = np.asarray(self.S, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        if not (len(self.P) == len(self.S) == len(self.valid)):
            raise ValueError("P, S and valid mask must have the same length")
        if self.rate <= 0:
            raise ValueError("Sampling rate must be positive")

    @property
    def n(self) -> int:
        return len(self.P)

    @property
    def end_time(self) -> pd.Timestamp:
        return self.start_time + pd.Timedelta(seconds=(self.n - 1) / self.rate)

    def prob(self, phase):
        return {"P": self.P, "S": self.S}[phase]

    def covered(self, times) -> np.ndarray:
        """True where the sample nearest each time lies on the grid and is valid."""
        t_ns = _to_ns(times)
        if len(t_ns) == 0:
            return np.zeros(0, dtype=bool)
        sec = (t_ns - self.start_time.value) / 1e9
        idx = np.round(sec * self.rate).astype(np.int64)
        inside = (idx >= 0) & (idx < self.n)
        out = np.zeros(len(idx), dtype=bool)
        out[inside] = self.valid[idx[inside]]
        return out


def valid_mask_from_stream(stream, start_time, rate, n, zero_run_s=1.0) -> np.ndarray:
    """Valid support of an annotation grid from the input stream.

    A grid sample is valid when every component id of the input stream has data
    there that is neither masked nor part of an exact-zero run of at least
    `zero_run_s` seconds. The held-out builder fills waveform gaps with zeros
    (audit 2026-09-10, finding 5), so zero runs are the only trace of a gap left
    in the stored MiniSEED. An empty stream gives no valid support.
    """
    n = int(n)
    if stream is None or len(stream) == 0:
        return np.zeros(n, dtype=bool)
    start = to_timestamp(start_time)
    valid = np.ones(n, dtype=bool)
    by_id = {}
    for tr in stream:
        by_id.setdefault(tr.id, []).append(tr)
    for traces in by_id.values():
        cov = np.zeros(n, dtype=bool)
        for tr in traces:
            data = tr.data
            good = ~np.ma.getmaskarray(data) if np.ma.isMaskedArray(data) else np.ones(len(data), dtype=bool)
            data = np.ma.getdata(data)
            if zero_run_s and len(data):
                min_run = max(1, int(round(zero_run_s * tr.stats.sampling_rate)))
                zero = np.asarray(data == 0)
                for a, b in _runs(zero):
                    if b - a >= min_run:
                        good[a:b] = False
            t0 = to_timestamp(tr.stats.starttime)
            for a, b in _runs(good):
                # trace samples [a, b) → grid indices whose time lies inside [t_a, t_(b-1)]
                ta = (t0.value - start.value) / 1e9 + a / tr.stats.sampling_rate
                tb = (t0.value - start.value) / 1e9 + (b - 1) / tr.stats.sampling_rate
                ia, ib = int(np.ceil(ta * rate - 1e-6)), int(np.floor(tb * rate + 1e-6))
                ia, ib = max(ia, 0), min(ib, n - 1)
                if ib >= ia:
                    cov[ia:ib + 1] = True
        valid &= cov
    return valid


def _runs(flags: np.ndarray):
    """[start, stop) index pairs of True runs."""
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return []
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def annotation_from_stream(annotations, stream=None, zero_run_s=1.0, meta=None) -> Annotation:
    """Annotation from a SeisBench-style annotate() output (traces named *_P and *_S).

    The P and S traces must share start time, rate and length; the valid mask
    is derived from the input `stream` on the annotation grid.
    """
    traces = {}
    for tr in annotations:
        label = str(tr.stats.channel).rsplit("_", 1)[-1]
        if label in PHASES:
            if label in traces:
                raise ValueError(f"More than one {label} annotation trace")
            traces[label] = tr
    missing = [p for p in PHASES if p not in traces]
    if missing:
        raise ValueError(f"Annotation stream lacks {missing} traces")
    p, s = traces["P"], traces["S"]
    same = (p.stats.starttime == s.stats.starttime and p.stats.sampling_rate == s.stats.sampling_rate
            and p.stats.npts == s.stats.npts)
    if not same:
        raise ValueError("P and S annotation traces are not on one time grid")
    rate, n = float(p.stats.sampling_rate), int(p.stats.npts)
    valid = valid_mask_from_stream(stream, p.stats.starttime, rate, n, zero_run_s=zero_run_s)
    return Annotation(p.stats.starttime, rate, p.data, s.data, valid, meta=dict(meta or {}))


class AnnotationStore:
    """Raw probability traces per (sequence key, window_id, station, model_id), as .npz plus an index parquet."""

    INDEX_COLUMNS = ["key", "window_id", "station", "model_id", "start_time", "rate", "n_samples", "n_valid", "path"]
    KEYS = ["key", "window_id", "station", "model_id"]

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.parquet"
        self._index = (pd.read_parquet(self.index_path) if self.index_path.exists()
                       else pd.DataFrame(columns=self.INDEX_COLUMNS))

    def _path(self, key, window_id, station, model_id) -> Path:
        return self.root / str(key) / str(window_id) / str(model_id) / f"{station}.npz"

    def has(self, key, window_id, station, model_id) -> bool:
        return self._path(key, window_id, station, model_id).exists()

    def put(self, key, window_id, station, model_id, annotation: Annotation) -> Path:
        path = self._path(key, window_id, station, model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, P=annotation.P, S=annotation.S, valid=annotation.valid,
            start_time=np.array(annotation.start_time.isoformat()), rate=np.array(annotation.rate),
            meta=np.array(json.dumps(annotation.meta, sort_keys=True, default=str)))
        row = dict(key=str(key), window_id=str(window_id), station=str(station), model_id=str(model_id),
                   start_time=annotation.start_time.isoformat(), rate=annotation.rate, n_samples=annotation.n,
                   n_valid=int(annotation.valid.sum()), path=str(path.relative_to(self.root)))
        idx = self._index
        same = np.ones(len(idx), dtype=bool)
        for k in self.KEYS:
            same &= (idx[k].astype(str) == row[k]).to_numpy() if len(idx) else same
        kept = idx[~same]
        new = pd.DataFrame([row])
        self._index = (pd.concat([kept, new], ignore_index=True) if len(kept) else new)[self.INDEX_COLUMNS]
        self._index.to_parquet(self.index_path, index=False)
        return path

    def get(self, key, window_id, station, model_id) -> Annotation:
        path = self._path(key, window_id, station, model_id)
        if not path.exists():
            raise FileNotFoundError(f"No annotation for {key}/{window_id}/{model_id}/{station}")
        with np.load(path) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            return Annotation(str(z["start_time"]), float(z["rate"]), z["P"], z["S"], z["valid"], meta=meta)

    def index(self) -> pd.DataFrame:
        return self._index.copy()

    def pairs(self, key, model_id) -> set:
        """(window_id, station) pairs annotated by one model for one sequence."""
        idx = self._index
        sub = idx[(idx["key"].astype(str) == str(key)) & (idx["model_id"].astype(str) == str(model_id))]
        return set(zip(sub["window_id"].astype(str), sub["station"].astype(str)))


# ── pick extraction ──────────────────────────────────────────────────────────

def extract_picks(prob, start_time, rate, threshold, valid_mask=None, phase="P", station="", id_prefix="") -> pd.DataFrame:
    """Picks from one probability trace at one threshold, by the SeisBench production rule.

    Reproduces seisbench.models.base.WaveformModel.picks_from_annotations
    (installed SeisBench base.py, lines 2494-2531):
      * triggers = obspy.signal.trigger.trigger_onset(trace.data, threshold, threshold / 2)   (line 2511)
      * for each trigger [s0, s1]: peak_value = max(data[s0:s1+1]), s_peak = s0 + argmax(data[s0:s1+1]),
        peak_time = starttime + times[s_peak]                                               (lines 2517-2519)
      * one Pick per trigger with that peak time and value                                  (lines 2521-2529)
    Trigger on/off use obspy's strict `>` comparisons; nothing here re-thresholds
    a pick list made at another threshold. Picks whose peak sample lies outside
    the valid support are dropped (a gap or padding cannot yield an emitted pick).
    """
    prob = np.asarray(prob, dtype=float)
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    start = to_timestamp(start_time)
    rate = float(rate)
    rows = []
    for s0, s1 in trigger_onset(prob, threshold, threshold / 2):
        s0, s1 = int(s0), int(s1)
        seg = prob[s0:s1 + 1]
        s_peak = s0 + int(np.argmax(seg))
        if valid_mask is not None and not bool(np.asarray(valid_mask)[s_peak]):
            continue
        rows.append(dict(
            pick_id=f"{id_prefix}{station}|{phase}|{threshold:g}|{s_peak}", station=station, phase=phase,
            time=start + pd.Timedelta(seconds=s_peak / rate), score=float(np.max(seg)), threshold=threshold,
            i_on=s0, i_off=s1, i_peak=s_peak))
    out = pd.DataFrame(rows, columns=PICK_COLUMNS)
    return out.astype({"time": "datetime64[ns, UTC]", "score": float, "threshold": float,
                       "i_on": np.int64, "i_off": np.int64, "i_peak": np.int64})


def extract_all(annotation: Annotation, thresholds, station="", id_prefix="") -> pd.DataFrame:
    """extract_picks for both phases at every (deduplicated) threshold of one annotation."""
    parts = [extract_picks(annotation.prob(phase), annotation.start_time, annotation.rate, thr,
                           valid_mask=annotation.valid, phase=phase, station=station, id_prefix=id_prefix)
             for phase in PHASES for thr in dedup_thresholds(thresholds)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PICK_COLUMNS)


# ── references ───────────────────────────────────────────────────────────────

def reference_picks(picks, stations, t0, t1, tiers=("manual",), require_reference_ok=True) -> pd.DataFrame:
    """Reference picks on `stations` inside [t0, t1], one row per (event, station, phase).

    Duplicate bulletin representations of one arrival (several agencies or
    methods) collapse to the row with the highest time_weight, then the
    earliest time. Different events are never merged, however close their
    arrivals. `tiers` selects the provenance tier by `mode` (None keeps every
    mode); the kept tier is returned so sensitivity analyses can split it.
    Rows without an event id cannot be identified and are kept as they are.
    """
    lo, hi = to_timestamp(t0), to_timestamp(t1)
    times = pd.to_datetime(picks["time"], utc=True)
    sel = (times >= lo) & (times <= hi) & picks["station"].isin(list(stations))
    if require_reference_ok and "reference_ok" in picks:
        sel &= picks["reference_ok"].astype(bool)
    mode = picks["mode"] if "mode" in picks else pd.Series("unknown", index=picks.index)
    if tiers is not None:
        sel &= mode.isin(list(tiers))
    sub = picks.loc[sel].copy()
    sub["time"] = times[sel]
    sub["tier"] = mode[sel].astype(str)
    if "time_weight" not in sub:
        sub["time_weight"] = np.nan
    sub["time_weight"] = pd.to_numeric(sub["time_weight"], errors="coerce")
    event = sub["event"] if "event" in sub else pd.Series(np.nan, index=sub.index)
    no_id = event.isna()
    sub["event"] = event.astype(object)
    sub.loc[no_id, "event"] = [f"noevent:{i}" for i in sub.index[no_id]]
    sub["_w"] = sub["time_weight"].fillna(-np.inf)
    sub = sub.sort_values(["event", "station", "phase", "_w", "time"], ascending=[True, True, True, False, True])
    ident = ["event", "station", "phase"]
    n_rows = sub.groupby(ident).size().rename("n_rows")
    out = sub.drop_duplicates(ident, keep="first").merge(n_rows, left_on=ident, right_index=True)
    out["ref_id"] = out["event"].astype(str) + "|" + out["station"].astype(str) + "|" + out["phase"].astype(str)
    cols = REFERENCE_COLUMNS + [c for c in ("agency", "source") if c in out]
    return out[cols].sort_values(["time", "station", "phase"]).reset_index(drop=True)


# ── matching ─────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    pairs: list                   # (reference index, candidate index, residual = candidate - reference, seconds)
    unmatched_reference: list
    unmatched_candidate: list

    @property
    def residuals(self) -> np.ndarray:
        return np.array([p[2] for p in self.pairs], dtype=float)


def match_picks(reference_times, candidate_times, tol) -> MatchResult:
    """Maximum-cardinality one-to-one matching, minimum total |residual| second.

    Cost is |candidate - reference| for pairs within `tol` (|d| == tol is
    feasible) and a constant larger than any sum of feasible costs otherwise,
    solved with scipy.optimize.linear_sum_assignment; infeasible assignments
    are then discarded. Times are seconds (numbers) or UTC instants.
    """
    r_ns, c_ns = _to_ns(reference_times), _to_ns(candidate_times)
    tol_ns = int(round(float(tol) * 1e9))
    if tol_ns < 0:
        raise ValueError("tol must be non-negative")
    if len(r_ns) == 0 or len(c_ns) == 0:
        return MatchResult([], list(range(len(r_ns))), list(range(len(c_ns))))
    d_ns = c_ns[None, :] - r_ns[:, None]
    feasible = np.abs(d_ns) <= tol_ns
    d_s = d_ns / 1e9
    big = (min(len(r_ns), len(c_ns)) + 1) * float(tol) + 1.0
    cost = np.where(feasible, np.abs(d_s), big)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(d_s[i, j])) for i, j in zip(rows, cols) if feasible[i, j]]
    used_r = {i for i, _, _ in pairs}
    used_c = {j for _, j, _ in pairs}
    return MatchResult(pairs, [i for i in range(len(r_ns)) if i not in used_r],
                       [j for j in range(len(c_ns)) if j not in used_c])


# ── scoring ──────────────────────────────────────────────────────────────────

def _residual_stats(res: np.ndarray) -> dict:
    if len(res) == 0:
        return dict(residual_mae=np.nan, residual_median=np.nan, residual_p90=np.nan)
    a = np.abs(res)
    return dict(residual_mae=float(a.mean()), residual_median=float(np.median(res)),
                residual_p90=float(np.percentile(a, 90)))


def score_window(reference, picks, support, *, window_id, model_id, thresholds, tol, phases=PHASES):
    """Rows per (model_id, phase, threshold, window_id) and the matched-pick table.

    `reference` is a reference_picks frame, `picks` an extract_picks frame for
    this model and window at every threshold, `support` {station: Annotation}
    for the (window, station) pairs under comparison. A reference on a station
    outside `support`, or at a time outside its valid mask, is uncovered: it
    leaves the recall denominator and is counted in n_reference_uncovered.
    Candidates on stations outside `support` are not counted. A window with
    no covered reference has recall NaN and its emitted picks still counted.
    """
    thresholds = dedup_thresholds(thresholds)
    ref = reference.reset_index(drop=True).copy()
    covered = np.zeros(len(ref), dtype=bool)
    for sta, g in ref.groupby("station"):
        if sta in support:
            covered[g.index.to_numpy()] = support[sta].covered(g["time"])
    ref["covered"] = covered
    picks = picks[picks["station"].isin(list(support))] if len(picks) else picks
    rows, matches = [], []
    for phase in phases:
        r_all = ref[ref["phase"] == phase]
        r_cov = r_all[r_all["covered"]]
        n_unc = int((~r_all["covered"]).sum())
        for thr in thresholds:
            c = picks[(picks["phase"] == phase) & (picks["threshold"] == thr)] if len(picks) else picks
            residuals = []
            for sta in sorted(set(r_cov["station"]) | set(c["station"])):
                rs = r_cov[r_cov["station"] == sta].reset_index(drop=True)
                cs = c[c["station"] == sta].reset_index(drop=True)
                m = match_picks(rs["time"], cs["time"], tol)
                for i, j, d in m.pairs:
                    matches.append(dict(model_id=model_id, window_id=window_id, station=sta, phase=phase, threshold=thr,
                                        pick_id=cs.at[j, "pick_id"], ref_id=rs.at[i, "ref_id"], event=rs.at[i, "event"],
                                        residual=d))
                    residuals.append(d)
            n_ref, n_m, emitted = len(r_cov), len(residuals), int(len(c))
            rows.append(dict(model_id=model_id, window_id=window_id, phase=phase, threshold=thr,
                             n_reference=n_ref, n_reference_uncovered=n_unc, matched=n_m, emitted=emitted,
                             unmatched_candidates=emitted - n_m, recall=(n_m / n_ref if n_ref else np.nan),
                             **_residual_stats(np.array(residuals))))
    return pd.DataFrame(rows, columns=ROW_COLUMNS), pd.DataFrame(matches, columns=MATCH_COLUMNS)


def aggregate(rows, window_ids, by=("model_id", "phase", "threshold"), matches=None) -> pd.DataFrame:
    """Sum counts over the listed window ids and recompute recall.

    The listed windows must all exist in `rows` and belong to one sequence
    (`key` column, when present). Per-window rows are left untouched; the
    result carries window_id = "+".join(window_ids) and n_windows. Residual MAE
    is recomputed exactly from the per-window MAE and matched counts; median
    and p90 need the residuals themselves and are NaN unless `matches` (the
    score_window match table) is given.
    """
    window_ids = [str(w) for w in window_ids]
    if not window_ids:
        raise ValueError("aggregate needs an explicit list of window ids")
    present = set(rows["window_id"].astype(str)) if len(rows) else set()
    missing = sorted(set(window_ids) - present)
    if missing:
        raise ValueError(f"Windows not in rows: {missing}")
    sub = rows[rows["window_id"].astype(str).isin(window_ids)]
    if "key" in sub and sub["key"].nunique() > 1:
        raise ValueError(f"Windows span more than one sequence: {sorted(sub['key'].unique())}")
    by = list(by)
    agg = sub.groupby(by, dropna=False, sort=True)[COUNT_COLUMNS].sum().reset_index()
    agg["n_windows"] = sub.groupby(by, dropna=False, sort=True)["window_id"].nunique().to_numpy()
    agg["recall"] = np.where(agg["n_reference"] > 0, agg["matched"] / agg["n_reference"].replace(0, np.nan), np.nan)
    weighted = (sub["residual_mae"].fillna(0) * sub["matched"]).groupby([sub[b] for b in by], dropna=False, sort=True).sum()
    agg["residual_mae"] = np.where(agg["matched"] > 0, weighted.to_numpy() / agg["matched"].replace(0, np.nan), np.nan)
    agg["residual_median"] = np.nan
    agg["residual_p90"] = np.nan
    if matches is not None and len(matches):
        m = matches[matches["window_id"].astype(str).isin(window_ids)]
        stats = m.groupby(by, dropna=False)["residual"].agg(
            residual_median=lambda r: float(np.median(r)), residual_p90=lambda r: float(np.percentile(np.abs(r), 90)))
        agg = agg.drop(columns=["residual_median", "residual_p90"]).merge(stats, how="left", left_on=by, right_index=True)
    agg["window_id"] = "+".join(window_ids)
    if "key" in sub and len(sub):
        agg["key"] = sub["key"].iloc[0]
    for col in ("model", "access_id"):
        if col in sub and sub.groupby(by)[col].nunique().max() == 1:
            agg[col] = sub.groupby(by, sort=True)[col].first().to_numpy()
    return agg[ROW_COLUMNS + ["n_windows"] + [c for c in ("key", "model", "access_id") if c in agg]]


def operating_points(rows) -> pd.DataFrame:
    """Attained (threshold, emitted, recall) per model and phase for one scope; no interpolation.

    `rows` must come from one scope (one window id, or one aggregate); a duplicate
    (model_id, phase, threshold) is an error rather than a silently dropped row.
    """
    if len(rows) == 0:
        return pd.DataFrame(columns=["model_id", "phase", "threshold", "emitted", "matched", "n_reference", "recall"])
    if rows["window_id"].nunique() > 1:
        raise ValueError("operating_points needs rows of one window or one aggregate; call aggregate first")
    keys = ["model_id", "phase", "threshold"]
    if rows.duplicated(keys).any():
        raise ValueError("Duplicate (model_id, phase, threshold) rows in one scope")
    return rows[keys + ["emitted", "matched", "n_reference", "recall"]].sort_values(keys).reset_index(drop=True)


def matched_budget(rows, target_emitted=None, reference_model=None, reference_threshold=None, tolerance=0.10) -> pd.DataFrame:
    """For each model and phase, the attained threshold whose emitted count is closest to the target.

    The target per phase is `target_emitted` (a number or {phase: number}) or the
    emitted count of `reference_model` at `reference_threshold`. A model whose
    closest attained point is farther than `tolerance` (relative) from the
    target gets threshold/emitted/recall None and a reason; a target of zero
    requires zero emitted. Ties go to the threshold nearest `reference_threshold`
    when one is given, else to the lower threshold. No interpolation.
    """
    pts = operating_points(rows)
    if target_emitted is None and reference_model is None:
        raise ValueError("matched_budget needs target_emitted or reference_model")
    out = []
    for phase, g in pts.groupby("phase", sort=True):
        if target_emitted is not None:
            target = target_emitted[phase] if isinstance(target_emitted, dict) else target_emitted
        else:
            r = g[(g["model_id"] == reference_model) & np.isclose(g["threshold"], float(reference_threshold))]
            if len(r) != 1:
                raise ValueError(f"{reference_model} has no single attained point at {reference_threshold} for {phase}")
            target = int(r["emitted"].iloc[0])
        target = float(target)
        slack = tolerance * target
        for model_id, gm in g.groupby("model_id", sort=True):
            tie = (gm["threshold"] - float(reference_threshold)).abs() if reference_threshold is not None else gm["threshold"]
            gm = gm.assign(dist=(gm["emitted"] - target).abs(), tie=tie).sort_values(["dist", "tie", "threshold"])
            best = gm.iloc[0]
            row = dict(phase=phase, model_id=model_id, target_emitted=target)
            if best["dist"] <= slack:
                row.update(threshold=float(best["threshold"]), emitted=int(best["emitted"]), matched=int(best["matched"]),
                           n_reference=int(best["n_reference"]), recall=best["recall"], within_tolerance=True, reason="")
            else:
                row.update(threshold=None, emitted=None, matched=None, n_reference=None, recall=None, within_tolerance=False,
                           reason=(f"no attained point within {tolerance:.0%} of target {target:g}: closest emitted "
                                   f"{int(best['emitted'])} at threshold {best['threshold']:g}"))
            out.append(row)
    return pd.DataFrame(out, columns=BUDGET_COLUMNS)


# ── failure and coverage policy ──────────────────────────────────────────────

def failure_row(model_id, window_id, station, stage, exc) -> dict:
    return dict(model_id=model_id, window_id=window_id, station=station, stage=stage,
                exception=type(exc).__name__, message=str(exc))


def common_support(annotated: dict):
    """(window, station) pairs annotated by every compared model, and the excluded pairs.

    `annotated` maps model_id to the set of (window_id, station) pairs it
    produced. Comparison rows are computed on the common set only; each
    excluded pair is listed with the models that lack it, so a failure removes
    the pair from every model rather than improving the model that failed.
    """
    sets = {m: set(map(tuple, s)) for m, s in annotated.items()}
    union = set().union(*sets.values()) if sets else set()
    common = set.intersection(*sets.values()) if sets else set()
    excluded = [dict(window_id=w, station=s, missing_models=sorted(m for m, ps in sets.items() if (w, s) not in ps))
                for w, s in sorted(union - common)]
    return common, pd.DataFrame(excluded, columns=EXCLUDED_COLUMNS)
