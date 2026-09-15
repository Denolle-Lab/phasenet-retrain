#!/usr/bin/env python3
"""
audit_source_labels.py

Model-independent label audit of the curated training sources (issue #41,
checkpoint 41B). No model is run. Every check is a pure function on numpy
arrays; the two readers produce one common row structure (LabelRow) from
either a built held-out case (analyst picks, the calibration of the checks)
or a SeisBench source read through the loader's own readers
(`manifest_dataset._fetch_sbd`, `SingleHDF5Reader`, `ChunkedHDF5Reader`) at
the stored rate.

Checks (rules and tolerances in the docstrings; docs/2026-09-15_41b_label_audit.md):

  C1  S-P consistency        Theil-Sen fit of ts - tp against distance per source
  C2  P/S onset vs energy    Maeda AIC onset around the labelled time; testable
                             only when the post/pre RMS ratio exceeds 2; late
                             labels flag, emergent onsets are reported
  C3  component sanity       Z/H energy after P versus after S (source-level)
  C4  pick placement         P position along the trace; edge picks; the modal sample
  C5  cross-source duplicates metadata only: origin coincidence + station code
  C6  unlabelled arrivals    STA/LTA screen on every row: triggers away from every
                             labelled arrival; Aguilar-flagged rows also classified

C2 is asymmetric by default: a label is flagged when the energy onset
arrives more than 0.5 s before it (a late label); an analyst pick earlier
than the energy rise (an emergent onset) is reported as `c2_emergent` and
never flagged. A late P is cross-tabulated with the C6 screen:
`c2_late_kind` is `unlabelled_earlier_event` when an unexplained trigger
precedes the label (keep the row, add the arrival) and `suspect_pick`
otherwise. Per-row outputs carry `suggested_tier` (manual, or unknown when
C1, C4 or a suspect late P flag it: the 41A tier that masks rather than
supervises) and `suggested_extra_arrival_s` (JSON list of trigger times) so
the 40A builder can add them as `automatic`-tier arrivals, which mask their
neighbourhood under label_targets.MASKED, instead of dropping the row.

Torch-free and SeisBench-free at import; the SeisBench reader imports
seisbench, h5py and manifest_dataset (torch) only when constructed.

    python scripts/audit_source_labels.py heldout --keys samos_2020 adriatic_2022 --out-dir data/label_audit/heldout --report
    python scripts/audit_source_labels.py seisbench --sources instancecounts ethz --sample 5000 --seed 0 \\
        --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/seisbench --report
    python scripts/audit_source_labels.py duplicates --sources stead instancecounts --cache-root $SEISBENCH_CACHE_ROOT \\
        --out-dir data/label_audit/seisbench
    python scripts/audit_source_labels.py report --out-dir data/label_audit/heldout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import butter, sosfilt
from scipy.stats import theilslopes

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import exclusion_bundle as eb  # noqa: E402  (pure pandas)
import heldout_sequences as hs  # noqa: E402  (pure pandas)
import label_error_filter as lef  # noqa: E402  (pure pandas)
from metrics import bootstrap_ci  # noqa: E402

AUDIT_VERSION = "41b-v1"
N_BOOT = 1000
BOOT_SEED = 0

# Band for every energy measure: 1 Hz to min(20, 0.4 * Nyquist) Hz, causal 2nd-order Butterworth.
BAND_LOW_HZ = 1.0
BAND_HIGH_HZ = 20.0
BAND_NYQUIST_FRACTION = 0.4

# C1
C1_MAD_FACTOR = 4.0
C1_MIN_RESIDUAL_S = 1.0
C1_MIN_ROWS = 5
VP_KM_S = 6.0
# C2
C2_HALFWIDTH_P_S = 3.0
C2_HALFWIDTH_S_S = 4.0
C2_TOL_S = 0.5
C2_RULES = ("asymmetric", "symmetric")
C2_RULE = "asymmetric"      # flag only late labels (energy more than C2_TOL_S before the label)
C2_LATE_KINDS = ("unlabelled_earlier_event", "suspect_pick")
C2_RMS_WINDOW_S = 1.0
C2_MIN_RMS_RATIO = 2.0
C2_MIN_SIDE_S = 0.25
# C3
C3_WINDOW_S = 1.0
C3_WARN_FRACTION = 0.5
# C4
C4_EDGE_S = 1.0
# C5
C5_TOL_S = 0.2
C5_TIME_TOL_S = 2.0
C5_DIST_TOL_DEG = 0.1
# C6
C6_STA_S = 0.5
C6_LTA_S = 10.0
C6_ON = 4.0
C6_OFF = 1.5
C6_NEAR_S = 1.0
C6_S_CODA_S = 3.0
C6_CLASSES = ("second_event", "wrong_first_pick", "no_detection", "single_detection", "not_testable")

HELDOUT_PRE_S = 30.0
HELDOUT_LENGTH_S = 120.0
REVIEW_SHEET_ROWS = 50
STRATUM_FLOOR = 50
DISTANCE_BINS = ("local", "regional", "teleseismic", "unknown")

P_STATUS_COLUMNS = ["trace_p_status", "trace_P_status", "trace_P_arrival_status", "trace_p_arrival_status",
                    "trace_Pg_status", "trace_Pn_status", "p_status"]
S_STATUS_COLUMNS = ["trace_s_status", "trace_S_status", "trace_S_arrival_status", "trace_s_arrival_status",
                    "trace_Sg_status", "trace_Sn_status", "s_status"]
DEFAULT_REPORT_DIRS = (REPO_ROOT / "data" / "labelerrors",
                       Path(os.path.expanduser("~/.cache/phasenet_retrain/label_errors")))
# Direct-route layouts of manifest_dataset (prefix of the waveform files); the path is cache_root/datasets/<source>.
CHUNKED_PREFIX = {"mlaapde": "waveforms_", "cwa": "waveforms_", "aq2009gm": "waveforms", "obs": "waveforms"}
SINGLE_HDF5 = ("pisdl", "meier2019jgr", "ross2018gpd")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── the common row ───────────────────────────────────────────────────────────

@dataclass
class LabelRow:
    """One labelled trace. Times are seconds from the first sample of `waveform`.

    waveform   (3, n) float32, ZNE, at the stored rate `rate_hz`
    p_s, s_s   labelled arrival times or None
    distance_km None when the source has no distance
    p_status, s_status the source's pick status text, "unknown" when not exposed
    flagged_multiplet  named in the Aguilar multiplet report for the source
    p_sample   the source's own P index when it has one (C4 modal sample)
    """
    source: str
    trace_id: str
    station: str
    event_id: object
    waveform: np.ndarray
    rate_hz: float
    p_s: object = None
    s_s: object = None
    distance_km: object = None
    p_status: str = "unknown"
    s_status: str = "unknown"
    flagged_multiplet: bool = False
    component_mask: tuple = (True, True, True)
    p_sample: object = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.waveform = np.asarray(self.waveform, dtype=np.float32)
        if self.waveform.ndim != 2 or self.waveform.shape[0] != 3:
            raise ValueError(f"waveform must be (3, n); got {self.waveform.shape}")
        if not (np.isfinite(self.rate_hz) and self.rate_hz > 0):
            raise ValueError("rate_hz must be positive and finite")
        self.p_s = _opt_float(self.p_s)
        self.s_s = _opt_float(self.s_s)
        self.distance_km = _opt_float(self.distance_km)
        self.component_mask = tuple(bool(x) for x in self.component_mask)

    @property
    def n_samples(self) -> int:
        return int(self.waveform.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate_hz


def _opt_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(value) else value


# ── signal helpers ───────────────────────────────────────────────────────────

def band_corners(rate_hz: float) -> tuple:
    """(low, high) Hz: 1 Hz to min(20 Hz, 0.4 * Nyquist). Rates below 5 Hz have no band."""
    high = min(BAND_HIGH_HZ, BAND_NYQUIST_FRACTION * rate_hz / 2.0)
    if high <= BAND_LOW_HZ:
        raise ValueError(f"No usable band at {rate_hz} Hz")
    return BAND_LOW_HZ, high


def bandpass(x, rate_hz: float) -> np.ndarray:
    """Demean, then a causal 2nd-order Butterworth band-pass between band_corners(rate).

    Causal (sosfilt, not filtfilt) so that filter ringing never precedes an
    onset; the group delay biases AIC onsets late by at most one or two
    samples at the upper corner (0.05 s at 20 Hz on the synthetic set).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if not np.any(x):
        return x
    low, high = band_corners(rate_hz)
    sos = butter(2, [low, high], btype="band", fs=rate_hz, output="sos")
    return sosfilt(sos, x)


def aic_curve(x, return_variances: bool = False):
    """Maeda (1985) AIC for a split after k samples, k = 1 .. N-1:
    AIC(k) = k ln var(x[:k]) + (N - k) ln var(x[k:]), via cumulative sums.
    Variances are floored at 1e-12 of the total variance so a constant
    segment (zero padding) gives a finite value. With return_variances the
    left and right segment variances come back too."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        raise ValueError("AIC needs at least three samples")
    k = np.arange(1, n, dtype=np.float64)
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    left = c2[:-1] / k - (c1[:-1] / k) ** 2
    r = n - k
    right = (c2[-1] - c2[:-1]) / r - ((c1[-1] - c1[:-1]) / r) ** 2
    floor = 1e-12 * max(float(np.var(x)), 1e-30)
    aic = k * np.log(np.maximum(left, floor)) + r * np.log(np.maximum(right, floor))
    return (aic, left, right) if return_variances else aic


def aic_onset(x, guard: float = 0.02) -> int:
    """Sample index of the onset: the first sample of the second segment at the
    AIC minimum over the splits where the variance after the split exceeds
    the variance before it (an energy increase; the decay of a short wavelet
    inside the window is a variance drop and is not an onset), the outer
    `guard` fraction of the window excluded because the one-sided variance
    estimates are unstable there. Falls back to the unconstrained minimum
    when no split shows an increase."""
    aic, left, right = aic_curve(x, return_variances=True)
    m = len(aic)
    g = max(int(guard * m), 1)
    lo, hi = g, max(m - g, g + 1)
    window = aic[lo:hi].copy()
    increase = right[lo:hi] > left[lo:hi]
    if increase.any():
        window[~increase] = np.inf
    return int(lo + np.argmin(window)) + 1


def _c2_empty() -> dict:
    return dict(onset_s=np.nan, residual_s=np.nan, rms_ratio=np.nan, testable=False, flag=False, late=False,
                emergent=False)


def energy_onset(series, amplitude, rate_hz: float, t_label: float, halfwidth_s: float, *,
                 t_min=None, t_max=None, rms_window_s: float = C2_RMS_WINDOW_S,
                 min_ratio: float = C2_MIN_RMS_RATIO, tol_s: float = C2_TOL_S, rule: str = C2_RULE) -> dict:
    """C2 rule. AIC onset of `series` inside [t_label - halfwidth_s, t_label + halfwidth_s],
    clipped to the trace and to [t_min, t_max] when given (the callers pass
    the midpoint between the labelled P and S so the P window never reaches
    the S and the S window never reaches the P); residual_s = onset - t_label.
    The row is testable when the RMS of `amplitude` over rms_window_s after
    the onset exceeds min_ratio times the RMS over rms_window_s before it and
    both sides of the window hold at least C2_MIN_SIDE_S of samples; a row
    that is not testable is never flagged.
    late = testable and residual_s < -tol_s (energy arrives more than tol_s
    before the label: the label misses an onset); emergent = testable and
    residual_s > tol_s (the analyst picked earlier than the energy rise, the
    usual emergent onset; reported, never a flag by default).
    flag = late under rule "asymmetric" (default), late or emergent under "symmetric"."""
    if rule not in C2_RULES:
        raise ValueError(f"Unknown C2 rule {rule!r}; expected one of {C2_RULES}")
    out = _c2_empty()
    series = np.asarray(series, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    n = len(series)
    lo_s, hi_s = t_label - halfwidth_s, t_label + halfwidth_s
    if t_min is not None:
        lo_s = max(lo_s, float(t_min))
    if t_max is not None:
        hi_s = min(hi_s, float(t_max))
    i0 = max(int(round(lo_s * rate_hz)), 0)
    i1 = min(int(round(hi_s * rate_hz)), n)
    min_side = max(int(C2_MIN_SIDE_S * rate_hz), 2)
    if i1 - i0 < 2 * min_side + 2:
        return out
    j = i0 + aic_onset(series[i0:i1])
    m = max(int(round(rms_window_s * rate_hz)), 1)
    pre, post = amplitude[max(j - m, 0):j], amplitude[j:j + m]
    if len(pre) < min_side or len(post) < min_side:
        return out
    rms_pre, rms_post = float(np.sqrt(np.mean(pre ** 2))), float(np.sqrt(np.mean(post ** 2)))
    if rms_pre > 0:
        ratio = rms_post / rms_pre
    else:
        ratio = np.inf if rms_post > 0 else np.nan
    out["onset_s"] = j / rate_hz
    out["residual_s"] = j / rate_hz - t_label
    out["rms_ratio"] = ratio
    out["testable"] = bool(np.isfinite(ratio) or ratio == np.inf) and bool(ratio > min_ratio)
    out["late"] = bool(out["testable"] and out["residual_s"] < -tol_s)
    out["emergent"] = bool(out["testable"] and out["residual_s"] > tol_s)
    out["flag"] = bool(out["late"] or (rule == "symmetric" and out["emergent"]))
    return out


def _midpoint(tp, ts):
    return None if tp is None or ts is None else 0.5 * (float(tp) + float(ts))


def check_p_onset(waveform, rate_hz: float, tp, component_mask=(True, True, True), ts=None,
                  rule: str = C2_RULE) -> dict:
    """C2 for P: AIC on the band-passed vertical channel, W = C2_HALFWIDTH_P_S,
    the window cut at the P-S midpoint when S is labelled."""
    if tp is None or not component_mask[0]:
        return _c2_empty()
    z = bandpass(waveform[0], rate_hz)
    return energy_onset(z, z, rate_hz, float(tp), C2_HALFWIDTH_P_S, t_max=_midpoint(tp, ts), rule=rule)


def check_s_onset(waveform, rate_hz: float, ts, component_mask=(True, True, True), tp=None,
                  rule: str = C2_RULE) -> dict:
    """C2 for S: AIC on the horizontal energy N^2 + E^2 of the band-passed
    channels (the available ones), W = C2_HALFWIDTH_S_S, the window starting
    no earlier than the P-S midpoint when P is labelled (at stations a few
    km away the P lies inside +-4 s of the S and would capture the minimum);
    the RMS ratio is that of the horizontal amplitude sqrt(N^2 + E^2)."""
    if ts is None or not (component_mask[1] or component_mask[2]):
        return _c2_empty()
    energy = np.zeros(waveform.shape[1], dtype=np.float64)
    for i in (1, 2):
        if component_mask[i]:
            energy += bandpass(waveform[i], rate_hz) ** 2
    return energy_onset(energy, np.sqrt(energy), rate_hz, float(ts), C2_HALFWIDTH_S_S, t_min=_midpoint(tp, ts),
                        rule=rule)


def late_kind(late: bool, n_extra_before: int) -> str:
    """Cross-tabulation of a late label with the C6 screen: a late label with an
    unexplained trigger before it is an unlabelled earlier event (the
    multi-event case: keep the row, add the extra arrival); a late label with
    no earlier trigger is a suspect pick. Empty when the label is not late."""
    if not late:
        return ""
    return "unlabelled_earlier_event" if n_extra_before >= 1 else "suspect_pick"


def zh_ratio(waveform, rate_hz: float, t, window_s: float = C3_WINDOW_S, component_mask=(True, True, True)) -> float:
    """C3 measure: vertical energy over horizontal energy (N^2 + E^2) of the
    band-passed channels in [t, t + window_s]. NaN when t is None, the window
    is empty, a component family is missing or the horizontal energy is zero."""
    if t is None or not component_mask[0] or not (component_mask[1] or component_mask[2]):
        return np.nan
    i0 = max(int(round(float(t) * rate_hz)), 0)
    i1 = min(i0 + max(int(round(window_s * rate_hz)), 1), waveform.shape[1])
    if i1 <= i0:
        return np.nan
    z = bandpass(waveform[0], rate_hz)[i0:i1]
    h = 0.0
    for i in (1, 2):
        if component_mask[i]:
            h += float(np.sum(bandpass(waveform[i], rate_hz)[i0:i1] ** 2))
    if h <= 0:
        return np.nan
    return float(np.sum(z ** 2)) / h


def pick_placement(tp, duration_s: float, edge_s: float = C4_EDGE_S) -> dict:
    """C4 per row: p_fraction = tp / duration; edge when tp < edge_s or tp > duration - edge_s."""
    if tp is None:
        return dict(p_fraction=np.nan, edge=False)
    tp = float(tp)
    return dict(p_fraction=tp / duration_s, edge=bool(tp < edge_s or tp > duration_s - edge_s))


def modal_share(values) -> tuple:
    """(mode, share) of the single most common finite value; (nan, nan) when empty."""
    v = pd.Series(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.empty:
        return np.nan, np.nan
    counts = v.value_counts()
    return float(counts.index[0]), float(counts.iloc[0] / len(v))


def sta_lta_triggers(waveform, rate_hz: float, component_mask=(True, True, True), *, sta_s: float = C6_STA_S,
                     lta_s: float = C6_LTA_S, on: float = C6_ON, off: float = C6_OFF) -> tuple:
    """C6 detector: obspy classic_sta_lta (STA sta_s, LTA lta_s) on the
    root-sum-square of the band-passed available components, trigger_onset(on, off).
    Returns (array of [on_s, off_s], testable). No trigger can occur inside
    the first lta_s (LTA warm-up); a trace shorter than LTA + STA is not testable."""
    nsta, nlta = max(int(round(sta_s * rate_hz)), 1), max(int(round(lta_s * rate_hz)), 2)
    n = waveform.shape[1]
    comps = [bandpass(waveform[i], rate_hz) for i in range(3) if component_mask[i]]
    if not comps or n <= nlta + nsta:
        return np.zeros((0, 2)), False
    amplitude = np.sqrt(np.sum(np.stack(comps) ** 2, axis=0))
    cft = classic_sta_lta(amplitude, nsta, nlta)
    trig = np.asarray(trigger_onset(cft, on, off), dtype=float).reshape(-1, 2)
    return trig / rate_hz, True


def extra_triggers(on_times, tp, ts, near_s: float = C6_NEAR_S, coda_s: float = C6_S_CODA_S) -> np.ndarray:
    """Trigger on-times farther than near_s from every labelled arrival and,
    when S is labelled, outside the S coda window [ts, ts + coda_s]."""
    on_times = np.asarray(on_times, dtype=float)
    keep = np.ones(len(on_times), dtype=bool)
    for t in (tp, ts):
        if t is not None:
            keep &= np.abs(on_times - float(t)) > near_s
    if ts is not None:
        keep &= ~((on_times >= float(ts)) & (on_times <= float(ts) + coda_s))
    return on_times[keep]


def classify_triggers(on_times, t_ref, extra=None, near_s: float = C6_NEAR_S) -> str:
    """Proposal for an Aguilar-flagged row. no_detection: no trigger;
    wrong_first_pick: >= 1 trigger and none within near_s of the labelled
    arrival t_ref (P, or S when there is no P); second_event: a trigger within
    near_s of t_ref and >= 1 trigger that no label explains (`extra`, from
    extra_triggers: the labelled S and its coda do not count as a second
    event); single_detection: a trigger near t_ref and nothing unexplained;
    not_testable when there is no labelled arrival. When `extra` is None every
    other trigger counts (>= 2 triggers, one near t_ref, gives second_event)."""
    if t_ref is None:
        return "not_testable"
    on_times = np.asarray(on_times, dtype=float)
    if len(on_times) == 0:
        return "no_detection"
    near = bool(np.any(np.abs(on_times - float(t_ref)) <= near_s))
    if not near:
        return "wrong_first_pick"
    n_other = len(on_times) - 1 if extra is None else len(np.asarray(extra, dtype=float))
    return "second_event" if n_other >= 1 else "single_detection"


def sp_consistency(tp, ts, dist_km, *, mad_factor: float = C1_MAD_FACTOR, min_residual_s: float = C1_MIN_RESIDUAL_S,
                   min_rows: int = C1_MIN_ROWS, vp_km_s: float = VP_KM_S) -> dict:
    """C1 rule. Theil-Sen fit (scipy.stats.theilslopes) of ts - tp against
    distance over the rows with both picks, a distance and ts > tp, when at
    least min_rows such rows exist and the distances are not all equal;
    residual_s = (ts - tp) - (a + b * D);
    flag when |residual_s| > max(mad_factor * MAD, min_residual_s), with
    MAD = median |r - median r| over the fitted rows, or when ts <= tp. Rows
    without a distance or without both picks get residual NaN and are flagged
    only when ts <= tp. slope is s/km; the implied Vp/Vs = 1 + Vp * slope
    (from 1/Vs - 1/Vp = slope with Vp = vp_km_s) is a sanity line, not a flag."""
    tp = np.asarray([np.nan if v is None else v for v in tp], dtype=float)
    ts = np.asarray([np.nan if v is None else v for v in ts], dtype=float)
    dist = np.asarray([np.nan if v is None else v for v in dist_km], dtype=float)
    both = np.isfinite(tp) & np.isfinite(ts)
    ts_le_tp = both & (ts <= tp)
    fit_rows = both & np.isfinite(dist) & (ts > tp)
    residual = np.full(len(tp), np.nan)
    flag = ts_le_tp.copy()
    out = dict(n_fit=int(fit_rows.sum()), slope_s_per_km=np.nan, intercept_s=np.nan, mad_s=np.nan,
               threshold_s=np.nan, implied_vp_vs=np.nan)
    if fit_rows.sum() >= min_rows and np.ptp(dist[fit_rows]) > 0:
        y, x = (ts - tp)[fit_rows], dist[fit_rows]
        slope, intercept, _, _ = theilslopes(y, x)
        r = y - (intercept + slope * x)
        mad = float(np.median(np.abs(r - np.median(r))))
        threshold = max(mad_factor * mad, min_residual_s)
        residual[fit_rows] = r
        flag |= fit_rows & (np.abs(residual) > threshold)
        out.update(slope_s_per_km=float(slope), intercept_s=float(intercept), mad_s=mad, threshold_s=float(threshold),
                   implied_vp_vs=float(1.0 + vp_km_s * slope))
    out.update(residual_s=residual, flag=flag, ts_le_tp=ts_le_tp)
    return out


# ── per-row audit ────────────────────────────────────────────────────────────

def audit_row(row: LabelRow, c2_rule: str = C2_RULE) -> dict:
    """Every per-row check except C1 (which needs the source). Waveforms are not kept."""
    w, rate, mask = row.waveform, row.rate_hz, row.component_mask
    out = dict(source=row.source, trace_id=row.trace_id, station=row.station, event_id=row.event_id,
               rate_hz=float(rate), n_samples=row.n_samples, duration_s=row.duration_s, p_s=row.p_s, s_s=row.s_s,
               distance_km=row.distance_km, p_status=row.p_status, s_status=row.s_status,
               flagged_multiplet=bool(row.flagged_multiplet),
               component_mask="".join(c for c, m in zip("ZNE", mask) if m),
               p_sample=float(row.p_sample) if row.p_sample is not None else (
                   float(np.round(row.p_s * rate)) if row.p_s is not None else np.nan))
    out.update({k: v for k, v in row.meta.items() if k not in out})
    p = check_p_onset(w, rate, row.p_s, mask, ts=row.s_s, rule=c2_rule)
    s = check_s_onset(w, rate, row.s_s, mask, tp=row.p_s, rule=c2_rule)
    out.update(c2_rule=c2_rule, c2_onset_s=p["onset_s"], c2_residual_s=p["residual_s"], c2_rms_ratio=p["rms_ratio"],
               c2_testable=p["testable"], c2_late=p["late"], c2_emergent=p["emergent"], c2_flag=p["flag"],
               c2s_onset_s=s["onset_s"], c2s_residual_s=s["residual_s"], c2s_rms_ratio=s["rms_ratio"],
               c2s_testable=s["testable"], c2s_late=s["late"], c2s_emergent=s["emergent"], c2s_flag=s["flag"])
    ratio_p = zh_ratio(w, rate, row.p_s, component_mask=mask)
    ratio_s = zh_ratio(w, rate, row.s_s, component_mask=mask)
    c3_ok = p["testable"] and s["testable"] and np.isfinite(ratio_p) and np.isfinite(ratio_s)
    out.update(c3_ratio_p=ratio_p, c3_ratio_s=ratio_s, c3_testable=bool(c3_ok),
               c3_p_gt_s=(bool(ratio_p > ratio_s) if c3_ok else None))
    placement = pick_placement(row.p_s, row.duration_s)
    out.update(c4_p_fraction=placement["p_fraction"], c4_edge=placement["edge"])
    trig, testable = sta_lta_triggers(w, rate, mask)
    on_times = trig[:, 0] if len(trig) else np.zeros(0)
    extra = extra_triggers(on_times, row.p_s, row.s_s)
    ref = row.p_s if row.p_s is not None else row.s_s
    before_p = extra[extra < float(ref)] if ref is not None else extra
    out.update(c6_testable=testable, c6_triggers_json=json.dumps([round(float(t), 3) for t in on_times]),
               c6_n_triggers=int(len(on_times)), c6_n_extra_triggers=int(len(extra)),
               c6_has_unlabelled_arrival=bool(len(extra) >= 1), c6_n_extra_before_p=int(len(before_p)),
               c6_class=classify_triggers(on_times, ref, extra) if testable else "not_testable",
               c6_p_in_blind=bool(row.p_s is not None and row.p_s < C6_LTA_S),
               suggested_extra_arrival_s=json.dumps([round(float(t), 3) for t in extra]))
    out["c2_late_kind"] = late_kind(bool(out["c2_late"]), int(out["c6_n_extra_before_p"]))
    return out


def finish_source(records: list, source: str) -> pd.DataFrame:
    """C1 over the source's rows, then the derived tier columns."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    c1 = sp_consistency(df["p_s"].tolist(), df["s_s"].tolist(), df["distance_km"].tolist())
    df["c1_residual_s"] = c1["residual_s"]
    df["c1_flag"] = c1["flag"]
    df["c1_ts_le_tp"] = c1["ts_le_tp"]
    df.attrs["c1"] = {k: v for k, v in c1.items() if k not in ("residual_s", "flag", "ts_le_tp")}
    # P to `unknown` on a S-P outlier, an edge pick or a suspect late pick; a late
    # label with an unlabelled earlier event keeps `manual` and gets the extra
    # arrival; an emergent onset flags only under the symmetric rule.
    suspect = df["c2_late_kind"].astype(str) == "suspect_pick"
    emergent_flag = df["c2_flag"].astype(bool) & ~df["c2_late"].astype(bool)
    df["suggested_tier"] = np.where(df["c1_flag"] | df["c4_edge"] | suspect | emergent_flag, "unknown", "manual")
    has_s = df["s_s"].notna()
    df["suggested_tier_s"] = np.where(~has_s, "", np.where(df["c1_flag"] | df["c2s_flag"], "unknown", "manual"))
    return df


def _frac_ci(indicator) -> tuple:
    """(fraction, lo, hi) of a 0/1 indicator; NaN when empty; interval NaN below five values."""
    v = np.asarray(pd.Series(indicator).dropna().astype(float))
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = bootstrap_ci(v, n_boot=N_BOOT, seed=BOOT_SEED)
    return float(v.mean()), lo, hi


def _nanmedian(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    return float(np.median(v)) if len(v) else np.nan


def _nanmad(v):
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    return float(np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def summarise(df: pd.DataFrame, source: str, n_read_errors: int = 0) -> dict:
    """One summary row per source: counts, medians, MADs, fractions and 95 %
    percentile-bootstrap intervals (metrics.bootstrap_ci, n_boot 1000, seed 0)."""
    s = dict(source=source, n_rows=int(len(df)), n_read_errors=int(n_read_errors))
    if df.empty:
        return s
    c1 = df.attrs.get("c1", {})
    s.update(n_with_p=int(df["p_s"].notna().sum()), n_with_s=int(df["s_s"].notna().sum()),
             n_with_distance=int(df["distance_km"].notna().sum()),
             n_flagged_multiplet=int(df["flagged_multiplet"].sum()),
             rates_hz=json.dumps(sorted(float(r) for r in df["rate_hz"].unique())),
             n_samples_median=_nanmedian(df["n_samples"]), n_samples_min=int(df["n_samples"].min()),
             n_samples_max=int(df["n_samples"].max()), duration_s_median=_nanmedian(df["duration_s"]))
    # C1
    both = df["p_s"].notna() & df["s_s"].notna()
    frac, lo, hi = _frac_ci(df.loc[both, "c1_flag"])
    s.update(c1_n_fit=int(c1.get("n_fit", 0)), c1_slope_s_per_km=c1.get("slope_s_per_km", np.nan),
             c1_intercept_s=c1.get("intercept_s", np.nan), c1_implied_vp_vs=c1.get("implied_vp_vs", np.nan),
             c1_residual_mad_s=c1.get("mad_s", np.nan), c1_threshold_s=c1.get("threshold_s", np.nan),
             c1_n_flag=int(df["c1_flag"].sum()), c1_n_ts_le_tp=int(df["c1_ts_le_tp"].sum()),
             c1_flag_frac=frac, c1_flag_lo=lo, c1_flag_hi=hi)
    # C2
    for tag, has in (("c2", df["p_s"].notna()), ("c2s", df["s_s"].notna())):
        testable = df[f"{tag}_testable"].astype(bool)
        t_frac, t_lo, t_hi = _frac_ci(testable[has])
        f_frac, f_lo, f_hi = _frac_ci(df.loc[testable, f"{tag}_flag"])
        res = df.loc[testable, f"{tag}_residual_s"].astype(float)
        l_frac, l_lo, l_hi = _frac_ci(df.loc[testable, f"{tag}_late"])
        e_frac, e_lo, e_hi = _frac_ci(df.loc[testable, f"{tag}_emergent"])
        w_frac, w_lo, w_hi = _frac_ci((res < -2 * C2_TOL_S).astype(float)) if len(res) else (np.nan, np.nan, np.nan)
        s.update({f"{tag}_n_testable": int(testable.sum()), f"{tag}_testable_frac": t_frac,
                  f"{tag}_testable_lo": t_lo, f"{tag}_testable_hi": t_hi,
                  f"{tag}_residual_median_s": _nanmedian(res), f"{tag}_residual_mad_s": _nanmad(res),
                  f"{tag}_residual_p05_s": float(np.percentile(res, 5)) if len(res) else np.nan,
                  f"{tag}_residual_p95_s": float(np.percentile(res, 95)) if len(res) else np.nan,
                  f"{tag}_n_flag": int(df[f"{tag}_flag"].sum()), f"{tag}_flag_frac": f_frac,
                  f"{tag}_flag_lo": f_lo, f"{tag}_flag_hi": f_hi,
                  f"{tag}_n_late": int(df[f"{tag}_late"].sum()), f"{tag}_late_frac": l_frac,
                  f"{tag}_late_lo": l_lo, f"{tag}_late_hi": l_hi,
                  f"{tag}_late_1s_frac": w_frac, f"{tag}_late_1s_lo": w_lo, f"{tag}_late_1s_hi": w_hi,
                  f"{tag}_n_emergent": int(df[f"{tag}_emergent"].sum()), f"{tag}_emergent_frac": e_frac,
                  f"{tag}_emergent_lo": e_lo, f"{tag}_emergent_hi": e_hi})
    testable = df["c2_testable"].astype(bool)
    kind = df["c2_late_kind"].astype(str)
    s["c2_rule"] = str(df["c2_rule"].iloc[0]) if "c2_rule" in df else C2_RULE
    for k, name in (("suspect_pick", "suspect"), ("unlabelled_earlier_event", "unlabelled_earlier")):
        frac, lo, hi = _frac_ci((kind[testable] == k).astype(float))
        s.update({f"c2_{name}_n": int((kind == k).sum()), f"c2_{name}_frac": frac, f"c2_{name}_lo": lo, f"c2_{name}_hi": hi})
    # C3
    c3 = df.loc[df["c3_testable"].astype(bool), "c3_p_gt_s"]
    frac, lo, hi = _frac_ci(c3)
    s.update(c3_n=int(len(c3)), c3_p_gt_s_frac=frac, c3_p_gt_s_lo=lo, c3_p_gt_s_hi=hi,
             c3_warning=bool(np.isfinite(frac) and frac < C3_WARN_FRACTION))
    # C4
    has_p = df["p_s"].notna()
    frac, lo, hi = _frac_ci(df.loc[has_p, "c4_edge"])
    mode, share = modal_share(df.loc[has_p, "p_sample"])
    s.update(c4_edge_frac=frac, c4_edge_lo=lo, c4_edge_hi=hi, c4_n_edge=int(df["c4_edge"].sum()),
             c4_mode_p_sample=mode, c4_mode_share=share, c4_p_fraction_median=_nanmedian(df.loc[has_p, "c4_p_fraction"]))
    # C6
    testable = df["c6_testable"].astype(bool)
    flagged = df["flagged_multiplet"].astype(bool)
    frac, lo, hi = _frac_ci(df.loc[testable, "c6_has_unlabelled_arrival"])
    b_frac, b_lo, b_hi = _frac_ci((df.loc[testable, "c6_n_extra_before_p"] > 0).astype(float))
    s.update(c6_n_testable=int(testable.sum()), c6_n_triggers_median=_nanmedian(df.loc[testable, "c6_n_triggers"]),
             c6_p_in_blind_frac=float(df.loc[has_p, "c6_p_in_blind"].mean()) if has_p.any() else np.nan,
             c6_unlabelled_frac=frac, c6_unlabelled_lo=lo, c6_unlabelled_hi=hi,
             c6_n_unlabelled=int(df["c6_has_unlabelled_arrival"].sum()),
             c6_unlabelled_before_p_frac=b_frac, c6_unlabelled_before_p_lo=b_lo, c6_unlabelled_before_p_hi=b_hi)
    for tag, sel in (("flagged", flagged), ("unflagged", ~flagged)):
        frac, lo, hi = _frac_ci(df.loc[testable & sel, "c6_has_unlabelled_arrival"])
        s.update({f"c6_unlabelled_{tag}_frac": frac, f"c6_unlabelled_{tag}_lo": lo, f"c6_unlabelled_{tag}_hi": hi,
                  f"c6_n_{tag}": int((testable & sel).sum())})
    fl = df.loc[testable & flagged, "c6_class"]
    s["c6_flagged_n"] = int(len(fl))
    for cls in ("second_event", "wrong_first_pick", "no_detection", "single_detection"):
        frac, lo, hi = _frac_ci((fl == cls).astype(float)) if len(fl) else (np.nan, np.nan, np.nan)
        s.update({f"c6_{cls}_frac": frac, f"c6_{cls}_lo": lo, f"c6_{cls}_hi": hi})
    # derived
    s.update(suggested_unknown_frac=float((df["suggested_tier"] == "unknown").mean()),
             n_suggested_unknown=int((df["suggested_tier"] == "unknown").sum()),
             n_extra_arrival_rows=int((df["c6_n_extra_triggers"] > 0).sum()))
    return s


def review_sheet(df: pd.DataFrame, seed: int = BOOT_SEED, n: int = REVIEW_SHEET_ROWS) -> pd.DataFrame:
    """Up to n random rows per source that a check or the multiplet report
    singled out, with the reasons, the labelled picks and the detector
    triggers; the classification is a proposal for the reviewer."""
    if df.empty:
        return pd.DataFrame(columns=["source", "trace_id"])
    reasons = {"multiplet_report": df["flagged_multiplet"].astype(bool),
               "unlabelled_arrival": df["c6_has_unlabelled_arrival"].astype(bool),
               "c1": df["c1_flag"].astype(bool), "c2_p": df["c2_flag"].astype(bool),
               "c2_p_emergent": df["c2_emergent"].astype(bool),
               "c2_s": df["c2s_flag"].astype(bool), "c4_edge": df["c4_edge"].astype(bool)}
    any_reason = np.zeros(len(df), dtype=bool)
    for v in reasons.values():
        any_reason |= v.to_numpy()
    cand = df.loc[any_reason].copy()
    if cand.empty:
        return pd.DataFrame(columns=["source", "trace_id"])
    cand["reasons"] = [";".join(k for k, v in reasons.items() if v.loc[i]) for i in cand.index]
    rng = np.random.default_rng(seed)
    if len(cand) > n:
        cand = cand.iloc[np.sort(rng.choice(len(cand), size=n, replace=False))]
    cols = ["source", "trace_id", "station", "event_id", "rate_hz", "p_s", "s_s", "distance_km", "p_status", "s_status",
            "flagged_multiplet", "reasons", "c1_residual_s", "c2_residual_s", "c2_late_kind", "c2s_residual_s",
            "c6_triggers_json", "c6_n_extra_triggers", "c6_class", "suggested_tier", "suggested_extra_arrival_s"]
    cols += [c for c in ("window_start_time",) if c in cand]
    return cand[cols].reset_index(drop=True)


# ── C5: cross-source duplicates (metadata only) ──────────────────────────────

def duplicate_pairs(meta: pd.DataFrame, *, tol_s: float = C5_TOL_S, time_tol_s: float = C5_TIME_TOL_S,
                    dist_tol_deg: float = C5_DIST_TOL_DEG, max_pairs_per_key: int = 10) -> tuple:
    """C5 rule. `meta` has one row per trace with columns source, trace_name,
    station, source_origin_time, source_latitude_deg, source_longitude_deg and,
    when known, trace_start_time, p_sample and rate_hz. Rows are joined into
    events by exclusion_bundle.origin_unions (identical fingerprints, or
    origins within time_tol_s and dist_tol_deg). For every (event, station)
    seen by more than one source, the cross-source row pairs are listed (at
    most max_pairs_per_key per key). A pair is comparable when both rows carry
    trace_start_time, p_sample and rate_hz: |P_i - P_j| in absolute time is
    compared with tol_s. Returns (pairs frame, summary dict with the fraction
    disagreeing and its bootstrap interval over comparable pairs)."""
    m = meta.reset_index(drop=True)
    n = len(m)
    uf = eb.UnionFind(n)
    for i, j in eb.origin_unions(m[hs.TIME_COL], m[hs.LAT_COL], m[hs.LON_COL], time_tol_s, dist_tol_deg):
        uf.union(int(i), int(j))
    m["_group"] = uf.labels()
    valid = m[hs.TIME_COL].notna() & m[hs.LAT_COL].notna() & m[hs.LON_COL].notna() & m["station"].notna()
    rows = []
    start = pd.to_datetime(m["trace_start_time"], utc=True, errors="coerce") if "trace_start_time" in m else pd.Series(pd.NaT, index=m.index)
    p_abs = pd.Series(pd.NaT, index=m.index, dtype="datetime64[ns, UTC]")
    if "p_sample" in m and "rate_hz" in m:
        offset = pd.to_numeric(m["p_sample"], errors="coerce") / pd.to_numeric(m["rate_hz"], errors="coerce")
        ok = start.notna() & offset.notna()
        p_abs[ok] = start[ok] + pd.to_timedelta(offset[ok], unit="s")
    for (_, station), grp in m[valid].groupby(["_group", "station"], sort=False):
        if grp["source"].nunique() < 2:
            continue
        idx = grp.index.tolist()
        count = 0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if m.at[i, "source"] == m.at[j, "source"]:
                    continue
                comparable = pd.notna(p_abs[i]) and pd.notna(p_abs[j])
                dt = abs((p_abs[i] - p_abs[j]).total_seconds()) if comparable else np.nan
                rows.append(dict(source_a=m.at[i, "source"], trace_a=m.at[i, "trace_name"], source_b=m.at[j, "source"],
                                 trace_b=m.at[j, "trace_name"], station=station, origin_time=str(m.at[i, hs.TIME_COL]),
                                 comparable=bool(comparable), dp_s=dt,
                                 disagree=(bool(dt > tol_s) if comparable else None)))
                count += 1
                if count >= max_pairs_per_key:
                    break
            if count >= max_pairs_per_key:
                break
    pairs = pd.DataFrame(rows, columns=["source_a", "trace_a", "source_b", "trace_b", "station", "origin_time",
                                        "comparable", "dp_s", "disagree"])
    comp = pairs.loc[pairs["comparable"].astype(bool), "disagree"] if len(pairs) else pd.Series([], dtype=float)
    frac, lo, hi = _frac_ci(comp.astype(float)) if len(comp) else (np.nan, np.nan, np.nan)
    summary = dict(n_pairs=int(len(pairs)), n_comparable=int(len(comp)), n_not_comparable=int(len(pairs) - len(comp)),
                   disagree_frac=frac, disagree_lo=lo, disagree_hi=hi, tol_s=tol_s,
                   pairs_by_sources={f"{a}|{b}": int(c) for (a, b), c in
                                     (pairs.groupby(["source_a", "source_b"]).size().items() if len(pairs) else [])})
    return pairs, summary


# ── readers ──────────────────────────────────────────────────────────────────

def status_text(row: dict, columns) -> str:
    for c in columns:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip().lower()
    return "unknown"


def load_multiplet_report(source: str, report_dirs=DEFAULT_REPORT_DIRS) -> tuple:
    """(set of flagged trace names, info) from a cached Aguilar multiplet report;
    never downloads. info records the path and sha256 used, or why none was."""
    stem = next((s for s, d in lef.REPORT_STEM_TO_DATASET.items() if d == source), None)
    if stem is None:
        return set(), dict(status="no_report_for_source")
    for d in report_dirs:
        path = lef._cache_path(d, stem)
        if path.exists():
            names = lef.load_bad_trace_names(source, extra_cache_dirs=[d])
            return names, dict(status="cached", path=str(path), sha256=_sha256(path), n_flagged=len(names))
    return set(), dict(status="report_not_cached", stem=stem, searched=[str(d) for d in report_dirs])


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class HeldoutReader:
    """Rows from a built held-out case: one per (event, station) with a manual
    P (S optional) on a station whose waveforms are on disk. The waveform is
    cut from the case's MiniSEED at native rate as a length_s window starting
    pre_s before P (shorter at file edges; `window_start_time` records the
    actual start). Distance from the catalogue and station coordinates when
    the event is catalogued, else the stations.csv `km` column. Regression and
    development roles only: the sealed panel is not read.
    """

    def __init__(self, key: str, data_root=None, pre_s: float = HELDOUT_PRE_S, length_s: float = HELDOUT_LENGTH_S,
                 tiers=("manual",)):
        import evaluation_policy as policy
        import heldout_testset_score as hts
        role = policy.role_for(key)
        if role not in policy.SCORING_ROLES:
            raise PermissionError(f"{key} is protected ({role}); the label audit reads regression/dev cases only")
        self.key, self.pre_s, self.length_s, self.tiers = key, float(pre_s), float(length_s), tuple(tiers)
        self.root = Path(data_root) if data_root is not None else hts.OUT_ROOT
        self.case_dir = self.root / key
        old = hts.OUT_ROOT
        hts.OUT_ROOT = self.root
        try:
            policy.record_access(key, "reference_qa", data_root=self.root)
            self.windows, self.picks = hts._load_sequence(key, tiers=self.tiers)
        finally:
            hts.OUT_ROOT = old
        self.stations = pd.read_csv(self.case_dir / "stations.csv")
        cat = self.case_dir / "catalog.parquet"
        self.catalog = pd.read_parquet(cat) if cat.exists() else pd.DataFrame(columns=["event", "lat", "lon"])
        self.status = {}
        if "status" in self.picks:
            for r in self.picks.itertuples():
                self.status.setdefault((r.event, r.station, r.phase), str(r.status))

    def file_hashes(self) -> dict:
        out = {}
        for name in ("picks.parquet", "windows.csv", "stations.csv", "catalog.parquet", "manifest.json"):
            p = self.case_dir / name
            out[name] = _sha256(p) if p.exists() else None
        out["waveform_files"] = sorted(f.name for f in (self.case_dir / "waveforms").glob("*.mseed"))
        return out

    def distance_km(self, event, station):
        sta = self.stations.loc[self.stations["station"] == station]
        ev = self.catalog.loc[self.catalog["event"] == event] if "event" in self.catalog else self.catalog.iloc[0:0]
        if len(sta) and len(ev) and pd.notna(sta["lat"].iloc[0]) and pd.notna(ev["lat"].iloc[0]):
            d = hs.gc_distance_deg(float(ev["lat"].iloc[0]), float(ev["lon"].iloc[0]),
                                   float(sta["lat"].iloc[0]), float(sta["lon"].iloc[0]))
            return float(d) * 111.2, "catalogue"
        if len(sta) and "km" in sta and pd.notna(sta["km"].iloc[0]):
            return float(sta["km"].iloc[0]), "stations_csv_km"
        return None, "none"

    def cut(self, stream, t_start):
        """(waveform (3, n) ZNE float32, rate, actual start UTCDateTime, mask, n_gaps) for [t_start, t_start + length_s]."""
        from obspy import Stream, UTCDateTime
        t0, t1 = UTCDateTime(t_start), UTCDateTime(t_start) + self.length_s
        seg = stream.slice(t0, t1).copy()
        n_gaps = len(seg.get_gaps())
        seg.merge(method=1, fill_value=0)
        rates = {float(tr.stats.sampling_rate) for tr in seg}
        if len(rates) != 1:
            raise ValueError(f"Mixed or missing sampling rates in the window: {sorted(rates)}")
        rate = rates.pop()
        slots = {}
        for tr in seg:
            code = tr.stats.channel[-1].upper()
            slot = {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}.get(code)
            if slot is not None and slot not in slots:
                slots[slot] = tr
        if 0 not in slots:
            raise ValueError("No vertical channel in the window")
        start = max(tr.stats.starttime for tr in slots.values())
        end = min(tr.stats.endtime for tr in slots.values())
        seg = Stream(list(slots.values())).slice(start, end)
        n = min(tr.stats.npts for tr in seg)
        out = np.zeros((3, n), dtype=np.float32)
        mask = [False, False, False]
        for tr in seg:
            slot = {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}[tr.stats.channel[-1].upper()]
            out[slot] = np.asarray(tr.data[:n], dtype=np.float32)
            mask[slot] = True
        return out, rate, start, tuple(mask), n_gaps

    def rows(self):
        for win in self.windows:
            ref = win["reference"]
            if ref.empty:
                continue
            for (event, station), grp in ref.groupby(["event", "station"], sort=False):
                p = grp.loc[grp["phase"] == "P", "time"]
                if p.empty or station not in win["streams"]:
                    continue
                tp = p.iloc[0]
                s = grp.loc[grp["phase"] == "S", "time"]
                ts = s.iloc[0] if len(s) else None
                try:
                    wave, rate, start, mask, n_gaps = self.cut(win["streams"][station], tp - pd.Timedelta(seconds=self.pre_s))
                except ValueError as exc:
                    log(f"    skip {self.key} {win['window_id']} {station} {event}: {exc}")
                    continue
                start_ts = pd.Timestamp(start.datetime, tz="UTC")
                p_s = (tp - start_ts).total_seconds()
                s_s = (ts - start_ts).total_seconds() if ts is not None else None
                if s_s is not None and not (0 <= s_s < wave.shape[1] / rate):
                    s_s = None
                dist, dist_src = self.distance_km(event, station)
                yield LabelRow(source=self.key, trace_id=f"{self.key}:{win['window_id']}:{station}:{event}",
                               station=station, event_id=event, waveform=wave, rate_hz=rate, p_s=p_s, s_s=s_s,
                               distance_km=dist, p_status=self.status.get((event, station, "P"), "unknown"),
                               s_status=self.status.get((event, station, "S"), "unknown") if ts is not None else "unknown",
                               flagged_multiplet=False, component_mask=mask,
                               meta=dict(window_id=win["window_id"], window_start_time=str(start_ts.isoformat()),
                                         distance_source=dist_src, n_gaps=int(n_gaps),
                                         tier=str(grp["tier"].iloc[0])))


def stratified_sample(bins: pd.Series, n: int, seed: int, floor: int = STRATUM_FLOOR) -> np.ndarray:
    """Deterministic index sample: proportional allocation across the strata of
    `bins`, each stratum floored at min(floor, its size), the total trimmed or
    topped up on the largest stratum to n; within a stratum the draw is a
    seeded choice over the index sorted by label, so the same metadata gives
    the same rows whatever its row order."""
    bins = bins.astype(str)
    total = len(bins)
    if total <= n:
        return np.asarray(sorted(bins.index))
    counts = bins.value_counts()
    alloc = {b: min(int(c), max(int(round(n * c / total)), min(floor, int(c)))) for b, c in counts.items()}
    largest = counts.index[0]
    diff = n - sum(alloc.values())
    alloc[largest] = int(np.clip(alloc[largest] + diff, 0, counts[largest]))
    rng = np.random.default_rng(seed)
    chosen = []
    for b in counts.index:
        idx = np.asarray(sorted(bins.index[bins == b]))
        k = alloc[b]
        if k > 0:
            chosen.extend(idx[np.sort(rng.choice(len(idx), size=k, replace=False))].tolist())
    return np.asarray(chosen)


def load_source_metadata(source: str, cache_root, route: str = "auto") -> tuple:
    """(metadata frame with a `chunk` column, route) the way build_training_dataset loads it.
    Imports seisbench here."""
    os.environ["SEISBENCH_CACHE_ROOT"] = str(cache_root)
    import seisbench
    import seisbench.data as sbd
    seisbench.cache_root = Path(cache_root)
    import build_training_dataset as btd
    path = Path(cache_root) / "datasets" / source
    if route == "auto":
        route = "chunked" if source in CHUNKED_PREFIX else "single" if source in SINGLE_HDF5 else "seisbench"
    if route == "chunked":
        prefix = "metadata" + CHUNKED_PREFIX.get(source, "waveforms_").removeprefix("waveforms")
        meta = btd._load_chunked_meta(path, prefix=prefix)
    elif route == "single":
        meta = sbd.WaveformDataset(str(path)).metadata.copy()
        meta["chunk"] = ""
    elif route == "seisbench":
        cfg = next((c for c in btd.DATASET_CONFIGS if c["name"] == source), None)
        if cfg is not None and cfg["cls"] is not None:
            ds = cfg["cls"](sampling_rate=None, component_order="ZNE", dimension_order="NCW", missing_components="pad")
        else:
            ds = sbd.WaveformDataset(str(path), sampling_rate=None, component_order="ZNE", dimension_order="NCW",
                                     missing_components="pad")
        meta = ds.metadata.copy()
        if "chunk" not in meta.columns:
            meta["chunk"] = meta["trace_chunk"].fillna("").astype(str) if "trace_chunk" in meta else ""
        meta.attrs["dataset"] = ds
    else:
        raise ValueError(f"Unknown route {route!r}")
    meta["chunk"] = meta["chunk"].fillna("").astype(str)
    return meta, route


# Distance columns the builder's configs use, tried in this order for a source it does not list.
DISTANCE_COLUMNS = (("path_ep_distance_km", "km"), ("source_distance_km", "km"), ("path_hyp_distance_km", "km"),
                    ("station_epicentral_distance", "km"), ("path_epicentral_distance_deg", "deg"),
                    ("source_distance_deg", "deg"), ("station_epicentral_distance_m", "m"))


def source_config(source: str, columns=None) -> dict:
    """dist_col, dist_unit, default_bin, use_s of build_training_dataset.DATASET_CONFIGS
    (imports seisbench); a source the builder does not list gets the first of
    DISTANCE_COLUMNS present in `columns`, no default bin and S used."""
    import build_training_dataset as btd
    cfg = next((c for c in btd.DATASET_CONFIGS if c["name"] == source), None)
    if cfg is None:
        col, unit = next(((c, u) for c, u in DISTANCE_COLUMNS if columns is not None and c in columns), (None, "km"))
        return dict(dist_col=col, dist_unit=unit, default_bin=None, use_s=True)
    return dict(dist_col=cfg["dist_col"], dist_unit=cfg["dist_unit"], default_bin=cfg["default_bin"], use_s=cfg["use_s"])


def prepare_picks(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pick indices, distance and bin per metadata row with the builder's
    conventions (coalesce_picks over P_PRIORITY/S_PRIORITY, to_km, distance_bin,
    S nulled on teleseismic rows, rows without any pick dropped)."""
    import build_training_dataset as btd
    p_vals, p_col = btd.coalesce_picks(meta, btd.P_PRIORITY)
    if cfg["use_s"]:
        s_vals, s_col = btd.coalesce_picks(meta, btd.S_PRIORITY)
    else:
        s_vals, s_col = pd.Series(np.nan, index=meta.index), None
    p_vals = p_vals.where(np.isfinite(p_vals) & (p_vals >= 0))
    s_vals = s_vals.where(np.isfinite(s_vals) & (s_vals >= 0))
    if cfg["dist_col"] and cfg["dist_col"] in meta.columns:
        dist = btd.to_km(meta[cfg["dist_col"]], cfg["dist_unit"])
    else:
        dist = pd.Series(np.nan, index=meta.index)
    dbin = dist.apply(lambda d: btd.distance_bin(d, default=cfg["default_bin"] or "unknown"))
    s_vals = s_vals.where(dbin != "teleseismic")
    out = pd.DataFrame({"p_sample": p_vals, "s_sample": s_vals, "distance_km": dist, "distance_bin": dbin,
                        "p_col": p_col or "", "s_col": s_col or ""}, index=meta.index)
    return out.loc[out["p_sample"].notna() | out["s_sample"].notna()]


class SeisBenchReader:
    """A stratified deterministic sample of one SeisBench source read through the
    loader's readers at the stored rate: `manifest_dataset._fetch_sbd` for the
    SeisBench route (via a namespace shim so the code path is the loader's),
    `SingleHDF5Reader`/`ChunkedHDF5Reader` (waveform_contract.read_hdf5_trace)
    for the direct routes. Strata are the builder's distance bins when the
    source has a distance column, else one stratum. Needs seisbench, h5py and
    torch (manifest_dataset) at construction, not at import.
    """

    def __init__(self, source: str, cache_root, sample: int = 5000, seed: int = 0, route: str = "auto",
                 report_dirs=DEFAULT_REPORT_DIRS):
        self.source, self.cache_root, self.sample, self.seed = source, Path(cache_root), int(sample), int(seed)
        os.environ["SEISBENCH_CACHE_ROOT"] = str(self.cache_root)
        try:
            import manifest_dataset as md
        except ImportError as exc:
            raise ImportError(f"the seisbench mode needs seisbench, h5py and torch (manifest_dataset): {exc}") from exc
        self.md = md
        meta, self.route = load_source_metadata(source, cache_root, route)
        self.cfg = source_config(source, meta.columns)
        self.dataset = meta.attrs.get("dataset")
        picks = prepare_picks(meta, self.cfg)
        self.n_candidates = int(len(picks))
        strata = picks["distance_bin"] if picks["distance_km"].notna().any() else pd.Series("all", index=picks.index)
        chosen = stratified_sample(strata, self.sample, self.seed)
        self.picks = picks.loc[chosen]
        self.meta = meta.loc[chosen]
        self.strata_counts = {str(k): int(v) for k, v in strata.loc[chosen].value_counts().items()}
        self.flagged, self.report_info = load_multiplet_report(source, report_dirs)
        self.read_errors = []
        path = self.cache_root / "datasets" / source
        if self.route == "chunked":
            wanted = {str(c): set(g["trace_name"].astype(str)) for c, g in self.meta.groupby("chunk")}
            self.reader = md.ChunkedHDF5Reader(path, CHUNKED_PREFIX.get(source, "waveforms_"), wanted)
        elif self.route == "single":
            self.reader = md.SingleHDF5Reader(path, wanted=set(self.meta["trace_name"].astype(str)))
        else:
            from types import SimpleNamespace
            ds = self.dataset
            index = md.ManifestDataset._build_name_index(ds)
            unique = {}
            for (_, name), position in index.items():
                unique[name] = None if name in unique else position
            self.shim = SimpleNamespace(_sbd_datasets={source: ds}, _sbd_name_to_idx={source: index},
                                        _sbd_unique_names={source: unique})
            self.reader = None

    def fetch(self, trace_name: str, chunk: str):
        if self.route == "chunked":
            return self.reader.get_record(chunk, trace_name, None)
        if self.route == "single":
            return self.reader.get_record(trace_name, None)
        return self.md.ManifestDataset._fetch_sbd(self.shim, self.source, trace_name, chunk, {})

    def rows(self):
        for idx, m in self.meta.iterrows():
            pk = self.picks.loc[idx]
            name, chunk = str(m["trace_name"]), str(m.get("chunk", "") or "")
            try:
                rec = self.fetch(name, chunk)
            except Exception as exc:  # noqa: BLE001  a row the loader would reject; counted, not hidden
                self.read_errors.append(dict(trace_name=name, chunk=chunk, error=f"{type(exc).__name__}: {exc}"))
                continue
            rate = float(rec.arrival_sampling_rate)
            p_sample = pk["p_sample"] if pd.notna(pk["p_sample"]) else None
            s_sample = pk["s_sample"] if pd.notna(pk["s_sample"]) else None
            row = m.to_dict()
            station = row.get("station_code")
            if pd.notna(row.get("station_network_code")) and pd.notna(station):
                station = f"{row['station_network_code']}.{station}"
            yield LabelRow(source=self.source, trace_id=f"{chunk}${name}" if chunk else name,
                           station=str(station) if pd.notna(station) else "", event_id=_event_id(row),
                           waveform=rec.waveform, rate_hz=float(rec.sampling_rate),
                           p_s=None if p_sample is None else p_sample / rate,
                           s_s=None if s_sample is None else s_sample / rate,
                           distance_km=pk["distance_km"] if pd.notna(pk["distance_km"]) else None,
                           p_status=status_text(row, P_STATUS_COLUMNS), s_status=status_text(row, S_STATUS_COLUMNS),
                           flagged_multiplet=name in self.flagged, component_mask=tuple(rec.component_mask),
                           p_sample=p_sample,
                           meta=dict(chunk=chunk, distance_bin=str(pk["distance_bin"]), p_col=pk["p_col"], s_col=pk["s_col"],
                                     trace_start_time=str(rec.start_time) if rec.start_time is not None else "",
                                     split=str(row.get("split", ""))))

    def close(self):
        if self.reader is not None:
            self.reader.close()


def _event_id(row: dict):
    for c in ("source_id", "event_id", "source_event_id", "trace_event_id"):
        v = row.get(c)
        if v is not None and pd.notna(v) and str(v).strip():
            return str(v)
    t = row.get(hs.TIME_COL)
    return str(t) if t is not None and pd.notna(t) else None


# ── outputs ──────────────────────────────────────────────────────────────────

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def versions() -> dict:
    import obspy
    import scipy
    out = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
               obspy=obspy.__version__)
    try:
        import seisbench
        out["seisbench"] = seisbench.__version__
    except ImportError:
        out["seisbench"] = None
    return out


def check_constants() -> dict:
    return {k: v for k, v in globals().items()
            if k.startswith(("C1_", "C2_", "C3_", "C4_", "C5_", "C6_", "BAND_", "HELDOUT_")) and not callable(v)
            and k != "C6_CLASSES"} | dict(VP_KM_S=VP_KM_S, N_BOOT=N_BOOT, BOOT_SEED=BOOT_SEED, STRATUM_FLOOR=STRATUM_FLOOR)


def run_source(name: str, rows_iter, out_dir: Path, n_read_errors_fn=None, c2_rule: str = C2_RULE) -> tuple:
    """Audit every row of one source, write rows.parquet and review_sheet.csv under out_dir/name."""
    records = []
    for i, row in enumerate(rows_iter):
        records.append(audit_row(row, c2_rule=c2_rule))
        if (i + 1) % 500 == 0:
            log(f"    {name}: {i + 1} rows")
    df = finish_source(records, name)
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "rows.parquet", index=False)
    review_sheet(df).to_csv(d / "review_sheet.csv", index=False)
    n_err = n_read_errors_fn() if n_read_errors_fn else 0
    summary = summarise(df, name, n_read_errors=n_err)
    log(f"    {name}: {len(df)} rows, C1 flags {summary.get('c1_n_flag', 0)}, C2 P flags {summary.get('c2_n_flag', 0)} "
        f"(suspect {summary.get('c2_suspect_n', 0)}, emergent {summary.get('c2_n_emergent', 0)}), "
        f"C4 edge {summary.get('c4_n_edge', 0)}, unlabelled arrivals {summary.get('c6_n_unlabelled', 0)}")
    return df, summary


def write_summary(out_dir: Path, summaries: list, provenance: dict, report: bool) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    if report:
        (out_dir / "report.md").write_text(render_report(summary, provenance))
    return summary


def _pct(frac, lo=None, hi=None) -> str:
    if frac is None or not np.isfinite(frac):
        return "n/a"
    s = f"{100 * frac:.1f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def _num(v, fmt="{:.2f}") -> str:
    return fmt.format(v) if v is not None and np.isfinite(v) else "n/a"


def render_report(summary: pd.DataFrame, provenance: dict = None) -> str:
    """Markdown table of the per-source results (percentages with 95 % bootstrap intervals)."""
    lines = ["# Label audit (41B): per-source summary", ""]
    if provenance:
        lines += [f"*mode `{provenance.get('mode')}`, commit `{provenance.get('git_commit', '')[:8]}`, "
                  f"created {provenance.get('created_utc', '')}, C2 rule `{provenance.get('c2_rule', C2_RULE)}`, "
                  f"bootstrap n={N_BOOT} seed={BOOT_SEED}.*", ""]
    if summary.empty:
        return "\n".join(lines + ["(no sources)", ""])
    head = ["source", "rows", "rate Hz", "C1 n fit", "C1 slope s/km", "Vp/Vs", "C1 flag %", "C2 P testable %",
            "C2 P late % (flag)", "of which suspect %", "C2 P emergent %", "C2 P median res s", "C2 S late % (flag)",
            "C2 S emergent %", "C3 P>S %", "C4 edge %", "C4 mode share", "C6 unlabelled %", "C6 before P %",
            "C6 unlabelled flagged/unflagged %", "C6 second/wrong/none % (flagged n)", "suggested unknown %"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    g = lambda r, k: r.get(k, np.nan)  # noqa: E731
    for _, r in summary.iterrows():
        cells = [str(r["source"]), str(int(r["n_rows"])), str(g(r, "rates_hz")), str(int(g(r, "c1_n_fit") or 0)) if np.isfinite(g(r, "c1_n_fit")) else "0",
                 _num(g(r, "c1_slope_s_per_km"), "{:.4f}"), _num(g(r, "c1_implied_vp_vs")),
                 _pct(g(r, "c1_flag_frac"), g(r, "c1_flag_lo"), g(r, "c1_flag_hi")),
                 _pct(g(r, "c2_testable_frac"), g(r, "c2_testable_lo"), g(r, "c2_testable_hi")),
                 _pct(g(r, "c2_late_frac"), g(r, "c2_late_lo"), g(r, "c2_late_hi")),
                 _pct(g(r, "c2_suspect_frac"), g(r, "c2_suspect_lo"), g(r, "c2_suspect_hi")),
                 _pct(g(r, "c2_emergent_frac"), g(r, "c2_emergent_lo"), g(r, "c2_emergent_hi")),
                 _num(g(r, "c2_residual_median_s")),
                 _pct(g(r, "c2s_late_frac"), g(r, "c2s_late_lo"), g(r, "c2s_late_hi")),
                 _pct(g(r, "c2s_emergent_frac"), g(r, "c2s_emergent_lo"), g(r, "c2s_emergent_hi")),
                 _pct(g(r, "c3_p_gt_s_frac"), g(r, "c3_p_gt_s_lo"), g(r, "c3_p_gt_s_hi")),
                 _pct(g(r, "c4_edge_frac"), g(r, "c4_edge_lo"), g(r, "c4_edge_hi")),
                 _pct(g(r, "c4_mode_share")),
                 _pct(g(r, "c6_unlabelled_frac"), g(r, "c6_unlabelled_lo"), g(r, "c6_unlabelled_hi")),
                 _pct(g(r, "c6_unlabelled_before_p_frac"), g(r, "c6_unlabelled_before_p_lo"), g(r, "c6_unlabelled_before_p_hi")),
                 f"{_pct(g(r, 'c6_unlabelled_flagged_frac'))} / {_pct(g(r, 'c6_unlabelled_unflagged_frac'))}",
                 f"{_pct(g(r, 'c6_second_event_frac'))} / {_pct(g(r, 'c6_wrong_first_pick_frac'))} / "
                 f"{_pct(g(r, 'c6_no_detection_frac'))} ({int(g(r, 'c6_flagged_n')) if np.isfinite(g(r, 'c6_flagged_n')) else 0})",
                 _pct(g(r, "suggested_unknown_frac"))]
        lines.append("| " + " | ".join(cells) + " |")
    warn = [str(r["source"]) for _, r in summary.iterrows() if bool(r.get("c3_warning", False))]
    lines += ["", f"C3 warning (Z/H after P not above Z/H after S on more than half of the testable rows): "
                  f"{', '.join(warn) if warn else 'none'}.", ""]
    return "\n".join(lines)


def base_provenance(mode: str, argv) -> dict:
    return dict(audit_version=AUDIT_VERSION, mode=mode, created_utc=datetime.now(timezone.utc).isoformat(),
                git_commit=git_commit(), command=" ".join(argv) if argv else "", versions=versions(),
                constants=check_constants())


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_heldout(a, argv) -> pd.DataFrame:
    out_dir = Path(a.out_dir)
    prov = base_provenance("heldout", argv)
    prov.update(keys=list(a.keys), data_root=str(a.data_root) if a.data_root else None, pre_s=a.pre_s,
                length_s=a.length_s, tiers=list(a.tiers), c2_rule=a.c2_rule, cases={})
    summaries = []
    for key in a.keys:
        log(f"  {key}")
        reader = HeldoutReader(key, data_root=a.data_root, pre_s=a.pre_s, length_s=a.length_s, tiers=a.tiers)
        df, summary = run_source(key, reader.rows(), out_dir, c2_rule=a.c2_rule)
        summaries.append(summary)
        prov["cases"][key] = dict(n_windows=len(reader.windows), n_rows=int(len(df)), files=reader.file_hashes())
    return write_summary(out_dir, summaries, prov, a.report)


def cmd_seisbench(a, argv) -> pd.DataFrame:
    out_dir = Path(a.out_dir)
    if not a.cache_root:
        raise SystemExit("--cache-root (or SEISBENCH_CACHE_ROOT) is required")
    report_dirs = [Path(d) for d in a.report_dirs] if a.report_dirs else list(DEFAULT_REPORT_DIRS)
    if a.download_reports:
        for source in a.sources:
            stem = next((s for s, d in lef.REPORT_STEM_TO_DATASET.items() if d == source), None)
            if stem is not None:
                lef.download_multiplet_report(stem, cache_dir=str(report_dirs[0]))
    prov = base_provenance("seisbench", argv)
    prov.update(cache_root=str(a.cache_root), sample=a.sample, seed=a.seed, c2_rule=a.c2_rule, sources={},
                label_error_reports={})
    summaries = []
    for source in a.sources:
        log(f"  {source}")
        reader = SeisBenchReader(source, a.cache_root, sample=a.sample, seed=a.seed, route=a.route, report_dirs=report_dirs)
        try:
            df, summary = run_source(source, reader.rows(), out_dir, n_read_errors_fn=lambda r=reader: len(r.read_errors),
                                     c2_rule=a.c2_rule)
        finally:
            reader.close()
        summaries.append(summary)
        prov["sources"][source] = dict(route=reader.route, n_candidates=reader.n_candidates, n_sampled=int(len(reader.meta)),
                                       strata=reader.strata_counts, n_rows=int(len(df)), config=reader.cfg,
                                       read_errors=reader.read_errors[:50], n_read_errors=len(reader.read_errors))
        prov["label_error_reports"][source] = reader.report_info
    return write_summary(out_dir, summaries, prov, a.report)


def cmd_duplicates(a, argv) -> dict:
    out_dir = Path(a.out_dir)
    if not a.cache_root:
        raise SystemExit("--cache-root (or SEISBENCH_CACHE_ROOT) is required")
    frames = []
    prov = base_provenance("duplicates", argv)
    prov.update(cache_root=str(a.cache_root), max_rows=a.max_rows, sources={})
    for source in a.sources:
        log(f"  {source}: metadata")
        meta, route = load_source_metadata(source, a.cache_root, a.route)
        picks = prepare_picks(meta, source_config(source, meta.columns))
        meta = meta.loc[picks.index]
        if a.max_rows and len(meta) > a.max_rows:
            meta = meta.iloc[:a.max_rows]
        rate = pd.Series(np.nan, index=meta.index)
        for c in ("trace_sampling_rate_hz", "sampling_rate"):
            if c in meta:
                rate = rate.fillna(pd.to_numeric(meta[c], errors="coerce"))
        if "trace_dt_s" in meta:
            rate = rate.fillna(1.0 / pd.to_numeric(meta["trace_dt_s"], errors="coerce"))
        if rate.isna().all() and meta.attrs.get("dataset") is not None:
            rate[:] = float(meta.attrs["dataset"].data_format.get("sampling_rate", np.nan))
        frames.append(pd.DataFrame({
            "source": source, "trace_name": meta["trace_name"].astype(str), "station": meta.get("station_code", pd.Series(np.nan, index=meta.index)),
            hs.TIME_COL: meta.get(hs.TIME_COL, pd.Series(np.nan, index=meta.index)),
            hs.LAT_COL: meta.get(hs.LAT_COL, pd.Series(np.nan, index=meta.index)),
            hs.LON_COL: meta.get(hs.LON_COL, pd.Series(np.nan, index=meta.index)),
            "trace_start_time": meta.get("trace_start_time", pd.Series(np.nan, index=meta.index)),
            "p_sample": picks.loc[meta.index, "p_sample"], "rate_hz": rate}))
        prov["sources"][source] = dict(route=route, n_rows=int(len(meta)))
    meta_all = pd.concat(frames, ignore_index=True)
    pairs, summary = duplicate_pairs(meta_all)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "duplicates.csv", index=False)
    prov["result"] = summary
    (out_dir / "duplicates_provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    log(f"  pairs {summary['n_pairs']}, comparable {summary['n_comparable']}, disagree > {C5_TOL_S} s: "
        f"{_pct(summary['disagree_frac'], summary['disagree_lo'], summary['disagree_hi'])} %")
    return summary


def cmd_report(a, argv=None) -> str:
    out_dir = Path(a.out_dir)
    summary = pd.read_csv(out_dir / "summary.csv")
    prov_path = out_dir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else None
    text = render_report(summary, prov)
    (out_dir / "report.md").write_text(text)
    print(text)
    return text


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heldout", help="audit the analyst picks of built held-out cases (regression/dev roles)")
    h.add_argument("--keys", nargs="+", required=True)
    h.add_argument("--out-dir", required=True)
    h.add_argument("--data-root", default=None, help="data/heldout_testset by default")
    h.add_argument("--pre-s", type=float, default=HELDOUT_PRE_S)
    h.add_argument("--length-s", type=float, default=HELDOUT_LENGTH_S)
    h.add_argument("--tiers", nargs="+", default=["manual"])
    h.add_argument("--c2-rule", default=C2_RULE, choices=list(C2_RULES),
                   help="asymmetric (default): flag late labels only; symmetric: also flag emergent onsets")
    h.add_argument("--report", action="store_true")
    s = sub.add_parser("seisbench", help="audit a stratified sample of SeisBench sources (needs the cache)")
    s.add_argument("--sources", nargs="+", required=True)
    s.add_argument("--sample", type=int, default=5000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cache-root", default=os.environ.get("SEISBENCH_CACHE_ROOT"))
    s.add_argument("--route", default="auto", choices=["auto", "seisbench", "single", "chunked"])
    s.add_argument("--report-dirs", nargs="*", default=None, help="Aguilar report caches; default data/labelerrors, ~/.cache/phasenet_retrain/label_errors")
    s.add_argument("--download-reports", action="store_true", help="fetch missing Aguilar reports into the first report dir")
    s.add_argument("--c2-rule", default=C2_RULE, choices=list(C2_RULES),
                   help="asymmetric (default): flag late labels only; symmetric: also flag emergent onsets")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--report", action="store_true")
    d = sub.add_parser("duplicates", help="C5: cross-source duplicate (event, station) rows from metadata only")
    d.add_argument("--sources", nargs="+", required=True)
    d.add_argument("--cache-root", default=os.environ.get("SEISBENCH_CACHE_ROOT"))
    d.add_argument("--route", default="auto", choices=["auto", "seisbench", "single", "chunked"])
    d.add_argument("--max-rows", type=int, default=None, help="per source, the first N rows with a pick")
    d.add_argument("--out-dir", required=True)
    r = sub.add_parser("report", help="render report.md from summary.csv")
    r.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "heldout":
        return cmd_heldout(a, argv)
    if a.cmd == "seisbench":
        return cmd_seisbench(a, argv)
    if a.cmd == "duplicates":
        return cmd_duplicates(a, argv)
    return cmd_report(a, argv)


if __name__ == "__main__":
    main()
