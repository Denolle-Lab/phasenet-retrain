#!/usr/bin/env python3
"""
heldout_testset_score.py

Loads a sequence directory built by scripts/build_heldout_testset.py into
the structures the QuakeScope notebooks use, and scores a set of PhaseNet
weights on it the way the notebooks do: recall against the reference picks
at a shared threshold, and recall at matched pick budget along a threshold
sweep. Pick-level only; the event-level scorer (association) is a separate
step of the plan.

    python scripts/heldout_testset_score.py --sequence adriatic_2022 --weights jma_wc instance
    python scripts/heldout_testset_score.py --all --weights jma_wc --out docs/audit_2026-09-07/heldout_scores.csv

Needs seisbench and torch for the picking; the matching and budget
functions are pure Python and unit-tested without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_testset_registry as reg  # noqa: E402
import evaluation_policy as policy  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "heldout_testset"
DETECT_FLOOR = 0.02
REPORT_THRESHOLD = 0.3
THRESHOLD_SWEEP = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MATCH_TOL = reg.MATCH_TOL_S


# ── the notebooks' functions, verbatim in behaviour ──────────────────────────

def reference_from(picks: pd.DataFrame, stations, t0, t1, use_reference_ok=True):
    """Reference picks on the chosen stations inside the window, keyed
    (station, phase), duplicates across events collapsed at MATCH_TOL."""
    from obspy import UTCDateTime
    lo, hi = pd.Timestamp(t0.datetime, tz="UTC"), pd.Timestamp(t1.datetime, tz="UTC")
    ok = picks["reference_ok"] if use_reference_ok and "reference_ok" in picks else (picks["mode"] == "manual")
    sel = picks[(picks["time"] >= lo) & (picks["time"] <= hi) & ok & picks.station.isin(stations)]
    out = {}
    for (sta, pha), g in sel.groupby(["station", "phase"]):
        times, kept = sorted(UTCDateTime(t) for t in g["time"]), []
        for t in times:
            if not kept or t - kept[-1] > MATCH_TOL:
                kept.append(t)
        out[(sta, pha)] = kept
    return out


def at_threshold(store, thr):
    return {k: sorted(t for t, v in vals if v >= thr) for k, vals in store.items()}


def match(reference, candidate, tol=MATCH_TOL):
    """Greedy nearest match; each candidate pick is consumed at most once."""
    used, residuals = set(), []
    for a in reference:
        best_i = best_d = None
        for i, m in enumerate(candidate):
            if i in used:
                continue
            d = m - a
            if abs(d) <= tol and (best_d is None or abs(d) < abs(best_d)):
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            residuals.append(best_d)
    return residuals, len(candidate) - len(used)


def matched_budget(sweep: pd.DataFrame, names, phase, sequence, n_points=4):
    sub = sweep[(sweep.phase == phase) & (sweep.sequence == sequence)]
    present = [n for n in names if n in set(sub.weights)]
    if len(present) < 2:
        return None
    lo = max(sub[sub.weights == n].emitted.min() for n in present)
    hi = min(sub[sub.weights == n].emitted.max() for n in present)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return None
    rows = []
    for target in np.linspace(lo, hi, n_points):
        row = {"picks_emitted": int(round(target))}
        for n in present:
            d = sub[sub.weights == n].sort_values("emitted")
            row[n] = round(float(np.interp(target, d.emitted, d.recall)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ── loading a built sequence ─────────────────────────────────────────────────

def load_sequence(key: str):
    """Load for reference QA, logged separately from model scoring."""
    policy.record_access(key, "reference_qa", data_root=OUT_ROOT)
    return _load_sequence(key)


def _load_sequence(key: str):
    """streams {station: Stream}, reference {(station, phase): [UTCDateTime]},
    windows [(t0, t1)], picks DataFrame, for one built sequence."""
    from obspy import UTCDateTime, read
    d = OUT_ROOT / key
    picks = pd.read_parquet(d / "picks.parquet")
    wins = [(UTCDateTime(r.t0), UTCDateTime(r.t1)) for r in pd.read_csv(d / "windows.csv").itertuples()]
    streams = defaultdict(dict)
    for f in sorted((d / "waveforms").glob("*.mseed")):
        sta, band, t0s, t1s = f.stem.split("__")
        streams[(t0s, t1s)][sta] = read(str(f))
    out = []
    for (t0, t1) in wins:
        tag = (t0.strftime("%Y%m%dT%H%M%S"), t1.strftime("%Y%m%dT%H%M%S"))
        st = streams.get(tag, {})
        ref = reference_from(picks, list(st), t0, t1)
        out.append(dict(t0=t0, t1=t1, streams=st, reference=ref))
    return out, picks


def score(key: str, models: dict):
    """Pick every window with every model; return (bench rows at 0.3, sweep rows)."""
    policy.authorize_scoring([key])
    access_id = policy.record_access(
        key, "model_scoring", data_root=OUT_ROOT,
        models={name: policy.model_fingerprint(model) for name, model in models.items()},
        settings=dict(detect_floor=DETECT_FLOOR, report_threshold=REPORT_THRESHOLD,
                      threshold_sweep=THRESHOLD_SWEEP, match_tol_s=MATCH_TOL),
    )
    windows, _ = _load_sequence(key)
    label = reg.BY_KEY[key]["label"]
    bench, sweep = [], []
    for w in windows:
        store = {}
        for name, model in models.items():
            per = defaultdict(list)
            for sta, st in w["streams"].items():
                try:
                    out = model.classify(st, P_threshold=DETECT_FLOOR, S_threshold=DETECT_FLOOR)
                except Exception as exc:  # noqa: BLE001
                    print(f"    {name} {sta}: {type(exc).__name__}"); continue
                for p in out.picks:
                    per[(sta, p.phase)].append((p.peak_time, float(p.peak_value)))
            store[name] = dict(per)
        for name in models:
            for thr in THRESHOLD_SWEEP + [REPORT_THRESHOLD]:
                view = at_threshold(store[name], thr)
                for phase in ("P", "S"):
                    hit = tot = emitted = extra = 0; residuals = []
                    for sta in w["streams"]:
                        ref = w["reference"].get((sta, phase), []); got = view.get((sta, phase), [])
                        r, ex = match(ref, got); hit += len(r); tot += len(ref); emitted += len(got); extra += ex; residuals += r
                    row = dict(sequence=label, key=key, access_id=access_id, window=str(w["t0"]), weights=name, phase=phase, thr=thr,
                               analyst=tot, matched=hit, recall=(hit / tot if tot else np.nan), emitted=emitted, extra=extra,
                               MAE=(float(np.mean(np.abs(residuals))) if residuals else np.nan),
                               bias=(float(np.median(residuals)) if residuals else np.nan))
                    sweep.append(row)
                    if thr == REPORT_THRESHOLD:
                        bench.append(row)
    return pd.DataFrame(bench), pd.DataFrame(sweep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence", action="append", default=[])
    selection.add_argument("--all", action="store_true", help="Score built regression/dev sequences only")
    ap.add_argument("--weights", nargs="+", default=["jma_wc"], help="SeisBench PhaseNet weight names, or paths to converted .pt/.json pairs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    keys = [key for key in policy.routine_keys(reg.BY_KEY)
            if (OUT_ROOT / key / "manifest.json").exists()] if a.all else a.sequence
    try:
        policy.authorize_scoring(keys)
        if not keys:
            raise ValueError("No built regression/dev sequences available")
    except (PermissionError, ValueError) as exc:
        ap.error(str(exc))
    # Authorize the entire selection before loading any weights or waveform data.
    import seisbench.models as sbm
    models = {}
    for w in a.weights:
        models[w] = sbm.PhaseNet.from_pretrained(w) if not Path(w).exists() else sbm.PhaseNet.load(Path(w))
    benches, sweeps = [], []
    for k in keys:
        print(f"== {k}")
        b, s = score(k, models); benches.append(b); sweeps.append(s)
        if len(b):
            print(b.pivot_table(index=["window", "phase"], columns="weights", values="recall").round(3).to_string())
    bench = pd.concat(benches, ignore_index=True); sweep = pd.concat(sweeps, ignore_index=True)
    if a.out:
        bench.to_csv(a.out, index=False); sweep.to_csv(str(a.out).replace(".csv", "_sweep.csv"), index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
