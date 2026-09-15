#!/usr/bin/env python3
"""
certify_evaluability.py  (checkpoint 37A)

Metadata-only eligibility of every built held-out case for two scoring
endpoints, pick_scoring and network_event_scoring. Reads the committed
reference files under data/heldout_testset/<key>/ and, when the MiniSEED
files are reachable, measures the waveform coverage of the fetched stations
inside the scoring windows. No model output is read or produced.

Outputs
  data/heldout_testset/evaluability.csv                 one row per case
  data/heldout_testset/<key>/evaluability_stations.csv  one row per registered
                                                        station and window

Rules (defaults printed at run time)
  provenance tier       mode x reference_ok of a pick: manual_ok, unknown_ok,
                        automatic_rejected, unknown_rejected, other
  reviewed-or-manual    manual_ok, plus unknown_ok when the case's bulletin is
                        recorded as reviewed in BULLETIN_REVIEWED
  covered station       fetched and, when measured, three-component common
                        coverage >= min_coverage_fraction of the window; when
                        the MiniSEED files are not reachable, fetched stations
                        count as covered and waveforms_measured is False
  reference in coverage a pick inside a three-component covered interval of
                        its station, duplicates collapsed at match_tol_s per
                        station and phase (the scorer's rule)
  not_evaluable         no_waveforms: no fetched station
                        zero_overlap: no reference_ok pick in coverage
                        single_station: fewer than two covered stations carry
                        reference_ok picks
  pick_scoring          not not_evaluable, >= min_stations_pick covered stations
                        with reviewed-or-manual picks, and >= min_refs_per_phase
                        reviewed-or-manual picks in coverage for P and for S
  network_event_scoring not not_evaluable, >= min_covered_stations covered
                        stations with reviewed-or-manual picks, and
                        >= min_events_3sta events with reviewed-or-manual P
                        picks in coverage on >= 3 covered stations
  verdict               both | pick_scoring | network_event_scoring | neither

Regression and dev cases get certified_by = 37A. Every other role gets
provisional = True and no certification; 37B freezes the acceptance panel.

Usage
  python scripts/certify_evaluability.py --waveform-root /path/to/clone/data/heldout_testset
  python scripts/certify_evaluability.py --sequence samos_2020 --min-refs-per-phase 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_policy as policy  # noqa: E402
import heldout_testset_registry as reg  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "heldout_testset"
CERTIFYING_ROLES = {"regression", "dev"}
CHECKPOINT = "37A"

DEFAULT_RULES = dict(
    min_refs_per_phase=100,      # N: reviewed-or-manual picks per phase in coverage
    min_events_3sta=10,          # M: events with P on >= 3 covered stations
    min_covered_stations=3,      # K: covered stations with picks, network endpoint
    min_stations_pick=2,         # covered stations with picks, pick endpoint
    min_coverage_fraction=0.5,   # 3-component coverage / window length
    match_tol_s=reg.MATCH_TOL_S,  # duplicate collapse per station and phase
)

TIERS = ["manual_ok", "unknown_ok", "automatic_rejected", "unknown_rejected", "other"]
PHASES = ["P", "S"]

# Unknown-mode ISC picks are reference_ok; whether the bulletin behind them is
# the reviewed one is taken from docs/2026-09-08_heldout_test_cases.md ("Data
# status") and the registry notes, not re-verified against ISC. Cases absent
# here have no unknown_ok picks or are treated as not reviewed.
BULLETIN_REVIEWED = {
    "petrinja_2020": True, "la_palma_2021": True, "fagradalsfjall_2021": True,
    "kahramanmaras_2023": False, "noto_2024": False, "hualien_2024": False, "reykjanes_2023": False,
}

STATION_COLUMNS = [
    "key", "window", "t0", "t1", "station", "network", "code", "band", "rate", "km", "fetched",
    "file", "measured", "n_locations", "n_channels", "rates_hz", "covered_any_s", "covered_3c_s",
    "coverage_fraction_3c", "gap_count", "gap_s", "covered", "read_error",
    "ref_P_window", "ref_S_window", "ref_P_covered", "ref_S_covered", "ref_P_covered_rm", "ref_S_covered_rm",
] + [f"{tier}_{phase}" for tier in TIERS for phase in PHASES]


# ── intervals ────────────────────────────────────────────────────────────────

def _union(intervals, tol=0.0):
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _clip(intervals, lo, hi):
    return [[max(a, lo), min(b, hi)] for a, b in intervals if min(b, hi) > max(a, lo)]


def _length(intervals):
    return float(sum(b - a for a, b in intervals))


def _complement(intervals, lo, hi, tol=0.0):
    """Uncovered parts of [lo, hi] longer than tol."""
    gaps, cursor = [], lo
    for a, b in intervals:
        if a - cursor > tol:
            gaps.append([cursor, a])
        cursor = max(cursor, b)
    if hi - cursor > tol:
        gaps.append([cursor, hi])
    return gaps


def _inside(times, intervals):
    """Boolean mask: which epoch times fall inside the sorted intervals."""
    times = np.asarray(times, dtype=float)
    if not len(intervals) or not len(times):
        return np.zeros(len(times), dtype=bool)
    starts = np.array([a for a, _ in intervals])
    ends = np.array([b for _, b in intervals])
    idx = np.searchsorted(starts, times, side="right") - 1
    ok = idx >= 0
    ok[ok] = times[ok] <= ends[idx[ok]]
    return ok


def measure_file(path, t0, t1):
    """Coverage of one station-window MiniSEED file inside [t0, t1] (epoch s).

    Per channel the trace spans are merged with a one-sample tolerance and
    clipped to the window. Three-component coverage is, per location code,
    the intersection of the three channels with the longest coverage; the
    station's covered intervals are the union over locations. Gaps are the
    uncovered parts of the window in that three-component coverage.
    """
    from obspy import read

    st = read(str(path), headonly=True)
    by_chan, rates = {}, set()
    for tr in st:
        sr = float(tr.stats.sampling_rate)
        if sr <= 0 or tr.stats.npts == 0:
            continue
        rates.add(sr)
        a = float(tr.stats.starttime.timestamp)
        b = float(tr.stats.endtime.timestamp) + 1.0 / sr
        by_chan.setdefault((tr.stats.location, tr.stats.channel, sr), []).append([a, b])
    any_cov, per_loc = [], {}
    for (loc, _chan, sr), spans in by_chan.items():
        merged = _clip(_union(spans, tol=1.0 / sr), t0, t1)
        any_cov += merged
        per_loc.setdefault(loc, []).append(merged)
    three = []
    for chans in per_loc.values():
        if len(chans) < 3:
            continue
        best = sorted(chans, key=_length, reverse=True)[:3]
        common = best[0]
        for c in best[1:]:
            common = _intersect(common, c)
        three += common
    three = _union(three)
    tol = 1.0 / min(rates) if rates else 0.0
    gaps = _complement(three, t0, t1, tol=tol)
    return dict(
        n_locations=len(per_loc), n_channels=len(by_chan),
        rates_hz="|".join(f"{r:g}" for r in sorted(rates)),
        min_rate=min(rates) if rates else np.nan,
        covered_any_s=_length(_union(any_cov)), covered_3c_s=_length(three),
        gap_count=len(gaps), gap_s=_length(gaps), intervals_3c=three,
    )


# ── picks ────────────────────────────────────────────────────────────────────

def tier_of(mode, ok):
    mode = "unknown" if mode is None or (isinstance(mode, float) and np.isnan(mode)) else str(mode)
    tier = f"{mode}_{'ok' if bool(ok) else 'rejected'}"
    return tier if tier in TIERS else "other"


def dedup_count(times, tol):
    """Number of picks left after collapsing neighbours within tol seconds."""
    kept, last = 0, None
    for t in sorted(float(x) for x in times):
        if last is None or t - last > tol:
            kept += 1
            last = t
    return kept


def dedup_by_station_phase(df, tol):
    if not len(df):
        return {phase: 0 for phase in PHASES}
    out = {phase: 0 for phase in PHASES}
    for (_sta, phase), g in df.groupby(["station", "phase"]):
        if phase in out:
            out[phase] += dedup_count(g["epoch"].to_numpy(), tol)
    return out


def _epoch(series):
    return (pd.to_datetime(series, utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def _bool_col(series):
    return series.map(lambda v: str(v).strip().lower() == "true").to_numpy()


def load_windows(case_dir):
    w = pd.read_csv(case_dir / "windows.csv")
    return [(pd.Timestamp(a).tz_convert("UTC") if pd.Timestamp(a).tzinfo else pd.Timestamp(a, tz="UTC"),
             pd.Timestamp(b).tz_convert("UTC") if pd.Timestamp(b).tzinfo else pd.Timestamp(b, tz="UTC"))
            for a, b in zip(w["t0"], w["t1"])]


def window_tag(t0, t1):
    return f"{t0.strftime('%Y%m%dT%H%M%S')}__{t1.strftime('%Y%m%dT%H%M%S')}"


def resolution_category(station, network, fetched, ambiguous_targets):
    if station in ambiguous_targets:
        return "ambiguous"
    if station in fetched:
        return "fetched"
    if network:
        return "unfetched"
    return "unresolved"


# ── one case ─────────────────────────────────────────────────────────────────

def certify_case(key, case_dir, waveform_dir, rules, role, log_path=None):
    """Compute the case row and the per-station table. Records a reference_qa access first."""
    access_id = policy.record_access(key, "reference_qa", data_root=case_dir.parent, log_path=log_path,
                                     settings=dict(rules, checkpoint=CHECKPOINT,
                                                   certify_evaluability_sha256=policy.file_hash(Path(__file__))))
    tol = float(rules["match_tol_s"])
    picks = pd.read_parquet(case_dir / "picks.parquet")
    catalog = pd.read_parquet(case_dir / "catalog.parquet") if (case_dir / "catalog.parquet").exists() \
        else pd.DataFrame(columns=["event", "origin", "mag"])
    stations = pd.read_csv(case_dir / "stations.csv")
    stations["fetched"] = _bool_col(stations["fetched"]) if "fetched" in stations else False
    smap = pd.read_csv(case_dir / "station_map.csv", dtype=str).fillna("") if (case_dir / "station_map.csv").exists() \
        else pd.DataFrame(columns=["code", "station", "ambiguous"])
    windows = load_windows(case_dir)
    manifest = json.loads((case_dir / "manifest.json").read_text()) if (case_dir / "manifest.json").exists() else {}
    registry = reg.BY_KEY.get(key, {})

    picks = picks.copy()
    picks["network"] = picks["network"].fillna("").astype(str) if "network" in picks else ""
    picks["reference_ok"] = picks["reference_ok"].astype(bool) if "reference_ok" in picks else (picks["mode"] == "manual")
    picks["epoch"] = _epoch(picks["time"])
    picks["tier"] = [tier_of(m, ok) for m, ok in zip(picks["mode"], picks["reference_ok"])]
    bulletin_reviewed = bool(BULLETIN_REVIEWED.get(key, False))
    picks["rm"] = (picks["tier"] == "manual_ok") | ((picks["tier"] == "unknown_ok") & bulletin_reviewed)

    fetched = set(stations.loc[stations["fetched"], "station"])
    ambiguous_targets = set(smap.loc[smap["ambiguous"].astype(str) != "", "station"]) if len(smap) else set()
    code_of_fetched = {str(c): s for s, c in zip(stations["station"], stations["code"]) if s in fetched}
    picks["category"] = [resolution_category(s, n, fetched, ambiguous_targets)
                         for s, n in zip(picks["station"], picks["network"])]

    # window membership: first window containing the pick, -1 otherwise
    win_idx = np.full(len(picks), -1)
    for i, (t0, t1) in enumerate(windows):
        m = (win_idx < 0) & (picks["epoch"] >= t0.timestamp()) & (picks["epoch"] <= t1.timestamp())
        win_idx[m] = i
    picks["window"] = win_idx
    inwin = picks[picks["window"] >= 0]

    # waveform coverage per fetched station and window
    measured_any = waveform_dir.is_dir()
    station_rows, coverage = [], {}   # coverage[(station, window)] -> intervals (epoch)
    fetched_seconds = 0.0             # window seconds expected from fetched stations
    for i, (t0, t1) in enumerate(windows):
        lo, hi = t0.timestamp(), t1.timestamp()
        for r in stations.itertuples():
            fetched_seconds += (hi - lo) if r.fetched else 0.0
            fname = f"{r.station}__{r.band}__{window_tag(t0, t1)}.mseed"
            path = waveform_dir / fname
            row = dict(key=key, window=i, t0=t0.isoformat(), t1=t1.isoformat(), station=r.station,
                       network=r.network, code=r.code, band=r.band, rate=r.rate, km=r.km, fetched=bool(r.fetched),
                       file=fname, measured=False, n_locations=np.nan, n_channels=np.nan, rates_hz="",
                       covered_any_s=np.nan, covered_3c_s=np.nan, coverage_fraction_3c=np.nan,
                       gap_count=np.nan, gap_s=np.nan, covered=False, read_error="")
            if r.fetched and measured_any:
                row["measured"] = True
                m = None
                if path.is_file():
                    try:
                        m = measure_file(path, lo, hi)
                    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the census
                        row["read_error"] = f"{type(exc).__name__}: {exc}"
                        print(f"    {key} window {i} {r.station}: unreadable MiniSEED, counted as a full gap "
                              f"({type(exc).__name__})", file=sys.stderr)
                if m is not None:
                    row.update({k: v for k, v in m.items() if k in row})
                    row["coverage_fraction_3c"] = m["covered_3c_s"] / (hi - lo)
                    intervals = m["intervals_3c"]
                else:
                    # missing or unreadable: a full-window gap, never a silent pass
                    row.update(n_locations=0, n_channels=0, covered_any_s=0.0, covered_3c_s=0.0,
                               coverage_fraction_3c=0.0, gap_count=1, gap_s=hi - lo)
                    intervals = []
                row["covered"] = row["coverage_fraction_3c"] >= rules["min_coverage_fraction"]
            elif r.fetched:
                intervals = [[lo, hi]]
                row["covered"] = True
            else:
                intervals = []
            coverage[(r.station, i)] = intervals
            sel = inwin[(inwin["station"] == r.station) & (inwin["window"] == i) & (inwin["category"] != "ambiguous")]
            ok = sel[sel["reference_ok"]]
            cov = ok[_inside(ok["epoch"].to_numpy(), intervals)] if len(ok) else ok
            rm = cov[cov["rm"]]
            for phase in PHASES:
                row[f"ref_{phase}_window"] = dedup_count(ok.loc[ok["phase"] == phase, "epoch"], tol)
                row[f"ref_{phase}_covered"] = dedup_count(cov.loc[cov["phase"] == phase, "epoch"], tol)
                row[f"ref_{phase}_covered_rm"] = dedup_count(rm.loc[rm["phase"] == phase, "epoch"], tol)
                for tier in TIERS:
                    row[f"{tier}_{phase}"] = int(((sel["tier"] == tier) & (sel["phase"] == phase)).sum())
            station_rows.append(row)
    station_table = pd.DataFrame(station_rows, columns=STATION_COLUMNS)

    # reference picks in coverage (fetched, non-ambiguous, inside 3c intervals)
    on_fetched = inwin[inwin["category"] == "fetched"]
    in_cov = np.zeros(len(on_fetched), dtype=bool)
    for (sta, i), intervals in coverage.items():
        m = ((on_fetched["station"] == sta) & (on_fetched["window"] == i)).to_numpy()
        if m.any():
            in_cov[m] = _inside(on_fetched.loc[m, "epoch"].to_numpy(), intervals)
    covered_picks = on_fetched[in_cov]
    ok_fetched = on_fetched[on_fetched["reference_ok"]]
    ok_covered = covered_picks[covered_picks["reference_ok"]]
    rm_covered = covered_picks[covered_picks["rm"]]

    fetched_dedup = dedup_by_station_phase(ok_fetched, tol)
    covered_dedup = dedup_by_station_phase(ok_covered, tol)
    rm_dedup = dedup_by_station_phase(rm_covered, tol)

    fetched_rows = station_table[station_table["fetched"]]
    covered_rows = fetched_rows[fetched_rows["covered"]]
    stations_covered = covered_rows["station"].nunique()
    stations_covered_with_refs = covered_rows.loc[(covered_rows["ref_P_covered"] + covered_rows["ref_S_covered"]) > 0,
                                                  "station"].nunique()
    stations_covered_with_rm = covered_rows.loc[(covered_rows["ref_P_covered_rm"] + covered_rows["ref_S_covered_rm"]) > 0,
                                                "station"].nunique()

    # event support
    window_s = float(sum((t1 - t0).total_seconds() for t0, t1 in windows))
    cat_epoch = _epoch(catalog["origin"]) if len(catalog) else np.array([])
    cat_in = np.zeros(len(catalog), dtype=bool)
    for t0, t1 in windows:
        cat_in |= (cat_epoch >= t0.timestamp()) & (cat_epoch <= t1.timestamp())
    cat_win = catalog[cat_in]
    mags = pd.to_numeric(cat_win["mag"], errors="coerce") if len(cat_win) else pd.Series(dtype=float)
    p_fetched = ok_fetched[ok_fetched["phase"] == "P"].groupby("event")["station"].nunique()
    p_rm = rm_covered[rm_covered["phase"] == "P"].groupby("event")["station"].nunique()
    events_with_refs_fetched = int(ok_fetched["event"].nunique())
    events_3sta_fetched = int((p_fetched >= 3).sum())
    events_3sta_rm = int((p_rm >= 3).sum())

    covered_hours = float(fetched_rows["covered_3c_s"].sum() / 3600.0) if measured_any else np.nan
    n_fetched = int(stations["fetched"].sum())
    n_ref_fetched = fetched_dedup["P"] + fetched_dedup["S"]
    density_fetched = n_ref_fetched / (n_fetched * window_s / 3600.0) if n_fetched and window_s else np.nan
    density_covered = (covered_dedup["P"] + covered_dedup["S"]) / covered_hours \
        if measured_any and covered_hours and covered_hours > 0 else np.nan

    unresolved = inwin[inwin["category"] == "unresolved"]
    top_unresolved = Counter(unresolved["station"]).most_common(5)
    unresolved_matching_fetched = int(unresolved["station"].map(lambda s: str(s) in code_of_fetched).sum())

    row = dict(
        key=key, label=registry.get("label", manifest.get("label", key)), role=role,
        regime=registry.get("regime", ""), certified_by=CHECKPOINT if role in CERTIFYING_ROLES else "",
        provisional=role not in CERTIFYING_ROLES,
        n_windows=len(windows), window_s_total=window_s, window_hours_total=window_s / 3600.0,
        stations_registered=int(len(stations)), stations_fetched=n_fetched,
        stations_covered=int(stations_covered), stations_covered_with_refs=int(stations_covered_with_refs),
        stations_covered_with_rm_refs=int(stations_covered_with_rm),
        waveforms_measured=bool(measured_any),
        files_expected=int(len(fetched_rows)),
        files_present=int(sum((waveform_dir / f).is_file() for f in fetched_rows["file"])) if measured_any else np.nan,
        covered_3c_s_total=float(fetched_rows["covered_3c_s"].sum()) if measured_any else np.nan,
        covered_3c_fraction=(float(fetched_rows["covered_3c_s"].sum()) / fetched_seconds)
        if measured_any and fetched_seconds else np.nan,
        gap_count_total=int(fetched_rows["gap_count"].sum()) if measured_any else np.nan,
        gap_s_total=float(fetched_rows["gap_s"].sum()) if measured_any else np.nan,
        stations_rate_lt_100hz=int(sum(
            (min(float(x) for x in r.rates_hz.split("|")) < 100.0) if (measured_any and r.rates_hz)
            else (float(r.rate) < 100.0 if r.rate == r.rate else False)
            for r in fetched_rows.drop_duplicates("station").itertuples())),
        sources="|".join(sorted(map(str, picks["source"].unique()))) if "source" in picks else "",
        picks_total=int(len(picks)), picks_in_windows=int(len(inwin)),
        res_fetched=int((inwin["category"] == "fetched").sum()),
        res_unfetched=int((inwin["category"] == "unfetched").sum()),
        res_ambiguous=int((inwin["category"] == "ambiguous").sum()),
        res_unresolved=int((inwin["category"] == "unresolved").sum()),
        res_unresolved_code_matches_fetched=unresolved_matching_fetched,
        top_unresolved_codes="|".join(f"{c}:{n}" for c, n in top_unresolved),
        bulletin_unknown_mode_reviewed=(bulletin_reviewed if (inwin["tier"] == "unknown_ok").any() else "n/a"),
    )
    for scope, df in (("all", inwin), ("fetched", on_fetched)):
        for phase in PHASES:
            for tier in TIERS:
                row[f"ref_{scope}_{phase}_{tier}"] = int(((df["tier"] == tier) & (df["phase"] == phase)).sum())
    for phase in PHASES:
        row[f"ref_fetched_{phase}_dedup"] = int(fetched_dedup[phase])
        row[f"ref_covered_{phase}_dedup"] = int(covered_dedup[phase])
        row[f"ref_rm_covered_{phase}"] = int(rm_dedup[phase])
    row.update(
        events_catalog_in_windows=int(len(cat_win)),
        mag_min=float(mags.min()) if mags.notna().any() else np.nan,
        mag_max=float(mags.max()) if mags.notna().any() else np.nan,
        events_mag_missing=int(mags.isna().sum()) if len(cat_win) else 0,
        events_with_refs_fetched=events_with_refs_fetched,
        events_3sta_P_fetched=events_3sta_fetched,
        events_3sta_P_covered_rm=events_3sta_rm,
        ref_per_station_hour_fetched=density_fetched,
        ref_per_covered_hour=density_covered,
    )
    row.update(verdict(row, rules))
    row.update(
        rule_params=";".join(f"{k}={v}" for k, v in rules.items()),
        git_commit=_git_commit(), computed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        access_id=access_id,
    )
    return row, station_table


def verdict(row, rules):
    """Endpoint eligibility from the row's counts; rules are the parameters."""
    n, m, k, kp = (rules["min_refs_per_phase"], rules["min_events_3sta"],
                   rules["min_covered_stations"], rules["min_stations_pick"])
    if row["stations_fetched"] == 0:
        ne = "no_waveforms"
    elif row["ref_covered_P_dedup"] + row["ref_covered_S_dedup"] == 0:
        ne = "zero_overlap"
    elif row["stations_covered_with_refs"] < 2:
        ne = "single_station"
    else:
        ne = ""
    out = dict(not_evaluable_reason=ne)
    p_ok = row["ref_rm_covered_P"] >= n
    s_ok = row["ref_rm_covered_S"] >= n
    out["pick_scoring_P_ok"] = bool(p_ok and not ne)
    out["pick_scoring_S_ok"] = bool(s_ok and not ne)
    if ne:
        pick, pick_reason = False, ne
        net, net_reason = False, ne
    else:
        fails = []
        if row["stations_covered_with_rm_refs"] < kp:
            fails.append(f"covered stations with reviewed-or-manual picks {row['stations_covered_with_rm_refs']} < {kp}")
        if not p_ok:
            fails.append(f"P {row['ref_rm_covered_P']} < {n}")
        if not s_ok:
            fails.append(f"S {row['ref_rm_covered_S']} < {n}")
        pick, pick_reason = not fails, "; ".join(fails) if fails else \
            f"P {row['ref_rm_covered_P']} and S {row['ref_rm_covered_S']} >= {n} on {row['stations_covered_with_rm_refs']} stations"
        fails = []
        if row["stations_covered_with_rm_refs"] < k:
            fails.append(f"covered stations with reviewed-or-manual picks {row['stations_covered_with_rm_refs']} < {k}")
        if row["events_3sta_P_covered_rm"] < m:
            fails.append(f"events with 3-station P support {row['events_3sta_P_covered_rm']} < {m}")
        net, net_reason = not fails, "; ".join(fails) if fails else \
            f"{row['events_3sta_P_covered_rm']} events with 3-station P support on {row['stations_covered_with_rm_refs']} stations"
    out.update(pick_scoring_eligible=pick, pick_scoring_reason=pick_reason,
               network_event_scoring_eligible=net, network_event_scoring_reason=net_reason,
               verdict="both" if pick and net else "pick_scoring" if pick else
               "network_event_scoring" if net else "neither")
    return out


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ── driver ───────────────────────────────────────────────────────────────────

def built_keys(data_root):
    return sorted(d.name for d in data_root.iterdir()
                  if d.is_dir() and all((d / f).is_file() for f in ("picks.parquet", "windows.csv", "stations.csv")))


def run(data_root, waveform_root=None, keys=None, rules=None, log_path=None, out=None, station_tables=True):
    rules = dict(DEFAULT_RULES, **(rules or {}))
    waveform_root = data_root if waveform_root is None else Path(waveform_root)
    roles = policy.load_policy()["roles"]
    keys = built_keys(data_root) if not keys else list(keys)
    unknown = [k for k in keys if k not in roles]
    if unknown:
        raise ValueError(f"Not in configs/evaluation_suites.json: {unknown}")
    print("rules: " + ", ".join(f"{k}={v}" for k, v in rules.items()))
    rows = []
    for key in keys:
        case_dir = data_root / key
        row, table = certify_case(key, case_dir, waveform_root / key / "waveforms", rules, roles[key], log_path)
        if station_tables:
            table.to_csv(case_dir / "evaluability_stations.csv", index=False, lineterminator="\n")
        rows.append(row)
        print(f"{key:<22} {row['role']:<10} fetched {row['stations_fetched']:>2} covered {row['stations_covered']:>2} "
              f"P/S rm-in-coverage {row['ref_rm_covered_P']:>5}/{row['ref_rm_covered_S']:<5} "
              f"events3sta {row['events_3sta_P_covered_rm']:>4}  -> {row['verdict']}"
              + (f" ({row['not_evaluable_reason']})" if row["not_evaluable_reason"] else ""))
    table = pd.DataFrame(rows)
    out = data_root / "evaluability.csv" if out is None else Path(out)
    table.to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {out} ({len(table)} rows)")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--waveform-root", type=Path, default=None,
                    help="directory holding <key>/waveforms/*.mseed (default: --data-root); read only")
    ap.add_argument("--sequence", action="append", default=[], help="restrict to these keys (default: every built case)")
    ap.add_argument("--access-log", type=Path, default=None, help="reference_qa log (default: evaluation_policy.ACCESS_LOG)")
    ap.add_argument("--out", type=Path, default=None, help="case table (default: <data-root>/evaluability.csv)")
    ap.add_argument("--no-station-tables", action="store_true")
    for name, default in DEFAULT_RULES.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    a = ap.parse_args(argv)
    rules = {name: getattr(a, name) for name in DEFAULT_RULES}
    run(a.data_root, a.waveform_root, a.sequence, rules, a.access_log, a.out, not a.no_station_tables)


if __name__ == "__main__":
    main()
