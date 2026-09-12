#!/usr/bin/env python3
"""
source_census.py  (checkpoint 39A of issue #39)

A stratified census of usable supervision, in two parts.

1. Bulletin operators (INGV, NOA, GeoNet, ISC, USGS ComCat, franceseisme):
   for every operator-year a deterministic day sample (default two days per
   quarter, chosen by a hash of operator, year, quarter and a seed tag, never
   by content), one day-sized arrivals query per sampled day along the query
   paths of scripts/build_heldout_testset.py (day query with includearrivals
   and 413 halving for NOA/ISC/franceseisme; day catalogue then one
   eventid query per event for INGV/GeoNet; ComCat catalogue then the
   phase-data product's QuakeML for USGS), and per sampled day: events,
   events with arrivals, manual/automatic/unknown P and S readings (from
   the pick's evaluation mode where the operator exposes it), distinct
   stations, readings per event, a magnitude histogram, the share of
   readings inside the held-out windows/places/years of
   scripts/heldout_sequences.py, and the acquisition cost (queries,
   seconds, bytes, failures). The operator-year total is the day mean
   times the days in the year, with a percentile bootstrap over sampled
   days (95 %). Raw responses are cached under data/census/raw/ (ignored),
   one file per request URL; the CSVs are committed.

2. SeisBench sources: data/census/seisbench_sources.csv from the committed
   records (notebooks/audit_results/summary_statistics.csv,
   notebooks/benchmark_pool_summary.csv, data/README.md, DATASET_CONFIGS of
   scripts/build_training_dataset.py and the metadata column record in
   notebooks/step_1_claude.ipynb). Held-out overlap and pick-status counts
   need the cache and are marked "server"; the server path is
       python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT
   which imports seisbench only then.

Usage (from the repository root):
    python scripts/source_census.py bulletin --operators INGV NOA --years 2018 2019 \
        --days-per-quarter 2 --max-minutes 20 --seed-tag 39a            # --sampler v2 (default) or v1
    python scripts/source_census.py summary
    python scripts/source_census.py seisbench
    python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT

Nothing is fetched at import; the tests monkeypatch fetch_day.
"""

from __future__ import annotations

import argparse
import ast
import calendar
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_sequences as hs  # noqa: E402  (pure pandas)

CENSUS_DIR = REPO_ROOT / "data" / "census"
RAW_DIR = CENSUS_DIR / "raw"
SUMMARY_STATS_CSV = REPO_ROOT / "notebooks" / "audit_results" / "summary_statistics.csv"
POOL_SUMMARY_CSV = REPO_ROOT / "notebooks" / "benchmark_pool_summary.csv"
BUILDER_PY = REPO_ROOT / "scripts" / "build_training_dataset.py"

N_BOOT = 2000
HTTP_TIMEOUT_S = 120
BACKOFF_S = (5, 15, 30)          # the builder's download() sleeps 15 * (i + 1); shorter here, the budget is the bound
MIN_PIECE_S = 900                # a 413 is halved down to 15 minutes, then recorded as a failure


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ── operators: the arrival paths scripts/build_heldout_testset.py has proven ──
# kind: "day" = one includearrivals query per sampled day, halved on 413 (harvest_fdsn_region);
#       "per_event" = day catalogue, then one eventid query per event (harvest_fdsn_per_event);
#       "usgs_phase_data" = ComCat catalogue, then the phase-data product's QuakeML per event.
# Service roots are obspy's URL_MAPPINGS (the builder's client_for), franceseisme as in the builder.
# KOERI, RESIF, IRIS and the EIDA/IRIS routing clients of NETWORK_ROUTE are waveform routes; the
# repository has no proven arrivals path for them, so they are not census operators.
OPERATORS = {
    "INGV": dict(kind="per_event", client="INGV", per_event_params={"includearrivals": "true"},
                 mode_note="evaluation_mode manual/automatic per pick"),
    "NOA": dict(kind="day", client="NOA", mode_note="evaluation_mode manual per pick"),
    "GEONET": dict(kind="per_event", client="GEONET", per_event_params={},
                   mode_note="per-event QuakeML carries the picks; evaluation_mode where set"),
    "ISC": dict(kind="day", client="ISC",
                mode_note="no evaluation mode; reviewed bulletin about two years behind real time"),
    "USGS": dict(kind="usgs_phase_data", client="USGS", mode_note="NEIC phase-data product; evaluation_mode where set"),
    "franceseisme": dict(kind="day", client="franceseisme", mode_note="evaluation_mode where set"),
}
FALLBACK_ROOTS = {"INGV": "http://webservices.ingv.it", "NOA": "http://eida.gein.noa.gr",
                  "GEONET": "http://service.geonet.org.nz", "ISC": "http://www.isc.ac.uk",
                  "USGS": "http://earthquake.usgs.gov", "franceseisme": "https://api.franceseisme.fr"}


def event_query_url(operator: str) -> str:
    """The fdsnws-event query URL, from obspy's mapping (what the builder's client_for uses)."""
    if operator == "franceseisme":
        return "https://api.franceseisme.fr/fdsnws/event/1/query"      # explicit in the builder's client_for
    try:
        from obspy.clients.fdsn.header import URL_MAPPINGS
        root = URL_MAPPINGS.get(OPERATORS[operator]["client"], FALLBACK_ROOTS[operator])
    except ImportError:
        root = FALLBACK_ROOTS[operator]
    return root.rstrip("/") + "/fdsnws/event/1/query"


# ── deterministic day sample ─────────────────────────────────────────────────

def quarter_days(year: int, quarter: int) -> list:
    m0 = 3 * (quarter - 1) + 1
    start = date(year, m0, 1)
    end = date(year + 1, 1, 1) if quarter == 4 else date(year, m0 + 3, 1)
    return [start + timedelta(days=i) for i in range((end - start).days)]


SAMPLERS = ("v1", "v2")
DEFAULT_SAMPLER = "v2"


def sample_days(operator: str, year: int, days_per_quarter: int, seed_tag: str, sampler: str = DEFAULT_SAMPLER) -> list:
    """[(quarter, date), ...]: days_per_quarter distinct days per quarter, chosen by
    operator, year, quarter and seed tag only (no content enters), the same inputs giving
    the same days. A quarter with at most days_per_quarter days is taken whole under
    either sampler; otherwise
      v2 (default): the quarter's days sorted by sha256(operator|year|Qq|seed_tag|date),
          the first days_per_quarter kept; one hash per day, and a larger
          days_per_quarter extends the smaller sample.
      v1: the rule of commit 7580636, kept so the committed 2026-09-12 demonstration
          (data/census/bulletin_*.csv, seed tag 39a) stays reproducible: for k = 0, 1, ...
          take days[sha256(operator|year|Qq|seed_tag|k)[:16] mod days-in-quarter],
          skipping repeats, until days_per_quarter days are in hand. The whole-quarter
          short-circuit bounds its worst case and returns the same days as the loop did.
    The two rules map the same seed tag to different days."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; known: {SAMPLERS}")
    out = []
    for q in (1, 2, 3, 4):
        days = quarter_days(year, q)
        if days_per_quarter >= len(days):
            chosen = days
        elif sampler == "v1":
            chosen, k = [], 0
            while len(chosen) < days_per_quarter:
                h = hashlib.sha256(f"{operator}|{year}|Q{q}|{seed_tag}|{k}".encode()).hexdigest()
                k += 1
                d = days[int(h[:16], 16) % len(days)]
                if d not in chosen:
                    chosen.append(d)
        else:
            chosen = sorted(days, key=lambda d: hashlib.sha256(
                f"{operator}|{year}|Q{q}|{seed_tag}|{d.isoformat()}".encode()).hexdigest())[:days_per_quarter]
        out += [(q, d) for d in sorted(chosen)]
    return out


def subset_events(event_ids: list, max_events, seed_tag: str) -> list:
    """Deterministic content-blind subset of a day's events for the per-event paths."""
    if max_events is None or len(event_ids) <= max_events:
        return list(event_ids)
    keyed = sorted(event_ids, key=lambda e: hashlib.sha256(f"{seed_tag}|{e}".encode()).hexdigest())
    return keyed[:max_events]


# ── HTTP with retries, backoff, 413 signalling, raw cache, cost accounting ────

class Budget:
    def __init__(self, max_minutes):
        self.t0 = time.time()
        self.limit = None if max_minutes is None else float(max_minutes) * 60.0

    def exhausted(self) -> bool:
        return self.limit is not None and (time.time() - self.t0) > self.limit


class Cost:
    """Per-day acquisition cost: every request made, retries, failures, seconds, bytes."""
    def __init__(self):
        self.n_queries = 0; self.n_retries = 0; self.n_failed = 0; self.n_cached = 0
        self.seconds = 0.0; self.bytes = 0

    def add(self, other):
        for k in ("n_queries", "n_retries", "n_failed", "n_cached", "seconds", "bytes"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self


class TooLarge(Exception):
    pass


class BadRequest(Exception):
    pass


def _is_too_large(code, body: str) -> bool:
    b = body.lower()
    return code == 413 or "too much" in b or "request too large" in b or "too many" in b


def cache_paths(url: str, dest: Path) -> tuple:
    """(body_path, meta_path) of one request: `dest` with the first 12 hex digits of
    sha256(url) inserted before its suffix (catalog.xml -> catalog.<hash>.xml), so two
    requests for the same day that differ only in their parameters (minmagnitude,
    includearrivals, ...) never share a cache file. The sidecar records the full URL."""
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    body = dest.with_name(f"{dest.stem}.{h}{dest.suffix}")
    return body, body.with_suffix(body.suffix + ".meta.json")


def http_get(url: str, dest: Path, cost: Cost, timeout: int = HTTP_TIMEOUT_S, tries: int = 3) -> tuple:
    """GET url into the file cache_paths(url, dest) names (with a .meta.json sidecar:
    status, seconds, bytes, when, url). A cached response is reused only when its sidecar
    records this exact URL, query string included; its recorded cost is then counted
    again, so a rerun reports the same acquisition cost. A cached file whose sidecar
    records another URL is refetched, not reused. Returns (status_code, path or None).
    204 -> (204, None). 413 / 'too much' -> TooLarge. 400 -> BadRequest (no retry).
    Other HTTP errors, URL errors and timeouts are retried with backoff."""
    dest, meta_path = cache_paths(url, dest)
    if dest.exists() and meta_path.exists():
        m = json.loads(meta_path.read_text())
        if m.get("url") == url:
            cost.n_queries += 1; cost.n_cached += 1; cost.seconds += m["seconds"]; cost.bytes += m["bytes"]
            if m["status"] == 204:
                return 204, None
            return m["status"], dest
        log(f"    cache {dest.name} records another URL ({str(m.get('url'))[:80]}); refetching")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for i in range(tries):
        cost.n_queries += 1
        if i:
            cost.n_retries += 1
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "phasenet-retrain source_census (obspy-compatible)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read(); status = r.status
            dt = time.time() - t0
            cost.seconds += dt; cost.bytes += len(body)
            if status == 204 or not body.strip():
                meta_path.write_text(json.dumps(dict(status=204, seconds=dt, bytes=len(body), url=url,
                                                     when=datetime.now(timezone.utc).isoformat(timespec="seconds"))))
                dest.write_bytes(b"")
                return 204, None
            dest.write_bytes(body)
            meta_path.write_text(json.dumps(dict(status=status, seconds=dt, bytes=len(body), url=url,
                                                 when=datetime.now(timezone.utc).isoformat(timespec="seconds"))))
            return status, dest
        except urllib.error.HTTPError as exc:
            dt = time.time() - t0; cost.seconds += dt
            try:
                text = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                text = ""
            cost.bytes += len(text)
            if exc.code == 204:
                return 204, None
            if _is_too_large(exc.code, text):
                raise TooLarge(f"{exc.code}: {text[:120]}")
            if exc.code == 400:
                cost.n_failed += 1
                raise BadRequest(f"400: {text[:160]}")
            last = f"HTTP {exc.code}: {text[:100]}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            cost.seconds += time.time() - t0
            last = f"{type(exc).__name__}: {str(exc)[:100]}"
        log(f"    {url[:90]} attempt {i + 1}: {last}")
        if i + 1 < tries:
            time.sleep(BACKOFF_S[min(i, len(BACKOFF_S) - 1)])
    cost.n_failed += 1
    raise RuntimeError(f"failed after {tries} tries: {last}")


def _read_events(path: Path):
    from obspy import read_events
    return read_events(str(path))


def _query(base: str, **params) -> str:
    return base + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S")


# ── fetching one sampled day ─────────────────────────────────────────────────

def fetch_day(operator: str, day: date, raw_dir: Path, cost: Cost, budget: Budget,
              min_magnitude=None, max_events=None, seed_tag: str = "") -> dict:
    """One sampled day for one operator, along the builder's path for that operator.
    Returns dict(catalog=DataFrame[event, origin, lat, lon, mag, source],
                 picks=DataFrame[event, station, phase, time, mode, status, ...],
                 events_fetched=int, note=str). Raises on a failed day."""
    import build_heldout_testset as bht   # _pick_rows, _catalog_rows, _event_id: the QuakeScope rows
    spec = OPERATORS[operator]
    base = event_query_url(operator)
    ddir = raw_dir / operator / f"{day.year}" / day.isoformat()
    t0 = datetime(day.year, day.month, day.day)
    t1 = t0 + timedelta(days=1)
    label = day.isoformat()
    cat_rows, pick_rows, notes = [], [], []

    if spec["kind"] == "day":
        pieces, seen = [(t0, t1)], set()
        while pieces:
            a, b = pieces.pop(0)
            if budget.exhausted():
                raise RuntimeError("budget exhausted")
            url = _query(base, starttime=_fmt(a), endtime=_fmt(b), includearrivals="true", minmagnitude=min_magnitude)
            dest = ddir / f"day_{a.strftime('%H%M')}_{b.strftime('%H%M') if b.date() == a.date() else '2400'}.xml"
            try:
                status, path = http_get(url, dest, cost)
            except TooLarge as exc:
                if (b - a).total_seconds() <= MIN_PIECE_S:
                    cost.n_failed += 1; notes.append(f"413 at {MIN_PIECE_S} s piece {a}..{b}: {exc}")
                    continue
                m = a + (b - a) / 2
                pieces = [(a, m), (m, b)] + pieces
                notes.append(f"413 split {a.strftime('%H:%M')}..{b.strftime('%H:%M') if b.date() == a.date() else '24:00'}")
                continue
            if path is None:
                continue
            for ev in _read_events(path):
                eid = bht._event_id(ev)
                if eid in seen:
                    continue
                seen.add(eid)
                cat_rows += bht._catalog_rows([ev], operator)
                pick_rows += bht._pick_rows(ev, label, operator)
        n_fetched = len(cat_rows)

    elif spec["kind"] in ("per_event", "usgs_phase_data"):
        if budget.exhausted():
            raise RuntimeError("budget exhausted")
        url = _query(base, starttime=_fmt(t0), endtime=_fmt(t1), minmagnitude=min_magnitude)
        status, path = http_get(url, ddir / "catalog.xml", cost)
        events = list(_read_events(path)) if path is not None else []
        by_id = {}
        for ev in events:
            eid = bht._event_id(ev).split("&")[0]
            by_id.setdefault(eid, ev)
        for eid, ev in by_id.items():
            for r in bht._catalog_rows([ev], operator):
                r["event"] = eid
                cat_rows.append(r)
        chosen = subset_events(list(by_id), max_events, seed_tag)
        n_fetched = 0
        for eid in chosen:
            if budget.exhausted():
                notes.append(f"budget exhausted after {n_fetched} of {len(chosen)} events")
                break
            try:
                if spec["kind"] == "per_event":
                    eurl = _query(base, eventid=eid, **spec["per_event_params"])
                    status, path = http_get(eurl, ddir / f"event_{eid}.xml", cost)
                else:
                    gurl = _query(base, eventid=eid, format="geojson")
                    status, gpath = http_get(gurl, ddir / f"event_{eid}.geojson", cost)
                    if gpath is None:
                        n_fetched += 1; continue
                    d = json.loads(gpath.read_text())
                    prods = d["properties"]["products"].get("phase-data", [])
                    if not prods:
                        n_fetched += 1; continue
                    c = prods[0]["contents"]
                    qml = next(k for k in c if k.endswith("quakeml.xml"))
                    status, path = http_get(c[qml]["url"], ddir / f"event_{eid}.quakeml.xml", cost)
            except (BadRequest, RuntimeError) as exc:
                notes.append(f"event {eid}: {exc}"[:160]); continue
            n_fetched += 1
            if path is None:
                continue
            for e in _read_events(path):
                rows = bht._pick_rows(e, label, operator if spec["kind"] == "per_event" else "USGS phase-data")
                for r in rows:
                    r["event"] = eid
                pick_rows += rows
    else:
        raise ValueError(spec["kind"])

    cat = pd.DataFrame(cat_rows, columns=["event", "origin", "lat", "lon", "depth_km", "mag", "source"])
    picks = pd.DataFrame(pick_rows, columns=bht.PICK_COLS)
    return dict(catalog=cat, picks=picks, events_fetched=int(n_fetched), note="; ".join(notes))


# ── per-day counting ─────────────────────────────────────────────────────────

MAG_BINS = [-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf]
MAG_LABELS = ["mag_lt1", "mag_1_2", "mag_2_3", "mag_3_4", "mag_ge4"]
TIME_WINDOWS = [n for n in hs.WINDOW_NAMES if n not in hs.PLACE_NAMES]
SCALED = ["events_with_arrivals", "p_manual", "s_manual", "p_automatic", "s_automatic",
          "p_unknown", "s_unknown", "p_total", "s_total", "readings", "readings_heldout_any"]
DAY_METRICS = ["events", "events_with_arrivals", "p_manual", "s_manual", "p_automatic", "s_automatic",
               "p_unknown", "s_unknown", "readings"]


def _mode(m) -> str:
    m = str(m).lower() if m is not None and m == m else "unknown"
    if m in ("manual", "automatic"):
        return m
    return "unknown"


def heldout_flags(catalog: pd.DataFrame) -> pd.DataFrame:
    """Per event: in_window (time-bounded hold-out), in_place (place hold-out), in_year
    (2016/2021), any, and the semicolon-joined names of the windows hit."""
    if len(catalog) == 0:
        return pd.DataFrame(columns=["event", "in_window", "in_place", "in_year", "any", "windows"])
    hits = hs.window_hits(catalog["origin"], catalog["lat"], catalog["lon"])
    hits.index = catalog.index
    in_window = hits[TIME_WINDOWS].any(axis=1) if TIME_WINDOWS else pd.Series(False, index=catalog.index)
    in_place = hits[hs.PLACE_NAMES].any(axis=1) if hs.PLACE_NAMES else pd.Series(False, index=catalog.index)
    in_year = hs.holdout_year_mask(catalog["origin"])
    in_year.index = catalog.index
    names = np.array(hs.WINDOW_NAMES, dtype=object)
    windows = hits.apply(lambda r: ";".join(names[r.to_numpy(dtype=bool)]), axis=1)
    return pd.DataFrame(dict(event=catalog["event"].values, in_window=in_window.values, in_place=in_place.values,
                             in_year=in_year.values, any=(in_window | in_place | in_year).values,
                             windows=windows.values))


def count_day(catalog: pd.DataFrame, picks: pd.DataFrame, scale: float = 1.0) -> dict:
    """Counts for one sampled day. `scale` = events / events_fetched on the per-event
    paths; the *_est columns are the observed counts times scale (day estimate)."""
    out = dict(events=int(catalog["event"].nunique()) if len(catalog) else 0)
    p = picks.copy()
    p["mode_n"] = p["mode"].map(_mode) if len(p) else pd.Series([], dtype=object)
    p["phase"] = p["phase"].astype(str).str.upper() if len(p) else p["phase"]
    for ph in ("p", "s"):
        sub = p[p["phase"] == ph.upper()]
        for m in ("manual", "automatic", "unknown"):
            out[f"{ph}_{m}"] = int((sub["mode_n"] == m).sum())
        out[f"{ph}_total"] = int(len(sub))
    out["readings"] = out["p_total"] + out["s_total"]
    out["events_with_arrivals"] = int(p["event"].nunique()) if len(p) else 0
    out["stations"] = int(p["station"].nunique()) if len(p) else 0
    out["stations_list"] = ";".join(sorted(map(str, p["station"].unique()))) if len(p) else ""
    out["readings_per_event"] = (out["readings"] / out["events_with_arrivals"]) if out["events_with_arrivals"] else np.nan
    mags = pd.to_numeric(catalog["mag"], errors="coerce") if len(catalog) else pd.Series([], dtype=float)
    cut = pd.cut(mags, MAG_BINS, labels=MAG_LABELS, right=False)
    counts = cut.value_counts()
    for lab in MAG_LABELS:
        out[lab] = int(counts.get(lab, 0))
    out["mag_unknown"] = int(mags.isna().sum())
    flags = heldout_flags(catalog)
    out["events_heldout_window"] = int(flags["in_window"].sum()) if len(flags) else 0
    out["events_heldout_place"] = int(flags["in_place"].sum()) if len(flags) else 0
    out["events_heldout_year"] = int(flags["in_year"].sum()) if len(flags) else 0
    out["events_heldout_any"] = int(flags["any"].sum()) if len(flags) else 0
    hit = set()
    for w in (flags["windows"] if len(flags) else []):
        hit.update(x for x in str(w).split(";") if x)
    out["heldout_windows"] = ";".join(sorted(hit))
    if len(p) and len(flags):
        joined = p.merge(flags[["event", "any"]], on="event", how="left")
        out["readings_heldout_any"] = int(joined["any"].fillna(False).astype(bool).sum())
    else:
        out["readings_heldout_any"] = 0
    out["heldout_readings_share"] = (out["readings_heldout_any"] / out["readings"]) if out["readings"] else np.nan
    out["heldout_events_share"] = (out["events_heldout_any"] / out["events"]) if out["events"] else np.nan
    out["scale"] = float(scale)
    for k in SCALED:
        out[f"{k}_est"] = out[k] * scale
    out["events_est"] = float(out["events"])
    return out


DAY_COLUMNS = (["operator", "year", "quarter", "day", "seed_tag", "sampler", "status", "error", "note",
                "n_queries", "n_retries", "n_failed_queries", "n_cached", "seconds", "bytes",
                "events", "events_fetched", "scale", "events_with_arrivals",
                "p_manual", "s_manual", "p_automatic", "s_automatic", "p_unknown", "s_unknown", "p_total", "s_total",
                "readings", "stations", "readings_per_event"] + MAG_LABELS + ["mag_unknown",
                "events_heldout_window", "events_heldout_place", "events_heldout_year", "events_heldout_any",
                "readings_heldout_any", "heldout_readings_share", "heldout_events_share", "heldout_windows",
                "events_est"] + [f"{k}_est" for k in SCALED] + ["stations_list"])


def _empty_day_row(operator, year, quarter, day, seed_tag, status, error, sampler=DEFAULT_SAMPLER) -> dict:
    row = {c: np.nan for c in DAY_COLUMNS}
    row.update(operator=operator, year=year, quarter=quarter, day=day.isoformat(), seed_tag=seed_tag, sampler=sampler,
               status=status, error=error, note="", n_queries=0, n_retries=0, n_failed_queries=0, n_cached=0,
               seconds=0.0, bytes=0, stations_list="", heldout_windows="")
    return row


def census_day(operator, year, quarter, day, seed_tag, raw_dir, budget, fetch=None,
               min_magnitude=None, max_events=None, sampler=DEFAULT_SAMPLER) -> dict:
    """One row of bulletin_<operator>_<year>.csv. `fetch` defaults to fetch_day (tests inject one).
    `sampler` is recorded in the row (the day itself was chosen by the caller)."""
    fetch = fetch or fetch_day
    cost = Cost()
    row = _empty_day_row(operator, year, quarter, day, seed_tag, "ok", "", sampler=sampler)
    if budget.exhausted():
        row.update(status="not_attempted", error="budget exhausted before the query")
        return row
    try:
        got = fetch(operator, day, raw_dir, cost, budget, min_magnitude=min_magnitude,
                    max_events=max_events, seed_tag=seed_tag)
    except Exception as exc:  # noqa: BLE001  every failure is a row, never a crash
        row.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        row.update(n_queries=cost.n_queries, n_retries=cost.n_retries, n_failed_queries=max(cost.n_failed, 1),
                   n_cached=cost.n_cached, seconds=round(cost.seconds, 3), bytes=cost.bytes)
        return row
    cat, picks = got["catalog"], got["picks"]
    n_ev = int(cat["event"].nunique()) if len(cat) else 0
    n_fetched = int(got.get("events_fetched", n_ev))
    scale = (n_ev / n_fetched) if n_fetched else 1.0
    row.update(count_day(cat, picks, scale=scale))
    row.update(events_fetched=n_fetched, note=got.get("note", ""),
               n_queries=cost.n_queries, n_retries=cost.n_retries, n_failed_queries=cost.n_failed,
               n_cached=cost.n_cached, seconds=round(cost.seconds, 3), bytes=cost.bytes)
    return row


def run_bulletin(operators, years, days_per_quarter, seed_tag, max_minutes, out_dir: Path = CENSUS_DIR,
                 raw_dir: Path = RAW_DIR, fetch=None, min_magnitude=None, max_events=None,
                 sampler: str = DEFAULT_SAMPLER) -> pd.DataFrame:
    """The bulletin census: writes bulletin_<operator>_<year>.csv per operator-year and
    bulletin_summary.csv over everything present in out_dir. Returns the day rows."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; known: {SAMPLERS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(max_minutes)
    frames = []
    for op in operators:
        if op not in OPERATORS:
            raise ValueError(f"unknown operator {op!r}; known: {sorted(OPERATORS)}")
        for year in years:
            rows = []
            for q, d in sample_days(op, year, days_per_quarter, seed_tag, sampler):
                log(f"{op} {year} Q{q} {d}")
                r = census_day(op, year, q, d, seed_tag, raw_dir, budget, fetch=fetch,
                               min_magnitude=min_magnitude, max_events=max_events, sampler=sampler)
                log(f"    {r['status']}: events={r.get('events')} readings={r.get('readings')} "
                    f"P manual={r.get('p_manual')} S manual={r.get('s_manual')} "
                    f"queries={r['n_queries']} {r['seconds']:.1f} s {r['bytes']} B {r['error']}")
                rows.append(r)
            df = pd.DataFrame(rows, columns=DAY_COLUMNS)
            path = out_dir / f"bulletin_{op}_{year}.csv"
            df.to_csv(path, index=False)
            log(f"  wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
            frames.append(df)
    write_summary(out_dir)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAY_COLUMNS)


# ── extrapolation and summary ────────────────────────────────────────────────

def bootstrap_year(values, days_in_year: int, seed: int, n_boot: int = N_BOOT):
    """Year estimate = days_in_year * mean(values); percentile 95 % interval from a
    bootstrap over the sampled days. Returns (est, lo, hi, boot_array) with NaN
    interval when fewer than two days."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan, np.full(n_boot, np.nan)
    est = days_in_year * float(v.mean())
    if len(v) < 2:
        return est, np.nan, np.nan, np.full(n_boot, est)
    rng = np.random.default_rng(seed)
    boots = days_in_year * rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), boots


def _seed(seed_tag: str, operator: str, year) -> int:
    return int(hashlib.sha256(f"{seed_tag}|{operator}|{year}|bootstrap".encode()).hexdigest()[:8], 16)


def summarise(days: pd.DataFrame) -> pd.DataFrame:
    """One row per operator-year plus one 'all' row per operator (sum of the year
    estimates; interval from the summed bootstrap draws)."""
    rows, boots_by_op = [], {}
    if len(days) == 0:
        return pd.DataFrame()
    for (op, year), g in days.groupby(["operator", "year"], sort=True):
        year = int(year)
        seed_tag = str(g["seed_tag"].iloc[0])
        sampler = ";".join(sorted(set(g["sampler"].astype(str)))) if "sampler" in g.columns else "v1"
        ok = g[g["status"] == "ok"]
        diy = 366 if calendar.isleap(year) else 365
        r = dict(operator=op, year=year, seed_tag=seed_tag, sampler=sampler, days_sampled=int(len(g)), days_ok=int(len(ok)),
                 days_failed=int((g["status"] == "failed").sum()),
                 days_not_attempted=int((g["status"] == "not_attempted").sum()), days_in_year=diy,
                 service_url=event_query_url(op) if op in OPERATORS else "")
        boots = {}
        for m in DAY_METRICS:
            col = f"{m}_est" if f"{m}_est" in ok.columns else m
            est, lo, hi, b = bootstrap_year(ok[col].to_numpy(dtype=float) if len(ok) else [], diy, _seed(seed_tag, op, year))
            r[f"{m}_per_day_mean"] = (est / diy) if est == est else np.nan
            r[f"{m}_year_est"] = est; r[f"{m}_year_lo"] = lo; r[f"{m}_year_hi"] = hi
            boots[m] = b
        boots_by_op.setdefault(op, []).append((diy, boots))
        r["stations_per_day_mean"] = float(ok["stations"].mean()) if len(ok) else np.nan
        seen = set()
        for s in ok["stations_list"].fillna(""):
            seen.update(x for x in str(s).split(";") if x)
        r["stations_distinct_sampled"] = len(seen)
        r["readings_per_event_mean"] = float(ok["readings_per_event"].mean()) if len(ok) else np.nan
        tot_ev = float(ok["events"].sum()) if len(ok) else 0.0
        for lab in MAG_LABELS + ["mag_unknown"]:
            r[f"{lab}_share"] = (float(ok[lab].sum()) / tot_ev) if tot_ev else np.nan
        tot_rd = float(ok["readings"].sum()) if len(ok) else 0.0
        r["heldout_readings_share"] = (float(ok["readings_heldout_any"].sum()) / tot_rd) if tot_rd else np.nan
        r["heldout_events_share"] = (float(ok["events_heldout_any"].sum()) / tot_ev) if tot_ev else np.nan
        hit = set()
        for w in ok["heldout_windows"].fillna(""):
            hit.update(x for x in str(w).split(";") if x)
        r["heldout_windows"] = ";".join(sorted(hit))
        r["manual_share_of_readings"] = ((float(ok["p_manual"].sum()) + float(ok["s_manual"].sum())) / tot_rd) if tot_rd else np.nan
        r["s_over_p_manual"] = (float(ok["s_manual"].sum()) / float(ok["p_manual"].sum())) if len(ok) and ok["p_manual"].sum() else np.nan
        r["n_queries"] = int(g["n_queries"].sum()); r["n_retries"] = int(g["n_retries"].sum())
        r["n_failed_queries"] = int(g["n_failed_queries"].sum()); r["n_cached"] = int(g["n_cached"].sum())
        r["seconds_total"] = float(g["seconds"].sum()); r["bytes_total"] = int(g["bytes"].sum())
        r["seconds_per_query"] = (r["seconds_total"] / r["n_queries"]) if r["n_queries"] else np.nan
        r["bytes_per_query"] = (r["bytes_total"] / r["n_queries"]) if r["n_queries"] else np.nan
        r["errors"] = " | ".join(f"{d}: {e}" for d, e in zip(g["day"], g["error"].fillna("")) if e)
        rows.append(r)
    for op, items in boots_by_op.items():
        yrs = [r for r in rows if r["operator"] == op and r["year"] != "all"]
        t = dict(operator=op, year="all", seed_tag=yrs[0]["seed_tag"],
                 sampler=";".join(sorted({s for r in yrs for s in r["sampler"].split(";")})),
                 days_sampled=sum(r["days_sampled"] for r in yrs),
                 days_ok=sum(r["days_ok"] for r in yrs), days_failed=sum(r["days_failed"] for r in yrs),
                 days_not_attempted=sum(r["days_not_attempted"] for r in yrs), days_in_year=sum(r["days_in_year"] for r in yrs),
                 service_url=yrs[0]["service_url"])
        for m in DAY_METRICS:
            ests = [r[f"{m}_year_est"] for r in yrs]
            t[f"{m}_year_est"] = float(np.nansum(ests)) if any(e == e for e in ests) else np.nan
            summed = np.nansum(np.vstack([b[m] for _, b in items]), axis=0)
            ok_any = any(np.isfinite(b[m]).any() for _, b in items)
            t[f"{m}_year_lo"], t[f"{m}_year_hi"] = (tuple(float(x) for x in np.percentile(summed, [2.5, 97.5]))
                                                   if ok_any and all(r["days_ok"] >= 2 for r in yrs) else (np.nan, np.nan))
            t[f"{m}_per_day_mean"] = np.nan
        for k in ("n_queries", "n_retries", "n_failed_queries", "n_cached", "bytes_total"):
            t[k] = int(sum(r[k] for r in yrs))
        t["seconds_total"] = float(sum(r["seconds_total"] for r in yrs))
        t["seconds_per_query"] = (t["seconds_total"] / t["n_queries"]) if t["n_queries"] else np.nan
        t["bytes_per_query"] = (t["bytes_total"] / t["n_queries"]) if t["n_queries"] else np.nan
        t["errors"] = " | ".join(r["errors"] for r in yrs if r["errors"])
        rows.append(t)
    cols = ["operator", "year", "seed_tag", "sampler", "days_sampled", "days_ok", "days_failed", "days_not_attempted",
            "days_in_year"]
    for m in DAY_METRICS:
        cols += [f"{m}_per_day_mean", f"{m}_year_est", f"{m}_year_lo", f"{m}_year_hi"]
    cols += ["stations_per_day_mean", "stations_distinct_sampled", "readings_per_event_mean",
             "manual_share_of_readings", "s_over_p_manual"]
    cols += [f"{lab}_share" for lab in MAG_LABELS + ["mag_unknown"]]
    cols += ["heldout_readings_share", "heldout_events_share", "heldout_windows",
             "n_queries", "n_retries", "n_failed_queries", "n_cached", "seconds_total", "seconds_per_query",
             "bytes_total", "bytes_per_query", "service_url", "errors"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["_yr"] = df["year"].map(lambda y: 10 ** 9 if y == "all" else int(y))
    return df.sort_values(["operator", "_yr"]).drop(columns="_yr")[cols].reset_index(drop=True)


def load_day_files(out_dir: Path = CENSUS_DIR) -> pd.DataFrame:
    files = sorted(p for p in out_dir.glob("bulletin_*_*.csv") if p.name != "bulletin_summary.csv")
    if not files:
        return pd.DataFrame(columns=DAY_COLUMNS)
    frames = []
    for p in files:
        df = pd.read_csv(p, dtype={"stations_list": str, "heldout_windows": str, "error": str, "note": str})
        if "sampler" not in df.columns:
            # day files written before the --sampler option (the committed 2026-09-12 run,
            # commit 7580636) were drawn by the rule that is now sampler v1
            df.insert(list(df.columns).index("seed_tag") + 1, "sampler", "v1")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def write_summary(out_dir: Path = CENSUS_DIR) -> pd.DataFrame:
    days = load_day_files(out_dir)
    summary = summarise(days)
    path = out_dir / "bulletin_summary.csv"
    summary.to_csv(path, index=False)
    log(f"  wrote {path} ({len(summary)} rows)")
    return summary


# ── SeisBench sources ────────────────────────────────────────────────────────

SUMMARY_NAME_MAP = {"STEAD": "stead", "INSTANCE": "instancecounts", "PNW": "pnw", "IQUIQUE": "iquique",
                    "LENDB": "lendb", "SCEDC": "scedc", "ETHZ": "ethz", "GEOFON": "geofon", "NEIC": "neic",
                    "OBST2024": "obst2024", "TXED": "txed", "VCSEIS": "vcseis", "CEED": "ceed", "PNW_ACCEL": "pnw_accel"}

# Trace counts data/README.md carries for sources the 2026-04-06 audit did not cover
# (its "Full traces" column; dagger = one .partial shard, a real row count of what is on disk).
README_TRACES = {"mlaapde": (510196, "partial shard"), "crew": (1599323, ""), "cwa": (346959, "partial shard"),
                 "pisdl": (142001, ""), "aq2009gm": (258984, "partial shard"), "meier2019jgr": (1060433, ""),
                 "ross2018gpd": (4773750, ""), "obs": (109208, ""), "ceed": (5009718, "")}

# Pick-status columns as recorded in notebooks/step_1_claude.ipynb (cell 8 output, five-row
# reads of each metadata.csv on the server): the column names that exist and the values seen.
# "not recorded" = the repository holds no record of that dataset's columns.
STATUS_RECORD = {
    "stead": ("trace_p_status;trace_s_status;trace_p_weight;trace_s_weight", "NaN in the sampled rows (noise rows)"),
    "neic": ("trace_p_status;trace_s_status", "trace_p_status = manual"),
    "geofon": ("trace_P_status;trace_S_status;trace_pP_status", "trace_P_status = manual"),
    "ethz": ("trace_P_status;trace_PmP_status;trace_SmS_status", "NaN in the sampled rows"),
    "scedc": ("trace_p_status;trace_s_status;trace_p_weight;trace_s_weight", "status NaN; weights 1.0/0.3/0.5"),
    "iquique": ("none", "no status column"),
    "lendb": ("trace_p_status", "trace_p_status = estimated"),
    "pnw": ("trace_P_arrival_uncertainty_s;trace_S_arrival_uncertainty_s;trace_P_onset", "no status column; uncertainties 0.01-0.08 s"),
    "obst2024": ("trace_p_status;trace_s_status;trace_p_weight;trace_s_weight", "NaN in the sampled rows"),
    "txed": ("trace_p_arrival_uncertainty_s;trace_s_arrival_uncertainty_s", "no status column"),
    "instancecounts": ("trace_P_uncertainty_s;trace_S_uncertainty_s;path_weight_phase_location_P;path_weight_phase_location_S",
                       "no status column; uncertainties 0.1 s (P) 0.6 s (S), location weights"),
    "ceed": ("not recorded", "metadata.csv absent when the notebook ran"),
    "vcseis": ("not recorded", "metadata.csv absent when the notebook ran"),
}

# Held-out overlap documented so far (docs/2026-09-07_training_history_audit.md section 5,
# docs/audit_2026-09-07/h3c_sequence_corpus_coverage.csv, docs/2026-09-08_heldout_test_cases.md).
HELDOUT_DOC = {
    "instancecounts": "Norcia 2016 sequence and year 2016 (Italy 2005-01 to 2020-01); Etna and Campi Flegrei place hold-outs",
    "stead": "year 2016; Kaikoura 2016 and Norcia 2016 possible (global 2005-2018)",
    "crew": "Thessaly 2021 and year 2021 possible (global regional)",
    "scedc": "Ridgecrest 2019", "ceed": "Ridgecrest 2019", "ross2018gpd": "Ridgecrest 2019",
    "pnw": "Monroe 2019", "vcseis": "Hawaii is tier 2 (usable only if VCSEIS ends before 2022)",
}


def parse_dataset_configs(path: Path = BUILDER_PY) -> pd.DataFrame:
    """name, cap, default_bin, use_s from DATASET_CONFIGS without importing the module
    (it imports seisbench at import time)."""
    tree = ast.parse(path.read_text())
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DATASET_CONFIGS" for t in node.targets):
            for call in node.value.elts:
                kw = {k.arg: k.value for k in call.keywords}
                rows.append(dict(source=ast.literal_eval(kw["name"]), cap=ast.literal_eval(kw["cap"]),
                                 default_bin=ast.literal_eval(kw["default_bin"]), use_s=ast.literal_eval(kw["use_s"])))
    return pd.DataFrame(rows)


def seisbench_table(summary_csv: Path = SUMMARY_STATS_CSV, pool_csv: Path = POOL_SUMMARY_CSV,
                    builder_py: Path = BUILDER_PY) -> pd.DataFrame:
    """The laptop-side table: what the committed records say about each source."""
    cfg = parse_dataset_configs(builder_py).set_index("source")
    stats = pd.read_csv(summary_csv)
    stats["source"] = stats["dataset"].map(SUMMARY_NAME_MAP)
    stats = stats.set_index("source")
    pool = pd.read_csv(pool_csv).set_index("dataset") if pool_csv.exists() else pd.DataFrame()
    lower_cols = {"stead", "lendb", "scedc", "neic", "obst2024", "txed", "vcseis", "ceed"}   # trace_p_arrival_sample (audit counted these)
    sources = list(cfg.index) + [s for s in stats.index if s not in cfg.index]
    rows = []
    for s in sources:
        in_pool = s in cfg.index
        st = stats.loc[s] if s in stats.index else None
        r = dict(source=s, in_training_pool=in_pool,
                 train_cap=int(cfg.loc[s, "cap"]) if in_pool else np.nan,
                 use_s=bool(cfg.loc[s, "use_s"]) if in_pool else np.nan,
                 default_bin=cfg.loc[s, "default_bin"] if in_pool else "")
        audit_partial = (st is not None and s in README_TRACES and int(st["n_traces"]) < 0.5 * README_TRACES[s][0])
        if st is not None and not audit_partial:
            r["n_traces"] = int(st["n_traces"]); r["n_traces_source"] = "summary_statistics.csv (2026-04-06 audit)"
            if s in README_TRACES and README_TRACES[s][0] != r["n_traces"]:
                r["n_traces_source"] += f"; data/README.md says {README_TRACES[s][0]:,}"
        elif audit_partial:       # ceed: the 2026-04-06 audit saw a 1,962-row partial copy
            r["n_traces"] = README_TRACES[s][0]
            r["n_traces_source"] = (f"data/README.md (2026-07-20 row count); summary_statistics.csv counted "
                                    f"{int(st['n_traces']):,} rows, a partial copy")
        elif s in README_TRACES:
            r["n_traces"] = README_TRACES[s][0]
            r["n_traces_source"] = "data/README.md (2026-07-20 row count" + (", partial shard)" if README_TRACES[s][1] else ")")
        else:
            r["n_traces"] = np.nan; r["n_traces_source"] = "server"
        if st is not None:
            rates = str(st["sampling_rates"])
            r["native_rates_hz"] = rates if rates not in ("[]", "nan") else "no rate column found by audit_metadata.py; server"
            r["dominant_rate_hz"] = st["dominant_sr"] if st["dominant_sr"] == st["dominant_sr"] else np.nan
            r["mag_median"] = st["mag_median"]
        else:
            r["native_rates_hz"] = "server"; r["dominant_rate_hz"] = np.nan; r["mag_median"] = np.nan
        if len(pool) and s in pool.index:
            r["n_p"] = int(pool.loc[s, "has_P"]); r["n_s"] = int(pool.loc[s, "has_S"]); r["n_both"] = int(pool.loc[s, "has_both"])
            r["pick_counts_source"] = "benchmark_pool_summary.csv"
        elif st is not None and s in lower_cols and not audit_partial:
            r["n_p"] = int(st["n_p_picks"]); r["n_s"] = int(st["n_s_picks"]); r["n_both"] = np.nan
            r["pick_counts_source"] = "summary_statistics.csv"
        elif audit_partial:
            r["n_p"] = np.nan; r["n_s"] = np.nan; r["n_both"] = np.nan
            r["pick_counts_source"] = f"server (summary_statistics.csv counted a {int(st['n_traces']):,}-row partial copy)"
        else:
            r["n_p"] = np.nan; r["n_s"] = np.nan; r["n_both"] = np.nan
            r["pick_counts_source"] = ("server (summary_statistics.csv counted trace_p_arrival_sample only, this source "
                                       "uses another column)" if st is not None else "server")
        cols, seen = STATUS_RECORD.get(s, ("not recorded", "no record in the repository"))
        r["status_columns"] = cols; r["status_values_seen"] = seen
        r["status_exposed"] = ("yes" if any(c.endswith("_status") for c in cols.split(";"))
                               else ("unknown" if cols == "not recorded" else "no"))
        r["heldout_overlap_documented"] = HELDOUT_DOC.get(s, "none documented")
        r["heldout_overlap_counts"] = "server"; r["pick_status_counts"] = "server"
        r["label_policy"] = _label_policy(s, r, cfg)
        r["server_required"] = "yes"
        r["server_command"] = "python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT"
        rows.append(r)
    return pd.DataFrame(rows)


def _label_policy(s: str, r: dict, cfg: pd.DataFrame) -> str:
    if not r["in_training_pool"]:
        return "not in the training pool" + (" (excluded: 0 % traces with both P and S)" if s == "neic" else "")
    use_s = bool(cfg.loc[s, "use_s"])
    n_s = r.get("n_s")
    if cfg.loc[s, "default_bin"] == "teleseismic":
        return "supervises P; S masked by the teleseismic P-only rule"
    if not use_s:
        return "supervises P; S masked (use_s=False)" + ("; needs review: P status 'estimated'" if s == "lendb" else "")
    if n_s == n_s and n_s is not None and r.get("n_traces") == r.get("n_traces") and n_s < 0.05 * r["n_traces"]:
        return f"supervises P; S present on {int(n_s):,} traces only, unknown-S mask needed"
    if r["status_exposed"] == "yes" and "NaN" in r["status_values_seen"]:
        return "supervises P and S; needs review: status column present, values unverified"
    if r["status_exposed"] == "unknown":
        return "supervises P and S; needs review: columns not recorded"
    if r["status_exposed"] == "no":
        return "supervises P and S; provenance by uncertainty/weight columns only"
    return "supervises P and S"


def status_counts(meta: pd.DataFrame) -> dict:
    """Server-side: counts of the pick-status values per phase from whichever status
    columns the metadata carries (trace_p_status / trace_P_status and the S forms)."""
    out = {}
    for ph, cands in (("p", ["trace_p_status", "trace_P_status"]), ("s", ["trace_s_status", "trace_S_status"])):
        col = next((c for c in cands if c in meta.columns), None)
        out[f"{ph}_status_column"] = col or ""
        vals = meta[col].astype(str).str.lower().where(meta[col].notna(), "nan") if col else pd.Series([], dtype=object)
        vc = vals.value_counts()
        out[f"{ph}_manual"] = int(vc.get("manual", 0)); out[f"{ph}_automatic"] = int(vc.get("automatic", 0))
        out[f"{ph}_estimated"] = int(vc.get("estimated", 0)); out[f"{ph}_status_nan"] = int(vc.get("nan", 0))
        out[f"{ph}_status_other"] = int(len(vals) - out[f"{ph}_manual"] - out[f"{ph}_automatic"]
                                        - out[f"{ph}_estimated"] - out[f"{ph}_status_nan"])
    return out


def heldout_counts(meta: pd.DataFrame) -> dict:
    """Server-side: rows inside any held-out window/place and in the held-out years."""
    flags = hs.flag_rows(meta)
    return dict(n_rows=int(len(meta)), n_in_window=int((flags["window"] != "").sum()),
                n_year_holdout=int(flags["year_holdout"].sum()), n_unverifiable=int((~flags["verifiable"]).sum()))


def seisbench_server_census(cache_root: str, out_dir: Path = CENSUS_DIR, sources=None) -> pd.DataFrame:
    """Runs on the lab server only: loads each DATASET_CONFIGS source the way the training
    build does and fills the two 'server' columns. Imports seisbench here, not at module import."""
    os.environ["SEISBENCH_CACHE_ROOT"] = str(cache_root)
    try:
        import build_training_dataset as btd  # noqa: F401  imports seisbench
    except ImportError as exc:
        raise SystemExit(f"seisbench is not importable here ({exc}); run this on the lab server") from exc
    rows = []
    for cfg in btd.DATASET_CONFIGS:
        if sources and cfg["name"] not in sources:
            continue
        log(f"  {cfg['name']}")
        try:
            meta = cfg["meta_fn"]() if cfg["meta_fn"] is not None else cfg["cls"]().metadata.copy()
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(source=cfg["name"], error=f"{type(exc).__name__}: {str(exc)[:120]}")); continue
        r = dict(source=cfg["name"], error="")
        r.update(status_counts(meta)); r.update(heldout_counts(meta))
        rows.append(r)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "seisbench_server.csv", index=False)
    log(f"  wrote {out_dir / 'seisbench_server.csv'}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bulletin", help="stratified day census of bulletin operators (network)")
    b.add_argument("--operators", nargs="+", default=["INGV", "NOA"], choices=sorted(OPERATORS))
    b.add_argument("--years", nargs="+", type=int, required=True)
    b.add_argument("--days-per-quarter", type=int, default=2)
    b.add_argument("--seed-tag", default="39a")
    b.add_argument("--sampler", choices=SAMPLERS, default=DEFAULT_SAMPLER,
                   help="day-drawing rule; v1 is the rule of the committed 2026-09-12 run (commit 7580636), "
                        "v2 (default) sorts the quarter's days by hash")
    b.add_argument("--max-minutes", type=float, default=20.0, help="wall-clock budget for all queries")
    b.add_argument("--max-events-per-day", type=int, default=80,
                   help="per-event paths (INGV, GeoNet, USGS): events fetched per day, deterministic subset; the rest is scaled")
    b.add_argument("--min-magnitude", type=float, default=None)
    b.add_argument("--out-dir", type=Path, default=CENSUS_DIR)
    b.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    s = sub.add_parser("summary", help="rebuild bulletin_summary.csv from the per-day CSVs")
    s.add_argument("--out-dir", type=Path, default=CENSUS_DIR)
    sb = sub.add_parser("seisbench", help="SeisBench source table; with --cache-root also the server counts")
    sb.add_argument("--cache-root", default=None, help="SEISBENCH_CACHE_ROOT; only on the lab server")
    sb.add_argument("--sources", nargs="*", default=None)
    sb.add_argument("--out-dir", type=Path, default=CENSUS_DIR)
    a = ap.parse_args(argv)

    if a.cmd == "bulletin":
        run_bulletin(a.operators, a.years, a.days_per_quarter, a.seed_tag, a.max_minutes, out_dir=a.out_dir,
                     raw_dir=a.raw_dir, min_magnitude=a.min_magnitude, max_events=a.max_events_per_day,
                     sampler=a.sampler)
    elif a.cmd == "summary":
        df = write_summary(a.out_dir)
        cols = ["operator", "year", "days_ok", "days_failed", "events_year_est", "p_manual_year_est", "p_manual_year_lo",
                "p_manual_year_hi", "s_manual_year_est", "s_manual_year_lo", "s_manual_year_hi", "heldout_readings_share"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    elif a.cmd == "seisbench":
        a.out_dir.mkdir(parents=True, exist_ok=True)
        table = seisbench_table()
        table.to_csv(a.out_dir / "seisbench_sources.csv", index=False)
        log(f"  wrote {a.out_dir / 'seisbench_sources.csv'} ({len(table)} sources)")
        if a.cache_root:
            seisbench_server_census(a.cache_root, out_dir=a.out_dir, sources=a.sources)


if __name__ == "__main__":
    main()
