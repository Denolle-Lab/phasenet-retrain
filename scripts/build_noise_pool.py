#!/usr/bin/env python3
"""
build_noise_pool.py

Checkpoint 42A of issue #42: harvest continuous data into a noise pool under
the ontology of `scripts/noise_ontology.py`.

For every (station, day):
  1. fetch the day of continuous 3-component data (FDSN, day chunks, bounded
     retries, raw miniSEED cached under data/noise_pools/raw/);
  2. cut it into 120 s windows at the NATIVE rate (no resampling, #34);
  3. apply the event-free rule of `docs/2026-09-07_training_plan.md` §5.2:
     a window is rejected when any catalogued event (USGS ComCat M >= 2.5
     globally, plus a local FDSN event service at M >= 0 where the station
     registry names one) has a predicted first P or first S (obspy TauP,
     iasp91) at the station inside the window or in the 120 s before it;
  4. compute the spectral features of `noise_ontology.spectral_features`;
  5. assign the class: a source flag first (registry flag, polar geography,
     catalogued-mainshock coda), the feature rules second;
  6. apply `exclusion_bundle.apply_exclusions(kind="noise")` on station
     location and window start (held-out places, windows and years), or
     record the bundle's absence with `--no-bundle`, which forces
     negative_support = unknown on every row;
  7. assign the station-disjoint split by hash;
  8. write the pool manifest (parquet), the windows as one .npz per
     station-day, and manifest.json (git commit, bundle hash, parameters,
     every catalogue query, every fetch, every failure).

No model is run on the data (§5.2). Nothing is fabricated: a service that
fails is recorded as a failure and the station-day is skipped.

    python scripts/build_noise_pool.py --pool pilot_2019 \\
        --station IU.KIP.00.BH --station IU.ANMO.00.BH --day 2019-03-12 \\
        --local-catalogue IU.KIP=USGS:3.0:0.0:2.0 --max-minutes 20 --no-bundle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import exclusion_bundle as eb  # noqa: E402
import noise_ontology as no  # noqa: E402

POOL_ROOT = REPO_ROOT / "data" / "noise_pools"
RAW_DIR = POOL_ROOT / "raw"

WINDOW_S = 120.0          # §5.2
LEAD_S = 120.0            # the 120 s before the window that must also be arrival-free
GLOBAL_MIN_MAG = 2.5      # ComCat global floor
LOCAL_MIN_MAG = 0.0       # local catalogue floor
CATALOGUE_LEAD_S = 2 * 3600.0   # events this long before the day can still arrive in it (S at 180 deg < 1 h)
POLAR_LAT_DEG = 60.0
CODA_SPANS = (             # (min magnitude, max distance deg (None = any), coda span s)
    (7.0, None, 3 * 3600.0),
    (5.0, 15.0, 30 * 60.0),
)
FDSN_TIMEOUT_S = 120
FDSN_TRIES = 3
FDSN_BACKOFF_S = 10.0

_LOG = []


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _LOG.append(line)


def _rel(path: Path) -> str:
    """Path relative to the repository when inside it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ── budget ────────────────────────────────────────────────────────────────────

class Budget:
    def __init__(self, max_minutes: Optional[float]):
        self.t0 = time.time()
        self.max_s = None if max_minutes is None else float(max_minutes) * 60.0
        self.exhausted_at: Optional[str] = None

    def elapsed(self) -> float:
        return time.time() - self.t0

    def left(self) -> Optional[float]:
        return None if self.max_s is None else self.max_s - self.elapsed()

    def check(self, what: str) -> bool:
        """True if there is budget left for `what`; records the first refusal."""
        if self.max_s is None or self.elapsed() < self.max_s:
            return True
        if self.exhausted_at is None:
            self.exhausted_at = what
            log(f"budget of {self.max_s / 60:.1f} min exhausted before: {what}")
        return False


# ── station registry ──────────────────────────────────────────────────────────

def parse_station(spec: str) -> dict:
    """'NET.STA[.LOC[.BAND]]' -> dict; LOC '' or '--' means blank, BAND default 'BH'."""
    parts = spec.split(".")
    if len(parts) < 2:
        raise ValueError(f"station spec must be NET.STA[.LOC[.BAND]], got {spec!r}")
    net, sta = parts[0].upper(), parts[1].upper()
    loc = parts[2] if len(parts) > 2 else "*"
    if loc in ("--", ""):
        loc = ""
    band = parts[3].upper() if len(parts) > 3 else "BH"
    if len(band) != 2:
        raise ValueError(f"channel band must be two letters (BH, HH, EH, HN), got {band!r}")
    return dict(network=net, station=sta, location=loc, channel_band=band, key=f"{net}.{sta}")


def parse_local_catalogue(spec: str) -> dict:
    """'NET.STA=CLIENT:radius_deg:min_mag:completeness_mag'."""
    key, _, rest = spec.partition("=")
    if not rest:
        raise ValueError(f"--local-catalogue must be NET.STA=CLIENT:radius:minmag:completeness, got {spec!r}")
    parts = rest.split(":")
    if len(parts) != 4:
        raise ValueError(f"--local-catalogue must be NET.STA=CLIENT:radius:minmag:completeness, got {spec!r}")
    return dict(key=key.upper(), client=parts[0], radius_deg=float(parts[1]),
                min_mag=float(parts[2]), completeness_mag=float(parts[3]))


# ── FDSN clients, bounded ─────────────────────────────────────────────────────

_clients: dict = {}


def client_for(name: str, timeout: int = FDSN_TIMEOUT_S):
    from obspy.clients.fdsn import Client
    if name not in _clients:
        if name.startswith("http"):
            _clients[name] = Client(base_url=name, timeout=timeout)
        else:
            _clients[name] = Client(name, timeout=timeout)
    return _clients[name]


def with_retries(fn, what: str, budget: Budget, failures: list, tries: Optional[int] = None,
                 backoff_s: Optional[float] = None):
    """Call fn() up to `tries` times with a fixed backoff; None on final failure."""
    tries = FDSN_TRIES if tries is None else int(tries)
    backoff_s = FDSN_BACKOFF_S if backoff_s is None else float(backoff_s)
    last = None
    for attempt in range(1, tries + 1):
        if not budget.check(what):
            failures.append(dict(what=what, attempt=attempt, error="budget_exhausted"))
            return None
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001  FDSNException, socket timeouts, parse errors
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            log(f"  {what}: attempt {attempt}/{tries} failed: {last}")
            if attempt < tries:
                time.sleep(backoff_s)
    failures.append(dict(what=what, attempt=tries, error=last))
    return None


# ── catalogue ─────────────────────────────────────────────────────────────────

def _event_rows(cat, source: str) -> List[dict]:
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        if o is None or o.time is None or o.latitude is None or o.longitude is None:
            continue
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        depth_km = (o.depth or 0.0) / 1000.0
        rid = str(ev.resource_id.id)
        for k in ("eventid=", "eventId=", "evid="):
            if k in rid:
                rid = rid.split(k)[-1]
                break
        rid = rid.split("&")[0].split("/")[-1]
        rows.append(dict(
            event_id=rid,
            time=float(o.time.timestamp),
            latitude=float(o.latitude), longitude=float(o.longitude),
            depth_km=float(depth_km if depth_km is not None and np.isfinite(depth_km) else 10.0),
            magnitude=float(m.mag) if m is not None and m.mag is not None else float("nan"),
            source=source,
        ))
    return rows


def fetch_catalogue(client_name: str, t0, t1, *, min_mag: float, cache_path: Path, budget: Budget,
                    failures: list, queries: list, lat=None, lon=None, radius_deg=None) -> Optional[List[dict]]:
    """ComCat-style event query, cached as QuakeML under RAW_DIR. None on failure."""
    from obspy import UTCDateTime, read_events
    t0, t1 = UTCDateTime(t0), UTCDateTime(t1)
    params = dict(starttime=str(t0), endtime=str(t1), minmagnitude=min_mag)
    if radius_deg is not None:
        params.update(latitude=lat, longitude=lon, maxradius=radius_deg)
    rec = dict(client=client_name, params=params, cache=_rel(cache_path),
               when=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if cache_path.is_file():
        cat = read_events(str(cache_path))
        rec.update(from_cache=True, n_events=len(cat))
        queries.append(rec)
        return _event_rows(cat, client_name)

    def _q():
        c = client_for(client_name)
        kw = dict(starttime=t0, endtime=t1, minmagnitude=min_mag)
        if radius_deg is not None:
            kw.update(latitude=lat, longitude=lon, maxradius=radius_deg)
        try:
            return c.get_events(**kw)
        except Exception as exc:  # noqa: BLE001
            # FDSN 204 "no data" is an empty catalogue, not a failure
            from obspy.clients.fdsn.header import FDSNNoDataException
            if isinstance(exc, FDSNNoDataException):
                from obspy import Catalog
                return Catalog()
            raise

    cat = with_retries(_q, f"get_events {client_name} {params}", budget, failures)
    if cat is None:
        rec.update(from_cache=False, n_events=None, failed=True)
        queries.append(rec)
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cat.write(str(cache_path), format="QUAKEML")
    rec.update(from_cache=False, n_events=len(cat))
    queries.append(rec)
    return _event_rows(cat, client_name)


# ── travel times ──────────────────────────────────────────────────────────────

_taup = None


def taup_model():
    global _taup
    if _taup is None:
        from obspy.taup import TauPyModel
        _taup = TauPyModel("iasp91")
    return _taup


@lru_cache(maxsize=65536)
def _first_ps(dist_deg_r: float, depth_km_r: float):
    """(first P travel time, first S travel time) in s, NaN when no phase arrives."""
    model = taup_model()
    arr = model.get_travel_times(source_depth_in_km=max(depth_km_r, 0.0), distance_in_degree=dist_deg_r,
                                 phase_list=["ttp", "tts"])
    p = [a.time for a in arr if a.name.upper().startswith(("P", "PK", "PDIFF")) and not a.name.upper().startswith("S")]
    s = [a.time for a in arr if a.name.upper().startswith(("S", "SK", "SDIFF"))]
    return (min(p) if p else float("nan"), min(s) if s else float("nan"))


def predicted_arrivals(event: dict, sta_lat: float, sta_lon: float) -> dict:
    """First P and first S epoch times at the station for one catalogue event (iasp91).

    Distance and depth are rounded to 0.01 deg and 1 km before the TauP call so
    the cache is effective; the rounding error is < 0.2 s, far inside LEAD_S.
    """
    from obspy.geodetics import locations2degrees
    d = float(locations2degrees(event["latitude"], event["longitude"], sta_lat, sta_lon))
    depth = float(event.get("depth_km", 10.0))
    if not np.isfinite(depth):
        depth = 10.0
    p, s = _first_ps(round(d, 2), round(min(depth, 700.0), 0))
    t = float(event["time"])
    return dict(event_id=event.get("event_id"), distance_deg=d, magnitude=event.get("magnitude"),
                p_time=(t + p) if np.isfinite(p) else float("nan"),
                s_time=(t + s) if np.isfinite(s) else float("nan"),
                source=event.get("source"))


def event_free_mask(window_starts: np.ndarray, arrivals: List[dict], *, duration_s: float = WINDOW_S,
                    lead_s: float = LEAD_S):
    """The §5.2 rule on precomputed arrivals.

    Returns (keep, nearest): keep[i] is False when any predicted P or S time
    lies in [start_i - lead_s, start_i + duration_s]; nearest[i] is the signed
    time from the window start to the closest arrival (NaN if none). Both ends
    of the interval are inclusive, so an arrival exactly at the window start
    or exactly 120 s before it rejects the window.
    """
    starts = np.asarray(window_starts, dtype=float)
    keep = np.ones(len(starts), dtype=bool)
    nearest = np.full(len(starts), np.nan)
    times = np.array([t for a in arrivals for t in (a.get("p_time"), a.get("s_time"))
                      if t is not None and np.isfinite(t)], dtype=float)
    if not len(times) or not len(starts):
        return keep, nearest
    for i, s0 in enumerate(starts):
        rel = times - s0
        j = int(np.argmin(np.abs(rel)))
        nearest[i] = rel[j]
        hit = (rel >= -lead_s) & (rel <= duration_s)
        keep[i] = not bool(hit.any())
    return keep, nearest


def coda_flags(window_starts: np.ndarray, arrivals: List[dict], *, spans=CODA_SPANS) -> np.ndarray:
    """True where a catalogued mainshock's first P precedes the window start by
    less than the coda span for its magnitude and distance (§5.1, earthquake
    coda and sequence hum). Such windows are kept but are unlabelled_interval."""
    starts = np.asarray(window_starts, dtype=float)
    flag = np.zeros(len(starts), dtype=bool)
    for a in arrivals:
        m = a.get("magnitude")
        p = a.get("p_time")
        if m is None or p is None or not np.isfinite(m) or not np.isfinite(p):
            continue
        d = a.get("distance_deg", float("inf"))
        for min_mag, max_dist, span in spans:
            if m >= min_mag and (max_dist is None or d <= max_dist):
                flag |= (starts > p) & (starts - p <= span)
                break
    return flag


# ── waveforms ─────────────────────────────────────────────────────────────────

def fetch_day(st_spec: dict, day, *, duration_s: float, client_name: str, budget: Budget,
              failures: list, fetches: list):
    """One station-day stream (3 components, merged, gaps zero-filled) or None."""
    from obspy import UTCDateTime, read
    t0 = UTCDateTime(day)
    t1 = t0 + duration_s
    loc = st_spec["location"]
    loc_tag = "any" if loc == "*" else (loc or "--")
    tag = f"{st_spec['network']}.{st_spec['station']}.{loc_tag}.{st_spec['channel_band']}_{t0.strftime('%Y-%m-%d')}_{int(duration_s)}s"
    cache = RAW_DIR / f"{tag}.mseed"
    rec = dict(station=st_spec["key"], day=t0.strftime("%Y-%m-%d"), duration_s=duration_s,
               client=client_name, cache=_rel(cache))
    t_start = time.time()
    if cache.is_file():
        st = read(str(cache))
        rec.update(from_cache=True)
    else:
        def _q():
            c = client_for(client_name)
            return c.get_waveforms(network=st_spec["network"], station=st_spec["station"],
                                  location=loc if loc else "--", channel=st_spec["channel_band"] + "?",
                                  starttime=t0, endtime=t1)
        st = with_retries(_q, f"get_waveforms {tag}", budget, failures)
        if st is None:
            rec.update(from_cache=False, failed=True, seconds=round(time.time() - t_start, 1))
            fetches.append(rec)
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        st.write(str(cache), format="MSEED")
        rec.update(from_cache=False)
    rec["seconds"] = round(time.time() - t_start, 1)
    if len(st) == 0:
        rec.update(failed=True, error="empty stream")
        fetches.append(rec)
        failures.append(dict(what=f"get_waveforms {tag}", error="empty stream"))
        return None
    n_traces_raw = len(st)
    # one location only: if the query returned several, keep the most complete
    locs = sorted({tr.stats.location for tr in st})
    if len(locs) > 1:
        best = max(locs, key=lambda L: sum(tr.stats.npts for tr in st if tr.stats.location == L))
        st = st.select(location=best)
        rec["location_chosen"] = best
    st.merge(method=1, fill_value=0)
    rec.update(n_traces_raw=n_traces_raw, n_channels=len(st),
               channels=sorted(tr.stats.channel for tr in st),
               rate_hz=float(st[0].stats.sampling_rate) if len(st) else None)
    if len(st) < 3:
        rec.update(failed=True, error=f"{len(st)} channels, need 3")
        fetches.append(rec)
        failures.append(dict(what=f"get_waveforms {tag}", error=rec["error"]))
        return None
    rates = {float(tr.stats.sampling_rate) for tr in st}
    if len(rates) != 1:
        rec.update(failed=True, error=f"mixed rates {sorted(rates)}")
        fetches.append(rec)
        failures.append(dict(what=f"get_waveforms {tag}", error=rec["error"]))
        return None
    fetches.append(rec)
    return st


def windows_from_stream(st, *, window_s: float = WINDOW_S, start=None, end=None):
    """Cut a merged 3-component stream into (data (n_win, 3, n), start_epochs, channels, rate).

    Windows are aligned to `start` (default: the stream's common start),
    non-overlapping, at the native rate; a trailing partial window is dropped.
    The stream is zero-padded to [start, end] so the grid is the day grid;
    windows that are entirely zero on every channel (no data at all) are
    dropped and counted in the fifth return value `n_empty`. Partially
    zero-filled windows (internal gaps) are kept for the instrument rule.
    The three channels are ordered Z first, then the horizontals by code
    (E/N or 1/2), so channel index 0 is always the vertical for the features.
    """
    from obspy import UTCDateTime
    if len(st) < 3:
        raise ValueError("need a 3-component stream")
    rate = float(st[0].stats.sampling_rate)
    n = int(round(window_s * rate))
    t0 = max(tr.stats.starttime for tr in st) if start is None else UTCDateTime(start)
    t1 = min(tr.stats.endtime for tr in st) if end is None else UTCDateTime(end)
    if t1 <= t0:
        return np.zeros((0, 3, n), dtype=np.float32), np.zeros(0), [tr.stats.channel for tr in st][:3], rate, 0
    st = st.copy().trim(t0, t1, pad=True, fill_value=0)

    def _order(tr):
        c = tr.stats.channel[-1].upper()
        return {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}.get(c, 3)
    trs = sorted(st, key=_order)[:3]
    chans = [tr.stats.channel for tr in trs]
    if trs[0].stats.channel[-1].upper() != "Z":
        raise ValueError(f"no vertical channel among {chans}")
    m = min(tr.stats.npts for tr in trs)
    n_win = m // n
    if n_win == 0:
        return np.zeros((0, 3, n), dtype=np.float32), np.zeros(0), chans, rate, 0
    data = np.stack([np.asarray(tr.data[: n_win * n], dtype=np.float32).reshape(n_win, n) for tr in trs], axis=1)
    starts = float(trs[0].stats.starttime.timestamp) + np.arange(n_win) * (n / rate)
    nonempty = np.any(data != 0, axis=(1, 2))
    n_empty = int((~nonempty).sum())
    return data[nonempty], starts[nonempty], chans, rate, n_empty


# ── labelling ─────────────────────────────────────────────────────────────────

def source_class_for(st_spec: dict, sta_lat: float, source_flags: Dict[str, str]) -> Optional[str]:
    """Registry source flag first, then the polar-geography rule."""
    flag = source_flags.get(st_spec["key"])
    if flag:
        if flag not in {c.value for c in no.NoiseClass}:
            raise ValueError(f"source flag {flag!r} for {st_spec['key']} is not a noise class")
        return flag
    if np.isfinite(sta_lat) and abs(sta_lat) >= POLAR_LAT_DEG:
        return no.NoiseClass.POLAR_ICE.value
    return None


def label_windows(features: List[dict], *, coda: np.ndarray, source_class: Optional[str]):
    """(class, class_source, confidence) per window: coda flag > source flag > features."""
    out = []
    for i, f in enumerate(features):
        if coda[i]:
            out.append((no.NoiseClass.EARTHQUAKE_CODA_SEQUENCE_HUM.value, "catalogue_coda", 1.0))
        elif source_class:
            out.append((source_class, "source_flag", 1.0))
        else:
            c, conf = no.classify_features(f)
            out.append((c.value, "features", float(conf)))
    return out


def support_column(df: pd.DataFrame, *, bundle_present: bool) -> pd.Series:
    """negative_support per row from category, event-free result and catalogue completeness.
    Without a bundle every row is unknown: nothing about its independence is known."""
    vals = []
    for _, r in df.iterrows():
        if not bundle_present:
            vals.append("unknown")
            continue
        vals.append(no.negative_support_for(
            r["ontology_category"], event_free=bool(r["event_free"]),
            local_catalogue=bool(r.get("catalogue_local", False)),
            completeness_mag=r.get("completeness_mag", float("nan"))))
    return pd.Series(vals, index=df.index, dtype=object)


def apply_bundle(df: pd.DataFrame, bundle: Optional[dict]):
    """apply_exclusions(kind="noise") on station location and start time, or the
    no-bundle path: nothing removed, every row flagged, support forced unknown.
    The trace list is passed empty: pool windows are FDSN cuts, not SeisBench
    traces, so (dataset, chunk, trace_name) identity does not apply to them."""
    if bundle is None:
        kept = df.copy()
        kept[eb.FLAG_COL] = True
        kept["exclusion_bundle_sha256"] = ""
        report = dict(kind="noise", bundle="absent", n_input=int(len(df)), n_kept=int(len(df)), n_removed=0,
                      n_quarantined_unknown=0, n_unknown_kept_flagged=int(len(df)), windows={})
        return kept, report
    kept, report = eb.apply_exclusions(df, bundle, kind="noise", dataset="noise_pool",
                                       station_lat_col="station_latitude_deg",
                                       station_lon_col="station_longitude_deg", start_col="start_time",
                                       exclusions={})
    kept = kept.copy()
    kept["exclusion_bundle_sha256"] = bundle.get("sha256", "")
    return kept, report


# ── the pool ──────────────────────────────────────────────────────────────────

def build_rows(*, pool: str, st_spec: dict, sta_lat: float, sta_lon: float, data: np.ndarray,
               starts: np.ndarray, rate: float, keep: np.ndarray, nearest: np.ndarray, coda: np.ndarray,
               source_class: Optional[str], catalogue_used: str, catalogue_local: bool,
               completeness_mag: float, npz_rel: str, source: str) -> pd.DataFrame:
    """Rows for the kept windows of one station-day (no bundle, no split yet)."""
    from obspy import UTCDateTime
    n_win = data.shape[0]
    feats = []
    for i in range(n_win):
        hour = UTCDateTime(float(starts[i])).hour
        feats.append(no.spectral_features(data[i], rate, hour_of_day=hour, vertical_index=0))
    # station references from this station's own windows (all windows of the harvest, kept or not)
    rms10 = np.array([f["rms_log10"] for f in feats], dtype=float)
    finite = rms10[np.isfinite(rms10)]
    p10 = float(np.percentile(finite, no.THRESH["quiet_rms_percentile"])) if finite.size else float("nan")
    sec = np.array([f["secondary_band_power_log10"] for f in feats], dtype=float)
    finite = sec[np.isfinite(sec)]
    p50 = float(np.median(finite)) if finite.size else float("nan")
    for f in feats:
        f["station_rms_log10_p10"] = p10
        f["station_secondary_p50"] = p50
    labels = label_windows(feats, coda=coda, source_class=source_class)
    rows = []
    for i in range(n_win):
        if not keep[i]:
            continue
        cls, cls_src, conf = labels[i]
        cat = no.CLASS_SPECS[cls]["ontology_category"]
        row = dict(
            pool=pool, noise_class=cls, ontology_category=cat, negative_support="unknown", arrivals_json="",
            network=st_spec["network"], station=st_spec["station"], location=st_spec["location"],
            channel_band=st_spec["channel_band"],
            station_latitude_deg=float(sta_lat), station_longitude_deg=float(sta_lon),
            rate_hz=float(rate), start_time=UTCDateTime(float(starts[i])).isoformat(),
            duration_s=float(data.shape[2] / rate), npz_path=npz_rel, npz_index=int(i),
            source=source, class_source=cls_src, class_confidence=float(conf),
            catalogue_used=catalogue_used, completeness_mag=float(completeness_mag),
            event_free=bool(keep[i]), nearest_arrival_s=float(nearest[i]),
            exclusion_bundle_sha256="", independence_unverified=False, split="",
        )
        for c in no.FEATURE_COLUMNS:
            row[c] = float(feats[i][c])
        row["catalogue_local"] = bool(catalogue_local)   # dropped before writing
        rows.append(row)
    return pd.DataFrame(rows)


def finalize(df: pd.DataFrame, *, bundle: Optional[dict], holdout_fraction: float, seed_tag: str):
    """Bundle, support, split; returns (manifest, exclusion report)."""
    if len(df) == 0:
        out = no.empty_manifest()
        return out, dict(kind="noise", n_input=0, n_kept=0, n_removed=0, bundle="absent" if bundle is None else bundle.get("sha256"))
    kept, report = apply_bundle(df, bundle)
    kept["negative_support"] = support_column(kept, bundle_present=bundle is not None)
    kept["split"] = [no.station_split(f"{n}.{s}", holdout_fraction, seed_tag)
                     for n, s in zip(kept["network"], kept["station"])]
    kept = kept.drop(columns=["catalogue_local"], errors="ignore")
    kept = kept[list(no.MANIFEST_COLUMNS)].reset_index(drop=True)
    no.check_manifest(kept)
    return kept, report


def census(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return dict(n_rows=0)
    return dict(
        n_rows=int(len(df)),
        by_station={k: int(v) for k, v in (df["network"] + "." + df["station"]).value_counts().sort_index().items()},
        by_class={k: int(v) for k, v in df["noise_class"].value_counts().sort_index().items()},
        by_category={k: int(v) for k, v in df["ontology_category"].value_counts().sort_index().items()},
        by_support={k: int(v) for k, v in df["negative_support"].value_counts().sort_index().items()},
        by_split={k: int(v) for k, v in df["split"].value_counts().sort_index().items()},
        by_class_source={k: int(v) for k, v in df["class_source"].value_counts().sort_index().items()},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool name; output under data/noise_pools/<pool>/")
    ap.add_argument("--station", action="append", required=True, help="NET.STA[.LOC[.BAND]] (repeatable)")
    ap.add_argument("--day", action="append", required=True, help="UTC day YYYY-MM-DD (repeatable)")
    ap.add_argument("--day-duration-s", type=float, default=86400.0,
                    help="seconds of each day to fetch, from 00:00 UTC (default the whole day)")
    ap.add_argument("--waveform-client", default="IRIS", help="FDSN client name or base URL for waveforms and stations")
    ap.add_argument("--global-catalogue", default="USGS", help="FDSN event client for the global M>=2.5 query")
    ap.add_argument("--local-catalogue", action="append", default=[],
                    help="NET.STA=CLIENT:radius_deg:min_mag:completeness_mag (repeatable)")
    ap.add_argument("--source-class", action="append", default=[],
                    help="NET.STA=<noise_class> source flag that overrides the feature rules (repeatable)")
    ap.add_argument("--bundle", default=str(eb.BUNDLE_PATH), help="exclusion bundle path")
    ap.add_argument("--no-bundle", action="store_true",
                    help="run without an exclusion bundle: absence recorded, every row negative_support=unknown")
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed-tag", default="42A")
    ap.add_argument("--max-minutes", type=float, default=20.0, help="wall-clock budget for the whole run")
    ap.add_argument("--out-root", default=str(POOL_ROOT))
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="repository root the bundle's input paths are relative to")
    args = ap.parse_args(argv)

    budget = Budget(args.max_minutes)
    failures: list = []
    queries: list = []
    fetches: list = []
    out_root = Path(args.out_root)
    pool_dir = out_root / args.pool
    (pool_dir / "windows").mkdir(parents=True, exist_ok=True)
    global RAW_DIR
    RAW_DIR = out_root / "raw"

    stations = [parse_station(s) for s in args.station]
    local = {c["key"]: c for c in (parse_local_catalogue(s) for s in args.local_catalogue)}
    source_flags = dict(s.split("=", 1) for s in args.source_class)

    # bundle
    bundle = None
    bundle_rec: dict
    if args.no_bundle:
        bundle_rec = dict(present=False, reason="--no-bundle", path=args.bundle)
        log("no exclusion bundle: every row will carry negative_support=unknown and independence_unverified=True")
    else:
        bundle = eb.load_bundle(args.bundle, require_certified=False, verify_sources=False, repo_root=args.repo_root)
        bundle_rec = dict(present=True, path=args.bundle, sha256=bundle.get("sha256"),
                          certified=bool(bundle.get("certified")), version=bundle.get("version"))
        log(f"exclusion bundle {args.bundle}: sha256 {bundle['sha256'][:12]} certified={bundle.get('certified')}")

    from obspy import UTCDateTime

    # station coordinates
    station_recs = []
    for s in stations:
        def _q(s=s):
            c = client_for(args.waveform_client)
            return c.get_stations(network=s["network"], station=s["station"], level="station",
                                  starttime=UTCDateTime(min(args.day)), endtime=UTCDateTime(max(args.day)) + 86400)
        inv = with_retries(_q, f"get_stations {s['key']}", budget, failures)
        if inv is None or len(inv) == 0 or len(inv[0]) == 0:
            s["latitude"], s["longitude"] = float("nan"), float("nan")
            log(f"{s['key']}: coordinates unknown (query failed); rows will be quarantined by the bundle")
        else:
            s["latitude"], s["longitude"] = float(inv[0][0].latitude), float(inv[0][0].longitude)
        station_recs.append(dict(s, local_catalogue=local.get(s["key"]), source_flag=source_flags.get(s["key"])))
        log(f"{s['key']} at {s['latitude']:.4f}, {s['longitude']:.4f}")

    frames = []
    day_records = []
    for day in args.day:
        d0 = UTCDateTime(day)
        d1 = d0 + args.day_duration_s
        # global catalogue, once per day
        gcat = fetch_catalogue(args.global_catalogue, d0 - CATALOGUE_LEAD_S, d1, min_mag=GLOBAL_MIN_MAG,
                               cache_path=RAW_DIR / "events" / f"{args.global_catalogue}_global_M{GLOBAL_MIN_MAG}_{day}_{int(args.day_duration_s)}s.xml",
                               budget=budget, failures=failures, queries=queries)
        if gcat is None:
            log(f"{day}: global catalogue query failed; the event-free rule cannot be applied, day skipped")
            day_records.append(dict(day=day, skipped="global catalogue failed"))
            continue
        log(f"{day}: {len(gcat)} global events M>={GLOBAL_MIN_MAG} in [{d0 - CATALOGUE_LEAD_S}, {d1}]")

        for s in stations:
            if not budget.check(f"station-day {s['key']} {day}"):
                day_records.append(dict(day=day, station=s["key"], skipped="budget_exhausted"))
                continue
            events = list(gcat)
            lc = local.get(s["key"])
            cat_used = f"{args.global_catalogue} global M>={GLOBAL_MIN_MAG}"
            cat_local_ok = False
            comp = float("nan")
            if lc is not None and np.isfinite(s["latitude"]):
                lcat = fetch_catalogue(lc["client"], d0 - CATALOGUE_LEAD_S, d1, min_mag=lc["min_mag"],
                                       lat=s["latitude"], lon=s["longitude"], radius_deg=lc["radius_deg"],
                                       cache_path=RAW_DIR / "events" / f"{lc['client']}_local_{s['key']}_r{lc['radius_deg']}_M{lc['min_mag']}_{day}_{int(args.day_duration_s)}s.xml",
                                       budget=budget, failures=failures, queries=queries)
                if lcat is None:
                    log(f"  {s['key']}: local catalogue failed; rows stay unknown (global rule only)")
                    cat_used += f" + {lc['client']} local FAILED"
                else:
                    seen = {e["event_id"] for e in events}
                    events += [e for e in lcat if e["event_id"] not in seen]
                    cat_used += f" + {lc['client']} local M>={lc['min_mag']} r<={lc['radius_deg']}deg"
                    cat_local_ok = True
                    comp = lc["completeness_mag"]
                    log(f"  {s['key']}: {len(lcat)} local events, completeness stated M{comp}")

            st = fetch_day(s, day, duration_s=args.day_duration_s, client_name=args.waveform_client,
                           budget=budget, failures=failures, fetches=fetches)
            if st is None:
                day_records.append(dict(day=day, station=s["key"], skipped="waveform fetch failed"))
                continue
            data, starts, chans, rate, n_empty = windows_from_stream(st, start=d0, end=d1)
            if data.shape[0] == 0:
                day_records.append(dict(day=day, station=s["key"], skipped="no complete window", n_empty=n_empty))
                continue

            t_tt = time.time()
            arrivals = ([predicted_arrivals(e, s["latitude"], s["longitude"]) for e in events]
                        if np.isfinite(s["latitude"]) else [])
            keep, nearest = event_free_mask(starts, arrivals)
            coda = coda_flags(starts, arrivals)
            log(f"  {s['key']} {day}: {data.shape[0]} windows at {rate:g} Hz, {int((~keep).sum())} rejected by "
                f"{len(arrivals)} events ({time.time() - t_tt:.1f} s TauP), {int((coda & keep).sum())} kept as coda")

            npz_rel = f"windows/{s['network']}.{s['station']}.{s['location'] or '--'}.{s['channel_band']}_{day}.npz"
            np.savez_compressed(pool_dir / npz_rel, data=data[keep], start_epoch=starts[keep],
                                channels=np.array(chans), rate_hz=rate, kept_index=np.flatnonzero(keep))
            src_cls = source_class_for(s, s["latitude"], source_flags)
            frame = build_rows(pool=args.pool, st_spec=s, sta_lat=s["latitude"], sta_lon=s["longitude"],
                               data=data, starts=starts, rate=rate, keep=keep, nearest=nearest, coda=coda,
                               source_class=src_cls, catalogue_used=cat_used, catalogue_local=cat_local_ok,
                               completeness_mag=comp, npz_rel=npz_rel, source=f"fdsn:{args.waveform_client}")
            # npz_index must address the kept array, not the pre-rejection index
            frame["npz_index"] = np.arange(len(frame), dtype=int)
            frames.append(frame)
            day_records.append(dict(day=day, station=s["key"], n_windows=int(data.shape[0]), n_empty=n_empty,
                                    n_rejected_arrival=int((~keep).sum()), n_coda=int((coda & keep).sum()),
                                    n_events=len(events), rate_hz=rate, channels=chans, npz=npz_rel))
            if not np.isfinite(s["latitude"]):
                failures.append(dict(what=f"{s['key']} coordinates", error="unknown; event-free rule not applied"))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest, excl = finalize(df, bundle=bundle, holdout_fraction=args.holdout_fraction, seed_tag=args.split_seed_tag)
    manifest.to_parquet(pool_dir / "manifest.parquet", index=False)

    meta = dict(
        pool=args.pool, checkpoint="42A", git_commit=git_commit(),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        parameters=dict(window_s=WINDOW_S, lead_s=LEAD_S, global_min_mag=GLOBAL_MIN_MAG,
                        catalogue_lead_s=CATALOGUE_LEAD_S, coda_spans=CODA_SPANS, polar_lat_deg=POLAR_LAT_DEG,
                        taup_model="iasp91", phases="first P of ttp, first S of tts",
                        day_duration_s=args.day_duration_s, days=args.day,
                        waveform_client=args.waveform_client, global_catalogue=args.global_catalogue,
                        holdout_fraction=args.holdout_fraction, split_seed_tag=args.split_seed_tag,
                        max_minutes=args.max_minutes, fdsn_timeout_s=FDSN_TIMEOUT_S, fdsn_tries=FDSN_TRIES,
                        feature_thresholds=no.THRESH, model_screening=False),
        stations=station_recs,
        bundle=bundle_rec,
        exclusion_report=excl,
        catalogue_queries=queries,
        fetches=fetches,
        station_days=day_records,
        failures=failures,
        budget=dict(max_minutes=args.max_minutes, elapsed_s=round(budget.elapsed(), 1),
                    exhausted_before=budget.exhausted_at),
        census=census(manifest),
        schema=list(no.MANIFEST_COLUMNS),
        log=_LOG,
    )
    (pool_dir / "manifest.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"wrote {pool_dir / 'manifest.parquet'} ({len(manifest)} rows) and manifest.json; "
        f"{len(failures)} failure(s); {budget.elapsed() / 60:.1f} min")
    return 0 if len(manifest) else 1


if __name__ == "__main__":
    sys.exit(main())
