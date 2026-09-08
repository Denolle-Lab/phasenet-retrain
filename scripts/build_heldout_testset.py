#!/usr/bin/env python3
"""
build_heldout_testset.py

Builds the held-out test data for the sequences in
scripts/heldout_testset_registry.py, the way the QuakeScope notebooks do
it, and reproducibly: every query is pinned by the registry entry, every
output is hashed, and the manifest records what was asked of which service
when.

Per sequence, under data/heldout_testset/<key>/:
  catalog.parquet     stage-1 catalogue (event, origin, lat, lon, depth, mag, source)
  windows.csv         the scoring windows and how they were chosen
  picks.parquet       QuakeScope schema: sequence, event, origin, mag, station,
                      channel, phase, time, mode, status, method, agency,
                      time_weight, onset, uncertainty  + network, source,
                      reference_ok (see REFERENCE_OK)
  station_map.csv     bare station code -> NET.STA resolution (ISC, JMA, Zenodo)
  stations.csv        candidate stations with manual-pick counts and distance,
                      and which were fetched
  waveforms/*.mseed   raw MiniSEED, one file per station and window (not committed)
  manifest.json       registry entry, git commit, versions, service URLs,
                      queries with timestamps, counts, sha256 of every file
  build.log           everything printed

The parquet, csv and json files are committed; they pin the reference. The
notebooks' rule "the cache is what pins the reference between runs" is kept:
a step is skipped when its output exists unless --force is given.

Usage (from repo root):
    python scripts/build_heldout_testset.py --sequence adriatic_2022 all
    python scripts/build_heldout_testset.py --sequence petrinja_2020 catalog picks windows stations waveforms manifest
    python scripts/build_heldout_testset.py --all all
    python scripts/build_heldout_testset.py --index         # rebuild data/heldout_testset/index.csv
    python scripts/build_heldout_testset.py --verify KEY    # re-hash outputs against manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_testset_registry as reg  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "heldout_testset"
PICK_COLS = ["sequence", "event", "origin", "mag", "station", "channel", "phase", "time",
             "mode", "status", "method", "agency", "time_weight", "onset", "uncertainty",
             "network", "source", "reference_ok"]
REFERENCE_OK = ("reference_ok = mode == 'manual', or mode unknown for ISC-sourced picks "
                "(the ISC bulletin carries agencies' reviewed readings without an evaluation mode)")

_log_fh = None


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n"); _log_fh.flush()


def _have(path: Path) -> bool:
    """An output exists and is non-empty (a crashed run can leave a 0-byte file)."""
    return path.exists() and path.stat().st_size > 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def utc(s):
    from obspy import UTCDateTime
    return UTCDateTime(s)


def download(url: str, dest: Path, tries: int = 4, timeout: int = 300) -> Path:
    """GET with retries and backoff; Zenodo answers 504 under load."""
    import urllib.error
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            dest.write_bytes(urllib.request.urlopen(req, timeout=timeout).read())
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc; log(f"    download {url.split('/')[-2:]} attempt {i + 1}: {type(exc).__name__} {str(exc)[:60]}")
            time.sleep(15 * (i + 1))
    raise RuntimeError(f"download failed after {tries} tries: {url}") from last


# ── clients ──────────────────────────────────────────────────────────────────

_clients: dict = {}
QUERIES: list = []          # every request made, for the manifest


def client_for(name: str):
    from obspy.clients.fdsn import Client, RoutingClient
    if name in _clients:
        return _clients[name]
    if name == "eida":
        c = RoutingClient("eida-routing", timeout=120)
    elif name == "iris-fed":
        c = RoutingClient("iris-federator", timeout=120)
    elif name == "franceseisme":
        c = Client(base_url="https://api.franceseisme.fr", timeout=120,
                   service_mappings={"event": "https://api.franceseisme.fr/fdsnws/event/1"})
    else:
        c = Client(name, timeout=120)
    _clients[name] = c
    return c


def base_url(name: str) -> str:
    c = client_for(name)
    return getattr(c, "base_url", None) or f"routing:{name}"


def _record(kind, client, **params):
    QUERIES.append(dict(kind=kind, client=client, base_url=base_url(client) if client else None,
                        when=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        params={k: (str(v) if v is not None else None) for k, v in params.items()}))


def _event_id(ev) -> str:
    rid = str(ev.resource_id.id)
    return rid.split("eventId=")[-1].split("eventid=")[-1].split("evid=")[-1].split("/")[-1]


# ── pick rows, the QuakeScope function plus network/source/reference_ok ────

def _pick_rows(ev, label, source):
    origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
    if origin is None:
        return []
    mag = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
    by_id = {p.resource_id.id: p for p in ev.picks}
    rows = []
    for arr in origin.arrivals:
        pick = by_id.get(arr.pick_id.id)
        if pick is None or not arr.phase:
            continue
        phase = arr.phase.strip().upper().lstrip("IE")[:1]
        if phase not in ("P", "S"):
            continue
        w = pick.waveform_id
        net = (w.network_code or "").strip()
        if net in ("IR", "--", "XX"):        # ISC's placeholder: network unknown
            net = ""
        mode = str(pick.evaluation_mode) if pick.evaluation_mode else "unknown"
        rows.append(dict(
            sequence=label, event=_event_id(ev), origin=origin.time.datetime,
            mag=mag.mag if mag else np.nan,
            station=f"{net}.{w.station_code}" if net else w.station_code,
            channel=w.channel_code or "",
            phase=phase, time=pick.time.datetime, mode=mode,
            status=str(pick.evaluation_status) if pick.evaluation_status else "unknown",
            method=str(pick.method_id.id).split("/")[-1] if pick.method_id else "",
            agency=(pick.creation_info.agency_id if pick.creation_info and pick.creation_info.agency_id else ""),
            time_weight=arr.time_weight if arr.time_weight is not None else np.nan,
            onset=str(pick.onset) if pick.onset else "",
            uncertainty=(pick.time_errors.uncertainty if pick.time_errors and pick.time_errors.uncertainty is not None else np.nan),
            network=net, source=source,
            reference_ok=(mode == "manual") or (mode == "unknown" and source == "ISC"),
        ))
    return rows


def _catalog_rows(cat, source):
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
        if o is None:
            continue
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows.append(dict(event=_event_id(ev), origin=o.time.datetime, lat=o.latitude, lon=o.longitude,
                         depth_km=(o.depth / 1000.0) if o.depth is not None else np.nan,
                         mag=m.mag if m else np.nan, source=source))
    return rows


# ── harvest kinds ────────────────────────────────────────────────────────────

def _region(seq, min_mag):
    d = dict(latitude=seq["lat"], longitude=seq["lon"], maxradius=seq["radius"])
    if min_mag is not None:
        d["minmagnitude"] = min_mag
    return d


def _chunked_region_events(client_name, seq, t0, t1, min_mag, arrivals, hours=1.0):
    """Region query in time chunks, splitting on 413/'too much', skipping 204."""
    from obspy.clients.fdsn.header import FDSNException
    client = client_for(client_name)
    region = _region(seq, min_mag)
    step = hours * 3600.0
    edges = np.arange(float(t0), float(t1), step)
    pieces = [(utc(a), utc(min(a + step, float(t1)))) for a in edges]
    events, seen = [], set()
    while pieces:
        a, b = pieces.pop(0)
        _record("get_events", client_name, starttime=a, endtime=b, includearrivals=arrivals, **region)
        try:
            cat = client.get_events(starttime=a, endtime=b, includearrivals=arrivals, **region)
        except Exception as exc:  # noqa: BLE001  FDSNException, socket timeouts, obspy's own parse errors
            s = str(exc)
            if "413" in s or "too much" in s.lower() or "Request too large" in s:
                m = a + (b - a) / 2
                pieces = [(a, m), (m, b)] + pieces
                continue
            if "204" in s or "No data" in s:
                continue
            log(f"    {client_name} {a}..{b}: {type(exc).__name__}: {s[:120]}; retrying once after 20 s")
            time.sleep(20)
            if "event service" in s or "No FDSN services" in s:
                _clients.pop(client_name, None)          # discovery failed once; rebuild the client
                client = client_for(client_name)
            try:
                cat = client.get_events(starttime=a, endtime=b, includearrivals=arrivals, **region)
            except Exception as exc2:  # noqa: BLE001
                log(f"    {client_name} {a}..{b}: skipped ({type(exc2).__name__})")
                continue
        for ev in cat:
            eid = _event_id(ev)
            if eid in seen:
                continue
            seen.add(eid); events.append(ev)
    return events


def harvest_fdsn_per_event(seq, spec, spans, label):
    """Catalogue by region over each span, then one includearrivals request per event."""
    from obspy.clients.fdsn.header import FDSNException
    client_name = spec["client"]; client = client_for(client_name)
    rows, cat_rows, seen = [], [], set()
    for (t0, t1) in spans:
        _record("get_events", client_name, starttime=t0, endtime=t1, **_region(seq, spec.get("min_mag")))
        try:
            cat = client.get_events(starttime=t0, endtime=t1, **_region(seq, spec.get("min_mag")))
        except FDSNException as exc:
            log(f"    {client_name}: {type(exc).__name__}: {str(exc)[:100]}"); continue
        log(f"    {client_name}: {len(cat)} events in {t0}..{t1}, fetching arrivals one by one")
        for ev in cat:
            eid = _event_id(ev)
            if eid in seen:
                continue
            seen.add(eid)
            try:
                if client_name == "INGV":
                    full = client.get_events(eventid=eid, includearrivals=True)
                else:                                   # GeoNet: the per-event QuakeML carries the picks
                    full = client.get_events(eventid=eid)
            except Exception as exc:  # noqa: BLE001
                log(f"    {client_name} event {eid}: {type(exc).__name__}; skipped"); continue
            for e in full:
                rows += _pick_rows(e, label, client_name)
            cat_rows += _catalog_rows(cat.filter(f"time > {t0 - 1}") if False else [ev], client_name)
    return rows, cat_rows


def harvest_fdsn_region(seq, spec, spans, label, arrivals=True):
    client_name = spec["client"]
    rows, cat_rows = [], []
    for (t0, t1) in spans:
        # arrivals: hourly chunks (NOA refuses more); catalogue only: 30-day chunks split on
        # 413, except NOA whose service streams a busy fortnight too slowly to finish: daily
        evs = _chunked_region_events(client_name, seq, t0, t1, spec.get("min_mag"), arrivals,
                                     hours=1.0 if arrivals else (24.0 if client_name == "NOA" else 24.0 * 30))
        log(f"    {client_name}: {len(evs)} events in {t0}..{t1}")
        for ev in evs:
            if arrivals:
                rows += _pick_rows(ev, label, client_name)
            cat_rows += _catalog_rows([ev], client_name)
    return rows, cat_rows


def harvest_usgs_phase_data(seq, spec, spans, label):
    """USGS ComCat catalogue, then the phase-data product's QuakeML per event."""
    from obspy import read_events
    from obspy.clients.fdsn.header import FDSNException
    client = client_for("USGS")
    rows, cat_rows = [], []
    for (t0, t1) in spans:
        _record("get_events", "USGS", starttime=t0, endtime=t1, **_region(seq, spec.get("min_mag", 4.5)))
        try:
            cat = client.get_events(starttime=t0, endtime=t1, **_region(seq, spec.get("min_mag", 4.5)))
        except FDSNException as exc:
            log(f"    USGS: {type(exc).__name__}: {str(exc)[:100]}"); continue
        log(f"    USGS: {len(cat)} events M>={spec.get('min_mag', 4.5)} in {t0}..{t1}")
        for ev in cat:
            eid = _event_id(ev).split("&")[0]
            url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?eventid={eid}&format=geojson"
            _record("phase-data", "USGS", eventid=eid)
            try:
                d = json.load(urllib.request.urlopen(url, timeout=120))
                prods = d["properties"]["products"].get("phase-data", [])
                if not prods:
                    continue
                c = prods[0]["contents"]
                qml = next(k for k in c if k.endswith("quakeml.xml"))
                raw = urllib.request.urlopen(c[qml]["url"], timeout=120).read()
                full = read_events(io.BytesIO(raw))
            except Exception as exc:  # noqa: BLE001
                log(f"    USGS {eid}: {type(exc).__name__}: {str(exc)[:80]}"); continue
            for e in full:
                rows += _pick_rows(e, label, "USGS phase-data")
            cat_rows += _catalog_rows([ev], "USGS")
    return rows, cat_rows


# JMA deck files: 96-byte fixed records, Japan Standard Time.
JST = 9 * 3600.0


def _jma_time(year, month, day, hh, mm, ss):
    from obspy import UTCDateTime
    return UTCDateTime(year, month, day, hh, mm) + float(ss) - JST


def _fixed(s: str, decimals: int):
    """A JMA fixed-format number with implied decimals: '1050' with 2 -> 10.50."""
    s = s.strip()
    if not s:
        return np.nan
    return float(s) / (10 ** decimals)


def parse_jma_deck(text: str, label: str):
    """Hypocenter records J/U/I, arrival records '_', end record 'E'."""
    from obspy import UTCDateTime
    picks, cats = [], []
    cur = None
    for line in text.splitlines():
        if not line:
            continue
        c = line[0]
        if c in "JUI":
            try:
                year = int(line[1:5]); mon = int(line[5:7]); day = int(line[7:9])
                hh = int(line[9:11]); mi = int(line[11:13]); ss = _fixed(line[13:17], 2)
                ss = 0.0 if ss != ss else ss
                lat = int(line[21:24]) + (_fixed(line[24:28], 2) if line[24:28].strip() else 0.0) / 60.0
                lon = int(line[32:36]) + (_fixed(line[36:40], 2) if line[36:40].strip() else 0.0) / 60.0
                dstr = line[44:49]
                # depth-free method: F5.2 with implied decimals; depth-slice: I3 then two blanks
                dep = (float(dstr[:3]) if dstr[3:5] == "  " and dstr[:3].strip() else _fixed(dstr, 2)) if dstr.strip() else np.nan
                mstr = line[52:54].strip()
                mag = (int(mstr) / 10.0 if mstr.lstrip("-").isdigit() and not mstr.startswith("-") else np.nan)
                if mstr.startswith("-") and mstr[1:].isdigit():
                    mag = -int(mstr[1:]) / 10.0
                ot = _jma_time(year, mon, day, hh, mi, ss)
                flag = line[95:96]
                cur = dict(event=f"jma{year:04d}{mon:02d}{day:02d}{hh:02d}{mi:02d}{ss:05.2f}", origin=ot.datetime,
                           lat=lat, lon=lon, depth_km=dep, mag=mag, source="JMA deck", year=year, month=mon,
                           det_flag=flag)
                if c == "J":
                    cats.append({k: v for k, v in cur.items() if k not in ("year", "month", "det_flag")})
            except ValueError:
                cur = None
        elif c == "_" and cur is not None:
            sta = line[1:7].strip()
            day = int(line[13:15]) if line[13:15].strip() else None
            year, mon = cur["year"], cur["month"]
            # day of arrival may roll over the month relative to the hypocenter
            flag92 = line[91:92]
            weight = line[95:96]
            mode = "automatic" if flag92.islower() and flag92.isalpha() else "manual"
            t_first = None
            for k, (ph_s, hh_s, mm_s, ss_s) in enumerate(((line[15:19], line[19:21], line[21:23], line[23:27]),
                                                          (line[27:31], line[19:21], line[31:33], line[33:37]))):
                ph = ph_s.strip().upper()
                if not ph or not ss_s.strip():
                    continue
                phase = ph.lstrip("IE")[:1]
                if phase not in ("P", "S") or day is None:
                    continue
                try:
                    t = _jma_time(year, mon, day, int(hh_s), int(mm_s), _fixed(ss_s, 2))
                except ValueError:
                    continue
                if k == 0:
                    t_first = t
                elif t_first is not None and t < t_first - 30:
                    t = t + 3600          # the second phase carries minutes only; it rolled into the next hour
                picks.append(dict(sequence=label, event=cur["event"], origin=cur["origin"], mag=cur["mag"],
                                  station=sta, channel="", phase=phase, time=t.datetime, mode=mode,
                                  status=cur["det_flag"], method=f"flag92={flag92}", agency="JMA",
                                  time_weight=(0.0 if weight == "0" else 1.0), onset=("impulsive" if ph.startswith("I") else "emergent" if ph.startswith("E") else ""),
                                  uncertainty=np.nan, network="", source="JMA deck",
                                  reference_ok=(mode == "manual")))
        elif c == "E":
            cur = None
    return picks, cats


def harvest_jma_deck(seq, spec, spans, label, cache_dir: Path):
    picks, cats = [], []
    for ym in spec["months"]:
        url = f"https://www.data.jma.go.jp/eqev/data/bulletin/data/deck/d{ym}.zip"
        local = cache_dir / f"d{ym}.zip"
        _record("jma_deck", None, url=url)
        if not local.exists():
            download(url, local)
        with zipfile.ZipFile(local) as z:
            for name in z.namelist():
                text = z.read(name).decode("utf-8", errors="replace")
                p, c = parse_jma_deck(text, label)
                picks += p; cats += c
        log(f"    JMA d{ym}: {len(cats)} hypocentres, {len(picks)} arrivals so far")
    # keep the region and spans only
    lo = min(t0 for t0, _ in spans) - reg.ORIGIN_LEAD_S; hi = max(t1 for _, t1 in spans)
    from obspy.geodetics import locations2degrees
    keep_ev = {c["event"] for c in cats
               if lo <= utc(c["origin"]) <= hi and locations2degrees(seq["lat"], seq["lon"], c["lat"], c["lon"]) <= seq["radius"]}
    return [p for p in picks if p["event"] in keep_ev], [c for c in cats if c["event"] in keep_ev]


def jma_station_coords(cache_dir: Path) -> pd.DataFrame:
    """JMA station list (deck/stations.zip, Shift-JIS): code in columns 1-6,
    longitude DDDMMmm in 15-21, latitude DDMMmm in 22-27, station number after."""
    local = cache_dir / "jma_stations.zip"
    if not local.exists():
        download("https://www.data.jma.go.jp/eqev/data/bulletin/data/deck/stations.zip", local)
    with zipfile.ZipFile(local) as z:
        text = z.read(z.namelist()[0]).decode("shift_jis", errors="replace")
    rows = []
    for line in text.splitlines():
        if len(line) < 27:
            continue
        code, lon_s, lat_s, num = line[:6].strip(), line[14:21], line[21:27], line[27:].strip()
        try:
            lon = int(lon_s[:3]) + int(lon_s[3:7]) / 100.0 / 60.0
            lat = int(lat_s[:2]) + int(lat_s[2:6]) / 100.0 / 60.0
        except ValueError:
            continue
        rows.append(dict(code=code, lat=lat, lon=lon, jma_number=num))
    return pd.DataFrame(rows).drop_duplicates("code")


def parse_hypodd_pha(text: str, label: str, network: str):
    """hypoDD phase format: '# YR MO DY HR MI SEC LAT LON DEP MAG EH EZ RMS ID' then 'STA TT WGT PHA'."""
    from obspy import UTCDateTime
    picks, cats, cur = [], [], None
    for line in text.splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "#":
            yr, mo, dy, hr, mi = map(int, s[1:6]); sec = float(s[6])
            lat, lon, dep, mag = map(float, s[7:11])
            eid = s[14] if len(s) > 14 else f"{yr}{mo:02d}{dy:02d}{hr:02d}{mi:02d}"
            ot = UTCDateTime(yr, mo, dy, hr, mi) + sec
            cur = dict(event=str(eid), origin=ot.datetime, lat=lat, lon=lon, depth_km=dep, mag=mag, source="Zenodo pha")
            cats.append(cur)
        elif cur is not None and len(s) >= 4:
            sta, tt, wgt, pha = s[0], float(s[1]), float(s[2]), s[3].upper()[:1]
            if pha not in ("P", "S"):
                continue
            picks.append(dict(sequence=label, event=cur["event"], origin=cur["origin"], mag=cur["mag"],
                              station=(f"{network}.{sta}" if network else sta), channel="", phase=pha,
                              time=(UTCDateTime(cur["origin"]) + tt).datetime, mode="manual", status="",
                              method="hypoDD pha", agency="WEBNET", time_weight=wgt, onset="", uncertainty=np.nan,
                              network=network, source="Zenodo pha", reference_ok=True))
    return picks, cats


def harvest_zenodo_pha(seq, spec, spans, label, cache_dir: Path):
    rec = spec["record"]
    wanted = [k for k in (spec["file"], spec.get("quality_file"), spec.get("stations_file")) if k]
    api_cache = cache_dir / f"zenodo_{rec}.json"
    if all((cache_dir / k).exists() for k in wanted) or (cache_dir / spec["file"]).exists() and api_cache.exists():
        api = json.loads(api_cache.read_text()) if api_cache.exists() else {"doi": f"10.5281/zenodo.{rec}", "files": []}
    else:
        try:
            if not api_cache.exists():
                download(f"https://zenodo.org/api/records/{rec}", api_cache, tries=3)   # 504 under load, 403 when rate-limited
            api = json.loads(api_cache.read_text())
        except RuntimeError:
            if not (cache_dir / spec["file"]).exists():
                raise
            log("    Zenodo API unreachable; using the cached phase file and the registry DOI")
            api = {"doi": f"10.5281/zenodo.{rec}", "files": []}
    # the API's /content links answer 403 under Zenodo's rate limiting; the record's
    # download links are the documented public form and are tried first
    files = {k: f"https://zenodo.org/records/{rec}/files/{k}?download=1" for k in wanted}
    api_links = {f["key"]: f["links"]["self"] for f in api.get("files", [])}
    for k in wanted:
        api_links.setdefault(k, files[k])
    _record("zenodo", None, record=rec, doi=api.get("doi"), files=wanted)
    texts = {}
    for key in (spec["file"], spec.get("quality_file"), spec.get("stations_file")):
        if not key:
            continue
        local = cache_dir / key
        if not local.exists():
            try:
                download(files[key], local, tries=2)
            except RuntimeError:
                try:
                    download(api_links[key], local, tries=2)
                except RuntimeError:
                    if key == spec["file"]:
                        raise
                    log(f"    optional file {key} not obtainable (Zenodo 403); continuing without it"); continue
        texts[key] = local.read_text(errors="replace")
    picks, cats = parse_hypodd_pha(texts[spec["file"]], label, spec.get("network", ""))
    if spec.get("quality_file") in texts:
        q_ids = {c["event"] for c in parse_hypodd_pha(texts[spec["quality_file"]], label, spec.get("network", ""))[1]}
        for p in picks:
            p["status"] = "quality1" if p["event"] in q_ids else "all"
    lo = min(t0 for t0, _ in spans) - reg.ORIGIN_LEAD_S; hi = max(t1 for _, t1 in spans)
    keep = {c["event"] for c in cats if lo <= utc(c["origin"]) <= hi}
    log(f"    Zenodo {rec}: {len(cats)} events, {len(picks)} picks; {len(keep)} events in the spans")
    return [p for p in picks if p["event"] in keep], [c for c in cats if c["event"] in keep]


# ── windows ──────────────────────────────────────────────────────────────────

def choose_windows(seq, catalog: pd.DataFrame):
    w = seq["windows"]
    if w["kind"] == "mainshock":
        t0 = utc(seq["mainshock"]) + w["start_s"]
        return [dict(t0=t0, t1=t0 + w["minutes"] * 60, rule=f"mainshock+{w['start_s']}s, {w['minutes']} min")]
    span0, span1 = utc(w["span"][0]), utc(w["span"][1])
    step = w["hours"] * 3600.0
    t = pd.to_datetime(catalog["origin"], utc=True)
    secs = (t.astype("int64") / 1e9).to_numpy()
    secs = secs[(secs >= float(span0)) & (secs <= float(span1))]
    # candidate starts on a 15-minute grid; count events in [s, s+step); greedy non-overlap
    starts = np.arange(float(span0), float(span1) - step + 1, 900.0)
    counts = np.array([((secs >= s) & (secs < s + step)).sum() for s in starts])
    order = np.lexsort((starts, -counts))             # most events first, earliest first on ties
    chosen = []
    for i in order:
        s = starts[i]
        if counts[i] == 0 or any(abs(s - c) < step for c in chosen):
            continue
        chosen.append(s)
        if len(chosen) >= w["n_windows"]:
            break
    chosen.sort()
    return [dict(t0=utc(s), t1=utc(s + step), rule=f"busiest {w['hours']} h window in {w['span'][0]}..{w['span'][1]}, "
                 f"{int(counts[np.where(starts == s)[0][0]])} catalogue events") for s in chosen]


# ── stations and waveforms ───────────────────────────────────────────────────

def _inventory(seq, route, t0, t1):
    c = client_for(route)          # obspy's RoutingClient is a factory, so no isinstance here
    kw = dict(latitude=seq["lat"], longitude=seq["lon"], maxradius=seq["radius"] + 0.5,
              channel="HH?,EH?,BH?", starttime=t0, endtime=t1, level="channel")
    _record("get_stations", route, **kw)
    try:
        return c.get_stations(**kw)
    except Exception as exc:  # noqa: BLE001
        log(f"    inventory via {route}: {type(exc).__name__}: {str(exc)[:80]}")
        return None


def station_table(seq, picks: pd.DataFrame, windows, routes):
    """Candidates within the radius with a pickable band, their route, and the
    reference-pick count inside the windows; bare pick station codes are
    resolved to NET.STA by matching against the inventories."""
    from obspy.geodetics import locations2degrees
    t0 = min(w["t0"] for w in windows); t1 = max(w["t1"] for w in windows)
    invs = [(r, _inventory(seq, r, t0, t1)) for r in routes]
    rows, by_code = [], {}
    for route, inv in invs:
        if inv is None:
            continue
        for net in inv:
            for sta in net:
                bands = {ch.code[:2]: ch.sample_rate for ch in sta}
                band = next((b for b in seq["channel_pref"] if b in bands), None)
                if band is None:
                    continue
                key = f"{net.code}.{sta.code}"
                rows.append(dict(station=key, network=net.code, code=sta.code, band=band, rate=bands[band],
                                 km=locations2degrees(seq["lat"], seq["lon"], sta.latitude, sta.longitude) * 111.19,
                                 route=reg.NETWORK_ROUTE.get(net.code, route)))
                by_code.setdefault(sta.code, []).append(key)
    for extra in seq.get("extra_stations", []):
        net, code = extra.split(".")
        if not any(r["station"] == extra for r in rows):
            rows.append(dict(station=extra, network=net, code=code, band="BH", rate=np.nan, km=np.nan,
                             route=reg.NETWORK_ROUTE.get(net, "IRIS")))
    table = pd.DataFrame(rows).drop_duplicates("station") if rows else pd.DataFrame(
        columns=["station", "network", "code", "band", "rate", "km", "route"])

    # resolve bare station codes in the picks
    resolved, ambiguous = {}, {}
    bare = sorted(set(picks.loc[(picks["network"] == "") | (picks["source"] == "JMA deck"), "station"]))
    for code in bare:
        cands = sorted(set(by_code.get(code, [])))
        if len(cands) == 1:
            resolved[code] = cands[0]
        elif len(cands) > 1:
            ambiguous[code] = cands; resolved[code] = cands[0]
    picks = picks.copy()
    picks["station"] = picks["station"].map(lambda s: resolved.get(s, s))
    picks["network"] = [("" if src == "JMA deck" else (sta.split(".")[0] if "." in sta else ""))
                        for sta, src in zip(picks["station"], picks["source"])]

    lo = pd.Timestamp(t0.datetime, tz="UTC"); hi = pd.Timestamp(t1.datetime, tz="UTC")
    inwin = picks[(picks["time"] >= lo) & (picks["time"] <= hi) & picks["reference_ok"]]
    counts = inwin.groupby(["station", "phase"]).size().unstack(fill_value=0)
    table["P"] = table["station"].map(lambda s: int(counts.loc[s, "P"]) if s in counts.index and "P" in counts else 0)
    table["S"] = table["station"].map(lambda s: int(counts.loc[s, "S"]) if s in counts.index and "S" in counts else 0)
    table["picks"] = table["P"] + table["S"]
    table = table.sort_values(["picks", "km", "station"], ascending=[False, True, True]).reset_index(drop=True)
    station_map = pd.DataFrame([dict(code=k, station=v, ambiguous=";".join(ambiguous.get(k, []))) for k, v in resolved.items()]
                               + [dict(code=k, station="", ambiguous="") for k in bare if k not in resolved])
    if "JMA deck" in set(picks["source"]) and len(station_map):
        from obspy.geodetics import locations2degrees
        coords = jma_station_coords(OUT_ROOT / seq["key"] / "cache").set_index("code")
        station_map["lat"] = station_map["code"].map(coords["lat"])
        station_map["lon"] = station_map["code"].map(coords["lon"])
        station_map["jma_number"] = station_map["code"].map(coords["jma_number"])
        station_map["km"] = [locations2degrees(seq["lat"], seq["lon"], la, lo) * 111.19 if la == la else np.nan
                             for la, lo in zip(station_map["lat"], station_map["lon"])]
        station_map = station_map.sort_values("km")
    return table, picks, station_map


def fetch_station(route, key, band, t0, t1):
    net, sta = key.split(".")
    c = client_for(route)
    _record("get_waveforms", route, network=net, station=sta, channel=band + "?", starttime=t0, endtime=t1)
    try:
        st = c.get_waveforms(network=net, station=sta, location="*", channel=band + "?", starttime=t0, endtime=t1)
    except Exception:  # noqa: BLE001
        return None
    if len(st) == 0:
        return None
    st.merge(fill_value=0)
    if len(st) < 3:
        return None
    expected = (t1 - t0) * st[0].stats.sampling_rate
    if st[0].stats.npts < 0.5 * expected:
        return None
    return st


# ── the build ────────────────────────────────────────────────────────────────

def build(key: str, steps: list, force: bool):
    global _log_fh
    seq = reg.BY_KEY[key]
    out = OUT_ROOT / key; out.mkdir(parents=True, exist_ok=True)
    (out / "waveforms").mkdir(exist_ok=True); cache = out / "cache"; cache.mkdir(exist_ok=True)
    _log_fh = open(out / "build.log", "a")
    log(f"=== {seq['label']} ({key}) steps={steps} force={force}")
    label = seq["label"]

    # stage 1: catalogue (needed to choose busiest windows); for mainshock windows it is the harvest span
    cat_path = out / "catalog.parquet"
    if "catalog" in steps and (force or not _have(cat_path)):
        w = seq["windows"]
        if w["kind"] == "busiest":
            spec = w["catalog"]
            span = (utc(w["span"][0]), utc(w["span"][1]))
            if spec["kind"] == "fdsn_region":
                _, cat_rows = harvest_fdsn_region(seq, spec, [span], label, arrivals=False)
            elif spec["kind"] == "zenodo_pha":
                _, cat_rows = harvest_zenodo_pha(seq, seq["picks"][0], [span], label, cache)
            else:
                raise ValueError(spec["kind"])
        else:
            t0 = utc(seq["mainshock"]) + w["start_s"]
            span = (t0 - reg.ORIGIN_LEAD_S, t0 + w["minutes"] * 60)
            cat_rows = []
            for spec in seq["picks"]:
                if spec["kind"] == "fdsn_per_event":
                    c = client_for(spec["client"]); _record("get_events", spec["client"], starttime=span[0], endtime=span[1], **_region(seq, spec.get("min_mag")))
                    try:
                        cat_rows += _catalog_rows(c.get_events(starttime=span[0], endtime=span[1], **_region(seq, spec.get("min_mag"))), spec["client"])
                    except Exception as exc:  # noqa: BLE001
                        log(f"    catalogue {spec['client']}: {type(exc).__name__}")
                elif spec["kind"] == "fdsn_region":
                    _, rows = harvest_fdsn_region(seq, spec, [span], label, arrivals=False); cat_rows += rows
                elif spec["kind"] == "jma_deck":
                    _, rows = harvest_jma_deck(seq, spec, [span], label, cache); cat_rows += rows
                elif spec["kind"] == "usgs_phase_data":
                    pass
        cat = pd.DataFrame(cat_rows, columns=["event", "origin", "lat", "lon", "depth_km", "mag", "source"])
        if len(cat):
            cat["origin"] = pd.to_datetime(cat["origin"], utc=True)
            cat = cat.drop_duplicates(["source", "event"]).sort_values("origin")
        cat.to_parquet(cat_path); log(f"  catalogue: {len(cat)} events -> {cat_path.name}")
    cat = pd.read_parquet(cat_path) if _have(cat_path) else pd.DataFrame(columns=["event", "origin"])

    win_path = out / "windows.csv"
    if "windows" in steps and (force or not _have(win_path)):
        wins = choose_windows(seq, cat)
        pd.DataFrame([dict(t0=str(w["t0"]), t1=str(w["t1"]), rule=w["rule"]) for w in wins]).to_csv(win_path, index=False)
        for w in wins:
            log(f"  window {w['t0']} .. {w['t1']}  ({w['rule']})")
    wins = [dict(t0=utc(r.t0), t1=utc(r.t1), rule=r.rule) for r in pd.read_csv(win_path).itertuples()] if _have(win_path) else []
    spans = [(w["t0"] - reg.ORIGIN_LEAD_S, w["t1"]) for w in wins]

    picks_path = out / "picks.parquet"
    if "picks" in steps and (force or not _have(picks_path)):
        rows, extra_cat = [], []
        for spec in seq["picks"]:
            k = spec["kind"]; log(f"  picks: {k} {spec.get('client', '')}")
            if k == "fdsn_per_event":
                r, c = harvest_fdsn_per_event(seq, spec, spans, label)
            elif k == "fdsn_region":
                r, c = harvest_fdsn_region(seq, spec, spans, label, arrivals=True)
            elif k == "usgs_phase_data":
                r, c = harvest_usgs_phase_data(seq, spec, spans, label)
            elif k == "jma_deck":
                r, c = harvest_jma_deck(seq, spec, spans, label, cache)
            elif k == "zenodo_pha":
                r, c = harvest_zenodo_pha(seq, spec, spans, label, cache)
            else:
                raise ValueError(k)
            rows += r; extra_cat += c
            log(f"    {len(r)} arrivals")
        picks = pd.DataFrame(rows, columns=PICK_COLS)
        if len(picks):
            picks["origin"] = pd.to_datetime(picks["origin"], utc=True)
            picks["time"] = pd.to_datetime(picks["time"], utc=True)
            picks = picks.drop_duplicates(["source", "event", "station", "phase", "time"]).sort_values(["time", "station"])
        picks.to_parquet(picks_path)
        log(f"  picks: {len(picks)} arrivals from {picks.event.nunique() if len(picks) else 0} events "
            f"({int(picks.reference_ok.sum()) if len(picks) else 0} reference_ok) -> {picks_path.name}")
    picks = pd.read_parquet(picks_path) if _have(picks_path) else pd.DataFrame(columns=PICK_COLS)

    sta_path = out / "stations.csv"
    if "stations" in steps and wins and (force or not _have(sta_path)):
        table, picks2, station_map = station_table(seq, picks, wins, seq["waveform_routes"])
        if len(station_map):
            station_map.to_csv(out / "station_map.csv", index=False)
            n_res = int((station_map.station != "").sum())
            log(f"  station map: {n_res}/{len(station_map)} bare codes resolved to NET.STA")
            picks2.to_parquet(picks_path)          # picks now carry resolved stations
        table["fetched"] = False
        table.to_csv(sta_path, index=False)
        log(f"  stations: {len(table)} candidates ({int((table.picks > 0).sum())} with reference picks in the windows)")

    if "waveforms" in steps and wins and _have(sta_path):
        table = pd.read_csv(sta_path)
        got = {}
        for w in wins:
            tag = f"{w['t0'].strftime('%Y%m%dT%H%M%S')}__{w['t1'].strftime('%Y%m%dT%H%M%S')}"
            n_ok = 0
            for r in table.itertuples():
                if n_ok >= seq["n_stations"] and r.station not in seq.get("extra_stations", []):
                    continue
                f = out / "waveforms" / f"{r.station}__{r.band}__{tag}.mseed"
                if f.exists() and not force:
                    n_ok += 1; got[r.station] = True; continue
                st = fetch_station(r.route, r.station, r.band, w["t0"], w["t1"])
                if st is None and r.route not in seq["waveform_routes"]:
                    for alt in seq["waveform_routes"]:
                        st = fetch_station(alt, r.station, r.band, w["t0"], w["t1"])
                        if st is not None:
                            break
                if st is None:
                    continue
                st.write(str(f), format="MSEED")
                n_ok += 1; got[r.station] = True
                log(f"    {tag} {r.station:<10} {r.band}@{st[0].stats.sampling_rate:g} Hz  {r.km if r.km == r.km else -1:5.0f} km  {r.P:3d} P {r.S:3d} S")
            log(f"  window {tag}: {n_ok} stations with data")
        table["fetched"] = table["station"].map(lambda s: bool(got.get(s, False)))
        table.to_csv(sta_path, index=False)

    if "manifest" in steps:
        files = {}
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name not in ("manifest.json", "build.log") and "cache" not in f.parts:
                files[str(f.relative_to(out))] = dict(sha256=sha256(f), bytes=f.stat().st_size)
        import obspy
        man = dict(
            key=key, label=label, built=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=git_commit(), python=sys.version.split()[0], obspy=obspy.__version__,
            pandas=pd.__version__, registry=seq, reference_ok=REFERENCE_OK,
            queries=QUERIES[-400:], n_queries=len(QUERIES),
            counts=dict(catalog_events=int(len(cat)), picks=int(len(picks)),
                        picks_reference_ok=int(picks.reference_ok.sum()) if len(picks) else 0,
                        picks_manual=int((picks["mode"] == "manual").sum()) if len(picks) else 0,
                        stations_fetched=int(pd.read_csv(sta_path).fetched.sum()) if _have(sta_path) else 0,
                        windows=len(wins)),
            files=files,
        )
        (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
        log(f"  manifest: {len(files)} files hashed")
    _log_fh.close(); _log_fh = None


def verify(key: str) -> bool:
    out = OUT_ROOT / key
    man = json.loads((out / "manifest.json").read_text())
    bad = 0
    for rel, info in man["files"].items():
        f = out / rel
        if not f.exists():
            print(f"MISSING {rel}"); bad += 1
        elif sha256(f) != info["sha256"]:
            print(f"CHANGED {rel}"); bad += 1
    print(f"{key}: {len(man['files'])} files, {bad} problems")
    return bad == 0


def index() -> None:
    rows = []
    for d in sorted(OUT_ROOT.iterdir()):
        m = d / "manifest.json"
        if not m.exists():
            continue
        man = json.loads(m.read_text()); s = man["registry"]
        rows.append(dict(key=man["key"], label=man["label"], regime=s["regime"], tier=s["tier"], suite=s["suite"],
                         built=man["built"], git_commit=man["git_commit"][:12], **man["counts"]))
    pd.DataFrame(rows).to_csv(OUT_ROOT / "index.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


ALL_STEPS = ["catalog", "windows", "picks", "stations", "waveforms", "manifest"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", default=["all"])
    ap.add_argument("--sequence", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--verify", metavar="KEY")
    a = ap.parse_args()
    if a.index:
        index(); return
    if a.verify:
        sys.exit(0 if verify(a.verify) else 1)
    steps = ALL_STEPS if "all" in a.steps else a.steps
    keys = [s["key"] for s in reg.SEQUENCES] if a.all else a.sequence
    failed = []
    for k in keys:
        try:
            build(k, steps, a.force)
        except Exception as exc:  # noqa: BLE001  one sequence must not take the others down
            import traceback
            log(f"!!! {k} failed: {type(exc).__name__}: {exc}"); traceback.print_exc()
            failed.append(k)
            global _log_fh
            if _log_fh:
                _log_fh.close(); _log_fh = None
    if failed:
        print(f"FAILED: {failed}"); sys.exit(1)


if __name__ == "__main__":
    main()
