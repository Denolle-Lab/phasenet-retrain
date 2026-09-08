#!/usr/bin/env python3
"""
make_heldout_dashboard.py

Renders docs/heldout_testset_dashboard.html: an overview map of the held-out
test sequences and, per sequence, a detail map of the catalogue events inside
the scoring windows and the stations (filled where waveforms were fetched,
hollow where the reference has picks but no open waveforms), with one line on
why the sequence was chosen, where the picks come from and what it tests.
Numbers and positions come from data/heldout_testset/*; the prose lives here.
Plotly is loaded from cdnjs.

    python scripts/make_heldout_dashboard.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_testset_registry as reg  # noqa: E402

DATA = REPO_ROOT / "data" / "heldout_testset"
OUT = REPO_ROOT / "docs" / "heldout_testset_dashboard.html"
LEAD = reg.ORIGIN_LEAD_S
MAX_EVENTS = 400
MAX_STATIONS = 40

REGIME = {"msas": "Mainshock–aftershock", "vt": "Volcano-tectonic", "swarm": "Fluid-driven swarm"}

# why chosen · what it tests · gap
TEXT = {
    "kaikoura_2016": ("M7.8 across a dozen crustal faults on a sparse permanent network; most GeoNet picks sit 80–120 km out.",
                      ["regional S", "sparse network", "overlapping coda"], ""),
    "norcia_2016": ("Largest shock of Amatrice–Visso–Norcia; dense permanent plus temporary stations, aftershocks seconds apart.",
                    ["near-field density", "in-domain for instance"], ""),
    "thessaly_2021": ("Normal-faulting doublet; NOA picks S on stations 45–90 km out where every weight finds P and misses S.",
                      ["regional S ceiling", "held-out year 2021"], ""),
    "kahramanmaras_2023": ("Mw 7.8 and 7.6 doublet on a 350 km rupture; AFAD read 566k P and 461k S by hand.",
                           ["dense overlapping aftershocks", "many operators", "100 Hz KO"], "AFAD bulk readings still to add; ISC preliminary"),
    "noto_2024": ("Mw 7.5 that ended a three-year swarm; JMA unified picks, after the parent's 2014–2021 training years.",
                  ["held out for jma_wc too", "dense sequence"], "Hi-net account needed for waveforms and picks"),
    "hualien_2024": ("Mw 7.4 with a conjugate fault system; CWA hand-picked thousands of aftershocks; SeisBench's CWA set ends 2021.",
                     ["near-field density", "collision–subduction"], "CWA GDMS account needed; only IU.TATO is open"),
    "petrinja_2020": ("Mw 6.4 intraplate in a slow-deforming region; 255,729 hand-picked onsets over six months, complete to ML 1.2.",
                      ["sparse 2020 network", "small aftershocks", "intraplate"], ""),
    "samos_2020": ("Mw 7.0 offshore normal fault between the Greek and Turkish networks.",
                   ["offshore source", "two operators"], ""),
    "adriatic_2022": ("Mw 5.5 offshore Pesaro recorded on land only: one-sided geometry, moderate magnitude.",
                      ["offshore, one-sided", "moderate mainshock"], ""),
    "reykjanes_2023": ("A 15 km dike under Grindavík: 20,000 earthquakes and forty M4+ within hours on 2023-11-10.",
                       ["dike propagation rate", "M4+ minutes apart"], "IMO SIL picks on request; ISC preliminary"),
    "fagradalsfjall_2021": ("A year of pre-eruptive unrest ending in the Feb–Mar 2021 dike on the plate boundary.",
                            ["dike seismicity", "boundary strike-slip", "held-out year 2021"], "IMO station codes barely resolve on EIDA"),
    "la_palma_2021": ("Nine pre-eruptive days of seismicity from 12 to 35 km depth; IGN hand picks, 1,412 events in the span.",
                      ["deep VT", "ocean island", "emergent onsets"], ""),
    "santorini_2025": ("A magmatic dike under Kolumbo triggering tectonic seismicity toward Anydros; ~100 events per 3 h at the peak.",
                       ["offshore VT", "magmatic–tectonic mix", "high rate"], ""),
    "etna_2022_2024": ("Flank and summit VT swarms of an open-conduit volcano under tremor; Etna is removed from INSTANCE by the place hold-out.",
                       ["VT under tremor", "shallow, dense network"], ""),
    "west_bohemia_2018": ("The type example of a CO₂-driven swarm at Nový Kostel; WEBNET processed more than 1,500 events by hand.",
                          ["fluid swarm", "ML 0.5–3.8", "station 1 km away"], "classic WEBNET waveforms only in the Zenodo tarball"),
    "maurienne_2017": ("71,000 Alpine events over 20 months, slip and pore pressure taking turns driving the migration.",
                       ["long-lived swarm", "migration", "small magnitudes"], ""),
    "noto_swarm_2023": ("The fluid-driven swarm of 2020–23 that led to the Mw 7.5, with the M6.5 Suzu event inside it; JMA unified picks.",
                        ["updip migration", "station 3 km away"], "Hi-net account needed for waveforms"),
    "campi_flegrei_2023": ("Hydrothermal–magmatic unrest under Naples with the M4.2 of 2023-09-27; INGV-OV hand picks.",
                           ["caldera VT", "urban noise", "shallow"], "INGV national service holds few caldera events; OV bulletin needed"),
    "corinth_thiva_2020": ("Rift swarms at Perachora and Thiva with pore-pressure migration fronts.",
                           ["fluid migration", "small events"], ""),
}
SOURCE_NAME = {"INGV": "INGV", "NOA": "NOA", "GEONET": "GeoNet", "ISC": "ISC bulletin", "franceseisme": "BCSF-RENASS"}


def source_label(seq):
    parts = []
    for p in seq["picks"]:
        k = p["kind"]
        if k in ("fdsn_per_event", "fdsn_region"):
            parts.append(SOURCE_NAME.get(p["client"], p["client"]))
        elif k == "usgs_phase_data":
            parts.append("NEIC M≥4.5")
        elif k == "jma_deck":
            parts.append("JMA deck file")
        elif k == "zenodo_pha":
            parts.append("WEBNET (Zenodo)")
    return " + ".join(parts)


def when(seq):
    if seq["mainshock"]:
        return seq["mainshock"][:10] + (f" · M{seq['mag']:g}" if seq.get("mag") else "")
    a, b = seq["windows"]["span"]
    return f"{a[:10]} → {b[:10]}"


def circle(lat, lon, r_deg, n=90):
    """Small-circle of angular radius r_deg (degrees of arc) around (lat, lon)."""
    la1, lo1, d = map(math.radians, (lat, lon, r_deg))
    out_lat, out_lon = [], []
    for k in range(n + 1):
        br = 2 * math.pi * k / n
        la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(br))
        lo2 = lo1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
        out_lat.append(round(math.degrees(la2), 4)); out_lon.append(round(math.degrees(lo2), 4))
    return out_lat, out_lon


def load_sequence(seq, idx):
    k = seq["key"]; d = DATA / k
    windows = pd.read_csv(d / "windows.csv") if (d / "windows.csv").exists() else pd.DataFrame(columns=["t0", "t1"])
    cat = pd.read_parquet(d / "catalog.parquet") if (d / "catalog.parquet").exists() else pd.DataFrame()
    events = []
    if len(cat) and len(windows):
        t = pd.to_datetime(cat["origin"], utc=True)
        mask = np.zeros(len(cat), dtype=bool)
        for w in windows.itertuples():
            t0 = pd.Timestamp(w.t0) - pd.Timedelta(seconds=LEAD); t1 = pd.Timestamp(w.t1)
            mask |= ((t >= t0) & (t <= t1)).to_numpy()
        sub = cat[mask].copy()
        sub["mag"] = pd.to_numeric(sub["mag"], errors="coerce")
        n_all = len(sub)
        sub = sub.sort_values("mag", ascending=False, na_position="last").head(MAX_EVENTS)
        for r in sub.itertuples():
            if not (r.lat == r.lat and r.lon == r.lon):
                continue
            events.append(dict(lat=round(float(r.lat), 4), lon=round(float(r.lon), 4),
                               mag=(round(float(r.mag), 1) if r.mag == r.mag else None),
                               dep=(round(float(r.depth_km), 1) if r.depth_km == r.depth_km else None),
                               t=str(pd.Timestamp(r.origin))[:19]))
    else:
        n_all = 0
    stations = []
    st = pd.read_csv(d / "stations.csv") if (d / "stations.csv").exists() else pd.DataFrame()
    if len(st) and "lat" in st.columns:
        keep = st[(st["fetched"] == True) | (st["picks"] > 0)].copy()  # noqa: E712
        keep = keep.sort_values(["fetched", "picks"], ascending=[False, False]).head(MAX_STATIONS)
        for r in keep.itertuples():
            if not (r.lat == r.lat and r.lon == r.lon):
                continue
            stations.append(dict(id=r.station, lat=round(float(r.lat), 4), lon=round(float(r.lon), 4),
                                 km=(round(float(r.km)) if r.km == r.km else None), band=str(r.band),
                                 rate=(float(r.rate) if r.rate == r.rate else None), P=int(r.P), S=int(r.S),
                                 wf=bool(r.fetched)))
    # JMA and other bare-code references with coordinates but no FDSN station
    smp = d / "station_map.csv"
    if smp.exists():
        sm = pd.read_csv(smp)
        if "lat" in sm.columns:
            picks = pd.read_parquet(d / "picks.parquet")
            cnt = picks[picks["reference_ok"]].groupby(["station", "phase"]).size().unstack(fill_value=0)
            have = {s["id"] for s in stations}
            for r in sm.dropna(subset=["lat", "lon"]).itertuples():
                if r.code in have or (isinstance(r.station, str) and r.station in have):
                    continue
                P = int(cnt.loc[r.code, "P"]) if r.code in cnt.index and "P" in cnt else 0
                S = int(cnt.loc[r.code, "S"]) if r.code in cnt.index and "S" in cnt else 0
                if P + S == 0:
                    continue
                stations.append(dict(id=r.code, lat=round(float(r.lat), 4), lon=round(float(r.lon), 4),
                                     km=(round(float(r.km)) if r.km == r.km else None), band="JMA", rate=None, P=P, S=S, wf=False))
            stations = stations[:MAX_STATIONS + 20]
    why, tags, gap = TEXT[k]
    i = idx.loc[k] if k in idx.index else None
    clat, clon = circle(seq["lat"], seq["lon"], seq["radius"])
    return dict(
        key=k, label=seq["label"], regime=seq["regime"], regime_name=REGIME[seq["regime"]], suite=seq["suite"],
        lat=seq["lat"], lon=seq["lon"], radius=seq["radius"], when=when(seq), source=source_label(seq),
        why=why, tags=tags, gap=gap,
        arrivals=int(i["picks_reference_ok"]) if i is not None else 0,
        n_stations_wf=int(i["stations_fetched"]) if i is not None else 0,
        windows=[dict(t0=str(w.t0)[:16].replace("T", " "), t1=str(w.t1)[:16].replace("T", " "), rule=w.rule) for w in windows.itertuples()],
        n_events_windows=int(n_all), events=events, stations=stations, circle=dict(lat=clat, lon=clon),
    )


def main():
    idx = pd.read_csv(DATA / "index.csv").set_index("key")
    refs = pd.read_csv(DATA / "references.csv") if (DATA / "references.csv").exists() else pd.DataFrame(columns=["key"])
    seqs = [load_sequence(s, idx) for s in reg.SEQUENCES]
    for s in seqs:
        rr = refs[refs.key == s["key"]]
        s["refs"] = [dict(kind=r.kind, citation=r.citation, doi=(r.doi if isinstance(r.doi, str) else ""),
                          status=r.status) for r in rr.itertuples()]
    totals = dict(n=len(seqs), arrivals=sum(s["arrivals"] for s in seqs),
                  stations=sum(sum(1 for st in s["stations"] if st["wf"]) for s in seqs),
                  events=sum(s["n_events_windows"] for s in seqs),
                  per_regime={r: sum(1 for s in seqs if s["regime"] == r) for r in REGIME})
    built = str(idx["built"].max())[:10]
    data_json = json.dumps(seqs, ensure_ascii=False, separators=(",", ":"))

    def nav(regime):
        items = []
        for s in seqs:
            if s["regime"] != regime:
                continue
            pill = "acc" if s["suite"] == "acceptance" else "dev"
            items.append(f'''<button type="button" class="nav r-{regime}" data-key="{s["key"]}" aria-pressed="false">
          <span class="nm">{s["label"]}</span><span class="cnt mono">{s["n_events_windows"]} ev · {sum(1 for st in s["stations"] if st["wf"])} sta</span><span class="pill {pill}">{s["suite"][:3]}</span></button>''')
        return "\n".join(items)

    html = f'''<title>Held-Out Sequence Atlas</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --ground: #f6f7f9; --surface: #ffffff; --ink: #14181f; --ink-2: #4b5563; --ink-3: #7a8391; --rule: #dfe3e8;
  --land: #e9ecf0; --ocean: #f6f7f9; --coast: #b9c2cd; --grid: #e3e7ec; --hover-bg: #ffffff;
  --msas: #2a78d6; --vt: #eb6834; --swarm: #1baf7a;
  --acc-bg: #e8f1fb; --acc-ink: #1b5fae; --dev-bg: #eef0f3; --dev-ink: #4b5563; --focus: #2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground: #0f1319; --surface: #171c24; --ink: #f2f4f7; --ink-2: #b8c0cc; --ink-3: #7f8a99; --rule: #2a323d;
    --land: #222a35; --ocean: #0f1319; --coast: #46515f; --grid: #232b36; --hover-bg: #171c24;
    --msas: #3987e5; --vt: #d95926; --swarm: #199e70;
    --acc-bg: #1b2a3f; --acc-ink: #8db8f0; --dev-bg: #1f2630; --dev-ink: #b8c0cc; --focus: #3987e5;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground: #0f1319; --surface: #171c24; --ink: #f2f4f7; --ink-2: #b8c0cc; --ink-3: #7f8a99; --rule: #2a323d;
  --land: #222a35; --ocean: #0f1319; --coast: #46515f; --grid: #232b36; --hover-bg: #171c24;
  --msas: #3987e5; --vt: #d95926; --swarm: #199e70;
  --acc-bg: #1b2a3f; --acc-ink: #8db8f0; --dev-bg: #1f2630; --dev-ink: #b8c0cc; --focus: #3987e5;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--ground); color: var(--ink); font: 14.5px/1.5 "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; }}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, Menlo, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1320px; margin: 0 auto; padding: 24px 22px 40px; }}
header.top {{ display: grid; grid-template-columns: 1fr auto; gap: 12px 28px; align-items: end; border-bottom: 1px solid var(--rule); padding-bottom: 16px; }}
.eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); margin: 0 0 6px; }}
h1 {{ font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-weight: 600; font-size: 32px; line-height: 1.1; margin: 0 0 6px; letter-spacing: -.01em; text-wrap: balance; }}
.lede {{ margin: 0; max-width: 64ch; color: var(--ink-2); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
.chip {{ display: inline-flex; align-items: baseline; gap: 6px; padding: 5px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); font-size: 13px; color: var(--ink-2); }}
.chip b {{ font-family: "IBM Plex Mono", monospace; font-weight: 500; color: var(--ink); }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; align-self: center; }}
.dot.msas, .r-msas .dot {{ background: var(--msas); }} .dot.vt {{ background: var(--vt); }} .dot.swarm {{ background: var(--swarm); }}
section.overview {{ margin-top: 16px; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }}
.bar {{ display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; padding: 9px 14px; border-bottom: 1px solid var(--rule); font-size: 13px; color: var(--ink-2); }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i.c {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.legend i.tri {{ width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid var(--ink); display: inline-block; }}
.legend i.tri.hollow {{ border-bottom-color: var(--ink-3); }}
.legend i.ring {{ width: 12px; height: 12px; border: 1.5px dashed var(--ink-3); border-radius: 50%; display: inline-block; }}
#overview {{ width: 100%; height: 300px; }}
section.detail {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 18px; margin-top: 18px; align-items: start; }}
@media (max-width: 960px) {{ section.detail {{ grid-template-columns: 1fr; }} header.top {{ grid-template-columns: 1fr; }} .chips {{ justify-content: flex-start; }} }}
.navcol h2 {{ font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 14px; letter-spacing: .02em; margin: 14px 0 6px; display: flex; gap: 8px; align-items: center; color: var(--ink-2); }}
.navcol h2:first-child {{ margin-top: 0; }}
.navcol h2 i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.nav {{ appearance: none; width: 100%; display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: center; text-align: left; font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-left-width: 4px; border-radius: 6px; padding: 7px 10px; margin-bottom: 6px; cursor: pointer; }}
.nav.r-msas {{ border-left-color: var(--msas); }} .nav.r-vt {{ border-left-color: var(--vt); }} .nav.r-swarm {{ border-left-color: var(--swarm); }}
.nav[aria-pressed="true"] {{ box-shadow: 0 0 0 2px var(--focus); }}
.nav:focus-visible, .views button:focus-visible {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
.nav .nm {{ font-weight: 500; font-size: 14px; }}
.nav .cnt {{ font-size: 11.5px; color: var(--ink-3); white-space: nowrap; }}
.pill {{ font-size: 10.5px; letter-spacing: .05em; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; white-space: nowrap; }}
.pill.acc {{ background: var(--acc-bg); color: var(--acc-ink); }} .pill.dev {{ background: var(--dev-bg); color: var(--dev-ink); }}
.panel {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }}
.panel .head {{ padding: 12px 16px 10px; border-bottom: 1px solid var(--rule); display: grid; grid-template-columns: 1fr auto; gap: 8px 16px; align-items: start; }}
.panel h3 {{ margin: 0; font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 22px; line-height: 1.15; display: flex; align-items: center; gap: 10px; }}
.panel h3 i {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.panel .meta {{ margin: 3px 0 0; font-size: 12.5px; color: var(--ink-3); }}
.panel .why {{ margin: 6px 0 0; color: var(--ink); max-width: 72ch; }}
.stats {{ display: grid; grid-auto-flow: column; gap: 14px; text-align: right; }}
.stats div {{ font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-3); }}
.stats b {{ display: block; font-family: "IBM Plex Mono", monospace; font-weight: 500; font-size: 18px; color: var(--ink); letter-spacing: 0; text-transform: none; }}
#detail {{ width: 100%; height: 520px; }}
.foot {{ padding: 10px 16px 12px; border-top: 1px solid var(--rule); display: grid; grid-template-columns: 62px 1fr; gap: 5px 10px; font-size: 13px; }}
.foot dt {{ color: var(--ink-3); font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; padding-top: 3px; }}
.foot dd {{ margin: 0; color: var(--ink-2); }}
.tags {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.tag {{ font-size: 12px; padding: 2px 8px; border: 1px solid var(--rule); border-radius: 999px; color: var(--ink-2); background: var(--ground); }}
.gap {{ color: var(--ink-3); }}
.refs {{ display: grid; gap: 3px; }}
.refs a {{ color: var(--ink-2); text-decoration: none; border-bottom: 1px solid var(--rule); }}
.refs a:hover, .refs a:focus-visible {{ color: var(--ink); border-bottom-color: var(--ink); outline: none; }}
.refs .k {{ font-family: "IBM Plex Mono", monospace; font-size: 10.5px; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-3); margin-right: 6px; }}
.refs .doi {{ font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: var(--ink-3); margin-left: 6px; }}
footer {{ margin-top: 26px; padding-top: 12px; border-top: 1px solid var(--rule); font-size: 12.5px; color: var(--ink-3); max-width: 84ch; }}
footer code {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-2); }}
</style>
<div class="wrap">
  <header class="top">
    <div>
      <p class="eyebrow">phasenet-retrain · held-out test set · built {built}</p>
      <h1>Held-Out Sequence Atlas</h1>
      <p class="lede">Nineteen sequences a picker never trains on, in the three regimes a global campaign has to get right. Pick a sequence to see the events inside its scoring windows and the stations the reference and the waveforms come from.</p>
    </div>
    <div class="chips">
      <span class="chip"><b>{totals["n"]}</b> sequences</span>
      <span class="chip"><i class="dot msas"></i><b>{totals["per_regime"]["msas"]}</b> mainshock–aftershock</span>
      <span class="chip"><i class="dot vt"></i><b>{totals["per_regime"]["vt"]}</b> volcano-tectonic</span>
      <span class="chip"><i class="dot swarm"></i><b>{totals["per_regime"]["swarm"]}</b> fluid-driven swarm</span>
      <span class="chip"><b>{totals["events"]:,}</b> events in the scoring windows</span>
      <span class="chip"><b>{totals["arrivals"]:,}</b> reference arrivals</span>
      <span class="chip"><b>{totals["stations"]}</b> stations with waveforms</span>
    </div>
  </header>

  <section class="overview" aria-label="Overview map">
    <div class="bar">
      <div class="legend">
        <span><i class="c" style="background:var(--msas)"></i>Mainshock–aftershock</span>
        <span><i class="c" style="background:var(--vt)"></i>Volcano-tectonic</span>
        <span><i class="c" style="background:var(--swarm)"></i>Fluid-driven swarm</span>
        <span class="mono" style="color:var(--ink-3)">click a sequence to open it below</span>
      </div>
    </div>
    <div id="overview"></div>
  </section>

  <section class="detail">
    <nav class="navcol" aria-label="Sequences">
      <h2><i style="background:var(--msas)"></i>Mainshock–aftershock</h2>
      {nav("msas")}
      <h2><i style="background:var(--vt)"></i>Volcano-tectonic</h2>
      {nav("vt")}
      <h2><i style="background:var(--swarm)"></i>Fluid-driven swarm</h2>
      {nav("swarm")}
    </nav>
    <div class="panel">
      <div class="head">
        <div>
          <h3 id="d-title"><i id="d-dot"></i><span id="d-label"></span></h3>
          <p class="meta mono" id="d-meta"></p>
          <p class="why" id="d-why"></p>
        </div>
        <div class="stats">
          <div>events<b id="d-ev"></b></div>
          <div>arrivals<b id="d-arr"></b></div>
          <div>stations<b id="d-sta"></b></div>
        </div>
      </div>
      <div class="bar">
        <div class="legend">
          <span><i class="c" id="d-legend-dot"></i>events in the windows, sized by magnitude</span>
          <span><i class="tri"></i>station with waveforms</span>
          <span><i class="tri hollow"></i>reference picks only</span>
          <span><i class="ring"></i>hold-out radius</span>
        </div>
        <span class="mono" id="d-windows" style="color:var(--ink-3)"></span>
      </div>
      <div id="detail"></div>
      <dl class="foot">
        <dt>Picks</dt><dd id="d-picks"></dd>
        <dt>Tests</dt><dd class="tags" id="d-tags"></dd>
        <dt id="d-gap-dt">Gap</dt><dd class="gap" id="d-gap"></dd>
        <dt>Papers</dt><dd class="refs" id="d-refs"></dd>
      </dl>
    </div>
  </section>

  <footer>
    Events are the stage-1 catalogue inside the scoring windows (the largest {MAX_EVENTS} where there are more); stations are those with fetched waveforms or with reference picks, up to {MAX_STATIONS}. Coastlines are Natural Earth 1:50m, coarse at this scale.
    Papers and data sets are listed in <code>data/heldout_testset/references.csv</code>, every DOI checked against Crossref or DataCite by <code>scripts/heldout_references.py</code>.
    Built from <code>data/heldout_testset/</code> by <code>scripts/make_heldout_dashboard.py</code>; rebuild a sequence with <code>python scripts/build_heldout_testset.py --sequence KEY all</code>.
  </footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
<script>
(function () {{
  const SEQ = {data_json};
  const BY = Object.fromEntries(SEQ.map((s) => [s.key, s]));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const REGS = ["msas", "vt", "swarm"];
  const font = () => ({{ family: "IBM Plex Sans, sans-serif", color: css("--ink") }});
  const hover = () => ({{ bgcolor: css("--hover-bg"), bordercolor: css("--rule"), font: {{ family: "IBM Plex Sans, sans-serif", color: css("--ink"), size: 13 }} }});
  const geoBase = (extra) => Object.assign({{
    showland: true, landcolor: css("--land"), showocean: true, oceancolor: css("--ocean"),
    showcoastlines: true, coastlinecolor: css("--coast"), coastlinewidth: 0.8, showframe: false, bgcolor: "rgba(0,0,0,0)",
    showlakes: false, showcountries: true, countrycolor: css("--coast"), countrywidth: 0.5,
  }}, extra);

  // ── overview ──
  const ov = document.getElementById("overview");
  function drawOverview() {{
    const traces = REGS.map((r) => {{
      const rows = SEQ.filter((s) => s.regime === r);
      return {{ type: "scattergeo", mode: "markers", name: rows[0].regime_name,
        lon: rows.map((s) => s.lon), lat: rows.map((s) => s.lat), text: rows.map((s) => s.label), customdata: rows.map((s) => s.key),
        hovertemplate: "<b>%{{text}}</b><extra></extra>",
        marker: {{ color: css("--" + r), size: 11, line: {{ color: css("--surface"), width: 1.5 }} }} }};
    }});
    Plotly.react(ov, traces, {{ margin: {{ l: 0, r: 0, t: 0, b: 0 }}, paper_bgcolor: "rgba(0,0,0,0)", showlegend: false, font: font(), hoverlabel: hover(),
      geo: geoBase({{ projection: {{ type: "natural earth" }}, lonaxis: {{ range: [-180, 180], showgrid: true, gridcolor: css("--grid"), dtick: 30 }},
                      lataxis: {{ range: [-60, 80], showgrid: true, gridcolor: css("--grid"), dtick: 30 }}, showcountries: false, resolution: 110 }}) }},
      {{ displayModeBar: false, responsive: true }});
  }}
  ov.on && drawOverview();
  ov.on("plotly_click", (ev) => select(ev.points[0].customdata, true));

  // ── detail ──
  const det = document.getElementById("detail");
  let current = null;
  function magSize(m) {{ if (m == null) return 5; return Math.max(4, Math.min(18, 4 + 2.4 * (m - 1))); }}
  function drawDetail(s) {{
    const col = css("--" + s.regime), ink = css("--ink"), ink3 = css("--ink-3");
    const pad = s.radius + 0.35, cosl = Math.cos(s.lat * Math.PI / 180);
    const lonR = [s.lon - pad / cosl, s.lon + pad / cosl], latR = [s.lat - pad, s.lat + pad];
    const wf = s.stations.filter((x) => x.wf), po = s.stations.filter((x) => !x.wf);
    const staTrace = (rows, filled) => ({{ type: "scattergeo", mode: "markers+text", name: filled ? "station with waveforms" : "reference picks only",
      lon: rows.map((x) => x.lon), lat: rows.map((x) => x.lat), text: rows.map((x) => x.id), textposition: "top center",
      textfont: {{ family: "IBM Plex Mono, monospace", size: 10, color: filled ? ink : ink3 }},
      customdata: rows.map((x) => [x.km ?? "?", x.band + (x.rate ? "@" + x.rate + " Hz" : ""), x.P, x.S, x.wf ? "waveforms fetched" : "picks only"]),
      hovertemplate: "<b>%{{text}}</b><br>%{{customdata[0]}} km · %{{customdata[1]}}<br>%{{customdata[2]}} P · %{{customdata[3]}} S reference picks<br>%{{customdata[4]}}<extra></extra>",
      marker: {{ symbol: "triangle-up", size: 11, color: filled ? ink : "rgba(0,0,0,0)", line: {{ color: filled ? css("--surface") : ink3, width: filled ? 1.5 : 1.8 }} }} }});
    const traces = [
      {{ type: "scattergeo", mode: "lines", name: "hold-out radius", lon: s.circle.lon, lat: s.circle.lat, hoverinfo: "skip",
        line: {{ color: ink3, width: 1, dash: "dot" }} }},
      {{ type: "scattergeo", mode: "markers", name: "events", lon: s.events.map((e) => e.lon), lat: s.events.map((e) => e.lat),
        customdata: s.events.map((e) => [e.t, e.mag ?? "?", e.dep ?? "?"]),
        hovertemplate: "%{{customdata[0]}} UTC<br>M %{{customdata[1]}} · %{{customdata[2]}} km<extra></extra>",
        marker: {{ color: col, size: s.events.map((e) => magSize(e.mag)), opacity: 0.8, line: {{ color: css("--surface"), width: 0.8 }} }} }},
      staTrace(po, false), staTrace(wf, true),
    ];
    Plotly.react(det, traces, {{ margin: {{ l: 0, r: 0, t: 0, b: 0 }}, paper_bgcolor: "rgba(0,0,0,0)", showlegend: false, font: font(), hoverlabel: hover(),
      geo: geoBase({{ projection: {{ type: "mercator" }}, resolution: 50,
        lonaxis: {{ range: lonR, showgrid: true, gridcolor: css("--grid"), dtick: 0.5 }}, lataxis: {{ range: latR, showgrid: true, gridcolor: css("--grid"), dtick: 0.5 }} }}) }},
      {{ displayModeBar: false, responsive: true }});
  }}
  function fill(s) {{
    document.getElementById("d-label").textContent = s.label;
    document.getElementById("d-dot").style.background = css("--" + s.regime);
    document.getElementById("d-legend-dot").style.background = css("--" + s.regime);
    document.getElementById("d-meta").textContent = `${{s.regime_name}} · ${{s.when}} · ${{s.lat.toFixed(2)}}, ${{s.lon.toFixed(2)}} · radius ${{s.radius}}° · ${{s.suite}}`;
    document.getElementById("d-why").textContent = s.why;
    document.getElementById("d-ev").textContent = s.n_events_windows.toLocaleString();
    document.getElementById("d-arr").textContent = s.arrivals.toLocaleString();
    document.getElementById("d-sta").textContent = `${{s.stations.filter((x) => x.wf).length}} / ${{s.stations.length}}`;
    document.getElementById("d-windows").textContent = s.windows.map((w) => `${{w.t0}} → ${{w.t1.slice(11)}} UTC`).join("   ·   ");
    document.getElementById("d-picks").textContent = s.source;
    document.getElementById("d-tags").innerHTML = s.tags.map((t) => `<span class="tag">${{t}}</span>`).join("");
    const gap = document.getElementById("d-gap"), gdt = document.getElementById("d-gap-dt");
    gap.textContent = s.gap || ""; gap.hidden = gdt.hidden = !s.gap;
    document.getElementById("d-refs").innerHTML = (s.refs || []).map((r) => r.doi
      ? `<div><span class="k">${{r.kind}}</span><a href="https://doi.org/${{r.doi}}" target="_blank" rel="noopener">${{r.citation}}</a><span class="doi">${{r.doi}}</span></div>`
      : `<div><span class="k">${{r.kind}}</span>${{r.citation}}<span class="doi">no DOI found</span></div>`).join("");
  }}
  function select(key, scroll) {{
    const s = BY[key]; if (!s) return; current = key;
    document.querySelectorAll(".nav").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === key)));
    fill(s); drawDetail(s); history.replaceState(null, "", "#" + key);
    if (scroll) document.querySelector(".panel").scrollIntoView({{ behavior: "smooth", block: "start" }});
  }}
  document.querySelectorAll(".nav").forEach((b) => b.addEventListener("click", () => select(b.dataset.key, false)));
  const first = (location.hash || "").slice(1);
  select(BY[first] ? first : SEQ[0].key, false);
  const redraw = () => {{ drawOverview(); if (current) drawDetail(BY[current]); }};
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});
}})();
</script>
'''
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB); {totals}")


if __name__ == "__main__":
    main()
