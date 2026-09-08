#!/usr/bin/env python3
"""
make_heldout_dashboard.py

Renders docs/heldout_testset_dashboard.html: a map of the held-out test
sequences colour-coded by regime (mainshock-aftershock, volcano-tectonic,
fluid-driven swarm), with one line on why each was chosen, where its picks
come from, and what it tests. Numbers come from data/heldout_testset/index.csv
and the registry; the prose lives here. Plotly is loaded from cdnjs.

    python scripts/make_heldout_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_testset_registry as reg  # noqa: E402

OUT = REPO_ROOT / "docs" / "heldout_testset_dashboard.html"
INDEX = REPO_ROOT / "data" / "heldout_testset" / "index.csv"

REGIME = {"msas": "Mainshock–aftershock", "vt": "Volcano-tectonic", "swarm": "Fluid-driven swarm"}

# why chosen · what it tests (2–3 tags) · gap, if any
TEXT = {
    "kaikoura_2016": ("M7.8 across a dozen crustal faults on a sparse permanent network; most GeoNet picks sit 80–120 km out.",
                      ["regional S", "sparse network", "overlapping coda"], ""),
    "norcia_2016": ("Largest shock of Amatrice–Visso–Norcia; dense permanent plus temporary stations, so near-field aftershocks seconds apart.",
                    ["near-field density", "in-domain for instance"], ""),
    "thessaly_2021": ("Normal-faulting doublet; NOA picks S on stations 45–90 km out where every weight finds P and misses S.",
                      ["regional S ceiling", "held-out year 2021"], ""),
    "kahramanmaras_2023": ("Mw 7.8 and 7.6 doublet on a 350 km rupture; AFAD read 566k P and 461k S by hand, the decade's largest manual set.",
                           ["dense overlapping aftershocks", "many operators", "100 Hz KO"], "AFAD bulk readings still to add; ISC is preliminary"),
    "noto_2024": ("Mw 7.5 that ended a three-year swarm; JMA unified picks, after the parent's 2014–2021 training years.",
                  ["held out for jma_wc too", "dense sequence"], "Hi-net account needed for waveforms and picks"),
    "hualien_2024": ("Mw 7.4 with a conjugate fault system; CWA hand-picked thousands of aftershocks; SeisBench's CWA set ends 2021.",
                     ["near-field density", "collision–subduction"], "CWA GDMS account needed; only IU.TATO open"),
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


def main():
    idx = pd.read_csv(INDEX).set_index("key")
    rows = []
    for s in reg.SEQUENCES:
        k = s["key"]
        why, tags, gap = TEXT[k]
        i = idx.loc[k] if k in idx.index else None
        rows.append(dict(
            key=k, label=s["label"], regime=s["regime"], regime_name=REGIME[s["regime"]], suite=s["suite"],
            lat=s["lat"], lon=s["lon"], when=when(s), source=source_label(s), why=why, tags=tags, gap=gap,
            arrivals=int(i["picks_reference_ok"]) if i is not None else 0,
            stations=int(i["stations_fetched"]) if i is not None else 0,
            windows=int(i["windows"]) if i is not None else 0,
        ))
    df = pd.DataFrame(rows)
    totals = dict(n=len(df), arrivals=int(df.arrivals.sum()), stations=int(df.stations.sum()),
                  per_regime={r: int((df.regime == r).sum()) for r in REGIME})
    built = idx["built"].max()[:10] if "built" in idx else "2026-09-08"
    data_json = json.dumps(rows, ensure_ascii=False)

    def cards(regime):
        out = []
        for r in rows:
            if r["regime"] != regime:
                continue
            tags = "".join(f'<span class="tag">{t}</span>' for t in r["tags"])
            gap = f'<p class="gap">Gap: {r["gap"]}</p>' if r["gap"] else ""
            suite = '<span class="pill acc">acceptance</span>' if r["suite"] == "acceptance" else '<span class="pill dev">development</span>'
            out.append(f'''
      <article class="card r-{regime}" id="card-{r["key"]}" tabindex="0">
        <header><h3>{r["label"]}</h3>{suite}</header>
        <p class="meta mono">{r["when"]} · {r["lat"]:.2f}, {r["lon"]:.2f}</p>
        <p class="why">{r["why"]}</p>
        <dl>
          <dt>Picks</dt><dd>{r["source"]} · <span class="mono">{r["arrivals"]:,}</span> reference arrivals · <span class="mono">{r["stations"]}</span> stations · <span class="mono">{r["windows"]}</span> window{"s" if r["windows"] != 1 else ""}</dd>
          <dt>Tests</dt><dd class="tags">{tags}</dd>
        </dl>
        {gap}
      </article>''')
        return "\n".join(out)

    html = f'''<title>Held-Out Sequence Atlas</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --ground: #f6f7f9; --surface: #ffffff; --ink: #14181f; --ink-2: #4b5563; --ink-3: #7a8391; --rule: #dfe3e8;
  --land: #e7eaee; --ocean: #f6f7f9; --coast: #c5ccd6; --grid: #e3e7ec;
  --msas: #2a78d6; --vt: #eb6834; --swarm: #1baf7a;
  --acc-bg: #e8f1fb; --acc-ink: #1b5fae; --dev-bg: #eef0f3; --dev-ink: #4b5563;
  --focus: #2a78d6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground: #0f1319; --surface: #171c24; --ink: #f2f4f7; --ink-2: #b8c0cc; --ink-3: #7f8a99; --rule: #2a323d;
    --land: #222a35; --ocean: #0f1319; --coast: #3a4553; --grid: #232b36;
    --msas: #3987e5; --vt: #d95926; --swarm: #199e70;
    --acc-bg: #1b2a3f; --acc-ink: #8db8f0; --dev-bg: #1f2630; --dev-ink: #b8c0cc;
    --focus: #3987e5;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground: #0f1319; --surface: #171c24; --ink: #f2f4f7; --ink-2: #b8c0cc; --ink-3: #7f8a99; --rule: #2a323d;
  --land: #222a35; --ocean: #0f1319; --coast: #3a4553; --grid: #232b36;
  --msas: #3987e5; --vt: #d95926; --swarm: #199e70;
  --acc-bg: #1b2a3f; --acc-ink: #8db8f0; --dev-bg: #1f2630; --dev-ink: #b8c0cc;
  --focus: #3987e5;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--ground); color: var(--ink); font: 15px/1.5 "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; }}
.mono {{ font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 24px 48px; }}
header.top {{ display: grid; grid-template-columns: 1fr auto; gap: 16px 32px; align-items: end; border-bottom: 1px solid var(--rule); padding-bottom: 18px; }}
.eyebrow {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3); margin: 0 0 6px; }}
h1 {{ font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif; font-weight: 600; font-size: 34px; line-height: 1.1; margin: 0 0 8px; text-wrap: balance; letter-spacing: -.01em; }}
.lede {{ margin: 0; max-width: 62ch; color: var(--ink-2); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
.chip {{ display: inline-flex; align-items: baseline; gap: 6px; padding: 6px 10px; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); font-size: 13px; color: var(--ink-2); }}
.chip b {{ font-family: "IBM Plex Mono", monospace; font-weight: 500; color: var(--ink); font-size: 14px; }}
.chip .dot {{ width: 9px; height: 9px; border-radius: 50%; align-self: center; }}
.dot.msas {{ background: var(--msas); }} .dot.vt {{ background: var(--vt); }} .dot.swarm {{ background: var(--swarm); }}
section.map {{ margin-top: 18px; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }}
.mapbar {{ display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; padding: 10px 14px; border-bottom: 1px solid var(--rule); font-size: 13px; color: var(--ink-2); }}
.views {{ display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }}
.views button {{ appearance: none; border: 0; background: transparent; color: var(--ink-2); font: inherit; padding: 5px 12px; cursor: pointer; }}
.views button[aria-pressed="true"] {{ background: var(--ink); color: var(--surface); }}
.views button:focus-visible, .card:focus-visible {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
#map {{ width: 100%; height: 520px; }}
.legend {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
section.regimes {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; margin-top: 26px; }}
@media (max-width: 960px) {{ section.regimes {{ grid-template-columns: 1fr; }} header.top {{ grid-template-columns: 1fr; }} .chips {{ justify-content: flex-start; }} }}
.col h2 {{ font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 19px; margin: 0 0 4px; display: flex; align-items: center; gap: 8px; }}
.col h2 i {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.col .sub {{ margin: 0 0 12px; font-size: 13px; color: var(--ink-3); }}
.cards {{ display: flex; flex-direction: column; gap: 10px; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-left-width: 4px; border-radius: 6px; padding: 12px 14px 10px; }}
.card.r-msas {{ border-left-color: var(--msas); }} .card.r-vt {{ border-left-color: var(--vt); }} .card.r-swarm {{ border-left-color: var(--swarm); }}
.card.hi {{ box-shadow: 0 0 0 2px var(--focus); }}
.card header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }}
.card h3 {{ margin: 0; font-size: 16px; font-weight: 600; }}
.pill {{ font-size: 11px; letter-spacing: .04em; text-transform: uppercase; padding: 2px 7px; border-radius: 4px; white-space: nowrap; }}
.pill.acc {{ background: var(--acc-bg); color: var(--acc-ink); }} .pill.dev {{ background: var(--dev-bg); color: var(--dev-ink); }}
.meta {{ margin: 2px 0 8px; font-size: 12px; color: var(--ink-3); }}
.why {{ margin: 0 0 8px; color: var(--ink); }}
dl {{ margin: 0; display: grid; grid-template-columns: 52px 1fr; gap: 4px 8px; font-size: 13px; }}
dt {{ color: var(--ink-3); font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; padding-top: 3px; }}
dd {{ margin: 0; color: var(--ink-2); }}
.tags {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.tag {{ font-size: 12px; padding: 2px 8px; border: 1px solid var(--rule); border-radius: 999px; color: var(--ink-2); background: var(--ground); }}
.gap {{ margin: 8px 0 0; font-size: 12px; color: var(--ink-3); border-top: 1px dashed var(--rule); padding-top: 6px; }}
footer {{ margin-top: 30px; padding-top: 14px; border-top: 1px solid var(--rule); font-size: 12.5px; color: var(--ink-3); max-width: 80ch; }}
footer code {{ font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-2); }}
</style>
<div class="wrap">
  <header class="top">
    <div>
      <p class="eyebrow">phasenet-retrain · held-out test set · built {built}</p>
      <h1>Held-Out Sequence Atlas</h1>
      <p class="lede">Nineteen sequences a picker never trains on, in the three regimes a global campaign has to get right. Every reference pick is an analyst's, stored in the QuakeScope schema; every query is pinned in the registry.</p>
    </div>
    <div class="chips">
      <span class="chip"><b>{totals["n"]}</b> sequences</span>
      <span class="chip"><i class="dot msas"></i><b>{totals["per_regime"]["msas"]}</b> mainshock–aftershock</span>
      <span class="chip"><i class="dot vt"></i><b>{totals["per_regime"]["vt"]}</b> volcano-tectonic</span>
      <span class="chip"><i class="dot swarm"></i><b>{totals["per_regime"]["swarm"]}</b> fluid-driven swarm</span>
      <span class="chip"><b>{totals["arrivals"]:,}</b> reference arrivals</span>
      <span class="chip"><b>{totals["stations"]}</b> stations with waveforms</span>
    </div>
  </header>

  <section class="map" aria-label="Map of held-out sequences">
    <div class="mapbar">
      <div class="legend">
        <span><i style="background:var(--msas)"></i>Mainshock–aftershock</span>
        <span><i style="background:var(--vt)"></i>Volcano-tectonic</span>
        <span><i style="background:var(--swarm)"></i>Fluid-driven swarm</span>
        <span class="mono" style="color:var(--ink-3)">marker area ∝ reference arrivals · click a marker to jump to its card</span>
      </div>
      <div class="views" role="group" aria-label="Map view">
        <button type="button" data-view="world" aria-pressed="true">World</button>
        <button type="button" data-view="europe" aria-pressed="false">Europe</button>
        <button type="button" data-view="japan" aria-pressed="false">Japan · Taiwan</button>
      </div>
    </div>
    <div id="map"></div>
  </section>

  <section class="regimes">
    <div class="col">
      <h2><i style="background:var(--msas)"></i>Mainshock–aftershock</h2>
      <p class="sub">The first hours after an M6+: events seconds apart, coda everywhere. Scored from 10 min after the mainshock for 2–3 h.</p>
      <div class="cards">{cards("msas")}</div>
    </div>
    <div class="col">
      <h2><i style="background:var(--vt)"></i>Volcano-tectonic</h2>
      <p class="sub">Dikes and pre-eruptive unrest: emergent onsets, weak S, tremor underneath. Held out as places at all times, not as dates.</p>
      <div class="cards">{cards("vt")}</div>
    </div>
    <div class="col">
      <h2><i style="background:var(--swarm)"></i>Fluid-driven swarm</h2>
      <p class="sub">Magmatic and hydrothermal swarms that migrate for weeks to months. The two busiest 3 h windows of each are scored.</p>
      <div class="cards">{cards("swarm")}</div>
    </div>
  </section>

  <footer>
    Built from <code>scripts/heldout_testset_registry.py</code> and <code>data/heldout_testset/index.csv</code> by <code>scripts/make_heldout_dashboard.py</code>.
    Reference arrivals are picks flagged <code>reference_ok</code>: manual, or without an evaluation mode when relayed by the ISC bulletin.
    Acceptance sequences are never read during development; development ones decide between runs. Ocean-bottom observations are out of scope this round.
    Rebuild any sequence with <code>python scripts/build_heldout_testset.py --sequence KEY all</code>; verify with <code>--verify KEY</code>.
  </footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
<script>
(function () {{
  const SEQ = {data_json};
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const REG = {{ msas: "Mainshock–aftershock", vt: "Volcano-tectonic", swarm: "Fluid-driven swarm" }};
  const VIEWS = {{
    world:  {{ lon: [-180, 180], lat: [-60, 80], text: false }},
    europe: {{ lon: [-26, 40],  lat: [27, 67],  text: true }},
    japan:  {{ lon: [118, 146], lat: [20, 46],  text: true }},
  }};
  let view = "world";
  function traces() {{
    return ["msas", "vt", "swarm"].map((r) => {{
      const rows = SEQ.filter((s) => s.regime === r);
      return {{
        type: "scattergeo", name: REG[r], mode: VIEWS[view].text ? "markers+text" : "markers",
        lon: rows.map((s) => s.lon), lat: rows.map((s) => s.lat),
        text: rows.map((s) => s.label), textposition: "top center",
        textfont: {{ family: "IBM Plex Sans, sans-serif", size: 11, color: css("--ink-2") }},
        customdata: rows.map((s) => [s.key, s.when, s.source, s.arrivals, s.stations, s.windows, REG[r]]),
        hovertemplate: "<b>%{{text}}</b><br>%{{customdata[6]}} · %{{customdata[1]}}<br>picks: %{{customdata[2]}}<br>%{{customdata[3]:,}} reference arrivals · %{{customdata[4]}} stations · %{{customdata[5]}} window(s)<extra></extra>",
        marker: {{ color: css("--" + r), size: rows.map((s) => Math.max(9, 3 * Math.sqrt(Math.max(s.arrivals, 1) / 60))),
                  line: {{ color: css("--surface"), width: 2 }}, opacity: 0.92 }},
      }};
    }});
  }}
  function layout() {{
    const v = VIEWS[view];
    return {{
      margin: {{ l: 0, r: 0, t: 0, b: 0 }}, paper_bgcolor: "rgba(0,0,0,0)", showlegend: false,
      font: {{ family: "IBM Plex Sans, sans-serif", color: css("--ink") }},
      hoverlabel: {{ bgcolor: css("--surface"), bordercolor: css("--rule"), font: {{ family: "IBM Plex Sans, sans-serif", color: css("--ink"), size: 13 }} }},
      geo: {{
        projection: {{ type: view === "world" ? "natural earth" : "mercator" }},
        lonaxis: {{ range: v.lon, showgrid: true, gridcolor: css("--grid"), dtick: view === "world" ? 30 : 5 }},
        lataxis: {{ range: v.lat, showgrid: true, gridcolor: css("--grid"), dtick: view === "world" ? 30 : 5 }},
        showland: true, landcolor: css("--land"), showocean: true, oceancolor: css("--ocean"),
        showcoastlines: true, coastlinecolor: css("--coast"), coastlinewidth: 0.8,
        showcountries: view !== "world", countrycolor: css("--coast"), countrywidth: 0.5,
        showlakes: false, showframe: false, bgcolor: "rgba(0,0,0,0)", resolution: view === "world" ? 110 : 50,
      }},
    }};
  }}
  const el = document.getElementById("map");
  function draw() {{ Plotly.react(el, traces(), layout(), {{ displayModeBar: false, responsive: true }}); }}
  draw();
  el.on("plotly_click", (ev) => {{
    const key = ev.points[0].customdata[0]; const card = document.getElementById("card-" + key);
    if (!card) return;
    document.querySelectorAll(".card.hi").forEach((c) => c.classList.remove("hi"));
    card.classList.add("hi"); card.scrollIntoView({{ behavior: "smooth", block: "center" }}); card.focus({{ preventScroll: true }});
  }});
  document.querySelectorAll(".views button").forEach((b) => b.addEventListener("click", () => {{
    view = b.dataset.view; document.querySelectorAll(".views button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw();
  }}));
  const mq = window.matchMedia("(prefers-color-scheme: dark)"); mq.addEventListener("change", draw);
  new MutationObserver(draw).observe(document.documentElement, {{ attributes: true, attributeFilter: ["data-theme"] }});
}})();
</script>
'''
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB); {totals}")


if __name__ == "__main__":
    main()
