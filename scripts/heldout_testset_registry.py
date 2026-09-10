#!/usr/bin/env python3
"""
heldout_testset_registry.py

Every held-out test sequence of docs/2026-09-08_heldout_test_cases.md as a
fully specified, reproducible query: where the reference picks come from,
how the scoring windows are chosen, where the waveforms come from. The
pipeline (scripts/build_heldout_testset.py) reads nothing else, and writes
this entry into each sequence's manifest.json, so a data directory can
always be traced back to the exact query that produced it.

Pick sources (`picks`, applied in order, results concatenated):
  fdsn_per_event  catalogue by region, then one includearrivals request per
                  event (INGV, GeoNet)          -- the QuakeScope path
  fdsn_region     region query with includearrivals, chunked by the hour and
                  split on HTTP 413 (NOA, ISC, franceseisme)
  usgs_phase_data USGS ComCat catalogue by region, then the phase-data
                  product's QuakeML per event (NEIC manual picks, M >= min_mag)
  jma_deck        JMA monthly arrival-time files dYYYYMM.zip (public through
                  2023-12), Japan Standard Time converted to UTC
  zenodo_pha      a hypoDD .pha phase file from a Zenodo record

Scoring windows (`windows`):
  mainshock       [T + start_s, T + start_s + minutes*60], the notebook rule
  busiest         the n_windows most populated non-overlapping windows of
                  `hours` inside `span`, by catalogue event count

Waveforms (`waveforms`): routes tried in order for stations whose network is
unknown; known networks go to NETWORK_ROUTE. `extra` stations are always
attempted (the USGS GSN stations). `channel_pref` is the band order.

Times are ISO-8601 UTC. Nothing here is fetched at import.
"""

from __future__ import annotations

import evaluation_policy as policy

# network code -> obspy client name ("eida" = RoutingClient("eida-routing"),
# "iris-fed" = RoutingClient("iris-federator"))
NETWORK_ROUTE = {
    "IV": "INGV", "MN": "INGV", "OX": "INGV", "ST": "INGV", "NI": "INGV",
    "HL": "NOA", "HA": "NOA", "HT": "NOA", "HP": "NOA", "HC": "NOA", "HI": "NOA", "CQ": "NOA",
    "NZ": "GEONET",
    "IU": "IRIS", "II": "IRIS", "TW": "IRIS", "US": "IRIS", "N4": "IRIS", "GS": "IRIS",
    "FR": "RESIF", "RA": "RESIF", "RD": "RESIF",
    "KO": "KOERI",
    "CR": "eida", "SL": "eida", "HU": "eida", "OE": "eida", "WB": "eida", "ES": "eida",
    "VI": "eida", "GE": "eida", "CH": "eida", "BW": "eida", "GR": "eida", "CZ": "eida",
    "MK": "eida", "SJ": "eida", "RO": "eida", "BS": "eida",
}
DEFAULT_ROUTES = ["eida", "iris-fed"]
CHANNEL_PREF = ["HH", "EH", "BH"]
N_STATIONS = 6
ORIGIN_LEAD_S = 180      # an origin this far before a window can still put arrivals in it
MATCH_TOL_S = 0.5        # duplicate collapse across events, as in the notebooks

GSN = {"BORG": "II.BORG", "MAJO": "IU.MAJO", "TATO": "IU.TATO", "ANTO": "IU.ANTO"}


def _seq(**kw):
    kw["suite"] = policy.role_for(kw["key"])
    kw.setdefault("channel_pref", CHANNEL_PREF)
    kw.setdefault("n_stations", N_STATIONS)
    kw.setdefault("waveform_routes", DEFAULT_ROUTES)
    kw.setdefault("extra_stations", [])
    kw.setdefault("notes", "")
    return kw


SEQUENCES = [
    # ── the three sequences already scored in QuakeScope, same parameters ──
    _seq(key="kaikoura_2016", label="Kaikoura 2016", regime="msas", tier=1,
         lat=-42.69, lon=173.02, radius=2.0, mainshock="2016-11-13T11:02:56Z", mag=7.8,
         picks=[dict(kind="fdsn_per_event", client="GEONET", min_mag=2.5)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["GEONET"],
         notes="as tutorials/phasenet_global_sequences.ipynb"),
    _seq(key="norcia_2016", label="Norcia 2016", regime="msas", tier=1,
         lat=42.83, lon=13.11, radius=0.8, mainshock="2016-10-30T06:40:18Z", mag=6.5,
         picks=[dict(kind="fdsn_per_event", client="INGV", min_mag=2.0)],
         windows=dict(kind="mainshock", start_s=600, minutes=120),
         waveform_routes=["INGV"], notes="as the notebook"),
    _seq(key="thessaly_2021", label="Thessaly 2021", regime="msas", tier=1,
         lat=39.75, lon=22.20, radius=1.0, mainshock="2021-03-03T10:16:08Z", mag=6.3,
         picks=[dict(kind="fdsn_region", client="NOA", min_mag=2.0)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["NOA"], notes="as the notebook"),

    # ── regime 1: mainshock-aftershock ──────────────────────────────────────
    _seq(key="kahramanmaras_2023", label="Kahramanmaras 2023", regime="msas", tier=1,
         lat=37.23, lon=37.01, radius=2.5, mainshock="2023-02-06T01:17:35Z", mag=7.8,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None),
                dict(kind="usgs_phase_data", min_mag=4.5)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["KOERI", "eida", "iris-fed"], extra_stations=[GSN["ANTO"]],
         notes="ISC is preliminary for 2023; AFAD manual readings to be added when bulk access is confirmed"),
    _seq(key="noto_2024", label="Noto 2024", regime="msas", tier=1,
         lat=37.50, lon=137.27, radius=1.0, mainshock="2024-01-01T07:10:09Z", mag=7.5,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None),
                dict(kind="usgs_phase_data", min_mag=4.5)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["IRIS"], extra_stations=[GSN["MAJO"]],
         notes="JMA deck files end 2023-12; the JMA unified picks need a Hi-net account (HinetPy get_arrivaltime). Hi-net waveforms likewise."),
    _seq(key="hualien_2024", label="Hualien 2024", regime="msas", tier=1,
         lat=23.82, lon=121.56, radius=1.0, mainshock="2024-04-02T23:58:11Z", mag=7.4,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None),
                dict(kind="usgs_phase_data", min_mag=4.5)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["IRIS"], extra_stations=[GSN["TATO"]],
         notes="ISC carries CWB picks with station codes; CWASN waveforms need a GDMS account, TW (BATS) is on EarthScope"),
    _seq(key="petrinja_2020", label="Petrinja 2020", regime="msas", tier=1,
         lat=45.42, lon=16.26, radius=0.7, mainshock="2020-12-29T11:19:54Z", mag=6.4,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["eida"],
         notes="ISC reviewed bulletin, Zagreb (ZAG) and neighbours; CR and SL on EIDA"),
    _seq(key="samos_2020", label="Samos 2020", regime="msas", tier=1,
         lat=37.90, lon=26.79, radius=1.0, mainshock="2020-10-30T11:51:27Z", mag=7.0,
         picks=[dict(kind="fdsn_region", client="NOA", min_mag=2.0)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["NOA", "KOERI"], extra_stations=[GSN["ANTO"]]),
    _seq(key="adriatic_2022", label="Adriatic 2022", regime="msas", tier=1,
         lat=43.96, lon=13.32, radius=0.7, mainshock="2022-11-09T06:07:25Z", mag=5.5,
         picks=[dict(kind="fdsn_per_event", client="INGV", min_mag=1.5)],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["INGV"], notes="offshore, land stations only"),

    # ── regime 2: volcano-tectonic ──────────────────────────────────────────
    _seq(key="reykjanes_2023", label="Reykjanes 2023 dike", regime="vt", tier=1,
         lat=63.87, lon=-22.38, radius=0.7, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None),
                dict(kind="usgs_phase_data", min_mag=4.5)],
         windows=dict(kind="busiest", span=("2023-11-10T12:00:00Z", "2023-11-11T12:00:00Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="ISC", min_mag=None)),
         waveform_routes=["eida", "IRIS"], extra_stations=[GSN["BORG"]],
         notes="IMO SIL manual picks on request; ISC preliminary for 2023-11; VI stations on EIDA"),
    _seq(key="fagradalsfjall_2021", label="Fagradalsfjall 2021", regime="vt", tier=1,
         lat=63.90, lon=-22.27, radius=0.7, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None)],
         windows=dict(kind="busiest", span=("2021-02-24T00:00:00Z", "2021-03-05T00:00:00Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="ISC", min_mag=None)),
         waveform_routes=["eida", "IRIS"], extra_stations=[GSN["BORG"]],
         notes="ISC reviewed; 2021 is a held-out year"),
    _seq(key="la_palma_2021", label="La Palma 2021", regime="vt", tier=1,
         lat=28.61, lon=-17.87, radius=0.5, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_region", client="ISC", min_mag=None)],
         windows=dict(kind="busiest", span=("2021-09-11T00:00:00Z", "2021-09-20T00:00:00Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="ISC", min_mag=None)),
         waveform_routes=["eida"], notes="IGN picks through ISC; ES stations on EIDA"),
    _seq(key="santorini_2025", label="Santorini-Amorgos 2025", regime="vt", tier=1,
         lat=36.60, lon=25.60, radius=0.7, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_region", client="NOA", min_mag=None)],
         windows=dict(kind="busiest", span=("2025-01-28T00:00:00Z", "2025-02-12T00:00:00Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="NOA", min_mag=None, arrivals=False)),
         waveform_routes=["NOA"]),
    _seq(key="etna_2022_2024", label="Etna 2022-2024", regime="vt", tier=1,
         lat=37.75, lon=15.00, radius=0.5, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_per_event", client="INGV", min_mag=None)],
         windows=dict(kind="busiest", span=("2022-01-01T00:00:00Z", "2024-12-31T23:59:59Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="INGV", min_mag=1.0, arrivals=False)),
         waveform_routes=["INGV"]),

    # ── regime 3: fluid-driven swarms ───────────────────────────────────────
    _seq(key="west_bohemia_2018", label="West Bohemia 2018", regime="swarm", tier=1,
         lat=50.24, lon=12.45, radius=0.5, mainshock=None, mag=None,
         picks=[dict(kind="zenodo_pha", record=5016845, file="catalog_2018swarm.pha",
                     quality_file="catalog_2018swarm_quality1.pha", stations_file="station_coordinates.txt",
                     network="")],      # bare codes: the resolver maps them to the inventory (NKC -> CZ.NKC); the
                                        # classic WEBNET stations are not on EIDA under WB, their waveforms are in the tarball
         windows=dict(kind="busiest", span=("2018-05-10T00:00:00Z", "2018-06-01T00:00:00Z"), n_windows=2, hours=3,
                      catalog=dict(kind="zenodo_pha")),
         waveform_routes=["eida"], notes="WEBNET manual picks; only NKC (CZ) and the newer WB.*D stations are on EIDA, "
                                          "the classic WEBNET stations' waveforms are in the Zenodo tarball (waveforms.tar.gz, 966 MB)"),
    _seq(key="maurienne_2017", label="Maurienne 2017-2019", regime="swarm", tier=1,
         lat=45.30, lon=6.30, radius=0.5, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_region", client="franceseisme", min_mag=None)],
         windows=dict(kind="busiest", span=("2017-08-01T00:00:00Z", "2019-03-31T23:59:59Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="franceseisme", min_mag=None, arrivals=False)),
         waveform_routes=["RESIF", "eida"]),
    _seq(key="noto_swarm_2023", label="Noto swarm 2023", regime="swarm", tier=1,
         lat=37.50, lon=137.27, radius=1.0, mainshock="2023-05-05T05:42:04Z", mag=6.5,
         picks=[dict(kind="jma_deck", months=["202305"])],
         windows=dict(kind="mainshock", start_s=600, minutes=180),
         waveform_routes=["IRIS"], extra_stations=[GSN["MAJO"]],
         notes="JMA M6.5 (Mw 6.2) Suzu event during the swarm; picks from the public JMA deck file; Hi-net waveforms need an account"),
    _seq(key="campi_flegrei_2023", label="Campi Flegrei 2023", regime="swarm", tier=1,
         lat=40.83, lon=14.14, radius=0.15, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_per_event", client="INGV", min_mag=None)],
         windows=dict(kind="busiest", span=("2023-09-26T00:00:00Z", "2023-10-05T00:00:00Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="INGV", min_mag=0.5, arrivals=False)),
         waveform_routes=["INGV"], notes="the M4.2 of 2023-09-27 and its days"),
    _seq(key="corinth_thiva_2020", label="Corinth-Thiva 2020-2021", regime="swarm", tier=1,
         lat=38.20, lon=23.10, radius=0.7, mainshock=None, mag=None,
         picks=[dict(kind="fdsn_region", client="NOA", min_mag=None)],
         windows=dict(kind="busiest", span=("2020-12-01T00:00:00Z", "2021-06-30T23:59:59Z"), n_windows=2, hours=3,
                      catalog=dict(kind="fdsn_region", client="NOA", min_mag=1.0, arrivals=False)),
         waveform_routes=["NOA"]),
]

BY_KEY = {s["key"]: s for s in SEQUENCES}

if set(BY_KEY) != set(policy.load_policy()["roles"]):
    raise ValueError("Registry and evaluation policy keys differ")
