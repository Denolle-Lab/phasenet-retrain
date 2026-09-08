#!/usr/bin/env python3
"""
heldout_sequences.py

Definition of the permanently held-out external test sequences and the
whole-year hold-out, plus the pure-pandas helpers that
scripts/build_training_dataset.py, scripts/audit_heldout_sequences.py and
tests/test_heldout_sequences.py share. This module deliberately imports
nothing from SeisBench so it can be tested on a laptop without the cache.

Why this exists (docs/2026-09-07_generalization_audit_prompt.md, task 1):
the QuakeScope notebooks score every candidate weight on Kaikoura 2016,
Norcia 2016, Thessaly 2021, Ridgecrest 2019 and Monroe 2019 against
analyst picks. Those sequences are held out for the parent `jma_wc` by
construction (Japan only) but not for any fine-tune drawn from INSTANCE,
STEAD, CREW, SCEDC, CEED, ROSS2018GPD or PNW. For the sequences to be
test assets, nothing trained from now on may contain them.

Windows are great-circle radii (degrees of arc) around the mainshock
epicentre, the same convention as the QuakeScope notebooks' FDSN
`maxradius`. Each sequence has a mainshock window (the hours the notebooks
score) and a wider sequence span; a trace is excluded if its source origin
falls in either.

Whole-year hold-out: no 2016 and no 2021 origin in any training manifest,
so a future sequence from those years is clean too.

2026-09-08 extension (docs/2026-09-08_heldout_test_cases.md): three test
regimes, each with its own hold-outs.
  * mainshock-aftershock sequences: time-bounded windows as above
    (Kahramanmaras 2023, Noto 2024, Hualien 2024, Petrinja 2020-21,
    Samos 2020, Adriatic 2022);
  * volcano-tectonic sequences and fluid-driven swarms: PLACE hold-outs
    with no time bound (start/end None), because the same volcano or
    swarm zone recurs across years and appears in the SeisBench sets
    (Etna and Campi Flegrei in INSTANCE, for instance). A trace is
    excluded if its source lies inside the radius at any time.
Every window carries a `regime` and a `tier`: tier 1 places were never in
any training set and count for the generalisation claim; tier 2 are
known places at new times and are listed in the catalogue but not
enforced here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUSION_DIR = REPO_ROOT / "data" / "exclusions"
EXCLUSION_CSV = EXCLUSION_DIR / "heldout_sequences.csv"
COUNTS_CSV = EXCLUSION_DIR / "heldout_sequence_counts.csv"
WINDOWS_CSV = EXCLUSION_DIR / "heldout_windows.csv"

# Metadata column names (SeisBench convention; scripts/event_keys.py uses the
# same three as its fallback fingerprint).
TIME_COL = "source_origin_time"
LAT_COL = "source_latitude_deg"
LON_COL = "source_longitude_deg"

HOLDOUT_YEARS = frozenset({2016, 2021})

# name, lat, lon, radius (deg of arc), start (UTC), end (UTC, inclusive);
# start/end None = place hold-out at all times. regime: msas | vt | swarm |
# mixed; tier: 1 (never in any training set) | 2 (known place, new time).
WINDOWS = [
    dict(name="norcia_2016_mainshock", lat=42.83, lon=13.11, radius_deg=1.0,
         start="2016-10-30T06:50:00Z", end="2016-10-30T08:50:00Z"),
    dict(name="norcia_2016_sequence", lat=42.83, lon=13.11, radius_deg=1.0,
         start="2016-08-24T00:00:00Z", end="2017-01-31T23:59:59Z"),
    dict(name="kaikoura_2016_mainshock", lat=-42.69, lon=173.02, radius_deg=2.0,
         start="2016-11-13T11:13:00Z", end="2016-11-13T14:13:00Z"),
    dict(name="kaikoura_2016_sequence", lat=-42.69, lon=173.02, radius_deg=2.0,
         start="2016-11-13T00:00:00Z", end="2017-05-13T23:59:59Z"),
    dict(name="thessaly_2021_mainshock", lat=39.75, lon=22.20, radius_deg=1.0,
         start="2021-03-03T10:26:00Z", end="2021-03-03T13:26:00Z"),
    dict(name="thessaly_2021_sequence", lat=39.75, lon=22.20, radius_deg=1.0,
         start="2021-02-28T00:00:00Z", end="2021-05-31T23:59:59Z"),
    dict(name="ridgecrest_2019", lat=35.77, lon=-117.60, radius_deg=1.0,
         start="2019-07-06T00:00:00Z", end="2019-08-06T23:59:59Z"),
    dict(name="monroe_2019", lat=47.87, lon=-122.02, radius_deg=1.0,
         start="2019-07-12T00:00:00Z", end="2019-07-19T23:59:59Z"),
    # ── 2026-09-08: mainshock-aftershock sequences (time-bounded) ────────
    dict(name="kahramanmaras_2023", lat=37.4, lon=37.2, radius_deg=2.5,
         start="2023-02-06T00:00:00Z", end="2023-12-31T23:59:59Z", regime="msas", tier=1),
    dict(name="noto_2024", lat=37.5, lon=137.27, radius_deg=1.0,
         start="2024-01-01T00:00:00Z", end="2024-12-31T23:59:59Z", regime="msas", tier=1),
    dict(name="hualien_2024", lat=23.82, lon=121.56, radius_deg=1.0,
         start="2024-04-02T00:00:00Z", end="2024-12-31T23:59:59Z", regime="msas", tier=1),
    dict(name="petrinja_2020", lat=45.42, lon=16.26, radius_deg=0.7,
         start="2020-12-28T00:00:00Z", end="2021-06-30T23:59:59Z", regime="msas", tier=1),
    dict(name="samos_2020", lat=37.90, lon=26.79, radius_deg=1.0,
         start="2020-10-30T00:00:00Z", end="2021-02-28T23:59:59Z", regime="msas", tier=1),
    dict(name="adriatic_2022", lat=43.96, lon=13.32, radius_deg=0.7,
         start="2022-11-09T00:00:00Z", end="2023-03-31T23:59:59Z", regime="msas", tier=1),
    # ── 2026-09-08: fluid-driven swarm, time-bounded (Noto, before the 2024 mainshock)
    dict(name="noto_swarm_2020_2023", lat=37.5, lon=137.27, radius_deg=1.0,
         start="2020-12-01T00:00:00Z", end="2023-12-31T23:59:59Z", regime="swarm", tier=1),
    # ── 2026-09-08: volcano-tectonic places, all times ─────────────────────
    dict(name="reykjanes_peninsula", lat=63.9, lon=-22.3, radius_deg=0.7,
         start=None, end=None, regime="vt", tier=1),
    dict(name="la_palma", lat=28.61, lon=-17.87, radius_deg=0.5,
         start=None, end=None, regime="vt", tier=1),
    dict(name="santorini_amorgos", lat=36.6, lon=25.6, radius_deg=0.7,
         start=None, end=None, regime="vt", tier=1),
    dict(name="etna", lat=37.75, lon=15.0, radius_deg=0.5,
         start=None, end=None, regime="vt", tier=1),
    dict(name="mayotte", lat=-12.8, lon=45.5, radius_deg=1.0,
         start=None, end=None, regime="vt", tier=1),
    # ── 2026-09-08: fluid-driven swarm places, all times ───────────────────
    dict(name="campi_flegrei", lat=40.83, lon=14.14, radius_deg=0.15,  # 17 km: the caldera, not Vesuvius
         start=None, end=None, regime="swarm", tier=1),
    dict(name="west_bohemia", lat=50.24, lon=12.45, radius_deg=0.5,
         start=None, end=None, regime="swarm", tier=1),
    dict(name="maurienne", lat=45.30, lon=6.30, radius_deg=0.5,
         start=None, end=None, regime="swarm", tier=1),
    dict(name="corinth_thiva", lat=38.2, lon=23.1, radius_deg=0.7,
         start=None, end=None, regime="swarm", tier=1),
]
for _w in WINDOWS:
    _w.setdefault("regime", "msas"); _w.setdefault("tier", 1)
WINDOW_NAMES = [w["name"] for w in WINDOWS]
PLACE_NAMES = [w["name"] for w in WINDOWS if w["start"] is None]


def windows_frame() -> pd.DataFrame:
    return pd.DataFrame(WINDOWS)


def gc_distance_deg(lat1, lon1, lat2, lon2):
    """Great-circle distance in degrees of arc (haversine)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _to_utc(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, errors="coerce", utc=True)
    return t


def window_hits(origin_time, lat, lon) -> pd.DataFrame:
    """One boolean column per window. Rows with a missing time or location
    are False everywhere (they are reported as unverifiable elsewhere)."""
    t = _to_utc(pd.Series(origin_time))
    lat = pd.to_numeric(pd.Series(lat), errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(pd.Series(lon), errors="coerce").to_numpy(dtype=float)
    out = {}
    for w in WINDOWS:
        d = gc_distance_deg(w["lat"], w["lon"], lat, lon)
        in_space = np.isfinite(d) & (d <= w["radius_deg"])
        if w["start"] is None:            # place hold-out: any time, even unknown
            out[w["name"]] = in_space
            continue
        t0 = pd.Timestamp(w["start"])
        t1 = pd.Timestamp(w["end"])
        in_time = (t >= t0) & (t <= t1)
        out[w["name"]] = (in_time.to_numpy() & in_space)
    return pd.DataFrame(out, index=t.index)


def spatial_hits(lat, lon) -> pd.DataFrame:
    """Space-only version for tables that carry no origin time (an upper
    bound on membership, never a substitute for the spatiotemporal join)."""
    lat = pd.to_numeric(pd.Series(lat), errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(pd.Series(lon), errors="coerce").to_numpy(dtype=float)
    out = {}
    for w in WINDOWS:
        d = gc_distance_deg(w["lat"], w["lon"], lat, lon)
        out[w["name"]] = np.isfinite(d) & (d <= w["radius_deg"])
    return pd.DataFrame(out)


def flag_rows(df: pd.DataFrame, time_col=TIME_COL, lat_col=LAT_COL, lon_col=LON_COL) -> pd.DataFrame:
    """Per-row: `window` (semicolon-joined names of every window hit, '' if
    none), `origin_year` (Int64, <NA> if unknown), `year_holdout` (True when
    the origin year is in HOLDOUT_YEARS), `verifiable` (time and location
    both present)."""
    n = len(df)
    if not all(c in df.columns for c in (time_col, lat_col, lon_col)):
        return pd.DataFrame({
            "window": [""] * n,
            "origin_year": pd.array([pd.NA] * n, dtype="Int64"),
            "year_holdout": [False] * n,
            "verifiable": [False] * n,
        }, index=df.index)
    t = _to_utc(df[time_col])
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    hits = window_hits(t, lat, lon)
    hits.index = df.index
    names = np.array(WINDOW_NAMES, dtype=object)
    window = hits.apply(lambda r: ";".join(names[r.to_numpy(dtype=bool)]), axis=1) if n else pd.Series([], dtype=object)
    year = t.dt.year.astype("Int64")
    year_holdout = year.isin(list(HOLDOUT_YEARS)).fillna(False).astype(bool)
    verifiable = t.notna() & lat.notna() & lon.notna()
    return pd.DataFrame({
        "window": window.astype(str).values if n else [],
        "origin_year": year.values,
        "year_holdout": year_holdout.values,
        "verifiable": verifiable.values,
    }, index=df.index)


def holdout_year_mask(origin_time) -> pd.Series:
    """True where the origin year is 2016 or 2021 (drop these). Unknown
    years are False; callers decide what to do with them."""
    t = _to_utc(pd.Series(origin_time))
    return t.dt.year.isin(list(HOLDOUT_YEARS)).fillna(False).astype(bool)


def load_sequence_exclusions(path: Path = EXCLUSION_CSV, required: bool = True) -> dict:
    """{dataset: frozenset(trace_name)} from the committed exclusion list.

    Fails closed: if the list is missing and `required`, raise so that no
    manifest can be built without it. The list is produced on the lab
    server by scripts/audit_heldout_sequences.py.
    """
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Held-out sequence exclusion list not found at {path}. "
                "Run `python scripts/audit_heldout_sequences.py` on the server "
                "(needs the SeisBench cache) and commit its output before "
                "building any training manifest. Nothing trained from "
                "2026-09-07 on may contain the held-out sequences."
            )
        return {}
    df = pd.read_csv(path, usecols=["dataset", "trace_name"], dtype=str)
    return {ds: frozenset(g["trace_name"]) for ds, g in df.groupby("dataset")}


def check_manifest(manifest: pd.DataFrame, exclusions: dict,
                   time_col=TIME_COL, lat_col=LAT_COL, lon_col=LON_COL) -> dict:
    """Verify a manifest against the exclusion list and the year hold-out.

    Returns a dict with counts. `n_excluded_present` must be 0 and
    `n_year_holdout` must be 0 for the manifest to be usable. Rows with no
    origin time are counted in `n_year_unverifiable` (a manifest written
    by the patched build_training_dataset.py carries source_origin_time;
    older manifests need the metadata join done by
    scripts/audit_heldout_sequences.py --check-manifest).
    """
    ds_col = "dataset_name" if "dataset_name" in manifest.columns else "dataset"
    present = 0
    for ds, g in manifest.groupby(ds_col):
        bad = exclusions.get(ds)
        if bad:
            present += int(g["trace_name"].isin(bad).sum())
    flags = flag_rows(manifest, time_col, lat_col, lon_col)
    return dict(
        n_rows=int(len(manifest)),
        n_excluded_present=present,
        n_in_window=int((flags["window"] != "").sum()),
        n_year_holdout=int(flags["year_holdout"].sum()),
        n_year_unverifiable=int((~flags["verifiable"]).sum()),
    )
