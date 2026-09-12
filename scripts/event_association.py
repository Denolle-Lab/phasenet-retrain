#!/usr/bin/env python3
"""
event_association.py

Event association and catalogue-relative recovery engine (issue #36,
checkpoint 36A). Consumes the 35A pick store written by
scripts/heldout_testset_score.py (columns pick_id, model_id, threshold,
station, phase, time, score, matched_event, access_id, key, ...) together
with a station table and a reference catalogue, and produces events, pick
assignments, one-to-one event matches, split/merge diagnostics, recovery
tables with coverage flags, and a paired block bootstrap.

  * AssociatorConfig freezes every associator parameter (velocity model,
    tolerances, minimum picks, region) in configs/association/<region>.json.
    Its sha256 over the canonical content is written as config_sha256 into
    every events, assignments and matches row and into run.json.
  * PyOctoAssociator adapts the pick store and the station table to PyOcto
    (Münchmeyer 2024, Seismica 3(1)). pyocto is imported lazily; a missing
    install raises ImportError with the remedy. The PyOcto call itself was
    not executed in this checkpoint (pyocto is not installed here).
  * SyntheticAssociator is a deterministic back-projection over a coarse
    x/y/depth grid with homogeneous velocities. It is a test double for the
    pipeline, not a scientific associator, and must not be used for
    reported baselines.
  * match_events lifts continuous_scoring.match_picks to events: maximum
    cardinality first, minimum normalised residual second, under explicit
    origin-time, epicentral and optional depth tolerances.
  * split_merge_diagnostics, recovery_tables and paired_block_bootstrap
    produce what the 36B acceptance asks for, with a claim_supported flag on
    every table row tied to window coverage.

numpy, pandas, scipy only at module level; no torch, seisbench or pyocto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import continuous_scoring as cs  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "association"
HELDOUT_ROOT = REPO_ROOT / "data" / "heldout_testset"
EARTH_RADIUS_KM = 6371.0
KM_PER_DEG = 111.19
REGIMES = ("msas", "vt", "swarm")

# Matching tolerances: proposals (docs/2026-09-11_36a_event_association.md), not tuned values.
MATCH_TOL_TIME_S = 5.0
MATCH_TOL_KM = 30.0
MATCH_TOL_DEPTH_KM = 50.0
HOUR_BINS = (0.0, 1.0, 6.0, 24.0, 48.0, np.inf)
MAGNITUDE_BINS = (-np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf)
MIN_COVERAGE = 0.95          # a time-binned claim needs this fraction of the bin inside the windows
SPLIT_OVERLAP = 0.5          # a predicted event "belongs" to a reference event at this pick share

EVENT_COLUMNS = ["event_idx", "time", "lat", "lon", "depth_km", "n_picks", "n_p", "n_s", "n_stations", "n_p_and_s",
                 "misfit_s", "associator", "config_sha256"]
ASSIGNMENT_COLUMNS = ["event_idx", "pick_id", "station", "phase", "time", "score", "residual"]
PAIR_COLUMNS = ["event", "event_idx", "dt_s", "dist_km", "ddepth_km", "cost"]
SPLIT_COLUMNS = ["event", "n_predicted", "predicted", "rule"]
MERGE_COLUMNS = ["event_idx", "n_reference", "events", "reference_events_in_picks", "rule"]
PICK_REQUIRED = ("pick_id", "station", "phase", "time")
STATION_REQUIRED = ("station", "lat", "lon", "elev_m")


# ── configuration ────────────────────────────────────────────────────────────

HASH_EXCLUDED = ("notes", "versioned")
VERSIONED_NOTE = ("Frozen associator parameters. sha256 is computed over the canonical JSON of every field except "
                  "notes and versioned; it is recorded as config_sha256 in every events, assignments and matches row "
                  "and in run.json of each run. To change a value: edit, bump version, and rewrite the file with "
                  "AssociatorConfig.save (or `event_association.py --rehash <file>`) so the stored hash matches; "
                  "AssociatorConfig.load refuses a file whose stored hash disagrees with its content.")


@dataclass
class AssociatorConfig:
    """Every associator parameter, JSON-serialisable, with a content hash.

    Names follow PyOcto where one exists (Münchmeyer 2024): time_tolerance_s
    is the velocity-model `tolerance` and `pick_match_tolerance`,
    spatial_tolerance_km is `min_node_size`, location_tolerance_km is
    `min_node_size_location`, association_cutoff_km is
    `association_cutoff_distance`, time_before_s is `time_before`, min_picks
    is `n_picks`, min_p_picks/min_s_picks are `n_p_picks`/`n_s_picks`,
    n_p_and_s_picks is `n_p_and_s_picks` (stations with both phases) and is
    passed only when n_picks_p_and_s_policy == "require". The synthetic test
    double uses spatial_tolerance_km as its grid step and margin_km around
    the station extent as its search box.
    """
    region: str
    version: str = "1"
    velocity_model: str = "homogeneous"      # "homogeneous" (p_velocity, s_velocity) or a name for `layers`
    layers: list = field(default_factory=list)   # [{"depth_km", "vp", "vs"}], top of each layer, for VelocityModel1D
    p_velocity: float = 6.0                  # km/s; fallback when no layers
    s_velocity: float = 3.4
    time_tolerance_s: float = 1.5
    spatial_tolerance_km: float = 10.0
    location_tolerance_km: float = 2.0
    association_cutoff_km: float = 250.0
    depth_range_km: list = field(default_factory=lambda: [0.0, 60.0])
    margin_km: float = 50.0
    time_before_s: float = 300.0
    min_picks: int = 6
    min_p_picks: int = 3
    min_s_picks: int = 0
    n_p_and_s_picks: int = 2
    n_picks_p_and_s_policy: str = "require"  # "require" | "ignore"
    notes: str = ""
    versioned: str = VERSIONED_NOTE

    def __post_init__(self):
        self.region = str(self.region)
        self.version = str(self.version)
        self.velocity_model = str(self.velocity_model)
        self.layers = [dict(depth_km=float(l["depth_km"]), vp=float(l["vp"]), vs=float(l["vs"])) for l in self.layers]
        for name in ("p_velocity", "s_velocity", "time_tolerance_s", "spatial_tolerance_km", "location_tolerance_km",
                     "association_cutoff_km", "margin_km", "time_before_s"):
            setattr(self, name, float(getattr(self, name)))
        for name in ("min_picks", "min_p_picks", "min_s_picks", "n_p_and_s_picks"):
            setattr(self, name, int(getattr(self, name)))
        self.depth_range_km = [float(v) for v in self.depth_range_km]
        if len(self.depth_range_km) != 2 or self.depth_range_km[0] >= self.depth_range_km[1]:
            raise ValueError(f"depth_range_km must be [top, bottom] with top < bottom: {self.depth_range_km}")
        if self.p_velocity <= 0 or self.s_velocity <= 0 or self.s_velocity >= self.p_velocity:
            raise ValueError("velocities must satisfy 0 < s_velocity < p_velocity")
        for name in ("time_tolerance_s", "spatial_tolerance_km", "location_tolerance_km", "association_cutoff_km"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.margin_km < 0 or self.time_before_s < 0:
            raise ValueError("margin_km and time_before_s must be non-negative")
        if self.min_picks < 1 or min(self.min_p_picks, self.min_s_picks, self.n_p_and_s_picks) < 0:
            raise ValueError("pick minimums must be non-negative and min_picks >= 1")
        if self.n_picks_p_and_s_policy not in ("require", "ignore"):
            raise ValueError("n_picks_p_and_s_policy must be 'require' or 'ignore'")
        if self.velocity_model != "homogeneous" and not self.layers:
            raise ValueError("a named velocity model needs `layers`")
        for l in self.layers:
            if l["vp"] <= 0 or l["vs"] <= 0 or l["vs"] >= l["vp"]:
                raise ValueError(f"layer velocities must satisfy 0 < vs < vp: {l}")

    def canonical(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in HASH_EXCLUDED}

    def canonical_json(self) -> str:
        return json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sha256"] = self.sha256
        return d

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_dict(cls, d: dict, verify=True) -> "AssociatorConfig":
        d = dict(d)
        recorded = d.pop("sha256", None)
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown AssociatorConfig keys: {unknown}")
        cfg = cls(**d)
        if verify and recorded is not None and recorded != cfg.sha256:
            raise ValueError(f"Stored sha256 {recorded[:12]} does not match content hash {cfg.sha256[:12]}; "
                             "the file was edited without rewriting its hash (see the `versioned` note)")
        return cfg

    @classmethod
    def load(cls, path, verify=True) -> "AssociatorConfig":
        return cls.from_dict(json.loads(Path(path).read_text()), verify=verify)

    def replace(self, **changes) -> "AssociatorConfig":
        return AssociatorConfig(**{**asdict(self), **changes})


# Proposed defaults per regime; tune on development sequences only (36B), never on acceptance.
DEFAULT_CONFIGS = {
    "msas": dict(region="msas", p_velocity=6.0, s_velocity=3.4, time_tolerance_s=1.5, spatial_tolerance_km=10.0,
                 location_tolerance_km=2.0, association_cutoff_km=250.0, depth_range_km=[0.0, 60.0], margin_km=50.0,
                 time_before_s=300.0, min_picks=6, min_p_picks=3, min_s_picks=0, n_p_and_s_picks=2,
                 notes="Proposal for mainshock-aftershock sequences: crustal vp/vs 6.0/3.4 km/s, 1.5 s tolerance."),
    "vt": dict(region="vt", p_velocity=5.0, s_velocity=2.9, time_tolerance_s=1.0, spatial_tolerance_km=5.0,
               location_tolerance_km=1.0, association_cutoff_km=100.0, depth_range_km=[0.0, 30.0], margin_km=30.0,
               time_before_s=120.0, min_picks=5, min_p_picks=3, min_s_picks=0, n_p_and_s_picks=1,
               notes="Proposal for volcano-tectonic sequences: slow shallow edifice velocities, dense local networks."),
    "swarm": dict(region="swarm", p_velocity=5.8, s_velocity=3.3, time_tolerance_s=0.8, spatial_tolerance_km=5.0,
                  location_tolerance_km=0.5, association_cutoff_km=100.0, depth_range_km=[0.0, 25.0], margin_km=30.0,
                  time_before_s=120.0, min_picks=5, min_p_picks=3, min_s_picks=0, n_p_and_s_picks=2,
                  notes="Proposal for fluid-driven swarms: tight time tolerance for closely spaced small events."),
}


def config_path(region) -> Path:
    return CONFIG_DIR / f"{region}.json"


def default_config(region) -> AssociatorConfig:
    if region not in DEFAULT_CONFIGS:
        raise ValueError(f"No default config for region {region!r}; known: {sorted(DEFAULT_CONFIGS)}")
    return AssociatorConfig(**DEFAULT_CONFIGS[region])


def write_default_configs(directory=None) -> list:
    directory = CONFIG_DIR if directory is None else Path(directory)
    return [default_config(r).save(Path(directory) / f"{r}.json") for r in DEFAULT_CONFIGS]


def load_config(region_or_path) -> AssociatorConfig:
    """configs/association/<region>.json, or any JSON path."""
    p = Path(str(region_or_path))
    if p.suffix == ".json":
        return AssociatorConfig.load(p)
    return AssociatorConfig.load(config_path(region_or_path))


# ── geometry ─────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class LocalProjection:
    """Equirectangular km offsets around (lat0, lon0); adequate for the synthetic test double only."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.coslat = float(np.cos(np.radians(self.lat0)))

    @classmethod
    def from_stations(cls, stations) -> "LocalProjection":
        return cls(stations["lat"].astype(float).mean(), stations["lon"].astype(float).mean())

    def to_km(self, lat, lon):
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        return (lon - self.lon0) * KM_PER_DEG * self.coslat, (lat - self.lat0) * KM_PER_DEG

    def to_latlon(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return self.lat0 + y / KM_PER_DEG, self.lon0 + x / (KM_PER_DEG * self.coslat)


def _require(df, columns, name):
    missing = [c for c in columns if c not in df]
    if missing:
        raise ValueError(f"{name} lacks columns {missing}")
    return df


def _check_picks(picks) -> pd.DataFrame:
    p = _require(picks, PICK_REQUIRED, "picks").reset_index(drop=True).copy()
    p["station"] = p["station"].astype(str)
    p["phase"] = p["phase"].astype(str)
    if "score" not in p:
        p["score"] = np.nan
    if len(p) and p["pick_id"].duplicated().any():
        raise ValueError("pick_id must be unique")
    return p


def _check_stations(stations) -> pd.DataFrame:
    s = _require(stations, STATION_REQUIRED, "stations").reset_index(drop=True).copy()
    s["station"] = s["station"].astype(str)
    if s["station"].duplicated().any():
        raise ValueError("station table has duplicate station ids")
    s["elev_m"] = pd.to_numeric(s["elev_m"], errors="coerce").fillna(0.0)
    return s


def _empty_events():
    return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(columns=ASSIGNMENT_COLUMNS)


# ── associators ──────────────────────────────────────────────────────────────

class PyOctoAssociator:
    """Adapter to PyOcto (Münchmeyer 2024, Seismica 3(1); github.com/yetinam/pyocto).

    Every parameter comes from the AssociatorConfig; nothing is set here.
    `frames` maps the pick store and station table to PyOcto's input frames
    (picks: station, phase, time as POSIX seconds, probability; stations:
    id, latitude, longitude, elevation in m) and is testable without pyocto.
    `associate` imports pyocto lazily. Written against the PyOcto 0.1 API
    (VelocityModel0D/1D, OctoAssociator.from_area, transform_stations,
    associate, transform_events); it has not been executed in this
    repository because pyocto is not installed.
    """
    name = "pyocto"

    def __init__(self, config: AssociatorConfig, verbose=False, model_dir=None):
        self.config = config
        self.verbose = verbose
        self.model_dir = model_dir

    @staticmethod
    def frames(picks, stations):
        p = _check_picks(picks)
        s = _check_stations(stations)
        pick_frame = pd.DataFrame({
            "station": p["station"], "phase": p["phase"], "time": cs._to_ns(p["time"]) / 1e9,
            "probability": pd.to_numeric(p["score"], errors="coerce").fillna(1.0).astype(float),
            "pick_id": p["pick_id"]})
        station_frame = pd.DataFrame({
            "id": s["station"], "latitude": s["lat"].astype(float), "longitude": s["lon"].astype(float),
            "elevation": s["elev_m"].astype(float)})
        return pick_frame, station_frame

    @staticmethod
    def _import():
        try:
            import pyocto  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyOctoAssociator needs the `pyocto` package (Münchmeyer 2024, https://github.com/yetinam/pyocto): "
                "`pip install pyocto`. SyntheticAssociator is a pipeline test double, not a substitute.") from exc
        return pyocto

    def build(self, stations):
        pyocto = self._import()
        cfg = self.config
        if cfg.layers:
            model_dir = Path(self.model_dir) if self.model_dir is not None else Path.cwd()
            path = model_dir / f"velocity_{cfg.velocity_model}_{cfg.sha256[:12]}.bin"
            layers = pd.DataFrame(cfg.layers).rename(columns={"depth_km": "depth"})[["depth", "vp", "vs"]]
            pyocto.VelocityModel1D.create_model(layers, 1.0, cfg.association_cutoff_km, cfg.depth_range_km[1], str(path))
            velocity_model = pyocto.VelocityModel1D(str(path), tolerance=cfg.time_tolerance_s,
                                                    association_cutoff_distance=cfg.association_cutoff_km)
        else:
            velocity_model = pyocto.VelocityModel0D(p_velocity=cfg.p_velocity, s_velocity=cfg.s_velocity,
                                                    tolerance=cfg.time_tolerance_s,
                                                    association_cutoff_distance=cfg.association_cutoff_km)
        lat, lon = stations["latitude"].astype(float), stations["longitude"].astype(float)
        dlat = cfg.margin_km / KM_PER_DEG
        dlon = cfg.margin_km / (KM_PER_DEG * max(np.cos(np.radians(lat.mean())), 0.1))
        return pyocto.OctoAssociator.from_area(
            lat=(lat.min() - dlat, lat.max() + dlat), lon=(lon.min() - dlon, lon.max() + dlon),
            zlim=tuple(cfg.depth_range_km), time_before=cfg.time_before_s, velocity_model=velocity_model,
            n_picks=cfg.min_picks, n_p_picks=cfg.min_p_picks, n_s_picks=cfg.min_s_picks,
            n_p_and_s_picks=(cfg.n_p_and_s_picks if cfg.n_picks_p_and_s_policy == "require" else 0),
            pick_match_tolerance=cfg.time_tolerance_s, min_node_size=cfg.spatial_tolerance_km,
            min_node_size_location=cfg.location_tolerance_km)

    def associate(self, picks, stations, config=None):
        if config is not None:
            self.config = config
        pick_frame, station_frame = self.frames(picks, stations)
        associator = self.build(station_frame)
        station_frame = associator.transform_stations(station_frame)
        events, assignments = associator.associate(pick_frame, station_frame)
        if len(events):
            events = associator.transform_events(events)
        return self._to_schema(events, assignments, pick_frame)

    def _to_schema(self, events, assignments, pick_frame):
        if events is None or len(events) == 0:
            return _empty_events()
        a = assignments.merge(pick_frame.reset_index().rename(columns={"index": "pick_idx"}), on="pick_idx", how="left")
        a = a.rename(columns={"time": "time_s", "probability": "score"})
        a["time"] = pd.to_datetime((a["time_s"] * 1e9).round().astype(np.int64), unit="ns", utc=True)
        a = a[["event_idx", "pick_id", "station", "phase", "time", "score", "residual"]]
        per = a.groupby("event_idx")
        both = a.groupby(["event_idx", "station"])["phase"].nunique().eq(2).groupby("event_idx").sum()
        ev = pd.DataFrame({
            "event_idx": events["idx"].astype(int),
            "time": pd.to_datetime((events["time"].astype(float) * 1e9).round().astype(np.int64), unit="ns", utc=True),
            "lat": events["latitude"].astype(float), "lon": events["longitude"].astype(float),
            "depth_km": events["depth"].astype(float)})
        ev["n_picks"] = ev["event_idx"].map(per.size()).fillna(0).astype(int)
        ev["n_p"] = ev["event_idx"].map(a[a["phase"] == "P"].groupby("event_idx").size()).fillna(0).astype(int)
        ev["n_s"] = ev["event_idx"].map(a[a["phase"] == "S"].groupby("event_idx").size()).fillna(0).astype(int)
        ev["n_stations"] = ev["event_idx"].map(per["station"].nunique()).fillna(0).astype(int)
        ev["n_p_and_s"] = ev["event_idx"].map(both).fillna(0).astype(int)
        ev["misfit_s"] = ev["event_idx"].map(per["residual"].apply(lambda r: float(np.mean(np.abs(r)))))
        ev["associator"] = self.name
        ev["config_sha256"] = self.config.sha256
        return ev[EVENT_COLUMNS].reset_index(drop=True), a.reset_index(drop=True)


class SyntheticAssociator:
    """Deterministic coarse-grid back-projection with homogeneous velocities: a TEST DOUBLE.

    Origin-time votes of every remaining pick are cast on every node of an
    x/y/depth grid (step spatial_tolerance_km, station extent plus margin_km)
    into time bins of width time_tolerance_s; the (bin, node) cell with the
    most distinct station-phase votes over two adjacent bins seeds an event
    (ties: earliest bin, then lowest node index). One pick per station-phase
    is kept (closest implied origin), the location is refined on a local
    sub-grid, picks beyond the tolerance are released, and the minimum-pick
    rules of the config decide whether the event stands; its picks then leave
    the pool. Straight rays, no station corrections, no 1-D model: good enough
    to exercise the pipeline on synthetic picks, never a scientific result.
    """
    name = "synthetic"

    def __init__(self, config: AssociatorConfig, refine=True, refine_points=7, max_iterations=2000):
        self.config = config
        self.refine = refine
        self.refine_points = int(refine_points)
        self.max_iterations = int(max_iterations)

    def associate(self, picks, stations, config=None):
        cfg = self.config if config is None else config
        self.config = cfg
        picks, stations = _check_picks(picks), _check_stations(stations)
        if len(picks) == 0 or len(stations) == 0:
            return _empty_events()
        proj = LocalProjection.from_stations(stations)
        sx, sy = proj.to_km(stations["lat"].to_numpy(float), stations["lon"].to_numpy(float))
        sz = -stations["elev_m"].to_numpy(float) / 1000.0
        sta_index = {s: i for i, s in enumerate(stations["station"])}
        known = picks["station"].map(sta_index)
        usable = known.notna() & picks["phase"].isin(["P", "S"])
        pk = picks[usable].reset_index(drop=True)
        if len(pk) < cfg.min_picks:
            return _empty_events()
        st = known[usable].to_numpy(dtype=int)
        ph = (pk["phase"] == "S").to_numpy(dtype=int)
        t_ns = cs._to_ns(pk["time"])
        t_ref = int(t_ns.min())
        t = (t_ns - t_ref) / 1e9
        n_sp = 2 * len(stations)
        sp = st * 2 + ph

        step, margin = cfg.spatial_tolerance_km, cfg.margin_km
        xs = np.arange(sx.min() - margin, sx.max() + margin + step / 2, step)
        ys = np.arange(sy.min() - margin, sy.max() + margin + step / 2, step)
        z0, z1 = cfg.depth_range_km
        zs = np.arange(z0, z1 + step / 2, step)
        X, Y, Z = (g.ravel() for g in np.meshgrid(xs, ys, zs, indexing="ij"))
        m = len(X)
        V = np.array([cfg.p_velocity, cfg.s_velocity])
        D = np.sqrt((X[:, None] - sx[None, :]) ** 2 + (Y[:, None] - sy[None, :]) ** 2 + (Z[:, None] - sz[None, :]) ** 2)
        TT = D[:, :, None] / V[None, None, :]                      # node × station × phase
        tol = cfg.time_tolerance_s

        pool = np.ones(len(pk), dtype=bool)
        rejected = np.zeros(0, dtype=np.int64)
        events, assigns = [], []
        iteration = 0
        while pool.sum() >= cfg.min_picks:
            iteration += 1
            if iteration > self.max_iterations:
                raise RuntimeError(f"SyntheticAssociator exceeded {self.max_iterations} iterations; "
                                   "the test double is not meant for this pick volume")
            idx = np.flatnonzero(pool)
            k = len(idx)
            O = t[idx][None, :] - TT[:, st[idx], ph[idx]]           # node × pick implied origin times
            B = np.floor(O / tol).astype(np.int64)
            bmin = int(B.min())
            B -= bmin
            node = np.repeat(np.arange(m, dtype=np.int64), k)
            spk = np.tile(sp[idx].astype(np.int64), m)
            b = B.ravel()
            keys = np.concatenate([(b * m + node) * n_sp + spk, ((b + 1) * m + node) * n_sp + spk])
            cells, counts = np.unique(np.unique(keys) // n_sp, return_counts=True)
            abs_cells = (cells // m + bmin) * m + cells % m
            if len(rejected):
                counts = np.where(np.isin(abs_cells, rejected), 0, counts)
            best = int(np.argmax(counts))
            if counts[best] < cfg.min_picks:
                break
            j, node_i = int(cells[best] // m), int(cells[best] % m)
            Bn = B[node_i]
            cand = idx[(Bn == j - 1) | (Bn == j)]
            built = self._build(cand, node_i, X, Y, Z, sx, sy, sz, st, ph, sp, t, V, TT, tol, cfg, xs, ys, zs)
            if built is None:
                rejected = np.append(rejected, abs_cells[best])
                continue
            sel, x, y, z, origin, res = built
            e = len(events)
            lat, lon = proj.to_latlon(x, y)
            n_p, n_s = int((ph[sel] == 0).sum()), int((ph[sel] == 1).sum())
            both = int(pd.Series(ph[sel]).groupby(st[sel]).nunique().eq(2).sum())
            events.append(dict(event_idx=e, time=pd.Timestamp(t_ref + int(round(origin * 1e9)), unit="ns", tz="UTC"),
                               lat=float(lat), lon=float(lon), depth_km=float(z), n_picks=int(len(sel)), n_p=n_p,
                               n_s=n_s, n_stations=int(len(np.unique(st[sel]))), n_p_and_s=both,
                               misfit_s=float(np.mean(np.abs(res))), associator=self.name, config_sha256=cfg.sha256))
            for i, r in zip(sel, res):
                assigns.append(dict(event_idx=e, pick_id=pk.at[i, "pick_id"], station=pk.at[i, "station"],
                                    phase=pk.at[i, "phase"], time=pk.at[i, "time"], score=pk.at[i, "score"],
                                    residual=float(r)))
            pool[sel] = False
        ev = pd.DataFrame(events, columns=EVENT_COLUMNS)
        asg = pd.DataFrame(assigns, columns=ASSIGNMENT_COLUMNS)
        if len(asg):
            asg["time"] = pd.to_datetime(asg["time"], utc=True)
        return ev, asg

    def _one_per_station_phase(self, cand, o, origin, sp, tol):
        order = cand[np.argsort(np.abs(o - origin), kind="stable")]
        oo = o[np.argsort(np.abs(o - origin), kind="stable")]
        seen, keep, keep_o = set(), [], []
        for i, oi in zip(order, oo):
            if sp[i] in seen or abs(oi - origin) > tol:
                continue
            seen.add(sp[i])
            keep.append(i)
            keep_o.append(oi)
        return np.array(keep, dtype=int), np.array(keep_o, dtype=float)

    def _build(self, cand, node_i, X, Y, Z, sx, sy, sz, st, ph, sp, t, V, TT, tol, cfg, xs, ys, zs):
        o = t[cand] - TT[node_i, st[cand], ph[cand]]
        origin = float(np.median(o))
        sel, o_sel = self._one_per_station_phase(cand, o, origin, sp, tol)
        if len(sel) < cfg.min_picks:
            return None
        origin = float(np.median(o_sel))
        x, y, z = float(X[node_i]), float(Y[node_i]), float(Z[node_i])
        if self.refine:
            step = cfg.spatial_tolerance_km
            offs = np.linspace(-step, step, self.refine_points)
            gx, gy, gz = (g.ravel() for g in np.meshgrid(x + offs, y + offs, np.clip(z + offs, cfg.depth_range_km[0],
                                                                                          cfg.depth_range_km[1]), indexing="ij"))
            d = np.sqrt((gx[:, None] - sx[st[sel]][None, :]) ** 2 + (gy[:, None] - sy[st[sel]][None, :]) ** 2
                        + (gz[:, None] - sz[st[sel]][None, :]) ** 2)
            oo = t[sel][None, :] - d / V[ph[sel]][None, :]
            og = np.median(oo, axis=1)
            misfit = np.mean(np.abs(oo - og[:, None]), axis=1)
            bi = int(np.argmin(misfit))
            x, y, z, origin = float(gx[bi]), float(gy[bi]), float(gz[bi]), float(og[bi])
            d_all = np.sqrt((x - sx[st[cand]]) ** 2 + (y - sy[st[cand]]) ** 2 + (z - sz[st[cand]]) ** 2)
            o = t[cand] - d_all / V[ph[cand]]
            sel, o_sel = self._one_per_station_phase(cand, o, origin, sp, tol)
            if len(sel) < cfg.min_picks:
                return None
            origin = float(np.median(o_sel))
            keep = np.abs(o_sel - origin) <= tol
            sel, o_sel = sel[keep], o_sel[keep]
        n_p, n_s = int((ph[sel] == 0).sum()), int((ph[sel] == 1).sum())
        both = int(pd.Series(ph[sel]).groupby(st[sel]).nunique().eq(2).sum())
        if len(sel) < cfg.min_picks or n_p < cfg.min_p_picks or n_s < cfg.min_s_picks:
            return None
        if cfg.n_picks_p_and_s_policy == "require" and both < cfg.n_p_and_s_picks:
            return None
        return sel, x, y, z, origin, o_sel - origin


BACKENDS = {"synthetic": SyntheticAssociator, "pyocto": PyOctoAssociator}


def associate(picks, stations, config: AssociatorConfig, backend="synthetic", **kwargs):
    """(events, assignments) from one backend; every parameter comes from `config`."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown associator backend {backend!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[backend](config, **kwargs).associate(picks, stations)


# ── event matching ───────────────────────────────────────────────────────────

def _standard_events(df, kind) -> pd.DataFrame:
    """id, time_ns, lat, lon, depth_km (+ mag) from a catalogue (`event`, `origin`) or an events frame (`event_idx`, `time`)."""
    df = df.reset_index(drop=True)
    time_col = "origin" if "origin" in df else "time"
    if time_col not in df:
        raise ValueError(f"{kind} frame needs an `origin` or `time` column")
    id_col = "event" if kind == "reference" else "event_idx"
    ids = df[id_col] if id_col in df else pd.Series(np.arange(len(df)), index=df.index)
    out = pd.DataFrame({"id": ids.to_numpy(), "time_ns": cs._to_ns(df[time_col]),
                        "lat": pd.to_numeric(df["lat"], errors="coerce").to_numpy(float),
                        "lon": pd.to_numeric(df["lon"], errors="coerce").to_numpy(float),
                        "depth_km": (pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
                                     if "depth_km" in df else np.full(len(df), np.nan))})
    if "mag" in df:
        out["mag"] = pd.to_numeric(df["mag"], errors="coerce").to_numpy(float)
    if out["id"].duplicated().any():
        raise ValueError(f"{kind} ids are not unique")
    return out


@dataclass
class EventMatchResult:
    pairs: pd.DataFrame                # PAIR_COLUMNS; dt_s = predicted - reference
    unmatched_reference: pd.DataFrame  # reference rows (original columns)
    unmatched_predicted: pd.DataFrame  # predicted rows (original columns)
    reference: pd.DataFrame
    predicted: pd.DataFrame
    feasible: np.ndarray               # n_reference × n_predicted
    tolerances: dict

    @property
    def n_reference(self):
        return len(self.reference)

    @property
    def n_predicted(self):
        return len(self.predicted)

    @property
    def n_matched(self):
        return len(self.pairs)

    @property
    def recovery(self):
        return self.n_matched / self.n_reference if self.n_reference else np.nan

    def matched_reference_ids(self) -> set:
        return set(self.pairs["event"].astype(str))


def match_events(predicted, reference, tol_time_s=MATCH_TOL_TIME_S, tol_km=MATCH_TOL_KM,
                 tol_depth_km=MATCH_TOL_DEPTH_KM) -> EventMatchResult:
    """One-to-one maximum-cardinality event matching, minimum normalised residual second.

    A (reference, predicted) pair is feasible when |origin-time difference|
    <= tol_time_s (integer nanoseconds, so equality is feasible), epicentral
    distance <= tol_km (haversine), and, when tol_depth_km is given, |depth
    difference| <= tol_depth_km or either depth is missing. Feasible cost is
    dt/tol_time_s + dist/tol_km (+ ddepth/tol_depth_km), each term at most
    1; infeasible pairs cost (min(n_r, n_p) + 1) * n_terms + 1, above any
    feasible total, so scipy's linear assignment maximises cardinality first
    and minimises the summed normalised residual second, as
    continuous_scoring.match_picks does for picks. Reference ids come from
    `event`, predicted ids from `event_idx` (else the row position).
    """
    if tol_time_s <= 0 or tol_km <= 0 or (tol_depth_km is not None and tol_depth_km <= 0):
        raise ValueError("tolerances must be positive (tol_depth_km may be None)")
    r, p = _standard_events(reference, "reference"), _standard_events(predicted, "predicted")
    nr, npred = len(r), len(p)
    tol = dict(tol_time_s=float(tol_time_s), tol_km=float(tol_km),
               tol_depth_km=(None if tol_depth_km is None else float(tol_depth_km)))
    if nr == 0 or npred == 0:
        return EventMatchResult(pd.DataFrame(columns=PAIR_COLUMNS), reference.reset_index(drop=True),
                                predicted.reset_index(drop=True), r, p, np.zeros((nr, npred), dtype=bool), tol)
    dt_ns = p["time_ns"].to_numpy()[None, :] - r["time_ns"].to_numpy()[:, None]
    dt_s = dt_ns / 1e9
    tol_ns = int(round(tol_time_s * 1e9))
    dist = haversine_km(r["lat"].to_numpy()[:, None], r["lon"].to_numpy()[:, None],
                        p["lat"].to_numpy()[None, :], p["lon"].to_numpy()[None, :])
    dd = np.abs(p["depth_km"].to_numpy()[None, :] - r["depth_km"].to_numpy()[:, None])
    feasible = (np.abs(dt_ns) <= tol_ns) & (dist <= tol_km)
    cost = np.abs(dt_s) / tol_time_s + dist / tol_km
    n_terms = 2
    if tol_depth_km is not None:
        feasible &= (dd <= tol_depth_km) | np.isnan(dd)
        cost = cost + np.nan_to_num(dd, nan=0.0) / tol_depth_km
        n_terms = 3
    big = (min(nr, npred) + 1) * n_terms + 1.0
    rows, cols = linear_sum_assignment(np.where(feasible, cost, big))
    pairs = [dict(event=r.at[i, "id"], event_idx=p.at[j, "id"], dt_s=float(dt_s[i, j]), dist_km=float(dist[i, j]),
                  ddepth_km=float(dd[i, j]), cost=float(cost[i, j])) for i, j in zip(rows, cols) if feasible[i, j]]
    pairs = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    used_r = {int(i) for i, j in zip(rows, cols) if feasible[i, j]}
    used_p = {int(j) for i, j in zip(rows, cols) if feasible[i, j]}
    ref_out = reference.reset_index(drop=True)
    pred_out = predicted.reset_index(drop=True)
    return EventMatchResult(pairs, ref_out[~ref_out.index.isin(used_r)].reset_index(drop=True),
                            pred_out[~pred_out.index.isin(used_p)].reset_index(drop=True), r, p, feasible, tol)


# ── split and merge diagnostics ──────────────────────────────────────────────

@dataclass
class Diagnostics:
    splits: pd.DataFrame
    merges: pd.DataFrame
    overlap: float

    @property
    def n_splits(self):
        return int(len(self.splits))

    @property
    def n_merges(self):
        return int(len(self.merges))

    def to_dict(self) -> dict:
        return dict(n_splits=self.n_splits, n_merges=self.n_merges, overlap_threshold=self.overlap,
                    splits=self.splits.to_dict(orient="records"), merges=self.merges.to_dict(orient="records"))


def split_merge_diagnostics(match: EventMatchResult, assignments=None, overlap=SPLIT_OVERLAP) -> Diagnostics:
    """Splits and merges relative to the reference catalogue.

    Split: a reference event with two or more predicted events that either
    fall within the matching tolerance of it (`feasible`) or whose assigned
    picks carry that reference event as `matched_event` (the 35A pick-level
    match) for at least `overlap` of their picks. Merge: a predicted event
    within tolerance of two or more reference events. Rules are labelled per
    row; `reference_events_in_picks` lists the matched_event composition of
    a merged predicted event when assignments are given.
    """
    F = match.feasible
    ref_ids = [str(v) for v in match.reference["id"]]
    pred_ids = list(match.predicted["id"])
    by_ref = {rid: set() for rid in ref_ids}
    rule_ref = {rid: set() for rid in ref_ids}
    for i, rid in enumerate(ref_ids):
        for j in np.flatnonzero(F[i]) if F.size else []:
            by_ref[rid].add(pred_ids[j])
            rule_ref[rid].add("tolerance")
    composition = {}
    if assignments is not None and len(assignments) and "matched_event" in assignments:
        for eidx, g in assignments.groupby("event_idx"):
            me = g["matched_event"].dropna().astype(str)
            counts = me.value_counts()
            composition[eidx] = {str(k): int(v) for k, v in counts.items()}
            for rid, c in counts.items():
                if c / len(g) >= overlap and str(rid) in by_ref:
                    by_ref[str(rid)].add(eidx)
                    rule_ref[str(rid)].add("pick_overlap")
    splits = [dict(event=rid, n_predicted=len(preds), predicted=sorted(preds, key=str),
                   rule="+".join(sorted(rule_ref[rid])))
              for rid, preds in by_ref.items() if len(preds) >= 2]
    merges = []
    for j, pid in enumerate(pred_ids):
        refs = [ref_ids[i] for i in (np.flatnonzero(F[:, j]) if F.size else [])]
        if len(refs) >= 2:
            merges.append(dict(event_idx=pid, n_reference=len(refs), events=refs,
                               reference_events_in_picks=composition.get(pid, {}), rule="tolerance"))
    return Diagnostics(pd.DataFrame(splits, columns=SPLIT_COLUMNS), pd.DataFrame(merges, columns=MERGE_COLUMNS),
                       float(overlap))


# ── coverage and recovery tables ─────────────────────────────────────────────

def _windows_ns(windows):
    """[(t0_ns, t1_ns)] from a windows.csv frame (t0, t1) or an iterable of pairs; None stays None."""
    if windows is None:
        return None
    if isinstance(windows, pd.DataFrame):
        pairs = list(zip(windows["t0"], windows["t1"]))
    else:
        pairs = list(windows)
    out = []
    for t0, t1 in pairs:
        a, b = int(cs._to_ns([t0])[0]), int(cs._to_ns([t1])[0])
        if b < a:
            raise ValueError(f"window ends before it starts: {t0} .. {t1}")
        out.append((a, b))
    return sorted(out)


def _covered_ns(windows_ns, a, b) -> int:
    """Length (ns) of the union of windows inside [a, b]."""
    if b <= a:
        return 0
    total, cur = 0, None
    for w0, w1 in windows_ns:
        w0, w1 = max(w0, a), min(w1, b)
        if w1 <= w0:
            continue
        if cur is None or w0 > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [w0, w1]
        else:
            cur[1] = max(cur[1], w1)
    if cur is not None:
        total += cur[1] - cur[0]
    return int(total)


def _in_windows(t_ns, windows_ns, lead_ns=0) -> np.ndarray:
    t_ns = np.asarray(t_ns, dtype=np.int64)
    out = np.zeros(len(t_ns), dtype=bool)
    for w0, w1 in windows_ns:
        out |= (t_ns >= w0 - lead_ns) & (t_ns <= w1)
    return out


def _pairs_of(matches) -> pd.DataFrame:
    return matches.pairs if isinstance(matches, EventMatchResult) else matches


def _bin_label(a, b, unit=""):
    if np.isinf(b):
        return f">= {a:g}{unit}"
    return f"[{a:g}, {b:g}){unit}"


def recovery_tables(matches, reference, predicted, mainshock_time=None, windows=None, magnitude_bins=MAGNITUDE_BINS,
                    hour_bins=HOUR_BINS, day_bins=None, assignments=None, n_picks_total=None,
                    min_coverage=MIN_COVERAGE, lead_s=0.0) -> dict:
    """Recovery versus magnitude, hour after mainshock and day of sequence, with coverage flags.

    Denominators are reference events whose origin lies inside a window
    (t0 - lead_s <= origin <= t1); events outside every window are counted
    separately and never enter a recovery fraction. `claim_supported` is
    True on a row only when windows were given, the row has covered
    reference events, and, for time bins, the union of windows covers at
    least `min_coverage` of the bin's span measured from `mainshock_time`
    (else the first window start, else the first reference origin; the
    origin used is written in `time_origin`/`time_origin_label`). The open
    last hour/day bin ends at the later of the last reference origin and the
    last window end. Without windows every claim_supported is False.
    Returns by_magnitude, by_hour, by_day, station_support, coverage and
    summary frames.
    """
    pairs = _pairs_of(matches)
    ref = _standard_events(reference, "reference")
    ref["event"] = ref["id"].astype(str)
    matched = ref["event"].isin(set(pairs["event"].astype(str)))
    ref["matched"] = matched.to_numpy()
    t_ns = ref["time_ns"].to_numpy()
    win = _windows_ns(windows)
    coverage_known = win is not None
    if coverage_known:
        ref["in_window"] = _in_windows(t_ns, win, int(round(lead_s * 1e9)))
    else:
        ref["in_window"] = True
    if mainshock_time is not None:
        T, label = int(cs._to_ns([mainshock_time])[0]), "mainshock"
    elif coverage_known and len(win):
        T, label = win[0][0], "first_window_start"
    elif len(ref):
        T, label = int(t_ns.min()), "first_reference_event"
    else:
        T, label = 0, "undefined"
    T_iso = pd.Timestamp(T, unit="ns", tz="UTC").isoformat()
    ref["hours_after"] = (t_ns - T) / 3.6e12
    ref["days_after"] = np.floor(ref["hours_after"] / 24.0)
    ends = [T] + ([int(t_ns.max())] if len(ref) else []) + ([w[1] for w in win] if coverage_known else [])
    end_ns = max(ends)

    def counts(sub):
        cov = sub[sub["in_window"]]
        n_cov, n_m = int(len(cov)), int(cov["matched"].sum())
        return dict(n_reference=int(len(sub)), n_reference_covered=n_cov, n_reference_outside_windows=int(len(sub) - n_cov),
                    n_matched=n_m, recovery=(n_m / n_cov if n_cov else np.nan))

    # magnitude
    mb = list(magnitude_bins)
    mag = ref["mag"].to_numpy(float) if "mag" in ref else np.full(len(ref), np.nan)
    rows = []
    for a, b in zip(mb[:-1], mb[1:]):
        sub = ref[(mag >= a) & (mag < b)]
        row = dict(bin=_bin_label(a, b), mag_lo=a, mag_hi=b, **counts(sub))
        row["claim_supported"] = bool(coverage_known and row["n_reference_covered"] > 0)
        rows.append(row)
    sub = ref[np.isnan(mag)]
    if len(sub):
        row = dict(bin="unknown", mag_lo=np.nan, mag_hi=np.nan, **counts(sub))
        row["claim_supported"] = False
        rows.append(row)
    by_magnitude = pd.DataFrame(rows)

    def time_rows(edges, scale_ns, unit):
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = ref["hours_after"] if unit == "h" else ref["days_after"]
            sub = ref[(v >= a) & (v < b)] if unit == "h" else ref[(ref["hours_after"] / 24.0 >= a) & (ref["hours_after"] / 24.0 < b)]
            s0 = T + int(round(a * scale_ns))
            s1 = end_ns if np.isinf(b) else T + int(round(b * scale_ns))
            span = s1 - s0
            frac = (_covered_ns(win, s0, s1) / span) if (coverage_known and span > 0) else np.nan
            row = dict(bin=_bin_label(a, b, unit), lo=a, hi=b, **counts(sub), coverage_fraction=frac,
                       time_origin=T_iso, time_origin_label=label)
            row["claim_supported"] = bool(coverage_known and row["n_reference_covered"] > 0
                                          and np.isfinite(frac) and frac >= min_coverage)
            out.append(row)
        return pd.DataFrame(out)

    by_hour = time_rows(list(hour_bins), 3.6e12, "h")
    if day_bins is None:
        last_day = int(np.floor((end_ns - T) / 8.64e13)) if end_ns > T else 0
        day_bins = list(range(0, last_day + 2))
    by_day = time_rows(list(day_bins), 8.64e13, "d")

    # station support
    if assignments is not None and len(assignments):
        a = assignments
        per = a.groupby("event_idx")
        both = a.groupby(["event_idx", "station"])["phase"].nunique().eq(2).groupby("event_idx").sum()
        support = pd.DataFrame({"n_picks": per.size(), "n_p": a[a["phase"] == "P"].groupby("event_idx").size(),
                                "n_s": a[a["phase"] == "S"].groupby("event_idx").size(),
                                "n_stations": per["station"].nunique(), "n_p_and_s": both}).fillna(0).astype(int)
        support.index.name = "event_idx"
        support = support.reset_index()
        n_assigned = int(a["pick_id"].nunique())
    else:
        pred_ids = _standard_events(predicted, "predicted")["id"] if predicted is not None and len(predicted) else []
        support = pd.DataFrame({"event_idx": list(pred_ids)})
        for c in ("n_picks", "n_p", "n_s", "n_stations", "n_p_and_s"):
            support[c] = np.nan
        n_assigned = 0
    if len(pairs) and len(support):
        me = pairs[["event_idx", "event"]].rename(columns={"event": "matched_event"})
        support = support.merge(me.astype({"event_idx": support["event_idx"].dtype}), on="event_idx", how="left")
    else:
        support["matched_event"] = pd.Series(dtype=object)
    support["matched"] = support["matched_event"].notna()
    support["claim_supported"] = bool(coverage_known)

    # per-event coverage
    coverage = ref[["event", "time_ns", "lat", "lon", "depth_km"] + (["mag"] if "mag" in ref else [])].copy()
    coverage["origin"] = pd.to_datetime(coverage.pop("time_ns"), unit="ns", utc=True)
    coverage["hours_after"] = ref["hours_after"]
    coverage["in_window"] = ref["in_window"]
    coverage["matched"] = ref["matched"]
    coverage["claim_supported"] = bool(coverage_known)

    n_pred = int(len(predicted)) if predicted is not None else int(len(support))
    total = counts(ref)
    n_unmatched_pred = n_pred - int(len(pairs))
    summary = pd.DataFrame([dict(
        n_reference=total["n_reference"], n_reference_covered=total["n_reference_covered"],
        n_reference_outside_windows=total["n_reference_outside_windows"], n_predicted=n_pred,
        n_matched=total["n_matched"], n_pairs=int(len(pairs)), recovery=total["recovery"],
        n_unmatched_reference_covered=total["n_reference_covered"] - total["n_matched"],
        n_unmatched_reference_outside_windows=int((~ref["in_window"] & ~ref["matched"]).sum()),
        n_matched_outside_windows=int((~ref["in_window"] & ref["matched"]).sum()),
        n_unmatched_predicted=n_unmatched_pred,
        n_picks_total=(int(n_picks_total) if n_picks_total is not None else np.nan), n_picks_assigned=n_assigned,
        n_picks_unassociated=(int(n_picks_total) - n_assigned if n_picks_total is not None else np.nan),
        n_windows=(len(win) if coverage_known else 0), time_origin=T_iso, time_origin_label=label,
        min_coverage=min_coverage, claim_supported=bool(coverage_known and total["n_reference_covered"] > 0))])
    return dict(by_magnitude=by_magnitude, by_hour=by_hour, by_day=by_day, station_support=support,
                coverage=coverage, summary=summary)


# ── paired block bootstrap ───────────────────────────────────────────────────

def paired_block_bootstrap(matches_a, matches_b, reference, block="event", n_boot=2000, ci=0.95, seed=0,
                           mainshock_time=None) -> dict:
    """Bootstrap interval for recovery(b) - recovery(a) over the same reference events.

    The unit is the reference event, so P and S observations and stations of
    one event are never separated. block="event" resamples events;
    block="hour" or "day" resamples whole clock-hour or day blocks measured
    from `mainshock_time` (else the first reference origin); any other string
    names a column of `reference` to block on. Percentile interval of level
    `ci`; `excludes_zero` is the paired verdict. Restrict `reference` to the
    covered events (recovery_tables' coverage frame) before calling.
    """
    ref = _standard_events(reference, "reference")
    if len(ref) == 0:
        raise ValueError("paired_block_bootstrap needs at least one reference event")
    ids = ref["id"].astype(str)
    a = ids.isin({str(v) for v in _pairs_of(matches_a)["event"]}).to_numpy(float)
    b = ids.isin({str(v) for v in _pairs_of(matches_b)["event"]}).to_numpy(float)
    if block == "event":
        g = np.arange(len(ref))
    elif block in ("hour", "day"):
        T = int(cs._to_ns([mainshock_time])[0]) if mainshock_time is not None else int(ref["time_ns"].min())
        g = np.floor((ref["time_ns"].to_numpy() - T) / (3.6e12 if block == "hour" else 8.64e13)).astype(np.int64)
    elif isinstance(reference, pd.DataFrame) and block in reference:
        g = reference.reset_index(drop=True)[block].to_numpy()
    else:
        raise ValueError(f"block must be 'event', 'hour', 'day' or a column of reference, not {block!r}")
    codes, uniq = pd.factorize(pd.Series(g), sort=True)
    G = len(uniq)
    n_g = np.bincount(codes, minlength=G).astype(float)
    a_g = np.bincount(codes, weights=a, minlength=G)
    b_g = np.bincount(codes, weights=b, minlength=G)
    rng = np.random.default_rng(seed)
    samp = rng.integers(0, G, size=(int(n_boot), G))
    N = n_g[samp].sum(axis=1)
    diffs = (b_g[samp].sum(axis=1) - a_g[samp].sum(axis=1)) / N
    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return dict(block=block, n_events=int(len(ref)), n_blocks=int(G), n_boot=int(n_boot), ci=float(ci), seed=int(seed),
                recovery_a=float(a.mean()), recovery_b=float(b.mean()), diff=float(b.mean() - a.mean()),
                ci_low=float(lo), ci_high=float(hi), excludes_zero=bool(lo > 0 or hi < 0))


# ── run and CLI ──────────────────────────────────────────────────────────────

def select_picks(picks, model_id=None, threshold=None, key=None) -> pd.DataFrame:
    """One sequence, one access id, one model and one threshold from the 35A
    pick store; unambiguous or an error.

    A store holding more than one `key` must be narrowed with `key`; a store
    holding more than one `access_id` for the selected key is refused (two
    scoring runs were mixed; re-run the scorer or pre-filter). Nothing is
    ever merged across sequences or runs.
    """
    sub = picks
    if "key" in sub:
        keys = sorted(sub["key"].dropna().astype(str).unique())
        if key is None:
            if len(keys) > 1:
                raise ValueError(f"the pick store holds several sequences {keys}; pass key= to select one")
        else:
            sub = sub[sub["key"].astype(str) == str(key)]
            if len(sub) == 0:
                raise ValueError(f"key {key} not in the pick store ({keys})")
    if "access_id" in sub:
        access_ids = sorted(sub["access_id"].dropna().astype(str).unique())
        if len(access_ids) > 1:
            raise ValueError(f"the pick store mixes {len(access_ids)} scoring runs (access_id {access_ids}); "
                             "pre-filter to one run before associating")
    if "model_id" in sub:
        ids = sorted(sub["model_id"].astype(str).unique())
        if model_id is None:
            if len(ids) != 1:
                raise ValueError(f"--model-id is required: the pick store holds {ids}")
            model_id = ids[0]
        sub = sub[sub["model_id"].astype(str) == str(model_id)]
        if len(sub) == 0:
            raise ValueError(f"model_id {model_id} not in the pick store ({ids})")
    if "threshold" in sub:
        thr = sorted(sub["threshold"].astype(float).unique())
        if threshold is None:
            if len(thr) != 1:
                raise ValueError(f"--threshold is required: the pick store holds {thr}")
            threshold = thr[0]
        sub = sub[np.isclose(sub["threshold"].astype(float), float(threshold))]
        if len(sub) == 0:
            raise ValueError(f"threshold {threshold} not in the pick store ({thr})")
    return sub.reset_index(drop=True)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(picks, stations, catalog, config: AssociatorConfig, *, model_id=None, threshold=None, windows=None,
        mainshock_time=None, key=None, backend="synthetic", tol_time_s=MATCH_TOL_TIME_S, tol_km=MATCH_TOL_KM,
        tol_depth_km=MATCH_TOL_DEPTH_KM, restrict_stations=True, out_dir=None, sources=None) -> dict:
    """Associate, match, diagnose and tabulate one (model, threshold) of a pick store; write if out_dir."""
    picks_sel = select_picks(picks, model_id, threshold, key=key)
    if restrict_stations and len(picks_sel):
        stations = stations[stations["station"].astype(str).isin(set(picks_sel["station"].astype(str)))]
    events, assignments = associate(picks_sel, stations, config, backend=backend)
    if len(assignments) and "matched_event" in picks_sel:
        assignments = assignments.merge(picks_sel[["pick_id", "matched_event"]], on="pick_id", how="left")
    match = match_events(events, catalog, tol_time_s=tol_time_s, tol_km=tol_km, tol_depth_km=tol_depth_km)
    diag = split_merge_diagnostics(match, assignments)
    tables = recovery_tables(match, catalog, events, mainshock_time=mainshock_time, windows=windows,
                             assignments=assignments, n_picks_total=len(picks_sel))
    access_ids = sorted(picks_sel["access_id"].dropna().astype(str).unique()) if "access_id" in picks_sel else []
    model_ids = sorted(picks_sel["model_id"].astype(str).unique()) if "model_id" in picks_sel else []
    thresholds = sorted(picks_sel["threshold"].astype(float).unique()) if "threshold" in picks_sel else []
    meta = dict(
        checkpoint="36A", key=key, access_id=(access_ids[0] if access_ids else None),
        model_id=(model_ids[0] if len(model_ids) == 1 else model_ids or model_id),
        threshold=(thresholds[0] if len(thresholds) == 1 else thresholds or threshold),
        associator=backend, config=config.to_dict(), config_sha256=config.sha256,
        match_tolerances=dict(tol_time_s=tol_time_s, tol_km=tol_km, tol_depth_km=tol_depth_km),
        mainshock_time=(None if mainshock_time is None else cs.to_timestamp(mainshock_time).isoformat()),
        n_picks=int(len(picks_sel)), n_stations=int(len(stations)), n_events=int(len(events)),
        n_reference=int(len(catalog)), n_matched=match.n_matched, n_splits=diag.n_splits, n_merges=diag.n_merges,
        git_commit=_git_commit(), sources={k: dict(path=str(v), sha256=_sha256_file(v)) for k, v in (sources or {}).items()},
        pyocto_version=None)
    if backend == "pyocto":
        import pyocto
        meta["pyocto_version"] = getattr(pyocto, "__version__", "unknown")
    stamp = dict(config_sha256=config.sha256, model_id=meta["model_id"], threshold=meta["threshold"],
                 access_id=meta["access_id"], key=key)
    events = events.assign(**{k: v for k, v in stamp.items() if k != "config_sha256"})
    assignments = assignments.assign(**stamp)
    pairs = match.pairs.assign(**stamp)
    result = dict(meta=meta, events=events, assignments=assignments, matches=pairs,
                  unmatched_predicted=match.unmatched_predicted, unmatched_reference=match.unmatched_reference,
                  diagnostics=diag, tables=tables, match=match)
    if out_dir is not None:
        write_run(result, out_dir)
    return result


def write_run(result, out_dir) -> Path:
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name in ("events", "assignments", "matches", "unmatched_predicted", "unmatched_reference"):
        result[name].to_parquet(out / f"{name}.parquet", index=False)
    diag = result["diagnostics"]
    diag.splits.assign(predicted=diag.splits["predicted"].map(json.dumps) if len(diag.splits) else None).to_parquet(out / "splits.parquet", index=False)
    if len(diag.merges):
        mg = diag.merges.assign(events=diag.merges["events"].map(json.dumps),
                                reference_events_in_picks=diag.merges["reference_events_in_picks"].map(json.dumps))
    else:
        mg = diag.merges
    mg.to_parquet(out / "merges.parquet", index=False)
    (out / "diagnostics.json").write_text(json.dumps(diag.to_dict(), indent=2, default=str) + "\n")
    for name, frame in result["tables"].items():
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)
    (out / "run.json").write_text(json.dumps(result["meta"], indent=2, sort_keys=True, default=str) + "\n")
    result["out_dir"] = out
    return out


def format_tables(tables: dict) -> str:
    parts = []
    for name in ("summary", "by_magnitude", "by_hour", "by_day"):
        parts.append(f"-- {name}")
        parts.append(tables[name].round(3).to_string(index=False))
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", help="35A pick store parquet (picks.parquet of a heldout_testset_score run)")
    ap.add_argument("--stations", help="stations.csv (station, lat, lon, elev_m); default from --sequence")
    ap.add_argument("--catalog", help="catalog.parquet (event, origin, lat, lon, depth_km, mag); default from --sequence")
    ap.add_argument("--windows", help="windows.csv (t0, t1); default from --sequence; without it no time claim is supported")
    ap.add_argument("--config", help="configs/association/<region>.json or a path", default=None)
    ap.add_argument("--sequence", default=None, help="Held-out sequence key; authorises the run (regression/dev only) and supplies defaults")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--associator", choices=sorted(BACKENDS), default="pyocto")
    ap.add_argument("--mainshock", default=None, help="ISO UTC origin of the mainshock; default from the registry for --sequence")
    ap.add_argument("--tol-time-s", type=float, default=MATCH_TOL_TIME_S)
    ap.add_argument("--tol-km", type=float, default=MATCH_TOL_KM)
    ap.add_argument("--tol-depth-km", type=float, default=MATCH_TOL_DEPTH_KM, help="<= 0 disables the depth tolerance")
    ap.add_argument("--all-stations", action="store_true", help="Keep station rows without picks (default: restrict to stations with picks)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--write-default-configs", default=None, metavar="DIR", help="Write the proposed regime configs and exit")
    ap.add_argument("--rehash", default=None, metavar="FILE", help="Rewrite a config file's sha256 after an edit and exit")
    a = ap.parse_args(argv)

    if a.write_default_configs:
        for p in write_default_configs(a.write_default_configs):
            print("wrote", p)
        return None
    if a.rehash:
        cfg = AssociatorConfig.load(a.rehash, verify=False)
        cfg.save(a.rehash)
        print("rewrote", a.rehash, cfg.sha256)
        return None

    key, mainshock = a.sequence, a.mainshock
    stations_path, catalog_path, windows_path, picks_path = a.stations, a.catalog, a.windows, a.picks
    try:
        if key is not None:
            import evaluation_policy as policy
            policy.authorize_scoring([key])
            import heldout_testset_registry as reg
            entry = reg.BY_KEY[key]
            stations_path = stations_path or HELDOUT_ROOT / key / "stations.csv"
            catalog_path = catalog_path or HELDOUT_ROOT / key / "catalog.parquet"
            windows_path = windows_path or HELDOUT_ROOT / key / "windows.csv"
            mainshock = mainshock or entry.get("mainshock")
            if a.config is None:
                a.config = str(config_path(entry["regime"]))
        if picks_path is None or stations_path is None or catalog_path is None:
            raise ValueError("--picks, --stations and --catalog are required unless --sequence supplies them")
        if a.config is None:
            raise ValueError("--config is required (configs/association/<region>.json)")
        config = load_config(a.config)
    except (PermissionError, ValueError, KeyError) as exc:
        ap.error(str(exc))
    picks = pd.read_parquet(picks_path)
    pick_keys = sorted(picks["key"].dropna().astype(str).unique()) if "key" in picks else []
    try:
        if pick_keys:
            import evaluation_policy as policy
            policy.authorize_scoring(pick_keys)
        if key is None and len(pick_keys) == 1:
            key = pick_keys[0]
        elif pick_keys and key is not None and pick_keys != [key]:
            raise ValueError(f"pick store keys {pick_keys} differ from --sequence {key}")
    except (PermissionError, ValueError) as exc:
        ap.error(str(exc))
    stations = pd.read_csv(stations_path)
    catalog = pd.read_parquet(catalog_path) if str(catalog_path).endswith(".parquet") else pd.read_csv(catalog_path)
    windows = pd.read_csv(windows_path) if windows_path is not None else None
    tol_depth = a.tol_depth_km if a.tol_depth_km and a.tol_depth_km > 0 else None
    sources = dict(picks=picks_path, stations=stations_path, catalog=catalog_path, config=a.config)
    if windows_path is not None:
        sources["windows"] = windows_path
    try:
        res = run(picks, stations, catalog, config, model_id=a.model_id, threshold=a.threshold, windows=windows,
                  mainshock_time=mainshock, key=key, backend=a.associator, tol_time_s=a.tol_time_s, tol_km=a.tol_km,
                  tol_depth_km=tol_depth, restrict_stations=not a.all_stations, out_dir=a.out_dir, sources=sources)
    except (ValueError, ImportError) as exc:
        ap.error(str(exc))
    m = res["meta"]
    print(f"config {m['config_sha256'][:12]} ({config.region} v{config.version}), associator {m['associator']}, "
          f"model {m['model_id']}, threshold {m['threshold']}, access {m['access_id']}")
    print(f"picks {m['n_picks']}, events {m['n_events']}, reference {m['n_reference']}, matched {m['n_matched']}, "
          f"splits {m['n_splits']}, merges {m['n_merges']}")
    print(format_tables(res["tables"]))
    if res.get("out_dir") is not None:
        print("wrote", res["out_dir"])
    return res


if __name__ == "__main__":
    main()
