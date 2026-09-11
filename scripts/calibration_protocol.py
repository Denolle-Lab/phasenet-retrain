"""Operating-point calibration protocol on independent station-days (#38A).

Numbers here are the pre-registered proposal of
docs/2026-09-11_38a_calibration_protocol.md. They are frozen when 38A is
checked off; candidates are evaluated only after that. Pure pandas/numpy.

Vocabulary
----------
station-day      one station, one UTC day, one instrument epoch.
stratum          (region_class, instrument_class, season, condition).
condition        "quiet" or "disturbed", from the station's own 1–20 Hz
                 power percentile over the calibration year (below the 30th
                 percentile is quiet, above the 70th is disturbed; the middle
                 is not used for calibration).
unmatched-pick rate  emitted picks per station-day with no catalogue arrival
                 within MATCH_TOL_S at that station. Not a false-positive
                 rate: a reviewed sample of unmatched picks estimates the
                 fraction that are real events absent from the catalogue.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "38a-v1"
MATCH_TOL_S = 0.5

REGION_CLASSES = ("dense_local", "sparse_regional", "island_coastal", "volcanic", "polar")
INSTRUMENT_CLASSES = ("broadband", "short_period", "strong_motion", "geophone_lowcost")
SEASONS = ("DJF", "MAM", "JJA", "SON")
CONDITIONS = ("quiet", "disturbed")

# Station-day roles are exclusive; a station-day used anywhere else is not a
# calibration day (training, mining, development, sealed acceptance).
ROLES = ("calibration", "training", "mining", "development", "sealed", "unassigned")


@dataclass(frozen=True)
class Budget:
    """Nuisance-pick budget per phase in unmatched picks per station-day on
    quiet days, with the sensitivity values reported alongside."""
    p_per_station_day: float = 50.0
    s_per_station_day: float = 50.0
    sensitivity: tuple = (20.0, 100.0)
    ci_level: float = 0.95
    max_relative_halfwidth: float = 0.20     # of the budget, on the rate estimate
    min_stations_per_stratum: int = 20
    min_days_per_station: int = 3
    reviewed_sample_per_weight: int = 200    # unmatched picks reviewed per weight and region


@dataclass
class Protocol:
    version: str = PROTOCOL_VERSION
    budget: Budget = field(default_factory=Budget)
    thresholds: tuple = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))
    match_tol_s: float = MATCH_TOL_S

    def to_dict(self) -> dict:
        return dict(version=self.version, budget=self.budget.__dict__, thresholds=list(map(float, self.thresholds)),
                    match_tol_s=self.match_tol_s)


def station_day_key(station: str, day: str, epoch: str = "") -> str:
    """Deterministic identity of a station-day; `day` is YYYY-MM-DD (UTC)."""
    return f"{station}|{pd.Timestamp(day).strftime('%Y-%m-%d')}|{epoch}"


def calibration_hash(key: str) -> float:
    """Uniform in [0, 1) from the station-day key; used to draw calibration
    days without reference to any model output."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def condition_from_percentile(pct: float) -> str | None:
    if pct < 30:
        return "quiet"
    if pct > 70:
        return "disturbed"
    return None


def select_calibration_days(availability: pd.DataFrame, fraction: float = 0.15, seed_tag: str = "38a",
                            excluded_keys=()) -> pd.DataFrame:
    """Draw calibration station-days from an availability table.

    availability columns: station, day, epoch (optional), region_class,
    instrument_class, season, power_percentile, role. Only rows with role
    "unassigned" are eligible; rows whose key is in `excluded_keys` (held-out
    places and years, development and sealed windows) are dropped first. The
    draw is by hash so it is reproducible and independent of any model.
    Returns the selected rows with `condition` and `key` columns.
    """
    required = {"station", "day", "region_class", "instrument_class", "season", "power_percentile", "role"}
    missing = required - set(availability.columns)
    if missing:
        raise ValueError(f"availability table lacks columns {sorted(missing)}")
    df = availability.copy()
    df["epoch"] = df["epoch"].fillna("") if "epoch" in df else ""
    df["key"] = [station_day_key(s, d, e) for s, d, e in zip(df.station, df.day, df.epoch)]
    df = df[~df["key"].isin(set(excluded_keys))]
    df = df[df["role"] == "unassigned"]
    for column, allowed in (("region_class", REGION_CLASSES), ("instrument_class", INSTRUMENT_CLASSES), ("season", SEASONS)):
        bad = set(df[column]) - set(allowed)
        if bad:
            raise ValueError(f"Unknown {column} values {sorted(bad)}")
    df["condition"] = df["power_percentile"].map(condition_from_percentile)
    df = df[df["condition"].notna()]
    draw = df["key"].map(lambda k: calibration_hash(f"{seed_tag}|{k}"))
    out = df[draw < fraction].copy()
    out["role"] = "calibration"
    return out.reset_index(drop=True)


def exposure_check(days: pd.DataFrame, budget: Budget = Budget()) -> pd.DataFrame:
    """Per stratum: stations, station-days, and whether the minimum exposure
    holds. A stratum below the minimum is reported `pooled_with` its nearest
    region class rather than silently passing."""
    rows = []
    for keys, g in days.groupby(["region_class", "instrument_class", "season", "condition"]):
        n_sta = g["station"].nunique()
        per_sta = g.groupby("station").size()
        ok = n_sta >= budget.min_stations_per_stratum and (per_sta >= budget.min_days_per_station).sum() >= budget.min_stations_per_stratum
        rows.append(dict(zip(["region_class", "instrument_class", "season", "condition"], keys),
                         n_stations=int(n_sta), n_station_days=int(len(g)), exposure_ok=bool(ok)))
    return pd.DataFrame(rows)


def unmatched_rate(picks: pd.DataFrame, exposure_days: float) -> float:
    """Unmatched picks per station-day over the given exposure."""
    if exposure_days <= 0:
        raise ValueError("exposure must be positive")
    return float((~picks["matched"]).sum() / exposure_days)


def block_bootstrap_rate(per_day: pd.DataFrame, n_boot: int = 2000, ci=0.95, seed: int = 0) -> tuple:
    """Rate and percentile interval with station-day blocks resampled
    (columns: key, unmatched). Stations, not picks, are the unit."""
    if per_day.empty:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    counts = per_day["unmatched"].to_numpy(dtype=float)
    rate = counts.mean()
    boots = rng.choice(counts, size=(n_boot, counts.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(rate), (float(lo), float(hi))


def operating_threshold(sweep: pd.DataFrame, budget_value: float, phase: str) -> dict | None:
    """Lowest threshold whose unmatched-rate upper bound is within budget.

    sweep columns: phase, threshold, rate, ci_low, ci_high (from
    `block_bootstrap_rate` on calibration days). Returns the attained row,
    or None with no threshold meeting the budget."""
    sub = sweep[sweep["phase"] == phase].sort_values("threshold")
    ok = sub[sub["ci_high"] <= budget_value]
    if ok.empty:
        return None
    row = ok.iloc[0]
    return dict(phase=phase, threshold=float(row["threshold"]), rate=float(row["rate"]),
                ci_low=float(row["ci_low"]), ci_high=float(row["ci_high"]), budget=float(budget_value))


def precision_ok(rate: float, ci: tuple, budget_value: float, budget: Budget = Budget()) -> bool:
    """The interval half-width must not exceed the allowed fraction of the budget."""
    return (ci[1] - ci[0]) / 2 <= budget.max_relative_halfwidth * budget_value
