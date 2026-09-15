"""Arrival lists, provenance tiers and window samples (#41A).

Shared by the label policy (`label_targets.py`), the augmentation transforms
(#43A) and the manifest loader. Pure numpy; no torch, no SeisBench.

Conventions
-----------
* Times are seconds from the first sample of the waveform they belong to:
  the trace origin before cropping, the crop origin after. Rates are Hz.
* A phase arrival carries a provenance tier. Only `manual` and `reviewed`
  arrivals supervise the loss; `automatic` and `unknown` arrivals mask their
  neighbourhood instead (see `label_targets.LabelPolicy`).
* `negative_support` says where the absence of a phase is known:
  `certified` (catalogue-complete window certified event-free outside the
  listed arrivals), `reviewed` (a reviewed negative window) or `unknown`
  (a bulletin pick list is not proof that nothing else arrived).
* `unknown_intervals` are [start, end) seconds that carry no information
  (gaps, padding, masked artefacts); they are never supervised.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np

PHASES = ("P", "S")
TIERS = ("manual", "reviewed", "automatic", "unknown")
NEGATIVE_SUPPORT = ("certified", "reviewed", "unknown")

# Manifest columns carrying the schema (see docs/2026-09-11_41a_label_policy.md).
ARRIVALS_COLUMN = "arrivals_json"
NEGATIVE_SUPPORT_COLUMN = "negative_support"
SCHEMA_VERSION = "41a-v1"


@dataclass(frozen=True)
class Arrival:
    phase: str
    time_s: float
    tier: str = "manual"
    event_id: str | None = None
    uncertainty_s: float | None = None

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"Unknown phase {self.phase!r}; expected one of {PHASES}")
        if self.tier not in TIERS:
            raise ValueError(f"Unknown provenance tier {self.tier!r}; expected one of {TIERS}")
        if not np.isfinite(float(self.time_s)):
            raise ValueError("Arrival time must be finite")
        if self.uncertainty_s is not None and not (np.isfinite(self.uncertainty_s) and self.uncertainty_s >= 0):
            raise ValueError("Arrival uncertainty must be a finite non-negative number of seconds")

    @property
    def supervises(self) -> bool:
        return self.tier in ("manual", "reviewed")

    def shifted(self, dt_s: float) -> "Arrival":
        return replace(self, time_s=float(self.time_s) + float(dt_s))

    def to_dict(self) -> dict:
        out = {"phase": self.phase, "time_s": float(self.time_s), "tier": self.tier}
        if self.event_id is not None:
            out["event_id"] = str(self.event_id)
        if self.uncertainty_s is not None:
            out["uncertainty_s"] = float(self.uncertainty_s)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Arrival":
        return cls(phase=d["phase"], time_s=float(d["time_s"]), tier=d.get("tier", "manual"),
                   event_id=d.get("event_id"), uncertainty_s=d.get("uncertainty_s"))


def arrivals_to_json(arrivals) -> str:
    """Serialise for the `arrivals_json` manifest column. Sorted by time."""
    items = sorted((a.to_dict() for a in arrivals), key=lambda d: (d["time_s"], d["phase"]))
    return json.dumps(items, separators=(",", ":"), sort_keys=True)


def arrivals_from_json(text) -> list:
    """Parse the `arrivals_json` column. Empty, None or NaN gives an empty list."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    text = str(text).strip()
    if not text:
        return []
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError("arrivals_json must hold a JSON list")
    return [Arrival.from_dict(d) for d in items]


def arrivals_from_columns(p_sample, s_sample, rate_hz: float, tier: str = "manual",
                          event_id=None) -> list:
    """Legacy manifests: one P and one S index on the `rate_hz` grid."""
    if not (np.isfinite(rate_hz) and rate_hz > 0):
        raise ValueError("rate_hz must be positive and finite")
    out = []
    for phase, value in (("P", p_sample), ("S", s_sample)):
        if value is None:
            continue
        value = float(value)
        if np.isnan(value):
            continue
        out.append(Arrival(phase, value / float(rate_hz), tier=tier, event_id=event_id))
    return out


def shift_arrivals(arrivals, dt_s: float) -> list:
    return [a.shifted(dt_s) for a in arrivals]


def arrivals_inside(arrivals, duration_s: float) -> list:
    """Arrivals with 0 <= time < duration_s."""
    return [a for a in arrivals if 0.0 <= a.time_s < duration_s]


@dataclass
class WindowSample:
    """A waveform with everything the label policy and augmentation need.

    waveform         (3, n) float32, ZNE order, channel-wise.
    rate_hz          sampling rate of `waveform`.
    arrivals         list of Arrival, times relative to sample 0.
    component_mask   which of Z, N, E are real (False = zero-filled).
    negative_support one of NEGATIVE_SUPPORT.
    unknown_intervals [start_s, end_s) intervals carrying no information.
    valid_samples    number of leading samples with real support (None = all).
    meta             free provenance (dataset, trace_name, snr_db, ...).
    """

    waveform: np.ndarray
    rate_hz: float
    arrivals: list = field(default_factory=list)
    component_mask: tuple = (True, True, True)
    negative_support: str = "unknown"
    unknown_intervals: list = field(default_factory=list)
    valid_samples: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.waveform = np.asarray(self.waveform, dtype=np.float32)
        if self.waveform.ndim != 2 or self.waveform.shape[0] != 3:
            raise ValueError(f"waveform must be (3, n); got {self.waveform.shape}")
        if not (np.isfinite(self.rate_hz) and self.rate_hz > 0):
            raise ValueError("rate_hz must be positive and finite")
        if self.negative_support not in NEGATIVE_SUPPORT:
            raise ValueError(f"negative_support must be one of {NEGATIVE_SUPPORT}")
        self.component_mask = tuple(bool(x) for x in self.component_mask)
        if len(self.component_mask) != 3:
            raise ValueError("component_mask must have three entries (Z, N, E)")
        if self.valid_samples is not None:
            self.valid_samples = int(self.valid_samples)
            if not 0 <= self.valid_samples <= self.n_samples:
                raise ValueError("valid_samples outside the waveform length")
        self.unknown_intervals = [(float(a), float(b)) for a, b in self.unknown_intervals]
        for a, b in self.unknown_intervals:
            if not (np.isfinite(a) and np.isfinite(b) and b > a):
                raise ValueError(f"Invalid unknown interval {(a, b)}")
        self.arrivals = list(self.arrivals)

    @property
    def n_samples(self) -> int:
        return int(self.waveform.shape[-1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate_hz

    @property
    def n_valid(self) -> int:
        return self.n_samples if self.valid_samples is None else self.valid_samples

    def copy(self) -> "WindowSample":
        return WindowSample(self.waveform.copy(), self.rate_hz, list(self.arrivals), self.component_mask,
                            self.negative_support, list(self.unknown_intervals), self.valid_samples,
                            dict(self.meta))

    def supervised_arrivals(self) -> list:
        return [a for a in self.arrivals if a.supervises]
