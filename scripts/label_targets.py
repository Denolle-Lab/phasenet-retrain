"""Label validity policy and target/mask builder (#41A).

Two policies:

* `legacy`  reproduces the v7 target formula exactly: Gaussian P and S from
  every listed arrival, N = clip(1 - max(P, S)), every sample supervised.
  It exists so the alignment diagnostic (46A) can hold the target formula
  fixed while the loader changes.
* `masked`  supervises only what the label provenance supports. Gaussian
  targets come from `manual` and `reviewed` arrivals; the PSN triple is
  renormalised to sum to one; the loss mask is 1 where the absence or
  presence of a phase is known and 0 elsewhere:
    - negative support `certified` or `reviewed`: every sample supervised;
    - negative support `unknown`: only ±`supervised_halfwidth_s` around a
      supervising arrival;
    - `automatic` and `unknown`-tier arrivals mask ±`unknown_halfwidth_s`,
      except the ±`positive_core_sigmas`·σ core of a supervising arrival,
      which is always supervised;
    - `unknown_intervals` (gaps, masked artefacts) and padding beyond
      `valid_samples` are never supervised.

The mask multiplies the per-sample soft cross-entropy; the distillation
term, when used, is not masked. See docs/2026-09-11_41a_label_policy.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arrivals import NEGATIVE_SUPPORT, PHASES, WindowSample

POLICIES = ("legacy", "masked")
SIGMA_S = 0.10  # seconds; ten samples at 100 Hz, the parent's label width


@dataclass(frozen=True)
class LabelPolicy:
    name: str = "masked"
    sigma_s: float = SIGMA_S
    supervised_halfwidth_s: float = 0.5
    unknown_halfwidth_s: float = 1.0
    positive_core_sigmas: float = 3.0

    def __post_init__(self):
        if self.name not in POLICIES:
            raise ValueError(f"Unknown label policy {self.name!r}; expected one of {POLICIES}")
        for key in ("sigma_s", "supervised_halfwidth_s", "unknown_halfwidth_s", "positive_core_sigmas"):
            value = getattr(self, key)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"LabelPolicy.{key} must be positive and finite")

    @classmethod
    def from_config(cls, cfg) -> "LabelPolicy":
        """`None` or missing -> legacy (unchanged behaviour); a string names
        the policy; a dict may override any field."""
        if cfg is None:
            return LEGACY
        if isinstance(cfg, str):
            return cls(name=cfg)
        if isinstance(cfg, dict):
            return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})
        raise TypeError("label policy config must be None, a string or a dict")


LEGACY = LabelPolicy(name="legacy")
MASKED = LabelPolicy(name="masked")


@dataclass
class Targets:
    targets: np.ndarray            # (3, n) float32, PSN
    mask: np.ndarray               # (n,) float32 in {0, 1}
    info: dict = field(default_factory=dict)

    @property
    def n_supervised(self) -> int:
        return int(self.mask.sum())


def gaussian(n_samples: int, centre_sample: float, sigma_samples: float) -> np.ndarray:
    x = np.arange(n_samples, dtype=np.float32)
    return np.exp(-((x - np.float32(centre_sample)) ** 2) / np.float32(2 * sigma_samples ** 2)).astype(np.float32)


def _set_interval(mask: np.ndarray, centre_sample: float, halfwidth_samples: float, value: float) -> None:
    lo = int(np.floor(centre_sample - halfwidth_samples))
    hi = int(np.ceil(centre_sample + halfwidth_samples))
    lo, hi = max(lo, 0), min(hi, mask.shape[0] - 1)
    if hi >= lo:
        mask[lo:hi + 1] = value


def build_targets(arrivals, n_samples: int, rate_hz: float, *, negative_support: str = "unknown",
                  unknown_intervals=(), valid_samples=None, policy: LabelPolicy = MASKED) -> Targets:
    """Targets and loss mask for one window. Arrival times are seconds from
    sample 0 of the window; arrivals outside [0, n) are reported, not used."""
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not (np.isfinite(rate_hz) and rate_hz > 0):
        raise ValueError("rate_hz must be positive and finite")
    if negative_support not in NEGATIVE_SUPPORT:
        raise ValueError(f"negative_support must be one of {NEGATIVE_SUPPORT}")
    if valid_samples is None:
        valid_samples = n_samples
    valid_samples = int(valid_samples)
    if not 0 <= valid_samples <= n_samples:
        raise ValueError("valid_samples outside the window")

    sigma = policy.sigma_s * rate_hz
    channels = {"P": np.zeros(n_samples, np.float32), "S": np.zeros(n_samples, np.float32)}
    used, outside, ignored = {"P": 0, "S": 0}, [], []
    supervising, non_supervising = [], []
    for a in arrivals:
        centre = a.time_s * rate_hz
        contributes = policy.name == "legacy" or a.supervises
        if not (0.0 <= centre < n_samples):
            outside.append(a)
            continue
        if contributes:
            channels[a.phase] = np.maximum(channels[a.phase], gaussian(n_samples, centre, sigma))
            used[a.phase] += 1
            supervising.append((a, centre))
        else:
            ignored.append(a)
            non_supervising.append((a, centre))

    p, s = channels["P"], channels["S"]
    noise = np.clip(1.0 - np.maximum(p, s), 0.0, 1.0).astype(np.float32)
    targets = np.stack([p, s, noise]).astype(np.float32)

    if policy.name == "legacy":
        mask = np.ones(n_samples, np.float32)
    else:
        total = targets.sum(axis=0, keepdims=True)
        targets = (targets / np.where(total > 0, total, 1.0)).astype(np.float32)
        if negative_support in ("certified", "reviewed"):
            mask = np.ones(n_samples, np.float32)
        else:
            mask = np.zeros(n_samples, np.float32)
            for _, centre in supervising:
                _set_interval(mask, centre, policy.supervised_halfwidth_s * rate_hz, 1.0)
        for _, centre in non_supervising:
            _set_interval(mask, centre, policy.unknown_halfwidth_s * rate_hz, 0.0)
        for _, centre in supervising:
            _set_interval(mask, centre, policy.positive_core_sigmas * sigma, 1.0)
        for start_s, end_s in unknown_intervals:
            lo = max(int(np.floor(start_s * rate_hz)), 0)
            hi = min(int(np.ceil(end_s * rate_hz)), n_samples)
            if hi > lo:
                mask[lo:hi] = 0.0
        if valid_samples < n_samples:
            mask[valid_samples:] = 0.0

    info = dict(
        policy=policy.name, sigma_samples=float(sigma), n_supervised=int(mask.sum()),
        supervised_fraction=float(mask.mean()), used=used,
        n_outside=len(outside), n_ignored_tier=len(ignored),
        outside=[a.to_dict() for a in outside], ignored=[a.to_dict() for a in ignored],
        negative_support=negative_support,
    )
    return Targets(targets, mask, info)


def targets_for(sample: WindowSample, policy: LabelPolicy = MASKED) -> Targets:
    return build_targets(sample.arrivals, sample.n_samples, sample.rate_hz,
                         negative_support=sample.negative_support,
                         unknown_intervals=sample.unknown_intervals,
                         valid_samples=sample.valid_samples, policy=policy)


def has_supervision(sample: WindowSample, policy: LabelPolicy = MASKED) -> bool:
    """False when a window would contribute nothing to the loss: no
    supervising arrival and unknown negative support. Loaders reject such
    signal rows instead of serving them."""
    if policy.name == "legacy":
        return True
    if sample.negative_support in ("certified", "reviewed"):
        return True
    return any(0.0 <= a.time_s < sample.duration_s for a in sample.supervised_arrivals())
