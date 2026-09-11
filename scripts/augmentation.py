"""Label-consistent augmentation on the 41A sample contract (#43A).

Every transform is a callable ``(sample, rng) -> WindowSample`` that returns a
new object (the input is never mutated) and moves arrivals, unknown intervals,
valid support and the component mask together with the waveform. Pure
numpy/scipy, no torch. Every rate change goes through
``waveform_contract.resample_waveform``, the loader's polyphase resampler.

Provider protocols, implemented by the caller (reference implementations
``ArrayNoiseProvider`` and ``ListSampleProvider`` below):

* ``NoiseProvider.draw(rng, n_samples, rate_hz)`` returns
  ``(waveform (3, n_samples) float32 ZNE at rate_hz, class_name)``.
  Class-balanced sampling and resampling of stored noise to ``rate_hz`` are
  the provider's job; the transform only scales and adds.
* ``SampleProvider.draw(rng, n_samples, rate_hz)`` returns a ``WindowSample``
  (any length; resampled here if its rate differs) for event superposition.

SNR (``NoiseSuperposition``, ``NonStationaryNoise``): mean square of the real
channels of the source window in [tP - 0.5 s, tP + 2.0 s] over the mean
square of the added noise in the same interval; the S window when there is no
P inside the valid support; the whole valid window when there is neither.
The "signal" is the stored window with its own background, so the realised
SNR is relative to the source window, not to a clean signal; the source's own
SNR, when known, stays in ``meta``.

Transforms record what they did in ``meta["augmentations"]`` (a list of
dicts, one per applied transform). See
docs/2026-09-11_43a_augmentation_contract.md for the contract and the config.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

import numpy as np
from scipy.signal import butter, sosfiltfilt

from arrivals import PHASES, WindowSample
from label_targets import MASKED, LabelPolicy, targets_for
from waveform_contract import resample_waveform

SUPPORT_RANK = {"certified": 2, "reviewed": 1, "unknown": 0}
DEFAULT_SNR_BINS = ((0.4, 0.0, 5.0), (0.3, 5.0, 10.0), (0.3, 10.0, 25.0))
SNR_PRE_S, SNR_POST_S = 0.5, 2.0
CONFIG_SECTION = "augmentation_v3"


class NoiseProvider(Protocol):
    def draw(self, rng: np.random.Generator, n_samples: int, rate_hz: float) -> tuple: ...


class SampleProvider(Protocol):
    def draw(self, rng: np.random.Generator, n_samples: int, rate_hz: float) -> WindowSample: ...


# --------------------------------------------------------------------------- helpers

def _pair(value, name):
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, IndexError, ValueError):
        raise ValueError(f"{name} must be a (low, high) pair; got {value!r}")
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"{name} must satisfy low <= high; got {value!r}")
    return lo, hi


def _prob(value):
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must lie in [0, 1]; got {value!r}")
    return p


def _snr_bins(bins):
    out = []
    for item in bins:
        weight, lo, hi = float(item[0]), float(item[1]), float(item[2])
        if weight < 0 or not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
            raise ValueError(f"SNR bin must be (weight >= 0, low_db <= high_db); got {item!r}")
        out.append((weight, lo, hi))
    if not out or sum(w for w, _, _ in out) <= 0:
        raise ValueError("SNR bins need positive total weight")
    return tuple(out)


def real_channels(sample: WindowSample) -> np.ndarray:
    return np.asarray(sample.component_mask, dtype=bool)


def reference_rms(sample: WindowSample) -> float:
    """RMS over the real channels within the valid support."""
    real, n_valid = real_channels(sample), sample.n_valid
    if not real.any() or n_valid == 0:
        return 0.0
    seg = sample.waveform[real, :n_valid].astype(np.float64)
    return float(np.sqrt(np.mean(seg * seg)))


def reference_peak(sample: WindowSample) -> float:
    """Largest absolute amplitude over the real channels within the valid support."""
    real, n_valid = real_channels(sample), sample.n_valid
    if not real.any() or n_valid == 0:
        return 0.0
    return float(np.abs(sample.waveform[real, :n_valid]).max())


def clip_intervals(intervals, dt_s: float, duration_s: float) -> list:
    """Shift [start, end) intervals by dt_s and clip them to [0, duration_s)."""
    out = []
    for a, b in intervals:
        a, b = max(float(a) + dt_s, 0.0), min(float(b) + dt_s, float(duration_s))
        if b > a:
            out.append((a, b))
    return out


def merge_support(a: str, b: str) -> str:
    """Weakest negative support wins: unknown over reviewed over certified."""
    return min((a, b), key=SUPPORT_RANK.__getitem__)


def anchor_time(sample: WindowSample):
    """Earliest supervising arrival inside the valid support, else the earliest
    arrival of any tier, else None."""
    dur = sample.n_valid / sample.rate_hz
    for pool in (sample.supervised_arrivals(), sample.arrivals):
        times = [a.time_s for a in pool if 0.0 <= a.time_s < dur]
        if times:
            return float(min(times))
    return None


def snr_window(sample: WindowSample, pre_s: float = SNR_PRE_S, post_s: float = SNR_POST_S):
    """[tP - pre_s, tP + post_s) clipped to the valid support; the S window
    when there is no P; the whole valid window otherwise. Supervising
    arrivals are preferred over automatic ones. Returns (lo_s, hi_s, kind)."""
    dur = sample.n_valid / sample.rate_hz
    for phase in PHASES:
        for pool in (sample.supervised_arrivals(), sample.arrivals):
            times = [a.time_s for a in pool if a.phase == phase and 0.0 <= a.time_s < dur]
            if times:
                t = min(times)
                lo, hi = max(t - pre_s, 0.0), min(t + post_s, dur)
                if hi > lo:
                    return float(lo), float(hi), phase
    return 0.0, float(dur), "whole"


def window_power(waveform: np.ndarray, real: np.ndarray, lo_s: float, hi_s: float, rate_hz: float) -> float:
    """Mean square over the real channels in [lo_s, hi_s)."""
    i0 = max(int(math.floor(lo_s * rate_hz)), 0)
    i1 = min(int(math.ceil(hi_s * rate_hz)), waveform.shape[-1])
    if i1 <= i0 or not real.any():
        return 0.0
    seg = waveform[real, i0:i1].astype(np.float64)
    return float(np.mean(seg * seg))


def draw_snr_db(rng: np.random.Generator, bins=DEFAULT_SNR_BINS):
    """Draw a bin by weight, then uniformly inside it. Returns (snr_db, bin index)."""
    bins = _snr_bins(bins)
    weights = np.array([w for w, _, _ in bins], dtype=np.float64)
    index = int(rng.choice(len(bins), p=weights / weights.sum()))
    _, lo, hi = bins[index]
    return float(rng.uniform(lo, hi)), index


def _record(target: WindowSample, name: str, **info) -> None:
    """Append a provenance entry without touching lists shared with the input."""
    entry = {"name": name}
    entry.update(info)
    target.meta["augmentations"] = list(target.meta.get("augmentations", ())) + [entry]


def resample_sample(sample: WindowSample, target_rate: float) -> WindowSample:
    """Resample the valid support through the loader's resampler. Arrival and
    interval seconds are unchanged; padding is dropped (the output is all valid)."""
    if float(target_rate) == float(sample.rate_hz):
        return sample.copy()
    n_valid = sample.n_valid
    if n_valid < 2:
        raise ValueError("At least two valid samples are required for resampling")
    wave = resample_waveform(sample.waveform[:, :n_valid], sample.rate_hz, target_rate)
    meta = dict(sample.meta)
    meta["resampled_from_hz"] = float(sample.rate_hz)
    return WindowSample(wave, float(target_rate), list(sample.arrivals), sample.component_mask,
                        sample.negative_support, list(sample.unknown_intervals), None, meta)


# --------------------------------------------------------------------------- providers

class ArrayNoiseProvider:
    """Reference NoiseProvider over in-memory pools.

    pools: {class_name: [(3, m) arrays, ...]} all at `rate_hz` (ZNE order).
    draw() picks a class uniformly (class balance), a member uniformly, resamples
    it to the requested rate when the rates differ, and returns a random
    contiguous segment of n_samples. Members shorter than the request raise.
    """

    def __init__(self, pools: dict, rate_hz: float):
        if not pools:
            raise ValueError("ArrayNoiseProvider needs at least one class")
        self.rate_hz = float(rate_hz)
        self.pools = {}
        for name, members in pools.items():
            members = [np.asarray(m, dtype=np.float32) for m in members]
            if not members or any(m.ndim != 2 or m.shape[0] != 3 for m in members):
                raise ValueError(f"noise class {name!r} needs (3, m) members")
            self.pools[str(name)] = members
        self.classes = tuple(sorted(self.pools))

    def draw(self, rng, n_samples, rate_hz):
        name = self.classes[int(rng.integers(len(self.classes)))]
        members = self.pools[name]
        member = members[int(rng.integers(len(members)))]
        if float(rate_hz) != self.rate_hz:
            member = resample_waveform(member, self.rate_hz, rate_hz)
        if member.shape[-1] < n_samples:
            raise ValueError(f"noise member of class {name!r} has {member.shape[-1]} samples at "
                             f"{rate_hz} Hz; {n_samples} requested")
        start = int(rng.integers(0, member.shape[-1] - n_samples + 1))
        return np.array(member[:, start:start + n_samples], dtype=np.float32), name


class ListSampleProvider:
    """Reference SampleProvider: uniform draw from a list of WindowSamples,
    resampled to the requested rate when needed. n_samples is ignored; the
    superposition offset decides which part of the drawn window is used."""

    def __init__(self, samples):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("ListSampleProvider needs at least one sample")

    def draw(self, rng, n_samples, rate_hz):
        sample = self.samples[int(rng.integers(len(self.samples)))]
        return resample_sample(sample, rate_hz)


# --------------------------------------------------------------------------- base

class Transform:
    """Callable (sample, rng) -> new WindowSample. `prob` gates `apply`;
    a gated-off draw returns a copy with no record."""

    name = "transform"

    def __init__(self, prob: float = 1.0):
        self.prob = _prob(prob)

    def __call__(self, sample: WindowSample, rng: np.random.Generator) -> WindowSample:
        if self.prob < 1.0 and rng.random() >= self.prob:
            return sample.copy()
        return self.apply(sample, rng)

    def apply(self, sample: WindowSample, rng: np.random.Generator) -> WindowSample:
        raise NotImplementedError

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in sorted(vars(self).items()) if not callable(v))
        return f"{type(self).__name__}({fields})"


# --------------------------------------------------------------------------- crop

class RandomCrop(Transform):
    """Crop `window_samples` from a longer stored window.

    When a supervising arrival exists, one is chosen at random and placed at a
    fraction of the window drawn from `anchor_range`, so it is always inside;
    without supervising arrivals any arrival anchors; without arrivals the
    start is uniform. Arrivals and unknown intervals shift by the crop start;
    arrivals outside are dropped and listed in meta["dropped_arrivals"] (crop
    frame). valid_samples is recomputed; a window shorter than the crop is
    zero-padded and the padding is beyond valid_samples.
    """

    name = "crop"

    def __init__(self, window_samples: int, anchor_range=(0.1, 0.7), prob: float = 1.0):
        super().__init__(prob)
        self.window_samples = int(window_samples)
        if self.window_samples <= 0:
            raise ValueError("window_samples must be positive")
        self.anchor_range = _pair(anchor_range, "anchor_range")
        if not (0.0 <= self.anchor_range[0] and self.anchor_range[1] <= 1.0):
            raise ValueError("anchor_range must lie within [0, 1]")

    def apply(self, sample, rng):
        n, width, rate = sample.n_samples, self.window_samples, sample.rate_hz
        anchors = [a for a in (sample.supervised_arrivals() or sample.arrivals) if 0.0 <= a.time_s < sample.duration_s]
        if width >= n:
            start = 0
        elif anchors:
            anchor = anchors[int(rng.integers(len(anchors)))]
            fraction = float(rng.uniform(*self.anchor_range))
            start = int(round(anchor.time_s * rate - fraction * width))
            start = min(max(start, 0), n - width)
        else:
            start = int(rng.integers(0, n - width + 1))
        wave = np.zeros((3, width), dtype=np.float32)
        copied = min(width, n - start)
        wave[:, :copied] = sample.waveform[:, start:start + copied]
        dt, duration = -start / rate, width / rate
        kept, dropped = [], []
        for a in sample.arrivals:
            b = a.shifted(dt)
            (kept if 0.0 <= b.time_s < duration else dropped).append(b)
        valid = min(max(sample.n_valid - start, 0), width)
        meta = dict(sample.meta)
        meta["dropped_arrivals"] = list(meta.get("dropped_arrivals", ())) + [a.to_dict() for a in dropped]
        out = WindowSample(wave, rate, kept, sample.component_mask, sample.negative_support,
                           clip_intervals(sample.unknown_intervals, dt, duration),
                           None if valid == width else valid, meta)
        _record(out, self.name, start_sample=start, start_s=start / rate, window_samples=width,
                n_dropped=len(dropped), n_kept=len(kept))
        return out


# --------------------------------------------------------------------------- noise

def _draw_noise(provider, rng, n_samples, rate_hz):
    noise, class_name = provider.draw(rng, n_samples, rate_hz)
    noise = np.array(noise, dtype=np.float32, copy=True)
    if noise.shape != (3, n_samples):
        raise ValueError(f"noise provider returned shape {noise.shape}; expected {(3, n_samples)}")
    if not np.isfinite(noise).all():
        raise ValueError("noise provider returned non-finite samples")
    return noise, str(class_name)


class NoiseSuperposition(Transform):
    """Add provider noise scaled to a target SNR drawn from `snr_bins`
    ((weight, low_db, high_db), ...). Noise is zeroed on missing channels and
    beyond valid_samples. Labels and masks are unchanged; the realised SNR is
    recorded (meta["realised_snr_db"] and the augmentation entry)."""

    name = "noise"

    def __init__(self, noise_provider, prob: float = 0.6, snr_bins=DEFAULT_SNR_BINS,
                 pre_s: float = SNR_PRE_S, post_s: float = SNR_POST_S):
        super().__init__(prob)
        if noise_provider is None:
            raise ValueError("NoiseSuperposition needs a noise provider")
        self.provider = noise_provider
        self.snr_bins = _snr_bins(snr_bins)
        self.pre_s, self.post_s = float(pre_s), float(post_s)

    def apply(self, sample, rng):
        out = sample.copy()
        real, n_valid, rate = real_channels(out), out.n_valid, out.rate_hz
        noise, class_name = _draw_noise(self.provider, rng, out.n_samples, rate)
        noise[~real] = 0.0
        noise[:, n_valid:] = 0.0
        target_db, bin_index = draw_snr_db(rng, self.snr_bins)
        lo, hi, kind = snr_window(out, self.pre_s, self.post_s)
        p_signal = window_power(out.waveform, real, lo, hi, rate)
        p_noise = window_power(noise, real, lo, hi, rate)
        if p_signal <= 0.0 or p_noise <= 0.0:
            _record(out, self.name, skipped="zero signal or noise power in the SNR window",
                    noise_class=class_name, snr_window=kind)
            return out
        scale = math.sqrt(p_signal / (p_noise * 10.0 ** (target_db / 10.0)))
        out.waveform += np.float32(scale) * noise
        realised_db = 10.0 * math.log10(p_signal / (scale * scale * p_noise))
        out.meta["realised_snr_db"] = realised_db
        _record(out, self.name, noise_class=class_name, snr_target_db=target_db, snr_db=realised_db,
                snr_bin=bin_index, snr_window=kind, snr_window_s=(lo, hi), scale=scale)
        return out


class NonStationaryNoise(Transform):
    """Noise whose level changes inside the window: a linear ramp, a step, or
    a burst over part of the window (cosine tapers of `taper_s`). Noise comes
    from the provider, or is white when there is none. The target SNR is
    defined at full level in the SNR window; the realised SNR in that window
    (which the envelope may raise) is recorded. Labels are unchanged."""

    name = "nonstationary"
    MODES = ("ramp", "step", "partial")

    def __init__(self, noise_provider=None, prob: float = 0.3, snr_bins=DEFAULT_SNR_BINS,
                 taper_s: float = 0.5, min_fraction: float = 0.2, pre_s: float = SNR_PRE_S,
                 post_s: float = SNR_POST_S):
        super().__init__(prob)
        self.provider = noise_provider
        self.snr_bins = _snr_bins(snr_bins)
        self.taper_s, self.min_fraction = float(taper_s), float(min_fraction)
        self.pre_s, self.post_s = float(pre_s), float(post_s)

    def _envelope(self, mode, n_valid, rate, rng):
        env = np.zeros(n_valid, dtype=np.float32)
        if mode == "ramp":
            rising = bool(rng.random() < 0.5)
            x = np.linspace(0.0, 1.0, n_valid, dtype=np.float32)
            env[:] = x if rising else 1.0 - x
            return env, {"direction": "up" if rising else "down"}
        if mode == "step":
            k = int(rng.integers(max(int(0.2 * n_valid), 1), max(int(0.8 * n_valid), 2)))
            rising = bool(rng.random() < 0.5)
            if rising:
                env[k:] = 1.0
            else:
                env[:k] = 1.0
            return env, {"step_s": k / rate, "direction": "up" if rising else "down"}
        length = int(rng.integers(max(int(self.min_fraction * n_valid), 1), n_valid + 1))
        k0 = int(rng.integers(0, n_valid - length + 1))
        env[k0:k0 + length] = 1.0
        taper = min(int(self.taper_s * rate), length // 2)
        if taper > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper) / taper)).astype(np.float32)
            env[k0:k0 + taper] *= ramp
            env[k0 + length - taper:k0 + length] *= ramp[::-1]
        return env, {"start_s": k0 / rate, "end_s": (k0 + length) / rate}

    def apply(self, sample, rng):
        out = sample.copy()
        real, n_valid, rate, n = real_channels(out), out.n_valid, out.rate_hz, out.n_samples
        if self.provider is not None:
            noise, class_name = _draw_noise(self.provider, rng, n, rate)
        else:
            noise, class_name = rng.standard_normal((3, n)).astype(np.float32), "white"
        noise[~real] = 0.0
        noise[:, n_valid:] = 0.0
        mode = self.MODES[int(rng.integers(len(self.MODES)))]
        target_db, bin_index = draw_snr_db(rng, self.snr_bins)
        lo, hi, kind = snr_window(out, self.pre_s, self.post_s)
        p_signal = window_power(out.waveform, real, lo, hi, rate)
        p_noise = window_power(noise, real, lo, hi, rate)
        if n_valid < 2 or p_signal <= 0.0 or p_noise <= 0.0:
            _record(out, self.name, skipped="zero signal or noise power in the SNR window", mode=mode)
            return out
        env, description = self._envelope(mode, n_valid, rate, rng)
        scale = math.sqrt(p_signal / (p_noise * 10.0 ** (target_db / 10.0)))
        shaped = np.float32(scale) * noise
        shaped[:, :n_valid] *= env[None, :]
        out.waveform += shaped
        p_added = window_power(shaped, real, lo, hi, rate)
        realised_db = 10.0 * math.log10(p_signal / p_added) if p_added > 0.0 else None
        _record(out, self.name, mode=mode, noise_class=class_name, snr_target_db=target_db,
                snr_db=realised_db, snr_bin=bin_index, snr_window=kind, snr_window_s=(lo, hi),
                envelope=description)
        return out


# --------------------------------------------------------------------------- event superposition

class EventSuperposition(Transform):
    """Add a second event window from `sample_provider`.

    The second window's anchor (its earliest supervising arrival, else earliest
    arrival) is placed at the base anchor plus an offset of magnitude drawn
    from `offset_s`, sign at random, restricted to offsets that land inside
    the base's valid support; the shift is rounded to whole samples. The
    second waveform is scaled so its peak equals `amplitude_ratio` times the
    base peak (one scalar; relative component amplitudes preserved). Merged:
    arrivals (shifted; those outside are listed as dropped; missing event_ids
    become the second window's trace_name or "superposed"), unknown intervals
    (union), negative_support (weakest wins), valid_samples (minimum of the
    base support and the end of the second's support in the base frame),
    component_mask (and; channels missing in either are zeroed).
    Targets from overlapping arrivals follow label_targets (per-channel
    maximum, then renormalisation).
    """

    name = "event_superposition"

    def __init__(self, sample_provider, prob: float = 0.3, offset_s=(2.0, 40.0), amplitude_ratio=(0.1, 1.0)):
        super().__init__(prob)
        if sample_provider is None:
            raise ValueError("EventSuperposition needs a sample provider")
        self.provider = sample_provider
        self.offset_s = _pair(offset_s, "offset_s")
        self.amplitude_ratio = _pair(amplitude_ratio, "amplitude_ratio")
        if self.offset_s[0] < 0 or self.amplitude_ratio[0] < 0:
            raise ValueError("offset_s and amplitude_ratio must be non-negative")

    def apply(self, sample, rng):
        out = sample.copy()
        rate, n, n_valid = out.rate_hz, out.n_samples, out.n_valid
        second = self.provider.draw(rng, n, rate)
        if float(second.rate_hz) != float(rate):
            second = resample_sample(second, rate)
        t_base, t_second = anchor_time(out), anchor_time(second)
        if t_base is None or t_second is None:
            _record(out, self.name, skipped="no arrival to anchor on")
            return out
        lo, hi = self.offset_s
        dur_valid = n_valid / rate
        ranges = [(+1, (lo, min(hi, dur_valid - t_base - 1.0 / rate))), (-1, (lo, min(hi, t_base)))]
        ranges = [(sign, r) for sign, r in ranges if r[1] >= r[0]]
        if not ranges:
            _record(out, self.name, skipped="no offset fits inside the valid support")
            return out
        sign, (a, b) = ranges[int(rng.integers(len(ranges)))]
        offset = sign * float(rng.uniform(a, b))
        ratio = float(rng.uniform(*self.amplitude_ratio))
        base_peak, second_peak = reference_peak(out), reference_peak(second)
        if base_peak <= 0.0 or second_peak <= 0.0:
            _record(out, self.name, skipped="zero peak amplitude")
            return out
        shift = int(round((t_base + offset - t_second) * rate))
        s0, s1 = max(0, shift), min(n, shift + second.n_valid)
        if s1 <= s0:
            _record(out, self.name, skipped="second window does not overlap the base window")
            return out
        gain = ratio * base_peak / second_peak
        real = real_channels(out) & real_channels(second)
        out.waveform[:, s0:s1] += np.float32(gain) * second.waveform[:, s0 - shift:s1 - shift]
        out.waveform[~real] = 0.0
        dt = shift / rate
        fallback_id = second.meta.get("event_id") or second.meta.get("trace_name") or "superposed"
        added, dropped = [], []
        for arrival in second.arrivals:
            b = arrival.shifted(dt)
            if b.event_id is None:
                b = replace(b, event_id=str(fallback_id))
            (added if 0.0 <= b.time_s < out.duration_s else dropped).append(b)
        out.arrivals = list(out.arrivals) + added
        out.unknown_intervals = list(out.unknown_intervals) + clip_intervals(second.unknown_intervals, dt, out.duration_s)
        out.negative_support = merge_support(out.negative_support, second.negative_support)
        valid = max(min(n_valid, shift + second.n_valid, n), 0)
        out.valid_samples = None if valid >= n else valid
        out.component_mask = tuple(bool(x) for x in real)
        out.meta["dropped_arrivals"] = list(out.meta.get("dropped_arrivals", ())) + [a.to_dict() for a in dropped]
        _record(out, self.name, offset_s=dt + t_second - t_base, offset_drawn_s=offset, shift_s=dt,
                amplitude_ratio=ratio, gain=gain, n_added=len(added), n_dropped=len(dropped),
                second_trace=second.meta.get("trace_name"), second_support=second.negative_support)
        return out


# --------------------------------------------------------------------------- artefacts

class Spike(Transform):
    """One or a few samples offset by `amplitude` times the window RMS on one
    real channel (or all, half the time). Labels unchanged."""

    name = "spike"

    def __init__(self, prob: float = 0.03, amplitude=(5.0, 50.0), width_samples=(1, 3)):
        super().__init__(prob)
        self.amplitude = _pair(amplitude, "amplitude")
        self.width_samples = tuple(int(x) for x in _pair(width_samples, "width_samples"))

    def apply(self, sample, rng):
        out = sample.copy()
        real, n_valid, ref = real_channels(out), out.n_valid, reference_rms(out)
        width = int(rng.integers(self.width_samples[0], self.width_samples[1] + 1))
        if ref <= 0.0 or n_valid < width or not real.any():
            _record(out, self.name, skipped="no support")
            return out
        i0 = int(rng.integers(0, n_valid - width + 1))
        amplitude = float(rng.uniform(*self.amplitude)) * ref * (1.0 if rng.random() < 0.5 else -1.0)
        channels = np.flatnonzero(real)
        if rng.random() < 0.5:
            channels = channels[[int(rng.integers(len(channels)))]]
        out.waveform[channels, i0:i0 + width] += np.float32(amplitude)
        _record(out, self.name, start_sample=i0, time_s=i0 / out.rate_hz, width=width, amplitude=amplitude,
                channels=[int(c) for c in channels])
        return out


class DCStep(Transform):
    """A step of `amplitude` times the window RMS on one real channel from a
    random time to the end of the valid support. Labels unchanged."""

    name = "dc_step"

    def __init__(self, prob: float = 0.03, amplitude=(0.5, 5.0)):
        super().__init__(prob)
        self.amplitude = _pair(amplitude, "amplitude")

    def apply(self, sample, rng):
        out = sample.copy()
        real, n_valid, ref = real_channels(out), out.n_valid, reference_rms(out)
        if ref <= 0.0 or n_valid < 2 or not real.any():
            _record(out, self.name, skipped="no support")
            return out
        i0 = int(rng.integers(1, n_valid))
        amplitude = float(rng.uniform(*self.amplitude)) * ref * (1.0 if rng.random() < 0.5 else -1.0)
        channel = int(np.flatnonzero(real)[int(rng.integers(int(real.sum())))])
        out.waveform[channel, i0:n_valid] += np.float32(amplitude)
        _record(out, self.name, start_sample=i0, time_s=i0 / out.rate_hz, amplitude=amplitude, channel=channel)
        return out


class Gap(Transform):
    """Zero every channel over a gap of `duration_s` and add the gap as an
    unknown interval, so targets_for masks it."""

    name = "gap"

    def __init__(self, prob: float = 0.03, duration_s=(0.1, 3.0)):
        super().__init__(prob)
        self.duration_s = _pair(duration_s, "duration_s")
        if self.duration_s[0] <= 0:
            raise ValueError("gap duration must be positive")

    def apply(self, sample, rng):
        out = sample.copy()
        n_valid, rate = out.n_valid, out.rate_hz
        if n_valid < 1:
            _record(out, self.name, skipped="no support")
            return out
        length = int(round(float(rng.uniform(*self.duration_s)) * rate))
        length = min(max(length, 1), n_valid)
        i0 = int(rng.integers(0, n_valid - length + 1))
        i1 = i0 + length
        out.waveform[:, i0:i1] = 0.0
        out.unknown_intervals = list(out.unknown_intervals) + [(i0 / rate, i1 / rate)]
        _record(out, self.name, start_s=i0 / rate, end_s=i1 / rate, start_sample=i0, end_sample=i1)
        return out


class Clip(Transform):
    """Clip every channel at `level` times the window peak (one threshold, so
    relative component amplitudes above it are lost as in a real clip)."""

    name = "clip"

    def __init__(self, prob: float = 0.03, level=(0.3, 0.9)):
        super().__init__(prob)
        self.level = _pair(level, "level")
        if not (0.0 < self.level[0] and self.level[1] <= 1.0):
            raise ValueError("clip level must lie in (0, 1]")

    def apply(self, sample, rng):
        out = sample.copy()
        peak = reference_peak(out)
        level = float(rng.uniform(*self.level))
        if peak <= 0.0:
            _record(out, self.name, skipped="zero peak amplitude")
            return out
        threshold = level * peak
        clipped = int((np.abs(out.waveform) > threshold).sum())
        np.clip(out.waveform, -threshold, threshold, out=out.waveform)
        _record(out, self.name, level=level, threshold=threshold, n_clipped=clipped)
        return out


class MainsHum(Transform):
    """A sinusoid at a mains frequency, `amplitude` times the window RMS, with
    a random phase per real channel. It is evaluated on the sample grid, so a
    frequency at or above Nyquist aliases as it would in a digitiser without
    anti-alias filtering (60 Hz at 100 Hz sampling appears at 40 Hz)."""

    name = "mains_hum"

    def __init__(self, prob: float = 0.02, freq_hz=(50.0, 60.0), amplitude=(0.05, 0.5)):
        super().__init__(prob)
        self.freq_hz = tuple(float(f) for f in freq_hz)
        if not self.freq_hz or any(f <= 0 for f in self.freq_hz):
            raise ValueError("freq_hz must list positive frequencies")
        self.amplitude = _pair(amplitude, "amplitude")

    def apply(self, sample, rng):
        out = sample.copy()
        real, n_valid, rate, ref = real_channels(out), out.n_valid, out.rate_hz, reference_rms(out)
        if ref <= 0.0 or n_valid < 1 or not real.any():
            _record(out, self.name, skipped="no support")
            return out
        freq = self.freq_hz[int(rng.integers(len(self.freq_hz)))]
        amplitude = float(rng.uniform(*self.amplitude)) * ref
        t = np.arange(n_valid, dtype=np.float64) / rate
        phases = []
        for channel in np.flatnonzero(real):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            phases.append(phase)
            out.waveform[channel, :n_valid] += (amplitude * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)
        _record(out, self.name, freq_hz=freq, amplitude=amplitude, phases=phases, aliased=bool(freq >= rate / 2))
        return out


class Drift(Transform):
    """A slow sinusoid (period drawn from `period_s`, normally longer than the
    window) of `amplitude` times the window RMS with a random phase per real
    channel: unfiltered long-period drift. Labels unchanged."""

    name = "drift"

    def __init__(self, prob: float = 0.03, amplitude=(1.0, 10.0), period_s=(60.0, 600.0)):
        super().__init__(prob)
        self.amplitude = _pair(amplitude, "amplitude")
        self.period_s = _pair(period_s, "period_s")
        if self.period_s[0] <= 0:
            raise ValueError("period_s must be positive")

    def apply(self, sample, rng):
        out = sample.copy()
        real, n_valid, rate, ref = real_channels(out), out.n_valid, out.rate_hz, reference_rms(out)
        if ref <= 0.0 or n_valid < 1 or not real.any():
            _record(out, self.name, skipped="no support")
            return out
        period = float(rng.uniform(*self.period_s))
        amplitude = float(rng.uniform(*self.amplitude)) * ref
        t = np.arange(n_valid, dtype=np.float64) / rate
        phases = []
        for channel in np.flatnonzero(real):
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            phases.append(phase)
            out.waveform[channel, :n_valid] += (amplitude * np.sin(2.0 * np.pi * t / period + phase)).astype(np.float32)
        _record(out, self.name, period_s=period, amplitude=amplitude, phases=phases)
        return out


class ChannelDrop(Transform):
    """Zero one real channel among `candidates` (Z, N, E = 0, 1, 2) and mark it
    False in component_mask. The last real channel is never dropped."""

    name = "channel_drop"

    def __init__(self, prob: float = 0.05, candidates=(0, 1, 2)):
        super().__init__(prob)
        self.candidates = tuple(int(c) for c in candidates)
        if any(c not in (0, 1, 2) for c in self.candidates):
            raise ValueError("candidates must be channel indices 0, 1, 2")

    def apply(self, sample, rng):
        out = sample.copy()
        real = real_channels(out)
        options = [c for c in self.candidates if real[c]]
        if int(real.sum()) <= 1 or not options:
            _record(out, self.name, skipped="would leave no real channel")
            return out
        channel = options[int(rng.integers(len(options)))]
        out.waveform[channel] = 0.0
        mask = list(out.component_mask)
        mask[channel] = False
        out.component_mask = tuple(mask)
        _record(out, self.name, channel=channel, component="ZNE"[channel])
        return out


# --------------------------------------------------------------------------- rate and band

class RateTransform(Transform):
    """Decimate the valid support to an intermediate rate drawn from `rates`
    (only rates below the sample rate are eligible) and return to the sample
    rate, both through resample_waveform. The first-sample time is unchanged,
    so arrival and interval seconds are unchanged; the returned support is
    trimmed to the source support, so valid_samples shrinks by up to a few
    samples and the rest is zero padding."""

    name = "rate_transform"

    def __init__(self, prob: float = 0.3, rates=(20.0, 40.0, 50.0)):
        super().__init__(prob)
        self.rates = tuple(float(r) for r in rates)
        if not self.rates or any(r <= 0 for r in self.rates):
            raise ValueError("rates must list positive rates")

    def apply(self, sample, rng):
        out = sample.copy()
        rate, n, n_valid = out.rate_hz, out.n_samples, out.n_valid
        options = [r for r in self.rates if r < rate]
        if not options or n_valid < 2:
            _record(out, self.name, skipped="no intermediate rate below the sample rate or no support")
            return out
        intermediate = options[int(rng.integers(len(options)))]
        down = resample_waveform(out.waveform[:, :n_valid], rate, intermediate)
        if down.shape[-1] < 2:
            _record(out, self.name, skipped="support too short at the intermediate rate", intermediate_hz=intermediate)
            return out
        up = resample_waveform(down, intermediate, rate)
        n_up = min(int(up.shape[-1]), n_valid)
        wave = np.zeros((3, n), dtype=np.float32)
        wave[:, :n_up] = up[:, :n_up]
        out.waveform = wave
        out.valid_samples = None if n_up >= n else n_up
        _record(out, self.name, intermediate_hz=intermediate, n_valid_before=n_valid, n_valid_after=n_up)
        return out


class BandLimit(Transform):
    """Zero-phase Butterworth low-pass of the valid support at a corner drawn
    from `corner_hz` (capped at 0.9 Nyquist). Zero phase keeps arrival times."""

    name = "bandlimit"

    def __init__(self, prob: float = 0.0, corner_hz=(8.0, 20.0), order: int = 4):
        super().__init__(prob)
        self.corner_hz = _pair(corner_hz, "corner_hz")
        if self.corner_hz[0] <= 0:
            raise ValueError("corner_hz must be positive")
        self.order = int(order)

    def apply(self, sample, rng):
        out = sample.copy()
        rate, n_valid = out.rate_hz, out.n_valid
        corner = min(float(rng.uniform(*self.corner_hz)), 0.9 * rate / 2.0)
        sos = butter(self.order, corner, btype="low", fs=rate, output="sos")
        padlen = 3 * (2 * len(sos) + 1)
        if n_valid <= padlen:
            _record(out, self.name, skipped="support shorter than the filter padding", corner_hz=corner)
            return out
        out.waveform[:, :n_valid] = sosfiltfilt(sos, out.waveform[:, :n_valid].astype(np.float64), axis=-1).astype(np.float32)
        _record(out, self.name, corner_hz=corner, order=self.order)
        return out


# --------------------------------------------------------------------------- always-on

class AmplitudeJitter(Transform):
    """Multiply every channel by one log-uniform scalar from `scale`."""

    name = "amplitude_jitter"

    def __init__(self, scale=(0.5, 2.0), prob: float = 1.0):
        super().__init__(prob)
        self.scale = _pair(scale, "scale")
        if self.scale[0] <= 0:
            raise ValueError("scale must be positive")

    def apply(self, sample, rng):
        out = sample.copy()
        factor = float(np.exp(rng.uniform(np.log(self.scale[0]), np.log(self.scale[1]))))
        out.waveform *= np.float32(factor)
        _record(out, self.name, scale=factor)
        return out


class PolarityFlip(Transform):
    """Multiply every channel by -1 (the gate probability is the flip probability)."""

    name = "polarity_flip"

    def __init__(self, prob: float = 0.5):
        super().__init__(prob)

    def apply(self, sample, rng):
        out = sample.copy()
        out.waveform *= np.float32(-1.0)
        _record(out, self.name)
        return out


# --------------------------------------------------------------------------- composition

class Compose:
    """Apply `transforms` in order, then `always` (jitter and flip by default).
    A draw below `untouched_fraction` skips `transforms` and applies only
    `always`; meta["untouched"] records which. `pure_noise_fraction` is
    carried for the caller's mix_pure_noise call."""

    def __init__(self, transforms, untouched_fraction: float = 0.3, always=None, pure_noise_fraction: float = 0.0):
        self.transforms = list(transforms)
        self.untouched_fraction = _prob(untouched_fraction)
        self.always = [AmplitudeJitter(), PolarityFlip()] if always is None else list(always)
        self.pure_noise_fraction = _prob(pure_noise_fraction)

    def __call__(self, sample: WindowSample, rng: np.random.Generator) -> WindowSample:
        untouched = bool(self.untouched_fraction > 0.0 and rng.random() < self.untouched_fraction)
        out = sample
        if not untouched:
            for transform in self.transforms:
                out = transform(out, rng)
        for transform in self.always:
            out = transform(out, rng)
        if out is sample:
            out = sample.copy()
        out.meta["untouched"] = untouched
        return out

    def __repr__(self):
        return (f"Compose(transforms={self.transforms!r}, untouched_fraction={self.untouched_fraction}, "
                f"always={self.always!r}, pure_noise_fraction={self.pure_noise_fraction})")


def pure_noise_sample(noise_provider, n_samples: int, rate_hz: float, rng: np.random.Generator,
                      negative_support: str = "reviewed", meta=None) -> WindowSample:
    """An arrival-free WindowSample from the noise provider. component_mask is
    True where the channel is not identically zero; negative_support is the
    caller's statement about the pool (reviewed for a 42A pool)."""
    noise, class_name = _draw_noise(noise_provider, rng, int(n_samples), float(rate_hz))
    mask = tuple(bool(np.any(noise[c] != 0.0)) for c in range(3))
    info = dict(meta or {})
    info.update(pure_noise=True, noise_class=class_name)
    return WindowSample(noise, float(rate_hz), [], mask, negative_support, [], None, info)


def mix_pure_noise(samples, noise_provider, fraction: float, rng: np.random.Generator,
                   negative_support: str = "reviewed"):
    """Replace round(fraction * len(samples)) positions of a batch, chosen
    without replacement, by pure-noise windows of the same length and rate as
    the sample they replace. Returns (new list, replaced indices)."""
    samples = list(samples)
    count = int(round(_prob(fraction) * len(samples)))
    if count == 0:
        return samples, []
    indices = sorted(int(i) for i in rng.choice(len(samples), size=count, replace=False))
    for i in indices:
        source = samples[i]
        samples[i] = pure_noise_sample(noise_provider, source.n_samples, source.rate_hz, rng,
                                       negative_support, meta={"replaced": source.meta.get("trace_name")})
    return samples, indices


def augment_then_target(sample: WindowSample, pipeline, policy: LabelPolicy = MASKED,
                        rng: np.random.Generator = None):
    """Run the pipeline, then label_targets.targets_for on the result.
    Returns (waveform (3, n) float32, targets (3, n) PSN, mask (n,), info);
    info carries the target info, the augmentation records and the augmented
    sample under "sample"."""
    if rng is None:
        rng = np.random.default_rng()
    out = pipeline(sample, rng)
    built = targets_for(out, policy)
    info = dict(built.info)
    info.update(
        augmentations=list(out.meta.get("augmentations", ())),
        untouched=out.meta.get("untouched"),
        dropped_arrivals=list(out.meta.get("dropped_arrivals", ())),
        arrivals=[a.to_dict() for a in out.arrivals],
        component_mask=out.component_mask,
        unknown_intervals=list(out.unknown_intervals),
        valid_samples=out.valid_samples,
        sample=out,
    )
    return out.waveform, built.targets, built.mask, info


# --------------------------------------------------------------------------- config

ARTEFACTS = {
    "spike": Spike, "dc_step": DCStep, "gap": Gap, "clip": Clip,
    "mains_hum": MainsHum, "drift": Drift, "channel_drop": ChannelDrop,
}


def _section(cfg):
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("augmentation config must be a dict (full config, data section or augmentation_v3 section)")
    if CONFIG_SECTION in cfg:
        return cfg[CONFIG_SECTION]
    if "data" in cfg and isinstance(cfg["data"], dict):
        return cfg["data"].get(CONFIG_SECTION)
    return cfg


def _group(parent, key, default=None):
    """A group is off when absent (unless `default`), None, False,
    `enabled: false` or `prob: 0`. Returns the kwargs dict or None."""
    value = parent.get(key, default) if isinstance(parent, dict) else default
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"augmentation group {key!r} must be a mapping, true/false or null")
    if value.get("enabled", True) is False:
        return None
    if "prob" in value and float(value["prob"]) == 0.0:
        return None
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in value.items() if k != "enabled"}


def from_config(cfg, noise_provider=None, sample_provider=None) -> Compose:
    """Build the pipeline from `data.augmentation_v3` (the full config, the
    data section or the augmentation section itself). A missing or empty
    section or `enabled: false` gives an identity pipeline (copy only). Application
    order: crop, event_superposition, noise, nonstationary, artefacts (spike,
    dc_step, gap, clip, mains_hum, drift, channel_drop), rate_transform,
    bandlimit, then amplitude_jitter and polarity_flip on every sample."""
    section = _section(cfg)
    if section is not None and not isinstance(section, dict):
        raise TypeError(f"{CONFIG_SECTION} must be a mapping")
    if not section or section.get("enabled", True) is False:
        return Compose([], untouched_fraction=0.0, always=[], pure_noise_fraction=0.0)
    transforms = []
    crop = _group(section, "crop")
    if crop:
        transforms.append(RandomCrop(**crop))
    group = _group(section, "event_superposition")
    if group:
        if sample_provider is None:
            raise ValueError("event_superposition is enabled but no sample_provider was given")
        transforms.append(EventSuperposition(sample_provider, **group))
    group = _group(section, "noise")
    if group:
        if noise_provider is None:
            raise ValueError("noise is enabled but no noise_provider was given")
        transforms.append(NoiseSuperposition(noise_provider, **group))
    group = _group(section, "nonstationary")
    if group:
        transforms.append(NonStationaryNoise(noise_provider, **group))
    artefacts = section.get("artefacts") or {}
    if not isinstance(artefacts, dict):
        raise TypeError("artefacts must be a mapping of artefact name to settings")
    unknown = set(artefacts) - set(ARTEFACTS)
    if unknown:
        raise ValueError(f"unknown artefact groups {sorted(unknown)}; known: {sorted(ARTEFACTS)}")
    for key, cls in ARTEFACTS.items():
        group = _group(artefacts, key)
        if group:
            transforms.append(cls(**group))
    group = _group(section, "rate_transform")
    if group:
        transforms.append(RateTransform(**group))
    group = _group(section, "bandlimit")
    if group:
        transforms.append(BandLimit(**group))
    always = []
    group = _group(section, "amplitude_jitter", default={})
    if group is not None:
        always.append(AmplitudeJitter(**group))
    group = _group(section, "polarity_flip", default={})
    if group is not None:
        always.append(PolarityFlip(**group))
    return Compose(transforms, untouched_fraction=section.get("untouched_fraction", 0.3), always=always,
                   pure_noise_fraction=section.get("pure_noise_fraction", 0.0))
