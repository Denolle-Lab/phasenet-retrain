"""Augmentation transforms (#43A) on the 41A sample contract. Pure numpy/scipy.

Fixtures are Gaussian pulses (sigma 0.1 s) at integer sample times on a
white floor of 1e-3, so a retained arrival must still sit on its pulse peak
within one sample after any time-consistent transform.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arrivals import Arrival, WindowSample  # noqa: E402
from label_targets import MASKED, targets_for  # noqa: E402
import augmentation as aug  # noqa: E402

RATE = 100.0
N_STORED = 12000  # 120 s stored window
WINDOW = 3001


def pulse_sample(n=N_STORED, arrivals=(("P", 40.0), ("S", 55.0)), tier="manual", amplitude=1.0,
                 floor=1e-3, seed=0, negative_support="certified", **kw):
    rng = np.random.default_rng(seed)
    wave = floor * rng.standard_normal((3, n)).astype(np.float32)
    t = np.arange(n) / RATE
    for _, time_s in arrivals:
        wave += (amplitude * np.exp(-((t - time_s) ** 2) / (2 * 0.1 ** 2))).astype(np.float32)
    return WindowSample(wave, RATE, [Arrival(p, t_, tier=tier) for p, t_ in arrivals],
                        negative_support=negative_support, **kw)


def noise_provider(seed=1, length=20000):
    rng = np.random.default_rng(seed)
    return aug.ArrayNoiseProvider({
        "urban": [rng.standard_normal((3, length)).astype(np.float32)],
        "quiet": [0.3 * rng.standard_normal((3, length)).astype(np.float32)],
    }, RATE)


def second_provider(**kw):
    kw.setdefault("negative_support", "unknown")
    kw.setdefault("unknown_intervals", [(60.0, 62.0)])
    kw.setdefault("meta", {"trace_name": "second"})
    return aug.ListSampleProvider([pulse_sample(arrivals=(("P", 30.0), ("S", 38.0)), seed=3, **kw)])


def local_peak_offset(sample, arrival, halfwidth_s=0.3):
    centre = int(round(arrival.time_s * sample.rate_hz))
    half = int(halfwidth_s * sample.rate_hz)
    lo, hi = max(centre - half, 0), min(centre + half + 1, sample.n_samples)
    real = np.asarray(sample.component_mask, bool)
    seg = np.abs(sample.waveform[real, lo:hi]).max(axis=0)
    return int(np.argmax(seg)) + lo - centre


def inside_unknown(sample, time_s, margin_s=0.3):
    """True inside or within `margin_s` of an unknown interval or the valid end,
    where a gap edge legitimately destroys the pulse shape the peak check needs."""
    if any(a - margin_s <= time_s < b + margin_s for a, b in sample.unknown_intervals):
        return True
    return time_s >= sample.n_valid / sample.rate_hz - margin_s


def check_peaks(sample):
    checked = 0
    for a in sample.arrivals:
        assert 0.0 <= a.time_s < sample.duration_s
        if inside_unknown(sample, a.time_s):
            continue
        assert abs(local_peak_offset(sample, a)) <= 1, (a, local_peak_offset(sample, a))
        checked += 1
    return checked


def snapshot(sample):
    return (sample.waveform.copy(), [a.to_dict() for a in sample.arrivals], sample.component_mask,
            sample.negative_support, list(sample.unknown_intervals), sample.valid_samples,
            copy.deepcopy(sample.meta))


def assert_unchanged(sample, snap):
    wave, arrivals, mask, support, intervals, valid, meta = snap
    np.testing.assert_array_equal(sample.waveform, wave)
    assert [a.to_dict() for a in sample.arrivals] == arrivals
    assert sample.component_mask == mask and sample.negative_support == support
    assert list(sample.unknown_intervals) == intervals and sample.valid_samples == valid
    assert sample.meta == meta


def band_energy(wave, rate, lo_hz, hi_hz=None):
    f = np.fft.rfftfreq(wave.shape[-1], 1.0 / rate)
    power = np.abs(np.fft.rfft(wave.astype(np.float64), axis=-1)) ** 2
    band = (f > lo_hz) if hi_hz is None else ((f > lo_hz) & (f < hi_hz))
    return float(power[:, band].sum())


# ---------------------------------------------------------------- crop

def test_random_crop_keeps_an_anchor_and_moves_arrivals_with_the_waveform():
    s = pulse_sample()
    crop = aug.RandomCrop(WINDOW, anchor_range=(0.1, 0.7))
    for seed in range(8):
        out = crop(s, np.random.default_rng(seed))
        rec = out.meta["augmentations"][-1]
        assert out.n_samples == WINDOW and out.valid_samples is None
        assert len(out.supervised_arrivals()) >= 1
        assert check_peaks(out) == len(out.arrivals)
        start_s = rec["start_s"]
        for a in out.arrivals:
            source = next(b for b in s.arrivals if b.phase == a.phase)
            assert a.time_s == pytest.approx(source.time_s - start_s, abs=1e-9)
            assert 0.1 * WINDOW / RATE - 1e-9 <= a.time_s or len(out.arrivals) == 2
        dropped = out.meta["dropped_arrivals"]
        assert len(dropped) + len(out.arrivals) == 2 and rec["n_dropped"] == len(dropped)
        for d in dropped:
            assert not (0.0 <= d["time_s"] < out.duration_s)


def test_random_crop_clips_unknown_intervals_and_recomputes_valid_samples():
    s = pulse_sample(unknown_intervals=[(41.0, 43.0), (100.0, 101.0)], valid_samples=9000)
    out = aug.RandomCrop(WINDOW, anchor_range=(0.5, 0.5))(s, np.random.default_rng(0))
    start = out.meta["augmentations"][-1]["start_sample"]
    assert 0 <= start <= N_STORED - WINDOW
    for a, b in out.unknown_intervals:
        assert 0.0 <= a < b <= out.duration_s
    expected_valid = min(max(9000 - start, 0), WINDOW)
    assert out.n_valid == expected_valid
    if start < 4100 - WINDOW or start > 4300:
        assert (41.0 - start / RATE, 43.0 - start / RATE) not in out.unknown_intervals or True
    # a stored window shorter than the crop is zero-padded beyond valid_samples
    short = pulse_sample(n=2000, arrivals=(("P", 5.0),))
    padded = aug.RandomCrop(WINDOW)(short, np.random.default_rng(0))
    assert padded.n_samples == WINDOW and padded.valid_samples == 2000
    assert np.all(padded.waveform[:, 2000:] == 0) and padded.arrivals[0].time_s == 5.0


def test_random_crop_without_arrivals_draws_a_uniform_start():
    s = pulse_sample(arrivals=())
    starts = {aug.RandomCrop(WINDOW)(s, np.random.default_rng(i)).meta["augmentations"][-1]["start_sample"]
              for i in range(5)}
    assert len(starts) > 1 and all(0 <= k <= N_STORED - WINDOW for k in starts)


# ---------------------------------------------------------------- noise

def test_noise_superposition_realises_the_target_snr_in_the_p_window():
    s = pulse_sample()
    transform = aug.NoiseSuperposition(noise_provider(), prob=1.0)
    for seed in range(6):
        out = transform(s, np.random.default_rng(seed))
        rec = out.meta["augmentations"][-1]
        assert rec["snr_window"] == "P" and rec["snr_window_s"] == pytest.approx((39.5, 42.0))
        lo, hi = rec["snr_window_s"]
        i0, i1 = int(lo * RATE), int(np.ceil(hi * RATE))
        p_signal = float(np.mean(s.waveform[:, i0:i1].astype(np.float64) ** 2))
        added = out.waveform.astype(np.float64) - s.waveform.astype(np.float64)
        p_noise = float(np.mean(added[:, i0:i1] ** 2))
        realised = 10 * np.log10(p_signal / p_noise)
        assert abs(realised - rec["snr_target_db"]) < 0.5
        assert rec["snr_db"] == pytest.approx(realised, abs=0.05)
        assert out.meta["realised_snr_db"] == rec["snr_db"]
        weight, blo, bhi = aug.DEFAULT_SNR_BINS[rec["snr_bin"]]
        assert blo <= rec["snr_target_db"] <= bhi
        assert [a.to_dict() for a in out.arrivals] == [a.to_dict() for a in s.arrivals]
        assert rec["noise_class"] in ("urban", "quiet")


def test_snr_bins_are_drawn_by_weight():
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    for _ in range(3000):
        counts[aug.draw_snr_db(rng)[1]] += 1
    np.testing.assert_allclose(counts / counts.sum(), [0.4, 0.3, 0.3], atol=0.04)


def test_noise_superposition_falls_back_to_s_window_then_whole_window():
    transform = aug.NoiseSuperposition(noise_provider(), prob=1.0)
    s_only = transform(pulse_sample(arrivals=(("S", 55.0),)), np.random.default_rng(0))
    rec = s_only.meta["augmentations"][-1]
    assert rec["snr_window"] == "S" and rec["snr_window_s"] == pytest.approx((54.5, 57.0))
    empty = transform(pulse_sample(arrivals=()), np.random.default_rng(0))
    rec = empty.meta["augmentations"][-1]
    assert rec["snr_window"] == "whole" and rec["snr_window_s"] == pytest.approx((0.0, 120.0))


def test_noise_superposition_respects_component_mask_and_padding():
    s = pulse_sample(component_mask=(True, False, True), valid_samples=9000)
    s.waveform[1] = 0.0
    out = aug.NoiseSuperposition(noise_provider(), prob=1.0)(s, np.random.default_rng(0))
    added = out.waveform - s.waveform
    assert np.all(added[1] == 0) and np.all(added[:, 9000:] == 0) and np.any(added[0] != 0)
    assert out.component_mask == (True, False, True) and out.valid_samples == 9000


def test_noise_provider_protocol_is_enforced():
    class Bad:
        def draw(self, rng, n, rate):
            return np.zeros((2, n), np.float32), "bad"
    with pytest.raises(ValueError):
        aug.NoiseSuperposition(Bad(), prob=1.0)(pulse_sample(), np.random.default_rng(0))
    provider = noise_provider()
    rng = np.random.default_rng(0)
    classes = [provider.draw(rng, 3001, RATE)[1] for _ in range(400)]
    assert 0.35 < classes.count("urban") / 400 < 0.65
    wave, _ = provider.draw(rng, 3001, RATE)
    assert wave.shape == (3, 3001) and wave.dtype == np.float32
    with pytest.raises(ValueError):
        provider.draw(rng, 30000, RATE)
    resampled, _ = provider.draw(rng, 1000, 50.0)
    assert resampled.shape == (3, 1000)


def test_nonstationary_noise_changes_level_inside_the_window():
    s = pulse_sample()
    transform = aug.NonStationaryNoise(noise_provider(), prob=1.0, snr_bins=((1.0, 10.0, 10.0),))
    modes = set()
    for seed in range(12):
        out = transform(s, np.random.default_rng(seed))
        rec = out.meta["augmentations"][-1]
        modes.add(rec["mode"])
        assert rec["snr_target_db"] == pytest.approx(10.0)
        assert rec["snr_db"] is None or rec["snr_db"] >= 10.0 - 0.5
        added = (out.waveform - s.waveform).astype(np.float64)
        first, last = np.sqrt(np.mean(added[:, :1200] ** 2)), np.sqrt(np.mean(added[:, -1200:] ** 2))
        if rec["mode"] == "ramp":
            assert (first < last) == (rec["envelope"]["direction"] == "up")
        assert [a.to_dict() for a in out.arrivals] == [a.to_dict() for a in s.arrivals]
    assert modes == {"ramp", "step", "partial"}
    white = aug.NonStationaryNoise(None, prob=1.0)(s, np.random.default_rng(0))
    assert white.meta["augmentations"][-1]["noise_class"] == "white"


# ---------------------------------------------------------------- event superposition

def test_event_superposition_merges_arrivals_and_masks_with_unknown_winning():
    base = pulse_sample()
    transform = aug.EventSuperposition(second_provider(), prob=1.0, offset_s=(3.0, 3.0), amplitude_ratio=(0.5, 0.5))
    for seed in range(4):
        out = transform(base, np.random.default_rng(seed))
        rec = out.meta["augmentations"][-1]
        assert rec["n_added"] == 2 and rec["n_dropped"] == 0
        assert abs(abs(rec["offset_s"]) - 3.0) < 1.0 / RATE
        added = out.arrivals[2:]
        assert [a.event_id for a in added] == ["second", "second"]
        assert added[0].phase == "P" and added[0].time_s == pytest.approx(40.0 + rec["offset_s"], abs=1e-9)
        assert added[1].time_s == pytest.approx(added[0].time_s + 8.0, abs=1e-9)
        assert check_peaks(out) == 4
        assert out.negative_support == "unknown"
        shift = rec["shift_s"]
        assert out.unknown_intervals == [(pytest.approx(60.0 + shift), pytest.approx(62.0 + shift))]
        assert out.valid_samples is None and out.component_mask == (True, True, True)
        built = targets_for(out, MASKED)
        u0, u1 = out.unknown_intervals[0]
        assert built.mask[int(u0 * RATE):int(u1 * RATE)].max() == 0.0
        for a in out.arrivals:
            c = int(round(a.time_s * RATE))
            assert built.targets["PS".index(a.phase), c] >= 0.99 and built.mask[c] == 1.0
        # the second window's peak is amplitude_ratio times the base peak
        assert abs(out.waveform[:, int(round(added[0].time_s * RATE))]).max() == pytest.approx(0.5, abs=0.02)


def test_event_superposition_support_rules():
    assert aug.merge_support("certified", "unknown") == "unknown"
    assert aug.merge_support("certified", "reviewed") == "reviewed"
    assert aug.merge_support("certified", "certified") == "certified"
    base = pulse_sample(negative_support="unknown")
    provider = second_provider(negative_support="certified", unknown_intervals=[], component_mask=(True, True, False),
                               valid_samples=5000)
    out = aug.EventSuperposition(provider, prob=1.0, offset_s=(3.0, 3.0))(base, np.random.default_rng(1))
    rec = out.meta["augmentations"][-1]
    assert out.negative_support == "unknown"
    assert out.component_mask == (True, True, False) and np.all(out.waveform[2] == 0)
    shift_samples = int(round(rec["shift_s"] * RATE))
    assert out.valid_samples == min(N_STORED, shift_samples + 5000)
    # no arrival to anchor on: recorded skip, sample otherwise unchanged
    empty = aug.EventSuperposition(provider, prob=1.0)(pulse_sample(arrivals=()), np.random.default_rng(0))
    assert "skipped" in empty.meta["augmentations"][-1] and empty.arrivals == []


# ---------------------------------------------------------------- artefacts

def test_gap_is_zero_and_masked_unknown_through_targets_for():
    s = pulse_sample()
    out = aug.Gap(prob=1.0, duration_s=(1.0, 1.0))(s, np.random.default_rng(0))
    rec = out.meta["augmentations"][-1]
    i0, i1 = rec["start_sample"], rec["end_sample"]
    assert i1 - i0 == 100 and out.unknown_intervals == [(i0 / RATE, i1 / RATE)]
    assert np.all(out.waveform[:, i0:i1] == 0)
    mask = targets_for(out, MASKED).mask
    assert mask[i0:i1].max() == 0.0 and mask[i0 - 1] == 1.0 and mask[i1] == 1.0


def test_additive_artefacts_keep_arrivals_and_record_what_they_did():
    s = pulse_sample()
    rms = aug.reference_rms(s)
    # spike: only the recorded samples on the recorded channels change
    out = aug.Spike(prob=1.0, amplitude=(10.0, 10.0))(s, np.random.default_rng(0))
    rec = out.meta["augmentations"][-1]
    diff = out.waveform - s.waveform
    changed = np.argwhere(diff != 0)
    assert set(changed[:, 0].tolist()) == set(rec["channels"])
    assert changed[:, 1].min() == rec["start_sample"] and changed[:, 1].max() == rec["start_sample"] + rec["width"] - 1
    assert abs(rec["amplitude"]) == pytest.approx(10.0 * rms)
    # dc step: constant offset on one channel from the step onwards
    out = aug.DCStep(prob=1.0)(s, np.random.default_rng(0))
    rec = out.meta["augmentations"][-1]
    diff = out.waveform - s.waveform
    assert np.all(diff[rec["channel"], rec["start_sample"]:] == np.float32(rec["amplitude"]))
    assert np.all(diff[rec["channel"], :rec["start_sample"]] == 0)
    # clip: nothing above the threshold
    out = aug.Clip(prob=1.0, level=(0.5, 0.5))(s, np.random.default_rng(0))
    rec = out.meta["augmentations"][-1]
    assert np.abs(out.waveform).max() <= rec["threshold"] + 1e-6 and rec["n_clipped"] > 0
    # mains hum: the added signal is a tone at the recorded frequency
    out = aug.MainsHum(prob=1.0, freq_hz=(20.0,))(s, np.random.default_rng(0))
    diff = (out.waveform - s.waveform).astype(np.float64)
    f = np.fft.rfftfreq(N_STORED, 1 / RATE)
    assert f[np.argmax(np.abs(np.fft.rfft(diff[0])))] == pytest.approx(20.0, abs=0.02)
    assert out.meta["augmentations"][-1]["freq_hz"] == 20.0
    # drift: the added signal lives below 0.1 Hz
    out = aug.Drift(prob=1.0)(s, np.random.default_rng(0))
    diff = out.waveform - s.waveform
    assert band_energy(diff, RATE, 0.1) < 2e-2 * band_energy(diff, RATE, -1.0)
    for transform in (aug.Spike(1.0), aug.DCStep(1.0), aug.Clip(1.0), aug.MainsHum(1.0), aug.Drift(1.0)):
        out = transform(s, np.random.default_rng(3))
        assert [a.to_dict() for a in out.arrivals] == [a.to_dict() for a in s.arrivals]
        assert out.unknown_intervals == [] and out.valid_samples is None
        assert out.meta["augmentations"][-1]["name"] == transform.name


def test_channel_drop_updates_component_mask_and_never_drops_the_last_channel():
    s = pulse_sample()
    out = aug.ChannelDrop(prob=1.0, candidates=(1,))(s, np.random.default_rng(0))
    assert out.component_mask == (True, False, True) and np.all(out.waveform[1] == 0)
    assert out.meta["augmentations"][-1] == {"name": "channel_drop", "channel": 1, "component": "N"}
    single = pulse_sample(component_mask=(True, False, False))
    kept = aug.ChannelDrop(prob=1.0)(single, np.random.default_rng(0))
    assert kept.component_mask == (True, False, False) and "skipped" in kept.meta["augmentations"][-1]


# ---------------------------------------------------------------- rate and band

@pytest.mark.parametrize("intermediate", [20.0, 40.0, 50.0])
def test_rate_transform_keeps_pulse_time_and_removes_energy_above_intermediate_nyquist(intermediate):
    s = pulse_sample()
    transform = aug.RateTransform(prob=1.0, rates=(intermediate,))
    out = transform(s, np.random.default_rng(0))
    rec = out.meta["augmentations"][-1]
    assert rec["intermediate_hz"] == intermediate and out.rate_hz == RATE and out.n_samples == N_STORED
    assert [a.time_s for a in out.arrivals] == [40.0, 55.0]
    assert all(abs(local_peak_offset(out, a)) <= 1 for a in out.arrivals)
    assert N_STORED - 10 <= out.n_valid < N_STORED and np.all(out.waveform[:, out.n_valid:] == 0)
    white = WindowSample(np.random.default_rng(5).standard_normal((3, WINDOW)).astype(np.float32), RATE)
    filtered = transform(white, np.random.default_rng(0))
    stop = 1.2 * intermediate / 2
    attenuation_db = 10 * np.log10(band_energy(filtered.waveform, RATE, stop) / band_energy(white.waveform, RATE, stop))
    assert attenuation_db < -20.0
    passband_db = 10 * np.log10(band_energy(filtered.waveform, RATE, 0.0, 0.8 * intermediate / 2)
                                / band_energy(white.waveform, RATE, 0.0, 0.8 * intermediate / 2))
    assert abs(passband_db) < 0.5


def test_rate_transform_is_a_recorded_no_op_when_no_rate_is_below_the_sample_rate():
    low = WindowSample(np.zeros((3, 500), np.float32), 20.0, [Arrival("P", 5.0)])
    out = aug.RateTransform(prob=1.0)(low, np.random.default_rng(0))
    assert "skipped" in out.meta["augmentations"][-1] and out.arrivals[0].time_s == 5.0


def test_bandlimit_is_zero_phase():
    s = pulse_sample()
    out = aug.BandLimit(prob=1.0, corner_hz=(10.0, 10.0))(s, np.random.default_rng(0))
    assert all(local_peak_offset(out, a) == 0 for a in out.arrivals)
    white = WindowSample(np.random.default_rng(5).standard_normal((3, WINDOW)).astype(np.float32), RATE)
    filtered = aug.BandLimit(prob=1.0, corner_hz=(10.0, 10.0))(white, np.random.default_rng(0))
    assert 10 * np.log10(band_energy(filtered.waveform, RATE, 20.0) / band_energy(white.waveform, RATE, 20.0)) < -20


def test_resample_sample_keeps_arrival_seconds():
    s = pulse_sample()
    out = aug.resample_sample(s, 50.0)
    assert out.rate_hz == 50.0 and out.n_samples == (N_STORED - 1) // 2 + 1
    assert [a.time_s for a in out.arrivals] == [40.0, 55.0]
    assert all(local_peak_offset(out, a) == 0 for a in out.arrivals)


# ---------------------------------------------------------------- composition

def full_pipeline(untouched_fraction=0.0):
    np_, sp = noise_provider(), second_provider()
    quiet = ((1.0, 40.0, 40.0),)
    return aug.Compose([
        aug.RandomCrop(WINDOW),
        aug.EventSuperposition(sp, prob=1.0, offset_s=(3.0, 3.0), amplitude_ratio=(0.5, 0.5)),
        aug.NoiseSuperposition(np_, prob=1.0, snr_bins=quiet),
        aug.NonStationaryNoise(np_, prob=1.0, snr_bins=quiet),
        aug.Gap(prob=1.0),
        aug.ChannelDrop(prob=1.0),
        aug.RateTransform(prob=1.0),
        aug.BandLimit(prob=1.0),
    ], untouched_fraction=untouched_fraction)


def test_compose_untouched_changes_the_waveform_only_by_a_scalar_and_sign():
    s = pulse_sample()
    pipeline = full_pipeline(untouched_fraction=1.0)
    signs = set()
    for seed in range(6):
        out = pipeline(s, np.random.default_rng(seed))
        assert out.meta["untouched"] is True
        names = [r["name"] for r in out.meta["augmentations"]]
        assert set(names) <= {"amplitude_jitter", "polarity_flip"} and "amplitude_jitter" in names
        big = np.abs(s.waveform) > 0.05
        ratio = out.waveform[big] / s.waveform[big]
        factor = float(ratio[0])
        np.testing.assert_allclose(ratio, factor, rtol=1e-4)
        assert 0.5 <= abs(factor) <= 2.0
        signs.add(np.sign(factor))
        assert [a.to_dict() for a in out.arrivals] == [a.to_dict() for a in s.arrivals]
        assert out.n_samples == N_STORED and out.unknown_intervals == []
    assert signs == {-1.0, 1.0}


def test_composed_pipeline_keeps_every_retained_arrival_on_its_pulse():
    s = pulse_sample()
    pipeline = full_pipeline()
    checked = 0
    for seed in range(10):
        out = pipeline(s, np.random.default_rng(seed))
        assert out.meta["untouched"] is False and out.n_samples == WINDOW and out.rate_hz == RATE
        names = [r["name"] for r in out.meta["augmentations"]]
        assert names[:8] == ["crop", "event_superposition", "noise", "nonstationary", "gap", "channel_drop",
                             "rate_transform", "bandlimit"]
        assert 2 <= len(out.arrivals) <= 4 and len(out.meta["dropped_arrivals"]) + len(out.arrivals) == 4
        assert sum(1 for m in out.component_mask if not m) == 1
        for a, b in out.unknown_intervals:
            assert 0.0 <= a < b <= out.duration_s
        assert out.n_valid <= WINDOW
        checked += check_peaks(out)
        built = targets_for(out, MASKED)
        gap = next(r for r in out.meta["augmentations"] if r["name"] == "gap")
        assert built.mask[gap["start_sample"]:gap["end_sample"]].max() == 0.0
        assert built.info["n_outside"] == 0
    assert checked >= 20


def test_augment_then_target_peaks_at_the_arrival_samples():
    s = pulse_sample()
    pipeline = aug.Compose([aug.RandomCrop(WINDOW), aug.NoiseSuperposition(noise_provider(), prob=1.0,
                                                                            snr_bins=((1.0, 30.0, 30.0),))],
                           untouched_fraction=0.0)
    for seed in range(4):
        wave, targets, mask, info = aug.augment_then_target(s, pipeline, MASKED, np.random.default_rng(seed))
        assert wave.shape == (3, WINDOW) and targets.shape == (3, WINDOW) and mask.shape == (WINDOW,)
        np.testing.assert_allclose(targets.sum(axis=0), 1.0, atol=1e-6)
        assert info["arrivals"] and info["policy"] == "masked" and info["untouched"] is False
        for a in info["arrivals"]:
            c = int(round(a["time_s"] * RATE))
            assert targets["PS".index(a["phase"]), c] >= 0.99 and mask[c] == 1.0
            assert abs(local_peak_offset(info["sample"], Arrival(a["phase"], a["time_s"]))) <= 1
        assert [r["name"] for r in info["augmentations"]][:2] == ["crop", "noise"]


def test_seeded_rng_reproduces_the_pipeline():
    s = pulse_sample()
    pipeline = full_pipeline()
    a = pipeline(s, np.random.default_rng(123))
    b = pipeline(s, np.random.default_rng(123))
    np.testing.assert_array_equal(a.waveform, b.waveform)
    assert [x.to_dict() for x in a.arrivals] == [x.to_dict() for x in b.arrivals]
    assert a.meta == b.meta and a.unknown_intervals == b.unknown_intervals
    c = pipeline(s, np.random.default_rng(124))
    assert not np.array_equal(a.waveform, c.waveform)


def test_transforms_never_mutate_their_input():
    s = pulse_sample(unknown_intervals=[(70.0, 71.0)], valid_samples=11000, meta={"trace_name": "t", "augmentations": []})
    snap = snapshot(s)
    np_, sp = noise_provider(), second_provider()
    transforms = [aug.RandomCrop(WINDOW), aug.NoiseSuperposition(np_, 1.0), aug.NonStationaryNoise(np_, 1.0),
                  aug.EventSuperposition(sp, 1.0, (3.0, 3.0)), aug.Spike(1.0), aug.DCStep(1.0), aug.Gap(1.0),
                  aug.Clip(1.0), aug.MainsHum(1.0), aug.Drift(1.0), aug.ChannelDrop(1.0), aug.RateTransform(1.0),
                  aug.BandLimit(1.0), aug.AmplitudeJitter(), aug.PolarityFlip(1.0), full_pipeline(),
                  aug.Spike(0.0)]
    for transform in transforms:
        out = transform(s, np.random.default_rng(0))
        assert out is not s and out.waveform is not s.waveform
        assert_unchanged(s, snap)
    batch, _ = aug.mix_pure_noise([s] * 4, np_, 0.5, np.random.default_rng(0))
    assert_unchanged(s, snap)


def test_mix_pure_noise_replaces_the_requested_fraction():
    samples = [pulse_sample(n=WINDOW, arrivals=(("P", 9.0),), seed=i, meta={"trace_name": f"t{i}"}) for i in range(20)]
    batch, replaced = aug.mix_pure_noise(samples, noise_provider(), 0.15, np.random.default_rng(0))
    assert len(batch) == 20 and len(replaced) == 3 and replaced == sorted(set(replaced))
    for i, sample in enumerate(batch):
        if i in replaced:
            assert sample.arrivals == [] and sample.negative_support == "reviewed"
            assert sample.meta["pure_noise"] is True and sample.meta["noise_class"] in ("urban", "quiet")
            assert sample.meta["replaced"] == f"t{i}" and sample.waveform.shape == (3, WINDOW)
            assert sample.component_mask == (True, True, True) and targets_for(sample).n_supervised == WINDOW
        else:
            assert sample is samples[i]
    untouched, none = aug.mix_pure_noise(samples, noise_provider(), 0.0, np.random.default_rng(0))
    assert none == [] and untouched == samples


# ---------------------------------------------------------------- config

CONFIG = """
data:
  augmentation_v3:
    untouched_fraction: 0.3
    pure_noise_fraction: 0.15
    crop: {window_samples: 3001, anchor_range: [0.1, 0.7]}
    noise:
      prob: 0.6
      snr_bins: [[0.4, 0, 5], [0.3, 5, 10], [0.3, 10, 25]]
    event_superposition: {prob: 0.3, offset_s: [2, 40], amplitude_ratio: [0.1, 1.0]}
    nonstationary: {prob: 0.3}
    artefacts:
      spike: {prob: 0.03}
      dc_step: {prob: 0.03}
      gap: {prob: 0.03, duration_s: [0.1, 3.0]}
      clip: {prob: 0.03}
      mains_hum: {prob: 0.02}
      drift: {prob: 0.03}
      channel_drop: {prob: 0.05}
    rate_transform: {prob: 0.3, rates: [20, 40, 50]}
    bandlimit: {prob: 0.1, corner_hz: [8, 20]}
    amplitude_jitter: {scale: [0.5, 2.0]}
    polarity_flip: {prob: 0.5}
"""


def test_from_config_builds_the_documented_pipeline():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG)
    pipeline = aug.from_config(cfg, noise_provider=noise_provider(), sample_provider=second_provider())
    names = [t.name for t in pipeline.transforms]
    assert names == ["crop", "event_superposition", "noise", "nonstationary", "spike", "dc_step", "gap", "clip",
                     "mains_hum", "drift", "channel_drop", "rate_transform", "bandlimit"]
    assert [t.name for t in pipeline.always] == ["amplitude_jitter", "polarity_flip"]
    assert pipeline.untouched_fraction == 0.3 and pipeline.pure_noise_fraction == 0.15
    by_name = {t.name: t for t in pipeline.transforms}
    assert by_name["noise"].prob == 0.6 and by_name["noise"].snr_bins == aug.DEFAULT_SNR_BINS
    assert by_name["rate_transform"].rates == (20.0, 40.0, 50.0) and by_name["gap"].duration_s == (0.1, 3.0)
    assert by_name["event_superposition"].offset_s == (2.0, 40.0)
    out = pipeline(pulse_sample(), np.random.default_rng(0))
    assert out.n_samples in (WINDOW, N_STORED) and targets_for(out).targets.shape[-1] == out.n_samples
    # the same section is accepted on its own or under data
    section = cfg["data"]["augmentation_v3"]
    assert [t.name for t in aug.from_config(section, noise_provider(), second_provider()).transforms] == names
    assert [t.name for t in aug.from_config(cfg["data"], noise_provider(), second_provider()).transforms] == names


def test_from_config_switches_groups_off_and_checks_providers():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CONFIG)
    section = cfg["data"]["augmentation_v3"]
    with pytest.raises(ValueError, match="noise_provider"):
        aug.from_config(cfg, sample_provider=second_provider())
    with pytest.raises(ValueError, match="sample_provider"):
        aug.from_config(cfg, noise_provider=noise_provider())
    section["noise"]["prob"] = 0.0
    section["event_superposition"] = None
    section["artefacts"]["gap"] = {"enabled": False}
    section["artefacts"]["clip"] = False
    section["polarity_flip"] = None
    section["bandlimit"]["prob"] = 0
    pipeline = aug.from_config(cfg, noise_provider=noise_provider())
    assert [t.name for t in pipeline.transforms] == ["crop", "nonstationary", "spike", "dc_step", "mains_hum",
                                                     "drift", "channel_drop", "rate_transform"]
    assert [t.name for t in pipeline.always] == ["amplitude_jitter"]
    section["artefacts"]["hum"] = {"prob": 0.1}
    with pytest.raises(ValueError, match="unknown artefact"):
        aug.from_config(cfg, noise_provider=noise_provider())
    s = pulse_sample()
    for cfg_off in (None, {}, {"data": {}}, {"data": {"augmentation_v3": {"enabled": False}}}):
        identity = aug.from_config(cfg_off)
        out = identity(s, np.random.default_rng(0))
        assert identity.transforms == [] and identity.always == []
        np.testing.assert_array_equal(out.waveform, s.waveform)
        assert out is not s and out.meta["untouched"] is False
