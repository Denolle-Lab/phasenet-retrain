"""Label validity policy (#41A): targets, masks and the arrival schema. Pure numpy."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arrivals import (Arrival, WindowSample, arrivals_from_columns, arrivals_from_json,  # noqa: E402
                      arrivals_to_json, shift_arrivals)
from label_targets import (LEGACY, MASKED, LabelPolicy, build_targets, gaussian,  # noqa: E402
                           has_supervision, targets_for)

RATE = 100.0
N = 3001


def legacy_formula(p_off, s_off, n=N, sigma=10.0):
    """The v7 formula from scripts/manifest_dataset.py::make_labels, replicated
    here because that module imports torch."""
    x = np.arange(n, dtype=np.float32)
    def g(c):
        return np.zeros(n, np.float32) if c is None else np.exp(-((x - c) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    p, s = g(p_off), g(s_off)
    return np.stack([p, s, np.clip(1 - np.maximum(p, s), 0, 1)]).astype(np.float32)


def test_legacy_policy_reproduces_make_labels_exactly():
    arr = arrivals_from_columns(900.0, 1350.5, RATE)
    t = build_targets(arr, N, RATE, policy=LEGACY)
    np.testing.assert_allclose(t.targets, legacy_formula(900.0, 1350.5), atol=1e-6)
    assert t.mask.shape == (N,) and t.mask.min() == 1.0
    assert t.info["policy"] == "legacy" and t.n_supervised == N


def test_legacy_policy_p_only_leaves_s_zero_and_supervises_everything():
    t = build_targets([Arrival("P", 9.0)], N, RATE, policy=LEGACY)
    np.testing.assert_allclose(t.targets, legacy_formula(900.0, None), atol=1e-6)
    assert t.n_supervised == N


def test_masked_targets_sum_to_one_and_coincident_arrivals_split():
    t = build_targets([Arrival("P", 5.0), Arrival("S", 5.0)], 1001, RATE, negative_support="certified")
    np.testing.assert_allclose(t.targets.sum(axis=0), 1.0, atol=1e-6)
    assert t.targets[0, 500] == pytest.approx(0.5) and t.targets[1, 500] == pytest.approx(0.5)
    assert t.targets[2, 500] == pytest.approx(0.0)
    assert t.targets[2, 0] == pytest.approx(1.0)


def test_masked_same_phase_overlap_takes_the_maximum_not_the_sum():
    t = build_targets([Arrival("P", 5.0), Arrival("P", 5.05)], 1001, RATE, negative_support="certified")
    assert t.targets[0].max() <= 1.0 + 1e-6
    assert t.info["used"] == {"P": 2, "S": 0}


def test_certified_negative_support_supervises_every_sample():
    t = build_targets([Arrival("P", 3.0)], N, RATE, negative_support="certified")
    assert t.n_supervised == N
    t = build_targets([], N, RATE, negative_support="reviewed")
    assert t.n_supervised == N and t.targets[2].min() == 1.0


def test_unknown_negative_support_supervises_only_around_supervising_arrivals():
    t = build_targets([Arrival("P", 10.0), Arrival("S", 15.0)], N, RATE, negative_support="unknown")
    hw = int(MASKED.supervised_halfwidth_s * RATE)
    assert t.mask[1000 - hw:1000 + hw + 1].min() == 1.0
    assert t.mask[1500 - hw:1500 + hw + 1].min() == 1.0
    assert t.mask[1000 - hw - 2] == 0.0 and t.mask[1000 + hw + 2] == 0.0
    assert t.mask[0] == 0.0 and t.mask[-1] == 0.0
    assert t.n_supervised == 2 * (2 * hw + 1)
    assert 0 < t.info["supervised_fraction"] < 0.1


def test_automatic_arrival_masks_its_neighbourhood_but_not_a_manual_core():
    manual = Arrival("P", 10.0, tier="manual")
    auto = Arrival("S", 10.4, tier="automatic")
    t = build_targets([manual, auto], N, RATE, negative_support="certified")
    # the automatic S contributes no target and no positive support
    assert t.targets[1].max() == 0.0 and t.info["n_ignored_tier"] == 1
    # its ±1 s halo is unsupervised ...
    assert t.mask[1130] == 0.0 and t.mask[1090] == 0.0
    # ... except the ±3σ core of the manual P, which stays supervised
    assert t.mask[1000 - 30:1000 + 30 + 1].min() == 1.0
    assert t.mask[1000 + 31] == 0.0
    # far from both, certified support is supervised
    assert t.mask[0] == 1.0 and t.mask[-1] == 1.0


def test_unknown_intervals_and_padding_are_never_supervised():
    t = build_targets([Arrival("P", 5.0)], 1001, RATE, negative_support="certified",
                      unknown_intervals=[(2.0, 3.0)], valid_samples=800)
    assert t.mask[200:300].max() == 0.0 and t.mask[199] == 1.0 and t.mask[300] == 1.0
    assert t.mask[800:].max() == 0.0 and t.mask[799] == 1.0
    # a gap wins over positive support
    t = build_targets([Arrival("P", 5.0)], 1001, RATE, negative_support="unknown",
                      unknown_intervals=[(4.9, 5.1)])
    assert t.mask[490:510].max() == 0.0 and t.mask[489] == 1.0


def test_arrivals_outside_the_window_are_reported_not_used():
    t = build_targets([Arrival("P", -0.5), Arrival("S", 40.0), Arrival("P", 1.0)], N, RATE,
                      negative_support="certified")
    assert t.info["n_outside"] == 2 and t.info["used"] == {"P": 1, "S": 0}
    assert {a["time_s"] for a in t.info["outside"]} == {-0.5, 40.0}


def test_fractional_arrival_centre_is_kept():
    t = build_targets([Arrival("P", 5.004)], 1001, RATE, negative_support="certified")
    assert t.targets[0, 500] == pytest.approx(gaussian(1001, 500.4, 10.0)[500])


def test_targets_for_window_sample_and_has_supervision():
    wf = np.zeros((3, 1001), np.float32)
    s = WindowSample(wf, RATE, [Arrival("P", 5.0, tier="automatic")], negative_support="unknown")
    assert not has_supervision(s)
    assert targets_for(s).n_supervised == 0
    s.negative_support = "certified"
    assert has_supervision(s)
    s = WindowSample(wf, RATE, [Arrival("P", 5.0)], negative_support="unknown")
    assert has_supervision(s)
    assert has_supervision(WindowSample(wf, RATE, [], negative_support="unknown"), policy=LEGACY)
    assert not has_supervision(WindowSample(wf, RATE, [Arrival("P", 12.0)], negative_support="unknown"))


def test_policy_from_config_and_validation():
    assert LabelPolicy.from_config(None) is LEGACY
    assert LabelPolicy.from_config("masked") == MASKED
    p = LabelPolicy.from_config({"name": "masked", "supervised_halfwidth_s": 2.0, "ignored": 1})
    assert p.supervised_halfwidth_s == 2.0
    with pytest.raises(ValueError):
        LabelPolicy(name="soft")
    with pytest.raises(ValueError):
        LabelPolicy(sigma_s=0.0)
    with pytest.raises(ValueError):
        build_targets([], N, RATE, negative_support="maybe")
    with pytest.raises(ValueError):
        build_targets([], N, RATE, valid_samples=N + 1)


def test_arrival_schema_round_trip_and_validation():
    arr = [Arrival("S", 7.25, tier="reviewed", event_id="ev1", uncertainty_s=0.05), Arrival("P", 3.0)]
    text = arrivals_to_json(arr)
    back = arrivals_from_json(text)
    assert back == sorted(arr, key=lambda a: a.time_s)
    assert arrivals_from_json("") == [] and arrivals_from_json(None) == [] and arrivals_from_json(float("nan")) == []
    assert shift_arrivals(back, -1.0)[0].time_s == 2.0
    assert arrivals_from_columns(np.nan, 250.0, 50.0) == [Arrival("S", 5.0)]
    with pytest.raises(ValueError):
        Arrival("Pg", 1.0)
    with pytest.raises(ValueError):
        Arrival("P", 1.0, tier="guess")
    with pytest.raises(ValueError):
        Arrival("P", float("nan"))
    with pytest.raises(ValueError):
        arrivals_from_json("{}")


def test_window_sample_validation():
    with pytest.raises(ValueError):
        WindowSample(np.zeros((2, 10)), RATE)
    with pytest.raises(ValueError):
        WindowSample(np.zeros((3, 10)), RATE, negative_support="none")
    with pytest.raises(ValueError):
        WindowSample(np.zeros((3, 10)), RATE, valid_samples=11)
    with pytest.raises(ValueError):
        WindowSample(np.zeros((3, 10)), RATE, unknown_intervals=[(1.0, 0.5)])
    s = WindowSample(np.zeros((3, 10)), RATE, [Arrival("P", 0.05)], component_mask=(1, 0, 1), valid_samples=8)
    assert s.n_valid == 8 and s.duration_s == pytest.approx(0.1) and s.component_mask == (True, False, True)
    c = s.copy()
    c.arrivals.append(Arrival("S", 0.07))
    assert len(s.arrivals) == 1
