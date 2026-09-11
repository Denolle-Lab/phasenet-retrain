"""#38A calibration protocol: deterministic day selection, exposure, thresholds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibration_protocol as cp  # noqa: E402


def availability(n_stations=60, days=("2022-01-05", "2022-01-06", "2022-01-07", "2022-07-01")):
    rows = []
    for i in range(n_stations):
        for d in days:
            rows.append(dict(station=f"XX.S{i:03d}", day=d, region_class="dense_local",
                             instrument_class="broadband", season="DJF" if d.startswith("2022-01") else "JJA",
                             power_percentile=(i * 7 + int(d[-1])) % 100, role="unassigned"))
    return pd.DataFrame(rows)


def test_selection_is_deterministic_and_model_free():
    av = availability()
    a = cp.select_calibration_days(av, fraction=0.5)
    b = cp.select_calibration_days(av, fraction=0.5)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["condition"]) <= {"quiet", "disturbed"}
    assert (a["role"] == "calibration").all()
    # the middle band of the power percentile is never used
    assert ((a["power_percentile"] < 30) | (a["power_percentile"] > 70)).all()
    # a different tag gives a different, not a shifted, draw
    c = cp.select_calibration_days(av, fraction=0.5, seed_tag="other")
    assert set(c["key"]) != set(a["key"])


def test_roles_and_exclusions_are_respected():
    av = availability()
    av.loc[0, "role"] = "training"
    excluded = {cp.station_day_key(av.station[1], av.day[1])}
    out = cp.select_calibration_days(av, fraction=1.0, excluded_keys=excluded)
    assert cp.station_day_key(av.station[0], av.day[0]) not in set(out["key"])
    assert cp.station_day_key(av.station[1], av.day[1]) not in set(out["key"])
    with pytest.raises(ValueError):
        cp.select_calibration_days(av.drop(columns=["season"]))
    bad = av.copy(); bad.loc[5, "region_class"] = "moon"   # an eligible row
    with pytest.raises(ValueError):
        cp.select_calibration_days(bad)


def test_exposure_check_flags_thin_strata():
    av = availability(n_stations=60)
    days = cp.select_calibration_days(av, fraction=1.0)
    table = cp.exposure_check(days)
    assert set(table.columns) >= {"region_class", "condition", "n_stations", "n_station_days", "exposure_ok"}
    thin = cp.exposure_check(days[days.station.isin(days.station.unique()[:5])])
    assert not thin["exposure_ok"].any()


def test_rates_bootstrap_and_operating_threshold():
    per_day = pd.DataFrame({"key": [f"k{i}" for i in range(40)], "unmatched": np.r_[np.full(20, 10.0), np.full(20, 30.0)]})
    rate, (lo, hi) = cp.block_bootstrap_rate(per_day, n_boot=500)
    assert rate == pytest.approx(20.0) and lo < 20.0 < hi
    assert cp.unmatched_rate(pd.DataFrame({"matched": [True, False, False]}), 2.0) == 1.0
    sweep = pd.DataFrame([
        dict(phase="P", threshold=0.1, rate=80.0, ci_low=70.0, ci_high=90.0),
        dict(phase="P", threshold=0.2, rate=48.0, ci_low=40.0, ci_high=56.0),
        dict(phase="P", threshold=0.3, rate=30.0, ci_low=25.0, ci_high=36.0),
        dict(phase="S", threshold=0.3, rate=70.0, ci_low=60.0, ci_high=80.0),
    ])
    op = cp.operating_threshold(sweep, 50.0, "P")
    assert op["threshold"] == 0.3   # 0.2 has an upper bound above budget
    assert cp.operating_threshold(sweep, 50.0, "S") is None
    assert cp.precision_ok(30.0, (25.0, 35.0), 50.0)
    assert not cp.precision_ok(30.0, (10.0, 50.0), 50.0)
    assert cp.Protocol().to_dict()["version"] == cp.PROTOCOL_VERSION
