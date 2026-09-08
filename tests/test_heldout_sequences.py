"""
Unit tests for scripts/heldout_sequences.py (task 1 of the 2026-09-07
generalization audit): window membership, the whole-year hold-out, the
fail-closed exclusion loader, and the manifest check.

These run without SeisBench or the lab cache.  Run with:  pytest tests/ -v
"""
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import heldout_sequences as hs  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows, columns=["trace_name", hs.TIME_COL, hs.LAT_COL, hs.LON_COL])


def test_gc_distance_is_symmetric_and_zero_on_self():
    assert hs.gc_distance_deg(42.83, 13.11, 42.83, 13.11) == pytest.approx(0.0)
    d1 = hs.gc_distance_deg(42.83, 13.11, 43.83, 13.11)
    d2 = hs.gc_distance_deg(43.83, 13.11, 42.83, 13.11)
    assert d1 == pytest.approx(d2) and d1 == pytest.approx(1.0, abs=1e-6)


def test_norcia_mainshock_window_hits_inside_and_not_outside():
    df = _frame([
        ("in_time_in_space", "2016-10-30T07:00:00", 42.9, 13.2),
        ("in_time_out_space", "2016-10-30T07:00:00", 45.0, 13.2),   # 2.2 deg away
        ("out_time_in_space", "2016-10-30T09:00:00", 42.9, 13.2),   # after the 2h window
        ("in_sequence_span", "2016-12-15T00:00:00", 42.9, 13.2),
    ])
    flags = hs.flag_rows(df)
    got = dict(zip(df.trace_name, flags.window))
    assert got["in_time_in_space"] == "norcia_2016_mainshock;norcia_2016_sequence"
    assert got["in_time_out_space"] == ""
    assert got["out_time_in_space"] == "norcia_2016_sequence"
    assert got["in_sequence_span"] == "norcia_2016_sequence"


def test_kaikoura_uses_two_degree_radius_and_southern_latitude():
    df = _frame([
        ("near", "2016-11-13T12:00:00", -42.69 + 1.8, 173.02),
        ("far", "2016-11-13T12:00:00", -42.69 + 2.2, 173.02),
    ])
    flags = hs.flag_rows(df)
    got = dict(zip(df.trace_name, flags.window))
    assert got["near"] == "kaikoura_2016_mainshock;kaikoura_2016_sequence"
    assert got["far"] == ""


def test_ridgecrest_and_monroe_windows():
    df = _frame([
        ("ridgecrest_day2", "2019-07-07T00:00:00", 35.8, -117.5),
        ("ridgecrest_late", "2019-09-01T00:00:00", 35.8, -117.5),
        ("monroe", "2019-07-15T00:00:00", 47.9, -122.0),
        ("monroe_before", "2019-07-11T00:00:00", 47.9, -122.0),
    ])
    got = dict(zip(df.trace_name, hs.flag_rows(df).window))
    assert got == {"ridgecrest_day2": "ridgecrest_2019", "ridgecrest_late": "",
                   "monroe": "monroe_2019", "monroe_before": ""}


def test_year_holdout_flags_2016_and_2021_only_and_marks_unverifiable():
    df = _frame([
        ("y2016", "2016-01-01T00:00:00", 0.0, 0.0),
        ("y2021", "2021-12-31T23:59:59", 0.0, 0.0),
        ("y2017", "2017-01-01T00:00:00", 0.0, 0.0),
        ("no_time", None, 0.0, 0.0),
        ("no_loc", "2016-05-05T00:00:00", None, None),
    ])
    flags = hs.flag_rows(df)
    assert flags.year_holdout.tolist() == [True, True, False, False, True]
    assert flags.verifiable.tolist() == [True, True, True, False, False]
    assert hs.holdout_year_mask(df[hs.TIME_COL]).tolist() == [True, True, False, False, True]


def test_flag_rows_without_fingerprint_columns_is_all_unverifiable():
    df = pd.DataFrame({"trace_name": ["a", "b"]})
    flags = hs.flag_rows(df)
    assert (~flags.verifiable).all() and (flags.window == "").all() and (~flags.year_holdout).all()


def test_load_sequence_exclusions_fails_closed_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        hs.load_sequence_exclusions(tmp_path / "missing.csv")
    assert hs.load_sequence_exclusions(tmp_path / "missing.csv", required=False) == {}


def test_load_sequence_exclusions_groups_by_dataset(tmp_path):
    p = tmp_path / "excl.csv"
    pd.DataFrame({"dataset": ["stead", "stead", "pnw"], "trace_name": ["t1", "t2", "t3"],
                  "chunk": ["", "", ""], "window": ["kaikoura_2016_sequence"] * 3}).to_csv(p, index=False)
    excl = hs.load_sequence_exclusions(p)
    assert excl == {"stead": frozenset({"t1", "t2"}), "pnw": frozenset({"t3"})}


def test_check_manifest_counts_violations():
    excl = {"stead": frozenset({"bad"})}
    man = pd.DataFrame({
        "dataset_name": ["stead", "stead", "pnw"],
        "trace_name": ["bad", "ok", "x"],
        hs.TIME_COL: ["2016-11-20T00:00:00", "2018-01-01T00:00:00", None],
        hs.LAT_COL: [-42.7, 0.0, 0.0],
        hs.LON_COL: [173.0, 0.0, 0.0],
    })
    rep = hs.check_manifest(man, excl)
    assert rep == dict(n_rows=3, n_excluded_present=1, n_in_window=1, n_year_holdout=1, n_year_unverifiable=1)


def test_windows_csv_roundtrip_matches_definitions():
    wf = hs.windows_frame()
    assert list(wf.name) == hs.WINDOW_NAMES
    assert set(wf.columns) >= {"name", "lat", "lon", "radius_deg", "start", "end"}
