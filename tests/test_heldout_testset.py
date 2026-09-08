"""
Unit tests for scripts/build_heldout_testset.py parsers and the window
chooser, on synthetic inputs (no network).  Run with:  pytest tests/ -v
"""
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("obspy")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_heldout_testset as b  # noqa: E402
import heldout_testset_registry as reg  # noqa: E402


def _rec(cols):
    """Build a 96-byte JMA record from {1-based column: text}."""
    line = [" "] * 96
    for start, text in cols.items():
        for i, ch in enumerate(text):
            line[start - 1 + i] = ch
    return "".join(line)


def test_parse_jma_deck_converts_jst_and_flags_manual_vs_auto():
    # JMA hypocentre 2023-05-05 14:42:04.30 JST (= 05:42:04.30 UTC), 37 deg 31.00', 137 deg 16.00', 12 km, M6.5 (J)
    hypo = _rec({1: "J", 2: "2023", 6: "05", 8: "05", 10: "14", 12: "42", 14: "0430", 22: " 37", 25: "3100",
                   33: " 137", 37: "1600", 45: "  12 ", 53: "65", 55: "J", 96: "K"})
    # arrival: station SUZU  , day 05, IP at 14:42:10.50 JST, then ES at 42:14.20; col 92 blank = manual
    arr_manual = _rec({1: "_", 2: "SUZU  ", 14: "05", 16: "IP  ", 20: "14", 22: "42", 24: "1050", 28: "ES  ", 32: "42", 34: "1420", 88: "23", 90: "05"})
    arr_auto = _rec({1: "_", 2: "WAJM  ", 14: "05", 16: "P   ", 20: "14", 22: "42", 24: "2000", 88: "23", 90: "05", 92: "a", 96: "0"})
    text = "\n".join([hypo, arr_manual, arr_auto, "E" + " " * 95])
    picks, cats = b.parse_jma_deck(text, "test")
    assert len(cats) == 1 and abs(cats[0]["lat"] - (37 + 31 / 60)) < 1e-6 and abs(cats[0]["lon"] - (137 + 16 / 60)) < 1e-6
    assert cats[0]["mag"] == 6.5 and str(cats[0]["origin"]).startswith("2023-05-05 05:42:04.3")
    assert [p["phase"] for p in picks] == ["P", "S", "P"]
    assert [p["mode"] for p in picks] == ["manual", "manual", "automatic"]
    assert str(picks[0]["time"]).startswith("2023-05-05 05:42:10.5") and str(picks[1]["time"]).startswith("2023-05-05 05:42:14.2")
    assert picks[0]["onset"] == "impulsive" and picks[1]["onset"] == "emergent"
    assert picks[2]["time_weight"] == 0.0 and picks[2]["reference_ok"] is False and picks[0]["reference_ok"] is True


def test_parse_hypodd_pha_builds_absolute_pick_times():
    text = """# 2018 05 21 10 12 30.50 50.2400 12.4500 8.50 2.1 0.1 0.2 0.05 12345
KVC 1.234 1.0 P
KVC 2.100 0.5 S
NKC 1.500 1.0 P
# 2018 05 21 10 20 00.00 50.2500 12.4600 9.00 1.3 0.1 0.2 0.05 12346
KVC 1.300 1.0 P
"""
    picks, cats = b.parse_hypodd_pha(text, "wb", "WB")
    assert len(cats) == 2 and cats[0]["event"] == "12345" and cats[0]["mag"] == 2.1
    assert [p["station"] for p in picks] == ["WB.KVC", "WB.KVC", "WB.NKC", "WB.KVC"]
    assert str(picks[0]["time"]).startswith("2018-05-21 10:12:31.734")
    assert picks[1]["phase"] == "S" and picks[1]["time_weight"] == 0.5 and all(p["reference_ok"] for p in picks)
    bare, _ = b.parse_hypodd_pha(text, "wb", "")
    assert [p["station"] for p in bare][:2] == ["KVC", "KVC"] and bare[0]["network"] == ""


def test_choose_busiest_windows_is_deterministic_and_non_overlapping():
    seq = dict(windows=dict(kind="busiest", span=("2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), n_windows=2, hours=3))
    # 30 events at 05:00-05:30, 10 events at 20:00-20:10, 5 events at 06:30 (inside the first window)
    times = (["2020-01-01T05:%02d:00Z" % m for m in range(30)] + ["2020-01-01T20:%02d:00Z" % m for m in range(10)]
             + ["2020-01-01T06:3%d:00Z" % m for m in range(5)])
    cat = pd.DataFrame({"origin": pd.to_datetime(times, utc=True)})
    wins = b.choose_windows(seq, cat)
    assert len(wins) == 2
    assert str(wins[0]["t0"]).startswith("2020-01-01T0") and str(wins[1]["t0"]).startswith("2020-01-01T1") or str(wins[1]["t0"]).startswith("2020-01-01T2")
    assert (wins[1]["t0"] - wins[0]["t0"]) >= 3 * 3600
    assert "35 catalogue events" in wins[0]["rule"]
    wins2 = b.choose_windows(seq, cat)
    assert [str(w["t0"]) for w in wins] == [str(w["t0"]) for w in wins2]


def test_mainshock_windows_follow_the_notebook_rule():
    seq = dict(windows=dict(kind="mainshock", start_s=600, minutes=180), mainshock="2022-11-09T06:07:25Z")
    wins = b.choose_windows(seq, pd.DataFrame(columns=["origin"]))
    assert len(wins) == 1 and str(wins[0]["t0"]) == "2022-11-09T06:17:25.000000Z" and (wins[0]["t1"] - wins[0]["t0"]) == 3 * 3600


def test_registry_entries_are_complete_and_match_the_heldout_windows():
    import heldout_sequences as hs
    for s in reg.SEQUENCES:
        assert s["regime"] in {"msas", "vt", "swarm"} and s["suite"] in {"acceptance", "dev"}
        assert s["windows"]["kind"] in {"mainshock", "busiest"}
        if s["windows"]["kind"] == "mainshock":
            assert s["mainshock"] is not None
        else:
            assert "catalog" in s["windows"] and "span" in s["windows"]
        for p in s["picks"]:
            assert p["kind"] in {"fdsn_per_event", "fdsn_region", "usgs_phase_data", "jma_deck", "zenodo_pha"}
        # every registry sequence sits inside a held-out window or place of heldout_sequences.py
        flags = hs.spatial_hits([s["lat"]], [s["lon"]])
        assert flags.iloc[0].any(), s["key"]


def test_match_and_budget_functions_behave_like_the_notebook():
    import heldout_testset_score as sc
    from obspy import UTCDateTime
    t = UTCDateTime("2020-01-01T00:00:00")
    ref = [t, t + 10, t + 20]
    cand = [t + 0.2, t + 10.6, t + 30]          # second is outside the 0.5 s tolerance
    res, extra = sc.match(ref, cand)
    assert len(res) == 1 and abs(res[0] - 0.2) < 1e-9 and extra == 2
    sweep = pd.DataFrame([
        dict(sequence="s", weights="a", phase="P", thr=0.1, recall=0.8, emitted=100),
        dict(sequence="s", weights="a", phase="P", thr=0.5, recall=0.4, emitted=40),
        dict(sequence="s", weights="b", phase="P", thr=0.1, recall=0.9, emitted=120),
        dict(sequence="s", weights="b", phase="P", thr=0.5, recall=0.5, emitted=60),
    ])
    tab = sc.matched_budget(sweep, ["a", "b"], "P", "s", n_points=3)
    assert list(tab.picks_emitted) == [60, 80, 100]
    assert tab.loc[0, "a"] == pytest.approx(0.533, abs=1e-3) and tab.loc[0, "b"] == pytest.approx(0.5)
    store = {("X", "P"): [(t, 0.9), (t + 5, 0.2)]}
    assert sc.at_threshold(store, 0.3) == {("X", "P"): [t]}


def test_reference_from_uses_reference_ok_and_collapses_duplicates():
    import heldout_testset_score as sc
    from obspy import UTCDateTime
    t0 = UTCDateTime("2020-01-01T00:00:00"); t1 = t0 + 3600
    picks = pd.DataFrame({
        "station": ["IV.A", "IV.A", "IV.A", "IV.B"],
        "phase": ["P", "P", "S", "P"],
        "time": pd.to_datetime(["2020-01-01T00:10:00Z", "2020-01-01T00:10:00.2Z", "2020-01-01T00:10:05Z", "2020-01-01T00:20:00Z"], utc=True, format="ISO8601"),
        "mode": ["manual", "manual", "manual", "automatic"],
        "reference_ok": [True, True, True, False],
    })
    ref = sc.reference_from(picks, ["IV.A", "IV.B"], t0, t1)
    assert len(ref[("IV.A", "P")]) == 1 and len(ref[("IV.A", "S")]) == 1 and ("IV.B", "P") not in ref


def test_jma_second_phase_rolls_into_next_hour():
    hypo = _rec({1: "J", 2: "2023", 6: "05", 8: "05", 10: "14", 12: "59", 14: "3282", 22: " 37", 25: "3100",
                 33: " 137", 37: "1600", 45: "  12 ", 53: "35", 55: "D", 96: "K"})
    arr = _rec({1: "_", 2: "N.TOYH", 14: "05", 16: "P   ", 20: "14", 22: "59", 24: "4500", 28: "S   ", 32: "00", 34: "0017", 88: "23", 90: "05"})
    picks, _ = b.parse_jma_deck("\n".join([hypo, arr, "E" + " " * 95]), "t")
    assert str(picks[0]["time"]).startswith("2023-05-05 05:59:45") and str(picks[1]["time"]).startswith("2023-05-05 06:00:00.17")
