"""Offline tests for scripts/certify_evaluability.py (#37A) on synthetic case directories."""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("obspy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import certify_evaluability as ce  # noqa: E402
import evaluation_policy as policy  # noqa: E402

T0 = pd.Timestamp("2021-01-01T00:00:00Z")
WIN_S = 600
ROLES = {"synth_reg": "regression", "synth_dev": "dev", "synth_acc": "acceptance"}
FAST = dict(min_refs_per_phase=5, min_events_3sta=10)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Synthetic policy with three keys, a temporary access log, a data root."""
    pol = tmp_path / "suites.json"
    pol.write_text(json.dumps({"version": 1, "acceptance_status": "provisional", "roles": ROLES}))
    monkeypatch.setattr(policy, "POLICY_PATH", pol)
    monkeypatch.setattr(policy, "ACCESS_LOG", tmp_path / "access.jsonl")
    root = tmp_path / "heldout"
    root.mkdir()
    return root


def pick(event, station, phase, offset_s, mode="manual", ok=None, source="TEST"):
    network = station.split(".")[0] if "." in station else ""
    return dict(sequence="synth", event=str(event), origin=T0 + pd.Timedelta(seconds=offset_s - 5), mag=2.0,
                station=station, channel="HHZ", phase=phase, time=T0 + pd.Timedelta(seconds=offset_s),
                mode=mode, status="unknown", method="", agency="", time_weight=1.0, onset="", uncertainty=np.nan,
                network=network, source=source, reference_ok=(mode == "manual") if ok is None else ok)


def station(name, fetched=True, rate=100.0, band="HH"):
    net, code = name.split(".")
    return dict(station=name, network=net, code=code, band=band, rate=rate, lat=0.0, lon=0.0, elev_m=0.0,
                km=10.0, route="x", P=0, S=0, picks=0, fetched=fetched)


def mseed(path, name, channels, rate=100.0):
    """channels: {channel: [(start_s, end_s), ...]} covered segments from T0."""
    from obspy import Stream, Trace, UTCDateTime
    net, sta = name.split(".")
    st = Stream()
    for chan, segments in channels.items():
        for a, b in segments:
            tr = Trace(np.zeros(int(round((b - a) * rate)), dtype=np.int32))
            tr.stats.network, tr.stats.station, tr.stats.channel = net, sta, chan
            tr.stats.sampling_rate = rate
            tr.stats.starttime = UTCDateTime(T0.to_pydatetime()) + a
            st += tr
    st.write(str(path), format="MSEED")


def make_case(root, key, picks, stations, catalog=None, station_map=None, waveforms=None):
    d = root / key
    d.mkdir()
    t1 = T0 + pd.Timedelta(seconds=WIN_S)
    pd.DataFrame([dict(t0=T0.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), t1=t1.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       rule="synthetic")]).to_csv(d / "windows.csv", index=False)
    pd.DataFrame(picks).to_parquet(d / "picks.parquet", index=False)
    events = {p["event"]: p["origin"] for p in picks}
    cat = catalog if catalog is not None else [dict(event=e, origin=o, lat=0.0, lon=0.0, depth_km=5.0, mag=2.0, source="TEST")
                                                for e, o in events.items()]
    pd.DataFrame(cat, columns=["event", "origin", "lat", "lon", "depth_km", "mag", "source"]).to_parquet(
        d / "catalog.parquet", index=False)
    pd.DataFrame(stations).to_csv(d / "stations.csv", index=False)
    if station_map is not None:
        pd.DataFrame(station_map, columns=["code", "station", "ambiguous"]).to_csv(d / "station_map.csv", index=False)
    (d / "manifest.json").write_text(json.dumps({"key": key, "label": key}))
    if waveforms:
        (d / "waveforms").mkdir()
        tag = ce.window_tag(T0, t1)
        for name, (channels, rate) in waveforms.items():
            band = next(s["band"] for s in stations if s["station"] == name)
            mseed(d / "waveforms" / f"{name}__{band}__{tag}.mseed", name, channels, rate)
    return d


def certify(root, key, rules=None):
    table = ce.run(root, keys=[key], rules=rules, log_path=root / "log.jsonl")
    return table.iloc[0]


def test_resolution_categories_tiers_and_dedup(env, monkeypatch):
    picks = [
        pick(1, "NET.A", "P", 10.0), pick(2, "NET.A", "P", 20.0), pick(3, "NET.A", "P", 20.3),   # 20.3 collapses into 20.0
        pick(1, "NET.A", "S", 15.0),
        pick(4, "NET.A", "P", 30.0, mode="unknown", ok=True, source="ISC"),
        pick(5, "NET.A", "P", 40.0, mode="automatic"),
        pick(1, "NET.B", "P", 11.0), pick(1, "NET.B", "S", 16.0),          # resolved, not fetched
        pick(1, "NET.X", "P", 12.0), pick(2, "NET.X", "P", 22.0),          # ambiguous code, quarantined
        pick(1, "A", "P", 13.0),                                           # bare code equal to a fetched code
        pick(9, "NET.A", "P", WIN_S + 100.0),                              # outside the window
    ]
    stations = [station("NET.A"), station("NET.B", fetched=False), station("NET.X")]
    make_case(env, "synth_dev", picks, stations, station_map=[dict(code="X", station="NET.X", ambiguous="NET.X;OTH.X")])
    row = certify(env, "synth_dev")
    assert row["picks_total"] == 12 and row["picks_in_windows"] == 11
    assert (row["res_fetched"], row["res_unfetched"], row["res_ambiguous"], row["res_unresolved"]) == (6, 2, 2, 1)
    assert row["res_unresolved_code_matches_fetched"] == 1 and row["top_unresolved_codes"] == "A:1"
    assert row["ref_fetched_P_manual_ok"] == 3 and row["ref_fetched_S_manual_ok"] == 1
    assert row["ref_fetched_P_unknown_ok"] == 1 and row["ref_fetched_P_automatic_rejected"] == 1
    assert row["ref_all_P_manual_ok"] == 7 and row["ref_all_P_unknown_ok"] == 1     # B, X and bare A count on all stations
    assert row["ref_fetched_P_dedup"] == 3 and row["ref_covered_P_dedup"] == 3      # 2 manual after dedup + 1 unknown_ok
    assert row["ref_rm_covered_P"] == 2 and row["bulletin_unknown_mode_reviewed"] == False  # noqa: E712
    assert not row["waveforms_measured"] and row["stations_covered"] == 2
    assert row["stations_covered_with_refs"] == 1 and row["not_evaluable_reason"] == "single_station"
    assert row["verdict"] == "neither" and row["certified_by"] == "37A" and not row["provisional"]
    monkeypatch.setitem(ce.BULLETIN_REVIEWED, "synth_dev", True)
    row = certify(env, "synth_dev")
    assert row["ref_rm_covered_P"] == 3 and row["bulletin_unknown_mode_reviewed"] == True  # noqa: E712


def test_three_station_event_support_and_verdicts(env):
    picks = []
    for e in range(12):                                     # 12 events with P on three fetched stations
        for s in ("NET.A", "NET.B", "NET.C"):
            picks.append(pick(e, s, "P", 20.0 + 10.0 * e))
    for e in range(12, 15):                                 # 3 events with P on two stations
        for s in ("NET.A", "NET.B"):
            picks.append(pick(e, s, "P", 20.0 + 10.0 * e))
    for e in range(6):
        picks.append(pick(e, "NET.A", "S", 25.0 + 10.0 * e))
    stations = [station("NET.A"), station("NET.B"), station("NET.C"), station("NET.D")]
    make_case(env, "synth_reg", picks, stations)
    row = certify(env, "synth_reg", FAST)
    assert row["events_catalog_in_windows"] == 15 and row["events_3sta_P_covered_rm"] == 12
    assert row["events_with_refs_fetched"] == 15 and row["stations_covered_with_rm_refs"] == 3
    assert row["ref_rm_covered_P"] == 42 and row["ref_rm_covered_S"] == 6
    assert row["verdict"] == "both" and row["pick_scoring_eligible"] and row["network_event_scoring_eligible"]
    assert row["ref_per_station_hour_fetched"] == pytest.approx(48 / (4 * WIN_S / 3600))
    row = certify(env, "synth_reg", dict(FAST, min_events_3sta=13))
    assert row["verdict"] == "pick_scoring" and "13" in row["network_event_scoring_reason"]
    row = certify(env, "synth_reg", dict(FAST, min_refs_per_phase=7))
    assert row["verdict"] == "network_event_scoring" and row["pick_scoring_reason"] == "S 6 < 7"
    assert row["pick_scoring_P_ok"] and not row["pick_scoring_S_ok"]


def test_zero_overlap_single_station_and_no_waveforms(env):
    picks = [pick(1, "NET.B", "P", 10.0), pick(1, "NET.B", "S", 15.0)]
    make_case(env, "synth_dev", picks, [station("NET.A"), station("NET.B", fetched=False)])
    row = certify(env, "synth_dev", FAST)
    assert row["not_evaluable_reason"] == "zero_overlap" and row["verdict"] == "neither"
    assert row["pick_scoring_reason"] == row["network_event_scoring_reason"] == "zero_overlap"
    assert row["res_unfetched"] == 2 and row["ref_all_P_manual_ok"] == 1

    picks = [pick(e, "NET.A", ph, 10.0 * e + dt) for e in range(1, 8) for ph, dt in (("P", 0.0), ("S", 3.0))]
    make_case(env, "synth_reg", picks, [station("NET.A"), station("NET.B")])
    row = certify(env, "synth_reg", FAST)
    assert row["not_evaluable_reason"] == "single_station" and row["verdict"] == "neither"
    assert row["ref_rm_covered_P"] == 7 and row["stations_covered"] == 2 and row["stations_covered_with_refs"] == 1

    make_case(env, "synth_acc", picks, [station("NET.A", fetched=False)])
    row = certify(env, "synth_acc", FAST)
    assert row["not_evaluable_reason"] == "no_waveforms" and row["stations_fetched"] == 0


def test_waveform_coverage_gaps_and_components(env):
    full = [(0, WIN_S)]
    waveforms = {
        "NET.A": ({"HHE": full, "HHN": full, "HHZ": full}, 100.0),
        "NET.B": ({"HHE": full, "HHN": full, "HHZ": [(0, 300), (320, WIN_S)]}, 100.0),   # 20 s gap on Z
        "NET.C": ({"HHE": full, "HHZ": full}, 50.0),                                     # two components only
    }
    picks = [pick(1, "NET.A", "P", 100.0), pick(1, "NET.A", "S", 105.0),
             pick(1, "NET.B", "P", 100.0), pick(2, "NET.B", "P", 310.0),                 # second is inside the gap
             pick(1, "NET.C", "P", 100.0), pick(1, "NET.D", "P", 100.0)]
    stations = [station("NET.A"), station("NET.B"), station("NET.C", rate=50.0), station("NET.D")]   # D: file missing
    make_case(env, "synth_dev", picks, stations, waveforms=waveforms)
    row = certify(env, "synth_dev", FAST)
    t = pd.read_csv(env / "synth_dev" / "evaluability_stations.csv").set_index("station")
    assert bool(row["waveforms_measured"]) and row["files_expected"] == 4 and row["files_present"] == 3
    assert t.loc["NET.A", "covered_3c_s"] == pytest.approx(WIN_S, abs=0.02) and t.loc["NET.A", "gap_count"] == 0
    assert t.loc["NET.B", "covered_3c_s"] == pytest.approx(WIN_S - 20, abs=0.02)
    assert t.loc["NET.B", "gap_count"] == 1 and t.loc["NET.B", "gap_s"] == pytest.approx(20, abs=0.02)
    assert t.loc["NET.B", "ref_P_window"] == 2 and t.loc["NET.B", "ref_P_covered"] == 1
    assert t.loc["NET.C", "covered_any_s"] == pytest.approx(WIN_S, abs=0.05) and t.loc["NET.C", "covered_3c_s"] == 0
    assert not t.loc["NET.C", "covered"] and t.loc["NET.C", "n_channels"] == 2
    assert t.loc["NET.D", "measured"] and not t.loc["NET.D", "covered"] and t.loc["NET.D", "gap_s"] == WIN_S
    assert t["read_error"].fillna("").eq("").all()
    assert row["stations_covered"] == 2 and row["stations_rate_lt_100hz"] == 1
    assert row["ref_fetched_P_dedup"] == 5 and row["ref_covered_P_dedup"] == 2 and row["ref_rm_covered_P"] == 2
    # gaps are counted in three-component coverage: B 20 s, C (two components) and D (no file) the whole window
    assert row["gap_count_total"] == 3 and row["gap_s_total"] == pytest.approx(2 * WIN_S + 20, abs=0.02)
    assert row["ref_per_covered_hour"] == pytest.approx(3 / ((2 * WIN_S - 20) / 3600), rel=1e-3)


def test_unreadable_mseed_counts_as_a_full_gap_and_does_not_abort(env):
    full = [(0, WIN_S)]
    waveforms = {"NET.A": ({"HHE": full, "HHN": full, "HHZ": full}, 100.0),
                 "NET.B": ({"HHE": full, "HHN": full, "HHZ": full}, 100.0)}
    picks = [pick(1, "NET.A", "P", 100.0), pick(1, "NET.B", "P", 100.0)]
    make_case(env, "synth_dev", picks, [station("NET.A"), station("NET.B")], waveforms=waveforms)
    # corrupt B's file after it was written
    bad = next((env / "synth_dev" / "waveforms").glob("NET.B__*.mseed"))
    bad.write_bytes(b"not a miniseed record at all")
    row = certify(env, "synth_dev", FAST)
    t = pd.read_csv(env / "synth_dev" / "evaluability_stations.csv").set_index("station")
    assert t.loc["NET.A", "covered"] and (pd.isna(t.loc["NET.A", "read_error"]) or t.loc["NET.A", "read_error"] == "")
    assert not t.loc["NET.B", "covered"] and t.loc["NET.B", "gap_s"] == WIN_S
    assert isinstance(t.loc["NET.B", "read_error"], str) and t.loc["NET.B", "read_error"]
    assert row["stations_covered"] == 1


def test_acceptance_case_gets_only_a_provisional_row_and_access_is_logged(env):
    picks = [pick(e, s, "P", 20.0 + 10.0 * e) for e in range(12) for s in ("NET.A", "NET.B", "NET.C")]
    picks += [pick(e, "NET.A", "S", 25.0 + 10.0 * e) for e in range(6)]
    make_case(env, "synth_acc", picks, [station("NET.A"), station("NET.B"), station("NET.C")])
    log = env / "log.jsonl"
    table = ce.run(env, keys=["synth_acc"], rules=FAST, log_path=log, out=env / "out.csv")
    row = table.iloc[0]
    assert row["role"] == "acceptance" and bool(row["provisional"]) and row["certified_by"] == ""
    assert row["verdict"] == "both"                      # computed, but not certified
    written = pd.read_csv(env / "out.csv")
    assert bool(written.loc[0, "provisional"]) and pd.isna(written.loc[0, "certified_by"])
    assert "min_refs_per_phase=5" in written.loc[0, "rule_params"]
    entry = json.loads(log.read_text().splitlines()[-1])
    assert entry["operation"] == "reference_qa" and entry["role"] == "acceptance" and entry["models"] == {}
    assert entry["settings"]["checkpoint"] == "37A" and entry["settings"]["min_events_3sta"] == 10
    assert entry["id"] == row["access_id"]


def test_unknown_key_is_rejected_before_any_access(env):
    with pytest.raises(ValueError):
        ce.run(env, keys=["nope"], log_path=env / "log.jsonl")
    assert not (env / "log.jsonl").exists()


def test_interval_helpers():
    assert ce._union([[0, 1], [1.05, 2], [3, 4]], tol=0.1) == [[0, 2], [3, 4]]
    assert ce._intersect([[0, 10]], [[2, 3], [8, 12]]) == [[2, 3], [8, 10]]
    assert ce._complement([[1, 2], [3, 4]], 0, 5) == [[0, 1], [2, 3], [4, 5]]
    assert list(ce._inside([0.5, 1.5, 2.5, 9.0], [[1, 2], [2.4, 3]])) == [False, True, True, False]
    assert ce.dedup_count([10.0, 10.3, 10.9, 20.0], 0.5) == 3
