"""Behavioural tests for scripts/event_association.py (issue #36, checkpoint 36A).

Synthetic station networks, catalogues and picks only: no network, no torch,
no seisbench, no pyocto, no model inference. Run with
    python -m pytest tests/test_event_association.py -q
"""
import ast
import builtins
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import event_association as ea  # noqa: E402

T0 = pd.Timestamp("2020-10-30T12:00:00Z")
LAT0, LON0 = 38.0, 26.5
PROJ = ea.LocalProjection(LAT0, LON0)
STATION_XY = [(-40, -30), (35, -25), (-30, 40), (45, 35), (0, 5), (-10, -45)]


def test_config(**changes) -> ea.AssociatorConfig:
    """Small grid: the same rules as the shipped configs, cheap enough for tests."""
    base = dict(region="test", p_velocity=6.0, s_velocity=3.4, time_tolerance_s=1.0, spatial_tolerance_km=8.0,
                location_tolerance_km=2.0, association_cutoff_km=200.0, depth_range_km=[0.0, 20.0], margin_km=20.0,
                min_picks=6, min_p_picks=3, min_s_picks=0, n_p_and_s_picks=2)
    base.update(changes)
    return ea.AssociatorConfig(**base)


test_config.__test__ = False


def stations(n=len(STATION_XY)) -> pd.DataFrame:
    rows = []
    for i, (x, y) in enumerate(STATION_XY[:n]):
        lat, lon = PROJ.to_latlon(x, y)
        rows.append(dict(station=f"XX.S{i:02d}", lat=float(lat), lon=float(lon), elev_m=float(50 * i), rate=100.0))
    return pd.DataFrame(rows)


def catalogue(offsets_s=(0, 600, 1800, 3600), seed=0, mags=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k, s in enumerate(offsets_s):
        x, y, z = rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(4, 14)
        lat, lon = PROJ.to_latlon(x, y)
        mag = float(mags[k]) if mags is not None else float(rng.uniform(1.2, 4.5))
        rows.append(dict(event=f"ref{k:02d}", origin=T0 + pd.Timedelta(seconds=float(s)), lat=float(lat),
                         lon=float(lon), depth_km=float(z), mag=mag, source="synthetic"))
    return pd.DataFrame(rows)


def synth_picks(cat, sta, cfg, noise_s=0.05, n_noise=0, seed=1, drop_stations=()) -> pd.DataFrame:
    """Arrivals from every catalogue event with homogeneous velocities, plus noise picks."""
    rng = np.random.default_rng(seed)
    sx, sy = PROJ.to_km(sta["lat"].to_numpy(float), sta["lon"].to_numpy(float))
    sz = -sta["elev_m"].to_numpy(float) / 1000.0
    rows = []
    for r in cat.itertuples():
        ex, ey = PROJ.to_km(r.lat, r.lon)
        for i, name in enumerate(sta["station"]):
            if name in drop_stations:
                continue
            d = float(np.sqrt((ex - sx[i]) ** 2 + (ey - sy[i]) ** 2 + (r.depth_km - sz[i]) ** 2))
            for phase, v in (("P", cfg.p_velocity), ("S", cfg.s_velocity)):
                t = r.origin + pd.Timedelta(seconds=d / v + float(rng.normal(0, noise_s)))
                rows.append(dict(pick_id=f"{r.event}|{name}|{phase}", station=name, phase=phase, time=t,
                                 score=0.9, matched_event=r.event))
    span = float(max(1.0, (cat["origin"].max() - T0).total_seconds() + 600))
    for k in range(n_noise):
        name = sta["station"].iloc[int(rng.integers(len(sta)))]
        phase = "PS"[int(rng.integers(2))]
        rows.append(dict(pick_id=f"noise{k:03d}|{name}|{phase}", station=name, phase=phase,
                         time=T0 + pd.Timedelta(seconds=float(rng.uniform(0, span))), score=0.35, matched_event=None))
    out = pd.DataFrame(rows)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.sort_values("time").reset_index(drop=True)


def events_frame(rows) -> pd.DataFrame:
    """Predicted events straight from (event_idx, seconds after T0, x_km, y_km, depth) tuples."""
    out = []
    for idx, s, x, y, z in rows:
        lat, lon = PROJ.to_latlon(x, y)
        out.append(dict(event_idx=idx, time=T0 + pd.Timedelta(seconds=float(s)), lat=float(lat), lon=float(lon),
                        depth_km=float(z), n_picks=8, n_p=4, n_s=4, n_stations=4, n_p_and_s=4, misfit_s=0.1,
                        associator="fixture", config_sha256="x"))
    return pd.DataFrame(out, columns=ea.EVENT_COLUMNS)


def windows_frame(pairs) -> pd.DataFrame:
    return pd.DataFrame([dict(t0=(T0 + pd.Timedelta(seconds=a)).isoformat(),
                              t1=(T0 + pd.Timedelta(seconds=b)).isoformat()) for a, b in pairs])


# ── module hygiene ───────────────────────────────────────────────────────────

def test_module_imports_no_torch_seisbench_or_pyocto_at_import_time():
    tree = ast.parse((REPO_ROOT / "scripts" / "event_association.py").read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"torch", "seisbench", "pyocto"}


# ── configuration ────────────────────────────────────────────────────────────

def test_config_hash_changes_with_any_parameter():
    cfg = test_config()
    changes = dict(region="other", version="2", velocity_model="crust1", p_velocity=6.1, s_velocity=3.5,
                   time_tolerance_s=1.6, spatial_tolerance_km=9.0, location_tolerance_km=2.5,
                   association_cutoff_km=260.0, depth_range_km=[0.0, 61.0], margin_km=51.0, time_before_s=301.0,
                   min_picks=7, min_p_picks=4, min_s_picks=1, n_p_and_s_picks=3, n_picks_p_and_s_policy="ignore",
                   layers=[dict(depth_km=0.0, vp=5.5, vs=3.2)])
    seen = {cfg.sha256}
    for name, value in changes.items():
        kw = {name: value}
        if name == "velocity_model":
            kw["layers"] = [dict(depth_km=0.0, vp=5.5, vs=3.2)]
        other = cfg.replace(**kw)
        assert other.sha256 != cfg.sha256, name
        seen.add(other.sha256)
    assert len(seen) == len(changes) + 1


def test_notes_and_versioned_note_do_not_change_the_hash():
    cfg = test_config()
    assert cfg.replace(notes="anything", versioned="other").sha256 == cfg.sha256


def test_config_roundtrip_and_tamper_detection(tmp_path):
    cfg = test_config(notes="proposal")
    path = cfg.save(tmp_path / "test.json")
    assert ea.AssociatorConfig.load(path).sha256 == cfg.sha256
    d = json.loads(path.read_text())
    assert d["sha256"] == cfg.sha256
    d["time_tolerance_s"] = 99.0
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="does not match content hash"):
        ea.AssociatorConfig.load(path)
    edited = ea.AssociatorConfig.load(path, verify=False)
    assert edited.time_tolerance_s == 99.0 and edited.sha256 != cfg.sha256


def test_config_rejects_unknown_keys_and_invalid_values(tmp_path):
    with pytest.raises(ValueError, match="Unknown AssociatorConfig keys"):
        ea.AssociatorConfig.from_dict(dict(region="x", vp=6.0))
    for bad in (dict(p_velocity=3.0, s_velocity=6.0), dict(time_tolerance_s=0.0), dict(min_picks=0),
                dict(depth_range_km=[10.0, 1.0]), dict(n_picks_p_and_s_policy="maybe"),
                dict(velocity_model="named", layers=[])):
        with pytest.raises(ValueError):
            test_config(**bad)


def test_shipped_regime_configs_exist_load_and_are_self_consistent():
    for regime in ea.REGIMES:
        cfg = ea.load_config(regime)
        assert cfg.region == regime
        stored = json.loads(ea.config_path(regime).read_text())
        assert stored["sha256"] == cfg.sha256
        assert cfg.versioned and "config_sha256" in cfg.versioned
        assert cfg.notes.startswith("Proposal")
    hashes = {ea.load_config(r).sha256 for r in ea.REGIMES}
    assert len(hashes) == len(ea.REGIMES)


def test_write_default_configs_is_reproducible(tmp_path):
    paths = ea.write_default_configs(tmp_path)
    assert {p.name for p in paths} == {f"{r}.json" for r in ea.REGIMES}
    for p in paths:
        assert p.read_text() == (ea.CONFIG_DIR / p.name).read_text()


# ── the synthetic associator (test double) ───────────────────────────────────

def test_synthetic_associator_recovers_the_synthetic_catalogue():
    cfg, sta, cat = test_config(), stations(), catalogue()
    picks = synth_picks(cat, sta, cfg, n_noise=20)
    events, assignments = ea.SyntheticAssociator(cfg).associate(picks, sta)
    assert len(events) == len(cat)
    match = ea.match_events(events, cat)
    assert match.n_matched == len(cat)
    assert match.pairs["dt_s"].abs().max() < 2.0
    assert match.pairs["dist_km"].max() < 15.0
    # noise picks are not associated into the events
    assigned = set(assignments["pick_id"])
    assert not any(p.startswith("noise") for p in assigned)
    assert set(events["config_sha256"]) == {cfg.sha256}
    assert (events["n_stations"] == len(sta)).all()
    assert set(assignments.columns) == set(ea.ASSIGNMENT_COLUMNS)


def test_synthetic_associator_with_missing_stations():
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0, 900))
    dropped = ("XX.S00", "XX.S05")
    picks = synth_picks(cat, sta, cfg, drop_stations=dropped)
    events, assignments = ea.SyntheticAssociator(cfg).associate(picks, sta)
    assert len(events) == 2
    assert not set(assignments["station"]) & set(dropped)
    assert (events["n_stations"] == len(sta) - len(dropped)).all()
    assert ea.match_events(events, cat).n_matched == 2


def test_too_few_picks_yields_no_event():
    cfg, sta, cat = test_config(min_picks=10), stations(), catalogue(offsets_s=(0,))
    picks = synth_picks(cat, sta, cfg, drop_stations=("XX.S00", "XX.S01", "XX.S02"))
    events, assignments = ea.SyntheticAssociator(cfg).associate(picks, sta)
    assert len(events) == 0 and len(assignments) == 0
    assert list(events.columns) == ea.EVENT_COLUMNS


def test_p_and_s_policy_is_enforced_and_can_be_ignored():
    cfg, sta, cat = test_config(min_picks=5, n_p_and_s_picks=4), stations(), catalogue(offsets_s=(0,))
    picks = synth_picks(cat, sta, cfg)
    picks = picks[(picks["phase"] == "P") | picks["station"].isin(["XX.S00", "XX.S01"])]
    assert len(ea.SyntheticAssociator(cfg).associate(picks, sta)[0]) == 0
    relaxed = cfg.replace(n_picks_p_and_s_policy="ignore")
    assert len(ea.SyntheticAssociator(relaxed).associate(picks, sta)[0]) == 1


def test_associator_handles_empty_and_unknown_stations():
    cfg, sta = test_config(), stations()
    empty = pd.DataFrame(columns=["pick_id", "station", "phase", "time", "score"])
    events, assignments = ea.SyntheticAssociator(cfg).associate(empty, sta)
    assert len(events) == 0 and len(assignments) == 0
    cat = catalogue(offsets_s=(0,))
    picks = synth_picks(cat, sta, cfg)
    picks.loc[len(picks)] = dict(pick_id="ghost", station="ZZ.NONE", phase="P", time=T0, score=0.9, matched_event=None)
    events, assignments = ea.SyntheticAssociator(cfg).associate(picks, sta)
    assert "ghost" not in set(assignments["pick_id"]) and len(events) == 1


def test_duplicate_pick_ids_are_refused():
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0,))
    picks = synth_picks(cat, sta, cfg)
    picks = pd.concat([picks, picks.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="pick_id must be unique"):
        ea.SyntheticAssociator(cfg).associate(picks, sta)


def test_associate_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown associator backend"):
        ea.associate(pd.DataFrame(), pd.DataFrame(), test_config(), backend="octoloc")


# ── the PyOcto adapter ───────────────────────────────────────────────────────

def test_pyocto_adapter_raises_a_clear_error_when_pyocto_is_absent(monkeypatch):
    real_import = builtins.__import__

    def no_pyocto(name, *args, **kwargs):
        if name == "pyocto" or name.startswith("pyocto."):
            raise ImportError("No module named 'pyocto'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyocto)
    monkeypatch.delitem(sys.modules, "pyocto", raising=False)
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0,))
    picks = synth_picks(cat, sta, cfg)
    with pytest.raises(ImportError, match="pip install pyocto"):
        ea.PyOctoAssociator(cfg).associate(picks, sta)
    with pytest.raises(ImportError, match="Münchmeyer"):
        ea.PyOctoAssociator._import()


def test_pyocto_frames_map_the_pick_store_and_station_table():
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0,))
    picks = synth_picks(cat, sta, cfg)
    pick_frame, station_frame = ea.PyOctoAssociator.frames(picks, sta)
    assert list(pick_frame.columns) == ["station", "phase", "time", "probability", "pick_id"]
    assert list(station_frame.columns) == ["id", "latitude", "longitude", "elevation"]
    assert pick_frame["time"].dtype == float
    first = picks["time"].iloc[0].timestamp()
    assert abs(pick_frame["time"].iloc[0] - first) < 1e-6
    assert set(pick_frame["phase"]) <= {"P", "S"}
    assert station_frame["elevation"].tolist() == sta["elev_m"].tolist()
    no_score = picks.drop(columns=["score"])
    assert (ea.PyOctoAssociator.frames(no_score, sta)[0]["probability"] == 1.0).all()


# ── event matching ───────────────────────────────────────────────────────────

def test_matching_is_one_to_one_and_maximum_cardinality():
    # The greedy counterexample of continuous_scoring.match_picks, at event scale:
    # nearest-first would pair ref00 with the 0.0 s prediction and leave ref01 unmatched.
    cat = catalogue(offsets_s=(0, 6), mags=[2.0, 2.0])
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, 5.0], [LAT0, LON0, 5.0]]
    pred = events_frame([(0, -4.0, 0, 0, 5.0), (1, 1.0, 0, 0, 5.0)])
    match = ea.match_events(pred, cat, tol_time_s=5.0, tol_km=30.0, tol_depth_km=50.0)
    assert match.n_matched == 2
    assert dict(zip(match.pairs["event"], match.pairs["event_idx"])) == {"ref00": 0, "ref01": 1}
    assert len(match.unmatched_reference) == 0 and len(match.unmatched_predicted) == 0


def test_tolerance_boundaries_are_inclusive_in_time_distance_and_depth():
    cat = catalogue(offsets_s=(0,))
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, 10.0]]
    on_time = events_frame([(0, 5.0, 0, 0, 10.0)])
    assert ea.match_events(on_time, cat, tol_time_s=5.0).n_matched == 1
    just_over = events_frame([(0, 5.001, 0, 0, 10.0)])
    assert ea.match_events(just_over, cat, tol_time_s=5.0).n_matched == 0
    # distance: a prediction 30 km east at the tolerance, and one past it
    d_on = ea.haversine_km(LAT0, LON0, *PROJ.to_latlon(30.0, 0.0))
    assert abs(float(d_on) - 30.0) < 0.2
    near = events_frame([(0, 0.0, 29.8, 0, 10.0)])
    far = events_frame([(0, 0.0, 31.0, 0, 10.0)])
    assert ea.match_events(near, cat, tol_km=30.0).n_matched == 1
    assert ea.match_events(far, cat, tol_km=30.0).n_matched == 0
    deep = events_frame([(0, 0.0, 0, 0, 60.0)])
    assert ea.match_events(deep, cat, tol_depth_km=50.0).n_matched == 1
    assert ea.match_events(deep, cat, tol_depth_km=49.0).n_matched == 0
    assert ea.match_events(deep, cat, tol_depth_km=None).n_matched == 1


def test_missing_depth_never_blocks_a_match():
    cat = catalogue(offsets_s=(0,))
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, np.nan]]
    pred = events_frame([(0, 0.0, 0, 0, 12.0)])
    match = ea.match_events(pred, cat, tol_depth_km=5.0)
    assert match.n_matched == 1 and np.isnan(match.pairs["ddepth_km"].iloc[0])


def test_unmatched_counts_and_residual_signs():
    cat = catalogue(offsets_s=(0, 3600, 7200))
    pred = events_frame([(0, 2.0, 0, 0, 8.0)])
    pred.loc[0, ["lat", "lon"]] = [cat.loc[0, "lat"], cat.loc[0, "lon"]]
    pred = pd.concat([pred, events_frame([(1, 20000.0, 0, 0, 8.0)])], ignore_index=True)
    match = ea.match_events(pred, cat)
    assert match.n_matched == 1 and match.n_reference == 3 and match.n_predicted == 2
    assert len(match.unmatched_reference) == 2 and len(match.unmatched_predicted) == 1
    assert match.unmatched_predicted["event_idx"].tolist() == [1]
    assert sorted(match.unmatched_reference["event"]) == ["ref01", "ref02"]
    assert match.pairs["dt_s"].iloc[0] == pytest.approx(2.0)     # predicted minus reference
    assert match.recovery == pytest.approx(1 / 3)


def test_matching_with_empty_sides_and_duplicate_ids():
    cat = catalogue(offsets_s=(0,))
    empty = pd.DataFrame(columns=ea.EVENT_COLUMNS)
    assert ea.match_events(empty, cat).n_matched == 0
    assert len(ea.match_events(empty, cat).unmatched_reference) == 1
    assert ea.match_events(events_frame([(0, 0.0, 0, 0, 5.0)]), cat.iloc[:0]).n_matched == 0
    doubled = pd.concat([cat, cat], ignore_index=True)
    with pytest.raises(ValueError, match="reference ids are not unique"):
        ea.match_events(events_frame([(0, 0.0, 0, 0, 5.0)]), doubled)
    with pytest.raises(ValueError, match="tolerances must be positive"):
        ea.match_events(events_frame([(0, 0.0, 0, 0, 5.0)]), cat, tol_time_s=0.0)


def test_secondary_cost_prefers_the_smaller_normalised_residual():
    cat = catalogue(offsets_s=(0,))
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, 10.0]]
    pred = events_frame([(0, 4.0, 0, 0, 10.0), (1, 0.5, 0, 0, 10.0)])
    match = ea.match_events(pred, cat)
    assert match.pairs["event_idx"].tolist() == [1]


# ── splits and merges ────────────────────────────────────────────────────────

def test_duplicate_predicted_event_is_reported_as_a_split():
    cat = catalogue(offsets_s=(0,))
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, 10.0]]
    pred = events_frame([(0, 0.2, 1, 0, 10.0), (1, 1.4, 2, 0, 10.0)])
    match = ea.match_events(pred, cat)
    assert match.n_matched == 1 and len(match.unmatched_predicted) == 1
    diag = ea.split_merge_diagnostics(match)
    assert diag.n_splits == 1 and diag.n_merges == 0
    row = diag.splits.iloc[0]
    assert row["event"] == "ref00" and row["n_predicted"] == 2 and sorted(row["predicted"]) == [0, 1]
    assert row["rule"] == "tolerance"


def test_split_detected_from_shared_picks_outside_the_distance_tolerance():
    cat = catalogue(offsets_s=(0,))
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, 10.0]]
    pred = events_frame([(0, 0.2, 0, 0, 10.0), (1, 0.4, 200, 0, 10.0)])   # the second is 200 km away
    match = ea.match_events(pred, cat)
    assert match.n_matched == 1
    assignments = pd.DataFrame([
        dict(event_idx=e, pick_id=f"{e}-{i}", station=f"XX.S{i:02d}", phase="P", time=T0, score=0.9,
             residual=0.0, matched_event=("ref00" if i < 3 else None))
        for e in (0, 1) for i in range(4)])
    diag = ea.split_merge_diagnostics(match, assignments)
    assert diag.n_splits == 1
    assert diag.splits.iloc[0]["rule"] == "pick_overlap+tolerance"
    # below the overlap threshold the pick rule no longer fires
    weak = assignments.copy()
    weak.loc[(weak["event_idx"] == 1) & (weak["pick_id"] != "1-0"), "matched_event"] = None
    diag_weak = ea.split_merge_diagnostics(match, weak)
    assert diag_weak.n_splits == 0


def test_one_predicted_event_covering_two_reference_events_is_a_merge():
    cat = catalogue(offsets_s=(0, 4))
    cat.loc[:, ["lat", "lon", "depth_km"]] = [[LAT0, LON0, 10.0], [LAT0 + 0.05, LON0, 10.0]]
    pred = events_frame([(0, 2.0, 0, 0, 10.0)])
    match = ea.match_events(pred, cat, tol_time_s=5.0, tol_km=30.0)
    assert match.n_matched == 1 and len(match.unmatched_reference) == 1
    assignments = pd.DataFrame([
        dict(event_idx=0, pick_id=f"p{i}", station=f"XX.S{i:02d}", phase="P", time=T0, score=0.9, residual=0.0,
             matched_event=("ref00" if i < 3 else "ref01")) for i in range(6)])
    diag = ea.split_merge_diagnostics(match, assignments)
    assert diag.n_merges == 1 and diag.n_splits == 0
    row = diag.merges.iloc[0]
    assert row["event_idx"] == 0 and sorted(row["events"]) == ["ref00", "ref01"]
    assert row["reference_events_in_picks"] == {"ref00": 3, "ref01": 3}
    assert diag.to_dict()["n_merges"] == 1


def test_clean_one_to_one_case_reports_no_split_or_merge():
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0, 1200))
    events, assignments = ea.SyntheticAssociator(cfg).associate(synth_picks(cat, sta, cfg), sta)
    diag = ea.split_merge_diagnostics(ea.match_events(events, cat), assignments)
    assert diag.n_splits == 0 and diag.n_merges == 0
    assert diag.splits.empty and diag.merges.empty


# ── recovery tables and coverage ─────────────────────────────────────────────

def _tables(offsets_s, matched_idx, windows=None, **kw):
    cat = catalogue(offsets_s=offsets_s, mags=kw.pop("mags", None))
    pred = events_frame([(k, offsets_s[k], 0, 0, 10.0) for k in matched_idx])
    for n, k in enumerate(matched_idx):
        pred.loc[n, ["lat", "lon", "depth_km"]] = cat.loc[k, ["lat", "lon", "depth_km"]].to_numpy()
    match = ea.match_events(pred, cat)
    return cat, pred, match, ea.recovery_tables(match, cat, pred, windows=windows, mainshock_time=T0, **kw)


def test_recovery_by_magnitude_bin():
    mags = [1.5, 2.5, 2.7, 4.2]
    windows = windows_frame([(0, 4000)])
    cat, pred, match, tabs = _tables((0, 600, 1800, 3600), (0, 2, 3), windows=windows, mags=mags)
    by_mag = tabs["by_magnitude"].set_index("bin")
    assert by_mag.loc["[1, 2)", "n_reference_covered"] == 1 and by_mag.loc["[1, 2)", "recovery"] == 1.0
    assert by_mag.loc["[2, 3)", "n_reference_covered"] == 2 and by_mag.loc["[2, 3)", "recovery"] == 0.5
    assert by_mag.loc["[4, 5)", "recovery"] == 1.0
    assert np.isnan(by_mag.loc[">= 5", "recovery"]) and not by_mag.loc[">= 5", "claim_supported"]
    assert by_mag.loc["[2, 3)", "claim_supported"]
    assert tabs["summary"]["recovery"].iloc[0] == pytest.approx(3 / 4)


def test_first_48_hour_claim_is_unsupported_when_the_windows_do_not_cover_it():
    offsets = (0, 1800, 5 * 3600, 30 * 3600)
    short = windows_frame([(0, 3 * 3600)])                      # three hours only
    _, _, _, tabs = _tables(offsets, (0, 1, 2, 3), windows=short)
    by_hour = tabs["by_hour"].set_index("bin")
    assert by_hour.loc["[0, 1)h", "claim_supported"]             # fully inside the window
    assert by_hour.loc["[0, 1)h", "coverage_fraction"] == pytest.approx(1.0)
    assert not by_hour.loc["[1, 6)h", "claim_supported"]         # only 2 of 5 hours covered
    assert by_hour.loc["[1, 6)h", "coverage_fraction"] == pytest.approx(0.4)
    assert by_hour.loc["[1, 6)h", "n_reference_covered"] == 0    # the +5 h event is outside the windows
    assert not by_hour.loc["[24, 48)h", "claim_supported"]
    assert by_hour.loc["[24, 48)h", "coverage_fraction"] == 0.0
    # with 48 h of windows the same bins become claimable
    _, _, _, full = _tables(offsets, (0, 1, 2, 3), windows=windows_frame([(0, 48 * 3600)]))
    full_hours = full["by_hour"].set_index("bin")
    assert full_hours.loc["[1, 6)h", "claim_supported"] and full_hours.loc["[24, 48)h", "claim_supported"]
    assert full_hours.loc["[24, 48)h", "recovery"] == 1.0
    assert set(full["by_hour"]["time_origin_label"]) == {"mainshock"}


def test_no_windows_means_no_supported_claim():
    _, _, _, tabs = _tables((0, 1800, 7200), (0, 1), windows=None)
    for name in ("by_magnitude", "by_hour", "by_day"):
        assert not tabs[name]["claim_supported"].any()
    assert not tabs["summary"]["claim_supported"].iloc[0]
    assert tabs["summary"]["n_windows"].iloc[0] == 0
    assert tabs["summary"]["n_reference_covered"].iloc[0] == 3   # everything counted, nothing claimable


def test_reference_events_outside_the_windows_leave_the_denominator():
    windows = windows_frame([(0, 1200)])
    _, _, _, tabs = _tables((0, 600, 7200), (0, 1), windows=windows)
    s = tabs["summary"].iloc[0]
    assert s["n_reference"] == 3 and s["n_reference_covered"] == 2 and s["n_reference_outside_windows"] == 1
    assert s["recovery"] == 1.0 and s["n_unmatched_reference_covered"] == 0
    cov = tabs["coverage"].set_index("event")
    assert cov.loc["ref02", "in_window"] == False   # noqa: E712
    assert cov["hours_after"].max() == pytest.approx(2.0)


def test_day_bins_cover_the_sequence_and_count_per_day():
    windows = windows_frame([(0, 3 * 86400)])
    _, _, _, tabs = _tables((0, 90000, 180000), (0, 1, 2), windows=windows)
    by_day = tabs["by_day"]
    assert by_day["bin"].tolist()[:3] == ["[0, 1)d", "[1, 2)d", "[2, 3)d"]
    assert by_day["n_reference_covered"].tolist()[:3] == [1, 1, 1]
    assert by_day["claim_supported"].tolist()[:3] == [True, True, True]


def test_station_support_and_unassociated_pick_counts():
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0, 1500))
    picks = synth_picks(cat, sta, cfg, n_noise=12)
    events, assignments = ea.SyntheticAssociator(cfg).associate(picks, sta)
    match = ea.match_events(events, cat)
    tabs = ea.recovery_tables(match, cat, events, mainshock_time=T0, windows=windows_frame([(0, 3600)]),
                              assignments=assignments, n_picks_total=len(picks))
    support = tabs["station_support"]
    assert len(support) == len(events)
    assert (support["n_stations"] == len(sta)).all()
    assert (support["n_picks"] == 2 * len(sta)).all()
    assert support["matched"].all()
    s = tabs["summary"].iloc[0]
    assert s["n_picks_total"] == len(picks)
    assert s["n_picks_assigned"] == len(assignments)
    assert s["n_picks_unassociated"] == len(picks) - len(assignments) >= 12
    assert s["n_unmatched_predicted"] == 0


def test_recovery_tables_without_assignments_still_report_events():
    _, pred, _, tabs = _tables((0, 600), (0,), windows=windows_frame([(0, 1200)]))
    assert len(tabs["station_support"]) == len(pred)
    assert tabs["station_support"]["n_picks"].isna().all()
    assert np.isnan(tabs["summary"]["n_picks_total"].iloc[0])


def test_time_origin_falls_back_to_the_first_window_then_the_first_event():
    cat = catalogue(offsets_s=(0, 3600))
    pred = events_frame([])
    match = ea.match_events(pred, cat)
    windows = windows_frame([(-600, 7200)])
    tabs = ea.recovery_tables(match, cat, pred, windows=windows)
    assert set(tabs["by_hour"]["time_origin_label"]) == {"first_window_start"}
    tabs2 = ea.recovery_tables(match, cat, pred)
    assert set(tabs2["by_hour"]["time_origin_label"]) == {"first_reference_event"}


# ── paired block bootstrap ───────────────────────────────────────────────────

def _match_subset(cat, keep_idx):
    pred = events_frame([(n, (cat.loc[k, "origin"] - T0).total_seconds(), 0, 0, 10.0) for n, k in enumerate(keep_idx)])
    for n, k in enumerate(keep_idx):
        pred.loc[n, ["lat", "lon", "depth_km"]] = cat.loc[k, ["lat", "lon", "depth_km"]].to_numpy()
    return ea.match_events(pred, cat)


def test_paired_bootstrap_interval_contains_zero_for_identical_inputs():
    cat = catalogue(offsets_s=tuple(range(0, 40 * 300, 300)))
    match = _match_subset(cat, list(range(0, 40, 2)))
    out = ea.paired_block_bootstrap(match, match, cat, n_boot=500, seed=0)
    assert out["diff"] == 0.0 and out["ci_low"] <= 0 <= out["ci_high"]
    assert not out["excludes_zero"]
    assert out["n_events"] == 40 and out["n_blocks"] == 40 and out["recovery_a"] == out["recovery_b"] == 0.5


def test_paired_bootstrap_excludes_zero_for_a_clear_difference():
    cat = catalogue(offsets_s=tuple(range(0, 40 * 300, 300)))
    weak = _match_subset(cat, list(range(0, 8)))            # 8 of 40
    strong = _match_subset(cat, list(range(0, 38)))         # 38 of 40
    out = ea.paired_block_bootstrap(weak, strong, cat, n_boot=1000, seed=0)
    assert out["recovery_a"] == pytest.approx(0.2) and out["recovery_b"] == pytest.approx(0.95)
    assert out["diff"] == pytest.approx(0.75)
    assert out["ci_low"] > 0 and out["excludes_zero"]
    flipped = ea.paired_block_bootstrap(strong, weak, cat, n_boot=1000, seed=0)
    assert flipped["ci_high"] < 0 and flipped["excludes_zero"]


def test_bootstrap_blocks_keep_events_together_and_are_deterministic():
    cat = catalogue(offsets_s=tuple(range(0, 24 * 1800, 1800)))   # 24 events over 12 hours
    a = _match_subset(cat, list(range(0, 24, 2)))
    b = _match_subset(cat, list(range(0, 24, 3)))
    by_event = ea.paired_block_bootstrap(a, b, cat, block="event", n_boot=400, seed=3, mainshock_time=T0)
    by_hour = ea.paired_block_bootstrap(a, b, cat, block="hour", n_boot=400, seed=3, mainshock_time=T0)
    assert by_event["n_blocks"] == 24 and by_hour["n_blocks"] == 12
    assert by_event["diff"] == by_hour["diff"]
    again = ea.paired_block_bootstrap(a, b, cat, block="hour", n_boot=400, seed=3, mainshock_time=T0)
    assert (again["ci_low"], again["ci_high"]) == (by_hour["ci_low"], by_hour["ci_high"])
    assert by_hour["ci_low"] <= by_hour["diff"] <= by_hour["ci_high"]
    with pytest.raises(ValueError, match="block must be"):
        ea.paired_block_bootstrap(a, b, cat, block="station")
    with pytest.raises(ValueError, match="at least one reference event"):
        ea.paired_block_bootstrap(a, b, cat.iloc[:0])


# ── run, artifacts and CLI ───────────────────────────────────────────────────

def pick_store(cat, sta, cfg, key="samos_2020", model_id="m" * 64, threshold=0.3, access_id="acc-1", **kw):
    picks = synth_picks(cat, sta, cfg, **kw)
    return picks.assign(model_id=model_id, model="synthetic_weights", threshold=threshold, access_id=access_id, key=key)


def test_run_writes_every_artifact_with_the_config_hash_and_access_id(tmp_path):
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0, 1800))
    picks = pick_store(cat, sta, cfg, n_noise=10)
    windows = windows_frame([(0, 3600)])
    res = ea.run(picks, sta, cat, cfg, windows=windows, mainshock_time=T0, key="samos_2020", backend="synthetic",
                 out_dir=tmp_path / "run")
    out = tmp_path / "run"
    for name in ("events", "assignments", "matches", "unmatched_predicted", "unmatched_reference", "splits", "merges"):
        assert (out / f"{name}.parquet").exists()
    for name in ("by_magnitude", "by_hour", "by_day", "station_support", "coverage", "summary"):
        assert (out / "tables" / f"{name}.csv").exists()
    meta = json.loads((out / "run.json").read_text())
    assert meta["config_sha256"] == cfg.sha256 and meta["config"]["sha256"] == cfg.sha256
    assert meta["access_id"] == "acc-1" and meta["key"] == "samos_2020" and meta["checkpoint"] == "36A"
    assert meta["model_id"] == "m" * 64 and meta["threshold"] == 0.3 and meta["associator"] == "synthetic"
    assert meta["match_tolerances"] == dict(tol_time_s=5.0, tol_km=30.0, tol_depth_km=50.0)
    assert meta["n_matched"] == 2 and meta["n_events"] == 2 and meta["n_splits"] == 0 and meta["n_merges"] == 0
    events = pd.read_parquet(out / "events.parquet")
    assert set(events["config_sha256"]) == {cfg.sha256} and set(events["access_id"]) == {"acc-1"}
    assignments = pd.read_parquet(out / "assignments.parquet")
    assert set(assignments["config_sha256"]) == {cfg.sha256}
    assert "matched_event" in assignments      # the 35A pick-level assignment is carried through
    matches = pd.read_parquet(out / "matches.parquet")
    assert set(matches["key"]) == {"samos_2020"} and len(matches) == 2
    assert json.loads((out / "diagnostics.json").read_text())["n_splits"] == 0
    assert res["tables"]["summary"]["n_picks_unassociated"].iloc[0] >= 10


def test_run_selects_one_model_and_threshold_from_the_pick_store():
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0,))
    a = pick_store(cat, sta, cfg, model_id="a" * 64, threshold=0.2)
    b = pick_store(cat, sta, cfg, model_id="b" * 64, threshold=0.5)
    b["pick_id"] = b["pick_id"] + "|b"
    both = pd.concat([a, b], ignore_index=True)
    with pytest.raises(ValueError, match="--model-id is required"):
        ea.select_picks(both)
    with pytest.raises(ValueError, match="--threshold is required"):
        ea.select_picks(both.assign(model_id="a" * 64))
    sel = ea.select_picks(both, model_id="b" * 64, threshold=0.5)
    assert len(sel) == len(b) and set(sel["model_id"]) == {"b" * 64}
    with pytest.raises(ValueError, match="not in the pick store"):
        ea.select_picks(both, model_id="c" * 64)
    with pytest.raises(ValueError, match="not in the pick store"):
        ea.select_picks(both, model_id="a" * 64, threshold=0.9)


def _write_inputs(tmp_path, key="samos_2020", noise=6):
    cfg, sta, cat = test_config(), stations(), catalogue(offsets_s=(0, 1800))
    picks = pick_store(cat, sta, cfg, key=key, n_noise=noise)
    picks.to_parquet(tmp_path / "picks.parquet", index=False)
    sta.to_csv(tmp_path / "stations.csv", index=False)
    cat.to_parquet(tmp_path / "catalog.parquet", index=False)
    windows_frame([(0, 3600)]).to_csv(tmp_path / "windows.csv", index=False)
    cfg.save(tmp_path / "assoc.json")
    return cfg, sta, cat


def test_cli_runs_end_to_end_on_a_pick_store(tmp_path, capsys):
    _write_inputs(tmp_path)
    res = ea.main(["--picks", str(tmp_path / "picks.parquet"), "--stations", str(tmp_path / "stations.csv"),
                   "--catalog", str(tmp_path / "catalog.parquet"), "--windows", str(tmp_path / "windows.csv"),
                   "--config", str(tmp_path / "assoc.json"), "--associator", "synthetic",
                   "--model-id", "m" * 64, "--threshold", "0.3", "--out-dir", str(tmp_path / "out")])
    assert res["meta"]["n_matched"] == 2
    meta = json.loads((tmp_path / "out" / "run.json").read_text())
    assert meta["sources"]["picks"]["sha256"] and meta["sources"]["config"]["sha256"]
    assert meta["key"] == "samos_2020"            # taken from the pick store
    text = capsys.readouterr().out
    assert "by_hour" in text and meta["config_sha256"][:12] in text


def test_cli_refuses_a_protected_sequence(tmp_path):
    _write_inputs(tmp_path, key="la_palma_2021")
    with pytest.raises(SystemExit):
        ea.main(["--picks", str(tmp_path / "picks.parquet"), "--stations", str(tmp_path / "stations.csv"),
                 "--catalog", str(tmp_path / "catalog.parquet"), "--config", str(tmp_path / "assoc.json"),
                 "--associator", "synthetic"])
    with pytest.raises(SystemExit):
        ea.main(["--sequence", "la_palma_2021", "--picks", str(tmp_path / "picks.parquet"),
                 "--associator", "synthetic"])


def test_cli_rehash_and_default_config_writing(tmp_path, capsys):
    cfg = test_config()
    path = cfg.save(tmp_path / "edit.json")
    d = json.loads(path.read_text())
    d["min_picks"] = 9
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        ea.AssociatorConfig.load(path)
    ea.main(["--rehash", str(path)])
    reloaded = ea.AssociatorConfig.load(path)
    assert reloaded.min_picks == 9 and reloaded.sha256 != cfg.sha256
    ea.main(["--write-default-configs", str(tmp_path / "cfgs")])
    assert {p.name for p in (tmp_path / "cfgs").glob("*.json")} == {f"{r}.json" for r in ea.REGIMES}


def test_select_picks_refuses_mixed_sequences_and_runs():
    base = pd.DataFrame({"pick_id": ["p1", "p2", "p3", "p4"], "model_id": ["m" * 64] * 4, "threshold": [0.3] * 4,
                         "station": ["XX.A"] * 4, "phase": ["P"] * 4, "time": pd.to_datetime(["2020-01-01"] * 4, utc=True),
                         "score": [0.5] * 4, "key": ["samos_2020", "samos_2020", "adriatic_2022", "adriatic_2022"],
                         "access_id": ["r1", "r1", "r2", "r2"]})
    with pytest.raises(ValueError, match="several sequences"):
        ea.select_picks(base)
    one = ea.select_picks(base, key="samos_2020")
    assert set(one["key"]) == {"samos_2020"} and len(one) == 2
    with pytest.raises(ValueError, match="not in the pick store"):
        ea.select_picks(base, key="etna_2022_2024")
    mixed = base.assign(key="samos_2020")
    with pytest.raises(ValueError, match="mixes 2 scoring runs"):
        ea.select_picks(mixed)
    clean = base[base["key"] == "samos_2020"]
    assert len(ea.select_picks(clean)) == 2
