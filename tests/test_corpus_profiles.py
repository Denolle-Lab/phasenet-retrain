"""Corpus profiles of scripts/build_training_dataset.py --profile (#40A).

No SeisBench, no cache: build_training_dataset is imported with a stub
seisbench module (the same trick as tests/test_exclusion_bundle.py), and the
end-to-end build runs on synthetic sources through the fixture bundle.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import exclusion_bundle as eb  # noqa: E402
import heldout_sequences as hs  # noqa: E402
from test_exclusion_bundle import build, repo, _fill_cache  # noqa: E402,F401  (fixtures)

PROFILES = REPO / "configs" / "corpus_profiles.yaml"


def _import_btd(monkeypatch):
    fake_data = types.ModuleType("seisbench.data")
    fake_data.__getattr__ = lambda name: object
    fake_ek = types.ModuleType("event_keys")
    fake_ek.trace_key_map = lambda name: {}
    fake_ek.trace_key_map_any_chunk = lambda name: {}
    monkeypatch.setitem(sys.modules, "seisbench", types.ModuleType("seisbench"))
    monkeypatch.setitem(sys.modules, "seisbench.data", fake_data)
    monkeypatch.setitem(sys.modules, "event_keys", fake_ek)
    monkeypatch.delitem(sys.modules, "build_training_dataset", raising=False)
    import build_training_dataset as btd
    return btd


# ── loading and the legacy equivalence ───────────────────────────────────────

def test_profiles_file_loads_and_lists_both_profiles(monkeypatch):
    btd = _import_btd(monkeypatch)
    profiles, file_sha = btd.load_profiles(PROFILES)
    assert {"legacy_v2", "t0_pilot"} <= set(profiles) and len(file_sha) == 64
    assert btd.profile_sha256(profiles["t0_pilot"]) != btd.profile_sha256(profiles["legacy_v2"])
    # the hash is layout-independent: same mapping, same digest
    assert btd.profile_sha256(json.loads(json.dumps(profiles["t0_pilot"]))) == btd.profile_sha256(profiles["t0_pilot"])


def test_legacy_profile_equals_module_defaults(monkeypatch):
    """--profile legacy_v2 resolves to exactly DATASET_CONFIGS, TARGET_FRACTIONS
    and SKIP_SOURCES_THIS_ROUND, so a build without --profile is unchanged."""
    btd = _import_btd(monkeypatch)
    r = btd.resolve_profile("legacy_v2", path=PROFILES)
    assert r["configs"] == btd.DATASET_CONFIGS                      # same entries, same order, same loader objects
    assert [c["name"] for c in r["configs"]] == [c["name"] for c in btd.DATASET_CONFIGS]
    assert r["target_fractions"] == btd.TARGET_FRACTIONS
    assert r["skip_sources"] == btd.SKIP_SOURCES_THIS_ROUND
    assert r["max_distance_km"] is None and r["allow_p_only"] is True
    assert not any("require_status_columns" in c for c in r["configs"])


def test_t0_profile_matches_the_strategy(monkeypatch):
    btd = _import_btd(monkeypatch)
    r = btd.resolve_profile("t0_pilot", path=PROFILES)
    caps = {c["name"]: c["cap"] for c in r["configs"]}
    assert caps == {"ethz": 10_000, "pnw": 10_000, "cwa": 10_000, "scedc": 10_000, "ceed": 10_000,
                    "txed": 10_000, "iquique": 5_000, "instancecounts": 15_000, "vcseis": 10_000}
    assert sum(caps.values()) == 90_000
    assert all(c["use_s"] for c in r["configs"])
    assert r["target_fractions"] == {"local": 0.55, "regional": 0.40, "teleseismic": 0.0, "unknown": 0.05}
    assert r["max_distance_km"] == 2000.0 and r["allow_p_only"] is False
    assert r["skip_sources"] == frozenset({"obst2024", "obs"})
    inst = next(c for c in r["configs"] if c["name"] == "instancecounts")
    assert inst["require_status_columns"] and inst["manual_values"] == ["manual"]
    # every listed status column is a name the repository records
    import audit_source_labels as asl
    assert set(inst["require_status_columns"]) <= set(asl.P_STATUS_COLUMNS + asl.S_STATUS_COLUMNS)
    # DATASET_CONFIGS order, not the profile's
    order = [c["name"] for c in btd.DATASET_CONFIGS]
    assert [c["name"] for c in r["configs"]] == sorted(caps, key=order.index)
    # no OBS, no P-only, no teleseismic-only source
    assert not {"obs", "obst2024", "geofon", "lendb", "meier2019jgr"} & set(caps)


def test_p_only_source_is_refused_when_not_allowed(monkeypatch, tmp_path):
    btd = _import_btd(monkeypatch)
    doc = {"version": 1, "profiles": {
        "bad": {"allow_p_only": False, "sources": {"geofon": {"cap": 10, "use_s": False}}},
        "ok": {"allow_p_only": True, "sources": {"geofon": {"cap": 10, "use_s": False}}},
    }}
    f = tmp_path / "p.yaml"
    import yaml
    f.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="P-only"):
        btd.resolve_profile("bad", path=f)
    assert btd.resolve_profile("ok", path=f)["configs"][0]["use_s"] is False


def test_resolve_profile_refuses_unknown_names_and_keys(monkeypatch, tmp_path):
    btd = _import_btd(monkeypatch)
    import yaml
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump({"version": 1, "profiles": {
        "nosrc": {"sources": {"nope": {"cap": 1, "use_s": True}}},
        "badkey": {"sources": {"ethz": {"cap": 1, "use_s": True, "dist_col": "x"}}},
        "nocap": {"sources": {"ethz": {"use_s": True}}},
        "badfrac": {"sources": {"ethz": {"cap": 1, "use_s": True}}, "target_fractions": {"near": 1}},
    }}))
    with pytest.raises(ValueError, match="unknown profile"):
        btd.resolve_profile("missing", path=f)
    with pytest.raises(ValueError, match="does not know"):
        btd.resolve_profile("nosrc", path=f)
    with pytest.raises(ValueError, match="unknown keys"):
        btd.resolve_profile("badkey", path=f)
    with pytest.raises(ValueError, match="cap and use_s"):
        btd.resolve_profile("nocap", path=f)
    with pytest.raises(ValueError, match="unknown bins"):
        btd.resolve_profile("badfrac", path=f)
    f.write_text(yaml.safe_dump({"version": 2, "profiles": {}}))
    with pytest.raises(ValueError, match="version 1"):
        btd.load_profiles(f)


def test_fractions_are_renormalised(monkeypatch):
    btd = _import_btd(monkeypatch)
    out = btd.normalise_fractions({"local": 55, "regional": 40, "unknown": 5})
    assert out == pytest.approx({"local": 0.55, "regional": 0.40, "teleseismic": 0.0, "unknown": 0.05})
    assert abs(sum(out.values()) - 1.0) < 1e-12
    assert btd.normalise_fractions(btd.TARGET_FRACTIONS) == btd.TARGET_FRACTIONS   # untouched when already 1
    with pytest.raises(ValueError):
        btd.normalise_fractions({"local": -1, "regional": 2})
    with pytest.raises(ValueError):
        btd.normalise_fractions({"local": 0})
    with pytest.raises(ValueError):
        btd.normalise_fractions({})


def test_zero_fraction_removes_the_bin_from_training(monkeypatch):
    btd = _import_btd(monkeypatch)
    df = pd.DataFrame({"distance_bin": ["local"] * 55 + ["regional"] * 40 + ["teleseismic"] * 20 + ["unknown"] * 5,
                       "x": range(120)})
    out = btd.stratify_training(df, np.random.default_rng(0),
                                fractions={"local": 0.55, "regional": 0.40, "teleseismic": 0.0, "unknown": 0.05})
    assert "teleseismic" not in set(out["distance_bin"])
    counts = out["distance_bin"].value_counts().to_dict()
    # int(n / frac) truncation can lose one row of the target total; the mix is the point
    assert all(abs(counts[b] - n) <= 1 for b, n in {"local": 55, "regional": 40, "unknown": 5}.items())


# ── status filter on a synthetic frame ───────────────────────────────────────

def test_status_filter_nulls_non_manual_picks_per_phase(monkeypatch):
    btd = _import_btd(monkeypatch)
    meta = pd.DataFrame({
        "trace_P_status": ["manual", "automatic", "manual", None, " Manual ", "manual"],
        "trace_S_status": ["manual", "manual", "automatic", "manual", None, "manual"],
    })
    p = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, np.nan])
    s = pd.Series([2.0, 2.0, 2.0, np.nan, 2.0, 2.0])
    p2, s2, rep = btd.apply_status_filter(meta, p, s, ["trace_P_status", "trace_S_status", "trace_p_arrival_status"])
    assert p2.notna().tolist() == [True, False, True, False, True, False]   # automatic and NaN P nulled; case/space tolerated
    assert s2.notna().tolist() == [True, True, False, False, False, True]   # automatic and NaN S nulled; absent S untouched
    assert rep["status_columns_present"] == "trace_P_status;trace_S_status"
    assert rep["status_columns_listed"].endswith("trace_p_arrival_status")
    assert rep["n_p_dropped_by_status"] == 2 and rep["n_s_dropped_by_status"] == 2
    assert rep["n_rows_removed_by_status"] == 1                               # row 3: P NaN status, no S pick
    # absent columns: a no-op with the listing recorded
    p3, s3, rep3 = btd.apply_status_filter(meta[[]], p, s, ["trace_p_status"])
    assert p3.equals(p) and s3.equals(s) and rep3["status_columns_present"] == ""
    # a column without a phase in its name constrains both picks
    meta2 = pd.DataFrame({"pick_status": ["manual", "reviewed"]})
    p4, s4, rep4 = btd.apply_status_filter(meta2, pd.Series([1.0, 1.0]), pd.Series([2.0, 2.0]), ["pick_status"],
                                          manual_values=["manual"])
    assert p4.notna().tolist() == [True, False] and s4.notna().tolist() == [True, False]
    assert btd._status_phase("trace_Pg_status") == "P" and btd._status_phase("trace_s_status") == "S"
    assert btd._status_phase("pick_status") is None


# ── end to end on synthetic sources ──────────────────────────────────────────

def _source(name, rows, dist=None):
    cols = ["trace_name", "chunk", "trace_P_arrival_sample", "trace_S_arrival_sample",
            hs.TIME_COL, hs.LAT_COL, hs.LON_COL, "split", "trace_P_status", "trace_S_status"]
    frame = pd.DataFrame(rows, columns=cols)
    if dist is not None:
        frame["path_ep_distance_km"] = dist
    return dict(name=name, cls=None, meta_fn=lambda: frame.copy(), dist_col="path_ep_distance_km" if dist is not None else None,
                dist_unit="km", cap=10_000, default_bin="local", use_s=True)


def test_build_with_profile_filters_status_distance_and_records_provenance(repo, monkeypatch, tmp_path):
    btd = _import_btd(monkeypatch)
    _fill_cache(repo)
    monkeypatch.setenv("SEISBENCH_CACHE_ROOT", str(repo / "cache"))
    path, bundle = build(repo)
    monkeypatch.setattr(eb, "REPO_ROOT", repo)
    monkeypatch.setattr(eb, "BUNDLE_PATH", path)
    monkeypatch.setattr(eb, "USER_LABEL_ERROR_CACHE", repo / "nowhere")
    monkeypatch.setattr(btd, "BENCHMARK_CSV", repo / "notebooks" / "benchmark_manifest.csv")
    eb._cached_trace_exclusions.cache_clear()

    origins = pd.date_range("2018-01-01", periods=60, freq="6h").strftime("%Y-%m-%dT%H:%M:%S")
    inst = []
    for i in range(60):
        p_status = "automatic" if i % 10 == 0 else "manual"          # 6 automatic P
        s_status = "manual" if i % 3 else None                        # 20 NaN S
        s_pick = np.nan if i % 7 == 0 else 900.0                      # 9 rows without S
        inst.append((f"i{i}", "", 500.0, s_pick, origins[i], 40.0 + i * 0.3, 12.0 + i * 0.3, "train", p_status, s_status))
    dist = [50.0] * 30 + [800.0] * 20 + [2500.0] * 5 + [np.nan] * 5   # 5 beyond 2000 km, 5 unknown
    pnw = [(f"p{i}", "", 500.0, 900.0, origins[i], 45.0 + i * 0.2, -120.0 + i * 0.2, "train", None, None) for i in range(40)]

    import yaml
    prof = tmp_path / "profiles.yaml"
    prof.write_text(yaml.safe_dump({"version": 1, "profiles": {"pilot": {
        "allow_p_only": False, "max_distance_km": 2000, "skip_sources": ["obs"],
        "target_fractions": {"local": 0.55, "regional": 0.40, "unknown": 0.05},
        "sources": {"instancecounts": {"cap": 100, "use_s": True,
                                       "require_status_columns": ["trace_P_status", "trace_S_status", "trace_P_arrival_status"]},
                    "pnw": {"cap": 100, "use_s": True}},
    }}}))
    monkeypatch.setattr(btd, "DATASET_CONFIGS", [_source("pnw", pnw), _source("instancecounts", inst, dist=dist),
                                                 dict(_source("obs", pnw), name="obs")])
    out = tmp_path / "m"
    btd.main(str(out), seed=5, label_error_filter=False, profile="pilot", profiles_file=str(prof))

    written = pd.concat([pd.read_csv(out / f"{n}.csv").assign(split=n) for n in ("train", "val", "test")], ignore_index=True)
    assert set(written.dataset_name) == {"instancecounts", "pnw"}          # obs skipped, DATASET_CONFIGS order kept
    inst_w = written[written.dataset_name == "instancecounts"].set_index("trace_name")
    assert not any(f"i{i}" in inst_w.index for i in range(50, 55))          # beyond 2000 km
    assert inst_w.loc[[f"i{i}" for i in (10, 20, 40) if f"i{i}" in inst_w.index], "p_arrival_sample"].isna().all()
    assert inst_w.loc[[f"i{i}" for i in (3, 6, 9) if f"i{i}" in inst_w.index], "s_arrival_sample"].isna().all()
    assert not {"i0", "i30"} & set(inst_w.index)                             # automatic P and no manual S: no pick left
    assert not (written.distance_km > 2000).any()

    rep = pd.read_csv(out / "heldout_removal_report.csv").set_index("dataset")
    assert (rep["profile"] == "pilot").all() and rep["profile_sha256"].nunique() == 1
    assert rep.loc["instancecounts", "status_columns_present"] == "trace_P_status;trace_S_status"
    assert rep.loc["instancecounts", "n_p_dropped_by_status"] == 6
    assert rep.loc["instancecounts", "n_s_dropped_by_status"] == 20 - 3          # NaN-status rows that carry an S pick
    assert rep.loc["instancecounts", "n_rows_removed_by_status"] == 2           # i0 (no S pick) and i30 (S status NaN)
    assert rep.loc["instancecounts", "n_beyond_max_distance"] == 5
    assert pd.isna(rep.loc["pnw", "status_columns_present"]) or rep.loc["pnw", "status_columns_present"] == ""

    prov = json.loads((out / "provenance.json").read_text())
    assert prov["profile"]["name"] == "pilot" and prov["profile"]["sha256"] == rep["profile_sha256"].iloc[0]
    assert prov["profile"]["file_sha256"] and prov["profile"]["max_distance_km"] == 2000.0
    assert prov["profile"]["allow_p_only"] is False and prov["options"]["profile"] == "pilot"
    assert prov["target_fractions"] == pytest.approx({"local": 0.55, "regional": 0.40, "teleseismic": 0.0, "unknown": 0.05})
    assert [s["name"] for s in prov["sources"]] == ["pnw", "instancecounts"]
    assert prov["sources"][1]["require_status_columns"] == ["trace_P_status", "trace_S_status", "trace_P_arrival_status"]


def test_build_without_profile_records_no_profile(repo, monkeypatch, tmp_path):
    btd = _import_btd(monkeypatch)
    _fill_cache(repo)
    monkeypatch.setenv("SEISBENCH_CACHE_ROOT", str(repo / "cache"))
    path, _ = build(repo)
    monkeypatch.setattr(eb, "REPO_ROOT", repo)
    monkeypatch.setattr(eb, "BUNDLE_PATH", path)
    monkeypatch.setattr(eb, "USER_LABEL_ERROR_CACHE", repo / "nowhere")
    monkeypatch.setattr(btd, "BENCHMARK_CSV", repo / "notebooks" / "benchmark_manifest.csv")
    eb._cached_trace_exclusions.cache_clear()
    origins = pd.date_range("2018-01-01", periods=30, freq="6h").strftime("%Y-%m-%dT%H:%M:%S")
    rows = [(f"p{i}", "", 500.0, 900.0, origins[i], 45.0 + i * 0.2, -120.0 + i * 0.2, "train", None, None) for i in range(30)]
    monkeypatch.setattr(btd, "DATASET_CONFIGS", [_source("pnw", rows)])
    monkeypatch.setattr(btd, "SKIP_SOURCES_THIS_ROUND", frozenset())
    out = tmp_path / "m"
    btd.main(str(out), seed=1, label_error_filter=False)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["profile"]["name"] is None and "legacy_v2" in prov["profile"]["reason"]
    assert prov["target_fractions"] == btd.TARGET_FRACTIONS
    rep = pd.read_csv(out / "heldout_removal_report.csv")
    assert rep["profile"].isna().all() or (rep["profile"] == "").all()


def test_removal_report_s_counts_match_the_written_rows(repo, monkeypatch):
    """A synthetic source with teleseismic rows: the teleseismic P-only rule
    nulls their S and drops an S-only teleseismic row, and the report's
    n_written / n_with_s_written equal what process_dataset returns."""
    btd = _import_btd(monkeypatch)
    _fill_cache(repo)
    monkeypatch.setenv("SEISBENCH_CACHE_ROOT", str(repo / "cache"))
    path, bundle = build(repo)
    monkeypatch.setattr(eb, "REPO_ROOT", repo)
    monkeypatch.setattr(eb, "BUNDLE_PATH", path)
    monkeypatch.setattr(eb, "USER_LABEL_ERROR_CACHE", repo / "nowhere")
    eb._cached_trace_exclusions.cache_clear()
    origins = pd.date_range("2018-01-01", periods=12, freq="6h").strftime("%Y-%m-%dT%H:%M:%S")
    rows, dist = [], []
    for i in range(12):
        tele = i >= 6                                             # 6 local rows, 6 teleseismic rows
        s_pick = np.nan if i in (2, 8) else 900.0                 # one local and one teleseismic row without S
        p_pick = np.nan if i == 11 else 500.0                     # one S-only teleseismic row
        rows.append((f"t{i}", "", p_pick, s_pick, origins[i], 10.0 + i, 20.0 + i, "train", None, None))
        dist.append(1800.0 if tele else 30.0)
    cfg = _source("geo", rows, dist=dist)
    report = []
    out = btd.process_dataset(cfg, np.random.default_rng(0), bundle=bundle, holdout_report=report)
    rep = report[0]
    assert rep["n_after_cap"] == 12
    assert rep["n_s_nulled_teleseismic"] == 5                     # six teleseismic rows, one had no S
    assert rep["n_written"] == len(out) == 11                     # the S-only teleseismic row lost its last label
    assert rep["n_with_s_written"] == int(out["s_arrival_sample"].notna().sum()) == 5
    assert out.loc[out.distance_bin == "teleseismic", "s_arrival_sample"].isna().all()
    assert "t11" not in set(out.trace_name)
