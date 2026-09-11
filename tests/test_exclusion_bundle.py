"""
Fixture tests for scripts/exclusion_bundle.py (issue #33, checkpoint 33A):
the versioned bundle (missing and stale lists, rule and policy changes, the
self hash), (dataset, chunk, trace_name) identity, per-reason exclusion counts
and the quarantine policy, cross-source event grouping and split assignment,
noise-window checks on station coordinates, interval overlap, and the
read-only `check` command.

Pure pandas on temp directories; no SeisBench cache.  Run:  pytest tests/ -q
"""
import json
import pathlib
import shutil
import sys
import types

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import exclusion_bundle as eb  # noqa: E402
import heldout_sequences as hs  # noqa: E402

SEQ_ROWS = pd.DataFrame({
    "dataset": ["stead", "mlaapde", "mlaapde"],
    "trace_name": ["bad", "slot1", "slot2"],
    "chunk": ["", "2019-01", ""],          # "" = every chunk of that trace name
    "window": ["norcia_2016_sequence"] * 3,
    "source": ["fixture"] * 3,
})
T0 = "2018-06-01T00:00:00"


@pytest.fixture
def repo(tmp_path):
    """A repository layout with every input the bundle hashes and an empty cache."""
    root = tmp_path / "repo"
    for d in ("data/exclusions", "data/labelerrors", "configs", "notebooks", "cache"):
        (root / d).mkdir(parents=True)
    shutil.copy(REPO_ROOT / "configs" / "evaluation_suites.json", root / "configs" / "evaluation_suites.json")
    SEQ_ROWS.to_csv(root / "data/exclusions/heldout_sequences.csv", index=False)
    pd.DataFrame({"dataset": ["stead"], "trace_name": ["bm"]}).to_csv(root / "notebooks/benchmark_manifest.csv", index=False)
    return root


def _kw(root):
    return dict(repo_root=root, label_error_dirs=[root / "data" / "labelerrors"], cache_root=root / "cache")


def build(root, **extra):
    """Build, write and reload the bundle of the fixture repo."""
    bundle = eb.build_bundle(**_kw(root), **extra)
    path = eb.write_bundle(bundle, root / "data/exclusions/bundle.json")
    return path, eb.load_bundle(path, **_kw(root))


def _signal(rows):
    return pd.DataFrame(rows, columns=["dataset_name", "trace_name", "chunk", hs.TIME_COL, hs.LAT_COL, hs.LON_COL])


def _counts(rep):
    return {k: v for k, v in rep.items() if k.startswith("n_")}


def _fill_cache(root):
    for name in eb.SOURCE_DATASETS:
        d = root / "cache" / "datasets" / name
        d.mkdir(parents=True)
        (d / "metadata.csv").write_text(f"trace_name\n{name}_0\n")


# ── bundle: missing, stale, tampered, rules, policy, self hash ───────────────

def test_missing_sequence_list_fails_closed(repo):
    (repo / "data/exclusions/heldout_sequences.csv").unlink()
    with pytest.raises(eb.MissingExclusionInputError):
        eb.build_bundle(**_kw(repo))
    path, bundle = build(repo, allow_missing_sequence_list=True)
    assert bundle["certified"] is False and bundle["sequence_list_present"] is False
    assert bundle["inputs"]["heldout_sequences_csv"]["sha256"] is None
    with pytest.raises(eb.UncertifiedBundleError):
        eb.load_bundle(path, require_certified=True, **_kw(repo))
    # without the list the trace check cannot run and the report says so
    _, rep = eb.apply_exclusions(_signal([("stead", "bad", "", T0, 0.0, 0.0)]), bundle, kind="signal", repo_root=repo)
    assert rep["trace_list_checked"] is False and rep["n_trace_listed"] == 0
    # the builder path: a bundle recorded absent becomes stale once the list appears
    SEQ_ROWS.to_csv(repo / "data/exclusions/heldout_sequences.csv", index=False)
    with pytest.raises(eb.StaleBundleError):
        eb.load_bundle(path, **_kw(repo))


def test_no_bundle_fails_closed(repo):
    with pytest.raises(eb.MissingExclusionInputError):
        eb.load_bundle(repo / "data/exclusions/bundle.json", **_kw(repo))


def test_stale_sequence_list_fails(repo):
    path, bundle = build(repo)
    (repo / "data/exclusions/heldout_sequences.csv").write_text("dataset,trace_name,chunk\nstead,other,\n")
    with pytest.raises(eb.StaleBundleError, match="heldout_sequences_csv"):
        eb.load_bundle(path, **_kw(repo))
    # the list read for application is hash-checked too, not only at load
    eb._cached_trace_exclusions.cache_clear()
    with pytest.raises(eb.StaleBundleError):
        eb.trace_exclusions(bundle, repo_root=repo)


def test_recorded_list_removed_is_missing_not_stale(repo):
    path, _ = build(repo)
    (repo / "data/exclusions/heldout_sequences.csv").unlink()
    with pytest.raises(eb.MissingExclusionInputError):
        eb.load_bundle(path, **_kw(repo))


def test_tampered_bundle_fails_self_hash(repo):
    path, _ = build(repo)
    raw = json.loads(path.read_text())
    raw["quarantine_policy"]["allow_unknown"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(eb.StaleBundleError, match="self sha256"):
        eb.load_bundle(path, **_kw(repo))
    raw["version"] = 2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="version"):
        eb.load_bundle(path, **_kw(repo))


def test_suite_policy_change_invalidates_bundle(repo):
    path, _ = build(repo)
    policy = json.loads((repo / "configs/evaluation_suites.json").read_text())
    policy["roles"]["samos_2020"] = "acceptance"
    (repo / "configs/evaluation_suites.json").write_text(json.dumps(policy, indent=2))
    with pytest.raises(eb.StaleBundleError, match="evaluation_suites"):
        eb.load_bundle(path, **_kw(repo))


def test_self_hash_stable_across_rebuilds_and_changes_with_windows(repo, monkeypatch):
    a = eb.build_bundle(**_kw(repo))
    b = eb.build_bundle(**_kw(repo))
    assert a["sha256"] == b["sha256"] == eb.self_hash(a)
    assert [w["name"] for w in a["rules"]["windows"]] == hs.WINDOW_NAMES
    assert a["rules"]["holdout_years"] == sorted(hs.HOLDOUT_YEARS)
    path = eb.write_bundle(a, repo / "data/exclusions/bundle.json")
    monkeypatch.setattr(hs, "WINDOWS", hs.WINDOWS + [dict(name="fixture_place", lat=0.0, lon=0.0, radius_deg=0.5,
                                                            start=None, end=None, regime="vt", tier=1)])
    with pytest.raises(eb.StaleBundleError, match="rules"):
        eb.load_bundle(path, **_kw(repo))
    c = eb.build_bundle(**_kw(repo))
    assert c["sha256"] != a["sha256"] and c["rules"]["sha256"] != a["rules"]["sha256"]


def test_certified_only_with_every_source_snapshot(repo):
    _fill_cache(repo)
    path, bundle = build(repo)
    assert bundle["certified"] is True and bundle["uncertified_sources"] == []
    assert all(rec["files"]["metadata.csv"]["n_rows"] == 1 for rec in bundle["sources"].values())
    assert eb.load_bundle(path, require_certified=True, **_kw(repo))["sha256"] == bundle["sha256"]
    (repo / "cache/datasets/stead/metadata.csv").write_text("trace_name\nstead_0\nstead_1\n")
    with pytest.raises(eb.StaleBundleError, match="sources/stead"):
        eb.load_bundle(path, **_kw(repo))
    # sources are certification, not exclusion definitions: without a cache they are unchecked
    rep = {}
    eb.load_bundle(path, repo_root=repo, label_error_dirs=[repo / "data/labelerrors"],
                   cache_root=repo / "nocache", report=rep)
    assert not rep["stale"] and sum("sources/" in m for m in rep["unchecked"]) == len(eb.SOURCE_DATASETS)


def test_label_error_report_appearing_later_is_stale(repo):
    path, _ = build(repo)
    (repo / "data/labelerrors/stead_report.csv").write_text("trace_name\nx\n")
    with pytest.raises(eb.StaleBundleError, match="label_error_reports/stead"):
        eb.load_bundle(path, **_kw(repo))


# ── apply_exclusions: signal kinds ───────────────────────────────────────────

def test_unknown_time_or_location_quarantined_and_counted(repo):
    _, bundle = build(repo)
    df = _signal([
        ("stead", "no_time", "", None, 0.0, 0.0),
        ("stead", "no_loc", "", T0, None, None),
        ("stead", "no_lon", "", T0, 0.0, None),
        ("stead", "ok", "", T0, 0.0, 0.0),
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="signal", repo_root=repo)
    assert kept.trace_name.tolist() == ["ok"]
    assert rep["n_quarantined_unknown"] == 3 and rep["n_unknown_kept_flagged"] == 0 and rep["n_removed"] == 3
    assert not kept[eb.FLAG_COL].any() and rep["allow_unknown"] is False


def test_unknown_rows_kept_and_flagged_when_policy_allows(repo):
    _, bundle = build(repo, allow_unknown=True)
    assert bundle["quarantine_policy"]["allow_unknown"] is True
    df = _signal([
        ("stead", "no_time", "", None, 0.0, 0.0),
        ("stead", "ok", "", T0, 0.0, 0.0),
        ("stead", "bad", "", None, 0.0, 0.0),        # listed rows are removed even without a time
        ("stead", "cf_no_time", "", None, 40.83, 14.14),   # Campi Flegrei: place window, any time
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="signal", repo_root=repo)
    assert kept.trace_name.tolist() == ["no_time", "ok"]
    assert kept[eb.FLAG_COL].tolist() == [True, False]
    assert rep["n_unknown_kept_flagged"] == 1 and rep["n_quarantined_unknown"] == 0
    assert rep["n_trace_listed"] == 1 and rep["n_in_window"] == 1 and rep["n_removed"] == 2


def test_year_exclusion(repo):
    _, bundle = build(repo)
    df = _signal([
        ("stead", "y2016", "", "2016-03-01T00:00:00", 0.0, 0.0),
        ("stead", "y2021", "", "2021-12-31T23:59:59", 0.0, 0.0),
        ("stead", "y2017", "", "2017-01-01T00:00:00", 0.0, 0.0),
        ("stead", "y2022", "", "2022-01-01T00:00:00", 0.0, 0.0),
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="signal", repo_root=repo)
    assert kept.trace_name.tolist() == ["y2017", "y2022"]
    assert rep["n_year_holdout"] == 2 and rep["n_in_window"] == 0 and rep["windows"] == {}


def test_place_exclusion_at_any_time(repo):
    _, bundle = build(repo)
    df = _signal([
        ("instancecounts", "etna_2012", "", "2012-05-01T00:00:00", 37.7, 15.0),
        ("instancecounts", "etna_no_time", "", None, 37.7, 15.0),
        ("instancecounts", "vesuvius", "", "2023-01-01T00:00:00", 40.82, 14.43),
        ("instancecounts", "campi_flegrei_2005", "", "2005-01-01T00:00:00", 40.83, 14.14),
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="signal", repo_root=repo)
    assert kept.trace_name.tolist() == ["vesuvius"]
    assert rep["n_in_window"] == 3 and rep["n_quarantined_unknown"] == 0
    assert rep["windows"] == {"etna": 2, "campi_flegrei": 1}


def test_time_window_exclusion(repo):
    _, bundle = build(repo)
    df = _signal([
        ("instancecounts", "norcia_seq", "", "2016-12-15T00:00:00", 42.9, 13.2),
        ("instancecounts", "norcia_2018", "", "2018-06-01T00:00:00", 42.9, 13.2),
        ("scedc", "ridgecrest_day2", "", "2019-07-07T00:00:00", 35.8, -117.5),
        ("scedc", "ridgecrest_late", "", "2019-09-01T00:00:00", 35.8, -117.5),
        ("scedc", "far_2019", "", "2019-07-07T00:00:00", 34.0, -118.5),
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="signal", repo_root=repo)
    assert kept.trace_name.tolist() == ["norcia_2018", "ridgecrest_late", "far_2019"]
    # the Norcia 2016 row is counted once, as a window hit, not as a 2016 origin
    assert rep["n_in_window"] == 2 and rep["n_year_holdout"] == 0
    assert rep["windows"] == {"norcia_2016_sequence": 1, "ridgecrest_2019": 1}


def test_chunk_aware_trace_keys(repo):
    _, bundle = build(repo)
    df = _signal([
        ("mlaapde", "slot1", "2019-01", T0, 0.0, 0.0),   # listed with this chunk
        ("mlaapde", "slot1", "2020-05", T0, 0.0, 0.0),   # same name, other chunk: a different event
        ("mlaapde", "slot2", "2019-01", T0, 0.0, 0.0),   # listed with chunk "": every chunk
        ("mlaapde", "slot2", "2020-05", T0, 0.0, 0.0),
        ("mlaapde", "slot3", "2019-01", T0, 0.0, 0.0),
        ("stead", "bad", "", T0, 0.0, 0.0),
        ("stead", "slot1", "", T0, 0.0, 0.0),            # listed name in another dataset: kept
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="signal", repo_root=repo)
    assert list(zip(kept.trace_name, kept.chunk)) == [("slot1", "2020-05"), ("slot3", "2019-01"), ("slot1", "")]
    assert rep["n_trace_listed"] == 4 and rep["n_removed"] == 4


def test_legacy_loader_and_chunk_aware_check_manifest(repo):
    path = repo / "data/exclusions/heldout_sequences.csv"
    legacy = hs.load_sequence_exclusions(path)
    assert legacy == {"stead": frozenset({"bad"}), "mlaapde": frozenset({"slot1", "slot2"})}
    aware = hs.load_sequence_exclusions(path, chunk_aware=True)
    assert aware == {"stead": {"bad": frozenset({""})},
                     "mlaapde": {"slot1": frozenset({"2019-01"}), "slot2": frozenset({""})}}
    man = _signal([("mlaapde", "slot1", "2019-01", T0, 0.0, 0.0), ("mlaapde", "slot1", "2020-05", T0, 0.0, 0.0)])
    assert hs.check_manifest(man, aware)["n_excluded_present"] == 1
    assert hs.check_manifest(man, legacy)["n_excluded_present"] == 2   # name-only over-excludes
    assert hs.listed_mask(["slot1", "slot1"], [None, "2019-01"], aware["mlaapde"]).tolist() == [False, True]


def test_reason_counts_partition_removed_rows_and_kind_is_checked(repo):
    _, bundle = build(repo)
    df = _signal([
        ("stead", "bad", "", "2016-12-15T00:00:00", 42.9, 13.2),   # listed AND in-window AND 2016: counted once
        ("stead", "w", "", "2016-12-15T00:00:00", 42.9, 13.2),
        ("stead", "y", "", "2016-01-01T00:00:00", 0.0, 0.0),
        ("stead", "u", "", None, 0.0, 0.0),
        ("stead", "ok", "", T0, 0.0, 0.0),
    ])
    kept, rep = eb.apply_exclusions(df, bundle, kind="mining", repo_root=repo)
    assert _counts(rep) == dict(n_input=5, n_kept=1, n_removed=4, n_trace_listed=1, n_in_window=1,
                                n_year_holdout=1, n_quarantined_unknown=1, n_unknown_kept_flagged=0)
    assert rep["bundle_sha256"] == bundle["sha256"] and df.shape[1] == 6   # input not modified
    with pytest.raises(ValueError):
        eb.apply_exclusions(df, bundle, kind="test", repo_root=repo)


def test_mixed_time_formats_are_not_quarantined():
    t = hs._to_utc(pd.Series(["2018-01-01", "2016-10-30T07:00:00", "2013-01-06 11:29:23.29",
                              "2019-01-01T00:00:01.5Z", None, "not a time"]))
    assert t.notna().tolist() == [True, True, True, True, False, False]
    assert hs._to_utc(pd.Series([np.nan, np.nan])).isna().all()


# ── cross-source duplicates and split assignment ─────────────────────────────

def _events():
    return pd.DataFrame({
        "dataset": ["stead", "instancecounts", "stead", "pnw", "pnw", "stead", "scedc", "scedc"],
        "event_keys": ["usgs:1", "usgs:1", "", frozenset(), "", "", "", None],
        hs.TIME_COL: ["2018-01-01T00:00:00", "2018-01-01T00:00:00",
                      "2018-02-01T00:00:00", "2018-02-01T00:00:01.5", "2018-02-01T00:00:06",
                      "2018-02-01T00:00:00", "2018-02-01T00:00:01", None],
        hs.LAT_COL: [10.0, 10.0, 20.0, 20.05, 20.0, 20.0, 20.3, 20.0],
        hs.LON_COL: [10.0, 10.0, 20.0, 20.05, 20.0, 20.0, 20.0, 20.0],
    })


def test_cross_source_duplicates_by_key_and_by_origin_tolerance():
    fr = _events()
    g = eb.event_groups(fr)
    assert g[0] == g[1]                         # shared key across datasets
    assert g[2] == g[3] == g[5]                 # identical origin, and 1.5 s / 0.07 deg away
    assert g[4] != g[2] and g[6] != g[2] and g[7] != g[2]   # 6 s late, 0.3 deg away, no time
    assert len(set(g)) == 5
    dup = eb.cross_source_duplicates(fr)
    assert sorted(dup.index.tolist()) == [0, 1, 2, 3, 5]
    assert dup.groupby("group_id")["dataset"].nunique().tolist() == [2, 2]
    assert eb.cross_source_duplicates(fr, time_tol_s=0.5, dist_tol_deg=0.01).index.tolist() == [1, 0]
    assert eb.cross_source_duplicates(fr, time_tol_s=10.0, dist_tol_deg=0.5).groupby("group_id").size().tolist() == [2, 5]


def test_origin_unions_collapse_identical_fingerprints():
    n = 300
    t = ["2018-02-01T00:00:00"] * n + ["2018-02-01T00:00:01"] * n
    pairs = eb.origin_unions(t, [20.0] * (2 * n), [20.0] * (2 * n))
    assert len(pairs) == 2 * (n - 1) + 1    # one pair per duplicate row, one per near pair of fingerprints
    assert len(eb.origin_unions([None, None], [0, 0], [0, 0])) == 0


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_groups_land_in_one_split(seed):
    rng = np.random.default_rng(seed)
    fr = _events().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    extra = pd.DataFrame({"dataset": ["geofon"] * 120, "event_keys": [""] * 120,
                          hs.TIME_COL: pd.date_range("2015-01-01", periods=120, freq="1h").strftime("%Y-%m-%dT%H:%M:%S"),
                          hs.LAT_COL: np.linspace(-60, 60, 120), hs.LON_COL: np.linspace(-170, 170, 120)})
    fr = pd.concat([fr, extra], ignore_index=True)
    groups = eb.event_groups(fr)
    split = eb.fill_splits_by_group(groups, rng)
    fr["split"], fr["group"] = split, groups
    assert (fr.groupby("group")["split"].nunique() == 1).all()
    n = len(fr)
    assert (split == "val").sum() >= int(0.1 * n) and (split == "test").sum() >= int(0.1 * n)
    assert set(split) == {"train", "val", "test"}
    both = fr[fr["event_keys"].astype(str) == "usgs:1"]
    assert both["split"].nunique() == 1 and both["dataset"].nunique() == 2


# ── noise rows: station coordinates and start time ───────────────────────────

def test_noise_windows_on_station_coordinates_and_start_time(repo):
    _, bundle = build(repo)
    nd = pd.DataFrame({
        "trace_name": ["etna_any_time", "norcia_in_seq", "norcia_2019", "no_station", "no_start", "y2021", "far"],
        "latitude":   [37.7, 42.9, 42.9, None, 0.0, 0.0, 0.0],
        "longitude":  [15.0, 13.2, 13.2, None, 0.0, 0.0, 0.0],
        "starttime":  ["2019-01-01T00:00:00", "2016-11-01T00:00:00", "2019-01-01T00:00:00",
                       "2019-01-01T00:00:00", None, "2021-05-05T00:00:00", "2019-01-01T00:00:00"],
        hs.TIME_COL: [None] * 7,   # a source origin is never used for noise rows
    })
    kept, rep = eb.apply_exclusions(nd, bundle, kind="noise", dataset="noise_global", repo_root=repo)
    assert kept.trace_name.tolist() == ["norcia_2019", "far"]
    assert _counts(rep) == dict(n_input=7, n_kept=2, n_removed=5, n_trace_listed=0, n_in_window=2,
                                n_year_holdout=1, n_quarantined_unknown=2, n_unknown_kept_flagged=0)
    assert rep["columns"] == {"time": "starttime", "lat": "latitude", "lon": "longitude"}
    assert rep["windows"] == {"norcia_2016_sequence": 1, "etna": 1}
    # SeisBench column names are detected too
    sb = pd.DataFrame({"station_latitude_deg": [37.7], "station_longitude_deg": [15.0], "trace_start_time": ["2019-01-01"]})
    _, rep2 = eb.apply_exclusions(sb, bundle, kind="noise", dataset="vcseis", repo_root=repo)
    assert rep2["n_in_window"] == 1 and rep2["columns"]["time"] == "trace_start_time"


def test_overlapping_intervals_same_station_across_datasets_or_roles():
    fr = pd.DataFrame({
        "station": ["IV.X", "IV.X", "IV.X", "IV.X", "IV.Y", "IV.Y"],
        "dataset": ["instancecounts", "noise_global", "instancecounts", "instancecounts", "instancecounts", "stead"],
        "role":    ["signal", "noise", "signal", "signal", "signal", "signal"],
        "start":   ["2019-01-01T00:00:00", "2019-01-01T00:00:30", "2019-01-01T00:00:10",
                    "2019-01-01T00:01:00", "2019-01-01T00:00:00", "2019-01-01T00:02:00"],
        "end":     ["2019-01-01T00:01:00", "2019-01-01T00:01:30", "2019-01-01T00:00:40",
                    "2019-01-01T00:02:00", "2019-01-01T00:01:00", "2019-01-01T00:03:00"],
    })
    got = eb.overlapping_intervals(fr)
    assert list(got.columns) == eb.OVERLAP_COLS
    assert [(r.left, r.right, r.overlap_s) for r in got.itertuples()] == [(0, 1, 30.0), (2, 1, 10.0), (1, 3, 30.0)]
    assert set(got.role_left) | set(got.role_right) == {"signal", "noise"}
    # same dataset and role (rows 0 and 2), touching intervals (0 and 3) and different stations are not flagged
    assert eb.overlapping_intervals(fr.iloc[[0, 2, 3, 4, 5]]).empty
    assert len(eb.overlapping_intervals(fr.drop(columns=["dataset", "role"]))) == 4   # every overlap without identity columns


# ── check command, historical manifests ──────────────────────────────────────

def test_check_command_exit_code_and_read_only(repo, capsys):
    path, _ = build(repo)
    argv = ["--bundle", str(path), "--repo-root", str(repo), "--label-error-dir", str(repo / "data/labelerrors"),
            "--cache-root", str(repo / "cache")]
    dirty = repo / "dirty.csv"
    _signal([("stead", "bad", "", T0, 0.0, 0.0), ("stead", "ok", "", T0, 0.0, 0.0),
             ("stead", "u", "", None, 0.0, 0.0)]).to_csv(dirty, index=False)
    before = dirty.read_bytes()
    assert eb.main(["check", str(dirty)] + argv) == 2
    assert dirty.read_bytes() == before
    out = capsys.readouterr().out
    assert '"n_trace_listed": 1' in out and "FAIL: 1 excluded row(s) present; 1 row(s) of unknown independence" in out
    clean = repo / "clean.csv"
    _signal([("stead", "ok", "", T0, 0.0, 0.0)]).to_csv(clean, index=False)
    assert eb.main(["check", str(clean), "--kind", "validation"] + argv) == 0
    assert "PASS: 0 excluded row(s)" in capsys.readouterr().out
    assert eb.main(["show", "--bundle", str(path)]) == 0
    assert "self hash           : ok" in capsys.readouterr().out


def test_historical_manifests_are_refused(tmp_path):
    csv = tmp_path / "manifest_checksums.csv"
    pd.DataFrame({"manifest": ["data/manifests_v2/train.csv"], "n_rows": [1], "sha256_of_sorted_keys": ["x"]}).to_csv(csv, index=False)
    with pytest.raises(PermissionError, match="immutable"):
        eb.assert_not_historical(tmp_path / "data/manifests_v2/train.csv", checksums_csv=csv, repo_root=tmp_path)
    eb.assert_not_historical(tmp_path / "data/manifests_v9/train.csv", checksums_csv=csv, repo_root=tmp_path)
    p = eb.append_provenance(tmp_path, "noise_appends", {"n": 1})
    eb.append_provenance(tmp_path, "noise_appends", {"n": 2})
    assert [r["n"] for r in json.loads(p.read_text())["noise_appends"]] == [1, 2]


# ── the manifest builder end to end, without SeisBench ───────────────────────

def _fake_source(name, rows):
    """A DATASET_CONFIGS entry whose metadata is the given frame (no cache needed)."""
    frame = pd.DataFrame(rows, columns=["trace_name", "chunk", "trace_P_arrival_sample", "trace_S_arrival_sample",
                                        hs.TIME_COL, hs.LAT_COL, hs.LON_COL, "split"])
    return dict(name=name, cls=None, meta_fn=lambda: frame.copy(), dist_col=None, dist_unit="km",
                cap=10_000, default_bin="local", use_s=True)


def test_build_training_dataset_applies_bundle_and_writes_provenance(repo, monkeypatch, tmp_path):
    """build_training_dataset.main with two synthetic sources: the bundle is
    applied per source, the final gate passes, provenance.json and
    heldout_removal_report.csv carry the counts, and an event present in both
    sources lands in one split."""
    fake_sb = types.ModuleType("seisbench")
    fake_data = types.ModuleType("seisbench.data")
    fake_data.__getattr__ = lambda name: object          # sbd.STEAD etc. at import time
    fake_ek = types.ModuleType("event_keys")
    fake_ek.trace_key_map = lambda name: {}
    fake_ek.trace_key_map_any_chunk = lambda name: {}
    monkeypatch.setitem(sys.modules, "seisbench", fake_sb)
    monkeypatch.setitem(sys.modules, "seisbench.data", fake_data)
    monkeypatch.setitem(sys.modules, "event_keys", fake_ek)
    monkeypatch.delitem(sys.modules, "build_training_dataset", raising=False)
    import build_training_dataset as btd

    _fill_cache(repo)                                     # every source hashed: certified
    monkeypatch.setenv("SEISBENCH_CACHE_ROOT", str(repo / "cache"))
    path, bundle = build(repo)
    assert bundle["certified"] is True
    monkeypatch.setattr(eb, "REPO_ROOT", repo)
    monkeypatch.setattr(eb, "BUNDLE_PATH", path)
    monkeypatch.setattr(eb, "USER_LABEL_ERROR_CACHE", repo / "nowhere")
    monkeypatch.setattr(btd, "BENCHMARK_CSV", repo / "notebooks" / "benchmark_manifest.csv")
    eb._cached_trace_exclusions.cache_clear()

    rng = np.random.default_rng(3)
    origins = pd.date_range("2018-01-01", periods=40, freq="6h").strftime("%Y-%m-%dT%H:%M:%S")
    stead = [(f"s{i}", "", 500.0, 900.0, origins[i], 30.0 + i * 0.5, 10.0 + i * 0.5, "train") for i in range(40)]
    stead += [("bad", "", 500.0, 900.0, "2018-05-01T00:00:00", 0.0, 0.0, "train"),         # listed
              ("norcia", "", 500.0, 900.0, "2016-12-15T00:00:00", 42.9, 13.2, "train"),     # window
              ("y2021", "", 500.0, 900.0, "2021-07-01T00:00:00", 0.0, 0.0, "train"),        # year
              ("unknown", "", 500.0, 900.0, None, 0.0, 0.0, "train")]                      # quarantined
    pnw = [(f"p{i}", "", 500.0, 900.0, origins[i], 30.0 + i * 0.5 + 0.02, 10.0 + i * 0.5, "test") for i in range(0, 40, 4)]
    pnw += [("bm", "", 500.0, 900.0, "2018-05-02T00:00:00", 0.0, 0.0, "train")]           # benchmark trace
    monkeypatch.setattr(btd, "DATASET_CONFIGS", [_fake_source("stead", stead), _fake_source("pnw", pnw)])
    monkeypatch.setattr(btd, "SKIP_SOURCES_THIS_ROUND", frozenset())

    out = tmp_path / "manifests_fixture"
    btd.main(str(out), seed=int(rng.integers(1000)), label_error_filter=False)

    splits = {name: pd.read_csv(out / f"{name}.csv") for name in ("train", "val", "test")}
    written = pd.concat([df.assign(split=name) for name, df in splits.items()], ignore_index=True)
    assert set(written.columns) >= {"dataset_name", "trace_name", "chunk", hs.TIME_COL, eb.FLAG_COL}
    assert not written[eb.FLAG_COL].astype(bool).any()
    assert not set(written.trace_name) & {"bad", "norcia", "y2021", "unknown", "bm"}
    # the pnw rows duplicate every fourth stead origin (0.02 deg away): one split per event
    for i in range(0, 40, 4):
        pair = written[written.trace_name.isin([f"s{i}", f"p{i}"])]
        assert pair["split"].nunique() <= 1, (i, pair[["dataset_name", "trace_name", "split"]].to_dict("records"))
    assert any(written[written.trace_name.isin([f"s{i}", f"p{i}"])]["dataset_name"].nunique() == 2 for i in range(0, 40, 4))

    report = pd.read_csv(out / "heldout_removal_report.csv").set_index("dataset")
    assert report.loc["stead", ["n_trace_listed", "n_in_window", "n_year_holdout", "n_quarantined_unknown"]].tolist() == [1, 1, 1, 1]
    assert report.loc["pnw", "n_removed"] == 0 and (report["bundle_sha256"] == bundle["sha256"]).all()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["bundle_sha256"] == bundle["sha256"] and prov["bundle_certified"] is True
    assert prov["options"]["allow_uncertified_bundle"] is False and prov["git_commit"] is None
    assert prov["bundle_policy_sha256"] == bundle["inputs"]["evaluation_suites"]["policy_sha256"]
    assert set(prov["manifests"]) == {"train.csv", "val.csv", "test.csv"}
    assert all(prov["gate"][k]["bundle"]["n_removed"] == 0 for k in prov["gate"])
    assert sum(m["n_rows"] for m in prov["manifests"].values()) == len(written)


def test_build_training_dataset_refuses_without_certified_bundle(repo, monkeypatch, tmp_path):
    fake_data = types.ModuleType("seisbench.data")
    fake_data.__getattr__ = lambda name: object
    fake_ek = types.ModuleType("event_keys")
    fake_ek.trace_key_map_any_chunk = lambda name: {}
    monkeypatch.setitem(sys.modules, "seisbench", types.ModuleType("seisbench"))
    monkeypatch.setitem(sys.modules, "seisbench.data", fake_data)
    monkeypatch.setitem(sys.modules, "event_keys", fake_ek)
    monkeypatch.delitem(sys.modules, "build_training_dataset", raising=False)
    import build_training_dataset as btd
    monkeypatch.setattr(eb, "REPO_ROOT", repo)
    monkeypatch.setattr(eb, "BUNDLE_PATH", repo / "data/exclusions/bundle.json")
    monkeypatch.setattr(eb, "USER_LABEL_ERROR_CACHE", repo / "nowhere")
    monkeypatch.setattr(btd, "BENCHMARK_CSV", repo / "notebooks" / "benchmark_manifest.csv")
    monkeypatch.setattr(btd, "DATASET_CONFIGS", [])
    with pytest.raises(eb.MissingExclusionInputError):          # no bundle at all
        btd.main(str(tmp_path / "out"), seed=0, label_error_filter=False)
    build(repo)                                                  # a bundle without source hashes
    with pytest.raises(eb.UncertifiedBundleError):
        btd.main(str(tmp_path / "out"), seed=0, label_error_filter=False)
    with pytest.raises(SystemExit, match="no datasets loaded"):  # escape hatch: proceeds to the source loop
        btd.main(str(tmp_path / "out"), seed=0, label_error_filter=False, allow_uncertified_bundle=True)
    with pytest.raises(ValueError, match="exclusion bundle"):
        btd.process_dataset(_fake_source("stead", [("s0", "", 500.0, 900.0, T0, 0.0, 0.0, "train")]),
                            np.random.default_rng(0))
