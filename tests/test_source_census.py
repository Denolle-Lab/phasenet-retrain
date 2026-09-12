"""
Unit tests for scripts/source_census.py (checkpoint 39A of issue #39), on
synthetic QuakeML-like frames and without network: the query function is
monkeypatched.  Run with:  python -m pytest tests -q
"""
import datetime
import json
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import heldout_sequences as hs  # noqa: E402
import source_census as sc  # noqa: E402

PICK_COLS = ["sequence", "event", "origin", "mag", "station", "channel", "phase", "time",
             "mode", "status", "method", "agency", "time_weight", "onset", "uncertainty",
             "network", "source", "reference_ok"]


def _catalog(rows):
    """rows: (event, origin iso, lat, lon, mag)"""
    return pd.DataFrame([dict(event=e, origin=pd.Timestamp(o, tz="UTC"), lat=la, lon=lo, depth_km=10.0, mag=m, source="T")
                         for e, o, la, lo, m in rows], columns=["event", "origin", "lat", "lon", "depth_km", "mag", "source"])


def _picks(rows):
    """rows: (event, station, phase, mode)"""
    out = []
    for e, s, ph, mode in rows:
        out.append(dict(sequence="d", event=e, origin=pd.Timestamp("2018-01-27T00:00:00Z"), mag=1.0, station=s,
                        channel="HHZ", phase=ph, time=pd.Timestamp("2018-01-27T00:00:05Z"), mode=mode, status="unknown",
                        method="", agency="", time_weight=np.nan, onset="", uncertainty=np.nan, network="IV", source="T",
                        reference_ok=(mode == "manual")))
    return pd.DataFrame(out, columns=PICK_COLS)


# ── deterministic day sampling ─────────────────────────────────────────────────

def test_sample_days_is_deterministic_stratified_and_seed_dependent():
    a = sc.sample_days("INGV", 2018, 2, "39a")
    b = sc.sample_days("INGV", 2018, 2, "39a")
    assert a == b
    assert [q for q, _ in a] == [1, 1, 2, 2, 3, 3, 4, 4]
    for q, d in a:
        assert d.year == 2018 and (d.month - 1) // 3 + 1 == q
    assert len({d for _, d in a}) == 8                       # distinct days
    assert sc.sample_days("INGV", 2018, 2, "other") != a     # the seed tag moves the sample
    assert sc.sample_days("NOA", 2018, 2, "39a") != a        # so does the operator
    assert sc.sample_days("INGV", 2019, 2, "39a") != a       # and the year


def test_sample_days_takes_whole_quarters_when_the_target_reaches_the_quarter_length():
    full = sc.sample_days("INGV", 2018, 92, "39a")               # Q3 and Q4 of 2018 have 92 days
    assert [d for _, d in full] == [d for q in (1, 2, 3, 4) for d in sc.quarter_days(2018, q)]
    assert sc.sample_days("INGV", 2018, 10 ** 6, "39a") == full
    mostly = sc.sample_days("INGV", 2018, 91, "39a")             # Q1 (90) and Q2 (91) whole, Q3 and Q4 drawn
    per_q = {q: [d for qq, d in mostly if qq == q] for q in (1, 2, 3, 4)}
    assert per_q[1] == sc.quarter_days(2018, 1) and per_q[2] == sc.quarter_days(2018, 2)
    assert len(per_q[3]) == len(per_q[4]) == 91 and set(per_q[3]) < set(sc.quarter_days(2018, 3))
    assert per_q[3] == sorted(per_q[3])


def test_sample_days_grows_as_a_prefix():
    one = sc.sample_days("NOA", 2019, 1, "39a"); two = sc.sample_days("NOA", 2019, 2, "39a")
    assert len(one) == 4 and len(two) == 8 and set(one) < set(two)


def test_quarter_days_cover_the_year_exactly_once():
    days = [d for q in (1, 2, 3, 4) for d in sc.quarter_days(2020, q)]
    assert len(days) == 366 and len(set(days)) == 366
    assert days[0] == datetime.date(2020, 1, 1) and days[-1] == datetime.date(2020, 12, 31)


def test_subset_events_is_deterministic_and_bounded():
    ids = [f"e{i}" for i in range(50)]
    s1 = sc.subset_events(ids, 10, "39a"); s2 = sc.subset_events(ids, 10, "39a")
    assert s1 == s2 and len(s1) == 10 and set(s1) <= set(ids)
    assert sc.subset_events(ids, None, "39a") == ids and sc.subset_events(ids[:5], 10, "39a") == ids[:5]


# ── per-day counting ─────────────────────────────────────────────────────────

def test_count_day_counts_modes_phases_stations_and_magnitudes():
    cat = _catalog([("e1", "2018-01-27T01:00:00", 42.0, 13.0, 0.5),
                    ("e2", "2018-01-27T02:00:00", 42.0, 13.0, 1.5),
                    ("e3", "2018-01-27T03:00:00", 42.0, 13.0, 2.5),
                    ("e4", "2018-01-27T04:00:00", 42.0, 13.0, 3.5),
                    ("e5", "2018-01-27T05:00:00", 42.0, 13.0, 4.5),
                    ("e6", "2018-01-27T06:00:00", 42.0, 13.0, np.nan)])
    picks = _picks([("e1", "IV.AAA", "P", "manual"), ("e1", "IV.AAA", "S", "manual"),
                    ("e1", "IV.BBB", "P", "automatic"), ("e2", "IV.AAA", "P", "manual"),
                    ("e2", "IV.CCC", "S", None), ("e3", "IV.AAA", "P", "unknown")])
    r = sc.count_day(cat, picks)
    assert r["events"] == 6 and r["events_with_arrivals"] == 3
    assert r["p_manual"] == 2 and r["s_manual"] == 1 and r["p_automatic"] == 1 and r["s_automatic"] == 0
    assert r["p_unknown"] == 1 and r["s_unknown"] == 1 and r["p_total"] == 4 and r["s_total"] == 2
    assert r["readings"] == 6 and r["stations"] == 3 and r["readings_per_event"] == pytest.approx(2.0)
    assert [r[k] for k in sc.MAG_LABELS] == [1, 1, 1, 1, 1] and r["mag_unknown"] == 1
    assert r["scale"] == 1.0 and r["readings_est"] == 6 and r["p_manual_est"] == 2


def test_count_day_scales_the_per_event_subset():
    cat = _catalog([(f"e{i}", "2018-01-27T01:00:00", 42.0, 13.0, 1.0) for i in range(10)])
    picks = _picks([("e0", "IV.AAA", "P", "manual"), ("e1", "IV.AAA", "P", "manual")])
    r = sc.count_day(cat, picks, scale=5.0)
    assert r["p_manual"] == 2 and r["p_manual_est"] == 10.0 and r["events_est"] == 10.0
    assert r["events_with_arrivals"] == 2 and r["events_with_arrivals_est"] == 10.0


def test_count_day_on_empty_inputs():
    r = sc.count_day(_catalog([]), _picks([]))
    assert r["events"] == 0 and r["readings"] == 0 and r["stations"] == 0
    assert np.isnan(r["readings_per_event"]) and np.isnan(r["heldout_readings_share"])


# ── held-out share ───────────────────────────────────────────────────────────

def test_heldout_share_uses_windows_places_and_years():
    cat = _catalog([("norcia", "2016-10-30T07:00:00", 42.9, 13.2, 2.0),      # Norcia mainshock window + year 2016
                    ("etna", "2018-01-27T01:00:00", 37.75, 15.0, 1.0),      # Etna place hold-out, any time
                    ("y2021", "2021-06-01T00:00:00", 45.0, 8.0, 1.0),       # year hold-out only
                    ("clean", "2018-01-27T02:00:00", 45.0, 8.0, 1.0)])
    picks = _picks([("norcia", "IV.A", "P", "manual"), ("etna", "IV.A", "P", "manual"),
                    ("y2021", "IV.A", "P", "manual"), ("clean", "IV.A", "P", "manual"), ("clean", "IV.B", "S", "manual")])
    r = sc.count_day(cat, picks)
    assert r["events_heldout_window"] == 1 and r["events_heldout_place"] == 1 and r["events_heldout_year"] == 2
    assert r["events_heldout_any"] == 3 and r["readings_heldout_any"] == 3
    assert r["heldout_readings_share"] == pytest.approx(3 / 5) and r["heldout_events_share"] == pytest.approx(3 / 4)
    assert "norcia_2016_mainshock" in r["heldout_windows"] and "etna" in r["heldout_windows"]
    assert set(hs.HOLDOUT_YEARS) == {2016, 2021}


# ── extrapolation with intervals ─────────────────────────────────────────────

def test_bootstrap_year_scales_the_day_mean_and_brackets_it():
    est, lo, hi, boots = sc.bootstrap_year([10, 20, 30, 40], 365, seed=1, n_boot=500)
    assert est == pytest.approx(365 * 25)
    assert lo <= est <= hi and 365 * 10 <= lo and hi <= 365 * 40 and len(boots) == 500
    est2, lo2, hi2, _ = sc.bootstrap_year([10, 20, 30, 40], 365, seed=1, n_boot=500)
    assert (est2, lo2, hi2) == (est, lo, hi)                # seeded


def test_bootstrap_year_edge_cases():
    est, lo, hi, _ = sc.bootstrap_year([12], 365, seed=1)
    assert est == 365 * 12 and np.isnan(lo) and np.isnan(hi)     # one day: no interval
    est, lo, hi, _ = sc.bootstrap_year([], 365, seed=1)
    assert np.isnan(est)
    est, lo, hi, _ = sc.bootstrap_year([np.nan, 5, 5], 365, seed=1)
    assert est == 365 * 5 and lo == est == hi                     # NaN dropped, constant days


# ── census_day with an injected fetch: failure accounting ────────────────────

def _fetch_ok(operator, day, raw_dir, cost, budget, min_magnitude=None, max_events=None, seed_tag=""):
    cost.n_queries += 3; cost.seconds += 1.5; cost.bytes += 3000
    cat = _catalog([("a", f"{day.isoformat()}T01:00:00", 42.0, 13.0, 1.2), ("b", f"{day.isoformat()}T02:00:00", 42.0, 13.0, 2.2)])
    picks = _picks([("a", "IV.A", "P", "manual"), ("a", "IV.A", "S", "manual"), ("b", "IV.B", "P", "automatic")])
    return dict(catalog=cat, picks=picks, events_fetched=2, note="")


def _fetch_fail(operator, day, raw_dir, cost, budget, **kw):
    cost.n_queries += 3; cost.n_retries += 2; cost.n_failed += 1; cost.seconds += 40.0
    raise RuntimeError("failed after 3 tries: HTTP 503")


def test_census_day_ok_and_failed_rows(tmp_path):
    b = sc.Budget(None)
    ok = sc.census_day("INGV", 2018, 1, datetime.date(2018, 1, 27), "t", tmp_path, b, fetch=_fetch_ok)
    assert ok["status"] == "ok" and ok["events"] == 2 and ok["p_manual"] == 1 and ok["s_manual"] == 1 and ok["p_automatic"] == 1
    assert ok["n_queries"] == 3 and ok["seconds"] == 1.5 and ok["bytes"] == 3000 and ok["n_failed_queries"] == 0
    assert set(sc.DAY_COLUMNS) <= set(ok)
    bad = sc.census_day("INGV", 2018, 1, datetime.date(2018, 1, 28), "t", tmp_path, b, fetch=_fetch_fail)
    assert bad["status"] == "failed" and "503" in bad["error"] and bad["n_failed_queries"] == 1 and bad["n_retries"] == 2
    assert np.isnan(bad["events"]) and np.isnan(bad["p_manual"])


def test_census_day_respects_an_exhausted_budget(tmp_path):
    b = sc.Budget(0.0); b.t0 -= 10.0          # already over
    r = sc.census_day("NOA", 2018, 1, datetime.date(2018, 1, 27), "t", tmp_path, b, fetch=_fetch_ok)
    assert r["status"] == "not_attempted" and r["n_queries"] == 0


def test_http_get_reuses_the_raw_cache_and_counts_its_cost(tmp_path, monkeypatch):
    url = "http://example/query?starttime=2018-01-27T00:00:00&endtime=2018-01-28T00:00:00"
    path, meta = sc.cache_paths(url, tmp_path / "x.xml")
    assert path.parent == tmp_path and path.suffix == ".xml" and path.name.startswith("x.") and path.name != "x.xml"
    assert meta == path.with_name(path.name + ".meta.json")
    path.write_bytes(b"<q/>")
    meta.write_text(json.dumps(dict(status=200, seconds=2.5, bytes=4, url=url, when="w")))
    monkeypatch.setattr(sc.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    cost = sc.Cost()
    status, got = sc.http_get(url, tmp_path / "x.xml", cost)
    assert status == 200 and got == path and cost.n_queries == 1 and cost.n_cached == 1 and cost.seconds == 2.5 and cost.bytes == 4


class _Response:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_get_refetches_when_the_cached_url_differs(tmp_path, monkeypatch):
    url_a = "http://example/query?starttime=2018-01-27T00:00:00&endtime=2018-01-28T00:00:00"
    url_b = url_a + "&minmagnitude=2"
    path_a, meta_a = sc.cache_paths(url_a, tmp_path / "x.xml")
    path_b, meta_b = sc.cache_paths(url_b, tmp_path / "x.xml")
    assert path_a != path_b                                # the parameters change the cache file
    path_b.write_bytes(b"<stale/>")                        # a body under url_b's name whose sidecar records url_a
    meta_b.write_text(json.dumps(dict(status=200, seconds=2.5, bytes=8, url=url_a, when="w")))
    calls = []
    monkeypatch.setattr(sc.urllib.request, "urlopen",
                        lambda req, timeout=None: calls.append(req.full_url) or _Response(b"<fresh/>"))
    cost = sc.Cost()
    status, got = sc.http_get(url_b, tmp_path / "x.xml", cost)
    assert status == 200 and got == path_b and calls == [url_b]
    assert path_b.read_bytes() == b"<fresh/>" and json.loads(meta_b.read_text())["url"] == url_b
    assert cost.n_queries == 1 and cost.n_cached == 0 and cost.bytes == 8
    status, got = sc.http_get(url_b, tmp_path / "x.xml", cost)          # now cached under its own URL
    assert got == path_b and calls == [url_b] and cost.n_cached == 1
    status, got = sc.http_get(url_a, tmp_path / "x.xml", cost)          # the other parameter set is its own file
    assert got == path_a and calls == [url_b, url_a] and path_b.read_bytes() == b"<fresh/>"


# ── run_bulletin, summary assembly ───────────────────────────────────────────

def test_run_bulletin_writes_day_files_and_summary(tmp_path):
    calls = []

    def fetch(operator, day, raw_dir, cost, budget, min_magnitude=None, max_events=None, seed_tag=""):
        calls.append((operator, day))
        if operator == "NOA" and day.month in (4, 5, 6):          # NOA's Q2 day fails, whichever day is drawn
            return _fetch_fail(operator, day, raw_dir, cost, budget)
        return _fetch_ok(operator, day, raw_dir, cost, budget)

    days = sc.run_bulletin(["INGV", "NOA"], [2018, 2019], 1, "t", None, out_dir=tmp_path, raw_dir=tmp_path / "raw", fetch=fetch)
    assert len(days) == 16 and len(calls) == 16
    for op in ("INGV", "NOA"):
        for y in (2018, 2019):
            assert (tmp_path / f"bulletin_{op}_{y}.csv").exists()
    summary = pd.read_csv(tmp_path / "bulletin_summary.csv")
    assert list(summary["operator"]) == ["INGV", "INGV", "INGV", "NOA", "NOA", "NOA"]
    assert list(summary["year"].astype(str)) == ["2018", "2019", "all"] * 2
    ingv18 = summary[(summary.operator == "INGV") & (summary.year.astype(str) == "2018")].iloc[0]
    assert ingv18["days_sampled"] == 4 and ingv18["days_ok"] == 4 and ingv18["days_failed"] == 0
    assert ingv18["p_manual_year_est"] == pytest.approx(365 * 1.0) and ingv18["s_manual_year_est"] == pytest.approx(365.0)
    assert ingv18["events_year_est"] == pytest.approx(365 * 2.0)
    assert ingv18["p_manual_year_lo"] == pytest.approx(365.0) and ingv18["p_manual_year_hi"] == pytest.approx(365.0)  # constant days
    assert ingv18["n_queries"] == 12 and ingv18["bytes_total"] == 12000 and ingv18["seconds_per_query"] == pytest.approx(0.5)
    assert ingv18["manual_share_of_readings"] == pytest.approx(2 / 3) and ingv18["stations_distinct_sampled"] == 2
    noa18 = summary[(summary.operator == "NOA") & (summary.year.astype(str) == "2018")].iloc[0]
    assert noa18["days_ok"] == 3 and noa18["days_failed"] == 1 and noa18["n_failed_queries"] == 1 and "503" in noa18["errors"]
    assert noa18["p_manual_year_est"] == pytest.approx(365.0)        # the failed day is not a zero
    all_ingv = summary[(summary.operator == "INGV") & (summary.year.astype(str) == "all")].iloc[0]
    assert all_ingv["events_year_est"] == pytest.approx(365 * 2.0 + 365 * 2.0) and all_ingv["days_sampled"] == 8
    # the summary is rebuilt from the day files alone
    again = sc.write_summary(tmp_path)
    assert len(again) == 6 and again["p_manual_year_est"].tolist() == summary["p_manual_year_est"].tolist()


def test_summarise_with_all_days_failed_keeps_the_row_with_nan_estimates():
    row = sc._empty_day_row("ISC", 2019, 1, datetime.date(2019, 1, 1), "t", "failed", "HTTP 503")
    row["n_queries"] = 3; row["n_failed_queries"] = 1
    s = sc.summarise(pd.DataFrame([row], columns=sc.DAY_COLUMNS))
    assert len(s) == 2 and s.iloc[0]["days_failed"] == 1 and np.isnan(s.iloc[0]["p_manual_year_est"])
    assert s.iloc[0]["n_failed_queries"] == 1


def test_run_bulletin_rejects_unknown_operator(tmp_path):
    with pytest.raises(ValueError):
        sc.run_bulletin(["NOPE"], [2018], 1, "t", None, out_dir=tmp_path, raw_dir=tmp_path / "raw", fetch=_fetch_ok)


# ── SeisBench table (laptop side, from the committed records) ────────────────

def test_seisbench_table_reads_configs_and_marks_server_columns():
    t = sc.seisbench_table()
    assert set(t["source"]) >= {"stead", "instancecounts", "geofon", "lendb", "neic", "pnw", "scedc"}
    assert (t["server_required"] == "yes").all() and (t["heldout_overlap_counts"] == "server").all()
    assert (t["server_command"] == "python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT").all()
    by = t.set_index("source")
    assert by.loc["geofon", "train_cap"] == 150000 and by.loc["geofon", "use_s"] == False  # noqa: E712
    assert "teleseismic" in by.loc["geofon", "label_policy"]
    assert by.loc["lendb", "status_exposed"] == "yes" and "estimated" in by.loc["lendb", "status_values_seen"]
    assert by.loc["neic", "in_training_pool"] == False and "not in the training pool" in by.loc["neic", "label_policy"]  # noqa: E712
    assert by.loc["pnw", "status_exposed"] == "no" and by.loc["pnw", "n_p"] == 183909
    assert by.loc["stead", "n_traces"] == 1265657


def test_parse_dataset_configs_matches_the_builder():
    cfg = sc.parse_dataset_configs().set_index("source")
    assert len(cfg) == 20 and cfg.loc["ross2018gpd", "cap"] == 200000 and cfg.loc["meier2019jgr", "use_s"] == False  # noqa: E712


def test_status_counts_and_heldout_counts_server_helpers():
    meta = pd.DataFrame({
        "trace_p_status": ["manual", "manual", "automatic", None, "estimated"],
        "trace_s_status": [None, "manual", None, None, None],
        hs.TIME_COL: ["2016-10-30T07:00:00", "2018-01-01T00:00:00", "2021-03-03T11:00:00", None, "2019-01-01T00:00:00"],
        hs.LAT_COL: [42.9, 45.0, 39.8, 40.0, 37.75],
        hs.LON_COL: [13.2, 8.0, 22.2, 10.0, 15.0],
    })
    sc_ = sc.status_counts(meta)
    assert sc_["p_status_column"] == "trace_p_status" and sc_["p_manual"] == 2 and sc_["p_automatic"] == 1
    assert sc_["p_estimated"] == 1 and sc_["p_status_nan"] == 1 and sc_["s_manual"] == 1 and sc_["s_status_nan"] == 4
    h = sc.heldout_counts(meta)
    assert h["n_rows"] == 5 and h["n_in_window"] == 3 and h["n_year_holdout"] == 2 and h["n_unverifiable"] == 1
