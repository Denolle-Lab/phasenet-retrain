"""Checkpoint 42A: the pool builder without network.

Synthetic streams, a fake FDSN client and a fake catalogue; the exclusion
bundle is the fixture repo of tests/test_exclusion_bundle.py. Base interpreter
(numpy, pandas, obspy, pytest).
"""
import itertools
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
obspy = pytest.importorskip("obspy")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_noise_pool as bnp  # noqa: E402
import exclusion_bundle as eb  # noqa: E402
import noise_ontology as no  # noqa: E402

from obspy import Stream, Trace, UTCDateTime  # noqa: E402
from obspy.core.event import Catalog, Event, Magnitude, Origin, ResourceIdentifier  # noqa: E402
from obspy.core.inventory import Inventory, Network, Station  # noqa: E402

DAY = "2019-03-12"
D0 = UTCDateTime(DAY)
RATE = 40.0
STA_LAT, STA_LON = 34.9459, -106.4572     # IU.ANMO

# ── the event-free rule on precomputed arrivals ───────────────────────────────

def test_event_free_rule_rejects_inside_and_lead_keeps_clear():
    starts = np.array([0.0, 120.0, 240.0, 360.0, 480.0])
    arrivals = [dict(p_time=130.0, s_time=float("nan"))]        # inside window 1 (120-240)
    keep, nearest = bnp.event_free_mask(starts, arrivals)
    # window 1: arrival inside; window 2 (240-360): arrival 110 s before its start -> in the lead
    assert keep.tolist() == [True, False, False, True, True]
    assert nearest[1] == pytest.approx(10.0) and nearest[2] == pytest.approx(-110.0)
    # an S alone rejects as well
    keep, _ = bnp.event_free_mask(starts, [dict(p_time=float("nan"), s_time=365.0)])
    assert keep.tolist() == [True, True, True, False, False]
    # no arrivals at all: everything kept, nearest NaN
    keep, nearest = bnp.event_free_mask(starts, [])
    assert keep.all() and np.isnan(nearest).all()


def test_event_free_rule_boundaries_are_inclusive():
    starts = np.array([1000.0])
    for t, expect in [(1000.0 - 120.0, False), (1000.0 - 120.0 - 0.01, True), (1000.0 + 120.0, False),
                      (1000.0 + 120.01, True)]:
        keep, _ = bnp.event_free_mask(starts, [dict(p_time=t, s_time=float("nan"))])
        assert keep[0] == expect, (t, expect)


def test_predicted_arrivals_taup_p_before_s():
    ev = dict(event_id="e1", time=1_000_000.0, latitude=STA_LAT + 2.0, longitude=STA_LON, depth_km=10.0, magnitude=3.0)
    a = bnp.predicted_arrivals(ev, STA_LAT, STA_LON)
    assert a["distance_deg"] == pytest.approx(2.0, abs=0.01)
    assert 20 < a["p_time"] - ev["time"] < 40          # iasp91 Pn/P at 2 deg, ~30 s
    assert a["s_time"] > a["p_time"]
    far = dict(ev, latitude=STA_LAT - 50.0, longitude=STA_LON + 60.0, depth_km=float("nan"))
    b = bnp.predicted_arrivals(far, STA_LAT, STA_LON)
    assert b["p_time"] - far["time"] > 400 and b["s_time"] > b["p_time"]


def test_coda_flags_mainshock_only():
    starts = np.arange(0, 4 * 3600.0, 120.0)
    p0 = 600.0
    small = [dict(p_time=p0, magnitude=3.0, distance_deg=1.0)]
    assert not bnp.coda_flags(starts, small).any()
    regional = [dict(p_time=p0, magnitude=5.5, distance_deg=5.0)]
    f = bnp.coda_flags(starts, regional)
    assert f.sum() == 15 and f[(starts > p0) & (starts - p0 <= 1800)].all()      # 30 min after P
    tele = [dict(p_time=p0, magnitude=7.2, distance_deg=80.0)]
    f = bnp.coda_flags(starts, tele)
    assert f.sum() == 90                                                            # 3 h after P
    far_m5 = [dict(p_time=p0, magnitude=5.5, distance_deg=80.0)]
    assert not bnp.coda_flags(starts, far_m5).any()


# ── windows ───────────────────────────────────────────────────────────────────

def _stream(t0, seconds, rate=RATE, seed=0, order=("BH1", "BHZ", "BH2"), loc="00"):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    st = Stream()
    for ch in order:
        tr = Trace(rng.standard_normal(n).astype(np.float32),
                   header=dict(network="XX", station="TST", location=loc, channel=ch, sampling_rate=rate, starttime=t0))
        st.append(tr)
    return st


def test_windows_from_stream_native_rate_z_first_empty_dropped():
    st = _stream(D0, 3600)
    data, starts, chans, rate, n_empty = bnp.windows_from_stream(st, start=D0, end=D0 + 7200)
    assert rate == RATE and data.shape == (30, 3, 4800) and data.dtype == np.float32
    assert chans[0] == "BHZ" and n_empty == 30                         # second hour has no data
    assert starts[0] == D0.timestamp and np.diff(starts).tolist() == [120.0] * 29
    # a stream starting off-grid is aligned to the day grid, the leading partial window padded
    st = _stream(D0 + 30, 600)
    data, starts, _, _, _ = bnp.windows_from_stream(st, start=D0, end=D0 + 720)
    assert starts[0] == D0.timestamp and data.shape[0] == 6 and (data[0, :, :1200] == 0).all()
    with pytest.raises(ValueError):
        bnp.windows_from_stream(_stream(D0, 600, order=("BH1", "BH2", "BHN")), start=D0, end=D0 + 600)


# ── labelling and the bundle ──────────────────────────────────────────────────

def _spec(**kw):
    s = bnp.parse_station("XX.TST.00.BH")
    s.update(kw)
    return s


def _rows(*, lat=STA_LAT, lon=STA_LON, coda=None, source_class=None, local=False, comp=float("nan"),
          day=D0, seconds=1200, keep=None):
    st = _stream(day, seconds)
    data, starts, chans, rate, _ = bnp.windows_from_stream(st, start=day, end=day + seconds)
    n = data.shape[0]
    keep = np.ones(n, dtype=bool) if keep is None else keep
    coda = np.zeros(n, dtype=bool) if coda is None else coda
    return bnp.build_rows(pool="fx", st_spec=_spec(), sta_lat=lat, sta_lon=lon, data=data, starts=starts, rate=rate,
                          keep=keep, nearest=np.full(n, np.nan), coda=coda, source_class=source_class,
                          catalogue_used="fake", catalogue_local=local, completeness_mag=comp,
                          npz_rel="windows/x.npz", source="fixture")


def test_no_bundle_path_marks_every_row_unknown_and_flagged():
    df = _rows(local=True, comp=1.5)
    man, rep = bnp.finalize(df, bundle=None, holdout_fraction=0.2, seed_tag="42A")
    assert len(man) == 10 and rep["bundle"] == "absent"
    assert set(man["negative_support"]) == {"unknown"}
    assert man["independence_unverified"].all()
    assert set(man["exclusion_bundle_sha256"]) == {""}
    assert set(man["split"]) == {no.station_split("XX.TST", 0.2, "42A")}
    no.check_manifest(man)


@pytest.fixture
def bundle(tmp_path):
    """The fixture repo of tests/test_exclusion_bundle.py, built and loaded."""
    root = tmp_path / "repo"
    for d in ("data/exclusions", "data/labelerrors", "configs", "notebooks", "cache"):
        (root / d).mkdir(parents=True)
    shutil.copy(REPO_ROOT / "configs" / "evaluation_suites.json", root / "configs" / "evaluation_suites.json")
    pd.DataFrame({"dataset": ["stead"], "trace_name": ["bad"], "chunk": [""], "window": ["x"], "source": ["f"]}
                 ).to_csv(root / "data/exclusions/heldout_sequences.csv", index=False)
    kw = dict(repo_root=root, label_error_dirs=[root / "data" / "labelerrors"], cache_root=root / "cache")
    b = eb.build_bundle(allow_missing_sequence_list=False, **kw)
    path = eb.write_bundle(b, root / "data/exclusions/bundle.json")
    return path, eb.load_bundle(path, **kw), root


def test_bundle_path_quarantines_unknown_coordinates(bundle):
    _, b, _ = bundle
    known = _rows(local=True, comp=1.5)
    unknown = _rows(lat=float("nan"), lon=float("nan"), local=True, comp=1.5)
    df = pd.concat([known, unknown], ignore_index=True)
    man, rep = bnp.finalize(df, bundle=b, holdout_fraction=0.2, seed_tag="42A")
    assert rep["n_input"] == 20 and rep["n_quarantined_unknown"] == 10 and rep["n_kept"] == 10
    assert man["station_latitude_deg"].notna().all()
    assert not man["independence_unverified"].any()
    assert set(man["exclusion_bundle_sha256"]) == {b["sha256"]}
    # event-free + local catalogue + stated completeness -> certified, on the background rows
    assert set(man["negative_support"]) == {"certified"}


def test_bundle_path_drops_holdout_year_and_place(bundle):
    _, b, _ = bundle
    y2016 = _rows(day=UTCDateTime("2016-03-12"))
    etna = _rows(lat=37.75, lon=15.0)                                # all-time place hold-out
    ok = _rows()
    df = pd.concat([y2016, etna, ok], ignore_index=True)
    man, rep = bnp.finalize(df, bundle=b, holdout_fraction=0.2, seed_tag="42A")
    assert rep["n_year_holdout"] == 10 and rep["n_in_window"] == 10 and rep["n_kept"] == 10
    assert rep["windows"] == {"etna": 10}
    # no local catalogue: unknown even though every row is event-free
    assert set(man["negative_support"]) == {"unknown"}


def test_coda_and_source_flag_precedence(bundle):
    _, b, _ = bundle
    n = 10
    coda = np.zeros(n, dtype=bool); coda[:3] = True
    df = _rows(coda=coda, source_class="volcanic_tremor_hydrothermal", local=True, comp=1.0)
    man, _ = bnp.finalize(df, bundle=b, holdout_fraction=0.2, seed_tag="42A")
    assert man["noise_class"].tolist()[:3] == ["earthquake_coda_sequence_hum"] * 3
    assert man["class_source"].tolist()[:3] == ["catalogue_coda"] * 3
    assert set(man["negative_support"].iloc[:3]) == {"unknown"}          # unlabelled_interval, never certified
    assert set(man["noise_class"].iloc[3:]) == {"volcanic_tremor_hydrothermal"}
    assert set(man["ontology_category"].iloc[3:]) == {"task_excluded"}
    assert set(man["negative_support"].iloc[3:]) == {"certified"}        # event-free of catalogued tectonic events
    with pytest.raises(ValueError):
        bnp.source_class_for(_spec(), STA_LAT, {"XX.TST": "not_a_class"})
    assert bnp.source_class_for(_spec(), -75.0, {}) == "polar_ice"
    assert bnp.source_class_for(_spec(), STA_LAT, {}) is None


def _support_column_reference(df, *, bundle_present):
    """The per-row form support_column() had before it was vectorised; kept as the oracle."""
    vals = []
    for _, r in df.iterrows():
        if not bundle_present:
            vals.append("unknown")
            continue
        vals.append(no.negative_support_for(
            r["ontology_category"], event_free=bool(r["event_free"]),
            local_catalogue=bool(r.get("catalogue_local", False)),
            completeness_mag=r.get("completeness_mag", float("nan"))))
    return pd.Series(vals, index=df.index, dtype=object)


def test_support_column_matches_per_row_reference():
    # every category x event_free x local catalogue x completeness (NaN, 0.0, stated), non-default index
    grid = list(itertools.product(no.CATEGORIES, (True, False), (True, False), (float("nan"), 0.0, 1.2)))
    df = pd.DataFrame(grid, columns=["ontology_category", "event_free", "catalogue_local", "completeness_mag"],
                      index=[f"r{i:02d}" for i in range(len(grid))][::-1])
    for present in (True, False):
        got = bnp.support_column(df, bundle_present=present)
        pd.testing.assert_series_equal(got, _support_column_reference(df, bundle_present=present))
    got = bnp.support_column(df, bundle_present=True)
    cert = got == "certified"
    assert cert.sum() == 2 * 2                                  # 2 certifiable categories x (0.0, 1.2)
    assert set(df.loc[cert, "ontology_category"]) == set(no.CERTIFIABLE_CATEGORIES)
    assert df.loc[cert, "event_free"].all() and df.loc[cert, "catalogue_local"].all()
    assert set(got[df["ontology_category"] == "reviewed_negative"]) == {"reviewed"}
    assert set(got[df["ontology_category"] == "unlabelled_interval"]) == {"unknown"}
    # the optional columns absent: local catalogue False, completeness NaN -> never certified
    bare = df[["ontology_category", "event_free"]]
    pd.testing.assert_series_equal(bnp.support_column(bare, bundle_present=True),
                                   _support_column_reference(bare, bundle_present=True))
    assert set(bnp.support_column(bare, bundle_present=True)) == {"unknown", "reviewed"}
    # None for completeness (object column) is 'not stated', as in the scalar rule
    none = df.assign(completeness_mag=None)
    pd.testing.assert_series_equal(bnp.support_column(none, bundle_present=True),
                                   _support_column_reference(none, bundle_present=True))
    # an empty frame and an unknown category behave as before
    assert len(bnp.support_column(df.iloc[:0], bundle_present=True)) == 0
    with pytest.raises(ValueError):
        bnp.support_column(df.assign(ontology_category="not_a_category"), bundle_present=True)


def test_rejected_windows_are_not_written():
    n = 10
    keep = np.ones(n, dtype=bool); keep[[2, 5]] = False
    df = _rows(keep=keep)
    assert len(df) == 8 and df["event_free"].all()


def test_parsers():
    s = bnp.parse_station("iu.kip.00.bh")
    assert (s["network"], s["station"], s["location"], s["channel_band"], s["key"]) == ("IU", "KIP", "00", "BH", "IU.KIP")
    assert bnp.parse_station("IU.KIP")["location"] == "*"
    assert bnp.parse_station("IU.KIP.--")["location"] == ""
    lc = bnp.parse_local_catalogue("IU.KIP=USGS:3.0:0.0:2.0")
    assert lc == dict(key="IU.KIP", client="USGS", radius_deg=3.0, min_mag=0.0, completeness_mag=2.0)
    for bad in ("IU", "IU.KIP.00.BHZ"):
        with pytest.raises(ValueError):
            bnp.parse_station(bad)
    with pytest.raises(ValueError):
        bnp.parse_local_catalogue("IU.KIP=USGS:3.0")
    # --source-class keys are upper-cased like parse_station(), so the flag matches its station
    assert bnp.parse_source_class("iu.kip=polar_ice") == ("IU.KIP", "polar_ice")
    assert bnp.parse_source_class(" IU.kip = hydrological ") == ("IU.KIP", "hydrological")
    for bad in ("IU.KIP", "IU.KIP=", "=polar_ice", "IU.KIP=not_a_class"):
        with pytest.raises(ValueError):
            bnp.parse_source_class(bad)


def test_provenance_path_is_never_absolute(tmp_path):
    inside = bnp.REPO_ROOT / "data" / "exclusions" / "bundle.json"
    assert bnp._provenance_path(inside) == os.path.join("data", "exclusions", "bundle.json")
    assert bnp._provenance_path(str(inside)) == os.path.join("data", "exclusions", "bundle.json")
    outside = tmp_path / "scratch" / "bundle_laptop.json"
    assert bnp._provenance_path(outside) == "bundle_laptop.json"
    # an extra root (the --repo-root of the run) is tried first
    assert bnp._provenance_path(outside, roots=(tmp_path, bnp.REPO_ROOT)) == os.path.join("scratch", "bundle_laptop.json")
    for p in (inside, outside):
        assert not os.path.isabs(bnp._provenance_path(p))


# ── end to end with a fake FDSN client ────────────────────────────────────────

class FakeClient:
    """Stations, waveforms and events without the network. `fail` names the
    calls that raise, to exercise the failure records."""
    calls: list = []

    def __init__(self, fail=()):
        self.fail = fail            # shared with the fixture so a test can add failures mid-run

    def get_stations(self, network, station, level, starttime, endtime):
        FakeClient.calls.append(("get_stations", network, station))
        if "get_stations" in self.fail:
            raise RuntimeError("stations down")
        if station == "NOCO":
            return Inventory(networks=[], source="fixture")
        return Inventory(networks=[Network(code=network, stations=[
            Station(code=station, latitude=STA_LAT, longitude=STA_LON, elevation=1000.0)])], source="fixture")

    def get_waveforms(self, network, station, location, channel, starttime, endtime):
        FakeClient.calls.append(("get_waveforms", network, station))
        if "get_waveforms" in self.fail or station == "DOWN":
            raise RuntimeError("waveforms down")
        return _stream(starttime, float(endtime - starttime), seed=hash(station) % 1000)

    def get_events(self, **kw):
        FakeClient.calls.append(("get_events", kw.get("minmagnitude"), kw.get("maxradius")))
        if "get_events" in self.fail:
            raise RuntimeError("events down")
        cat = Catalog()
        if kw.get("maxradius") is None:
            # global: an M3 at 2 deg one hour into the day -> P ~ +30 s, S ~ +55 s
            cat.append(_event("gl1", D0 + 3600, STA_LAT + 2.0, STA_LON, 10.0, 3.0))
            # an M7.5 at ~80 deg, 1 h before the day: its P and S arrive before the day, coda covers 3 h after P
            cat.append(_event("gl2", D0 - 3600, STA_LAT - 60.0, STA_LON + 70.0, 30.0, 7.5))
        else:
            # local: an M0.8 at 0.3 deg, 2 h into the day
            cat.append(_event("lo1", D0 + 7200, STA_LAT + 0.3, STA_LON, 5.0, 0.8))
        return cat


def _event(eid, t, lat, lon, depth_km, mag):
    ev = Event(resource_id=ResourceIdentifier(f"quakeml:fixture/query?eventid={eid}&format=quakeml"))
    o = Origin(time=t, latitude=lat, longitude=lon, depth=depth_km * 1000.0)
    ev.origins = [o]; ev.preferred_origin_id = o.resource_id
    m = Magnitude(mag=mag, magnitude_type="ml"); ev.magnitudes = [m]; ev.preferred_magnitude_id = m.resource_id
    return ev


@pytest.fixture
def fake_net(monkeypatch, tmp_path):
    FakeClient.calls = []
    fails = set()
    monkeypatch.setattr(bnp, "client_for", lambda name, timeout=0: FakeClient(fails))
    monkeypatch.setattr(bnp, "FDSN_BACKOFF_S", 0.0)
    monkeypatch.setattr(bnp, "_clients", {})
    return fails, tmp_path / "pools"


def _run(out, *extra):
    return bnp.main(["--pool", "fx", "--station", "XX.TST.00.BH", "--day", DAY, "--day-duration-s", "10800",
                     "--out-root", str(out), "--max-minutes", "5", *extra])


def test_main_no_bundle_end_to_end(fake_net):
    fails, out = fake_net
    rc = _run(out, "--no-bundle", "--local-catalogue", "XX.TST=LOCAL:1.0:0.0:1.2")
    assert rc == 0
    man = pd.read_parquet(out / "fx" / "manifest.parquet")
    meta = json.loads((out / "fx" / "manifest.json").read_text())
    no.check_manifest(man)
    # 90 windows in 3 h; the M3 at +3600 s rejects the windows whose [start-120, start+120] holds +3630/+3655:
    # starts 3600 and 3720; the local M0.8 at +7200 s (P ~ +7207, S ~ +7212) rejects starts 7200 and 7320
    sd = meta["station_days"][0]
    assert sd["n_windows"] == 90 and sd["n_rejected_arrival"] == 4 and sd["n_events"] == 3
    assert len(man) == 86 and man["event_free"].all()
    starts = pd.to_datetime(man["start_time"])
    rejected = {3600, 3720, 7200, 7320}
    assert not any(((starts - pd.Timestamp(DAY)).dt.total_seconds()).isin(rejected))
    # the M7.5's P arrives ~ +11 min after its origin, i.e. ~ 49 min before the day: coda spans 3 h after that,
    # so windows in the first ~2 h 11 min are coda -> unlabelled_interval -> unknown
    coda = man[man["noise_class"] == "earthquake_coda_sequence_hum"]
    assert sd["n_coda"] == len(coda) and 60 <= len(coda) <= 70
    assert set(coda["ontology_category"]) == {"unlabelled_interval"} and set(coda["class_source"]) == {"catalogue_coda"}
    # --no-bundle: every row unknown and flagged, bundle absence recorded
    assert set(man["negative_support"]) == {"unknown"} and man["independence_unverified"].all()
    assert meta["bundle"] == {"present": False, "reason": "--no-bundle",
                              "path": os.path.join("data", "exclusions", "bundle.json")}
    assert not os.path.isabs(meta["bundle"]["path"])
    assert meta["parameters"]["model_screening"] is False
    assert set(man["rate_hz"]) == {RATE} and set(man["duration_s"]) == {120.0}
    # the windows on disk address the manifest
    z = np.load(out / "fx" / sd["npz"])
    assert z["data"].shape == (86, 3, 4800) and list(z["channels"]) == ["BHZ", "BH1", "BH2"]
    assert man["npz_index"].tolist() == list(range(86))
    assert np.allclose(z["start_epoch"], starts.map(lambda t: t.timestamp()).to_numpy())
    # queries recorded, raw cache written, no failures
    assert [q["client"] for q in meta["catalogue_queries"]] == ["USGS", "LOCAL"]
    assert meta["failures"] == [] and (out / "raw" / "events").is_dir()
    assert all(c in man["catalogue_used"].iloc[0] for c in ("USGS", "LOCAL"))
    assert meta["census"]["by_support"] == {"unknown": 86}


def test_main_with_bundle_certifies_background_rows(fake_net, bundle):
    fails, out = fake_net
    path, b, root = bundle
    rc = _run(out, "--bundle", str(path), "--repo-root", str(root), "--local-catalogue", "XX.TST=LOCAL:1.0:0.0:1.2")
    assert rc == 0
    man = pd.read_parquet(out / "fx" / "manifest.parquet")
    meta = json.loads((out / "fx" / "manifest.json").read_text())
    # the fixture bundle is uncertified (empty cache: all 20 sources unhashed); noise rows do not need
    # certification, which is about the SeisBench sequence list, but the fact is recorded
    assert meta["bundle"]["sha256"] == b["sha256"] and meta["bundle"]["certified"] is False
    # the bundle path is recorded relative to --repo-root, never as the workstation's absolute path
    assert meta["bundle"]["path"] == os.path.join("data", "exclusions", "bundle.json")
    assert str(path) not in json.dumps(meta)
    assert set(man["exclusion_bundle_sha256"]) == {b["sha256"]}
    assert not man["independence_unverified"].any()
    coda = man["ontology_category"] == "unlabelled_interval"
    assert set(man.loc[coda, "negative_support"]) == {"unknown"}
    assert set(man.loc[~coda, "negative_support"]) == {"certified"}
    assert set(man.loc[~coda, "completeness_mag"]) == {1.2}
    assert meta["exclusion_report"]["n_removed"] == 0


def test_main_lower_case_source_class_is_honoured(fake_net):
    fails, out = fake_net
    rc = _run(out, "--no-bundle", "--source-class", "xx.tst=volcanic_tremor_hydrothermal")
    assert rc == 0
    man = pd.read_parquet(out / "fx" / "manifest.parquet")
    meta = json.loads((out / "fx" / "manifest.json").read_text())
    assert meta["stations"][0]["source_flag"] == "volcanic_tremor_hydrothermal"
    flagged = man[man["class_source"] != "catalogue_coda"]
    assert len(flagged) > 0
    assert set(flagged["noise_class"]) == {"volcanic_tremor_hydrothermal"}
    assert set(flagged["class_source"]) == {"source_flag"}
    assert "features" not in set(man["class_source"])


def test_main_records_failures_without_fabricating(fake_net):
    fails, out = fake_net
    # waveform service down: station-day skipped, failure recorded, empty manifest, exit 1
    rc = bnp.main(["--pool", "fx", "--station", "XX.DOWN.00.BH", "--day", DAY, "--day-duration-s", "1200",
                   "--out-root", str(out), "--max-minutes", "5", "--no-bundle"])
    assert rc == 1
    meta = json.loads((out / "fx" / "manifest.json").read_text())
    assert meta["station_days"] == [dict(day=DAY, station="XX.DOWN", skipped="waveform fetch failed")]
    assert meta["failures"][0]["error"].startswith("RuntimeError: waveforms down") and meta["failures"][0]["attempt"] == 3
    assert meta["fetches"][0]["failed"] is True
    man = pd.read_parquet(out / "fx" / "manifest.parquet")
    assert len(man) == 0 and list(man.columns) == list(no.MANIFEST_COLUMNS)
    # global catalogue down: the event-free rule cannot run, the day is skipped, nothing is written as noise
    # (a fresh out-root: the first run cached its catalogue under raw/events and a cache hit is not a query)
    fails.add("get_events")
    rc = bnp.main(["--pool", "fx2", "--station", "XX.TST.00.BH", "--day", DAY, "--day-duration-s", "1200",
                   "--out-root", str(out.parent / "pools2"), "--max-minutes", "5", "--no-bundle"])
    assert rc == 1
    meta = json.loads((out.parent / "pools2" / "fx2" / "manifest.json").read_text())
    assert meta["station_days"] == [dict(day=DAY, skipped="global catalogue failed")]
    assert meta["catalogue_queries"][0]["failed"] is True


def test_main_budget_exhausted_is_recorded(fake_net):
    fails, out = fake_net
    rc = bnp.main(["--pool", "fx", "--station", "XX.TST.00.BH", "--day", DAY, "--day-duration-s", "1200",
                   "--out-root", str(out), "--max-minutes", "0", "--no-bundle"])
    assert rc == 1
    meta = json.loads((out / "fx" / "manifest.json").read_text())
    assert meta["budget"]["exhausted_before"].startswith("get_stations")
    assert any(f["error"] == "budget_exhausted" for f in meta["failures"])


def test_main_unknown_coordinates_are_quarantined_by_bundle(fake_net, bundle):
    fails, out = fake_net
    path, b, root = bundle
    rc = bnp.main(["--pool", "fx", "--station", "XX.NOCO.00.BH", "--day", DAY, "--day-duration-s", "1200",
                   "--out-root", str(out), "--max-minutes", "5", "--bundle", str(path), "--repo-root", str(root)])
    assert rc == 1
    meta = json.loads((out / "fx" / "manifest.json").read_text())
    assert meta["exclusion_report"]["n_quarantined_unknown"] == 10 and meta["exclusion_report"]["n_kept"] == 0
    assert any("coordinates" in f["what"] for f in meta["failures"])
