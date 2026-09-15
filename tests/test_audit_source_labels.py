"""Behavioural tests for scripts/audit_source_labels.py (checkpoint 41B of issue #41).

Synthetic three-component records only: no network, no SeisBench cache, no
model. The SeisBench-reader test writes a fake HDF5+CSV source and runs only
where h5py, seisbench and torch (manifest_dataset) are importable.
Run with:  python -m pytest tests/test_audit_source_labels.py -q
"""
import ast
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("obspy")

from obspy import Stream, Trace, UTCDateTime  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_source_labels as asl  # noqa: E402
import evaluation_policy as policy  # noqa: E402

RATES = (20.0, 40.0, 50.0, 100.0, 200.0)


# ── synthetic records ────────────────────────────────────────────────────────

def wavelet(t, t_on, f=3.0, amp=1.0, sigma=0.5):
    """Gaussian-enveloped cosine starting exactly at t_on (impulsive onset, peak at t_on + sigma)."""
    env = np.where(t >= t_on, np.exp(-((t - t_on - sigma) / sigma) ** 2), 0.0)
    return amp * env * np.cos(2 * np.pi * f * (t - t_on))


def record(rate, tp=30.0, ts=38.0, length_s=120.0, seed=0, noise=0.05, events=()):
    """(3, n) float32 ZNE: P on Z (amp 1, 0.3 on horizontals), a larger S on the
    horizontals (amp 2, 0.3 on Z); `events` adds further (tp, ts) pairs."""
    rng = np.random.default_rng(seed)
    n = int(length_s * rate)
    t = np.arange(n) / rate
    w = rng.normal(scale=noise, size=(3, n))
    for p, s in ((tp, ts),) + tuple(events):
        if p is not None:
            w[0] += wavelet(t, p, amp=1.0)
            w[1] += wavelet(t, p, amp=0.3)
            w[2] += wavelet(t, p, amp=0.3)
        if s is not None:
            w[0] += wavelet(t, s, amp=0.3)
            w[1] += wavelet(t, s, amp=2.0)
            w[2] += wavelet(t, s, amp=2.0)
    return w.astype(np.float32)


def row(rate=100.0, tp=30.0, ts=38.0, label_p=None, label_s=None, dist=40.0, seed=0, source="syn", trace="t", **kw):
    w = record(rate, tp, ts, seed=seed, **kw)
    return asl.LabelRow(source=source, trace_id=trace, station="XX.A", event_id="e", waveform=w, rate_hz=rate,
                        p_s=tp if label_p is None else label_p, s_s=ts if label_s is None else label_s, distance_km=dist)


# ── module hygiene ───────────────────────────────────────────────────────────

def test_module_imports_no_torch_seisbench_h5py_at_top_level():
    tree = ast.parse((REPO_ROOT / "scripts" / "audit_source_labels.py").read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"torch", "seisbench", "h5py", "manifest_dataset", "build_training_dataset"}


# ── C2: onset recovery and tolerance ─────────────────────────────────────────

@pytest.mark.parametrize("rate", RATES)
def test_c2_recovers_p_and_s_onsets_within_0p1_s(rate):
    r = row(rate=rate)
    p = asl.check_p_onset(r.waveform, rate, r.p_s, ts=r.s_s)
    s = asl.check_s_onset(r.waveform, rate, r.s_s, tp=r.p_s)
    assert p["testable"] and abs(p["residual_s"]) <= 0.1
    assert s["testable"] and abs(s["residual_s"]) <= 0.1
    assert not p["flag"] and not s["flag"]


@pytest.mark.parametrize("rate", (20.0, 100.0))
def test_c2_flags_a_late_label_reports_an_early_one_and_ignores_0p3_s(rate):
    r = row(rate=rate)
    late = asl.check_p_onset(r.waveform, rate, r.p_s + 1.0, ts=r.s_s)          # energy 1 s before the label
    early = asl.check_p_onset(r.waveform, rate, r.p_s - 1.0, ts=r.s_s)         # analyst 1 s before the energy
    near = asl.check_p_onset(r.waveform, rate, r.p_s + 0.3, ts=r.s_s)
    assert late["testable"] and late["late"] and late["flag"] and not late["emergent"] and late["residual_s"] < -0.5
    assert early["testable"] and early["emergent"] and not early["flag"] and not early["late"] and early["residual_s"] > 0.5
    assert near["testable"] and not near["flag"] and not near["late"] and not near["emergent"]
    assert abs(near["residual_s"] + 0.3) <= 0.1
    sym = asl.check_p_onset(r.waveform, rate, r.p_s - 1.0, ts=r.s_s, rule="symmetric")
    assert sym["flag"] and sym["emergent"]
    late_s = asl.check_s_onset(r.waveform, rate, r.s_s + 1.0, tp=r.p_s)
    early_s = asl.check_s_onset(r.waveform, rate, r.s_s - 1.0, tp=r.p_s)
    near_s = asl.check_s_onset(r.waveform, rate, r.s_s - 0.3, tp=r.p_s)
    assert late_s["flag"] and late_s["late"] and late_s["residual_s"] < -0.5
    assert early_s["emergent"] and not early_s["flag"] and early_s["residual_s"] > 0.5
    assert not near_s["flag"] and not near_s["emergent"]
    with pytest.raises(ValueError):
        asl.check_p_onset(r.waveform, rate, r.p_s, rule="lenient")


def test_c2_noise_window_is_not_testable_and_never_flagged():
    rng = np.random.default_rng(5)
    w = rng.normal(scale=0.05, size=(3, 12000)).astype(np.float32)
    p = asl.check_p_onset(w, 100.0, 30.0)
    assert not p["testable"] and not p["flag"]
    assert asl.check_p_onset(w, 100.0, None)["testable"] is False


def test_c2_s_window_starts_at_the_p_s_midpoint():
    """A station 2 s (S-P) away: without the midpoint cut the AIC lands on the P."""
    r = row(rate=100.0, tp=30.0, ts=32.0)
    s = asl.check_s_onset(r.waveform, 100.0, 32.0, tp=30.0)
    assert s["testable"] and abs(s["residual_s"]) <= 0.1


# ── C1 ───────────────────────────────────────────────────────────────────────

def test_c1_swapped_pair_fails_and_outlier_is_the_only_flag():
    rng = np.random.default_rng(1)
    d = rng.uniform(5, 150, size=40)
    tp = rng.uniform(20, 40, size=40)
    ts = tp + 0.5 + 0.125 * d + rng.normal(scale=0.03, size=40)
    ts[7] += 3.0                                              # the outlier
    out = asl.sp_consistency(tp, ts, d)
    assert out["n_fit"] == 40
    assert abs(out["slope_s_per_km"] - 0.125) < 0.01
    assert abs(out["implied_vp_vs"] - 1.75) < 0.06
    assert np.flatnonzero(out["flag"]).tolist() == [7]
    swapped = asl.sp_consistency([38.0, 30.0], [30.0, 38.0], [40.0, 40.0])
    assert swapped["flag"].tolist() == [True, False]
    assert swapped["ts_le_tp"].tolist() == [True, False]
    assert np.isnan(swapped["slope_s_per_km"])              # below C1_MIN_ROWS: no fit
    missing = asl.sp_consistency([30.0], [None], [None])
    assert missing["flag"].tolist() == [False] and np.isnan(missing["residual_s"][0])


# ── C3 / C4 / C6 on audited rows ─────────────────────────────────────────────

def synthetic_source(n=12, rate=100.0):
    rows = []
    rng = np.random.default_rng(2)
    for i in range(n):
        d = float(rng.uniform(10, 100))
        tp = 30.0
        ts = tp + 0.5 + 0.125 * d
        rows.append(row(rate=rate, tp=tp, ts=ts, dist=d, seed=i, trace=f"t{i}"))
    return rows


def test_c3_fraction_and_summary_schema_on_synthetic_set():
    df = asl.finish_source([asl.audit_row(r) for r in synthetic_source()], "syn")
    s = asl.summarise(df, "syn")
    assert s["c3_p_gt_s_frac"] >= 0.8 and s["c3_n"] >= 5
    assert s["c1_n_fit"] == 12 and s["c1_flag_frac"] == 0.0
    for key in ("c1_flag", "c2_flag", "c2s_flag", "c3_p_gt_s", "c4_edge", "c6_unlabelled", "c6_unlabelled_before_p",
                "c6_unlabelled_flagged", "c6_unlabelled_unflagged", "c2_testable", "c2_late", "c2_late_1s",
                "c2_emergent", "c2_suspect", "c2_unlabelled_earlier", "c2s_late", "c2s_emergent",
                "c6_second_event", "c6_wrong_first_pick", "c6_no_detection"):
        frac, lo, hi = s[f"{key}_frac"], s[f"{key}_lo"], s[f"{key}_hi"]
        if np.isfinite(frac) and np.isfinite(lo):
            assert lo <= frac <= hi
    assert s["c6_unlabelled_frac"] == 0.0 and s["n_extra_arrival_rows"] == 0
    assert s["c2_rule"] == "asymmetric" and s["c2_late_frac"] == 0.0 and s["c2_emergent_frac"] == 0.0
    assert set(df["suggested_tier"]) == {"manual"} and set(df["suggested_tier_s"]) == {"manual"}
    assert set(df["c2_late_kind"]) == {""}
    assert json.loads(df["rates_hz"].iloc[0]) == [100.0] if "rates_hz" in df else json.loads(s["rates_hz"]) == [100.0]
    sheet = asl.review_sheet(df)
    assert sheet.empty


def test_c4_flags_an_edge_pick_and_reports_the_mode_share():
    rows = synthetic_source(6)
    edge = row(rate=100.0, tp=0.5, ts=8.0, trace="edge")
    df = asl.finish_source([asl.audit_row(r) for r in rows + [edge]], "syn")
    assert df.loc[df["trace_id"] == "edge", "c4_edge"].item()
    assert not df.loc[df["trace_id"] != "edge", "c4_edge"].any()
    s = asl.summarise(df, "syn")
    assert s["c4_n_edge"] == 1 and abs(s["c4_edge_frac"] - 1 / 7) < 1e-9
    assert s["c4_mode_p_sample"] == 3000.0 and abs(s["c4_mode_share"] - 6 / 7) < 1e-9
    assert df.loc[df["trace_id"] == "edge", "suggested_tier"].item() == "unknown"


def test_c6_two_events_one_label_gives_extra_triggers_and_single_event_none():
    two = row(rate=100.0, tp=30.0, ts=38.0, events=((70.0, 76.0),), trace="two")
    one = row(rate=100.0, tp=30.0, ts=38.0, trace="one")
    a, b = asl.audit_row(two), asl.audit_row(one)
    assert a["c6_n_extra_triggers"] >= 1 and a["c6_has_unlabelled_arrival"]
    assert a["c6_class"] == "second_event"
    extra = json.loads(a["suggested_extra_arrival_s"])
    assert any(abs(t - 70.0) < 0.5 for t in extra)
    assert b["c6_n_extra_triggers"] == 0 and not b["c6_has_unlabelled_arrival"]
    assert b["c6_class"] == "single_detection" and json.loads(b["suggested_extra_arrival_s"]) == []
    assert a["c6_n_extra_before_p"] == 0
    # a second event before the labelled one counts as "before P"
    before = asl.audit_row(row(rate=100.0, tp=60.0, ts=68.0, events=((30.0, 38.0),), trace="before"))
    assert before["c6_n_extra_before_p"] >= 1


def test_late_label_kind_from_the_c6_screen():
    """A label 1 s after the true onset: with an unexplained earlier event it is an
    unlabelled earlier event (keep manual, add the arrival); alone it is a suspect pick."""
    multi = row(rate=100.0, tp=60.0, ts=68.0, label_p=61.0, events=((30.0, 38.0),), dist=60.0, trace="multi")
    alone = row(rate=100.0, tp=60.0, ts=68.0, label_p=61.0, dist=60.0, trace="alone")
    df = asl.finish_source([asl.audit_row(multi), asl.audit_row(alone)] + [asl.audit_row(r) for r in synthetic_source(6)],
                           "syn").set_index("trace_id")
    assert df.loc["multi", "c2_late"] and df.loc["multi", "c2_flag"] and df.loc["multi", "c6_n_extra_before_p"] >= 1
    assert df.loc["multi", "c2_late_kind"] == "unlabelled_earlier_event" and df.loc["multi", "suggested_tier"] == "manual"
    assert any(abs(t - 30.0) < 0.5 for t in json.loads(df.loc["multi", "suggested_extra_arrival_s"]))
    assert df.loc["alone", "c2_late"] and df.loc["alone", "c6_n_extra_before_p"] == 0
    assert df.loc["alone", "c2_late_kind"] == "suspect_pick" and df.loc["alone", "suggested_tier"] == "unknown"
    s = asl.summarise(df.reset_index(), "syn")
    assert s["c2_suspect_n"] == 1 and s["c2_unlabelled_earlier_n"] == 1 and s["c2_n_late"] == 2
    assert abs(s["c2_suspect_frac"] - 1 / 8) < 1e-9 and s["c2_suspect_lo"] <= s["c2_suspect_frac"] <= s["c2_suspect_hi"]
    assert asl.late_kind(False, 3) == "" and asl.late_kind(True, 0) == "suspect_pick"


def test_c6_classification_wrong_first_pick_and_no_detection():
    # the label sits in noise 5 s before the real onset
    wrong = asl.audit_row(row(rate=100.0, tp=40.0, ts=48.0, label_p=35.0, label_s=None, trace="wrong"))
    assert wrong["c6_class"] == "wrong_first_pick"
    rng = np.random.default_rng(9)
    noise = asl.LabelRow("syn", "noise", "XX.A", "e", rng.normal(scale=0.05, size=(3, 12000)).astype(np.float32), 100.0,
                         p_s=30.0, s_s=None)
    quiet = asl.audit_row(noise)
    assert quiet["c6_class"] == "no_detection" and quiet["c6_n_triggers"] == 0
    assert asl.classify_triggers([], 30.0) == "no_detection"
    assert asl.classify_triggers([10.0], 30.0) == "wrong_first_pick"
    assert asl.classify_triggers([30.2, 70.0], 30.0, extra=[70.0]) == "second_event"
    assert asl.classify_triggers([30.2, 38.1], 30.0, extra=[]) == "single_detection"
    assert asl.classify_triggers([30.2], None) == "not_testable"


def test_extra_triggers_exclude_labels_and_s_coda():
    on = np.array([10.0, 29.5, 38.4, 40.5, 41.5, 90.0])
    extra = asl.extra_triggers(on, 30.0, 38.0)
    assert extra.tolist() == [10.0, 41.5, 90.0]
    assert asl.extra_triggers(on, 30.0, None).tolist() == [10.0, 38.4, 40.5, 41.5, 90.0]


def test_c6_summary_split_by_multiplet_flag():
    rows = synthetic_source(8)
    recs = [asl.audit_row(r) for r in rows]
    two = row(rate=100.0, tp=30.0, ts=38.0, dist=60.0, events=((70.0, 76.0),), trace="two")   # on the S-P line
    two.flagged_multiplet = True
    recs.append(asl.audit_row(two))
    df = asl.finish_source(recs, "syn")
    s = asl.summarise(df, "syn")
    assert s["n_flagged_multiplet"] == 1 and s["c6_flagged_n"] == 1
    assert s["c6_unlabelled_flagged_frac"] == 1.0 and s["c6_unlabelled_unflagged_frac"] == 0.0
    assert s["c6_second_event_frac"] == 1.0 and s["c6_wrong_first_pick_frac"] == 0.0 and s["c6_no_detection_frac"] == 0.0
    sheet = asl.review_sheet(df)
    assert len(sheet) == 1 and sheet["reasons"].iloc[0] == "multiplet_report;unlabelled_arrival"
    assert sheet["c6_class"].iloc[0] == "second_event"


# ── derived tier ─────────────────────────────────────────────────────────────

def base_record(**kw):
    rec = dict(source="s", trace_id="t", station="a", event_id="e", rate_hz=100.0, n_samples=12000, duration_s=120.0,
               p_s=30.0, s_s=38.0, distance_km=40.0, p_status="unknown", s_status="unknown", flagged_multiplet=False,
               component_mask="ZNE", p_sample=3000.0, c2_rule="asymmetric", c2_onset_s=30.0, c2_residual_s=0.0,
               c2_rms_ratio=5.0, c2_testable=True, c2_late=False, c2_emergent=False, c2_flag=False, c2_late_kind="",
               c2s_onset_s=38.0, c2s_residual_s=0.0, c2s_rms_ratio=5.0, c2s_testable=True, c2s_late=False,
               c2s_emergent=False, c2s_flag=False, c3_ratio_p=2.0, c3_ratio_s=0.1, c3_testable=True,
               c3_p_gt_s=True, c4_p_fraction=0.25, c4_edge=False, c6_testable=True, c6_triggers_json="[]",
               c6_n_triggers=0, c6_n_extra_triggers=0, c6_has_unlabelled_arrival=False, c6_n_extra_before_p=0,
               c6_class="no_detection", c6_p_in_blind=False, suggested_extra_arrival_s="[]")
    rec.update(kw)
    return rec


def test_suggested_tier_mapping():
    recs = [base_record(trace_id="clean"),
            base_record(trace_id="suspect", c2_late=True, c2_flag=True, c2_late_kind="suspect_pick"),
            base_record(trace_id="multi", c2_late=True, c2_flag=True, c2_late_kind="unlabelled_earlier_event",
                        c6_n_extra_before_p=1, c6_n_extra_triggers=1, c6_has_unlabelled_arrival=True),
            base_record(trace_id="emergent", c2_emergent=True),
            base_record(trace_id="sym", c2_rule="symmetric", c2_emergent=True, c2_flag=True),
            base_record(trace_id="edge", c4_edge=True), base_record(trace_id="c2s", c2s_flag=True, c2s_late=True),
            base_record(trace_id="swap", p_s=38.0, s_s=30.0), base_record(trace_id="nos", s_s=None)]
    df = asl.finish_source(recs, "s").set_index("trace_id")
    assert df.loc["clean", "suggested_tier"] == "manual" and df.loc["clean", "suggested_tier_s"] == "manual"
    assert df.loc["suspect", "suggested_tier"] == "unknown"
    assert df.loc["multi", "suggested_tier"] == "manual"          # keeps the label, gets the extra arrival
    assert df.loc["emergent", "suggested_tier"] == "manual"       # reported, never a flag by default
    assert df.loc["sym", "suggested_tier"] == "unknown"           # the symmetric rule flags emergent onsets
    assert df.loc["edge", "suggested_tier"] == "unknown"
    assert df.loc["c2s", "suggested_tier"] == "manual" and df.loc["c2s", "suggested_tier_s"] == "unknown"
    assert df.loc["swap", "c1_flag"] and df.loc["swap", "suggested_tier"] == "unknown"
    assert df.loc["nos", "suggested_tier_s"] == ""


# ── C5 on metadata ───────────────────────────────────────────────────────────

def test_duplicate_pairs_from_metadata():
    t0 = "2020-01-01T00:00:00Z"
    meta = pd.DataFrame([
        dict(source="a", trace_name="a1", station="STA", source_origin_time=t0, source_latitude_deg=10.0,
             source_longitude_deg=20.0, trace_start_time="2020-01-01T00:00:10Z", p_sample=500, rate_hz=100.0),
        dict(source="b", trace_name="b1", station="STA", source_origin_time="2020-01-01T00:00:01Z", source_latitude_deg=10.01,
             source_longitude_deg=20.0, trace_start_time="2020-01-01T00:00:12Z", p_sample=300, rate_hz=100.0),   # P 15.0 vs 15.0
        dict(source="a", trace_name="a2", station="STB", source_origin_time=t0, source_latitude_deg=10.0,
             source_longitude_deg=20.0, trace_start_time="2020-01-01T00:00:10Z", p_sample=500, rate_hz=100.0),
        dict(source="b", trace_name="b2", station="STB", source_origin_time=t0, source_latitude_deg=10.0,
             source_longitude_deg=20.0, trace_start_time="2020-01-01T00:00:10Z", p_sample=550, rate_hz=100.0),   # 0.5 s apart
        dict(source="a", trace_name="a3", station="STC", source_origin_time=t0, source_latitude_deg=10.0,
             source_longitude_deg=20.0, trace_start_time=None, p_sample=500, rate_hz=100.0),
        dict(source="b", trace_name="b3", station="STC", source_origin_time=t0, source_latitude_deg=10.0,
             source_longitude_deg=20.0, trace_start_time="2020-01-01T00:00:10Z", p_sample=500, rate_hz=100.0),
        dict(source="b", trace_name="b4", station="STA", source_origin_time="2020-01-01T01:00:00Z", source_latitude_deg=10.0,
             source_longitude_deg=20.0, trace_start_time="2020-01-01T01:00:10Z", p_sample=500, rate_hz=100.0),   # another event
    ])
    pairs, summary = asl.duplicate_pairs(meta)
    assert summary["n_pairs"] == 3 and summary["n_comparable"] == 2 and summary["n_not_comparable"] == 1
    assert summary["disagree_frac"] == 0.5
    by_station = pairs.set_index("station")
    assert by_station.loc["STA", "disagree"] == False and abs(by_station.loc["STA", "dp_s"]) < 1e-9  # noqa: E712
    assert by_station.loc["STB", "disagree"] == True and abs(by_station.loc["STB", "dp_s"] - 0.5) < 1e-9  # noqa: E712
    assert by_station.loc["STC", "comparable"] == False  # noqa: E712
    assert summary["pairs_by_sources"] == {"a|b": 3}


# ── multiplet reports (no download) ─────────────────────────────────────────

def test_multiplet_report_is_read_from_cache_only(tmp_path, monkeypatch):
    names, info = asl.load_multiplet_report("stead", report_dirs=[tmp_path])
    assert names == set() and info["status"] == "report_not_cached"
    pd.DataFrame({"trace_name": ["x1", "x2"], "score": [0.9, 0.8]}).to_csv(tmp_path / "stead_report.csv", index=False)
    names, info = asl.load_multiplet_report("stead", report_dirs=[tmp_path])
    assert names == {"x1", "x2"} and info["status"] == "cached" and info["n_flagged"] == 2 and len(info["sha256"]) == 64
    assert asl.load_multiplet_report("geofon", report_dirs=[tmp_path])[1]["status"] == "no_report_for_source"


# ── the held-out reader on a synthetic case ──────────────────────────────────

KEY = "samos_2020"                        # role "dev"
W_T0 = UTCDateTime("2020-10-30T12:01:27Z")
W_T1 = W_T0 + 300
STATIONS = ("HT.CHOS", "HL.PRK")
RATE = 100.0


def picks_frame(rows):
    base = dict(sequence="s", origin=pd.Timestamp("2020-10-30T12:01:00Z"), mag=1.0, channel="HHZ", mode="manual",
                status="reviewed", method="m", agency="A", time_weight=1.0, onset="impulsive", uncertainty=0.1,
                network="XX", source="svc", reference_ok=True)
    return pd.DataFrame([{**base, **r} for r in rows])


@pytest.fixture
def case_dir(tmp_path, monkeypatch):
    """A built-sequence directory the way tests/test_continuous_scoring.py builds one,
    with synthetic events in the waveforms: e1 P at 60 s / S at 68 s on both
    stations, e2 P at 5 s (window start edge) on HT.CHOS, e3 P at 200 s / S at 206 s on HL.PRK."""
    root = tmp_path / "heldout"
    d = root / KEY
    (d / "waveforms").mkdir(parents=True)
    n = int(300 * RATE)
    for sta in STATIONS:
        net, code = sta.split(".")
        events = ((60.0, 68.0), (5.0, None)) if sta == "HT.CHOS" else ((60.0, 68.0), (200.0, 206.0))
        w = record(RATE, events[0][0], events[0][1], length_s=300.0, seed=hash(sta) % 1000, events=events[1:])
        traces = [Trace(w[i], header=dict(network=net, station=code, channel=ch, sampling_rate=RATE, starttime=W_T0))
                  for i, ch in enumerate(("HHZ", "HHN", "HHE"))]
        Stream(traces).write(str(d / "waveforms" / f"{sta}__HH__{W_T0.strftime('%Y%m%dT%H%M%S')}__{W_T1.strftime('%Y%m%dT%H%M%S')}.mseed"),
                             format="MSEED")
    t = lambda s: pd.Timestamp(W_T0.datetime, tz="UTC") + pd.Timedelta(seconds=s)  # noqa: E731
    picks = picks_frame([
        dict(event="e1", station="HT.CHOS", phase="P", time=t(60.0)),
        dict(event="e1", station="HT.CHOS", phase="S", time=t(68.0)),
        dict(event="e1", station="HL.PRK", phase="P", time=t(60.0)),
        dict(event="e1", station="HL.PRK", phase="S", time=t(67.0)),
        dict(event="e2", station="HT.CHOS", phase="P", time=t(5.0)),
        dict(event="e3", station="HL.PRK", phase="P", time=t(200.0)),
        dict(event="e3", station="HL.PRK", phase="S", time=t(206.0)),
        dict(event="e4", station="HT.CHOS", phase="S", time=t(100.0)),                      # S only: no row
        dict(event="e5", station="HT.CHOS", phase="P", time=t(120.0), mode="automatic"),    # not manual: no row
        dict(event="e1", station="HL.XXX", phase="P", time=t(60.0)),                        # no waveform: no row
    ])
    picks.to_parquet(d / "picks.parquet", index=False)
    pd.DataFrame([dict(t0=str(W_T0), t1=str(W_T1), rule="test")]).to_csv(d / "windows.csv", index=False)
    pd.DataFrame([dict(station=s, network=s.split(".")[0], code=s.split(".")[1], band="HH", rate=RATE, lat=38.0 + i, lon=26.0,
                       elev_m=0.0, km=25.0 * (i + 1), route="x", P=1, S=1, picks=2, fetched=True)
                  for i, s in enumerate(STATIONS)]).to_csv(d / "stations.csv", index=False)
    pd.DataFrame([dict(event="e1", origin=t(50.0), lat=38.0, lon=26.0, depth_km=10.0, mag=3.0, source="T")]).to_parquet(
        d / "catalog.parquet", index=False)
    (d / "manifest.json").write_text(json.dumps(dict(key=KEY, label="test")))
    monkeypatch.setattr(policy, "ACCESS_LOG", tmp_path / "access.jsonl")
    return root


def test_heldout_reader_rows_windows_distances_and_ids(case_dir, tmp_path):
    reader = asl.HeldoutReader(KEY, data_root=case_dir)
    rows = {r.trace_id: r for r in reader.rows()}
    assert {r.split(":")[2] + ":" + r.split(":")[3] for r in rows} == {"HT.CHOS:e1", "HL.PRK:e1", "HT.CHOS:e2", "HL.PRK:e3"}
    r = rows[f"{KEY}:{list(rows)[0].split(':')[1]}:HT.CHOS:e1"]
    assert r.rate_hz == RATE and r.waveform.shape == (3, int(120 * RATE) + 1)
    assert abs(r.p_s - 30.0) < 1e-6 and abs(r.s_s - 38.0) < 1e-6
    assert r.distance_km is not None and abs(r.distance_km) < 1e-6 and r.meta["distance_source"] == "catalogue"  # station at the epicentre
    assert r.event_id == "e1" and r.p_status == "reviewed" and r.s_status == "reviewed" and r.component_mask == (True, True, True)
    edge = next(v for k, v in rows.items() if k.endswith("HT.CHOS:e2"))
    assert abs(edge.p_s - 5.0) < 1e-6 and edge.s_s is None                  # shorter window at the file edge
    assert edge.meta["window_start_time"].startswith("2020-10-30T12:01:27")
    e3 = next(v for k, v in rows.items() if k.endswith("HL.PRK:e3"))
    assert e3.meta["distance_source"] == "stations_csv_km" and e3.distance_km == 50.0   # e3 is not catalogued
    audited = asl.finish_source([asl.audit_row(x) for x in rows.values()], KEY)
    assert audited.loc[audited["event_id"] == "e1", "c2_flag"].tolist() == [False, False]
    assert (tmp_path / "access.jsonl").exists()
    assert "picks.parquet" in reader.file_hashes() and len(reader.file_hashes()["waveform_files"]) == 2


def test_heldout_reader_refuses_protected_keys(case_dir):
    with pytest.raises(PermissionError):
        asl.HeldoutReader("noto_2024", data_root=case_dir)


def test_heldout_cli_writes_summary_provenance_report_and_sheet(case_dir, tmp_path):
    out = tmp_path / "out"
    asl.main(["heldout", "--keys", KEY, "--data-root", str(case_dir), "--out-dir", str(out), "--report"])
    summary = pd.read_csv(out / "summary.csv")
    assert summary["source"].tolist() == [KEY] and summary["n_rows"].item() == 4
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["mode"] == "heldout" and prov["keys"] == [KEY] and KEY in prov["cases"]
    assert prov["constants"]["C2_TOL_S"] == 0.5 and prov["c2_rule"] == "asymmetric"
    assert "sha256" not in prov["cases"][KEY]["files"]["picks.parquet"]
    assert (out / KEY / "rows.parquet").exists() and (out / KEY / "review_sheet.csv").exists()
    assert (out / "report.md").read_text().startswith("# Label audit (41B)")
    text = asl.main(["report", "--out-dir", str(out)])
    assert KEY in text


# ── the SeisBench reader on a fake HDF5 + CSV source ─────────────────────────

def _fake_source(cache_root, rate=100.0, bucket=False):
    h5py = pytest.importorskip("h5py")
    d = cache_root / "datasets" / "fake"
    d.mkdir(parents=True)
    rows, waves = [], []
    for i in range(5):
        tp, ts = 10.0 + i, 16.0 + 2 * i
        w = record(rate, tp, ts, length_s=60.0, seed=i)
        waves.append(w)
        rows.append(dict(trace_name=f"tr{i}", trace_sampling_rate_hz=rate, trace_component_order="ZNE",
                         trace_start_time=f"2020-01-01T00:00:{i:02d}Z", trace_p_arrival_sample=int(tp * rate),
                         trace_s_arrival_sample=int(ts * rate) if i != 4 else np.nan, station_code=f"S{i}",
                         station_network_code="XX", path_ep_distance_km=10.0 * (i + 1), trace_p_status="manual",
                         trace_s_status="automatic", source_origin_time="2020-01-01T00:00:00Z", source_latitude_deg=1.0,
                         source_longitude_deg=2.0, source_id=f"ev{i}", split="train"))
    with h5py.File(d / "waveforms.hdf5", "w") as f:
        fmt = f.create_group("data_format")
        fmt["sampling_rate"], fmt["component_order"], fmt["dimension_order"] = rate, "ZNE", "CW"
        data = f.create_group("data")
        if bucket:
            data["bucket0"] = np.stack(waves)
            for i, r in enumerate(rows):
                r["trace_name"] = f"bucket0${i},:3,:{waves[i].shape[1]}"
        else:
            for i, w in enumerate(waves):
                data[f"tr{i}"] = w
    pd.DataFrame(rows).to_csv(d / "metadata.csv", index=False)
    return d


@pytest.mark.parametrize("route", ["seisbench", "single"])
def test_seisbench_reader_reads_fake_source_through_loader_readers(tmp_path, monkeypatch, route):
    pytest.importorskip("h5py")
    pytest.importorskip("seisbench")
    pytest.importorskip("torch")
    cache = tmp_path / "cache"
    _fake_source(cache, bucket=(route == "single"))
    reports = tmp_path / "reports"
    reports.mkdir()
    import label_error_filter as lef
    monkeypatch.setitem(lef.REPORT_STEM_TO_DATASET, "fake", "fake")
    pd.DataFrame({"trace_name": ["tr1"]}).to_csv(reports / "fake_report.csv", index=False)
    reader = asl.SeisBenchReader("fake", cache, sample=4, seed=0, route=route, report_dirs=[reports])
    assert reader.n_candidates == 5 and len(reader.meta) == 4 and reader.route == route
    rows = list(reader.rows())
    reader.close()
    assert len(rows) == 4 and reader.read_errors == []
    by = {r.trace_id.split("$")[0] if route == "seisbench" else r.meta["chunk"] + r.trace_id: r for r in rows}
    r = rows[0]
    i = int(r.event_id[2:])
    assert r.rate_hz == 100.0 and r.waveform.shape == (3, 6000) and r.component_mask == (True, True, True)
    assert abs(r.p_s - (10.0 + i)) < 1e-9 and r.p_sample == (10.0 + i) * 100
    assert r.distance_km == 10.0 * (i + 1) and r.station == f"XX.S{i}" and r.p_status == "manual" and r.s_status == "automatic"
    assert r.meta["distance_bin"] == "local" and r.meta["trace_start_time"].startswith("2020-01-01")
    flagged = [x for x in rows if x.flagged_multiplet]
    assert all(x.event_id == "ev1" for x in flagged) and reader.report_info["status"] == "cached"
    df = asl.finish_source([asl.audit_row(x) for x in rows], "fake")
    assert not df["c2_flag"].any() and df["c2_testable"].all()
    assert by  # rows carry a chunk in meta on every route
