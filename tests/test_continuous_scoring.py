"""Behavioural tests for scripts/continuous_scoring.py and the 35A scorer driver.

Synthetic fixtures only: no network, no torch, no seisbench. Run with
    python -m pytest tests/test_continuous_scoring.py -q
"""
import ast
import itertools
import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("obspy")
pytest.importorskip("scipy")

from obspy import Stream, Trace, UTCDateTime  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import continuous_scoring as cs  # noqa: E402
import evaluation_policy as policy  # noqa: E402
import heldout_testset_score as scorer  # noqa: E402

T0 = pd.Timestamp("2020-01-01T00:00:00Z")
RATE = 100.0


def sec(s):
    return T0 + pd.Timedelta(seconds=s)


def gaussians(n, rate, peaks, sigma_s=0.05):
    """Probability trace of length n with Gaussian peaks [(seconds, amplitude)]."""
    t = np.arange(n) / rate
    out = np.zeros(n)
    for centre, amp in peaks:
        out += amp * np.exp(-0.5 * ((t - centre) / sigma_s) ** 2)
    return out


# ── module hygiene ───────────────────────────────────────────────────────────

def test_engine_imports_no_torch_or_seisbench():
    tree = ast.parse((REPO_ROOT / "scripts" / "continuous_scoring.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"torch", "seisbench"}


# ── extraction: the production rule at each threshold ────────────────────────

def test_threshold_induced_peak_splitting():
    # audit 2026-09-10 finding 2: peaks 0.8 and 0.7 joined by 0.03
    prob = np.zeros(400)
    prob[100], prob[150] = 0.8, 0.7
    prob[101:150] = 0.03
    low = cs.extract_picks(prob, T0, RATE, 0.02, phase="P", station="X")
    assert len(low) == 1 and len(low[low.score >= 0.3]) == 1
    direct = cs.extract_picks(prob, T0, RATE, 0.3, phase="P", station="X")
    assert len(direct) == 2
    assert list(direct.score) == [0.8, 0.7] and list(direct.i_peak) == [100, 150]
    assert list(direct.time) == [sec(1.0), sec(1.5)]
    assert list(direct.columns[:6]) == ["pick_id", "station", "phase", "time", "score", "threshold"]


def seisbench_rule(trace, threshold):
    """Verbatim SeisBench picks_from_annotations (installed base.py lines 2511-2519), for cross-checking."""
    from obspy.signal.trigger import trigger_onset
    triggers = trigger_onset(trace.data, threshold, threshold / 2)
    times = trace.times()
    out = []
    for s0, s1 in triggers:
        peak_value = np.max(trace.data[s0: s1 + 1])
        s_peak = s0 + np.argmax(trace.data[s0: s1 + 1])
        out.append((trace.stats.starttime + times[s_peak], float(peak_value)))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_extract_matches_seisbench_rule_on_random_traces(seed):
    rng = np.random.default_rng(seed)
    n = 6000
    peaks = [(float(c), float(a)) for c, a in zip(rng.uniform(0, n / RATE, 25), rng.uniform(0.02, 1.0, 25))]
    prob = np.clip(gaussians(n, RATE, peaks, sigma_s=rng.uniform(0.03, 0.4)) + rng.normal(0, 0.01, n), 0, 1)
    start = UTCDateTime("2021-06-01T12:34:56.789Z")
    trace = Trace(prob.astype(np.float32), header=dict(sampling_rate=RATE, starttime=start, channel="PhaseNet_P"))
    for thr in (0.05, 0.1, 0.3, 0.5, 0.9):
        want = seisbench_rule(trace, thr)
        got = cs.extract_picks(trace.data, start, RATE, thr, phase="P", station="X")
        assert len(got) == len(want)
        for (t_ref, v_ref), (t_got, v_got) in zip(want, zip(got.time, got.score)):
            assert abs((cs.to_timestamp(t_ref) - t_got).total_seconds()) < 1e-6
            assert v_got == pytest.approx(v_ref, abs=1e-6)


def test_peak_in_invalid_support_is_dropped():
    prob = gaussians(1000, RATE, [(2.0, 0.9), (6.0, 0.9)])
    valid = np.ones(1000, bool)
    valid[550:650] = False
    picks = cs.extract_picks(prob, T0, RATE, 0.3, valid_mask=valid, phase="S", station="X")
    assert list(picks.time) == [sec(2.0)] and list(picks.phase) == ["S"]


def test_thresholds_are_deduplicated_and_sorted():
    assert cs.dedup_thresholds([0.3, 0.1, 0.3, 0.05, 0.1]) == [0.05, 0.1, 0.3]
    with pytest.raises(ValueError):
        cs.dedup_thresholds([0.3, 0.0])


# ── references: identity, not proximity ──────────────────────────────────────

def picks_frame(rows):
    base = dict(sequence="s", origin=T0, mag=1.0, channel="HHZ", mode="manual", status="reviewed", method="m",
                agency="A", time_weight=1.0, onset="impulsive", uncertainty=0.1, network="XX", source="svc",
                reference_ok=True)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_two_close_events_both_kept_and_matchable():
    picks = picks_frame([
        dict(event="e1", station="XX.A", phase="P", time=sec(10.0)),
        dict(event="e2", station="XX.A", phase="P", time=sec(10.3)),
    ])
    ref = cs.reference_picks(picks, ["XX.A"], T0, sec(60))
    assert list(ref.event) == ["e1", "e2"] and list(ref.time) == [sec(10.0), sec(10.3)]
    m = cs.match_picks(ref.time, [sec(10.05), sec(10.32)], tol=0.5)
    assert sorted(p[:2] for p in m.pairs) == [(0, 0), (1, 1)]
    assert m.residuals == pytest.approx([0.05, 0.02])


def test_duplicate_agency_picks_reduce_to_one_row():
    picks = picks_frame([
        dict(event="e1", station="XX.A", phase="P", time=sec(10.05), agency="B", time_weight=50.0),
        dict(event="e1", station="XX.A", phase="P", time=sec(10.00), agency="A", time_weight=100.0),
        dict(event="e1", station="XX.A", phase="P", time=sec(10.20), agency="C", time_weight=100.0),
        dict(event="e1", station="XX.A", phase="S", time=sec(18.0)),
    ])
    ref = cs.reference_picks(picks, ["XX.A"], T0, sec(60))
    assert len(ref) == 2
    p = ref[ref.phase == "P"].iloc[0]
    assert p.time == sec(10.0) and p.agency == "A" and p.n_rows == 3 and p.tier == "manual"
    assert p.ref_id == "e1|XX.A|P"


def test_reference_picks_filters_window_station_tier_and_reference_ok():
    picks = picks_frame([
        dict(event="e1", station="XX.A", phase="P", time=sec(10.0)),
        dict(event="e0", station="XX.A", phase="P", time=sec(-5.0)),                       # before the window
        dict(event="e1", station="XX.B", phase="P", time=sec(11.0)),                       # station not fetched
        dict(event="e3", station="XX.A", phase="S", time=sec(20.0), mode="automatic", reference_ok=False),
        dict(event="e4", station="XX.A", phase="S", time=sec(30.0), mode="unknown", reference_ok=True),
    ])
    ref = cs.reference_picks(picks, ["XX.A"], T0, sec(60))
    assert list(ref.event) == ["e1"]
    both = cs.reference_picks(picks, ["XX.A"], T0, sec(60), tiers=("manual", "unknown"))
    assert list(both.event) == ["e1", "e4"] and list(both.tier) == ["manual", "unknown"]
    every = cs.reference_picks(picks, ["XX.A"], T0, sec(60), tiers=None, require_reference_ok=False)
    assert list(every.event) == ["e1", "e3", "e4"]


# ── matching: maximum cardinality, then minimum residual ─────────────────────

def brute_force(ref, cand, tol):
    """(max cardinality, min total |residual| at that cardinality) by enumeration."""
    best = (0, 0.0)
    for k in range(1, min(len(ref), len(cand)) + 1):
        for ri in itertools.combinations(range(len(ref)), k):
            for cj in itertools.permutations(range(len(cand)), k):
                d = [abs(cand[j] - ref[i]) for i, j in zip(ri, cj)]
                if all(x <= tol + 1e-12 for x in d):
                    cost = sum(d)
                    if k > best[0] or (k == best[0] and cost < best[1]):
                        best = (k, cost)
    return best


def test_greedy_counterexample_gives_two_matches():
    # audit 2026-09-10 finding 3: nearest-first greedy finds one match here
    m = cs.match_picks([0.0, 0.6], [-0.4, 0.1], tol=0.5)
    assert len(m.pairs) == 2 and m.unmatched_reference == [] and m.unmatched_candidate == []
    assert sorted(p[:2] for p in m.pairs) == [(0, 0), (1, 1)]
    assert brute_force([0.0, 0.6], [-0.4, 0.1], 0.5) == (2, pytest.approx(0.9))


@pytest.mark.parametrize("seed", range(40))
def test_matching_agrees_with_brute_force(seed):
    rng = np.random.default_rng(seed)
    ref = sorted(rng.uniform(0, 5, rng.integers(0, 5)))
    cand = sorted(rng.uniform(0, 5, rng.integers(0, 5)))
    tol = float(rng.choice([0.2, 0.5, 1.0]))
    m = cs.match_picks(ref, cand, tol)
    card, cost = brute_force(ref, cand, tol)
    assert len(m.pairs) == card
    assert np.abs(m.residuals).sum() == pytest.approx(cost, abs=1e-8)   # engine rounds to integer ns
    assert len(m.pairs) + len(m.unmatched_reference) == len(ref)
    assert len(m.pairs) + len(m.unmatched_candidate) == len(cand)
    assert len({i for i, _, _ in m.pairs}) == len(m.pairs) == len({j for _, j, _ in m.pairs})


def test_exact_tolerance_boundary_is_feasible():
    assert cs.match_picks([0.0], [0.5], tol=0.5).pairs == [(0, 0, 0.5)]
    assert cs.match_picks([0.0], [-0.5], tol=0.5).pairs == [(0, 0, -0.5)]
    assert cs.match_picks([0.0], [0.5 + 1e-6], tol=0.5).pairs == []
    assert cs.match_picks([sec(10.0)], [sec(10.5)], tol=0.5).pairs == [(0, 0, 0.5)]
    assert cs.match_picks([sec(10.0)], [sec(10.500001)], tol=0.5).pairs == []
    assert cs.match_picks([], [1.0], tol=0.5).unmatched_candidate == [0]


# ── window scoring, coverage, aggregation ────────────────────────────────────

def annotation(n=6000, valid=None, p_peaks=(), s_peaks=()):
    valid = np.ones(n, bool) if valid is None else valid
    return cs.Annotation(T0, RATE, gaussians(n, RATE, p_peaks), gaussians(n, RATE, s_peaks), valid)


def test_gap_makes_reference_uncovered_and_leaves_denominator():
    valid = np.ones(6000, bool)
    valid[2000:3000] = False                                     # 20-30 s gap
    ann = annotation(valid=valid, p_peaks=[(10.0, 0.9), (25.0, 0.9)])
    ref = cs.reference_picks(picks_frame([
        dict(event="e1", station="XX.A", phase="P", time=sec(10.0)),
        dict(event="e2", station="XX.A", phase="P", time=sec(25.0)),
        dict(event="e3", station="XX.B", phase="P", time=sec(40.0)),   # station without annotation
    ]), ["XX.A", "XX.B"], T0, sec(60))
    picks = cs.extract_all(ann, [0.3], station="XX.A")
    rows, matches = cs.score_window(ref, picks, {"XX.A": ann}, window_id="w", model_id="m", thresholds=[0.3], tol=0.5)
    p = rows[rows.phase == "P"].iloc[0]
    assert p.n_reference == 1 and p.n_reference_uncovered == 2 and p.matched == 1 and p.recall == 1.0
    assert p.emitted == 1 and p.unmatched_candidates == 0     # the peak inside the gap was never emitted
    assert list(matches.event) == ["e1"]


def test_zero_reference_window_has_nan_recall_and_counted_emissions():
    ann = annotation(p_peaks=[(5.0, 0.9), (7.0, 0.4)], s_peaks=[(9.0, 0.8)])
    ref = cs.reference_picks(picks_frame([]), ["XX.A"], T0, sec(60)) if False else pd.DataFrame(columns=cs.REFERENCE_COLUMNS)
    picks = cs.extract_all(ann, [0.3, 0.5], station="XX.A")
    rows, matches = cs.score_window(ref, picks, {"XX.A": ann}, window_id="w", model_id="m", thresholds=[0.5, 0.3, 0.3], tol=0.5)
    assert list(rows.threshold) == [0.3, 0.5, 0.3, 0.5] and list(rows.phase) == ["P", "P", "S", "S"]
    assert rows.recall.isna().all() and rows.n_reference.eq(0).all()
    assert list(rows.emitted) == [2, 1, 1, 1] and list(rows.unmatched_candidates) == [2, 1, 1, 1]
    assert len(matches) == 0


def rows_for(key, window_id, model_id, n_ref, matched, emitted, threshold=0.3, phase="P", mae=0.1):
    return dict(model_id=model_id, window_id=window_id, phase=phase, threshold=threshold, n_reference=n_ref,
                n_reference_uncovered=0, matched=matched, emitted=emitted, unmatched_candidates=emitted - matched,
                recall=matched / n_ref if n_ref else np.nan, residual_mae=mae, residual_median=0.0,
                residual_p90=0.2, key=key)


def test_aggregate_sums_listed_windows_of_one_sequence_only():
    rows = pd.DataFrame([
        rows_for("seqA", "w1", "m", 10, 8, 20, mae=0.10),
        rows_for("seqA", "w2", "m", 30, 12, 40, mae=0.20),
        rows_for("seqA", "w3", "m", 5, 5, 5),
        rows_for("seqB", "w4", "m", 100, 100, 100),
    ])
    agg = cs.aggregate(rows, ["w1", "w2"])
    assert len(agg) == 1
    a = agg.iloc[0]
    assert a.n_reference == 40 and a.matched == 20 and a.emitted == 60 and a.recall == 0.5 and a.n_windows == 2
    assert a.residual_mae == pytest.approx((0.10 * 8 + 0.20 * 12) / 20)
    assert a.window_id == "w1+w2" and a.key == "seqA"
    assert len(rows) == 4 and set(rows.window_id) == {"w1", "w2", "w3", "w4"}   # per-window rows untouched
    with pytest.raises(ValueError):
        cs.aggregate(rows, ["w1", "w4"])           # two sequences
    with pytest.raises(ValueError):
        cs.aggregate(rows, ["w1", "missing"])
    with pytest.raises(ValueError):
        cs.aggregate(rows, [])


def test_operating_points_and_matched_budget_use_attained_points_only():
    rows = pd.DataFrame([
        rows_for("s", "w", "a", 100, 80, 100, threshold=0.1), rows_for("s", "w", "a", 100, 60, 60, threshold=0.3),
        rows_for("s", "w", "a", 100, 40, 40, threshold=0.5),
        rows_for("s", "w", "b", 100, 90, 120, threshold=0.1), rows_for("s", "w", "b", 100, 50, 30, threshold=0.5),
        rows_for("s", "w", "a", 50, 20, 20, threshold=0.3, phase="S"), rows_for("s", "w", "b", 50, 25, 21, threshold=0.3, phase="S"),
    ])
    pts = cs.operating_points(rows)
    assert list(pts[pts.phase == "P"].emitted) == [100, 60, 40, 120, 30]
    budget = cs.matched_budget(rows, reference_model="a", reference_threshold=0.3)
    p = budget[budget.phase == "P"].set_index("model_id")
    assert p.loc["a", "threshold"] == 0.3 and p.loc["a", "emitted"] == 60 and p.loc["a", "recall"] == 0.6
    assert not p.loc["b", "within_tolerance"] and pd.isna(p.loc["b", "threshold"])
    assert "closest emitted 30 at threshold 0.5" in p.loc["b", "reason"]
    s = budget[budget.phase == "S"].set_index("model_id")
    assert s.loc["b", "within_tolerance"] and s.loc["b", "emitted"] == 21 and s.loc["b", "recall"] == 0.5
    explicit = cs.matched_budget(rows, target_emitted={"P": 110, "S": 20}).set_index(["phase", "model_id"])
    assert explicit.loc[("P", "a"), "emitted"] == 100 and explicit.loc[("P", "b"), "emitted"] == 120
    with pytest.raises(ValueError):
        cs.matched_budget(rows)
    two_windows = pd.concat([rows, pd.DataFrame([rows_for("s", "w2", "a", 1, 1, 1)])])
    with pytest.raises(ValueError):
        cs.operating_points(two_windows)
    dup = pd.concat([rows, rows.iloc[[0]]])
    with pytest.raises(ValueError):
        cs.operating_points(dup)


def test_common_support_and_failure_rows():
    common, excluded = cs.common_support({"a": {("w", "X"), ("w", "Y")}, "b": {("w", "X")}, "c": {("w", "X"), ("w", "Y"), ("w", "Z")}})
    assert common == {("w", "X")}
    assert excluded.to_dict("records") == [dict(window_id="w", station="Y", missing_models=["b"]),
                                           dict(window_id="w", station="Z", missing_models=["a", "b"])]
    row = cs.failure_row("m", "w", "X", "annotate", RuntimeError("CUDA out of memory"))
    assert row == dict(model_id="m", window_id="w", station="X", stage="annotate", exception="RuntimeError",
                       message="CUDA out of memory")
    assert cs.common_support({})[0] == set()


def test_annotation_store_roundtrip(tmp_path):
    store = cs.AnnotationStore(tmp_path / "ann")
    valid = np.ones(500, bool)
    valid[10:20] = False
    ann = cs.Annotation(T0, RATE, np.linspace(0, 1, 500), np.linspace(1, 0, 500), valid, meta=dict(model="x"))
    store.put("seq", "w1", "XX.A", "m" * 64, ann)
    store.put("seq", "w1", "XX.A", "m" * 64, ann)                       # overwrite keeps one index row
    assert store.has("seq", "w1", "XX.A", "m" * 64) and not store.has("seq", "w1", "XX.B", "m" * 64)
    back = store.get("seq", "w1", "XX.A", "m" * 64)
    assert back.start_time == T0 and back.rate == RATE and back.meta == dict(model="x")
    assert np.allclose(back.P, ann.P) and np.allclose(back.S, ann.S) and (back.valid == valid).all()
    idx = cs.AnnotationStore(tmp_path / "ann").index()             # reload from disk
    assert len(idx) == 1 and idx.iloc[0].n_valid == 490 and idx.iloc[0].n_samples == 500
    assert cs.AnnotationStore(tmp_path / "ann").pairs("seq", "m" * 64) == {("w1", "XX.A")}


def test_valid_mask_from_stream_marks_zero_runs_masked_samples_and_missing_components():
    n = 1000
    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=n).astype(np.float32) for c in ("HHZ", "HHN", "HHE")}
    data["HHZ"][300:500] = 0.0                                          # 2 s zero fill on one component
    data["HHN"][700:705] = 0.0                                          # 0.05 s of zeros: not a gap
    st = Stream([Trace(d, header=dict(network="XX", station="A", channel=c, sampling_rate=RATE,
                                      starttime=UTCDateTime(T0.to_pydatetime()))) for c, d in data.items()])
    st[2].data = np.ma.masked_array(st[2].data, mask=np.arange(n) >= 900)
    valid = cs.valid_mask_from_stream(st, T0, RATE, n)
    assert not valid[300:500].any() and valid[299] and valid[500]
    assert valid[700:705].all()
    assert not valid[900:].any() and valid[899]
    assert valid.sum() == n - 200 - 100
    assert not cs.valid_mask_from_stream(Stream(), T0, RATE, n).any()
    short = Stream([st[0].copy().trim(endtime=st[0].stats.starttime + 4.99)])
    assert cs.valid_mask_from_stream(short, T0, RATE, n)[:500].sum() == 300 and not cs.valid_mask_from_stream(short, T0, RATE, n)[500:].any()


# ── end to end through the guarded scorer ────────────────────────────────────

KEY = "samos_2020"                       # role "dev" in configs/evaluation_suites.json
W_T0 = UTCDateTime("2020-10-30T12:01:27Z")
W_T1 = W_T0 + 120
WID = "20201030T120127_20201030T120327"
STATIONS = ("HT.CHOS", "HL.PRK")

PEAKS = {
    "A": {"HT.CHOS": dict(P=[(10.0, 0.8), (10.3, 0.7)], S=[(18.0, 0.9), (80.0, 0.25)]),
          "HL.PRK": dict(P=[(10.0, 0.9), (50.0, 0.9)], S=[(17.0, 0.9)])},
    "B": {"HT.CHOS": dict(P=[(10.0, 0.6)], S=[(18.0, 0.9)]),
          "HL.PRK": RuntimeError("CUDA error: device-side assert")},
}
FINGERPRINTS = {"A": dict(state_sha256="a" * 64, class_="test.Model"), "B": dict(state_sha256="b" * 64, class_="test.Model")}


def synth_annotate(model, stream):
    spec = PEAKS[model.name][f"{stream[0].stats.network}.{stream[0].stats.station}"]
    if isinstance(spec, Exception):
        raise spec
    n, hdr = stream[0].stats.npts, stream[0].stats
    traces = []
    for label in ("P", "S"):
        traces.append(Trace(gaussians(n, hdr.sampling_rate, spec[label]).astype(np.float32),
                            header=dict(network=hdr.network, station=hdr.station, channel=f"PhaseNet_{label}",
                                        sampling_rate=hdr.sampling_rate, starttime=hdr.starttime)))
    traces.append(Trace(np.ones(n, np.float32), header=dict(network=hdr.network, station=hdr.station,
                                                             channel="PhaseNet_N", sampling_rate=hdr.sampling_rate,
                                                             starttime=hdr.starttime)))
    return Stream(traces)


class Model:
    def __init__(self, name):
        self.name = name

    def annotate(self, stream):
        return synth_annotate(self, stream)


@pytest.fixture
def sequence_dir(tmp_path, monkeypatch):
    """A built-sequence directory for KEY: picks.parquet, windows.csv, stations.csv, manifest.json, waveforms."""
    root = tmp_path / "heldout"
    d = root / KEY
    (d / "waveforms").mkdir(parents=True)
    rng = np.random.default_rng(1)
    n = int(120 * RATE)
    for sta in STATIONS:
        net, code = sta.split(".")
        traces = []
        for comp in ("HHZ", "HHN", "HHE"):
            data = rng.normal(size=n).astype(np.float32)
            if sta == "HL.PRK":
                data[4500:5500] = 0.0                                    # zero-filled gap 45-55 s
            traces.append(Trace(data, header=dict(network=net, station=code, channel=comp, sampling_rate=RATE, starttime=W_T0)))
        Stream(traces).write(str(d / "waveforms" / f"{sta}__HH__{W_T0.strftime('%Y%m%dT%H%M%S')}__{W_T1.strftime('%Y%m%dT%H%M%S')}.mseed"), format="MSEED")
    t = lambda s: cs.to_timestamp(W_T0) + pd.Timedelta(seconds=s)  # noqa: E731
    picks = picks_frame([
        dict(event="e1", station="HT.CHOS", phase="P", time=t(10.0), agency="A", time_weight=100.0),
        dict(event="e1", station="HT.CHOS", phase="P", time=t(10.05), agency="B", time_weight=50.0),
        dict(event="e1", station="HT.CHOS", phase="S", time=t(18.0)),
        dict(event="e1", station="HL.PRK", phase="P", time=t(10.0)),
        dict(event="e1", station="HL.PRK", phase="S", time=t(17.0)),
        dict(event="e2", station="HT.CHOS", phase="P", time=t(10.3)),
        dict(event="e3", station="HL.PRK", phase="P", time=t(50.0)),
        dict(event="e0", station="HT.CHOS", phase="P", time=t(-5.0)),
        dict(event="e4", station="HT.CHOS", phase="S", time=t(60.0), mode="automatic", reference_ok=False),
    ])
    picks.to_parquet(d / "picks.parquet", index=False)
    pd.DataFrame([dict(t0=str(W_T0), t1=str(W_T1), rule="test")]).to_csv(d / "windows.csv", index=False)
    pd.DataFrame([dict(station=s, network=s.split(".")[0], code=s.split(".")[1], band="HH", rate=RATE, lat=0.0, lon=0.0,
                       elev_m=0.0, km=1.0, route="x", P=1, S=1, picks=2, fetched=True) for s in STATIONS]).to_csv(d / "stations.csv", index=False)
    (d / "manifest.json").write_text(json.dumps(dict(key=KEY, label="test")))
    monkeypatch.setattr(scorer, "OUT_ROOT", root)
    monkeypatch.setattr(policy, "ACCESS_LOG", tmp_path / "access.jsonl")
    monkeypatch.setattr(policy, "model_fingerprint", lambda model: FINGERPRINTS[model.name])
    return tmp_path


def test_score_end_to_end_with_failure_common_support_and_pick_store(sequence_dir):
    tmp = sequence_dir
    models = {"A": Model("A"), "B": Model("B")}
    res = scorer.score(KEY, models, annotate_fn=synth_annotate, thresholds=[0.3, 0.1, 0.5, 0.3],
                       annotations_root=tmp / "ann", out_dir=tmp / "scores")
    a, b = "a" * 64, "b" * 64
    # failure table and common-support gate
    assert res.failures.to_dict("records") == [dict(model_id=b, window_id=WID, station="HL.PRK", stage="annotate",
                                                    exception="RuntimeError", message="CUDA error: device-side assert")]
    assert res.excluded.to_dict("records") == [dict(window_id=WID, station="HL.PRK", missing_models=[b])]
    idx = cs.AnnotationStore(tmp / "ann").index()
    assert sorted(zip(idx.model_id, idx.station)) == [(a, "HL.PRK"), (a, "HT.CHOS"), (b, "HT.CHOS")]
    # per-window rows: thresholds deduplicated, identical denominators for both models
    win = res.rows[res.rows.scope == "window"].set_index(["model_id", "phase", "threshold"]).sort_index()
    assert list(win.reset_index().threshold.unique()) == [0.1, 0.3, 0.5]
    assert len(win) == 12
    assert (win.groupby("phase").n_reference_uncovered.nunique() == 1).all()   # same denominator for both models
    for thr in (0.1, 0.3, 0.5):
        assert win.loc[(a, "P", thr), ["n_reference", "n_reference_uncovered", "matched", "emitted"]].tolist() == [2, 2, 2, 2]
        assert win.loc[(b, "P", thr), ["n_reference", "n_reference_uncovered", "matched", "emitted"]].tolist() == [2, 2, 1, 1]
        assert win.loc[(a, "S", thr), ["n_reference", "n_reference_uncovered", "matched"]].tolist() == [1, 1, 1]
    assert win.loc[(a, "P", 0.3), "recall"] == 1.0 and win.loc[(b, "P", 0.3), "recall"] == 0.5
    assert win.loc[(a, "S", 0.1), "emitted"] == 2 and win.loc[(a, "S", 0.1), "unmatched_candidates"] == 1
    assert win.loc[(a, "S", 0.3), "emitted"] == 1
    assert win.loc[(a, "P", 0.3), "residual_mae"] == pytest.approx(0.0, abs=1e-6)
    # aggregate rows exist for the listed window and the matched-budget table was computed
    agg = res.rows[res.rows.scope == "aggregate"]
    assert set(agg.window_id) == {WID} and len(agg) == 12 and (agg.n_windows == 1).all()
    bud = res.budget.set_index(["phase", "model"])
    assert bud.loc[("P", "A"), "emitted"] == 2 and bud.loc[("P", "A"), "target_emitted"] == 2
    assert not bud.loc[("P", "B"), "within_tolerance"] and "closest emitted 1" in bud.loc[("P", "B"), "reason"]
    assert bud.loc[("S", "B"), "within_tolerance"] and bud.loc[("S", "B"), "recall"] == 1.0
    # pick store: every emitted pick, with the exact matched reference event or null
    store = res.picks
    assert set(scorer.PICK_STORE_COLUMNS) <= set(store.columns) and store.pick_id.is_unique
    assert (store.access_id == res.access_id).all() and (store.key == KEY).all() and set(store.window_id) == {WID}
    assert set(store.station) == {"HT.CHOS"}                             # excluded pair not extracted
    p03 = store[(store.model == "A") & (store.phase == "P") & (store.threshold == 0.3)].sort_values("time")
    assert list(p03.matched_event) == ["e1", "e2"] and list(p03.ref_id) == ["e1|HT.CHOS|P", "e2|HT.CHOS|P"]
    assert list(p03.time) == [cs.to_timestamp(W_T0 + 10.0), cs.to_timestamp(W_T0 + 10.3)]
    false_alarm = store[(store.model == "A") & (store.phase == "S") & (store.threshold == 0.1) & (store.score < 0.5)]
    assert len(false_alarm) == 1 and pd.isna(false_alarm.matched_event.iloc[0])
    assert (store[store.model == "B"].matched_event.dropna() == "e1").all()
    # artifacts and access record
    out = tmp / "scores" / KEY / res.access_id
    assert {f.name for f in out.iterdir()} >= {"rows.parquet", "picks.parquet", "matches.parquet", "failures.parquet",
                                              "excluded.parquet", "budget.parquet", "models.csv"}
    assert pd.read_parquet(out / "picks.parquet").pick_id.is_unique
    entry = json.loads((tmp / "access.jsonl").read_text().splitlines()[-1])
    assert entry["id"] == res.access_id and entry["operation"] == "model_scoring" and entry["role"] == "dev"
    assert entry["settings"]["thresholds"] == [0.1, 0.3, 0.5] and entry["settings"]["match_tol_s"] == 0.5
    assert entry["models"] == FINGERPRINTS


def test_score_single_model_counts_gap_reference_as_uncovered(sequence_dir):
    tmp = sequence_dir
    res = scorer.score(KEY, {"A": Model("A")}, annotate_fn=synth_annotate, thresholds=[0.3], annotations_root=tmp / "ann")
    assert len(res.failures) == 0 and len(res.excluded) == 0
    win = res.rows[res.rows.scope == "window"].set_index("phase")
    assert win.loc["P", ["n_reference", "n_reference_uncovered", "matched", "emitted"]].tolist() == [3, 1, 3, 3]
    assert win.loc["S", ["n_reference", "n_reference_uncovered", "matched", "emitted"]].tolist() == [2, 0, 2, 2]
    assert not ((res.picks.station == "HL.PRK") & (res.picks.time > cs.to_timestamp(W_T0 + 45))).any()
    # rerun reuses the stored annotations: no annotate_fn call
    def forbidden(model, stream):
        pytest.fail("annotation recomputed although stored")
    again = scorer.score(KEY, {"A": Model("A")}, annotate_fn=forbidden, thresholds=[0.3, 0.1], annotations_root=tmp / "ann")
    assert list(again.rows[again.rows.scope == "window"].threshold.unique()) == [0.1, 0.3]


def test_main_prints_window_and_budget_tables_and_writes_artifacts(sequence_dir, monkeypatch, capsys):
    tmp = sequence_dir
    fake_models = types.ModuleType("seisbench.models")
    fake_models.PhaseNet = types.SimpleNamespace(from_pretrained=lambda name: Model(name))
    monkeypatch.setitem(sys.modules, "seisbench", types.ModuleType("seisbench"))
    monkeypatch.setitem(sys.modules, "seisbench.models", fake_models)
    results = scorer.main(["--sequence", KEY, "--weights", "A", "B", "--thresholds", "0.3", "0.1", "0.3",
                           "--annotations-root", str(tmp / "ann"), "--out-dir", str(tmp / "scores"),
                           "--out", str(tmp / "rows.csv")])
    out = capsys.readouterr().out
    assert "-- per-window rows" in out and "-- matched budget" in out and "-- failures" in out
    assert "closest emitted 1 at threshold" in out and "RuntimeError" in out
    assert len(results) == 1 and (tmp / "rows.csv").exists()
    rows = pd.read_csv(tmp / "rows.csv")
    assert sorted(rows.threshold.unique()) == [0.1, 0.3] and set(rows.scope) == {"window", "aggregate"}
    with pytest.raises(SystemExit):
        scorer.main(["--sequence", KEY, "--weights", "A", "--budget-reference", "Z"])
