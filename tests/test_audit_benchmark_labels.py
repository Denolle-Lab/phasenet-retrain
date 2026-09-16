"""Behavioural tests for scripts/audit_benchmark_labels.py (checkpoint 41B, benchmark part).

Synthetic results frames only: no parquet of the real benchmark, no network,
no model. Run with:  python -m pytest tests/test_audit_benchmark_labels.py -q
"""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import audit_benchmark_labels as abl  # noqa: E402

W = abl.PUBLIC_WEIGHTS


def results_frame(traces):
    """One results row per (weight, trace) from a list of dicts with keys
    dataset, trace_name, trained_models, p_in_window, s_in_window, snr_db and
    per-weight overrides: residual (dict weight -> value or scalar), prob,
    s_residual, s_prob."""
    rows = []
    for t in traces:
        for w in W:
            res = t.get("residual", 0.0)
            prob = t.get("prob", 0.9)
            sres = t.get("s_residual", 0.0)
            sprob = t.get("s_prob", 0.9)
            rows.append(dict(weight=w, dataset=t["dataset"], trace_name=t["trace_name"], dist_bin=t.get("dist_bin", "local"),
                             trained_models=t.get("trained_models", "nan"), snr_db=t.get("snr_db", 15.0),
                             p_in_window=t.get("p_in_window", 1500), s_in_window=t.get("s_in_window", -1),
                             p_prob=res_of(prob, w), s_prob=res_of(sprob, w),
                             p_residual_s=res_of(res, w) if t.get("p_in_window", 1500) >= 0 else np.nan,
                             s_residual_s=res_of(sres, w) if t.get("s_in_window", -1) >= 0 else np.nan))
    return pd.DataFrame(rows)


def res_of(value, weight):
    return value.get(weight, value.get("default", 0.0)) if isinstance(value, dict) else value


def screen(results):
    traces = abl.trace_table(results)
    indep = abl.independence(traces, W)
    return abl.screen_phase(results, traces, indep, "p").set_index(traces.index), traces, indep


def test_eight_agreeing_independent_weights_flag_a_0p8_s_offset():
    eight = {w: 0.8 for w in W[:8]}
    eight["default"] = np.nan
    prob = {w: 0.9 for w in W[:8]}
    prob["default"] = 0.05
    r = results_frame([dict(dataset="ceed", trace_name="a", residual=eight, prob=prob)])
    out, _, _ = screen(r)
    row = out.iloc[0]
    assert row["n_independent"] == 16 and row["n_detect"] == 8
    assert row["consensus_residual_s"] == pytest.approx(0.8) and row["consensus_mad_s"] == 0.0
    assert bool(row["s1_flag"]) and not bool(row["consensus_at_edge"]) and not bool(row["s2_flag"])


def test_consensus_at_the_source_trace_start_is_edge_not_flag():
    # label 5 s into a 60 s source trace (P at sample 500 of a 6000-sample bucket name); the window put it at 15 s;
    # eight weights agree on -4.95 s: the consensus sits at 0.05 s of the source, before its first second
    r = results_frame([dict(dataset="stead", trace_name="bucket0$1,:3,:6000", residual={w: -4.95 for w in W[:8]} | {"default": np.nan},
                            prob={w: 0.9 for w in W[:8]} | {"default": 0.1}, p_in_window=1500)])
    traces = abl.trace_table(r)
    traces["p_source_s"] = 5.0
    traces["s_source_s"] = np.nan
    traces["source_length_s"] = 60.0
    indep = abl.independence(traces, W)
    out = abl.screen_phase(r, traces, indep, "p")
    row = out.iloc[0]
    assert row["n_detect"] == 8 and row["consensus_residual_s"] == pytest.approx(-4.95)
    assert row["consensus_source_position_s"] == pytest.approx(0.05)
    assert bool(row["consensus_at_edge"]) and not bool(row["s1_flag"])
    # the same offset away from every edge flags
    traces["p_source_s"] = 30.0
    out = abl.screen_phase(r, traces, indep, "p")
    assert bool(out.iloc[0]["s1_flag"]) and not bool(out.iloc[0]["consensus_at_edge"])


def test_window_edge_and_source_end_are_edges_too():
    prob = np.full((1, 16), 0.9)
    indep = np.ones((1, 16), dtype=bool)
    # label at 3 s of the window, consensus at -2.5 s -> window position 0.5 s
    out = abl.consensus(prob, np.full((1, 16), -2.5), indep, np.array([3.0]), np.array([40.0]), np.array([60.0]))
    assert bool(out["consensus_at_edge"][0]) and not bool(out["s1_flag"][0])
    # label at 59.5 s of a 60 s source, consensus +0.4 s -> 59.9 s, within the last second
    out = abl.consensus(prob, np.full((1, 16), 0.4), indep, np.array([15.0]), np.array([59.5]), np.array([60.0]))
    assert bool(out["consensus_at_edge"][0])
    # unknown source length: only the start applies
    out = abl.consensus(prob, np.full((1, 16), 0.4), indep, np.array([15.0]), np.array([59.5]), np.array([np.nan]))
    assert not bool(out["consensus_at_edge"][0]) and bool(out["s1_flag"][0])


def test_own_dataset_weights_are_excluded_from_the_consensus():
    # a stead trace: stead and original trained on STEAD and must not count; the six others agreeing are enough
    agree = {w: 1.0 for w in ("stead", "original", "jma", "lendb", "iquique", "ethz", "scedc", "geofon")}
    agree["default"] = np.nan
    prob = {w: 0.95 for w in agree}
    prob["default"] = 0.0
    r = results_frame([dict(dataset="stead", trace_name="t", trained_models="stead", residual=agree, prob=prob)])
    out, traces, indep = screen(r)
    assert not indep.loc[("stead", "t"), "stead"] and not indep.loc[("stead", "t"), "original"]
    assert indep.loc[("stead", "t"), "jma_wc"] and indep.loc[("stead", "t"), "instance"]
    row = out.iloc[0]
    assert row["n_independent"] == 14 and row["n_detect"] == 6 and bool(row["s1_flag"])
    # with only five independent agreeing weights there is no consensus
    r2 = results_frame([dict(dataset="stead", trace_name="t", trained_models="stead",
                             residual={k: v for k, v in agree.items() if k != "geofon"} | {"default": np.nan},
                             prob={k: v for k, v in prob.items() if k != "geofon"} | {"default": 0.0})])
    out2, _, _ = screen(r2)
    assert out2.iloc[0]["n_detect"] == 5 and not bool(out2.iloc[0]["s1_flag"]) and not bool(out2.iloc[0]["has_consensus"])


def test_disagreeing_weights_have_no_consensus():
    spread = {w: (-1.0) ** i * 0.6 * (i + 1) for i, w in enumerate(W)}
    r = results_frame([dict(dataset="ceed", trace_name="a", residual=spread, prob=0.9)])
    out, _, _ = screen(r)
    assert out.iloc[0]["n_detect"] == 16 and out.iloc[0]["consensus_mad_s"] > abl.S1_MAX_MAD_S
    assert not bool(out.iloc[0]["s1_flag"])


def test_s2_fires_only_above_10_db_and_with_at_most_one_detection():
    lone = {"jma": 0.9, "default": 0.05}
    r = results_frame([dict(dataset="ceed", trace_name="hi", prob=lone, snr_db=15.0),
                       dict(dataset="ceed", trace_name="lo", prob=lone, snr_db=8.0),
                       dict(dataset="ceed", trace_name="two", prob={"jma": 0.9, "lendb": 0.5, "default": 0.05}, snr_db=15.0),
                       dict(dataset="ceed", trace_name="nolabel", prob=lone, snr_db=15.0, p_in_window=-1)])
    out, _, _ = screen(r)
    flags = out["s2_flag"].to_dict()
    assert flags[("ceed", "hi")] and not flags[("ceed", "lo")] and not flags[("ceed", "two")] and not flags[("ceed", "nolabel")]


def test_saturated_consensus_is_marked():
    r = results_frame([dict(dataset="ceed", trace_name="a", residual=5.0, prob=0.9, p_in_window=1000)])
    out, _, _ = screen(r)
    assert bool(out.iloc[0]["consensus_saturated"]) and bool(out.iloc[0]["s1_flag"])


def manifest_frame(n=30, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(5, 120, n)
    z = np.full(n, 8.0)
    ts = 0.125 * np.sqrt(d ** 2 + z ** 2) + rng.normal(0, 0.05, n)
    m = pd.DataFrame(dict(dataset="syn", trace_name=[f"bucket0${i},:3,:6000" for i in range(n)], trained_models="nan",
                          distance_km=d, depth_km=z, ts_tp_s=ts, p_arrival_sample=500.0, s_arrival_sample=500.0 + ts * 100,
                          has_p_pick=True, has_s_pick=True, evaluate_s=True, p_in_s_window=ts <= 15, multi_arrival=False,
                          clipped_flag=False, dist_bin="local", magnitude=2.0))
    return m


def test_c1_flags_a_swapped_pair_and_reports_slope_and_vp_vs():
    m = manifest_frame()
    # swap the S-P of the nearest and the farthest row
    i, j = int(np.argmin(m["distance_km"])), int(np.argmax(m["distance_km"]))
    m.loc[[i, j], "ts_tp_s"] = m.loc[[j, i], "ts_tp_s"].to_numpy()
    rows, per = abl.c1_manifest(m)
    assert rows["c1_flag"].sum() == 2 and set(rows.index[rows["c1_flag"]]) == {i, j}
    assert rows["c1_epi_flag"].sum() == 2
    assert per["syn"]["c1_hyp_slope_s_per_km"] == pytest.approx(0.125, abs=0.01)
    assert per["syn"]["c1_hyp_implied_vp_vs"] == pytest.approx(1.75, abs=0.06)
    assert 0.10 < per["syn"]["c1_slope_s_per_km"] < 0.14 and per["syn"]["c1_n_fit"] == 30


def test_c4_mode_share_and_source_edge():
    m = manifest_frame()
    m["notebook_rate_hz"] = 100.0
    m["source_trace_samples"] = 6000.0
    m.loc[0, "p_arrival_sample"] = 50.0     # 0.5 s into a 60 s trace
    c4 = abl.c4_manifest(m)
    assert c4["c4_p_sample_mode"] == 500.0 and c4["c4_p_sample_mode_share"] == pytest.approx(29 / 30)
    assert c4["c4_n_source_edge"] == 1 and c4["c4_source_edge_frac"] == pytest.approx(1 / 30)
    assert c4["c4_source_samples_mode"] == 6000.0 and c4["c4_notebook_rate_hz_mode"] == 100.0


def test_paired_difference_reproduces_recall_and_the_bootstrap_brackets_it():
    detect = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=float)
    clean = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0], dtype=bool)
    d = abl.paired_difference(detect, clean, n_boot=200, seed=1)
    assert d["n_all"] == 10 and d["n_clean"] == 7
    assert d["value_all"] == pytest.approx(0.7) and d["value_clean"] == pytest.approx(5 / 7)
    assert d["diff"] == pytest.approx(0.7 - 5 / 7) and d["diff_lo"] <= d["diff"] <= d["diff_hi"]


def test_sensitivity_table_reproduces_recall_from_a_tiny_frame():
    r = results_frame([dict(dataset="ceed", trace_name=f"t{i}", prob=0.9 if i % 3 else 0.1, s_in_window=1800,
                            s_prob=0.5 if i % 2 else 0.1) for i in range(12)])
    r = r[r["weight"].isin(("jma_wc", "instance"))].copy()
    flags = pd.DataFrame(dict(dataset="ceed", trace_name=[f"t{i}" for i in range(12)], has_label=True, has_label_s=True,
                              s1_flag=[i == 0 for i in range(12)], s2_flag=False, s1_flag_s=False, s2_flag_s=False,
                              c1_flag=[i == 11 for i in range(12)]))
    sens = abl.sensitivity_table(r, flags, weights=("jma_wc", "instance"))
    row = sens[(sens["weight"] == "jma_wc") & (sens["dataset"] == "ceed") & (sens["phase"] == "P")
               & (sens["metric"] == "recall_t03") & (sens["clean_rule"] == "s1_s2_c1")].iloc[0]
    assert row["n_all"] == 12 and row["n_clean"] == 10
    assert row["value_all"] == pytest.approx(8 / 12) and row["value_clean"] == pytest.approx(7 / 10)
    s_row = sens[(sens["weight"] == "instance") & (sens["dataset"] == "all") & (sens["phase"] == "S")
                 & (sens["metric"] == "recall_t03") & (sens["clean_rule"] == "s1_c1")].iloc[0]
    assert s_row["value_all"] == pytest.approx(6 / 12) and s_row["n_clean"] == 11
    assert set(sens["clean_rule"]) == set(abl.CLEAN_RULES)


def test_review_sheet_keeps_at_most_30_rows_per_dataset_strongest_first():
    n = 80
    flags = pd.DataFrame(dict(dataset=["ceed"] * n + ["pnw"] * 10, trace_name=[f"t{i}" for i in range(n + 10)]))
    m = len(flags)
    rng = np.random.default_rng(0)
    for sfx in ("", "_s"):
        flags[f"label_s{sfx}"] = 10.0
        flags[f"consensus_residual_s{sfx}"] = rng.uniform(0.4, 4.0, m)
        flags[f"consensus_position_s{sfx}"] = 10.0 + flags[f"consensus_residual_s{sfx}"]
        flags[f"consensus_source_position_s{sfx}"] = flags[f"consensus_position_s{sfx}"]
        flags[f"n_detect{sfx}"] = 12
        flags[f"n_independent{sfx}"] = 15
        flags[f"consensus_mad_s{sfx}"] = 0.05
        flags[f"consensus_saturated{sfx}"] = False
        flags[f"s1_flag{sfx}"] = True
        flags[f"s2_flag{sfx}"] = False
    flags["snr_db"] = 12.0
    flags["c1_flag"] = False
    flags["p_in_s_window"] = False
    flags["multi_arrival"] = False
    flags["aguilar_flagged"] = False
    flags["p_arrival_sample"] = 500.0
    flags["s_arrival_sample"] = 1500.0
    sheet = abl.review_sheet(flags)
    counts = sheet.groupby("dataset").size().to_dict()
    assert counts == {"ceed": 30, "pnw": 20}      # 10 pnw traces x (P, S)
    ceed = sheet[sheet["dataset"] == "ceed"]
    assert ceed["score"].is_monotonic_decreasing and (ceed["reason"] == "s1_consensus_offset").all()
    assert {"trace_name", "phase", "consensus_residual_s", "n_detect", "consensus_mad_s", "snr_db"} <= set(sheet.columns)


def test_run_writes_every_output_without_a_report_cache(tmp_path):
    m = manifest_frame(n=40)
    m.loc[:19, "dataset"] = "stead"
    m.loc[:19, "trained_models"] = "stead"
    m["trace_name"] = [f"bucket0${i},:3,:6000" for i in range(40)]
    m.to_csv(tmp_path / "bench.csv", index=False)
    traces = [dict(dataset=ds, trace_name=tn, trained_models=tm, s_in_window=2000, snr_db=12.0,
                   residual=({w: 1.0 for w in W[:9]} | {"default": np.nan}) if i == 3 else 0.0,
                   prob=({w: 0.9 for w in W[:9]} | {"default": 0.1}) if i == 3 else 0.8)
              for i, (ds, tn, tm) in enumerate(zip(m["dataset"], m["trace_name"], m["trained_models"]))]
    r = results_frame(traces)
    extra = r[r["weight"] == "jma_wc"].assign(weight="jma_wc_ft_global_v7")
    pd.concat([r, extra]).to_parquet(tmp_path / "res.parquet", index=False)
    out = abl.run(tmp_path / "bench.csv", tmp_path / "res.parquet", tmp_path / "out", report_dirs=[tmp_path / "reports"],
                  download=False)
    for name in ("per_trace_flags.parquet", "per_dataset_summary.csv", "sensitivity.csv", "sensitivity_gap.csv",
                 "review_sheet.csv", "provenance.json", "tables.md"):
        assert (tmp_path / "out" / name).exists(), name
    prov = json.loads((tmp_path / "out" / "provenance.json").read_text())
    assert prov["label_error_reports"]["stead"]["status"] == "report_not_cached"
    assert prov["label_error_reports"]["syn"]["status"] == "no_report_for_dataset"
    assert prov["weights_missing"] == [] and prov["n_traces"] == 40
    flags = out["flags"]
    assert flags["s1_flag"].sum() == 1 and flags.loc[flags["s1_flag"], "trace_name"].iloc[0] == "bucket0$3,:3,:6000"
    summary = out["summary"].set_index("dataset")
    assert summary.loc["stead", "n_results"] == 20 and summary.loc["stead", "p_n_independent_median"] == 14
    assert summary.loc["syn", "p_n_independent_median"] == 16
    assert set(out["sensitivity"]["weight"]) == {"jma_wc", "instance", "jma_wc_ft_global_v7"}
    assert (out["gaps"]["pair"] == "jma_wc_ft_global_v7 - jma_wc").any()
