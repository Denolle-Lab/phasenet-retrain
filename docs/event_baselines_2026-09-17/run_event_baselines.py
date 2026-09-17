#!/usr/bin/env python3
"""
run_event_baselines.py  (issue #36, checkpoint 36B)

Event-level baselines of jma_wc, instance and quakescope2026 on the seven
regression/development cases with PyOcto, from the 35A pick stores written
by scripts/heldout_testset_score.py. One scripts/event_association.py run
per case and weight through its CLI entry point (`--associator pyocto`,
the regime config of the registry, the frozen matching tolerances), then the
committed tables of this directory.

Operating points (35B, docs/baselines_2026-09-13/matched_budget.csv): the
parent jma_wc at 0.3 for P and S; each other weight at the threshold at
which it attains the parent's emitted count, per case and phase
(`thr_<weight>` columns). Nothing here tunes the associator: a case on which
PyOcto returns nothing is recorded as such in runs.csv with the reason the
inputs suggest.

    python docs/event_baselines_2026-09-17/run_event_baselines.py \
        --scores data/evaluation/scores --events data/evaluation/events \
        --out docs/event_baselines_2026-09-17
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import event_association as ea  # noqa: E402
import heldout_testset_registry as reg  # noqa: E402
import evaluation_policy as policy  # noqa: E402

KEYS = ["kaikoura_2016", "norcia_2016", "thessaly_2021", "samos_2020", "adriatic_2022", "etna_2022_2024",
        "corinth_thiva_2020"]
WEIGHTS = ["jma_wc", "instance", "quakescope2026"]
PARENT = "jma_wc"
PARENT_THRESHOLD = 0.3
MATCHED_BUDGET = REPO_ROOT / "docs" / "baselines_2026-09-13" / "matched_budget.csv"
EVALUABILITY = REPO_ROOT / "data" / "heldout_testset" / "evaluability.csv"
N_BOOT, CI, SEED = 2000, 0.95, 0


def operating_points(budget_path=MATCHED_BUDGET) -> pd.DataFrame:
    """key, weight, phase, threshold, source: the 35B matched-budget thresholds, the parent at 0.3."""
    mb = pd.read_csv(budget_path)
    rows = []
    for r in mb.itertuples():
        for w in WEIGHTS:
            thr = PARENT_THRESHOLD if w == PARENT else float(getattr(r, f"thr_{w}"))
            rows.append(dict(key=r.key, weight=w, phase=r.phase, threshold=thr,
                             budget_target_picks=int(r.target),
                             source=("parent operating point (35B budget reference)" if w == PARENT
                                     else "35B matched budget, docs/baselines_2026-09-13/matched_budget.csv")))
    return pd.DataFrame(rows)


def find_pick_store(scores_root: Path, key: str) -> Path:
    """The one scoring run under <scores_root>/<key>/ that holds the three weights; refuse ambiguity."""
    candidates = []
    for d in sorted((scores_root / key).glob("*")):
        if not (d / "picks.parquet").exists() or not (d / "models.csv").exists():
            continue
        models = pd.read_csv(d / "models.csv")
        if set(WEIGHTS) <= set(models["model"]):
            candidates.append(d)
    if len(candidates) != 1:
        raise ValueError(f"{key}: expected exactly one scoring run with {WEIGHTS} under {scores_root / key}, "
                         f"found {[c.name for c in candidates]}")
    return candidates[0]


def run_one(key, weight, store_dir: Path, thr_p, thr_s, out_dir: Path, associator="pyocto") -> dict:
    models = pd.read_csv(store_dir / "models.csv").set_index("model")["model_id"]
    argv = ["--sequence", key, "--picks", str(store_dir / "picks.parquet"), "--model-id", models[weight],
            "--threshold-p", f"{thr_p:.2f}", "--threshold-s", f"{thr_s:.2f}", "--associator", associator,
            "--out-dir", str(out_dir)]
    print(f"\n### {key} {weight}: python scripts/event_association.py {' '.join(argv)}", flush=True)
    t0 = time.perf_counter()
    status, note, res = "ok", "", None
    try:
        res = ea.main(argv)
    except SystemExit as exc:                     # argparse error (policy refusal, missing threshold, ...)
        status, note = "error", f"SystemExit {exc.code}"
    except Exception as exc:  # noqa: BLE001
        status, note = "error", f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    row = dict(key=key, weight=weight, status=status, wall_s=round(wall, 1), note=note, out_dir=str(out_dir),
               command="python scripts/event_association.py " + " ".join(argv))
    if res is not None:
        m = res["meta"]
        row.update(access_id=m["access_id"], model_id=m["model_id"], config_region=m["config"]["region"],
                   config_version=m["config"]["version"], config_sha256=m["config_sha256"],
                   pyocto_version=m["pyocto_version"], associate_runtime_s=round(m["associate_runtime_s"], 2),
                   n_picks=m["n_picks"], n_stations=m["n_stations"], n_events=m["n_events"],
                   n_reference=m["n_reference"], n_matched=m["n_matched"], git_commit=m["git_commit"])
        if m["n_events"] == 0:
            row["status"] = "no_events"
    return row, res


def _prefixed_pairs(res, key):
    pairs = res["match"].pairs[["event"]].copy()
    pairs["event"] = key + ":" + pairs["event"].astype(str)
    return pairs


def _covered_reference(res, key):
    cov = res["tables"]["coverage"]
    cov = cov[cov["in_window"]].copy()
    cov["event"] = key + ":" + cov["event"].astype(str)
    return cov[["event", "origin", "lat", "lon", "depth_km"] + (["mag"] if "mag" in cov else [])]


def recovery_interval(res, key):
    """Percentile interval of recovery itself: the paired bootstrap against an empty comparator."""
    ref = _covered_reference(res, key)
    if len(ref) == 0:
        return np.nan, np.nan
    empty = pd.DataFrame(columns=["event"])
    b = ea.paired_block_bootstrap(empty, _prefixed_pairs(res, key), ref, block="event", n_boot=N_BOOT, ci=CI, seed=SEED)
    return b["ci_low"], b["ci_high"]


def paired_rows(results, cases, label):
    """instance - parent and quakescope2026 - parent over the pooled covered events of `cases`."""
    rows = []
    ref = pd.concat([_covered_reference(results[(k, PARENT)], k) for k in cases], ignore_index=True)
    if len(ref) == 0:
        return rows
    for w in WEIGHTS:
        if w == PARENT:
            continue
        a = pd.concat([_prefixed_pairs(results[(k, PARENT)], k) for k in cases], ignore_index=True)
        b = pd.concat([_prefixed_pairs(results[(k, w)], k) for k in cases], ignore_index=True)
        boot = ea.paired_block_bootstrap(a, b, ref, block="event", n_boot=N_BOOT, ci=CI, seed=SEED)
        rows.append(dict(scope=label, cases=";".join(cases), comparison=f"{w} - {PARENT}", weight=w,
                         n_events=boot["n_events"], n_blocks=boot["n_blocks"], block=boot["block"],
                         recovery_parent=round(boot["recovery_a"], 4), recovery_weight=round(boot["recovery_b"], 4),
                         diff=round(boot["diff"], 4), ci_low=round(boot["ci_low"], 4), ci_high=round(boot["ci_high"], 4),
                         excludes_zero=boot["excludes_zero"], n_boot=boot["n_boot"], ci=boot["ci"], seed=boot["seed"]))
    return rows


def reference_station_support(key, scored_stations) -> pd.Series:
    """event -> number of scored stations with a reference_ok P pick (37A's 3-station rule, on the scored stations)."""
    ref = pd.read_parquet(reg_root(key) / "picks.parquet")
    ref = ref[ref["reference_ok"] & (ref["phase"] == "P") & ref["station"].isin(set(scored_stations))]
    return ref.groupby("event")["station"].nunique()


def nearest_catalogue(unmatched: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """For each unmatched predicted event: nearest catalogue event in time and its distance (review aid)."""
    return _nearest(unmatched, "time", catalog, "origin", "event", "nearest_event")


def nearest_predicted(unmatched_ref: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """For each unmatched covered catalogue event: nearest predicted event in time and its distance."""
    return _nearest(unmatched_ref, "origin", events, "time", "event_idx", "nearest_event_idx")


def _nearest(left, left_time, right, right_time, right_id, id_name):
    if len(left) == 0 or len(right) == 0:
        return left.assign(**{id_name: None, "nearest_dt_s": np.nan, "nearest_dist_km": np.nan})
    rt = ea.cs._to_ns(right[right_time]).astype(np.int64)
    lt = ea.cs._to_ns(left[left_time]).astype(np.int64)
    dt = (lt[:, None] - rt[None, :]) / 1e9
    j = np.abs(dt).argmin(axis=1)
    dist = ea.haversine_km(left["lat"].to_numpy(), left["lon"].to_numpy(),
                           right["lat"].to_numpy()[j], right["lon"].to_numpy()[j])
    return left.assign(**{id_name: right[right_id].to_numpy()[j], "nearest_dt_s": dt[np.arange(len(j)), j].round(2),
                          "nearest_dist_km": dist.round(1)})


def proximity_class(dt_s, dist_km, tol_time_s=ea.MATCH_TOL_TIME_S, tol_km=ea.MATCH_TOL_KM):
    dt = abs(dt_s)
    if dt <= tol_time_s and dist_km <= tol_km:
        return "within tolerances"
    if dt <= tol_time_s:
        return f"|dt|<={tol_time_s:g}s, dist>{tol_km:g}km"
    if dt <= 30:
        return f"{tol_time_s:g}<|dt|<=30s"
    if dt <= 120:
        return "30<|dt|<=120s"
    return "|dt|>120s"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(REPO_ROOT / "data" / "evaluation" / "scores"))
    ap.add_argument("--events", default=str(REPO_ROOT / "data" / "evaluation" / "events"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--keys", nargs="+", default=KEYS)
    ap.add_argument("--associator", default="pyocto", choices=sorted(ea.BACKENDS))
    a = ap.parse_args(argv)
    policy.authorize_scoring(a.keys)
    scores, events_root, out = Path(a.scores), Path(a.events), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ops = operating_points()
    ops = ops[ops["key"].isin(a.keys)]
    ops.to_csv(out / "operating_points.csv", index=False)
    evaluability = pd.read_csv(EVALUABILITY).set_index("key")

    runs, results = [], {}
    for key in a.keys:
        store = find_pick_store(scores, key)
        for w in WEIGHTS:
            thr = ops[(ops["key"] == key) & (ops["weight"] == w)].set_index("phase")["threshold"]
            row, res = run_one(key, w, store, thr["P"], thr["S"], events_root / key / w, associator=a.associator)
            row["regime"] = reg.BY_KEY[key]["regime"]
            if res is not None:
                results[(key, w)] = res
            runs.append(row)
    runs = pd.DataFrame(runs)

    # reasons for empty runs, from the inputs (never a parameter change)
    for i, r in runs.iterrows():
        if r["status"] != "no_events":
            continue
        res = results[(r["key"], r["weight"])]
        m = res["meta"]
        cfg = m["config"]
        runs.at[i, "note"] = (f"PyOcto returned no event: {m['n_picks']} picks on {m['n_stations']} stations, "
                              f"config needs min_picks={cfg['min_picks']}, min_p_picks={cfg['min_p_picks']}, "
                              f"{cfg['n_p_and_s_picks']} stations with P and S")
    cols = ["key", "regime", "weight", "status", "access_id", "model_id", "config_region", "config_version", "config_sha256",
            "pyocto_version", "n_picks", "n_stations", "n_events", "n_reference", "n_matched", "associate_runtime_s",
            "wall_s", "git_commit", "note", "out_dir", "command"]
    runs = runs.reindex(columns=cols)
    runs.to_csv(out / "runs.csv", index=False)

    # per case and weight
    by_case, by_mag, by_hour, by_day, support, unmatched, unmatched_ref = [], [], [], [], [], [], []
    stations_by_key = {key: set(pd.read_parquet(find_pick_store(scores, key) / "picks.parquet", columns=["station"])["station"])
                       for key in a.keys}
    for (key, w), res in results.items():
        s = res["tables"]["summary"].iloc[0]
        lo, hi = recovery_interval(res, key)
        st = res["tables"]["station_support"]
        ev = res["events"]
        elig = evaluability.loc[key]
        cov = res["tables"]["coverage"]
        cov = cov[cov["in_window"]]
        n_p_sta = cov["event"].astype(str).map(reference_station_support(key, stations_by_key[key])).fillna(0).astype(int)
        sup3 = cov[n_p_sta >= 3]
        by_case.append(dict(
            key=key, regime=reg.BY_KEY[key]["regime"], weight=w,
            event_scoring_eligible_37A=bool(elig["network_event_scoring_eligible"]),
            n_catalogue=int(s["n_reference"]), n_catalogue_in_windows=int(s["n_reference_covered"]),
            n_matched=int(s["n_matched"]), recovery=round(float(s["recovery"]), 4) if pd.notna(s["recovery"]) else np.nan,
            recovery_ci_low=round(lo, 4) if pd.notna(lo) else np.nan, recovery_ci_high=round(hi, 4) if pd.notna(hi) else np.nan,
            n_catalogue_in_windows_3staP=int(len(sup3)), n_matched_3staP=int(sup3["matched"].sum()),
            recovery_3staP=(round(float(sup3["matched"].mean()), 4) if len(sup3) else np.nan),
            n_predicted=int(s["n_predicted"]), n_unmatched_predicted=int(s["n_unmatched_predicted"]),
            n_matched_outside_windows=int(s["n_matched_outside_windows"]),
            n_splits=res["diagnostics"].n_splits, n_merges=res["diagnostics"].n_merges,
            n_picks=int(s["n_picks_total"]), n_picks_associated=int(s["n_picks_assigned"]),
            n_picks_unassociated=int(s["n_picks_unassociated"]),
            n_stations_with_picks=int(res["meta"]["n_stations"]),
            median_stations_per_event=(float(ev["n_stations"].median()) if len(ev) else np.nan),
            median_picks_per_event=(float(ev["n_picks"].median()) if len(ev) else np.nan),
            time_origin_label=s["time_origin_label"], claim_supported=bool(s["claim_supported"])))
        by_mag.append(res["tables"]["by_magnitude"].assign(key=key, weight=w))
        by_hour.append(res["tables"]["by_hour"].assign(key=key, weight=w))
        bd = res["tables"]["by_day"]
        by_day.append(bd[bd["n_reference_covered"] > 0].assign(key=key, weight=w))
        # station support: predicted events by station count, matched or not
        if len(st):
            g = st.groupby("n_stations").agg(n_events=("event_idx", "size"), n_matched=("matched", "sum")).reset_index()
            support.append(g.assign(key=key, weight=w))
        um = res["unmatched_predicted"]
        if len(um):
            catalog = pd.read_parquet(reg_root(key) / "catalog.parquet")
            um = nearest_catalogue(um[["event_idx", "time", "lat", "lon", "depth_km", "n_picks", "n_p", "n_s",
                                       "n_stations", "n_p_and_s", "misfit_s"]].reset_index(drop=True), catalog)
            um["nearest_class"] = [proximity_class(a, b) for a, b in zip(um["nearest_dt_s"], um["nearest_dist_km"])]
            unmatched.append(um.assign(key=key, weight=w))
        ur = res["unmatched_reference"]
        cov = res["tables"]["coverage"]
        ur = ur[ur["event"].astype(str).isin(set(cov.loc[cov["in_window"], "event"].astype(str)))]
        if len(ur):
            ur = nearest_predicted(ur[["event", "origin", "lat", "lon", "depth_km", "mag"]].reset_index(drop=True), ev)
            ur["nearest_class"] = [proximity_class(a, b) for a, b in zip(ur["nearest_dt_s"], ur["nearest_dist_km"])]
            unmatched_ref.append(ur.assign(key=key, weight=w))

    front = ["key", "weight"]
    pd.DataFrame(by_case).to_csv(out / "recovery_by_case.csv", index=False)
    mag = pd.concat(by_mag, ignore_index=True)
    mag[front + [c for c in mag.columns if c not in front]].to_csv(out / "recovery_by_magnitude.csv", index=False)
    hour = pd.concat(by_hour, ignore_index=True)
    hour[front + [c for c in hour.columns if c not in front]].to_csv(out / "recovery_by_hour.csv", index=False)
    day = pd.concat(by_day, ignore_index=True)
    day[front + [c for c in day.columns if c not in front]].to_csv(out / "recovery_by_day.csv", index=False)
    sup = pd.concat(support, ignore_index=True) if support else pd.DataFrame(columns=front + ["n_stations", "n_events", "n_matched"])
    sup[front + [c for c in sup.columns if c not in front]].to_csv(out / "station_support.csv", index=False)
    um = pd.concat(unmatched, ignore_index=True) if unmatched else pd.DataFrame(columns=front)
    um[front + [c for c in um.columns if c not in front]].to_csv(out / "unmatched_predicted.csv", index=False)
    ur = pd.concat(unmatched_ref, ignore_index=True) if unmatched_ref else pd.DataFrame(columns=front)
    ur[front + [c for c in ur.columns if c not in front]].to_csv(out / "unmatched_reference.csv", index=False)
    # proximity summaries: how far the nearest counterpart is, per case and weight
    prox = []
    for name, frame in (("unmatched_predicted", um), ("unmatched_reference_covered", ur)):
        if len(frame):
            g = frame.groupby(["key", "weight", "nearest_class"]).size().rename("n").reset_index()
            prox.append(g.assign(table=name))
    prox = pd.concat(prox, ignore_index=True) if prox else pd.DataFrame(columns=["table"] + front + ["nearest_class", "n"])
    prox[["table"] + front + ["nearest_class", "n"]].to_csv(out / "proximity_summary.csv", index=False)

    # paired bootstrap: per case, pooled over the certified mainshock cases, pooled over every 37A event-eligible case
    rows = []
    complete = {k for k in a.keys if all((k, w) in results for w in WEIGHTS)}
    for key in a.keys:
        if key in complete:
            rows += paired_rows(results, [key], "case")
    mainshock = [k for k in a.keys if reg.BY_KEY[k]["regime"] == "msas" and reg.BY_KEY[k].get("mainshock")
                 and bool(evaluability.loc[k, "network_event_scoring_eligible"]) and k in complete]
    if mainshock:
        rows += paired_rows(results, mainshock, "pooled_mainshock_37A_eligible")
    eligible = [k for k in a.keys if bool(evaluability.loc[k, "network_event_scoring_eligible"]) and k in complete]
    if eligible:
        rows += paired_rows(results, eligible, "pooled_all_37A_eligible")
    pd.DataFrame(rows).to_csv(out / "paired_bootstrap.csv", index=False)

    print("\n== runs"); print(runs[["key", "weight", "status", "n_picks", "n_stations", "n_events", "n_matched",
                                  "associate_runtime_s", "wall_s"]].to_string(index=False))
    print("\n== recovery by case"); print(pd.DataFrame(by_case).to_string(index=False))
    print("\n== paired bootstrap"); print(pd.DataFrame(rows).to_string(index=False))
    return runs


def reg_root(key) -> Path:
    return REPO_ROOT / "data" / "heldout_testset" / key


if __name__ == "__main__":
    main()
