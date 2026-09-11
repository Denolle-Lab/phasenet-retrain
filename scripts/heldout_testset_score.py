#!/usr/bin/env python3
"""
heldout_testset_score.py

Scores PhaseNet weights on a sequence directory built by
scripts/build_heldout_testset.py, pick-level only (issue #35, checkpoint 35A;
event association is #36). Suite roles and access records of 44A
(scripts/evaluation_policy.py) guard every entry point.

    python scripts/heldout_testset_score.py --sequence adriatic_2022 --weights jma_wc instance
    python scripts/heldout_testset_score.py --all --weights jma_wc instance --thresholds 0.1 0.2 0.3 0.5
    python scripts/heldout_testset_score.py --sequence samos_2020 --weights jma_wc --annotations-root data/evaluation/annotations --out-dir data/evaluation/scores

Per model and station the continuous P/S probabilities are computed once
(`annotate_fn`, default `model.annotate(stream)`), stored raw by
scripts/continuous_scoring.AnnotationStore, and picks are extracted at each
threshold with the production trigger rule. References are deduplicated by
event identity, matched one-to-one with maximum cardinality, and every
emitted pick is persisted with its matched reference event so #36 can
associate exactly what was emitted. Model failures go to a failure table and
remove that (window, station) pair from every compared model.

Needs seisbench and torch only for the default adapter and the model
fingerprint; the engine and this driver are testable with synthetic
annotations.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import heldout_testset_registry as reg  # noqa: E402
import evaluation_policy as policy  # noqa: E402
import continuous_scoring as cs  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "heldout_testset"
ANNOTATIONS_ROOT = REPO_ROOT / "data" / "evaluation" / "annotations"
SCORES_ROOT = REPO_ROOT / "data" / "evaluation" / "scores"
REPORT_THRESHOLD = 0.3
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MATCH_TOL = reg.MATCH_TOL_S
REFERENCE_TIERS = ("manual",)
BUDGET_TOLERANCE = 0.10
PICK_STORE_COLUMNS = ["pick_id", "model_id", "model", "threshold", "station", "phase", "time", "score",
                      "matched_event", "ref_id", "residual", "window_id", "access_id", "key"]


@dataclass
class ScoreResult:
    key: str = ""
    access_id: str = ""
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)       # per-window (scope="window") and aggregate rows
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)      # the pick store
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)    # pick_id -> reference assignments
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)   # (window, station) outside common support
    budget: pd.DataFrame = field(default_factory=pd.DataFrame)
    models: dict = field(default_factory=dict)                     # name -> model_id
    out_dir: Path = None

    def write(self, out_dir) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("rows", "picks", "matches", "failures", "excluded", "budget"):
            frame = getattr(self, name)
            if name == "excluded" and len(frame):
                frame = frame.assign(missing_models=frame["missing_models"].map(lambda m: ",".join(m)))
            frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        pd.DataFrame([dict(model=k, model_id=v) for k, v in self.models.items()]).to_csv(out_dir / "models.csv", index=False)
        self.out_dir = out_dir
        return out_dir


# ── loading a built sequence ─────────────────────────────────────────────────

def window_id_for(t0, t1) -> str:
    """The waveform-file tag of a window, <t0>_<t1> as %Y%m%dT%H%M%S."""
    return f"{t0.strftime('%Y%m%dT%H%M%S')}_{t1.strftime('%Y%m%dT%H%M%S')}"


def load_sequence(key: str):
    """Load for reference QA, logged separately from model scoring."""
    policy.record_access(key, "reference_qa", data_root=OUT_ROOT)
    return _load_sequence(key)


def _load_sequence(key: str, tiers=REFERENCE_TIERS):
    """(windows, picks): windows are dicts with window_id, t0, t1 (UTCDateTime),
    streams {station: Stream} (one band per station, by registry preference)
    and the reference frame of continuous_scoring.reference_picks."""
    from obspy import UTCDateTime, read
    d = OUT_ROOT / key
    picks = pd.read_parquet(d / "picks.parquet")
    wins = [(UTCDateTime(r.t0), UTCDateTime(r.t1)) for r in pd.read_csv(d / "windows.csv").itertuples()]
    pref = {b: i for i, b in enumerate(reg.BY_KEY[key].get("channel_pref", reg.CHANNEL_PREF))}
    found = {}
    for f in sorted((d / "waveforms").glob("*.mseed")):
        sta, band, t0s, t1s = f.stem.split("__")
        found.setdefault((f"{t0s}_{t1s}", sta), []).append((pref.get(band, len(pref)), band, f))
    out = []
    for (t0, t1) in wins:
        wid = window_id_for(t0, t1)
        streams = {sta: read(str(sorted(found[(w, sta)])[0][2])) for (w, sta) in found if w == wid}
        ref = cs.reference_picks(picks, list(streams), t0, t1, tiers=tiers)
        out.append(dict(window_id=wid, t0=t0, t1=t1, streams=streams, reference=ref))
    return out, picks


# ── model adapter ────────────────────────────────────────────────────────────

def default_annotate(model, stream):
    """SeisBench WaveformModel.annotate: a Stream of *_P, *_S (and *_N) probability traces."""
    return model.annotate(stream)


# ── scoring ──────────────────────────────────────────────────────────────────

def score(key: str, models: dict, annotate_fn=None, thresholds=THRESHOLDS, annotations_root=None,
          out_dir=None, tiers=REFERENCE_TIERS, budget_reference=None, budget_threshold=REPORT_THRESHOLD,
          budget_tolerance=BUDGET_TOLERANCE) -> ScoreResult:
    """Annotate once per model and station, store, extract at each threshold, match, persist.

    `models` maps a display name to a model object; `annotate_fn(model, stream)`
    must return an obspy Stream with traces named *_P and *_S (default:
    model.annotate). The model id is the 44A state hash. Stored annotations
    are reused, so a rerun with more thresholds does no inference.
    """
    policy.authorize_scoring([key])
    thresholds = cs.dedup_thresholds(thresholds)
    fingerprints = {name: policy.model_fingerprint(model) for name, model in models.items()}
    model_ids = {name: fp["state_sha256"] for name, fp in fingerprints.items()}
    if len(set(model_ids.values())) != len(model_ids):
        raise ValueError(f"Two weight names share one state hash: {model_ids}")
    annotations_root = ANNOTATIONS_ROOT if annotations_root is None else Path(annotations_root)
    access_id = policy.record_access(
        key, "model_scoring", data_root=OUT_ROOT, models=fingerprints,
        settings=dict(thresholds=thresholds, report_threshold=REPORT_THRESHOLD, match_tol_s=MATCH_TOL,
                      reference_tiers=list(tiers), annotations_root=str(annotations_root),
                      budget_reference=budget_reference, budget_threshold=budget_threshold,
                      budget_tolerance=budget_tolerance),
    )
    windows, _ = _load_sequence(key, tiers=tiers)
    annotate_fn = default_annotate if annotate_fn is None else annotate_fn
    store = cs.AnnotationStore(annotations_root)
    label = reg.BY_KEY[key]["label"]
    failures = []

    # 1. annotate once per (window, station, model); failures are recorded, never skipped silently
    for w in windows:
        for name, model in models.items():
            mid = model_ids[name]
            for sta, st in w["streams"].items():
                if store.has(key, w["window_id"], sta, mid):
                    continue
                try:
                    ann = cs.annotation_from_stream(annotate_fn(model, st), st,
                                                    meta=dict(model=name, key=key, window_id=w["window_id"]))
                    store.put(key, w["window_id"], sta, mid, ann)
                except Exception as exc:  # noqa: BLE001
                    failures.append(cs.failure_row(mid, w["window_id"], sta, "annotate", exc))
    failures = pd.DataFrame(failures, columns=cs.FAILURE_COLUMNS)

    # 2. common support: every compared model must have annotated the (window, station) pair
    wanted = {(w["window_id"], sta) for w in windows for sta in w["streams"]}
    annotated = {mid: store.pairs(key, mid) & wanted for mid in model_ids.values()}
    common, excluded = cs.common_support(annotated)

    # 3. extract at each threshold, match, score
    rows, matches, picks = [], [], []
    for w in windows:
        wid = w["window_id"]
        for name in models:
            mid = model_ids[name]
            support, parts = {}, []
            for sta in sorted(w["streams"]):
                if (wid, sta) not in common:
                    continue
                ann = store.get(key, wid, sta, mid)
                support[sta] = ann
                parts.append(cs.extract_all(ann, thresholds, station=sta, id_prefix=f"{wid}|{mid[:12]}|"))
            cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cs.PICK_COLUMNS)
            r, m = cs.score_window(w["reference"], cand, support, window_id=wid, model_id=mid,
                                   thresholds=thresholds, tol=MATCH_TOL)
            rows.append(r.assign(model=name)); matches.append(m)
            picks.append(cand.assign(model_id=mid, model=name, window_id=wid))
    rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cs.ROW_COLUMNS + ["model"])
    matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=cs.MATCH_COLUMNS)
    picks = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=cs.PICK_COLUMNS + ["model_id", "model", "window_id"])
    rows = rows.assign(scope="window", key=key, sequence=label, access_id=access_id, n_windows=1)

    # 4. aggregate over this sequence's windows, by explicit window ids
    window_ids = [w["window_id"] for w in windows]
    if window_ids and len(rows):
        agg = cs.aggregate(rows, window_ids, matches=matches).assign(scope="aggregate", sequence=label)
        agg["model"] = agg["model_id"].map({v: k for k, v in model_ids.items()})
        rows = pd.concat([rows, agg], ignore_index=True)

    # 5. pick store: every emitted pick with its matched reference event (null when unmatched)
    assign = matches[["pick_id", "event", "ref_id", "residual"]].rename(columns={"event": "matched_event"})
    picks = picks.merge(assign, on="pick_id", how="left").assign(access_id=access_id, key=key)
    picks = picks[PICK_STORE_COLUMNS + ["i_on", "i_off", "i_peak"]] if len(picks) else pd.DataFrame(columns=PICK_STORE_COLUMNS)
    if len(picks) and picks["pick_id"].duplicated().any():
        raise ValueError("Pick ids are not unique")

    # 6. matched budget on the aggregate rows: target = reference model's emitted count at budget_threshold
    budget = pd.DataFrame(columns=cs.BUDGET_COLUMNS)
    agg_rows = rows[rows["scope"] == "aggregate"] if len(rows) else rows
    if len(agg_rows):
        ref_name = next(iter(models)) if budget_reference is None else budget_reference
        budget = cs.matched_budget(agg_rows, reference_model=model_ids[ref_name], reference_threshold=budget_threshold,
                                   tolerance=budget_tolerance)
        budget["model"] = budget["model_id"].map({v: k for k, v in model_ids.items()})
        budget["budget_reference"] = ref_name

    result = ScoreResult(key=key, access_id=access_id, rows=rows, picks=picks, matches=matches, failures=failures,
                         excluded=excluded, budget=budget, models=model_ids)
    if out_dir is not None:
        result.write(Path(out_dir) / key / access_id)
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def format_windows(rows: pd.DataFrame) -> str:
    cols = ["window_id", "model", "phase", "threshold", "n_reference", "n_reference_uncovered", "matched", "emitted",
            "unmatched_candidates", "recall", "residual_mae"]
    sub = rows[rows["scope"] == "window"] if "scope" in rows else rows
    if not len(sub):
        return "(no window rows)"
    return sub[cols].sort_values(["window_id", "model", "phase", "threshold"]).round(3).to_string(index=False)


def format_budget(budget: pd.DataFrame) -> str:
    if not len(budget):
        return "(no matched-budget rows)"
    cols = ["phase", "model", "target_emitted", "threshold", "emitted", "recall", "within_tolerance", "reason"]
    return budget[cols].round(3).to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence", action="append", default=[])
    selection.add_argument("--all", action="store_true", help="Score built regression/dev sequences only")
    ap.add_argument("--weights", nargs="+", default=["jma_wc"], help="SeisBench PhaseNet weight names, or paths to converted .pt/.json pairs")
    ap.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS, help="Trigger thresholds; deduplicated and sorted")
    ap.add_argument("--annotations-root", default=str(ANNOTATIONS_ROOT), help="AnnotationStore root (raw P/S probabilities, reused on rerun)")
    ap.add_argument("--out-dir", default=str(SCORES_ROOT), help="Per-run parquet artifacts under <out-dir>/<key>/<access_id>/")
    ap.add_argument("--budget-reference", default=None, help="Weight name whose emitted count at --budget-threshold sets the budget (default: first --weights)")
    ap.add_argument("--budget-threshold", type=float, default=REPORT_THRESHOLD)
    ap.add_argument("--out", default=None, help="Optional CSV of all score rows (window and aggregate scopes)")
    a = ap.parse_args(argv)
    keys = [key for key in policy.routine_keys(reg.BY_KEY)
            if (OUT_ROOT / key / "manifest.json").exists()] if a.all else a.sequence
    try:
        policy.authorize_scoring(keys)
        if not keys:
            raise ValueError("No built regression/dev sequences available")
        thresholds = cs.dedup_thresholds(a.thresholds)
        if a.budget_reference is not None and a.budget_reference not in a.weights:
            raise ValueError(f"--budget-reference {a.budget_reference} is not among --weights")
    except (PermissionError, ValueError) as exc:
        ap.error(str(exc))
    # Authorize the entire selection before loading any weights or waveform data.
    import seisbench.models as sbm
    models = {}
    for w in a.weights:
        models[w] = sbm.PhaseNet.from_pretrained(w) if not Path(w).exists() else sbm.PhaseNet.load(Path(w))
    results = []
    for k in keys:
        print(f"== {k}")
        res = score(k, models, thresholds=thresholds, annotations_root=a.annotations_root, out_dir=a.out_dir,
                    budget_reference=a.budget_reference, budget_threshold=a.budget_threshold)
        results.append(res)
        print("-- per-window rows")
        print(format_windows(res.rows))
        print(f"-- matched budget (aggregate over {', '.join(sorted(set(res.rows.get('window_id', pd.Series(dtype=str)))))})")
        print(format_budget(res.budget))
        if len(res.failures):
            print("-- failures"); print(res.failures.to_string(index=False))
        if len(res.excluded):
            print("-- excluded from common support"); print(res.excluded.to_string(index=False))
        if res.out_dir is not None:
            print("wrote", res.out_dir)
    if a.out and results:
        pd.concat([r.rows for r in results], ignore_index=True).to_csv(a.out, index=False)
        print("wrote", a.out)
    return results


if __name__ == "__main__":
    main()
