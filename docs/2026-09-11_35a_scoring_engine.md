# 35A: continuous pick scoring engine

Branch `issue/35a-continuous-scorer`, stacked on `issue/44a-suite-policy`
(PR #51). Checkpoint 35A of issue #35. Nothing here runs model inference:
the engine is exercised on synthetic annotations only, and no baseline
numbers were produced. Baseline artifacts are 35B, calibrated comparison
is 35C.

## What changed and why

The scorer in `scripts/heldout_testset_score.py` reproduced the QuakeScope
notebooks, and the 2026-09-10 audit (findings 2 and 3 of
[`2026-09-10_picker_and_issue_roadmap_audit.md`](2026-09-10_picker_and_issue_roadmap_audit.md))
showed that the notebook logic is not a measurement of production picking.
The table lists each defect and where the replacement lives.

| Old behaviour (44A branch, `heldout_testset_score.py`) | Consequence | Replacement |
|---|---|---|
| one `classify()` at 0.02, then score filtering per threshold (`:37`, `:133`, `:140-146`) | a low threshold joins two peaks into one trigger; filtering keeps one pick where direct extraction gives two | `continuous_scoring.AnnotationStore` keeps the raw probabilities; `extract_picks` runs the production trigger at each threshold |
| `reference_from` collapses picks within 0.5 s regardless of event (`:43-56`) | close aftershock arrivals disappear from the denominator | `reference_picks` deduplicates by (event, station, phase) identity only |
| nearest-first greedy `match` (`:64-78`) | undercounts feasible one-to-one matches | `match_picks`: maximum cardinality, then minimum total residual, by linear assignment |
| `matched_budget` interpolates across windows and is never called by `main` (`:81-97`) | no attained operating point, no budget table | `aggregate` over explicit window ids, `operating_points`, `matched_budget` on attained points; `main` prints the table |
| 0.3 appended to the sweep twice (`:37`, `:140`, `:152`) | duplicate rows | `dedup_thresholds` |
| inference exceptions printed and skipped (`:134-135`) | a failing model loses only its hard stations | failure table plus a common-support gate over every compared model |
| picks discarded after aggregation | #36 cannot associate what was emitted | pick store parquet with matched reference event per pick |

`scripts/continuous_scoring.py` holds the engine (numpy, pandas, scipy,
obspy; no torch or seisbench import, checked by a test).
`scripts/heldout_testset_score.py` is the driver: it keeps the 44A guards
(`policy.authorize_scoring`, `policy.record_access`,
`policy.model_fingerprint`) in the same order as before and adds the
annotate-store-extract-match-persist pipeline behind them.

## Extraction rule

`extract_picks(prob, start_time, rate, threshold, valid_mask, phase, station)`
reproduces `seisbench.models.base.WaveformModel.picks_from_annotations`
of the installed SeisBench (`base.py` lines 2494-2531):
`obspy.signal.trigger.trigger_onset(data, threshold, threshold / 2)`
(line 2511), one pick per trigger at `s0 + argmax(data[s0:s1+1])` with
value `max(data[s0:s1+1])` (lines 2517-2519). obspy's trigger uses strict
`>` on both thresholds; nothing is changed. A pick whose peak sample falls
outside the valid mask is dropped, since a gap or padding cannot yield a
production pick. `tests/test_continuous_scoring.py::test_extract_matches_seisbench_rule_on_random_traces`
compares the output with a verbatim copy of those lines on random traces at
five thresholds; `test_threshold_induced_peak_splitting` is the audit's
0.8/0.7/0.03 case (one pick at 0.02 then filtered, two at 0.3).

## References

`reference_picks(picks, stations, t0, t1, tiers=("manual",), require_reference_ok=True)`
keeps rows on the fetched stations inside the window whose `mode` is in
`tiers` and whose `reference_ok` is set, then keeps one row per
(event, station, phase): highest `time_weight`, then earliest time. The
`tier` column carries the provenance mode so the reviewed-manual and
unknown-mode (ISC) tiers can be scored separately; the default tier is
manual only. Rows with no event id are kept as they are. Two events 0.3 s
apart on one station are two references.

## Matching objective

`match_picks(reference_times, candidate_times, tol)` builds the cost
matrix |candidate - reference| for pairs within `tol` and a constant
`(min(n_r, n_c) + 1) * tol + 1` otherwise, solves it with
`scipy.optimize.linear_sum_assignment`, and discards assigned infeasible
pairs. The constant exceeds any sum of feasible costs, so cardinality is
maximised first and total |residual| second. Times are converted to
integer nanoseconds, so |d| == tol is feasible exactly and no float slack
is needed. Residuals are candidate minus reference in seconds. The tests
include the audit's greedy counterexample (references [0.0, 0.6],
candidates [-0.4, 0.1], tol 0.5: two matches) and 40 random small cases
checked against exhaustive enumeration.

Matching is per (window, station, phase). Tolerance stays
`heldout_testset_registry.MATCH_TOL_S = 0.5`.

## Coverage and failure policy

- Every `(window, station, model)` annotation attempt that raises is a row
  of the failure table (`model_id, window_id, station, stage, exception,
  message`). Nothing is printed and skipped.
- Comparison rows are computed only on `(window, station)` pairs that every
  compared model annotated (`common_support`). Excluded pairs are listed
  with the models that lack them. A failing model therefore removes the
  pair from every model; it cannot improve its own recall by failing on
  hard stations, and denominators are identical across models.
- A reference on a station outside the common support, or at a time where
  the valid mask is False (nearest sample), leaves the denominator and is
  counted in `n_reference_uncovered`. The valid mask comes from the input
  stream on the annotation grid: masked samples, samples not covered by
  every component, and exact-zero runs of at least 1 s (the builder
  zero-fills gaps) are invalid.
- Zero covered references give recall NaN; emitted picks are still
  counted.
- Emitted counts every pick extracted from the window's annotation on the
  common-support stations, at that threshold; nothing is re-thresholded.

## Aggregation and operating points

`score_window` writes one row per (model_id, phase, threshold, window_id):
`n_reference, n_reference_uncovered, matched, emitted,
unmatched_candidates, recall, residual_mae, residual_median, residual_p90`.
`aggregate(rows, window_ids)` sums the counts over an explicit list of
window ids, refuses windows of two sequences or ids not present, and
recomputes recall; MAE is recomputed from per-window MAE and matched
counts, median and p90 from the match table. Per-window rows are written
unchanged next to the aggregate rows (`scope` column).

`operating_points(rows)` lists attained (threshold, emitted, recall) per
model and phase for one scope and raises on duplicate rows.
`matched_budget(rows, target_emitted=None, reference_model=None,
reference_threshold=None, tolerance=0.10)` picks, per model and phase, the
attained threshold whose emitted count is closest to the target; a model
with no point within 10 % of the target gets `None` and a reason. No
interpolation anywhere. The driver's budget rule is the emitted count of
the first (or `--budget-reference`) weight at `--budget-threshold`
(default 0.3) on the aggregate rows.

## Artifacts

Annotation store (`--annotations-root`, default
`data/evaluation/annotations`, gitignored):
`<key>/<window_id>/<model_id>/<station>.npz` with `P`, `S` (float32),
`valid` (bool), `start_time` (ISO UTC), `rate`, `meta`; `index.parquet`
with `key, window_id, station, model_id, start_time, rate, n_samples,
n_valid, path`. `model_id` is the 44A `state_sha256`. A rerun with more
thresholds reuses stored annotations and does no inference.

Per run (`--out-dir`, default `data/evaluation/scores`, under
`<key>/<access_id>/`): `rows.parquet`, `picks.parquet`, `matches.parquet`,
`failures.parquet`, `excluded.parquet`, `budget.parquet`, `models.csv`.

Pick store (`picks.parquet`), one row per emitted pick per threshold:

| column | content |
|---|---|
| `pick_id` | `<window_id>\|<model_id[:12]>\|<station>\|<phase>\|<threshold>\|<peak sample>`, unique per run |
| `model_id`, `model` | state hash and weight name |
| `threshold` | extraction threshold |
| `station`, `phase` | `NET.STA`, P or S |
| `time` | peak time, UTC |
| `score` | peak probability |
| `matched_event` | reference event id, null when unmatched |
| `ref_id`, `residual` | `event\|station\|phase` of the matched reference and candidate minus reference (s) |
| `window_id` | `<t0>_<t1>` as `%Y%m%dT%H%M%S`, the waveform-file tag |
| `access_id`, `key` | 44A access record and sequence key |
| `i_on`, `i_off`, `i_peak` | trigger on, off and peak sample indices on the annotation grid |

## Commands

```sh
python -m pytest tests -q
python scripts/heldout_testset_score.py --sequence samos_2020 --weights jma_wc instance
python scripts/heldout_testset_score.py --all --weights jma_wc --thresholds 0.05 0.1 0.2 0.3 0.5 \
    --annotations-root data/evaluation/annotations --out-dir data/evaluation/scores --out scores.csv
```

Protected and unknown selections still fail before any model or waveform
is loaded. `--all` still selects built regression/dev cases only.

## Verified here

`python -m pytest tests -q` on the conda base interpreter (Python 3.9,
numpy 2.0.2, pandas 2.3.2, scipy 1.13.1, obspy 1.4.2): 106 passed.
`tests/test_continuous_scoring.py` (63 tests) covers peak splitting, the
SeisBench rule on random traces, invalid-support drops, close events,
duplicate agency rows, tier filters, the greedy counterexample, brute-force
agreement, the exact tolerance boundary, gap coverage, zero-reference
windows, aggregation by window id and its refusals, attained operating
points and budgets, common support and failure rows, the store round trip,
the valid mask, and an end-to-end `score()` and `main()` on a synthetic
`samos_2020` directory with two synthetic models, one of which fails on one
station. The two legacy tests of the greedy matcher and proximity collapse
in `tests/test_heldout_testset.py` were removed with the functions.

No real sequence was scored: torch and seisbench are not installed on this
machine, and no annotation of a real model exists yet.

## Remaining for 35B and 35C

- 35B: run parent, v7, INSTANCE and any available v11 through this driver
  on regression/dev cases certified by 37A; record training-domain and
  parent-overlap provenance per baseline (Norcia versus INSTANCE); archive
  the old QuakeScope tables as superseded; ensemble members need aligned
  labels, grids and valid masks (`Annotation` carries all three; the
  combination rule is not written).
- 35C: P/S recall, residual distributions, matched workload and separately
  calibrated nuisance budgets with paired block uncertainty, after 38A
  supplies calibration station-days. `matched_budget` gives the attained
  points; the uncertainty and the nuisance-pick calibration are not in
  this checkpoint.
- The default valid mask infers gaps from zero runs because the builder
  discards gap information; 37A should persist gap masks so the inference
  can be dropped.
- The QuakeScope notebooks still carry the old logic; they are outside
  this repository.
