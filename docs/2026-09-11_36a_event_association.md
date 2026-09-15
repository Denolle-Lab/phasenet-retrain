# 36A: event association and catalogue-relative recovery engine

Branch `issue/36a-event-scorer`, stacked on `issue/35a-continuous-scorer`
(PR #68). Checkpoint 36A of issue #36. Nothing here runs model inference or
PyOcto: the engine is exercised on synthetic picks only, pyocto is not
installed on this machine, and no baseline number was produced. Development
event baselines are 36B.

## What is frozen

`scripts/event_association.py::AssociatorConfig` holds every associator
parameter as a JSON-serialisable dataclass. `configs/association/<region>.json`
stores it with a `version` and a `sha256` over the canonical JSON of every
field except `notes` and `versioned`. `AssociatorConfig.load` refuses a file
whose stored hash disagrees with its content, so a hand edit without a
rehash (`python scripts/event_association.py --rehash <file>`) cannot run.
The hash is written as `config_sha256` into every row of `events.parquet`,
`assignments.parquet` and `matches.parquet` and into `run.json`, together
with the 35A `access_id`, `model_id`, `threshold` and sequence `key` taken
from the pick store.

Field names follow PyOcto (Münchmeyer 2024, Seismica 3(1)) where a
counterpart exists: `time_tolerance_s` is the velocity model `tolerance` and
`pick_match_tolerance`, `spatial_tolerance_km` is `min_node_size`,
`location_tolerance_km` is `min_node_size_location`, `association_cutoff_km`
is `association_cutoff_distance`, `time_before_s` is `time_before`,
`min_picks`/`min_p_picks`/`min_s_picks` are `n_picks`/`n_p_picks`/`n_s_picks`,
and `n_p_and_s_picks` is passed only when `n_picks_p_and_s_policy` is
`require`. `velocity_model` is `homogeneous` (`p_velocity`, `s_velocity`,
PyOcto `VelocityModel0D`) or a name with `layers` (`depth_km`, `vp`, `vs`;
`VelocityModel1D.create_model`). `margin_km` pads the station extent for the
search area; `depth_range_km` is `zlim`.

Shipped defaults, one per regime. Every value is a proposal to be tuned on
development sequences in 36B and never on acceptance cases; none was fitted
to data here.

| region | vp | vs | time tol s | node km | location km | cutoff km | depth km | margin km | time_before s | min picks | min P | P+S stations | sha256[:12] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| msas | 6.0 | 3.4 | 1.5 | 10 | 2 | 250 | 0-60 | 50 | 300 | 6 | 3 | 2 | be32d8ffcaa3 |
| vt | 5.0 | 2.9 | 1.0 | 5 | 1 | 100 | 0-30 | 30 | 120 | 5 | 3 | 1 | 703aa7c593a1 |
| swarm | 5.8 | 3.3 | 0.8 | 5 | 0.5 | 100 | 0-25 | 30 | 120 | 5 | 3 | 2 | b9ed1db8b94c |

`min_s_picks` is 0 in all three. `--sequence <key>` picks the regime file
from `heldout_testset_registry.BY_KEY[key]["regime"]`.

## Associators

`associate(picks, stations, config, backend)` returns `(events, assignments)`
with columns `event_idx, time, lat, lon, depth_km, n_picks, n_p, n_s,
n_stations, n_p_and_s, misfit_s, associator, config_sha256` and
`event_idx, pick_id, station, phase, time, score, residual`. Picks are 35A
pick-store rows (`pick_id, station, phase, time, score`; one `model_id` and
one `threshold`, selected by `select_picks`, which refuses an ambiguous
store). Stations are `stations.csv` rows (`station, lat, lon, elev_m`).

`PyOctoAssociator` maps the pick store to PyOcto's frame (`station, phase,
time` as POSIX seconds, `probability` from `score`, 1.0 when absent) and the
station table to `id, latitude, longitude, elevation`, builds the velocity
model and `OctoAssociator.from_area` from the config alone, and maps events
and assignments back to the schema above. `pyocto` is imported inside
`associate`; a missing install raises `ImportError` naming the package and
the paper. The adapter was written against the PyOcto 0.1 API and has not
been executed: the test monkeypatches the import away and checks the error
and the frame mapping only.

`SyntheticAssociator` is a test double. It back-projects every pick's
implied origin time onto a coarse x/y/depth grid (step
`spatial_tolerance_km`, station extent plus `margin_km`, straight rays,
homogeneous vp/vs), seeds an event at the cell with the most distinct
station-phase votes over two adjacent `time_tolerance_s` bins, keeps one pick
per station-phase, refines the location on a local sub-grid, releases picks
outside the tolerance, applies the config's minimum-pick rules, and removes
the event's picks from the pool. It is deterministic and grid-free in the
sense that it needs no precomputed travel-time table. It is not a
scientific associator and must not appear in a reported baseline; the CLI
default is `--associator pyocto`.

## Matching rule and tolerances

`match_events(predicted, reference, tol_time_s, tol_km, tol_depth_km)` is
the event-level version of `continuous_scoring.match_picks`. A
(reference, predicted) pair is feasible when |origin-time difference| <=
`tol_time_s` (integer nanoseconds, equality feasible), haversine epicentral
distance <= `tol_km`, and, when `tol_depth_km` is not None, |depth
difference| <= `tol_depth_km` or either depth is missing. The feasible cost
is `|dt|/tol_time_s + dist/tol_km (+ ddepth/tol_depth_km)`, each term at most
1; an infeasible pair costs `(min(n_r, n_p) + 1) * n_terms + 1`, above any
feasible total, so `scipy.optimize.linear_sum_assignment` maximises
cardinality first and minimises the summed normalised residual second.
`EventMatchResult` carries the pairs (`event, event_idx, dt_s, dist_km,
ddepth_km, cost`; `dt_s` is predicted minus reference), the unmatched
reference rows, the unmatched predicted rows and the feasibility matrix.

Proposed tolerances, the CLI defaults: origin time 5 s, epicentre 30 km,
depth 50 km (`--tol-depth-km 0` disables depth). They are wide on purpose so
that the catalogue's own location scatter does not fail a match; the
diagnostics below catch what wide tolerances let through. They are not
tuned and should be revisited in 36B per regime.

## Diagnostics

`split_merge_diagnostics(match, assignments, overlap=0.5)`:

- split: a reference event with two or more predicted events that are
  within the matching tolerance of it (`rule=tolerance`) or whose assigned
  picks carry that reference as `matched_event` (the 35A pick-level match)
  for at least 50 % of the picks (`rule=pick_overlap`; both rules can fire).
- merge: a predicted event within tolerance of two or more reference
  events; `reference_events_in_picks` lists how its picks split between
  reference events when assignments are given.

Counts go to `run.json` and `diagnostics.json`, lists to `splits.parquet`
and `merges.parquet`.

## Tables and the coverage rule

`recovery_tables(matches, reference, predicted, mainshock_time, windows,
magnitude_bins, hour_bins, day_bins, assignments, n_picks_total)` writes
six frames to `<out-dir>/tables/*.csv`:

| table | rows | content |
|---|---|---|
| `by_magnitude` | bins (-inf,1), [1,2), [2,3), [3,4), [4,5), >=5, unknown | reference, covered, outside windows, matched, recovery |
| `by_hour` | [0,1), [1,6), [6,24), [24,48), >=48 h after `mainshock_time` | as above plus `coverage_fraction` |
| `by_day` | one row per day from the time origin to the last event or window | as above |
| `station_support` | one row per predicted event | picks, P, S, stations, stations with both phases, matched reference |
| `coverage` | one row per reference event | origin, hours after, in_window, matched |
| `summary` | one row | totals, `n_pairs`, unassociated picks, windows, time origin |

Denominators are reference events whose origin lies inside a window
(`windows.csv` `t0`..`t1`); events outside every window are counted in
`n_reference_outside_windows` and never enter a recovery fraction. Matched
events outside the windows are reported in `n_matched_outside_windows`.

`claim_supported` is False on every row unless windows were given, the row
has covered reference events and, for hour and day rows, the union of
windows covers at least 95 % (`MIN_COVERAGE`) of the bin's span measured
from the time origin (`mainshock_time`, else the first window start, else
the first reference origin; `time_origin_label` says which). On the
current held-out windows (mainshock + 600 s, 120 or 180 min) the [0,1) h
row has `coverage_fraction` 0.833 and the [1,6) h row 0.233 (Norcia, 120
min) or 0.433 (Samos, 180 min), computed from the committed `windows.csv`
of those two sequences with `recovery_tables` on an empty prediction, so no
first-48-hour claim can be made from them; that is the audit's requirement
(`2026-09-10_picker_and_issue_roadmap_audit.md`, #36 row) made mechanical.

`paired_block_bootstrap(matches_a, matches_b, reference, block="event",
n_boot=2000, ci=0.95, seed=0, mainshock_time=None)` resamples reference
events (or whole hour or day blocks from the time origin, or any column of
`reference`) and returns the percentile interval of recovery(b) minus
recovery(a) with `excludes_zero`. The unit is the event, so P and S
observations and stations of one event are never separated. Pass the
covered subset of the catalogue (the `coverage` table) as `reference`.

## Command

```sh
python scripts/event_association.py --picks data/evaluation/scores/<key>/<access_id>/picks.parquet \
    --sequence samos_2020 --model-id <state_sha256> --threshold 0.3 --out-dir data/evaluation/events/<key>
```

`--sequence` calls `evaluation_policy.authorize_scoring([key])` before any
file is read (regression/dev only), supplies `stations.csv`,
`catalog.parquet`, `windows.csv`, the mainshock time and the regime config;
the `key` column of the pick store is authorised the same way and must agree
with `--sequence`. `--stations`, `--catalog`, `--windows`, `--config` and
`--mainshock` override. Outputs: `events.parquet`, `assignments.parquet`
(with the 35A `matched_event` per pick), `matches.parquet`,
`unmatched_predicted.parquet`, `unmatched_reference.parquet`,
`splits.parquet`, `merges.parquet`, `diagnostics.json`, `tables/*.csv`,
`run.json` (config, hashes of every input file, git commit, tolerances,
counts, `pyocto_version` when used).

## Verified here

`python -m pytest tests -q` on the conda base interpreter (Python 3.9.20,
numpy 2.0.2, pandas 2.3.2, scipy 1.13.1): 148 passed, of which 42 in
`tests/test_event_association.py`. They cover: no torch, seisbench or pyocto
import at module level; the hash changing with each of 18 parameters and
not with `notes`; save/load round trip, tamper refusal and `--rehash`; the
three shipped configs loading with matching hashes; the synthetic
associator recovering a 4-event catalogue with 20 noise picks, with two
stations removed, refusing an event below `min_picks`, and enforcing or
ignoring the P+S station rule; the PyOcto adapter's error without pyocto
and its frame mapping; the greedy counterexample at event scale; inclusive
tolerance boundaries in time, distance and depth; missing depth; unmatched
counts and residual sign; empty sides and duplicate ids; a duplicate
predicted event reported as a split by tolerance, a 200 km distant duplicate
reported by pick overlap only, one predicted event over two 4 s apart
references reported as a merge; recovery by magnitude; `claim_supported`
False for [1,6) h and [24,48) h with a 3 h window and True with a 48 h
window; no claim without windows; events outside windows leaving the
denominator; day bins; station support and unassociated pick counts; the
bootstrap interval containing zero for identical inputs, excluding zero for
8/40 versus 38/40, hour blocks, determinism by seed; `run` writing every
artifact with the config hash and access id; `select_picks` refusals; the
CLI end to end and its refusal of `la_palma_2021` (acceptance role).

One CLI run outside the tests: `--sequence samos_2020` with the real
`stations.csv` (6 fetched stations), `catalog.parquet` (16 events) and
`windows.csv`, on a pick store synthesised from that catalogue with the
`msas` velocities plus 40 noise picks, `--associator synthetic`, written to
the session scratchpad. It produced 16 events, 16 matches, 0 splits, 0
merges, 40 unassociated picks, and `claim_supported` False on every hour
row (coverage 0.833 and 0.433). This checks file formats and the policy
guard only; the picks are not model output and the numbers mean nothing.

## Remaining for 36B

- Install pyocto and execute `PyOctoAssociator` once on the synthetic
  fixture; the adapter's API calls (`VelocityModel0D`, `VelocityModel1D`,
  `OctoAssociator.from_area`, `transform_stations`, `associate`,
  `transform_events`) were written from the 0.1 documentation, not run.
- Replace the homogeneous velocities with versioned regional 1-D models
  (`layers`) per sequence and bump `version`; tune tolerances and minimum
  picks on development sequences only (`samos_2020`, `adriatic_2022`,
  `etna_2022_2024`, `corinth_thiva_2020`, if 37A certifies them).
- Run on 35C calibrated picks: Kaikōura as the known regression, Etna and
  Corinth-Thiva as development; choose substitutes before scoring if 37A
  does not certify them.
- Review a sample of unmatched predicted events independently before any is
  called false; catalogue absence is not ground truth. Report catalogue
  completeness (magnitude of completeness per window) next to the magnitude
  table.
- Time coverage: the present windows cannot support first-48-hour or
  migration claims; 37A has to add the excluded first ten minutes and later
  hour/day blocks before those rows can turn `claim_supported`.
- The `station_support` table reports the associated picks per event; valid
  exposure per station (the 35A valid mask) is not yet joined to it.
