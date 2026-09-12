# Operating-point calibration on independent station-days (#38A)

*2026-09-11, branch `issue/38a-threshold-calibration` from
`issue/44a-suite-policy` (PR #51). Checkpoint 38A of issue #38: the protocol
and its numbers, pre-registered before any candidate is scored. The numbers
are proposals until 38A is checked off; after that they are frozen and a
change is a new protocol version. Encoded in `scripts/calibration_protocol.py`
(`PROTOCOL_VERSION = "38a-v1"`). Nothing here has been run on data.*

## Why a protocol

Every threshold in the history was 0.3 for every weight. The 2026-09-07 audit
showed the fine-tunes' probabilities sit lower than the parent's (H1), so a
shared threshold compares different operating points. The scorer (35A) gives
recall at each attained threshold; this protocol decides which threshold each
weight deploys with, from data that no model was selected on, and reports the
false-alarm side the benchmark cannot see.

## Roles of station-days

A station-day is one station, one UTC day, one instrument epoch. Roles are
exclusive: `training`, `mining` (hard-negative or noise harvest), `development`,
`sealed`, `calibration`, `unassigned`. Calibration days are drawn from
`unassigned` station-days only, after the held-out places and years
(`scripts/heldout_sequences.py`), the development windows and the sealed
panel are removed, by a hash of the station-day key
(`calibration_protocol.select_calibration_days`): reproducible, and blind to
any model output. Achieved rates are then reported on *distinct* evaluation
days without retuning; a calibration day never appears in a reported rate.

## Strata

| Axis | Classes |
|---|---|
| Region | dense_local, sparse_regional, island_coastal, volcanic, polar |
| Instrument | broadband, short_period, strong_motion, geophone_lowcost |
| Season | DJF, MAM, JJA, SON |
| Condition | quiet (station's own 1–20 Hz power below its 30th percentile over the calibration year), disturbed (above the 70th); the middle band is not used |

Thresholds are set on quiet days; disturbed days are reported, not used for
setting, so that a storm season does not silence the picker for the year.

## Budgets and precision (proposed, to freeze)

| Quantity | Value |
|---|---|
| Nuisance-pick budget, P | 50 unmatched picks per station-day on quiet days |
| Nuisance-pick budget, S | 50 unmatched picks per station-day on quiet days |
| Sensitivity values reported | 20 and 100 |
| Interval | 95 % percentile bootstrap with stations as blocks (all days of a drawn station move together, because days of one station are not independent) |
| Precision target | interval half-width ≤ 20 % of the budget |
| Minimum exposure per stratum | 20 stations with ≥ 3 calibration days each |
| Reviewed sample | 200 unmatched picks per weight and region, reviewed by a person, to estimate the fraction that are real events absent from the catalogue |
| Fallback | a stratum below minimum exposure is pooled with the first neighbour region class (same instrument class, season and condition; neighbour order in `REGION_NEIGHBOURS`) whose days make the combination sufficient, and the table row carries `pooled_with` and `exposure_ok_pooled`; when no neighbour suffices the stratum publishes no threshold |

The unmatched-pick rate is emitted picks with no catalogue arrival within
0.5 s at that station, per station-day. It is reported as "unmatched-pick
rate", never as a false-positive rate; the reviewed sample gives the
interval on the fraction of unmatched picks that are missed events.

## Procedure

1. Build the availability table (station, day, epoch, region class,
   instrument class, season, power percentile, role) from the campaign's
   station inventory and the continuous archives (*server*).
2. Draw calibration days (`select_calibration_days`, fraction 0.15 of the
   unassigned quiet and disturbed days) and check exposure per stratum
   (`exposure_check`).
3. For each weight, annotate every calibration station-day once (the 35A
   annotation store), extract picks at every threshold of the sweep
   (0.05 to 0.95 in steps of 0.05) with the production trigger rule, match
   to the catalogue, and compute the unmatched rate per station-day.
4. Per stratum, phase and threshold: rate and station-block bootstrap interval
   (`block_bootstrap_rate`, `block="station"`). The operating threshold is the lowest threshold
   whose interval upper bound is within budget (`operating_threshold`); if the
   precision target fails (`precision_ok`), the stratum needs more exposure
   before a threshold is published.
5. Publish, per weight: the thresholds per stratum, the attained rates with
   intervals on calibration days, the achieved rates on distinct evaluation
   days, the reviewed-sample fraction, and both the total emitted workload
   and the nuisance rate (matching one does not imply matching the other).
6. Candidates are refit by the same procedure on the same calibration days;
   parent and candidate event comparisons (#36) share the associator.

## What 38A releases and what it does not

Released here: the protocol, its parameters, the deterministic selection and
the estimators, with fixtures. Not released: the availability table and the
baseline thresholds, which need the continuous archives and the 35B baseline
annotations (*server*); those complete 38A on the server and unblock 35C.

## Validation

```bash
python -m pytest tests/test_calibration_protocol.py -q   # 4 tests, pure pandas
```
