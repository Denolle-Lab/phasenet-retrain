# Stratified source census, checkpoint 39A of issue #39

Branch `issue/39a-source-census`, stacked on `issue/44a-suite-policy` (PR #51).
Script `scripts/source_census.py`; outputs under `data/census/`; tests
`tests/test_source_census.py`. Every number below is read from the committed
CSVs named in each section. The bulletin part was run from a laptop on
2026-09-12 (UTC) with the command in section 2; the SeisBench cache was not
reachable, so section 3 is what the committed records say and section 5 lists
what the server run has to add.

## 1. Method

**Bulletin operators.** The census treats an operator-year as the population
and samples days from it: `days_per_quarter` distinct days per quarter, a
quarter taken whole when it has no more days than that, otherwise its days
sorted by `sha256(operator|year|Qq|seed_tag|date)` and the first
`days_per_quarter` kept. The sample depends on nothing in the data, a rerun
with the same `--seed-tag` asks the same days, and a larger
`days_per_quarter` extends the smaller sample. (The 16 days of section 2 were
drawn at commit 7580636 by the earlier rule,
`sha256(operator|year|Qq|seed_tag|k) mod days-in-quarter` with repeats
rejected, which PR #75 replaced after review; seed tag `39a` maps to other
days under the current rule, and the days that ran are the `day` column of
the committed CSVs.) Each sampled day is queried along the path
`scripts/build_heldout_testset.py` already uses for that operator:

| Operator | Path (builder function) | Query | Evaluation mode exposed |
|---|---|---|---|
| INGV, GeoNet | `harvest_fdsn_per_event` | day catalogue, then one `eventid` request per event (INGV with `includearrivals=true`) | INGV: manual/automatic per pick; GeoNet: where set |
| NOA, ISC, franceseisme | `harvest_fdsn_region` | one day query with `includearrivals=true`, halved on HTTP 413 or "too much" down to 15 minutes, then recorded as a failure | NOA: manual; ISC: none (reviewed bulletin) |
| USGS ComCat | `harvest_usgs_phase_data` | day catalogue, the `phase-data` product's QuakeML per event | where set |

The rows per pick come from the builder's `_pick_rows` (the QuakeScope rule:
only picks referenced by an arrival of the preferred origin, phase P or S
after stripping I/E), the catalogue rows from `_catalog_rows`. Per sampled
day the script counts events, events with any arrival, P and S readings by
evaluation mode (manual, automatic, unknown), distinct stations, readings
per event, a magnitude histogram (bins <1, 1-2, 2-3, 3-4, >=4, unknown),
and the events and readings whose origin falls in a held-out window, a
place hold-out or a held-out year of `scripts/heldout_sequences.py`
(`window_hits`, `holdout_year_mask`, the same functions the manifest
builder uses). On the per-event paths a deterministic subset of at most
`--max-events-per-day` events (80 by default; the day's identifiers sorted
by `sha256(seed_tag|eventid)`) is fetched and the counts are scaled by
events/fetched; the `*_est` columns carry the scaled values and `scale`
records the factor.

Every request goes through one HTTP function with a 120 s timeout, three
tries with 5/15/30 s backoff, a `--max-minutes` wall-clock budget checked
before each request, and a raw cache under `data/census/raw/` (ignored by
git; one file per request URL, the first 12 hex digits of `sha256(url)` in the
file name and the full URL in the `.meta.json` sidecar; a rerun reuses a
cached body only when the sidecar records the same URL, query parameters
included, and re-counts its recorded cost; the files the section 2 run wrote
at commit 7580636 carry no URL hash and are not read). A day
that fails after the retries is a row with `status=failed` and the error
text; a day the budget did not reach is `status=not_attempted`. Neither
enters the estimates.

**Extrapolation.** For each operator-year, each metric's estimate is
`days_in_year x mean over ok days`, and the 95 % interval is the 2.5 and
97.5 percentiles of 2000 bootstrap resamples of the ok days (seeded from
the seed tag). One ok day gives an estimate and no interval. The `all` row
per operator sums the year estimates and takes the percentiles of the
summed bootstrap draws. With four days per year (one per quarter, this
demonstration) the intervals are wide, and the bootstrap reports that
width rather than hiding it; the default of two days per quarter is the
smallest sample a real census should use.

**SeisBench sources.** `python scripts/source_census.py seisbench` writes
`data/census/seisbench_sources.csv` from `DATASET_CONFIGS` in
`scripts/build_training_dataset.py` (parsed with `ast`, so the module's
`import seisbench` is never executed here), the 2026-04-06 metadata audit
(`notebooks/audit_results/summary_statistics.csv`), the P/S availability
count in `notebooks/benchmark_pool_summary.csv`, the on-disk row counts in
`data/README.md`, and the pick-status columns recorded when
`notebooks/step_1_claude.ipynb` read five rows of each metadata file on the
server. The columns `heldout_overlap_counts` and `pick_status_counts` are
the string `server`; `python scripts/source_census.py seisbench
--cache-root $SEISBENCH_CACHE_ROOT` loads each source the way the training
build does and writes `data/census/seisbench_server.csv` with the status
value counts per phase and the rows inside held-out windows and years.

## 2. Demonstration: INGV and NOA, 2018 and 2019, one day per quarter

Command (run 2026-09-12 13:09 to 13:22 UTC; 12.5 minutes of the 30-minute
budget; the NOA 2019 file was regenerated from the raw cache after a
note-format fix, same counts):

```sh
python scripts/source_census.py bulletin --operators INGV NOA --years 2018 2019 \
    --days-per-quarter 1 --max-minutes 30 --seed-tag 39a
```

All 16 sampled days returned; no day failed and none was skipped by the
budget. The one service incident was NOA on 2019-05-21, which answered the
full-day request with HTTP 413 and the two half-day requests normally
(`note` column: `413 split 00:00..24:00`). INGV was fetched at most 80
events per day: 2018-04-20 (134 events, scale 1.675) and 2018-08-22
(82 events, scale 1.025) are scaled, all other days complete.

### Operator-year estimates (bulletin_summary.csv)

| Operator | Year | Days ok/failed | Events/yr est. | Events with arrivals/yr | P manual/yr est. [95 %] | S manual/yr est. [95 %] | P auto/yr | Readings/event | Stations (sampled days) | Manual share | Held-out readings share | Queries | s/query | MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INGV | 2018 | 4/0 | 29,748 | 29,565 | 489,527 [290,175, 744,096] | 364,138 [212,140, 588,487] | 0 | 28.6 | 420 | 100.0 % | 2.4 % | 274 | 1.0 | 45.8 |
| INGV | 2019 | 4/0 | 17,338 | 15,330 | 281,415 [222,924, 326,858] | 203,670 [162,790, 244,550] | 0 | 32.6 | 455 | 100.0 % | 1.2 % | 194 | 0.9 | 27.6 |
| INGV | all | 8/0 | 47,085 | 44,895 | 770,942 [574,834, 1,023,237] | 567,808 [415,810, 792,598] | 0 | n/a | n/a | n/a | n/a | 468 | 0.9 | 73.4 |
| NOA | 2018 | 4/0 | 18,706 | 18,706 | 267,636 [175,930, 392,466] | 115,796 [80,118, 151,475] | 0 | 20.6 | 162 | 100.0 % | 4.7 % | 4 | 29.7 | 4.0 |
| NOA | 2019 | 4/0 | 21,626 | 21,626 | 391,554 [294,372, 497,404] | 201,298 [149,102, 255,135] | 0 | 27.6 | 209 | 100.0 % | 10.9 % | 6 | 27.9 | 6.8 |
| NOA | all | 8/0 | 40,332 | 40,332 | 659,190 [517,844, 822,080] | 317,094 [256,321, 379,237] | 0 | n/a | n/a | n/a | n/a | 10 | 28.6 | 10.8 |

### Magnitude mix of sampled events (share of catalogue events, bulletin_summary.csv)

| Operator | Year | M<1 | 1-2 | 2-3 | 3-4 | >=4 | unknown |
|---|---|---|---|---|---|---|---|
| INGV | 2018 | 26.4 | 64.1 | 7.7 | 0.9 | 0.9 | 0.0 |
| INGV | 2019 | 19.5 | 70.0 | 8.9 | 0.0 | 1.6 | 0.0 |
| NOA | 2018 | 0.0 | 42.0 | 43.4 | 9.3 | 0.5 | 4.9 |
| NOA | 2019 | 0.8 | 40.9 | 52.3 | 4.6 | 1.3 | 0.0 |

Columns: estimates are `days_in_year x day mean` over the ok days, the
bracket is the percentile bootstrap (2.5, 97.5) over the four days; "P
auto" is the automatic-mode count among the readings of the preferred
origin; "Manual share" is manual P and S over all readings; "Held-out
readings share" is the share of readings whose event lies in a held-out
window, place or year; "Stations" is the number of distinct station codes
over the year's sampled days. The `all` rows sum the year estimates.

### Sampled days (bulletin_<operator>_<year>.csv)

| Operator | Day | Status | Events | Fetched | With arrivals | P manual | S manual | P auto | S auto | P unknown | Stations | Held-out events | Held-out windows | Queries | s | MB | Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INGV | 2018-01-27 | ok | 69 | 69 | 69 | 1,200 | 786 | 0 | 0 | 0 | 275 | 3 | etna;maurienne | 70 | 69 | 12.1 |  |
| INGV | 2018-04-20 | ok | 134 | 80 | 80 | 1,388 | 1,127 | 0 | 0 | 0 | 265 | 6 | etna | 81 | 80 | 13.9 |  |
| INGV | 2018-08-22 | ok | 82 | 80 | 80 | 1,151 | 793 | 0 | 0 | 0 | 257 | 1 | etna | 81 | 78 | 13.3 |  |
| INGV | 2018-11-25 | ok | 41 | 41 | 39 | 660 | 504 | 0 | 0 | 0 | 217 | 2 | etna | 42 | 37 | 6.5 |  |
| INGV | 2019-01-01 | ok | 61 | 61 | 41 | 773 | 467 | 0 | 0 | 0 | 270 | 21 | etna | 62 | 45 | 7.5 |  |
| INGV | 2019-05-03 | ok | 47 | 47 | 46 | 908 | 702 | 0 | 0 | 0 | 306 | 1 | etna | 48 | 44 | 8.4 |  |
| INGV | 2019-07-27 | ok | 26 | 26 | 25 | 520 | 425 | 0 | 0 | 0 | 219 | 0 |  | 27 | 26 | 4.4 |  |
| INGV | 2019-11-02 | ok | 56 | 56 | 56 | 883 | 638 | 0 | 0 | 0 | 256 | 0 |  | 57 | 50 | 7.3 |  |
| NOA | 2018-01-28 | ok | 64 | 64 | 64 | 698 | 400 | 0 | 0 | 0 | 98 | 1 | corinth_thiva | 1 | 24 | 1.0 |  |
| NOA | 2018-04-08 | ok | 36 | 36 | 36 | 587 | 233 | 0 | 0 | 0 | 76 | 1 | corinth_thiva | 1 | 28 | 0.8 |  |
| NOA | 2018-08-13 | ok | 31 | 31 | 31 | 410 | 206 | 0 | 0 | 0 | 91 | 3 | corinth_thiva | 1 | 21 | 0.6 |  |
| NOA | 2018-12-29 | ok | 74 | 74 | 74 | 1,238 | 430 | 0 | 0 | 0 | 127 | 8 | corinth_thiva | 1 | 46 | 1.6 |  |
| NOA | 2019-02-05 | ok | 68 | 68 | 68 | 1,139 | 581 | 0 | 0 | 0 | 89 | 1 | corinth_thiva | 1 | 28 | 1.7 |  |
| NOA | 2019-05-21 | ok | 83 | 83 | 83 | 1,539 | 761 | 0 | 0 | 0 | 134 | 10 | corinth_thiva | 3 | 73 | 2.5 |  |
| NOA | 2019-08-18 | ok | 42 | 42 | 42 | 834 | 351 | 0 | 0 | 0 | 121 | 9 | corinth_thiva | 1 | 27 | 1.3 |  |
| NOA | 2019-10-24 | ok | 44 | 44 | 44 | 779 | 513 | 0 | 0 | 0 | 145 | 4 | corinth_thiva | 1 | 40 | 1.4 |  |

Columns as in `bulletin_<operator>_<year>.csv`: "Fetched" is the number of
events whose arrivals were requested (INGV per-event path, at most 80),
"Held-out events" counts events inside any hold-out, "Held-out windows"
names them, "s" and "MB" are the day's acquisition time and bytes.

What the numbers say, and what they cannot say yet:

- **Both operators expose manual P and S at the reading level.** Every
  reading returned by both services on all 16 days carries
  `evaluation_mode = manual` (P auto = 0, P unknown = 0 on every row).
  For INGV this is the mode of the picks referenced by the preferred
  origin's arrivals; the per-event QuakeML also carries automatic picks
  that no arrival references, and the census, like the held-out builder,
  does not count them. NOA returns manual picks only.
- **Volume.** INGV's preferred-origin readings extrapolate to 490 k manual
  P and 364 k manual S in 2018 (95 % interval 290 k to 744 k for P) and
  281 k P and 204 k S in 2019; NOA to 268 k P and 116 k S in 2018 and
  392 k P and 201 k S in 2019. Four days per year is a thin sample: the
  INGV 2018 interval is a factor 2.6 wide because 2018-04-20 (134 events)
  sits beside 2018-11-25 (41 events). The `all` rows put INGV at 771 k
  manual P and 568 k manual S over the two years and NOA at 659 k P and
  317 k S. These are readings, not traces; a trace is one station-event
  pair and carries P, S or both, so the station-event count lies between
  the P count and the P plus S count.
- **S completeness differs.** INGV's manual S per manual P is 0.73 in both
  years (`s_over_p_manual`), NOA's 0.43 in 2018 and 0.51 in 2019. NOA
  traces will be P-only about half the time, which is exactly the
  partial-label case #40's target schema has to carry as "S unknown", not
  "no S".
- **Magnitude range.** INGV's sampled events are 20 to 26 % M < 1 and 64 to
  70 % M 1 to 2; NOA's are 41 to 42 % M 1 to 2 and 43 to 52 % M 2 to 3
  with almost no M < 1 (0 and 0.8 %). The two operators supervise different parts of the
  low-SNR range; the low-SNR measurement itself (issue #39 acceptance)
  needs waveforms and is not in this census.
- **Held-out overlap is small but real and concentrated.** 1.2 to 2.4 % of
  INGV readings fall in the Etna place hold-out (one 2018-01-27 event
  also lies in the Maurienne place), 4.7 to 10.9 % of NOA readings in the
  Corinth-Thiva place hold-out (26 % of the readings on 2019-08-18). No
  sampled day is in a held-out year (2016, 2021 were not sampled). A
  pilot corpus from these operators loses that share to the place rules
  and nothing to the year rules for 2018-2019.
- **Cost.** INGV: 468 requests, 0.9 s per request, 73 MB for eight days
  (the per-event path costs one request per event). NOA: 10 requests,
  29 s per request, 11 MB for eight days (one day query, halved once).
  At these rates (INGV 54 s and 9 MB per day, NOA 36 s and 1.3 MB per
  day, `seconds_total` and `bytes_total` over eight days each) a
  two-days-per-quarter census of both operators over ten years (80 days
  each) is about 70 minutes of INGV requests and 50 minutes of NOA
  requests, inside one session. The raw cache for these 16 days is
  83 MB (`du -sh data/census/raw`).

Not measured here: readings per unique station-event pair across
operators (shared events between INGV and NOA would need a spatial join
of the two catalogues; the day samples of the two operators are different
days by construction, so no join is possible in this run), event
multiplicity within windows, waveform accessibility and response state
(a station-side census), and the licence and access terms of either
service, which were not checked in this run.

## 3. SeisBench sources (data/census/seisbench_sources.csv)

Traces, P and S counts are the committed audit numbers; `server` means the
count is not in any committed record and needs the cache. The status
column is what `notebooks/step_1_claude.ipynb` saw in five rows per file
on the server, so "NaN in the sampled rows" is a five-row observation, not
a count.

| Source | In pool | Cap | use_s | Traces | P | S | P and S | Native rate (Hz) | Status column | Values seen (5 rows) | Label-policy consequence | Held-out overlap documented |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geofon | yes | 150,000 | no | 275,274 | 274,474 | 2,648 | 2,583 | 20 dominant; 20 to 200 | yes: trace_P_status, trace_S_status, trace_pP_status | trace_P_status = manual | supervises P; S masked by the teleseismic P-only rule | none documented |
| stead | yes | 100,000 | yes | 1,265,657 | 1,030,231 | 1,030,231 | 1,030,231 | server | yes: trace_p_status, trace_s_status, trace_p_weight, trace_s_weight | NaN in the sampled rows (noise rows) | supervises P and S; needs review: status column present, values unverified | year 2016; Kaikoura 2016 and Norcia 2016 possible (global 2005-2018) |
| ceed | yes | 100,000 | yes | 5,009,718 | server | server | server | 100 | unknown: not recorded | metadata.csv absent when the notebook ran | supervises P and S; needs review: columns not recorded | Ridgecrest 2019 |
| instancecounts | yes | 100,000 | yes | 1,159,249 | 1,159,249 | 713,883 | 713,883 | server | no: trace_P_uncertainty_s, trace_S_uncertainty_s, path_weight_phase_location_P, path_weight_phase_location_S | no status column; uncertainties 0.1 s (P) 0.6 s (S), location weights | supervises P and S; provenance by uncertainty/weight columns only | Norcia 2016 sequence and year 2016 (Italy 2005-01 to 2020-01); Etna and Campi Flegrei place hold-outs |
| mlaapde | yes | 80,000 | yes | 510,196 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | none documented |
| ethz | yes | 60,000 | yes | 36,743 | 1,674 | 86 | 57 | 100, 120, 200, 250, 500 | yes: trace_P_status, trace_PmP_status, trace_SmS_status | NaN in the sampled rows | supervises P; S present on 86 traces only, unknown-S mask needed | none documented |
| crew | yes | 30,000 | yes | 1,599,323 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | Thessaly 2021 and year 2021 possible (global regional) |
| cwa | yes | 30,000 | yes | 346,959 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | none documented |
| iquique | yes | 13,400 | yes | 13,400 | 13,327 | 11,361 | 11,288 | 100 | no: none | no status column | supervises P and S; provenance by uncertainty/weight columns only | none documented |
| txed | yes | 40,000 | yes | 519,689 | 312,231 | 312,231 | 312,231 | server | no: trace_p_arrival_uncertainty_s, trace_s_arrival_uncertainty_s | no status column | supervises P and S; provenance by uncertainty/weight columns only | none documented |
| pnw | yes | 40,000 | yes | 183,909 | 183,909 | 183,909 | 183,909 | 100 | no: trace_P_arrival_uncertainty_s, trace_S_arrival_uncertainty_s, trace_P_onset | no status column; uncertainties 0.01-0.08 s | supervises P and S; provenance by uncertainty/weight columns only | Monroe 2019 |
| lendb | yes | 40,000 | no | 1,244,942 | 629,095 | 0 | 0 | server | yes: trace_p_status | trace_p_status = estimated | supervises P; S masked (use_s=False); needs review: P status 'estimated' | none documented |
| pisdl | yes | 10,000 | yes | 142,001 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | none documented |
| vcseis | yes | 30,000 | yes | 160,278 | 147,863 | 147,863 | server | 100 | unknown: not recorded | metadata.csv absent when the notebook ran | supervises P and S; needs review: columns not recorded | Hawaii is tier 2 (usable only if VCSEIS ends before 2022) |
| aq2009gm | yes | 60,000 | yes | 258,984 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | none documented |
| obst2024 | yes | 60,000 | yes | 60,394 | 35,394 | 35,394 | 35,394 | server | yes: trace_p_status, trace_s_status, trace_p_weight, trace_s_weight | NaN in the sampled rows | supervises P and S; needs review: status column present, values unverified | none documented |
| scedc | yes | 60,000 | yes | 8,035,833 | 7,501,488 | 4,317,447 | 3,783,102 | 40, 100 | yes: trace_p_status, trace_s_status, trace_p_weight, trace_s_weight | status NaN; weights 1.0/0.3/0.5 | supervises P and S; needs review: status column present, values unverified | Ridgecrest 2019 |
| meier2019jgr | yes | 150,000 | no | 1,060,433 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P; S masked (use_s=False) | none documented |
| ross2018gpd | yes | 200,000 | yes | 4,773,750 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | Ridgecrest 2019 |
| obs | yes | 100,000 | yes | 109,208 | server | server | server | server | unknown: not recorded | no record in the repository | supervises P and S; needs review: columns not recorded | none documented |
| neic | no |  |  | 1,354,789 | 1,025,000 | 329,789 | 0 | server | yes: trace_p_status, trace_s_status | trace_p_status = manual | not in the training pool (excluded: 0 % traces with both P and S) | none documented |
| pnw_accel | no |  |  | 6,107 | server | server | server | 100 | unknown: not recorded | no record in the repository | not in the training pool | none documented |

Reading the table:

- **Manual status is verified for two sources only**: GEOFON
  (`trace_P_status = manual`, P-only by the teleseismic rule) and NEIC
  (`trace_p_status = manual`, not in the pool). LenDB's P status is
  `estimated` (picks derived from the catalogue, not read on the trace),
  so its 629 k P labels supervise a P onset the operator never picked;
  that source needs review before it stays in a pilot.
- **STEAD, SCEDC and OBST2024 carry status columns whose values the
  repository has never counted** (the five sampled STEAD rows were noise
  traces; SCEDC's sampled rows had NaN status and weights 1.0/0.3/0.5).
  The server run fills `pick_status_counts` for them.
- **INSTANCE, PNW, TXED and Iquique expose no manual/automatic flag.**
  INSTANCE carries per-pick uncertainties (0.1 s P, 0.6 s S in the sampled
  rows) and location weights; PNW carries uncertainties of 0.01 to 0.08 s.
  Provenance for these is the dataset paper, not a column, and the census
  records that as "provenance by uncertainty/weight columns only".
- **Ten pool sources have no column record at all** (MLAAPDE, CREW, CWA,
  PiSDL, AQ2009GM, MEIER2019JGR, ROSS2018GPD, OBS, and CEED and VCSEIS
  whose metadata was absent when the notebook ran): 790,000 of the
  1,453,400-trace training cap (`DATASET_CONFIGS` caps summed) comes from
  sources whose pick status the repository has never looked at.
- **S availability.** ETHZ has S on 86 of 36,743 traces and `use_s=True`;
  GEOFON has S on 2,648 traces and is P-only by rule; LenDB has none.
  Everything else with counts has S on 54 to 100 % of traces (SCEDC 54 %,
  OBST2024 59 %, TXED 60 %, INSTANCE 62 %, STEAD 81 %, Iquique 85 %,
  VCSEIS 92 %, PNW 100 %).
- **Native rates** are on record for GEOFON (20 Hz dominant, 20 to 200),
  ETHZ (100 to 500, 200 dominant), SCEDC (40 and 100), PNW, VCSEIS, CEED,
  Iquique (100); STEAD, INSTANCE, TXED, LenDB and OBST2024 have no rate
  column the 2026-04-06 audit could read, and the nine unaudited sources
  are `server`.

## 4. What this says about the T0 pilot

The pilot #40A/#46 needs inspected examples at every native rate and S
condition (`docs/2026-09-10_picker_and_issue_roadmap_audit.md`, gate C).
From this census:

- **Manual P and S with an explicit per-reading mode** exist today from
  INGV and NOA (bulletin side) and, among SeisBench sources, only GEOFON
  and NEIC have a verified manual flag, both P-only in practice. A pilot
  that wants verified manual S has to harvest it from the bulletins or
  verify STEAD/SCEDC/OBST2024 status values on the server first.
- **Native rates represented.** The bulletin readings carry no waveform,
  so their native rate is the station's and is measured only when
  waveforms are fetched; the held-out builder's `stations.csv` tables
  show IV HH and EH at 100 Hz (IV BH at 20 Hz, MN HH at 80 or 100 Hz)
  for the Italian sequences and HL/HA/HT/HP HH at 100 Hz for the Greek
  ones. SeisBench sources with a known rate cover 20, 40, 80, 100, 120,
  125, 200, 250 and 500 Hz; of the sources with a rate on record, only
  GEOFON (20 Hz), ETHZ (200 Hz dominant) and part of SCEDC (40 Hz) are
  not at 100 Hz.
- **Held-out overlap is largest where the pilot would most like to draw**:
  INSTANCE holds Norcia 2016 and the Etna and Campi Flegrei places,
  SCEDC/CEED/ROSS2018GPD hold Ridgecrest, PNW holds Monroe, and any 2018
  to 2019 INGV or NOA harvest loses 1 to 11 % of readings to the Etna and
  Corinth-Thiva places. The place rules bite on every Italian and Greek
  source; the year rules bite on none of the sampled years.
- **A development region outside Europe** (issue #39 acceptance) is not
  answered by this run: GeoNet, USGS and ISC are implemented in the same
  script and were not run. `python scripts/source_census.py bulletin
  --operators GEONET USGS ISC --years 2018 2019 --days-per-quarter 1
  --max-minutes 30 --seed-tag 39a` is the next command.

## 5. What remains, and where

On the lab server (`SEISBENCH_CACHE_ROOT` set):

```sh
python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT
```

writes `data/census/seisbench_server.csv` with, per pool source, the
value counts of the P and S status columns (manual, automatic, estimated,
NaN, other), the rows inside held-out windows and years
(`heldout_sequences.flag_rows`, the manifest builder's function) and the
rows without an origin time. That fills the two `server` columns of
`seisbench_sources.csv` and replaces the five-row observations above with
counts. Until it runs, no SeisBench source's manual share is a number.

Still open on the bulletin side, all runnable here: the other operators
(GeoNet, USGS, ISC, franceseisme), more years including the held-out 2016
and 2021 so the year rule is exercised, two days per quarter, and a
shared-event join between operators on common days. A rerun of INGV and NOA
2018 and 2019 with `--seed-tag 39a` draws different days from the 16 of
section 2 (the sampling rule changed in PR #75, section 1) and refetches them;
the 83 MB cache of the section 2 run can be deleted.

Not in scope of this checkpoint: waveform accessibility, response state
and native rate per station (a station-side census against the FDSN
station services), the low-SNR and multi-event measurements (need
waveforms), and any nomination of pilot sources beyond the reading above.
