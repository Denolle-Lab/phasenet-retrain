# Model-independent label audit of the curated sources (#41B)

*2026-09-15, branch `issue/41b-label-audit` from `audit/2026-09-07-generalization`
at `2581125`. Checkpoint 41B of issue #41, widened from the pilot corpus to the
curated SeisBench sources. No model is run. The script is
`scripts/audit_source_labels.py`, the tests `tests/test_audit_source_labels.py`,
the calibration outputs `data/label_audit/heldout/`.*

## Why

The September audits found the training path mishandled labels (absent labels
supervised as negatives, native-rate misalignment, zero windows). The curated
labels themselves had never been checked. Today `scripts/label_error_filter.py`
drops every row named in Aguilar's multiplet reports
(`docs/LABEL_ERROR_FILTERING.md`): a flag means an unlabelled second arrival,
so the drop removes exactly the multi-event windows the aftershock objective
needs, and the flags are model-derived. This audit is model-independent. It
produces a per-source label-quality table with bootstrap intervals and a
per-row flag the 41A arrival schema (`scripts/arrivals.py`, tiers `manual,
reviewed, automatic, unknown`) can carry, so a suspect label masks its
neighbourhood instead of supervising it (`scripts/label_targets.py`, policy
`masked`).

## Row structure and readers

Every row is a `LabelRow`: waveform `(3, n)` ZNE float32 at the stored rate,
`rate_hz`, `p_s`/`s_s` (seconds from the first sample, None when absent),
`distance_km` (None when unknown), `source`, `trace_id`, `station`, `event_id`,
`p_status`/`s_status` (the source's pick status text, else `unknown`),
`flagged_multiplet` (from a cached Aguilar report; never downloaded by default).

| Reader | Rows | Waveform path | Distance |
|---|---|---|---|
| `HeldoutReader(key)` | one per (event, station) with a manual P on a station whose MiniSEED is on disk; S optional | the case's MiniSEED at native rate, 120 s from 30 s before P, shorter at file edges (`window_start_time`, `n_samples` record the actual cut); regression and dev roles only, access recorded as `reference_qa` | catalogue origin and station coordinates, else `stations.csv` `km` |
| `SeisBenchReader(source, cache_root, sample, seed)` | stratified deterministic sample (strata: the builder's distance bins when the source has a distance column; proportional allocation, floor 50 per stratum) | `manifest_dataset._fetch_sbd` through a namespace shim, `SingleHDF5Reader` / `ChunkedHDF5Reader` for `mlaapde, cwa, aq2009gm, obs, pisdl, meier2019jgr, ross2018gpd`, so the audit reads what the loader reads; picks through `build_training_dataset.coalesce_picks` over `P_PRIORITY` / `S_PRIORITY`, S nulled on teleseismic rows as the builder does | `DATASET_CONFIGS` `dist_col` / `dist_unit` |

The module is torch- and SeisBench-free at import; the SeisBench reader
imports `seisbench`, `h5py` and `manifest_dataset` (torch) when constructed.

## The six checks

All energy measures use the demeaned trace through a causal 2nd-order
Butterworth band-pass 1 Hz to min(20 Hz, 0.4 × Nyquist) (1–4 Hz at 20 Hz,
1–10 Hz at 50 Hz, 1–20 Hz at 100 Hz and above). Causal, so filter ringing
never precedes an onset; the group delay biases AIC onsets late by at most
0.05 s on the synthetic set.

| Check | Rule | Tolerance / decision | Per row | Per source |
|---|---|---|---|---|
| C1 S−P consistency | Theil–Sen fit (`scipy.stats.theilslopes`) of ts − tp against distance over rows with both picks, a distance and ts > tp (≥ 5 rows); residual = (ts − tp) − (a + b·D) | `c1_flag` when \|residual\| > max(4·MAD, 1.0 s) or ts ≤ tp; rows without distance flagged only when ts ≤ tp | `c1_residual_s`, `c1_flag`, `c1_ts_le_tp` | slope (s/km), implied Vp/Vs = 1 + 6.0·slope (sanity line, not a flag), MAD, threshold, flag fraction with interval |
| C2 P onset vs energy | Maeda AIC (AIC(k) = k ln var(x[:k]) + (N−k) ln var(x[k:]), minimum over the splits where the variance increases) on the vertical channel in [tp − 3 s, tp + 3 s], cut at the P–S midpoint when S is labelled | testable when the RMS over 1 s after the AIC onset exceeds 2× the RMS over 1 s before it (else never flagged); `c2_flag` when testable and \|onset − tp\| > 0.5 s; `c2_early` when onset − tp < −0.5 s (energy before the label) | `c2_onset_s`, `c2_residual_s`, `c2_rms_ratio`, `c2_testable`, `c2_flag`, `c2_early` | testable fraction, median and MAD of the residual, p05/p95, flag fraction at 0.5 s and at 1.0 s, early fraction, each with interval |
| C2 S onset vs energy | the same on the horizontal energy N² + E², W = 4 s, window starting no earlier than the P–S midpoint (at stations a few km away the P lies inside ±4 s of the S and captured the minimum before this cut); RMS ratio on √(N² + E²) | as C2 P | `c2s_*` | as C2 P |
| C3 component sanity | Z/H energy ratio over 1 s after P and 1 s after S on rows where both onsets are testable | source-level: fraction of rows with ratio_P > ratio_S; below 0.5 is a warning in the summary, never a per-row flag | `c3_ratio_p`, `c3_ratio_s`, `c3_p_gt_s` | fraction with interval, `c3_warning` |
| C4 pick placement | P position along the trace | `c4_edge` when tp < 1 s or tp > duration − 1 s | `c4_p_fraction`, `c4_edge`, `p_sample` | edge fraction with interval, the single most common P sample and its share (a cropping convention), n_samples min/median/max, rates |
| C5 cross-source duplicates | metadata only (`duplicates` subcommand): `exclusion_bundle.origin_unions` (2 s, 0.1°) over the union of the listed sources' metadata, then (event group, `station_code`) seen by more than one source | comparable when both rows carry `trace_start_time`, a P sample and a rate: absolute P times disagreeing by more than 0.2 s | `duplicates.csv`: one line per cross-source pair | fraction disagreeing with interval; pairs without a start time reported as not comparable |
| C6 unlabelled arrivals | obspy `classic_sta_lta` (STA 0.5 s, LTA 10 s) on the root-sum-square of the band-passed components over the whole window, `trigger_onset(4, 1.5)`; run on every row | `n_extra_triggers` = triggers farther than 1 s from every labelled arrival and outside [tS, tS + 3 s] when S is labelled; `has_unlabelled_arrival` = n_extra ≥ 1. Aguilar-flagged rows also classified: `second_event` (a trigger within 1 s of P and ≥ 1 unexplained trigger), `wrong_first_pick` (≥ 1 trigger, none within 1 s of P), `no_detection`, `single_detection` (one trigger at P, nothing unexplained) | `c6_triggers_json`, `c6_n_triggers`, `c6_n_extra_triggers`, `c6_n_extra_before_p`, `c6_has_unlabelled_arrival`, `c6_class`, `c6_p_in_blind` | unlabelled fraction with interval, overall and split by `flagged_multiplet` true/false (what the Aguilar report catches and what it misses); before-P fraction; the class fractions over flagged rows |

The detector is blind in the first 10 s of a trace (LTA warm-up;
`c6_p_in_blind` records a P inside it) and, after a large arrival, the inflated
LTA suppresses smaller ones for tens of seconds. It over-triggers on noise
bursts and on coda (regional S coda lasts far longer than the 3 s window
excluded here), and it does not trigger at all on emergent onsets at ratio 4.
Every trigger it reports is therefore a proposal, not a pick.

Bootstrap intervals are 95 % percentile intervals from `metrics.bootstrap_ci`
(n_boot 1000, seed 0); an interval is NaN below five values.

## What a flag does downstream

`rows.parquet` carries, per row:

- `suggested_tier`: `manual` when no flag; `unknown` when `c1_flag` or
  `c2_flag` or `c4_edge`. Under `label_targets.MASKED` an `unknown`-tier
  arrival supervises nothing and masks ±1.0 s around itself, so a suspect P
  neither teaches its time nor lets the surrounding samples be supervised as
  noise. `suggested_tier_s` does the same for the S from `c1_flag` or
  `c2s_flag` (tiers are per arrival, so the S is not masked for a P defect).
- `suggested_extra_arrival_s`: the JSON list of unexplained trigger times, for
  every row with `n_extra_triggers ≥ 1`. The 40A builder adds these as
  `automatic`-tier arrivals in `arrivals_json`, which mask ±1.0 s under the
  41A policy until a reviewer promotes or deletes them; the row is kept. They
  are never supervised picks: the screen over-triggers on noise bursts and
  coda.

`review_sheet.csv` (up to 50 random rows per source among those a check or
the multiplet report singled out, with the reasons, the labelled picks, the
trigger times and the class) is the human-review deliverable; the class is a
proposal.

Nothing is dropped by this audit. The Aguilar rows stay in the pool with
their extra arrivals masked, which is the change #41 asks for relative to
`label_error_filter.py`.

## Held-out calibration (analyst picks)

Run on the laptop in the base interpreter after the code commit;
`provenance.json` names that commit and the sha256 of every case file:

```
python scripts/audit_source_labels.py heldout --keys kaikoura_2016 norcia_2016 thessaly_2021 \
    samos_2020 adriatic_2022 etna_2022_2024 corinth_thiva_2020 --out-dir data/label_audit/heldout --report
```

2592 rows over the seven regression/development cases (the stations with
waveforms on disk: six per case, two windows for Etna and Corinth–Thiva),
all with a manual P, 2145 with a manual S, every row with a catalogue
distance, 100 Hz except Samos KO.DKL at 50 Hz. From
`data/label_audit/heldout/summary.csv`:

| case | rows | C1 slope s/km | Vp/Vs | C1 flag % | C2 P testable % | C2 P flag % (0.5 s) | at 1.0 s | C2 P early % | C2 P median res s | C2 S flag % | C2 S median res s | C3 P>S % | C6 unlabelled % | before P % | suggested unknown % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kaikoura_2016 | 357 | 0.112 | 1.67 | 8.5 [5.4, 11.9] | 88.5 | 17.1 [13.0, 21.2] | 7.9 | 3.2 | +0.15 | 18.1 [13.6, 22.7] | +0.14 | 88.7 [84.5, 92.5] | 98.0 [96.4, 99.4] | 58.5 | 21.0 |
| norcia_2016 | 701 | 0.127 | 1.76 | 1.8 [0.9, 2.9] | 71.6 | 15.1 [12.0, 18.5] | 9.4 | 13.1 | +0.02 | 3.6 [2.2, 5.2] | +0.04 | 84.7 [81.6, 87.9] | 99.7 [99.3, 100.0] | 65.6 | 11.7 |
| thessaly_2021 | 395 | 0.119 | 1.71 | 1.5 [0.3, 3.0] | 87.6 | 21.1 [17.0, 25.4] | 10.7 | 7.8 | +0.06 | 51.9 [46.3, 57.9] | +0.60 | 95.9 [93.2, 98.1] | 94.9 [92.7, 97.0] | 48.1 | 19.5 |
| samos_2020 | 34 | 0.109 | 1.66 | 0.0 [0.0, 0.0] | 64.7 | 45.5 [27.3, 63.6] | 22.7 | 4.5 | +0.32 | 65.0 [45.0, 85.0] | +0.78 | 81.2 [62.5, 100.0] | 91.2 [82.4, 100.0] | 58.8 | 29.4 |
| adriatic_2022 | 400 | 0.117 | 1.70 | 2.0 [0.8, 3.6] | 68.5 | 8.8 [5.8, 12.0] | 2.9 | 4.0 | +0.04 | 19.8 [15.4, 25.1] | +0.19 | 93.6 [89.8, 96.8] | 73.8 [69.0, 77.8] | 15.2 | 8.2 |
| etna_2022_2024 | 597 | 0.054 | 1.32 | 1.0 [0.3, 2.0] | 88.3 | 5.9 [4.0, 8.0] | 2.3 | 2.8 | +0.05 | 20.1 [15.5, 25.0] | +0.24 | 100.0 [100.0, 100.0] | 59.3 [54.9, 63.3] | 14.7 | 5.9 |
| corinth_thiva_2020 | 108 | 0.115 | 1.69 | 6.9 [2.3, 12.6] | 67.6 | 24.7 [15.1, 34.3] | 16.4 | 9.6 | +0.13 | 41.7 [30.0, 55.0] | +0.28 | 76.6 [61.7, 87.2] | 72.2 [63.9, 79.6] | 23.1 | 22.2 |

Reading the table:

- **C1.** Slopes 0.109–0.127 s/km, implied Vp/Vs 1.66–1.76 on the six
  crustal sequences; Etna's 0.054 s/km (1.32) is not a velocity ratio: at
  4–34 km epicentral distance with shallow sources the S−P time is set by
  depth, and the sanity line says so. No ts ≤ tp anywhere. Flag rates 0–2 %
  except Kaikōura 8.5 % (25 rows, residuals −3.6 to +4.0 s at 7–129 km on a
  multi-fault rupture with GeoNet picks: the Pn/Pg and Sn/Sg convention
  question of #41, to be settled by the reviewed sample, not by the check)
  and Corinth–Thiva 6.9 % (6 rows).
- **C2 P.** Testable on 65–89 % of rows (the RMS-ratio gate removes the
  low-SNR picks). Median residuals are positive everywhere (+0.02 to
  +0.32 s): the energy onset follows the analyst's pick, as it must for an
  emergent first motion plus the filter delay. At the 0.5 s tolerance the
  check flags 6–45 % of analyst P picks (286 of 2060 testable rows; 137 with
  energy before the label, 149 with energy after); at 1.0 s, 2–23 %. Samos
  (34 rows, three stations 46–168 km from M3.7–4.1 events in the three hours
  after the Mw 7.0 mainshock) is the outlier at 45.5 % [27.3, 63.6] and is the case where
  emergent regional P dominates. The 0.5 s symmetric rule therefore masks
  correct emergent picks; `suggested_unknown` reaches 6–29 % on analyst
  labels. The `c2_early` fraction (energy before the label, 2.8–13.1 %,
  Norcia highest with 13.1 %: in that sequence density a preceding event's
  arrival inside the 3 s window is the likely cause) is the part of the flag
  that hurts training under the legacy formula, because those samples were
  supervised as noise.
- **C2 S.** Norcia (INGV) S picks agree with the horizontal energy onset to
  a MAD of 0.05 s (3.6 % flagged). The Greek cases do not: Thessaly 51.9 %,
  Samos 65.0 %, Corinth–Thiva 41.7 %. The residuals are station-wise
  constants, not scatter (from `rows.parquet`, stations with ≥ 5 testable
  S): Thessaly HL.KZN +1.95 s (MAD 0.66, 74 km), HT.AGG +1.47 (0.48,
  79 km), HL.TETR +1.56 (0.69, 90 km), HT.LIT +1.11 (0.41, 49 km) against
  HL.THL +0.06 (0.12, 24 km) and HT.TYRN +0.04 (0.08, 7 km); Corinth–Thiva
  HL.LKR +1.33 (0.11, 80 km), HA.ACOR +1.15 (0.26, 55 km) against HL.KLV
  +0.10 (0.05, 13 km). Beyond ~50 km the NOA S pick sits 1–2 s before the
  horizontal energy jump. This is a phase or channel convention of the
  operator (an early emergent S, or a pick on a different channel), not
  random error, and it is exactly the case the #41 acceptance text names:
  emergent S is not to be rejected because a detector misses it. For the
  reviewed sample these stations are the first to look at.
- **C3.** 77–100 % of rows have a larger Z/H ratio after P than after S;
  no warning. The check is not diagnostic at this level; it would catch
  swapped or mislabelled channels, none found.
- **C4.** By construction (the held-out cut starts 30 s before P) every P
  sits at sample 3000, so the mode share is 100 % and says nothing here; it
  is meaningful only for the SeisBench sources. Two Adriatic rows are within
  1 s of a file edge.
- **C6.** 59–100 % of the 120 s windows contain a trigger no label
  explains, and 15–66 % contain one before the labelled P. These are
  aftershock sequences: the windows genuinely hold other events, which is
  the multi-event material the aftershock objective wants, and the screen
  also re-triggers on regional S coda beyond the 3 s exclusion. Median
  triggers per window 2 (Adriatic, Etna, Corinth–Thiva) to 5 (Kaikōura,
  Norcia). No case has Aguilar flags, so the flagged/unflagged split and the
  three-way classification are empty here; they fill on the server run.

Verdict of the calibration: C1, C3 and C4 behave; C2 at 0.5 s is too strict
for regional emergent onsets and its early-energy half is the part to act
on; C2 S exposes an operator-level S convention on Greek stations beyond
50 km; C6 works as a screen and, in sequences, flags nearly everything,
which is the truth of those windows and the reason the extra triggers enter
as masked `automatic` arrivals rather than as drops or as supervised picks.
The tolerances are left at the specification values in the code; the
reviewed sample (below) decides whether the mask trigger becomes `c2_early`
or a 1.0 s tolerance. `summary.csv` reports both.

## Server commands

Added as step 6b of `docs/2026-09-13_server_session_runbook.md`:

```bash
python scripts/audit_source_labels.py seisbench --sources instancecounts ethz stead ceed pnw txed aq2009gm \
    geofon crew cwa iquique lendb vcseis scedc pisdl meier2019jgr ross2018gpd mlaapde \
    --sample 5000 --seed 0 --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/seisbench --report
python scripts/audit_source_labels.py duplicates --sources stead instancecounts ethz ceed scedc ross2018gpd \
    --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/seisbench
git add data/label_audit/seisbench/summary.csv data/label_audit/seisbench/provenance.json \
    data/label_audit/seisbench/report.md data/label_audit/seisbench/*/review_sheet.csv \
    data/label_audit/seisbench/duplicates.csv data/label_audit/seisbench/duplicates_provenance.json
```

The Aguilar reports must be cached under `data/labelerrors/` or
`~/.cache/phasenet_retrain/label_errors/` (`--download-reports` fetches the
missing ones into the first `--report-dirs` entry); `provenance.json` records
each report's sha256. `rows.parquet` stays out of git (`.gitignore`). The
`seisbench` and `duplicates` paths were exercised here only on a fake
HDF5+CSV source in the torch venv (both the SeisBench route and the direct
route); the cache is not on this machine.

## What remains for 41B proper

1. The server run above: the per-source table for the curated sources, the
   C6 split by Aguilar flag (what the report catches, what it misses), the
   three-way classification of the flagged rows, and C5 on the sources that
   overlap (STEAD/SCEDC/ross2018gpd in southern California; INSTANCE/ETHZ/
   aq2009gm in Italy).
2. The reviewed sample of the issue's acceptance text: per-operator reviewed
   rows from `review_sheet.csv` (the Greek far-station S picks first, then
   the `c2_early` P rows, then a random slice of `second_event` proposals),
   with corrected arrivals or explicit uncertainty masks and retained,
   removed and quarantined counts with reasons. The reviewer sees the
   waveform and the label, not which check raised the row.
3. The tolerance decision after that review (0.5 s symmetric, 1.0 s, or
   `c2_early` only) and the wiring of `suggested_tier` and
   `suggested_extra_arrival_s` into the 40A builder's `arrivals_json`.
4. A depth- and phase-aware C1 (Pg/Pn, Sg/Sn) for the sources that label
   them; the Theil–Sen line over epicentral distance is the model-free
   version and is enough for the outlier screen at the rates seen here.

Training-label review and sealed reference review stay separate records:
this audit reads the regression and development cases only and refuses the
acceptance keys.
