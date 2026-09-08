# Generalization audit, 2026-09-07: results

*Marine Denolle with Claude, branch `audit/2026-09-07-generalization`, from
commit 77f581d. Answers the five tasks of
`docs/2026-09-07_generalization_audit_prompt.md`. Every table in this
directory was produced by `scripts/audit_generalization_hypotheses.py`
(local, from the committed `notebooks/step3_results.parquet` and
`notebooks/step3_metrics.csv`) except the two transcriptions named below.
Every number carries a 95 % percentile-bootstrap interval from
`scripts/metrics.py::bootstrap_ci` unless the table says why not.*

## What could not be done, and why

The lab servers were unreachable from the laptop this session ran on:
`siletzia`, `cascadia`, `psound` and `marine1.ess.washington.edu` do not
resolve off the UW network and `dasway` refuses SSH on ports 22 and 7777.
The clone has no SeisBench cache (`~/.seisbench` holds only `models/`), no
`data/manifests*`, an empty `checkpoints/` and an empty `results/`. Nothing
was reconstructed. Consequently:

| Needed for | Asset (server only) | Status |
|---|---|---|
| Task 1 counts and exclusion list | `data/manifests_v2/{train,val}.csv`, full metadata of the ten corpora | not computed; script written, self-tested, wired |
| Task 2 per-trace `clean_holdout` | `results/event_leakage_row_mask__data__manifests_v2__train.csv` | aggregates read from `step3_metrics.csv`; paired re-score done on the full benchmark instead |
| Task 2 / H00 noise-pool detection MCC | `results/noise_fp_audit.csv`, `results/detection_metrics.csv`, `checkpoints/finetune_jma_wc_global_v*/best.pt` | 2026-08-10 table transcribed from `paper_draft.qmd` (`detection_metrics_2026-08-10_snapshot.csv`); 14 of 20 versions were never scored |
| H2 S fraction by distance bin | `data/manifests_v2/train.csv` | cap-based bound only |
| Training curves, `results/*_metrics.csv` | server | not read |

The "What is known from outside this repository" and "Hypotheses" sections
that the prompt file says to read are not in it (the file has "Prompt" and
"Notes for Marine" only; no branch, stash or QuakeScope document carries
them). The external numbers were taken from the rendered QuakeScope reports
(`reports/phasenet_global_sequences.html`, run 2026-09-07, and
`reports/phasenet_sequence_comparison.html`, notebook of 2026-08-20;
transcribed in `external_results_2026-09-07.csv`), and H00 to H4 were
reconstructed from `docs/2026-09-07_training_history_audit.md` §4 and the
task 3 hints:

- H00: the selection metric (unconditional P-MAE on a positives-only benchmark) ordered the versions in a way a false-positive-aware metric does not reproduce.
- H0: positional prior from the fixed P placement in training windows. Tested and dropped in the audit (§4.2).
- H1: v7's probabilities on true arrivals are uniformly lower (T = 4 distillation, hard targets), so a shared threshold under-counts it; matched budget should close the gap.
- H2: S was under-supervised, especially at regional distance, so S recall fell most there.
- H3: the external sequences sit in v7's corpus; the results are contaminated in v7's favour.
- H4: v7 lost capability at low SNR, where the corpus had no curation and the only noise augmentation was white noise.

## Task 1. Independence of the test sequences

**Built and verified locally**

- `scripts/heldout_sequences.py`: the eight windows (mainshock hours and sequence spans for Norcia, Kaikōura, Thessaly; spans for Ridgecrest and Monroe; great-circle radii of 1° or 2° as the QuakeScope `maxradius`), the 2016/2021 year hold-out, the flagging helpers, the fail-closed loader and the manifest check. Pure pandas.
- `scripts/audit_heldout_sequences.py`: the spatiotemporal join over `manifests_v2/{train,val}` (joined to source metadata on `(trace_name, chunk)` after a checksum check against `data/manifest_checksums.csv`), the ten full corpora and `notebooks/benchmark_manifest.csv`; writes `data/exclusions/heldout_sequences.csv` (dataset, trace_name, chunk, window, origin fingerprint, source) and `heldout_sequence_counts.csv` (per source, dataset, window, plus year counts). `--check-manifest` verifies any manifest. `--selftest` passes.
- `scripts/build_training_dataset.py`: loads the list beside `benchmark_exclude` and refuses to run without it; drops every row with a 2016 or 2021 origin (rows with no origin time are kept and counted as `year_unverifiable`, or dropped with `--strict-year-holdout`); writes `source_origin_time`, `source_latitude_deg`, `source_longitude_deg` into every manifest; gates the written train/val/test with `check_manifest` and aborts on any violation; writes `heldout_removal_report.csv`.
- `tests/test_heldout_sequences.py`: ten tests, all passing with the existing ten in `tests/test_metrics.py`.
- `.gitignore`: `data/exclusions/` is un-ignored so the list is committed when it exists.

**Not done: the counts.** They need the server. Run there:

```bash
export SEISBENCH_CACHE_ROOT=/path/to/cache
python scripts/audit_heldout_sequences.py          # writes data/exclusions/*.csv, prints the counts table
git add data/exclusions && git commit -m "Task 1: held-out sequence exclusion list and counts"
```

What can be said locally is the space-only pre-screen of the benchmark
(`h3b_benchmark_spatial_prescreen.md`): 916 INSTANCE, 17 STEAD and all
1,346 AQ2009GM benchmark traces lie within 1° of Norcia (AQ2009GM is
L'Aquila 2009, outside the time window; the INSTANCE and STEAD ones are
unresolved), 406 CEED and 49 STEAD within 1° of Ridgecrest, 987 PNW and 70
STEAD within 1° of Monroe, 43 STEAD within 2° of Kaikōura, 23 STEAD within
1° of Thessaly. Origin times for these are on the server. The benchmark's
only dated slice, MLAAPDE, is 2013-07 to 2014-10 and touches no window.

## Task 2. v7 against the parent, honestly

Tables: `task2a_clean_holdout_aggregates`, `task2b_paired_full_benchmark`,
`task2c_recall_curves`, `task2d_noise_pool_detection_mcc`.

On `clean_holdout` as committed (server-computed, populations differ by
about 1,000 traces, not paired):

| | P recall | S recall | P-MAE cond. (s) | S-MAE cond. (s) |
|---|--:|--:|--:|--:|
| jma_wc, all | 0.909 [0.904, 0.912] | 0.527 [0.519, 0.536] | 0.250 | 0.641 |
| v7, all | 0.876 [0.872, 0.880] | 0.477 [0.468, 0.485] | 0.211 | 0.598 |
| jma_wc, regional | 0.878 [0.872, 0.884] | 0.361 [0.346, 0.377] | 0.333 | 0.707 |
| v7, regional | 0.824 [0.817, 0.831] | 0.277 [0.261, 0.291] | 0.285 | 0.634 |

Paired on the same 31,880 P and 16,967 S traces of the full benchmark
(which is v7's `in_domain` split, see H3), threshold 0.3, v7 minus parent:

| Phase, bin | Recall | AUC-recall (mean prob.) | Cond. MAE on picks both make (s) |
|---|--:|--:|--:|
| P local | −0.012 [−0.015, −0.009] | −0.021 [−0.023, −0.019] | −0.014 [−0.019, −0.010] |
| P regional | −0.043 [−0.047, −0.038] | −0.047 [−0.049, −0.045] | −0.026 [−0.032, −0.019] |
| P teleseismic | −0.101 [−0.119, −0.082] | −0.055 [−0.060, −0.049] | +0.011 [−0.001, 0.026] |
| P all | −0.030 [−0.032, −0.027] | −0.034 [−0.036, −0.033] | −0.019 [−0.023, −0.015] |
| S local | −0.032 [−0.037, −0.028] | −0.012 [−0.014, −0.011] | −0.002 [−0.005, 0.001] |
| S regional | −0.073 [−0.082, −0.065] | −0.030 [−0.032, −0.027] | +0.003 [−0.002, 0.006] |
| S all | −0.044 [−0.047, −0.040] | −0.017 [−0.018, −0.015] | −0.001 [−0.004, 0.001] |

Noise-pool detection MCC at each model's own best threshold, from the
2026-08-10 table: parent 0.776 [0.771, 0.780], v7 0.771 [0.766, 0.776],
v7_eventclean 0.745 [0.740, 0.750].

**Matched pick budget** cannot be formed on this benchmark. It stores one
probability per trace at the argmax inside ±5 s of the true pick
(`scripts/eval_finetuned.py:157-171`) and has no negative windows, so the
number of picks emitted at a threshold is recall times n
(`task2c`: the v7 threshold that emits the parent's 28,174 P picks is 0.224
and gives, by construction, the parent's recall). The threshold-independent
summary the repository uses instead, AUC-recall from
`scripts/threshold_independent_ranking.py`, is in the table above; the
matched-budget numbers come from the QuakeScope notebooks.

**Does any v7 advantage survive?** One: 19 ms [15, 23] of P timing on the
picks both models make, 14 ms locally and 26 ms regionally. No S-timing
gain, lower recall and lower AUC-recall in every bin, and a detection MCC
that is tied or slightly worse. Externally the timing residuals agree to
within 6 ms on every sequence and phase.

## Task 3. Hypotheses, one table each

**H1, calibration** (`h1a_peak_probability_distributions`,
`h1b_recalibration_residual`). Median probability at the true arrival, v7
minus parent: P −0.034 [−0.038, −0.031], S −0.070 [−0.076, −0.063]; Spearman
rank agreement 0.87 (P) and 0.97 (S). At 0.3 the parent detects 1,457 P and
936 S traces v7 does not; the reverse sets are 515 and 195. Rank-mapping v7
onto the parent's probability scale (pooled per phase) leaves no residual in
the local bins, in regional P, or in any SNR bin above 5 dB for either
phase, but leaves −0.009 [−0.016, −0.002] at regional S, −0.047 [−0.065,
−0.029] at teleseismic P, −0.045 [−0.057, −0.034] at P below 0 dB and
−0.008 [−0.014, −0.003] at 0 to 5 dB, compensated by +0.005 to +0.013 in
the high-SNR bins. **Verdict: most of the shared-threshold deficit is
calibration, which a matched budget removes and which explains why the
external gap shrinks from 3–4 points to 1–2. The part that is left is not
calibration.**

**H2, S supervision** (`h2a_s_fraction_by_distance_bound`,
`h2a_dataset_configs_parsed`, `h2b_recall_deficit_by_distance`). The
manifest is on the server, so only bounds: 340,000 of the 1,453,400 cap
budget (23.4 %) is P-only by policy (`geofon`, `lendb`, `meier2019jgr`), the
training split targets 25 % teleseismic windows whose S is nulled, and
`data/README.md` records 38 % S overall. The observable H2 predicts is
there: the S deficit is larger than the P deficit and grows with distance
(local −0.032, regional −0.073 against P −0.012, −0.043), and it is uniform
across SNR (H4a, −0.031 to −0.062 in every bin) with a uniform −0.07 median
probability shift, i.e. a loss of confidence on S everywhere rather than a
low-SNR effect. **Verdict: consistent with H2; the by-bin S fraction is
still to be computed with `audit_generalization_hypotheses.py h2` on the
server.**

**H3, contamination** (`h3a_in_domain_vs_all`,
`h3b_benchmark_spatial_prescreen`, `h3c_sequence_corpus_coverage`). For v7
`in_domain` equals `all` (32k traces): the benchmark tests v7 only on
datasets it trained on, and it loses there by 0.028 P and 0.044 S recall.
The external sequences may well be in its corpus (Norcia in INSTANCE,
Ridgecrest in SCEDC/CEED/ROSS2018GPD, Monroe in PNW, by date coverage), and
the pre-screen above shows the benchmark itself may carry hundreds of
traces from the Norcia, Ridgecrest and Monroe regions. **Verdict:
contamination can only have flattered v7; it cannot explain a loss. Its
extent is task 1's pending count.**

**H4, low SNR** (`h4a_recall_by_snr`, `h4b_misses_by_snr`). Paired P
recall deficit by SNR: −0.007 [−0.011, −0.002] above 20 dB, −0.005 at 10–20,
−0.026 [−0.033, −0.021] at 5–10, −0.056 [−0.063, −0.050] at 0–5, −0.096
[−0.109, −0.084] below 0 dB; median probability deficit −0.007 above 20 dB
against −0.136 [−0.150, −0.122] below 0 dB. Of v7's P misses, 71 % [69, 72]
are below 5 dB (the v13 header said 82 %), against 70 % of the parent's own
misses; the traces only the parent detects have a median SNR of 3.5 dB and
58 % [56, 61] lie below 5 dB, the traces only v7 detects have a median SNR
of 12.3 dB. The S deficit does not depend on SNR. **Verdict: supported for
P. This is the deficit that survives recalibration and the one a matched
budget on regional continuous data would see.**

**H00, the selection metric** (`h00_detection_mcc_ordering`). Nine own
weights have a noise-pool detection MCC. Their rank under it against their
rank under the unconditional P-MAE that drove selection has Spearman 0.65;
v7 is second on P-MAE and fifth on detection MCC, one place below the
parent; v11 (T = 1.5) is the only single fine-tune above the parent, 0.804
[0.799, 0.808] against 0.776 [0.771, 0.780], and the v7+v11 ensemble reaches
0.823 [0.819, 0.827]. Under P recall the parent is second only to the
trigger-happy v18. **Verdict: the selection metric did not pick the best
detector among the fine-tunes, and no single fine-tune it could have picked
beats the parent on recall. The full v1–v20 ordering does not exist:
`jma_wc_ft_global_v2, v4, v5, v6, v8, v9, v10, v12, v13, v14, v15, v16, v17, v19`
and the four pre-v2 runs were never scored on the noise pool.**

**External results with intervals** (`external_recall_with_intervals`):
individually, no non-US sequence separates v7 from the parent at 0.3
(every unpaired difference interval spans zero; the paired interval would
be narrower but the per-pick data is not stored); in the western set only
Mendocino P does, −0.070 [−0.140, −0.005]. The evidence is the
consistency: at matched budget v7 is below the parent on 6 of 6 non-US
sequence-phase pairs (sign test p = 0.016), on Mendocino S, and above it
only on Ridgecrest S (+0.024).

## Task 4

Written into `paper_draft.qmd` as item 15 of §Critical audit (Blocking),
with in-place retraction callouts under §Leaderboard, §By-distance and the
§Trajectory selection-bias callout, and a dated note in the Roadmap. The
original text and tables are kept.

## Task 5

Task 3 identifies causes the recipe can address (the calibration offset,
the low-SNR P loss, the S under-confidence) and one it cannot (teleseismic
P), and the external results say a recalibrated v7 would close most but not
all of the gap. A conditional v21 is proposed in
`docs/2026-09-07_v21_proposal.md`: analyst P and S only, no P-only sets, no
teleseismic rebalancing, exclusion list and year hold-out applied, SNR
stratification, random window position, real-noise superposition from the
existing pool, band-limiting and resampling to 40–100 Hz, channel dropout,
soft Gaussian targets, a selection split never read during development, and
a pre-registered acceptance test. Its first step costs no training: convert
v11 with the QuakeScope path and run the two notebooks. Not run.

## Reproducing the tables

```bash
python -m pytest tests/ -v                                   # 20 tests
python scripts/audit_heldout_sequences.py --selftest
python scripts/audit_generalization_hypotheses.py all        # writes this directory's csv/md
```
