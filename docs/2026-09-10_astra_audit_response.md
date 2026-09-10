# Response to the 2026-09-10 independent audit

*2026-09-10, Marine Denolle with Claude, branch `audit/2026-09-07-generalization`.
Answers `docs/2026-09-10_picker_and_issue_roadmap_audit.md` (the Astra audit
of the plan and issues #33–#50). Each finding was checked against the code
and the committed data before this was written; what could be verified from
the laptop is marked so, what needs the server is marked so.*

## Verdicts, in order of consequence

**1. The training loader mislabels every source not natively sampled at
100 Hz. Confirmed; the most consequential finding on the table, and one the
2026-09-07 audit missed.**

`scripts/manifest_dataset.py` resamples the waveform to 100 Hz
(`_resample_if_needed`, line 399) and then cuts the window and builds the
Gaussian target from the manifest's `p_arrival_sample` (`_window`, line 414),
a native-rate index that is never rescaled. The chunked and single-HDF5
readers assert 100 Hz outright (lines 343–350). The native rates are in
the repository's own dataset audit, `notebooks/audit_results/summary_statistics.csv`:

| Source | Native rate | Training cap | Effect of the bug |
|---|---|--:|---|
| GEOFON | 20 Hz dominant, 20–200 Hz spread | 150,000 | P index 5× too small at 20 Hz: window cut around the wrong sample, target seconds to tens of seconds before the onset, onset trained as N |
| ETHZ | 200 Hz dominant, 100–500 Hz | 60,000 | index 2× too large |
| SCEDC | 40 and 100 Hz | 60,000 | 40 Hz part 2.5× too small |
| LenDB | 20 Hz (SeisBench converter) | 40,000 | 5× too small; matches the audit's −3.2 s probe |
| MLAAPDE | to be read on the server | 80,000 | reader asserts 100 Hz |

GEOFON alone fills the 25 % teleseismic target of `TARGET_FRACTIONS`, so on
the order of a third of the 527,477-window v7 corpus carried displaced P
targets. This explains the recall–timing seesaw, the teleseismic collapse
(v7 0.12 against the parent's 0.22 on the benchmark) and why every
augmentation that touched temporal structure made things worse, more
economically than any hypothesis in the 2026-09-07 audit. The benchmark is
not affected: on its 200 Hz ETHZ rows the parent scores P recall 0.980 and
MAE 0.115 s (`notebooks/step3_results.parquet`), so notebook 05 rescaled
correctly; only the training path is broken.

Consequence: the 2026-09-07 conclusion that "the corpus was the problem"
was stated more strongly than the evidence allowed. The v7 deficit may be a
label-alignment defect, not a data-volume limit. Whether more data helps is
untested until the loader is fixed.

**2. Unlabelled arrivals are trained as explicit negatives. Confirmed.**
`make_labels` (lines 258–266) sets N = 1 − max(P, S): an absent S or a
second event in the window becomes a confident noise target, and the
distillation term then argues with it. "Every catalogued arrival labelled"
(plan §4) narrows this but does not close it, because catalogued is not
physical. A label-validity mask or a partial-label loss belongs in #40 and
#43, and the overlapping-target rule (channel sums reaching 2.0 for
coincident peaks) must be defined before event superposition is used.

**3. The scorer critiques are correct, all of them.** Extracting picks once
at 0.02 and filtering is not extraction at each threshold, because the
trigger's off-threshold (half the on-threshold) merges adjacent peaks; both
QuakeScope notebooks share this, so the published matched-budget numbers
need re-extraction from stored annotations, and the direction of the bias
is unknown until then. The 0.5 s cross-event collapse in `reference_from`,
the greedy nearest-first matcher, the window mixing in `matched_budget`,
the helper never being called, the duplicated 0.3 row and the silently
skipped inference failures are all defects of `scripts/heldout_testset_score.py`
as committed.

**4. Reference coverage.** The audit's table is fair. The Noto swarm has no
reference on any fetched station; Hualien and Noto 2024 cannot support a
network test; the ISC unknown-mode tier must be reported apart from
reviewed-manual picks. #37 is therefore a dependency of the acceptance
gate, not enrichment.

**5. The suite freeze contradicted itself.** Kaikōura, Norcia, Thessaly and
the western sequences motivated the recipe; naming them acceptance restores
no blindness, and #35 as written selects a campaign winner on them. Known
cases become regression sets; the sealed panel is the new sequences,
characterised for evaluability only until the single acceptance run.

**6. Qualifications.** The batch-normalisation running-statistics
observation (input-layer variance ratio 0.76) is a cheap and worthwhile
ablation; a buffer swap is a diagnostic, not a candidate. "AUC-recall is not
calibration-independent" is right; it was the repository's summary
statistic and its limits were stated, but it should not carry weight in
selection. The 576 GB estimate for a 4 M-window RAM cache is correct and had
not been confronted; sharded, memory-mapped storage is a prerequisite for
#45. The reordering (small correct pilot, then scale) is the right one, and
the warning that a flat scaling curve cannot separate bad labels from
over-regularisation is exactly what finding 1 makes concrete.

## Amendments to the issues

| Issue | Amendment |
|---|---|
| #33 | Keep first. Exclude by `(dataset, chunk, trace_name)`; record source-snapshot hashes; state the policy for rows with no origin time (currently kept). |
| #34 | Becomes data-path forensics before any training: for every manifests_v2 row, native rate versus assumed rate, index misalignment in seconds, effective P and S supervision after resampling and cropping, S-only rows, fetch failures. P and S separately. |
| #35 | Blocked by the scorer rewrite and by #37 evaluability. Store annotations once and extract picks at each threshold; maximum-cardinality matching; per-window aggregation; validity masks; a structured failure table. Baselines are characterised on regression sets, never on the sealed panel. |
| #36 | Needs the persisted pick store, station coordinates, frozen associator settings; one-to-one event matching with split, merged and unmatched reported separately. |
| #37 | Promoted to an acceptance dependency; zero-overlap cases are marked not evaluable. |
| #40 | Rate and index invariants, all-arrival lists, S-only crops and label-validity masks before any volume target; composition targets are pilot hypotheses. |
| #41 | Rewritten: `label_error_filter.py` downloads existing trace-name reports and does not run confident learning; its multiplet flags mark additional arrivals to keep, not rows to drop. |
| #42 | Noise ontology separated: background noise, task-excluded sources, unlabelled intervals, reviewed negatives; aftershock coda and LFEs are not "N" by catalogue absence. |
| #43 | After the loader fix. SNR from defined pre- and post-arrival windows; anti-aliased rate transforms; label shifts and masks; overlapping-target rule. |
| #44 | Moved to Phase 0. Regression sets (already examined) versus a sealed panel (never scored); every scoring entry point guarded; hashes logged. |
| #45 | Deferred behind a small faithful fine-tune on correctly aligned labels (the audit's Gate C: alignment fix, missing-label handling, BN freeze versus adaptation, anchor on and off, real noise on and off; several seeds; per-term loss logging). E1 then runs with event-grouped nested subsets and disk-backed shards. |
| #46 | A small version moves ahead of E1; at scale a full init × distillation factorial with three seeds. |
| #48 | Inference contract first: SeisBench PhaseNet forwards at 3001 and 6000 samples and fails at 6001, and `in_samples` stays 3001; a config change does not deploy 60 s context. |
| #49 | Gate strengthened: pre-registered practical margin and per-regime non-inferiority on independent blocks; one primary candidate; frozen thresholds and associator; a failed panel becomes development. |
| #50 | Verify the cached `geofon` weights' rate before proposing 20 Hz input; retrain the transfer; overlap and deduplication at the 300 km handoff. |

## What happens next, in order

1. Fix `ManifestDataset` to carry the native rate through the pick index,
   with fixtures at 20, 40, 50, 100 and 200 Hz, missing channels, S-only
   rows and crop-boundary offsets; audit the HDF5 readers' rate assumptions.
2. On the server, count the misaligned rows of manifests_v2 (#34).
3. Rewrite the scorer (item 3) and re-extract the three published
   sequences from stored annotations.
4. Split #44; re-label the existing cases as regression sets.
5. Gate C on a small, correctly labelled corpus, before E1.

The v7 checkpoint checks in the audit (student equals the deployed export,
teacher equals the parent, ZNE/PSN/std/filter factor 2) are accepted as
settled.
