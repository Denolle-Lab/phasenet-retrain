# Applied training roadmap

GitHub issues #33–#50 were updated on 2026-09-10. Rendered offline by `scripts/open_plan_issues.py` from the verified [applied snapshot](issue_revision_2026-09-10/applied_roadmap.json).

See the [execution branches and order](2026-09-10_issue_execution.md). Earlier issue descriptions and scientific proposals are historical; the checkpoint gates below govern new work.

## Phase 0: debugging and trustworthy evaluation

Gate-based debugging and trustworthy evaluation: verified waveform/label/timebase and model contracts, corrected continuous scoring, eligible suites, calibrated pick and event baselines. Issues 33-38 and 44; release early checkpoints without waiting for whole-issue closure.

### [#33 — [Phase 0] Enforce versioned exclusions and trace/event identity across every data path](https://github.com/Denolle-Lab/phasenet-retrain/issues/33)

**Why this revision.** The exclusion file must certify a particular source snapshot and suite policy, not merely exist.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- [ ] **33A — Exclusion contract released:** [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Preserve `(dataset, chunk, trace_name)` and canonical event identity. Hash the source metadata, registry, exclusion rules and resulting manifests. Apply exclusions to signal, validation, negative examples, augmentation pools and hard-negative mining. Audit duplicate events across source datasets and overlapping waveform intervals. Require a documented quarantine policy for rows whose time/location/identity cannot establish independence; any explicitly allowed unknown rows remain outside independence claims.

**Acceptance.**

- A versioned exclusion bundle and per-source removal/unknown counts are committed and checked by every builder.
- Reused trace names in different chunks are distinguished; equivalent events across datasets remain in one split.
- Fixtures cover missing/stale lists, unknown coordinates/times, year/place exclusions, cross-source duplicates and noise-window overlap.
- Historical manifests remain immutable. Exclusions constrain new training; they do not delay read-only forensics in #34.

**Dependency detail.** 33A uses the initial suite policy 44A. Subsequent suite additions invalidate the relevant exclusion bundle and require a new hash. Server-wide counts are required for a server-wide certificate; a verified pilot snapshot may be certified separately.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#34 — [Phase 0] Repair and quantify waveform–label, timebase and checkpoint-contract defects](https://github.com/Denolle-Lab/phasenet-retrain/issues/34)

**Why this revision.** Promote from hypothesis tables to the principal debugging issue, including the benchmark path.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- [ ] **34A — Loader invariants pass:** no upstream checkpoint.
- [ ] **34B — Historical row-level forensics complete:** no upstream checkpoint.
- [ ] **34C — Model and benchmark contracts verified:** [34A](https://github.com/Denolle-Lab/phasenet-retrain/issues/34).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Distinguish original recording rate, effective rate returned by the reader, and the coordinate system of each arrival index. Rescale waveform and indices exactly once. Read HDF5 rate/component metadata instead of assuming 100 Hz; handle bucketed names and trim to actual trace support. Support S-only rows. Replace fetch-error zero/noise substitution with a rejected-row ledger and failure gates.

**Acceptance.**

- **34A:** Fixtures at 20/40/50/100/200 Hz plus a non-integer rate preserve absolute P/S times within one target sample, including crop boundaries, missing channels and explicitly missing rates. Check 120/250/500 Hz when represented by production sources.
- **34B:** An immutable row-level report for the actual v7 train/val/test manifests records source/effective/assumed rates, old/new offsets, effective labels after crop, outside-window arrivals and fetch status. Aggregate by source, distance and phase; report unique rows and training exposures separately. Restore H2 and noise-pool ordering as secondary tables with independent negative populations.
- **34C:** Verify raw source → benchmark window → seconds against independent timestamps. Notebook 05's ETHZ=100 and MLAAPDE=40 assumptions both need source evidence. Check crop-cache invalidation, array orientation, normalization and parent → wrapper → export probability parity at the pinned runtime. Hash the actual deployed pair, not only the laptop cache.
- Quantify affected rates/rows before publishing a corpus fraction. Revise historical result provenance if the benchmark changes.

**Dependency detail.** A and B start immediately. C follows the local reader contract; server evidence remains necessary to certify historical artifacts. This issue does not itself establish which defect caused v7's generalization loss; #46 supplies controlled training.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#35 — [Phase 0] Correct continuous pick scoring and recompute regression/development baselines](https://github.com/Denolle-Lab/phasenet-retrain/issues/35)

**Why this revision.** Separate scorer implementation from data acquisition and calibrated comparison; never run --all over sealed acceptance cases.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- [ ] **35A — Scoring engine released:** [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).
- [ ] **35B — Eligible baseline annotations and picks persisted:** [35A](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [34C](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [37A](https://github.com/Denolle-Lab/phasenet-retrain/issues/37).
- [ ] **35C — Calibrated baseline comparison complete:** [35B](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [38A](https://github.com/Denolle-Lab/phasenet-retrain/issues/38).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Store annotations once and invoke the production trigger extractor independently at each threshold. Deduplicate bulletin representations by event/station/phase identity, preserving distinct close events. Match maximum cardinality first, minimum residual second. Store picks, reference assignments, unmatched candidates, coverage and failures. Aggregate counts within model/phase/threshold across explicitly identified windows; remove duplicate thresholds and use actual attained operating points.

**Acceptance.**

- **35A:** Tests cover threshold-induced peak splitting, two close events, greedy-matching counterexamples, duplicate references, multiple windows, gaps, model failures and zero-reference cases. Failure/coverage policy is explicit and cannot silently make a candidate look better.
- **35B:** Parent, v7, instance and available v11 produce versioned artifacts on regression/development only. Record training-domain/parent-overlap provenance for every baseline; Norcia is not automatically independent of INSTANCE weights. Existing QuakeScope tables are re-extracted from annotations if saved; otherwise rerun inference. Old tables remain archived and marked superseded. Ensembles align phase labels, time grids and valid support while retaining each member's preprocessing.
- **35C:** Publish P/S recall, residual distributions, matched total-pick workload and separately calibrated nuisance-pick budgets, with paired block uncertainty and unavailable models identified.
- Persist enough information for #36 to associate the exact emitted picks.

**Dependency detail.** Scorer development starts using synthetic fixtures without #37. Baseline inference needs only the particular cases certified by 37A, not every access request. Threshold curves precede 38A; calibrated comparison follows it. #35 does not authorize replacing the campaign picker; #49 owns release.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#36 — [Phase 0] Validate event association and catalogue-relative recovery on continuous data](https://github.com/Denolle-Lab/phasenet-retrain/issues/36)

**Why this revision.** An event-level endpoint is necessary, but it must not consume sealed La Palma/West Bohemia results during debugging.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- [ ] **36A — Association/matching engine released:** [35A](https://github.com/Denolle-Lab/phasenet-retrain/issues/35).
- [ ] **36B — Development event baselines complete:** [36A](https://github.com/Denolle-Lab/phasenet-retrain/issues/36), [35C](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [37A](https://github.com/Denolle-Lab/phasenet-retrain/issues/37).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Associate the persisted picks with PyOcto under versioned regional travel-time models and station metadata. Freeze associator parameters across picker comparisons; tune them only on development. Match predicted/catalogue events one-to-one with explicit origin-time, distance and depth tolerances, retaining diagnostics for association splits and merges.

**Acceptance.**

- **36A:** Synthetic cases and a reviewed small catalogue establish matching behavior for missing stations, duplicate events, false associations, splits and merges.
- **36B:** Report recovery versus magnitude, hour after mainshock and day of swarm, plus station support, valid exposure and unassociated/unmatched counts. Use Kaikōura as a known regression and Etna/Corinth-Thiva as development examples if 37A certifies them; choose substitutes before scoring if necessary.
- Report catalogue completeness limitations and independently review a sample of unmatched events. “Absent from the catalogue” is not automatically a false event.
- Paired uncertainty uses independent event/time blocks; P/S observations and stations belonging to the same event remain linked.
- First-48-hour or migration claims require the corresponding time coverage; busy-hour excerpts alone cannot pass those claims.

**Dependency detail.** Build the engine after the 35A artifact schema. Final development comparisons use calibrated picks from 35C. Sealed event-level baselines and candidate results are generated together under #49.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#37 — [Phase 0] Certify test-case evaluability and close targeted reference/waveform gaps](https://github.com/Denolle-Lab/phasenet-retrain/issues/37)

**Why this revision.** Replace a binary built/unobtainable list with eligibility for specific scientific claims.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0 for eligibility; access work case-dependent.

**Checkpoint dependencies.**

- [ ] **37A — Existing regression/development cases certified:** [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).
- [ ] **37B — Proposed sealed panel eligibility frozen:** [37A](https://github.com/Denolle-Lab/phasenet-retrain/issues/37).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** For each case produce valid component/station/time coverage, reference overlap, P/S counts, event support and reference-provenance tiers. Resolve NET.STA.LOC/channel/epoch with inventory and coordinates; quarantine ambiguous station codes. Preserve gap masks and report availability independently of model performance.

**Acceptance.**

- **37A:** Existing cases are explicitly eligible for pick scoring, network/event scoring, both, or neither. Noto swarm's current zero overlap and Hualien's one-station coverage cannot be treated as acceptance passes.
- **37B:** The proposed sealed panel has an immutable eligibility table, minimum support/precision requirements per endpoint and predeclared fallback cases. No candidate scores are examined to choose cases.
- Retain the Hi-net, CWA GDMS, IMO, AFAD, Zagreb, WEBNET and INGV-OV acquisition tasks, each with owner, access state and expected eligibility gain. Marine handles external account/contact actions when authorized.
- Unknown-mode bulletin picks, explicitly manual picks and automatic picks remain separate provenance tiers. Hash pinning is distinct from reference validation.
- Extend selected cases to early coda, later hours/days and ordinary conditions where those claims are planned; record station-selection bias.

**Dependency detail.** Existing open cases unblock #35 immediately when certified. Unresolved access blocks only the affected endpoint/case; #49 requires a complete, predefined adequate panel. A case replaced before scoring still requires exclusion-policy verification in #33/#44.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#38 — [Phase 0] Calibrate P/S operating points on independent station-days](https://github.com/Denolle-Lab/phasenet-retrain/issues/38)

**Why this revision.** This issue was not amended in Fable's response; it currently conflates catalogue absence with false picks and has no calibration/test separation.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- [ ] **38A — Calibration protocol and baseline thresholds released:** [35B](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [37A](https://github.com/Denolle-Lab/phasenet-retrain/issues/37), [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Define calibration days separately from training/mining, development evaluation and sealed acceptance. Sample quiet and disturbed conditions by region, station/instrument class and season. Use corrected threshold-specific extraction from #35, measured valid exposure and independent review of unmatched picks.

**Acceptance.**

- A protocol specifies P/S nuisance-pick budgets, uncertainty targets, minimum exposure and a fallback for regions with too few calibration data. Numeric values are fixed before evaluating candidates.
- Publish thresholds and attained rates on calibration days; report achieved rates on distinct evaluation days without retuning.
- Where no reviewed truth exists, call the metric “unmatched-pick rate,” retain reviewed-sample uncertainty, and do not equate it with false-positive rate.
- Candidate-specific thresholds are refit by the same frozen procedure on calibration data only. Parent/candidate event comparisons share the associator.
- Retain both total emitted workload and nuisance-pick rate; matching one does not imply matching the other.

**Dependency detail.** Requires the baseline annotations/picks checkpoint 35B, not closure of #35. Its output unblocks 35C. Station-day calibration may proceed without the final eleven-class training-noise corpus.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#44 — [Phase 0] Separate regression, development, calibration and sealed acceptance roles](https://github.com/Denolle-Lab/phasenet-retrain/issues/44)

**Why this revision.** Amend the actual registry/policy and all entrypoints, not just add an override flag or another issue comment.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0; begin now.

**Checkpoint dependencies.**

- [ ] **44A — Initial roles and read policy released:** no upstream checkpoint.
- [ ] **44B — Eligible sealed panel and decision rule frozen:** [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44), [37B](https://github.com/Denolle-Lab/phasenet-retrain/issues/37).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Mark Kaikōura, Norcia, Thessaly and previously inspected western cases as regression. Keep Samos, Adriatic, Etna and Corinth-Thiva as development, with additional geographically useful development cases assigned before scoring. Define disjoint calibration station-days. Treat new sequences as sealed only after checking whether model results were previously viewed anywhere.

**Acceptance.**

- **44A:** Registry roles and access logs distinguish metadata/reference QA from model-performance access. All CLI/notebook/API scoring entrypoints use the same policy; `--all` excludes sealed cases by default.
- **44B:** After #37's metadata-only eligibility review, freeze the panel, reference/model-provenance strata, fallback policy, practical effect margins, non-inferiority endpoints and paired uncertainty rule.
- Record hashes for suite, source references, code, candidate and configuration at every authorized scoring run. Years are exclusion rules, not sequence keys.
- A changed/new sealed case triggers recertification of training exclusions. A consumed panel cannot be restored by renaming; future selection needs fresh acceptance cases.
- Review of sealed picks for scientific quality is allowed without candidate predictions; it does not authorize tuning models or thresholds on that panel.

**Dependency detail.** 44A starts immediately and unblocks infrastructure and read-only census. 44B waits for eligibility, not access to every desired archive. 44B precedes pilot/model selection on the new roadmap; debugging with synthetic/regression fixtures can proceed meanwhile.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

## Phase 1: validated corpora and controlled pilots

Validated corpora and controlled pilots: reviewed labels and noise, consistent augmentation, bounded-memory storage, and a same-row alignment diagnostic before recipe selection. Issues 39-43 and 46; expanded acquisition follows pilot evidence.

### [#39 — [Phase 1] Census usable bulletin supervision, source diversity and acquisition cost](https://github.com/Denolle-Lab/phasenet-retrain/issues/39)

**Why this revision.** This unamended issue needs a census of useful independent labels, rather than extrapolation from a single week.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1; start early.

**Checkpoint dependencies.**

- [ ] **39A — Stratified source census released:** [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Sample multiple periods per operator/year: quiet intervals, ordinary seismicity, aftershock activity and relevant volcanic/swarm regimes. Record native/effective rate, channel identity, units/response state, provenance, uncertainty, P-only/S-only/P+S availability, event multiplicity and waveform accessibility.

**Acceptance.**

- Publish operator/year counts with sampling uncertainty and separate total rows, unique events, stations and duplicate events shared across corpora.
- Estimate low-SNR and multi-event availability using defined measurements; uncertain label completeness remains explicit.
- Identify at least one development region/operator outside the existing European concentration, subject to #44's role decision before scoring.
- Report query/storage/processing costs and licensing/access constraints; nominate pilot sources based on useful supervision and diversity.
- Treat composition targets and the 4M maximum as hypotheses to test, not quantities the census must force.
- Apply existing held-out rules to estimates, including source-year/place overlaps.

**Dependency detail.** Read-only reconnaissance starts early with 44A. The training-ready pilot additionally needs #33, #34 and #41; a census alone does not certify a usable corpus.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#40 — [Phase 1] Build a versioned partial-label corpus and scalable waveform store](https://github.com/Denolle-Lab/phasenet-retrain/issues/40)

**Why this revision.** The corpus must preserve time, all known arrivals and unknown-label regions; scaling must not depend on a dense RAM cache.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1.

**Checkpoint dependencies.**

- [ ] **40A — Small validated corpus and target schema released:** [33A](https://github.com/Denolle-Lab/phasenet-retrain/issues/33), [34A](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [39A](https://github.com/Denolle-Lab/phasenet-retrain/issues/39), [41A](https://github.com/Denolle-Lab/phasenet-retrain/issues/41).
- [ ] **40B — Sharded corpus and composition audit released:** [40A](https://github.com/Denolle-Lab/phasenet-retrain/issues/40), [41B](https://github.com/Denolle-Lab/phasenet-retrain/issues/41), [46B](https://github.com/Denolle-Lab/phasenet-retrain/issues/46).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Store waveform intervals with explicit sample rate, absolute start time, component identity and valid support. Store lists of all known arrivals with event ID, phase family, provenance/uncertainty and label-validity information. Include P-only, S-only, noise and multi-event crops. Harvest arrivals capable of reaching the station from before/outside the nominal event query window.

**Acceptance.**

- **40A:** A small reviewed pilot passes loader/time invariants. The target contract distinguishes observed absence from unlabelled time and defines normalized overlapping P/S targets or an explicitly tested alternative loss. A mask excludes uncertain supervision without accidentally penalizing the missing phase through softmax normalization.
- **40B:** Disk-backed shards or memory mapping support worker-local reads, raw/long-window caching and labels generated per batch. Validate a bounded-memory stress run before millions of windows.
- Report effective P/S, SNR, distance, native-rate, event-count, operator and station mix after cropping/augmentation, plus unique-event counts and removal reasons.
- Preserve immutable source and manifest hashes; maintain event/station/operator transfer splits with overlap audits.
- Keep stored context independent of training context. Begin diagnostics at the parent's verified 3001-sample contract; 60 s training requires #48's explicit inference contract.

**Dependency detail.** 40A is a pilot release, not the full corpus. Full expansion 40B follows successful label review and the small recipe pilot 46B; #46 does not wait for 40B.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#41 — [Phase 1] Define label validity and repair uncertain or incomplete arrivals](https://github.com/Denolle-Lab/phasenet-retrain/issues/41)

**Why this revision.** The existing helper downloads exclusions; it neither runs new confident learning nor justifies rejecting every multiplet.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P0 policy; P1 review.

**Checkpoint dependencies.**

- [ ] **41A — Target/provenance and review policy released:** [39A](https://github.com/Denolle-Lab/phasenet-retrain/issues/39).
- [ ] **41B — Pilot labels reviewed and repaired:** [40A](https://github.com/Denolle-Lab/phasenet-retrain/issues/40).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Define provenance tiers, accepted phase families and task scope with a domain reviewer. Model disagreement and physical residuals prioritize review; they are not ground truth. Review examples without exposing which candidate produced the suggestion. If prediction-based triage is used, produce it out of fold.

**Acceptance.**

- **41A:** A documented decision table distinguishes valid labels, additional unlabelled arrivals, uncertain timing/phase, source artefacts and unusable records. Specify loss/mask behavior for each.
- **41B:** The pilot has per-operator reviewed-sample results, corrected arrivals or explicit uncertainty masks, and retained/removed/quarantined counts with reasons.
- Existing multiplet flags route examples to relabelling or uncertainty handling. Do not automatically discard them, and do not retain irreparable labels simply because multiplets are desired.
- S–P checks account for depth, distance, velocity uncertainty and Pg/Pn/Sg/Sn conventions; emergent S and LFEs are not rejected solely because the parent misses them.
- Training-label review and sealed reference review are separate records. Model-based review must not mine the sealed panel.

**Dependency detail.** Release the policy before constructing 40A. Review 40A before 46B/40B. The new-corpus review implementation must be built explicitly; running `label_error_filter.py` alone cannot close this issue.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#42 — [Phase 1] Build reviewed background and uncertain-event pools with station/time holdouts](https://github.com/Denolle-Lab/phasenet-retrain/issues/42)

**Why this revision.** Retain natural-noise diversity without teaching the picker to suppress legitimate physical arrivals.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1.

**Checkpoint dependencies.**

- [ ] **42A — Noise ontology and pilot pools released:** [33A](https://github.com/Denolle-Lab/phasenet-retrain/issues/33), [34A](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [41A](https://github.com/Denolle-Lab/phasenet-retrain/issues/41), [44A](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).
- [ ] **42B — Expanded class/station/season census released:** [42A](https://github.com/Denolle-Lab/phasenet-retrain/issues/42), [47A](https://github.com/Denolle-Lab/phasenet-retrain/issues/47).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Separate background noise, reviewed negatives, task-excluded sources, and uncertain event-bearing intervals. Preserve the eleven environmental/site categories as metadata, not automatic all-N labels. Aftershock coda may contain arrivals; LFEs and blasts may contain meaningful P/S phases. Fix the general-picker/source-classifier boundary in 41A.

**Acceptance.**

- **42A:** Reviewed pilot examples and uncertainty masks exist for each included class; no label depends solely on catalogue absence or the parent's confidence.
- **42B:** Expanded pools have station/time/season/source provenance, valid exposure, composition summaries, holdout lists and overlap checks against all evaluation/calibration intervals.
- Any existing noise_global data retain their original model-screening history and are independently reassessed; they cannot silently become an unbiased negative set.
- Keep OBS excluded under the current scope. Do not impose 20k/class until class validity and useful diversity are established.
- Training/mining and evaluation pools are distinct. Rate-preserving waveform operations follow #34, and pools use #40's storage pattern.

**Dependency detail.** A small pilot supports #43/#47 before full harvest. Expand after 47A demonstrates that the intended use adds value or after a documented decision to expand for coverage, rather than interpreting failed augmentation as a mandate for volume.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#43 — [Phase 1] Implement time-consistent augmentation with explicit label and SNR rules](https://github.com/Denolle-Lab/phasenet-retrain/issues/43)

**Why this revision.** Augmentation should be switchable and verifiable independently of whether it improves the model.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1.

**Checkpoint dependencies.**

- [ ] **43A — Augmentation transforms and fixtures released:** [34A](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [40A](https://github.com/Denolle-Lab/phasenet-retrain/issues/40), [42A](https://github.com/Denolle-Lab/phasenet-retrain/issues/42).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Apply random cropping/placement at sample access, transform arrival lists and validity masks together, and normalize consistently with deployment after physically meaningful mixing. Implement real-noise addition, controlled multi-event superposition, component dropout, valid anti-aliased bandwidth/rate changes and separately switchable artefact groups.

**Acceptance.**

- Time transformations preserve absolute arrival time and mask support within one output sample; superposition preserves distinct event IDs and uses the normalized target rule from #40.
- Define signal/noise measurement windows and target achieved SNR after mixing; handle already noisy source waveforms and cases where clean-signal SNR is unidentifiable.
- Do not treat whole-window unit variance as clean signal amplitude. Preserve relative component amplitudes and require compatible units/response/bandwidth before mixing.
- Deterministic fixtures cover gaps, clipping, missing components, overlapping phases and crop boundaries. Stored seeds reproduce mixtures and sampling.
- An inspection sheet includes untouched and augmented real examples. Every group can be disabled in config; untouched/noise fractions are explicit tunable values.
- Augmentation never samples calibration/development/acceptance intervals.

**Dependency detail.** 43A can be built from pilot pools; it does not depend on full 40B/42B or #45. Transform correctness is assessed here; scientific benefit is assessed in #47.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#46 — [Phase 1] Run controlled alignment and recipe pilots before initialization/KD scaling](https://github.com/Denolle-Lab/phasenet-retrain/issues/46)

**Why this revision.** Own causal attribution explicitly; do not bundle alignment, labels, BN, augmentation and dataset expansion into the first retrain.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P0 first pilot; P2 later confirmation.

**Checkpoint dependencies.**

- [ ] **46A — Paired alignment diagnostic complete:** [33A](https://github.com/Denolle-Lab/phasenet-retrain/issues/33), [34A](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [34B](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [34C](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [35C](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [44B](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).
- [ ] **46B — Small corrected recipe selected:** [46A](https://github.com/Denolle-Lab/phasenet-retrain/issues/46), [40A](https://github.com/Denolle-Lab/phasenet-retrain/issues/40), [41B](https://github.com/Denolle-Lab/phasenet-retrain/issues/41).
- [ ] **46C — Matched initialization/KD confirmation complete:** [45B](https://github.com/Denolle-Lab/phasenet-retrain/issues/45).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work and acceptance.**
- **46A:** Compare the same parent initialization and the same verified, exclusion-clean row list under legacy versus corrected time/index handling. Use the repaired reader in both arms, deliberately toggling only legacy versus corrected time/index mapping; do not reproduce unrelated silent fetch failures. Hold loss, KD, BN policy, augmentation, batch size and update budget fixed. Use three paired seeds and untouched verified evaluation; keep old v7 as a historical baseline, not the experimental control. Restrict deliberate faulty-label reproduction to this diagnostic; never deploy it. A 100-Hz-only stratum is a useful control but cannot replace the same-row comparison because source mix changes.
- Report effects by affected/unaffected rate strata, source, distance and phase. If historical data cannot be reconstructed, document the causal limit rather than declaring the bug explains all loss.
- **46B:** On the small reviewed corpus, test a minimal corrected recipe, then controlled soft-label/partial-label handling, BN frozen versus adaptive, and KD off/on. Separate BN statistics from affine-parameter freezing; treat replay as an optional additional arm with certified exclusions. Use identical row lists and paired seeds for each comparison; log loss components, gradients, learning rate, effective labels and continuous validation.
- **46C:** At #45's chosen size, compare matched-width parent/scratch initialization and KD off/on explicitly. Scratch+KD is a defined arm, not an accidental implementation. Match data and report compute costs.

**Dependency detail.** 46A uses the verified existing corpus and does not wait for new bulletin harvest. 46B needs only pilot 40A/41B, not bulk 40B. Noise/augmentation benefit is #47. Only 46C depends on #45, eliminating the original E1/E2 cycle.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

## Phase 2: scaling and generalization experiments

Scaling and generalization experiments: conditional data-scaling evidence, controlled augmentation and initialization confirmations, deployable width/context decisions and measured compute cost. Issues 45, 47 and 48; pilot checkpoints precede large runs.

### [#45 — [Phase 2] Measure data scaling for a validated recipe under bounded memory and compute](https://github.com/Denolle-Lab/phasenet-retrain/issues/45)

**Why this revision.** The scaling curve must follow causal debugging, not serve as a test of several unvalidated assumptions at once.

**Milestone:** Phase 2: scaling and generalization experiments. **Priority:** P2; defer large runs.

**Checkpoint dependencies.**

- [ ] **45A — Scaling design and throughput budget frozen:** [40B](https://github.com/Denolle-Lab/phasenet-retrain/issues/40), [42B](https://github.com/Denolle-Lab/phasenet-retrain/issues/42), [46B](https://github.com/Denolle-Lab/phasenet-retrain/issues/46), [47A](https://github.com/Denolle-Lab/phasenet-retrain/issues/47).
- [ ] **45B — Scaling curve and next-size decision complete:** [45A](https://github.com/Denolle-Lab/phasenet-retrain/issues/45).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Select a validated recipe and event-grouped nested subsets with controlled source/phase/noise composition. Start at the available pilot size; 0.25/0.5/1/2/4M are possible tiers, not mandatory commitments. Distinguish unique events/stations/windows from repeats and synthetic mixtures.

**Acceptance.**

- **45A:** Memory/throughput measurements support the next tier. The old 4M × 6000 × 3 × float32 waveform-plus-label cache is about 576 GB before other allocations; it is not the implementation plan.
- **45B:** Use at least three seeds for decision-making tiers and report matched-budget continuous metrics by regime, with baseline/pilot included.
- Report fixed-epoch results and a matched-update or matched-compute comparison; larger data change optimization exposure as well as diversity.
- Keep development/threshold calibration rules fixed and log model-selection compute. Stop/escalate based on predefined practical margins and uncertainty, not a blanket “2M failure means fine-tuning cannot work.”
- A flat curve is conditional on this recipe/data mixture. Diagnose whether label quality, coverage or optimization limits interpretation before acquiring the next tier.

**Dependency detail.** This waits for checkpoints 46B and 47A, not for closure of their later full-scale confirmation work. If a signal-only recipe is selected, 42B may be explicitly waived with that scope recorded before 45A; negative evaluation remains mandatory.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#47 — [Phase 2] Test real-noise and multi-event augmentation before full-scale confirmation](https://github.com/Denolle-Lab/phasenet-retrain/issues/47)

**Why this revision.** This unamended issue belongs partly before scaling and needs no-augmentation and uncertainty-handling controls.

**Milestone:** Phase 2: scaling and generalization experiments. **Priority:** P1 pilot; P2 confirmation.

**Checkpoint dependencies.**

- [ ] **47A — Pilot augmentation value established:** [46B](https://github.com/Denolle-Lab/phasenet-retrain/issues/46), [42A](https://github.com/Denolle-Lab/phasenet-retrain/issues/42), [43A](https://github.com/Denolle-Lab/phasenet-retrain/issues/43).
- [ ] **47B — Selected ablations confirmed at chosen scale:** [45B](https://github.com/Denolle-Lab/phasenet-retrain/issues/45), [47A](https://github.com/Denolle-Lab/phasenet-retrain/issues/47).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Start from the selected small corrected recipe, then compare no added noise, measured real-noise mixtures and white-noise control; separately test event superposition, non-stationary/artefact groups and class balancing. Change one interpretable group at a time before testing important interactions.

**Acceptance.**

- **47A:** Paired seeds, equal source examples/update budgets and explicit untouched fractions isolate augmentation effects. Report P/S recall on weak and close arrivals, nuisance picks by noise class, valid station-day exposure and timing tails.
- Demonstrate that target uncertainty/merged labels remain correct; do not let an augmentation ablation accidentally change label policy.
- Evaluate on untouched real continuous data at held-out stations/times, including disturbed conditions absent from the added-noise recipe.
- **47B:** Repeat the shortlisted informative controls at the selected scale. Record harmful or null effects as results; no augmentation group is mandatory because it sounds physically realistic.
- Any hard-negative mining uses a separate training pool and records iteration and source; calibration/development/acceptance data are never recycled into it.

**Dependency detail.** 47A precedes #45 and full noise harvest 42B. 47B follows #45; #45 must not depend on closure of the whole issue.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#48 — [Phase 2] Validate and compare deployable PhaseNet width, context and ensembles](https://github.com/Denolle-Lab/phasenet-retrain/issues/48)

**Why this revision.** A direct forward shape or YAML length is insufficient to establish continuous inference at that context.

**Milestone:** Phase 2: scaling and generalization experiments. **Priority:** P2; context contract before any long-window run.

**Checkpoint dependencies.**

- [ ] **48A — Alternative context/export contract released:** [34C](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [35A](https://github.com/Denolle-Lab/phasenet-retrain/issues/35).
- [ ] **48B — Width/context experiments and deployment cost complete:** [48A](https://github.com/Denolle-Lab/phasenet-retrain/issues/48), [45B](https://github.com/Denolle-Lab/phasenet-retrain/issues/45).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Pin the SeisBench implementation and explicitly configure training length, `in_samples`, `pred_sample`, overlap, padding/blinding and exported metadata. Test supported lengths before choosing the 60 s arm. The prior 6001-sample failure is evidence about the tested runtime, not a timeless architectural law.

**Acceptance.**

- **48A:** Direct forward, continuous annotation and save/reload agree on dimensions, time alignment and probabilities at each supported context, including edges and gaps. Keep original 3001-sample diagnostics independent of this optional extension.
- **48B:** Compare width and context separately where possible, holding data/recipe fixed. A narrow network cannot inherit incompatible wide weights without a specified transfer; use matched scratch comparisons or a documented compatible initialization to isolate width.
- Report effective receptive-field/context interpretation, P/S performance in late-S/close-event regimes, memory, latency and throughput.
- Ensembles align phases/time support and retain each model's normalization; weights/thresholds are selected on development/calibration only.
- Negative or null results close the experiment with a decision. Acceptance does not require trying every context or deploying an ensemble.

**Dependency detail.** 48A can run early; #40 stores long/raw intervals without requiring 60 s model training. 48B follows scaling. #49 accepts a documented decision to retain the original width/context.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

## Phase 3: acceptance and deployment

Acceptance and deployment: one frozen primary candidate, preregistered practical improvement and per-regime non-inferiority, verified deployment artifact and rollback. Issues 49 and 50; the distant-P line has a separate acceptance and handoff decision.

### [#49 — [Phase 3] Execute a sealed acceptance decision and verify the deployed artifact](https://github.com/Denolle-Lab/phasenet-retrain/issues/49)

**Why this revision.** Replace correlated five-of-six wins and multiple candidate reads with one interpretable release decision.

**Milestone:** Phase 3: acceptance and deployment. **Priority:** P2; protocol fixed earlier.

**Checkpoint dependencies.**

- [ ] **49A — Primary candidate and frozen release package selected:** [35C](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [36B](https://github.com/Denolle-Lab/phasenet-retrain/issues/36), [37B](https://github.com/Denolle-Lab/phasenet-retrain/issues/37), [38A](https://github.com/Denolle-Lab/phasenet-retrain/issues/38), [44B](https://github.com/Denolle-Lab/phasenet-retrain/issues/44), [45B](https://github.com/Denolle-Lab/phasenet-retrain/issues/45), [46C](https://github.com/Denolle-Lab/phasenet-retrain/issues/46), [47B](https://github.com/Denolle-Lab/phasenet-retrain/issues/47), [48B](https://github.com/Denolle-Lab/phasenet-retrain/issues/48).
- [ ] **49B — Acceptance decision and deployment verification complete:** [49A](https://github.com/Denolle-Lab/phasenet-retrain/issues/49).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Freeze one primary candidate, the parent/declared baselines, thresholds, associator, suite/reference hashes and software/container versions before the sealed run. Record approved omissions of optional experiments before candidate selection; an unrun experiment cannot be reported as a pass.

**Acceptance.**

- **49A:** The preregistered endpoint policy contains numeric practical improvement/non-inferiority margins, minimum usable coverage, primary/secondary metrics, paired block uncertainty and handling of missing cases/multiple endpoints.
- **49B:** Score baseline and candidate together on the sealed panel once. Require the planned overall benefit and regime-specific non-inferiority; report pick workload, nuisance-rate evidence, event recovery, timing tails, operational failures and computational cost.
- Insufficient independent events/days yields “inconclusive,” not “no degradation.” Do not treat P/S pairs from one sequence as independent replicates.
- Hash and reload the exported artifact offline, verify preprocessing/probability parity on fixed fixtures, and run the production notebooks at frozen settings. Stage rollout with monitored rates and a reproducible rollback to the parent.
- A failed or inconclusive acceptance result retains the parent. Once results inform another development cycle, that panel is consumed and a fresh one is required.

**Dependency detail.** #35 characterizes baselines without shipping them. Any earlier no-training baseline replacement must use a separate preregistered acceptance decision and consume its own panel; it cannot silently reuse the future sealed panel.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).

### [#50 — [Phase 3] Pilot distant-P transfer with a verified timebase and regional handoff](https://github.com/Denolle-Lab/phasenet-retrain/issues/50)

**Why this revision.** Keep sparse-region/offshore coverage as a separate experiment, without assuming 20 Hz transfer or catalogue completeness.

**Milestone:** Phase 3: acceptance and deployment. **Priority:** P2; independent pilot after shared contracts.

**Checkpoint dependencies.**

- [ ] **50A — Distant-P timebase/data pilot validated:** [33A](https://github.com/Denolle-Lab/phasenet-retrain/issues/33), [34C](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), [35A](https://github.com/Denolle-Lab/phasenet-retrain/issues/35), [38A](https://github.com/Denolle-Lab/phasenet-retrain/issues/38), [41A](https://github.com/Denolle-Lab/phasenet-retrain/issues/41), [42A](https://github.com/Denolle-Lab/phasenet-retrain/issues/42), [44B](https://github.com/Denolle-Lab/phasenet-retrain/issues/44).
- [ ] **50B — Distant-P acceptance and handoff decision complete:** [50A](https://github.com/Denolle-Lab/phasenet-retrain/issues/50).

A checkpoint such as `34A` means checkpoint A in issue #34, not closure of the entire issue. Follow checkpoint-level dependencies; whole-issue blocking links can create artificial cycles.

**Work.** Inspect the actual geofon checkpoint's sample rate, preprocessing and input contract. Compare verified native-rate deployment with a deliberately retrained 20 Hz/long-window candidate; feeding 20 Hz arrays to unchanged 100 Hz weights changes physical time/frequency scales. Apply the same arrival-index and metadata invariants as #34.

**Acceptance.**

- **50A:** Define P branch coverage (e.g. P/Pn and later branches), magnitude/distance support, source-label provenance and off-Japan/held-out-year evaluation. Validate proposed context through annotation/export, not forward alone.
- Include cultural, instrument and coda conditions at continental sparse-network stations, alongside coastal/microseism/polar noise.
- Predefine overlapping regional/distant eligibility around the nominal 300 km handoff, deduplicate common picks/events and keep appropriate travel-time uncertainty in association.
- **50B:** Freeze a separate candidate and sealed panel; evaluate station/event recovery and nuisance workload, including double detections and handoff misses. Report recovery relative to reference completeness; claiming a new completeness magnitude requires additional evidence.
- Preserve the no-OBS scope. A failed distant-P pilot does not block a valid main-picker release, and a successful main picker does not certify the distant-P branch.

**Dependency detail.** This shares infrastructure and label policy but not the full #45–#48 training sequence. It owns its own candidate freeze, acceptance criteria and deployment verification under #49's common release contract.

---
This body supersedes the original roadmap and incorporates the audit-response amendments. [Full revised roadmap](https://github.com/Denolle-Lab/phasenet-retrain/blob/2cbe758/docs/2026-09-10_revised_issue_roadmap.md).
