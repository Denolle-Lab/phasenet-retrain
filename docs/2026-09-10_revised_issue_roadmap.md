# Proposed replacement roadmap for GitHub issues #33–#50

2026-09-10 · Branch `audit/2026-09-07-generalization` · Proposal only; no GitHub writes.

**Recommendation: reorganize around verified implementation, controlled attribution, and measured generalization—in that order.** Retain the 18 issue numbers, replace their titles/bodies, and move #44 and the small #46 pilot ahead of scaling. The present issue comments should become operative scope and acceptance criteria, not sit underneath contradictory original bodies.

This proposal incorporates [Fable's response](2026-09-10_astra_audit_response.md), [the independent audit](2026-09-10_picker_and_issue_roadmap_audit.md), the code and dataset statistics, and a fresh read of all 18 live issues. Fifteen issues already carry amendment comments from the response; **#38, #39 and #47 do not**. All remain open, and the original titles/bodies still describe the previous roadmap. [The current GitHub snapshot](issue_revision_2026-09-10/github_before.json) preserves that starting state. [Machine-readable proposed replacements](issue_revision_2026-09-10/proposed_updates.json) contain a complete body for each issue.

## Review of Fable's response

I agree with the prioritization of alignment, incomplete-label supervision, continuous scoring, evaluability, suite roles and bounded-memory scaling. Three qualifications materially change the next experiments:

1. **The fraction affected is not yet measured.** The dataset summary establishes non-100-Hz source populations: GEOFON is predominantly 20 Hz, ETHZ 200 Hz, and SCEDC includes 40/100 Hz. It does not establish their row counts in v7 or prove GEOFON alone fills the 25% teleseismic target. Caps, source totals and the manifest sampler are different quantities. Replace “about a third of v7 was displaced” with “potentially substantial; #34 will count actual rows and exposures.” A plausible mechanism is not a completed causal attribution.
2. **There are distinct timebase failure modes.** Resampling a waveform while leaving an index in the old coordinates displaces the target relative to the waveform. Falsely declaring native-rate data to be 100 Hz may keep waveform and target aligned in array indices while corrupting physical duration, frequency interpretation and residual seconds. An API may already have transformed the waveform. #34 must identify the actual reader output and label coordinate system before applying a ratio, or a repair can double-resample.
3. **The benchmark is not certified unaffected.** Notebook 05 does rescale indices when it chooses to resample, but code cell 3 hardcodes `NATIVE_SR["ethz"] = 100.0`; cell 9 uses that value rather than per-row metadata. It also assumes MLAAPDE=40, whereas the training reader assumes 100. High oracle recall on ETHZ is not an independent timebase check: waveform and reference may share the same incorrect physical scale. Inspect the exact historical source/cache and benchmark artifacts before deciding which numbers to recompute. These are confirmed code assumptions, not proof that every persisted ETHZ benchmark trace is wrong.

Evidence: [dataset summary](../notebooks/audit_results/summary_statistics.csv), [benchmark notebook](../notebooks/05_benchmark_waveform_processing.ipynb), [training loader](../scripts/manifest_dataset.py), and [training sampler](../scripts/build_training_dataset.py). The issue #34 amendment comment currently repeats “benchmark unaffected”; its replacement below explicitly supersedes that claim.

The response's multiplet proposal also needs a practical distinction: **repair or mask useful additional arrivals; quarantine irreparable examples**. Neither automatic deletion nor unconditional retention is appropriate. The checkpoint identity comparison is settled for the inspected local files; the actual deployment image still needs its own artifact/version check.

## Organization and sequencing

Retain the four milestone slots but revise their names/descriptions as follows. Replace the old calendar promises with evidence gates; give execution estimates after source access, pilot runtime and reviewer workload are known.

| Milestone | Issues primarily owned here | Deliverable |
|---|---|---|
| Phase 0: debugging and trustworthy evaluation | #33–#38, #44 | Verified time/label/model contracts, corrected scorer, eligible suites and calibrated baselines |
| Phase 1: validated corpora and controlled pilots | #39–#43, #46 | Small reviewed corpus, interpretable causal comparisons, scalable storage and a selected recipe |
| Phase 2: scaling and generalization experiments | #45, #47, #48 | Conditional scaling evidence, augmentation/context/width decisions and measured deployment cost |
| Phase 3: acceptance and deployment | #49, #50 | Frozen main-picker release decision and separately gated distant-P experiment |

**Checkpoint notation.** `34A` means checkpoint A inside issue #34, not a new issue number. A checkpoint can unblock another issue while its parent issue remains open for later work. Track those releases as linked checkboxes/artifacts or child tasks. Do not use whole-issue blocking links where only an early checkpoint is required. Each issue below specifies proposed released artifacts and closure criteria; completed experiments may close with a negative result.

| Execution stage | Work that can proceed | Requirement to advance |
|---|---|---|
| Start now | 44A roles, 34A loader repair, 34B historical read-only forensics; then 33A exclusions, 35A scorer, 37A eligibility and 39A census | Local correctness and provenance fixtures; no training claim yet |
| Trustworthy measurements | 34C benchmark/export verification; 35B raw baseline artifacts; 38A independent calibration; 35C comparison; 36A/B event evaluation | Measured baselines on eligible regression/development cases, with no sealed model scoring |
| Small causal experiment | 44B frozen eligible panel and 46A same-row alignment diagnostic | Separate the alignment effect from other recipe changes |
| Small generalization pilot | 41A policy → 40A corpus → 41B review → 46B recipe; 42A/43A augmentation infrastructure → 47A value test | Correct supervision and an interpretable recipe that supports the next investment |
| Scale only after pilot | 40B/42B expanded data; 45A memory/compute plan → 45B curve | Evidence on the selected data/recipe, not a universal data-volume claim |
| Confirm and release | 46C initialization/KD, 47B augmentation confirmation, 48B width/context; 49A/B sealed decision | Frozen candidate, numeric practical margins and per-regime evidence; preserve parent on failure/inconclusive result |
| Separate distant-P line | 50A/B after shared contracts and its own suite freeze | Own acceptance and handoff checks; does not automatically block the main picker |

Restricted-data requests do **not** block synthetic scorer tests, source inspection or measurements on already eligible cases. They block the affected endpoint. Before 46A, freeze an adequate panel from unexamined eligible cases, including a predefined fallback for unavailable archives. If final panel membership changes later, revalidate exclusions and any resulting independence claim before continuing.

**Proposed ownership.** Assign a data-pipeline implementer (#33/#34/#40/#43), evaluation implementer (#35/#36/#38/#44), training implementer (#45–#48), domain/reference reviewer (#37/#39/#41/#42), and deployment owner (#49/#50). These are roles, not assignments to named people. Marine owns the external access requests already identified in #37. Estimate review effort explicitly; “manual” or “catalogued” labels are not a substitute for that work.

**Operating definition of generalization.** Preserve the existing land-station local/regional scope and separate distant-P branch; OBS remains out of this round. Evaluate weak P, regional/late S, closely spaced arrivals, volcanic/swarm regimes, unfamiliar stations/instruments, gaps, and real noise. Track event-disjoint, station-disjoint and region/operator transfer separately, with each pretrained model's exposure identified. A general phase picker and a source-type classifier have different targets; #41/#42 must establish which physical onsets remain valid.

## Proposed issue replacements

The following titles and bodies are ready for issue editing. “P0/P1/P2” denotes proposed urgency rather than a claim that such labels already exist.

### #33 — [Phase 0] Enforce versioned exclusions and trace/event identity across every data path

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/33)

**Why this revision.** The exclusion file must certify a particular source snapshot and suite policy, not merely exist.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- **33A — Exclusion contract released:** 44A.

**Work.** Preserve `(dataset, chunk, trace_name)` and canonical event identity. Hash the source metadata, registry, exclusion rules and resulting manifests. Apply exclusions to signal, validation, negative examples, augmentation pools and hard-negative mining. Audit duplicate events across source datasets and overlapping waveform intervals. Require a documented quarantine policy for rows whose time/location/identity cannot establish independence; any explicitly allowed unknown rows remain outside independence claims.

**Acceptance.**
- A versioned exclusion bundle and per-source removal/unknown counts are committed and checked by every builder.
- Reused trace names in different chunks are distinguished; equivalent events across datasets remain in one split.
- Fixtures cover missing/stale lists, unknown coordinates/times, year/place exclusions, cross-source duplicates and noise-window overlap.
- Historical manifests remain immutable. Exclusions constrain new training; they do not delay read-only forensics in #34.

**Dependency detail.** 33A uses the initial suite policy 44A. Subsequent suite additions invalidate the relevant exclusion bundle and require a new hash. Server-wide counts are required for a server-wide certificate; a verified pilot snapshot may be certified separately.

### #34 — [Phase 0] Repair and quantify waveform–label, timebase and checkpoint-contract defects

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/34)

**Why this revision.** Promote from hypothesis tables to the principal debugging issue, including the benchmark path.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- **34A — Loader invariants pass:** no upstream checkpoint.
- **34B — Historical row-level forensics complete:** no upstream checkpoint.
- **34C — Model and benchmark contracts verified:** 34A.

**Work.** Distinguish original recording rate, effective rate returned by the reader, and the coordinate system of each arrival index. Rescale waveform and indices exactly once. Read HDF5 rate/component metadata instead of assuming 100 Hz; handle bucketed names and trim to actual trace support. Support S-only rows. Replace fetch-error zero/noise substitution with a rejected-row ledger and failure gates.

**Acceptance.**
- **34A:** Fixtures at 20/40/50/100/200 Hz plus a non-integer rate preserve absolute P/S times within one target sample, including crop boundaries, missing channels and explicitly missing rates. Check 120/250/500 Hz when represented by production sources.
- **34B:** An immutable row-level report for the actual v7 train/val/test manifests records source/effective/assumed rates, old/new offsets, effective labels after crop, outside-window arrivals and fetch status. Aggregate by source, distance and phase; report unique rows and training exposures separately. Restore H2 and noise-pool ordering as secondary tables with independent negative populations.
- **34C:** Verify raw source → benchmark window → seconds against independent timestamps. Notebook 05's ETHZ=100 and MLAAPDE=40 assumptions both need source evidence. Check crop-cache invalidation, array orientation, normalization and parent → wrapper → export probability parity at the pinned runtime. Hash the actual deployed pair, not only the laptop cache.
- Quantify affected rates/rows before publishing a corpus fraction. Revise historical result provenance if the benchmark changes.

**Dependency detail.** A and B start immediately. C follows the local reader contract; server evidence remains necessary to certify historical artifacts. This issue does not itself establish which defect caused v7's generalization loss; #46 supplies controlled training.

### #35 — [Phase 0] Correct continuous pick scoring and recompute regression/development baselines

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/35)

**Why this revision.** Separate scorer implementation from data acquisition and calibrated comparison; never run --all over sealed acceptance cases.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- **35A — Scoring engine released:** 44A.
- **35B — Eligible baseline annotations and picks persisted:** 35A, 34C, 37A.
- **35C — Calibrated baseline comparison complete:** 35B, 38A.

**Work.** Store annotations once and invoke the production trigger extractor independently at each threshold. Deduplicate bulletin representations by event/station/phase identity, preserving distinct close events. Match maximum cardinality first, minimum residual second. Store picks, reference assignments, unmatched candidates, coverage and failures. Aggregate counts within model/phase/threshold across explicitly identified windows; remove duplicate thresholds and use actual attained operating points.

**Acceptance.**
- **35A:** Tests cover threshold-induced peak splitting, two close events, greedy-matching counterexamples, duplicate references, multiple windows, gaps, model failures and zero-reference cases. Failure/coverage policy is explicit and cannot silently make a candidate look better.
- **35B:** Parent, v7, instance and available v11 produce versioned artifacts on regression/development only. Record training-domain/parent-overlap provenance for every baseline; Norcia is not automatically independent of INSTANCE weights. Existing QuakeScope tables are re-extracted from annotations if saved; otherwise rerun inference. Old tables remain archived and marked superseded. Ensembles align phase labels, time grids and valid support while retaining each member's preprocessing.
- **35C:** Publish P/S recall, residual distributions, matched total-pick workload and separately calibrated nuisance-pick budgets, with paired block uncertainty and unavailable models identified.
- Persist enough information for #36 to associate the exact emitted picks.

**Dependency detail.** Scorer development starts using synthetic fixtures without #37. Baseline inference needs only the particular cases certified by 37A, not every access request. Threshold curves precede 38A; calibrated comparison follows it. #35 does not authorize replacing the campaign picker; #49 owns release.

### #36 — [Phase 0] Validate event association and catalogue-relative recovery on continuous data

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/36)

**Why this revision.** An event-level endpoint is necessary, but it must not consume sealed La Palma/West Bohemia results during debugging.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- **36A — Association/matching engine released:** 35A.
- **36B — Development event baselines complete:** 36A, 35C, 37A.

**Work.** Associate the persisted picks with PyOcto under versioned regional travel-time models and station metadata. Freeze associator parameters across picker comparisons; tune them only on development. Match predicted/catalogue events one-to-one with explicit origin-time, distance and depth tolerances, retaining diagnostics for association splits and merges.

**Acceptance.**
- **36A:** Synthetic cases and a reviewed small catalogue establish matching behavior for missing stations, duplicate events, false associations, splits and merges.
- **36B:** Report recovery versus magnitude, hour after mainshock and day of swarm, plus station support, valid exposure and unassociated/unmatched counts. Use Kaikōura as a known regression and Etna/Corinth-Thiva as development examples if 37A certifies them; choose substitutes before scoring if necessary.
- Report catalogue completeness limitations and independently review a sample of unmatched events. “Absent from the catalogue” is not automatically a false event.
- Paired uncertainty uses independent event/time blocks; P/S observations and stations belonging to the same event remain linked.
- First-48-hour or migration claims require the corresponding time coverage; busy-hour excerpts alone cannot pass those claims.

**Dependency detail.** Build the engine after the 35A artifact schema. Final development comparisons use calibrated picks from 35C. Sealed event-level baselines and candidate results are generated together under #49.

### #37 — [Phase 0] Certify test-case evaluability and close targeted reference/waveform gaps

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/37)

**Why this revision.** Replace a binary built/unobtainable list with eligibility for specific scientific claims.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0 for eligibility; access work case-dependent.

**Checkpoint dependencies.**

- **37A — Existing regression/development cases certified:** 44A.
- **37B — Proposed sealed panel eligibility frozen:** 37A.

**Work.** For each case produce valid component/station/time coverage, reference overlap, P/S counts, event support and reference-provenance tiers. Resolve NET.STA.LOC/channel/epoch with inventory and coordinates; quarantine ambiguous station codes. Preserve gap masks and report availability independently of model performance.

**Acceptance.**
- **37A:** Existing cases are explicitly eligible for pick scoring, network/event scoring, both, or neither. Noto swarm's current zero overlap and Hualien's one-station coverage cannot be treated as acceptance passes.
- **37B:** The proposed sealed panel has an immutable eligibility table, minimum support/precision requirements per endpoint and predeclared fallback cases. No candidate scores are examined to choose cases.
- Retain the Hi-net, CWA GDMS, IMO, AFAD, Zagreb, WEBNET and INGV-OV acquisition tasks, each with owner, access state and expected eligibility gain. Marine handles external account/contact actions when authorized.
- Unknown-mode bulletin picks, explicitly manual picks and automatic picks remain separate provenance tiers. Hash pinning is distinct from reference validation.
- Extend selected cases to early coda, later hours/days and ordinary conditions where those claims are planned; record station-selection bias.

**Dependency detail.** Existing open cases unblock #35 immediately when certified. Unresolved access blocks only the affected endpoint/case; #49 requires a complete, predefined adequate panel. A case replaced before scoring still requires exclusion-policy verification in #33/#44.

### #38 — [Phase 0] Calibrate P/S operating points on independent station-days

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/38)

**Why this revision.** This issue was not amended in Fable's response; it currently conflates catalogue absence with false picks and has no calibration/test separation.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0.

**Checkpoint dependencies.**

- **38A — Calibration protocol and baseline thresholds released:** 35B, 37A, 44A.

**Work.** Define calibration days separately from training/mining, development evaluation and sealed acceptance. Sample quiet and disturbed conditions by region, station/instrument class and season. Use corrected threshold-specific extraction from #35, measured valid exposure and independent review of unmatched picks.

**Acceptance.**
- A protocol specifies P/S nuisance-pick budgets, uncertainty targets, minimum exposure and a fallback for regions with too few calibration data. Numeric values are fixed before evaluating candidates.
- Publish thresholds and attained rates on calibration days; report achieved rates on distinct evaluation days without retuning.
- Where no reviewed truth exists, call the metric “unmatched-pick rate,” retain reviewed-sample uncertainty, and do not equate it with false-positive rate.
- Candidate-specific thresholds are refit by the same frozen procedure on calibration data only. Parent/candidate event comparisons share the associator.
- Retain both total emitted workload and nuisance-pick rate; matching one does not imply matching the other.

**Dependency detail.** Requires the baseline annotations/picks checkpoint 35B, not closure of #35. Its output unblocks 35C. Station-day calibration may proceed without the final eleven-class training-noise corpus.

### #39 — [Phase 1] Census usable bulletin supervision, source diversity and acquisition cost

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/39)

**Why this revision.** This unamended issue needs a census of useful independent labels, rather than extrapolation from a single week.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1; start early.

**Checkpoint dependencies.**

- **39A — Stratified source census released:** 44A.

**Work.** Sample multiple periods per operator/year: quiet intervals, ordinary seismicity, aftershock activity and relevant volcanic/swarm regimes. Record native/effective rate, channel identity, units/response state, provenance, uncertainty, P-only/S-only/P+S availability, event multiplicity and waveform accessibility.

**Acceptance.**
- Publish operator/year counts with sampling uncertainty and separate total rows, unique events, stations and duplicate events shared across corpora.
- Estimate low-SNR and multi-event availability using defined measurements; uncertain label completeness remains explicit.
- Identify at least one development region/operator outside the existing European concentration, subject to #44's role decision before scoring.
- Report query/storage/processing costs and licensing/access constraints; nominate pilot sources based on useful supervision and diversity.
- Treat composition targets and the 4M maximum as hypotheses to test, not quantities the census must force.
- Apply existing held-out rules to estimates, including source-year/place overlaps.

**Dependency detail.** Read-only reconnaissance starts early with 44A. The training-ready pilot additionally needs #33, #34 and #41; a census alone does not certify a usable corpus.

### #40 — [Phase 1] Build a versioned partial-label corpus and scalable waveform store

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/40)

**Why this revision.** The corpus must preserve time, all known arrivals and unknown-label regions; scaling must not depend on a dense RAM cache.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1.

**Checkpoint dependencies.**

- **40A — Small validated corpus and target schema released:** 33A, 34A, 39A, 41A.
- **40B — Sharded corpus and composition audit released:** 40A, 41B, 46B.

**Work.** Store waveform intervals with explicit sample rate, absolute start time, component identity and valid support. Store lists of all known arrivals with event ID, phase family, provenance/uncertainty and label-validity information. Include P-only, S-only, noise and multi-event crops. Harvest arrivals capable of reaching the station from before/outside the nominal event query window.

**Acceptance.**
- **40A:** A small reviewed pilot passes loader/time invariants. The target contract distinguishes observed absence from unlabelled time and defines normalized overlapping P/S targets or an explicitly tested alternative loss. A mask excludes uncertain supervision without accidentally penalizing the missing phase through softmax normalization.
- **40B:** Disk-backed shards or memory mapping support worker-local reads, raw/long-window caching and labels generated per batch. Validate a bounded-memory stress run before millions of windows.
- Report effective P/S, SNR, distance, native-rate, event-count, operator and station mix after cropping/augmentation, plus unique-event counts and removal reasons.
- Preserve immutable source and manifest hashes; maintain event/station/operator transfer splits with overlap audits.
- Keep stored context independent of training context. Begin diagnostics at the parent's verified 3001-sample contract; 60 s training requires #48's explicit inference contract.

**Dependency detail.** 40A is a pilot release, not the full corpus. Full expansion 40B follows successful label review and the small recipe pilot 46B; #46 does not wait for 40B.

### #41 — [Phase 1] Define label validity and repair uncertain or incomplete arrivals

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/41)

**Why this revision.** The existing helper downloads exclusions; it neither runs new confident learning nor justifies rejecting every multiplet.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P0 policy; P1 review.

**Checkpoint dependencies.**

- **41A — Target/provenance and review policy released:** 39A.
- **41B — Pilot labels reviewed and repaired:** 40A.

**Work.** Define provenance tiers, accepted phase families and task scope with a domain reviewer. Model disagreement and physical residuals prioritize review; they are not ground truth. Review examples without exposing which candidate produced the suggestion. If prediction-based triage is used, produce it out of fold.

**Acceptance.**
- **41A:** A documented decision table distinguishes valid labels, additional unlabelled arrivals, uncertain timing/phase, source artefacts and unusable records. Specify loss/mask behavior for each.
- **41B:** The pilot has per-operator reviewed-sample results, corrected arrivals or explicit uncertainty masks, and retained/removed/quarantined counts with reasons.
- Existing multiplet flags route examples to relabelling or uncertainty handling. Do not automatically discard them, and do not retain irreparable labels simply because multiplets are desired.
- S–P checks account for depth, distance, velocity uncertainty and Pg/Pn/Sg/Sn conventions; emergent S and LFEs are not rejected solely because the parent misses them.
- Training-label review and sealed reference review are separate records. Model-based review must not mine the sealed panel.

**Dependency detail.** Release the policy before constructing 40A. Review 40A before 46B/40B. The new-corpus review implementation must be built explicitly; running `label_error_filter.py` alone cannot close this issue.

### #42 — [Phase 1] Build reviewed background and uncertain-event pools with station/time holdouts

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/42)

**Why this revision.** Retain natural-noise diversity without teaching the picker to suppress legitimate physical arrivals.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1.

**Checkpoint dependencies.**

- **42A — Noise ontology and pilot pools released:** 33A, 34A, 41A, 44A.
- **42B — Expanded class/station/season census released:** 42A, 47A.

**Work.** Separate background noise, reviewed negatives, task-excluded sources, and uncertain event-bearing intervals. Preserve the eleven environmental/site categories as metadata, not automatic all-N labels. Aftershock coda may contain arrivals; LFEs and blasts may contain meaningful P/S phases. Fix the general-picker/source-classifier boundary in 41A.

**Acceptance.**
- **42A:** Reviewed pilot examples and uncertainty masks exist for each included class; no label depends solely on catalogue absence or the parent's confidence.
- **42B:** Expanded pools have station/time/season/source provenance, valid exposure, composition summaries, holdout lists and overlap checks against all evaluation/calibration intervals.
- Any existing noise_global data retain their original model-screening history and are independently reassessed; they cannot silently become an unbiased negative set.
- Keep OBS excluded under the current scope. Do not impose 20k/class until class validity and useful diversity are established.
- Training/mining and evaluation pools are distinct. Rate-preserving waveform operations follow #34, and pools use #40's storage pattern.

**Dependency detail.** A small pilot supports #43/#47 before full harvest. Expand after 47A demonstrates that the intended use adds value or after a documented decision to expand for coverage, rather than interpreting failed augmentation as a mandate for volume.

### #43 — [Phase 1] Implement time-consistent augmentation with explicit label and SNR rules

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/43)

**Why this revision.** Augmentation should be switchable and verifiable independently of whether it improves the model.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P1.

**Checkpoint dependencies.**

- **43A — Augmentation transforms and fixtures released:** 34A, 40A, 42A.

**Work.** Apply random cropping/placement at sample access, transform arrival lists and validity masks together, and normalize consistently with deployment after physically meaningful mixing. Implement real-noise addition, controlled multi-event superposition, component dropout, valid anti-aliased bandwidth/rate changes and separately switchable artefact groups.

**Acceptance.**
- Time transformations preserve absolute arrival time and mask support within one output sample; superposition preserves distinct event IDs and uses the normalized target rule from #40.
- Define signal/noise measurement windows and target achieved SNR after mixing; handle already noisy source waveforms and cases where clean-signal SNR is unidentifiable.
- Do not treat whole-window unit variance as clean signal amplitude. Preserve relative component amplitudes and require compatible units/response/bandwidth before mixing.
- Deterministic fixtures cover gaps, clipping, missing components, overlapping phases and crop boundaries. Stored seeds reproduce mixtures and sampling.
- An inspection sheet includes untouched and augmented real examples. Every group can be disabled in config; untouched/noise fractions are explicit tunable values.
- Augmentation never samples calibration/development/acceptance intervals.

**Dependency detail.** 43A can be built from pilot pools; it does not depend on full 40B/42B or #45. Transform correctness is assessed here; scientific benefit is assessed in #47.

### #44 — [Phase 0] Separate regression, development, calibration and sealed acceptance roles

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/44)

**Why this revision.** Amend the actual registry/policy and all entrypoints, not just add an override flag or another issue comment.

**Milestone:** Phase 0: debugging and trustworthy evaluation. **Priority:** P0; begin now.

**Checkpoint dependencies.**

- **44A — Initial roles and read policy released:** no upstream checkpoint.
- **44B — Eligible sealed panel and decision rule frozen:** 44A, 37B.

**Work.** Mark Kaikōura, Norcia, Thessaly and previously inspected western cases as regression. Keep Samos, Adriatic, Etna and Corinth-Thiva as development, with additional geographically useful development cases assigned before scoring. Define disjoint calibration station-days. Treat new sequences as sealed only after checking whether model results were previously viewed anywhere.

**Acceptance.**
- **44A:** Registry roles and access logs distinguish metadata/reference QA from model-performance access. All CLI/notebook/API scoring entrypoints use the same policy; `--all` excludes sealed cases by default.
- **44B:** After #37's metadata-only eligibility review, freeze the panel, reference/model-provenance strata, fallback policy, practical effect margins, non-inferiority endpoints and paired uncertainty rule.
- Record hashes for suite, source references, code, candidate and configuration at every authorized scoring run. Years are exclusion rules, not sequence keys.
- A changed/new sealed case triggers recertification of training exclusions. A consumed panel cannot be restored by renaming; future selection needs fresh acceptance cases.
- Review of sealed picks for scientific quality is allowed without candidate predictions; it does not authorize tuning models or thresholds on that panel.

**Dependency detail.** 44A starts immediately and unblocks infrastructure and read-only census. 44B waits for eligibility, not access to every desired archive. 44B precedes pilot/model selection on the new roadmap; debugging with synthetic/regression fixtures can proceed meanwhile.

### #45 — [Phase 2] Measure data scaling for a validated recipe under bounded memory and compute

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/45)

**Why this revision.** The scaling curve must follow causal debugging, not serve as a test of several unvalidated assumptions at once.

**Milestone:** Phase 2: scaling and generalization experiments. **Priority:** P2; defer large runs.

**Checkpoint dependencies.**

- **45A — Scaling design and throughput budget frozen:** 40B, 42B, 46B, 47A.
- **45B — Scaling curve and next-size decision complete:** 45A.

**Work.** Select a validated recipe and event-grouped nested subsets with controlled source/phase/noise composition. Start at the available pilot size; 0.25/0.5/1/2/4M are possible tiers, not mandatory commitments. Distinguish unique events/stations/windows from repeats and synthetic mixtures.

**Acceptance.**
- **45A:** Memory/throughput measurements support the next tier. The old 4M × 6000 × 3 × float32 waveform-plus-label cache is about 576 GB before other allocations; it is not the implementation plan.
- **45B:** Use at least three seeds for decision-making tiers and report matched-budget continuous metrics by regime, with baseline/pilot included.
- Report fixed-epoch results and a matched-update or matched-compute comparison; larger data change optimization exposure as well as diversity.
- Keep development/threshold calibration rules fixed and log model-selection compute. Stop/escalate based on predefined practical margins and uncertainty, not a blanket “2M failure means fine-tuning cannot work.”
- A flat curve is conditional on this recipe/data mixture. Diagnose whether label quality, coverage or optimization limits interpretation before acquiring the next tier.

**Dependency detail.** This waits for checkpoints 46B and 47A, not for closure of their later full-scale confirmation work. If a signal-only recipe is selected, 42B may be explicitly waived with that scope recorded before 45A; negative evaluation remains mandatory.

### #46 — [Phase 1] Run controlled alignment and recipe pilots before initialization/KD scaling

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/46)

**Why this revision.** Own causal attribution explicitly; do not bundle alignment, labels, BN, augmentation and dataset expansion into the first retrain.

**Milestone:** Phase 1: validated corpora and controlled pilots. **Priority:** P0 first pilot; P2 later confirmation.

**Checkpoint dependencies.**

- **46A — Paired alignment diagnostic complete:** 33A, 34A, 34B, 34C, 35C, 44B.
- **46B — Small corrected recipe selected:** 46A, 40A, 41B.
- **46C — Matched initialization/KD confirmation complete:** 45B.

**Work and acceptance.**
- **46A:** Compare the same parent initialization and the same verified, exclusion-clean row list under legacy versus corrected time/index handling. Use the repaired reader in both arms, deliberately toggling only legacy versus corrected time/index mapping; do not reproduce unrelated silent fetch failures. Hold loss, KD, BN policy, augmentation, batch size and update budget fixed. Use three paired seeds and untouched verified evaluation; keep old v7 as a historical baseline, not the experimental control. Restrict deliberate faulty-label reproduction to this diagnostic; never deploy it. A 100-Hz-only stratum is a useful control but cannot replace the same-row comparison because source mix changes.
- Report effects by affected/unaffected rate strata, source, distance and phase. If historical data cannot be reconstructed, document the causal limit rather than declaring the bug explains all loss.
- **46B:** On the small reviewed corpus, test a minimal corrected recipe, then controlled soft-label/partial-label handling, BN frozen versus adaptive, and KD off/on. Separate BN statistics from affine-parameter freezing; treat replay as an optional additional arm with certified exclusions. Use identical row lists and paired seeds for each comparison; log loss components, gradients, learning rate, effective labels and continuous validation.
- **46C:** At #45's chosen size, compare matched-width parent/scratch initialization and KD off/on explicitly. Scratch+KD is a defined arm, not an accidental implementation. Match data and report compute costs.

**Dependency detail.** 46A uses the verified existing corpus and does not wait for new bulletin harvest. 46B needs only pilot 40A/41B, not bulk 40B. Noise/augmentation benefit is #47. Only 46C depends on #45, eliminating the original E1/E2 cycle.

### #47 — [Phase 2] Test real-noise and multi-event augmentation before full-scale confirmation

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/47)

**Why this revision.** This unamended issue belongs partly before scaling and needs no-augmentation and uncertainty-handling controls.

**Milestone:** Phase 2: scaling and generalization experiments. **Priority:** P1 pilot; P2 confirmation.

**Checkpoint dependencies.**

- **47A — Pilot augmentation value established:** 46B, 42A, 43A.
- **47B — Selected ablations confirmed at chosen scale:** 45B, 47A.

**Work.** Start from the selected small corrected recipe, then compare no added noise, measured real-noise mixtures and white-noise control; separately test event superposition, non-stationary/artefact groups and class balancing. Change one interpretable group at a time before testing important interactions.

**Acceptance.**
- **47A:** Paired seeds, equal source examples/update budgets and explicit untouched fractions isolate augmentation effects. Report P/S recall on weak and close arrivals, nuisance picks by noise class, valid station-day exposure and timing tails.
- Demonstrate that target uncertainty/merged labels remain correct; do not let an augmentation ablation accidentally change label policy.
- Evaluate on untouched real continuous data at held-out stations/times, including disturbed conditions absent from the added-noise recipe.
- **47B:** Repeat the shortlisted informative controls at the selected scale. Record harmful or null effects as results; no augmentation group is mandatory because it sounds physically realistic.
- Any hard-negative mining uses a separate training pool and records iteration and source; calibration/development/acceptance data are never recycled into it.

**Dependency detail.** 47A precedes #45 and full noise harvest 42B. 47B follows #45; #45 must not depend on closure of the whole issue.

### #48 — [Phase 2] Validate and compare deployable PhaseNet width, context and ensembles

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/48)

**Why this revision.** A direct forward shape or YAML length is insufficient to establish continuous inference at that context.

**Milestone:** Phase 2: scaling and generalization experiments. **Priority:** P2; context contract before any long-window run.

**Checkpoint dependencies.**

- **48A — Alternative context/export contract released:** 34C, 35A.
- **48B — Width/context experiments and deployment cost complete:** 48A, 45B.

**Work.** Pin the SeisBench implementation and explicitly configure training length, `in_samples`, `pred_sample`, overlap, padding/blinding and exported metadata. Test supported lengths before choosing the 60 s arm. The prior 6001-sample failure is evidence about the tested runtime, not a timeless architectural law.

**Acceptance.**
- **48A:** Direct forward, continuous annotation and save/reload agree on dimensions, time alignment and probabilities at each supported context, including edges and gaps. Keep original 3001-sample diagnostics independent of this optional extension.
- **48B:** Compare width and context separately where possible, holding data/recipe fixed. A narrow network cannot inherit incompatible wide weights without a specified transfer; use matched scratch comparisons or a documented compatible initialization to isolate width.
- Report effective receptive-field/context interpretation, P/S performance in late-S/close-event regimes, memory, latency and throughput.
- Ensembles align phases/time support and retain each model's normalization; weights/thresholds are selected on development/calibration only.
- Negative or null results close the experiment with a decision. Acceptance does not require trying every context or deploying an ensemble.

**Dependency detail.** 48A can run early; #40 stores long/raw intervals without requiring 60 s model training. 48B follows scaling. #49 accepts a documented decision to retain the original width/context.

### #49 — [Phase 3] Execute a sealed acceptance decision and verify the deployed artifact

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/49)

**Why this revision.** Replace correlated five-of-six wins and multiple candidate reads with one interpretable release decision.

**Milestone:** Phase 3: acceptance and deployment. **Priority:** P2; protocol fixed earlier.

**Checkpoint dependencies.**

- **49A — Primary candidate and frozen release package selected:** 35C, 36B, 37B, 38A, 44B, 45B, 46C, 47B, 48B.
- **49B — Acceptance decision and deployment verification complete:** 49A.

**Work.** Freeze one primary candidate, the parent/declared baselines, thresholds, associator, suite/reference hashes and software/container versions before the sealed run. Record approved omissions of optional experiments before candidate selection; an unrun experiment cannot be reported as a pass.

**Acceptance.**
- **49A:** The preregistered endpoint policy contains numeric practical improvement/non-inferiority margins, minimum usable coverage, primary/secondary metrics, paired block uncertainty and handling of missing cases/multiple endpoints.
- **49B:** Score baseline and candidate together on the sealed panel once. Require the planned overall benefit and regime-specific non-inferiority; report pick workload, nuisance-rate evidence, event recovery, timing tails, operational failures and computational cost.
- Insufficient independent events/days yields “inconclusive,” not “no degradation.” Do not treat P/S pairs from one sequence as independent replicates.
- Hash and reload the exported artifact offline, verify preprocessing/probability parity on fixed fixtures, and run the production notebooks at frozen settings. Stage rollout with monitored rates and a reproducible rollback to the parent.
- A failed or inconclusive acceptance result retains the parent. Once results inform another development cycle, that panel is consumed and a fresh one is required.

**Dependency detail.** #35 characterizes baselines without shipping them. Any earlier no-training baseline replacement must use a separate preregistered acceptance decision and consume its own panel; it cannot silently reuse the future sealed panel.

### #50 — [Phase 3] Pilot distant-P transfer with a verified timebase and regional handoff

[Current issue](https://github.com/Denolle-Lab/phasenet-retrain/issues/50)

**Why this revision.** Keep sparse-region/offshore coverage as a separate experiment, without assuming 20 Hz transfer or catalogue completeness.

**Milestone:** Phase 3: acceptance and deployment. **Priority:** P2; independent pilot after shared contracts.

**Checkpoint dependencies.**

- **50A — Distant-P timebase/data pilot validated:** 33A, 34C, 35A, 38A, 41A, 42A, 44B.
- **50B — Distant-P acceptance and handoff decision complete:** 50A.

**Work.** Inspect the actual geofon checkpoint's sample rate, preprocessing and input contract. Compare verified native-rate deployment with a deliberately retrained 20 Hz/long-window candidate; feeding 20 Hz arrays to unchanged 100 Hz weights changes physical time/frequency scales. Apply the same arrival-index and metadata invariants as #34.

**Acceptance.**
- **50A:** Define P branch coverage (e.g. P/Pn and later branches), magnitude/distance support, source-label provenance and off-Japan/held-out-year evaluation. Validate proposed context through annotation/export, not forward alone.
- Include cultural, instrument and coda conditions at continental sparse-network stations, alongside coastal/microseism/polar noise.
- Predefine overlapping regional/distant eligibility around the nominal 300 km handoff, deduplicate common picks/events and keep appropriate travel-time uncertainty in association.
- **50B:** Freeze a separate candidate and sealed panel; evaluate station/event recovery and nuisance workload, including double detections and handoff misses. Report recovery relative to reference completeness; claiming a new completeness magnitude requires additional evidence.
- Preserve the no-OBS scope. A failed distant-P pilot does not block a valid main-picker release, and a successful main picker does not certify the distant-P branch.

**Dependency detail.** This shares infrastructure and label policy but not the full #45–#48 training sequence. It owns its own candidate freeze, acceptance criteria and deployment verification under #49's common release contract.

## Applying the proposal without losing the plan history

1. Review this package as the proposed replacement text. The issue snapshot records the bodies and amendment comments it incorporates; the local audit and response remain historical evidence.
2. When issue edits are authorized, refresh GitHub state and reconcile any new comments before updating the existing issue numbers. Replace titles, descriptions, dependencies and milestone descriptions together. Preserve the discussion history.
3. Update `docs/2026-09-07_training_plan.md`, the rendered issue plan and the source in `scripts/open_plan_issues.py` to the accepted version. Do not rerun its existing `--create` workflow to amend these issues: it is a creation workflow, not an issue-number-based update contract.
4. Record checkpoint releases with code/config/data hashes. No roadmap item is marked complete solely because its description was corrected or its experiment gave the desired result.
5. First implementation batch: 44A + 34A/B + 35A, with 33A/37A/39A following the initial role policy. The first retraining request is 46A, not the 4M-window run.

**Validation of this proposal:** all 18 issue numbers have replacement titles/bodies; every checkpoint dependency resolves within the package and the graph is acyclic. This turn produced planning artifacts and source review only. It did not modify GitHub, retrain a model, or run models on sealed cases.
