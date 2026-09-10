# SU picker implementation plan

Owner: **Akash Kharita (@Akashkharita)**. All SU implementation issues are assigned
to him and labeled **`su-picker`**. Scientific coordination: Marine Denolle;
an independent annotation reviewer and the sprint start date still need to be
recorded. This plan is the entry point for future SU work; implementation remains
open. The [design rationale](2026-09-08_surface_event_picker_plan.md),
[audit](2026-09-10_surface_event_picker_audit.md) and
[review response](2026-09-10_surface_picker_review_response.md) explain the choices.

## Goal and first deliverable

Build SUNet, a retrospective station-level surface-event onset picker. Its U
output marks the first observable seismic arrival associated with a defined
surface process. U is not an earthquake P phase, source origin time, flow-front
passage or location. Supported processes and instruments must follow the evidence.
Classification, duration, network association and location are optional later
deliverables with their own tests.

The first bounded corpus uses post-2002 Rainier/St. Helens continuous records and
reviewed negatives from permitted stations, with Newberry/Hood development
station-days after overlap checks. Keep separate calibration and acceptance
assets. Catalogue absence is not proof of a negative. The starting model candidate
is UN softmax at 50 Hz/6000 samples (120 s); UPN, initialization and generalization
changes follow controlled pilot evidence.

## How to execute

1. Start SU-01; release the taxonomy and pilot partitions. SU-02 read-only census
   can start meanwhile. Do not train or mine until the relevant exclusions pass.
2. Complete the bounded SU-02 corpus and SU-03 preprocessing/model contracts.
   Synthetic SU-03 checks need not wait for a bulk harvest.
3. Complete SU-04 continuous baselines, separate calibration and acceptance
   eligibility/protocol freeze. Then run the SU-05 minimal pilot.
4. Use SU-06 only for diagnostic, controlled improvements. SU-07 is optional:
   document whether each additional stage earns its place.
5. Freeze one candidate for SU-08 acceptance. Publish either a supported release
   decision or a no-release result; an inconclusive experiment is useful evidence.

Work on one ready checkpoint branch/PR at a time, using a separate worktree per
session. Create branches from the latest reviewed surface integration head,
`audit/2026-09-10-surface-picker`. Each PR links its SU issue and includes code,
config/data hashes, validation results and remaining criteria. Close a task only
when its acceptance criteria are satisfied or an explicitly permitted no-go/omit
decision is documented. Plans and assignments do not count as completed work.

SU-01 and SU-02 each have a proposed 10-working-day timebox. T0 is the agreed
start with access and reviewer availability; no calendar due date is implied.
The SU-02 timebox delivers a census and bounded pilot, not all post-2002 data.
Estimate later compute/review time from SU-02/03/05 measurements.

## Shared dependencies with the earthquake picker

Reuse #33/#44 exclusions and suite access, #34 time/resampling/normalization/export
contracts, #35 continuous scoring, #37 evaluability and #38 calibration; #43
supplies compatible augmentation transforms. Keep their earthquake scope and
assignees unchanged. Link to the required checkpoint or verified artifact rather
than waiting for an entire multiphase issue to close. Do not build a separate SU
resampler or silently reuse the known-broken loader/scorer.

## Task index

Roadmap tracker: [#53](https://github.com/Denolle-Lab/phasenet-retrain/issues/53).

| Task | GitHub | Deliverable | Branch |
|---|---|---|---|
| SU-01 | [#54](https://github.com/Denolle-Lab/phasenet-retrain/issues/54) | Define target taxonomy, suite roles and exclusion registry | `surface/01-taxonomy-suites` |
| SU-02 | [#55](https://github.com/Denolle-Lab/phasenet-retrain/issues/55) | Census, review and build the bounded pilot corpus | `surface/02-corpus-label-audit` |
| SU-03 | [#56](https://github.com/Denolle-Lab/phasenet-retrain/issues/56) | Repair preprocessing, windowing, labels and model export contracts | `surface/03-model-data-contract` |
| SU-04 | [#57](https://github.com/Denolle-Lab/phasenet-retrain/issues/57) | Establish continuous baselines, calibration and acceptance protocol | `surface/04-continuous-baselines` |
| SU-05 | [#58](https://github.com/Denolle-Lab/phasenet-retrain/issues/58) | Train and diagnose the minimal UN pilot | `surface/05-minimal-pilot` |
| SU-06 | [#59](https://github.com/Denolle-Lab/phasenet-retrain/issues/59) | Measure source transfer and controlled generalization improvements | `surface/06-generalization-arms` |
| SU-07 | [#60](https://github.com/Denolle-Lab/phasenet-retrain/issues/60) | Evaluate optional cascade and coverage-aware network association | `surface/07-cascade-association` |
| SU-08 | [#61](https://github.com/Denolle-Lab/phasenet-retrain/issues/61) | Run sealed acceptance and deliver a verified SU artifact | `surface/08-acceptance-deployment` |

## Task specifications

### SU-01 — Define target taxonomy, suite roles and exclusion registry

Issue: [#54](https://github.com/Denolle-Lab/phasenet-retrain/issues/54). Owner: @Akashkharita.

**Dependencies:** No preceding SU task. Reuse the verified exclusion/suite mechanisms in #33 and #44; release only the specific needed checkpoints, not whole-issue closure.

**Timebox:** 10 working days from the agreed start; estimate, not a fixed calendar deadline.

**Work:**

- Define U as a station-level first observable surface-event arrival. Distinguish source initiation, flow-front passage, duration, source classification and location; document supported process classes and ambiguous glacier/tremor cases.
- Create a versioned registry for train, mining, regression, development, calibration and sealed acceptance. Store event-family IDs/aliases, station-day UTC bounds, evidence tier, source hashes, quality/coverage, reviewer and model-exposure history.
- Deduplicate raw/curated PNSN and cross-catalogue ESEC/SED/GeoNet/ComCat events. Apply event/station/time exclusions to positives, background, donors, pseudo-labels and transferred/classifier weights.
- Propose post-2002 Rainier/St. Helens training/mining and Newberry/Hood development partitions; resolve shared stations/events and hold separate calibration days. Define reference-QA versus model-scoring access guards.
- Coordinate scientific decisions with Marine; record an independent label reviewer and the agreed sprint start.

**Done when:**

- [ ] A committed taxonomy, registry schema, populated pilot partition and exclusion report identify all unknown/colliding cases.
- [ ] Entry-point checks refuse missing/stale roles and unauthorized acceptance scoring; fixtures cover aliases, overlapping windows, donor leakage and missing identities.
- [ ] Previous output exposure is documented; a consumed acceptance panel cannot be relabeled blind.
- [ ] Pilot partition is released for SU-02 or an explicit unresolved collision/access list is published.

### SU-02 — Census, review and build the bounded pilot corpus

Issue: [#55](https://github.com/Denolle-Lab/phasenet-retrain/issues/55). Owner: @Akashkharita.

**Dependencies:** SU-01 ([#54](https://github.com/Denolle-Lab/phasenet-retrain/issues/54)) releases the pilot partitions. Read-only census may begin earlier; training/mining may not. Reuse #37 reference/coverage eligibility principles.

**Timebox:** 10 working days after the pilot partition release; read-only census can start sooner.

**Work:**

- Census retrievable post-2002 Rainier/St. Helens continuous surface-event records and reviewed non-target windows from permitted stations; census Newberry/Hood development and separate calibration station-days. Do not commit to harvesting every year.
- Join and deduplicate the PNSN raw export, curated exotic data and labelled CSV. Count unique event families, valid hours, actual components/native bandwidth, SNR, process evidence and geographic/instrument coverage; report missing waveforms and retrieval cost.
- Verify original timestamp precision, P/S/first-arrival interpretation and quality-code semantics. Conduct blinded stratified re-picking; preserve uncertainty intervals and ambiguous labels rather than deleting them by a global move-out cutoff.
- Keep reviewed negatives, unreviewed background and uncertain event-bearing intervals distinct. Review earthquakes/codas, thunder/blasts, tremor, environmental noise, gaps and clipping. Catalogue absence alone is not a negative label.
- Audit observer start/end clock and station alignment before using duration masks. Register ESEC/other global cases as candidates with exposure/overlap and observability checks, not automatic acceptance examples.
- Publish a bounded versioned waveform/arrival/validity manifest for the pilot, plus a label-review sheet and donor-pool provenance. Longer context must support real onset-position diversity and coda-only windows.

**Done when:**

- [ ] Unique-event counts and usable waveform hours are reproducible from hashed sources; raw/curated duplication is removed.
- [ ] Timestamp/quality definitions, blinded residual/uncertainty report and actual component inventory are recorded.
- [ ] Pilot positives, reviewed negatives, development and calibration manifests pass SU-01 exclusions; missing/invalid rows are logged.
- [ ] Per-process/distance/instrument candidate counts and review exposure establish which later acceptance claims are supportable.
- [ ] Akash records a measured pilot size, access gaps and review workload at the end of the timebox.

### SU-03 — Repair preprocessing, windowing, labels and model export contracts

Issue: [#56](https://github.com/Denolle-Lab/phasenet-retrain/issues/56). Owner: @Akashkharita.

**Dependencies:** SU-01 ([#54](https://github.com/Denolle-Lab/phasenet-retrain/issues/54)) plus the rate/component/label schema from SU-02 ([#55](https://github.com/Denolle-Lab/phasenet-retrain/issues/55)). Synthetic work can start before the full corpus. Adopt #34A/#34C shared fixes; coordinate #43 augmentation semantics.

**Work:**

- Pin model/runtime versions, rates, units, component order, response epoch, normalization type/axis, filtering, overlap/blinding and validity masks. Candidate input is 50 Hz/6000 samples; quantify alternatives rather than assume an optimum.
- Repair the installed VariableLengthPhaseNet save/load failure, then check numerical probability equality across training, CPU export/import and offline QuakeScope loading under the correct model class/cache namespace.
- Use one verified training/inference resampling contract; measure passband, stopband, impulse timing and boundaries for 20/40/50/80/100/200 Hz and represented noninteger rates. Include the shared Hann attenuation on upsampling; do not build an independent surface-only fix.
- Transport every onset, end time, uncertainty interval and validity mask through resampling/stretch/crop in absolute UTC. Do not double-resample, duplicate Z as three channels or turn fetch failures into noise.
- Implement explicit onset-containing, coda-only, negative and mixed-event sampling. Standard 180 s/P-at-70 s records cannot yield onset-free 120 s crops; use longer context or declared valid-support restrictions.
- Implement UN soft targets and the UPN arm contract: raw u,p in [0,1], n=max(0,1-u-p), divide all three by their sum, use masked soft-label CE on logits. Missing annotations are not negative labels. Duration supervision is a later independent masked head.

**Done when:**

- [ ] CPU/offline export round trip preserves model outputs within a stated tolerance; class/labels/threshold arguments are recorded.
- [ ] Rate/label fixtures preserve absolute arrivals within one target sample and quantify filter response; train/inference parity is demonstrated.
- [ ] Crop/offset distributions, gap/component masks and failure rejection pass synthetic boundary/overlap tests.
- [ ] UPN targets sum to one, including coincident arrivals; UN/UPN label order and missing-label masks are tested.
- [ ] A reviewed example sheet and code/config hashes release the contract for baseline scoring.

### SU-04 — Establish continuous baselines, calibration and acceptance protocol

Issue: [#57](https://github.com/Denolle-Lab/phasenet-retrain/issues/57). Owner: @Akashkharita.

**Dependencies:** SU-02 ([#55](https://github.com/Denolle-Lab/phasenet-retrain/issues/55)) reviewed data and SU-03 ([#56](https://github.com/Denolle-Lab/phasenet-retrain/issues/56)) contracts. Reuse verified #35 candidate extraction/matching and #38 calibration mechanisms; SU-01 governs all access.

**Work:**

- Persist continuous annotations/candidates for STA/LTA, envelope-duration detection, earthquake picker/ELEP baselines and the current QuakeScope cascade where available. Record model training/exposure history.
- Extract triggers independently at each threshold, deduplicate seams, use one-to-one matching and account for missing support/model failures. Avoid oracle-centered maxima or filtered peaks from merged low-threshold triggers.
- Calibrate on separate permitted station-days, with a global fallback and prespecified supported instrument strata; give baselines comparable tuning opportunities.
- Report onset recall/timing, misses and reviewed false U candidates per 24 valid observed hours by process/noise/instrument/year; use paired event-family/station-day uncertainty. Unreviewed unmatched candidates are a separate metric.
- Translate candidate rates into fleet exposure, review workload and precision over a plausible prevalence range. Keep station-candidate and network-event counts distinct.
- Freeze reference/coverage eligibility and the acceptance protocol before training: matching tolerances, per-regime recall floors, reviewed false-alarm budget/exposure, interval width, fallback for unavailable data and panel-retirement rules.

**Done when:**

- [ ] Baseline candidate/reference assignments, valid-hour denominators, failures and hashes are reproducible on permitted data.
- [ ] Calibration days are disjoint from training/mining/development/acceptance; operating points are frozen for comparisons.
- [ ] Continuous scorer fixtures cover close events, threshold splitting, seams, gaps, duplicates, zero references and failures.
- [ ] Acceptance candidates are checked for ESEC/SED/Alaska/classifier overlap; independent event counts support the stated per-regime interval widths or strata are explicitly exploratory.
- [ ] A committed protocol/baseline report releases the first pilot without reading sealed model scores.

### SU-05 — Train and diagnose the minimal UN pilot

Issue: [#58](https://github.com/Denolle-Lab/phasenet-retrain/issues/58). Owner: @Akashkharita.

**Dependencies:** SU-01 ([#54](https://github.com/Denolle-Lab/phasenet-retrain/issues/54)), SU-02 ([#55](https://github.com/Denolle-Lab/phasenet-retrain/issues/55)), SU-03 ([#56](https://github.com/Denolle-Lab/phasenet-retrain/issues/56)) and SU-04 ([#57](https://github.com/Denolle-Lab/phasenet-retrain/issues/57)) release their required artifacts before training.

**Work:**

- Train a small UN softmax feasibility baseline on the reviewed pilot. Begin with the 50 Hz/120 s candidate, explicit sampling fractions, a minimal documented optimizer recipe and matched training/inference preprocessing.
- Use training-internal validation for stopping, separate calibration for operating points, and development for model choices. Log effective labels, gradients, losses, compute, unique data and repeated exposures.
- First establish clean-data learning and continuous regression behavior. Then compare UN versus normalized UPN and scratch versus documented transferred initialization as controlled arms with task heads reinitialized.
- Use identical permitted row lists/update budgets and three paired seeds for shortlisted comparisons. Record pretraining exposure and retain each model artifact.
- Publish successes and failure cases by weak/emergent onset, component availability, close/coincident events, earthquake coda and real noise.

**Done when:**

- [ ] A reproducible baseline trains and exports under SU-03; run/config/data/split hashes and seeds are preserved.
- [ ] Continuous development comparisons include recall, timing/misses, calibrated candidate workload and grouped uncertainty against SU-04 baselines.
- [ ] A concise decision report selects a minimal recipe or diagnoses failure; weak/inconclusive results do not trigger automatic bulk scaling.
- [ ] No mining or model selection used calibration/acceptance outputs; a passing candidate is not yet a deployment release.

### SU-06 — Measure source transfer and controlled generalization improvements

Issue: [#59](https://github.com/Denolle-Lab/phasenet-retrain/issues/59). Owner: @Akashkharita.

**Dependencies:** SU-05 ([#58](https://github.com/Denolle-Lab/phasenet-retrain/issues/58)) establishes the minimal recipe or actionable diagnostics; all changes retain SU-01–04 data/evaluation controls.

**Work:**

- Prioritize experiments from pilot failures: rate at fixed duration, context at fixed rate, bandwidth-matched controls, Z-only versus 3C, and meaningful context measured by perturbing development examples.
- Test real-noise mixing with measured achieved SNR/validity, cross-event mixtures with all labels, component dropout, gaps/clipping, and valid instrument-response/filter changes one group at a time.
- Mine only designated training/mining days; independently review candidates, retain uncertain labels and keep disjoint evaluation days. A contamination-aware background arm needs a measured upper bound, capped contribution and sensitivity checks.
- Run nested event-family data subsets with fixed negative exposure and comparable update budgets. Distinguish extra examples from extra mechanisms/regions; flat curves do not prove global sufficiency.
- Evaluate process × distance × instrument transfer and cross-volcano swaps. Compare shortlisted arms with paired seeds and report uncertainty, cost and unsupported regimes.
- Keep independent sigmoid onset heads/duration masks as conditional arms with appropriate reviewed labels and loss contracts, not automatic additions.

**Done when:**

- [ ] Each experiment has a predeclared question, one controlled change or stated interaction, paired artifacts and an interpretable result.
- [ ] No additional donor/mining/transfer data breach the registry; uncertainty/low-SNR masks and all time transforms pass checks.
- [ ] A selected candidate improves the relevant development tradeoff or the report explicitly retains the simpler recipe.
- [ ] A coverage/power report states supported and exploratory regimes, and a measured compute/data budget supports any proposed scaling.

### SU-07 — Evaluate optional cascade and coverage-aware network association

Issue: [#60](https://github.com/Denolle-Lab/phasenet-retrain/issues/60). Owner: @Akashkharita.

**Dependencies:** SU-05 ([#58](https://github.com/Denolle-Lab/phasenet-retrain/issues/58)) and the selected SU-06 ([#59](https://github.com/Denolle-Lab/phasenet-retrain/issues/59)) station-level candidate; SU-04 supplies fixed evaluation/calibration rules. Optional: a documented decision to omit these stages is a valid result.

**Work:**

- Compare the picker alone with pick-centered QuakeXNet scores and, if warranted, a binary reranker trained on out-of-fold first-stage candidates from training/mining data.
- Audit each classifier weight for ESEC/PNW exposure. Measure conditional and end-to-end performance, placement errors, additional latency and an evaluated Z-only/EHZ fallback.
- Census observable stations per event before testing multi-station criteria. Compare interval-aware timing, envelope coherence and combined rules; do not assume emergence removes all move-out information.
- Report all-event and network-eligible recall, false network events and correlated-noise failures. A fixed 30 s gate over 50 km or unmodified P/S PyOcto is not a general U associator.
- Claim locations only after a process-appropriate source/propagation model and independent location validation. Record a decision to retain, omit or defer each stage.

**Done when:**

- [ ] Complete candidate-stream comparisons show a measured benefit at comparable false-alarm/workload budgets, or optional stages are explicitly omitted.
- [ ] Classifier exposure, fallback behavior and first-stage misses remain visible in the end-to-end report.
- [ ] Association denominators include station observability and independent reference events; correlated noise and sparse-network cases are tested.
- [ ] No unvalidated U picks are passed into earthquake P/S association; no location claim follows from coherence alone.

### SU-08 — Run sealed acceptance and deliver a verified SU artifact

Issue: [#61](https://github.com/Denolle-Lab/phasenet-retrain/issues/61). Owner: @Akashkharita.

**Dependencies:** SU-04 ([#57](https://github.com/Denolle-Lab/phasenet-retrain/issues/57)) freezes eligibility/protocol; SU-05 ([#58](https://github.com/Denolle-Lab/phasenet-retrain/issues/58))/SU-06 ([#59](https://github.com/Denolle-Lab/phasenet-retrain/issues/59)) select the candidate; SU-07 ([#60](https://github.com/Denolle-Lab/phasenet-retrain/issues/60)) records inclusion or omission of optional stages. All foundational contracts remain valid.

**Work:**

- Freeze the selected weight, runtime, preprocessing, U threshold/calibration, candidate extraction, supported components, optional cascade/association and complete provenance before sealed scoring.
- Evaluate once on the eligible held-out panel and reviewed continuous exposures. Apply the preregistered per-regime recall/timing/false-alarm criteria and uncertainty; retain exploratory labels for underpowered regimes.
- On failure/inconclusive evidence, keep the existing workflow and document limitations. A panel used to revise the model loses blind status; plan fresh confirmation or disclose repeated evaluation.
- Prepare QuakeScope model-class/cache loading, U threshold/schema, task-specific run identity, resume/deduplication, component fallback and separate association handling.
- Verify CPU/offline loading, station-day throughput, memory, overlap/I/O/latency and frozen integration fixtures preserving earthquake outputs. Publish the model card and rollout/rollback decision.

**Done when:**

- [ ] A signed-off release decision references frozen artifacts, all exclusions/exposure history and the complete acceptance report; no aggregate score hides failed supported-regime criteria.
- [ ] The exported artifact reproduces validation outputs in the production container, including offline load and supported channel configurations.
- [ ] SU outputs and resume identity are distinct from earthquake runs; integration fixtures preserve earthquake results.
- [ ] Positive release evidence supports the documented rollout scope, or a no-release result and next-data requirements are recorded. Fleet execution follows the recorded rollout decision.

## Tracking and evidence

The task definitions live in `su_picker/issue_definitions.json`; the verified
creation snapshot is `su_picker/github_issues.json`. The unchecked lists above
describe acceptance criteria, not synchronized live completion status. Follow
the linked GitHub issues for progress. Regenerate this document offline with
`python docs/su_picker/render_plan.py` after changing definitions or issue links.

Every experiment preserves model/preprocessing versions, source and manifest
hashes, split/exposure history, candidate/reference assignments, seeds, calibration
and valid-hour denominators. Sealed model scoring happens only after candidate
freeze; reference QA is separately logged. A panel used to redesign the model is
no longer blind. No acceptance, training or production deployment has been
completed by creating this roadmap.
