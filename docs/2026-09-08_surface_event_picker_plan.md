# Plan: a dedicated onset picker for surface events

*Revised 2026-09-10 on `audit/2026-09-10-surface-picker` after the
[surface-picker audit](2026-09-10_surface_event_picker_audit.md). The expanded
2026-09-08 proposal and its source inventory are preserved in commit `98543f3`.
This revision supersedes its training recipe and acceptance protocol. No
surface model has been trained or evaluated by this audit.*

## 1. Product and scientific scope

Working name: SUNet. First deliverable: a retrospective station-level model
that proposes the first observable seismic arrival associated with a specified
surface process. Its `U` output denotes an annotated surface onset, not a P
phase, source origin time, flow-front arrival, source duration or location.
Keep the earthquake picker running independently. Store U picks separately or
with an explicit task identifier; do not route them into earthquake association
as if they were P picks.

The target processes are rockfalls, landslides, avalanches and debris/lahar
flows with identifiable seismic onsets. The first validated coverage will be
limited by available labels. PNSN `su` examples include glacier-related sources
and uncertain mechanisms; the label alone does not certify a landslide.
Separate process-confirmed positives, analyst surface-category positives and
ambiguous/adjacent sources. Icequakes, calving and tremor require explicit
mapping; ambiguous cases remain diagnostic rather than forced negatives.

Do not assume every process is emergent, shares a spindle envelope, lacks
separable phases, or fits within 120 s. State the distance, frequency, process
and duration coverage actually tested. Long-period source inversion and
amplitude/size estimation are outside this first deliverable. Network detection,
location and a QuakeXNet cascade are later gated tasks. Operational warning
latency requires its own causal/finite-lookahead design and measurements.

## 2. Governance before model design

Use a dedicated surface registry, with explicit roles for **train, mining,
regression, development, calibration and sealed acceptance**. Reuse the
mechanisms being repaired in earthquake issues #33 and #44; do not automatically
inherit their event lists or declare all existing earthquake hold-outs blind
for this task.

For every event family and station-day record:

- Canonical event ID plus aliases across PNSN, curated files, ESEC, ComCat,
  SED, GeoNet, observer catalogues and model-derived catalogues.
- Process class, evidence tier, label source, source location/uncertainty when
  known, receiving station and inventory epoch, actual components and UTC bounds.
- Role and reason, source-file hashes, waveform availability and annotation
  completeness, past model/classifier exposure and reviewer provenance.
- All linked traces, noise donors, synthetic parents and pseudo-labels. They
  inherit the strictest applicable split restrictions.

Split event families before traces. Keep overlapping time windows and common
station-day background out of separate roles. Check spatial and temporal
buffers and duplicate events across source catalogues. Receiving-station
proximity to a volcano is a station-domain proxy, not a source location.
Resolve shared stations and events across nominal volcano groups explicitly.

Retain St. Helens/Rainier training and other Cascade volcanoes development as
candidate partitions until counts, aliases and station overlap are verified.
The proposed 2016/2021 temporal hold-outs are candidates too; register their
role deliberately. Add out-of-region development separately from acceptance.
Existing Alaska and ESEC classifier experiments must be recorded as exposure.

Metadata and blinded reference QA may inspect acceptance candidates before
freeze. Model-output access is separate and logged. Complete reference and
coverage eligibility before freezing the acceptance panel; do not select cases
by whether a model succeeds. Once a panel informs a training or design decision,
it is development/regression. A failed acceptance run consumes its blind status.
Do not mine development, calibration or acceptance predictions for training.

## 3. Data and labels: inventory before volume

The earlier inventory is a starting point, not a verified training manifest.
The following local counts were checked in the audit; curated/API counts remain
reported values until a versioned census reproduces them.

| Source | Current evidence | Proposed use and condition |
|---|---|---|
| PNSN raw `su` export | 21,721 picks, 12,129 event IDs; 9,380 single-pick events; 20,777 P and 944 S entries | Primary onset source after quality/time semantics and waveform retrieval are verified |
| PNW curated exotic data | Earlier census reports 8,912 surface traces / 5,425 events, 180 s at 100 Hz with most P onsets at 70 s | Bootstrap corpus; deduplicate against export; recover native rate and true component availability |
| PNSN labelled CSV | Earlier census reports 13,465 picks / 7,523 events with impulsivity | Join by event, station and time; do not add as independent examples |
| Thelen observer catalogues | 133 Rainier, 59 St. Helens, 18 Hood rows; schemas differ; 130 Rainier rows have named start/end values | Ground-confirmation and duration-review material, not ready station-level masks |
| ESEC | Earlier inventory reports 245 local catalogue rows, 173 events with downloaded waveforms | Acceptance candidates after duplicate/exposure review and station-onset annotation; no training by default |
| SED / GeoNet / Alaska ComCat | Earlier plan records accessible event/phase products | Census retrieval and type/arrival semantics; candidate SED training and GeoNet/Alaska development, excluding acceptance duplicates |
| Illgraben / Ruapehu / OVPF / INGV-OV | Access requests previously deferred | Keep deferred; determine role before use if data become available |
| Model-derived Rainier detections | Candidate pool, not truth | Training-pool review/mining only; retain provenance and unknown class |

The raw export contains 7,880 pre-2002 picks. Continuous PNSN archiving starts
in 2002 in the dataset description, so earlier samples require an explicit
triggered-archive retrieval audit. Deduplicate first; do not sum export and
curated counts. Publish usable **unique event families**, station records,
observed hours, process types and geographic/instrument coverage, with exclusion
reasons. Native-rate samples and 100 Hz curated resamples are different assets.
[PNW dataset description](https://seismica.library.mcgill.ca/article/view/368/868)

For first arrivals, preserve the original phase annotation, converted UTC,
timestamp precision, pick uncertainty, quality code and its documented meaning.
An S-only record must not be silently treated as a first P-like arrival. An
integer-second export does not prove integer-second source precision and does
not impose Gaussian sigma of at least one second. Verify formatting against
higher-precision originals and conduct blinded re-picking of a stratified subset.
Use interval uncertainty where appropriate; sigma=1 s is a candidate U kernel,
not a physical constant. Earthquake P targets have a separate uncertainty policy.

Do not weight loss by an unexplained quality number or infer uncertainty solely
from interstation move-out. With unknown/moving sources, geometry and propagation
also contribute to residuals. Apparent speed above 6 km/s is a review flag only.
Confident-learning and other disagreement tools prioritize review; they do not
automatically delete label errors. Observer start/end times require a defined
reference clock and station-specific conversion before a duration head is trained.

## 4. Negatives, background and field cases

The key rejection populations are earthquake P/S/codas and aftershock sequences,
explosions, thunder/sonic booms, tremor/long-period activity, storm/wind/river noise,
traffic/helicopters, glitches, gaps and clipped traces. Source identity and onset
annotation are separate: an earthquake-only coda may be non-U without containing
a P onset. Absence from a surface catalogue does not establish a negative.

Maintain three annotation states: reviewed target, reviewed non-target, and
unreviewed/ambiguous. Use reviewed non-target windows for supervised negatives;
mask uncertain intervals or use an explicitly tested partial-label objective.
Random unreviewed background may support annotation/mining, but must not silently
become all-N targets. Keep catalogues from all relevant source types in the QA
join, including exotic events absent from ComCat.

For a later scaling arm, unreviewed background may enter an explicitly specified
contamination-aware objective. First estimate contamination by blinded stratified
review, record an upper bound and sensitivity analysis, cap its sampling/loss
contribution, and compare against reviewed negatives on disjoint development
days. This is not a general prohibition on learning from background; the small
first pilot uses reviewed negatives because its contamination is not yet bounded.

Hard-negative mining occurs only on registered training/mining days. Review
candidates with a source-neutral protocol, retaining uncertain or possible
uncatalogued surface events. Store the model version that generated each
candidate. Use disjoint days for reporting improvement. Do not train on every
unmatched pick, particularly in earthquake codas that can contain real failures.

The field panel must contain continuous hours, including pre-onset background,
long codas, multiple/coincident events, sparse networks, vertical-only sensors,
mixed sampling rates, clipping/gaps and seasonal environmental noise. Small local
confirmed events matter alongside spectacular distant landslides. Register the
previously examined atmospheric-river examples as development/regression; reserve
fresh station-days for acceptance when needed. Curated event windows alone cannot
establish a deployment false-alarm rate.

## 5. Preprocessing, windowing and model contract

Candidate backbone: small SeisBench `VariableLengthPhaseNet`, **50 Hz, 6000
samples, 120 s**, initially three-component input with valid component masks and
a separately evaluated Z-only option. This is a starting candidate, not a
validated optimum. The installed model has 268,443 parameters for three outputs,
but its save/load round trip currently fails. Repair/pin that contract before
training; no assumption of a drop-in `phasenet/` weight is permitted.

The choice is motivated by the earlier 150-trace PNW spectrum summary: about 75%
of energy at 1–10 Hz and 10% at 10–20 Hz, with 120 s accommodating much of the
local envelope population. Preserve that rationale while treating it as limited
PNW evidence, not coverage of every process or instrument. The cited 4 s effective
receptive-field figure was a legitimate literature citation; useful context for
the installed/trained model still requires the proposed perturbation experiment.

Record and test stored rate, original instrument rate, units, channel order,
response epoch, filtering and resampling paths, normalization type/axis,
window overlap, edge blinding, gap/padding masks and model output semantics.
Explicitly choose normalization: this class defaults to peak, whereas the old
proposal assumed standard deviation. Train and inference must implement the same
contract. Test CPU export/import for output equality and offline loading in the
actual QuakeScope environment.

The installed resampler uses different paths for integer downsampling and other
ratios. The audit measured strong attenuation on the default 40→50 Hz path and
the shared 20/40→100 Hz earthquake paths;
“use SeisBench everywhere” alone is not a specification. Verify 20/40/50/100 Hz
routes using tones, impulses, absolute timing and real development traces. Choose
and version an anti-alias/interpolation method with measured passband and
stopband behavior. Do not infer restored high frequencies after upsampling.
A 0.5 Hz high-pass is a candidate preprocessing choice with explicit phase and
edge behavior, not a universal surface-process filter.

The noninteger/upsampling route uses ObsPy's frequency-domain Hann window even
with `no_filter=True`. The added probes reproduce its predicted −6.02 dB gain at
half the source Nyquist and −32.23 dB at 0.9 of it. Track the common correction
under [#34](https://github.com/Denolle-Lab/phasenet-retrain/issues/34#issuecomment-5621902058),
with #43 consuming the same contract for augmentation and SU-03 for the surface
model. Verify 20/40/50/80/100/200 Hz and represented noninteger routes. Select an
explicit polyphase or filter-plus-interpolation implementation using passband,
alias rejection, timing and boundary checks; do not presume a method name alone
certifies it. Already resampled 100 Hz inputs need separate provenance. This
is a shared input mismatch, not a proven cause of historical model failure.
[ObsPy resampling contract](https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.resample.html)

Represent annotations in absolute UTC and transform **all** onsets, end times,
uncertainty intervals and validity masks through resampling, stretching and
cropping. Reuse loaders only after #34A/#34C or equivalent verified fixes; current
loaders still have alignment/component assumptions identified by the audit.
Never turn an I/O failure or an unlabelled interval into a valid noise example.

A 120 s crop of the standard 180 s trace with onset at 70 s always contains the
onset, at offsets 10–70 s. Therefore random cropping cannot provide either full
position coverage or onset-free tails. Obtain longer continuous spans, or declare
the restricted offsets and use reviewed context extension with validity masks.
Draw onset-containing, coda-only, reviewed-negative and mixed-event windows
explicitly; log realized proportions and position distributions. No circular
wrap or event repetition disguised as continuous context.

## 6. Targets and the first controlled pilot

Start with **UN softmax** to establish a working surface-onset baseline. N means
non-U onset at that sample, not absence of all seismic energy. The first output
ablation is **UPN**, with P denoting an earthquake P arrival only where validly
annotated; it is a hypothesis about rejection, not a guaranteed improvement.
A P-onset head alone does not label the entire earthquake coda.

For the first UPN arm choose normalized categorical targets: form raw kernels
`u,p` in [0,1], set `n=max(0,1-u-p)`, then divide **all three** by `u+p+n`.
Within each class use the maximum of that class's onset kernels, preserving the
individual onset list for scoring. This gives a unit-mass soft target; coincident
unit U/P kernels become `[0.5,0.5,0]`. Setting N alone is insufficient when
`u+p>1`. Use soft-label cross-entropy on logits with validity masks. This is
deliberate categorical competition, not two simultaneous near-unit probabilities.
An independent U/P sigmoid+BCE arm remains an alternative if overlap performance
justifies it; use no redundant N sigmoid. Missing annotations are not zero labels.
Document simultaneous processes and uncertainty/masks, and check every transform.

An optional duration head comes later, with independent masked supervision.
Switching `output_activation="sigmoid"` on the stock model changes every output;
it does not create a mixed softmax/sigmoid head. Keep weak duration estimates
separate from manual ends and out of acceptance references.

First verify clean-data learning and continuous regression behavior on a small,
audited pilot. Use scratch as one initialization and a documented transferable
encoder as a controlled arm; reinitialize task-specific outputs and account for
pretraining exposure. AdamW, a documented learning-rate schedule and early
stopping are provisional defaults, with the same update budget across comparisons.
Use training-internal validation for stopping, separate calibration for operating
points, and development for model choices. Run three seeds for shortlisted arms;
seed spread and sampling uncertainty answer different questions.

Add augmentations incrementally after the pilot works:

- Real-noise mixing from permitted donors, with SNR defined in a stated band and
  valid pre/post intervals. Track template noise, achieved SNR and target masking
  when arrivals become unobservable. Include a low/negative-SNR challenge if it
  occurs in deployment; the old uniform 0–20 dB recipe did not provide that range.
- Reviewed cross-event mixtures with all parent labels and masks transformed.
  Avoid contradictory categorical targets for overlapping processes.
- Rate/context augmentation with label transport and alias checks. Earthquake
  stretching/low-pass filtering is an optional controlled challenge, not a
  substitute for real emergent earthquakes and codas.
- Component dropout, gaps and clipping with matching availability/loss masks.
  Use verified component order; zero missing components rather than duplicating Z.
- Instrument-response variation only with valid source/target responses and stable
  passbands. A high-pass approximation is a filter experiment; short-period data
  cannot be transformed into genuine missing broadband information.

Compare amplitude scaling before/after normalization explicitly: simple scaling
before unit normalization may cancel. Do not ban Gaussian noise, polarity changes,
teachers or loss families solely because a confounded earthquake retrain failed;
keep them outside the minimal pilot unless a controlled question justifies them.

## 7. Continuous evaluation and acceptance

Build the surface scorer from repaired #35 extraction/matching contracts, not
oracle-centered maxima. Candidate generation, thresholding, hysteresis/peak
merging, seam deduplication and one-to-one matching must be consistent across
models. In a threshold sweep, extract candidates according to each threshold's
actual rule rather than filtering peaks of merged low-threshold triggers.

Calibrate on **separate registered days**, with a global fallback threshold and
only prespecified instrument strata that have enough exposure. No target-region
threshold tuning on acceptance. Use the same threshold-selection opportunity for
STA/LTA, envelope-duration triggers, earthquake pickers/ELEP and the existing
QuakeScope cascade. Current baseline results are diagnostic until their evaluation
and training exposure are reconciled.

Report:

1. Station-onset recall at frozen operating points and matched calibrated budgets,
   with the same reference/matching definition across suites. Pre-register a
   tolerance curve (for example 1/2/5/10 s) and one headline rule or interval-aware
   criterion before model selection; no easier tolerance for acceptance.
2. Conditional timing residuals plus the missed fraction, by annotation quality,
   SNR, process, distance where known, station/instrument, region and year.
3. Reviewed false U alarms per 24 **valid observed hours**, separately for quiet,
   environmental and earthquake periods, plus unmatched candidate rate where
   review is incomplete. Report exposure and review completeness explicitly.
4. Paired event-family and station-day block uncertainty for comparisons. Do not
   reuse the existing independent-value bootstrap unchanged or substitute a
   two-proportion random-window test for deployment precision.

Before acceptance, set the false-alarm budget (one per station-day remains a
candidate, not an achieved rate), required reviewed exposure, recall floors per
supported regime, uncertainty rules and decision procedure. A point estimate on
a few correlated days is insufficient. Report strata with inadequate power as
such; do not fill their gaps with an aggregate success claim.

Also pre-register campaign workload and precision sensitivity, not just the
station-normalized rate. If `D=sum(valid station-hours)/24` and the false-pick
rate is `f`, expected false station candidates are `f*D`. At 1,000 full station-days
and f=1 this is 1,000 candidates/day, not 1,000 distinct events. With an
*illustrative*, unmeasured true-arrival rate of 0.01 per station-day and recall
0.8, station-candidate precision is `0.008/(0.008+1)=0.79%`. Replace these examples
with census exposure and a plausible prevalence range; retain separate counts of
station candidates, reviewed workload and associated events. Require measured
end-to-end precision/false-event rates if the product depends on association.
Correlated noise invalidates a simple independent-station suppression calculation.

For each process × distance × instrument stratum, publish eligible independent
event-family counts, planned confidence interval, desired width and attainable
recall floor **before scoring**. The 173 downloaded ESEC events are neither 173
independent examples per regime nor all eligible arrivals. Illustrative two-sided
95% exact binomial intervals are 0.692–1 for 10/10 detections, 0.832–1 for 20/20,
and 0.929–1 for 50/50; 16/20 gives 0.563–0.943. The committed probe calculates
these examples. They assume independent events; clustered data need the grouped
analysis above. Label underpowered strata exploratory or acquire more independent
events; do not pool different mechanisms merely to clear an acceptance floor.

Acceptance candidates include out-of-PNW ESEC/confirmed surface events and fresh
continuous noise/earthquake periods. Check every candidate against SED, Alaska,
GeoNet, classifier pretraining and previous examinations. Keep independent
station-onset annotations; catalogue source times alone cannot score station
picking. Determine waveform/label eligibility and rejection reasons before model
access. The 100 km cutoff is a proposed coverage stratum, not a guarantee of an
observable arrival. Freeze one candidate including preprocessing, thresholds and
cascade configuration, then evaluate once. A failure makes that panel used;
subsequent changes require a fresh blind confirmation panel or an explicit
repeated-evaluation disclosure.

## 8. Optional cascade, network detection and deployment

A QuakeXNet stage earns its place by improving complete-pipeline recall versus
reviewed false-alarm rate on development candidates. Use pick-centered windows
and record latency/lookahead. Train a binary reranker on out-of-fold first-stage
candidates from training/mining data, not on hand-centered positives versus
unreviewed misses. Record each classifier weight's ESEC/PNW exposure. Measure
conditional and end-to-end performance: rejection cannot recover stage-one
misses. Maintain an evaluated fallback for missing horizontals/EHZ. The existing
16–78% Alaska placement example is overall agreement on 32 examples, not a
surface-only accuracy result.

Network association is a separate experiment. First measure the number of
observable stations per event. Report single-station recall, all-event recall,
network-eligible recall and false network events with distinct denominators.
Three stations are a candidate criterion only where coverage supports it. Derive
move-out windows from geometry, process and empirical arrivals; a fixed 30 s
window across 50 km is not valid over the whole proposed 1–3 km/s range. Review
extended/moving-source cases before claiming point locations.

Emergence increases arrival uncertainty; it does not imply the absence of all
useful travel-time structure. Compare interval-aware timing, envelope coherence
and their combination on reviewed development events. Reject the unmodified
earthquake PyOcto contract for U arrivals, but do not assume envelope coherence
will reject regional coherent noise or deliver a location. Measure that gain.

QuakeScope integration must cover model class/cache resolution, U threshold
configuration, output schema, component grouping, task-specific run identity,
resume/deduplication and separate association semantics. Its current association
uses P/S velocities and a backend-specific database path. Add integration checks
on frozen fixtures that preserve earthquake outputs. Offline CPU loading,
station-day throughput, memory, overlap and I/O costs must be measured in the
production container. No station-day cost claim follows from parameter count
alone. No campaign deployment is part of this plan audit.

## 9. Sequential branches and gates

These are **proposed surface checkpoint IDs**, not new GitHub issue numbers.
Create a separate surface epic/tickets when implementing; keep earthquake issues
#33–#50 scoped to their shared contracts and earthquake work. Branch one ready
checkpoint at a time from the surface integration head; use focused reviewed
imports of shared fixes rather than copying unfinished loader/scorer code.

| Checkpoint / branch | Deliverable and release criterion | Dependencies |
|---|---|---|
| SU-01 `surface/01-taxonomy-suites` | Task taxonomy, evidence tiers, exposure/alias registry and enforced suite roles; no ambiguous background promoted to negatives | Adopt #33/#44 mechanisms or equivalent verified guards |
| SU-02 `surface/02-corpus-label-audit` | Deduplicated retrievable census; quality/time definitions; blinded onset review; components, durations, station coverage and negative-review plan | SU-01; shared #37 eligibility rules |
| SU-03 `surface/03-model-data-contract` | Numeric CPU export/import, native-rate/filter tests, all-label time transport, crop coverage and component/mask parity | SU-01; #34A/#34C or verified equivalent; consumes SU-02 metadata |
| SU-04 `surface/04-continuous-baselines` | Reproducible baseline streams, one-to-one scorer, separate calibration days, reviewed exposure and block intervals; acceptance eligibility freeze | SU-02/03; shared #35/#38 contracts |
| SU-05 `surface/05-minimal-pilot` | UN feasibility, then UN/UPN and scratch/transfer controlled comparisons with fixed exposure/update budgets | SU-01–04 gates passed |
| SU-06 `surface/06-generalization-arms` | Targeted rate/context, Z-only, augmentation, mining and data-diversity tests; shortlisted three-seed comparisons | SU-05 diagnostics; training/mining review only |
| SU-07 `surface/07-cascade-association` | Optional reranker and network-detection gains with coverage-aware denominators; locations only if justified | SU-05/06 station-level evidence |
| SU-08 `surface/08-acceptance-deployment` | Frozen candidate/protocol, sealed evaluation, supported-regime decision and tested QuakeScope artifact | SU-04 freeze and selected SU-05–07 configuration |

### First sprint and ownership proposal

Use two **10-working-day timeboxes** for SU-01 and SU-02. T0 is the agreed sprint
start with data access and implementers assigned; these are estimates, not
calendar commitments or permission to skip gates. Proposed scientific decision
owner: Marine Denolle. Named implementation and independent label-review owners
must be recorded at T0; they are currently unassigned, so the sprint is not yet
scheduled. SU-03 likewise needs a named shared preprocessing/export owner and
a measured fixture-review estimate at T0.

| Timebox | Concrete scope | End-of-timebox decision |
|---|---|---|
| SU-01, T0 to working day 10 | Taxonomy, role/alias registry, exposure history, source/station map and review rubric for the pilot | Release permitted pilot partitions or publish the unresolved collision/access list |
| SU-02, 10 working days after the pilot partition is released | Post-2002 Rainier/St. Helens continuous candidate positives and reviewed negatives from permitted stations; Newberry/Hood station-days for development, subject to SU-01 overlap checks; blinded onset review and retrievable census | Publish counts, reviewed exposure, label uncertainty and per-regime power; decide a bounded pilot size from usable data |

Read-only census may start while roles are drafted; training/mining waits for
the role checks. Keep disjoint calibration days and event-family/station-day
separation inside this named corpus. Do not count all post-2002 data as a
two-week harvest commitment.

Proposed tracking change: split shared executable contracts into child tickets
under #34 (time/coordinate invariants; resampling/normalization/export parity) and
#35 (candidate extraction/matching; exposure/grouped uncertainty). Cross-link
SU-03/SU-04 to those tickets and migrate the existing checkpoint checklists when
creating them, avoiding two competing definitions of done. These are ticket
proposals, not newly created issues or completed dependencies. Use one worktree
per active session; the PR #52 review follow-up uses an isolated worktree without
switching the other session's branch.

Within SU-06, vary rate at fixed duration and context at fixed rate; use
bandwidth-matched controls before changing kernels/depth. Measure useful context
on trained development examples; it is not guaranteed by input length. Data
scaling uses nested event-family subsets (not trace subsets), fixed negative
exposure and comparable updates, and compares added regions/processes as well as
added rows. A flat 50–100% curve is evidence about this experiment, not proof that
more diverse labels cannot help. Preserve acceptance assets regardless of curve
shape. Set schedule and compute budget from census and pilot measurements, not
an unconditional 19-week commitment.

Next work: SU-01, followed by SU-02 census and SU-03 contract repairs. This audit
completes the planning review; the implementation checkpoints remain open.

## Evidence and reading

- [Audit findings and measured probes](2026-09-10_surface_event_picker_audit.md).
- [Applied earthquake roadmap](2026-09-09_issue_plan.md) and
  [checkpoint execution plan](2026-09-10_issue_execution.md).
- [PNW data and annotation conventions](https://seismica.library.mcgill.ca/article/view/368/868).
- [Picker evaluation and sampling protocol](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023499).
- [QuakeXNet study](https://seismica.library.mcgill.ca/article/view/2068).
- [Low-frequency earthquake picker precedent](https://arxiv.org/abs/2311.13971):
  relevant motivation, not direct evidence of surface-process transfer.

The original proposal's larger bibliography and detailed access inventory remain
in `98543f3`. Literature analogies motivate controlled arms; they do not certify
this pipeline's labels, supported processes, sampling routes or deployment cost.
