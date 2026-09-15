# Audit of the surface-event onset picker plan

2026-09-10. Branch: `audit/2026-09-10-surface-picker`, based on the earthquake
integration branch at `6b5297c`. The complete user-edited proposal is preserved
in commit `98543f3`; section references below refer to that version. The
[revised plan](2026-09-08_surface_event_picker_plan.md) is the execution document.

Follow-up: [review of Copilot and Fable's PR #52 comments](2026-09-10_surface_picker_review_response.md)
records accepted changes, qualifications and the extra shared-path probes.

## Decision

Pursue a separate surface-onset project and model, sharing repaired data and
evaluation contracts with the earthquake project. The proposal contains useful
source inventories and realistic confusers, but it is **not ready for training**.
The principal risks are target ambiguity, false-negative supervision, evaluation
reuse, a deterministic cropping error, and unverified preprocessing/export.
Increasing the number of augmentation arms would not resolve those defects.

Keep the first deliverable narrow: retrospective, station-level first-arrival
candidates for specified surface processes. Network detection, location, warning
latency and global process coverage each need separate evidence. A broad `su`
label is not ground truth for all landslides, lahars and avalanches.

## Evidence and limits

Read the full 787-line proposal; inspected the local loaders, labels, metrics,
SeisBench model/resampling source, QuakeScope picker and classifier plan; examined
local PNSN pick and observer-catalogue files; checked primary publications.
[Probes](surface_audit_2026-09-10/probes.py) and
[results](surface_audit_2026-09-10/probe_results.json) record reproducible synthetic
checks, input hashes and the installed environment. No real waveform was scored,
no model was trained, no server census was performed, and no external repository
was modified. Public API catalogue counts and the 8,912-row curated surface subset
were not independently recomputed in this audit. They remain census inputs.

## Findings, ordered by what must change before training

### S01 — Blocking: the task label is broader than the proposed product (§1, §4)

The PNW `su` class is an analyst category, dominated by glacier-related sources;
it is not a uniformly ground-verified mass-movement catalogue. The original
dataset describes incomplete analyst attention, sparse station picks, missing
source locations and origin times. Treat source identity, station first arrival,
source initiation and flow-front passage as distinct annotations.
[Ni et al.](https://seismica.library.mcgill.ca/article/view/368/868)

**Change:** add a taxonomy and evidence tier to each event; map uncertain `su`
subtypes to an explicit unknown/diagnostic category. Preserve station uncertainty
separately. Observer start/end times are not station waveform masks without a
station/time-reference join. The local observer files have 133/59/18 rows, but
different schemas; only 130 Rainier rows have both explicitly named start/end
fields. These are not 210 verified, aligned station masks. No `S` output is a
reasonable scope choice; 944 S entries are 4.35% of *picks*, not of events, and
624/8,912 is about 7% of traces. Sparse annotations do not prove absent physics.

### S02 — Blocking: absent catalogue entries become false negatives (§4.2, §4.2b)

The proposal declares whole earthquake days negative and labels every unmatched
`U` candidate as a negative. Both can suppress real uncatalogued surface events,
including earthquake-triggered failures. The statement that PNSN catalogue
absence verifies noise is particularly unsafe when the target exotic catalogue
is not fully represented in ComCat. This follows from the catalogue limitations
above, not from an observed SUNet error rate.

**Change:** separate reviewed negatives from unreviewed background. Independently
review mined candidates, retain an ambiguous class or ignored interval, and use
audited background hours for false-positive denominators. Otherwise report
unmatched candidates per day. Mine only designated training/mining days.

This is a requirement for the small audited pilot, not a prohibition on using
unreviewed background at scale. A later contamination-aware objective can admit
it with a measured contamination bound, bounded contribution and sensitivity
checks against disjoint reviewed data; it cannot certify those samples as truth.

### S03 — Blocking: development and acceptance roles contradict themselves (§4.3, §6–7)

Section 4.2b excludes development from mining, but §6.2 sends development errors
into the next training arm. Once that happens, the same set no longer measures
unseen generalization. Section 6.2 also scores held-in aftershock training hours.
The final schedule says a failed acceptance run leaves the suite unread; it does
not. ESEC can duplicate SED training events or Alaska/GeoNet development events.
The optional classifier has its own training and evaluation exposure history.

**Change:** version a joint event-family/station-day registry with train, mining,
regression, development, calibration and sealed-acceptance roles. Check aliases,
space/time overlap, raw/curated duplicates, noise donors, transferred weights and
classifier exposure. Freeze eligibility independently of model scores. Existing
QuakeXNet Alaska results are development evidence, not a fresh blind panel.
Acceptance failures retire the panel from blind status. Metadata/label QA is
allowed and logged independently from scoring. Share #33/#44 mechanisms, not
their earthquake-specific case assignments.

### S04 — Blocking: the crop recipe cannot deliver what it promises (§5.2)

For the standard 180 s record with a P-aligned onset at 70 s, valid 120 s crops
start between 0 and 60 s. The onset therefore lies between 10 and 70 s in **every**
crop. The synthetic probe enumerates all 3,001 starts at 50 Hz and finds zero
onset-free crops. A random-third selection is not a tail-third selection. The
published sampling rule explicitly permits picks in its random windows.
[Münchmeyer et al., §2.4](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023499)

**Change:** obtain sufficiently long continuous data for genuinely random onset
positions and reviewed coda-only windows. Where unavailable, explicitly handle
short context with validity masks and report the attainable offset range. Do not
manufacture an onset-free tail by forgetting a label or silently wrapping a trace.
Apply rate, stretch and crop transforms to every annotation using absolute UTC.

### S05 — Blocking: the proposed export fails in the installed environment (§3, §7)

`VariableLengthPhaseNet(in_samples=6000, sampling_rate=50, phases="UPN")` has
268,443 parameters and returns `[1, 3, 6000]`. However, `save()` followed by the
same class's `load()` fails in SeisBench `0.9.1.dev16+g3667d44` with
`TypeError: SeisBenchModel.__init__() got an unexpected keyword argument
'norm_amp_per_comp'`. Inherited export writes constructor arguments that this
class forwards incompatibly. This is an installed-version finding, not a claim
about every SeisBench release.

**Change:** require a pinned, corrected model serialization contract before a
pilot; test numeric output equality after CPU save/load and offline container
loading. Use the actual model class's cache namespace, not `phasenet/` by default.
QuakeScope currently sets only P/S thresholds (`picker.py:388–395`), and its
association routine uses a P/S velocity model and DocumentDB methods
(`picker.py:398–445`). A `U_threshold` edit alone does not implement surface-event
association or certify the production output and resume contracts.

### S06 — Blocking: the preprocessing contract is incomplete (§2, §3.2, §5)

The installed variable-length model defaults to **peak**, not standard-deviation,
normalization. Its resampler low-passes and decimates for integer downsampling,
but calls ObsPy FFT resampling otherwise. Tone tests measure 40→50 Hz amplitude
changes of −6.02 dB at 10 Hz and −32.23 dB at 18 Hz. For 100→25 Hz they measure
−1.16 dB at 10 Hz, −4.59 dB at 12 Hz and −15.91 dB at 15 Hz (the latter folds
to 10 Hz). These controlled long-sinusoid tests expose implementation behavior;
they are not broadband surface-event performance measurements.

The review follow-up also reproduces the same Hann gains on 20/40→100 Hz:
−6.02 dB at 5/10 Hz respectively and −32.23 dB at 9/18 Hz. This belongs to the
shared #34 preprocessing contract, with #43 and SU-03 consuming the fix. The
issue applies to inputs taking this installed resampling route; it is not proof
that every historical waveform took that route or that this caused the retrain
failure. Source/stored-rate provenance and controlled comparisons remain needed.

**Change:** pin stored rate and instrument rate separately, units, channel order,
validity masks, response epoch, filter order/corners/phase, resampling method,
normalization axis, and inference overlap/blinding. Validate every supported
native→model route, including noninteger and upsampling paths, impulses and gaps.
The current manifest loader still resamples without updating pick indices and
duplicates a 1D signal across three channels. Reuse requires #34A/#34C repairs;
the claim that those loaders are reusable unchanged must be removed.

### S07 — High: window length, receptive field and bandwidth are conflated (§3.2)

Changing `in_samples` does not add layers in the installed class. The probe's
center output has structural convolutional support spanning 3,331 samples
(66.62 s at 50 Hz), excluding normalization. This is neither a trained effective
receptive field nor evidence that the entire 120 s envelope is used. The 4 s
statement appears in an archived 2022 paper PDF; it is not a measured constant
for this model/version. Do not infer an 8 s effective field from it.
[Archived paper, §2.3](https://publikationen.bibliothek.kit.edu/1000143103/146775569)

The original 4 s citation was legitimate; the qualification concerns transfer
to this implementation. Likewise, 50 Hz/120 s had a useful PNW spectral/context
rationale. Preserve that rationale while measuring its applicability elsewhere.

**Change:** retain 50 Hz/120 s as a candidate. Compare rate at fixed duration and
context at fixed rate, with bandwidth-matched controls and measured throughput.
Twenty/25/50 Hz with simultaneously changed duration confounds the experiment.
Upsampling cannot recover missing instrument bandwidth; a simple high-pass is
not a reversible broadband↔short-period response conversion. Measure useful
context with perturbations on development examples after training. Window
normalization and zero-phase filtering also matter for latency; first delivery
should be explicitly retrospective.

### S08 — Blocking: overlapping labels and the event-mask loss are underspecified (§3.3, §5.1)

The suggested `N=1-max(U,P)` copied from `make_labels()` yields target mass
`1+min(U,P)`. Two σ=1 s onsets separated by 0.5 s produce mass 1.9668 in the
probe. That is not the declared categorical distribution. The installed
`output_activation="sigmoid"` switches *all* channels, not a separate mask head.
Missing earthquake P annotations, gaps over an onset and uncertain surface
intervals also need a loss-validity policy.

**Change:** make UN the minimal feasibility baseline; UPN is a controlled arm.
For the first UPN arm, choose `n=max(0,1-u-p)` followed by normalization of all
three raw targets by their sum and soft-label cross-entropy. Clipping N alone
does not repair the distribution. Independent U/P sigmoids with BCE remain a
separate arm if simultaneous-arrival behavior justifies it. A duration head uses
separate masked supervision after station-level end labels exist. Validate
shape, channel order, probability mass and time transforms before optimization.

### S09 — High: timestamp precision is mistaken for uncertainty (§5.1, §7)

The local export confirms 21,721 picks, 12,129 events, 9,380 single-pick events
and no displayed fractions. That proves export precision, not the precision of
the source database. Even known rounding to one second gives a quantization-only
standard deviation of 0.289 s, not a mandatory Gaussian σ≥1 s. The meaning and
direction of `quality` must be documented before using it as a confidence weight;
769 picks have zero. A multi-station move-out residual mixes source position,
source motion, phase identity, propagation and pick error. Apparent interstation
speed above 6 km/s can result from geometry and is not a valid automatic veto.

**Change:** re-pick a stratified, blinded subset and record interval uncertainty;
confirm timestamp formatting and quality semantics with the data owner/schema.
Use separate uncertainty policies for U and earthquake P. Review outliers instead
of deleting uncertain but physically interesting events with a global velocity
cut. Pseudo-label disagreement is a review queue, not an error certificate.

### S10 — High: continuous metrics need reviewed denominators and grouped uncertainty (§6)

The existing `bootstrap_ci` resamples individual values, not event families or
station-days. Three seeds do not replace grouped uncertainty. “One false pick per
station-day” needs valid observed hours, review coverage, outage treatment,
duplicates, matching and exposure duration. False picks per earthquake also vary
with catalogue completeness. A two-proportion test against random windows does
not measure field precision or account for correlated detections.

**Change:** use the repaired #35 extraction/matching rules, calibrate thresholds
on separate days (#38), then freeze them for comparison. Report reviewed false
alarms and unreviewed candidates separately, per 24 valid hours and per noise
regime; bootstrap paired event-family/station-day blocks. Pre-register consistent
timing tolerances, uncertainty intervals and candidate-merging rules. Calibrate
all baselines comparably; a fixed QuakeXNet threshold of 0.5 is not a matched
operating point. Zero reviewed false alarms in T days has an approximate 95%
Poisson upper rate 3/T, subject to the Poisson assumption; do not call a short,
correlated exposure a demonstrated deployment rate.

The revised plan now also requires fleet exposure, prevalence/precision
sensitivity, reviewer workload and per-process/distance/instrument interval
widths. A thousand false station picks are not a thousand independent events;
neither association nor per-regime power can be assumed from that count.

### S11 — High: association is treated as a guaranteed filter (§6.2, §6.4)

Most exported events have one pick. That does not prove one observable station,
but it prevents validating a three-station requirement without new waveform QA.
A 30 s tolerance over a 50 km aperture excludes some arrivals at 1 km/s. Extended
and moving sources further weaken a single travel-time assumption. A reranker
can only retain or reject stage-one detections; it cannot recover missed ones.

**Change:** station-onset performance first; separately report network coverage,
eligible multi-station recall, all-event recall and false network events. Inspect
move-out/envelope hypotheses by regime before fitting locations. Train the cascade
on out-of-fold stage-one candidates from training/mining data and test complete
candidate streams, including Z-only fallback and placement errors. Exclude or
declare ESEC exposure for every classifier weight. The local 16–78% placement
example was overall agreement on 32 Alaska examples, not a landslide-only score;
surface-class agreement in that table ranges 38–62%.

### S12 — High: corpus growth and experimental scope are overstated (§4, §7–8)

The export overlaps the curated dataset; adding their counts does not yield
20,000 unique training examples. The export has 7,880 pre-2002 picks, while
continuous PNSN archiving began in 2002; those records may require triggered
archives or be unavailable. A station-nearest-volcano tag is a receiving-station
proxy, not an event location. A flat 50→100% learning curve with three seeds does
not establish that the model is not data-limited.

**Change:** deduplicate and census retrievable waveform hours, unique event
families, actual components, process confirmation and native bandwidth first.
Define the sampler's fractions across positives, reviewed negatives, unreviewed
background and donors. Use nested event-family subsets with fixed negative
exposure and optimizer budget. Start with one minimal pilot, expand only on
diagnostics, and replace the 19-week promise with evidence-based checkpoint gates.

## Branch and issue organization

This audit branch contains the preserved proposal, revised plan and evidence.
It does not contain the unrelated #44A implementation from PR #51. Keep the
surface work as its own epic when implementation tickets are created; do not
relabel earthquake issues #33–#50 into surface-event work.

The revised plan defines SU-01 through SU-08 with branch names, deliverables and
dependencies. Shared contracts are #33 exclusions, #34 alignment/model contract,
#35 continuous scoring, #37 evaluability, #38 calibration and #44 suite access;
surface targets, corpus, pilot, cascade and association stay separate. The next
action is SU-01 (taxonomy and exposure registry), then the SU-02 label/corpus
census and SU-03 preprocessing/export checks, before any training.
