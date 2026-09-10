# PR #52: review of Copilot and Fable's comments

Reviewed 2026-09-10 against PR head `c39cbcf`. Fable's substantive review appears
under the repository owner's account in the
[PR reviews](https://github.com/Denolle-Lab/phasenet-retrain/pull/52#pullrequestreview-5169626443).
The response below evaluates its claims, rather than counting reviewer agreement
as validation. The follow-up uses an isolated worktree; the shared working tree
and the other session's branch are left untouched.

## Copilot: accept all four comments, with one qualification

| Review comment | Decision and change |
|---|---|
| [Generated absolute paths](https://github.com/Denolle-Lab/phasenet-retrain/pull/52#discussion_r3981230866) | Accepted. Regenerated results use named relative roots for repository, packages and sibling sources. Original content hashes remain evidence. |
| [Generator paths and date parsing](https://github.com/Denolle-Lab/phasenet-retrain/pull/52#discussion_r3981230916) | Accepted as robustness work. Dates are converted to strings explicitly, parsed with UTC/error handling, and invalid/missing/valid rows are reported separately. The audited export still has 21,721 valid timestamps and no invalid/missing dates; no numerical finding was caused by this concern. |
| [Source-hash keys](https://github.com/Denolle-Lab/phasenet-retrain/pull/52#discussion_r3981230956) | Accepted. Keys are `repo/...`, `package/seisbench/...`, `package/obspy/...`, or `sources/<repository>/...`. A root mismatch fails instead of emitting a workstation path. |
| [Run instructions](https://github.com/Denolle-Lab/phasenet-retrain/pull/52#discussion_r3981230988) | Accepted. README uses the active environment's `python`; an optional external-source root supports worktrees. No contributor-specific interpreter/cache path is required. |

Portability is the substantive defect. These paths are not credentials; no
history rewrite is needed for this correction. The current evidence is portable,
but the earlier commits retain their original contents. Copilot reviewed artifact
hygiene; its review is not scientific validation of the plan.

## Fable: accept the direction, sharpen the specifications

### 1. Shared resampling defect — accept and reproduce

The installed SeisBench upsampling path calls ObsPy `Trace.resample` with its
default spectral Hann window. `no_filter=True` does not disable that window.
Our added 20→100 Hz and 40→100 Hz tone checks reproduce −6.02 dB at half the
source Nyquist and −32.23 dB at 0.9 of it, matching
`20*log10(0.5*(1+cos(2*pi*f/fs)))`. The existing 40→50 Hz finding was therefore
too narrowly framed as a surface concern.
[ObsPy API](https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.resample.html)

Keep ownership in [#34's existing comment](https://github.com/Denolle-Lab/phasenet-retrain/issues/34#issuecomment-5621902058),
with #43 and SU-03 consuming the same implementation/fixtures. Do not create a
second surface resampler. Qualify “every non-100 Hz station”: already resampled
records and integer downsampling have different provenance/routes. The defect is
demonstrated in the pinned implementation; its contribution to historical model
failure still requires waveform-route census and controlled runs. A polyphase
implementation is a candidate remedy, not a certification by name alone.

### 2. Choose the UPN target — accept, with complete normalization

UN stays first. For UPN we now choose `n=max(0,1-u-p)`, then normalize **all**
channels by `u+p+n` and apply soft-label cross-entropy. Fable says to renormalize,
but the N formula alone is insufficient when U/P overlap. Coincident unit
kernels must become `[0.5,0.5,0]` in the categorical arm. Independent U/P sigmoid
heads with BCE are a different, optional arm; a later duration mask does not
itself require changing every onset output to sigmoid. Synthetic unit-mass
checks were added; no training objective has yet been implemented.

### 3. Campaign base rates — accept; do not equate picks and events

Station false-pick budgets need translation into valid fleet exposure, workload
and plausible precision. The plan now includes explicit formulas and an example:
1,000 station-days at one false pick/day produce 1,000 false **station candidates**.
With an illustrative true-arrival rate of 0.01 per station-day and recall 0.8,
candidate precision is about 0.79%. These are not measured campaign numbers.

Fable's expectation that association will be the decisive filter is a hypothesis.
Emergent signals can retain arrival/move-out information, though uncertain and
process-dependent. Compare interval-aware timing, envelope coherence and combined
rules; measure false network events under correlated noise. Do not reuse the
earthquake P/S PyOcto contract unchanged or claim a location from coherence alone.
Station picking and network products keep distinct acceptance criteria.

### 4. Mechanism/distance/instrument transfer and statistical power — accept

“Out of region” was too compressed. The revised plan requires process × distance
× instrument coverage and independent event-family counts before scoring.
The 173 waveform-bearing ESEC events do not provide 173 eligible trials per
regime. Added exact-binomial examples show how weak a small panel can be:

| Detected / independent events | Two-sided 95% exact recall interval |
|---|---|
| 10/10 | 0.692–1.000 |
| 20/20 | 0.832–1.000 |
| 50/50 | 0.929–1.000 |
| 16/20 | 0.563–0.943 |

These are planning illustrations, not a claim about actual panel power. Use
grouped uncertainty when event families/station-days are correlated. Set desired
width and attainable floors from the eligible census; mark small strata
exploratory or acquire independent examples instead of asserting a blanket lack
of power. The probe calculates the table using
[SciPy's exact proportion interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

### 5. Unreviewed background — accept the qualification

The audit should prohibit treating catalogue absence as proven non-target truth,
not all learning from unreviewed background. A later contamination-aware arm is
now allowed with blinded prevalence estimation, an upper bound, bounded sampling/
loss influence and sensitivity checks. Keep reviewed negatives for the first
small pilot. Evaluation false alarms still require an auditable reference/review
protocol, even if training tolerates contaminated negatives.

### 6. Sampling/context rationale — accept the qualification

The original 50 Hz/120 s choice had a concrete 150-trace PNW spectral rationale;
the plan now states it again. The 4 s effective-field claim was cited honestly
from the paper. Neither establishes the trained effective field or global
process coverage of the installed model. Keep the candidate and measure rate
and context separately using development perturbations and bandwidth controls.

### 7. Dates, owners and shared tickets — accept as a tracking proposal

Added two 10-working-day timeboxes for SU-01 and SU-02, a named post-2002
Rainier/St. Helens pilot and Newberry/Hood development station-days, conditional
on event/station overlap checks. The timebox delivers a registry/census and a
bounded pilot recommendation; it does not promise harvesting every year in two
weeks. Marine is the **proposed** scientific decision owner; implementation and
independent annotation owners are explicitly unassigned. T0 is not yet scheduled.
SU-03 also needs a named shared-contract owner and fixture-review estimate.

Proposed child-ticket scopes under #34/#35 are recorded in the plan. Existing
34A/34C etc. already have acceptance criteria in tracked parent issues; child
tickets would improve ownership, not create previously nonexistent scope. Migrate
the parent checkpoint references when creating them so two definitions of done
cannot drift. No new GitHub issues or assignees were created in this review.
The worktree recommendation was applied immediately for these changes.

## Validation and next decision

Four portability/date tests pass. The regenerated synthetic probes retain the
original crop, invalid-target and model-export findings, add a valid target
normalization check and confirm the shared Hann response. Invalid/missing date
counts are explicit. No real sequence was scored and no training was run.

The revised plan is ready for an implementation-scope decision, not training
approval by reviewer consensus. Next: assign SU-01/02 owners and T0, track the
shared contracts, then execute the taxonomy/role registry and retrievable census.
