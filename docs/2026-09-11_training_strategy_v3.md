# Training strategy v3: a general PhaseNet under verified contracts

*2026-09-11, Marine Denolle with Claude, branch `docs/training-strategy-v3`
from `audit/2026-09-07-generalization` at `e400e16`. This is the scientific
strategy for the next model. It supersedes the recipe parts of
`docs/2026-09-07_training_plan.md` (v2); the target, the three regimes and
the noise taxonomy of v2 stand. Execution stays on the checkpoint order of
`docs/2026-09-10_issue_execution.md` and issues #33–#50; where this document
changes what a checkpoint should build, it says so. Nothing here has been run.*

## 0. Status on the integration branch

| Checkpoint | State | Where |
|---|---|---|
| 34A loader repair | merged | PR #62, `scripts/waveform_contract.py`, `scripts/manifest_dataset.py`, contract in `docs/2026-09-10_34a_loader_contract.md` |
| 34B historical row forensics | runner under review, server run pending | PR #63 |
| 44A suite roles | under review | PR #51 |
| 33A, 34C, 35A–C, 36A–B, 37A–B, 38A, 44B | not started | |
| First training experiment (46A) | blocked on 33A, 34B, 34C, 35C, 44B | |

The server is reachable only from the UW network. Everything below that
needs the SeisBench cache, the historical manifests or a GPU is marked
*server*.

## 1. What failed, and the rule each failure imposes

Twenty fine-tunes of `jma_wc` did not beat it. The 2026-09-07 audit read
that as a data-volume limit. The 2026-09-10 audit and the review of PR #63
found that the corpus the model saw was not the corpus the manifest
described. Each defect below is a code fact with a citation, and each fixes
a rule for v3.

| Defect | Evidence | Rule for v3 |
|---|---|---|
| Failed waveform reads became zero windows labelled noise and were cached for all 44 epochs; bucket-indexed SeisBench names failed on the direct HDF5 readers | `3bf98c4:scripts/manifest_dataset.py` lines 57–64, 84–90, 370–375; `docs/2026-09-11_silent_zero_windows_report.md` | Failure budget zero. A run starts only if the rejection ledger is empty and the per-dataset count of rows read equals the manifest count; both are written into the run card (§9). |
| Waveforms resampled to 100 Hz while pick indices stayed in native coordinates; direct readers asserted 100 Hz | `3bf98c4:scripts/manifest_dataset.py`: `_resample_if_needed` called at line 379, `_window` at line 394 with the native index, readers return `TARGET_SR` unconditionally at lines 337 and 341 | One coordinate contract (34A): stored rate from metadata, arrival index rate explicit, single resampling, fixtures at every production rate. |
| An absent S or an unlabelled second event was trained as a confident noise target | `scripts/manifest_dataset.py:258` still forms N = 1 − max(P, S) | Partial-label loss with a validity mask (§5). N is supervised only where the window is certified event-free or reviewed. |
| Hard-argmax cross-entropy discarded the Gaussian target; timing, presence and focal terms bolted on to recover it collapsed recall (v4, v6, v13, v18) | `docs/2026-09-07_training_history_audit.md` lines 51–53, 60, 65; `scripts/fine_tune_model.py` lines 181–209 | Soft-label cross-entropy on the Gaussian target, no auxiliary terms. |
| Distillation at T = 4 lowered every probability; a shared threshold then under-counted the student | H1 in `docs/audit_2026-09-07/README.md` | Distillation is an experimental arm, not a default. Thresholds are set per weight on calibration station-days (#38), never shared. |
| Selection by unconditional P-MAE on a positives-only benchmark drawn from the training datasets (v7 `in_domain` = `all`) | H00 and H3, same file | Selection on the development suite at a frozen nuisance-pick budget with negatives; the sealed panel is scored once (#44, #49). |
| White Gaussian noise as the only noise model; crops frozen in RAM before augmentation | `scripts/fast_manifest_dataset.py` lines 116–124; `CachedManifestDataset` | Real-noise corpus mixed on the fly at controlled SNR; crops drawn per epoch from longer stored windows. |
| Teleseismic rebalancing inside the regional model hurt regional recall (v13, v16, v17) | history audit lines 60, 63–64 | Two models: regional picker here, distant-P model in #50. |
| One seed per version, one variable per version, no scaling curve, validation loss that did not predict the benchmark | history audit §4.4; `scripts/finetune.py` lines 272–276 | Pre-registered factorial pilots with three seeds; early stopping and model selection on a development metric with negatives, not on validation loss. |
| A 4 M-window RAM cache would need 576 GB; all three splits are loaded eagerly | `docs/2026-09-10_picker_and_issue_roadmap_audit.md` Gate D; `scripts/manifest_data_module.py` | Sharded, memory-mapped store with worker-local readers; labels built per batch. |

Two v7 comparisons are therefore uninformative about their nominal
variable: v20 (soft-label CE, "worse on both") and v11 (T = 1.5) carried the
zero-window and misalignment defects, so nothing in their results argues
against soft targets or lower temperature. They are rerun as arms of E1.

## 2. The target, and what "better" means before any run

The campaign runs one PhaseNet over every land network EarthScope holds,
for local and regional sensitivity, mainshock-aftershock sequences,
volcano-tectonic sequences and fluid-driven swarms. The parent's measured
weaknesses are the targets (from the 2026-09-07 audit, 95 % bootstrap
intervals):

| Weakness of `jma_wc` | Where measured | Size |
|---|---|--:|
| Regional S | benchmark, 150–1500 km | S recall 0.36 [0.35, 0.38] |
| Low-SNR P | benchmark, below 0 dB | P recall 0.66 [0.64, 0.67] |
| Precision per pick | Kaikōura, Norcia, Thessaly at matched budget | `instance` leads P by 3–12 points along the overlap |
| False triggers on noise | noise pool, own best threshold | precision 0.83 |
| Second arrivals in a window | never measured; the loader trained them as noise | to be measured by 35A on the aftershock cases |

**Decision rule, to be frozen at 44B and 49A before any candidate is
scored.** One primary candidate. Thresholds per weight from the #38
protocol on calibration station-days, frozen. Association settings frozen.
Paired block bootstrap over events and station-days, P and S of one event
kept in one block. Proposed margins, absolute:

| Endpoint | Requirement |
|---|---|
| Primary: P recall at the frozen nuisance budget, sealed mainshock-aftershock panel | candidate − parent ≥ +0.03, 95 % lower bound > 0 |
| S recall, every regime | ≥ parent − 0.02 (lower bound) |
| Event recovery after association, per regime and per hour or day | ≥ parent − 0.02 |
| Conditional P and S residual (MAE, picks both make) | ≤ parent + 0.010 s |
| Unmatched-pick rate on held-out noise stations at the frozen thresholds | ≤ parent |
| Missing-channel and 20/40/50 Hz inputs | non-inferior on the same margins |
| Runtime per station-day | ≤ 1.2 × parent |

A candidate that fails any row keeps the parent in deployment and the panel
it was scored on becomes development. The margins are proposals; the
freeze is the act that makes them a test.

## 3. The model

- **Architecture.** PhaseNetWC, the parent's width, ZNE in, PSN out, 100 Hz
  target grid. No architecture change in this round; the levers are data,
  targets, loss and context.
- **Context.** The production contract is 3001 samples at 100 Hz. SeisBench
  PhaseNet forwards at 3001 and 6000 samples and fails at 6001, and the
  deployed `in_samples` is 3001 (audit finding on #48). Sixty-second
  context is an experiment (48A contract first, then 48B), motivated by
  regional S and by overlapping events; it is not assumed.
- **Initialisation.** From `jma_wc`, the parent that still leads the
  benchmark and the non-US sequences. The SeisBench `instance` weights are
  the second parent: they lead `jma_wc` on precision per pick along the
  whole matched-budget overlap on Kaikōura, Norcia and Thessaly (§2), at
  standard PhaseNet width. Both enter E4 as initialisations, with the
  caveat that fine-tune exclusions cannot remove Italian sequences already
  in the `instance` weights, so every candidate carries its parent's
  exposure as provenance (35B). A from-scratch arm at matched width runs
  at T1 scale and above (§8, E4); the one scratch run in the history was a
  narrower model on the defective corpus and proves nothing.
- **Batch normalisation.** The v7 input-layer running variance is 0.758 of
  the parent's (audit probes). Adaptive versus frozen BN statistics, with
  affine parameters controlled separately, is an arm of E1.
- **Distant P.** A separate 20 Hz, 120 s, P-only model (#50) after the
  shared contracts; it never shares weights with the regional picker.
- **Ensembles.** Probability averaging of the candidate with the parent is
  a deployment option scored like any candidate (35B ensemble rules).

## 4. Data

### 4.1 Signal corpus, in tiers

Windows are cut from continuous data at the station's native rate, stored
uncropped at 120 s with every catalogued arrival inside them, and resampled
on the fly by the 34A polyphase contract. The manifest carries the 34A
fields (stored rate, sample interval, component and dimension order,
support, start time, arrival index rate) plus, per window: the arrival list
as JSON with phase, time, provenance tier (manual, automatic, unknown) and
event id; SNR on defined pre- and post-arrival windows; epicentral
distance; magnitude; operator; year; event type where the operator gives
one; the number of catalogued events in the window; and the event-group
key used for splitting.

| Tier | Size | Sources | Purpose | Checkpoint |
|---|--:|---|---|---|
| T0 pilot | 50,000–100,000 windows | SeisBench sets with manual P and S (ETHZ, PNW, CWA, SCEDC, CEED, TXED, Iquique, INSTANCE where the pick status is manual, VCSEIS) and one bulletin-harvest slice each from INGV, NOA and GeoNet through the pipeline of `scripts/build_heldout_testset.py`; every native rate represented; S-only rows present | Alignment diagnostic, recipe factorial, augmentation value | 40A |
| T1 | about 500,000 | T0 sources at full allowed volume plus the operators that pass the #39 census | Scaling curve start, init × KD factorial | 40B |
| T2 | 2–4 million | bulletin harvest over a decade, only if the T1 → T2 point of the scaling curve justifies the storage and compute | Final candidate | 45B |

Composition targets from v2 §4 are pilot hypotheses, measured on T0 and
enforced on T1 by stratified sampling: at least 35 % of windows below 5 dB
on the P window, at least 30 % with more than one event, an S label in at
least 60 %, at least 15 % from volcano-tectonic and swarm settings outside
the held-out places, regional-heavy distance mix, no operator above 30 %.
No P-only sets, nothing beyond 2000 km, no ocean-bottom data.

### 4.2 Label validity policy (#41)

Every arrival has a provenance tier. The loss (§5) uses three masks per
window, built from the arrival list and the window's catalogue status:

- *Positive support*: samples within ±3σ of a manual or reviewed arrival
  carry a Gaussian target on that phase channel.
- *Negative support*: the N channel is supervised only on windows certified
  event-free (no catalogued event with a predicted arrival in the window or
  the 120 s before it, in a region whose completeness magnitude the census
  records) or on reviewed negatives from the noise pools.
- *Unknown*: everything else. Samples in unknown support contribute nothing
  to the loss; they still pass through the network and the distillation
  term if that arm is on.

Automatic or unknown-tier arrivals mask their neighbourhood as unknown
rather than supervise it. Overlapping targets from two arrivals of the same
phase take the per-sample maximum, and the PSN triple is renormalised to
sum to one on supervised samples. Irreparable rows are quarantined with a
reason, not deleted silently and not kept unconditionally.

### 4.3 Noise corpus (#42)

The eleven-flavour taxonomy of v2 §5.1 stands, harvested from the
campaign's own archives at native rate, 120 s windows, no model-based
screening, class labels from the source or from spectral features. The
ontology is split as the audit asked: *background noise* (supervised N),
*task-excluded sources* (surface events, explosions, long-period volcanic
events: supervised N for this picker, kept for the surface-event picker),
*unlabelled intervals* (aftershock coda, tremor, LFEs: unknown, never
supervised N by catalogue absence), *reviewed negatives*. Pilot pool
100,000 windows with at least 5,000 per class, the held-out places and
years excluded, split by station so the unmatched-pick rate is measured on
stations the model never saw.

### 4.4 Hold-outs, identity and roles (#33, #44)

The 23 windows and places of `scripts/heldout_sequences.py`, the 2016 and
2021 years, the benchmark traces and events, applied to signal windows,
noise windows, validation and any mining. Identity is
`(dataset, chunk, trace_name)` plus a canonical event key; equivalent events
across sources stay in one split. Roles: *regression* (Kaikōura, Norcia,
Thessaly, Ridgecrest, Monroe and every case already inspected),
*development* (Samos, Adriatic 2022, Etna, Corinth–Thiva, tier-2 Hawaii and
Alaska), *calibration* (station-days, #38), *sealed* (the unexamined tier-1
cases certified evaluable by 37B). The exclusion bundle and every manifest
are hashed; the hashes go in the run card.

### 4.5 Storage and throughput

Sharded HDF5 with memory-mapped access, worker-local file handles, labels
and masks built per batch from the arrival list, no dense label array on
disk. Throughput is measured on T0 (windows per second per GPU with the
full augmentation stack) before T1 is planned; the v7 per-epoch time in
`results/finetune_jma_wc_global_v7_metrics.csv` (server) is the reference
point, not an assumption of linear scaling.

## 5. Targets and loss

- Gaussian targets, σ = 0.10 s (ten samples at 100 Hz, as the parent), on
  P and S; N = 1 − max(P, S) only on supervised samples.
- Loss = masked soft-label cross-entropy, −Σ over supervised samples of
  y · log p, averaged over supervised samples per batch, P and S and N
  terms logged separately.
- Optional distillation term α · T² · KL(teacher ‖ student) over all
  samples, α ∈ {0, 0.3}, T ∈ {1.5, 4}, as arms of E1 and E4. Optional
  parent-domain replay (JMA-like windows with exclusions) as a
  forgetting control, an arm of E4.
- No timing, presence, focal or class-weight terms.
- Logged per step: each loss term, gradient norm, effective positive-label
  count per phase, supervised-sample fraction, sampling composition by
  source and by augmentation branch.

The code today has `soft_ce`, distillation and the auxiliary terms
(`scripts/fine_tune_model.py` lines 169–209) but no mask and no per-term
logging; both are 41A/43A work and are prerequisites of E1.

## 6. Augmentation (#43, #47)

Every transform acts on waveform, arrival list and masks together and has
a fixture that checks label times to within one target sample.

- Random crop of the training window (3001 or 6000 samples) from the 120 s
  stored window, arrivals kept if inside, masks recomputed.
- Real-noise superposition, class-balanced draw with probability 0.6,
  resampled to the signal's rate, scaled to a target SNR on the P window
  drawn 40 % in 0–5 dB, 30 % in 5–10 dB, 30 % in 10–25 dB.
- Event superposition with probability 0.3: a second window at an offset
  of 2–40 s, arrivals merged, masks merged (unknown wins over negative).
- Non-stationary noise (level ramp or step, second class over part of the
  window) with probability 0.3.
- Instrument artefacts, each 0.02–0.05: spike, DC step, gap of 0.1–3 s with
  the gap masked unknown, clipping, mains hum, drift, one channel zeroed
  with the component mask updated.
- Rate transform with probability 0.3: anti-aliased decimation to 20, 40
  or 50 Hz and polyphase return to 100 Hz through the same kernel the
  loader uses.
- Pure-noise negatives, 15 % of each batch, class-balanced, N supervised.
- Untouched fraction 30 %; amplitude jitter and polarity flip on all.

## 7. Optimisation

- AdamW, weight decay 1e-4, gradient clipping 1.0, mixed precision, batch
  256 at T0 and 1024 at T1 and above.
- Learning rate: warm-up of 2 epochs then cosine to 1e-6. The rate is an
  E1 arm, {2e-6, 5e-6, 2e-5}; 5e-6 was the only stable value in the
  history, but it was chosen against the defects, so it is not carried
  forward untested.
- Early stopping and checkpoint selection on the development metric
  (P and S recall at the frozen nuisance budget on the development
  sequences, computed by the 35A scorer every epoch on a fixed excerpt),
  patience 10. Validation loss is logged, not used for selection.
- Three seeds for any comparison that guides the campaign. Results are
  reported at fixed epochs and at fixed optimiser updates, both.
- Frozen BN statistics is an arm; when frozen, affine parameters still
  train unless the arm says otherwise.

## 8. Experiments, in order, mapped to checkpoints

**E0, alignment diagnostic (46A).** Same v7 manifest rows, same v7 recipe
(hard-argmax CE, α = 0.3, T = 4, LR 5e-6), three arms: the pinned legacy
loader through PR #63's `load_legacy`, the 34A loader with the legacy
target formula, the 34A loader with the mask. Three seeds. Scored on the
regression sets. This separates the loader defects from everything else
and is the only experiment that uses the v7 corpus. *Server.*

**E1, recipe factorial on T0 (46B).** Base arm: 34A loader, mask on, soft
CE, α = 0, adaptive BN, LR 5e-6, no augmentation beyond crop, jitter and
flip. One-factor deviations from the base: mask off; α = 0.3 at T = 1.5;
α = 0.3 at T = 4; frozen BN; LR 2e-6; LR 2e-5. Seven arms, three seeds,
21 short runs. Selection of the recipe on the development suite with paired
intervals.

**E2, augmentation value on T0 then T1 (47A).** From the E1 recipe: no
noise; white noise at the v13 setting; real noise class-balanced; real
noise plus event superposition; the full §6 stack. Scored on unmatched-pick
rate by noise class, recall versus SNR, and the aftershock development
case. A flavour the model still fires on is under-represented in the pool
and the census is adjusted, not the threshold.

**E3, scaling (45A, 45B).** The E2 recipe at nested event-grouped subsets
of T1 and, if built, T2: 0.1, 0.25, 0.5, 1, 2 million windows at fixed
composition, three seeds at the two smallest sizes, one at the rest.
Matched-budget recall per regime against corpus size. The T2 build is
authorised only if the 0.5 → 1 M step still gains.

**E4, initialisation and anchor at the chosen size (46C).** Initialisation
∈ {`jma_wc`, `instance`, matched-width scratch}, crossed with α ∈ {0, 0.3}
(the teacher being the arm's own parent) and with replay on or off, three
seeds. The `instance` arm runs at standard width unless E5 has already
shown the width difference to matter.

**E5, context and width (48A, 48B).** Export contract for 6000 samples
first; then 3001 against 6000 at the chosen size, and PhaseNetWC against
standard width; ensembles with the parent.

**E6, acceptance (49A, 49B).** One candidate, frozen thresholds and
associator, the sealed panel once, the §2 rule, deployment hash and
probability parity verified before rollout.

## 9. The run card

Every run writes `results/<run>/run_card.json` before the first optimiser
step and again at the end, with: config hash; manifest hashes for train,
validation and the development excerpt; exclusion bundle hash; loader
contract version; per-dataset rows read and rows rejected (rejected must
be 0); achieved composition by source, distance bin, SNR bin, S-label
fraction and multi-event fraction; augmentation parameters; seeds; Python,
PyTorch, SeisBench and SciPy versions; git commit; and, at the end, the
epoch chosen, the development metric at that epoch, and the checkpoint
SHA-256. A run without a complete card is not a result. This is the
document the zero-window defect would have made impossible to miss.

## 10. What can start now, and what waits for the server

On the laptop, in checkpoint order (PRs opened 2026-09-11 in this order;
each is stacked on the checkpoint it depends on):

1. Review and merge PR #51 (44A) and PR #63 (34B runner).
2. 33A: versioned exclusion bundle with hashes and the quarantine policy.
3. 35A: the scorer rewrite (per-threshold extraction, maximum-cardinality
   matching, per-window aggregation, failure table), synthetic fixtures.
4. 37A: evaluability of the built regression and development cases.
5. 41A: the label-validity policy and the mask semantics of §4.2 as a
   schema, with fixtures, wired through the loader and the loss (PR #66).
6. 43A: the augmentation transforms of §6 as label-consistent functions on
   the 41A sample contract, with fixtures.
7. The run card writer with the rejection-ledger gate in
   `scripts/finetune.py`.

On the server, first session:

```bash
# 34B: count what v7 actually trained on
python scripts/audit_v7_rows.py --manifest-dir data/manifests_v2 --cache-root $SEISBENCH_CACHE_ROOT --inventory-only --output results/34b/inventory
python scripts/audit_v7_rows.py --manifest-dir data/manifests_v2 --cache-root $SEISBENCH_CACHE_ROOT --output results/34b/full
grep -h "fetch" results/finetune_jma_wc_global_v1[89]*_metrics.csv results/finetune_*_clean*_metrics.csv   # post-2026-07-12 failure counters
# Task 1 counts for the 23 windows and places
python scripts/audit_heldout_sequences.py
# Rates and formats per dataset, for the T0 census
python scripts/audit_v7_rows.py --inventory-only ...   # same inventory, read per source
```

Then 34C (benchmark timebase, deployment parity), 37A and 38A on the
built cases, 39A census, and 40A builds T0. E0 runs when 33A, 34B, 34C,
35C and 44B are released.

## 11. What this strategy does not do

No EQTransformer or multi-station model. No ocean-bottom data or noise
this round. No further runs on the v7 corpus except E0. No selection on
`notebooks/step3_metrics.csv` or on any positives-only table. No shared
threshold across weights. No claim of a better picker before E6. No
training run before the run card, the ledger gate and the frozen decision
rule exist.
