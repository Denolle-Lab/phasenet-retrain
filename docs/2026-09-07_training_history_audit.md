# Training history and strategy audit

*2026-09-07. Read-only audit of this repository by Marine Denolle with Claude,
after two external benchmarks (QuakeScope `phasenet_sequence_comparison` and
`phasenet_global_sequences`) found that the deployed fine-tune
`jma_wc_ft_global_v7` recovers fewer analyst picks than its parent `jma_wc`
in the western United States and on Kaikōura, Norcia and Thessaly. Every
claim below cites a file in this clone at commit 77f581d. Nothing on the lab
server was read; what only exists there is listed in §1.*

## 1. What exists, and what does not

**Checkpoints.** Only `models/jma_wc_ft_global_v7.pt` is in the clone. The
other twenty runs wrote to `checkpoints/finetune_jma_wc_global_v*/best.pt`,
which is git-ignored and lives on the lab server, if it was kept.

**Training curves.** None are committed. Each run wrote
`results/<run>_metrics.csv` per epoch (`scripts/finetune.py`, `logging` block
of every config) and `scripts/plot_training_curves.py` reads them, but
`results/` holds only `.gitkeep`. Whether any version overfit, when its
validation P-MAE turned, and what the loss did after the LR plateau can only be
answered on the server. The per-config header comments are the sole record of
training behaviour in git, and they are self-reported.

**Quantitative record.** `notebooks/step3_metrics.csv` (699 rows, 58 weight
sets, five splits, four distance bins) is the only committed measurement. The
training manifests are not committed; `data/manifest_checksums.csv` holds
their fingerprints.

**Timeline** (66 commits, 56 by the student). The benchmark was built
2026-05-12 to 06-04. v3 was evaluated 06-12, v7 on 06-15, v5–v13 landed
together on 06-17, v14–v19 on 06-25. Twenty fine-tunes in three weeks. The
review issues (#2–#15) were opened 06-26; fixes ran to 07-21; the noise-pool
detection audit came 08-10; the checkpoint was delivered 08-16.

## 2. The versions

Numbers are from `step3_metrics.csv`, split `all`, all distances, unless
marked. Recall is at threshold 0.3. "cond" is the detected-only MAE added in
#8; the unconditional P-MAE that drove selection is in the config headers.

| Version | Change (from the config header) | P recall | S recall | P-MAE cond | Header verdict |
|---|---|--:|--:|--:|---|
| `jma_wc` | parent, no training | 0.881 | 0.549 | 0.239 | baseline |
| `jma_wc_ft` | LR 1e-5, no KD, manifests v1 | 0.574 | 0.436 | | forgetting |
| `_frozen` | freeze encoder, LR 3e-5 | 0.438 | 0.431 | | worse |
| `_global` | +aq2009gm/obst2024/scedc | 0.863 | 0.533 | | |
| `_noise` | +noise pool, 200 epochs | 0.437 | 0.409 | | worse |
| v2 | manifests v2 (527k), no KD | 0.477 | 0.452 | | "catastrophic forgetting" |
| v3 | **KD α=0.3, T=4, LR 5e-6** | 0.872 | 0.511 | 0.253 | first stable |
| v4 | timing loss β=0.1, cosine LR 1e-4 | 0.296 | 0.544 | | best.pt at epoch 3 |
| v5 | v4, early stop on val_loss | 0.411 | 0.504 | | LR too disruptive |
| v6 | v3 + β=0.01 | 0.522 | 0.518 | | "even 0.01 is lethal" |
| **v7** | v6 with β=0 (= v3, longer) | 0.853 | 0.505 | 0.198 | **champion on P-MAE** |
| v8 | class weights [20,10,1], α=0.5 | 0.828 | 0.506 | | worse P-MAE |
| v9 | S-balanced manifests v3, monitor val_p_mae | 0.830 | 0.539 | | stopped epoch 13 |
| v10 | stage 2 from v9, LR 1e-6 | 0.770 | 0.492 | | P recall fell |
| v11 | T=1.5 | 0.810 | 0.517 | | calibration fix, no |
| v12 | α=0, init v7 | crashed | | | key mismatch |
| v13 | α=0, noise 40% at 0–10 dB, presence γ=0.5, 2× tele | 0.888 | 0.498 | 0.870 | recall up, timing gone |
| v14 | noise 20% at 5–10 dB | 0.851 | 0.522 | | worse |
| v15 | v14 without presence loss | 0.820 | 0.527 | | P-MAE 0.48 |
| v16 | v7 + 2× tele + noise 2–8 dB | 0.818 | 0.508 | | "noise hurts timing" |
| v17 | v7 + 2× tele, 300 epochs | 0.792 | 0.509 | | tele hurts regional |
| v18 | S-balanced + 1.5× tele + focal γ=1 | 0.896 | 0.575 | 0.350 | fires on 85% of noise |
| v19 | local+regional only, P-MAE focus | 0.810 | 0.502 | 0.197 | did not beat v7 |
| v20 | v7 + soft-label CE | 0.833 | 0.508 | 0.253 | worse on both |
| `scratch_v1` | from random init, 527k, strong augment | 0.485 | 0.472 | | far worse |
| v3/v7/v8/v9/v10/v11 `_clean`, v7 `_eventclean` | same recipes on leak-free manifests | 0.87/0.86 for v7 | 0.51 | 0.26/0.21 | leakage did not inflate v7 |

On the leak-corrected `clean_holdout` split the ranking is the same and the
gap is wider: `jma_wc` 0.909 P / 0.528 S against v7's 0.876 / 0.477; by
distance, v7 loses 1 point of P and 4 of S locally and 5 of P and 9 of S at
regional range. v7's conditional P-MAE is better, 0.211 s against 0.250 s.

## 3. Why v7 was chosen

For the lowest unconditional P-MAE on the benchmark, 0.340 s against the
parent's 0.374 s, read across nineteen versions on the same 32,000 traces.
Recall was below the parent from v3 onward and never recovered in any version
that kept its timing. The paper records this plainly (`paper_draft.qmd`
§Leaderboard: "no single fine-tuned model dominates the baseline") and flags
the selection as iterated selection on the test set (§Trajectory, callout).
Two later corrections weakened the choice further: the unconditional MAE was
found circular (#8), and the noise-pool detection MCC of 2026-08-10 puts v7
at 0.771 against the parent's 0.776 with overlapping intervals, so on the one
metric that sees false positives there is no advantage at all.

The honest summary of the campaign is that the parent was never beaten, and
the deployment decision took the one metric it was beaten on.

## 4. What failed, by the questions asked

### 4.1 The training data

**Volume.** The fine-tune corpus is 527,477 windows, 38% with an S label
(`data/README.md`). The parent was trained on 6.1 million three-component
waveforms from the JMA unified catalogue, 2014–2021, with consistent analyst
P and S labels (Naoi et al. 2024, *EPS* 76, doi:10.1186/s40623-024-02091-8).
The fine-tune therefore held under a tenth of the parent's data, drawn from
twenty sources with twenty labelling conventions. That is enough to perturb a
model and not enough to re-teach it, and `phasenet_scratch_v1` measures the
point: the same corpus from random initialisation gives P-MAE 1.7 s and
recall 0.49.

**P and S.** Three of the twenty sources are P-only by policy
(`build_training_dataset.py:209–319`, `use_s=False` for `geofon`, `lendb`,
`meier2019jgr`); their caps sum to 340,000 of the 1,453,400 cap budget, 23%.
The pool is then resampled to 25% teleseismic, where S is nulled by design.
S supervision was a minority of every window the student ever showed the
model, and the S-recall loss (0.549 to 0.505; 0.528 to 0.477 on clean
holdout) follows. v9 and v18 raised the S fraction to 60% and recovered S
recall to 0.54–0.58, each time at a timing cost the selection rule would not
accept.

**SNR.** No signal-to-noise selection or stratification exists at build time
(`build_training_dataset.py` filters on pick validity, caps and benchmark
exclusion only). The post-mortems report that 82% of v7's misses lie below
5 dB (v13 header). The response was to add white noise at training time
(v13–v16), which cost 0.2–0.6 s of P-MAE, rather than to curate or weight the
low-SNR population.

**Label quality.** Confident-learning flags 8.0% of INSTANCE, 10.3% of
AQ2009GM and 5.1% of CEED as mislabelled (`docs/LABEL_ERROR_FILTERING.md`).
The deployed v7 was trained before that filter was wired in (#10); only the
`_clean` variants used it, and v7_clean is not better. STEAD and INSTANCE
picks are largely automatic; JMA's are analyst picks.

**Leakage.** The deployed v7 trained on the pre-#32 `manifests_v2`, which
shared 75–95% of events with the benchmark for `mlaapde`, `aq2009gm` and
`cwa`. Retraining on leak-free manifests (v7_clean, v7_eventclean) moved v7 by
about one point, so leakage did not manufacture its result; it was simply
never excluded from the weight that shipped.

### 4.2 Augmentation

What the fine-tune path applies (`scripts/fast_manifest_dataset.py:102–116`):
amplitude jitter 0.5–2×, a 10% polarity flip, and, in v13–v16 only, additive
white Gaussian noise at a random SNR.

What it does not apply:

- **No time shift.** Every training window is cut with the P pick at 30% of
  the window, sample 900 of 3001, deterministically
  (`scripts/manifest_dataset.py:227–247`, `_window`), and the cached dataset
  stores those fixed windows before any augmentation. For 44 epochs the model
  saw P at the same sample in every example. The parent, like every SeisBench
  training run, saw arrivals at random positions. The benchmark places its
  anchor anywhere between samples 300 and 2700
  (`notebooks/05_benchmark_waveform_processing.ipynb`, `extract_window`,
  `P_JITTER_MIN`/`P_JITTER_MAX`), so it can test this, and the committed
  per-trace results (`notebooks/step3_results.parquet`) do: v7's P-recall
  deficit against the parent is the same in every position bin, 2.2 to 3.3
  points across eight bins from 300 to 2700, and its median peak probability
  is flat at 0.74–0.75 across the window. **The fixed training position did
  not leave a positional prior in v7.** It did in v3, whose recall falls from
  0.905 at samples 600–900 to 0.826 at 2400–2700, so the omission is a real
  risk of the recipe, but it is not the mechanism of v7's loss. v7's S
  deficit does grow with S position (+0.7 points for S before sample 600,
  −6.0 after 2400), and late S in a 30 s window means regional distance, which
  points at §4.1's S supervision rather than at position.
- No band-limiting, filtering, or sampling-rate variation: everything is
  resampled to 100 Hz once, and the model never sees the 40 Hz and 50 Hz
  instruments the campaign upsamples.
- No real-noise superposition. A quality-screened noise pool of 82,000
  traces was built (`scripts/build_noise_dataset.py`) and used only as
  negative examples; it was never added to signal windows at controlled SNR,
  which is what the low-SNR misses called for. The white noise that was used
  instead is what smeared timing.
- No channel dropout, gaps, or clipping in the fine-tune path. The
  from-scratch path (`scripts/scratch_dataset.py`) has channel masking, a
  wider amplitude range and a 50% flip; those never reached the v-series.
- The README's advertised time-shift, noise and channel-dropout augmentation
  belongs to a pipeline that was never used (`paper_draft.qmd` §Noise
  augmentation).

### 4.3 The loss

The v7 objective is `(1-α) CE_hard + α T² KL(teacher || student)` with α=0.3,
T=4 (`scripts/fine_tune_model.py:223–244`).

- **Hard-argmax cross-entropy.** The Gaussian label (σ=10 samples) is
  collapsed to its argmax class per sample, so a pick becomes a box about
  ±12 samples wide and the timing information the label carries is discarded.
  The parent was trained on the soft Gaussian. v20 swapped in soft CE alone
  and was worse on both recall and timing, so the swap is not sufficient by
  itself, but every version was trained with a loss shape the parent never
  saw.
- **The distillation anchor.** At T=4 the teacher's targets are heavily
  smoothed, and the student's peak probabilities dropped below the parent's
  (median P probability 0.747 against 0.781, v11 header). The student finds
  the same onsets with less confidence, which is exactly the calibration
  offset the external benchmarks measure at 0.3. Lowering T to 1.5 (v11) did
  not recover it. The header comments call α=0.3 "the indispensable
  cross-domain regularizer"; it is more accurately the term that stops the
  corpus from damaging the parent, which is not the same as improving it.
- **Class imbalance.** About 98% of samples are class N. The class-weight
  experiment (v8, [20,10,1]) produced a model with median P probability 0.94
  that the benchmark scored at 0.942 P recall and 0.612 S recall, the best
  numbers in the table, because the benchmark cannot see false positives
  (§4.5). Its conditional P-MAE was 0.54 s. The loss experiments were steered
  by a metric that rewards confidence.
- The explicit timing loss (soft-argmax L1) and the presence loss both
  collapsed recall or timing at any weight tried. With hard labels, an
  additional timing term fights the CE rather than refining it; this was
  observed and not analysed.

### 4.4 Optimisation and selection

LR 5e-6 flat for the whole run (ReduceLROnPlateau on `val_loss` never fired,
v8 header), early stopping on `val_loss` with patience 20, best checkpoint at
epoch 44. The headers state that validation metrics did not predict benchmark
behaviour, so every decision from v3 to v20 was made on the benchmark. There
is no test split that was not used for selection or threshold tuning
(`paper_draft.qmd` §Critical audit, items 1 and 7).

### 4.5 The evaluation

The benchmark is 35,392 single-arrival windows from 12 datasets, source
months 2013-07 to 2014-10, 31% Italy. Inference reads the model's probability
at the argmax inside a ±5 s window around the true pick
(`scripts/eval_finetuned.py:67,157–171`, `SEARCH_WIN_S = 5.0`); recall is that
probability exceeding the threshold. There are no negative windows, no sliding-window picking, no
association of a pick to an arrival, and no penalty for firing elsewhere. A
model that outputs 0.9 everywhere scores perfect recall. This is the metric
that ranked twenty versions.

The 2026-08-10 noise-pool audit is the first measurement that sees false
positives, and on it v7 and the parent tie. The two QuakeScope notebooks are
the first measurements on continuous data with the production picker, and on
them the parent wins.

## 5. Independence of the test sequences

`jma_wc` saw Japan only, so Kaikōura, Norcia and Thessaly are held out for it
by construction. For v7 they are not: INSTANCE (Jan 2005–Jan 2020) holds the
Amatrice–Visso–Norcia sequence; STEAD (2005–2018) and CREW (global regional)
may hold Kaikōura; CREW may hold Thessaly 2021. The western benchmark is worse
in this respect: SCEDC, CEED and ROSS2018GPD cover Ridgecrest 2019, PNW covers
Monroe 2019, and all four are in the corpus. Every external result so far is
in-domain or contaminated for v7 and clean for the parent, and the parent
wins.

To make these sequences permanent test assets:

1. Build an exclusion list by spatiotemporal join (the method of
   `scripts/audit_parent_leakage.py`) over `manifests_v2/{train,val}` and the
   full local copies of `instancecounts`, `stead`, `crew`, `lendb`,
   `mlaapde`, `meier2019jgr`, `scedc`, `ceed`, `ross2018gpd`, `pnw`, for the
   windows and sequence spans in `2026-09-07_generalization_audit_prompt.md`
   task 1. Commit the list as `(dataset, trace_name)` rows.
2. Apply it in `build_training_dataset.py` beside `benchmark_exclude`, so no
   manifest built from now on contains them.
3. Hold out whole years, not traces: no 2016 and no 2021 events in any
   training manifest, verified by the audit script, so a future sequence in
   those years is also clean.
4. Score every candidate with the two QuakeScope notebooks at matched pick
   budgets, and with the noise-pool detection MCC, before reading any
   benchmark number. Selection happens on a split that was never read during
   development.

## 6. What a v21 would have to change, in order

1. Evaluation before training: continuous-data picking with SeisBench's own
   annotate, matched-budget recall against analyst picks, a false-positive
   rate on noise, on held-out sequences and years.
2. Random window position in training. This is one line in `_window` and
   invalidates the cached-window dataset, which is why it was never done. It
   is hygiene rather than the fix: §4.2 shows v7 carries no positional prior,
   v3 does.
3. Real-noise superposition from the existing noise pool at 5–30 dB, random
   band-limiting and resampling to 40–100 Hz, channel dropout, gaps.
4. Soft Gaussian targets as the parent used; drop or shrink the distillation
   anchor once the data no longer damage the parent.
5. Fewer sources, all with analyst P and S, label-error cleaned, no P-only
   global sets, no teleseismic rebalancing for a local and regional campaign.
6. Selection on a held-out split with confidence intervals.

And the possibility the audit cannot exclude: a parent trained on six million
consistently labelled waveforms may not be improvable by a half-million-window
heterogeneous fine-tune at all. The evidence so far, twenty versions and two
external benchmarks, is consistent with that. The decision to keep the global
campaign on `jma_wc` stands regardless of what v21 finds.
