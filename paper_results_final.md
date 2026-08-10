# PhaseNet Retraining Project — Final Results Summary (for paper drafting)

This document consolidates the project context, methodology, and the **final,
leakage-corrected benchmark results** for a project retraining PhaseNet (a CNN
seismic phase picker) into a globally-deployable model. It is meant to be
pasted into a fresh conversation to help draft a research paper — it is
self-contained (no repo access needed) but abbreviated relative to the full
internal working draft (`paper_draft.qmd`).

---

## 1. Project context and motivation

**Goal.** Fine-tune PhaseNet (starting from SeisBench's `jma_wc` pretrained
weights) into a globally-deployable phase picker for onshore seismic networks
(Phase 1), eventually extended to ocean-bottom seismometers (Phase 2, not yet
started).

**Why.**
1. Münchmeyer et al. (2022) showed deep-learning phase pickers (PhaseNet,
   EQTransformer, GPD, etc.) generalize unevenly across regions/regimes — no
   single released model is uniformly best.
2. SeisBench has since added many more datasets (subduction zones, induced
   seismicity, volcanic settings, OBS deployments), enabling a genuinely
   global, hybrid training set for the first time.
3. Our group already deployed PhaseNet at cloud scale (4.3 billion P/S picks,
   >47,000 stations, 23-year span — Ni et al. 2025a *GJI*, 2025b *Seismica*).
   A downstream team is re-analyzing the resulting catalogs; pick quality and
   **reliability** (low false-positive rate, calibrated probabilities) bound
   the value of that catalog, especially for temporary networks in
   under-instrumented regions the deployed model was never specialized for.

**`jma_wc`** = SeisBench's "PhaseNetWC" (WC = wide/doubled conv channels),
trained on Japan Meteorological Agency's unified catalog (Naoi et al. 2024,
*Earth Planets Space*, 10.1186/s40623-024-02091-8). Empirically the
best-performing pretrained PhaseNet weight in initial evaluation, hence chosen
as the fine-tuning base and teacher.

**Design requirements:** metrics reported comparably to Münchmeyer et al.
(2022); a hybrid training set with spurious labels removed (Aguilar et al.
label-error method); rebalancing toward uniform coverage in space, depth,
tectonic regime, and P/S feature diversity.

## 2. Data and model

**Training pool:** ~20 SeisBench datasets (STEAD, INSTANCE, ETHZ, PNW, TXED,
GEOFON, SCEDC, LEN-DB, Iquique, MLAAPDE, Ross2018-GPD, Meier2019, CEED, CWA,
OBST2024, PISDL, CREW, VCSEIS, AQ2009GM, OBS), distance-stratified
(target local 40% / regional 25% / teleseismic 25% / unknown 10%), per-dataset
capped, benchmark traces excluded (event-level, not just exact-trace-name, as
of the final fix — see §5), Aguilar et al. label-error filtering applied.

**Benchmark/evaluation pool:** 35,392 candidate traces (31,992 usable) drawn
from 11 of the above datasets. Local 15,869 / regional 14,469 / teleseismic
1,346 traces (teleseismic ≈4.2%). S picks present in only ≈53% of traces and
in **zero** teleseismic traces. Mean magnitude ≈2.95; M>6 events ≈449 traces.

**⚠️ Critical benchmark limitation:** picks are searched only within **±5s of
the known true arrival** (oracle window). This means **no false positives are
possible and precision is never measured** on this population alone — recall
is the only detection metric this benchmark itself can produce. This flatters
every model relative to real continuous-data deployment and was the single
biggest thing a reviewer would flag.

**Partially resolved 2026-08-10 (§4f):** a separate pure-noise pool (no true
arrival in any trace) now supplies the missing negative population, giving a
real detection precision/recall/MCC for the headline models — see §4f. This
still isn't a deployment-accurate false-positive *rate* (the noise:arrival
ratio in the pooled test doesn't match real continuous-data base rates), but
it replaces "unmeasurable in principle" with an actual computed number.

**Model:** full fine-tuning of `jma_wc` (all layers trainable) with a
self-distillation anchor — a frozen copy of `jma_wc` as teacher, KL-distillation
loss to prevent catastrophic forgetting. Champion config (v7):

```
L = (1-α)·CE_hard(z_s, ŷ) + α·T²·KL(σ(z_t/T) ‖ σ(z_s/T)),  α=0.3, T=4
```

where `ŷ` is the **argmax-hardened** version of the original Gaussian soft
label. Key design finding: every cited baseline model (including the `jma_wc`
teacher) was trained with **soft-label cross-entropy that preserves sub-sample
timing**; the fine-tune path instead uses **hard-argmax CE**, discarding that
timing signal, and relies on the KD term to claw it back (this works — v7 beats
`jma_wc` on P-MAE — but is an internal contradiction: it fine-tunes a soft-CE
model with a hard-CE loss). Explicit timing-regression and pick-presence loss
terms were tried and found **lethal to recall** even at tiny weights, and are
disabled in the champion. A **soft-CE variant (v20) was tested as the fix for
this contradiction and made both P-MAE and recall worse** — the hypothesis did
not pan out (see trajectory table, §6).

## 3. The leakage story (why "clean" matters here)

An internal audit found the initial "cross-domain" evaluation split was a
**no-op** — fine-tuned models' `trained_on` field was hardcoded `None`, so
"cross-domain" numbers were byte-identical to "all" (i.e., partly evaluated on
data the models could have seen in training). Fixing this surfaced **three
compounding leakage issues**, each found by refusing to trust a suspiciously
clean first-pass fix:

1. **Event-level leakage** (same earthquake, different station/trace-name):
   despite 0% exact-trace overlap, several datasets showed 90%+ *event*
   overlap between training and benchmark. Fixed by grouping train/val splits
   by event identity, not trace name.
2. **A chunk-collision bug**: for sharded datasets (mlaapde, aq2009gm, cwa),
   `trace_name` is reused as a positional index per file shard, so a naive
   `{trace_name: event_id}` dict silently collapsed unrelated earthquakes
   across shards onto one arbitrary (usually wrong) ID — this had *also*
   undermined the event-grouping fix in (1). Fixed by keying on
   `(trace_name, chunk)`.
3. **Cross-dataset (parent-model) leakage**: the same physical earthquake can
   appear under different names in different datasets (e.g. a STEAD-trained
   model's "cross-domain" PNW performance was partly the same events it saw in
   STEAD). Detected via spatiotemporal matching (origin time ±2s, location
   ±10km) since raw catalog IDs aren't comparable across providers. Found
   substantial overlap: STEAD↔PNW 38.5% of events, INSTANCE↔ETHZ 32.7%,
   PISDL↔ETHZ 30.9%.

**Two honest, leakage-corrected evaluation splits now exist and are used for
every number below:**
- **`clean_holdout`** — for our own fine-tuned models: excludes benchmark
  traces whose *event* also appears in that specific model's training
  manifest.
- **`cross_domain_clean`** — for pretrained/parent models (EQTransformer,
  public PhaseNet weights): excludes benchmark traces whose event
  spatiotemporally matches an event in that model's known training corpus.

**Residual, irreducible leakage:** CWA retains ~9.9% train/benchmark event
overlap even after all fixes — a genuine ambiguity in CWA's own catalog (the
identical waveform is a candidate match to two nearby real earthquakes; no
further data disambiguates it), not a bug. volpick's Japan-region training
slice remains unverified locally (only its non-Japan corpus, `vcseis`, is
available to audit). `obst2024` and `neic` are marked unverifiable rather than
falsely clean (no usable event identifier / no complete local copy).

**The benchmark ground truth itself (`benchmark_manifest.csv`) was never
modified by any of this — only training-side exclusion logic and evaluation
masks changed.**

## 4. Final results — headline leaderboards

All numbers below are from the leakage-corrected splits. 95% CIs are
percentile bootstrap (1000 resamples). "Detected-only" MAE is the
Münchmeyer-style conditional metric (misses excluded); the plain "P-MAE" is
unconditional (misses count as saturated ±5s errors) — report both, the
unconditional number alone can hide real costs.

### 4a. Own fine-tuned models — `clean_holdout` split (event-leakage-corrected)

| Model | n | P-recall | S-recall | P-MAE (s) [95% CI] | P-MAE, detected-only | MCC* |
|---|--:|--:|--:|--:|--:|--:|
| **v7 (champion fine-tune)** | 20,827 | 0.876 | 0.477 | **0.290** [0.280, 0.300] | 0.211 | 0.753 |
| Ensemble v7+v11 | 21,853 | 0.856 | 0.489 | 0.290 [0.279, 0.300] | 0.200 | 0.718 |
| v7_eventclean (retrained on event-grouped-only data) | 20,096 | 0.881 | 0.466 | 0.300 [0.289, 0.310] | — | 0.760 |
| Ensemble v3+v7 | 21,853 | 0.881 | 0.482 | 0.302 [0.291, 0.313] | — | 0.745 |
| v11 (calibration variant, T=1.5) | 20,827 | 0.828 | 0.492 | 0.310 [0.300, 0.320] | — | 0.687 |
| v3 (first stable KD config) | 20,827 | 0.886 | 0.483 | 0.313 [0.303, 0.324] | — | 0.744 |
| **jma_wc (parent / un-fine-tuned baseline)** | 21,864 | **0.909** | **0.528** | 0.319 [0.308, 0.330] | 0.250 | **0.781** |
| v20 (soft-label CE test) | 20,827 | 0.843 | 0.480 | 0.330 [0.318, 0.341] | — | 0.714 |
| v18 (S-balanced + teleseismic oversample + focal) | 24,080 | **0.888** | 0.552 | 0.459 [0.445, 0.470] | — | 0.714 |

*MCC here is a within-window **P-vs-S argmax discriminability** score, **not**
Münchmeyer's detection MCC — this benchmark has no negative/noise windows so a
true detection MCC cannot currently be computed. Label it explicitly as
"P-vs-S MCC" in any paper table, not as "detection MCC."

**Headline finding:** v7 beats the `jma_wc` parent on pick timing (P-MAE 0.290
vs 0.319s, ~9% better) but **loses on recall and MCC** — no fine-tuned variant
dominates the baseline on every axis, across 19 numbered experiments.

### 4b. Pretrained / parent models — `cross_domain_clean` split (spatiotemporal-leakage-corrected)

| Model | n | P-recall | S-recall | P-MAE (s) [95% CI] | P-MAE, detected-only | MCC |
|---|--:|--:|--:|--:|--:|--:|
| eqt_ensemble (volpick + eqt_original_nonconservative)† | 21,034 | 0.911 | 0.735 | **0.263** [0.252, 0.274] | 0.098 | 0.864 |
| eqt_volpick | 31,752 | 0.897 | **0.837** | 0.316 [0.307, 0.325] | 0.175 | 0.734 |
| eqt_original_nonconservative (STEAD-trained EQTransformer) | 21,034 | 0.918 | 0.659 | 0.338 [0.324, 0.351] | 0.138 | **0.930** |
| eqt_scedc | 31,119 | 0.866 | 0.417 | 0.356 [0.347, 0.366] | — | 0.851 |
| eqt_instance | 22,835 | 0.833 | 0.384 | 0.364 [0.352, 0.376] | — | 0.815 |
| volpick (PhaseNet variant) | 32,022 | 0.829 | 0.577 | 0.374 [0.363, 0.383] | — | 0.515 |

†eqt_ensemble's "volpick half" only inherits the STEAD leakage mask (its
composite-model bookkeeping maps it to STEAD only) — so this row's population
(n=21,034) is **not the same population** as `eqt_volpick` alone (n=31,752).
Flag this before citing a direct ensemble-vs-solo comparison — it is currently
an apples-to-a-different-basket-of-apples comparison, not fully resolved.

**Headline finding:** `eqt_original_nonconservative` wins P-vs-S MCC/recall by
a wide margin over our best fine-tune (0.930 vs 0.753). v7 beats every
*fully-verified* parent model on P-MAE except the population-caveated ensemble
row above.

### 4c. Does retraining fully leak-free change the ranking? (matched-population comparison)

For each fine-tune variant, its "\_clean" counterpart (retrained after all
three leakage bugs in §3 were fixed) is scored **on the original variant's own
leakage mask/population** — not its own (much larger, ~leak-free) mask — so the
comparison is apples-to-apples.

| Weight | Original (leaky-trained) P-MAE | Retrained leak-free P-MAE | Outcome |
|---|--:|--:|---|
| v7 | 0.2899 | 0.3214 | leak-free retrain loses |
| v11 | 0.3097 | 0.3123 | loses (closest margin) |
| v3 | 0.3129 | 0.3163 | loses |
| v9 | 0.3769 | 0.3763 | tie |
| v10 | 0.3431 | 0.3407 | tiny win for leak-free |
| v8 | 0.3274 | 0.5643 | loses badly (see note) |

**v8_clean is an outlier**: best recall/MCC of the six leak-free variants
(0.942/0.822) but by far the worst P-MAE (0.588 vs ~0.36–0.39 for its
siblings). Working hypothesis (not yet confirmed): its P=20 class-weight
pushes many low-confidence borderline samples over the detection threshold,
inflating recall while dragging in imprecise picks.

**Conclusion for the paper:** the original leaky v7 (0.2899) is the single
best P-MAE of any model, on any split, in this entire project. No fully
leak-free retrain beat it. The honest framing is: *v7's reported number is
somewhat leakage-inflated (retraining leak-free costs ~11% on P-MAE), but no
alternative — leaky or clean — has since beaten it.* If a fully leakage-free
claim is required, **v11_clean (0.3123 matched P-MAE)** is the best
alternative to cite.

### 4d. Distance-bin breakdown

| Model | Local <150km (recall / P-MAE) | Regional 150–1500km (recall / P-MAE) | Teleseismic >1500km (recall / P-MAE) |
|---|---|---|---|
| v7 | 0.932 / 0.195 | 0.824 / 0.380 | *not computed for clean_holdout — recompute-script gap, not zero underlying data* |
| jma_wc | 0.942 / 0.216 | 0.878 / 0.415 | *same gap* |
| eqt_ensemble (caveated, §4b) | 0.959 / 0.136 | 0.937 / 0.213 | 0.210 / 1.981 |
| eqt_original_nonconservative | 0.961 / 0.170 | 0.949 / 0.286 | 0.247 / 2.440 |
| eqt_volpick | 0.952 / 0.186 | 0.893 / 0.312 | 0.264 / 1.936 |

No model exceeds ~26% teleseismic recall. This is the field's shared weak
point, not specific to our fine-tunes — the teleseismic-trained baseline
(`geofon`, not leakage-audited, recall 0.775 uncorrected) is the only model
with genuine teleseismic capability, at the cost of being far worse
local/regional.

### 4e. Threshold-independent re-ranking — the definitive fairness check

The tables above (§4a–§4d) report recall/MCC at a single fixed detection
threshold (0.30), applied identically to every model. A fair objection: what
if a model's *ranking* only looks bad (or good) because of that specific
cutoff? This was checked directly, not just addressed by argument.

**What actually needed fixing.** Inspecting the metric code
(`scripts/metrics.py`) shows P-MAE, S-MAE, RMSE, outlier fraction, and MCC
were **already threshold-independent** by construction — each is computed
over *every* in-window trace (or, for MCC, purely from the relative ranking
of P-probability vs. S-probability, no absolute cutoff involved). **Recall is
the only threshold-dependent metric** in the whole suite.

**Fix for recall — an exact, non-arbitrary summary.** For probabilities
bounded in [0,1], the area under the recall-vs-threshold curve has a closed
form:

$$
\text{AUC\_recall} = \int_0^1 \text{recall}(t)\,dt = \int_0^1 P(\text{prob} \ge t)\,dt = E[\text{prob at the true-arrival pick location}]
$$

i.e. **the mean probability the model assigns at the correct pick location**.
This is not a discretized approximation or a choice of curve-summary
statistic — it is the literal, exact area under the full continuous
recall-vs-threshold curve, immune to any threshold choice. Computed directly
from the cached per-trace inference results (`notebooks/step3_results.parquet`,
1.95M rows) — **no retraining or re-inference required**, since raw
probabilities were already stored. EQTransformer was excluded from this
comparison (out of deployment scope per project decision), which as a side
effect makes this a same-architecture (3-class softmax) comparison throughout
— removing the earlier softmax-vs-independent-sigmoid scale concern entirely.
Full output: `results/threshold_independent_ranking.csv` (44 PhaseNet-family
weights, degenerate rows excluded) and
`results/threshold_independent_recall_curves.csv` (full curves, 101
threshold points per weight).

**Ranked by P-MAE (exact, threshold-independent):**

| Rank | Weight | Population | n | P-MAE (s) | P-AUC-recall | S-MAE (s) | S-AUC-recall | MCC (P-vs-S) |
|--:|---|---|--:|--:|--:|--:|--:|--:|
| 1 | **v7** | clean_holdout | 20,827 | **0.290** | 0.669 | 1.459 | 0.339 | 0.753 |
| 2 | Ensemble v7+v11 | clean_holdout | 21,853 | 0.290 | 0.661 | 1.428 | 0.335 | 0.718 |
| 3 | v7_eventclean | clean_holdout | 20,096 | 0.300 | 0.658 | 1.503 | 0.324 | 0.760 |
| 4 | Ensemble v3+v7 | clean_holdout | 21,853 | 0.302 | 0.655 | 1.426 | 0.338 | 0.745 |
| 5 | v11 | clean_holdout | 20,827 | 0.310 | 0.675 | 1.459 | 0.339 | 0.687 |
| 6 | v3 | clean_holdout | 20,827 | 0.313 | 0.653 | 1.457 | 0.336 | 0.744 |
| 7 | v1 (jma_wc_ft_global) | clean_holdout | 21,853 | 0.318 | 0.707 | 1.452 | 0.366 | 0.672 |
| 8 | **jma_wc (parent)** | clean_holdout | 21,864 | 0.319 | **0.705** | 1.448 | 0.359 | **0.781** |
| 9 | v8 | clean_holdout | 20,827 | 0.327 | 0.757 | 1.474 | 0.349 | 0.742 |
| 10 | v20 (soft-CE) | clean_holdout | 20,827 | 0.329 | 0.588 | 1.468 | 0.318 | 0.714 |
| 12 | v11_clean (leak-free retrain) | clean_holdout | 31,970 | 0.367 | 0.674 | 1.403 | 0.351 | 0.688 |
| 16 | instance (pretrained baseline) | cross_domain_clean | 23,098 | 0.461 | 0.695 | 1.684 | **0.258** | 0.848 |
| — | v13 | clean_holdout | 20,892 | 0.945 | **0.864** | 1.383 | 0.355 | 0.942 |

**Ranked by P-AUC-recall (exact, threshold-independent) — top of table:**

| Rank | Weight | P-MAE (s) | P-AUC-recall | Note |
|--:|---|--:|--:|---|
| 1 | v13 | 0.945 | 0.864 | best recall, worst-tier timing |
| 2 | v8_clean (leak-free retrain) | 0.588 | 0.858 | same pattern |
| 3 | v14 | 1.024 | 0.818 | same pattern |
| 4 | v8 | 0.327 | 0.757 | best recall *without* wrecking timing |
| 5 | v1 (jma_wc_ft_global) | 0.318 | 0.707 | |
| 6 | **jma_wc (parent)** | 0.319 | **0.705** | |
| 15 | **v7** | **0.290** | 0.669 | best timing, mid-pack recall |

**What this confirms under a metric immune to threshold choice, not just
argument:**

1. **v7 is still the single best-timed model** (P-MAE 0.290s) — this
   conclusion was not an artifact of the 0.30 cutoff.
2. **The recall gap vs. the `jma_wc` parent is real, not a threshold-choice
   artifact.** jma_wc's AUC-recall (0.705) beats v7's (0.669) using a metric
   involving no cutoff at all — this closes the door on "v7 might win at a
   different threshold." It doesn't, on average across the whole curve.
3. **The recall↔timing seesaw (§5) is reconfirmed exactly**: v13/v8_clean/v14
   top the AUC-recall ranking and are simultaneously three of the worst
   models on P-MAE — the same pattern as the single-threshold table, now on a
   metric that cannot be gamed by threshold choice.
4. **v20 (soft-label CE) is confirmed as a straight regression, not a
   seesaw trade**: worse P-MAE (0.329) *and* worse AUC-recall (0.588, near
   the bottom of the table) than v7 — it does not even win on the axis it was
   designed to help.
5. `instance`'s S-detection weakness (§1's "why jma_wc, not instance"
   rationale) holds up under this stricter test too: S-AUC-recall 0.258 vs.
   jma_wc's 0.359, regardless of any threshold.

**One gap this does *not* close** (recorded for completeness, not pursued
further here): AUC-recall is a detection-*confidence* metric, not true
precision/recall — it still cannot penalize false positives, because this
benchmark has zero negative/noise windows (every trace has a real arrival).
A fully complete detection evaluation (real precision, a real F1-optimal
threshold, a genuine Münchmeyer-style detection MCC) would require building a
new negative-window benchmark subset and rerunning every model against it —
a data-collection task, not a recomputation, and out of scope for this pass.

### 4f. Detection precision — closing the false-positive gap (resolves GitHub #8's remaining item)

Every metric in §4a–§4e is measured on traces that **always contain a true
arrival** — the benchmark has no negative/noise windows, so none of them can
distinguish a well-calibrated model from one that fires constantly. Built the
missing negative population from datasets' own noise classes
(STEAD/LenDB/TXED/VCSEIS/OBST2024 — 94,405 pure-noise traces, no true arrival;
`scripts/audit_noise_fp_leaderboard.py`), ran the 15 headline models over it,
then combined it with the existing leak-corrected positive population
(`clean_holdout` / `cross_domain_clean`) into a real confusion matrix
(`scripts/compute_detection_metrics.py`) — genuine precision, recall, F1, and
a Münchmeyer-style **detection MCC**, not the P-vs-S discriminability proxy
used everywhere above.

**Caveat, applies to every number below:** the positive:negative ratio here
(~21–32k : ~94k) reflects benchmark/noise-pool *construction*, not the true
earthquake:noise ratio of real continuous data (far more noise-dominated) —
this is a real, computed detection score on a pooled test set, not a
deployment-accurate false-alarm rate.

| Weight | Precision | Recall | Detection MCC [95% CI] |
|---|--:|--:|--:|
| **Ensemble v7+v11** | **0.845** | 0.856 | **0.816** [0.811, 0.820] |
| v11 | 0.838 | 0.828 | 0.797 [0.792, 0.801] |
| eqt_scedc | 0.794 | 0.866 | 0.770 [0.766, 0.774] |
| **v7 (champion fine-tune)** | 0.725 | 0.876 | **0.748** [0.743, 0.752] |
| v7_eventclean | 0.685 | 0.881 | 0.723 [0.718, 0.728] |
| eqt_volpick | 0.708 | 0.897 | 0.719 [0.715, 0.724] |
| eqt_ensemble (volpick+nc) | 0.636 | 0.911 | 0.699 [0.694, 0.704] |
| Ensemble v3+v7 | 0.657 | 0.881 | 0.696 [0.692, 0.701] |
| **jma_wc (parent/teacher)** | 0.607 | 0.909 | **0.671** [0.666, 0.676] |
| volpick | 0.649 | 0.829 | 0.629 [0.625, 0.634] |
| **eqt_original_nonconservative** | 0.466 | 0.918 | **0.550** [0.546, 0.555] |
| eqt_instance | 0.501 | 0.833 | 0.535 [0.530, 0.540] |
| v20 | 0.450 | 0.843 | 0.501 [0.496, 0.506] |
| v3 | 0.392 | 0.886 | 0.456 [0.451, 0.460] |
| v18 | 0.211 | 0.888 | 0.047 [0.042, 0.052] |

**What this changes, not just confirms:**

1. **v18's recall was fool's gold.** 2nd-best recall of any single fine-tune
   (0.888) — but it fires on **85% of pure-noise windows**. Detection MCC
   0.047, barely above chance. High recall alone was never sufficient
   evidence of a good detector, exactly as this benchmark design couldn't
   previously rule out.
2. **`eqt_original_nonconservative`'s "decisive MCC win" over v7 (§4b: 0.930
   vs 0.753) does not survive this check.** That 0.930 is P-vs-S
   discriminability, not detection. Its real detection MCC (0.550) is
   *substantially worse* than v7's (0.748) — part of its recall advantage was
   bought with a false-positive rate ~3× v7's. **This supersedes §4b's
   "eqt wins recall/MCC by a wide margin" framing.**
3. **v7 beats its own `jma_wc` teacher on real detection MCC** (0.748 vs
   0.671), despite losing on raw recall (0.876 vs 0.909). Fine-tuning traded
   some recall for a measured drop in false triggers — the opposite of
   "made the model worse at its primary job" (abstract-critique Q3), now that
   the false-positive axis is no longer invisible.
4. **Ensemble v7+v11 is the strongest single entry by this metric** (MCC
   0.816, precision 0.845) — a better-balanced detector than v7 alone, worth
   promoting to co-headline status rather than a secondary row.

Full data: `results/noise_fp_audit.csv` (1.4M per-trace rows),
`results/noise_fp_leaderboard.csv`, `results/detection_metrics.csv`.

## 5. Experimental trajectory (v1 → v20) — the recall↔timing seesaw

19 versioned fine-tuning experiments plus one from-scratch run, each a
single-variable change with a written post-mortem. The recurring pattern:
**interventions that raise recall (noise augmentation, pick-presence loss,
removing distillation, S-balancing, teleseismic oversampling) degrade P-MAE,
and vice-versa** — no version escapes this trade-off.

| Ver | Key change | Outcome |
|--|--|--|
| v3 | KD α=0.3, T=4, low LR | First stable win, P-MAE 0.313 |
| v6 | tiny explicit timing loss β=0.01 | Recall collapsed 0.872→0.522 |
| **v7** | v6 with β=0 (timing loss removed) | **Champion**: P-MAE 0.290, recall 0.876 |
| v8 | + class weights [P=20,S=10,N=1] | Best recall/MCC among leak-free variants but worst P-MAE (0.564–0.588) |
| v11 | KD temperature T=4→1.5 | P-MAE 0.310; ensemble w/ v7 = 0.290 |
| v13 | α=0 (no KD) + noise aug + presence loss + 2× teleseismic | Best recall (0.888)/MCC of any single non-v18 fine-tune, but P-MAE collapsed to 0.945 |
| v18 | S-balanced + 1.5× tele oversample + focal γ=1 | Highest recall of any single fine-tune (0.888), P-MAE 0.459 (58% worse than v7) |
| v19 | local+regional only | P-MAE 0.328, recall 0.825 — did not beat v7 despite being P-MAE-focused |
| v20 | v7's exact recipe + soft-label CE (test of the loss-design hypothesis in §2) | P-MAE 0.330 (+14% vs v7) **and** recall 0.843 (worse) — a straight regression on both axes, refuting the hypothesis |
| from-scratch (no `jma_wc` transfer) | Random init, full retrain | P-MAE 1.696, MCC 0.160 — far worse on every metric; direct evidence pretrain+fine-tune is the right strategy for this data volume |

**Key conclusions:** (1) KD distillation (α≈0.3) is an indispensable
cross-domain regularizer — removing it collapses timing; (2) explicit
timing/presence loss terms backfire; (3) in-distribution validation metrics do
not predict cross-domain benchmark P-MAE, which complicates model selection.

**⚠️ Selection-bias caveat for the paper:** conclusion (3) is double-edged —
because validation metrics were unreliable, "which version wins" was decided
by repeatedly reading the **benchmark** leaderboard across 19+ versions. That
is iterated model selection on the test set. A truly held-out split, never
used for version selection or threshold tuning, has not yet been evaluated;
the reported v7-vs-jma_wc margin should be described as provisional pending
that check.

## 6. What "clean" still does not mean — required limitations section

- **No precision / false-positive measurement on the arrival benchmark
  alone.** ✅ *Partially resolved 2026-08-10 (§4f).* The oracle ±5s search
  window makes false positives impossible on that population by
  construction; a separate pure-noise pool now supplies real precision/F1/
  detection-MCC (§4f). **Still open:** the noise:arrival population ratio
  used doesn't match real continuous-data base rates, so this is not yet a
  deployment-accurate false-positive *rate*.
- **MCC is not Münchmeyer's detection MCC.** ✅ *Resolved 2026-08-10 (§4f).*
  The MCC reported in §4a–§4e measures P-vs-S discriminability within a
  window that already contains a true arrival, not noise-vs-signal
  classification — that limitation stands for those sections specifically,
  but a genuine detection MCC (needing a noise/negative-window benchmark
  subset) has now been built and is reported in §4f.
- **Threshold selection on the evaluation set.** ✅ *Partially resolved (§4e).*
  Whether model *rankings* depend on the arbitrary 0.30 cutoff was checked
  directly via an exact threshold-independent metric (AUC-recall) — rankings
  hold. **Still open:** the specific 0.30-vs-0.10 sweep numbers cited for v7
  earlier in this document were tuned on the same traces used for reporting
  results, and a genuine F1-optimal threshold still can't be computed at all
  (needs precision, which needs negative windows — see the bullet above).
- **Residual, quantified leakage:** CWA ~9.9% (irreducible catalog ambiguity);
  volpick's Japan-region training data unverified (only non-Japan corpus
  auditable locally); `obst2024`/`neic` unverifiable (no usable ID / no
  complete local copy) rather than falsely assumed clean.
- **Reproducibility:** only the PyTorch global seed is set; NumPy/Python
  `random` (used for augmentation and noise placement) are unseeded;
  `cudnn.benchmark=True`. Runs are not bit-reproducible.
- **Rebalancing is partially aspirational.** Distance rebalancing and
  per-dataset caps are implemented; depth/tectonic-regime/magnitude
  rebalancing and spectral-content/travel-time/polarity balancing of the
  *training* set are not — they exist only as post-hoc visualizations of the
  benchmark, not as training-time targets.
- **Degenerate rows exist in raw exports** (e.g. some baseline weights show
  recall=MCC=1.0, a broken-eval artifact, not a real result) — exclude any
  such row from headline comparisons rather than citing it.

## 7. Suggested honest headline claims for a paper

1. Fine-tuning `jma_wc` with self-distillation (v7) improves pick **timing**
   over the un-fine-tuned parent (P-MAE 0.290s vs 0.319s, ~9%, though see §4c
   — this specific number is leaky-trained; the leak-free-matched comparison
   is closer to a wash) but **does not close the recall gap** (0.876 vs 0.909
   at threshold 0.30; 0.669 vs 0.705 on the exact threshold-independent
   AUC-recall, §4e). **New (§4f): v7 beats the parent on real detection MCC**
   (0.748 vs 0.671) — the lost recall comes with a measured drop in false
   triggers, so this is a genuine trade with a quantified upside, not simply
   "worse at the primary job."
2. The best hypothesis for closing the recall gap (matching the teacher's
   original soft-label training objective instead of the fine-tune path's
   hard-argmax CE) was tested directly (v20) and **failed on both P-MAE and
   AUC-recall** — worth reporting as a negative result. This shows *this
   specific fix attempt* (a full soft-label CE swap) doesn't work; it does
   not establish that the hard/soft-label mismatch isn't the mechanism at
   all — untried alternatives (label smoothing, a hard+soft blend, a smaller
   explicit timing loss with different scheduling) remain open.
3. Event-level and cross-dataset leakage were real, non-trivial (up to ~95%
   event overlap for some datasets before fixing), and materially inflated
   some reported numbers — but after full correction, **P-MAE-based ranking
   of the top fine-tunes was preserved**; only absolute magnitudes shifted.
   Recall/MCC rankings were not separately re-checked across that same
   before/after boundary (AUC-recall and the §4f detection metrics didn't
   exist pre-fix, so there is no "before" to compare against) — don't
   over-read this as "leakage didn't affect any conclusion here."
4. No released or fine-tuned PhaseNet/EQTransformer variant in this study
   dominates on every metric — different models win on timing vs. recall vs.
   detection MCC, and the "best model" answer depends on which axis a
   deployment prioritizes. **`eqt_original_nonconservative`'s apparent
   dominance on recall/MCC is the weakest of these wins** — §4f shows its
   real detection MCC (0.550) trails v7 (0.748) and the v7+v11 ensemble
   (0.816) once false positives are counted.
5. The evaluation methodology's biggest gap — no precision/FP measurement,
   MCC ≠ detection MCC — is ✅ **substantially resolved (§4f)**: a real
   detection precision/MCC now exists for the headline models. What remains
   open is deployment-accurate false-positive *rate* (the pooled test's
   noise:arrival ratio isn't the real-world one) and the still-unaddressed
   threshold-tuned-on-eval-set issue (§6).
6. Stating any single percentage (e.g. "9% better timing") next to an
   admission of iterated model selection across 19+ variants is only
   defensible if the selection-bias caveat (§5) travels with it every time —
   narrow, non-overlapping bootstrap CIs rule out sampling noise as the
   explanation, but say nothing about selection bias, which is unquantified
   and could plausibly exceed 9%.

---
*Source data: `notebooks/step3_metrics.csv`, `results/leakfree_retrain_report.txt`,
`results/threshold_independent_ranking.csv`, `results/noise_fp_audit.csv`,
`results/noise_fp_leaderboard.csv`, `results/detection_metrics.csv`, project
git history through commit `40582f9` plus this session's uncommitted work
(2026-08-10 snapshot).*
