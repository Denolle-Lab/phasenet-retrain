# Phase 2 — Retraining PhaseNet for Offshore / OBS data

**Audience:** Michael (and anyone learning the retraining workflow).
**Goal:** specialize our PhaseNet picker for **Ocean-Bottom Seismometer (OBS)**
data, where noise and signal characteristics differ markedly from land stations.

This guide is deliberately pedagogical: it explains *why* each step exists, not
just the commands. Work through it top to bottom. When in doubt, read the
corresponding code (every step points to the file) and the project paper
(`../paper_draft.html`).

---

## 0. The mental model (read this first)

We are **not** training from scratch. We do **transfer learning + knowledge
distillation**:

```
Phase-1 onshore base model  ──fine-tune on OBS data──▶  OBS-specialized picker
        (the "teacher")            (low LR, KD anchor)
```

- The **base / teacher** is the best Phase-1 onshore model (or, until Phase 1 is
  finalized, `jma_wc` or `jma_wc_ft_global_v7` as a stand-in).
- We fine-tune it on OBS waveforms at a **very low learning rate** (5e-6) so it
  *adapts* rather than *forgets*.
- A frozen copy of the base provides a **distillation (KD) loss** — it keeps the
  model anchored to its proven onshore timing behavior while it learns OBS noise.
  This is the single most important lesson from Phase 1: **removing KD collapses
  timing** (see paper §Trajectory, versions v13–v15).

Why OBS needs its own model at all: OBS records are dominated by noise sources
that simply don't exist on land (next section). A land-trained picker fires on
that noise (false positives) and misses emergent OBS onsets (low recall). Many
groups maintain their own site-tuned OBS pickers; our aim is one robust,
distillation-anchored model that generalizes across OBS deployments.

---

## 1. What makes OBS data different (the seismology you must respect)

| OBS noise / signal feature | Why it matters for picking |
|---|---|
| **Infragravity waves & ocean microseism** | Strong long-period energy; can dominate vertical channel below ~0.1 Hz. |
| **Seafloor compliance & tilt noise** | Horizontal channels especially noisy; tilt couples seafloor currents into the seismometer. |
| **Water-column multiples / reverberation** | Repeated arrivals that a picker can mistake for P/S. |
| **Unknown horizontal orientation** | OBS horizontals (H1/H2) are often *not* aligned to N/E until post-deployment orientation analysis. Channel handling must not assume N/E. |
| **Instrument diversity** | Broadband OBS vs. short-period; differential pressure gauge (DPG)/hydrophone as a 4th channel. PhaseNet uses 3 channels — decide which. |
| **Biological & anthropogenic** | Whale calls, ship noise, mooring strum — non-seismic transients that must be learned as "noise." |
| **Emergent onsets** | Lower SNR and softer onsets than land → recall is the hard problem (same as teleseismic in Phase 1). |

**Takeaway:** the *noise corpus* (Step 3) is where most of the Phase-2 value is.
Spend your effort there.

---

## 2. Get oriented (do this before touching OBS data)

1. Read the project paper end to end (`../paper_draft.html`), especially
   §Loss design, §Trajectory, and §Critical audit.
2. Run the **onshore** pipeline once on a small config so you understand the
   moving parts (manifest → finetune → eval). Use an existing config and a tiny
   `max_epochs` to smoke-test.
3. Read these three files until you can explain them:
   - [`../scripts/build_training_dataset.py`](../scripts/build_training_dataset.py) — how manifests are built.
   - [`../scripts/fine_tune_model.py`](../scripts/fine_tune_model.py) — the model and the loss (`compute_loss_and_metrics`).
   - [`../scripts/finetune.py`](../scripts/finetune.py) — the training loop and `--init-from`.

---

## 3. Build the OBS training manifest

The OBS-relevant SeisBench datasets already wired into
`build_training_dataset.py` are **`obst2024`** and **`obs`** (see
`../data/README.md` for volumes). Build an **OBS-only** manifest analogous to the
onshore ones.

```bash
# On the server, in the repo root, with the env activated.
# Easiest path: add an --obs-only flag (or a new DATASET_CONFIGS subset) that
# keeps only obst2024 + obs, then:
python scripts/build_training_dataset.py --obs-only       # flag to be added
```

What to check after building (this is real data hygiene, not box-ticking):

- **Distance distribution** — OBS arrays are often regional/teleseismic; expect a
  different mix than the onshore `{0.40/0.25/0.25/0.10}` targets. Re-tune the
  target fractions for OBS.
- **S-pick availability** — many OBS catalogs are P-only at distance. Decide your
  S policy explicitly (mirror the teleseismic P-only rule).
- **Channel sanity** — confirm how OBS horizontals are stored and whether they are
  oriented. Do **not** silently assume Z/N/E.
- **Event-level split** — split train/val/test by **event**, not by trace, so the
  same earthquake recorded at many OBS doesn't leak across splits. (This is an
  open issue even in Phase 1 — fix it here from the start; see paper §Audit.)

---

## 4. Build the OBS noise corpus (the most important step)

A land-trained model fails on OBS mainly by **firing on OBS noise**. Teach it
what OBS noise looks like.

```bash
# Mirror the onshore noise builder, sourcing OBS noise windows.
python scripts/build_noise_dataset.py        # adapt sources to OBS (obst2024, obs)
python scripts/audit_noise_picks.py          # run the base picker over noise...
python scripts/add_noise_to_manifests.py     # ...keep only windows the base
                                             #    model does NOT fire on
```

The audit step (Phase-1 lesson) is essential: run the base picker over every
candidate noise window and **discard windows where it already triggers** — those
are ambiguous and confuse training (see paper §Noise augmentation; in Phase 1,
26% of "pre-phase" noise had spurious P-prob and was dropped). For OBS, make sure
the corpus includes: microseism/infragravity segments, tilt/compliance-noisy
horizontals, water-column reverberation, and biological/ship transients.

---

## 5. Configure the fine-tune

Copy a champion onshore config and change only what OBS needs. Start from the v7
recipe (the proven, KD-anchored, no-extra-loss recipe).

```yaml
# configs/finetune_obs_v1.yaml  (sketch — copy v7 and edit)
model:
  pretrained: { use_pretrained: true, model_name: "jma_wc" }  # teacher source
training:
  learning_rate: 0.000005          # low — adapt, don't forget
  distillation: { alpha: 0.3, temperature: 4.0 }   # KEEP KD — do not set alpha 0
  timing_beta: 0.0                  # leave OFF (collapses recall in Phase 1)
  presence_gamma: 0.0               # leave OFF
  focal_gamma: 0.0                  # try 1.0 later if OBS recall is poor
  scheduler: { name: ReduceLROnPlateau, monitor: val_loss, patience: 7 }
  early_stopping: { monitor: val_loss, patience: 25 }
data:
  train_manifest: data/manifests_obs/train.csv
  val_manifest:   data/manifests_obs/val.csv
  test_manifest:  data/manifests_obs/test.csv
  window_length: 3001
```

Then train, **initializing from the Phase-1 base checkpoint**:

```bash
python scripts/finetune.py \
  --config configs/finetune_obs_v1.yaml \
  --init-from checkpoints/<phase1_base>/best.pt
```

`--init-from` loads the base weights with a fresh optimizer
(`finetune.py:255–268`). Until Phase 1 is final, use `jma_wc`'s own weights as
both teacher and init to prototype the OBS pipeline.

---

## 6. Evaluate — and compare to the models people already use

```bash
python scripts/eval_finetuned.py     # → notebooks/step3_metrics.csv + figures
```

Report the **Münchmeyer-style** metrics (recall, MAE, outlier %, MCC) broken
down by distance bin, and judge success against two references:

1. **The onshore base** (does OBS fine-tuning improve OBS picks without wrecking
   onshore performance? Check both.)
2. **Site-tuned OBS models** that other groups use (this is the bar the community
   will hold us to).

Apply the same caveats the paper raises: report **detected-only (conditional)**
MAE, add a **precision / false-positive** evaluation on OBS noise windows (this is
*especially* important offshore), and use an **event-level** held-out test set.

---

## 7. What "done" looks like

- An OBS-specialized checkpoint that **beats the onshore base on OBS recall**
  without a timing collapse, and **does not regress** badly on onshore data.
- A **false-positive rate on OBS noise** at or below the site-tuned models'.
- A short results note (mirroring a config header post-mortem) documenting what
  you changed and what happened — so the next person learns from it.

---

## 8. Pitfalls to avoid (hard-won from Phase 1)

- **Don't remove distillation** (`alpha: 0`) — timing collapses.
- **Don't add the timing or presence losses** to "fix" timing — they collapse
  recall (`timing_beta`/`presence_gamma` ≥ small values were lethal).
- **Don't tune the detection threshold on your test set** — keep a separate
  tuning split.
- **Don't trust `val_loss`/`val_p_mae_s` as proxies for OBS generalization** —
  Phase 1 showed validation metrics don't predict cross-domain benchmark
  performance. Judge on a real held-out OBS benchmark.
- **Don't assume channel order / orientation** for OBS horizontals.

---

## 9. Learning resources

- PhaseNet: Zhu & Beroza (2019), *GJI*, `10.1093/gji/ggy423`.
- SeisBench docs & datasets: <https://seisbench.readthedocs.io>.
- Benchmark philosophy: Münchmeyer et al. (2022), `10.1029/2021JB023499`.
- Our deployment context: Ni et al. (2025), *Seismica*,
  `10.26443/seismica.v4i2.1738`.
- The project paper (`../paper_draft.html`) — start with §Loss design and
  §Critical audit.

Questions → ask Marine; for code specifics, read the file the step points to
first, then ask. Keep a running log of what you try and the result.
