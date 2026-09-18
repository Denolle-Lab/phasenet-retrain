# Train again: from a fresh server clone to a scored T0 model

*2026-09-18, branch `issue/40a-t0-profile` from
`audit/2026-09-07-generalization` at `083c45c`. The forward path only: build
the T0 pilot corpus with a corpus profile, train the E1 base arm from the
`instance` weights, export and score it on the development cases at matched
budget. Checkpoints 40A (corpus) and 46B (E1) of
`docs/2026-09-11_training_strategy_v3.md`; the historical forensics (34B,
E0) are not on this path. Every command below was exercised on the laptop
against fixtures and, for the scoring step, against the four built
development cases with a perturbed `instance` checkpoint (CPU); no model
has been trained.*

## 0. What this produces, and what it does not

A T0 model is a pilot for choosing the recipe (mask, distillation, BN
statistics, learning rate) among the E1 arms of strategy §8. It is not the
campaign picker: the T0 corpus is 50,000 to 100,000 windows against the
2 to 4 million of T2, has no real-noise corpus and no augmentation beyond
the loader's crop, amplitude jitter and polarity flip. The decision rule
that would make any model the picker is strategy §2 (frozen thresholds per
weight, paired block bootstrap on the sealed panel, margins per endpoint);
nothing here scores the sealed panel. What the path answers: which E1 arm
leads the base recipe on the development cases at matched budget with
three seeds, and whether the corrected loader plus `instance` init beats
the parents there at all.

## 1. Pieces added on this branch

| Piece | File | Role |
|---|---|---|
| Corpus profiles | `configs/corpus_profiles.yaml`; `--profile`, `--profiles-file`, `--list-profiles` in `scripts/build_training_dataset.py` | `t0_pilot` (strategy §4.1) and `legacy_v2` (the module defaults, byte-identical build) |
| Manual-status filter | `apply_status_filter` in the builder; `require_status_columns` per source | picks whose status column is present and not `manual` are nulled; absent columns are a printed no-op |
| E1 configs | `configs/e1_t0/{base,mask_off,kd_t1p5,kd_t4,bn_frozen,lr_2e-6,lr_2e-5,base_jma_wc}.yaml`; seeds 1 and 2 under `configs/e1_t0/seeds/` from `scripts/e1_seed_configs.py` | the base recipe and the one-factor arms of strategy §8 E1 |
| Frozen BN statistics | `training.freeze_bn_stats` in `scripts/fine_tune_model.py` (recorded by the run card) | BatchNorm modules in eval mode during training, affine parameters trainable |
| Window normalisation contract | `waveform_contract.NORMS`, `normalise_waveform`; `norm` on `ManifestDataset` and `CachedManifestDataset`; `manifest_data_module.resolve_norm`; `data.norm` in the configs; `norm` in every checkpoint and in the run card | the loader normalises the way the parent's `annotate()` does: `std` for `jma_wc`, `peak` for `instance` (SeisBench 0.12.5 `phasenet.py` lines 192, 199-204) |
| Post-run scoring | `scripts/score_checkpoint.py`; `load_weights` in `scripts/heldout_testset_score.py` | `best.pt` to a SeisBench pair declaring the run's norm, scored with the 35A engine against `instance` and `jma_wc` at matched budget on the dev cases |
| Tests | `tests/test_corpus_profiles.py`, `tests/test_train_again.py` | profiles, status filter, P-only refusal, fraction renormalisation, configs, seed copies, BN freeze, `instance` build, export round trip, scoring wrapper |

## 2. The path, in order

Paths follow `docs/2026-09-13_server_session_runbook.md` as amended by
PR #87 (a clone of your own; Akash's clone and cache under
`/data/wsd04/ak287/` are read-only inputs). Steps 2.1 and 2.2 are runbook
steps 0, 4 and 5; the rest is new.

### 2.1 Environment and tests (runbook §0)

```bash
curl -fsSL https://pixi.sh/install.sh | sh && exec $SHELL   # once; installs pixi to ~/.pixi, no root
cd /data/<your area>                                   # writable, with room for the manifests, checkpoints and results
git clone https://github.com/Denolle-Lab/phasenet-retrain.git && cd phasenet-retrain
git checkout audit/2026-09-07-generalization
pixi install                                           # CPU env for every script and test
pixi install -e cuda                                   # on the GPU node; see the CUDA note below

# a cache of your own: datasets are Akash's by symlink (read-only), models are yours (writable;
# score_checkpoint.py writes nothing there, but from_pretrained may fetch a missing parent)
mkdir -p $HOME/.seisbench_phasenet/models
ln -sfn /data/wsd04/ak287/.seisbench/datasets $HOME/.seisbench_phasenet/datasets
export SEISBENCH_CACHE_ROOT=$HOME/.seisbench_phasenet
ls $SEISBENCH_CACHE_ROOT/datasets | head

# the historical clone, read-only; its manifests only add counts in step 2.2 and may be absent
HIST=/data/wsd04/ak287/<clone>
mkdir -p data/manifests_v2 && cp $HIST/data/manifests_v2/*.csv data/manifests_v2/ 2>/dev/null || echo "no historical manifests (fine for this path)"

pixi run versions                    # python 3.11, numpy 1.26, torch 2.13, seisbench 0.12.6, obspy, pyocto, cuda
pixi run test                        # laptop: 450 passed
pixi shell                           # every command below runs inside this shell (GPU node: pixi shell -e cuda)
```

CUDA note: the lock resolves PyTorch's CUDA 12.9 build for `linux-64`. If
`nvidia-smi` shows a driver below CUDA 12.9, add
`cuda-version = "12.<x>.*"` to `[feature.cuda.dependencies]` in `pixi.toml`,
run `pixi lock`, then `pixi install -e cuda`, and commit the lock change
with the run.

Every output of this path (manifests, checkpoints, results, exports,
scores) goes under your clone; nothing writes into `$HIST` or into the
symlinked `datasets`.

### 2.2 Exclusion bundle (runbook steps 4 and 5)

```bash
python scripts/audit_heldout_sequences.py                                       # task 1: data/exclusions/heldout_sequences.csv
python scripts/exclusion_bundle.py build --cache-root $SEISBENCH_CACHE_ROOT    # 33A: hashes every input and source snapshot
python scripts/exclusion_bundle.py show                                         # must say certified: true
git add data/exclusions/heldout_sequences.csv data/exclusions/heldout_sequence_counts.csv data/exclusions/bundle.json
```

The builder refuses to run without a certified bundle
(`--allow-uncertified-bundle` is the recorded escape hatch).

### 2.3 Recommended before the build: the 39A census (runbook step 6)

```bash
python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/census
```

This fills `status_columns` and `pick_status_counts` per source in
`data/census/seisbench_sources.csv`. The `t0_pilot` profile lists
`trace_P_status`, `trace_S_status`, `trace_P_arrival_status` and
`trace_S_arrival_status` for `instancecounts`; the committed record
(`source_census.STATUS_RECORD`) shows no status column in the five rows
the notebook sampled, only uncertainty and location-weight columns. If the
census confirms none exists, the build prints
`WARNING: none of the status columns ... exists in instancecounts` and the
"manual picks only" clause of strategy §4.1 is not expressible from
metadata for INSTANCE. Then decide: drop the source, or add an
uncertainty-based rule (not written; the profile schema takes only status
columns). Do not silently accept the warning.

### 2.4 Build the T0 manifests

```bash
python scripts/build_training_dataset.py --list-profiles
python scripts/build_training_dataset.py --profile t0_pilot --output-dir data/manifests_t0 --seed 42
python scripts/exclusion_bundle.py check data/manifests_t0/train.csv --kind signal
cat data/manifests_t0/heldout_removal_report.csv
python -c "import json; p=json.load(open('data/manifests_t0/provenance.json')); print(p['profile'], p['target_fractions'], p['manifests'])"
```

What the profile does, per source (caps sum to 90,000 before the bundle,
the status filter and the split):

| Source | Cap | Note |
|---|--:|---|
| ethz | 10,000 | 100 to 500 Hz native; S on few rows |
| pnw | 10,000 | 100 Hz; no status column |
| cwa | 10,000 | chunked loader |
| scedc | 10,000 | 40 and 100 Hz |
| ceed | 10,000 | 100 Hz |
| txed | 10,000 | no status column |
| iquique | 5,000 | no status column |
| instancecounts | 15,000 | status filter (§2.3) |
| vcseis | 10,000 | volcanic; Hawaii tier-2 hold-out via the bundle |

Rules: no `use_s: false` source (the profile refuses one); rows with a
known distance above 2,000 km dropped before the cap (rows without a
distance kept); training-split fractions local 0.55, regional 0.40,
unknown 0.05, teleseismic 0 (a bin at 0 is absent from the training split;
val and test keep their rows); `obst2024` and `obs` skipped. S-only rows
stay (the builder keeps any row with a P or an S). The row-level rule that
nulls S on a teleseismic-bin row is unchanged from the legacy path; with
the teleseismic fraction at 0 it touches no training row. The removal
report carries, per source, the bundle counts, `n_beyond_max_distance`,
the status-filter counts, `n_after_cap`, `n_s_nulled_teleseismic`,
`n_written` and `n_with_s_written` (rows and S labels the source
contributes to the pool after the teleseismic P-only rule, so the S count
matches the written rows) and the profile name and hash;
`provenance.json` carries the profile record, the normalised fractions and
the source list.

### 2.5 Loader smoke test and the ledger gate (runbook step 9, on the new rows)

```bash
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
import pandas as pd
from fast_manifest_dataset import CachedManifestDataset
pd.read_csv("data/manifests_t0/train.csv").sample(2000, random_state=0).to_csv("/tmp/t0_smoke.csv", index=False)
ds = CachedManifestDataset("/tmp/t0_smoke.csv", label_policy="masked", return_mask=True, load_workers=8)
print(len(ds), ds.n_supervised_samples)
PY
ls /tmp/t0_smoke.rejected.*.jsonl 2>/dev/null && echo "rejections: repair the manifest, do not skip rows"
```

The 34A loader raises on the first row it cannot read and writes the
ledger; a rejection on the T0 rows is a builder or source problem to fix
before training (`docs/2026-09-10_34a_loader_contract.md`). `finetune.py`
refuses a manifest with a ledger beside it.

### 2.6 Train the base arm, three seeds

```bash
python scripts/e1_seed_configs.py --check      # committed seed copies match the arm files
for cfg in configs/e1_t0/base.yaml configs/e1_t0/seeds/base_seed1.yaml configs/e1_t0/seeds/base_seed2.yaml; do
  run=$(python -c "import yaml,sys; print(yaml.safe_load(open('$cfg'))['logging']['run_name'])")
  mkdir -p results/$run
  nohup python scripts/finetune.py --config $cfg > results/$run/train.log 2>&1
  python scripts/run_card.py check results/$run/run_card.json    # exit 0: ledger gate passed, checkpoint hashed
done
```

The base recipe (`configs/e1_t0/base.yaml`): `instance` init, masked label
policy, `data.norm: peak` (the parent's normalisation), soft cross-entropy,
no distillation, adaptive BN, AdamW 5e-6 with weight decay 1e-4, batch 256,
warm-up 2 epochs then cosine to 1e-6, gradient clip 1.0, at most 60 epochs,
early stopping on `val_loss` with patience 10. The loader prints
`Window normalisation: peak (config)` at preload; the run card records
`config.norm` and `extra.norm` (value and origin), and every checkpoint
carries `norm` and `parent`. Early stopping on `val_loss` is the interim rule: strategy §7
wants the development metric per epoch on a fixed excerpt, and that scorer
is not wired into the loop yet; the checkpoint choice therefore uses
validation loss and the arm choice uses §2.7. Run the loop sequentially or
one run per GPU; `torch.compile` and AMP are on in the config.

### 2.7 Export and score

```bash
python scripts/score_checkpoint.py --run results/e1_t0_base_seed0
python scripts/score_checkpoint.py --run results/e1_t0_base_seed1
python scripts/score_checkpoint.py --run results/e1_t0_base_seed2
```

Per run: `data/evaluation/exports/<run>.{json,pt,export.json}` (the
SeisBench pair, reloadable with `sbm.PhaseNet.load` or
`heldout_testset_score.load_weights`; the pair declares the norm the run
trained with, read from the checkpoint, else the run card, else the
config's `data.norm`, else `--norm`; a checkpoint whose norm none of these
states is refused; the sidecar has the checkpoint, `.pt` and `.json`
sha256, epoch, parent, parent norm, export norm and its origin), then the 35A engine on
every built case with role `dev` in `configs/evaluation_suites.json`
(Samos, Adriatic 2022, Etna, Corinth-Thiva) with the candidate,
`instance` and `jma_wc`, thresholds 0.02 to 0.90 in steps of 0.02, budget
set by `instance` at 0.3 (`--budget-reference`, `--budget-threshold`).
Output: `data/evaluation/scores/<key>/<access_id>/` (the pick store and
matches, for `paired_bootstrap.py` and 36B) and
`data/evaluation/summary/<run>/matched_budget.csv`, printed as one row per
case and phase with each model's recall and threshold at the budget and
the candidate minus each parent. The dev cases need their waveforms
(`data/heldout_testset/<key>/waveforms/`, built or linked as
`docs/baselines_2026-09-13/README.md` shows; another built-sequence root
is `--sequences-root`, separate from `--out-root`); the annotations are
cached, so rescoring a parent costs nothing.

Read the table against the parents' own numbers in
`docs/baselines_2026-09-13/matched_budget.csv` (same engine, `jma_wc` as
the budget reference there). For an interval, run
`docs/baselines_2026-09-13/paired_bootstrap.py data/evaluation` on the
written artifacts. Three seeds per arm; report the seed spread, not the
best seed.

### 2.8 The arms

```bash
for arm in mask_off kd_t1p5 kd_t4 bn_frozen lr_2e-6 lr_2e-5; do
  for cfg in configs/e1_t0/$arm.yaml configs/e1_t0/seeds/${arm}_seed1.yaml configs/e1_t0/seeds/${arm}_seed2.yaml; do
    run=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['logging']['run_name'])")
    mkdir -p results/$run
    python scripts/finetune.py --config $cfg > results/$run/train.log 2>&1
    python scripts/run_card.py check results/$run/run_card.json
    python scripts/score_checkpoint.py --run results/$run
  done
done
# the second parent, same recipe (not one of the seven E1 arms; brackets E4)
python scripts/finetune.py --config configs/e1_t0/base_jma_wc.yaml
```

Each arm file changes one key of the base and says so in its header:
`mask_off` (`data.label_policy: legacy`), `kd_t1p5` and `kd_t4`
(`distillation` alpha 0.3 at T 1.5 and 4, teacher = the arm's parent),
`bn_frozen` (`training.freeze_bn_stats: true`), `lr_2e-6`, `lr_2e-5`.
`tests/test_train_again.py` proves the one-factor property.

## 3. What to expect, and two confounds to carry

- **Normalisation follows the parent.** `instance` was trained with
  SeisBench norm `peak`, `jma_wc` with `std`. The loader now takes
  `data.norm` (every E1 config states it: `peak` for the `instance` arms,
  `std` for `base_jma_wc`) and, when a config omits it, reads the parent's
  `model_args.norm` through `PhaseNet.from_pretrained(name).norm`
  (`manifest_data_module.resolve_norm`, printed at preload, recorded in
  the run card with its origin). `waveform_contract._normalise_peak`
  matches SeisBench 0.12.5 `phasenet.py` `annotate_batch_pre` lines 192
  and 202-204 (per-component demean, divide by max |x| over time), checked
  on a random batch against the installed routine in
  `tests/test_train_again.py`; the one difference is a flat channel with a
  DC offset, whose float32 residue SeisBench serves at unit amplitude and
  the loader keeps at zero (a 34C item). The std path is unchanged
  (population std, guard, clip at 10; SeisBench's unbiased std differs by
  a factor 1.00017 at 3001 samples). The input statistics therefore match
  the parent from the first step, the `bn_frozen` arm is no longer
  confounded by the input scale, and the export declares the run's norm so
  `annotate()` sees what training saw. `PhaseNetFinetune` warns when
  `data.norm` disagrees with the parent's norm.
- **Probability scale.** `instance` attains the parent budget at 0.04 to
  0.24 (`docs/baselines_2026-09-13/README.md`); a fine-tune from it will
  sit on its own scale, which is why the table reports recall at matched
  budget and the threshold that attains it, never a shared threshold.
- **Val loss is not the selection metric.** It is logged and drives early
  stopping for now; arm selection is on the development cases (§2.7).
- **Samos** has 34 P and 24 S references on three stations; 37A marks it
  not evaluable. Its row is reported, not read.

## 4. Deliberately left out

| Left out | Where it lives |
|---|---|
| E0, the alignment diagnostic on the v7 rows | 46A, `configs/e0_46a_*_targets.yaml`; needs 34B |
| Noise pools beyond the pilot (no real-noise corpus at all here: `noise_prob 0`) | 42A |
| Augmentation beyond crop, jitter and flip; per-epoch crops (the cached loader draws one crop per row at preload) | 43A, 47A (E2) |
| The bulletin-harvest slices (INGV, NOA, GeoNet) of the T0 tier | 40A proper, through `scripts/build_heldout_testset.py` |
| Per-epoch development metric in the training loop | 35A wiring, strategy §7 |
| Per-weight thresholds on calibration station-days | #38 |
| 6000-sample context, width | E5 (48A, 48B) |
| The sealed panel | E6, after 44B |

## 5. Compute estimate

Assumptions, stated so they can be replaced by the first measured epoch:
about 70,000 training windows after the bundle, the status filter, the
10/10 event-grouped split and the distance stratification (90,000 at cap
minus exclusions; the removal report gives the real number); standard-width
PhaseNet (268,443 parameters, 3001 samples); batch 256 with AMP on one
RTX 3090; the in-RAM cached loader, so no I/O in the epoch. At a guessed
1,500 windows per second for forward plus backward, one epoch is about
50 s of training plus a few seconds of validation; 60 epochs are about
1 h per run and early stopping at patience 10 usually ends earlier. Three
seeds of the base: about 3 GPU-hours. The six arms at three seeds: about
18 GPU-hours, the KD arms slower by the teacher forward (about 1.3 times).
`base_jma_wc` has four times the parameters (1,070,899) and runs about
three times slower per epoch. RAM for the cache:
`N * 3 * 3001 * 4 * 2 / 1e9` GB (the loader's own formula, waveforms plus
labels) is 6.5 GB at 90,000 windows plus the masks; validation adds a
tenth. The v7 per-epoch time in `results/finetune_jma_wc_global_v7_metrics.csv`
(server) is the reference for the PhaseNetWC number; the metrics CSV has no
wall-clock column, so read `t(s)` from the training log instead. Replace the
1,500 windows per second with the first epoch's `t(s)` before scheduling
the arms.

## 6. Validation on 2026-09-18

Pixi environment (2026-09-18): `pixi run test`, 450 passed with the cached weights. Before pixi: base `python` 3.9.20 (no torch): `python -m pytest tests -q`, 400 passed,
20 skipped (the torch and cached-weight tests).
Torch venv (Python 3.11, torch 2.2.2, SeisBench 0.12.5, cached `instance`
and `jma_wc`): 450 passed. `scripts/score_checkpoint.py` was run on the
laptop CPU against the four built development cases with a checkpoint made
from `instance` through `PhaseNetFinetune` (`configs/e1_t0/base.yaml`, so
`norm: peak`) with every parameter perturbed by 1e-3, against `instance`
and `jma_wc`: the export reloaded with norm `peak` read from the
checkpoint, the four cases scored, and the summary table was written. That
run is a path check, not a result; its output directory is not committed.
It did show the point of the contract: the perturbed `instance` exported
with `norm: peak` reproduces the parent's operating point (recall within
0.01 of `instance` at the same threshold 0.3 on all eight case-phase rows),
whereas the same weights exported under the earlier fixed `std` sat 0.06
to 0.10 below the parent and attained the budget only at thresholds of
0.04 to 0.06, with two S rows not attaining it at all.
