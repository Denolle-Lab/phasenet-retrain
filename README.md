# PhaseNet Retraining Framework

Code and configuration for retraining **PhaseNet** seismic phase pickers
(P- and S-wave arrival-time detection) on a cleaned, hybrid, rebalanced
multi-dataset corpus, toward a **globally deployable** picker for onshore
(Phase 1) and offshore / ocean-bottom (Phase 2) networks.

Developed by the **Denolle Lab**, University of Washington, Department of Earth
and Space Sciences.

> **This repository is the *code*, not the data or the science.**
> Like most research-software READMEs, it documents how to run the pipeline.
> The motivation, datasets, methods, results, leaderboard, and a full internal
> audit are written up in the **project paper**:
> [`paper_draft.html`](paper_draft.html) (source: [`paper_draft.qmd`](paper_draft.qmd),
> render with `quarto render paper_draft.qmd`).
>
> The **data is not hosted here** — the SeisBench waveform datasets are hundreds
> of GB to multiple TB and live on the lab back-end Linux servers. See
> [`data/README.md`](data/README.md) for the dataset inventory, volumes, and
> server location.

## What this pipeline actually does

We **fine-tune the SeisBench `jma_wc` PhaseNet** on a hybrid corpus assembled
from ~20 SeisBench datasets, using **knowledge distillation** from a frozen copy
of `jma_wc` as a regularizer against catastrophic forgetting. Models are
benchmarked with **Münchmeyer et al. (2022)-style** metrics (recall, MAE,
outlier rate, MCC) across local / regional / teleseismic distance bins.

The end-to-end flow:

```
SeisBench datasets ──build_training_dataset.py──▶ manifest CSVs (data/manifests_*)
                                                      │
                          build_noise_dataset.py──▶ noise corpus
                                                      │
                                   finetune.py ──────▶ fine-tuned checkpoints
                                  (config v1..v19)     │
                              eval_finetuned.py ──────▶ step3_metrics.csv + figures
```

> **Note on the older docs.** Earlier docs in `docs/` (and prior versions of
> this README) described a 5-dataset, pytorch_lightning-based scaffold
> (`scripts/model.py` + `scripts/data_module.py` + `scripts/train.py` +
> `scripts/evaluate.py`) that was never how the v1–v19 models were trained.
> Those files were unused dead code and have been deleted (GitHub #12); the
> real pipeline is the manifest-based one above (`build_training_dataset.py` →
> `manifest_dataset.py` → `finetune.py`). `scripts/label_error_filter.py`
> started life in that same scaffold but is *not* dead — it's now wired into
> `build_training_dataset.py` (GitHub #10) and does apply to the real
> training pool.

## Repository layout

```
configs/      finetune_jma_wc_global_v1..v19.yaml  — versioned experiments (with post-mortems)
scripts/      build_training_dataset.py            — assemble hybrid manifests
              build_noise_dataset.py               — assemble noise corpus
              fine_tune_model.py                    — model + loss (CE + KD + optional terms)
              finetune.py                           — training driver
              manifest_dataset.py                   — manifest-backed Dataset
              eval_finetuned.py                     — benchmark evaluation → step3_metrics.csv
              scratch_model.py / train_scratch.py  — train-from-scratch path (soft-CE)
notebooks/    benchmark construction + step-3 evaluation figures (step3_*.png)
data/         manifest CSVs only (git-ignored); see data/README.md — no waveforms
checkpoints/  trained weights (git-ignored)
results/      metrics + figures (git-ignored)
paper_draft.qmd / .html   — the project paper (why / what / how + audit)
```

## Installation

```bash
git clone https://github.com/Denolle-Lab/phasenet-retrain.git
cd phasenet-retrain
python -m venv venv && source venv/bin/activate   # or conda
pip install -r requirements.txt
```

Core stack: PyTorch ≥ 2.0, SeisBench ≥ 0.4, ObsPy ≥ 1.4 (see
[`requirements.txt`](requirements.txt)).

> **Portability caveat.** The SeisBench cache path is currently **hard-coded**
> in the build/train/eval scripts (update it to your local path before running).
> Making it configurable via `SEISBENCH_CACHE_ROOT` is on the TODO list.

## Usage

```bash
# 1. Build the hybrid training manifests from the SeisBench cache
python scripts/build_training_dataset.py            # add --s-balanced for manifests_v3

# 2. (optional) Build the noise corpus
python scripts/build_noise_dataset.py

# 3. Fine-tune (champion recipe = v7)
python scripts/finetune.py --config configs/finetune_jma_wc_global_v7.yaml
#    resume / stage from a checkpoint:
python scripts/finetune.py --config <cfg.yaml> --init-from checkpoints/.../best.pt

# 4. Evaluate on the benchmark → notebooks/step3_metrics.csv + figures
python scripts/eval_finetuned.py
```

Each `configs/finetune_jma_wc_global_v*.yaml` header documents the change it
tests and its measured outcome — read them top-to-bottom for the experimental
narrative (summarized in the paper, §Trajectory).

## Status and known issues

This is **active research code**, not a production release. Before any model is
promoted, see the **audit section of the paper** (`paper_draft.html`, §Critical
audit). The headline open items:

- The benchmark **"cross-domain" split is currently a no-op** for the fine-tuned
  models, and the training manifests are not committed, so **train/test
  independence is unverified**.
- Headline **P-MAE is unconditional** (averaged over undetected traces, residuals
  saturated at the ±5 s search window) — not yet matched to the Münchmeyer
  detected-only definition.
- The benchmark uses an **oracle ±5 s window**, so **precision / false-positive
  rate is not measured** — the reliability metric the project most needs.
- **No fine-tuned model yet beats the `jma_wc` baseline on all metrics**: v7
  improves timing (P-MAE 0.340 vs 0.374 s) but loses recall and MCC.

## Citation

If you use this code, please cite PhaseNet, SeisBench, and the deployment papers
this work supports:

- **Zhu, W. & Beroza, G. C. (2019).** PhaseNet: a deep-neural-network-based
  seismic arrival-time picking method. *GJI* 216(1), 261–273.
  `10.1093/gji/ggy423`
- **Woollam, J., et al. (2022).** SeisBench — A toolbox for machine learning in
  seismology. *SRL* 93(3), 1695–1709. `10.1785/0220210324`
- **Münchmeyer, J., et al. (2022).** Which picker fits my data? *JGR Solid Earth*
  127, e2021JB023499. `10.1029/2021JB023499`
- **Ni, Y., et al. (2025).** A Global-scale Database of Seismic Phases from
  Cloud-based Picking at Petabyte Scale. *Seismica* 4(2), 1738.
  `10.26443/seismica.v4i2.1738`
- **Ni, Y., et al. (2025).** A review of cloud computing and storage in
  seismology. *GJI* 243(1), ggaf322. `10.1093/gji/ggaf322`
- Label-error method: **Aguilar, A. L., et al.** —
  [`albertleonardo/labelerrors`](https://github.com/albertleonardo/labelerrors).

Full dataset references and DOIs: see the paper, §Datasets.

## Contact

**Denolle Lab**, University of Washington — [denolle-lab.github.io](https://denolle-lab.github.io).
For questions or collaboration, open an issue.
