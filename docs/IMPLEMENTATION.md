# Implementation — retired doc

This document described the `PhaseNetLightning`/`scripts/model.py` +
`scripts/data_module.py` + `scripts/train.py` + `scripts/evaluate.py`
pytorch_lightning-based implementation — a template scaffold (Pipeline A in
`paper_draft.qmd`'s terminology) that was never used to train any of the
v1–v19 models and had drifted out of sync with the real pipeline. Those four
files were unused dead code and have been deleted (GitHub #12).

**The real, actually-used implementation:**

- Model + loss: `scripts/fine_tune_model.py` (`PhaseNetFinetune`, pure
  PyTorch — cross-entropy + knowledge distillation, optional timing/presence
  terms)
- Data loading: `scripts/manifest_dataset.py` (`ManifestDataset`, on-the-fly
  waveform fetch from the manifests built by
  `scripts/build_training_dataset.py`)
- Training driver: `scripts/finetune.py` (`python scripts/finetune.py
  --config configs/finetune_jma_wc_global_v7.yaml`)
- Evaluation: `scripts/eval_finetuned.py` → `notebooks/step3_metrics.csv`

See the top-level [`README.md`](../README.md) for the end-to-end flow
diagram, and [`../paper_draft.qmd`](../paper_draft.qmd) §"Two pipelines
exist" for the full history of why this Lightning path existed and was
abandoned.
