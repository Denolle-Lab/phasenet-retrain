# Datasets — retired doc, see data/README.md

This document previously described a "5 focused datasets" template
(STEAD/INSTANCE/ETHZ/PNW/TXED, driven by `scripts/data_module.py`'s YAML
config) that was never the pipeline used to train v2–v19. That path is
unused/dead code (GitHub #12) — see `paper_draft.qmd` §"Two pipelines exist"
for the full Pipeline A vs. Pipeline B distinction.

**The real, actually-used dataset inventory** (20 SeisBench datasets, real
on-disk volumes, real trace counts, how the hybrid training pool is
assembled) lives in [`../data/README.md`](../data/README.md). That is the
document to read and to keep updated going forward — this file is retired
and should not be edited further.

For dataset-level *label-error* cleaning (which datasets are Aguilar-audited
and their real removal fractions), see
[`LABEL_ERROR_FILTERING.md`](LABEL_ERROR_FILTERING.md).
