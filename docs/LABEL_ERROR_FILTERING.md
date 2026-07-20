# Label Error Filtering for PhaseNet Retraining

## Overview

This document describes the label error filtering applied to the training
pool. The approach is based on the confident-learning analysis in
[albertleonardo/labelerrors](https://github.com/albertleonardo/labelerrors)
(arXiv:2511.09805), which flags traces where a trained model is
systematically more confident in a different pick than the assigned label —
typically an unlabeled second earthquake in the window (a "multiplet") that
the model detects but the label doesn't record.

## What is actually applied

`scripts/label_error_filter.py` downloads (and caches under
`~/.cache/phasenet_retrain/label_errors/`, or reuses `data/labelerrors/` if
already populated) the multiplet report for each dataset the labelerrors repo
publishes one for, and returns the flagged `trace_name` strings. Matching is
keyed on `trace_name` — the fix for GitHub #10 (the previous version matched
the report CSV's positional row index against the dataset's own row index
instead, which is not a valid shared key and silently excluded nothing).

`scripts/build_training_dataset.py` (`load_label_error_exclusions()`, wired
into `process_dataset()`) applies this exclusion to the training pool by
default — pass `--no-label-error-filter` to skip it. Every run writes
`label_error_removal_report.csv` alongside the manifests, with the exact
per-dataset before/after row counts for that specific build (cap, benchmark
exclusion, and S-pick filtering all interact with how many flagged traces are
actually present in the pool, so the in-pool removal count differs from the
raw dataset-level count below).

**Only multiplet reports are used** (not the noise reports the labelerrors
repo also publishes for instance/pnw/stead/txed) — this keeps the removal
counts directly comparable to the benchmark pool's cleaning
(`notebooks/04_creating_benchmark_dataset.ipynb` §1.4b), which uses the same
multiplets-only methodology.

**`iquique` is intentionally excluded** — a PI decision recorded in the
benchmark notebook, carried over here for consistency.

## Real removal rates (dataset-level, verified 2026-07-20)

Flagged trace_names as a fraction of the full SeisBench dataset (before any
of our own cap/distance/benchmark filtering) — run
`python scripts/label_error_filter.py` to reproduce:

| Dataset (internal name) | Bad labels flagged | % of full dataset |
|---|--:|--:|
| `instancecounts` | 92,561 | 7.99% |
| `aq2009gm`        | 26,739 | 10.33% |
| `ceed`            | 253,309 | 5.06% |
| `pnw`             | 4,970 | 2.70% |
| `ethz`            | 895 | 2.44% |
| `txed`            | 4,672 | 0.90% |
| `stead`           | 149 | 0.01% |

`instancecounts`/`pnw`/`ethz`/`txed`/`stead` match the counts already
published in the benchmark notebook. `ceed` and `aq2009gm` reports were not
yet published when that notebook was written; they exist now and are applied
here on the training side (not backported to the benchmark notebook in this
pass, since editing the benchmark pool needs separate sign-off).

Datasets with no labelerrors report at all (not analysed by Aguilar):
`mlaapde`, `obst2024`, `pisdl`, `vcseis`, `cwa`, `geofon`, `lendb`, `crew`,
`scedc`, `meier2019jgr`, `ross2018gpd`, `obs`. Included in the training pool
without label-error filtering — a real limitation, not a placeholder.

## Usage

```python
from label_error_filter import load_bad_trace_names

bad = load_bad_trace_names("instancecounts")  # -> set of trace_name strings
```

```bash
# Rebuild manifests with label-error filtering (on by default):
python scripts/build_training_dataset.py --output-dir data/manifests_v4

# Inspect what would be flagged, without building anything:
python scripts/label_error_filter.py
```

## Cache management

Reports are cached to avoid repeated downloads at
`~/.cache/phasenet_retrain/label_errors/<stem>_report.csv`, or reused directly
from `data/labelerrors/` if already populated there. To force a re-download,
delete the relevant cached file.

## References

1. albertleonardo/labelerrors: https://github.com/albertleonardo/labelerrors (arXiv:2511.09805)
2. PhaseNet paper: Zhu & Beroza (2019)
3. EQTransformer paper: Mousavi et al. (2020)
4. SeisBench: Woollam et al. (2022)
