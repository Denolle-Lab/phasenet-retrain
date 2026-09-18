# tests/

Starter unit tests for the project. Currently covers the benchmark-metric and
domain-split helpers that the audit + the code-review-graph coverage analysis
flagged as **untested**.

## Run

```bash
pip install -r requirements.txt   # includes pytest, numpy, pandas, scikit-learn
pytest tests/ -v
```

Tests `importorskip` `numpy` / `pandas` / `scikit-learn`, so they skip cleanly if
a dependency is missing.

## What's covered

`tests/test_metrics.py`:

- **`domain_registry.split_masks`** (imported normally) — the in_domain /
  cross_domain split. Regression guard for the **#7 fix** (commit `244473f`):
  public weights split on `trained_models` vs their known corpus; our fine-tunes
  split on the `dataset` column vs the training manifest's datasets — this is the
  logic that replaced the old all-True no-op.
- **`compute_metrics`** (from `compare_v7_thresholds.py`) — pins the correct
  recall / MAE / outlier math, plus one **bug-documenting** test,
  `test_mae_is_currently_unconditional_and_threshold_independent` → issue **#8**
  (MAE must become detected-only / Münchmeyer-comparable). Flip that assertion
  when fixed.

## Known limitation (issue #12)

`compare_v7_thresholds.py` runs its analysis driver at **import time** (reads
`step3_results.parquet`, writes CSV/PNG), so it can't be imported in a test. The
test file extracts `compute_metrics` via an AST shim as a temporary workaround.
Once the driver is guarded under `if __name__ == "__main__":`, replace the shim
with a plain `from scripts.compare_v7_thresholds import compute_metrics` and
delete `_load_compute_metrics()`. (`domain_registry.py` is already clean and is
imported normally.)

## General-picker loader (#34A)

`test_manifest_dataset.py` generates temporary HDF5/CSV fixtures and checks
waveform/arrival timing, metadata validation, S-only support, explicit noise,
bucket/chunk handling, rejection ledgers, cache failure, and spawned workers.
It uses actual local SeisBench readers without downloading data. Requires the
training dependencies; skips when PyTorch/SeisBench/h5py are unavailable.
It can also run without pytest:

```bash
python -m unittest discover -s tests -p test_manifest_dataset.py -v
```

See [the loader contract](../docs/2026-09-10_34a_loader_contract.md) for migration
requirements and the remaining historical/deployment gates.

## Historical v7 row replay (#34B)

`test_v7_row_audit.py` exercises the pinned legacy loader and repaired loader
against synthetic source files, checks timing/identity/failure distinctions,
and verifies input/output hashes, duplicate-row accounting, and exposure
provenance. It does not read server data or download datasets.

```bash
python -m unittest discover -s tests -p test_v7_row_audit.py -v
```

See [the #34B runbook](../docs/2026-09-11_34b_row_forensics.md) for server inputs,
artifact interpretation, and the remaining historical acceptance criteria.
