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
