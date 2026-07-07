# tests/

Starter unit tests for the project. Currently covers the benchmark-metric
functions that the audit + the code-review-graph coverage analysis flagged as
**untested and buggy**.

## Run

```bash
pip install pytest          # already in requirements.txt
pytest tests/ -v
```

Requires `numpy`, `pandas`, `scikit-learn` (present in the `surface` env). Tests
`importorskip` those, so they skip cleanly if a dep is missing.

## What's covered

`tests/test_metrics.py` — `compute_metrics` and `get_split_mask` from
`scripts/compare_v7_thresholds.py`:

- **Regression guards** pin the correct recall / MAE / outlier math and the
  `get_split_mask` in-domain exclusion.
- **Bug-documenting tests** encode the two known defects so the fix is obvious:
  - `test_mae_is_currently_unconditional_and_threshold_independent` → issue **#8**
    (MAE must become detected-only / Münchmeyer-comparable). Flip the assertion
    when fixed.
  - `test_get_split_mask_is_a_noop_for_finetuned_weights` → issue **#7**
    (cross-domain split is a no-op for `jma_wc_ft_*`). Flip when fixed.

## Known limitation (issue #12)

`compare_v7_thresholds.py` runs its analysis driver at **import time** (reads
`step3_results.parquet`, writes CSV/PNG), so it can't be imported in a test. The
test file extracts the two pure functions via AST as a temporary shim. Once the
driver is guarded under `if __name__ == "__main__":`, replace `_load_pure()`
with a plain `from scripts.compare_v7_thresholds import ...` and delete the shim.
