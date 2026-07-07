"""
Starter unit tests for the benchmark-metric functions in
`scripts/compare_v7_thresholds.py`.

Why this file exists
--------------------
The code-review-graph knowledge graph flagged `compute_metrics` and
`get_split_mask` as *untested*, and the internal audit found real correctness
issues in both:

  * `compute_metrics`  -> MAE/outlier are UNCONDITIONAL (averaged over undetected
    traces, residuals saturated at the +/-5 s search window), so they are not
    comparable to the Munchmeyer et al. (2022) detected-only definition.
    (GitHub issue #8)
  * `get_split_mask`   -> returns all-True for any weight not listed in
    `TRAINED_ON`, so the benchmark "cross_domain" split is a no-op for every
    `jma_wc*` / `jma_wc_ft_*` model (cross_domain == all).  (GitHub issue #7)

These tests (a) pin the *correct* math that must not regress, and (b) DOCUMENT
the two bugs with tests that will need to be flipped when the bugs are fixed.

Loading note (temporary shim)
-----------------------------
`compare_v7_thresholds.py` executes its analysis driver (reads
`step3_results.parquet`, writes CSV/PNG) at *import* time, so it cannot be
imported directly in a test.  Until that driver is guarded under
`if __name__ == "__main__":` (issue #12), we extract just the two pure
functions via the AST below.  After that refactor, replace `_load_pure()` with:

    from scripts.compare_v7_thresholds import compute_metrics, get_split_mask

Run with:
    pytest tests/ -v
"""
import ast
import pathlib

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")  # compute_metrics uses matthews_corrcoef

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "compare_v7_thresholds.py"


def _load_pure():
    """Exec only the pure functions + the constants they need, skipping the
    module-level analysis driver.  TEMPORARY — see module docstring."""
    tree = ast.parse(SRC.read_text())
    want_funcs = {"compute_metrics", "get_split_mask"}
    want_consts = {"OUTLIER_THR", "TRAINED_ON"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_funcs:
            body.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            k in ast.dump(node) for k in ("numpy", "pandas", "sklearn")
        ):
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in want_consts for t in node.targets
        ):
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    ns: dict = {}
    exec(compile(module, str(SRC), "exec"), ns)  # noqa: S102 - trusted local source
    return ns


_NS = _load_pure()
compute_metrics = _NS["compute_metrics"]
get_split_mask = _NS["get_split_mask"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _p_only(p_prob, p_res):
    """Build a minimal benchmark slice with only P arrivals in window."""
    n = len(p_prob)
    return pd.DataFrame(
        {
            "p_in_window": np.zeros(n),          # >= 0  => P is in window
            "s_in_window": -np.ones(n),          # <  0  => no S in window
            "p_prob": np.asarray(p_prob, float),
            "s_prob": np.zeros(n),
            "p_residual_s": np.asarray(p_res, float),
            "s_residual_s": np.full(n, np.nan),
            "dist_bin": ["local (<150km)"] * n,
            "weight": ["test"] * n,
        }
    )


# ── compute_metrics: correct math (regression guards) ─────────────────────────

def test_p_recall_is_fraction_at_or_above_threshold():
    df = _p_only(p_prob=[0.90, 0.50, 0.20, 0.05], p_res=[0.0, 0.0, 0.0, 0.0])
    m = compute_metrics(df, thr_p=0.30, thr_s=0.30)
    assert m["p_recall"] == 0.5          # 2 of 4 have p_prob >= 0.30


def test_p_mae_and_outlier_fraction():
    df = _p_only(p_prob=[0.9, 0.9, 0.9, 0.9], p_res=[0.1, -0.2, 2.0, -3.0])
    m = compute_metrics(df, thr_p=0.30, thr_s=0.30)
    assert m["p_mae_s"] == pytest.approx(1.325, abs=1e-6)   # mean(|res|)
    assert m["p_outlier"] == pytest.approx(0.5)             # 2 of 4 exceed 1.5 s


def test_n_traces_counts_the_whole_slice():
    df = _p_only(p_prob=[0.9, 0.1], p_res=[0.0, 0.0])
    assert compute_metrics(df, 0.3, 0.3)["n_traces"] == 2


# ── compute_metrics: DOCUMENTS the unconditional-MAE bug (issue #8) ────────────

def test_mae_is_currently_unconditional_and_threshold_independent():
    """MAE averages over ALL in-window traces, including undetected ones, so it
    does not change with the detection threshold.  This is the audit's core
    complaint (issue #8).  When MAE is switched to detected-only, FLIP this to
    `assert hi["p_mae_s"] != lo["p_mae_s"]`.
    """
    df = _p_only(p_prob=[0.90, 0.05], p_res=[0.1, 4.0])
    hi = compute_metrics(df, thr_p=0.90, thr_s=0.90)   # only 1 "detected"
    lo = compute_metrics(df, thr_p=0.01, thr_s=0.01)   # both "detected"
    assert hi["p_mae_s"] == lo["p_mae_s"] == pytest.approx(2.05)


# ── get_split_mask: correct behavior + DOCUMENTS the no-op bug (issue #7) ──────

def test_get_split_mask_excludes_in_domain_rows_for_a_known_weight():
    wdf = pd.DataFrame({"trained_models": ["stead,ethz", "instance", "geofon"]})
    mask = get_split_mask(wdf, "stead")     # TRAINED_ON["stead"] == "stead"
    assert list(mask) == [False, True, True]  # row 0 trained on stead -> in-domain


def test_get_split_mask_is_a_noop_for_finetuned_weights():
    """`jma_wc_ft_*` weights are absent from TRAINED_ON, so EVERY row is labelled
    cross-domain regardless of provenance -> cross_domain == all (issue #7).
    When #7 is fixed so fine-tuned models exclude their own training datasets,
    replace this with an assertion that in-domain rows are masked out.
    """
    wdf = pd.DataFrame({"trained_models": ["stead", "instance", "pnw"]})
    mask = get_split_mask(wdf, "jma_wc_ft_global_v7")
    assert mask.all()   # current (buggy) behavior: nothing is excluded
