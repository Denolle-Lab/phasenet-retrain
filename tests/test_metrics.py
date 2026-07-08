"""
Starter unit tests for the benchmark-metric + domain-split helpers.

Coverage targets (flagged as untested by the code-review-graph analysis):
  * scripts/domain_registry.py :: split_masks   — in_domain / cross_domain split
  * scripts/compare_v7_thresholds.py :: compute_metrics — recall / MAE / outlier

Split logic (domain_registry)
-----------------------------
The cross-domain split no-op has been FIXED (commit 244473f, closes #7): the
logic now lives in `domain_registry.split_masks()`. For public pretrained
weights it compares the benchmark trace's `trained_models` column to that
weight's known corpus; for our own fine-tunes (`jma_wc*` / `jma_wc_ft_*`) it
compares the trace's `dataset` column to the set of datasets in the specific
train manifest that weight was fine-tuned on (each own-model weight resolves
its own manifest via `domain_registry.WEIGHT_MANIFESTS`, since different
weights train on different manifest families). These tests are the regression
guard for that fix. `domain_registry` is cleanly importable, so they use a
normal import.

Metric math (compute_metrics)
-----------------------------
`compare_v7_thresholds.py` still runs its analysis driver (reads
`step3_results.parquet`, writes CSV/PNG) at *import* time, so it can't be
imported in a test. Until that driver is guarded under
`if __name__ == "__main__":` (issue #12), we extract just `compute_metrics` via
the AST shim below; afterwards, replace it with a plain
`from scripts.compare_v7_thresholds import compute_metrics`.

`compute_metrics` computes MAE/outlier UNCONDITIONALLY (over all in-window
traces, including undetected ones), which the audit flags as not
Münchmeyer-comparable (issue #8). One test below documents that behavior and
should be flipped when MAE becomes detected-only.

Run with:  pytest tests/ -v
"""
import ast
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = SCRIPTS / "compare_v7_thresholds.py"
sys.path.insert(0, str(SCRIPTS))

import domain_registry  # noqa: E402  (cleanly importable — no import-time driver)


# ── AST shim: load only compute_metrics, skipping the import-time driver ──────

def _load_compute_metrics():
    """Exec only `compute_metrics` (+ OUTLIER_THR and its numpy/pandas/sklearn
    imports), skipping the module-level analysis driver. TEMPORARY — remove once
    the driver is guarded under `if __name__ == "__main__":` (issue #12)."""
    pytest.importorskip("sklearn")  # matthews_corrcoef is imported at module top
    tree = ast.parse(SRC.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "compute_metrics":
            body.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            k in ast.dump(node) for k in ("numpy", "pandas", "sklearn")
        ):
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "OUTLIER_THR" for t in node.targets
        ):
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    ns: dict = {}
    exec(compile(module, str(SRC), "exec"), ns)  # noqa: S102 - trusted local source
    return ns["compute_metrics"]


compute_metrics = _load_compute_metrics()


# ── helpers ───────────────────────────────────────────────────────────────────

def _p_only(p_prob, p_res):
    """Minimal benchmark slice with only P arrivals in window."""
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


def _wdf(**cols):
    return pd.DataFrame(cols)


# ══ compute_metrics: correct math (regression guards) ═════════════════════════

def test_p_recall_is_fraction_at_or_above_threshold():
    df = _p_only(p_prob=[0.90, 0.50, 0.20, 0.05], p_res=[0.0, 0.0, 0.0, 0.0])
    assert compute_metrics(df, thr_p=0.30, thr_s=0.30)["p_recall"] == 0.5


def test_p_mae_and_outlier_fraction():
    df = _p_only(p_prob=[0.9, 0.9, 0.9, 0.9], p_res=[0.1, -0.2, 2.0, -3.0])
    m = compute_metrics(df, thr_p=0.30, thr_s=0.30)
    assert m["p_mae_s"] == pytest.approx(1.325, abs=1e-6)   # mean(|res|)
    assert m["p_outlier"] == pytest.approx(0.5)             # 2 of 4 exceed 1.5 s


def test_n_traces_counts_the_whole_slice():
    df = _p_only(p_prob=[0.9, 0.1], p_res=[0.0, 0.0])
    assert compute_metrics(df, 0.3, 0.3)["n_traces"] == 2


def test_mae_is_currently_unconditional_and_threshold_independent():
    """MAE averages over ALL in-window traces, incl. undetected ones, so it does
    not change with the detection threshold (issue #8). When MAE becomes
    detected-only, FLIP this to `assert hi["p_mae_s"] != lo["p_mae_s"]`."""
    df = _p_only(p_prob=[0.90, 0.05], p_res=[0.1, 4.0])
    hi = compute_metrics(df, thr_p=0.90, thr_s=0.90)   # only 1 "detected"
    lo = compute_metrics(df, thr_p=0.01, thr_s=0.01)   # both "detected"
    assert hi["p_mae_s"] == lo["p_mae_s"] == pytest.approx(2.05)


# ══ domain_registry.split_masks: regression guard for the #7 fix ══════════════

def test_public_weight_with_known_corpus_marks_indomain():
    wdf = _wdf(trained_models=["stead,ethz", "instance", "geofon"])
    in_mask, cross_mask = domain_registry.split_masks(wdf, "stead")
    assert list(in_mask) == [True, False, False]     # row 0 trained on stead
    assert list(cross_mask) == [False, True, True]


def test_eqt_prefix_resolves_to_base_corpus():
    wdf = _wdf(trained_models=["scedc", "stead"])
    in_mask, _ = domain_registry.split_masks(wdf, "eqt_scedc")
    assert list(in_mask) == [True, False]


def test_public_weight_with_unknown_corpus_is_all_cross_domain():
    wdf = _wdf(trained_models=["stead", "instance"])
    in_mask, cross_mask = domain_registry.split_masks(wdf, "geofon")  # corpus None
    assert not in_mask.any()
    assert cross_mask.all()


def test_own_model_splits_on_dataset_not_trained_models(monkeypatch):
    """The #7 fix: our fine-tunes split on the benchmark trace's `dataset`
    column vs the manifest's datasets — NOT the old all-True no-op. Each
    own-model weight now resolves its OWN train manifest (parsed from its
    finetune config, since different weights train on different manifest
    families — see domain_registry.WEIGHT_MANIFESTS); monkeypatch
    `_load_trained_datasets` (rather than a single global set) so the test
    is deterministic on any machine regardless of which manifest a given
    weight resolves to."""
    monkeypatch.setattr(
        domain_registry,
        "_load_trained_datasets",
        lambda train_path: frozenset({"stead", "instance"}),
    )
    wdf = _wdf(dataset=["stead", "pnw", "instance"], trained_models=["x", "y", "z"])
    in_mask, cross_mask = domain_registry.split_masks(wdf, "jma_wc_ft_global_v7")
    assert list(in_mask) == [True, False, True]      # in the manifest -> in-domain
    assert list(cross_mask) == [False, True, False]  # not a no-op anymore


def test_own_model_requires_a_dataset_column():
    """Own-model split needs a `dataset` column (the contract Copilot flagged);
    calling without it is a KeyError the caller must satisfy."""
    with pytest.raises(KeyError):
        domain_registry.split_masks(_wdf(trained_models=["stead"]), "jma_wc")
