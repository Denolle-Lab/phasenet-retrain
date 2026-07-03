#!/usr/bin/env python3
"""
domain_registry.py

Single source of truth for in_domain/cross_domain split logic across the
eval scripts. Previously this dict + mask logic was copy-pasted into 7
scripts and had drifted out of sync (e.g. pisdl trained_on was "pisdl" in
eval_finetuned.py but None in eval_ensemble.py; eqt_scedc trained_on was
"scedc" in eval_eqtransformer.py but None in eval_ensemble_eqt.py).

Two different notions of "trained_on" are handled here:

  - Public SeisBench pretrained weights (stead, instance, neic, ...): their
    training corpus is a fixed, publicly documented dataset. We compare the
    benchmark trace's `trained_models` column (a per-dataset label of which
    public corpora a trace's source dataset belongs to) against that weight's
    known corpus.

  - Our own fine-tunes (jma_wc, jma_wc_ft_*): trained on a custom manifest
    (data/manifests/train.csv) assembled from nearly every dataset the
    benchmark also draws from. `trained_models` is meaningless here (it only
    describes public pretrained corpora, not our manifest composition), so
    instead we check the benchmark trace's own `dataset` column against the
    actual set of datasets our manifest was built from.

Consequence: cross_domain will come out near-empty for jma_wc*/jma_wc_ft_*
under this corrected logic, since our training manifest already draws from
every benchmark dataset. That is the mathematically honest answer given how
the manifest was assembled — it is not a bug. Genuine train/benchmark
independence for these models has to be established at the event level
(see scripts/audit_event_leakage.py), not the dataset-name level.
"""

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
TRAIN_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "train.csv"

# ── Public pretrained weights: base name -> known training dataset ──────────
# (base name = weight string with any "eqt_" prefix stripped)
BASE_TRAINED_ON = {
    "stead":                     "stead",
    "instance":                  "instance",
    "neic":                      "neic",
    "scedc":                     "scedc",
    "ethz":                      "ethz",
    "iquique":                   "iquique",
    "obs":                       "obst2024",
    "pisdl":                     "pisdl",
    "original":                  "stead",
    "original_nonconservative":  "stead",
    # Unknown / undocumented public training corpus -> treated as fully
    # cross_domain (conservative default, unchanged from prior behavior).
    "diting":                    None,
    "volpick":                   None,
    "phasenet_sn":               None,
    "jma":                       None,
    "lendb":                     None,
    "geofon":                    None,
    "pnw":                       None,
}

# Special-cased composite weights whose trained_on doesn't reduce to a
# single base name via the "eqt_" prefix rule.
COMPOSITE_TRAINED_ON = {
    # original_nonconservative component was trained on STEAD.
    "eqt_ensemble_volpick_nc": "stead",
}

OWN_MODEL_RE = re.compile(r"^jma_wc($|_ft)")


def _is_own_model(weight_name: str) -> bool:
    return bool(OWN_MODEL_RE.match(weight_name))


def _load_own_trained_datasets() -> frozenset:
    if not TRAIN_MANIFEST_PATH.exists():
        return frozenset()
    ds = pd.read_csv(TRAIN_MANIFEST_PATH, usecols=["dataset_name"], low_memory=False)
    return frozenset(ds["dataset_name"].unique())


# Dataset names our own fine-tunes were actually trained on (gradient
# updates only — data/manifests/val.csv is confirmed early-stopping-only
# and never backpropagated, so it is intentionally excluded).
OWN_TRAINED_DATASETS = _load_own_trained_datasets()


def _resolve_public_trained_on(weight_name: str):
    if weight_name in COMPOSITE_TRAINED_ON:
        return COMPOSITE_TRAINED_ON[weight_name]
    base = weight_name[4:] if weight_name.startswith("eqt_") else weight_name
    return BASE_TRAINED_ON.get(base)


def split_masks(wdf: pd.DataFrame, weight_name: str):
    """Return (in_domain_mask, cross_domain_mask) for a weight's result rows."""
    if _is_own_model(weight_name):
        in_mask = wdf["dataset"].isin(OWN_TRAINED_DATASETS)
        cross_mask = ~in_mask
        return in_mask, cross_mask

    trained_on = _resolve_public_trained_on(weight_name)
    if trained_on:
        in_mask = wdf["trained_models"].str.contains(trained_on, na=False, regex=False)
        cross_mask = ~in_mask
    else:
        in_mask = pd.Series(False, index=wdf.index)
        cross_mask = pd.Series(True, index=wdf.index)
    return in_mask, cross_mask


if __name__ == "__main__":
    print(f"OWN_TRAINED_DATASETS ({len(OWN_TRAINED_DATASETS)}):")
    for name in sorted(OWN_TRAINED_DATASETS):
        print(f"  {name}")
    print(f"\nBASE_TRAINED_ON entries: {len(BASE_TRAINED_ON)}")
    print(f"COMPOSITE_TRAINED_ON entries: {len(COMPOSITE_TRAINED_ON)}")
