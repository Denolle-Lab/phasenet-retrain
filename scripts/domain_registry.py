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

  - Our own fine-tunes (jma_wc, jma_wc_ft_*): each was trained on its own
    manifest (data/manifests*/train*.csv — NOT uniformly data/manifests/,
    see WEIGHT_MANIFESTS below), assembled from nearly every dataset the
    benchmark also draws from. `trained_models` is meaningless here (it only
    describes public pretrained corpora, not our manifest composition), so
    instead we check the benchmark trace's own `dataset` column against the
    actual set of datasets that specific weight's manifest was built from.

Consequence: cross_domain will come out near-empty for jma_wc*/jma_wc_ft_*
under this corrected logic, since our training manifests already draw from
every benchmark dataset. That is the mathematically honest answer given how
the manifests were assembled — it is not a bug. Genuine train/benchmark
independence for these models has to be established at the event level
(see scripts/audit_event_leakage.py), not the dataset-name level.

Per-weight manifest resolution (WEIGHT_MANIFESTS): different fine-tune runs
used materially different manifests — e.g. jma_wc_ft_global_v7 trained on
data/manifests_v2/train.csv (527k rows) while data/manifests/train.csv (the
old hardcoded default, 371k rows) omits 3 datasets v2 includes (obs,
meier2019jgr, ross2018gpd) and adds one v2 doesn't (noise_prephase). Checking
every weight against the wrong manifest silently under/over-counts both
in_domain datasets and clean_holdout leakage for that specific weight. The
mapping below is parsed directly from configs/finetune_jma_wc*.yaml — the
actual source of truth each training run read — rather than a second
hand-maintained table that can drift out of sync with it.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = REPO_ROOT / "configs"

# Legacy default — used only as a fallback for own-model weights with no
# matching finetune config (e.g. the bare pretrained "jma_wc" parent
# checkpoint, which isn't itself a product of any finetune_*.yaml run).
TRAIN_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "train.csv"
VAL_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "val.csv"

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
    # volpick (Zhong & Tan 2024): "vcseis" is the benchmark manifest's own
    # trained_models label AND, per the volpick repo's README, VCSEIS is
    # literally "a subset of the dataset in Zhong and Tan (2024), with the
    # data from Japan excluded" -- i.e. volpick's own non-Japan training
    # data, already downloaded locally. split_masks() excludes the 200
    # vcseis-labeled rows via this; scripts/audit_parent_leakage.py
    # additionally spatiotemporal-audits VCSEIS's full corpus against all
    # 12 benchmark datasets for parent_clean_cross_domain_mask() (see
    # PARENT_DOMAINS["volpick"] there). Only gap: volpick's Japan-region
    # training slice isn't available locally, so overlap specific to that
    # portion remains unverified.
    "volpick":                   "volpick",
    # Unknown / undocumented public training corpus -> treated as fully
    # cross_domain (conservative default, unchanged from prior behavior).
    "diting":                    None,
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


# Public alias — used outside this module to gate clean_holdout_mask() to
# the models the row-leak mask was actually computed against.
is_own_model = _is_own_model


def _weight_name_for_config_stem(stem: str):
    """finetune_jma_wc.yaml -> jma_wc_ft ; finetune_jma_wc_global_v7.yaml ->
    jma_wc_ft_global_v7. Returns None for configs outside this naming scheme
    (e.g. phasenet_scratch_v1.yaml — from-scratch training, not a fine-tune,
    and not matched by OWN_MODEL_RE either)."""
    prefix = "finetune_jma_wc"
    if not stem.startswith(prefix):
        return None
    return "jma_wc_ft" + stem[len(prefix):]


@lru_cache(maxsize=None)
def _load_weight_manifests() -> dict:
    registry = {}
    for cfg_path in sorted(CONFIGS_DIR.glob("finetune_jma_wc*.yaml")):
        weight_name = _weight_name_for_config_stem(cfg_path.stem)
        if weight_name is None:
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        data_cfg = cfg.get("data", {})
        train = data_cfg.get("train_manifest")
        val = data_cfg.get("val_manifest")
        if not train or not val:
            continue
        registry[weight_name] = {
            "train": (REPO_ROOT / train).resolve(),
            "val": (REPO_ROOT / val).resolve(),
        }
    return registry


# weight_name -> {"train": Path, "val": Path}, parsed from configs/*.yaml —
# the same files each training run actually read, so this can't drift out
# of sync with them the way a hand-maintained copy could.
WEIGHT_MANIFESTS = _load_weight_manifests()


def resolve_own_manifests(weight_name: str):
    """(train_path, val_path) for an own-model weight. Falls back to the
    legacy data/manifests/ pair for weights with no matching finetune
    config (currently just the bare pretrained "jma_wc" parent)."""
    paths = WEIGHT_MANIFESTS.get(weight_name)
    if paths:
        return paths["train"], paths["val"]
    return TRAIN_MANIFEST_PATH, VAL_MANIFEST_PATH


def family_key_for_manifest(train_path) -> str:
    """Stable filename-safe id for a train manifest, used to key per-family
    row-leak mask CSVs. e.g. data/manifests_v2/train_tele2x.csv ->
    'data__manifests_v2__train_tele2x'."""
    rel = Path(train_path).resolve().relative_to(REPO_ROOT).with_suffix("")
    return str(rel).replace("/", "__")


@lru_cache(maxsize=None)
def _load_trained_datasets(train_path) -> frozenset:
    train_path = Path(train_path)
    if not train_path.exists():
        return frozenset()
    ds = pd.read_csv(train_path, usecols=["dataset_name"], low_memory=False)
    return frozenset(ds["dataset_name"].unique())


def _resolve_public_trained_on(weight_name: str):
    if weight_name in COMPOSITE_TRAINED_ON:
        return COMPOSITE_TRAINED_ON[weight_name]
    base = weight_name[4:] if weight_name.startswith("eqt_") else weight_name
    return BASE_TRAINED_ON.get(base)


def split_masks(wdf: pd.DataFrame, weight_name: str):
    """Return (in_domain_mask, cross_domain_mask) for a weight's result rows."""
    if _is_own_model(weight_name):
        train_path, _ = resolve_own_manifests(weight_name)
        own_datasets = _load_trained_datasets(train_path)
        in_mask = wdf["dataset"].isin(own_datasets)
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


def _row_leak_csv_path(train_path) -> Path:
    return REPO_ROOT / "results" / f"event_leakage_row_mask__{family_key_for_manifest(train_path)}.csv"


@lru_cache(maxsize=None)
def _load_row_leak_df(row_mask_path) -> pd.DataFrame:
    row_mask_path = Path(row_mask_path)
    if not row_mask_path.exists():
        return pd.DataFrame(columns=["dataset", "trace_name", "verifiable", "leaked"])
    df = pd.read_csv(row_mask_path)
    # A handful of (dataset, trace_name) pairs repeat verbatim in
    # notebooks/benchmark_manifest.csv (mlaapde/aq2009gm chunked metadata);
    # leaked/verifiable are identical across those repeats, so dedup here
    # rather than fanning out matches in the merge below.
    return df.drop_duplicates(subset=["dataset", "trace_name"])


def clean_holdout_mask(wdf: pd.DataFrame, weight_name: str) -> pd.Series:
    """Event-level holdout mask: True where the trace's earthquake never
    appeared in the train/val manifest THIS SPECIFIC WEIGHT was trained on
    (see resolve_own_manifests() / scripts/audit_event_leakage.py). Only
    defined for our own fine-tunes; returns all-False for public pretrained
    weights, whose actual training split isn't verifiable this way. Rows
    with no verifiable event id (obst2024) are excluded, not assumed clean.
    """
    if not _is_own_model(weight_name):
        return pd.Series(False, index=wdf.index)

    train_path, _ = resolve_own_manifests(weight_name)
    row_leak_df = _load_row_leak_df(_row_leak_csv_path(train_path))
    if row_leak_df.empty:
        return pd.Series(False, index=wdf.index)

    merged = wdf[["dataset", "trace_name"]].merge(
        row_leak_df, on=["dataset", "trace_name"], how="left"
    )
    clean = merged["verifiable"].fillna(False) & ~merged["leaked"].fillna(True)
    clean.index = wdf.index
    return clean


@lru_cache(maxsize=None)
def _load_parent_row_leak_df(trained_on: str) -> pd.DataFrame:
    path = REPO_ROOT / "results" / f"parent_event_leakage_row_mask__{trained_on}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["dataset", "trace_name", "verifiable", "leaked"])
    df = pd.read_csv(path)
    return df.drop_duplicates(subset=["dataset", "trace_name"])


def parent_clean_cross_domain_mask(wdf: pd.DataFrame, weight_name: str):
    """Cross-domain mask for a PUBLIC pretrained weight, additionally
    excluding benchmark rows whose earthquake also appears — via
    spatiotemporal match, not dataset-name match — anywhere in that weight's
    full public training corpus (see scripts/audit_parent_leakage.py). This
    catches leakage split_masks() can't: an event nominally in an
    "independent" benchmark dataset (e.g. cwa) that's actually the same
    earthquake as one in the model's own training corpus (e.g. stead),
    recorded at a different station.

    Returns None if the weight is one of our own fine-tunes (use
    clean_holdout_mask instead) or if its trained_on corpus isn't locally
    verifiable this way (unknown corpus, or corpus unavailable locally —
    e.g. neic, obst2024; see audit_parent_leakage.py's UNVERIFIABLE_DOMAINS).
    """
    if _is_own_model(weight_name):
        return None
    trained_on = _resolve_public_trained_on(weight_name)
    if not trained_on:
        return None
    row_leak_df = _load_parent_row_leak_df(trained_on)
    if row_leak_df.empty:
        return None

    _, cross_mask = split_masks(wdf, weight_name)
    merged = wdf[["dataset", "trace_name"]].merge(
        row_leak_df, on=["dataset", "trace_name"], how="left"
    )
    clean = merged["verifiable"].fillna(False) & ~merged["leaked"].fillna(True)
    clean.index = wdf.index
    return cross_mask & clean


if __name__ == "__main__":
    print(f"WEIGHT_MANIFESTS ({len(WEIGHT_MANIFESTS)} weights resolved from configs/):")
    for weight, paths in sorted(WEIGHT_MANIFESTS.items()):
        print(f"  {weight:28s} train={paths['train'].relative_to(REPO_ROOT)}")
    print(f"\nBASE_TRAINED_ON entries: {len(BASE_TRAINED_ON)}")
    print(f"COMPOSITE_TRAINED_ON entries: {len(COMPOSITE_TRAINED_ON)}")
