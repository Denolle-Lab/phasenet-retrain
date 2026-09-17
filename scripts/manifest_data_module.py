"""
manifest_data_module.py

DataLoaders for fine-tuning built from manifest CSVs.
Uses CachedManifestDataset: pre-loads all waveforms into RAM once, then every
epoch is served from memory with zero HDF5 I/O.

`data.label_policy` (#41A) selects the target/mask policy; when it is set
(to "legacy", "masked" or a dict) every loader yields (x, y, mask) so the
loss can be restricted to supervised samples. Unset keeps the historical
(x, y) API and targets.

`data.norm` (2026-09-18) is the window normalisation, "std" or "peak"
(manifest_dataset.NORMS). When absent it follows the parent weights'
SeisBench norm (`model_args.norm` of `model.pretrained.model_name`, read
through seisbench.models.PhaseNet.from_pretrained(name).norm), so training
sees what annotate() will see: `jma_wc` is std, `instance` is peak.
resolve_norm returns the value and its origin for the run card.
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_manifest_dataset import CachedManifestDataset
from waveform_contract import NORMS


def parent_norm(model_name: str) -> str:
    """The SeisBench norm of the cached parent weights (`model_args.norm`)."""
    import seisbench.models as sbm
    norm = getattr(sbm.PhaseNet.from_pretrained(model_name), "norm", None)
    if norm not in NORMS:
        raise ValueError(f"parent {model_name!r} declares norm {norm!r}; the loader knows {NORMS}")
    return norm


def resolve_norm(config: dict):
    """(norm, source): `data.norm` from the config ("config"), else the parent
    weights' norm ("parent <name>"). An unknown value is refused."""
    data_cfg = config.get("data", {}) or {}
    norm = data_cfg.get("norm")
    if norm is not None:
        if norm not in NORMS:
            raise ValueError(f"data.norm {norm!r} is not one of {NORMS}")
        return norm, "config"
    model_name = ((config.get("model", {}) or {}).get("pretrained", {}) or {}).get("model_name", "jma_wc")
    return parent_norm(model_name), f"parent {model_name}"

SPLITS = ("train", "val", "test")


def build_dataloaders(config: dict, splits=SPLITS):
    """
    Build and return (train_loader, val_loader, test_loader); entries for
    splits not requested are None. Only the requested splits are pre-loaded
    into RAM (training needs train and val; `--test` loads test alone).
    """
    data_cfg    = config.get("data", {})
    train_cfg   = config.get("training", {})
    batch_size  = train_cfg.get("batch_size", 1024)
    # With in-memory data, num_workers=0 is fastest (no IPC overhead).
    # Use workers only for the initial parallel HDF5 extraction.
    num_workers = 0
    window_len  = data_cfg.get("window_length", 3001)
    load_workers = train_cfg.get("num_workers", 8)

    aug_cfg   = data_cfg.get("augmentation", {})
    noise_prob        = aug_cfg.get("noise_prob", 0.0)
    noise_snr_db_range = aug_cfg.get("noise_snr_db_range", [0, 10])
    label_policy = data_cfg.get("label_policy", None)
    return_mask  = label_policy is not None
    norm, norm_source = resolve_norm(config)
    print(f"Window normalisation: {norm} ({norm_source})")

    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown splits {sorted(unknown)}; expected a subset of {SPLITS}")

    datasets = {}
    for split in SPLITS:
        if split not in splits:
            continue
        print(f"Pre-loading {split} set:")
        extra = dict(noise_prob=noise_prob, noise_snr_db_range=noise_snr_db_range) if split == "train" else {}
        datasets[split] = CachedManifestDataset(
            data_cfg[f"{split}_manifest"], augment=(split == "train"),
            window_len=window_len, load_workers=load_workers,
            label_policy=label_policy, return_mask=return_mask, norm=norm, **extra,
        )

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,   # 0 = main process; no IPC for in-memory data
        pin_memory  = True,          # async CPU→GPU transfer
    )
    loaders = {split: DataLoader(ds, shuffle=(split == "train"), **loader_kwargs)
               for split, ds in datasets.items()}

    for split, ds in datasets.items():
        print(f"  {split:<5} : {len(ds):>7,} traces  ({len(loaders[split]):,} batches @ {batch_size})"
              + (f"  supervised samples {ds.n_supervised_samples:,}" if return_mask else ""))

    return loaders.get("train"), loaders.get("val"), loaders.get("test")
