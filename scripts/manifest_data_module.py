"""
manifest_data_module.py

DataLoaders for fine-tuning built from manifest CSVs.
Uses CachedManifestDataset: pre-loads all waveforms into RAM once, then every
epoch is served from memory with zero HDF5 I/O.

`data.label_policy` (#41A) selects the target/mask policy; when it is set
(to "legacy", "masked" or a dict) every loader yields (x, y, mask) so the
loss can be restricted to supervised samples. Unset keeps the historical
(x, y) API and targets.
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_manifest_dataset import CachedManifestDataset

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
            label_policy=label_policy, return_mask=return_mask, **extra,
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
