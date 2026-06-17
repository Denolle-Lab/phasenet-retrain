"""
manifest_data_module.py

DataLoaders for fine-tuning built from manifest CSVs.
Uses CachedManifestDataset: pre-loads all waveforms into RAM once, then every
epoch is served from memory with zero HDF5 I/O.
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_manifest_dataset import CachedManifestDataset


def build_dataloaders(config: dict):
    """
    Build and return (train_loader, val_loader, test_loader).

    Pre-loads all three splits into RAM before returning.  Subsequent epochs
    incur no HDF5 I/O — the GPU will not be data-starved.
    """
    data_cfg    = config.get("data", {})
    train_cfg   = config.get("training", {})
    batch_size  = train_cfg.get("batch_size", 1024)
    # With in-memory data, num_workers=0 is fastest (no IPC overhead).
    # Use 4 workers only for the initial parallel HDF5 extraction.
    num_workers = 0
    window_len  = data_cfg.get("window_length", 3001)
    load_workers = train_cfg.get("num_workers", 8)

    aug_cfg   = data_cfg.get("augmentation", {})
    noise_prob        = aug_cfg.get("noise_prob", 0.0)
    noise_snr_db_range = aug_cfg.get("noise_snr_db_range", [0, 10])

    print("Pre-loading training set:")
    train_ds = CachedManifestDataset(
        data_cfg["train_manifest"], augment=True,
        window_len=window_len, load_workers=load_workers,
        noise_prob=noise_prob,
        noise_snr_db_range=noise_snr_db_range,
    )
    print("Pre-loading validation set:")
    val_ds = CachedManifestDataset(
        data_cfg["val_manifest"], augment=False,
        window_len=window_len, load_workers=load_workers,
    )
    print("Pre-loading test set:")
    test_ds = CachedManifestDataset(
        data_cfg["test_manifest"], augment=False,
        window_len=window_len, load_workers=load_workers,
    )

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,   # 0 = main process; no IPC for in-memory data
        pin_memory  = True,          # async CPU→GPU transfer
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"\n  Train : {len(train_ds):>7,} traces  ({len(train_loader):,} batches @ {batch_size})")
    print(f"  Val   : {len(val_ds):>7,} traces  ({len(val_loader):,} batches)")
    print(f"  Test  : {len(test_ds):>7,} traces  ({len(test_loader):,} batches)")

    return train_loader, val_loader, test_loader
