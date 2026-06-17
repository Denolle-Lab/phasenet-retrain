"""
fast_manifest_dataset.py

CachedManifestDataset: pre-loads ALL waveforms and labels into RAM on first
use, then serves them from numpy arrays — zero I/O overhead during training.

Pre-loading uses a DataLoader with multiple workers to parallelise the initial
HDF5 reads.  On a 186k-trace training set this takes ~2–3 minutes once; every
subsequent epoch is GPU-bound rather than I/O-bound.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from manifest_dataset import ManifestDataset, WINDOW_LEN


class CachedManifestDataset(Dataset):
    """
    Thin wrapper around ManifestDataset that pre-loads all samples into RAM.

    Parameters
    ----------
    manifest_csv      : path to train/val/test CSV
    augment           : apply amplitude jitter + polarity flip during __getitem__
    window_len        : waveform window in samples (default 3001)
    load_workers      : worker count for the initial parallel HDF5 extraction
    load_batch        : batch size used during extraction (larger = faster)
    noise_prob        : fraction of training examples to inject additive noise
    noise_snr_db_range: [low, high] SNR range in dB for noise injection;
                        lower = harder (0 dB = noise power equals signal power)
    """

    def __init__(
        self,
        manifest_csv: str,
        augment: bool = False,
        window_len: int = WINDOW_LEN,
        load_workers: int = 8,
        load_batch: int = 512,
        noise_prob: float = 0.0,
        noise_snr_db_range: tuple = (0, 10),
    ):
        self.augment           = augment
        self.window_len        = window_len
        self.noise_prob        = noise_prob
        self.noise_snr_db_low  = noise_snr_db_range[0]
        self.noise_snr_db_high = noise_snr_db_range[1]

        # ── extract all samples from HDF5 into RAM ────────────────────────────
        raw = ManifestDataset(manifest_csv, augment=False, window_len=window_len)
        N   = len(raw)
        print(f"  Pre-loading {N:,} samples into RAM "
              f"({N * 3 * window_len * 4 * 2 / 1e9:.1f} GB) ...")

        waveforms = np.empty((N, 3, window_len), dtype=np.float32)
        labels    = np.empty((N, 3, window_len), dtype=np.float32)

        loader = DataLoader(
            raw,
            batch_size        = load_batch,
            num_workers       = load_workers,
            shuffle           = False,
            pin_memory        = False,
            persistent_workers= False,
            prefetch_factor   = 2,
        )

        idx = 0
        for x_batch, y_batch in loader:
            b = x_batch.shape[0]
            waveforms[idx: idx + b] = x_batch.numpy()
            labels   [idx: idx + b] = y_batch.numpy()
            idx += b
            # simple text progress every ~10 %
            if idx % max(1, N // 10) < load_batch:
                pct = 100 * idx / N
                print(f"    {pct:5.1f}%  ({idx:,}/{N:,})", flush=True)

        print(f"  Done — {N:,} samples in RAM")

        self._waveforms = waveforms
        self._labels    = labels

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._waveforms)

    def __getitem__(self, idx):
        x = self._waveforms[idx].copy()
        y = self._labels[idx].copy()

        if self.augment:
            # amplitude jitter and polarity flip (existing)
            x *= np.random.uniform(0.5, 2.0)
            if np.random.random() < 0.1:
                x = -x

            # hard-noise injection: teach the model to detect picks at low SNR.
            # Waveforms are already unit-std normalised so we can treat their
            # per-component RMS ≈ 1.  A noise_std of σ gives a per-component
            # SNR ≈ 1/σ² (linear), i.e. SNR_dB ≈ -20*log10(σ).
            if self.noise_prob > 0 and np.random.random() < self.noise_prob:
                snr_db  = np.random.uniform(self.noise_snr_db_low,
                                             self.noise_snr_db_high)
                noise_std = 10 ** (-snr_db / 20.0)
                x = x + np.random.randn(*x.shape).astype(np.float32) * noise_std

        return torch.from_numpy(x), torch.from_numpy(y)
