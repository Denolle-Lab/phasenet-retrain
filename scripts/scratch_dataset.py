"""
scratch_dataset.py

ScratchDataset: CachedManifestDataset subclass with stronger augmentation
for training PhaseNet from random initialisation.

Augmentation stack applied during training:
  1. Log-uniform amplitude scaling  [0.05, 20]  — wider than fine-tune [0.5, 2]
  2. Polarity flip                  p=0.5       — more aggressive (was 0.1)
  3. Additive Gaussian noise        SNR 5–30 dB, applied 70 % of batches
  4. Single-channel masking         p=0.15      — simulates missing components
"""

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).parent.resolve()
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_manifest_dataset import CachedManifestDataset


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────────────────────────────────────

def _strong_augment(x: np.ndarray) -> np.ndarray:
    """
    Apply stronger on-the-fly augmentation to a (3, T) waveform (already
    normalised, loaded from RAM cache).  Returns modified copy.
    """
    # 1. Log-uniform amplitude scaling over a wider range than fine-tuning
    log_scale = np.random.uniform(np.log(0.05), np.log(20.0))
    x = x * np.exp(log_scale)

    # 2. Polarity flip (50 %)
    if np.random.random() < 0.5:
        x = -x

    # 3. Additive Gaussian noise at a random SNR
    if np.random.random() < 0.7:
        snr_db       = np.random.uniform(5.0, 30.0)
        sig_power    = np.mean(x ** 2) + 1e-10
        snr_linear   = 10.0 ** (snr_db / 10.0)
        noise_std    = np.sqrt(sig_power / snr_linear)
        x = x + (np.random.randn(*x.shape).astype(np.float32) * noise_std)

    # 4. Single-channel masking (simulate a dead or clipped component)
    if np.random.random() < 0.15:
        ch = np.random.randint(0, x.shape[0])
        x[ch] = 0.0

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ScratchDataset(CachedManifestDataset):
    """
    Drop-in replacement for CachedManifestDataset that applies _strong_augment
    instead of the default minimal augmentation.

    All pre-loading logic (parallel HDF5 extraction → numpy RAM arrays) is
    inherited unchanged — only __getitem__ is overridden.
    """

    def __getitem__(self, idx: int):
        x = self._waveforms[idx].copy()
        y = self._labels[idx].copy()

        if self.augment:
            x = _strong_augment(x)

        return torch.from_numpy(x), torch.from_numpy(y)
