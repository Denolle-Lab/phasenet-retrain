"""
scratch_model.py

PhaseNetScratch — PhaseNet trained from random initialisation.

Key differences from PhaseNetFinetune (fine_tune_model.py):
  * No pretrained weights — sbm.PhaseNet() with random init
  * Soft cross-entropy loss on the full Gaussian label distribution
    (hard-argmax CE throws away timing information; soft CE preserves it)
  * No knowledge distillation, no timing_beta auxiliary loss
  * AdamW + CosineAnnealingWarmup by default (better for from-scratch)
  * Class-weighted loss for P/S emphasis (configurable)
"""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import seisbench.models as sbm

# Reuse pick-residual metric and CSV logger from the fine-tune module
from fine_tune_model import MetricsLogger, pick_residuals   # noqa: F401 (re-exported)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class PhaseNetScratch(nn.Module):
    """
    PhaseNet with random weight initialisation.

    Loss: soft cross-entropy
        L = -Σ_{t,c} w_c · y_{t,c} · log p_{t,c}
    where y is the Gaussian soft-label distribution (PSN ordering) and
    w_c is the per-class weight (upweights P and S samples vs noise).

    Using the full soft labels — rather than hard argmax — means the model
    receives a gradient that is proportional to distance from the true pick,
    which drives better timing accuracy.
    """

    def __init__(self, config: dict):
        super().__init__()
        training_cfg = config.get("training", {})

        print("Initialising PhaseNet from scratch (random weights)")
        self.model = sbm.PhaseNet(
            in_channels=3, classes=3, phases="PSN", sampling_rate=100,
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters : {n_params:,}")

        self.learning_rate = training_cfg.get("learning_rate", 1e-3)

        cw_list = training_cfg.get("class_weights", None)
        if cw_list is not None:
            cw = torch.tensor(cw_list, dtype=torch.float32)
            print(f"  Class weights : P={cw[0]:.1f}  S={cw[1]:.1f}  N={cw[2]:.1f}")
        else:
            cw = None
        self.register_buffer("class_weight", cw)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        return self.model(x, logits=logits)

    def compute_loss_and_metrics(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass + soft cross-entropy loss + per-phase accuracy metrics.

        Parameters
        ----------
        x : (B, 3, T)  normalised waveform
        y : (B, 3, T)  Gaussian soft labels  [P, S, N]  (PSN ordering)

        Returns
        -------
        metrics : dict with 'loss', 'acc', 'P_acc', 'S_acc', 'N_acc'
        probs   : (B, 3, T) softmax probabilities
        """
        logits = self(x, logits=True)           # (B, 3, T)
        probs  = F.softmax(logits, dim=1)

        # Flatten spatial dimension: (B, 3, T) → (B*T, 3)
        y_flat      = y.permute(0, 2, 1).reshape(-1, 3)
        logits_flat = logits.permute(0, 2, 1).reshape(-1, 3)

        # ── soft cross-entropy: -Σ p_true · log(p_pred) ──────────────────────
        log_p = F.log_softmax(logits_flat, dim=1)          # (B*T, 3)
        if self.class_weight is not None:
            weighted_y = y_flat * self.class_weight.unsqueeze(0)
            loss = -(weighted_y * log_p).sum(dim=1).mean()
        else:
            loss = -(y_flat * log_p).sum(dim=1).mean()

        # ── monitoring accuracy (hard prediction vs hard label) ───────────────
        y_cls    = y_flat.argmax(dim=1)
        pred_cls = logits_flat.argmax(dim=1)
        acc      = (pred_cls == y_cls).float().mean()

        metrics  = {"loss": loss, "acc": acc}

        true_cls  = y.argmax(dim=1).reshape(-1)
        pred_flat = probs.permute(0, 2, 1).reshape(-1, 3).argmax(dim=1)
        for idx, name in enumerate(["P", "S", "N"]):
            mask = true_cls == idx
            if mask.any():
                metrics[f"{name}_acc"] = (pred_flat[mask] == true_cls[mask]).float().mean()

        return metrics, probs

    # ──────────────────────────────────────────────────────────────────────────

    def build_optimiser(self, config: dict):
        """
        Build AdamW optimiser + scheduler from config.

        Supports:
          CosineAnnealingWarmup  (default — best for from-scratch training)
          ReduceLROnPlateau      (fallback if name is anything else)
        """
        training_cfg  = config.get("training", {})
        scheduler_cfg = training_cfg.get("scheduler", {})

        weight_decay = training_cfg.get("weight_decay", 1e-4)
        trainable    = [p for p in self.parameters() if p.requires_grad]
        opt          = torch.optim.AdamW(
            trainable, lr=self.learning_rate, weight_decay=weight_decay,
        )

        sched_name    = scheduler_cfg.get("name", "CosineAnnealingWarmup")
        max_epochs    = training_cfg.get("max_epochs", 300)
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 10)
        min_lr        = scheduler_cfg.get("min_lr", 1e-6)
        start_factor  = scheduler_cfg.get("warmup_start_factor", 0.01)

        if sched_name == "CosineAnnealingWarmup":
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=start_factor, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, max_epochs - warmup_epochs), eta_min=min_lr,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
            print(f"  Scheduler : CosineAnnealingWarmup  "
                  f"warmup={warmup_epochs}  total={max_epochs}  min_lr={min_lr:.0e}")
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min",
                factor  = scheduler_cfg.get("factor",   0.5),
                patience= scheduler_cfg.get("patience",  5),
                min_lr  = min_lr,
            )
            print(f"  Scheduler : ReduceLROnPlateau  patience={scheduler_cfg.get('patience', 5)}")

        return opt, scheduler
