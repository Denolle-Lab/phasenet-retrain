"""
fine_tune_model.py

Trainable wrapper for SeisBench PhaseNet (jma_wc fine-tuning).
Pure PyTorch — no pytorch_lightning required.

Provides:
  PhaseNetFinetune   — model + loss + optimiser + metric helpers
  MetricsLogger      — writes per-epoch scalars to a CSV file
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seisbench.models as sbm


# ──────────────────────────────────────────────────────────────────────────────
# Pick-residual helper
# ──────────────────────────────────────────────────────────────────────────────

def pick_residuals(
    probs: torch.Tensor,
    labels: torch.Tensor,
    conf_thresh: float = 0.3,
    label_thresh: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute P and S pick residuals (samples) for a batch.

    Only records a residual when the label peak > label_thresh AND the
    predicted peak > conf_thresh.

    Parameters
    ----------
    probs  : (B, 3, T)  softmax probabilities [N, P, S]
    labels : (B, 3, T)  Gaussian soft labels  [N, P, S]

    Returns
    -------
    p_res, s_res : 1-D float tensors (may be empty)
    """
    true_p_conf = labels[:, 0].max(dim=1).values  # P=ch0 (PSN ordering)
    true_s_conf = labels[:, 1].max(dim=1).values  # S=ch1
    true_p_loc  = labels[:, 0].argmax(dim=1).float()
    true_s_loc  = labels[:, 1].argmax(dim=1).float()

    pred_p_conf = probs[:, 0].max(dim=1).values
    pred_s_conf = probs[:, 1].max(dim=1).values
    pred_p_loc  = probs[:, 0].argmax(dim=1).float()
    pred_s_loc  = probs[:, 1].argmax(dim=1).float()

    p_mask = (true_p_conf > label_thresh) & (pred_p_conf > conf_thresh)
    s_mask = (true_s_conf > label_thresh) & (pred_s_conf > conf_thresh)

    return (pred_p_loc - true_p_loc)[p_mask], (pred_s_loc - true_s_loc)[s_mask]


def soft_pick_mae(
    probs: torch.Tensor,
    labels: torch.Tensor,
    phase_idx: int = 0,
    label_thresh: float = 0.1,
) -> torch.Tensor:
    """
    Differentiable pick timing loss (seconds at 100 Hz) for one phase.

    Computes the soft argmax of the predicted phase channel (expected position
    under the normalised probability mass) vs the hard argmax of the label, then
    returns the mean absolute error in seconds.  Returns scalar zero when no
    picks are present in the batch.

    Parameters
    ----------
    probs      : (B, 3, T)  per-sample softmax probs  [P, S, N] (PSN ordering)
    labels     : (B, 3, T)  Gaussian soft labels
    phase_idx  : 0 = P, 1 = S
    """
    phase_label = labels[:, phase_idx]                              # (B, T)
    has_pick    = phase_label.max(dim=1).values > label_thresh      # (B,)
    if not has_pick.any():
        return probs.new_zeros(())
    T         = probs.shape[-1]
    true_loc  = phase_label[has_pick].argmax(dim=1).float()         # (N,)
    pred_ch   = probs[has_pick, phase_idx]                          # (N, T)
    pred_ch   = pred_ch / pred_ch.sum(dim=1, keepdim=True).clamp(min=1e-8)
    t_idx     = torch.arange(T, device=probs.device, dtype=probs.dtype)
    pred_loc  = (pred_ch * t_idx).sum(dim=1)                        # (N,) soft argmax
    return (pred_loc - true_loc).abs().mean() / 100.0               # samples → seconds


# ──────────────────────────────────────────────────────────────────────────────
# Metrics CSV logger
# ──────────────────────────────────────────────────────────────────────────────

class MetricsLogger:
    """Appends one row per epoch to a CSV file."""

    FIELDS = [
        "epoch",
        "train_loss", "train_acc", "train_grad_norm",
        "val_loss",   "val_acc",
        "val_N_acc",  "val_P_acc",  "val_S_acc",
        "val_p_mae_s", "val_s_mae_s",
        "lr",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.csv_path.exists()

    def log(self, row: dict):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow({k: row.get(k, "") for k in self.FIELDS})


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class PhaseNetFinetune(nn.Module):
    """
    Fine-tuning wrapper around SeisBench PhaseNet (jma_wc).

    Parameters
    ----------
    config : dict  — loaded from finetune_jma_wc.yaml
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg    = config.get("model", {})
        pretrained   = model_cfg.get("pretrained", {})
        training_cfg = config.get("training", {})

        model_name = pretrained.get("model_name", "jma_wc")
        print(f"Loading pretrained PhaseNet: {model_name}")
        self.model = sbm.PhaseNet.from_pretrained(model_name)

        for layer_name in pretrained.get("freeze_layers", []):
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = False
                    print(f"  frozen: {name}")

        self.learning_rate = training_cfg.get("learning_rate", 3e-4)

        cw_list = training_cfg.get("class_weights", None)
        if cw_list is not None:
            cw = torch.tensor(cw_list, dtype=torch.float32)
            print(f"  class weights: P={cw[0]:.1f}  S={cw[1]:.1f}  N={cw[2]:.1f}")
        else:
            cw = None
        self.register_buffer("class_weight", cw)  # auto-moves with .to(device)

        # Knowledge distillation: frozen teacher = original pretrained weights.
        # Penalises deviation from the teacher's predictions (prevents catastrophic forgetting).
        distill_cfg = training_cfg.get("distillation", {})
        self.distill_alpha = distill_cfg.get("alpha", 0.0)  # 0 = off
        self.distill_T     = distill_cfg.get("temperature", 4.0)
        if self.distill_alpha > 0:
            print(f"  knowledge distillation ON  alpha={self.distill_alpha}  T={self.distill_T}")
            self.teacher = sbm.PhaseNet.from_pretrained(model_name)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None

        self.timing_beta = training_cfg.get("timing_beta", 0.0)
        if self.timing_beta > 0:
            print(f"  timing loss     ON  beta={self.timing_beta}")

        # Focal loss: down-weights easy (already-correct) time steps so the
        # gradient concentrates on hard cases — weak teleseismic onsets, S picks.
        # gamma=0 → standard cross-entropy.  gamma=1 is a mild but effective boost.
        self.focal_gamma = training_cfg.get("focal_gamma", 0.0)
        if self.focal_gamma > 0:
            print(f"  focal loss      ON  gamma={self.focal_gamma}")

        # Pick-presence loss: directly penalise low model probability at the
        # true pick sample.  Unlike CE (which weights all time steps equally),
        # this term concentrates gradient on the exact pick location for traces
        # the model currently misses.  gamma=0 disables it.
        self.presence_gamma = training_cfg.get("presence_gamma", 0.0)
        if self.presence_gamma > 0:
            print(f"  pick-presence   ON  gamma={self.presence_gamma}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()   # teacher must always stay in eval (no dropout/BN in train mode)
        return self

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        return self.model(x, logits=logits)

    def compute_loss_and_metrics(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass + loss + per-phase accuracy.
        When distillation is enabled, loss = alpha * KL(student||teacher) + (1-alpha) * CE.
        Returns a dict with 'loss', 'acc', 'N_acc', 'P_acc', 'S_acc'.
        """
        logits = self(x, logits=True)
        probs  = F.softmax(logits, dim=1)

        # Flatten spatial dimension for cross-entropy
        y_flat      = y.permute(0, 2, 1).reshape(-1, 3)
        logits_flat = logits.permute(0, 2, 1).reshape(-1, 3)
        y_cls       = y_flat.argmax(dim=1)

        if self.focal_gamma > 0:
            ce_per = F.cross_entropy(logits_flat, y_cls, weight=self.class_weight, reduction="none")
            pt = torch.exp(-ce_per)
            ce_loss = ((1 - pt) ** self.focal_gamma * ce_per).mean()
        else:
            ce_loss = F.cross_entropy(logits_flat, y_cls, weight=self.class_weight)

        if self.distill_alpha > 0 and self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(x, logits=True)
            T = self.distill_T
            # KL divergence on softened distributions (temperature scaling)
            s_flat = (logits_flat / T)
            t_flat = (teacher_logits.permute(0, 2, 1).reshape(-1, 3) / T)
            kl_loss = F.kl_div(
                F.log_softmax(s_flat, dim=-1),
                F.softmax(t_flat, dim=-1),
                reduction="batchmean",
            ) * (T ** 2)
            loss = (1 - self.distill_alpha) * ce_loss + self.distill_alpha * kl_loss
        else:
            loss = ce_loss

        if self.timing_beta > 0:
            p_timing = soft_pick_mae(probs, y, phase_idx=0)  # P = ch0
            s_timing = soft_pick_mae(probs, y, phase_idx=1)  # S = ch1
            loss = loss + self.timing_beta * (p_timing + s_timing)

        if self.presence_gamma > 0:
            # For each trace with a P pick, penalise low model probability at
            # the true pick sample: L = -log(p_model[t_pick] + eps).
            # y[:, 0, :] is the P-channel Gaussian label; its argmax = t_pick.
            p_lbl   = y[:, 0, :]                                   # (B, T)
            has_p   = p_lbl.max(dim=1).values > 0.5               # (B,)
            if has_p.any():
                t_pick       = p_lbl[has_p].argmax(dim=1)         # (N,)
                p_prob_pick  = probs[has_p, 0, :].gather(          # (N,)
                    1, t_pick.unsqueeze(1)).squeeze(1)
                presence_loss = -torch.log(p_prob_pick + 1e-6).mean()
                loss = loss + self.presence_gamma * presence_loss

        pred_cls = logits_flat.argmax(dim=1)
        acc      = (pred_cls == y_cls).float().mean()

        metrics = {"loss": loss, "acc": acc}

        true_cls = y.argmax(dim=1).reshape(-1)
        pred_flat = probs.permute(0, 2, 1).reshape(-1, 3).argmax(dim=1)
        for idx, name in enumerate(["P", "S", "N"]):  # PSN — matches jma_wc convention
            mask = true_cls == idx
            if mask.any():
                metrics[f"{name}_acc"] = (pred_flat[mask] == true_cls[mask]).float().mean()

        return metrics, probs

    def build_optimiser(self, config: dict):
        training_cfg  = config.get("training", {})
        scheduler_cfg = training_cfg.get("scheduler", {})

        weight_decay = training_cfg.get("weight_decay", 0.0)
        opt_name = training_cfg.get("optimizer", "adam").lower()
        trainable = [p for p in self.parameters() if p.requires_grad]
        if opt_name == "adamw":
            opt = torch.optim.AdamW(trainable, lr=self.learning_rate,
                                    weight_decay=weight_decay)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(
                trainable, lr=self.learning_rate, momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            opt = torch.optim.Adam(trainable, lr=self.learning_rate,
                                   weight_decay=weight_decay)

        sched_name = scheduler_cfg.get("name", "ReduceLROnPlateau")
        if sched_name == "CosineAnnealingWarmup":
            max_epochs    = config.get("training", {}).get("max_epochs", 100)
            warmup_epochs = scheduler_cfg.get("warmup_epochs", 5)
            min_lr        = scheduler_cfg.get("min_lr", 1e-6)
            start_factor  = scheduler_cfg.get("warmup_start_factor", 0.01)
            warmup_sched  = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=start_factor, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, max_epochs - warmup_epochs), eta_min=min_lr,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode     = scheduler_cfg.get("mode",     "min"),
                factor   = scheduler_cfg.get("factor",   0.5),
                patience = scheduler_cfg.get("patience", 5),
                min_lr   = scheduler_cfg.get("min_lr",   1e-6),
            )
        return opt, scheduler
