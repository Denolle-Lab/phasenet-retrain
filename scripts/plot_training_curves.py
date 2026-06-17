#!/usr/bin/env python3
"""
plot_training_curves.py

Generate training visualisations from the metrics CSV written by
MetricsCSVCallback (and optionally from a PyTorch Lightning CSVLogger
metrics.csv if present).

Usage
-----
  python scripts/plot_training_curves.py
  python scripts/plot_training_curves.py \
      --metrics-csv results/finetune_jma_wc_metrics.csv \
      --output-dir  results/finetune_jma_wc

Plots produced
--------------
  training_loss_curves.png      — train + val loss vs epoch
  training_acc_curves.png       — overall + per-phase accuracy vs epoch
  pick_residual_curves.png      — P- and S-pick MAE (seconds) vs epoch
  learning_rate_curve.png       — learning-rate schedule
  training_dashboard.png        — 2×2 combined overview figure
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Replace empty strings with NaN for numeric columns
    df = df.replace("", np.nan)
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def try_load_pl_csv(output_dir: Path) -> Optional[pd.DataFrame]:
    """Try to load a PyTorch Lightning CSVLogger metrics.csv for step-level data."""
    candidates = list(output_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    # Pick the most recently modified one
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        df = pd.read_csv(best)
        return df
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Individual plots
# ──────────────────────────────────────────────────────────────────────────────

def _epoch_col(df: pd.DataFrame) -> pd.Series:
    return df["epoch"] if "epoch" in df.columns else pd.RangeIndex(len(df))


def plot_loss(df: pd.DataFrame, out_path: Path):
    epochs = _epoch_col(df)
    fig, ax = plt.subplots(figsize=(8, 5))

    if "train_loss" in df.columns and df["train_loss"].notna().any():
        ax.plot(epochs, df["train_loss"], label="Train loss", color="#2196F3", lw=2)
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        ax.plot(epochs, df["val_loss"],   label="Val loss",   color="#F44336", lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Mark best val_loss
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        best_idx = df["val_loss"].idxmin()
        best_e   = epochs.iloc[best_idx] if hasattr(epochs, "iloc") else best_idx
        best_v   = df["val_loss"].iloc[best_idx]
        ax.axvline(best_e, color="#F44336", ls="--", alpha=0.5, lw=1)
        ax.annotate(
            f"best = {best_v:.4f}\n@ epoch {int(best_e)}",
            xy=(best_e, best_v), xytext=(best_e + 1, best_v * 1.05),
            fontsize=9, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336"),
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_accuracy(df: pd.DataFrame, out_path: Path):
    epochs = _epoch_col(df)
    fig, ax = plt.subplots(figsize=(8, 5))

    pairs = [
        ("train_acc",  "#2196F3", "Train overall", "-"),
        ("val_acc",    "#F44336", "Val overall",   "-"),
        ("val_N_acc",  "#9C27B0", "Val Noise",     "--"),
        ("val_P_acc",  "#4CAF50", "Val P-wave",    "--"),
        ("val_S_acc",  "#FF9800", "Val S-wave",    "--"),
    ]
    for col, color, label, ls in pairs:
        if col in df.columns and df[col].notna().any():
            ax.plot(epochs, df[col], label=label, color=color, lw=2, ls=ls)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pixel-level accuracy")
    ax.set_title("Accuracy (pixel-level N/P/S class)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_residuals(df: pd.DataFrame, out_path: Path):
    epochs = _epoch_col(df)
    fig, ax = plt.subplots(figsize=(8, 5))

    has_p = "val_p_mae_s" in df.columns and df["val_p_mae_s"].notna().any()
    has_s = "val_s_mae_s" in df.columns and df["val_s_mae_s"].notna().any()

    if has_p:
        ax.plot(epochs, df["val_p_mae_s"], label="P-pick MAE", color="#4CAF50", lw=2)
    if has_s:
        ax.plot(epochs, df["val_s_mae_s"], label="S-pick MAE", color="#FF9800", lw=2)

    if not has_p and not has_s:
        ax.text(0.5, 0.5, "No pick residual data yet\n(computed from validation)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean absolute error (s)")
    ax.set_title("Pick Timing Residuals (val set)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Reference lines
    for y, label in [(0.1, "0.1 s"), (0.5, "0.5 s")]:
        ax.axhline(y, color="grey", ls=":", alpha=0.6, lw=1)
        ax.text(epochs.iloc[0] if hasattr(epochs, "iloc") else 0,
                y, label, fontsize=8, color="grey", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_lr(df: pd.DataFrame, out_path: Path):
    epochs = _epoch_col(df)
    fig, ax = plt.subplots(figsize=(8, 4))

    if "lr" in df.columns and df["lr"].notna().any():
        ax.semilogy(epochs, df["lr"], color="#607D8B", lw=2)
        # Mark LR drops
        lr = df["lr"].dropna()
        drops = lr[lr.diff() < 0].index
        for idx in drops:
            e = epochs.iloc[idx] if hasattr(epochs, "iloc") else idx
            ax.axvline(e, color="#F44336", ls="--", alpha=0.4, lw=1)
    else:
        ax.text(0.5, 0.5, "No LR data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dashboard(df: pd.DataFrame, out_path: Path):
    """2×2 combined figure with all four panels."""
    epochs = _epoch_col(df)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0])
    for col, color, label in [
        ("train_loss", "#2196F3", "Train"),
        ("val_loss",   "#F44336", "Val"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax_loss.plot(epochs, df[col], label=label, color=color, lw=2)
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, 1])
    for col, color, label, ls in [
        ("train_acc", "#2196F3", "Train", "-"),
        ("val_acc",   "#F44336", "Val",   "-"),
        ("val_P_acc", "#4CAF50", "P",     "--"),
        ("val_S_acc", "#FF9800", "S",     "--"),
        ("val_N_acc", "#9C27B0", "N",     ":"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax_acc.plot(epochs, df[col], label=label, color=color, lw=1.8, ls=ls)
    ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Pixel accuracy"); ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(ncol=2, fontsize=9); ax_acc.grid(alpha=0.3)

    # ── Pick residuals ────────────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs[1, 0])
    has_res = False
    for col, color, label in [
        ("val_p_mae_s", "#4CAF50", "P MAE (s)"),
        ("val_s_mae_s", "#FF9800", "S MAE (s)"),
    ]:
        if col in df.columns and df[col].notna().any():
            ax_res.plot(epochs, df[col], label=label, color=color, lw=2)
            has_res = True
    if not has_res:
        ax_res.text(0.5, 0.5, "No residual data yet", ha="center", va="center",
                    transform=ax_res.transAxes, color="gray")
    ax_res.set_title("Pick MAE"); ax_res.set_xlabel("Epoch")
    ax_res.set_ylabel("seconds"); ax_res.legend(); ax_res.grid(alpha=0.3)

    # ── Learning rate ─────────────────────────────────────────────────────────
    ax_lr = fig.add_subplot(gs[1, 1])
    if "lr" in df.columns and df["lr"].notna().any():
        ax_lr.semilogy(epochs, df["lr"], color="#607D8B", lw=2)
    else:
        ax_lr.text(0.5, 0.5, "No LR data", ha="center", va="center",
                   transform=ax_lr.transAxes, color="gray")
    ax_lr.set_title("Learning Rate"); ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR"); ax_lr.grid(alpha=0.3, which="both")

    # Model name / summary
    n_epochs = int(df["epoch"].max()) + 1 if "epoch" in df.columns else len(df)
    best_val = df["val_loss"].min() if "val_loss" in df.columns else float("nan")
    fig.suptitle(
        f"PhaseNet jma_wc fine-tuning — {n_epochs} epochs  |  best val_loss = {best_val:.4f}",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Step-level loss plot (from PL CSVLogger)
# ──────────────────────────────────────────────────────────────────────────────

def plot_step_loss(df_step: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))

    if "train_loss_step" in df_step.columns:
        col = "train_loss_step"
    elif "train_loss" in df_step.columns:
        col = "train_loss"
    else:
        plt.close(fig)
        return

    s = df_step[col].dropna()
    ax.plot(s.index, s.values, color="#2196F3", lw=0.8, alpha=0.7, label="train loss (step)")

    # Rolling average
    window = max(10, len(s) // 100)
    ax.plot(s.index, s.rolling(window, min_periods=1).mean(),
            color="#0D47A1", lw=2, label=f"rolling avg ({window} steps)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Step-level Training Loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    csv_path = Path(args.metrics_csv)
    if not csv_path.exists():
        print(f"ERROR: metrics CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from : {csv_path}")
    df = load_metrics(str(csv_path))
    print(f"  {len(df)} epoch rows, columns: {list(df.columns)}")

    plot_loss(df,       out_dir / "training_loss_curves.png")
    plot_accuracy(df,   out_dir / "training_acc_curves.png")
    plot_residuals(df,  out_dir / "pick_residual_curves.png")
    plot_lr(df,         out_dir / "learning_rate_curve.png")
    plot_dashboard(df,  out_dir / "training_dashboard.png")

    # Try to add step-level loss from PL CSVLogger
    df_step = try_load_pl_csv(out_dir.parent)
    if df_step is not None:
        plot_step_loss(df_step, out_dir / "step_loss_curve.png")

    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from metrics CSV")
    parser.add_argument(
        "--metrics-csv",
        default="results/finetune_jma_wc_metrics.csv",
        help="Path to MetricsCSVCallback output CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="results/finetune_jma_wc",
        help="Directory to write plot PNGs",
    )
    args = parser.parse_args()
    main(args)
