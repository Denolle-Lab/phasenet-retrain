#!/usr/bin/env python3
"""
train_scratch.py  —  Train PhaseNet from random initialisation.

Uses the same GPU-saturating infrastructure as finetune.py:
  * All waveforms pre-loaded into RAM
  * AMP (fp16) + torch.compile
  * pin_memory + non_blocking transfers
  * Batch size 1024

Key differences from finetune.py:
  * Model: PhaseNetScratch (random init, no pretrained weights)
  * Dataset: ScratchDataset (stronger augmentation — see scratch_dataset.py)
  * Loss: soft cross-entropy on full Gaussian label distribution
  * Hybrid early stopping: val_loss until P-MAE is first seen, then P-MAE

Usage
-----
  conda activate surface
  cd /home/ak287/phasenet-retrain
  nohup python scripts/train_scratch.py \\
        --config configs/phasenet_scratch_v1.yaml \\
        > results/phasenet_scratch_v1/train.log 2>&1 &
  tail -f results/phasenet_scratch_v1/train.log

  # Resume a run:
  python scripts/train_scratch.py --resume checkpoints/phasenet_scratch_v1/last.pt

  # Evaluate best checkpoint on test set:
  python scripts/train_scratch.py --test-only checkpoints/phasenet_scratch_v1/best.pt
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("SEISBENCH_CACHE_ROOT", "/data/wsd04/ak287/.seisbench")
import seisbench
seisbench.cache_root = "/data/wsd04/ak287/.seisbench"

from scratch_model   import MetricsLogger, PhaseNetScratch, pick_residuals
from scratch_dataset import ScratchDataset
from plot_training_curves import load_metrics, plot_dashboard, plot_loss, plot_accuracy, plot_residuals, plot_lr


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(config: dict):
    """Build train/val/test loaders using ScratchDataset (strong augmentation)."""
    data_cfg     = config.get("data",     {})
    train_cfg    = config.get("training", {})
    batch_size   = train_cfg.get("batch_size",  1024)
    window_len   = data_cfg.get("window_length", 3001)
    load_workers = train_cfg.get("num_workers",    16)

    print("Pre-loading training set (strong augment):")
    train_ds = ScratchDataset(
        data_cfg["train_manifest"], augment=True,
        window_len=window_len, load_workers=load_workers,
    )
    print("Pre-loading validation set:")
    val_ds = ScratchDataset(
        data_cfg["val_manifest"], augment=False,
        window_len=window_len, load_workers=load_workers,
    )
    print("Pre-loading test set:")
    test_ds = ScratchDataset(
        data_cfg["test_manifest"], augment=False,
        window_len=window_len, load_workers=load_workers,
    )

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = 0,      # 0 = main process; fastest for in-memory data (no IPC)
        pin_memory  = True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"\n  Train : {len(train_ds):>7,} traces  ({len(train_loader):,} batches @ {batch_size})")
    print(f"  Val   : {len(val_ds):>7,} traces  ({len(val_loader):,} batches)")
    print(f"  Test  : {len(test_ds):>7,} traces  ({len(test_loader):,} batches)")

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimiser, scaler, epoch, val_loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     raw.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scaler":    scaler.state_dict(),
    }, path)


def load_checkpoint(path: Path, model, optimiser=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    raw  = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.load_state_dict(ckpt["model"])
    if optimiser and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ──────────────────────────────────────────────────────────────────────────────
# Epoch loop
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, optimiser=None, scaler=None, grad_clip=1.0):
    training = optimiser is not None
    model.train(training)
    use_amp = (scaler is not None) and device.type == "cuda"

    tot_loss = tot_acc = tot_grad_norm = 0.0
    phase_correct = {"N": 0.0, "P": 0.0, "S": 0.0}
    phase_total   = {"N": 0,   "P": 0,   "S": 0}
    p_residuals, s_residuals = [], []
    n_batches = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                metrics, probs = model.compute_loss_and_metrics(x, y)
                loss = metrics["loss"]

            if training:
                optimiser.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimiser.step()
                tot_grad_norm += gn.item()

            loss_val = loss.item()
            acc_val  = metrics["acc"].item()
            if not (math.isfinite(loss_val) and math.isfinite(acc_val)):
                n_batches -= 1
                continue
            tot_loss += loss_val
            tot_acc  += acc_val

            n = x.shape[0] * x.shape[-1]
            for ph in ("N", "P", "S"):
                if f"{ph}_acc" in metrics:
                    phase_correct[ph] += metrics[f"{ph}_acc"].item() * n
                    phase_total[ph]   += n

            if not training:
                p_r, s_r = pick_residuals(probs.float(), y)
                if p_r.numel():
                    p_residuals.append(p_r.cpu())
                if s_r.numel():
                    s_residuals.append(s_r.cpu())

            n_batches += 1

    results = {
        "loss":      tot_loss / n_batches,
        "acc":       tot_acc  / n_batches,
        "grad_norm": tot_grad_norm / n_batches if training else float("nan"),
    }
    for ph in ("N", "P", "S"):
        if phase_total[ph] > 0:
            results[f"{ph}_acc"] = phase_correct[ph] / phase_total[ph]

    sr = 100.0
    if p_residuals:
        p_all = torch.cat(p_residuals)
        results["p_mae_s"]  = p_all.abs().float().mean().item() / sr
        results["p_rmse_s"] = (p_all.float() ** 2).mean().sqrt().item() / sr
    if s_residuals:
        s_all = torch.cat(s_residuals)
        results["s_mae_s"]  = s_all.abs().float().mean().item() / sr
        results["s_rmse_s"] = (s_all.float() ** 2).mean().sqrt().item() / sr

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Training driver
# ──────────────────────────────────────────────────────────────────────────────

def _plot_progress(metrics_csv: str, log_cfg: dict):
    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        return
    out_dir = (
        Path(log_cfg.get("save_dir", "results"))
        / log_cfg.get("run_name", "phasenet_scratch_v1")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df = load_metrics(str(csv_path))
        plot_loss(df,      out_dir / "training_loss_curves.png")
        plot_accuracy(df,  out_dir / "training_acc_curves.png")
        plot_residuals(df, out_dir / "pick_residual_curves.png")
        plot_lr(df,        out_dir / "learning_rate_curve.png")
        plot_dashboard(df, out_dir / "training_dashboard.png")
    except Exception as e:
        print(f"  [plot] warning: {e}", flush=True)


def train(config: dict, resume_path=None):
    torch.manual_seed(config.get("seed", 42))

    hw_cfg    = config.get("hardware", {})
    train_cfg = config.get("training", {})
    log_cfg   = config.get("logging",  {})

    device  = torch.device(
        "cuda" if torch.cuda.is_available()
        and hw_cfg.get("accelerator", "auto") != "cpu" else "cpu"
    )
    use_amp = hw_cfg.get("amp", True) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device : {device}  |  AMP : {use_amp}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── data ──────────────────────────────────────────────────────────────────
    t_load = time.time()
    print("\nPre-loading datasets into RAM ...")
    train_loader, val_loader, _ = build_dataloaders(config)
    print(f"Data ready in {(time.time()-t_load)/60:.1f} min\n")

    # ── model ─────────────────────────────────────────────────────────────────
    model_raw = PhaseNetScratch(config).to(device)
    try:
        model = torch.compile(model_raw)
        print("torch.compile: enabled")
    except Exception:
        model = model_raw
        print("torch.compile: skipped")

    optimiser, scheduler = model_raw.build_optimiser(config)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_dir    = Path(log_cfg.get("checkpoint_dir", "checkpoints/phasenet_scratch_v1"))
    best_path   = ckpt_dir / "best.pt"
    last_path   = ckpt_dir / "last.pt"
    metrics_csv = log_cfg.get("metrics_csv", "results/phasenet_scratch_v1_metrics.csv")

    start_epoch    = 0
    best_val_loss  = float("inf")
    best_val_p_mae = float("inf")

    if resume_path:
        print(f"Resuming from: {resume_path}")
        start_epoch, best_val_loss = load_checkpoint(
            Path(resume_path), model, optimiser, scaler
        )
        start_epoch += 1
        new_lr = train_cfg["learning_rate"]
        for pg in optimiser.param_groups:
            pg["lr"] = new_lr
        print(f"LR overridden to: {new_lr}")

    metrics_logger = MetricsLogger(metrics_csv)

    es_cfg      = train_cfg.get("early_stopping", {})
    es_patience = es_cfg.get("patience", 25)
    es_counter  = 0
    p_mae_seen  = False    # delay P-MAE early stopping until first finite value

    max_epochs = train_cfg.get("max_epochs", 300)
    grad_clip  = train_cfg.get("gradient_clip_val", 1.0)

    n_params = sum(p.numel() for p in model_raw.parameters())
    print(f"\nModel params : {n_params:,}")
    print(f"Batch size   : {train_cfg.get('batch_size', 1024)}")
    print(f"Max epochs   : {max_epochs}  (early-stop patience={es_patience})")
    print()
    print(f"{'Epoch':>6}  {'TrainLoss':>10} {'TrainAcc':>9} {'GradNorm':>9} "
          f"{'ValLoss':>9} {'ValAcc':>8} "
          f"{'P-MAE(s)':>9} {'S-MAE(s)':>9}  {'LR':>9}  {'t(s)':>5}")
    print("-" * 103)

    t_start = time.time()

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        train_m = run_epoch(
            model, train_loader, device,
            optimiser=optimiser, scaler=scaler, grad_clip=grad_clip,
        )
        val_m = run_epoch(model, val_loader, device, scaler=scaler)

        lr_now = optimiser.param_groups[0]["lr"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_m["loss"])
        else:
            scheduler.step()

        p_mae     = val_m.get("p_mae_s", float("nan"))
        s_mae     = val_m.get("s_mae_s", float("nan"))
        grad_norm = train_m.get("grad_norm", float("nan"))
        ep_t      = time.time() - t0

        # ── improvement check ─────────────────────────────────────────────────
        # Phase 1: monitor val_loss until P-MAE first appears.
        # Phase 2: switch to P-MAE once it has been seen (more informative).
        if math.isfinite(p_mae):
            p_mae_seen = True

        if p_mae_seen:
            improved = math.isfinite(p_mae) and p_mae < best_val_p_mae
        else:
            improved = val_m["loss"] < best_val_loss

        mark = " ✓" if improved else ""
        print(
            f"{epoch:>6}  {train_m['loss']:>10.4f} {train_m['acc']:>9.4f} {grad_norm:>9.3f} "
            f"{val_m['loss']:>9.4f} {val_m['acc']:>8.4f} "
            f"{p_mae:>9.4f} {s_mae:>9.4f}  {lr_now:>9.2e}  {ep_t:>5.0f}s"
            f"{mark}",
            flush=True,
        )

        metrics_logger.log({
            "epoch":           epoch,
            "train_loss":      round(train_m["loss"],    6),
            "train_acc":       round(train_m["acc"],     6),
            "train_grad_norm": round(grad_norm,          4) if grad_norm == grad_norm else "",
            "val_loss":        round(val_m["loss"],      6),
            "val_acc":         round(val_m["acc"],       6),
            "val_N_acc":       round(val_m.get("N_acc",  float("nan")), 6),
            "val_P_acc":       round(val_m.get("P_acc",  float("nan")), 6),
            "val_S_acc":       round(val_m.get("S_acc",  float("nan")), 6),
            "val_p_mae_s":     round(p_mae, 6) if p_mae == p_mae else "",
            "val_s_mae_s":     round(s_mae, 6) if s_mae == s_mae else "",
            "lr":              lr_now,
        })

        _plot_progress(metrics_csv, log_cfg)

        save_checkpoint(model, optimiser, scaler, epoch, val_m["loss"], last_path)
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
        if math.isfinite(p_mae) and p_mae < best_val_p_mae:
            best_val_p_mae = p_mae

        if improved:
            save_checkpoint(model, optimiser, scaler, epoch, val_m["loss"], best_path)
            es_counter = 0
        else:
            es_counter += 1

        if es_patience and es_counter >= es_patience:
            print(f"\nEarly stopping triggered after {es_patience} epochs without improvement.")
            break

    total = time.time() - t_start
    print("=" * 103)
    print(f"Done — {total/60:.1f} min  |  best val_loss={best_val_loss:.4f}  best val_p_mae={best_val_p_mae:.4f}s")
    print(f"Best checkpoint : {best_path}")
    print(f"Metrics CSV     : {metrics_csv}")
    return best_path, metrics_csv


# ──────────────────────────────────────────────────────────────────────────────
# Test evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_test(config: dict, ckpt_path: str):
    hw_cfg  = config.get("hardware", {})
    device  = torch.device(
        "cuda" if torch.cuda.is_available()
        and hw_cfg.get("accelerator", "auto") != "cpu" else "cpu"
    )
    use_amp = hw_cfg.get("amp", True) and device.type == "cuda"

    print(f"\nLoading checkpoint for test: {ckpt_path}")
    _, _, test_loader = build_dataloaders(config)

    model = PhaseNetScratch(config).to(device)
    load_checkpoint(Path(ckpt_path), model)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    test_m  = run_epoch(model, test_loader, device, scaler=scaler)

    print("\n── Test Results ──────────────────")
    for k, v in sorted(test_m.items()):
        print(f"  {k:<20}: {v:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    config = load_config(args.config)

    if args.test_only:
        run_test(config, args.test_only)
        return

    best_ckpt, metrics_csv = train(config, resume_path=args.resume)

    if args.test:
        run_test(config, str(best_ckpt))

    plot_script = SCRIPTS_DIR / "plot_training_curves.py"
    log_cfg     = config.get("logging", {})
    out_dir = (
        Path(log_cfg.get("save_dir", "results"))
        / log_cfg.get("run_name", "phasenet_scratch_v1")
    )
    if plot_script.exists() and Path(metrics_csv).exists():
        print(f"\nGenerating plots → {out_dir}/")
        subprocess.run(
            [sys.executable, str(plot_script),
             "--metrics-csv", metrics_csv,
             "--output-dir",  str(out_dir)],
            check=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/phasenet_scratch_v1.yaml")
    parser.add_argument("--resume",    default=None)
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--test-only", default=None, metavar="CKPT")
    args = parser.parse_args()
    main(args)
