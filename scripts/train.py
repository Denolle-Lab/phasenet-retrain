"""
Training script for PhaseNet retraining
"""
import os
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model import PhaseNetLightning
from data_module import PhaseNetDataModule


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_callbacks(config):
    """Setup training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get("logging", {}).get("checkpoint_dir", "checkpoints"),
        filename="phasenet-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_config = config.get("training", {}).get("early_stopping", {})
    if early_stop_config:
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get("monitor", "val_loss"),
            patience=early_stop_config.get("patience", 10),
            mode=early_stop_config.get("mode", "min"),
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(config):
    """Setup experiment logger"""
    logging_config = config.get("logging", {})
    
    loggers = []
    
    # TensorBoard logger
    if logging_config.get("use_tensorboard", True):
        tb_logger = TensorBoardLogger(
            save_dir=logging_config.get("save_dir", "results"),
            name="phasenet_training"
        )
        loggers.append(tb_logger)
    
    # Weights & Biases logger
    if logging_config.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=logging_config.get("wandb_project", "phasenet-retrain"),
            entity=logging_config.get("wandb_entity", None),
            save_dir=logging_config.get("save_dir", "results")
        )
        loggers.append(wandb_logger)
    
    return loggers if loggers else None


def main(args):
    """Main training function"""
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    pl.seed_everything(config.get("seed", 42))
    
    # Setup data module
    data_module = PhaseNetDataModule(
        config=config,
        batch_size=config.get("training", {}).get("batch_size", 128),
        num_workers=config.get("training", {}).get("num_workers", 4)
    )
    
    # Setup model
    model = PhaseNetLightning(config)
    
    # Setup callbacks and loggers
    callbacks = setup_callbacks(config)
    loggers = setup_logger(config)
    
    # Setup trainer
    hardware_config = config.get("hardware", {})
    training_config = config.get("training", {})
    
    trainer = pl.Trainer(
        max_epochs=training_config.get("max_epochs", 100),
        accelerator=hardware_config.get("accelerator", "auto"),
        devices=hardware_config.get("devices", 1),
        precision=hardware_config.get("precision", "32-true"),
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        log_every_n_steps=config.get("logging", {}).get("log_every_n_steps", 50),
        deterministic=True,
    )
    
    # Train model
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)
    
    # Test model
    if args.test:
        trainer.test(model, data_module)
    
    print("Training complete!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhaseNet model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run testing after training"
    )
    
    args = parser.parse_args()
    main(args)
