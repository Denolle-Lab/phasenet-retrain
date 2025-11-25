"""
PhaseNet model module for PyTorch Lightning
Based on SeisBench implementation: https://github.com/seisbench/seisbench/blob/main/seisbench/models/phasenet.py
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import seisbench.models as sbm
from typing import Dict, Any, Optional


class PhaseNetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for SeisBench PhaseNet model
    
    This module wraps the PhaseNet architecture from SeisBench for training with PyTorch Lightning.
    PhaseNet is a U-Net style architecture with encoder-decoder structure for seismic phase picking.
    
    Architecture overview:
    - Encoder: 5 convolutional blocks with downsampling
    - Bottleneck: Deepest convolutional block
    - Decoder: 4 upsampling blocks with skip connections
    - Output: 3 channels for P, S, and Noise probabilities
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Load model configuration
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        arch_config = model_config.get("architecture", {})
        
        # Initialize PhaseNet from SeisBench
        if model_config.get("pretrained", {}).get("use_pretrained", True):
            pretrained_name = model_config.get("pretrained", {}).get("model_name", "stead")
            print(f"Loading pretrained PhaseNet model: {pretrained_name}")
            self.model = sbm.PhaseNet.from_pretrained(pretrained_name)
            
            # Optionally freeze layers for fine-tuning
            freeze_layers = model_config.get("pretrained", {}).get("freeze_layers", [])
            if freeze_layers:
                self._freeze_layers(freeze_layers)
        else:
            # Create model from scratch
            print("Creating PhaseNet model from scratch")
            self.model = sbm.PhaseNet(
                in_channels=arch_config.get("in_channels", 3),
                classes=arch_config.get("classes", 3),
                phases=arch_config.get("phases", "NPS"),  # Noise, P, S
                sampling_rate=arch_config.get("sampling_rate", 100),
                norm=arch_config.get("norm", "std"),
                filter_factor=arch_config.get("filter_factor", 1)
            )
        
        # Loss function (CrossEntropyLoss expects class indices or logits)
        # For PhaseNet, we use the softmax outputs, so we need to handle this carefully
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate
        self.learning_rate = training_config.get("learning_rate", 0.001)
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def _freeze_layers(self, layer_names):
        """Freeze specified layers for transfer learning"""
        for name, param in self.model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    print(f"Frozen layer: {name}")
        
    def forward(self, x, logits=False):
        """
        Forward pass through PhaseNet
        
        Args:
            x: Input waveform tensor of shape (batch, 3, samples)
            logits: If True, return logits; if False, return softmax probabilities
        
        Returns:
            Predictions of shape (batch, 3, samples) for N, P, S channels
        """
        return self.model(x, logits=logits)
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (waveforms, labels)
                waveforms: (batch, 3, samples) - Z, N, E components
                labels: (batch, 3, samples) - N, P, S probabilities
        """
        x, y = batch
        
        # Get predictions (with logits for numerical stability with CrossEntropyLoss)
        y_hat = self(x, logits=True)
        
        # Reshape for loss calculation: (batch * samples, classes)
        y_hat_reshaped = y_hat.permute(0, 2, 1).reshape(-1, y_hat.shape[1])
        y_reshaped = y.permute(0, 2, 1).reshape(-1, y.shape[1])
        
        # Convert soft labels to class indices if needed
        if y.shape[1] == 3:  # Soft labels
            y_indices = torch.argmax(y_reshaped, dim=1)
        else:
            y_indices = y_reshaped.long()
        
        loss = self.criterion(y_hat_reshaped, y_indices)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Calculate accuracy on picks (where probability > 0.5)
        with torch.no_grad():
            y_hat_probs = F.softmax(y_hat, dim=1)
            pred_class = torch.argmax(y_hat_probs, dim=1)
            true_class = torch.argmax(y, dim=1)
            acc = (pred_class == true_class).float().mean()
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with comprehensive metrics"""
        x, y = batch
        
        # Get predictions
        y_hat = self(x, logits=True)
        
        # Reshape for loss calculation
        y_hat_reshaped = y_hat.permute(0, 2, 1).reshape(-1, y_hat.shape[1])
        y_reshaped = y.permute(0, 2, 1).reshape(-1, y.shape[1])
        y_indices = torch.argmax(y_reshaped, dim=1)
        
        loss = self.criterion(y_hat_reshaped, y_indices)
        
        # Calculate metrics
        y_hat_probs = F.softmax(y_hat, dim=1)
        pred_class = torch.argmax(y_hat_probs, dim=1)
        true_class = torch.argmax(y, dim=1)
        acc = (pred_class == true_class).float().mean()
        
        # Calculate per-phase metrics
        for phase_idx, phase_name in enumerate(['N', 'P', 'S']):
            phase_mask = true_class == phase_idx
            if phase_mask.sum() > 0:
                phase_acc = (pred_class[phase_mask] == true_class[phase_mask]).float().mean()
                self.log(f"val_{phase_name}_acc", phase_acc, on_epoch=True, sync_dist=True)
        
        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"val_loss": loss, "val_acc": acc}
    
    def test_step(self, batch, batch_idx):
        """Test step with detailed evaluation"""
        x, y = batch
        
        # Get predictions
        y_hat = self(x, logits=False)  # Get probabilities for analysis
        
        # Calculate loss
        y_hat_logits = self(x, logits=True)
        y_hat_reshaped = y_hat_logits.permute(0, 2, 1).reshape(-1, y_hat_logits.shape[1])
        y_reshaped = y.permute(0, 2, 1).reshape(-1, y.shape[1])
        y_indices = torch.argmax(y_reshaped, dim=1)
        loss = self.criterion(y_hat_reshaped, y_indices)
        
        # Calculate overall metrics
        pred_class = torch.argmax(y_hat, dim=1)
        true_class = torch.argmax(y, dim=1)
        acc = (pred_class == true_class).float().mean()
        
        # Per-phase metrics
        phase_metrics = {}
        for phase_idx, phase_name in enumerate(['N', 'P', 'S']):
            phase_mask = true_class == phase_idx
            if phase_mask.sum() > 0:
                phase_acc = (pred_class[phase_mask] == true_class[phase_mask]).float().mean()
                phase_metrics[f"test_{phase_name}_acc"] = phase_acc
        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, sync_dist=True)
        
        for key, val in phase_metrics.items():
            self.log(key, val, sync_dist=True)
        
        return {"test_loss": loss, "test_acc": acc, **phase_metrics}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        training_config = self.config.get("training", {})
        
        # Optimizer
        optimizer_name = training_config.get("optimizer", "adam").lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_config = training_config.get("scheduler", {})
        scheduler_name = scheduler_config.get("name", "ReduceLROnPlateau")
        
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5),
                min_lr=scheduler_config.get("min_lr", 1e-6)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config.get("max_epochs", 100)
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
