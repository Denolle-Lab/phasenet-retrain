"""
Data module for PhaseNet training with PyTorch Lightning
Uses SeisBench data formats and generators
"""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import seisbench.data as sbd
import seisbench.generate as sbg
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from label_error_filter import LabelErrorFilter

logger = logging.getLogger(__name__)


class PhaseNetDataset(Dataset):
    """
    Custom dataset for PhaseNet training using SeisBench
    
    This dataset can load data from:
    1. SeisBench format (HDF5 with metadata)
    2. Custom format (implement _load_custom_data)
    """
    def __init__(
        self,
        data_path: str,
        window_length: int = 3001,
        augmentation: bool = False,
        config: Optional[Dict[str, Any]] = None,
        sampling_rate: int = 100
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.window_length = window_length
        self.augmentation = augmentation
        self.config = config or {}
        self.sampling_rate = sampling_rate
        
        # Load data using SeisBench or custom loader
        self.data, self.metadata = self._load_dataset()
        
        # Setup augmentation pipeline if requested
        if self.augmentation:
            self.augmentation_pipeline = self._setup_augmentation()
        else:
            self.augmentation_pipeline = None
        
    def _load_dataset(self):
        """
        Load dataset from disk
        
        Returns:
            data: Waveform data array
            metadata: Associated metadata (picks, event info, etc.)
        """
        # Check if using SeisBench dataset directly
        if self.config.get('use_seisbench_dataset', False):
            dataset_name = self.config.get('dataset_name', 'STEAD')
            logger.info(f"Loading SeisBench dataset: {dataset_name}")
            return self._load_direct_seisbench_dataset(dataset_name)
        
        # Try to load as SeisBench format from local files
        elif self.data_path and (self.data_path / "metadata.csv").exists():
            logger.info(f"Loading SeisBench format data from {self.data_path}")
            return self._load_seisbench_data()
        else:
            # Implement custom data loading
            logger.info(f"Loading custom format data from {self.data_path}")
            return self._load_custom_data()
    
    def _load_direct_seisbench_dataset(self, dataset_name):
        """
        Load dataset directly from SeisBench with optional label error filtering
        
        Args:
            dataset_name: Name of the dataset (STEAD, INSTANCE, ETHZ, PNW, TXED)
            
        Returns:
            Tuple of (waveforms, metadata) compatible with PhaseNetDataset
        """
        try:
            # Map dataset names to SeisBench classes
            DATASET_MAP = {
                'STEAD': sbd.STEAD,
                'INSTANCE': sbd.InstanceCountsCombined,
                'ETHZ': sbd.ETHZ,
                'PNW': sbd.LenDB,  # PNW data is in LenDB
                'TXED': sbd.TXED
            }
            
            if dataset_name not in DATASET_MAP:
                raise ValueError(
                    f"Dataset {dataset_name} not supported. "
                    f"Choose from: {list(DATASET_MAP.keys())}"
                )
            
            # Load the dataset
            logger.info(f"Loading SeisBench dataset: {dataset_name}")
            dataset_class = DATASET_MAP[dataset_name]
            dataset = dataset_class()
            
            # Download if needed and specified
            if self.config.get('download_dataset', False):
                logger.info("Downloading dataset if needed...")
                dataset.download()
            
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            
            # Apply label error filtering if enabled
            filter_config = self.config.get('label_error_filtering', {})
            if filter_config.get('enabled', False):
                logger.info("Applying label error filtering...")
                label_filter = LabelErrorFilter(
                    cache_dir=filter_config.get('cache_dir', None)
                )
                
                dataset = label_filter.filter_dataset(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    include_multiplets=filter_config.get('filter_multiplets', True),
                    include_noise=filter_config.get('filter_noise', True)
                )
                
                logger.info(f"Dataset after filtering: {len(dataset)} samples")
            
            # For SeisBench datasets, we return the dataset object directly
            # and handle it differently in __getitem__
            return dataset, None
            
        except Exception as e:
            raise RuntimeError(
                f"Error loading SeisBench dataset {dataset_name}: {e}"
            )
    
    def _load_seisbench_data(self):
        """
        Load data from SeisBench benchmark datasets
        
        Supports: STEAD, INSTANCE, ETHZ, PNW, TXED
        """
        try:
            # Map dataset name to SeisBench class
            dataset_map = {
                'STEAD': 'STEAD',
                'INSTANCE': 'InstanceCountsCombined',
                'ETHZ': 'ETHZ',
                'PNW': 'LenDB',  # PNW data in LenDB
                'TXED': 'TXED'
            }
            
            dataset_name = self.config.get('dataset_name', 'STEAD')
            if dataset_name not in dataset_map:
                raise ValueError(
                    f"Dataset {dataset_name} not supported. "
                    f"Choose from: {list(dataset_map.keys())}"
                )
            
            # Load dataset from SeisBench
            dataset_class_name = dataset_map[dataset_name]
            print(f"Loading SeisBench dataset: {dataset_class_name}")
            
            # Get the dataset class
            dataset_class = getattr(sbd, dataset_class_name)
            dataset = dataset_class()
            
            # Download if needed
            if self.config.get('download_dataset', False):
                print("Downloading dataset...")
                dataset.download()
            
            # Extract waveforms and metadata
            print(f"Dataset loaded: {len(dataset)} samples")
            waveforms = []
            metadata = []
            
            for i in range(len(dataset)):
                trace = dataset.get_waveforms(i)
                picks = dataset.get_sample(i)
                
                waveforms.append(trace)
                metadata.append(picks)
            
            return waveforms, metadata
            
        except Exception as e:
            raise NotImplementedError(
                f"Error loading SeisBench data: {e}. "
                f"Please check dataset name and availability. "
                f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}"
            )
    
    def _load_custom_data(self):
        """
        Load data in custom format
        
        This should be implemented based on your specific data format.
        Expected to return:
            - data: List or array of waveforms
            - metadata: List of dicts with 'trace_p_arrival_sample' and 'trace_s_arrival_sample'
        """
        raise NotImplementedError(
            "Custom data loading not implemented. "
            "Please implement this method for your data format, or use SeisBench format data."
        )
    
    def _setup_augmentation(self):
        """Setup SeisBench data augmentation pipeline"""
        aug_config = self.config.get("augmentation", {})
        
        augmentations = []
        
        # Add noise
        if aug_config.get("noise_addition", False):
            augmentations.append(
                sbg.OneOf([
                    sbg.AddGaussianNoise(scale=aug_config.get("noise_level", 0.1)),
                    sbg.AddGapNoise(),
                ])
            )
        
        # Time shift
        if aug_config.get("time_shift", False):
            max_shift = aug_config.get("max_shift_samples", 50)
            augmentations.append(
                sbg.RandomShift(max_shift=max_shift, key=("X", "y"))
            )
        
        # Amplitude scaling
        if aug_config.get("amplitude_scaling", False):
            scale_range = aug_config.get("scale_range", [0.8, 1.2])
            augmentations.append(
                sbg.ScaleAmplitude(scale_range=scale_range)
            )
        
        # Create pipeline
        if augmentations:
            return sbg.Compose(augmentations)
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single training sample
        
        Returns:
            waveform: (3, window_length) tensor - Z, N, E components
            labels: (3, window_length) tensor - N, P, S probabilities (Gaussian around picks)
        """
        # Get waveform and metadata
        waveform = self.data[idx]
        meta = self.metadata[idx]
        
        # Ensure waveform is correct shape (3, samples)
        if isinstance(waveform, np.ndarray):
            if waveform.shape[0] != 3:
                waveform = waveform.T
        
        # Create labels (Gaussian distributions around P and S picks)
        labels = self._create_labels(meta)
        
        # Apply augmentation if enabled
        if self.augmentation_pipeline is not None:
            sample = {"X": waveform, "y": labels}
            sample = self.augmentation_pipeline(sample)
            waveform = sample["X"]
            labels = sample["y"]
        
        # Convert to tensors
        waveform = torch.from_numpy(waveform).float()
        labels = torch.from_numpy(labels).float()
        
        return waveform, labels
    
    def _create_labels(self, metadata, sigma=10):
        """
        Create Gaussian labels for phase picks
        
        Args:
            metadata: Dict with 'trace_p_arrival_sample' and 'trace_s_arrival_sample'
            sigma: Standard deviation of Gaussian in samples
        
        Returns:
            labels: Array of shape (3, window_length) for N, P, S channels
        """
        labels = np.zeros((3, self.window_length), dtype=np.float32)
        
        # Noise channel (default to 1, will be updated)
        labels[0, :] = 1.0
        
        x = np.arange(self.window_length)
        
        # P-wave label
        p_pick = metadata.get('trace_p_arrival_sample', None)
        if p_pick is not None and 0 <= p_pick < self.window_length:
            labels[1, :] = np.exp(-((x - p_pick) ** 2) / (2 * sigma ** 2))
        
        # S-wave label
        s_pick = metadata.get('trace_s_arrival_sample', None)
        if s_pick is not None and 0 <= s_pick < self.window_length:
            labels[2, :] = np.exp(-((x - s_pick) ** 2) / (2 * sigma ** 2))
        
        # Update noise channel: 1 - max(P, S)
        labels[0, :] = 1.0 - np.maximum(labels[1, :], labels[2, :])
        
        return labels


class PhaseNetDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for PhaseNet
    
    Supports loading from:
    1. SeisBench datasets: STEAD, INSTANCE, ETHZ, PNW, TXED
    2. Custom local data
    """
    
    # Mapping of our focused datasets
    SUPPORTED_DATASETS = {
        'STEAD': 'STEAD',
        'INSTANCE': 'INSTANCE', 
        'ETHZ': 'ETHZ',
        'PNW': 'PNW',
        'TXED': 'TXED'
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Check if using SeisBench datasets
        self.use_seisbench = config.get("data", {}).get("use_seisbench_dataset", False)
        self.dataset_name = config.get("data", {}).get("dataset_name", "STEAD")
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing
        """
        data_config = self.config.get("data", {})
        
        if self.use_seisbench:
            # Load directly from SeisBench dataset
            print(f"Using SeisBench dataset: {self.dataset_name}")
            self._setup_seisbench_datasets(stage)
        else:
            # Load from local data
            print("Using local data files")
            if stage == "fit" or stage is None:
                # Training dataset with augmentation
                self.train_dataset = PhaseNetDataset(
                    data_path=data_config.get("train_data"),
                    window_length=data_config.get("window_length", 3001),
                    augmentation=True,
                    config=data_config
                )
                
                # Validation dataset without augmentation
                self.val_dataset = PhaseNetDataset(
                    data_path=data_config.get("val_data"),
                    window_length=data_config.get("window_length", 3001),
                    augmentation=False,
                    config=data_config
                )
            
            if stage == "test" or stage is None:
                self.test_dataset = PhaseNetDataset(
                    data_path=data_config.get("test_data"),
                    window_length=data_config.get("window_length", 3001),
                    augmentation=False,
                    config=data_config
                )
    
    def _setup_seisbench_datasets(self, stage):
        """
        Setup datasets from SeisBench benchmark datasets
        Automatically splits into train/val/test
        """
        # Load the full dataset
        dataset_config = {'dataset_name': self.dataset_name}
        full_dataset = PhaseNetDataset(
            data_path="",  # Not used for SeisBench datasets
            window_length=self.config.get("data", {}).get("window_length", 3001),
            augmentation=False,
            config={**self.config.get("data", {}), **dataset_config}
        )
        
        # Split dataset (default: 70% train, 15% val, 15% test)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(self.config.get("seed", 42))
        train_data, val_data, test_data = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        if stage == "fit" or stage is None:
            self.train_dataset = train_data
            self.val_dataset = val_data
        
        if stage == "test" or stage is None:
            self.test_dataset = test_data
        
        print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
