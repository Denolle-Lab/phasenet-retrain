"""
Label Error Filtering Module for PhaseNet Retraining

This module provides functionality to filter out problematic samples from seismic datasets
based on label errors documented in the albertleonardo/labelerrors repository.

The methodology:
1. Use pretrained PhaseNet and EQTransformer models to identify issues
2. Detect unlabeled earthquakes in event samples (multiplets)
3. Detect earthquakes in noise samples
4. Identify inaccurate or missing labels

Reference: https://github.com/albertleonardo/labelerrors
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class LabelErrorFilter:
    """
    Filter for removing samples with known label errors from seismic datasets.
    
    This class manages loading and applying filters from the labelerrors repository
    to clean training data before PhaseNet retraining.
    """
    
    # GitHub raw content base URL for the labelerrors repository
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/albertleonardo/labelerrors/main"
    
    # Mapping of dataset names to their error report files
    ERROR_REPORTS = {
        "STEAD": {
            "multiplets": "multiplet_reports/stead_multiplets.csv",
            "noise": "noise_reports/stead_noise.csv"
        },
        "INSTANCE": {
            "multiplets": "multiplet_reports/instance_multiplets.csv", 
            "noise": "noise_reports/instance_noise.csv"
        },
        "PNW": {
            "multiplets": "multiplet_reports/pnw_multiplets.csv",
            "noise": "noise_reports/pnw_noise.csv"
        },
        "TXED": {
            "multiplets": "multiplet_reports/txed_multiplets.csv",
            "noise": "noise_reports/txed_noise.csv"
        },
        "ETHZ": {
            "multiplets": "multiplet_reports/ethz_multiplets.csv",
            "noise": None  # ETHZ doesn't have noise samples
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the label error filter.
        
        Args:
            cache_dir: Directory to cache downloaded error reports. 
                      Defaults to ~/.cache/phasenet_retrain/label_errors/
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/phasenet_retrain/label_errors")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store loaded error indices
        self.error_indices: Dict[str, Set[int]] = {}
        
    def download_error_report(self, dataset_name: str, report_type: str) -> Optional[Path]:
        """
        Download error report CSV file from GitHub if not already cached.
        
        Args:
            dataset_name: Name of dataset (e.g., "STEAD", "INSTANCE")
            report_type: Type of report ("multiplets" or "noise")
            
        Returns:
            Path to cached CSV file, or None if report doesn't exist
        """
        dataset_upper = dataset_name.upper()
        
        if dataset_upper not in self.ERROR_REPORTS:
            logger.warning(f"No error reports available for dataset: {dataset_name}")
            return None
            
        report_path = self.ERROR_REPORTS[dataset_upper].get(report_type)
        if report_path is None:
            logger.info(f"No {report_type} report for {dataset_name}")
            return None
            
        # Check cache first
        cache_file = self.cache_dir / f"{dataset_name}_{report_type}.csv"
        if cache_file.exists():
            logger.info(f"Using cached error report: {cache_file}")
            return cache_file
            
        # Download from GitHub
        url = f"{self.GITHUB_RAW_BASE}/{report_path}"
        logger.info(f"Downloading error report from: {url}")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(url, cache_file)
            logger.info(f"Downloaded error report to: {cache_file}")
            return cache_file
        except Exception as e:
            logger.error(f"Failed to download error report: {e}")
            return None
            
    def load_error_indices(self, dataset_name: str, 
                          include_multiplets: bool = True,
                          include_noise: bool = True) -> Set[int]:
        """
        Load indices of samples with label errors for a dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., "STEAD", "INSTANCE")
            include_multiplets: Include multiplet error indices
            include_noise: Include noise sample error indices
            
        Returns:
            Set of integer indices to exclude from training
        """
        cache_key = f"{dataset_name}_{'M' if include_multiplets else ''}{'N' if include_noise else ''}"
        
        if cache_key in self.error_indices:
            return self.error_indices[cache_key]
            
        error_idx = set()
        
        # Load multiplet errors
        if include_multiplets:
            multiplet_file = self.download_error_report(dataset_name, "multiplets")
            if multiplet_file and multiplet_file.exists():
                try:
                    df = pd.read_csv(multiplet_file)
                    # Assuming CSV has an 'index' or similar column with bad sample indices
                    if 'index' in df.columns:
                        indices = df['index'].values
                    elif 'trace_name' in df.columns:
                        # May need to parse trace names to get indices
                        indices = df.index.values
                    else:
                        # Use row numbers as indices
                        indices = df.index.values
                    error_idx.update(indices)
                    logger.info(f"Loaded {len(indices)} multiplet errors for {dataset_name}")
                except Exception as e:
                    logger.error(f"Error loading multiplet report: {e}")
                    
        # Load noise errors  
        if include_noise:
            noise_file = self.download_error_report(dataset_name, "noise")
            if noise_file and noise_file.exists():
                try:
                    df = pd.read_csv(noise_file)
                    if 'index' in df.columns:
                        indices = df['index'].values
                    elif 'trace_name' in df.columns:
                        indices = df.index.values
                    else:
                        indices = df.index.values
                    error_idx.update(indices)
                    logger.info(f"Loaded {len(indices)} noise errors for {dataset_name}")
                except Exception as e:
                    logger.error(f"Error loading noise report: {e}")
                    
        self.error_indices[cache_key] = error_idx
        logger.info(f"Total error indices for {dataset_name}: {len(error_idx)}")
        
        return error_idx
        
    def filter_dataset(self, dataset, dataset_name: str,
                      include_multiplets: bool = True,
                      include_noise: bool = True) -> 'Dataset':
        """
        Filter a SeisBench dataset to remove samples with label errors.
        
        Args:
            dataset: SeisBench dataset object
            dataset_name: Name of the dataset
            include_multiplets: Filter out multiplet errors
            include_noise: Filter out noise errors
            
        Returns:
            Filtered dataset with problematic samples removed
        """
        error_idx = self.load_error_indices(dataset_name, include_multiplets, include_noise)
        
        if len(error_idx) == 0:
            logger.info(f"No error indices to filter for {dataset_name}")
            return dataset
            
        # Filter the dataset metadata
        original_len = len(dataset.metadata)
        
        # Create boolean mask for samples to keep
        keep_mask = ~dataset.metadata.index.isin(error_idx)
        
        # Apply filter
        dataset.metadata = dataset.metadata[keep_mask].reset_index(drop=True)
        
        filtered_len = len(dataset.metadata)
        removed = original_len - filtered_len
        
        logger.info(f"Filtered {dataset_name}: removed {removed} samples "
                   f"({removed/original_len*100:.2f}%), kept {filtered_len}")
        
        return dataset
        
    def get_filter_statistics(self, dataset_name: str) -> Dict[str, int]:
        """
        Get statistics about available error filters for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with counts of different error types
        """
        stats = {
            'multiplet_errors': 0,
            'noise_errors': 0,
            'total_errors': 0
        }
        
        # Count multiplet errors
        multiplet_idx = self.load_error_indices(dataset_name, include_multiplets=True, include_noise=False)
        stats['multiplet_errors'] = len(multiplet_idx)
        
        # Count noise errors
        noise_idx = self.load_error_indices(dataset_name, include_multiplets=False, include_noise=True)
        stats['noise_errors'] = len(noise_idx)
        
        # Total unique errors
        all_idx = self.load_error_indices(dataset_name, include_multiplets=True, include_noise=True)
        stats['total_errors'] = len(all_idx)
        
        return stats


def create_label_error_filter(cache_dir: Optional[str] = None) -> LabelErrorFilter:
    """
    Factory function to create a label error filter.
    
    Args:
        cache_dir: Optional custom cache directory
        
    Returns:
        Initialized LabelErrorFilter instance
    """
    return LabelErrorFilter(cache_dir=cache_dir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Label Error Filter - Test Run")
    print("=" * 60)
    
    filter_obj = create_label_error_filter()
    
    # Test with each dataset
    for dataset_name in ["STEAD", "INSTANCE", "PNW", "TXED", "ETHZ"]:
        print(f"\n{dataset_name}:")
        stats = filter_obj.get_filter_statistics(dataset_name)
        print(f"  Multiplet errors: {stats['multiplet_errors']}")
        print(f"  Noise errors: {stats['noise_errors']}")
        print(f"  Total errors: {stats['total_errors']}")
