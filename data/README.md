# Data Directory

This directory contains the training, validation, and test datasets for PhaseNet retraining.

## Expected Structure

```
data/
├── train/          # Training data
├── val/            # Validation data
├── test/           # Test data
└── README.md       # This file
```

## Data Format

The data loading implementation in `scripts/data_module.py` needs to be customized based on your specific data format. Common formats include:

- **SeisBench format**: HDF5 files with standardized structure
- **MSEED files**: Raw waveform data with metadata
- **SAC files**: Seismic Analysis Code format
- **Custom HDF5/NPY**: Your own data structure

## Data Requirements

Each waveform should include:
- **3 channels**: Z (vertical), N (north), E (east)
- **Sampling rate**: 100 Hz (default, configurable)
- **Window length**: ~30 seconds (3001 samples at 100 Hz)
- **Phase picks**: P and S arrival times

## Preparing Your Data

1. **Organize raw data**: Place your seismic data files in appropriate subdirectories
2. **Create metadata**: Prepare catalogs with event information and phase picks
3. **Preprocess**: Apply any necessary preprocessing (filtering, normalization)
4. **Split datasets**: Divide into train/val/test sets (typical: 70/15/15 split)

## Example Data Loading

Customize the `_load_dataset` method in `scripts/data_module.py`:

```python
def _load_dataset(self):
    # Example: Load from SeisBench format
    from seisbench.data import WaveformDataset
    dataset = WaveformDataset(self.data_path)
    return dataset
```

## Data Augmentation

Data augmentation is applied during training to improve model robustness:
- Noise addition
- Time shifting
- Amplitude scaling
- Bandpass filtering

Configure augmentation in `configs/train_config.yaml`.

## Large Datasets

For large datasets that don't fit in memory:
- Use lazy loading with generators
- Implement on-the-fly data loading in the Dataset class
- Consider using memory-mapped files (e.g., HDF5)

## Citation

If using public datasets, please cite the original data sources appropriately.
