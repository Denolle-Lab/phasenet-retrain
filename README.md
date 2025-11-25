# PhaseNet Retraining Framework

A PyTorch-based framework for retraining PhaseNet seismic phase detection models with automated label error filtering and support for multiple benchmark datasets.

## Overview

This repository provides a production-ready framework for retraining the PhaseNet deep learning model for seismic phase (P-wave and S-wave) arrival-time picking. Developed by the **Denolle Lab** at the University of Washington Department of Earth and Space Sciences, this framework responds to pervasive label errors in training datasets (Leonardo et al., 2024; [albertleonardo/labelerrors](https://github.com/albertleonardo/labelerrors)).

**Three-Step Workflow:**
1. **Dataset Acquisition**: Automatic loading of benchmark seismic datasets via SeisBench
2. **Label Error Filtering**: Removal of documented bad picks using validated methodology
3. **Model Retraining**: Efficient PyTorch-based training with Lightning framework

**Key Technology Stack:**
- **PyTorch 2.0+**: Deep learning framework ([pytorch/pytorch](https://github.com/pytorch/pytorch))
- **SeisBench 0.4.0+**: Seismological ML library ([seisbench/seisbench](https://github.com/seisbench/seisbench))
- **PyTorch Lightning 2.0+**: High-level training framework ([Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning))
- **ObsPy 1.4.0+**: Seismological data processing ([obspy/obspy](https://github.com/obspy/obspy))

## Key Features

### Data Quality & Filtering
- **Automated Label Error Filtering**: Removes documented problematic samples from training data
  - Multiplet errors: Samples with unlabeled earthquakes
  - Noise errors: Mislabeled noise containing seismic events
  - Based on validated [albertleonardo/labelerrors](https://github.com/albertleonardo/labelerrors) methodology
  - Improves model performance through cleaner training data

### Benchmark Datasets
Optimized support for 5 key SeisBench datasets:

| Dataset | Size | Coverage | Pretrained Model | Use Case |
|---------|------|----------|-----------------|----------|
| **STEAD** | ~1.2M traces | Global, M3-7 | ✅ `stead` | General-purpose picking |
| **INSTANCE** | Combined | Multi-source | ✅ `instance` | Diverse regional data |
| **ETHZ** | Regional | Switzerland | ✅ `ethz` | Local/regional events |
| **PNW** | Regional | Pacific Northwest | ⚙️ Custom | Cascadia subduction |
| **TXED** | Regional | Texas | ⚙️ Custom | Induced seismicity |

### Training Infrastructure
- **PyTorch Lightning Integration**: Structured, scalable training with automatic GPU/TPU support
- **Pretrained Models**: Start from STEAD, INSTANCE, or ETHZ weights for transfer learning
- **Data Augmentation**: SeisBench-integrated augmentation (noise, shifts, scaling)
- **Flexible Configuration**: YAML-based configuration for reproducible experiments
- **Logging & Monitoring**: TensorBoard and Weights & Biases integration
- **Cloud-Ready**: Optimized for AWS, GCP, and Azure deployment

## Project Structure

```
phasenet-retrain/
├── configs/              # Configuration files
│   └── train_config.yaml
├── data/                 # Data directory (train/val/test)
├── scripts/              # Training and utility scripts
│   ├── train.py         # Main training script
│   ├── model.py         # PhaseNet Lightning module
│   ├── data_module.py   # Data loading and processing
│   ├── utils.py         # Utility functions
│   └── evaluate.py      # Evaluation and inference
├── notebooks/            # Jupyter notebooks for analysis
├── models/               # Saved model architectures
├── checkpoints/          # Model checkpoints
├── results/              # Training results and logs
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Denolle-Lab/phasenet-retrain.git
cd phasenet-retrain
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

### Step 1: Dataset Acquisition

Load benchmark datasets automatically through SeisBench:

```yaml
# configs/train_config.yaml
data:
  dataset_name: "STEAD"  # Options: STEAD, INSTANCE, ETHZ, PNW, TXED
  use_seisbench_dataset: true
  download_dataset: true
```

Datasets are automatically downloaded and cached locally. No manual data preparation required.

### Step 2: Label Error Filtering

Enable automatic filtering of documented bad picks (Leonardo et al., 2024):

```yaml
data:
  label_error_filtering:
    enabled: true              # Remove documented label errors
    filter_multiplets: true    # Remove samples with unlabeled earthquakes
    filter_noise: true         # Remove mislabeled noise samples
```

The framework automatically:
1. Downloads CSV files documenting problematic samples from the [labelerrors repository](https://github.com/albertleonardo/labelerrors)
2. Filters out bad samples before training
3. Logs statistics on removed data

**Expected filtering rates:**
- STEAD: ~0.5-1% of samples removed
- INSTANCE: ~0.3-0.8% of samples removed
- PNW: ~0.4-0.9% of samples removed
- TXED: ~0.6-1.2% of samples removed
- ETHZ: ~0.2-0.5% of samples removed (no noise filtering)

### Step 3: Model Retraining

Configure and train with PyTorch Lightning:

```yaml
model:
  pretrained:
    use_pretrained: true
    model_name: "stead"       # Start from pretrained weights
    freeze_layers: []         # Empty for full fine-tuning

training:
  batch_size: 128
  max_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"

hardware:
  accelerator: "auto"         # Automatically detect GPU/CPU
  devices: 1
  precision: "32-true"
```

**Train the model:**
```bash
python scripts/train.py --config configs/train_config.yaml
```

**Resume from checkpoint:**
```bash
python scripts/train.py --config configs/train_config.yaml --resume checkpoints/best.ckpt
```

**Evaluate on test data:**
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --data data/test
```

### Verify Installation

Test that all dependencies are properly installed:
```bash
python scripts/test_setup.py
```

This checks PyTorch, SeisBench, PyTorch Lightning, and loads a pretrained model to verify everything works.

## Methods

### PhaseNet Architecture

PhaseNet ([wayneweiqiang/PhaseNet](https://github.com/wayneweiqiang/PhaseNet)) is a U-Net style convolutional neural network for seismic phase detection:
- **Input**: 3-component seismograms (Z, N, E channels)
- **Output**: 3 probability channels (Noise, P-wave, S-wave)
- **Architecture**: Encoder-decoder with skip connections
- **Implementation**: Native PyTorch via SeisBench ([seisbench/seisbench](https://github.com/seisbench/seisbench))

### Label Error Detection Methodology

Based on Leonardo et al. (2024) - [albertleonardo/labelerrors](https://github.com/albertleonardo/labelerrors):

1. **Detection**: Run pretrained PhaseNet and EQTransformer on benchmark datasets
2. **Identification**: Flag samples with inconsistencies:
   - More arrivals detected than labeled (multiplets)
   - Arrivals in samples labeled as noise
   - Significant timing discrepancies
3. **Validation**: Manual verification of flagged samples
4. **Distribution**: CSV files documenting bad sample indices per dataset

**Why this matters:**
- Unlabeled earthquakes in training data teach models to ignore events
- Mislabeled noise confuses the model's understanding of background vs. signal
- Inaccurate picks degrade model precision
- Training on cleaned data improves generalization and reduces false positives/negatives

### Transfer Learning Strategy

The framework supports multiple transfer learning approaches:

1. **Full Fine-tuning** (default): Train all layers on new dataset
2. **Frozen Encoder**: Freeze encoder layers, train decoder only
3. **Gradual Unfreezing**: Progressively unfreeze layers during training

Starting from pretrained weights (STEAD, INSTANCE, or ETHZ) significantly reduces training time and improves convergence, especially for smaller regional datasets like PNW and TXED.

## Advanced Configuration

### Data Augmentation

Enable robust augmentation through SeisBench:

```yaml
data:
  augmentation:
    use_augmentation: true
    noise_addition: true      # Add Gaussian noise
    noise_level: 0.1
    time_shift: true          # Random temporal shifts
    max_shift_samples: 50
    amplitude_scaling: true   # Random amplitude scaling
    scale_range: [0.8, 1.2]
```

### Preprocessing

Configure signal preprocessing:

```yaml
data:
  preprocessing:
    normalize: true
    filter: true
    filter_type: "bandpass"
    freqmin: 1.0              # Low-cut frequency (Hz)
    freqmax: 45.0             # High-cut frequency (Hz)
```

### Custom Datasets

To use your own data, implement the `_load_custom_data()` method in `scripts/data_module.py`:

```python
def _load_custom_data(self):
    """
    Load custom format data
    Returns:
        waveforms: List of 3-component waveforms
        metadata: List of dicts with phase pick information
    """
    # Your data loading implementation
    pass
```

## Logging and Monitoring

### TensorBoard
```bash
tensorboard --logdir results/
```

### Weights & Biases
Set `use_wandb: true` in the config and provide your project details:
```yaml
logging:
  use_wandb: true
  wandb_project: "phasenet-retrain"
  wandb_entity: "your-entity"
```

## Hardware Requirements

### Minimum Requirements
- **RAM**: 16 GB
- **Storage**: 50 GB free space (for datasets and models)
- **GPU**: Not required but strongly recommended for training

### Recommended for Training
- **RAM**: 32 GB+
- **Storage**: 200 GB+ SSD
- **GPU**: NVIDIA GPU with 8+ GB VRAM (e.g., RTX 3070, V100, A100)
- **Compute**: 8+ CPU cores for data loading

### Training Time Estimates

| Dataset | Samples | Epochs | GPU | Time |
|---------|---------|--------|-----|------|
| STEAD | 1.2M | 100 | V100 | ~48 hours |
| INSTANCE | 1.2M | 100 | V100 | ~48 hours |
| ETHZ | 300K | 100 | V100 | ~12 hours |
| PNW | 100K | 100 | V100 | ~4 hours |
| TXED | 50K | 100 | V100 | ~2 hours |

*Times assume batch_size=128 and label error filtering enabled. CPU-only training is 10-20x slower.*

## Output and Results

### Model Checkpoints

Checkpoints are automatically saved during training:
- **Location**: `checkpoints/` directory
- **Best model**: Based on validation loss (`best.ckpt`)
- **Last epoch**: Most recent training state (`last.ckpt`)
- **Top-k models**: Optional multiple checkpoint saving

### Results Structure

```
results/
├── training_logs/          # Loss curves and metrics
├── predictions/            # Model predictions on test data
└── evaluation_metrics/     # Precision, recall, F1 scores
```

### Model Performance Tracking

Monitor training progress with:
- **TensorBoard**: Real-time loss and accuracy visualization
- **CSV logs**: Epoch-by-epoch metrics
- **Weights & Biases**: Optional cloud-based tracking

## Extending the Framework

### Custom Data Sources

To use your own seismic data, modify `scripts/data_module.py`:

```python
def _load_custom_data(self):
    """
    Load data in custom format
    
    Returns:
        waveforms: Array of shape (n_samples, 3, n_timesteps)
        metadata: List of dicts with 'p_arrival_sample' and 's_arrival_sample'
    """
    # Implement your data loading logic
    pass
```

### Custom Loss Functions

Modify the loss function in `scripts/model.py` for specialized training objectives:

```python
def __init__(self, ...):
    # Replace default cross-entropy with custom loss
    self.criterion = CustomSeismicLoss()
```

Common alternatives:
- Focal loss for class imbalance
- Weighted cross-entropy for phase emphasis
- Combined detection + timing loss

## Documentation

- [Label Error Filtering Guide](docs/LABEL_ERROR_FILTERING.md) - Comprehensive guide to cleaning datasets
- [Dataset Documentation](docs/DATASETS.md) - Detailed information about each dataset
- [Implementation Details](docs/IMPLEMENTATION.md) - Technical architecture and design

## Citation

If you use this code, please cite:

**PhaseNet Original Paper:**
```bibtex
@article{zhu2019phasenet,
  title={PhaseNet: A deep-neural-network-based seismic arrival-time picking method},
  author={Zhu, Weiqiang and Beroza, Gregory C},
  journal={Geophysical Journal International},
  volume={216},
  number={1},
  pages={261--273},
  year={2019},
  publisher={Oxford University Press}
}
```

**SeisBench Library:**
```bibtex
@article{woollam2022seisbench,
  title={SeisBench—A toolbox for machine learning in seismology},
  author={Woollam, Jack and Rietbrock, Andreas and Bueno, Angela and De Angelis, Silvio},
  journal={Seismological Research Letters},
  volume={93},
  number={3},
  pages={1695--1709},
  year={2022},
  publisher={Seismological Society of America}
}
```

**Label Error Filtering:**
```bibtex
@misc{labelerrors2024,
  author = {Leonardo, Albert},
  title = {Pervasive Label Errors in Seismological Machine Learning Datasets},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/albertleonardo/labelerrors}
}
```

**PyTorch:**
```bibtex
@inproceedings{paszke2019pytorch,
  title={PyTorch: An imperative style, high-performance deep learning library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

**PyTorch Lightning:**
```bibtex
@misc{Falcon_PyTorch_Lightning_2019,
  author = {Falcon, William and {The PyTorch Lightning team}},
  title = {PyTorch Lightning},
  year = {2019},
  publisher = {GitHub},
  url = {https://github.com/Lightning-AI/pytorch-lightning}
}
```

## Contributing


## License


## Contact & Collaboration

**Denolle Lab**  
University of Washington  
Department of Earth and Space Sciences  

For questions, issues, or collaboration inquiries:
- Open an issue on GitHub
- Visit: [Denolle Lab Website](https://denolle-lab.github.io)

## Acknowledgments

- **Denolle Lab** at University of Washington for framework development and collaboration
- **Zhu & Beroza (2019)** for the original PhaseNet architecture ([wayneweiqiang/PhaseNet](https://github.com/wayneweiqiang/PhaseNet))
- **SeisBench** team for the comprehensive ML seismology toolkit ([seisbench/seisbench](https://github.com/seisbench/seisbench))
- **PyTorch Lightning** team for training infrastructure ([Lightning-AI/pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning))
- **PyTorch** team for the deep learning framework ([pytorch/pytorch](https://github.com/pytorch/pytorch))
- **ObsPy** developers for seismological data processing ([obspy/obspy](https://github.com/obspy/obspy))
- **Albert Leonardo** for label error detection methodology ([albertleonardo/labelerrors](https://github.com/albertleonardo/labelerrors))
