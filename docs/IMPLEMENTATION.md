# PhaseNet Implementation: Framework Update

## Summary of Changes

This document outlines how the phasenet-retrain framework has been adapted to use the **SeisBench PyTorch implementation** of PhaseNet.

## Key Points

### 1. **PyTorch, Not TensorFlow**
- SeisBench already implements PhaseNet in **PyTorch** (not TensorFlow)
- Original PhaseNet paper used TensorFlow, but SeisBench provides a native PyTorch implementation
- Our framework leverages this existing PyTorch implementation

### 2. **Architecture Source**
- Based on: https://github.com/seisbench/seisbench/blob/main/seisbench/models/phasenet.py
- U-Net style architecture with:
  - **Encoder**: 5 convolutional blocks with downsampling
  - **Bottleneck**: Deepest convolutional block  
  - **Decoder**: 4 upsampling blocks with skip connections
  - **Output**: 3 channels (Noise, P-wave, S-wave probabilities)

### 3. **Focused Datasets and Models**
This framework focuses on 5 key datasets from SeisBench:

**Primary Datasets:**
1. **STEAD** (Stanford Earthquake Dataset)
   - ~1.2M three-component waveforms
   - Global coverage, magnitude 3-7
   - Pretrained model available: `stead`
   
2. **INSTANCE** (Instance Counts Combined)
   - Combined dataset from multiple sources
   - Pretrained model available: `instance`
   
3. **ETHZ** (ETH Zurich)
   - Local and regional earthquakes
   - Swiss Seismological Service data
   - Pretrained model available: `ethz`
   
4. **PNW** (Pacific Northwest)
   - Accessed via LenDB in SeisBench
   - Regional US data
   - Can train custom model
   
5. **TXED** (Texas Earthquake Dataset)
   - Induced seismicity focus
   - Texas region
   - Can train custom model

**Pretrained Models Available:**
- `stead`: Best for general-purpose picking
- `instance`: Good for diverse data
- `ethz`: Optimized for local/regional events

## Updated Framework Components

### 1. **scripts/model.py**
- `PhaseNetLightning` class wraps SeisBench PhaseNet
- Supports loading pretrained models with `from_pretrained()`
- Implements PyTorch Lightning training/validation/test steps
- Handles proper loss calculation with soft labels (Gaussian distributions)
- Calculates per-phase metrics (N, P, S accuracy)
- Supports layer freezing for transfer learning

**Key Features:**
```python
# Load pretrained model
model = sbm.PhaseNet.from_pretrained('stead')

# Or create from scratch
model = sbm.PhaseNet(
    in_channels=3,
    classes=3, 
    phases="NPS",
    sampling_rate=100
)
```

### 2. **scripts/data_module.py**
- `PhaseNetDataModule` for PyTorch Lightning
- `PhaseNetDataset` with SeisBench data augmentation
- Supports SeisBench HDF5 format and custom formats
- Creates Gaussian labels around P and S picks
- Augmentation pipeline:
  - Noise addition
  - Time shifting
  - Amplitude scaling
  - Bandpass filtering

**Data Format:**
- Input: `(batch, 3, samples)` - Z, N, E components
- Labels: `(batch, 3, samples)` - N, P, S probabilities
- Default window: 3001 samples (30 seconds at 100 Hz)

### 3. **scripts/train.py**
- Command-line training script
- Loads YAML configuration
- Sets up PyTorch Lightning trainer
- Configures callbacks (checkpointing, early stopping)
- Supports TensorBoard and Weights & Biases logging

**Usage:**
```bash
python scripts/train.py --config configs/train_config.yaml
python scripts/train.py --config configs/train_config.yaml --resume checkpoint.ckpt
```

### 4. **scripts/evaluate.py**
- Model evaluation and inference
- Visualization of predictions
- Pick detection from probability curves
- Comprehensive metrics calculation

### 5. **scripts/utils.py**
- Data preprocessing utilities:
  - Normalization (std, minmax, max)
  - Bandpass filtering
  - Resampling
- Data augmentation:
  - Noise addition
  - Time shifting
  - Amplitude scaling
- Label creation (Gaussian distributions)

### 6. **configs/train_config.yaml**
Complete configuration including:
- Model settings (architecture, pretrained options)
- Training hyperparameters
- Data augmentation settings
- Hardware configuration (GPU/CPU/MPS)
- Logging options

### 7. **notebooks/01_phasenet_intro.ipynb**
Interactive tutorial demonstrating:
- Loading pretrained models
- Model architecture exploration
- Inference on synthetic data
- Training setup examples

### 8. **scripts/test_setup.py**
Diagnostic script to verify:
- PyTorch installation
- SeisBench availability
- PyTorch Lightning setup
- Model loading capability
- Dependencies

## PhaseNet Architecture Details

### Input
- Shape: `(batch, 3, 3001)` 
- 3 channels: Z (vertical), N (north), E (east)
- 3001 samples = 30 seconds at 100 Hz

### Encoder (Downsampling Path)
```
Layer 0: 3 -> 8 filters
Layer 1: 8 -> 16 filters (downsample by 4)
Layer 2: 16 -> 32 filters (downsample by 4)
Layer 3: 32 -> 64 filters (downsample by 4)
Layer 4: 64 -> 128 filters (bottleneck)
```

### Decoder (Upsampling Path)
```
Layer 0: 128 -> 64 filters (upsample by 4, concat with encoder layer 3)
Layer 1: 64 -> 32 filters (upsample by 4, concat with encoder layer 2)
Layer 2: 32 -> 16 filters (upsample by 4, concat with encoder layer 1)
Layer 3: 16 -> 8 filters (upsample by 4, concat with encoder layer 0)
```

### Output
- Shape: `(batch, 3, 3001)`
- 3 channels: Noise, P-wave, S-wave
- Softmax activation (probabilities sum to 1)

### Key Operations
- Convolutions: kernel_size=7, padding='same'
- Downsampling: stride=4 convolutions
- Upsampling: transposed convolutions
- Activation: ReLU
- Normalization: Batch normalization (eps=1e-3)
- Skip connections: concatenation

## Training Strategy

### Transfer Learning (Recommended)
1. Load pretrained model: `PhaseNet.from_pretrained('stead')`
2. Optionally freeze early layers
3. Fine-tune on your dataset
4. Lower learning rate (e.g., 1e-4)

### Training from Scratch
1. Initialize model: `PhaseNet(in_channels=3, classes=3)`
2. Larger dataset required
3. Standard learning rate (e.g., 1e-3)
4. More epochs needed

### Loss Function
- CrossEntropyLoss on soft labels
- Labels are Gaussian distributions around picks
- Sigma = 10 samples (default)

### Metrics
- Overall accuracy
- Per-phase accuracy (N, P, S)
- Validation loss
- Learning rate scheduling

## Next Steps for Users

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test setup:**
   ```bash
   python scripts/test_setup.py
   ```

3. **Prepare data:**
   - Organize in `data/train`, `data/val`, `data/test`
   - Implement `_load_custom_data()` in `scripts/data_module.py`

4. **Configure training:**
   - Edit `configs/train_config.yaml`
   - Set paths, hyperparameters, hardware

5. **Train model:**
   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```

6. **Monitor training:**
   ```bash
   tensorboard --logdir results/
   ```

7. **Evaluate:**
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --data data/test
   ```

## References

- **PhaseNet Paper**: Zhu & Beroza (2019) - https://doi.org/10.1093/gji/ggy423
- **SeisBench Library**: Woollam et al. (2022) - https://doi.org/10.1785/0220210324
- **SeisBench GitHub**: https://github.com/seisbench/seisbench
- **PyTorch**: https://pytorch.org
- **PyTorch Lightning**: https://lightning.ai
