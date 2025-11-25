# Focused Datasets Guide

This framework is optimized for 5 key datasets from SeisBench: **STEAD, INSTANCE, ETHZ, PNW, and TXED**.

## Dataset Overview

### 1. STEAD (Stanford Earthquake Dataset)
- **Full Name**: Stanford Earthquake Dataset
- **Size**: ~1.2 million three-component waveforms
- **Coverage**: Global
- **Magnitude Range**: 3.0 - 7.0
- **Sampling Rate**: 100 Hz
- **Features**: High-quality labeled data, diverse tectonic settings
- **Pretrained Model**: `stead` ✅
- **Best For**: General-purpose phase picking, transfer learning base

**Usage:**
```yaml
data:
  dataset_name: "STEAD"
  use_seisbench_dataset: true
  
model:
  pretrained:
    model_name: "stead"
```

**Citation:**
```
Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019).
STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic 
Signals for AI. IEEE Access, 7, 179464-179476.
```

---

### 2. INSTANCE (Instance Counts Combined)
- **Full Name**: Instance Counts Combined Dataset
- **Size**: Variable (combined from multiple sources)
- **Coverage**: Multiple regions
- **Features**: Diverse event types, multiple networks
- **Pretrained Model**: `instance` ✅
- **Best For**: Training on combined regional datasets

**Usage:**
```yaml
data:
  dataset_name: "INSTANCE"
  use_seisbench_dataset: true
  
model:
  pretrained:
    model_name: "instance"
```

---

### 3. ETHZ (ETH Zurich Dataset)
- **Full Name**: Swiss Seismological Service Dataset
- **Size**: ~30,000+ events
- **Coverage**: Switzerland and surrounding regions (local/regional)
- **Sampling Rate**: 100 Hz
- **Features**: High-quality local/regional data, diverse event sizes
- **Pretrained Model**: `ethz` ✅
- **Best For**: Local/regional seismicity, European tectonic settings

**Usage:**
```yaml
data:
  dataset_name: "ETHZ"
  use_seisbench_dataset: true
  
model:
  pretrained:
    model_name: "ethz"
```

**Citation:**
```
Swiss Seismological Service (SED) at ETH Zurich.
```

---

### 4. PNW (Pacific Northwest)
- **Full Name**: Pacific Northwest Seismic Data (via LenDB)
- **Size**: Large regional dataset
- **Coverage**: Pacific Northwest United States
- **Features**: Subduction zone, volcanic activity, local events
- **Pretrained Model**: Train your own
- **Best For**: Cascadia subduction zone, western US seismicity

**Usage:**
```yaml
data:
  dataset_name: "PNW"
  use_seisbench_dataset: true
  
model:
  pretrained:
    use_pretrained: true
    model_name: "stead"  # Start with STEAD, fine-tune on PNW
```

**Note:** Access through LenDB in SeisBench. May require additional setup.

---

### 5. TXED (Texas Earthquake Dataset)
- **Full Name**: Texas Earthquake Dataset
- **Size**: Focused regional dataset
- **Coverage**: Texas, United States
- **Features**: Induced seismicity, oil/gas production areas
- **Pretrained Model**: Train your own
- **Best For**: Induced seismicity, Texas region monitoring

**Usage:**
```yaml
data:
  dataset_name: "TXED"
  use_seisbench_dataset: true
  
model:
  pretrained:
    use_pretrained: true
    model_name: "stead"  # Start with STEAD, fine-tune on TXED
```

---

## Comparison Table

| Dataset | Size | Coverage | Pretrained | Best Use Case |
|---------|------|----------|------------|---------------|
| **STEAD** | ~1.2M | Global | ✅ Yes | General purpose, transfer learning |
| **INSTANCE** | Variable | Multi-region | ✅ Yes | Combined regional data |
| **ETHZ** | ~30K+ | Switzerland | ✅ Yes | Local/regional European events |
| **PNW** | Large | Pacific NW | ❌ No | Cascadia, western US |
| **TXED** | Moderate | Texas | ❌ No | Induced seismicity |

---

## Training Strategies

### Strategy 1: Use Pretrained Model (Fastest)
Best for: Quick deployment, similar data to pretrained model

```yaml
model:
  pretrained:
    use_pretrained: true
    model_name: "stead"  # or instance, ethz
    freeze_layers: []
```

No training needed, use directly for inference.

---

### Strategy 2: Fine-tune Pretrained Model (Recommended)
Best for: Your data differs somewhat from pretrained dataset

```yaml
model:
  pretrained:
    use_pretrained: true
    model_name: "stead"
    freeze_layers: []  # Or freeze early layers: ["down_branch.0", "down_branch.1"]

training:
  learning_rate: 0.0001  # Lower LR for fine-tuning
  max_epochs: 50
```

**Steps:**
1. Load pretrained model (e.g., STEAD)
2. Optionally freeze early layers
3. Train on your target dataset (e.g., PNW or TXED)
4. Use lower learning rate

---

### Strategy 3: Train from Scratch
Best for: Very different data characteristics

```yaml
model:
  pretrained:
    use_pretrained: false

training:
  learning_rate: 0.001  # Standard LR
  max_epochs: 100
```

Requires more data and training time.

---

## Loading Datasets in Code

### Load from SeisBench
```python
from data_module import PhaseNetDataModule

config = {
    "data": {
        "dataset_name": "STEAD",
        "use_seisbench_dataset": True,
        "download_dataset": True,
        "window_length": 3001
    },
    "training": {
        "batch_size": 128,
        "num_workers": 4
    },
    "seed": 42
}

data_module = PhaseNetDataModule(
    config=config,
    batch_size=128,
    num_workers=4
)

data_module.setup("fit")
train_loader = data_module.train_dataloader()
```

### Access Dataset Statistics
```python
import seisbench.data as sbd

# Load STEAD
stead = sbd.STEAD()
print(f"STEAD samples: {len(stead)}")
print(f"Metadata columns: {stead.metadata.columns.tolist()}")

# Load INSTANCE
instance = sbd.InstanceCountsCombined()
print(f"INSTANCE samples: {len(instance)}")

# Load ETHZ
ethz = sbd.ETHZ()
print(f"ETHZ samples: {len(ethz)}")
```

---

## Dataset-Specific Considerations

### STEAD
- **Pros**: Largest, most diverse, best pretrained model
- **Cons**: Global → may not be optimized for specific regions
- **Recommendation**: Start here for most applications

### INSTANCE
- **Pros**: Multi-source data, good variety
- **Cons**: Less documentation on composition
- **Recommendation**: Good alternative to STEAD

### ETHZ
- **Pros**: High-quality local data, well-documented
- **Cons**: Regional focus (Switzerland)
- **Recommendation**: Best for European/Alpine seismicity

### PNW
- **Pros**: Subduction zone data, relevant for Cascadia
- **Cons**: No pretrained model, requires training
- **Recommendation**: Fine-tune STEAD model on PNW data

### TXED
- **Pros**: Focused on induced seismicity
- **Cons**: Smaller regional scope, no pretrained model
- **Recommendation**: Fine-tune STEAD model on TXED data

---

## Example Workflows

### Workflow 1: Train on STEAD
```bash
# Edit config
# Set: dataset_name: "STEAD", use_seisbench_dataset: true

# Train
python scripts/train.py --config configs/train_config.yaml
```

### Workflow 2: Fine-tune STEAD on PNW
```bash
# Step 1: Edit config for PNW data
# dataset_name: "PNW"
# model_name: "stead"
# learning_rate: 0.0001

# Step 2: Train
python scripts/train.py --config configs/train_config.yaml
```

### Workflow 3: Compare Models
```bash
# Train on STEAD
python scripts/train.py --config configs/stead_config.yaml

# Train on INSTANCE  
python scripts/train.py --config configs/instance_config.yaml

# Train on ETHZ
python scripts/train.py --config configs/ethz_config.yaml

# Compare results
python scripts/compare_models.py
```

---

## Data Download and Caching

SeisBench automatically caches downloaded datasets:
- **Location**: `~/.seisbench/` (Unix/Mac) or `%USERPROFILE%\.seisbench\` (Windows)
- **Size**: 
  - STEAD: ~100 GB
  - INSTANCE: Varies
  - ETHZ: ~10 GB
  - PNW: Varies
  - TXED: Varies

**First download** will take time. Subsequent loads use cached data.

---

## References

- **SeisBench**: https://github.com/seisbench/seisbench
- **STEAD Paper**: Mousavi et al. (2019), IEEE Access
- **PhaseNet Paper**: Zhu & Beroza (2019), GJI
- **SeisBench Paper**: Woollam et al. (2022), SRL
