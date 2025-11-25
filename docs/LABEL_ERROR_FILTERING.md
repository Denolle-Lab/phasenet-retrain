# Label Error Filtering for PhaseNet Retraining

## Overview

This document describes the label error filtering methodology implemented in this PhaseNet retraining framework. The approach is based on the work documented in the [albertleonardo/labelerrors](https://github.com/albertleonardo/labelerrors) repository, which identifies pervasive label errors in seismological machine learning datasets.

## Motivation

Seismic datasets contain significant label errors that can negatively impact model performance:

1. **Unlabeled earthquakes in event samples (multiplets)**: Training data may contain multiple earthquakes in a single window, but only one is labeled. The model learns to ignore unlabeled events, which is exactly what we DON'T want.

2. **Earthquakes in noise samples**: Samples labeled as "noise" may contain unlabeled seismic events, confusing the model about what constitutes noise.

3. **Inaccurate labels**: Some P/S phase picks are incorrect due to:
   - Archival issues
   - Instrumental problems
   - Human picking errors
   - Timing inaccuracies

## Performance Improvements

Research has shown that performance improvements from cleaning datasets can exceed gains from adding model complexity with fixed datasets. By training on cleaner data, models learn more accurate representations of seismic phases.

## Methodology

### 1. Error Detection

The labelerrors methodology uses:
- **PhaseNet** (pretrained on original datasets)
- **EQTransformer** (pretrained on original datasets)

These models are run on each dataset to detect:
- Extra arrivals not in labels → multiplet errors
- Arrivals in noise samples → mislabeled noise
- Discrepancies between model picks and labels → inaccurate labels

### 2. Error Types

#### Multiplet Errors
Samples where pretrained models detect more P/S arrivals than are labeled in the metadata.

**Example**: A 3-channel waveform has labels for 1 P and 1 S arrival, but PhaseNet detects 3 P arrivals and 2 S arrivals, indicating unlabeled earthquakes in the sample.

#### Noise Sample Errors  
Samples categorized as "noise" where pretrained models detect seismic phase arrivals.

**Example**: A sample in the "noise" category shows clear P and S arrivals when processed by PhaseNet and EQTransformer, indicating it should be an "event" sample.

#### Inaccurate Labels
Samples where:
- Labels exist but the data doesn't contain the corresponding arrivals (archival issues)
- Pick times are significantly off from actual phase arrivals
- Wrong phase types are labeled

### 3. Dataset-Specific Reports

The labelerrors repository provides CSV files containing indices of problematic samples for each dataset:

```
multiplet_reports/
├── stead_multiplets.csv
├── instance_multiplets.csv
├── pnw_multiplets.csv
├── txed_multiplets.csv
└── ethz_multiplets.csv

noise_reports/
├── stead_noise.csv
├── instance_noise.csv
├── pnw_noise.csv
└── txed_noise.csv
```

Each CSV file contains references to bad examples using SeisBench metadata identifiers.

## Implementation

### Basic Usage

Label error filtering is controlled via the configuration file:

```yaml
data:
  dataset_name: "STEAD"
  use_seisbench_dataset: true
  
  label_error_filtering:
    enabled: true  # Enable filtering
    filter_multiplets: true  # Remove multiplet errors
    filter_noise: true  # Remove noise errors
    cache_dir: null  # Optional custom cache directory
```

### How It Works

1. **Download Error Reports**: CSV files are automatically downloaded from GitHub and cached locally
2. **Load Bad Indices**: Problematic sample indices are loaded for the specified dataset
3. **Filter Dataset**: Bad samples are removed from the dataset metadata before training
4. **Training**: Model trains only on cleaned data

### Code Example

```python
from label_error_filter import LabelErrorFilter
import seisbench.data as sbd

# Create filter instance
label_filter = LabelErrorFilter()

# Load STEAD dataset
dataset = sbd.STEAD()
print(f"Original dataset size: {len(dataset)}")

# Apply filtering
filtered_dataset = label_filter.filter_dataset(
    dataset=dataset,
    dataset_name="STEAD",
    include_multiplets=True,
    include_noise=True
)
print(f"Filtered dataset size: {len(filtered_dataset)}")

# Get statistics
stats = label_filter.get_filter_statistics("STEAD")
print(f"Multiplet errors: {stats['multiplet_errors']}")
print(f"Noise errors: {stats['noise_errors']}")
print(f"Total removed: {stats['total_errors']}")
```

### Manual Filter Testing

Test the label error filter before training:

```bash
cd scripts
python label_error_filter.py
```

This will display statistics for all supported datasets.

## Expected Results

### STEAD Dataset
- Multiplet errors: ~X samples (actual count from labelerrors repo)
- Noise errors: ~Y samples
- Total filtered: ~Z%

### INSTANCE Dataset
- Multiplet errors: ~X samples
- Noise errors: ~Y samples  
- Total filtered: ~Z%

### PNW Dataset
- Multiplet errors: ~X samples
- Noise errors: ~Y samples
- Total filtered: ~Z%

### TXED Dataset
- Multiplet errors: ~X samples
- Noise errors: ~Y samples
- Total filtered: ~Z%

### ETHZ Dataset
- Multiplet errors: ~X samples
- No noise samples (ETHZ doesn't have separate noise category)
- Total filtered: ~Z%

*Note: Run `python scripts/label_error_filter.py` to see actual counts*

## Training Recommendations

### With Label Filtering (Recommended)
```yaml
data:
  label_error_filtering:
    enabled: true
    filter_multiplets: true
    filter_noise: true
```

**Benefits**:
- Cleaner training signal
- Better generalization
- Fewer false positives/negatives
- More robust to unseen data

### Without Label Filtering
```yaml
data:
  label_error_filtering:
    enabled: false
```

**Use when**:
- Comparing with baseline models trained on original data
- Researching the impact of label errors
- Working with custom datasets not covered by labelerrors

### Selective Filtering
Filter only specific error types:

```yaml
# Only remove multiplet errors
data:
  label_error_filtering:
    enabled: true
    filter_multiplets: true
    filter_noise: false

# Or only remove noise errors
data:
  label_error_filtering:
    enabled: true
    filter_multiplets: false
    filter_noise: true
```

## Cache Management

Error reports are cached to avoid repeated downloads:

**Default cache location**: `~/.cache/phasenet_retrain/label_errors/`

**Custom cache location**:
```yaml
data:
  label_error_filtering:
    cache_dir: "/path/to/custom/cache"
```

**Clear cache**:
```bash
rm -rf ~/.cache/phasenet_retrain/label_errors/
```

## Citation

If you use this label error filtering in your research, please cite:

```bibtex
@misc{labelerrors2024,
  author = {Leonardo, Albert},
  title = {Pervasive Label Errors in Seismological Machine Learning Datasets},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/albertleonardo/labelerrors}
}
```

Also see related work on data-centric AI:
- [https://labelerrors.com/](https://labelerrors.com/) - Label errors in other ML domains
- Andrew Ng's talks on data-centric AI (minute 5-10 of [this video](https://www.youtube.com/watch?v=06-AZXmwHjo))

## Future Work

- Automated detection of label errors using ensemble methods
- Active learning to identify and correct borderline samples
- Per-sample confidence scores based on label quality
- Extension to other seismic datasets beyond the focused five

## Troubleshooting

### Issue: Downloads fail
**Solution**: Check internet connection and GitHub accessibility. Error reports can also be manually downloaded from https://github.com/albertleonardo/labelerrors

### Issue: Cache directory permissions
**Solution**: Specify a custom cache directory with write permissions in the config

### Issue: Unknown dataset name
**Solution**: Ensure dataset_name is one of: STEAD, INSTANCE, ETHZ, PNW, TXED (case-sensitive)

### Issue: CSV format mismatch
**Solution**: Clear cache and re-download. The labelerrors repository may have updated their format.

## References

1. albertleonardo/labelerrors: https://github.com/albertleonardo/labelerrors
2. PhaseNet paper: Zhu & Beroza (2019)
3. EQTransformer paper: Mousavi et al. (2020)
4. SeisBench: Woollam et al. (2022)
