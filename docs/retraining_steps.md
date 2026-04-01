# Retraining PhaseNet

## Download datasets
Download relevant datasets used in PhaseNet training: STEAD, PNW, INSTANCE, TXED.
Where are they stored? PNWstore, Siletzia

## Remove unlabeled and mislabeled data with Leonardo et al. label error method
Use the method found here: https://github.com/albertleonardo/labelerrors.
It is also described in the paper here: https://arxiv.org/pdf/2511.09805.

## Visualize corrected dataset distributions
We will plot histograms of spectral content, P-pick time, S-pick time minus P-pick time (t_S - t_P), magnitude distribution, at 100 Hz sampling rate.

## Curate training dataset
We need:
- High SNR picks
- A diverse distribution of P-pick times across the analysis window to use all neurons.
- A distribution of earthquake magnitudes that respects true physics
- A diverse distribution of spectral content
- A diverse distribution of t_S - t_P

## Data augmentation
We will randomly turn off neurons, mute channels, and introduce data gaps to keep the model performance robust across instrumentation issues.

## Retrain PhaseNet on the new datasets, completely from scratch
Perform training on all curated datasets, but otherwise follow the method of the original PhaseNet:
https://github.com/AI4EPS/PhaseNet.
And the paper here: https://arxiv.org/pdf/1803.03211.

## Testing
We will test the model by picking on 20, 40, and 100 Hz data from different sites to test model generalizability.

## Site-specific implementation
Curate noise datasets for each site of interest, including cultural / anthropogenic noise, biological signals, explosions, 
and non-seismic events that will be excluded from our picks. 
Perform transfer learning on our new base PhaseNet model to learn the noise profile of the specific site.

### AI Documentation
We will use Claude throughout the development and implementation of this plan.
