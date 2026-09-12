# Augmentation transform contract (#43A)

*2026-09-11, branch `issue/43a-augmentation` from `issue/41a-label-policy` at
`b712541` (PR #66). Checkpoint 43A of issue #43: the transforms, the provider
protocols, the SNR definition, the config section and the fixtures. It wires
nothing into a loader and trains nothing.*

## Scope

`scripts/augmentation.py` implements the training-strategy recipe on the 41A
`WindowSample` contract (`scripts/arrivals.py`): waveform (3, n) ZNE float32,
`rate_hz`, arrivals in seconds from sample 0 with a provenance tier,
`component_mask`, `negative_support`, `unknown_intervals` [start, end) s,
`valid_samples`, `meta`. Pure numpy/scipy; no torch, h5py or SeisBench. Every
rate change goes through `waveform_contract.resample_waveform` (polyphase FIR,
Kaiser beta 5, output trimmed to the source support), the loader's resampler.

Every transform is a callable `(sample, rng: numpy.random.Generator) ->
WindowSample`. It returns a new object, never mutates its input, and moves
arrivals, unknown intervals, valid support and the component mask with the
waveform. A `prob` gate decides whether `apply` runs; a gated-off draw returns
a copy with no record. Applied transforms append a dict to
`meta["augmentations"]` (name plus the drawn parameters), so a stored seed
and the record reproduce every mixture.

## Transforms

| Transform | Waveform | Arrivals | unknown_intervals | valid_samples | component_mask / support |
|---|---|---|---|---|---|
| `RandomCrop(window_samples, anchor_range=(0.1, 0.7))` | crop of `window_samples` starting at a whole sample; a shorter window is zero-padded | shifted by the start; a supervising arrival (any arrival when none supervises) is placed at a fraction of the window drawn from `anchor_range`, so it is always inside; arrivals outside are dropped and listed in `meta["dropped_arrivals"]` (crop frame) | shifted and clipped to the crop | `min(max(n_valid - start, 0), window)`; padding lies beyond it | unchanged |
| `NoiseSuperposition(noise_provider, prob=0.6, snr_bins)` | adds provider noise scaled to a target SNR (definition below); noise zeroed on missing channels and beyond `valid_samples` | unchanged | unchanged | unchanged | unchanged; the provider states what pool it serves |
| `NonStationaryNoise(noise_provider=None, prob=0.3, snr_bins)` | provider noise (white when no provider) times an envelope: linear ramp, step at 20 to 80 % of the support, or a burst over at least 20 % of it with 0.5 s cosine tapers; scaled to the target SNR at full level | unchanged | unchanged | unchanged | unchanged |
| `EventSuperposition(sample_provider, prob=0.3, offset_s=(2, 40), amplitude_ratio=(0.1, 1.0))` | adds a second window scaled by one scalar so its peak is `amplitude_ratio` times the base peak; the second anchor (earliest supervising arrival, else earliest arrival) lands at the base anchor plus an offset of magnitude from `offset_s`, sign at random, restricted to what fits inside the base valid support; shift rounded to whole samples | base plus the second's shifted arrivals; those outside dropped and listed; a missing `event_id` becomes the second's `trace_name` or `"superposed"` | union (second's shifted and clipped) | unchanged (the base's): outside its own support the second window adds silence, neither invalid nor unknown; its span in the mixture's time base is recorded as `second_support_s` | mask: and, zeroing channels missing in either; support: weakest wins (unknown over reviewed over certified) |
| `Spike(prob=0.03, amplitude=(5, 50), width_samples=(1, 3))` | offset of `amplitude` times window RMS over 1 to 3 samples on one real channel (all, half the time) | unchanged | unchanged | unchanged | unchanged |
| `DCStep(prob=0.03, amplitude=(0.5, 5))` | step of `amplitude` times RMS on one real channel to the end of the support | unchanged | unchanged | unchanged | unchanged |
| `Gap(prob=0.03, duration_s=(0.1, 3.0))` | zeros on every channel over the gap | unchanged | gap appended, so `targets_for` masks it | unchanged | unchanged |
| `Clip(prob=0.03, level=(0.3, 0.9))` | clipped at `level` times the window peak (one threshold for all channels) | unchanged | unchanged | unchanged | unchanged |
| `MainsHum(prob=0.02, freq_hz=(50, 60), amplitude=(0.05, 0.5))` | sinusoid at a listed mains frequency, `amplitude` times RMS, random phase per channel, evaluated on the sample grid (at or above Nyquist it aliases as in a digitiser without anti-alias filtering; 60 Hz at 100 Hz sampling appears at 40 Hz; `aliased` is recorded) | unchanged | unchanged | unchanged | unchanged |
| `Drift(prob=0.03, amplitude=(1, 10), period_s=(60, 600))` | slow sinusoid, `amplitude` times RMS, random phase per channel | unchanged | unchanged | unchanged | unchanged |
| `ChannelDrop(prob=0.05, candidates=(0, 1, 2))` | one real channel zeroed; the last real channel is never dropped | unchanged | unchanged | unchanged | mask False on the dropped channel |
| `RateTransform(prob=0.3, rates=(20, 40, 50))` | valid support decimated to an intermediate rate below the sample rate and returned through `resample_waveform` both ways; no eligible rate is a recorded no-op | seconds unchanged (first-sample time preserved; pulse peak within one sample, measured 0 at 20, 40 and 50 Hz) | unchanged | shrinks by up to a few samples (the resampler trims to the source support); the rest is zero padding | unchanged |
| `BandLimit(prob=0.0, corner_hz=(8, 20), order=4)` | zero-phase Butterworth low-pass of the valid support, corner capped at 0.9 Nyquist | unchanged | unchanged | unchanged | unchanged |
| `AmplitudeJitter(scale=(0.5, 2.0))` | one log-uniform scalar on every channel | unchanged | unchanged | unchanged | unchanged |
| `PolarityFlip(prob=0.5)` | times -1 on every channel | unchanged | unchanged | unchanged | unchanged |

Reference amplitude for the artefacts: the RMS (spike, step, hum, drift) or
the peak (clip) over the real channels within the valid support. A window
with no real channel or zero amplitude records a skipped entry and passes
through.

Overlapping targets after superposition follow `label_targets.build_targets`:
per-channel maximum of the Gaussians, then renormalisation of the PSN triple
(docs/2026-09-11_41a_label_policy.md). The transforms only merge arrival
lists; they never build targets.

`Compose(transforms, untouched_fraction=0.3, always=None, pure_noise_fraction=0.0)`
applies `transforms` in order and then `always` (default `AmplitudeJitter`,
`PolarityFlip`). A draw below `untouched_fraction` skips `transforms`; the
output then differs from the input by a scalar and a sign. `meta["untouched"]`
records which. `pure_noise_fraction` is carried for the caller.

`mix_pure_noise(samples, noise_provider, fraction, rng, negative_support="reviewed")`
replaces `round(fraction * len(samples))` positions of a batch, drawn without
replacement, by arrival-free windows of the same length and rate from the
provider (`component_mask` True where the channel is not identically zero,
`meta["pure_noise"]`, `meta["noise_class"]`). `negative_support` is the
caller's statement about the pool; `reviewed` is right for a 42A pool and
wrong for anything unreviewed.

`augment_then_target(sample, pipeline, policy=MASKED, rng)` returns
`(waveform, targets, mask, info)` from `label_targets.targets_for` on the
augmented sample; `info` carries the target info, the augmentation records,
the dropped arrivals and the augmented sample under `"sample"`.

## Provider protocols

The caller implements these; the module ships reference implementations.

- `NoiseProvider.draw(rng, n_samples, rate_hz) -> (waveform (3, n_samples) float32 ZNE at rate_hz, class_name)`.
  Class-balanced sampling and any resampling of stored noise to `rate_hz`
  are the provider's job; the transform checks the shape and finiteness,
  then scales and adds. `ArrayNoiseProvider(pools, rate_hz)` draws a class
  uniformly, a member uniformly, resamples through `resample_waveform` when
  the rates differ, and returns a random contiguous segment; a member shorter
  than the request raises.
- `SampleProvider.draw(rng, n_samples, rate_hz) -> WindowSample` for event
  superposition, any length, resampled here (`resample_sample`, arrival
  seconds unchanged, padding dropped) when its rate differs.
  `ListSampleProvider(samples)` draws uniformly from a list.

Neither provider knows about evaluation intervals. Keeping calibration,
development and acceptance station-days out of the pools is the pool
builder's job (42A, #44), not the transform's.

## SNR definition

Signal power is the mean square of the source window over the real channels
in [tP - 0.5 s, tP + 2.0 s], clipped to the valid support, with tP the
earliest P arrival inside the support (supervising arrivals preferred over
automatic ones). Without a P the S window with the same offsets is used;
without either, the whole valid window. Noise power is the mean square of the
added noise in the same interval, so the realised SNR is local to the P
window, not whole-window unit variance. The target is drawn from
`snr_bins = ((weight, low_db, high_db), ...)`, default 0.4 in 0 to 5 dB, 0.3
in 5 to 10 dB, 0.3 in 10 to 25 dB; the noise scale is
`sqrt(P_signal / (P_noise * 10^(target/10)))`.

The source window is used as stored, its own background included, so the
realised SNR is relative to the source, not to a clean signal; when the
source has a catalogued SNR it stays in `meta` and the added-noise SNR is
written to `meta["realised_snr_db"]` and the augmentation record
(`snr_target_db`, `snr_db`, `snr_bin`, `snr_window`, `snr_window_s`,
`noise_class`, `scale`). Zero signal or noise power in the window records a
skipped entry. For `NonStationaryNoise` the target is defined at full level
in the same window and the realised value after the envelope (which can only
raise it; `None` when the envelope is zero there) is recorded.

## Config

`data.augmentation_v3`, parsed by `from_config(cfg, noise_provider=None,
sample_provider=None)`; the full config, the `data` section or the section
itself is accepted. A missing section or `enabled: false` gives an identity
pipeline (copy only, no jitter or flip); so does an empty section. A group is off when absent, `null`,
`false`, `enabled: false` or `prob: 0`; `amplitude_jitter` and `polarity_flip`
are on unless set to `null` or `false`. Enabling `noise` without a
`noise_provider`, or `event_superposition` without a `sample_provider`, is an
error; `nonstationary` without a provider uses white noise. Application
order is fixed: crop, event_superposition, noise, nonstationary, artefacts
(spike, dc_step, gap, clip, mains_hum, drift, channel_drop), rate_transform,
bandlimit, then jitter and flip. The rate transform comes after the additive
groups so the whole mixture is band-limited the way a native low-rate record
would be.

```yaml
data:
  augmentation_v3:
    untouched_fraction: 0.3        # applies only jitter and flip
    pure_noise_fraction: 0.15      # for mix_pure_noise, called by the batch builder
    crop: {window_samples: 3001, anchor_range: [0.1, 0.7]}
    noise:
      prob: 0.6
      snr_bins: [[0.4, 0, 5], [0.3, 5, 10], [0.3, 10, 25]]
    event_superposition: {prob: 0.3, offset_s: [2, 40], amplitude_ratio: [0.1, 1.0]}
    nonstationary: {prob: 0.3}     # snr_bins as above by default
    artefacts:
      spike: {prob: 0.03}
      dc_step: {prob: 0.03}
      gap: {prob: 0.03, duration_s: [0.1, 3.0]}
      clip: {prob: 0.03}
      mains_hum: {prob: 0.02}
      drift: {prob: 0.03}
      channel_drop: {prob: 0.05}
    rate_transform: {prob: 0.3, rates: [20, 40, 50]}
    bandlimit: {prob: 0.0, corner_hz: [8, 20]}
    amplitude_jitter: {scale: [0.5, 2.0]}
    polarity_flip: {prob: 0.5}
```

Every keyword of a transform's constructor is accepted in its group
(`Spike.amplitude`, `MainsHum.freq_hz`, `BandLimit.order`, ...). The
per-artefact defaults above are the strategy's 0.02 to 0.05 range; they are
pilot values for 47A, not measured optima. The seed is not part of the
section: the caller passes a `numpy.random.Generator` and stores its seed.

## Validation

```bash
python -m pytest tests/test_augmentation.py -q    # 28 tests, base python (3.9, numpy 2.0.2, scipy 1.13.1)
python -m pytest tests -q                          # 73 passed, 1 skipped (torch/SeisBench loader tests skip)
```

`tests/test_augmentation.py` builds Gaussian pulses (sigma 0.1 s, integer
sample times) on a 1e-3 white floor and checks, per transform and for a
composed pipeline (crop, superposition, noise, non-stationary noise, gap,
channel drop, rate transform, band limit) over ten seeds, that every retained
arrival outside an unknown interval still sits on its pulse peak within one
sample; dropped arrivals are listed; a gap is masked to 0 through
`targets_for`; the realised SNR is within 0.5 dB of the target (recomputed
from the added noise, not read from the record); superposition merges
arrivals with distinct event ids, unknown support wins, the second's unknown
interval follows the shift and is masked, a shorter second window leaves the
base's `valid_samples` and its late S supervised, and `component_mask` takes
the and; `RateTransform` keeps the pulse time (offset 0)
and attenuates energy above 1.2 times the intermediate Nyquist by more than
20 dB (measured 35 to 41 dB at 20, 40, 50 Hz on white noise) with the
passband within 0.5 dB (measured under 0.02 dB); `ChannelDrop` updates the
mask and never drops the last channel; `Compose` with `untouched_fraction=1`
changes the waveform only by a scalar and a sign; `augment_then_target`
targets peak at the arrival samples with mask 1 there; a seeded generator
reproduces the pipeline; no transform mutates its input; `mix_pure_noise`
replaces the requested fraction; `from_config` builds the documented order
and switches every group off.

## What 43A does not do

- The pipeline is not wired into `scripts/fast_manifest_dataset.py` or
  `scripts/manifest_dataset.py`. The RAM cache stores fixed 3001-sample crops
  and dense labels, so on-the-fly cropping from 120 s stored windows has
  nowhere to read from; that belongs to the 40A/45A store (disk-backed shards,
  per-batch labels), which serves `WindowSample`s that this pipeline consumes
  unchanged.
- No real noise pool or second-event pool exists yet (42A). The reference
  providers are for tests and pilots; the pool builders own class balance,
  units and response compatibility, and the exclusion of evaluation
  station-days (#44).
- No inspection sheet with real examples; it needs the 40A pilot corpus.
- Scientific value is 47A: paired seeds, equal update budgets, no-augmentation
  and white-noise controls, one group at a time. None of the probabilities
  above is validated.
