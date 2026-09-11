# General picker: checkpoint 34A loader repair

This is the first implementation step in [issue #34](https://github.com/Denolle-Lab/phasenet-retrain/issues/34), based on `audit/2026-09-07-generalization` at `6b5297c`. The implementation branch is `issue/34a-loader-alignment`. It belongs to the general earthquake picker. SU work remains on its separate roadmap.

## What is repaired

The previous loader could resample a waveform while leaving P/S indices in source coordinates, assume 100 Hz for direct HDF5 reads, interpret an S-only record as noise, and replace failed reads with zero waveforms and noise labels. These are code defects; this repair does **not** establish how many historical v7 rows were affected or which defect caused its generalization loss.

The loader now:

- Resolves stored waveform rate from per-trace CSV metadata or HDF5 attributes, with file `data_format` as a fallback. Unknown rates fail. Conflicting rate aliases or per-trace CSV/HDF5 rates fail. A verified per-trace rate can override a file default.
- Explicitly requests stored-rate samples from SeisBench, even when its reader has a configured output rate. This prevents an implicit first resampling followed by a second conversion using stale metadata.
- Reads ordinary and bucket-indexed HDF5 traces, preserves chunk identities, trims declared `trace_npts`, and opens HDF5 handles separately in each worker process.
- Uses explicit component and dimension metadata to produce ZNE/CW arrays. Missing components are zero-filled and reported in a component mask. Unknown orientation, duplicate channels, unsupported 1/2 orientation, inconsistent shape, and non-finite waveforms fail.
- Converts each permitted arrival from its index rate to seconds, then to the target grid, then subtracts the crop origin. Fractional arrival positions are retained. A missing P can use S as the crop anchor.
- Rejects a signal row with no usable arrival instead of relabeling it as noise. Only the explicit `noise_global` and `noise_prephase` sources are treated as noise; those rows must have no supplied arrival labels.
- Writes a per-process rejection ledger and raises an exception on the first failed row. Cached loading propagates the exception, so an incomplete cache cannot be used for training.

The manifest builder retains S-only rows where the existing source/distance policy permits S. It carries known source rate, sample interval, component order, dimension order, support length, start time, and an explicit arrival-index rate through to exported manifests. Its existing teleseismic P-only rule still applies; S-only teleseismic rows therefore have no permitted label and are removed.

## Coordinate contract

| Quantity | Meaning |
| --- | --- |
| `trace_sampling_rate_hz`, `sampling_rate`, or inverse `trace_dt_s` | Rate of the waveform stored in the source, resolved from verified metadata. This does not reconstruct the instrument's earlier recording/resampling history. |
| Effective reader rate | Returned waveform rate. SeisBench is explicitly requested to return the stored rate; disagreement fails. |
| `arrival_sampling_rate_hz` | Optional manifest override describing the arrival indices' grid. If absent, indices refer to the stored waveform grid. An override needs source evidence, not a dataset-name guess. |
| Trace origin | Arrival indices refer to the first sample of the selected trace, including a bucket selection. `trace_start_time` is carried when available. Missing absolute timestamps are not fabricated. |
| Target rate | 100 Hz. |
| Crop origin | Integer target-grid sample; also reported in seconds from trace start. |
| Valid support | Samples within the stored trace's last-sample timestamp, after declared support trimming and resampling. Padding does not extend this support. |

For source arrival index `a`, arrival rate `f_a`, and target crop start `c`:

```text
arrival_seconds_from_trace_start = a / f_a
arrival_offset_in_crop = (a / f_a) * 100 - c
absolute_arrival = trace_start_time + c / 100 + arrival_offset_in_crop / 100
```

`ManifestDataset.get_sample_with_metadata(i)` returns `(waveform, labels, info)`. `info` includes the contract version, source/effective/arrival/target rates, source start time, crop offset, valid sample count, component mask, effective P/S offsets, and arrivals excluded by source/crop support. Ordinary `dataset[i]` retains the training API `(waveform, labels)`.

Arrivals beyond real support or outside the selected crop are omitted and reported. A signal row whose last usable arrival is lost fails. The output grid ends at the last target sample within source support: a fractional endpoint arrival beyond that target sample can be rejected; it is never moved into padded support.

## Resampling and experiment control

`waveform_contract.resample_waveform` uses a bounded rational ratio and SciPy polyphase FIR resampling, Kaiser beta 5, with line boundary extension and unchanged first-sample time. Output is trimmed to actual source support. Ratios that cannot meet the bounded approximation tolerance fail.

This replaces the training loader's FFT resampler. Fixtures check timing, passband fidelity, and rejection of an above-target-Nyquist tone. The implementation is a shared utility available for subsequent work, but the benchmark, noise exporters, and deployed SeisBench/QuakeScope path have **not** yet adopted it. Deployment parity remains **34C**; this commit makes no assertion that training and deployment preprocessing now match.

Record this kernel/version change in experiment provenance. In #46, hold the resampling contract fixed across the relevant comparison arms, or explicitly isolate it as an ablation. The existing PSN target formula, normalization, augmentations, and frozen RAM crops remain separate issues. In particular, partial-label loss masking and padding-aware loss are still #41 work; the valid-support/component metadata here does not implement those masks.

## Rejection ledger and migration

By default a manifest `train.csv` produces `train.rejected.<pid>.jsonl` alongside it. Pass `rejection_log=` to `ManifestDataset` or `CachedManifestDataset` to choose a writable output location. Each record contains UTC time, contract version, manifest SHA-256, row identity, exception type, and reason. Dataset initialization failures have no row index. The current failure budget is zero: this is a stop-on-error ledger, not an exhaustive census or automatic skip list.

Before using existing server manifests:

1. Keep the historical CSVs, waveforms, checkpoints, and benchmark caches immutable for 34B.
2. Resolve missing rate/orientation/dimension metadata from source evidence in a separate corrected artifact. Do not add assumed 100 Hz or ZNE values merely to pass validation.
3. Inspect each rejected record, repair its verified metadata or record an explicit dataset exclusion, and run again. Partial labels require review under #41 before scientific training claims.
4. Rebuild RAM caches and restart loader workers after any manifest, source metadata, or preprocessing change. Cached samples contain the old preprocessing until recreated.

The legacy noise builders do not supply a complete verified component/dimension contract and contain their own preprocessing assumptions. Their outputs may therefore be rejected. Repair and verify those exporters before using newly built noise pools with this loader; coordinate this follow-up with #34C and #42. This branch does not certify old noise crops by attaching new metadata to them.

The historical probes in `docs/audit_2026-09-10/probes.py` target the previous loader internals. Reproduce those results from base `6b5297c`; use the new fixture suite to validate this implementation.

## Validation and remaining gates

Run offline with PyTorch, SeisBench, h5py, NumPy, pandas, and SciPy installed:

```bash
python -m unittest discover -s tests -p test_manifest_dataset.py -v
python -m pytest tests -q
```

The loader suite has 15 tests covering direct and real SeisBench HDF5 readers, bucketed CW/WC storage, 20/40/50/62.5/100/120/200/250/500 Hz (plus 80 Hz direct), independent pulse times, fractional indices, P-only/S-only rows, crop endpoints, support trimming, missing channels, unknown rates/orientation, conflicting metadata, explicit noise, preload failure, initialization-error logging, and spawned workers. Waveform and label peak times must agree with independently generated source times within one target sample. No network dataset download or trained-model scoring is required.

Local validation on 2026-09-10: **15 loader tests passed**, using Python 3.11, PyTorch 2.7.1, SeisBench `0.9.1.dev16+g3667d44`, NumPy 1.26.4, and SciPy 1.16.0. The existing base-environment suite passed **31 tests**, with the loader module skipped there because that environment lacks PyTorch. Multiprocessing tests required normal OS shared-memory access outside the restricted execution sandbox.

This is local evidence for **34A**, pending review and checks against the pinned server environment. Issue #34 stays open:

- **34B:** Capture immutable actual-v7 train/val/test row forensics, with old/new offsets, effective labels, failures, source/distance/phase counts, unique identities, and training exposures. Quantify affected rows only from those artifacts.
- **34C:** Verify independent timestamps through the benchmark, remove unsupported ETHZ/MLAAPDE assumptions, define cache invalidation, reconcile all resamplers/orientations/normalizations, and check parent/wrapper/export probability parity for the actual deployed checkpoint pair.
- **Then #46:** Run controlled repair experiments once the corresponding evaluation and data gates in the revised issue roadmap are met. Passing these fixtures is not evidence of improved earthquake picking.
