# Label validity policy and target contract (#41A)

*2026-09-11, branch `issue/41a-label-policy` from `audit/2026-09-07-generalization`
at `e400e16`. Checkpoint 41A of issue #41: the target and provenance schema and
the review policy. It changes no historical manifest and trains nothing.*

## The defect this replaces

`make_labels` at `3bf98c4:scripts/manifest_dataset.py` lines 251–259 (still
the formula on the integration branch, line 258) sets N = 1 − max(P, S) from
at most one P and one S index. An S the bulletin did not report, a second
event in the window, an arrival the source labelled automatically, a gap and
the zero padding after a short trace all become confident noise targets, and
cross-entropy pushes the corresponding logits down at every epoch. Distillation
then argues with those samples. This is the mechanism behind the uniform S
under-confidence of the 2026-09-07 audit (H2) and a plausible part of the
aftershock deficit, which the benchmark cannot see.

## Schema

`scripts/arrivals.py`:

| Object | Fields | Meaning |
|---|---|---|
| `Arrival` | `phase` P or S; `time_s` seconds from sample 0 of the waveform it belongs to; `tier` manual, reviewed, automatic, unknown; `event_id`; `uncertainty_s` | one catalogued arrival |
| manifest column `arrivals_json` | JSON list of arrivals, times from the source trace start | overrides the legacy `p_arrival_sample` and `s_arrival_sample` for the targets; the crop anchor still comes from the columns |
| manifest column `negative_support` | certified, reviewed, unknown | where the absence of a phase is known |
| `WindowSample` | waveform (3, n) ZNE float32, `rate_hz`, arrivals, `component_mask`, `negative_support`, `unknown_intervals` [start, end) s, `valid_samples`, `meta` | the object the augmentation transforms (#43A) act on |

Provenance tiers: `manual` and `reviewed` supervise; `automatic` and `unknown`
do not. Negative support: `certified` means a catalogue-complete window
certified event-free outside the listed arrivals (the census of #39 records
the completeness magnitude of the region); `reviewed` means a reviewed
negative window (#42); `unknown` is every bulletin harvest and every legacy
manifest row, where a missing pick is not proof that nothing arrived.

## Policies (`scripts/label_targets.py`)

| | `legacy` | `masked` |
|---|---|---|
| Targets | Gaussian P and S (σ = 0.10 s) from every listed arrival; N = clip(1 − max(P, S)); channel sums can reach 2 at coincident peaks | Gaussian P and S from `manual` and `reviewed` arrivals only; same-phase overlap takes the maximum; the PSN triple renormalised to sum to one at every sample (coincident P and S give 0.5, 0.5, 0) |
| Mask | 1 everywhere, padding included: the v7 behaviour, kept for the alignment diagnostic 46A | 1 where the label provenance supports supervision, 0 elsewhere (below) |
| Rows with nothing to supervise | served | rejected into the ledger: a signal row with no supervising arrival and unknown negative support |

Mask rules of the `masked` policy, applied in this order:

1. `negative_support` certified or reviewed: every sample is supervised.
   Unknown: only ±0.5 s (`supervised_halfwidth_s`) around each supervising
   arrival.
2. Each `automatic` or `unknown`-tier arrival masks ±1.0 s
   (`unknown_halfwidth_s`) to 0.
3. The ±3σ core (`positive_core_sigmas`) of each supervising arrival is set
   back to 1: a manual pick is trusted over an automatic neighbour.
4. `unknown_intervals` (gaps, masked artefacts) are set to 0; they win over
   everything.
5. Samples beyond `valid_samples` (padding) are set to 0.

The mask multiplies the per-sample cross-entropy (hard, soft or focal) and
the mean is taken over supervised samples. The distillation term, when an arm
uses it, is computed on every sample; timing and presence terms are unchanged
and remain off in the strategy.

## Where it is wired

- `ManifestDataset(label_policy=None|"legacy"|"masked"|dict, return_mask=False)`:
  builds arrivals from `arrivals_json` or the legacy columns, resolves
  `negative_support` (column, else certified for the explicit noise pools and
  unknown for signal rows), calls `build_targets`, rejects unsupervised signal
  rows, and returns `(waveform, labels, mask)` when `return_mask` is set. The
  default keeps the historical `(waveform, labels)` API and targets;
  `make_labels` now delegates to the legacy policy.
- `CachedManifestDataset` caches the mask as uint8 (one byte per sample) and
  reports the number of supervised samples.
- `build_dataloaders(config, splits=...)` reads `data.label_policy` and, when
  it is set, yields `(x, y, mask)` from every loader; it now pre-loads only
  the requested splits (training no longer loads the test split).
- `PhaseNetFinetune.compute_loss_and_metrics(x, y, mask=None)` applies the
  mask and returns per-term metrics: `loss_ce`, `loss_kd`, the soft
  per-channel terms `loss_P`, `loss_S`, `loss_N`, `supervised_fraction` and the
  supervised positive counts `pos_P`, `pos_S`; accuracy is measured on
  supervised samples. `run_epoch` averages the terms and sums the counts, and
  `MetricsLogger` writes them as new columns (new runs only; an existing
  metrics CSV keeps its old header and should not be appended to).

Config:

```yaml
data:
  label_policy: masked          # or legacy, or {name: masked, supervised_halfwidth_s: 0.5}
```

`configs/e0_46a_legacy_targets.yaml` and `configs/e0_46a_masked_targets.yaml`
are the two corrected-loader arms of the alignment diagnostic (46A): the v7
recipe on the v7 manifests, differing only in the label policy. The third arm,
the pinned legacy loader, runs through PR #63's `load_legacy`.

## Review policy (the rest of #41)

- Manual or reviewed picks supervise; automatic and unknown-tier picks mask;
  nothing is deleted for being uncertain. A row is quarantined, with a
  reason in the rejection ledger, only when it carries no supervision at all.
- 41B reviews the pilot corpus (40A): every rejected row is inspected, the
  provenance tiers of each source are recorded from source evidence (a
  SeisBench `trace_p_status` or an operator's evaluation mode, never a dataset
  name), and the certified negative windows are those the #39 census supports.
- The overlapping-target rule for event superposition (#43A) is the
  per-channel maximum followed by renormalisation, the same as above; masks
  merge with unknown winning over negative and positive support.

## Validation

```bash
python -m pytest tests/test_label_targets.py -q      # 14 tests, pure numpy, run on the laptop
python -m pytest tests -q                             # loader and torch tests skip without torch/SeisBench/h5py
```

`tests/test_manifest_dataset.py` gained two masked-policy fixtures (mask and
target values on the 34A fixture; rejection of an unsupervised row and its
certified-support counterpart). They need PyTorch, SeisBench and h5py and were
not executed on the laptop that wrote this; they run in the pinned server
environment before 41A is checked off. The loss path in
`scripts/fine_tune_model.py` was compiled, not executed, for the same reason.

## What 41A does not do

No manifest carries `arrivals_json` or `negative_support` yet; that is the
40A builder. No noise pool is certified; that is #42. No training was run.
