# 33A: exclusion contract

Checkpoint 33A of issue #33, branch `issue/33a-versioned-exclusions`, stacked on
PR #51 (44A). Everything below was verified on 2026-09-11 with
`python -m pytest tests -q` in the base conda interpreter (Python 3.9, pandas
2.3.2, no SeisBench, no torch, no h5py). Nothing was run on the lab server;
the server-only steps are marked.

## What 33A releases

- `scripts/exclusion_bundle.py`: builds, verifies and applies
  `data/exclusions/bundle.json` (`BUNDLE_VERSION = 1`).
- Trace identity `(dataset, chunk, trace_name)`:
  `heldout_sequences.load_sequence_exclusions(chunk_aware=True)`,
  `heldout_sequences.listed_mask`, and a chunk-aware `check_manifest`. The
  name-only loader still works for its existing callers and tests.
- Event identity across datasets: `exclusion_bundle.origin_unions`,
  `event_groups`, `cross_source_duplicates`, `fill_splits_by_group`, wired
  into `build_training_dataset._event_group_ids` and `assign_splits`.
- Every builder loads the bundle fail-closed and applies it (table below),
  and writes what it removed beside its output.
- `tests/test_exclusion_bundle.py`: fixtures for every acceptance item that
  can be exercised without the cache, plus an end-to-end run of
  `build_training_dataset.main` on two synthetic sources with stubbed
  `seisbench` and `event_keys` modules (per-source counts, final gate,
  `provenance.json`, a cross-dataset event kept in one split).

## The bundle

`build_bundle` records, for every input that defines an exclusion, what it
found on the machine that built it. `load_bundle` recomputes every hash it
can and refuses to hand the bundle out if anything differs.

| Key | What is recorded | On `load_bundle` |
|---|---|---|
| `rules` | `heldout_sequences.WINDOWS` and `HOLDOUT_YEARS`, embedded verbatim, plus the sha256 of their canonical JSON | stale if the code's windows or years differ |
| `inputs.heldout_sequences_csv` | path, present, sha256, bytes, n_rows of `data/exclusions/heldout_sequences.csv` | stale if the hash differs; missing if recorded present and absent now; stale if recorded absent and present now |
| `inputs.benchmark_manifest` | same for `notebooks/benchmark_manifest.csv` (the trace source of `load_benchmark_exclusions`) | same rule |
| `inputs.label_error_reports` | same, per Aguilar stem (`aq2009, ceed, ethz, instance, pnw, stead, txed`), searched in `data/labelerrors/` then `~/.cache/phasenet_retrain/label_errors/` as `load_label_error_exclusions` does | stale if a report present here differs or was absent at build; unchecked if not cached here |
| `inputs.evaluation_suites` | sha256 of `configs/evaluation_suites.json`, `policy_sha256` (same expression as `evaluation_policy.record_access`, so it matches `data/evaluation/access.jsonl`), policy version, number of roles, acceptance status | stale if the file differs |
| `sources` | for each of the 20 `DATASET_CONFIGS` names: sha256, bytes and row count of every complete `metadata*.csv` under `$SEISBENCH_CACHE_ROOT/datasets/<name>/`, `.partial` shards listed by name; `null` when the source is absent | stale if present here and different; unchecked if absent here; `verify_sources=False` skips them (`check` and `show` skip unless `--verify-sources`) |
| `uncertified_sources` | the names whose entry is `null` | |
| `quarantine_policy` | `allow_unknown`, the flag column, the rule text | part of the hash |
| `sequence_list_present`, `certified` | `certified` is true only when the list was present and no source is uncertified | `require_certified=True` raises `UncertifiedBundleError` otherwise |
| `provenance` | git commit, creation time (UTC), cache root, host, builder | not hashed |
| `sha256` | sha256 of the canonical JSON (sorted keys, no whitespace) of everything except `sha256` and `provenance` | stale if it does not match the content (edited or truncated bundle) |

Consequences: two builds from the same inputs give the same `sha256`
whatever the time or host; adding a window, changing a year, editing the
sequence list, the benchmark manifest, a label-error report or the suite
policy gives a new hash and makes the old bundle stale. Adding a suite to
`configs/evaluation_suites.json` therefore invalidates the bundle, as the
issue requires; the bundle does not say whether that suite's cases are
eligible (#37, #44B).

Nothing in this PR commits a bundle. A bundle built on this laptop would
record the sequence list as absent and all 20 sources as uncertified, and
would become stale the moment the list is committed.

## Identity

Trace identity is `(dataset, chunk, trace_name)`, chunk `""` when the dataset
has none. A listed row with chunk `""` matches every chunk of that trace name
in that dataset. `mlaapde`, `cwa` and `aq2009gm` reuse `trace_name` as a
slot index across chunks (`event_keys.trace_key_map` docstring), so the
name-only key of the previous loader over-excluded unrelated slots; the
chunk-aware key excludes the listed slot only. `check_manifest` accepts both
shapes, so the final gate of `build_training_dataset.py` no longer fails on a
legitimately kept slot.

Event identity for split assignment is the `event_keys.py` fingerprint, as
before, plus origin coincidence: two rows whose source origins lie within
2 s and 0.1 deg (`origin_unions` defaults) are unioned into one group.
`_event_group_ids` applies both; `assign_splits` now treats a row with an
origin fingerprint as identified, so it is grouped instead of being handed to
the vendor split, and `fill_splits_by_group` keeps whole groups in one split
(the same greedy rule as before, moved into `exclusion_bundle.py` so the
fixture tests run the builder's code). `cross_source_duplicates` is the audit
view: rows that share an event with a row of another dataset. Identical
fingerprints are collapsed before pairing, so an event with many stations
costs one union per row.

`overlapping_intervals` flags same-station `[start, end]` overlaps across
datasets or between signal and noise roles. It is an audit tool; no builder
calls it yet because no builder has start and end for both signal and noise
rows today. The noise metadata gains `starttime` with this change, which is
the first half of that input.

## Quarantine policy

A row whose independence cannot be tested is quarantined. For signal,
validation, augmentation and mining rows that means a missing
`source_origin_time`, `source_latitude_deg` or `source_longitude_deg`; for
noise rows a missing station latitude/longitude or trace start time. A row
whose location alone lies in an all-time place window is excluded whatever
its time, before quarantine is considered.

| `allow_unknown` | Unknown rows | Count | Independence claim |
|---|---|---|---|
| `false` (default) | dropped | `n_quarantined_unknown` | none needed |
| `true` (`build --allow-unknown`) | kept, `independence_unverified = True` | `n_unknown_kept_flagged` | they are outside it |

The policy is part of the bundle hash, so a manifest's `provenance.json`
identifies which policy built it. `build_training_dataset.py
--strict-year-holdout` is now a no-op: the default policy is the strict one.
Every manifest and noise metadata file written from now on carries
`independence_unverified` (all false under the default policy).

Each `apply_exclusions` report counts one reason per row, in the order
`trace_listed > in_window > year_holdout > unknown`, so
`n_removed = n_trace_listed + n_in_window + n_year_holdout + n_quarantined_unknown`;
`windows` counts hits per window over all input rows.

## Builders

| Script | `kind` | Columns tested | Where the counts go |
|---|---|---|---|
| `build_training_dataset.py` | `signal`, once per `DATASET_CONFIGS` source | `source_origin_time`, `source_latitude_deg`, `source_longitude_deg` | `heldout_removal_report.csv` (per source: `n_input`, `n_kept`, `n_removed`, `n_trace_listed`, `n_in_window`, `n_year_holdout`, `n_quarantined_unknown`, `n_unknown_kept_flagged`, `trace_list_checked`, `bundle_sha256`) and `provenance.json` (bundle hash and certification, rules and policy hashes, git commit, options, per-source counts, the final gate reports, `hash_manifests` digests of train/val/test) |
| `add_noise_to_manifests.py` | `noise` | `latitude`, `longitude`, `starttime` of `data/noise_global/metadata.csv` | `<manifests-dir>/provenance.json`, list `noise_appends` |
| `add_prephase_to_manifests.py` | `noise` | same columns of `data/noise_prephase/metadata.csv`; `--manifests-dir` added | same |
| `build_noise_dataset.py` (server) | `noise` at extraction | `station_latitude_deg`, `station_longitude_deg`, `trace_start_time` of the source metadata | `<out-dir>/provenance.json`, list `extractions`; `metadata.csv` now fills `starttime` and adds `independence_unverified` |
| `build_prephase_noise.py` (server) | `signal` on the parent manifest row, then `noise` on the parent trace's station location and `trace_start_time` joined from the source metadata | as named | same; `metadata.csv` gains `starttime`, `independence_unverified` |

All five load the bundle with `require_certified=True`; `--allow-uncertified-bundle`
lowers that to "sequence list present" and is recorded in the provenance.
`build_training_dataset.py` keeps its final gate and runs it twice: the
chunk-aware `check_manifest`, then `apply_exclusions` on each written split,
which must remove nothing. The two append scripts refuse any manifest path
listed in `data/manifest_checksums.csv`. The two extraction scripts refuse to
resume a `metadata.csv` whose header predates this change (`--out-dir` starts
a new set), because those rows were extracted without the bundle and carry
no start time.

Two consequences to know before the next noise build. `data/noise_global/metadata.csv`
written before 2026-09-11 has an empty `starttime` for every row, so
`add_noise_to_manifests.py` quarantines all of it under the default policy;
re-extract with the patched `build_noise_dataset.py`. TXED noise has no
station coordinates (`build_noise_dataset.py`, `lat_col=None`), so it is
quarantined at extraction under the default policy; keeping it is a
deliberate `--allow-unknown` decision, flagged in the metadata.

Also changed: `heldout_sequences._to_utc` parses with `format="ISO8601"` on
pandas 2. The pandas 2 default infers one strptime format from the first
value and coerces every differently formatted value (with or without
fractional seconds, `T` or space) to NaT, which would have quarantined rows
for a formatting accident. Covered by a test.

## Certified locally versus on the server

| Claim | Laptop (this PR) | Server |
|---|---|---|
| Rules, policy, benchmark manifest and sequence-list hashes verified on load | yes, with fixtures | yes |
| Missing or stale list, tampered bundle, changed windows refused | yes, with fixtures | yes |
| Per-reason counts, quarantine, chunk-aware keys, cross-source groups in one split, noise windows on station coordinates, interval overlap | yes, with fixtures | same code |
| `build_training_dataset.py` wiring (`process_dataset`, gate, `provenance.json`, cross-dataset grouping) | yes, two synthetic sources, stubbed `seisbench`/`event_keys` | real sources, to run |
| `check` on a real manifest | not possible, manifests are not on this machine | `python scripts/exclusion_bundle.py check <manifest>` |
| Source snapshot hashes, `certified: true` | never (no cache) | after `build` with `SEISBENCH_CACHE_ROOT` set |
| Per-source removal and unknown counts for the real corpora | none | written by the first `build_training_dataset.py` run with a bundle |
| Noise extraction with the bundle | compiled only (h5py, seisbench absent) | to run |

## Commands

```sh
# server, once the audit list exists
python scripts/audit_heldout_sequences.py                     # writes data/exclusions/heldout_sequences.csv
python scripts/exclusion_bundle.py build                      # hashes list, rules, policy, reports, sources -> data/exclusions/bundle.json
python scripts/exclusion_bundle.py show
git add data/exclusions/heldout_sequences.csv data/exclusions/bundle.json

# anywhere: read-only forensics, exit 2 when an excluded row is present
python scripts/exclusion_bundle.py check data/manifests_v2/train.csv
python scripts/exclusion_bundle.py check data/noise_global/metadata.csv --kind noise

# server: builders (all fail closed without a valid bundle)
python scripts/build_training_dataset.py --output-dir data/manifests_v4
python scripts/build_noise_dataset.py --out-dir data/noise_global_v2
python scripts/add_noise_to_manifests.py --manifests-dir data/manifests_v4

# laptop
python -m pytest tests -q
```

`build --allow-missing-sequence-list` records the list as absent for an
uncertified bundle; the builders still refuse to run on it, `check` runs
without the trace-list test and says so.

## What 33A does not certify

- Server-wide counts. No bundle and no sequence list are committed by this
  PR, and no per-source removal or unknown count exists for the real corpora.
  The issue's server-wide certificate needs `audit_heldout_sequences.py`, then
  `exclusion_bundle.py build`, then one builder run, all on the server.
- Source snapshot hashes. Every entry of `sources` is `null` when built here.
- The historical v7 manifests. They are not on this machine; `check` is the
  read-only tool for #34B and does not modify them.
- The noise extraction scripts. They were compiled and patched by string
  replacement; they were not executed.
- Whether a listed or quarantined row is a real leak. The bundle records
  decisions and counts; the audit script establishes the list.

## Historical manifests

Nothing here modifies a historical manifest: `check` is read-only, the append
scripts refuse paths listed in `data/manifest_checksums.csv`, and
`hash_manifests.py` is unchanged. The new `independence_unverified` column is
not a key column, so `hash_manifests` digests of regenerated manifests remain
comparable with the committed ones.
