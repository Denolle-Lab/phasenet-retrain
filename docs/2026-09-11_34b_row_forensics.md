# #34B: historical v7 row forensics

Branch: `issue/34b-row-forensics`, from merged PR #62 (`e400e16`).
Implementation: [`scripts/audit_v7_rows.py`](../scripts/audit_v7_rows.py).

This runner replays the v7-era loader and the corrected #34A loader against preserved local files. It does not download datasets, rebuild manifests, run a model, or change source files. A completed replay is **not** a certificate that those files, reader settings, or random crops were used during training. Issue #34B remains open until the actual server artifacts and run provenance are verified and the independent noise comparison is supplied.

## Evidence currently available

The v7 configuration names `data/manifests_v2/{train,val,test}.csv`. The committed sorted-key inventory lists 527,477 / 125,218 / 275,660 occurrences respectively. Those are **reference inventory counts**, not counts measured by this audit. The CSVs and waveform cache are absent from this checkout.

The legacy reference is `3bf98c4:scripts/manifest_dataset.py`, the commit that introduced the v5–v13-era files. Its cache root was `/data/wsd04/ak287/.seisbench`. That source snapshot is evidence of the implementation, not proof of the exact uncommitted code or environment used in the historical run.

Read-only SSH checks on 2026-09-11 found:

| Documented host | Access result |
| --- | --- |
| Cascadia / Dasway | Authentication denied for the configured `mdenolle` account |
| Siletzia | Connection timed out on configured port 7777 |
| Psound / Marine1 | Connection refused on port 22 |

No server file was read. The local inventory command records missing inputs and exits with status 2; it produces no synthetic corpus statistics. Authenticated server access, the original repository path, and the current location of the preserved cache are the next operational requirements.

## Outputs and interpretation

Each invocation requires a **new** output directory. It contains:

- `inputs/`: byte-for-byte manifest/config snapshots, the committed reference checksums, and any supplied reader/exposure specifications.
- `code/` and `legacy_loader.py`: the audit and repaired loader sources, plus the pinned historical source. Only selected historical definitions execute; the old cache setup and download-capable dataset constructors do not.
- `rows.jsonl`: one record per audited manifest occurrence, retaining split, original row index, manifest byte hash, and `(dataset, chunk, trace_name)` identity.
- `phase_summary.csv`: source × distance × stored-rate × phase counts, with unique trace identities, manifest occurrences, declared/effective labels, rate/identity/time-axis mismatches, failed/rejected rows, comparable timing denominators, and available training exposures. This is the H2 supervision table; it does not establish causality or perform new model scoring.
- `provenance.json`: input/output hashes, current dependency versions, actual source file paths, missing evidence, and completion status.
- `CHECKSUMS.sha256`: hashes of the artifact files, including provenance, for later integrity verification.

The runner never overwrites an existing output directory. Source CSVs are hashed; HDF5 files have size/mtime checks before/after replay and hashes of consumed waveform arrays. Whole HDF5 files and original metadata CSVs are **not** copied into the report. Preserve them separately in an immutable server snapshot for historical certification. Size/mtime checks are weaker than full-file content hashes and do not establish historical identity.

The historical and corrected paths are evaluated independently. Important distinctions:

1. **Rate declared to the old resampler versus physical reader rate.** Direct HDF5 reads declared 100 Hz even when the stored grid differed. Such a row may have an aligned array index but an incorrect physical duration/frequency scale. SeisBench could instead resample implicitly before the loader resampled again. Both rates, lengths, and physical output rate are recorded when verifiable.
2. **Source identity.** The old SeisBench lookup used trace name alone; the repaired lookup includes chunk. Timing differences are not reported as paired errors when the two paths choose different traces. Historical CSV chunk type inference is preserved in the legacy replay.
3. **Time-axis interpretation.** An old shape heuristic can treat a component axis as time. A physical output rate or displacement is not fabricated for that case.
4. **Failure stage.** A failed row fetch could become zero/noise; a failed eager reader initialization would stop the original whole-loader setup. These have separate statuses. The runner enumerates row paths independently; an initialization error must not be counted as training noise contamination.
5. **Replay versus historical observation.** `historical_fetch_status` remains `unknown`. Contemporary logs or saved preload artifacts are required to establish that an observed replay failure occurred during training. The replay uses deterministic per-row RNG and restores the caller's RNG state; it does not reconstruct the original cached random crops.
6. **Label support.** The report retains old/new crop offsets and effective P/S labels, source-support exclusions, and old target times where physically interpretable. Missing references and failed processing remain unknown, with explicit denominators. A declared arrival outside source support is excluded from the displacement denominator.

Unique counts are within each summary stratum and refer to source trace identities, not independent earthquakes. Do not sum unique counts across phases or overlapping strata. Manifest occurrences describe one enumeration of the file, not the number of examples actually used for optimizer updates.

## Running on the server

Use a checkout of this branch with the training dependencies installed. Start with the original configuration and preserved files; do not regenerate `manifests_v2` to make its fingerprints match.

```bash
python scripts/audit_v7_rows.py \
  --manifest-dir /absolute/path/to/original/data/manifests_v2 \
  --cache-root /data/wsd04/ak287/.seisbench \
  --inventory-only \
  --output /absolute/path/to/audits/v7-34b-inventory
```

Inspect `provenance.json`. A sorted-key checksum match confirms only the old key/multiplicity fingerprint. It does not verify arrival columns or bind a manifest to the checkpoint. A mismatch is recorded rather than relabeled as an original-v7 match; actual-v7 attribution must then remain unresolved.

For SeisBench routes, supply explicit reader options in a JSON file via `--sources`. The runner uses `WaveformDataset` on local files, so it cannot trigger `BenchmarkDataset` downloads. The original subclass's filtering, metadata ordering, component/dimension defaults, and sampling-rate settings still need evidence from the historical runtime. Values below illustrate the schema; verify them before using them as historical settings:

```json
{
  "geofon": {
    "route": "seisbench",
    "path": "/data/wsd04/ak287/.seisbench/datasets/geofon",
    "legacy_reader_options": {
      "sampling_rate": null,
      "component_order": "ZNE",
      "dimension_order": "NCW",
      "missing_components": "pad"
    }
  }
}
```

Add all SeisBench sources present in the manifests. Direct sources can also override paths with `route: "single"` or `"chunked"`; chunked entries have `prefix`, such as `"waveforms_"`. Use absolute source paths. Noise roots default to this checkout's `data/noise_global` and `data/noise_prephase`; override them when auditing a separate snapshot. All effective source specifications are recorded.

First run a short diagnostic prefix:

```bash
python scripts/audit_v7_rows.py \
  --manifest-dir /absolute/path/to/original/data/manifests_v2 \
  --cache-root /data/wsd04/ak287/.seisbench \
  --sources /absolute/path/to/verified-reader-options.json \
  --max-rows 100 \
  --output /absolute/path/to/audits/v7-34b-diagnostic
```

Then use a new directory and omit `--max-rows` for the full replay. A diagnostic prefix is explicitly marked `partial_replay`; it is not a representative sample or a corpus estimate. Processing reads waveforms twice and hashes metadata, so benchmark the diagnostic on server storage before scheduling the complete pass. Direct-reader metadata is filtered to requested trace names. SeisBench still needs its full metadata/index to reproduce name collisions; this runner is not a RAM waveform cache.

To include documented optimizer exposures, pass `--exposures /path/to/exposures.csv`:

```text
split,row_index,manifest_sha256,training_exposures
train,0,<SHA256 of the exact train.csv bytes>,45
```

The number above is a schema example, not a v7 measurement. Counts must come from execution evidence, including resumes and any sampler/drop-last behavior. They are not derived from `max_epochs` or checkpoint epoch. Unlisted rows retain unknown exposures, and a summary reports a total only if every row in that stratum has evidence. Duplicate, fractional, out-of-range, or hash-mismatched exposure identities fail validation.

Verify an artifact after copying it:

```bash
cd /absolute/path/to/audits/v7-34b-full
shasum -a 256 -c CHECKSUMS.sha256
```

## Remaining #34B acceptance work

- Run on the actual original train/val/test CSVs and cache; review sorted-key differences and bind full byte hashes to run/checkpoint evidence.
- Verify the original SeisBench subclass behavior and runtime. Review row errors and actual-support metadata without changing the preserved inputs.
- Supply historical read/preload logs, RNG/cache artifacts where retained, and exposure evidence; keep unavailable quantities unknown.
- Review H2 with effective S labels and rejected-row denominators by distance/source, rather than the old cap-based estimates.
- Restore noise-pool ordering on independently verified negative populations. This runner intentionally produces no ranking from the old pool: its overlap with training/validation, model-based pool selection, and same-pool threshold tuning must be addressed first. Freeze calibration separately, bind identities and model histories, then report the ordering on untouched eligible negatives. Existing historical rankings remain exploratory.

## Validation

Local validation on 2026-09-11: **14 forensic tests**, **15 loader tests**, and **31 existing tests** passed. The two training-dependent modules skip in the base environment and were run separately in the PyTorch/SeisBench environment. The local inventory exited 2 as expected for missing original manifests.

`tests/test_v7_row_audit.py` uses synthetic local HDF5/CSV inputs and the actual pinned historical definitions. It covers native versus already-resampled SeisBench paths, direct-reader duration errors, bucket failures, numeric chunk inference, duplicate names, S-only rows, invalid support, invalid time axes, deterministic noise replay, source mutation, missing inputs, immutable output directories, byte hashes, repeated rows, and evidence-bound exposure totals. These fixtures validate the tool; their counts are not data-corpus findings.

```bash
python -m unittest discover -s tests -p test_v7_row_audit.py -v
python -m unittest discover -s tests -p test_manifest_dataset.py -v
python -m pytest tests -q
```
