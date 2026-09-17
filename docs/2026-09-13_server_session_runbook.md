# First server session: what to run, in what order, and what each step decides

*2026-09-13, branch `docs/server-session-runbook` from
`integrate/2026-09-13-checkpoints` (PR #76). Every command below was
exercised on the laptop against fixtures; none has run against the cache.
Server paths follow `3bf98c4:scripts/manifest_dataset.py`
(`SEISBENCH_CACHE = /data/wsd04/ak287/.seisbench`) and the historical
configs (`data/manifests_v2/{train,val,test}.csv`,
`results/finetune_*_metrics.csv`, `checkpoints/finetune_*/best.pt`); verify
each path exists before the step that needs it. Nothing here trains a model.*

## 0. Environment (10 min)

```bash
cd <server clone>; git fetch; git checkout audit/2026-09-07-generalization; git pull --ff-only
conda activate <env with torch, seisbench, h5py, scipy, pandas, obspy, pytest>
export SEISBENCH_CACHE_ROOT=/data/wsd04/ak287/.seisbench
export MPLCONFIGDIR=$PWD/.mpl
python -m pytest tests -q            # laptop: 361 passed in the torch venv; must pass here first
python -c "import sys, torch, seisbench, scipy, numpy; print(sys.version.split()[0], torch.__version__, seisbench.__version__, scipy.__version__)"
```

Record the four versions (Python, PyTorch, SeisBench, SciPy); the run card records them too. If the suite fails
here, stop and report the failure before anything else: the checkpoints
were validated on the laptop with SeisBench 0.12.5 and PyTorch 2.2.2, and
the server's pinned runtime is the one that matters.

## 1. Immutability of the historical inputs (5 min)

```bash
python scripts/hash_manifests.py --check            # every listed manifest against data/manifest_checksums.csv
ls -la data/manifests_v2/ checkpoints/finetune_jma_wc_global_v7/ results/ | head -40
sha256sum checkpoints/finetune_jma_wc_global_v7/best.pt models/jma_wc_ft_global_v7.pt
```

A checksum mismatch on `manifests_v2` means the 34B attribution to v7
stays unresolved; run 34B anyway and let its `provenance.json` record the
mismatch. Do not regenerate manifests.

## 2. The cheap number first: post-July fetch-failure counters (5 min)

Runs trained after 2026-07-12 (`71d7e2d`) logged fetch failures. Read
them before the replay:

```bash
grep -l -i "fetch" results/*_metrics.csv results/*/*.log 2>/dev/null | head
grep -h -i -E "fetch.*fail|zero.*sample|substitut" results/*/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -20
```

Note per run: run name, manifest, count of failed fetches, datasets named
in the messages. This is the first estimate of the zero-window count
(`docs/2026-09-11_silent_zero_windows_report.md` §6).

## 3. 34B: replay the v7 rows (inventory 15 min; full replay hours, benchmark first)

```bash
python scripts/audit_v7_rows.py --manifest-dir $PWD/data/manifests_v2 \
    --cache-root $SEISBENCH_CACHE_ROOT --inventory-only \
    --output $PWD/results/34b/inventory
cat results/34b/inventory/provenance.json | head -60     # manifest hashes vs checksums, per-source formats and rates
```

The inventory answers, per source: stored rate, HDF5 layout (bucketed or
not), which reader route the legacy loader took. That alone says whether
`meier2019jgr`, `ross2018gpd`, `pisdl`, `mlaapde`, `cwa`, `aq2009gm` were
bucketed and therefore zeros (report §5). Then write the reader-options
JSON for the SeisBench-route sources exactly as the runbook
`docs/2026-09-11_34b_row_forensics.md` shows (one entry per SeisBench
source in the manifests, absolute paths, the legacy options
`sampling_rate: null, component_order: ZNE, dimension_order: NCW,
missing_components: pad`), and run a timed diagnostic prefix:

```bash
time python scripts/audit_v7_rows.py --manifest-dir $PWD/data/manifests_v2 \
    --cache-root $SEISBENCH_CACHE_ROOT --sources $PWD/results/34b/reader_options.json \
    --max-rows 100 --output $PWD/results/34b/diagnostic
```

`--max-rows` is a prefix per split, so the diagnostic replays 300 rows
(100 from each of train, val and test). Scale its time by the total rows of
the three manifests divided by 300 (the training manifest alone is 527,477
rows) and schedule the full replay (new output directory, no `--max-rows`)
with `nohup`; it reads every waveform twice.
When it finishes, `phase_summary.csv` gives `legacy_zero_noise_rows` per
source and split, `legacy_label_displacement_s` for the rate defect, and
the effective P and S supervision after cropping (the H2 table). Commit
`results/34b/*/provenance.json`, `phase_summary.csv` and `CHECKSUMS`, not
the per-row parquet.

**Decision this step makes.** If the zero rows are a large fraction of the
direct-route sources, v7's result says nothing about the recipe and E0's
legacy arm is expected to reproduce the damage; if they are few, the rate
displacement is the dominant defect. Either way E0 runs; the number
decides what the report to the group says.

## 4. Task 1: the held-out exclusion list and its cost (30 min)

```bash
python scripts/audit_heldout_sequences.py            # joins manifests_v2 and the full corpora to the 23 windows
cat data/exclusions/heldout_sequence_counts.csv
git add data/exclusions/heldout_sequences.csv data/exclusions/heldout_sequence_counts.csv
```

The counts say what the place hold-outs (Etna, Campi Flegrei, Reykjanes,
La Palma, Santorini, West Bohemia, Maurienne, Corinth–Thiva) cost INSTANCE,
CREW and VCSEIS. A place that removes most of a source's volcanic traces
is a decision for the plan, not a reason to shrink the radius silently.

## 5. 33A: build and certify the exclusion bundle (10 min)

```bash
python scripts/exclusion_bundle.py build --cache-root $SEISBENCH_CACHE_ROOT     # hashes the sequence list, benchmark and label-error inputs, the suite policy and every source snapshot
python scripts/exclusion_bundle.py show
python scripts/exclusion_bundle.py check data/manifests_v2/train.csv --kind signal   # read-only: how many v7 rows the bundle would have excluded
git add data/exclusions/bundle.json
```

`check` on the historical manifests is the per-source removal count the
issue asks for. The bundle must report `certified: true`; if a source is
listed under `uncertified_sources`, its snapshot path is wrong.

## 6. 39A on the cache: pick status and held-out overlap per SeisBench source (20 min)

```bash
python scripts/source_census.py seisbench --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/census
git add data/census/seisbench_sources.csv
```

Fills the `server_required` columns: manual versus automatic pick status
where the source exposes it, rows inside held-out windows, native rates.
With step 4 this fixes which sources enter the T0 pilot
(`docs/2026-09-11_training_strategy_v3.md` §4.1).

## 6b. 41B on the cache: model-independent label audit of the curated sources (1–3 h)

```bash
python scripts/audit_source_labels.py seisbench --sources instancecounts ethz stead ceed pnw txed aq2009gm \
    geofon crew cwa iquique lendb vcseis scedc pisdl meier2019jgr ross2018gpd mlaapde \
    --sample 5000 --seed 0 --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/seisbench --report
python scripts/audit_source_labels.py duplicates --sources stead instancecounts ethz ceed scedc ross2018gpd \
    --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/seisbench
cat data/label_audit/seisbench/report.md
git add data/label_audit/seisbench/summary.csv data/label_audit/seisbench/provenance.json \
    data/label_audit/seisbench/report.md data/label_audit/seisbench/*/review_sheet.csv \
    data/label_audit/seisbench/duplicates.csv data/label_audit/seisbench/duplicates_provenance.json
```

5000 stratified rows per source (seed 0), read through the loader's own
readers at the stored rate; no model. The Aguilar multiplet reports must be
cached under `data/labelerrors/` (add `--download-reports` once if they are
not; `provenance.json` records their sha256). Per source the report gives
the S−P line and its flag rate, the P and S onset residuals against the AIC
energy onset, the edge and modal-sample shares, and the fraction of windows
with an unlabelled arrival split by Aguilar flag, with bootstrap intervals.
The laptop calibration on the seven regression/dev held-out cases
(`docs/2026-09-15_41b_label_audit.md`) says what to expect from analyst
picks: C1 flags 0–8.5 %, C2 late P labels 2.8–13.1 % (suspect picks 0–6.8 %,
the rest unlabelled earlier events), emergent onsets 2–41 % reported and not
flagged, C6 unlabelled arrivals in 59–100 % of aftershock-sequence windows. Time scales with rows × trace length; `meier2019jgr`, `ross2018gpd`
and `mlaapde` read 5000 traces each through h5py and are the slow ones.
`rows.parquet` stays out of git.

**Decision this step makes.** Which sources need the reviewed sample of 41B
first (a C2 early-energy fraction or a C1 flag rate far above the held-out
calibration), and whether the Aguilar flags mark rows the STA/LTA screen
also finds (the flagged/unflagged split): if the unflagged rows carry a
similar unlabelled-arrival rate, the report is not a filter and the extra
arrivals go in as masked `automatic` arrivals for every source alike.

## 6c. 41B on the cache: the benchmark test set and the historical manifests (1–2 h)

```bash
python scripts/audit_source_labels.py benchmark --benchmark notebooks/benchmark_manifest.csv \
    --sample 3000 --seed 0 --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/benchmark --report
for split in train val test; do
  python scripts/audit_source_labels.py manifest --manifest data/manifests_v2/$split.csv \
      --sample 3000 --seed 0 --cache-root $SEISBENCH_CACHE_ROOT --out-dir data/label_audit/manifest_v2_$split --report
done
cat data/label_audit/benchmark/report.md
python - <<'PY'
import json
for d in ("benchmark", "manifest_v2_train", "manifest_v2_val", "manifest_v2_test"):
    p = json.load(open(f"data/label_audit/{d}/provenance.json"))
    print(d, {s: (v["n_rows"], v["n_read_errors"], v["n_rate_mismatch"], v["stored_rates_hz"]) for s, v in p["sources"].items()})
PY
git add data/label_audit/benchmark/summary.csv data/label_audit/benchmark/provenance.json data/label_audit/benchmark/report.md \
    data/label_audit/benchmark/*/review_sheet.csv data/label_audit/manifest_v2_*/summary.csv \
    data/label_audit/manifest_v2_*/provenance.json data/label_audit/manifest_v2_*/report.md
```

Same C1–C6 as step 6b, on the rows the 2026 fine-tunes were tested on
(3000 benchmark rows stratified by dataset, floor 50, read through the
loader at the stored rate with bucket-style names) and on the rows v7 was
trained, validated and tested on (3000 rows per `data/manifests_v2` split).
The laptop part, `docs/audit_2026-09-16_benchmark_labels/README.md`, says
what to look for: the C2 S-onset residual on STEAD, TXED and INSTANCE, where
six or more independent public pickers put the S 0.3–5 s after the label on
11–14 % of the S labels; `n_rate_mismatch` per source in `provenance.json`
(the stored rate against notebook 05's 100 Hz: AQ2009GM is expected at
125 Hz and ETHZ at 120–500 Hz, which rescales their benchmark residuals);
and the loader's read errors on AQ2009GM, whose benchmark names carry no
chunk (an ambiguous name is a benchmark row whose waveform may not belong
to its label). `rows.parquet` stays out of git.

**Decision this step makes.** Whether the late-S consensus is a label
convention (C2 finds the energy onset at the label: keep the labels, the
pickers pick a later phase) or a label error (C2 finds the onset with the
pickers: the S labels of those sources go to `unknown` tier), and whether
the AQ2009GM and ETHZ benchmark rows must be recut at their stored rate
before the 34C numbers are recomputed.

## 7. Legacy noise pools: what 42A needs from them (10 min)

```bash
python - <<'PY'
import pandas as pd
for p in ("data/noise_global/metadata.csv", "data/noise_prephase/metadata.csv"):
    m = pd.read_csv(p, low_memory=False); print(p, len(m)); print(m[["starttime", "latitude", "longitude"]].isna().mean().round(3).to_dict())
PY
```

`data/noise_global` was written with an empty `starttime` on every row
(PR #71 finding); the pools are quarantined by the new exclusion policy
until re-extracted with the patched `build_noise_dataset.py`. Re-extraction
is a separate job; this step only confirms the state and its size.

## 8. Rebuild a manifest with the new builder, without training (30 min)

```bash
python scripts/build_training_dataset.py --output-dir data/manifests_v4_dryrun --seed 42
python scripts/exclusion_bundle.py check data/manifests_v4_dryrun/train.csv --kind signal
cat data/manifests_v4_dryrun/provenance.json data/manifests_v4_dryrun/heldout_removal_report.csv
```

This exercises the whole 33A path on real metadata (fail-closed bundle,
year hold-out, quarantine counts, event-grouped splits, provenance). It
writes no waveform. Keep the directory out of git.

## 9. Loader smoke test on real rows, with the ledger gate (30 min)

```bash
python - <<'PY'
import sys; sys.path.insert(0, "scripts")
from fast_manifest_dataset import CachedManifestDataset
import pandas as pd
pd.read_csv("data/manifests_v2/train.csv").sample(2000, random_state=0).to_csv("/tmp/smoke_train.csv", index=False)
ds = CachedManifestDataset("/tmp/smoke_train.csv", label_policy="masked", return_mask=True, load_workers=8)
print(len(ds), ds.n_supervised_samples)
PY
if ls /tmp/smoke_train.rejected.*.jsonl >/dev/null 2>&1; then
  echo "rejections written:"; head -3 /tmp/smoke_train.rejected.*.jsonl
  python scripts/run_card.py check /dev/null || true     # exit 2 shows the gate would refuse this manifest
else
  echo "no rejection: every sampled row was read under the 34A contract"
fi
```

The 34A loader raises on the first row it cannot read; the ledger names
it. On the historical manifests some rejections are expected (missing rate
metadata, S-only rows the old builder dropped, bucketed names on direct
routes): each is a row to repair in a corrected manifest, never a row to
skip. This is the migration list of
`docs/2026-09-10_34a_loader_contract.md`.

## 10. What is then unblocked

| Released by this session | Unblocks |
|---|---|
| 34B `phase_summary.csv` | the group report's missing number; 46A/E0 design |
| task 1 counts and the certified bundle (33A) | 40A pilot corpus build; every future manifest |
| 39A SeisBench table | T0 source list |
| loader smoke test | 46A configs `configs/e0_46a_*_targets.yaml` become runnable |

Still needed before E0 starts: 34C (benchmark timebase and deployment
parity; a laptop task with the torch venv and the local weights), 35C
(calibrated baselines; needs 38A's availability table from the continuous
archives, which is a server download), and 44B (panel freeze after 37B).
E0 itself is three arms × three seeds on the v7 rows; per-epoch time from
`results/finetune_jma_wc_global_v7_metrics.csv` sets the schedule.
