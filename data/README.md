# Data inventory — *not stored in this repository*

The waveform datasets used to train and benchmark these models are **far too
large to host on GitHub** (hundreds of GB to multiple TB). They live on the
Denolle Lab back-end Linux servers as the local **SeisBench cache**, and are
pulled from the SeisBench data repositories on first use.

- **Server cache location (current):** lab back-end Linux server, path set via
  the `SEISBENCH_CACHE_ROOT` environment variable (falls back to
  `~/.seisbench` if unset — see "Known issue" below).
- **This `data/` directory** holds only the generated **manifest CSVs**
  (lightweight index files listing `dataset_name`, `trace_name`, picks,
  distance, split) — and even those are git-ignored because the full manifests
  are large. Regenerate them with `scripts/build_training_dataset.py`.

> The science, provenance, and DOIs for every dataset are documented in the
> project paper (`../paper_draft.html`, §Datasets). This file is the operational
> inventory: counts, training caps, and where the bytes live.

## How the hybrid training set is assembled

20 SeisBench datasets → per-dataset cap + distance-bin stratification →
benchmark traces excluded → concatenated → resampled toward
`{local 0.40, regional 0.25, teleseismic 0.25, unknown 0.10}`.
Source of truth: [`../scripts/build_training_dataset.py`](../scripts/build_training_dataset.py)
(`DATASET_CONFIGS`, lines 209–319).

## Dataset volumes

`Full traces` = total traces in the SeisBench dataset (verified from
`../notebooks/benchmark_pool_summary.csv` where available, otherwise from a
direct row count of the on-disk `metadata*.csv` shard(s) — see below).
`Train cap` = max traces drawn into the hybrid pool before distance
stratification (`build_training_dataset.py`). `On-disk` = HDF5 waveforms + CSV
metadata in the SeisBench cache, measured directly on the server
(`du -sh datasets/*/`, 2026-07-20) — exact, not estimated.

| Dataset (SeisBench) | Full traces | Train cap | On-disk (HDF5+CSV) | Region / regime |
|---|--:|--:|--:|---|
| `scedc`          | 8,035,833 | 60,000  | 639 GB  | Southern California |
| `neic`*          | 1,354,789 | —       | 588 MB  | Global (benchmark candidate, excluded) |
| `stead`          | 1,265,657 | 100,000 | 86 GB   | Global |
| `lendb`          | 1,244,942 | 40,000  | 16 GB   | Global local + noise |
| `instancecounts` | 1,159,249 | 100,000 | 157 GB  | Italy |
| `txed`           |   519,689 | 40,000  | 70 GB   | Texas (induced) |
| `geofon`         |   275,274 | 150,000 | 26 GB   | Global / teleseismic |
| `pnw`            |   183,909 | 40,000  | 63 GB   | Cascadia / Pacific NW |
| `obst2024`       |    60,394 | 60,000  | 4.1 GB  | Ocean-bottom (Phase-2) |
| `ethz`           |    36,743 | 60,000  | 22 GB   | Switzerland / Alpine |
| `iquique`        |    13,400 | 13,400  | 5.0 GB  | N. Chile subduction |
| `ceed`           | 5,009,718 | 100,000 | 575 GB  | California event dataset |
| `mlaapde`        | 510,196†  | 80,000  | 62 GB   | Global (PDE), teleseismic |
| `crew`           | 1,599,323 | 30,000  | 1.1 TB  | confirm |
| `cwa`            | 346,959†  | 30,000  | 173 GB  | Taiwan |
| `pisdl`          | 142,001   | 10,000  | 35 GB   | confirm (volcanic?) |
| `vcseis`         | 160,278   | 30,000  | 47 GB   | Volcano seismicity |
| `aq2009gm`       | 258,984†  | 60,000  | 27 GB   | L'Aquila 2009 ground motion |
| `meier2019jgr`   | 1,060,433 | 150,000 | 24 GB   | Global |
| `ross2018gpd`    | 4,773,750 | 200,000 | 43 GB   | Southern California (GPD) |
| `obs`            | 109,208   | 100,000 | 33 GB   | Ocean-bottom (Phase-2) |

`†` `mlaapde`, `cwa`, and `aq2009gm` each have one trailing monthly/decade shard
with a `.partial` suffix (an interrupted SeisBench download) — their `Full
traces` counts are a real row count of the shards present on disk, not a
verified-complete total; a couple percent more may exist once the partial
shard is finished.

**Total cache size (all datasets + models, 2026-07-20): 3.1 TB.**

`*` `neic` was surveyed as a benchmark candidate but **excluded** (0% traces with
both P and S picks). It is not in the training pool.

**Derived training manifests** (built from the pool; not stored here):
`manifests_v2/{train,val,test}.csv` (~527k train, ~38% S),
`manifests_v3/{train,val,test}.csv` (S-balanced, ~60.5% S),
and the oversampled variants `train_tele2x.csv` (~662k, 2× teleseismic),
`train_v18.csv` (1.5× teleseismic), `train_p_focused.csv` / `val_p_focused.csv`
(local+regional only) — all four reproducible via
`../scripts/build_oversampled_manifests.py --all` (fixed 2026-07-20, GitHub
#6). Verify any regenerated manifest against the committed fingerprints with
`../scripts/hash_manifests.py --check` (hashes in
`manifest_checksums.csv`).

**Noise corpus** (`noise_global`, ~82k traces) and the excluded `noise_prephase`
set are built by `../scripts/build_noise_dataset.py` /
`build_prephase_noise.py`.

## Measuring on-disk sizes

The `On-disk` column above was filled in 2026-07-20 by running, on the server
(`$SEISBENCH_CACHE_ROOT`, see "Known issue" below):

```bash
# Per-dataset on-disk footprint in the SeisBench cache:
du -sh "$SEISBENCH_CACHE_ROOT/datasets/"*/ | sort -h

# Total cache size:
du -sh "$SEISBENCH_CACHE_ROOT"
```

Re-run this and update the table if datasets are added, re-downloaded, or a
`.partial` shard (see `†` note above) is completed.

## Cache path (fixed 2026-07-12, GitHub #9)

Every script now reads `SEISBENCH_CACHE_ROOT` (falling back to
`~/.seisbench`) instead of a hard-coded machine-specific path — a fresh clone
runs after setting that one env var.
