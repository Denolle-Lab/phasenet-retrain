# Data inventory — *not stored in this repository*

The waveform datasets used to train and benchmark these models are **far too
large to host on GitHub** (hundreds of GB to multiple TB). They live on the
Denolle Lab back-end Linux servers as the local **SeisBench cache**, and are
pulled from the SeisBench data repositories on first use.

- **Server cache location (current):** `/data/wsd04/ak287/.seisbench`
  *(hard-coded in the build/train/eval scripts — see "Known issue" below).*
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
`../notebooks/benchmark_pool_summary.csv` where available). `Train cap` = max
traces drawn into the hybrid pool before distance stratification
(`build_training_dataset.py`). `On-disk` = HDF5 waveforms + CSV metadata in the
SeisBench cache — **measure on the server** (command below); approximate values
are marked `~` and should be confirmed.

| Dataset (SeisBench) | Full traces | Train cap | On-disk (HDF5+CSV) | Region / regime |
|---|--:|--:|--:|---|
| `scedc`          | 8,035,833 | 60,000  | measure | Southern California |
| `neic`*          | 1,354,789 | —       | measure | Global (benchmark candidate, excluded) |
| `stead`          | 1,265,657 | 100,000 | ~75 GB  | Global |
| `lendb`          | 1,244,942 | 40,000  | measure | Global local + noise |
| `instancecounts` | 1,159,249 | 100,000 | ~160 GB | Italy |
| `txed`           |   519,689 | 40,000  | measure | Texas (induced) |
| `geofon`         |   275,274 | 150,000 | measure | Global / teleseismic |
| `pnw`            |   183,909 | 40,000  | measure | Cascadia / Pacific NW |
| `obst2024`       |    60,394 | 60,000  | measure | Ocean-bottom (Phase-2) |
| `ethz`           |    36,743 | 60,000  | ~10 GB  | Switzerland / Alpine |
| `iquique`        |    13,400 | 13,400  | measure | N. Chile subduction |
| `ceed`           | confirm   | 100,000 | measure | California event dataset |
| `mlaapde`        | confirm   | 80,000  | measure | Global (PDE), teleseismic |
| `crew`           | confirm   | 30,000  | measure | confirm |
| `cwa`            | confirm   | 30,000  | measure | Taiwan |
| `pisdl`          | confirm   | 10,000  | measure | confirm (volcanic?) |
| `vcseis`         | confirm   | 30,000  | measure | Volcano seismicity |
| `aq2009gm`       | confirm   | 60,000  | measure | L'Aquila 2009 ground motion |
| `meier2019jgr`   | confirm   | 150,000 | measure | Global |
| `ross2018gpd`    | confirm   | 200,000 | measure | Southern California (GPD) |
| `obs`            | confirm   | 100,000 | measure | Ocean-bottom (Phase-2) |

`*` `neic` was surveyed as a benchmark candidate but **excluded** (0% traces with
both P and S picks). It is not in the training pool.

**Derived training manifests** (built from the pool; not stored here):
`manifests_v2/{train,val,test}.csv` (~527k train, ~38% S),
`manifests_v3/{train,val,test}.csv` (S-balanced, ~60.5% S),
and the oversampled variants `train_tele2x.csv` (~662k, 2× teleseismic),
`train_v18.csv` (1.5× teleseismic), `train_p_focused.csv` / `val_p_focused.csv`
(local+regional only). **The scripts that derive the oversampled variants are
not yet committed — see the repo TODO list.**

**Noise corpus** (`noise_global`, ~82k traces) and the excluded `noise_prephase`
set are built by `../scripts/build_noise_dataset.py` /
`build_prephase_noise.py`.

## Measure actual on-disk sizes (run on the server)

```bash
# Per-dataset on-disk footprint in the SeisBench cache:
du -sh /data/wsd04/ak287/.seisbench/datasets/*/ | sort -h

# Total cache size:
du -sh /data/wsd04/ak287/.seisbench
```

Paste the output into the `On-disk` column above so the inventory is exact.

## Known issue — hard-coded cache path

The SeisBench cache path `/data/wsd04/ak287/.seisbench` is hard-coded in ~12
scripts. The repo will not run on another machine without editing those, or
without making the path configurable via the `SEISBENCH_CACHE_ROOT` environment
variable. Tracked in the repo TODO list.
