# PhaseNet Retraining Framework

Code, contracts and test assets for retraining **PhaseNet** (P and S onset
picking) into a general picker for a global campaign with local resolution:
mainshock–aftershock sequences, volcano-tectonic sequences and fluid-driven
swarms, on land stations at their native sampling rates. Developed by the
**Denolle Lab**, University of Washington, Department of Earth and Space
Sciences.

> **Status (2026-09-15).** No model has been trained with this code yet.
> Twenty fine-tunes of the SeisBench `jma_wc` weights made in 2026 were
> re-scored in September and found indistinguishable from their parent at
> matched pick budget; two audits then traced the training path's defects
> (`docs/2026-09-11_silent_zero_windows_report.md`,
> `docs/2026-09-10_picker_and_issue_roadmap_audit.md`). What is in this
> repository now is the repaired pipeline and the measurement apparatus,
> released checkpoint by checkpoint against issues #33–#50, validated on
> fixtures and on the built held-out cases, and awaiting its first session
> against the SeisBench cache on the lab server
> (`docs/2026-09-13_server_session_runbook.md`). The strategy for the next
> model is `docs/2026-09-11_training_strategy_v3.md`; the plan of record is
> `docs/2026-09-09_issue_plan.md`. The data are not hosted here: the
> SeisBench corpora live on the lab servers (`data/README.md`).

## What the pipeline enforces

Every stage fails closed. A manifest cannot be built without a certified
exclusion bundle; a waveform that cannot be read is rejected into a ledger
and stops the run; a training run writes a run card before its first
optimiser step and is not a result without one; a routine scoring call is
refused on a protected acceptance case.

```
SeisBench cache + bulletins
   │  exclusion_bundle.py   (33A) versioned (dataset, chunk, trace_name) identity, held-out windows/places/years, quarantine
   ▼
build_training_dataset.py  → manifests with rate, component, support and arrival metadata + provenance.json
   │  manifest_dataset.py   (34A) stored-rate contract, bucketed HDF5, polyphase resampling, rejection ledger
   │  label_targets.py      (41A) arrival provenance tiers → PSN targets + supervision mask
   │  augmentation.py       (43A) label-consistent transforms; noise_ontology.py / build_noise_pool.py (42A)
   ▼
finetune.py                → checkpoints + run_card.json (rows read, rejections, hashes, versions)
   │
   ▼  evaluation on continuous data, guarded by configs/evaluation_suites.json (44A)
heldout_testset_score.py   (35A) stored annotations, per-threshold trigger extraction, optimal matching, pick store
event_association.py       (36A) frozen associator config, one-to-one event matching, recovery vs magnitude/hour/day
certify_evaluability.py    (37A) what each built case can support;  calibration_protocol.py (38A) operating points
source_census.py           (39A) manual P/S supply per operator-year with intervals
```

## Repository layout

```
configs/      evaluation_suites.json            — suite roles: regression / dev / calibration / acceptance
              e0_46a_{legacy,masked}_targets.yaml — the first experiment's two arms (not yet run)
              association/{msas,vt,swarm}.json  — frozen, hashed associator parameters per regime
              finetune_jma_wc_global_v*.yaml    — the 2026 fine-tune history, kept as evidence
scripts/      the modules named in the diagram above, plus the historical builders and evaluators
tests/        361 tests (pytest); loader, loss and forensics tests need torch, SeisBench and h5py
data/         heldout_testset/ (19 sequences: picks, catalogues, windows, stations; waveforms not committed)
              census/, noise_pools/pilot_2019/ (committed manifests only), exclusions/ (bundle, built on the server)
docs/         audits, strategy, checkpoint contracts, baselines_2026-09-13/, the held-out atlas (GitHub Pages)
paper_draft.qmd / .html   — the project paper; §Critical audit carries the 2026-09 retractions in place
```

## Installation

[Pixi](https://pixi.sh) pins the whole stack per platform (macOS arm64 and
x86_64, Linux x86_64; CUDA 12 on Linux) in `pixi.lock`; no conda or venv
by hand.

```bash
curl -fsSL https://pixi.sh/install.sh | sh                 # once, no root; installs to ~/.pixi
git clone https://github.com/Denolle-Lab/phasenet-retrain.git && cd phasenet-retrain
pixi install                                               # CPU environment, all scripts and tests
export SEISBENCH_CACHE_ROOT=/path/to/seisbench/cache
pixi run versions                                          # python, numpy, scipy, torch, seisbench, obspy, pyocto, cuda
pixi run test                                              # 450 passed with the cached weights, 3 skipped without
pixi shell                                                 # or: an activated shell, then plain `python scripts/...`
```

GPU training on the lab server: `pixi install -e cuda` and prefix commands
with `pixi run -e cuda`. The lock resolves PyTorch's CUDA 12.9 build; if
`nvidia-smi` reports a driver below CUDA 12.9, add `cuda-version = "12.<x>.*"`
to the `cuda` feature in `pixi.toml` and run `pixi lock`. Every script is
also a task (`pixi run train --config ...`, `pixi run score ...`,
`pixi run bundle build ...`); `pixi task list` shows them.

`requirements.txt` remains for pip users; the pure-pandas parts (exclusions,
scoring, association, census, protocol) run without torch, and the loader,
loss and forensics tests skip when torch, SeisBench or h5py are missing.
The pixi environment is the supported one.

## Usage

```bash
# Exclusions: the held-out sequence list is produced on the server, then the bundle certifies every input
python scripts/audit_heldout_sequences.py
python scripts/exclusion_bundle.py build --cache-root $SEISBENCH_CACHE_ROOT && python scripts/exclusion_bundle.py show

# Manifests (refuses to run without a certified bundle; writes provenance.json and a removal report)
python scripts/build_training_dataset.py --output-dir data/manifests_v4

# Training (ledger gate, run card; the E0 arms are the only configs meant for the corrected pipeline)
python scripts/finetune.py --config configs/e0_46a_masked_targets.yaml
python scripts/run_card.py check results/<run_name>/run_card.json

# Scoring on continuous data (regression and development cases only; acceptance cases are refused)
python scripts/heldout_testset_score.py --all --weights jma_wc instance \
    --thresholds $(python -c "print(' '.join(f'{x/100:.2f}' for x in range(2, 92, 2)))")
python scripts/event_association.py --sequence samos_2020 --config configs/association/msas.json ...

# Historical forensics of the 2026 fine-tunes (read-only, server)
python scripts/audit_v7_rows.py --manifest-dir data/manifests_v2 --cache-root $SEISBENCH_CACHE_ROOT --inventory-only --output results/34b/inventory
```

Each checkpoint has a contract document under `docs/` (`2026-09-10_34a_loader_contract.md`,
`2026-09-11_35a_scoring_engine.md`, `2026-09-11_36a_event_association.md`,
`2026-09-11_41a_label_policy.md`, `2026-09-11_43a_augmentation_contract.md`,
`2026-09-11_33a_exclusion_contract.md`, `2026-09-11_38a_calibration_protocol.md`,
`2026-09-11_run_card.md`) stating what it guarantees and what it leaves open.

## Status and open items

- **Nothing trained.** The first experiment (E0, checkpoint 46A: same rows,
  legacy versus corrected loader, three seeds) waits for 33A, 34B, 34C, 35C
  and 44B, in that order (`docs/2026-09-10_issue_execution.md`).
- **Server-side checkpoints not run**: the 34B replay that counts how many
  historical rows were zero windows, the held-out exclusion counts, the
  bundle build, the SeisBench half of the source census.
- **Baselines** on the seven regression and development cases
  (`docs/baselines_2026-09-13/`): `instance` leads `jma_wc` on every case and
  phase at the parent's budget; the deployed fine-tune export does not.
  Provisional until 34C verifies the deployment path.
- **Held-out acceptance cases** are protected: their picks and waveforms are
  built, their evaluability is certified (`data/heldout_testset/evaluability.csv`),
  and no model output on them has been read.
- **Out of scope this round**: ocean-bottom data and the surface-event
  picker, which has its own roadmap (`docs/SU_PICKER_IMPLEMENTATION_PLAN.md`,
  issues #53–#61).

## Citation

If you use this code, please cite PhaseNet, SeisBench, and the deployment papers
this work supports:

- **Zhu, W. & Beroza, G. C. (2019).** PhaseNet: a deep-neural-network-based
  seismic arrival-time picking method. *GJI* 216(1), 261–273.
  `10.1093/gji/ggy423`
- **Woollam, J., et al. (2022).** SeisBench — A toolbox for machine learning in
  seismology. *SRL* 93(3), 1695–1709. `10.1785/0220210324`
- **Münchmeyer, J., et al. (2022).** Which picker fits my data? *JGR Solid Earth*
  127, e2021JB023499. `10.1029/2021JB023499`
- **Ni, Y., et al. (2025).** A Global-scale Database of Seismic Phases from
  Cloud-based Picking at Petabyte Scale. *Seismica* 4(2), 1738.
  `10.26443/seismica.v4i2.1738`
- **Ni, Y., et al. (2025).** A review of cloud computing and storage in
  seismology. *GJI* 243(1), ggaf322. `10.1093/gji/ggaf322`
- Label-error method: **Aguilar, A. L., et al.** —
  [`albertleonardo/labelerrors`](https://github.com/albertleonardo/labelerrors).

Full dataset references and DOIs: see the paper, §Datasets.

## Contact

**Denolle Lab**, University of Washington — [denolle-lab.github.io](https://denolle-lab.github.io).
For questions or collaboration, open an issue.
