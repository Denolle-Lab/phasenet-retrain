# Training plan as GitHub issues

*2026-09-09. Rendered by `scripts/open_plan_issues.py`; `--create` opens them on `Denolle-Lab/phasenet-retrain`. Branch `audit/2026-09-07-generalization`.*


## Phase 0: evaluation before training

*Weeks 1-3. Task 1 on the server, baselines on the held-out set, the event-level scorer, thresholds by region.*


### [Phase 0] Run the held-out sequence exclusion on the server and commit the list

`plan`, `server`

**Goal.** Make the 23 held-out windows and places (`scripts/heldout_sequences.py`) real exclusions: counts per dataset and window, and the committed `data/exclusions/heldout_sequences.csv` the builder refuses to run without.

**Steps** (on the server, branch `audit/2026-09-07-generalization`):
```bash
export SEISBENCH_CACHE_ROOT=/path/to/cache
python scripts/hash_manifests.py --check                    # manifests must match data/manifest_checksums.csv
python scripts/audit_heldout_sequences.py                    # joins manifests_v2, the ten full corpora, the benchmark
git add data/exclusions && git commit
```
**Acceptance.** `heldout_sequence_counts.csv` and the list committed; the counts table pasted into `docs/2026-09-08_heldout_test_cases.md`; `vcseis` and `crew` date ranges read from their metadata to settle the Hawaii and Alaska tiers.
**Depends on.** Server access. Blocks every manifest build.


### [Phase 0] Server-side hypothesis tables: S fraction by distance (H2) and the full noise-pool ordering (H00)

`plan`, `server`

**Goal.** Finish tasks 2-3 of the 2026-09-07 audit where the laptop could not: the S-label fraction of `manifests_v2/train.csv` by distance bin, and the detection MCC of every version on the noise pool.

**Steps.** `python scripts/audit_generalization_hypotheses.py h2 h00` (reads the manifest and `results/detection_metrics.csv` when present). For H00, extend `SINGLE_MODELS` in `scripts/audit_noise_fp_leaderboard.py` to every `checkpoints/finetune_jma_wc_global_v*/best.pt` still on disk, rerun it and `scripts/compute_detection_metrics.py`, then `h00` again.
**Acceptance.** `docs/audit_2026-09-07/h2a_s_fraction_by_distance_manifest.md` and an `h00` table covering every surviving checkpoint; the paper's item 15 updated with the numbers.
**Depends on.** Server access; which checkpoints were kept.


### [Phase 0] Rebuild the held-out test set on the server and score the existing weights at matched budget

`plan`, `server`

**Goal.** The baseline every candidate has to beat, on the 19 sequences, before any training: `jma_wc`, `instance`, `jma_wc_ft_global_v11` (best single fine-tune on the noise pool), and the PhaseNet ensembles `jma_wc`+`instance` and v7+v11.

**Steps.** The picks are committed; waveforms re-fetch from the registry: `python scripts/build_heldout_testset.py --all all` (about 20 min, 1 GB) then `--verify` each key. Convert v11 with the QuakeScope path (`docs/phasenet_v7_model_description.md` §Reproduction) and score: `python scripts/heldout_testset_score.py --all --weights jma_wc instance <v11 pair> --out docs/audit_2026-09-07/heldout_scores.csv`. Add ensemble support to the scorer (average the probability curves) if missing.
**Acceptance.** Recall at 0.3 and at matched budget per sequence, phase and window with bootstrap intervals; a decision on the campaign picker for this quarter written into the plan. Gate: a candidate replaces `jma_wc` only if it leads at matched budget on at least five of the six non-US mainshock pairs and does not lose on the volcano and swarm cases.
**Depends on.** Server GPU; `checkpoints/finetune_jma_wc_global_v11/best.pt`.


### [Phase 0] Event-level scorer: PyOcto association against the reference catalogues

`plan`

**Goal.** The metric the campaign is judged on is events, not picks. Associate each weight's picks with PyOcto, match to the operator catalogue, report events recovered by magnitude, by hour after the mainshock, and by day for swarms, plus picks per station-hour.

**Steps.** `scripts/heldout_testset_events.py` reading `data/heldout_testset/<key>/` and the scorer's pick store; regional 1-D velocity models per sequence in the registry; matching tolerance in origin time and location recorded with the results.
**Acceptance.** Tables for Kaikōura, La Palma and West Bohemia (one per regime) with intervals; unit tests on the matching.
**Depends on.** The pick-level scorer output (previous issue).


### [Phase 0] Obtain the restricted picks and waveforms

`plan`, `needs-access`

**Goal.** Close the gaps listed in `data/heldout_testset/README.md`.

- **Hi-net account** (NIED): Noto 2024 picks and waveforms, Noto swarm waveforms; `HinetPy` (`Client.get_arrivaltime`, `get_waveform`).
- **CWA GDMS account**: Hualien 2024 CWASN waveforms.
- **IMO**: SIL manual picks and the station list for Reykjanes 2023 and Fagradalsfjall 2021 (email).
- **AFAD**: bulk manual readings for Kahramanmaraş 2023 (or the ISC copy once reviewed).
- **Zagreb**: the 255,729 hand-picked onsets of Petrinja (Tectonophysics 2023 authors).
- **WEBNET**: the Zenodo tarball `waveforms.tar.gz` (966 MB) once Zenodo stops answering 403.
- **INGV-OV**: Campi Flegrei bulletin picks.

**Acceptance.** Each item either added to the registry and rebuilt (`build_heldout_testset.py --sequence KEY all --force`) or marked unobtainable with the reason.
**Owner.** Marine (accounts and emails); the pipeline already has a slot for each.


### [Phase 0] Thresholds per weight and per region from quiet-day false-pick rates

`plan`, `cross-repo`

**Goal.** Replace the shared 0.3 with a threshold set per weight set and region to a false-pick target on quiet station-days, recorded with the campaign.

**Steps.** In QuakeScope: sample quiet station-days per region (no catalogued event within the window), run each weight, count picks per station-day versus threshold, choose the threshold that meets the target; store in the campaign config with provenance.
**Acceptance.** A table of thresholds by weight and region and the false-pick rate at each; the two QuakeScope notebooks re-run at those thresholds.
**Depends on.** Nothing in this repo; needs the campaign archives.


## Phase 1: signal and noise corpora

*Weeks 4-11. Bulletin-labelled corpus at the scale that made jma_wc good, the eleven-flavour noise corpus, augmentation, frozen suites.*


### [Phase 1] Operator bulletin census: how many manual picks each FDSN event service can give

`plan`

**Goal.** Decide the corpus size and operator list from counts, not guesses: events per year, picked stations per event, manual fraction, open-waveform fraction, per operator.

**Steps.** `scripts/census_operator_bulletins.py` using the harvest code of `scripts/build_heldout_testset.py` on one week per year per operator: GeoNet, INGV, NOA, USGS ComCat (NC, CI, UW, NN, AK, HV), then tested one by one NRCan, GA, IMO, SED, KOERI, AFAD, CSN, SSN, GFZ, RESIF, IGN, ISC. Apply the held-out windows and years.
**Acceptance.** A table per operator and year; a target corpus size with the distance, SNR and event-type mix estimated; runs from the laptop.
**Depends on.** Nothing.


### [Phase 1] Bulletin-labelled corpus builder: 60 s windows with every arrival labelled

`plan`, `server`

**Goal.** The training corpus of the plan §4: windows cut from continuous data at native rate, every catalogued arrival in the window labelled (the aftershock regime), with SNR, distance, magnitude, rate, instrument, operator, year, event type and event count per window; SeisBench analyst P+S sets added with the same exclusions.

**Steps.** `scripts/build_bulletin_corpus.py` (harvest → windows → HDF5 + manifest), stratified sampling to the composition targets (≥35 % below 5 dB, ≥30 % multi-event, ≥60 % with S, regional-heavy, ≥15 % volcano-tectonic and swarm settings, no operator above 30 %), `hash_manifests.py` fingerprints, `audit_heldout_sequences.py --check-manifest` on train and val.
**Acceptance.** A manifest family `data/manifests_v4/` with composition summary and removal reports; the check passes; a 1 % sample plotted.
**Depends on.** The census; the exclusion list (Phase 0 issue 1).


### [Phase 1] Label quality: confident learning plus physical checks per operator

`plan`, `server`

**Goal.** Catch the reference problems the audit found (a per-operator S cap, automatic picks flagged manual).

**Steps.** Run `scripts/label_error_filter.py` on the new corpus; add S−P versus distance residual screening per operator; a per-operator recall cap test with `jma_wc` on a held-out slice, of the kind that exposed Thessaly's S at 0.5.
**Acceptance.** Removal fractions per operator in the manifest directory; a short note on any operator whose picks cannot be used as a reference.
**Depends on.** The corpus builder.


### [Phase 1] Noise corpus: eleven flavours harvested by catalogue exclusion, held out by station

`plan`, `server`

**Goal.** Plan §5: 120 s windows at native rate from the campaign's archives, in eleven classes (microseism, wind and tilt, cultural, hydrological, impulsive non-earthquake, volcanic tremor, tectonic tremor, sequence hum and coda, polar and ice, instrument and telemetry, quiet baseline), ≥20 k per class, no model screening, classes from source or spectral features, the held-out windows and places excluded, and a held-out split by station.

**Steps.** `scripts/build_noise_corpus.py`; the existing `data/noise_global` pool folded in with its obst2024 traces dropped; class census and a per-class spectrogram sheet.
**Acceptance.** `data/noise_v2/` with metadata (class, station, rate, site, season), the census, and the held-out station list.
**Depends on.** Nothing; can start now from the laptop for the FDSN-served archives.


### [Phase 1] Augmentation module: class-balanced real noise, event superposition, artefacts, resampling, random position

`plan`

**Goal.** Plan §5.3 in `CachedManifestDataset.__getitem__`: class-balanced noise draw with SNR weighted to 0-10 dB, non-stationary mixes, a second labelled window added with merged labels, coda backgrounds, spikes/steps/gaps/clipping/mains/drift, band-limiting and 20-100 Hz resampling, random window position, pure-noise negatives at 15 % of a batch, 30 % untouched.

**Steps.** `scripts/augment.py` with a config block; unit tests that labels survive every transform (positions shift with the window, merge on superposition); a notebook page of twenty augmented examples.
**Acceptance.** Tests pass; the config for each arm of Phase 2 is expressible.
**Depends on.** The noise corpus for real runs; tests can use synthetic noise.


### [Phase 1] Freeze the development and acceptance suites and enforce the never-read rule

`plan`

**Goal.** The acceptance suite (Kaikōura, Norcia, Thessaly, Ridgecrest, Monroe, 2016, 2021, Kahramanmaraş, Noto 2024, Hualien, Petrinja, Reykjanes, La Palma, Santorini-Amorgos, West Bohemia, Maurienne, Noto swarm, Campi Flegrei) is scored once, at the end. The development suite (Samos, Adriatic, Etna, Corinth-Thiva, Hawaii and Alaska as tier 2) decides between runs.

**Steps.** `suite` in the registry already; add a guard in the scorer that refuses acceptance keys unless `--acceptance-run` is passed and logs the run to `data/heldout_testset/acceptance_log.csv`.
**Acceptance.** The guard and the log exist; the plan's gate table points at them.
**Depends on.** Nothing.


## Phase 2: training experiments

*Weeks 12-22. The data-scaling curve first, then init/anchor, noise ablation, width and context. PhaseNet only.*


### [Phase 2] E1: the data-scaling curve, before any recipe work

`plan`, `server`

**Goal.** The experiment the twenty versions never ran: does more data of this kind move the parent at all? Init `jma_wc`, distillation T = 1.5, α = 0.3, soft Gaussian targets, LR 5e-6, the augmentation module on, on nested subsets of 0.25, 0.5, 1, 2 and 4 M windows.

**Acceptance.** Matched-budget recall against the parent on the development suite versus corpus size, per regime, with intervals; the decision written down: continue, or stop the fine-tune line and keep the Phase 0 winner (flat at or below the parent by 2 M means stop).
**Depends on.** The corpus, the noise corpus, the augmentation module, the frozen suites, GPU time (a 4 M run is about eight times v7 per epoch).


### [Phase 2] E2: initialisation and anchor at the size E1 chose

`plan`, `server`

`jma_wc` init against from-scratch PhaseNetWC, and α = 0 against α = 0.3, at the largest corpus size that helped in E1. Scored on the development suite at matched budget. **Depends on** E1.


### [Phase 2] E3: noise ablation

`plan`, `server`

Arms without the class-balanced draw, without event superposition, without the non-stationary and artefact groups, and with white Gaussian noise in place of the corpus. Scored on false picks per station-day by noise class (held-out stations) and on the aftershock regime. **Depends on** E1.


### [Phase 2] E4: PhaseNet width and input length

`plan`, `server`

PhaseNetWC (the parent's 2× filters) against standard width at the same data; 60 s input against 30 s (fully convolutional, so the change is the window). Ensembles of the best arm with `jma_wc` as the deployment option if they win at matched budget. **Depends on** E1.


### [Phase 2] The acceptance run and the QuakeScope notebooks

`plan`, `server`, `cross-repo`

One candidate per line scored once on the acceptance suite with `--acceptance-run`, then converted with the QuakeScope path, placed in the `quakescope2026` slot, and both notebooks re-run. It must beat `jma_wc` at matched pick budget on the non-US mainshock sequences and not lose on the volcano and swarm cases. Until then the campaign stays on `jma_wc` or the Phase 0 winner. **Depends on** E2-E4.


## Phase 3: distant P and deployment

*Weeks 22-26. The 20 Hz distant-P model, thresholds by weight and region, the standing acceptance test.*


### [Phase 3] Distant-P model at 20 Hz on 120 s windows for offshore events and sparse regions

`plan`, `server`

P only, from GEOFON, MLAAPDE, CREW and ISC-labelled P at 3-30° from M ≥ 4, initialised from the SeisBench `geofon` weights, the noise recipe restricted to microseism, polar and quiet classes. Runs only on oceanic islands, coasts facing offshore seismicity, and where the nearest station is more than 300 km away; picks enter the associator with a global velocity model. Scored against ISC and NEIC by completeness magnitude versus nearest-station distance on offshore and African regions held out by year. No ocean-bottom data this round. **Depends on** the noise corpus; independent of Phase 2.
