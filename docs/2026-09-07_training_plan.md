# Plan: a general PhaseNet for a global run with local resolution

> **Superseded execution plan, 2026-09-10.** GitHub issues #33–#50 now follow the [applied roadmap](2026-09-09_issue_plan.md) and [checkpoint branch order](2026-09-10_issue_execution.md). The text below is retained as the scientific design history, not current instructions. Its data-volume conclusions, blind-suite designation and training order were revised after the [independent audit](2026-09-10_picker_and_issue_roadmap_audit.md) and [response review](2026-09-10_revised_issue_roadmap.md).

*v2, 2026-09-08. Marine Denolle with Claude, branch
`audit/2026-09-07-generalization`. Re-audited after Marine's decisions of
2026-09-08: PhaseNet only, natural noise as a first-class part of the
training data, and three held-out test regimes. Nothing here has been run.*

**Changes from v1 (2026-09-07).** EQTransformer arms removed; the
architecture question is now width and input length inside PhaseNet, and
ensembles are PhaseNet-only. The single line on "real-noise superposition"
became §5, a noise corpus with its own taxonomy, harvest, hold-outs and
scoring. The acceptance suite is rebuilt around three regimes
(`docs/2026-09-08_heldout_test_cases.md`) and the volcano and swarm cases
are held out as places, not time windows. Akash's curated benchmark stays
as a unit test of timing on isolated arrivals.

**Scope decision, 2026-09-08.** Ocean-bottom observations are out of this
round: no OBST2024 or `obs` traces in the signal corpus, no ocean-bottom
noise class, and the obst2024 portion of the existing `data/noise_global`
pool (about 25,000 traces, identifiable by the `obst2024_` prefix of
`trace_name`) is dropped. Offshore *events* recorded on land and island
stations stay in scope; that is what the distant-P model of §7 is for.

## 1. What the audit fixes about the design

Twenty finetunes of `jma_wc` on 527,477 windows from twenty sources did
not beat the parent, which was trained on 6.1 million waveforms with one
labelling convention. The one gain that survived a paired comparison was
19 ms of P timing on picks both models make, and it did not show on
continuous data. The conclusion is about data, not optimisation: half a
million heterogeneous windows can perturb a model of this size but not
re-teach it, and every version paid for the perturbation in recall.

The parent's own weaknesses, measured, are the targets.

| Weakness of `jma_wc` | Where measured | Size |
|---|---|--:|
| Regional S | benchmark, 150–1500 km | S recall 0.36 [0.35, 0.38] |
| Low-SNR P | benchmark, below 0 dB | P recall 0.66 [0.64, 0.67] |
| Precision per pick | Kaikōura, Norcia, Thessaly at matched budget | `instance` leads P by 3–12 points along the whole overlap |
| False triggers on noise | noise pool, own best threshold | precision 0.83 |

Two more were built into the training pipeline and never measured. The
training windows carried one labelled pick each
(`scripts/manifest_dataset.py:227-247`); a second event in the same 30 s
was in the waveform and absent from the label, which teaches the model to
suppress exactly the arrivals an aftershock sequence is made of. And every
waveform was resampled to 100 Hz once, so nothing saw the 20, 40 and 50 Hz
instruments the campaign upsamples.

The teleseismic objective fought the regional one every time it was
raised (v13, v16, v17). Those two jobs get two models (§7).

## 2. The target, written so it can be scored

The campaign runs one PhaseNet over every network EarthScope holds. What
it has to do well, in order:

1. **Local and regional sensitivity**, 0 to 300 km, down to the operator's
   completeness in dense networks and below it in sparse ones. Scored as
   recall at matched pick budget against manual picks per phase and
   distance bin, and as false picks per station-day on quiet days, by
   noise class.
2. **Mainshock-aftershock sequences.** The first 48 hours after an M6+,
   events seconds apart, coda everywhere. Scored at the event level after
   association: the fraction of the operator's located events recovered,
   by magnitude and by hour after the mainshock; picks per station-hour;
   residuals against the manual picks.
3. **Volcano-tectonic sequences** (dike intrusions, pre-eruptive unrest)
   and **fluid-driven swarms** (magmatic and hydrothermal), which migrate
   for weeks to months, have emergent onsets, weak S, and tremor
   underneath. Scored as events recovered against the published catalogue
   by magnitude and by day, with the migration front recovered or not.
4. **Distant P for completeness where there are no stations**: offshore
   events recorded on land and island stations, and sparse regions such as
   most of Africa. A separate model (§7), scored against ISC and NEIC by
   completeness magnitude versus nearest station distance.

The acceptance suite is fixed now and never read during development: the
five sequences already held out, the years 2016 and 2021, and the tier-1
cases of `docs/2026-09-08_heldout_test_cases.md` marked "acceptance"
(Kahramanmaraş 2023, Noto 2024, Hualien 2024, Petrinja 2020–21; Reykjanes,
La Palma, Santorini–Amorgos; West Bohemia 2018, Maurienne 2017–19, the
Noto swarm, Campi Flegrei 2023–24). The cases marked "development" (Samos,
Adriatic 2022, Etna, Corinth–Thiva, and the tier-2 Hawaii and Alaska
cases) serve every decision before the final one. Mayotte, whose
reference catalogue rests on ocean-bottom instruments, waits for the OBS
round. All of them are
now enforced as exclusions in `scripts/heldout_sequences.py`.

## 3. Phase 0, before any training (two to three weeks)

1. **Run task 1 on the server** (`python scripts/audit_heldout_sequences.py`)
   with the 23 windows now defined; commit the list and the counts. The
   place hold-outs will cost INSTANCE its Etna and Campi Flegrei traces
   and CREW whatever it holds around the other places; the counts say
   whether that is affordable. Read the `vcseis` and `crew` metadata for
   their date ranges to settle the Hawaii and Alaska tiers.
2. **Rank the candidates that already exist at matched budget** on the
   external suite: `jma_wc`, `instance`, `jma_wc_ft_global_v11` (the only
   finetune above the parent on the noise pool, 0.804 [0.799, 0.808]
   against 0.776 [0.771, 0.780]), and two PhaseNet-only ensembles that
   cost no training, `jma_wc` + `instance` and v7 + v11, probability
   curves averaged. Ensembles were the best PhaseNet entries on the noise
   pool and no one has scored one on continuous data.
3. **Set thresholds per weight and per region** to a false-pick target on
   quiet station-days, not to 0.3.
4. **Build the event-level scorer**: PyOcto over the picks, matched to the
   reference catalogue, events recovered by magnitude, hour and day.
5. **Run the two extra regimes on the existing weights** (West Bohemia
   2018 and Campi Flegrei 2023 are open and small) to have the baseline
   numbers the training has to beat.

Gate: whichever candidate leads at matched budget on at least five of the
six non-US pairs and lowers the quiet-day false-pick rate replaces
`jma_wc` in the campaign now. Training continues regardless.

## 4. Phase 1, the signal corpus (six to eight weeks)

**Sources.** Arrivals harvested from every operator whose FDSN event
service returns manual picks, the code path of the notebooks: GeoNet,
INGV, NOA, USGS ComCat for the US networks, and, tested one by one,
NRCan, Geoscience Australia, IMO, SED, KOERI, AFAD, CSN, SSN, GFZ, RESIF,
IGN. Only manual P and S, only open waveforms. The SeisBench sets with
analyst P and S (ETHZ, PNW, CWA, SCEDC and CEED, TXED, Iquique, INSTANCE
where the pick status says manual, VCSEIS for its volcano-tectonic and
long-period supervision) are added with the exclusions and the
label-error filter. No P-only sets; nothing beyond 2000 km; no
ocean-bottom data (OBST2024 and `obs` are skipped by
`scripts/build_training_dataset.py` this round).

**Windows.** Cut from continuous data at native sampling rate, 60 s long,
with every arrival of every catalogued event inside the window labelled.
This is the change that addresses the aftershock regime directly, and the
bulletin harvest gives it for free: all picks on a station in a time
range, not one pick per trace. Each window carries SNR, epicentral
distance, magnitude, sampling rate, instrument, operator, year, event
type where the operator gives one (VT, LP, hybrid, tectonic) and the
number of events it contains.

**Size.** A one-week census first: events per year times picked stations
per event, per operator, over the years the waveforms are open; 3 to 5
million windows over a decade is the target to verify.

**Composition targets**, enforced by stratified sampling: at least 35 % of
windows below 5 dB on the P window; at least 30 % with more than one
event; an S label in at least 60 %; at least 15 % from volcano-tectonic
and swarm settings (VCSEIS, Hawaii and Alaska if tier 2, INGV volcano
observatories other than the held-out places, PNW Cascades); a distance
mix that matches the campaign's station geometry, regional-heavy; no
operator above 30 %.

**Exclusions and hygiene.** The 23 windows and places, the 2016 and 2021
years, the development suite, benchmark traces and events, the label-error
filter; `scripts/hash_manifests.py` fingerprints;
`scripts/audit_heldout_sequences.py --check-manifest` must pass on train
and val.

## 5. Phase 1b, the noise corpus and how it is used

The parent's failure mode outside Japan is low-SNR P, and the campaign's
false picks come from noise it never saw. White Gaussian noise (v13 to
v16) is the wrong model of either: real noise is coloured, non-stationary,
often impulsive, and site-specific. The noise corpus is built with the
same care as the signal corpus, and the mixing recipe is what carries the
low-SNR objective.

**5.1 Taxonomy.** Eleven flavours this round, each a class the pool is
balanced over and the false-pick rate is reported by. The ocean-bottom
flavour (current-induced tilt, compliance, whale calls, ship harmonics,
airguns, hydrophone self-noise) is defined for the OBS round and not
harvested now.

| Class | What it is | Where it comes from |
|---|---|---|
| Ocean microseism, storm-modulated | primary 0.05–0.1 Hz and secondary 0.1–0.5 Hz peaks, weeks-long modulation; dominates islands and coasts | coastal and island stations of the campaign; sampled by the station's own 0.1–0.5 Hz power percentile so storms are over-represented |
| Wind and site tilt | broadband 1–10 Hz gusts, horizontal tilt below 0.1 Hz on shallow vaults | exposed and high-elevation stations; sampled on horizontal-to-vertical low-frequency ratio and spectral flatness |
| Cultural, diurnal | traffic, machinery, trains, pumps; 50/60 Hz mains and harmonics; HVAC lines; wind turbines at 1–5 Hz | urban and industrial stations, day and night separately, by hour of day |
| Hydrological | rivers, waterfalls, rain on the enclosure, snowmelt | stations near rivers and glaciers, spring and monsoon months |
| Impulsive non-earthquake | thunder, sonic booms, explosions and quarry blasts, surface events (rockfalls, avalanches), plane and vehicle impacts | PNW exotic classes (Ni et al. 2023), operator "explosion" and "landslide" labels, VCSEIS long-period class where it should not be picked as P/S |
| Volcanic tremor and hydrothermal noise | harmonic and spasmodic tremor, hydrothermal boiling noise, gas-piston events | INGV-OE and INGV-OV, HVO, IMO and AVO during eruptions and unrest, outside the held-out places |
| Tectonic tremor and LFEs | Cascadia ETS and Nankai tremor bursts, hours long | PNSN tremor catalogue windows; Hi-net where accessible |
| Earthquake coda and sequence hum | regional coda minutes after M5+, teleseismic coda hours after M7+, the continuous overlap of small aftershocks | continuous data from aftershock sequences NOT held out (e.g. Ridgecrest 2019 is held out; use Monte Cristo 2020, Sparta 2020, Zagreb 2020) |
| Polar and ice | icequakes, calving, sea-ice noise, wind on ice | Antarctic and Greenland stations of the campaign |
| Instrument and telemetry | spikes, DC steps, mass recentring pulses, calibration pulses, clipping, gaps, dropouts, timing glitches, aliasing from decimation, 4.5 Hz geophone and low-cost sensor self-noise, accelerometer noise floor, temperature drift | synthesised on the fly from a small set of rules, plus real examples harvested from station-day QC flags |
| Quiet baseline | the station at its quietest, all instrument types and rates | every station in the campaign, lowest 10 % power windows |

**5.2 Harvest.** From the campaign's own archives (SCEDC and NCEDC S3,
EarthScope, the operators' FDSN services), windows of 120 s at native
rate, from stations chosen to span instrument type (broadband,
short-period, strong-motion, geophone, low-cost), sampling rate (20 to
250 Hz), site (urban, rural, coastal, island, high-elevation, polar) and
continent, with an explicit quota for Africa, South America and Oceania. A window is noise when no catalogued event, global
M ≥ 2.5 or local M ≥ 0 where a local catalogue exists, has a predicted P or
S at the station inside the window or the 120 s before it. No
model-based screening: running `jma_wc` to reject windows it fires on
would bias the pool toward what the model already ignores, which is the
opposite of what is needed. A separate flag records what `jma_wc` fires
on, for scoring.

Class labels come from the source (PNW exotic, tremor catalogues, eruption
periods, OBS deployments) or from simple spectral features (microseism
band power, mains line power, horizontal-to-vertical ratio, kurtosis for
impulsiveness, hour of day). Target: 1 million windows, at least 20,000
per class, the held-out places and times excluded exactly as for the
signal corpus, and a held-out noise split by station, never by window, so
the false-pick rate is measured on stations the model never saw.

**5.3 Mixing recipe**, applied on the fly in `CachedManifestDataset.__getitem__`
to a signal window drawn from Phase 1.

- *Class-balanced draw.* With probability 0.6 a noise window is drawn
  from a class chosen uniformly, then a window uniformly within it, so
  rare flavours are seen as often as common ones.
- *SNR-controlled superposition.* The noise is resampled to the signal's
  rate, scaled to a target SNR on the P window drawn from a distribution
  weighted toward the hard end (40 % of draws in 0 to 5 dB, 30 % in 5 to
  10 dB, 30 % in 10 to 25 dB), and added. Labels unchanged.
- *Non-stationarity.* With probability 0.3 the noise level ramps or steps
  inside the window, or a second noise window from a different class is
  added over part of it, because real noise changes within 60 s.
- *Event superposition.* With probability 0.3 a second labelled signal
  window is added with a random offset of 2 to 40 s and the labels
  merged, on top of the multi-event windows the harvest already
  contains. This is the aftershock regime in the loss.
- *Coda background.* With probability 0.1 the signal window is placed on
  a sequence-hum window rather than a quiet one.
- *Instrument artefacts*, each with probability 0.02 to 0.05: spike, DC
  step, gap of 0.1 to 3 s, clipping at a random level, mains hum, slow
  drift, a channel zeroed, decimation-then-upsampling through 20, 40 or
  50 Hz.
- *Band-limiting and rate.* With probability 0.3 a low-pass corner from 8
  to 20 Hz, then resampling through 20 to 100 Hz and back.
- *Pure-noise windows* as negatives, 15 % of every batch, all-zero
  labels, drawn class-balanced.
- *Untouched fraction.* 30 % of signal windows receive no augmentation at
  all, so the model keeps its behaviour on clean data.
- Random window position, amplitude jitter and polarity flip as before.

**5.4 What the noise buys and how it is checked.** False picks per
station-day by noise class on the held-out noise stations; recall versus
SNR on the development sequences; and the noise-pool detection MCC of
`scripts/compute_detection_metrics.py` extended per class. A flavour that
the model fires on after training is a flavour the pool under-represents,
and the census is adjusted, not the threshold.

## 6. Phase 2, training experiments (eight to twelve weeks)

PhaseNet only. Every run uses soft Gaussian targets, LR 5e-6, early
stopping on a validation loss, the §5 mixing recipe, and distillation
from `jma_wc` at T = 1.5, α = 0.3 unless the arm says otherwise. Selection
is on the development suite at matched budget with paired intervals; the
acceptance suite is run once, at the end.

**E1, the data-scaling curve, first.** Initialise from `jma_wc` and train
on nested subsets of 0.25, 0.5, 1, 2 and 4 million windows. Plot
matched-budget recall against the parent on the development suite versus
corpus size, per regime. This is the experiment the twenty versions never
ran. If the curve is flat at or below the parent by 2 million, the
finetune line stops and the effort goes to §7 and the associator.

**E2, initialisation and anchor**, at the largest size that helped:
`jma_wc` init against from-scratch PhaseNetWC, and α = 0 against 0.3.

**E3, noise ablation.** Arms without the class-balanced draw (uniform over
windows), without event superposition, without the non-stationary and
artefact groups, and with white Gaussian noise in place of the corpus.
Each arm is scored on the false-pick rate by class and on the aftershock
regime, which is where they should differ.

**E4, PhaseNet width and context.** PhaseNetWC (the parent's 2× filters)
against standard width at the same data, and 60 s input against 30 s.
Longer context is what regional S and overlapping events ask for; the
architecture is fully convolutional so the change is in the window
length, not the weights. Ensembles of the best arm with `jma_wc` are the
deployment option if they win at matched budget; two PhaseNets still cost
less than one EQTransformer.

Compute: v7 reached epoch 44 on 527k windows in one server session; the
per-epoch time is in `results/finetune_jma_wc_global_v7_metrics.csv` on
the server. A 4 million window run is about eight times v7 per epoch;
E1's subsets are planned to fit the GPUs available; E2 to E4 are ten to
twelve runs at one size.

## 7. Phase 3, a distant-P model for offshore and sparse regions

A 30 s window at 100 Hz is the wrong instrument for P at 10 to 30°. Train
a second PhaseNet at 20 Hz on 120 s windows, P only, from GEOFON, MLAAPDE,
CREW and ISC-labelled P at 3 to 30° from M ≥ 4, initialised from the
SeisBench `geofon` weights, with the §5 noise recipe restricted to the
microseism, polar and quiet classes. It runs only where it earns its
cost: oceanic islands, coasts facing offshore seismicity, and regions
where the nearest station is more than 300 km away. Its picks enter the
associator with a global velocity model. Scored against ISC and NEIC by
completeness magnitude versus nearest station distance, on offshore and
African test regions held out by year.

## 8. Phase 4, deployment

Thresholds belong to the weight and the region, set to a false-pick target
and recorded with the campaign. Association settings for sequences are
tuned on the aftershock and swarm metrics, not on picks. Every change of
weights re-runs the two QuakeScope notebooks and the event-level scorer
on the acceptance suite; until a candidate beats `jma_wc` at matched
budget on the non-US sequences and does not lose on the volcano and swarm
cases, the campaign stays on `jma_wc` or the Phase 0 winner.

## 9. Gates and timeline

| When | Deliverable | Decision |
|---|---|---|
| Week 3 | Task 1 counts with the 24 windows; matched-budget ranking of existing weights and PhaseNet ensembles; per-region thresholds; event scorer; baselines on West Bohemia and Campi Flegrei | Campaign picker for this quarter; affordability of the place hold-outs |
| Week 4 | Signal and noise harvest census per operator and per noise class | Corpus sizes and operator list |
| Week 11 | Signal corpus and noise corpus built, checked, fingerprinted; suites frozen | Go to E1 |
| Week 15 | E1 scaling curve, per regime | Continue, or stop the finetune line |
| Week 22 | E2 to E4; one candidate per line | Acceptance run, once |
| Week 24 | Distant-P model, first version | Offshore and sparse-region deployment |

## 10. What this plan does not do

No EQTransformer, no multi-station models. No ocean-bottom data or noise
this round. No more single-variable changes on the v7 corpus. No teleseismic rebalancing inside the regional
model. No white noise. No selection on `notebooks/step3_metrics.csv`. No
claim of a better picker before the acceptance run.

## References

- Zhu, W., & Beroza, G. C. (2019). PhaseNet. *GJI* 216, 261–273.
- Münchmeyer, J., et al. (2022). Which picker fits my data? *JGR Solid Earth* 127, e2021JB023499.
- Woollam, J., et al. (2022). SeisBench. *SRL* 93, 1695–1709.
- Naoi, M., et al. (2024). PhaseNet models trained on the JMA unified catalogue. *EPS* 76, doi:10.1186/s40623-024-02091-8.
- Ni, Y., et al. (2023). Curated Pacific Northwest AI-ready seismic dataset. *Seismica* 2(1).
- Zhong, Y., & Tan, Y. J. (2024). Deep-learning-based phase picking for volcano-tectonic and long-period earthquakes. *GRL* 51, e2024GL108438.
- Aguilar Suarez, A. L., & Beroza, G. C. (2024). CREW dataset. *Seismica* 3(1).
- Münchmeyer, J. (2024). PyOcto. *Seismica* 3(1).
