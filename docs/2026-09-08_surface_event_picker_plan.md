# Plan: a dedicated onset picker for surface events

*2026-09-08, Marine Denolle with Claude, branch `audit/2026-09-07-generalization`.
Companion to `docs/2026-09-07_training_plan.md` (earthquake picker) and
`~/GitHub/QuakeScope/docs/quakexnet_generalization_plan.md` (classifier).
Nothing here has been run. Counts marked "to verify" need the server or the
source paper.*

## 1. What the picker is for

A model that reads continuous seismograms and marks the onset of a
surface event: a landslide of any size, a debris flow or lahar front, a
snow, rock or debris avalanche, a rock or ice fall. It runs inside
QuakeScope on every station-day the campaign touches, so it has to be
cheap, and it has to work where it was never trained: Nepal, Greenland,
Alaska, the Cascades, the Andes and Central America, the Alps, Taiwan. The
onset feeds detection and a first location; amplitude and duration for
size are QuakeScope feature extraction and are out of scope here except
where a design choice now keeps them cheap later (§3.4).

The bar is set by the earthquake retrain. Twenty finetunes of `jma_wc`
never beat their parent, and the audit traced that to the corpus, the
loss shape, the absence of realistic augmentation, and a selection
metric that could not see false positives
(`docs/2026-09-07_training_history_audit.md`). Every one of those
findings is a rule here (§2), and the plan is written so the result can be
scored against a protocol fixed before training, not against a
leaderboard read twenty times.

Two properties of the signal drive the design. Surface events are
emergent and long: 10 to 100 s at local distance, minutes for the
largest, with a spindle envelope, energy at 1 to 10 Hz locally and
below 1 Hz for large events at regional distance, and no separable S
because the source lasts longer than the S minus P time. The onset is
therefore uncertain at the level of a second, not a hundredth, and the
window has to hold the whole envelope for the model to tell an onset from
a coda. The earthquake PhaseNet, 30 s at 100 Hz with 0.1 s labels, is the
wrong instrument on both counts.

## 2. Rules carried over from the retrain audit

| Finding in the audit | Rule for this picker |
|---|---|
| Positives-only benchmark with an oracle ±5 s window ranked twenty versions and never saw a false pick | Every selection metric is computed on continuous data with negatives; the primary metric is recall at a matched false-pick budget per station-day |
| Selection on the test set, twenty times | Three splits: train, validation (early stopping only), development (selection). An acceptance suite is frozen now and read once |
| Hard-argmax cross-entropy discarded the Gaussian label | Soft targets, cross-entropy against the label distribution |
| White Gaussian noise for augmentation cost 0.2 to 0.6 s of timing | Only real noise, superposed at controlled SNR, drawn from the stations and regions of deployment |
| Fixed onset position in every training window | Onset uniformly anywhere in the window, including windows that hold only the tail |
| Everything resampled to 100 Hz once; the model never saw 40 and 50 Hz instruments | Native rates 40, 50, 100 Hz all decimated to the model rate at train and inference time with the same SeisBench resampler |
| One labelled pick per window; other events in the window unlabelled | Every catalogued onset in the window labelled; earthquakes in the window labelled as what they are (§3.3) |
| Explicit timing, presence, focal and class-weight terms each collapsed recall or timing | None of them. One loss |
| No SNR curation; 71 % of misses below 5 dB | SNR stored per window at build time, stratified sampling, recall reported by SNR bin |
| Test sequences sat inside the training corpus | Hold-out by place and by time, verified by a spatiotemporal join before the manifest is written, and a test region list that the builder refuses to run without |
| Distillation at T = 4 lowered every probability | No teacher. Calibration measured on the development suite, thresholds set per region to a false-pick target |
| Half a million heterogeneous windows could not re-teach a six-million-window parent | There is no parent here; the corpus is small and the model is small. Data scaling is measured, not assumed (§7, E1) |

## 3. Architecture

### 3.1 Backbone

SeisBench `PhaseNet` (Zhu and Beroza 2019) with `filter_factor=1`, about
270 k parameters, or `VariableLengthPhaseNet` when the window or the
output activation departs from the defaults. Both are already in the
SeisBench install QuakeScope ships, so a trained weight is a `.pt` and
`.json` pair dropped into `sb_catalog/models/v3/<class>/` and selected
with `--weight` (`QuakeScope/docs/rerun_2026/02_weights_and_container.md`).
No new architecture is proposed. The U-Net is the right shape for a
per-sample onset probability, and the retrain showed that the failure
modes were data and protocol, not capacity.

### 3.2 Sampling rate and window

**25 Hz, 3001 samples, 120 s.** The constructor takes `sampling_rate`
directly, and `annotate` resamples the stream to it, so nothing changes in
the pipeline. At 25 Hz the model keeps 0.5 to 12 Hz, which covers the
local band of every target class, and the 120 s window holds a 100 s
event with room for the onset to sit anywhere. The receptive field of the
five-level U-Net at stride 4 spans the window. Per station-day this is a
quarter of PhaseNet's samples, so the picker costs about a quarter of an
earthquake picking pass, before the amplitude stages.

Arms to test (E3): 20 Hz over 150 s for the longest events, and 50 Hz over
3001 samples (60 s) as the control that shows what the lower rate costs on
short rock falls.

**Pre-filter.** A 0.5 Hz high-pass in the model's `filter_args`, applied
by SeisBench before normalisation, exactly as the `obs` weight does. The
per-window standard deviation normalisation otherwise puts a coastal or
ocean-bottom station's microseism in the denominator and a 1 to 10 Hz
landslide at a Cascade station becomes invisible after scaling. The
long-period detection of very large landslides at 20 to 150 s (Ekström
and Stark 2013) is a different instrument and a different model; out of
scope.

### 3.3 Output channels

Baseline, as proposed: **two channels, onset and not-onset**, softmax.
SeisBench names phases by single letters, so the onset class is `U`
(sUrface) and the model declares `phases="UN"`; QuakeScope's picker reads
the threshold as `U_threshold` once `sb_catalog/src/picker.py:370-371`
stops hard-coding `P` and `S` (one small change).

First ablation (E2): **three channels, `U`, `P`, `N`**, where `P` is an
earthquake P onset. The earthquake corpus (PNW ComCat, 184 k traces with
analyst picks) is two orders of magnitude larger than any surface-event
set and is the population the picker must not fire on. Giving earthquake
energy its own class instead of forcing it into "noise" is a cheap way to
sharpen the boundary that matters most in production, where the surface
picker sees the same aftershock sequences the earthquake picker does. The
`P` channel is discarded at inference. The hypothesis to score is fewer
false `U` picks on earthquake days at equal recall.

Second ablation (E2): a **sigmoid event-mask channel** in
`VariableLengthPhaseNet` (`output_activation="sigmoid"`), a boxcar from
onset to the analyst's end time where one exists. It is the hook for
duration in QuakeScope and it gives the encoder an envelope-shaped target
that the onset Gaussian alone does not. It is an arm, not the baseline,
because end times are labelled for only a subset of events (to verify,
§4.1) and a partially labelled channel needs masking in the loss.

### 3.4 Components

Three-component input with heavy channel dropout: half the training
draws keep only Z, so the same weight runs on the vertical-only
short-period stations that sit on the volcanoes (three of the 35 vertical
channels within 20 km of Rainier are `EHZ`, 27 are 50 Hz `BHZ`, 18 are
100 Hz `HHZ`; EarthScope station service, queried 2026-09-08). Surface
events carry little polarisation information at local distance, so a
Z-only model (`in_channels=1`) is the lighter alternative and is arm E4.
If E4 matches the three-component model at matched budget, it ships: one
weight, one channel, every station.

## 4. Data

### 4.1 Positives

**PNSN surface events.** Two forms of the same labels exist. The SeisBench
`pnw_exotic` file (Ni et al. 2023) stores 180 s three-component traces at
100 Hz with `source_type`, `trace_P_arrival_sample` (the analyst onset,
placed at 70 s by construction), `trace_snr_db` per component and station
metadata; the classification work drew 8,434 surface-event traces from it
(`PNW_Seismic_Event_Classification/deep_learning/testing_deep_learning_architectures.ipynb`;
the paper's count is to be verified). Behind it sits the raw PNSN pick
export, `~/GitHub/surface_events/data/events/su_picks.txt`: 21,721 `su`
picks on 12,129 events since 1981, with station, channel, and a pick
quality, and `PNSN_Pick_Label.csv` with 13,465 `su` picks on 7,523 events
carrying an `Impulsivity` flag. The raw export is the better source for
this picker, for three reasons: it has every pick of an event and not one
trace per station, so multi-onset windows can be cut from continuous data;
it reaches stations and years the curated file does not; and the
impulsivity flag is a per-pick handle on label width (§5.1). The picks are
dominated by St. Helens (SEP, EDM, HSR) and Rainier (RCS, RCM, STAR),
with Hood, Baker and Adams far behind; the located catalogue of the
directivity work holds 1,850 St. Helens, 1,252 Rainier and 65 Hood
events for 2001 to 2021 (`surface_events/events/Event_Data_*.csv`).
These are the only analyst-picked onsets of the target class at volume
anywhere, and they come from stations with different noise (glacier,
river, wind, snow). They are the training set. The other classes in the
curated file, thunder and sonic booms, are negatives with an onset (§4.2).

**End times.** Wes Thelen's video- and observer-verified catalogue
(`surface_events/events/Wes_Cat_{rainier,st_helens,hood}.csv`: 133, 59
and 18 events) is the only hand-picked start and end time per event in
the lab's holdings. It anchors the event-mask arm (§3.3) and calibrates
the two automatic duration estimators already written, the cumulative
fourth-power envelope of `surface_events/src/ML_Picker_all.ipynb` and the
5 to 95 % cumulative energy of
`Surface_Event_Detection/Common_Scripts/seis_feature.py`, which can then
supply weak end labels for the rest.

**Confirmed events from the exotic-event catalogues**, picked by hand for
this project. The Exotic Seismic Event Catalog (ESEC; Collins, Bahavar,
Allstadt) holds 245 events verified by non-seismic means, global, with
start and end time, volume, mass, runout and location uncertainty per event
and pointers to the waveforms at the DMC; a copy is at
`PNW_Seismic_Event_Classification/data/IRISExoticEventCatalog.txt`.
It has event times, not per-station onsets: station picks for the
acceptance suite are made by hand (§6.3). It is the only ground truth outside the Pacific Northwest
and it is small, so it is **never trained on**; it is the acceptance suite
(§6). The Alaska Earthquake Center's 36 catalogued landslides since 2005
(ComCat, `eventtype=landslide`, queried 2026-09-08; 57 worldwide, of which
23 in 2024 alone) are out-of-region positives for the development suite.

**Sources to request**, from the survey in the QuakeXNet plan §4a, which
apply unchanged: Illgraben debris flows (WSL and SED; the only labelled
debris-flow set found), Ruapehu lahars (GNS, ERLAWS), Piton de la Fournaise
rockfalls (OVPF, about 7,000 labelled events across seven classes),
Stromboli Sciara del Fuoco landslides (INGV-OV), Swiss Alpine landslides
and rockslides (SED, 39 a year by 2024). Whatever arrives with onset times
goes to training only if it is a place with more than one event, so that
the place can be split; a single event from a place is a test case.

**Model-derived detections** (Akash's 114,775 Rainier surface events over
15 years) are not labels. They are a pool for mining hard negatives and
for a later pseudo-label round after manual review of a sample.

**Adjacent classes.** Alaska ice quakes (16,242 in ComCat since 2005) are
neither positives nor negatives: calving and crevassing are impulsive,
mass movements are emergent, and the PNSN `su` label itself includes
glacier-sourced events on Rainier. They are excluded from training and
scored as a diagnostic set: what the picker calls them is reported, not
optimised.

### 4.2 Negatives

The picker fails in production by firing on things that are not surface
events, so the negative set is the larger half of the corpus and is built
from the populations that will be in the continuous data.

1. **Earthquakes with analyst picks** from `pnw` (ComCat), at every
   distance and magnitude, including regional and teleseismic arrivals
   whose long codas look like a spindle at 25 Hz. Labelled `N`, or `P` in
   the three-channel arm.
2. **Other exotic sources** in `pnw_exotic`: thunder, sonic booms,
   explosions, plane crashes. Thunder and sonic booms are the confusers
   with a long, emergent, low-frequency character.
3. **Volcanic tremor and long-period events** where labelled (VCSEIS holds
   Alaska, Hawaii, northern California and the Cascades; the label
   taxonomy is to be checked). Tremor is the closest natural analogue of a
   debris flow's sustained signal.
4. **Noise from continuous data at the training stations**, not from a
   global pool: storms, wind, high-flow river noise, glacier background,
   traffic and helicopters near Rainier, drawn at random times and
   verified against the PNSN catalogue to hold no event within the
   window. The existing `data/noise_global` pool (STEAD, LenDB, TXED,
   VCSEIS, OBST2024 noise) is a second source for superposition only.
5. **Aftershock hours.** Continuous days from the held-in earthquake
   sequences of the earthquake plan (not the acceptance ones), where
   arrivals are seconds apart and the coda never ends. No surface event
   labels; the whole day is negative. This is the regime that will produce
   the most false picks in a global run.

### 4.3 Splits and hold-outs

**By place, not by trace.** A volcano's surface events repeat at the same
stations for years, so a random split tests memory of a station. The
counts force the shape of the split: St. Helens and Rainier hold nearly
all the picks, Hood has 65 located events, Baker, Adams and Glacier Peak
fewer. Training is St. Helens and Rainier. The development suite is every
other Cascade volcano as a place (Hood, Baker, Adams, Glacier Peak,
Newberry, Crater Lake, with the 239 stations within 50 km listed in
`surface_events/data/station/Volcano_Metadata_50km.csv`) plus the
held-out years below at the two training volcanoes. A cross-volcano swap
(E8: train on St. Helens only, score on Rainier, and the reverse) measures
transfer between the two well-instrumented places directly and is the
cheapest generalisation test available before any request is answered.

**By time as well.** Hibert's Piton de la Fournaise classifier collapsed on
the same volcano five years later because the rockfall mechanism changed
(QuakeXNet plan §4b). Two whole years are held out of training everywhere,
chosen after the census so that each holds at least fifty events, and
recall by year is reported on the development suite so a drift of this
kind is seen rather than averaged away.

**Verification.** The same spatiotemporal join as
`scripts/heldout_sequences.py`, extended with place windows around each
volcano and each test region, run over every source before the manifest
is written; the builder refuses to run without the exclusion list, as it
now does for earthquakes. The noise corpus is filtered with the same
windows: a quiet day at a held-out station teaches that station's
character.

## 5. Labels, windows and augmentation

### 5.1 Onset label

A Gaussian at the analyst onset with σ = 1.0 s (25 samples at 25 Hz),
against 0.1 s for earthquakes. The width is the analyst's own uncertainty
on an emergent onset; it is to be measured, not assumed, from repeat picks
where the same event was picked by two analysts or at two stations of the
same distance (to verify on `pnw_exotic`). Arms at σ = 0.5 and 2.0 s
(E5). The `N` channel is one minus the maximum of the others, as in
`scripts/manifest_dataset.py::make_labels`.

For the event-mask arm: a boxcar from onset to end time, ramped over 2 s
at each edge, with the loss masked to zero on traces that carry no end
time.

### 5.2 Windows

Cut from the full stored trace (or from continuous data where the source
is a bulletin) at native rate, decimated to 25 Hz with SeisBench's
resampler, and cropped to 3001 samples per draw with the onset uniformly
between sample 0 and 3000. Windows whose onset falls before the crop hold
the event's body or tail with no onset label; these are kept at a fixed
fraction (20 %) because a coda without an onset is exactly what the model
must learn not to pick. Each window carries: SNR (10 s after onset over
10 s before, on Z, in dB), station, channel code, native rate, place,
year, event type, number of labelled onsets, and the operator's duration
where present.

### 5.3 Augmentation, all on the fly

Every item below is a lesson from the audit or a property of the target
signal; nothing is included because it is customary.

- **Real-noise superposition**, probability 0.5, target SNR uniform in 0 to
  20 dB, noise drawn first from the same station and season when available
  and otherwise from the training-station noise corpus. Labels unchanged.
- **Cross-class superposition**, probability 0.2: an earthquake window
  added to a surface-event window, or the reverse, with both labelled; a
  second surface event in the same window, both labelled. This is the
  multi-event rule of the earthquake plan, applied to the class the
  picker must reject.
- **Time stretch**, factor uniform in 0.8 to 1.25 by resampling, label
  moved with the onset. Duration varies by an order of magnitude across
  the target class and the training set cannot cover it by itself. Not
  applied to earthquake negatives (it would teach that a stretched P is a
  surface event).
- **Instrument response**: probability 0.3, the window is passed through a
  1 Hz short-period response or a 0.5 Hz high-pass before normalisation,
  so a broadband training trace looks like an `EHZ` record.
- **Channel dropout**: probability 0.5 keep Z only, 0.1 drop one
  horizontal. Gaps: probability 0.1, zero a 1 to 10 s segment. Clipping:
  probability 0.05, clip at 3 to 6 standard deviations, a common state of
  a short-period station during a large event nearby.
- Amplitude scaling 0.5 to 2 as before. Polarity flip is irrelevant to an
  envelope and is dropped.

No white noise. No fixed onset position. No teleseismic rebalancing.

## 6. Training, selection and the acceptance test

### 6.1 Optimisation

From scratch, AdamW, LR 1e-3 with a five-epoch warm-up and cosine decay to
1e-5, batch 256, weight decay 1e-4, soft-label cross-entropy, early stopping
on validation loss with patience 15. The corpus is small (tens of thousands
of positive windows before augmentation, to verify), so an epoch is minutes
and the whole arm table (§7) fits in a few GPU-days. Every arm is run with
three seeds and reported as mean and range, because at this corpus size the
seed variance can be as large as an arm effect.

Initialisation arm (E1b): the encoder weights of SeisBench's `original`
PhaseNet, which are defined in samples and therefore respond to four times
lower frequency at 25 Hz. Whether that is a useful prior or a burden is an
experiment; the retrain's experience with a strong parent argues for
measuring it rather than assuming it.

### 6.2 Selection on the development suite

Pre-registered, in this order, all on continuous data through SeisBench
`annotate` and `classify` exactly as QuakeScope runs them, with bootstrap
intervals from `scripts/metrics.py::bootstrap_ci`:

1. **Recall at matched false-pick budget.** The budget is the number of
   picks per station-day on quiet days (no catalogued event of any type
   within the window), fixed at one false pick per station-day for the
   headline number and swept from 0.1 to 10 for the curve. Recall is the
   fraction of analyst onsets on the held-out volcano with a pick within
   ±5 s, reported per SNR bin, per event type, per year, per channel code.
2. **Onset residual** on detected events: median and interquartile range,
   in seconds, against the analyst pick, per distance bin.
3. **False-pick rate on earthquake days**, from the held-in aftershock
   hours of §4.2 item 5 and from the held-out volcano's own earthquake
   days: picks per station-day and per catalogued earthquake.
4. **Event-level detection**: onsets on three or more stations within a
   plausible move-out window (30 s for a 50 km aperture) count as one
   detection; recall against the catalogue's events and false events per
   day.

Baselines scored on the same suite before any training, so a gain is a
gain over something: a long-window STA/LTA tuned on the training set
(Allstadt's parameters as a starting point), an envelope-duration
detector, the current QuakeScope path (PhaseNet picks passed to
QuakeXNet, `su` probability above 0.5), and the earthquake pickers themselves run as if their P channel were a
surface-event detector. The lab has one such number already: the ELEP
ensemble of six pretrained EQTransformers, applied to 10,997 PNSN
surface-event picks, lands within a median of 0.08 s of the analyst but
with a standard deviation of 5.65 s (`surface_events/data/bb_elep_picks_su.csv`);
that spread is what a dedicated picker has to beat, and `jma_wc` run the
same way measures how often the campaign's picker already fires on these
events.

Thresholds are set on the development suite per station class (broadband
against short-period) to the false-pick target, recorded with the weight,
and not touched afterwards.

### 6.3 Acceptance suite, read once

Three regimes the development suite does not test, all outside the
training footprint:

1. **Out of region.** ESEC events with waveforms at the DMC: [[TBD: list
   with dates from the catalogue; candidates are Langtang 2015 (Nepal),
   Karrat Fjord 2017 and Dickson Fjord 2023 (Greenland), Taan Fiord 2015
   and Lamplugh 2016 (Alaska), Piz Cengalo 2017 and Brienz 2023 (Alps),
   Hsiaolin 2009 and typhoon-triggered slides (Taiwan), Elliot Creek 2020
   (British Columbia), Volcán de Fuego 2018 lahars (Guatemala), Kaikōura
   2016 coseismic slides (New Zealand).]] Scored as detection at the
   station level (onset within ±10 s of a hand pick made for this suite by
   someone who has not seen the model output) and at the event level.
2. **Different noise.** Quiet station-days at stations the model has never
   seen, chosen for their noise: coastal and island stations where the
   microseism dominates (Alaska coast, Iceland, Greenland), an Alpine
   valley with a torrent (Illgraben's own stations if WSL agrees), an
   ocean-bottom deployment from `obst2024`, and a tropical volcano with
   tremor (Hawaii, `HV`). Scored as false picks per station-day at the
   thresholds fixed in §6.2. This is the test the earthquake retrain never
   had.
3. **Earthquake sequences.** Two days of Ridgecrest 2019 and of Kaikōura
   2016 continuous data (both already held out of everything), scored as
   false surface picks per station-day. A picker that calls aftershock
   codas landslides cannot run in a global campaign, whatever its recall.

Gate: the candidate ships to QuakeScope if it beats every baseline at
matched budget on the development suite, holds a false-pick rate below
one per station-day on the different-noise suite, and detects at least
[[TBD: fraction set after the census of ESEC waveform availability]] of
the out-of-region events at the event level.

## 7. Experiments and gates

**Phase 0, no training (two to three weeks).** Census of `pnw_exotic`:
events, traces, stations, years, onset and end-time completeness, by
volcano and event type; choice of the held-out volcanoes and years from
the counts. Label audit: onset residual between stations of the same
event against distance (a pick that moves faster than 6 km/s is wrong),
inter-analyst width where available, a confident-learning pass as in
`docs/LABEL_ERROR_FILTERING.md` adapted to a two-class target. ESEC
harvest: which of the 245 events have waveforms within 100 km, at what
rate; freeze the acceptance list and commit it as a window list beside
`heldout_sequences.py`. Baselines run on the development suite. Requests
sent to WSL, GNS, OVPF, INGV-OV, SED.

**Phase 1, corpus (four to six weeks).** Builder for surface-event
manifests with the window fields of §5.2; noise corpus from continuous
data at the training stations with the catalogue check; exclusion list
and year hold-out verified on every manifest; fingerprints committed.

**Phase 2, arms (six to eight weeks).** Each arm three seeds, selected on
§6.2.

| Arm | Question |
|---|---|
| E0 | Baseline: 25 Hz, 120 s, two channels, 3C with dropout, σ = 1 s, full augmentation |
| E1a | Data scaling: nested 25, 50, 100 % of the positive set at fixed negatives. Is the model data-limited? |
| E1b | Initialisation from `original` PhaseNet weights against scratch |
| E2 | Outputs: `UN` against `UPN` against `UN` plus sigmoid event mask |
| E3 | Rate and window: 20 Hz / 150 s, 50 Hz / 60 s |
| E4 | Z-only model |
| E5 | Label width σ = 0.5, 1, 2 s |
| E6 | Augmentation ablation, one arm each without: real noise, time stretch, cross-class superposition, instrument response |
| E7 | Negative composition: without aftershock hours; without exotic non-`su` classes |
| E8 | Cross-volcano swap: train St. Helens, score Rainier, and the reverse |

E1a runs first. If recall at matched budget is flat from 50 to 100 % of
the positives, the model is not data-limited and the request campaign of
§4.1 is about test coverage, not training; if it is still rising, every
onset that can be obtained goes to training and the plan's centre of
gravity moves to §4.1.

**Phase 3, acceptance and deployment (two weeks).** One candidate; §6.3
once; conversion to a SeisBench pair; the `U_threshold` change in the
QuakeScope picker; run on `EH`, `BH` and `HH` channel groups (the
classifier today runs on `BH` and `HH` only, `picker.py:598`); thresholds
per station class recorded in `sb_runs`; the two QuakeScope notebooks
re-run with the surface picks as a third phase to confirm nothing in the
earthquake catalogue changes.

| When | Deliverable | Decision |
|---|---|---|
| Week 3 | Census, label audit, acceptance list frozen, baselines scored | Held-out volcanoes and years; which requests are worth waiting for |
| Week 9 | Corpus, noise corpus, manifests verified | Go to E0 and E1a |
| Week 12 | E1a scaling curve | Training-limited or data-limited; scope of §4.1 |
| Week 17 | E2 to E7, one candidate | Acceptance run |
| Week 19 | Acceptance result, weight converted | Ships, or the plan returns to §4.1 with the acceptance suite still unread |

## 8. What already exists and what is new

Surveyed 2026-09-08 in `~/GitHub/surface_events`,
`PNW_Seismic_Event_Classification`, `Surface_Event_Detection` and
`PNW_ML_Classification`. None of them trains a picker on surface events;
onsets there come from the ELEP ensemble of pretrained EQTransformers
(`surface_events/src/mbf_elep_func.py`), and the only trigger in the
classifier runs on QuakeXNet's own noise probability. What carries over:

- Labels and catalogues: the PNSN pick export and the labelled pick file
  (§4.1), the Thelen start-and-end catalogue, the located catalogue of
  3,188 events with an envelope duration (median 17 s, a few negative
  values, so the estimator has failure modes), the ESEC copy, and the
  station lists within 50 km of each volcano.
- Data access: the classification repo reads the PNW HDF5 files directly
  from `/data/whd01/yiyu_data/PNWML` on the server rather than through
  SeisBench; the surface-event manifest builder can do either, and the
  SeisBench route is preferred so that the exclusion machinery of this
  repository applies unchanged.
- Windowing conventions to reuse: the classification work cut 100 s at
  50 Hz with the onset 5 to 20 s after the window start; the picker work
  used 150 s at 40 Hz. Both are evidence that 120 s at 25 Hz loses nothing
  the analysts used.
- Duration estimators (two, §4.1), envelope features, and the
  cross-station envelope correlation `pick_time()` of
  `surface_events/src/utils.py`, which can propagate a good pick to
  stations of the same event when the analyst picked only one.
- Deployment: QuakeXNet is wrapped as a SeisBench `WaveformModel` and runs
  in QuakeScope's classifier on `BH` and `HH` streams at 50 Hz; the same
  wrapper pattern serves the picker, which is a `PhaseNet` and needs none.

Reusable from this repository without change: `scripts/metrics.py`
(bootstrap intervals), `scripts/heldout_sequences.py` (windows, year
hold-out, manifest check; needs place windows for the volcanoes and the
ESEC sites), the manifest and cached-dataset pattern of
`scripts/manifest_dataset.py` and `scripts/fast_manifest_dataset.py`
(with the random crop moved into `__getitem__`, as the v21 proposal
already specifies), and `scripts/build_noise_dataset.py` as the template
for a station-specific noise corpus. New: a surface-event manifest
builder, the label and augmentation module of §5, a continuous-data
scorer that wraps `annotate` and `classify` and computes §6.2, and a
config family `configs/su_picker_v*.yaml` with the same header
convention as the earthquake configs.

## 9. What this plan does not do

No amplitude, duration or size estimation beyond the optional event-mask
channel; that is QuakeScope feature extraction. No long-period detection
of giant landslides at 20 to 150 s. No source classification beyond
onset against not-onset; QuakeXNet keeps that job and can run on the
surface picks as it runs on earthquake picks. No training on ESEC. No
selection on any number from the acceptance suite. No claim of a working
picker before the acceptance run.

## References

- Zhu, W., and Beroza, G. C. (2019). PhaseNet. *GJI* 216, 261–273.
- Ni, Y., et al. (2023). Curated Pacific Northwest AI-ready Seismic Dataset. *Seismica* 2(1). doi:10.26443/seismica.v2i1.368
- Kharita, A., Denolle, M., Hutko, A., Hartog, R., and Malone, S. (2026). Exploration of Machine Learning Methods to Seismic Event Discrimination in the Pacific Northwest. *Seismica*.
- Münchmeyer, J., et al. (2022). Which picker fits my data? *JGR Solid Earth* 127, e2021JB023499.
- Woollam, J., et al. (2022). SeisBench. *SRL* 93, 1695–1709.
- Ekström, G., and Stark, C. P. (2013). Simple scaling of catastrophic landslide dynamics. *Science* 339, 1416–1419.
- Allstadt, K. E., et al. (2018). Seismic and acoustic signatures of surficial mass movements at volcanoes. *JVGR* 364, 76–106.
- Hibert, C., et al. (2017). Automatic identification of rockfalls and volcano-tectonic earthquakes at Piton de la Fournaise using a Random Forest algorithm. *JVGR* 340, 130–142.
- Chmiel, M., et al. (2021). Near-real-time automated classification of seismic signals of slope failures with continuous random forests. *NHESS* 21, 339–361.
- Collins, E. A., Allstadt, K. E., Groult, C., Hibert, C., Malet, J.-P., Toney, L., and Bessette-Kirton, E. (2022). Seismogenic landslides and other mass movements. Exotic Seismic Event Catalog (ESEC), EarthScope SPUD and USGS ScienceBase.
- Zhong, Y., and Tan, Y. J. (2024). Deep-learning-based phase picking for volcano seismicity (VCSEIS / volpick). *GRL* 51.
