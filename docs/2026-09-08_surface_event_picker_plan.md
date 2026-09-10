# Plan: a dedicated onset picker for surface events

*2026-09-08, Marine Denolle with Claude, branch
`audit/2026-09-07-generalization`. Companion to
`docs/2026-09-07_training_plan.md` (earthquake picker) and
`~/GitHub/QuakeScope/docs/quakexnet_generalization_plan.md` (classifier).
Nothing here has been run. Counts from the public PNW exotic metadata were
computed on 2026-09-08; everything else marked as a census item needs the
server.*

## 1. What the picker is for

A model, working name SUNet, that reads continuous seismograms and marks the
onset of a surface event: a landslide of any size, a debris flow or lahar
front, a snow, rock or debris avalanche, a rock or ice fall. It runs inside
QuakeScope on every station-day the campaign touches, so it has to be cheap,
and it has to work where it was never trained: Nepal, Greenland, Alaska, the
Cascades, the Andes and Central America, the Alps, Taiwan. The onset feeds
detection and a first location; amplitude and duration for size are
QuakeScope feature extraction and are out of scope here except where a
design choice now keeps them cheap later (§3.3).

The bar is set by the earthquake retrain. Twenty finetunes of `jma_wc` never
beat their parent, and the audit traced that to the corpus, the loss shape,
the absence of realistic augmentation, and a selection metric that could not
see false positives (`docs/2026-09-07_training_history_audit.md`). Every one
of those findings is a rule here (§2), and the plan is written so the result
can be scored against a protocol fixed before training, not against a
leaderboard read twenty times.

Two properties of the signal drive the design. Surface events are emergent
and long: 10 to 100 s at local distance, minutes for the largest, with a
spindle envelope, energy at 1 to 10 Hz locally and below 1 Hz for large
events at regional distance, and no separable S because the source lasts
longer than the S minus P time. The onset is therefore uncertain at the
level of a second, not a hundredth, and the window has to hold the whole
envelope for the model to tell an onset from a coda. The earthquake
PhaseNet, 30 s at 100 Hz with 0.1 s labels, is the wrong instrument on both
counts.

## 2. Rules carried over from the retrain audit

| Finding in the audit | Rule for this picker |
|---|---|
| Positives-only benchmark with an oracle ±5 s window ranked twenty versions and never saw a false pick | Every selection metric is computed on continuous data with negatives; the primary metric is recall at a matched false-pick budget per station-day |
| Selection on the test set, twenty times | Three splits: train, validation (early stopping only), development (selection). An acceptance suite is frozen now and read once |
| Hard-argmax cross-entropy discarded the Gaussian label | Soft targets, cross-entropy against the label distribution |
| White Gaussian noise for augmentation cost 0.2 to 0.6 s of timing | Only real noise, superposed at controlled SNR, drawn from the stations and regions of deployment |
| Fixed onset position in every training window | Onset uniformly anywhere in the window, including windows that hold only the tail |
| Everything resampled to 100 Hz once; the model never saw 40 and 50 Hz instruments | Native rates 40, 50, 100 Hz all brought to the model rate at train and inference time with the same SeisBench resampler (a four-corner zero-phase Butterworth at the new Nyquist), never with obspy's default decimation filter, whose passband ends at 0.65 of Nyquist |
| One labelled pick per window; other events in the window unlabelled | Every catalogued onset in the window labelled; earthquakes in the window labelled as what they are (§3.3) |
| Explicit timing, presence, focal and class-weight terms each collapsed recall or timing | None of them. One loss |
| No SNR curation; 71 % of misses below 5 dB | SNR stored per window at build time, stratified sampling, recall reported by SNR bin |
| Test sequences sat inside the training corpus | Hold-out by place and by time, verified by a spatiotemporal join before the manifest is written, and a test region list that the builder refuses to run without |
| Distillation at T = 4 lowered every probability | No teacher. Calibration measured on the development suite, thresholds set per region to a false-pick target |
| Half a million heterogeneous windows could not re-teach a six-million-window parent | There is no parent here; the corpus is small and the model is small. Data scaling is measured, not assumed (§7, E1) |

### 2b. Practices taken from the picker literature

| Practice | Source | Where it lands here |
|---|---|---|
| Two thirds of training windows hold a pick, one third are drawn at random, so labels are not swamped by noise | Münchmeyer et al. 2022 | §5.2 window fractions |
| Diverse picks transfer best; a picker trained on one convention generalises worst | Münchmeyer et al. 2022 | §4.1 request campaign, §7 E1a |
| A 100 Hz picker fed 20 or 40 Hz data sees lower frequencies and detects low-frequency sources better; train at the low rate directly | Münchmeyer et al. 2022 | §3.2, the rate choice and the low-rate arms |
| PhaseNet's effective receptive field is about 4 s at 100 Hz despite the 30 s input | Münchmeyer et al. 2022 | §3.2, the rate sets the context the model actually uses |
| A PhaseNet at 20 Hz on 60 s windows, band-limited to 1 to 8 Hz, trained on templates superposed on real noise at controlled SNR, transfers across Cascadia, Guerrero and Nankai | Münchmeyer et al. 2024 (LFE picker) | §3.2, §5.3; the closest published precedent for this design |
| Wide labels for uncertain onsets: σ = 1.5 s in PhaseNet-DAS, 15 times the original | Zhu et al. 2023 | §5.1 |
| Waveform masks for earthquake and rockfall signals as segmentation targets, single-station model plus association model | Liao et al. 2023 (RockNet) | §3.3 event-mask arm, §6.2 item 4 |
| SeisBench generator blocks: `RandomWindow`, `ProbabilisticLabeller`, `RealNoise`, `AddGap`, `ChannelDropout`, `Filter`, `OneOf` | Woollam et al. 2022 | §5, implemented with these rather than by hand |
| Event-level train/test separation and a stated label-error rate (0.2 to 8 % per class) | Kharita et al. 2026 | §4.3, §7 Phase 0 label audit |
| A classifier trained on one volcano's rockfalls collapsed on the same volcano five years later | Hibert et al. 2017 | §4.3 year hold-out, recall by year |

## 3. Architecture

### 3.1 Backbone

SeisBench `PhaseNet` (Zhu and Beroza 2019) with `filter_factor=1`, 268 k
parameters, or `VariableLengthPhaseNet` when the window or the output
activation departs from the defaults. Both are already in the SeisBench
install QuakeScope ships, so a trained weight is a `.pt` and `.json` pair
dropped into `sb_catalog/models/v3/<class>/` and selected with `--weight`
(`QuakeScope/docs/rerun_2026/02_weights_and_container.md`). No new
architecture is proposed. The U-Net is the right shape for a per-sample
onset probability, and the retrain showed that the failure modes were data
and protocol, not capacity.

### 3.2 Sampling rate and window

**50 Hz, 6000 samples, 120 s**, in `VariableLengthPhaseNet` with
`in_samples=6000`. The rate was 25 Hz in the first draft and was raised on
2026-09-08 for a measured reason. SeisBench's `resample` brings a stream to
the model rate with a four-corner zero-phase Butterworth at the new Nyquist
and then decimates. On the PNW surface-event class the dominant frequency is
4.3 Hz, the spectral centroid 6.3 Hz, 74 % of the energy lies between 1 and
10 Hz and 10 % at 10 to 20 Hz (150-trace class summary,
`~/GitHub/thunderquakes/catalogs/pnwml_class_summary.csv`). At a 25 Hz
output that filter takes 1.4 dB at 10 Hz and 4.7 dB at 12 Hz, and lets 15 to
17 Hz through at −14 to −22 dB, where it aliases onto 8 to 10 Hz, the middle
of the band the picker is supposed to read; rock falls on short-period
stations, the highest-frequency members of the class, would lose their upper
half. At 50 Hz nothing below 15 Hz loses more than 0.2 dB and the aliased
band, 25 to 35 Hz, carries under 1 % of the class energy. 50 Hz is also
native to the CC broadbands that dominate the Rainier network today (27 of
48 vertical channels, §3.4) and an exact factor of two from the 100 Hz `EH`
and `HH` channels, so most traces are decimated once or not at all. The 120
s window holds a 100 s event with room for the onset to sit anywhere.

The cost stays small. Per second of data the model runs 50 samples through
268 k parameters, against 100 samples through the doubled filters of
`jma_wc`, whose convolutions cost about four times as much per sample:
roughly an eighth of the earthquake picker's arithmetic per station-day,
before the amplitude stages. The number is measured in Phase 0, not assumed.

What the lower rate would have bought is context. The nominal receptive
field of the five-level U-Net spans the window, but the effective one is
about 4 s at 100 Hz (Münchmeyer et al. 2022), so 8 s at 50 Hz and 16 s at 25
Hz. Whether 8 s is enough to tell a 30 s spindle from an earthquake coda is
the question the E3 arms answer: 25 Hz over 3001 samples (the first draft's
baseline, a sixteenth of `jma_wc`'s cost) and 20 Hz over 3001 samples, both
against the 50 Hz baseline on the earthquake-day false-pick rate and on
rock-fall recall. If a low-rate arm wins on rejection and loses on rock
falls, the fix is context without bandwidth loss: kernel 11 instead of 7, or
one more level, as a subclass of `PhaseNet`. That is the only architectural
change the plan allows, and only on that evidence. The precedent for a
low-rate, long-window PhaseNet trained on real noise is the low-frequency
earthquake picker of Münchmeyer et al. (2024), 20 Hz over 60 s, which
transferred across three subduction zones.

**Pre-filter.** A 0.5 Hz high-pass in the model's `filter_args`, applied by
SeisBench before normalisation, exactly as the `obs` weight does. The
per-window standard deviation normalisation otherwise puts a coastal or
ocean-bottom station's microseism in the denominator and a 1 to 10 Hz
landslide at a Cascade station becomes invisible after scaling. The
long-period detection of very large landslides at 20 to 150 s (Ekström and
Stark 2013) is a different instrument and a different model; out of scope.

### 3.3 Output channels

The pick export settles what the branches are not: of 21,721 PNSN
surface-event picks, 20,777 are `P` and 944 are `S`
(`~/GitHub/MTRainier/data/su_picks.txt`), and 624 of the 8,912 curated
surface-event traces carry an S sample. An S channel would be trained on 4 %
of the events and would learn the analysts' habit, not a phase. There is no
S branch.

Baseline: **three channels, `U`, `P`, `N`**, softmax, where `U` is a
surface-event onset (sUrface; SeisBench names phases by single letters) and
`P` is an earthquake P onset. The `P` channel exists to teach rejection. The
earthquake corpus (PNW ComCat, 184 k traces with analyst picks) is two
orders of magnitude larger than any surface-event set and is the population
the picker must never fire on, because in production it sees the same
aftershock sequences the earthquake picker does. Giving earthquake energy
its own class, instead of folding it into "not onset", puts a supervised
boundary exactly where the false picks would come from; at inference the `P`
channel is discarded, or kept as a free cross-check against the earthquake
picker's own picks. QuakeScope's picker reads the threshold as `U_threshold`
once `sb_catalog/src/picker.py:370-371` stops hard-coding `P` and `S` (one
small change).

First ablation (E2): **two channels, `U` and `N`**, the design as first
proposed. The hypothesis the pair tests is that the explicit earthquake
class lowers false `U` picks on earthquake days at equal recall (§6.2 item
3); if it does not, the two-channel model ships as the simpler one.

Second ablation (E2): a **sigmoid event-mask channel** in
`VariableLengthPhaseNet` (`output_activation="sigmoid"`), a boxcar from
onset to the analyst's end time where one exists. It is the hook for
duration in QuakeScope and it gives the encoder an envelope-shaped target
that the onset Gaussian alone does not; RockNet (Liao et al. 2023) used
exactly such masks for rockfall and earthquake signals at Luhu, Taiwan, and
transferred to Super-Sauze in the French Alps. It is an arm, not the
baseline, because end times exist for 210 events (§4.1) and a partially
labelled channel needs masking in the loss.

### 3.4 Components

Three-component input with heavy channel dropout: half the training draws
keep only Z, so the same weight runs on the vertical-only short-period
stations that sit on the volcanoes. The instrument mix has also inverted
over the life of the labels: the pick export is 80 % `EHZ` because it
reaches back to 1981, while the 35 stations with a vertical channel within
about 20 km of Rainier today carry 27 `BHZ` at 50 Hz, 18 `HHZ` at 100 Hz and
3 `EHZ` (EarthScope station service, queried 2026-09-08). The response
augmentation of §5.3 therefore runs both ways, broadband made to look
short-period and the reverse. Surface events carry little polarisation
information at local distance, so a Z-only model (`in_channels=1`) is the
lighter alternative and is arm E4. If E4 matches the three-component model
at matched budget, it ships: one weight, one channel, every station.

## 4. Data

### 4.1 Positives

**PNSN surface events.** Two forms of the same labels exist. The SeisBench
`pnw_exotic` file (Ni et al. 2023; public metadata read 2026-09-08) holds
9,267 three-component traces of 180 s at 100 Hz for 5,657 events: 8,912
surface-event traces on 5,425 events, 206 sonic booms, 146 thunder, 3 from
one plane crash. Columns: `source_type`, `trace_P_arrival_sample` (7,000 on
every trace that has one, so the analyst first arrival sits at 70 s),
`trace_P_onset` (4,931 impulsive, 3,873 emergent, 108 missing),
`trace_S_arrival_sample` (624 traces), `trace_snr_db` per component, station
code, channel and coordinates. No origin, location, magnitude, duration or
end time. The PNSN assigns `su` to a clear but emergent signal without
distinct P and S and with low frequency content, and the paper notes that
most such events are glacier-related icequakes and avalanches, with some
debris flows and rock falls; the analyst picks the first arrival at one or
two nearby stations. Behind the curated file sits the raw PNSN pick export,
`~/GitHub/surface_events/data/events/su_picks.txt`: 21,721 `su` picks on
12,129 events from 1981 to January 2024 with station, channel and pick
quality, and `PNSN_Pick_Label.csv` with 13,465 `su` picks on 7,523 events
and an `Impulsivity` flag. The export is the better source for this picker:
it has every pick of an event, so multi-onset windows can be cut from
continuous data; it reaches stations and years the curated file does not
(the curated file runs 2002 to 2021); and the impulsivity flag is a per-pick
handle on label width (§5.1).

Where and when the labels are, from the station coordinates (nearest volcano
within 50 km):

| Place | Traces | Events | Role |
|---|--:|--:|---|
| Mount St. Helens | 5,211 | 3,274 | training |
| Mount Rainier | 2,902 | 1,779 | training |
| Newberry | 429 | 218 | development, held out as a place |
| Mount Hood | 176 | 73 | development, held out as a place |
| Three Sisters, Baker, Adams, Jefferson, other | 194 | 81 | development, pooled |

Events per year run from 7 (2006, the network desensitised during the St.
Helens unrest) to 586 (2003), with 182 in 2016 and 413 in 2021, the two
years the earthquake plan already holds out; those two years are held out
here as well, so one convention covers the repository.

Four properties of the labels bind the design. Every start time in the
curated file ends in `.000000` and no row of the export carries a fraction:
**the onsets are quantised to 1 s**, and the label width of §5.1 cannot be
narrower. 4,214 of the 5,425 curated events have a single trace, and 9,380
of the 12,129 exported events a single pick; only the 1,200 to 2,700
multi-station events allow the move-out check of §7. The labelled set is a
high-SNR set: median 12.6 dB, 18 % below 5 dB, 7 % below 0 dB on the
component reported last in `trace_snr_db`, against 33 % below 5 dB on the
earthquake benchmark; the field will be worse, so real-noise superposition
(§5.3) is what supplies the low-SNR half of training, not the labels. And
the channels are 79 % `EH` (7,034 traces), 18 % `BH`, 3 % `HH`, a
short-period vertical corpus, which is the strongest argument for the Z-only
arm of §3.4. These are the only analyst-picked onsets of the target class at
volume anywhere, and they come from stations with different noise (glacier,
river, wind, snow). They are the training set. The other classes in the
curated file, thunder and sonic booms, are negatives with an onset (§4.2).

**End times.** Wes Thelen's video- and observer-verified catalogue
(`surface_events/events/Wes_Cat_{rainier,st_helens,hood}.csv`: 133, 59 and
18 events) is the only hand-picked start and end time per event in the lab's
holdings. It anchors the event-mask arm (§3.3) and calibrates the two
automatic duration estimators already written, the cumulative fourth-power
envelope of `surface_events/src/ML_Picker_all.ipynb` and the 5 to 95 %
cumulative energy of
`Surface_Event_Detection/Common_Scripts/seis_feature.py`, which can then
supply weak end labels for the rest.

**Confirmed events from the exotic-event catalogues**, picked by hand for
this project. The Exotic Seismic Event Catalog (ESEC; Collins, Bahavar,
Allstadt; Bahavar et al. 2019, Collins et al. 2022) holds 242 events in the
2022 release and 245 rows in the copy at
`PNW_Seismic_Event_Classification/data/IRISExoticEventCatalog.txt`, verified
by non-seismic means, global, with start and end time, volume, mass, runout
and location uncertainty per event and pointers to the waveforms at the DMC.
It has event times, not per-station onsets: station picks for the acceptance
suite are made by hand (§6.3). Waveforms for 173 of the events are already
on disk as 300 s cuts with StationXML (2,418 files,
`PNW_Seismic_Event_Classification/data/iris_esec_waveforms/`). By type: 48
rock falls, 30 rock avalanches, 27 landslides, 24 rock slides, 18 rock and
ice avalanches, 14 snow avalanches, 14 debris flows, 11 ice avalanches, 6
lahars, 5 outburst floods, and singletons; the copy ends in August 2021, so
Dickson Fjord 2023 and Brienz 2023 come from the ESEC v3.0 release of May
2025 or are added by hand. It is the only ground truth outside the Pacific
Northwest and it is small, so it is **never trained on**; it is the
acceptance suite (§6). One caution for the baselines: Kharita et al. (2026)
added 1,866 ESEC surface-event traces to QuakeXNet's training in one
experiment, so the classifier's score on ESEC is read as in-domain unless
the weight is the one trained without them. The Alaska Earthquake Center's
36 catalogued landslides since 2005 (ComCat, `eventtype=landslide`, queried
2026-09-08; 57 worldwide, of which 23 in 2024 alone) are out-of-region
positives for the development suite.

**Sources outside the Pacific Northwest, by access mode** (checked against
the services on 2026-09-08; decision of the same day: the two API sources
now, the requests deferred).

| Source | Labels | Access | Role |
|---|---|---|---|
| SED, Swiss landslides and rockslides | Analyst onsets: the FDSN event service honours `eventtype=landslide` (39 events in 2024) and returns picks with `includearrivals=true`; waveforms on EIDA | **API, now** | training, split by year, with 2016 and 2021 held out; the one out-of-region source with analyst onsets at volume |
| GeoNet, New Zealand landslides | `eventtype=landslide`, 41 events since 2000; waveforms open | **API, now** | development suite |
| USGS ComCat, Alaska landslides and ice quakes | Event list by `eventtype`; the service rejects `includearrivals`, but each event's `phase-data` product holds the AEC picks: 17 manual picks at millisecond precision on `BHZ` for each of the two events checked | **API, now** | landslides to the development suite; ice quakes as the diagnostic set of §4.1 |
| Illgraben debris flows (WSL) | Manual start and end times; waveforms behind the Geopraevent portal | email, deferred | acceptance first; training only with 2020 kept back |
| Ruapehu lahars (GNS, ERLAWS) | Not in the GeoNet event service | email, deferred | acceptance |
| Piton de la Fournaise rockfalls (OVPF) | The IPGP event service ignores `eventtype` and carries no type column; about 7,000 labelled events exist in the observatory's own catalogue | email, deferred | training if it comes, split by year |
| Stromboli landslides (INGV-OV) | INGV event service is earthquakes only | email, deferred | acceptance |

Whatever arrives with onset times goes to training only if it is a place
with more than one event, so that the place can be split by year; a single
event from a place is a test case.

**Model-derived detections** (Akash's 114,775 Rainier surface events over 15
years) are not labels. They are a pool for mining hard negatives and for a
later pseudo-label round after manual review of a sample.

**Adjacent classes.** Alaska ice quakes (16,242 in ComCat since 2005) are
neither positives nor negatives: calving and crevassing are impulsive, mass
movements are emergent, and the PNSN `su` label itself includes
glacier-sourced events on Rainier. They are excluded from training and
scored as a diagnostic set: what the picker calls them is reported, not
optimised.

### 4.2 Negatives

The picker fails in production by firing on things that are not surface
events, so the negative set is the larger half of the corpus and is built
from the populations that will be in the continuous data.

1. **Earthquakes with analyst picks** from `pnw` (ComCat), at every
   distance and magnitude, including regional and teleseismic arrivals
   whose long codas look like a spindle at 50 Hz. Labelled `P` in the
   baseline, `N` in the two-channel arm.
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
   River noise has been characterised already for the December 2025
   atmospheric river at UW.SKY, QPLV, MNRO and RVRV against USGS discharge
   (`~/GitHub/gaia-landslides-detect/notebooks/atmospheric_river_seismic_discharge.ipynb`);
   those station-days go to the different-noise suite, not to training.
5. **Aftershock hours.** Continuous days from the held-in earthquake
   sequences of the earthquake plan (not the acceptance ones), where
   arrivals are seconds apart and the coda never ends. No surface event
   labels; the whole day is negative. This is the regime that will produce
   the most false picks in a global run.

### 4.2b Making the earthquakes hard

The instruction of 2026-09-08 is that training should try to trick the
picker into calling an earthquake a surface event, and fail. A random draw
of earthquakes does not do that: most are impulsive, high-frequency and
clean, and a model rejects them for free while learning nothing about the
ones that look like the target. The earthquake negatives are therefore
chosen and dressed to resemble surface events, and every one keeps its
earthquake label.

- **Chosen.** Emergent onsets (`trace_P_onset = emergent` in the PNW ComCat
  metadata) are oversampled to half the earthquake set; SNR below 5 dB to a
  third; regional distances of 100 to 300 km, where Pn is emergent and the
  coda outlasts the source; deep events; M below 1.5 on short-period
  stations near the volcanoes, which is what a local earthquake looks like
  on the same instruments that record the surface events; teleseisms, whose
  P is a long-period emergent swell tens of seconds long; and earthquakes
  recorded during storms.
- **Dressed.** With probability 0.3 an earthquake window is low-passed at 8
  to 12 Hz so that its spectrum matches the surface-event class; with
  probability 0.2 it is time-stretched by 1.2 to 2, so that an impulsive
  onset becomes a ramp; with probability 0.2 it is cropped to start inside
  its own coda. All keep the earthquake label. Stretching a surface event is
  a property of the class; stretching an earthquake and still calling it one
  is what teaches the model that a ramp alone is not the target.
- **Superposed.** An earthquake and a surface event in one window, both
  labelled, probability 0.2 (§5.3): the picker must put `P` on one and `U`
  on the other and not merge them. A small earthquake inside a surface
  event's envelope is the hardest instance and is generated on purpose.
- **Mined.** After the baseline arm, the model runs over whole earthquake
  days (PNW earthquake days, Alaska swarm days, the held-in aftershock hours
  of item 5); every `U` pick above the working threshold with no surface
  event in the catalogue within 60 s becomes a negative window, with an
  earthquake label where one applies, and the model is retrained. Two rounds
  (E9). The mining days are disjoint from the development and acceptance
  days and are listed with the manifests, so the false-pick metric of §6.2
  is never scored on days the model was corrected on.

The scoring side of the same instruction is §6.2 item 3, false `U` picks per
catalogued earthquake, and §6.3 item 3, two days of Ridgecrest and Kaikōura.
A picker that passes recall and fails those does not ship.

### 4.3 Splits and hold-outs

**By place, not by trace.** A volcano's surface events repeat at the same
stations for years, so a random split tests memory of a station. The counts
(§4.1 table) force the shape of the split: St. Helens and Rainier hold 93 %
of the events. Training is St. Helens and Rainier. The development suite is
every other Cascade volcano as a place, Newberry (218 events) and Hood (73)
as the two scored separately, the rest pooled, with the 239 stations within
50 km of each volcano listed in
`surface_events/data/station/Volcano_Metadata_50km.csv`, plus the held-out
years below at the two training volcanoes. A cross-volcano swap (E8: train
on St. Helens only, score on Rainier, and the reverse) measures transfer
between the two well-instrumented places directly and is the cheapest
generalisation test available before any request is answered.

**By time as well.** Hibert's Piton de la Fournaise classifier collapsed on
the same volcano five years later because the rockfall mechanism changed
(QuakeXNet plan §4b). The years 2016 and 2021 are held out of training
everywhere, the same years as the earthquake plan (182 and 413 surface-event
events), and recall by year is reported on the development suite so a drift
of this kind is seen rather than averaged away.

**Verification.** The same spatiotemporal join as
`scripts/heldout_sequences.py`, extended with place windows around each
volcano and each test region, run over every source before the manifest is
written; the builder refuses to run without the exclusion list, as it now
does for earthquakes. The noise corpus is filtered with the same windows: a
quiet day at a held-out station teaches that station's character.

## 5. Labels, windows and augmentation

### 5.1 Onset label

A Gaussian at the analyst onset with σ = 1.0 s (50 samples at 50 Hz),
against 0.1 s for earthquakes. One second is the floor set by the 1 s
quantisation of every label (§4.1); wide labels have precedent where the
onset is uncertain, 1.5 s in PhaseNet-DAS (Zhu et al. 2023) against 0.2 s in
the volcano picker of Zhong and Tan (2024). The analyst's own uncertainty on
an emergent onset is probably larger than a second and is to be measured
from the multi-station events (residual of each pick against a move-out fit
at 1 to 3 km/s) and from the `Impulsivity` flag, with emergent picks given a
wider label than impulsive ones if the residuals say so. Arms at σ = 1.0,
2.0 and 4.0 s (E5); 0.5 s is below the label resolution and is not run. Pick
`quality` (0 to 1 in the export) weights each window's loss, so a weight-0
pick contributes nothing rather than a wrong onset. The `N` channel is one
minus the maximum of the others, as in
`scripts/manifest_dataset.py::make_labels`.

For the event-mask arm: a boxcar from onset to end time, ramped over 2 s at
each edge, with the loss masked to zero on traces that carry no end time.

### 5.2 Windows

Cut from the full stored trace (or from continuous data where the source is
a bulletin) at native rate, brought to 50 Hz with SeisBench's resampler, and
cropped to 6000 samples per draw with the onset uniformly across the window.
Two thirds of the draws from a positive trace hold the onset; one third is
cut at random over the trace, so that windows holding only the event's body
or tail, with no onset label, are a fixed fraction of training (Münchmeyer
et al. 2022): a coda without an onset is exactly what the model must learn
not to pick. Each window carries: SNR (10 s after onset over 10 s before, on
Z, in dB), station, channel code, native rate, place, year, event type,
number of labelled onsets, and the operator's duration where present.

### 5.3 Augmentation, all on the fly

Every item below is a lesson from the audit or a property of the target
signal; nothing is included because it is customary.

- **Real-noise superposition**, probability 0.5, target SNR uniform in 0 to
  20 dB, noise drawn first from the same station and season when available
  and otherwise from the training-station noise corpus. Labels unchanged.
- **Cross-class superposition**, probability 0.2: an earthquake window added
  to a surface-event window, or the reverse, with both labelled; a second
  surface event in the same window, both labelled. This is the multi-event
  rule of the earthquake plan, applied to the class the picker must reject.
  The earthquake-only dressings (low-pass, stretch, mid-coda crop) are in
  §4.2b.
- **Time stretch** of surface events, factor uniform in 0.8 to 1.25 by
  resampling, label moved with the onset. Duration varies by an order of
  magnitude across the target class and the training set cannot cover it by
  itself. Earthquakes are stretched separately and harder, and keep their
  label (§4.2b).
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
on validation loss with patience 15. The corpus is small (about 9,000
curated traces plus the export's extra picks, so of order 20,000 positive
traces before augmentation), so an epoch is minutes and the whole arm table
(§7) fits in a few GPU-days. Every arm is run with three seeds and reported
as mean and range, because at this corpus size the seed variance can be as
large as an arm effect.

Initialisation arm (E1b): the encoder weights of SeisBench's `original`
PhaseNet, which are defined in samples and therefore respond to half the
frequency at 50 Hz. Whether that is a useful prior or a burden is an
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
   fraction of analyst onsets on the development suite (held-out
   volcanoes and years) with a pick within ±5 s, reported per SNR bin, per event type, per year, per channel code.
2. **Onset residual** on detected events: median and interquartile range,
   in seconds, against the analyst pick, per distance bin.
3. **False-pick rate on earthquake days**, from the held-in aftershock
   hours of §4.2 item 5 and from the development suite's own earthquake
   days: picks per station-day and per catalogued earthquake.
4. **Event-level detection**: onsets on three or more stations within a
   plausible move-out window (30 s for a 50 km aperture) count as one
   detection; recall against the catalogue's events and false events per
   day.

Baselines scored on the same suite before any training, so a gain is a gain
over something: a long-window STA/LTA tuned on the training set (Allstadt's
parameters as a starting point), an envelope-duration detector, the current
QuakeScope path (PhaseNet picks passed to QuakeXNet, `su` probability above
0.5), and the earthquake pickers themselves run as if their P channel were a
surface-event detector. The lab has one such number already: the ELEP
ensemble of six pretrained EQTransformers, applied to 10,997 PNSN
surface-event picks, lands within a median of 0.08 s of the analyst but with
a standard deviation of 5.65 s (`surface_events/data/bb_elep_picks_su.csv`);
that spread is what a dedicated picker has to beat, and `jma_wc` run the
same way measures how often the campaign's picker already fires on these
events.

Thresholds are set on the development suite per station class (broadband
against short-period) to the false-pick target, recorded with the weight,
and not touched afterwards. Two habits from the thunderquake work carry over
(`~/GitHub/thunderquakes/report/manuscript.qmd`): false picks on the
development suite become hard negatives for the next arm, never for the one
being scored, and a detection rate on continuous data is reported against
the rate on a same-size random sample of windows with a two-proportion test,
so that a picker firing everywhere cannot look good.

### 6.3 Acceptance suite, read once

Three regimes the development suite does not test, all outside the training
footprint:

1. **Out of region.** Every ESEC event outside the Pacific Northwest
   with a station within 100 km, from the 173 with waveforms on disk plus
   the post-2021 additions, each with its regime named so that recall is
   read by regime and not as one number: Himalaya (the Chamoli 2021
   rock and ice avalanche on the dense Uttarakhand network, Cook et al.
   2021; Langtang 2015 only if waveforms exist), Greenland fjords (Karrat
   2017-06-17, Poli 2017; Dickson 2023-09-16, Svennevig et al. 2024, data
   on Zenodo), Alaska glaciated coast (Taan Fiord 2015-10-17 and the nine
   large southern-Alaska landslides of Karasözen and West 2024, Barry
   Glacier among them), Iceland (Askja 2014-07-21 on 58 stations, Schöpa
   et al. 2018), Alps (Piz Cengalo 2017-08-23, Brienz 2023-06-15, the
   Illgraben debris flows), Taiwan (Hsiaolin 2009 and the 40 Morakot-era
   landquakes of Chao et al. 2017, event times and locations only),
   British Columbia (Elliot Creek 2020-11-28), Central American volcanoes
   (Fuego 2018 lahars, Bejar et al. 2026). Kaikōura's coseismic slides
   have no seismic catalogue and sit in a Mw 7.8 coda; they are not
   scoreable and are dropped. The list is frozen at the
   Phase 0 census and committed beside `heldout_sequences.py` before any
   training. Scored as detection at the station level (onset within ±10 s
   of a hand pick made for this suite by someone who has not seen the
   model output) and at the event level.
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
matched budget on the development suite, holds a false-pick rate below one
per station-day on the different-noise suite, and detects, at the event
level, a fraction of the out-of-region events fixed at the Phase 0 census
(written into the committed acceptance list, per regime, before training)
once events with no station within 100 km are set aside.

### 6.4 Second stage: QuakeXNet on the picks, and association

The question of 2026-09-08: is SUNet sufficient on its own, or should
QuakeXNet sit on top of it as a binary classifier, geomorphological event
with a pick against everything else?

SUNet is already that classifier at the sample level: `U` against `P` and
`N`, trained on the negatives of §4.2b and scored on false picks per
earthquake. Whether it is sufficient is not a design question but the result
of §6.2 item 3 and §6.3 item 3. What a second stage can add is a different
view of the same 100 s, and it is cheap because it runs per pick, not per
station-day, the same cost class as the amplitude extractor.

The smart arrangement is a cascade trained on the first stage's mistakes:

1. **Cut at the pick.** Every `U` pick becomes a 100 s window with the
   onset 15 s in, the placement QuakeXNet was trained on. QuakeScope's own
   plan already asks for this for earthquake picks; on a blind slide the
   classifier's accuracy swung from 16 to 78 % with placement alone
   (`quakexnet_generalization_plan.md` §1).
2. **Score, do not argmax.** Store QuakeXNet's four probabilities with the
   pick. The reranker is the ratio of `su` to the rest, and the operating
   point is set on the development suite to the same false-pick budget as
   the picker, so the two stages are compared on one curve: does adding
   the classifier's score move recall at matched budget, or not. If it
   does not, the stage is dropped and the pipeline is one model.
3. **Retrain the head on the picker's errors.** If it helps, the version
   worth building is a binary head on QuakeXNet's spectrogram encoder,
   trained on pick-centred windows: positives are analyst surface events,
   negatives are SUNet's own false picks harvested from the mining days of
   §4.2b. A second stage earns its place by seeing what the first one gets
   wrong, not by re-learning the same classes on the same windows.
4. **Then associate.** Onsets on three or more stations with a plausible
   move-out and coherent envelopes (§6.2 item 4) remove single-station
   false picks that no single-station model can, and produce the first
   location. This is the strongest filter of the three and the only one
   that also delivers the product.

Two limits fix the order. QuakeXNet needs three components at 50 Hz on `BH`
or `HH`, so it cannot rerank a pick on a vertical-only `EHZ` station, which
is where most of the labels come from; there the `P` channel of SUNet and
the association step are the whole defence. And QuakeXNet's `su` class is
its weakest (62 % in Alaska, `su` mistaken for `px` on ESEC), so the stage
is admitted only on the measured gain of step 2, never assumed.

## 7. Experiments and gates

**Phase 0, no training (two to three weeks).** Census: the curated file is
done (§4.1, from the public metadata); still to do are the pick export by
station and year against the curated file, the join to the Thelen end-time
catalogue, and the `pnw` earthquake and `pnw_noise` volumes per training
station; the inference cost of the 50 Hz model per station-day measured on
one CPU core with the thunderquake timing harness (a 371 k-parameter CNN
there runs at 27 ms per window); choice of the held-out volcanoes and years
from the counts. Label audit: onset residual between stations of the same
event against distance (a pick that moves faster than 6 km/s is wrong),
inter-analyst width where available, a confident-learning pass as in
`docs/LABEL_ERROR_FILTERING.md` adapted to a two-class target. ESEC harvest:
which of the 245 events have waveforms within 100 km, at what rate; freeze
the acceptance list and commit it as a window list beside
`heldout_sequences.py`. Baselines run on the development suite. SED and
GeoNet landslide catalogues pulled through their FDSN services with
arrivals; the ComCat phase-data products for the Alaska landslides; the
email requests (WSL, GNS, OVPF, INGV-OV) deferred by decision of 2026-09-08.

**Phase 1, corpus (four to six weeks).** The PNW files are already on the
lab servers, so the only harvest is 180 s windows from EarthScope for the
export's picks outside 2002 to 2021, plus the SED and GeoNet events, through
obspy; builder for surface-event manifests with the window fields of §5.2;
noise corpus from continuous data at the training stations with the
catalogue check; exclusion list and year hold-out verified on every
manifest; fingerprints committed.

**Phase 2, arms (six to eight weeks).** Each arm three seeds, selected on
§6.2.

| Arm | Question |
|---|---|
| E0 | Baseline: 50 Hz, 6000 samples (120 s), three channels `UPN`, 3C with dropout, σ = 1 s, full augmentation, §4.2b negatives |
| E1a | Data scaling: nested 25, 50, 100 % of the positive set at fixed negatives. Is the model data-limited? |
| E1b | Initialisation from `original` PhaseNet weights against scratch |
| E2 | Outputs: `UPN` against `UN` against `UPN` plus sigmoid event mask |
| E3 | Rate and window: 25 Hz / 3001 samples and 20 Hz / 3001 samples against the 50 Hz baseline; then kernel 11 or six levels only if a low rate wins on rejection |
| E4 | Z-only model |
| E5 | Label width σ = 1, 2, 4 s; loss weighted by pick quality against unweighted |
| E6 | Augmentation ablation, one arm each without: real noise, time stretch, cross-class superposition, instrument response |
| E7 | Negative composition: without aftershock hours; without exotic non-`su` classes |
| E8 | Cross-volcano swap: train St. Helens, score Rainier, and the reverse |
| E9 | Hard-negative mining: zero, one and two rounds over earthquake days (§4.2b) |
| E10 | Second stage: picker alone, plus QuakeXNet score, plus retrained binary head on the picker's false picks (§6.4) |

E1a runs first. If recall at matched budget is flat from 50 to 100 % of the
positives, the model is not data-limited and the request campaign of §4.1 is
about test coverage, not training; if it is still rising, every onset that
can be obtained goes to training and the plan's centre of gravity moves to
§4.1.

**Phase 3, acceptance and deployment (two weeks).** One candidate; §6.3
once; export as a SeisBench pair under the contract of
`QuakeScope/sb_catalog/models/v3/phasenet/README.md` (every `.vN` the
installed SeisBench resolves, weights saved CPU-mapped because the image
runs CPU-only PyTorch, loaded once with the network blocked before it goes
near the fleet); the `U_threshold` change in the QuakeScope picker; run on
`EH`, `BH` and `HH` channel groups (the classifier today runs on `BH` and
`HH` only, `picker.py:598`); thresholds per station class recorded in
`sb_runs`; the two QuakeScope notebooks re-run with the surface picks as a
third phase to confirm nothing in the earthquake catalogue changes.

| When | Deliverable | Decision |
|---|---|---|
| Week 3 | Census, label audit, acceptance list frozen, baselines scored | Held-out volcanoes and years; which requests are worth waiting for |
| Week 9 | Corpus, noise corpus, manifests verified | Go to E0 and E1a |
| Week 12 | E1a scaling curve | Training-limited or data-limited; scope of §4.1 |
| Week 17 | E2 to E10, one candidate | Acceptance run |
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
  used 150 s at 40 Hz. Both are evidence that 120 s at 50 Hz loses nothing the analysts used.
- Duration estimators (two, §4.1), envelope features, and the
  cross-station envelope correlation `pick_time()` of
  `surface_events/src/utils.py`, which can propagate a good pick to
  stations of the same event when the analyst picked only one.
- Deployment: QuakeXNet is wrapped as a SeisBench `WaveformModel` and runs
  in QuakeScope's classifier on `BH` and `HH` streams at 50 Hz; the same
  wrapper pattern serves the picker, which is a `PhaseNet` and needs none.

Reusable from this repository without change: `scripts/metrics.py`
(bootstrap intervals), `scripts/heldout_sequences.py` (windows, year
hold-out, manifest check; needs place windows for the volcanoes and the ESEC
sites), the manifest and cached-dataset pattern of
`scripts/manifest_dataset.py` and `scripts/fast_manifest_dataset.py` (with
the random crop moved into `__getitem__`, as the v21 proposal already
specifies), and `scripts/build_noise_dataset.py` as the template for a
station-specific noise corpus. New: a surface-event manifest builder, the
label and augmentation module of §5, a continuous-data scorer that wraps
`annotate` and `classify` and computes §6.2, and a config family
`configs/su_picker_v*.yaml` with the same header convention as the
earthquake configs.

## 9. What this plan does not do

No amplitude, duration or size estimation beyond the optional event-mask
channel; that is QuakeScope feature extraction. No long-period detection of
giant landslides at 20 to 150 s. No source classification beyond onset,
earthquake and neither; QuakeXNet keeps that job and runs on the surface
picks under the terms of §6.4. No training on ESEC. No selection on any
number from the acceptance suite. No claim of a working picker before the
acceptance run.

## References

- Zhu, W., and Beroza, G. C. (2019). PhaseNet. *GJI* 216, 261–273.
- Zhu, W., et al. (2023). Seismic arrival-time picking on distributed acoustic sensing data using semi-supervised learning (PhaseNet-DAS). *Nat. Commun.* 14, 8192.
- Ni, Y., et al. (2023). Curated Pacific Northwest AI-ready Seismic Dataset. *Seismica* 2(1). doi:10.26443/seismica.v2i1.368
- Kharita, A., Denolle, M., Hutko, A., Hartog, R., and Malone, S. (2026). Exploration of Machine Learning Methods to Seismic Event Discrimination in the Pacific Northwest. *Seismica* 5(1). doi:10.26443/seismica.v5i1.2068
- Münchmeyer, J., et al. (2022). Which picker fits my data? *JGR Solid Earth* 127, e2021JB023499.
- Münchmeyer, J., et al. (2024). Deep learning picker for low-frequency earthquakes across subduction zones. arXiv:2311.13971.
- Woollam, J., et al. (2022). SeisBench. *SRL* 93, 1695–1709.
- Liao, W.-Y., Lee, E.-J., Chen, D.-Y., Chen, P., Mu, D., and Wu, Y.-M. (2023). RockNet: Rockfall and earthquake detection and association via multitask learning and transfer learning. *IEEE TGRS* 61, 1–12. doi:10.1109/TGRS.2023.3284008
- Zhong, Y., and Tan, Y. J. (2024). Deep-learning-based phase picking for volcano seismicity. *GRL* 51, e2024GL108438.
- Ekström, G., and Stark, C. P. (2013). Simple scaling of catastrophic landslide dynamics. *Science* 339, 1416–1419.
- Allstadt, K. E., et al. (2018). Seismic and acoustic signatures of surficial mass movements at volcanoes. *JVGR* 364, 76–106.
- Hibert, C., et al. (2017). Automatic identification of rockfalls and volcano-tectonic earthquakes at Piton de la Fournaise using a Random Forest algorithm. *JVGR* 340, 130–142.
- Chmiel, M., et al. (2021). Machine learning improves debris flow warning. *GRL* 48, e2020GL090874.
- Wenner, M., et al. (2021). Near-real-time automated classification of seismic signals of slope failures with continuous random forests. *NHESS* 21, 339–361.
- Huang, Q., et al. (2025). Unsupervised detection of debris flows at Illgraben. *GJI*. doi:10.1093/gji/ggaf353
- Bahavar, M., Allstadt, K. E., Van Fossen, M., Malone, S. D., and Trabant, C. (2019). Exotic Seismic Events Catalog (ESEC) data product. *SRL* 90, 1355–1363. doi:10.1785/0220180402
- Collins, E. A., et al. (2022). Seismogenic landslides and other mass movements. USGS data release. doi:10.5066/P90VGCSK (v3.0, May 2025, on ScienceBase)
- Svennevig, K., et al. (2024). A rockslide-generated tsunami in a Greenland fjord rang Earth for 9 days. *Science*. doi:10.1126/science.adm9247
- Poli, P. (2017). Creep and slip: seismic precursors to the Nuugaatsiaq landslide (Greenland). *GRL* 44, 8832–8836.
- Cook, K. L., et al. (2021). Detection and potential early warning of catastrophic flow events with regional seismic networks (Chamoli). *Science*. doi:10.1126/science.abj1227
- Karasözen, E., and West, M. E. (2024). Toward the rapid seismic assessment of landslides in coastal Alaska. *The Seismic Record* 4, 43–51. doi:10.1785/0320230044
- Schöpa, A., et al. (2018). Dynamics of the Askja caldera July 2014 landslide, Iceland, from seismic signal analysis. *Earth Surf. Dynam.* 6, 467–485.
- Chao, W.-A., et al. (2017). A first near real-time seismology-based landquake monitoring system. *Sci. Rep.* 7, 43510.
- Bejar, G., et al. (2026). Detection of lahars at Volcán de Fuego from seismic data. *JGR Solid Earth* 131. doi:10.1029/2025JB032019
