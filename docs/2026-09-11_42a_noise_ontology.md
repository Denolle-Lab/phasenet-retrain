# 42A: noise ontology, event-free harvest rule and pilot pool

Checkpoint 42A of issue #42, branch `issue/42a-noise-pilot`, stacked on
`issue/33a-versioned-exclusions` (PR #71). Everything below was run on
2026-09-12 in the base conda interpreter (Python 3.9.20, numpy 2.0.2, pandas
2.3.2, obspy 1.4.2; no torch, no h5py, no SeisBench). `python -m pytest
tests -q`: 110 passed (36 new). The two-station demonstration was run three
times against IRIS and USGS (the first with band-dominance-only features,
see "Features"; the second from the raw cache; the third, the committed one,
fresh from the network after clearing the cache). The census of the second
and third runs is identical.

## What 42A releases

- `scripts/noise_ontology.py`: the eleven classes of
  `docs/2026-09-07_training_plan.md` §5.1 as `NoiseClass`, each with a
  definition, a source rule, a feature rule (where one exists) and an
  ontology category; `negative_support_for`; `spectral_features` and
  `classify_features`; `station_split`; the manifest schema
  (`MANIFEST_COLUMNS`, `empty_manifest`, `check_manifest`).
- `scripts/build_noise_pool.py`: the harvest driver (FDSN day chunks with
  bounded retries and a wall-clock budget, 120 s windows at native rate,
  the event-free rule with iasp91 travel times, features, class assignment,
  `exclusion_bundle.apply_exclusions(kind="noise")`, station split, parquet
  manifest, one `.npz` per station-day, `manifest.json`).
- `data/noise_pools/pilot_2019/manifest.parquet` and `manifest.json`: the
  demonstration. The waveforms and the raw FDSN cache are gitignored
  (`data/noise_pools/raw/`, `data/noise_pools/*/windows/`).
- `tests/test_noise_ontology.py` (20) and `tests/test_build_noise_pool.py`
  (16): synthetic windows, a fake FDSN client, a fake catalogue, the fixture
  bundle of `tests/test_exclusion_bundle.py`. No network.

No model is run on any window (§5.2). Nothing in the pilot was reviewed by a
person, so no row carries `negative_support = reviewed`.

## The ontology

Two orthogonal axes. The class says what the ground and the instrument were
doing; the category says what the window may be used for and fixes
`negative_support`. The audit's rule (roadmap audit, #42 row) is enforced in
`negative_support_for` and re-checked on every written manifest by
`check_manifest`: aftershock coda and LFEs are never negatives by catalogue
absence.

| Class | Category | `negative_support` if event-free with a local catalogue | Label from | Source rule (abridged) |
|---|---|---|---|---|
| `ocean_microseism` | background | certified | features | coastal and island stations; the station's own 0.1-0.5 Hz power percentile |
| `wind_site_tilt` | background | certified | features | exposed and high-elevation stations |
| `cultural_diurnal` | background | certified | features | urban and industrial stations, by hour of day |
| `hydrological` | background | certified | source only | registry proximity to rivers and glaciers, season |
| `impulsive_non_earthquake` | task_excluded | certified | features | PNW exotic classes, operator explosion/landslide labels, VCSEIS long-period |
| `volcanic_tremor_hydrothermal` | task_excluded | certified | source only | INGV-OE/OV, HVO, IMO, AVO unrest periods outside the held-out places |
| `tectonic_tremor_lfe` | unlabelled_interval | unknown, always | source only | PNSN tremor catalogue, Hi-net |
| `earthquake_coda_sequence_hum` | unlabelled_interval | unknown, always | source, or the harvest's coda rule | non-held-out aftershock sequences; a catalogued M5+ within 15 deg in the last 30 min or M7+ anywhere in the last 3 h |
| `polar_ice` | background | certified | station geography | latitude at or beyond 60 deg |
| `instrument_telemetry` | background | certified | features | QC flags; the mixing recipe synthesises the rest |
| `quiet_baseline` | background | certified | features | every station, lowest 10% power windows |

`task_excluded` is certified only in the sense "no catalogued tectonic event
has a predicted arrival here"; blasts and surface events can carry real P and
S and the general picker is allowed to pick them (#41A). A future reviewed
pool sets `reviewed_negative` and `negative_support = reviewed`; the harvest
has no path to it.

`negative_support` per category (`noise_ontology.negative_support_for`):

| Category | Condition | Value |
|---|---|---|
| background, task_excluded | event-free rule passed, local catalogue queried, `completeness_mag` stated | certified |
| background, task_excluded | otherwise (global catalogue only, local query failed, no bundle) | unknown |
| unlabelled_interval | any | unknown |
| reviewed_negative | any | reviewed |

## The event-free rule and its completeness assumption

A 120 s window is rejected when any catalogued event has a predicted first P
or first S at the station inside `[start - 120 s, start + 120 s]`, both ends
inclusive (`build_noise_pool.event_free_mask`). Catalogues: USGS ComCat
M ≥ 2.5 globally, plus a local FDSN event service at M ≥ 0 where the station
registry names one (`--local-catalogue NET.STA=CLIENT:radius_deg:min_mag:completeness_mag`).
Both are queried from 2 h before the day to its end (`CATALOGUE_LEAD_S`), so
an event before midnight whose S arrives after it is not missed. Travel
times are obspy `TauPyModel("iasp91")`, the earliest of `ttp` and of `tts`,
with distance rounded to 0.01 deg and depth to 1 km for the cache (error
under 0.2 s). Rejected windows are not written. `nearest_arrival_s` records
the signed time from window start to the closest predicted arrival of every
kept window; its minimum modulus in the pilot is 120.1 s.

The completeness assumption is explicit and unmeasured in 42A:
`completeness_mag` is a number stated on the command line for the local
catalogue, and `certified` means "event-free above that magnitude according
to that catalogue". Nothing here estimates completeness. The pilot states
M2.0 for the HV-contributed ComCat events within 4 deg of IU.KIP; HVO's
network is on Hawaii Island, 300 km from Oahu, so M2.0 is optimistic for
events near the station and 42B must replace it with a measured value
(Gutenberg-Richter fit of the local query per station-month). IU.ANMO has
no local catalogue in the pilot and every one of its rows is `unknown`, by
the rule, not by any failure.

The global rule rejects 31% of the pilot's windows (447 of 1440) from 69
M2.5-4.9 events, most of them invisible at either station. That is the rule
as specified: catalogue-based, not visibility-based, because visibility
screening would reintroduce the model bias §5.2 forbids. A
magnitude-distance floor for the global catalogue is a 42B design decision.

## Features

`noise_ontology.spectral_features`, one Hann-tapered periodogram of the
demeaned vertical, band powers as fractions of the total above DC:

| Feature | Definition |
|---|---|
| `rms`, `rms_log10` | vertical rms in raw units |
| `primary_band_frac`, `secondary_band_frac`, `microseism_band_frac` | 0.05-0.1, 0.1-0.5, 0.05-0.5 Hz power over total |
| `secondary_band_power_log10` | log10 of the 0.1-0.5 Hz power, raw units: comparable within a station only |
| `high_freq_frac` | 1 Hz to Nyquist over total |
| `mains_line_frac`, `mains_hz` | power in ±0.5 Hz of 50 or 60 Hz and two harmonics, above the local median baseline, over total; whichever of 50/60 carries more |
| `hv_low_ratio` | horizontal over vertical rms below 0.1 Hz |
| `spectral_flatness` | geometric over arithmetic mean of the octave-band PSD from 0.05 Hz to Nyquist (1 for white noise) |
| `kurtosis` | Fisher kurtosis of the vertical |
| `n_zero_frac`, `clip_frac` | samples zero on every channel; samples at the trace extremum |
| `hour_of_day` | UTC hour of the window start |
| `station_rms_log10_p10`, `station_secondary_p50` | station references: 10th percentile of `rms_log10` and median of `secondary_band_power_log10` over the windows of the station-day being labelled (rejected windows included); a reference over every day a station contributes is 42B |

`classify_features` fires the first rule in this order, thresholds in
`noise_ontology.THRESH`: instrument (`n_zero_frac ≥ 0.05` or
`clip_frac ≥ 0.01`), impulsive (`kurtosis ≥ 8`), cultural by mains
(`mains_line_frac ≥ 0.05`), quiet (`rms_log10 ≤ station p10`), microseism
(`microseism_band_frac ≥ 0.5`, more than the high band, and
`secondary_band_power_log10 ≥ station median` when the reference exists),
wind (`hv_low_ratio ≥ 3` and `spectral_flatness ≥ 0.3`), cultural by hours
(06-20 UTC, `high_freq_frac ≥ 0.5`, flatness below 0.3), fallback
`quiet_baseline` at confidence 0.2. Confidence is distance past the
threshold, capped at 1.

Two things the pilot taught about the features on raw broadband data, both
to be settled in 42B with more than one day per station:

- The first run used band dominance alone and labelled all 993 windows,
  inland ANMO included, `ocean_microseism` with `microseism_band_frac` at
  0.97-1.00. Raw BH velocity counts are microseism-dominated at every quiet
  site. §5.1 defines the storm and the quiet classes against the station's
  own distribution, so the station references were added; with one day the
  reference is that day, and half of each station's windows are above the
  median by construction.
- `spectral_flatness` on the octave bands is 0.000-0.019 across the pilot
  (red spectrum), so the wind rule cannot fire on raw broadband data even
  where `hv_low_ratio` reaches 6.5. Either compute it after response
  removal and pre-whitening, or drop flatness from the wind rule.

## Split

`station_split(station, fraction=0.2, seed_tag="42A")`: sha256 of
`"42A|NET.STA"`, first 8 bytes as a uniform number, `holdout` below the
fraction. A pure function of the station, so every window of a station is in
the same split whatever the pool, the harvest order or the day; a different
`seed_tag` gives a different but equally deterministic split. Tested on 500
synthetic stations (holdout fraction 0.12-0.28 at 0.2) and on repeated
windows per station.

## Schema

43 columns, in `noise_ontology.MANIFEST_COLUMNS`; `check_manifest` rejects a
frame with a column missing or extra, a value outside the vocabularies, a
non-empty `arrivals_json`, or a category whose `negative_support` breaks the
mapping above.

`pool, noise_class, ontology_category, negative_support, arrivals_json,
network, station, location, channel_band, station_latitude_deg,
station_longitude_deg, rate_hz, start_time, duration_s, npz_path, npz_index,
source, class_source, class_confidence, catalogue_used, completeness_mag,
event_free, nearest_arrival_s, exclusion_bundle_sha256,
independence_unverified, split` plus the 17 feature columns.

`class_source` is `source_flag`, `catalogue_coda` or `features`.
`arrivals_json` is empty on every row. `npz_path`/`npz_index` address the
kept windows in `data/noise_pools/<pool>/windows/<NET.STA.LOC.BAND>_<day>.npz`
(`data` as `(n, 3, samples)` float32 with Z first, `start_epoch`, `channels`,
`rate_hz`, `kept_index` into the day grid).

## The exclusion bundle

`apply_exclusions(kind="noise")` runs on `station_latitude_deg`,
`station_longitude_deg` and `start_time` with the trace list passed empty,
because pool windows are FDSN cuts, not SeisBench traces. Rows with unknown
station coordinates are quarantined under the default policy (tested:
`test_bundle_path_quarantines_unknown_coordinates`, and end to end with a
station whose inventory query returns nothing). `--no-bundle` records the
absence in `manifest.json`, sets `independence_unverified = True` and
`negative_support = unknown` on every row, and writes an empty
`exclusion_bundle_sha256`.

No bundle is committed (33A). The pilot used one built on this laptop with
`python scripts/exclusion_bundle.py build --allow-missing-sequence-list --out
<scratch>/bundle.json` at commit `f6d193c`: sha256
`e419160ef68730180bf8e8b0cdfb0141401c71c8973d4c34a0cf15cc8aec0694`,
uncertified (sequence list absent, 20 sources unhashed), and reproduced
bit-for-bit by a second build. The hash depends only on committed inputs
(`heldout_sequences.WINDOWS`, `HOLDOUT_YEARS`, `notebooks/benchmark_manifest.csv`,
`configs/evaluation_suites.json`), so anyone at that commit without the cache
gets the same bundle. Certification concerns the SeisBench sequence list and
does not bear on FDSN noise windows; the 23 windows and the two years do,
and they were applied: 0 rows removed, none in a window, none in 2016/2021.

## Demonstration

`python scripts/build_noise_pool.py --pool pilot_2019 --station IU.KIP.00.BH
--station IU.ANMO.00.BH --day 2019-03-12 --local-catalogue
IU.KIP=USGS:4.0:0.0:2.0 --bundle <scratch>/bundle.json --max-minutes 20`

IU.KIP (Kipapa, Oahu, 21.4200 N 158.0112 W; island) and IU.ANMO
(Albuquerque, 34.9459 N 106.4572 W; inland), `00.BH?` at 40 Hz from IRIS,
2019-03-12 UTC, a day outside every window of `heldout_sequences.WINDOWS`
(Ridgecrest and Monroe are July 2019) and outside the held-out years.
Largest global event that day M4.9.

| | IU.KIP | IU.ANMO |
|---|---|---|
| waveform fetch (IRIS, committed run) | 1.2 s, 3 traces, 17.1 MB miniSEED | 1.0 s, 3 traces, 8.6 MB |
| windows on the day grid | 720 | 720 |
| empty windows | 0 | 0 |
| catalogue events applied | 69 global + 13 local (M1.1-2.4) | 69 global |
| rejected by a predicted arrival | 227 (211 by global events, 22 by local, overlapping) | 220 |
| kept | 493 | 500 |
| kept as coda (unlabelled_interval) | 0 | 0 |
| `ocean_microseism` | 245 | 252 |
| `quiet_baseline`, of which bottom-10% rms at confidence 0.8 | 248, 57 | 248, 48 |
| `negative_support` | certified 493 | unknown 500 |
| split | holdout | train |
| `rms_log10` min / median / max (counts) | 3.66 / 3.80 / 3.95 | 2.33 / 2.58 / 2.82 |
| `microseism_band_frac` min / median | 0.967 / 0.992 | 0.981 / 0.999 |
| `hv_low_ratio` max | 6.47 | 3.72 |
| `kurtosis` max | 3.31 | 3.27 |
| `spectral_flatness` max | 0.019 | 0.017 |

Totals: 993 rows, 0 failures, 0 removed by the bundle, 13.3 s of the
20-minute budget (the earlier runs took 15.1 s from the network and 10.4 s
from cache). Every number above is from the committed
`data/noise_pools/pilot_2019/manifest.json` and `manifest.parquet`. The
KIP `certified` rows rest on the stated M2.0 completeness discussed above;
they are certified against that statement, not against a measurement.

## Migration of the legacy pools

`data/noise_global` (`scripts/build_noise_dataset.py`: STEAD, LenDB, TXED,
VCSEIS, OBST2024 "noise" traces, resampled to 100 Hz, 60 s cap) and
`data/noise_prephase` (`scripts/build_prephase_noise.py`: the 30 s before P
of signal traces, 100 Hz). Neither exists on this laptop; what follows is
read from the builders and `docs/2026-09-11_33a_exclusion_contract.md`.

| Lacks | Consequence under 42A | Migration |
|---|---|---|
| `starttime` empty on every `noise_global` row written before 2026-09-11 | the bundle quarantines the row; the event-free rule cannot be applied | re-extract with the patched builder (33A) so `trace_start_time` is carried, or relabel `unknown` |
| TXED noise has no station coordinates | quarantined at extraction | keep only under `--allow-unknown`, flagged; `unknown` |
| native rate lost (resampled to 100 Hz), windows capped at 60 s | violates #34 rate preservation and the 120 s window | re-extract at native rate from the source archive where the source has it; otherwise `unknown` and excluded from certified pools |
| class: the source's "noise" category, whatever screening produced it (STEAD and LenDB noise were model- or analyst-screened by their authors) | the parent's or the vendor's screening history is exactly what the acceptance criterion says cannot silently become an unbiased negative set | carry the source label as `source`, run the feature rules, `ontology_category = background`, `negative_support = unknown` until independently reassessed |
| `noise_prephase`: pre-P windows end at the P arrival | by the event-free rule these windows are rejected (a predicted P is inside `[start, start + 120 s]`), and the window before P of a catalogued event is the definition of a window the rule forbids | do not migrate into a certified pool; if retained, `earthquake_coda_sequence_hum` is wrong too (pre-event); a new class is not warranted, keep them out of the noise pools and treat them in the signal corpus as label context |
| OBST2024 (25k rows) | ocean-bottom flavour is out of scope this round | drop from the migrated pool |

The migration is not run in 42A: it needs the cache on the lab server. The
rule for it is: a legacy row gets `certified` only by passing the same
event-free query as a fresh window, with a start time, station coordinates
and a local catalogue; everything else is `unknown`.

## What remains for 42B

- Class, station and season census: more than one day per station, so the
  station references are climatological; `completeness_mag` measured per
  station-month; the wind rule recomputed after response removal.
- Quotas per continent (Africa, South America, Oceania explicit), per site
  type, per instrument type and rate, as §5.2 lists; OBS stays excluded.
- Source-flagged classes: a station registry with `hydrological`,
  `volcanic_tremor_hydrothermal`, `tectonic_tremor_lfe` intervals from the
  named catalogues; the polar rule is the only geographic flag now.
- A reviewed-negative path (`reviewed_negative`, `negative_support =
  reviewed`) with the reviewer and the review date in the manifest.
- Overlap checks against every evaluation and calibration interval, holdout
  lists, composition summaries (acceptance 42B).
- The legacy migration above, on the server, and `add_noise_to_manifests.py`
  reading `negative_support` instead of assuming every noise row is a
  negative.
- The `jma_wc` firing flag of §5.2, recorded for scoring only, never for
  selection.

## Validation

```bash
python -m pytest tests -q                                  # 110 passed in 4.6 s
python scripts/noise_ontology.py                           # prints the class table and the schema
python -m pytest tests/test_noise_ontology.py tests/test_build_noise_pool.py -q   # 36 passed
```

The demonstration needs the network; rerunning it with the same arguments
reads `data/noise_pools/raw/` if present and refetches otherwise.
