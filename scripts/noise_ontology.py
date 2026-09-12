#!/usr/bin/env python3
"""
noise_ontology.py

Checkpoint 42A of issue #42: the noise ontology the pools carry, the
deterministic feature rules that assign a class when no source flag does, the
station-disjoint hold-out split, and the pool manifest schema.

Two orthogonal axes, and the whole point of 42A is that they are orthogonal.

  * CLASS: the eleven environmental/site flavours of
    `docs/2026-09-07_training_plan.md` §5.1 (the ocean-bottom flavour is
    defined for the OBS round and not harvested here). A class says what the
    ground and the instrument were doing. It never by itself says the window
    is a legitimate all-N training target.
  * ONTOLOGY CATEGORY: `background`, `task_excluded`, `unlabelled_interval`,
    `reviewed_negative` (`docs/2026-09-10_picker_and_issue_roadmap_audit.md`,
    #42 row). The category says what the window may be used for, and maps to
    the manifest's `negative_support`.

The audit's rule, restated: aftershock coda and LFEs are NOT negatives by
catalogue absence. A window whose class is `earthquake_coda_sequence_hum` or
`tectonic_tremor_lfe` is an `unlabelled_interval` and carries
`negative_support = unknown` however clean the catalogue query looked,
because the catalogue is not complete for the arrivals those intervals hold.
`impulsive_non_earthquake` (quarry blasts, surface events) is
`task_excluded`: those sources can carry real P and S, so the general picker
is allowed to pick them and only the source classifier rejects them (#41A);
they are certified event-free of *catalogued tectonic* events, never labelled
"no arrival here".

negative_support vocabulary (defined on another branch; the strings are used
verbatim and nothing is imported from there):
    certified  catalogue-complete window certified event-free
    reviewed   a reviewed negative window
    unknown    everything else
`arrivals_json` is empty for every noise row and provenance tiers are not
used for noise.

Nothing here runs a model on the data, by design (§5.2: model-based screening
would bias the pool toward what the model already ignores).
"""

from __future__ import annotations

import hashlib
import math
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

# ── ontology categories and their negative_support mapping ────────────────────

NEGATIVE_SUPPORT_VALUES = ("certified", "reviewed", "unknown")

CATEGORIES = ("background", "task_excluded", "unlabelled_interval", "reviewed_negative")

CATEGORY_DEFINITION = {
    "background": (
        "Ambient ground and site noise with no source event of interest. "
        "Eligible to be a certified negative when the event-free rule passes "
        "against a local catalogue of stated completeness."),
    "task_excluded": (
        "A real source the picker's task excludes (blasts, surface events, "
        "volcanic and hydrothermal processes). It may carry genuine arrivals; "
        "rejection belongs to the source classifier (#41A), not to an all-N "
        "label. Eligible to be certified event-free of catalogued tectonic "
        "events, which is a weaker statement than 'no arrival'."),
    "unlabelled_interval": (
        "An interval that very probably contains uncatalogued arrivals: "
        "aftershock coda and sequence hum, tectonic tremor and LFEs. Never a "
        "negative. Always negative_support = unknown, whatever the catalogue "
        "query returned, because no catalogue is complete for these."),
    "reviewed_negative": (
        "A window a human reviewed and recorded as arrival-free. The only "
        "source of negative_support = reviewed."),
}

# background and task_excluded are the only categories whose support can be
# upgraded by the event-free rule; the upgrade additionally needs a local
# catalogue with a stated completeness magnitude.
CERTIFIABLE_CATEGORIES = frozenset({"background", "task_excluded"})


def negative_support_for(category: str, *, event_free: bool = False,
                         local_catalogue: bool = False,
                         completeness_mag: Optional[float] = None,
                         reviewed: bool = False) -> str:
    """The manifest `negative_support` value for a window.

    reviewed_negative -> "reviewed" (also whenever `reviewed` is set).
    unlabelled_interval -> "unknown", always.
    background / task_excluded -> "certified" when the event-free rule passed
    against a local catalogue whose completeness magnitude is stated, else
    "unknown".
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown ontology category {category!r}; expected one of {CATEGORIES}")
    if reviewed or category == "reviewed_negative":
        return "reviewed"
    if category == "unlabelled_interval":
        return "unknown"
    if category in CERTIFIABLE_CATEGORIES:
        if event_free and local_catalogue and completeness_mag is not None \
                and not (isinstance(completeness_mag, float) and math.isnan(completeness_mag)):
            return "certified"
        return "unknown"
    return "unknown"


# ── the eleven classes ────────────────────────────────────────────────────────

class NoiseClass(Enum):
    """The eleven flavours of `docs/2026-09-07_training_plan.md` §5.1.

    The value is the manifest string. `.spec` holds definition, source rule,
    feature rule and ontology category.
    """

    OCEAN_MICROSEISM = "ocean_microseism"
    WIND_SITE_TILT = "wind_site_tilt"
    CULTURAL_DIURNAL = "cultural_diurnal"
    HYDROLOGICAL = "hydrological"
    IMPULSIVE_NON_EARTHQUAKE = "impulsive_non_earthquake"
    VOLCANIC_TREMOR_HYDROTHERMAL = "volcanic_tremor_hydrothermal"
    TECTONIC_TREMOR_LFE = "tectonic_tremor_lfe"
    EARTHQUAKE_CODA_SEQUENCE_HUM = "earthquake_coda_sequence_hum"
    POLAR_ICE = "polar_ice"
    INSTRUMENT_TELEMETRY = "instrument_telemetry"
    QUIET_BASELINE = "quiet_baseline"

    @property
    def spec(self) -> dict:
        return CLASS_SPECS[self.value]

    @property
    def category(self) -> str:
        return CLASS_SPECS[self.value]["ontology_category"]

    @property
    def feature_labelled(self) -> bool:
        """True when `classify_features` can produce this class."""
        return CLASS_SPECS[self.value]["feature_rule"] is not None


CLASS_SPECS: Dict[str, dict] = {
    "ocean_microseism": dict(
        definition=("Storm-modulated ocean loading: a primary peak at 0.05-0.1 Hz and a "
                    "secondary peak at 0.1-0.5 Hz, modulated over weeks; dominates island "
                    "and coastal stations."),
        source_rule=("Coastal and island stations of the campaign (EarthScope, SCEDC/NCEDC S3, "
                     "operator FDSN). Sampled by the station's own 0.1-0.5 Hz power percentile "
                     "so storm periods are over-represented."),
        feature_rule=("microseism_band_frac >= 0.5 of total band power, the 0.05-0.5 Hz band "
                      "power exceeds the high-frequency band power, and, when the station "
                      "reference is present, secondary_band_power_log10 >= the station's median "
                      "(storm-modulated: elevated against the station's own distribution)."),
        ontology_category="background",
    ),
    "wind_site_tilt": dict(
        definition=("Wind loading of the vault and the surface: broadband 1-10 Hz gusts plus "
                    "horizontal tilt below 0.1 Hz on shallow installations."),
        source_rule=("Exposed and high-elevation stations; station metadata (vault type, "
                     "elevation) and, when available, co-located wind records."),
        feature_rule=("hv_low_ratio >= 3.0 (horizontal-to-vertical below 0.1 Hz) and "
                      "spectral_flatness >= 0.3 (octave-band flatness, 0.05 Hz to Nyquist)."),
        ontology_category="background",
    ),
    "cultural_diurnal": dict(
        definition=("Anthropogenic noise with a diurnal cycle: traffic, machinery, trains, "
                    "pumps, HVAC; mains at 50 or 60 Hz and harmonics; wind turbines at 1-5 Hz."),
        source_rule=("Urban and industrial stations, day and night sampled separately by hour "
                     "of day."),
        feature_rule=("mains_line_frac >= 0.05 (power at 50/60 Hz and harmonics over total), "
                      "or hour_of_day in the local working day 06-20 with "
                      "high_freq_frac >= 0.5 and spectral_flatness < 0.3."),
        ontology_category="background",
    ),
    "hydrological": dict(
        definition="Rivers, waterfalls, rain on the enclosure, snowmelt.",
        source_rule=("Stations near rivers and glaciers (station registry proximity flag), "
                     "spring and monsoon months."),
        feature_rule=None,   # not separable from wind/cultural by these features alone
        ontology_category="background",
    ),
    "impulsive_non_earthquake": dict(
        definition=("Thunder, sonic booms, explosions and quarry blasts, rockfalls and "
                    "avalanches, plane and vehicle impacts."),
        source_rule=("PNW exotic classes (Ni et al. 2023), operator 'explosion' and 'landslide' "
                     "labels, VCSEIS long-period class."),
        feature_rule="kurtosis >= 8.0 on the vertical component (impulsiveness).",
        ontology_category="task_excluded",
    ),
    "volcanic_tremor_hydrothermal": dict(
        definition=("Harmonic and spasmodic volcanic tremor, hydrothermal boiling noise, "
                    "gas-piston events."),
        source_rule=("INGV-OE, INGV-OV, HVO, IMO and AVO station-days inside declared "
                     "eruption or unrest periods, outside the held-out places of "
                     "`heldout_sequences.WINDOWS` (etna, campi_flegrei, la_palma, "
                     "reykjanes_peninsula, santorini_amorgos are all-time place hold-outs)."),
        feature_rule=None,   # source only: tremor spectra overlap wind and cultural
        ontology_category="task_excluded",
    ),
    "tectonic_tremor_lfe": dict(
        definition="Cascadia ETS and Nankai tremor bursts and their low-frequency earthquakes.",
        source_rule="PNSN tremor catalogue windows; Hi-net where accessible.",
        feature_rule=None,
        ontology_category="unlabelled_interval",
    ),
    "earthquake_coda_sequence_hum": dict(
        definition=("Regional coda minutes after M5+, teleseismic coda hours after M7+, and "
                    "the continuous overlap of small aftershocks."),
        source_rule=("Continuous data from aftershock sequences that are NOT held out "
                     "(Monte Cristo 2020, Sparta 2020, Zagreb 2020; Ridgecrest 2019 is held "
                     "out). Also assigned by the harvest when a catalogued mainshock arrival "
                     "precedes the window by less than the coda span."),
        feature_rule=None,
        ontology_category="unlabelled_interval",
    ),
    "polar_ice": dict(
        definition="Icequakes, calving, sea-ice noise, wind over ice.",
        source_rule="Antarctic and Greenland stations of the campaign (|latitude| >= 60 deg).",
        feature_rule=None,   # station-geography rule, applied by the harvest, not by features
        ontology_category="background",
    ),
    "instrument_telemetry": dict(
        definition=("Spikes, DC steps, mass recentring and calibration pulses, clipping, gaps, "
                    "dropouts, timing glitches, decimation aliasing, sensor self-noise floors, "
                    "temperature drift."),
        source_rule=("Station-day QC flags and the miniSEED data-quality bits; plus rules-based "
                     "synthesis on the fly in the mixing recipe (§5.3)."),
        feature_rule=("n_zero_frac >= 0.05 (gaps or dropouts filled with zeros) or "
                      "clip_frac >= 0.01 (samples at the digitiser rail)."),
        ontology_category="background",
    ),
    "quiet_baseline": dict(
        definition="The station at its quietest, all instrument types and rates.",
        source_rule="Every station in the campaign, lowest 10% power windows.",
        feature_rule=("rms_log10 <= the station's 10th percentile (station_rms_log10_p10); "
                      "also the fallback when no rule fires, at confidence 0.2."),
        ontology_category="background",
    ),
}

assert set(CLASS_SPECS) == {c.value for c in NoiseClass}
assert all(s["ontology_category"] in CATEGORIES for s in CLASS_SPECS.values())


def class_table() -> "list":
    """Rows (class, category, negative_support_when_event_free, definition) for the doc."""
    rows = []
    for c in NoiseClass:
        s = c.spec
        rows.append(dict(
            noise_class=c.value,
            ontology_category=s["ontology_category"],
            negative_support_if_event_free=negative_support_for(
                s["ontology_category"], event_free=True, local_catalogue=True, completeness_mag=0.0),
            negative_support_default=negative_support_for(s["ontology_category"]),
            feature_labelled=s["feature_rule"] is not None,
            definition=s["definition"],
        ))
    return rows


# ── spectral features ─────────────────────────────────────────────────────────

FEATURE_COLUMNS = (
    "rms",
    "rms_log10",
    "microseism_band_frac",   # 0.05-0.5 Hz power / total power
    "primary_band_frac",      # 0.05-0.1 Hz
    "secondary_band_frac",    # 0.1-0.5 Hz
    "secondary_band_power_log10",  # log10 of the 0.1-0.5 Hz band power, raw units (station-relative use only)
    "high_freq_frac",         # 1 Hz .. Nyquist
    "mains_line_frac",        # 50/60 Hz and harmonics, narrow lines / total
    "mains_hz",               # 50 or 60, whichever carried more line power
    "hv_low_ratio",           # horizontal / vertical rms below 0.1 Hz
    "spectral_flatness",      # geometric / arithmetic mean over octave bands, 0.05 Hz to Nyquist
    "kurtosis",               # vertical component, Fisher (0 for a Gaussian)
    "n_zero_frac",            # samples exactly zero (gaps filled with zeros)
    "clip_frac",              # samples within 1e-6 of the trace extremum
    "hour_of_day",            # UTC hour of the window start
    "station_rms_log10_p10",  # station reference: 10th percentile of rms_log10 over the harvest (NaN if none)
    "station_secondary_p50",  # station reference: median secondary_band_power_log10 over the harvest (NaN if none)
)

STATION_REFERENCE_COLUMNS = ("station_rms_log10_p10", "station_secondary_p50")

# Band edges, Hz. Fixed here so the doc, the harvest and the tests agree.
BAND_PRIMARY = (0.05, 0.1)
BAND_SECONDARY = (0.1, 0.5)
BAND_MICROSEISM = (0.05, 0.5)
BAND_HIGH = 1.0            # to Nyquist
MAINS_CANDIDATES = (50.0, 60.0)
MAINS_HARMONICS = 3        # f0, 2 f0, 3 f0
MAINS_HALFWIDTH_HZ = 0.5

# classify_features thresholds. One place, so the tests and the doc cite the
# same numbers.
THRESH = dict(
    microseism_frac=0.5,
    mains_line_frac=0.05,
    hv_low_ratio=3.0,
    spectral_flatness_flat=0.3,
    kurtosis_impulsive=8.0,
    high_freq_frac_cultural=0.5,
    n_zero_frac=0.05,
    clip_frac=0.01,
    quiet_rms_percentile=10.0,
    work_hours=(6, 20),
)


def _band_power(freqs: np.ndarray, psd: np.ndarray, f0: float, f1: float) -> float:
    m = (freqs >= f0) & (freqs < f1)
    return float(psd[m].sum())


def spectral_features(data: np.ndarray, rate_hz: float, *, hour_of_day: int = 0,
                      vertical_index: int = 0,
                      station_rms_log10_p10: Optional[float] = None,
                      station_secondary_p50: Optional[float] = None) -> Dict[str, float]:
    """Features of one window.

    `data` is (n_channels, n_samples) or (n_samples,). Channel `vertical_index`
    is the vertical; any others are treated as horizontals for `hv_low_ratio`.
    The two station references are NaN unless given; the harvest fills them
    from the station's own distribution after a first pass over its windows.
    Deterministic: no random state, no model.
    """
    arr = np.atleast_2d(np.asarray(data, dtype=float))
    if arr.shape[0] > arr.shape[1]:      # given as (n_samples, n_channels)
        arr = arr.T
    nch, n = arr.shape
    if n < 8:
        raise ValueError(f"window too short for features: {n} samples")
    rate_hz = float(rate_hz)
    vi = min(int(vertical_index), nch - 1)
    z = arr[vi]

    zd = z - z.mean()
    rms = float(np.sqrt(np.mean(zd ** 2)))
    # Welch-free: a single periodogram with a Hann taper. The pool's features
    # only need band ratios, and one taper keeps the rule exactly reproducible.
    win = np.hanning(n)
    scale = float((win ** 2).sum()) * rate_hz
    spec = np.abs(np.fft.rfft(zd * win)) ** 2
    psd = spec / max(scale, 1e-30)
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    nyq = rate_hz / 2.0
    total = float(psd[1:].sum()) or 1e-30

    prim = _band_power(freqs, psd, *BAND_PRIMARY) / total
    sec = _band_power(freqs, psd, *BAND_SECONDARY) / total
    micro = _band_power(freqs, psd, *BAND_MICROSEISM) / total
    high = _band_power(freqs, psd, BAND_HIGH, nyq + 1.0) / total

    # mains lines: narrow bands at f0 and harmonics, minus a local baseline
    best_f0, best_frac = float("nan"), 0.0
    for f0 in MAINS_CANDIDATES:
        if f0 > nyq:
            continue
        p = 0.0
        for k in range(1, MAINS_HARMONICS + 1):
            fk = f0 * k
            if fk > nyq:
                break
            m = np.abs(freqs - fk) <= MAINS_HALFWIDTH_HZ
            if not m.any():
                continue
            side = (np.abs(freqs - fk) > MAINS_HALFWIDTH_HZ) & (np.abs(freqs - fk) <= 4 * MAINS_HALFWIDTH_HZ)
            base = float(np.median(psd[side])) * int(m.sum()) if side.any() else 0.0
            p += max(float(psd[m].sum()) - base, 0.0)
        frac = p / total
        if frac > best_frac or math.isnan(best_f0):
            best_f0, best_frac = f0, frac

    # horizontal-to-vertical below 0.1 Hz, from band rms
    def _low_rms(x: np.ndarray) -> float:
        xd = x - x.mean()
        s = np.abs(np.fft.rfft(xd * win)) ** 2 / max(scale, 1e-30)
        return float(np.sqrt(max(_band_power(freqs, s, 0.0, BAND_PRIMARY[1]), 0.0)))

    vz = _low_rms(z)
    horiz = [arr[i] for i in range(nch) if i != vi]
    if horiz and vz > 0:
        hv = float(np.sqrt(np.mean([_low_rms(h) ** 2 for h in horiz])) / vz)
    else:
        hv = float("nan")

    # spectral flatness over octave bands from 0.05 Hz to Nyquist: equal weight per
    # octave, so the many high-frequency bins of a red broadband spectrum do not
    # dominate; 1 for white noise, small for a steeply red spectrum
    edges = 0.05 * 2.0 ** np.arange(0, int(np.floor(np.log2(nyq / 0.05))) + 2)
    edges = edges[edges <= nyq * (1 + 1e-9)]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            bands.append(float(psd[m].mean()))
    bands = np.array([b for b in bands if b > 0])
    flat = float(np.exp(np.mean(np.log(bands))) / np.mean(bands)) if bands.size >= 2 else float("nan")
    sec_power = _band_power(freqs, psd, *BAND_SECONDARY)

    v = zd
    sd = v.std()
    kurt = float(np.mean((v / sd) ** 4) - 3.0) if sd > 0 else float("nan")

    zero_frac = float(np.mean(np.all(arr == 0.0, axis=0)))
    amax = float(np.max(np.abs(arr))) or 1.0
    clip = float(np.mean(np.abs(arr) >= amax * (1 - 1e-6)))

    f = dict(
        rms=rms,
        rms_log10=float(np.log10(rms)) if rms > 0 else float("nan"),
        microseism_band_frac=micro,
        primary_band_frac=prim,
        secondary_band_frac=sec,
        secondary_band_power_log10=float(np.log10(sec_power)) if sec_power > 0 else float("nan"),
        high_freq_frac=high,
        mains_line_frac=best_frac,
        mains_hz=best_f0,
        hv_low_ratio=hv,
        spectral_flatness=flat,
        kurtosis=kurt,
        n_zero_frac=zero_frac,
        clip_frac=clip,
        hour_of_day=float(int(hour_of_day)),
        station_rms_log10_p10=float("nan") if station_rms_log10_p10 is None else float(station_rms_log10_p10),
        station_secondary_p50=float("nan") if station_secondary_p50 is None else float(station_secondary_p50),
    )
    return f


def _g(features: Dict[str, float], key: str, default: float = float("nan")) -> float:
    v = features.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def classify_features(features: dict) -> Tuple[NoiseClass, float]:
    """Assign a class from features alone. Deterministic; first rule wins.

    Order, most specific first:
      1 instrument_telemetry   gaps/dropouts or clipping
      2 impulsive_non_earthquake  kurtosis
      3 cultural_diurnal       mains lines
      4 quiet_baseline         rms at or below the station's 10th percentile (needs the station reference)
      5 ocean_microseism       0.05-0.5 Hz dominance, and above the station's median
                               secondary-band power when the station reference is present
      6 wind_site_tilt         low-frequency H/V with a flat spectrum
      7 cultural_diurnal       working hours, high-frequency, peaked spectrum
      8 quiet_baseline         fallback

    The two station references make the microseism and quiet labels relative
    to the station's own distribution, as §5.1 defines them (storms
    over-represented by the station's 0.1-0.5 Hz power percentile; quiet
    baseline the lowest 10% power windows). Without a reference (a single
    window) the microseism rule fires on band dominance alone.

    Returns (class, confidence in [0, 1]). Confidence is how far past its
    threshold the firing rule went, capped at 1; the fallback returns 0.2.
    """
    t = THRESH
    zero = _g(features, "n_zero_frac", 0.0)
    clip = _g(features, "clip_frac", 0.0)
    kurt = _g(features, "kurtosis", 0.0)
    mains = _g(features, "mains_line_frac", 0.0)
    micro = _g(features, "microseism_band_frac", 0.0)
    high = _g(features, "high_freq_frac", 0.0)
    hv = _g(features, "hv_low_ratio", 0.0)
    flat = _g(features, "spectral_flatness", 0.0)
    hour = _g(features, "hour_of_day", -1.0)
    rms10 = _g(features, "rms_log10")
    p10 = _g(features, "station_rms_log10_p10")
    sec_pow = _g(features, "secondary_band_power_log10")
    sec_p50 = _g(features, "station_secondary_p50")

    def conf(value: float, thresh: float, span: float) -> float:
        if not np.isfinite(value) or span <= 0:
            return 0.5
        return float(min(1.0, max(0.0, (value - thresh) / span + 0.5)))

    if (np.isfinite(zero) and zero >= t["n_zero_frac"]) or (np.isfinite(clip) and clip >= t["clip_frac"]):
        c = max(conf(zero, t["n_zero_frac"], 0.2), conf(clip, t["clip_frac"], 0.05))
        return NoiseClass.INSTRUMENT_TELEMETRY, c
    if np.isfinite(kurt) and kurt >= t["kurtosis_impulsive"]:
        return NoiseClass.IMPULSIVE_NON_EARTHQUAKE, conf(kurt, t["kurtosis_impulsive"], 20.0)
    if np.isfinite(mains) and mains >= t["mains_line_frac"]:
        return NoiseClass.CULTURAL_DIURNAL, conf(mains, t["mains_line_frac"], 0.2)
    if np.isfinite(rms10) and np.isfinite(p10) and rms10 <= p10:
        return NoiseClass.QUIET_BASELINE, 0.8
    if np.isfinite(micro) and micro >= t["microseism_frac"] and micro > high:
        elevated = (not np.isfinite(sec_p50)) or (np.isfinite(sec_pow) and sec_pow >= sec_p50)
        if elevated:
            return NoiseClass.OCEAN_MICROSEISM, conf(micro, t["microseism_frac"], 0.4)
    if np.isfinite(hv) and hv >= t["hv_low_ratio"] and np.isfinite(flat) \
            and flat >= t["spectral_flatness_flat"]:
        return NoiseClass.WIND_SITE_TILT, conf(hv, t["hv_low_ratio"], 4.0)
    lo, hi = t["work_hours"]
    if lo <= hour <= hi and np.isfinite(high) and high >= t["high_freq_frac_cultural"] \
            and np.isfinite(flat) and flat < t["spectral_flatness_flat"]:
        return NoiseClass.CULTURAL_DIURNAL, conf(high, t["high_freq_frac_cultural"], 0.4)
    return NoiseClass.QUIET_BASELINE, 0.2


# ── station-disjoint hold-out ────────────────────────────────────────────────

SPLIT_VALUES = ("train", "holdout")


def station_split(station: str, fraction: float = 0.2, seed_tag: str = "42A") -> str:
    """'holdout' or 'train' for a station, by hash.

    The split is by STATION, never by window (§5.2), so the false-pick rate is
    measured on stations the model never saw. Deterministic given
    (station, seed_tag): the same station always lands in the same split
    whatever the order of the harvest or the composition of the pool.
    `station` is the full identifier the manifest carries, normally
    "NET.STA"; it is upper-cased and stripped before hashing.
    """
    if not (0.0 <= float(fraction) <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")
    key = f"{seed_tag}|{str(station).strip().upper()}".encode()
    h = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    u = h / float(1 << 64)          # in [0, 1)
    return "holdout" if u < float(fraction) else "train"


# ── the pool manifest schema ─────────────────────────────────────────────────

MANIFEST_COLUMNS = (
    # identity of the pool and the label
    "pool",
    "noise_class",
    "ontology_category",
    "negative_support",
    "arrivals_json",
    # station
    "network",
    "station",
    "location",
    "channel_band",
    "station_latitude_deg",
    "station_longitude_deg",
    # window
    "rate_hz",
    "start_time",
    "duration_s",
    "npz_path",
    "npz_index",
    # provenance of the label
    "source",
    "class_source",
    "class_confidence",
    "catalogue_used",
    "completeness_mag",
    "event_free",
    "nearest_arrival_s",
    # exclusions and split
    "exclusion_bundle_sha256",
    "independence_unverified",
    "split",
) + FEATURE_COLUMNS

# dtypes for an empty manifest, so a zero-row pool still round-trips parquet
_STR_COLS = frozenset({
    "pool", "noise_class", "ontology_category", "negative_support", "arrivals_json",
    "network", "station", "location", "channel_band", "start_time", "npz_path",
    "source", "class_source", "catalogue_used", "exclusion_bundle_sha256", "split",
})
_BOOL_COLS = frozenset({"event_free", "independence_unverified"})
_INT_COLS = frozenset({"npz_index"})


def empty_manifest():
    """An empty DataFrame with the full schema and stable dtypes."""
    import pandas as pd
    cols = {}
    for c in MANIFEST_COLUMNS:
        if c in _STR_COLS:
            cols[c] = pd.Series([], dtype=object)
        elif c in _BOOL_COLS:
            cols[c] = pd.Series([], dtype=bool)
        elif c in _INT_COLS:
            cols[c] = pd.Series([], dtype="int64")
        else:
            cols[c] = pd.Series([], dtype=float)
    return pd.DataFrame(cols)


def check_manifest(df) -> None:
    """Raise if the frame is not a valid pool manifest."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"pool manifest missing columns: {missing}")
    extra = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    if extra:
        raise ValueError(f"pool manifest has unexpected columns: {extra}")
    if len(df) == 0:
        return
    bad = sorted(set(df["negative_support"].astype(str)) - set(NEGATIVE_SUPPORT_VALUES))
    if bad:
        raise ValueError(f"negative_support values not in the vocabulary: {bad}")
    bad = sorted(set(df["ontology_category"].astype(str)) - set(CATEGORIES))
    if bad:
        raise ValueError(f"ontology_category values not in the vocabulary: {bad}")
    bad = sorted(set(df["noise_class"].astype(str)) - {c.value for c in NoiseClass})
    if bad:
        raise ValueError(f"noise_class values not in the vocabulary: {bad}")
    bad = sorted(set(df["split"].astype(str)) - set(SPLIT_VALUES))
    if bad:
        raise ValueError(f"split values not in the vocabulary: {bad}")
    nonempty = df["arrivals_json"].astype(str).str.strip()
    nonempty = nonempty[~nonempty.isin(("", "nan", "None"))]
    if len(nonempty):
        raise ValueError(f"arrivals_json must be empty for noise rows; {len(nonempty)} rows are not")
    # the category -> support mapping, re-checked on the written rows
    for cat, grp in df.groupby("ontology_category"):
        if cat == "unlabelled_interval" and set(grp["negative_support"]) - {"unknown"}:
            raise ValueError("unlabelled_interval rows must all carry negative_support=unknown")
        if cat == "reviewed_negative" and set(grp["negative_support"]) - {"reviewed"}:
            raise ValueError("reviewed_negative rows must all carry negative_support=reviewed")
        if cat in CERTIFIABLE_CATEGORIES:
            bad = set(grp["negative_support"]) - {"certified", "unknown"}
            if bad:
                raise ValueError(f"{cat} rows carry negative_support {sorted(bad)}")


def describe() -> str:
    """Human-readable ontology dump, used by `python scripts/noise_ontology.py`."""
    out = ["class | category | support if event-free | feature-labelled",
           "---|---|---|---"]
    for r in class_table():
        out.append(f"{r['noise_class']} | {r['ontology_category']} | "
                   f"{r['negative_support_if_event_free']} | {'yes' if r['feature_labelled'] else 'no'}")
    out.append("")
    out.append(f"manifest columns ({len(MANIFEST_COLUMNS)}): " + ", ".join(MANIFEST_COLUMNS))
    return "\n".join(out)


if __name__ == "__main__":
    print(describe())
