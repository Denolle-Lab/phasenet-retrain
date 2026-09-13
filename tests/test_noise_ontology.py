"""Checkpoint 42A: the noise ontology, feature rules, split and schema.

No network, no model: synthetic windows only. Base interpreter (numpy,
pandas, pytest).
"""
import pathlib
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import noise_ontology as no  # noqa: E402

RATE = 40.0
N = int(120 * RATE)
RNG = np.random.default_rng(42)


def _t():
    return np.arange(N) / RATE


def synth_microseism(seed=1):
    """Secondary microseism: a narrow band around 0.2 Hz over a weak white floor."""
    rng = np.random.default_rng(seed)
    t = _t()
    z = np.zeros(N)
    for f in np.linspace(0.12, 0.35, 12):
        z += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    z = 50 * z + 0.05 * rng.standard_normal(N)
    h = 0.7 * z + 0.05 * rng.standard_normal(N)
    return np.stack([z, h, h])


def synth_mains(seed=2, f0=60.0, rate=200.0):
    """Mains hum at f0 and 2 f0 over white noise, at a rate that resolves 120 Hz."""
    rng = np.random.default_rng(seed)
    n = int(120 * rate)
    t = np.arange(n) / rate
    z = rng.standard_normal(n) + 6 * np.sin(2 * np.pi * f0 * t) + 3 * np.sin(2 * np.pi * 2 * f0 * t)
    return np.stack([z, rng.standard_normal(n), rng.standard_normal(n)]), rate


def synth_impulsive(seed=3):
    """White noise with a single sharp spike: high kurtosis."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(N)
    z[N // 2] += 80.0
    return np.stack([z, rng.standard_normal(N), rng.standard_normal(N)])


def synth_quiet(seed=4):
    """Low-amplitude white noise, flat spectrum, no lines, no impulse."""
    rng = np.random.default_rng(seed)
    return 1e-3 * np.stack([rng.standard_normal(N) for _ in range(3)])


def synth_gap(seed=5):
    """White noise with 10 s of zeros on every channel (a gap filled by merge)."""
    x = synth_quiet(seed)
    x[:, 1000:1400] = 0.0
    return x


def synth_tilt(seed=6):
    """Wind/tilt: strong horizontal drift below 0.1 Hz, broadband otherwise."""
    rng = np.random.default_rng(seed)
    t = _t()
    z = rng.standard_normal(N)
    h = rng.standard_normal(N) + 200 * np.sin(2 * np.pi * 0.02 * t)
    return np.stack([z, h, h])


# ── feature rules ─────────────────────────────────────────────────────────────

def test_microseism_window_classified_as_ocean_microseism():
    f = no.spectral_features(synth_microseism(), RATE, hour_of_day=3)
    assert f["microseism_band_frac"] > 0.9
    c, conf = no.classify_features(f)
    assert c is no.NoiseClass.OCEAN_MICROSEISM
    assert 0.0 <= conf <= 1.0


def test_mains_hum_classified_as_cultural():
    x, rate = synth_mains()
    f = no.spectral_features(x, rate, hour_of_day=3)     # at night: only the mains rule can fire
    assert f["mains_hz"] == 60.0
    assert f["mains_line_frac"] >= no.THRESH["mains_line_frac"]
    c, _ = no.classify_features(f)
    assert c is no.NoiseClass.CULTURAL_DIURNAL


def test_mains_50hz_detected():
    x, rate = synth_mains(f0=50.0)
    f = no.spectral_features(x, rate, hour_of_day=3)
    assert f["mains_hz"] == 50.0
    assert no.classify_features(f)[0] is no.NoiseClass.CULTURAL_DIURNAL


def test_impulsive_window_classified_as_impulsive():
    f = no.spectral_features(synth_impulsive(), RATE, hour_of_day=12)
    assert f["kurtosis"] >= no.THRESH["kurtosis_impulsive"]
    c, _ = no.classify_features(f)
    assert c is no.NoiseClass.IMPULSIVE_NON_EARTHQUAKE
    assert c.category == "task_excluded"


def test_quiet_window_classified_as_quiet_baseline():
    f = no.spectral_features(synth_quiet(), RATE, hour_of_day=3)
    assert f["kurtosis"] < 1.0
    assert f["mains_line_frac"] < no.THRESH["mains_line_frac"]
    assert f["microseism_band_frac"] < no.THRESH["microseism_frac"]
    c, conf = no.classify_features(f)
    assert c is no.NoiseClass.QUIET_BASELINE
    assert conf == 0.2                                  # fallback, no station percentile
    f["station_rms_log10_p10"] = f["rms_log10"] + 1.0   # below the station's 10th percentile
    c, conf = no.classify_features(f)
    assert c is no.NoiseClass.QUIET_BASELINE and conf == 0.8


def test_gap_window_classified_as_instrument():
    f = no.spectral_features(synth_gap(), RATE)
    assert f["n_zero_frac"] == pytest.approx(400 / N)
    assert no.classify_features(f)[0] is no.NoiseClass.INSTRUMENT_TELEMETRY


def test_clipped_window_classified_as_instrument():
    x = synth_quiet()
    x = np.clip(x, -5e-4, 5e-4)
    f = no.spectral_features(x, RATE)
    assert f["clip_frac"] >= no.THRESH["clip_frac"]
    assert no.classify_features(f)[0] is no.NoiseClass.INSTRUMENT_TELEMETRY


def test_tilt_window_classified_as_wind_site_tilt():
    f = no.spectral_features(synth_tilt(), RATE, hour_of_day=3)
    assert f["hv_low_ratio"] >= no.THRESH["hv_low_ratio"]
    assert no.classify_features(f)[0] is no.NoiseClass.WIND_SITE_TILT


def test_instrument_rule_precedes_impulsive():
    """A spike inside a gappy window is an instrument window, not a blast."""
    x = synth_gap()
    x[0, N // 2] += 1.0
    f = no.spectral_features(x, RATE)
    assert f["kurtosis"] >= no.THRESH["kurtosis_impulsive"]
    assert no.classify_features(f)[0] is no.NoiseClass.INSTRUMENT_TELEMETRY


def test_classification_is_deterministic():
    f = no.spectral_features(synth_microseism(), RATE, hour_of_day=3)
    a = [no.classify_features(dict(f)) for _ in range(5)]
    assert len({(c, round(k, 12)) for c, k in a}) == 1


def test_features_accept_samples_by_channels_layout():
    x = synth_quiet()
    a = no.spectral_features(x, RATE)
    b = no.spectral_features(x.T, RATE)
    assert a["rms"] == pytest.approx(b["rms"])


def test_feature_columns_all_present():
    f = no.spectral_features(synth_quiet(), RATE, hour_of_day=7)
    assert set(no.FEATURE_COLUMNS) <= set(f)
    assert f["hour_of_day"] == 7.0


# ── ontology and negative_support ─────────────────────────────────────────────

def test_eleven_classes_and_categories():
    assert len(no.NoiseClass) == 11
    assert set(no.CLASS_SPECS) == {c.value for c in no.NoiseClass}
    for c in no.NoiseClass:
        s = c.spec
        assert s["definition"] and s["source_rule"]
        assert s["ontology_category"] in no.CATEGORIES
    assert no.NoiseClass.EARTHQUAKE_CODA_SEQUENCE_HUM.category == "unlabelled_interval"
    assert no.NoiseClass.TECTONIC_TREMOR_LFE.category == "unlabelled_interval"
    assert no.NoiseClass.IMPULSIVE_NON_EARTHQUAKE.category == "task_excluded"
    assert no.NoiseClass.VOLCANIC_TREMOR_HYDROTHERMAL.category == "task_excluded"
    assert no.NoiseClass.QUIET_BASELINE.category == "background"


def test_negative_support_mapping_per_category():
    # unlabelled_interval: unknown whatever the catalogue said
    assert no.negative_support_for("unlabelled_interval", event_free=True, local_catalogue=True,
                                   completeness_mag=1.0) == "unknown"
    # reviewed_negative: reviewed
    assert no.negative_support_for("reviewed_negative") == "reviewed"
    assert no.negative_support_for("background", reviewed=True) == "reviewed"
    # background / task_excluded: certified only with event-free + local catalogue + stated completeness
    for cat in ("background", "task_excluded"):
        assert no.negative_support_for(cat, event_free=True, local_catalogue=True, completeness_mag=2.0) == "certified"
        assert no.negative_support_for(cat, event_free=True, local_catalogue=False, completeness_mag=2.0) == "unknown"
        assert no.negative_support_for(cat, event_free=True, local_catalogue=True, completeness_mag=None) == "unknown"
        assert no.negative_support_for(cat, event_free=True, local_catalogue=True, completeness_mag=float("nan")) == "unknown"
        assert no.negative_support_for(cat, event_free=False, local_catalogue=True, completeness_mag=2.0) == "unknown"
    with pytest.raises(ValueError):
        no.negative_support_for("not_a_category")


def test_class_table_lists_every_class_once():
    rows = no.class_table()
    assert [r["noise_class"] for r in rows] == [c.value for c in no.NoiseClass]
    unk = {r["noise_class"] for r in rows if r["negative_support_if_event_free"] == "unknown"}
    assert unk == {"tectonic_tremor_lfe", "earthquake_coda_sequence_hum"}


# ── station split ─────────────────────────────────────────────────────────────

def test_station_split_deterministic_and_station_disjoint():
    stations = [f"XX.S{i:03d}" for i in range(500)]
    a = {s: no.station_split(s, 0.2, "42A") for s in stations}
    b = {s: no.station_split(s, 0.2, "42A") for s in reversed(stations)}
    assert a == b
    assert set(a.values()) <= set(no.SPLIT_VALUES)
    frac = sum(v == "holdout" for v in a.values()) / len(a)
    assert 0.12 < frac < 0.28
    # every window of a station lands in its station's split: the split is a pure function of the station
    df = pd.DataFrame({"station": np.repeat(stations[:20], 7)})
    df["split"] = [no.station_split(s, 0.2, "42A") for s in df["station"]]
    assert (df.groupby("station")["split"].nunique() == 1).all()
    # case and whitespace do not change the answer; the seed tag does
    assert no.station_split(" iu.anmo ", 0.2, "42A") == no.station_split("IU.ANMO", 0.2, "42A")
    diff = sum(no.station_split(s, 0.5, "42A") != no.station_split(s, 0.5, "other") for s in stations)
    assert diff > 100
    assert no.station_split("IU.ANMO", 0.0) == "train"
    assert no.station_split("IU.ANMO", 1.0) == "holdout"
    with pytest.raises(ValueError):
        no.station_split("IU.ANMO", 1.5)


# ── schema ────────────────────────────────────────────────────────────────────

REQUIRED = ["pool", "noise_class", "ontology_category", "negative_support", "network", "station", "location",
            "channel_band", "station_latitude_deg", "station_longitude_deg", "rate_hz", "start_time", "duration_s",
            "source", "catalogue_used", "completeness_mag", "exclusion_bundle_sha256", "split", "arrivals_json"]


def test_manifest_schema_complete():
    cols = set(no.MANIFEST_COLUMNS)
    assert set(REQUIRED) <= cols
    assert set(no.FEATURE_COLUMNS) <= cols
    assert len(no.MANIFEST_COLUMNS) == len(cols)          # no duplicates
    empty = no.empty_manifest()
    assert list(empty.columns) == list(no.MANIFEST_COLUMNS)
    no.check_manifest(empty)


def _row(**kw):
    r = {c: 0.0 for c in no.MANIFEST_COLUMNS}
    r.update(pool="p", noise_class="quiet_baseline", ontology_category="background", negative_support="unknown",
             arrivals_json="", network="XX", station="S1", location="", channel_band="BH", start_time="2019-03-12T00:00:00",
             source="fixture", catalogue_used="", exclusion_bundle_sha256="", split="train", event_free=True,
             independence_unverified=False, npz_path="", npz_index=0, class_source="features")
    r.update(kw)
    return r


def test_check_manifest_rejects_bad_vocabulary_and_mapping():
    good = pd.DataFrame([_row()])
    no.check_manifest(good)
    with pytest.raises(ValueError, match="negative_support"):
        no.check_manifest(pd.DataFrame([_row(negative_support="maybe")]))
    with pytest.raises(ValueError, match="arrivals_json"):
        no.check_manifest(pd.DataFrame([_row(arrivals_json='[{"phase": "P"}]')]))
    with pytest.raises(ValueError, match="unlabelled_interval"):
        no.check_manifest(pd.DataFrame([_row(noise_class="tectonic_tremor_lfe", ontology_category="unlabelled_interval",
                                             negative_support="certified")]))
    with pytest.raises(ValueError, match="missing columns"):
        no.check_manifest(good.drop(columns=["split"]))
    with pytest.raises(ValueError, match="unexpected"):
        no.check_manifest(good.assign(extra=1))


def test_station_references_make_microseism_and_quiet_relative():
    f = no.spectral_features(synth_microseism(), RATE, hour_of_day=3)
    assert np.isnan(f["station_rms_log10_p10"]) and np.isnan(f["station_secondary_p50"])
    assert no.classify_features(f)[0] is no.NoiseClass.OCEAN_MICROSEISM      # no reference: band dominance alone
    f["station_secondary_p50"] = f["secondary_band_power_log10"] + 0.5      # below the station median: not a storm
    c, conf = no.classify_features(f)
    assert c is no.NoiseClass.QUIET_BASELINE and conf == 0.2
    f["station_secondary_p50"] = f["secondary_band_power_log10"] - 0.5      # above the median: storm-modulated
    assert no.classify_features(f)[0] is no.NoiseClass.OCEAN_MICROSEISM
    f["station_rms_log10_p10"] = f["rms_log10"]                             # at the station's 10th percentile
    c, conf = no.classify_features(f)
    assert c is no.NoiseClass.QUIET_BASELINE and conf == 0.8                # quiet precedes microseism


def test_octave_flatness_white_vs_red():
    white = no.spectral_features(synth_quiet(), RATE)["spectral_flatness"]
    red = no.spectral_features(synth_microseism(), RATE)["spectral_flatness"]
    assert white > 0.8 and red < 0.3
