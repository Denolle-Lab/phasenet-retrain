"""Explicit waveform coordinates and a sample-aligned resampler for issue #34.

No dataset-name rate or channel-order defaults. Missing metadata is an error.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

CONTRACT_VERSION = "34a-v1"
METADATA_FIELDS = {
    "trace_name", "trace_sampling_rate_hz", "trace_dt_s", "sampling_rate",
    "trace_component_order", "component_order", "trace_dimension_order",
    "dimension_order", "trace_npts", "trace_start_time",
}


def present(value):
    return value is not None and not (isinstance(value, str) and not value.strip()) and not pd.isna(value)


def text_value(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def valid_rate(value, name="sampling rate"):
    if not present(value):
        raise ValueError(f"Missing {name}")
    rate = float(value)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid {name}: {value}")
    return rate


def metadata_rate(metadata):
    rates = [valid_rate(metadata[key]) for key in ("trace_sampling_rate_hz", "sampling_rate")
             if present(metadata.get(key))]
    if present(metadata.get("trace_dt_s")):
        rates.append(1 / valid_rate(metadata["trace_dt_s"], "sample interval"))
    if not rates:
        raise ValueError("Missing sampling rate; supply verified per-trace or HDF5 data_format metadata")
    if not all(np.isclose(rate, rates[0], rtol=1e-10, atol=0) for rate in rates):
        raise ValueError("Conflicting sampling-rate metadata")
    return rates[0]


def has_rate(metadata):
    return any(present(metadata.get(key)) for key in ("trace_sampling_rate_hz", "sampling_rate", "trace_dt_s"))


def metadata_field(layers, *keys):
    for layer in layers:
        values = [layer[key] for key in keys if present(layer.get(key))]
        if values:
            if len({text_value(value) for value in values}) != 1:
                raise ValueError(f"Conflicting metadata aliases: {keys}")
            return values[0]
    return None


def canonical_waveform(waveform, components, dimensions, npts=None):
    """Return ZNE/CW data with missing components zero-filled, never duplicated."""
    components = text_value(components) if present(components) else ""
    dimensions = text_value(dimensions) if present(dimensions) else ""
    if not components or len(set(components)) != len(components) or set(components) - set("ZNE"):
        raise ValueError("Explicit unique Z/N/E component metadata is required; orient 1/2 channels upstream")
    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim == 1 and dimensions == "W" and len(components) == 1:
        w = w[None, :]
    elif w.ndim == 2 and dimensions == "WC":
        w = w.T
    elif not (w.ndim == 2 and dimensions == "CW"):
        raise ValueError(f"Waveform shape {w.shape} disagrees with dimension order {dimensions!r}")
    if present(npts):
        count = float(npts)
        if not np.isfinite(count) or count != int(count) or not 0 < count <= w.shape[1]:
            raise ValueError("trace_npts is outside the stored trace support")
        w = w[:, :int(count)]
    if w.shape[0] != len(components) or w.shape[1] == 0 or not np.isfinite(w).all():
        raise ValueError("Invalid component count, empty waveform or non-finite samples")
    out = np.zeros((3, w.shape[1]), dtype=np.float32)
    mask = []
    for index, component in enumerate("ZNE"):
        mask.append(component in components)
        if component in components:
            out[index] = w[components.index(component)]
    return out, mask


def resample_waveform(waveform, source_rate, target_rate):
    """Polyphase FIR resampling with unchanged first-sample time and valid support.

    Kaiser beta=5, line boundary extension. The output is trimmed so no sample
    time exceeds the source's final sample. Rate/phase metadata are transported
    separately by the caller. Deployment adoption is checkpoint 34C.
    """
    source_rate, target_rate = valid_rate(source_rate), valid_rate(target_rate)
    if source_rate == target_rate:
        return waveform
    ratio = (Fraction(str(target_rate)) / Fraction(str(source_rate))).limit_denominator(10000)
    if not np.isclose(float(ratio), target_rate / source_rate, rtol=1e-10, atol=0):
        raise ValueError("Sampling-rate ratio cannot be represented accurately with bounded resampler size")
    if max(ratio.numerator, ratio.denominator) > 100000:
        raise ValueError("Sampling-rate ratio exceeds the supported resampler size")
    if waveform.shape[-1] < 2:
        raise ValueError("At least two samples are required for resampling")
    out = resample_poly(waveform, ratio.numerator, ratio.denominator, axis=-1,
                        window=("kaiser", 5.0), padtype="line")
    n_valid = int(np.floor((waveform.shape[-1] - 1) * target_rate / source_rate + 1e-9)) + 1
    return np.asarray(out[..., :n_valid], dtype=np.float32)


@dataclass
class TraceRecord:
    waveform: np.ndarray
    sampling_rate: float
    arrival_sampling_rate: float
    component_mask: list
    start_time: object = None


def read_hdf5_trace(handle, trace_name, metadata):
    """Read ordinary or SeisBench bucket-indexed records without eval()."""
    block, separator, location = str(trace_name).partition("$")
    dataset = handle["data"][block]
    if separator:
        selectors = []
        for field in location.split(","):
            parts = field.strip().split(":")
            if len(parts) == 1:
                selectors.append(int(parts[0]))
            elif len(parts) in (2, 3):
                selectors.append(slice(*(int(x) if x else None for x in parts)))
            else:
                raise ValueError("Invalid bucket slice")
        waveform = dataset[tuple(selectors)]
    else:
        waveform = dataset[()]
    defaults = {}
    if "data_format" in handle:
        defaults.update({key: value[()] for key, value in handle["data_format"].items()})
    # File data_format values are defaults; per-trace CSV/attributes can override.
    merged = dict(defaults)
    merged.update(dict(dataset.attrs))
    merged.update({key: value for key, value in metadata.items() if present(value)})
    attributes = dict(dataset.attrs)
    layers = [metadata, attributes, defaults]
    if has_rate(metadata) and has_rate(attributes):
        if not np.isclose(metadata_rate(metadata), metadata_rate(attributes), rtol=1e-10, atol=0):
            raise ValueError("CSV/HDF5 trace sampling-rate conflict")
    for keys in (("trace_component_order", "component_order"),
                 ("trace_dimension_order", "dimension_order")):
        csv_value = metadata_field([metadata], *keys)
        attr_value = metadata_field([attributes], *keys)
        if present(csv_value) and present(attr_value) and text_value(csv_value) != text_value(attr_value):
            raise ValueError(f"CSV/HDF5 trace conflict for {keys[0]}")
    if present(metadata.get("trace_npts")) and present(attributes.get("trace_npts")):
        if float(metadata["trace_npts"]) != float(attributes["trace_npts"]):
            raise ValueError("CSV/HDF5 trace_npts conflict")
    rate = next((metadata_rate(layer) for layer in layers if has_rate(layer)), None)
    rate = valid_rate(rate)
    components = metadata_field(layers, "trace_component_order", "component_order")
    dimensions = metadata_field(layers, "trace_dimension_order", "dimension_order")
    wave, mask = canonical_waveform(waveform, components, dimensions, merged.get("trace_npts"))
    return TraceRecord(wave, rate, rate, mask, merged.get("trace_start_time"))
