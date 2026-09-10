"""Read-only/synthetic probes for the surface-picker plan; no inference on real data.

Run with the agent-seisbench environment (numpy, pandas, torch, scipy, obspy).
Writes only probe_results.json beside this file; local catalogues are optional.
"""
import argparse
import hashlib
import inspect
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import seisbench
import seisbench.models as sbm
import torch
import obspy
from obspy import Stream, Trace
from scipy.stats import binomtest

from probe_support import date_summary, source_id

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--external-root", type=Path, default=ROOT.parent,
                    help="Parent directory of optional surface_events/QuakeScope/thunderquakes checkouts")
args = parser.parse_args()
SOURCE_ROOTS = [("repo", ROOT), ("package/seisbench", Path(seisbench.__file__).parent),
                ("package/obspy", Path(obspy.__file__).parent), ("sources", args.external_root)]
torch.set_num_threads(1)
torch.manual_seed(0)
result = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
          "seisbench": seisbench.__version__, "obspy": obspy.__version__,
          "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}

# Every valid 120 s crop of a 180 s record with onset at 70 s contains that onset.
starts = np.arange(9000 - 6000 + 1)
offsets = (3500 - starts) / 50
result["crop"] = {"minimum_onset_s": float(offsets.min()), "maximum_onset_s": float(offsets.max()),
                  "onset_free_fraction": float(np.mean((offsets < 0) | (offsets >= 120)))}
assert result["crop"]["onset_free_fraction"] == 0

# Existing make_labels uses N=1-max(P,S); overlapping positive targets exceed unit mass.
x = np.arange(6000)
u = np.exp(-0.5 * ((x - 3000) / 50) ** 2)
p = np.exp(-0.5 * ((x - 3025) / 50) ** 2)
result["softmax_target"] = {"maximum_sum": float((u + p + 1 - np.maximum(u, p)).max())}
assert result["softmax_target"]["maximum_sum"] > 1.8
raw_targets = np.stack([u, p, np.maximum(0, 1 - u - p)])
normalized_targets = raw_targets / raw_targets.sum(axis=0, keepdims=True)
assert np.allclose(normalized_targets.sum(axis=0), 1)
result["softmax_target"]["normalized_max_mass_error"] = float(
    np.abs(normalized_targets.sum(axis=0) - 1).max())
coincident_raw = np.array([1.0, 1.0, max(0, 1 - 1.0 - 1.0)])
result["softmax_target"]["coincident_upn_target"] = (coincident_raw / coincident_raw.sum()).tolist()
result["rounding_only_sigma_s"] = float(1 / np.sqrt(12))

model = sbm.VariableLengthPhaseNet(in_samples=6000, sampling_rate=50, phases="UPN", classes=3)
model.eval()
with torch.no_grad():
    shape = list(model(torch.randn(1, 3, 6000)).shape)
result["model"] = {"parameters": sum(p.numel() for p in model.parameters()), "output_shape": shape,
                   "default_norm": model.norm, "norm_axis": list(model.norm_axis),
                   "default_overlap": model._annotate_args["overlap"][1]}
assert shape == [1, 3, 6000]
try:
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / "surface"
        model.save(prefix)
        loaded = sbm.VariableLengthPhaseNet.load(prefix)
        result["model"]["roundtrip"] = {"success": True, "labels": loaded.labels,
                                          "norm": loaded.norm, "sampling_rate": loaded.sampling_rate}
except Exception as exc:
    result["model"]["roundtrip"] = {"success": False, "error": f"{type(exc).__name__}: {exc}"}

# Support in the convolutional graph, with preprocessing intentionally excluded.
# Positive weights/inputs keep all paths active. This is not a trained effective RF.
for module in model.modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        torch.nn.init.constant_(module.weight, 0.01)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
input_tensor = torch.ones(1, 3, 6000, requires_grad=True)
model(input_tensor, logits=True)[0, 0, 3000].backward()
support = torch.nonzero(input_tensor.grad.abs().sum((0, 1)) > 0).flatten()
result["convolution_support"] = {"first_sample": int(support.min()), "last_sample": int(support.max()),
    "span_samples": int(support.max() - support.min() + 1),
    "span_seconds_at_50hz": float((support.max() - support.min() + 1) / 50),
    "note": "Structural numerical support at center; not trained effective RF; window normalization excluded."}

# Measure installed integer-ratio resampler on tones away from transient boundaries.
tones = []
for source_rate, target_rate, frequency in [(100, 25, 10), (100, 25, 12), (100, 25, 15),
                                            (100, 50, 10), (100, 50, 20), (100, 50, 30),
                                            (40, 50, 10), (40, 50, 18),
                                            (40, 100, 10), (40, 100, 18),
                                            (20, 100, 5), (20, 100, 9)]:
    t = np.arange(source_rate * 120) / source_rate
    stream = Stream([Trace(np.sin(2 * np.pi * frequency * t), header={"sampling_rate": source_rate})])
    sbm.WaveformModel.resample(stream, target_rate)
    data = stream[0].data[int(10 * target_rate):int(110 * target_rate)]
    amplitude = np.sqrt(2) * np.sqrt(np.mean(data ** 2))
    tones.append({"source_hz": source_rate, "target_hz": target_rate, "tone_hz": frequency,
                  "output_amplitude_db": float(20 * np.log10(amplitude))})
result["resampler_tones"] = tones
for row in tones:
    if row["source_hz"] < row["target_hz"]:
        expected = 20 * np.log10(0.5 * (1 + np.cos(2 * np.pi * row["tone_hz"] / row["source_hz"])))
        row["hann_expected_db"] = float(expected)
        assert abs(expected - row["output_amplitude_db"]) < 0.01

result["recall_power_examples"] = []
for detected, count in [(10, 10), (20, 20), (50, 50), (8, 10), (16, 20), (40, 50)]:
    interval = binomtest(detected, count).proportion_ci(confidence_level=0.95, method="exact")
    result["recall_power_examples"].append({"detected": detected, "independent_events": count,
        "lower_95": interval.low, "upper_95": interval.high,
        "note": "Illustrative independent binomial events; not measured panel power or clustered uncertainty."})

external = args.external_root / "surface_events"
pick_path = external / "data/events/su_picks.txt"
if pick_path.exists():
    frame = pd.read_csv(pick_path, sep="|", skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    for col in frame.select_dtypes("object"):
        frame[col] = frame[col].str.strip()
    counts = frame.groupby("evid").size()
    result["pick_export"] = {"path": source_id(pick_path, SOURCE_ROOTS), "sha256": hashlib.sha256(pick_path.read_bytes()).hexdigest(),
        "rows": len(frame), "events": int(frame.evid.nunique()), "single_pick_events": int((counts == 1).sum()),
        "phase_counts": frame.iphase.value_counts().to_dict(),
        "quality_counts": {str(k): int(v) for k, v in frame.quality.value_counts().items()},
        **date_summary(frame.date)}
result["end_catalogues"] = {}
for path in sorted((external / "events").glob("Wes_Cat_*.csv")):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    result["end_catalogues"][path.name] = {"rows": len(frame), "columns": list(frame.columns),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nonempty_start_end": (int((frame["Time Start"].notna() & frame["Time End"].notna()).sum())
                               if {"Time Start", "Time End"} <= set(frame.columns) else None)}

result["source_hashes"] = {}
for path in [Path(inspect.getfile(sbm.VariableLengthPhaseNet)), Path(inspect.getfile(sbm.WaveformModel)),
             Path(inspect.getfile(Trace)), ROOT / "scripts/manifest_dataset.py", ROOT / "scripts/metrics.py",
             Path(__file__), Path(__file__).with_name("probe_support.py")]:
    result["source_hashes"][source_id(path, SOURCE_ROOTS)] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in [args.external_root / "QuakeScope/sb_catalog/src/picker.py",
             args.external_root / "QuakeScope/docs/quakexnet_generalization_plan.md",
             args.external_root / "thunderquakes/catalogs/pnwml_class_summary.csv"]:
    if path.exists():
        result["source_hashes"][source_id(path, SOURCE_ROOTS)] = hashlib.sha256(path.read_bytes()).hexdigest()
target = Path(__file__).with_name("probe_results.json")
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
