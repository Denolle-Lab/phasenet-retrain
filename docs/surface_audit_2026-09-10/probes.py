"""Read-only/synthetic probes for the surface-picker plan; no inference on real data.

Run with the agent-seisbench environment (numpy, pandas, torch, scipy, obspy).
Writes only probe_results.json beside this file; local catalogues are optional.
"""
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
from obspy import Stream, Trace

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
torch.manual_seed(0)
result = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
          "seisbench": seisbench.__version__, "numpy": np.__version__, "scipy": scipy.__version__}}

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
                                            (40, 50, 10), (40, 50, 18)]:
    t = np.arange(source_rate * 120) / source_rate
    stream = Stream([Trace(np.sin(2 * np.pi * frequency * t), header={"sampling_rate": source_rate})])
    sbm.WaveformModel.resample(stream, target_rate)
    data = stream[0].data[int(10 * target_rate):int(110 * target_rate)]
    amplitude = np.sqrt(2) * np.sqrt(np.mean(data ** 2))
    tones.append({"source_hz": source_rate, "target_hz": target_rate, "tone_hz": frequency,
                  "output_amplitude_db": float(20 * np.log10(amplitude))})
result["resampler_tones"] = tones

external = ROOT.parent / "surface_events"
pick_path = external / "data/events/su_picks.txt"
if pick_path.exists():
    frame = pd.read_csv(pick_path, sep="|", skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    for col in frame.select_dtypes("object"):
        frame[col] = frame[col].str.strip()
    counts = frame.groupby("evid").size()
    result["pick_export"] = {"path": str(pick_path), "sha256": hashlib.sha256(pick_path.read_bytes()).hexdigest(),
        "rows": len(frame), "events": int(frame.evid.nunique()), "single_pick_events": int((counts == 1).sum()),
        "phase_counts": frame.iphase.value_counts().to_dict(),
        "quality_counts": {str(k): int(v) for k, v in frame.quality.value_counts().items()},
        "pre_2002_picks": int((pd.to_datetime(frame.date).dt.year < 2002).sum()),
        "fractional_timestamp_rows": int(frame.date.str.contains(r"\.").sum())}
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
             ROOT / "scripts/manifest_dataset.py", ROOT / "scripts/metrics.py", Path(__file__)]:
    result["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in [ROOT.parent / "QuakeScope/sb_catalog/src/picker.py",
             ROOT.parent / "QuakeScope/docs/quakexnet_generalization_plan.md",
             ROOT.parent / "thunderquakes/catalogs/pnwml_class_summary.csv"]:
    if path.exists():
        result["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
target = Path(__file__).with_name("probe_results.json")
target.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
