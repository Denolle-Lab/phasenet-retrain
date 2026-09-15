"""Offline audit probes; no training, catalogue harvest, or held-out model scoring.

Run with an environment containing numpy, pandas, torch, seisbench and obspy.
Writes only beside this file. Synthetic loader inputs intentionally exercise
native-rate and missing-label paths without needing the server datasets.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import Stream, Trace, UTCDateTime
import seisbench
import seisbench.models as sbm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import manifest_dataset as md
import heldout_testset_score as hs
import heldout_testset_registry as registry

OUT = Path(__file__).parent
torch.set_num_threads(2)
result = {"versions": {"python": sys.version, "torch": torch.__version__,
                       "seisbench": seisbench.__version__}}

# Execute the actual loader using a fake source whose native pick time is known.
rates = []
for rate in (20, 40, 50, 100):
    ds = object.__new__(md.ManifestDataset)
    ds.manifest = pd.DataFrame([dict(dataset_name="synthetic", trace_name="impulse",
        p_arrival_sample=4 * rate, s_arrival_sample=np.nan)])
    ds.window_len, ds.augment = 3001, False
    ds._chunked, ds._single_hdf5 = {}, {}
    pulse = np.exp(-((np.arange(60 * rate) / rate - 4) / 0.5) ** 2)
    wave = np.tile(pulse, (3, 1)).astype(np.float32)
    ds._fetch_sbd = lambda *args, w=wave, r=rate: (w.copy(), r)
    x, y = ds[0]
    peak, label = int(x[0].argmax()), int(y[0].argmax())
    rates.append(dict(native_rate=rate, waveform_peak=peak, label_peak=label,
                      label_minus_waveform_seconds=(label - peak) / 100))
result["native_rate_alignment"] = rates

# A single low-floor trigger can span two higher-threshold triggers.
p = np.zeros(1000, dtype=np.float32)
p[200:501] = 0.03
p[200], p[500] = 0.8, 0.7
st = Stream([Trace(p, header={"sampling_rate": 100, "starttime": UTCDateTime(0)})])
low = sbm.PhaseNet.picks_from_annotations(st, 0.02, "P")
high = sbm.PhaseNet.picks_from_annotations(st, 0.3, "P")
result["threshold_sweep"] = dict(low_floor_picks=len(low),
    filtered_low_at_0p3=int(sum(v.peak_value >= 0.3 for v in low)),
    extracted_at_0p3=len(high))
result["greedy_matching"] = dict(reference=[0.0, 0.6], candidate=[-0.4, 0.1],
    tolerance=0.5, matched=len(hs.match([0.0, 0.6], [-0.4, 0.1], tol=0.5)[0]),
    feasible_maximum=2)
result["overlapping_soft_targets"] = dict(
    max_channel_sum=float(md.make_labels(100, 100).sum(axis=0).max()),
    unknown_s_noise_target=float(md.make_labels(100, None)[2, 500]))

cache = Path.home() / ".seisbench/models/v3/phasenet"
meta = json.loads((cache / "jma_wc.json.v1").read_text())
parent_state = torch.load(cache / "jma_wc.pt.v1", map_location="cpu", weights_only=True)
ckpt = torch.load(ROOT / "models/jma_wc_ft_global_v7.pt", map_location="cpu", weights_only=True)
student = {k.removeprefix("model."): v for k, v in ckpt["model"].items() if k.startswith("model.")}
teacher = {k.removeprefix("teacher."): v for k, v in ckpt["model"].items() if k.startswith("teacher.")}
export = torch.load(cache / "quakescope2026.pt.v1", map_location="cpu", weights_only=True)
result["checkpoint"] = dict(epoch=ckpt.get("epoch"), tensors=len(student),
    teacher_tensors=len(teacher), teacher_exactly_parent=bool(teacher) and set(teacher)==set(parent_state)
        and all(torch.equal(v, parent_state[k]) for k, v in teacher.items()),
    cached_export_exactly_student=set(export)==set(student)
        and all(torch.equal(v, student[k]) for k,v in export.items()),
    model_args=meta["model_args"])
model = sbm.PhaseNet(**meta["model_args"])
model.load_state_dict(student, strict=True)
model.eval()
lengths = []
with torch.no_grad():
    for length in (3001, 6000, 6001):
        try:
            y = model(torch.zeros(1, 3, length))
            lengths.append(dict(input=length, output=list(y.shape)))
        except Exception as exc:
            lengths.append(dict(input=length, error=str(exc)))
result["forward_lengths"] = lengths
result["inference_window_contract"] = dict(in_samples=model.in_samples, pred_sample=model.pred_sample)
bn = []
for k,v in student.items():
    if k.endswith("running_var"):
        ratio = v / parent_state[k].clamp_min(1e-12)
        bn.append(dict(key=k, median_ratio=float(ratio.median()),
                       min_ratio=float(ratio.min()), max_ratio=float(ratio.max())))
result["batchnorm_variance_ratios"] = bn

# One deterministic development excerpt illustrates deployment trigger behaviour.
# This is a scoring-mechanics probe, not an estimate of population performance.
from obspy import read
dev_file = sorted((ROOT / "data/heldout_testset/adriatic_2022/waveforms").glob("*.mseed"))[0]
dev_stream = read(str(dev_file))
t0 = max(t.stats.starttime for t in dev_stream)
dev_stream.trim(t0, t0 + 600)
dev = []
for name, state in (("jma_wc", parent_state), ("v7", student)):
    model.load_state_dict(state, strict=True)
    model.default_args.update(meta["default_args"])
    model.eval()
    ann = model.annotate(dev_stream)
    for phase in ("P", "S"):
        a = ann.select(channel=f"PhaseNet_{phase}")
        lp = model.picks_from_annotations(a, 0.02, phase)
        hp = model.picks_from_annotations(a, 0.3, phase)
        dev.append(dict(model=name, phase=phase,
            low_floor_filtered_at_0p3=int(sum(p.peak_value >= 0.3 for p in lp)),
            extracted_at_0p3=len(hp),
            low_floor_times=[str(p.peak_time) for p in lp if p.peak_value >= 0.3],
            direct_times=[str(p.peak_time) for p in hp]))
result["development_excerpt"] = dict(file=str(dev_file.relative_to(ROOT)),
    start=str(t0), seconds=600, results=dev)

(OUT / "probe_results.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item()) + "\n")
print(json.dumps(result, indent=2, default=lambda x: x.item()))
