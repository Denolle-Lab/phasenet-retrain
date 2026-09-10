"""Offline #34A fixtures: source timestamps are independent of manifest offsets."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import seisbench.data as sbd
except ImportError as exc:
    raise unittest.SkipTest(f"Loader fixtures require torch, SeisBench, h5py: {exc}")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import manifest_dataset as md
from waveform_contract import canonical_waveform, metadata_rate, resample_waveform
from fast_manifest_dataset import CachedManifestDataset
from build_training_dataset import process_dataset


class LoaderContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.datasets = []

    def source(self, rate=100, components="ENZ", dimensions="CW", suffix="", bucket=False,
               trace_npts=None, p_time=3.26, s_time=5.24):
        # Generate pulses at UTC-relative times, never using transported indices.
        time = np.arange(int(10 * rate) + 1) / rate
        channels = {"Z": np.exp(-0.5 * ((time - p_time) / .07) ** 2),
                    "N": np.zeros_like(time),
                    "E": np.exp(-0.5 * ((time - s_time) / .07) ** 2)}
        wave = np.stack([channels[c] for c in components])
        if dimensions == "WC":
            wave = wave.T
        name = f"bucket$0,:{wave.shape[0]},:{wave.shape[1]}" if bucket else "trace"
        with h5py.File(self.root / f"waveforms{suffix}.hdf5", "w") as f:
            fmt = f.create_group("data_format")
            fmt["sampling_rate"] = rate
            fmt["component_order"] = components
            fmt["dimension_order"] = dimensions
            f.create_group("data")["bucket" if bucket else name] = wave[None] if bucket else wave
        row = dict(trace_name=name, trace_sampling_rate_hz=rate,
                   trace_start_time="2020-01-01T00:00:00Z", trace_component_order=components)
        if trace_npts is not None:
            row["trace_npts"] = trace_npts
        pd.DataFrame([row]).to_csv(self.root / f"metadata{suffix}.csv", index=False)
        return name

    def dataset(self, name="trace", rate=100, p=3.26, s=5.24, dataset="fixture", chunk="", **kwargs):
        row = dict(dataset_name=dataset, trace_name=name, chunk=chunk,
                   p_arrival_sample=p * rate if p is not None else np.nan,
                   s_arrival_sample=s * rate if s is not None else np.nan)
        row.update(kwargs.pop("fields", {}))
        manifest = self.root / "manifest.csv"
        pd.DataFrame([row]).to_csv(manifest, index=False)
        with patch.dict(md._SINGLE_HDF5_DS, {"fixture": self.root}):
            ds = md.ManifestDataset(manifest, **kwargs)
        self.addCleanup(ds.close)
        self.datasets.append(ds)
        return ds

    def check_times(self, ds, p=3.26, s=5.24):
        wave, labels, info = ds.get_sample_with_metadata(0)
        self.assertEqual(info["source_start_time"], "2020-01-01T00:00:00Z")
        for phase, label, component, time in (("P", 0, 0, p), ("S", 1, 2, s)):
            if time is None:
                self.assertEqual(float(labels[label].max()), 0)
                continue
            label_time = (int(labels[label].argmax()) + info["crop_start_sample"]) / 100
            wave_time = (int(wave[component].argmax()) + info["crop_start_sample"]) / 100
            self.assertLessEqual(abs(label_time - time), .010001)
            self.assertLessEqual(abs(wave_time - time), .010001)
            self.assertAlmostEqual(info["arrival_offsets"][phase] / 100 + info["crop_start_offset_s"], time)
        return wave, labels, info

    def test_hdf5_rates_and_fractional_arrivals_preserve_times(self):
        for rate in (20, 40, 50, 62.5, 80, 100, 120, 200, 250, 500):
            with self.subTest(rate=rate):
                name = self.source(rate)
                ds = self.dataset(name, rate, window_len=801)
                self.check_times(ds)
                ds.close()

    def test_seisbench_configured_resampling_cannot_double_scale(self):
        for rate in (20, 40, 50, 62.5, 100, 120, 200, 250, 500):
            with self.subTest(rate=rate):
                name = self.source(rate, bucket=True)
                # Deliberately configure a reader target different from stored rate.
                def factory(**kwargs):
                    kwargs["sampling_rate"] = 100
                    return sbd.WaveformDataset(self.root, **kwargs)
                with patch.dict(md._SBD_CLASSES, {"fixture_sbd": factory}):
                    ds = self.dataset(name, rate, dataset="fixture_sbd", window_len=801)
                    _, _, info = self.check_times(ds)
                    self.assertEqual(info["source_rate_hz"], rate)
                    self.assertEqual(info["effective_reader_rate_hz"], rate)
                    self.assertEqual(info["arrival_rate_hz"], rate)
                ds.close()

    def test_explicit_arrival_coordinate_rate(self):
        self.source(40)
        ds = self.dataset(rate=100, window_len=801, fields={"arrival_sampling_rate_hz": 100})
        self.check_times(ds)

    def test_bucket_wc_and_missing_channel(self):
        name = self.source(50, components="Z", dimensions="WC", bucket=True)
        wave, _, info = self.dataset(name, 50, s=None, window_len=801).get_sample_with_metadata(0)
        self.assertEqual(info["component_mask"], [True, False, False])
        self.assertEqual(torch.count_nonzero(wave[1:]), 0)
        self.assertEqual(wave.shape, (3, 801))

    def test_s_only_and_p_only(self):
        self.source(40)
        self.check_times(self.dataset(rate=40, p=None, window_len=801), p=None)
        self.check_times(self.dataset(rate=40, s=None, window_len=801), s=None)

    def test_crop_boundaries_and_padding_support(self):
        self.source(40, p_time=0, s_time=10)
        self.check_times(self.dataset(rate=40, p=0, s=None, window_len=401), p=0, s=None)
        self.check_times(self.dataset(rate=40, p=None, s=10, window_len=401), p=None, s=10)
        for ds in self.datasets:
            ds.close()
        self.source(40, trace_npts=201, p_time=2, s_time=8)
        with h5py.File(self.root / "waveforms.hdf5", "a") as f:
            f["data/trace"][:, 201:] = np.nan  # Invalid storage padding is not signal support.
        _, labels, info = self.dataset(rate=40, p=2, s=8, window_len=801).get_sample_with_metadata(0)
        self.assertEqual(info["valid_samples"], 501)
        self.assertIsNone(info["arrival_offsets"]["S"])
        self.assertEqual(info["excluded_arrivals"]["S"], "outside source support")
        self.assertEqual(float(labels[1].max()), 0)

    def test_chunk_identity_and_lazy_pickle(self):
        for suffix, rate in (("_01", 40), ("_02", 50)):
            self.source(rate, suffix=suffix)
        with patch.dict(md._CHUNKED_DS, {"fixture_chunks": (self.root, "waveforms_")}):
            ds = self.dataset(rate=40, dataset="fixture_chunks", chunk="01", window_len=801)
            self.check_times(ds)
            reader = ds._chunked["fixture_chunks"]._reader("01")
            copied = pickle.loads(pickle.dumps(reader))
            self.addCleanup(copied.close)
            self.assertIsNone(copied._h5)
            self.assertEqual(copied.get_record("trace").sampling_rate, 40)
        reader = md.ChunkedHDF5Reader(self.root)
        self.addCleanup(reader.close)
        self.assertEqual(reader.get_record("02", "trace").sampling_rate, 50)
        fake = type("Fake", (), {"metadata": pd.DataFrame({"trace_name": ["x", "x"], "trace_chunk": ["01", "02"]})})()
        self.assertEqual(md.ManifestDataset._build_name_index(fake), {("01", "x"): 0, ("02", "x"): 1})
        fake.metadata["trace_chunk"] = "01"
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            md.ManifestDataset._build_name_index(fake)

    def test_unknown_rate_orientation_missing_trace_reject_and_log(self):
        for defect in ("rate", "orientation", "trace", "labels", "nan_wave"):
            with self.subTest(defect=defect):
                self.source(40)
                if defect == "rate":
                    with h5py.File(self.root / "waveforms.hdf5", "a") as f:
                        del f["data_format/sampling_rate"]
                    frame = pd.read_csv(self.root / "metadata.csv").drop(columns="trace_sampling_rate_hz")
                    frame.to_csv(self.root / "metadata.csv", index=False)
                if defect == "orientation":
                    with h5py.File(self.root / "waveforms.hdf5", "a") as f:
                        del f["data_format/component_order"]
                    frame = pd.read_csv(self.root / "metadata.csv").drop(columns="trace_component_order")
                    frame.to_csv(self.root / "metadata.csv", index=False)
                if defect == "nan_wave":
                    with h5py.File(self.root / "waveforms.hdf5", "a") as f:
                        f["data/trace"][0, 5] = np.nan
                ds = self.dataset("missing" if defect == "trace" else "trace", rate=40,
                                  p=None if defect == "labels" else 3.26, s=None)
                with self.assertRaisesRegex(RuntimeError, "Rejected manifest row 0"):
                    ds[0]
                records = [json.loads(line) for file in self.root.glob("*.rejected.*.jsonl")
                           for line in file.read_text().splitlines()]
                self.assertEqual(records[-1]["row_index"], 0)
                self.assertEqual(records[-1]["manifest_sha256"], ds.manifest_hash)
                self.assertTrue(records[-1]["reason"])
                ds.close()

    def test_per_trace_rate_overrides_file_default_and_conflicts_fail(self):
        self.source(40)
        with h5py.File(self.root / "waveforms.hdf5", "a") as f:
            f["data_format/sampling_rate"][()] = 100
        self.check_times(self.dataset(rate=40, window_len=801))
        with self.assertRaisesRegex(RuntimeError, "conflict"):
            self.dataset(rate=40, fields={"trace_sampling_rate_hz": 50})[0]
        for ds in self.datasets:
            ds.close()
        with h5py.File(self.root / "waveforms.hdf5", "a") as f:
            f["data/trace"].attrs["sampling_rate"] = 50
        with self.assertRaisesRegex(RuntimeError, "conflict"):
            self.dataset(rate=40)[0]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            metadata_rate({"trace_sampling_rate_hz": 50, "trace_dt_s": .025})
        self.assertEqual(metadata_rate({"trace_dt_s": .025}), 40)

    def test_explicit_noise_and_cached_preload_failure_gate(self):
        self.source(40)
        with patch.object(md, "_NOISE_GLOBAL_PATH", self.root):
            ds = self.dataset(rate=40, dataset="noise_global", p=None, s=None)
            wave, labels = ds[0]
            self.assertTrue(torch.isfinite(wave).all())
            self.assertTrue(torch.all(labels[:2] == 0))
            self.assertTrue(torch.all(labels[2] == 1))
        ds = self.dataset(rate=40)
        with patch.dict(md._SINGLE_HDF5_DS, {"fixture": self.root}):
            cached = CachedManifestDataset(ds.manifest_path, load_workers=0, load_batch=1)
            self.assertEqual(len(cached), 1)
            torch.testing.assert_close(cached[0][0], ds[0][0])
            for dataset in self.datasets:
                dataset.close()
            with h5py.File(self.root / "waveforms.hdf5", "a") as f:
                del f["data/trace"]
            with self.assertRaisesRegex(RuntimeError, "Rejected manifest row"):
                CachedManifestDataset(ds.manifest_path, load_workers=0)

    def test_spawn_workers_and_failure_ledger(self):
        self.source(40)
        ds = self.dataset(rate=40)
        expected = ds[0]  # Open a parent handle before pickling to workers.
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2,
                                             multiprocessing_context="spawn")
        batches = list(loader)
        torch.testing.assert_close(batches[0][0][0], expected[0])
        ds.close()
        bad = self.dataset(name="missing", rate=40)
        loader = torch.utils.data.DataLoader(bad, batch_size=1, num_workers=1,
                                             multiprocessing_context="spawn")
        with self.assertRaisesRegex(RuntimeError, "Rejected manifest row"):
            list(loader)
        self.assertTrue(list(self.root.glob("*.rejected.*.jsonl")))

    def test_invalid_rates_lengths_and_noise_labels(self):
        for rate in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                metadata_rate({"sampling_rate": rate})
        self.source(40, trace_npts=10000)
        with self.assertRaisesRegex(RuntimeError, "trace_npts"):
            self.dataset(rate=40)[0]
        for ds in self.datasets:
            ds.close()
        self.source(40)
        with patch.object(md, "_NOISE_GLOBAL_PATH", self.root):
            with self.assertRaisesRegex(RuntimeError, "Noise row contains"):
                self.dataset(rate=40, dataset="noise_global", p=-1, s=None)[0]

    def test_resampler_passband_alias_and_valid_endpoint(self):
        for rate in (20, 40, 62.5, 120, 250, 500):
            t = np.arange(int(rate * 20) + 1) / rate
            frequency = min(rate, 100) * .2
            x = np.sin(2 * np.pi * frequency * t)[None].astype(np.float32)
            y = resample_waveform(x, rate, 100)[0]
            expected = np.sin(2 * np.pi * frequency * np.arange(len(y)) / 100)
            self.assertLess(np.sqrt(np.mean((y[100:-100] - expected[100:-100]) ** 2)), .01)
            self.assertLessEqual((len(y) - 1) / 100, t[-1] + 1e-9)
        t = np.arange(5001) / 250
        y = resample_waveform(np.sin(2 * np.pi * 90 * t)[None], 250, 100)[0]
        self.assertLess(np.sqrt(np.mean(y[100:-100] ** 2)), .01)
        for components in (None, "Z12", "ZZE"):
            with self.assertRaises(ValueError):
                canonical_waveform(np.zeros((3, 10)), components, "CW")

    def test_builder_keeps_s_only_and_coordinate_metadata(self):
        frame = pd.DataFrame(dict(trace_name=["p", "s", "both", "none"],
            trace_P_arrival_sample=[100, np.nan, 100, np.nan],
            trace_S_arrival_sample=[np.nan, 200, 200, np.nan],
            trace_sampling_rate_hz=[40] * 4, trace_component_order=["ENZ"] * 4,
            trace_chunk=["01"] * 4, trace_npts=[1000] * 4,
            source_origin_time=["2010-01-01"] * 4))
        cfg = dict(name="fixture", meta_fn=lambda: frame.copy(), use_s=True,
                   dist_col=None, default_bin="local", cap=100)
        out = process_dataset(cfg, np.random.default_rng(0))
        self.assertEqual(set(out.trace_name), {"p", "s", "both"})
        self.assertEqual(out.chunk.tolist(), ["01"] * 3)
        self.assertEqual(out.trace_sampling_rate_hz.tolist(), [40] * 3)
        self.assertEqual(out.trace_component_order.tolist(), ["ENZ"] * 3)
        frame.drop(columns="trace_P_arrival_sample", inplace=True)
        out = process_dataset(cfg, np.random.default_rng(0))
        self.assertEqual(set(out.trace_name), {"s", "both"})
        cfg["default_bin"] = "teleseismic"
        self.assertTrue(process_dataset(cfg, np.random.default_rng(0)).empty)
        cfg["use_s"] = False
        self.assertIsNone(process_dataset(cfg, np.random.default_rng(0)))


if __name__ == "__main__":
    unittest.main()
