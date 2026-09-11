"""Offline #34B fixtures; synthetic results are never historical corpus evidence."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import seisbench
    import yaml
except ImportError as exc:
    raise unittest.SkipTest(f"Audit fixtures require the training environment: {exc}")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_v7_rows as audit


class RowAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.manifests = self.root / "manifests"
        self.manifests.mkdir()
        self.legacy, self.namespace = audit.load_legacy(audit.git("show", f"{audit.LEGACY_REF}:scripts/manifest_dataset.py"))

    def waveform(self, rate=20, bucket=False, suffix="", name="trace", count=None):
        t = np.arange(int(rate * 60) + 1) / rate
        wave = np.zeros((3, len(t)), dtype=np.float32)
        wave[0] = np.exp(-((t - 4) / .1) ** 2)
        wave[2] = np.exp(-((t - 12) / .1) ** 2)
        trace = f"bucket$0,:3,:{len(t)}" if bucket else name
        with h5py.File(self.source / f"waveforms{suffix}.hdf5", "w") as f:
            f.create_group("data")["bucket" if bucket else name] = wave[None] if bucket else wave
            fmt = f.create_group("data_format")
            fmt["sampling_rate"], fmt["component_order"], fmt["dimension_order"] = rate, "ZNE", "CW"
        meta = dict(trace_name=trace, trace_sampling_rate_hz=rate, trace_component_order="ZNE",
                    trace_start_time="2010-01-01T00:00:00Z")
        if count is not None:
            meta["trace_npts"] = count
        pd.DataFrame([meta]).to_csv(self.source / f"metadata{suffix}.csv", index=False)
        return trace

    def manifest(self, rate=20, trace="trace", chunk="", p=4, s=12, copies=1, name="fixture"):
        path = self.manifests / "train.csv"
        row = dict(dataset_name=name, trace_name=trace, chunk=chunk,
                   p_arrival_sample=None if p is None else rate * p,
                   s_arrival_sample=None if s is None else rate * s, distance_bin="regional")
        pd.DataFrame([row] * copies).to_csv(path, index=False)
        return path

    def replay(self, path, route="single", options=None, name="fixture"):
        spec = dict(path=str(self.source), route=route)
        if options is not None:
            spec["legacy_reader_options"] = options
        sources = audit.Sources({name: spec}, self.legacy)
        replay = audit.Replay(path, sources, self.legacy, self.namespace, 3001)
        self.addCleanup(sources.close)
        self.addCleanup(replay.close)
        return replay

    def test_direct_rate_assumption_keeps_indices_but_distorts_seconds(self):
        self.waveform(20)
        path = self.manifest()
        row = self.replay(path).row(0, "train", audit.digest(path))
        self.assertEqual(row["legacy_assumed_rate_hz"], 100)
        self.assertEqual(row["legacy_reader_rate_hz"], 20)
        self.assertEqual(row["legacy_physical_output_rate_hz"], 20)
        self.assertEqual(row["P"]["legacy_label_displacement_s"], 0)
        self.assertEqual(row["P"]["legacy_offset"], 80)
        self.assertEqual(row["P"]["corrected_offset"], 400)
        self.assertIsNone(row["training_exposures"])
        self.assertEqual(row["historical_fetch_status"], "unknown")
        summary = audit.summarize([row])
        self.assertEqual(summary[0]["legacy_rate_mismatch_rows"], 1)
        self.assertEqual(summary[0]["displaced_label_rows"], 0)

    def test_seisbench_native_and_already_resampled_rates(self):
        for effective in (None, 100):
            with self.subTest(reader_rate=effective):
                self.waveform(20)
                path = self.manifest()
                replay = self.replay(path, "seisbench", dict(sampling_rate=effective, component_order="ZNE", dimension_order="NCW", missing_components="pad"))
                row = replay.row(0, "train", audit.digest(path))
                self.assertEqual(row["legacy_assumed_rate_hz"], 20)
                self.assertEqual(row["legacy_reader_rate_hz"], effective or 20)
                self.assertEqual(row["corrected"]["arrival_rate_hz"], 20)
                self.assertAlmostEqual(row["P"]["legacy_label_displacement_s"], -3.2 if effective is None else -3.84)
                self.assertEqual(row["P"]["corrected_offset"], 400)
                replay.close()
                replay.sources.close()

    def test_bucket_fetch_failure_is_replay_evidence(self):
        trace = self.waveform(bucket=True)
        path = self.manifest(trace=trace)
        row = self.replay(path).row(0, "train", audit.digest(path))
        self.assertEqual(row["legacy_status"], "zero_noise_substitution")
        self.assertEqual(row["corrected_status"], "ok")
        self.assertFalse(row["P"]["legacy_effective_label"])
        self.assertEqual(row["historical_fetch_status"], "unknown")
        self.assertIsNone(row["same_source_identity"])
        self.assertEqual(audit.summarize([row])[0]["source_identity_mismatch_rows"], 0)

    def test_chunk_inference_failure_and_identity(self):
        self.waveform(suffix="_01")
        path = self.manifest(chunk="01")
        row = self.replay(path, "chunked").row(0, "train", audit.digest(path))
        self.assertEqual(row["identity"], ["fixture", "01", "trace"])
        self.assertEqual(row["legacy_status"], "zero_noise_substitution")
        self.assertEqual(row["corrected_status"], "ok")

    def test_s_only_and_outside_support(self):
        self.waveform(100, count=1001)
        path = self.manifest(rate=100, p=4, s=12)
        replay = self.replay(path)
        row = replay.row(0, "train", audit.digest(path))
        self.assertTrue(row["S"]["legacy_effective_label"])
        self.assertFalse(row["S"]["corrected_effective_label"])
        self.assertEqual(row["corrected"]["excluded_arrivals"]["S"], "outside source support")
        replay.close()
        path = self.manifest(rate=100, p=None, s=6)
        replay = self.replay(path)
        row = replay.row(0, "train", audit.digest(path))
        self.assertFalse(row["S"]["legacy_effective_label"])
        self.assertTrue(row["S"]["corrected_effective_label"])
        self.assertNotIn("legacy_crop_start_sample", row)

    def test_noise_replay_does_not_change_caller_rng(self):
        self.waveform(100)
        path = self.manifest(rate=100, p=None, s=None, name="noise_global")
        replay = self.replay(path, name="noise_global")
        np.random.seed(127)
        expected = np.random.random()
        np.random.seed(127)
        first = replay.row(0, "train", audit.digest(path))
        self.assertEqual(np.random.random(), expected)
        second = replay.row(0, "train", audit.digest(path))
        self.assertEqual(first, second)
        self.assertNotIn("legacy_crop_start_sample", first)
        self.assertFalse(first["P"]["legacy_effective_label"])

    def args(self):
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump({"data": {f"{split}_manifest": f"data/manifests_v2/{split}.csv" for split in ("train", "val", "test")}}))
        specs = self.root / "sources.json"
        specs.write_text(json.dumps({"fixture": {"path": str(self.source), "route": "single"}}))
        return argparse.Namespace(output=str(self.root / "output"), config=str(config), manifest_dir=str(self.manifests),
                                  cache_root=str(self.root), sources=str(specs), inventory_only=False, max_rows=None, exposures=None)

    def test_missing_inputs_leave_incomplete_receipt_no_fake_rows(self):
        args = self.args()
        self.assertEqual(audit.run(args), 2)
        result = json.loads((Path(args.output) / "provenance.json").read_text())
        self.assertEqual(len(result["missing"]), 3)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse((Path(args.output) / "rows.jsonl").exists())
        with self.assertRaises(FileExistsError):
            audit.run(args)

    def test_complete_artifact_hashes_occurrences_and_exposure_evidence(self):
        self.waveform(20)
        train = self.manifest(copies=2)
        for split in ("val", "test"):
            (self.manifests / f"{split}.csv").write_bytes(train.read_bytes())
        before = {p: audit.digest(p) for p in [*self.source.iterdir(), *self.manifests.iterdir()]}
        args = self.args()
        exposure = self.root / "exposures.csv"
        pd.DataFrame([dict(split="train", row_index=i, manifest_sha256=audit.digest(train), training_exposures=3) for i in range(2)]).to_csv(exposure, index=False)
        args.exposures = str(exposure)
        self.assertEqual(audit.run(args), 0)
        out = Path(args.output)
        prov = json.loads((out / "provenance.json").read_text())
        self.assertEqual(prov["status"], "replay_complete")
        self.assertEqual(prov["historical_attribution"], "unverified")
        self.assertFalse(prov["inputs"]["train"]["reference_keys_match"])
        rows = list(audit.jsonl_rows(out / "rows.jsonl"))
        self.assertEqual(len(rows), 6)
        summary = pd.read_csv(out / "phase_summary.csv")
        train_summary = summary[summary.split == "train"]
        self.assertTrue((train_summary.manifest_occurrences == 2).all())
        self.assertTrue((train_summary.unique_trace_identities == 1).all())
        self.assertTrue((train_summary.training_exposures == 6).all())
        self.assertTrue(summary[summary.split != "train"].training_exposures.isna().all())
        for name, expected_hash in prov["outputs"].items():
            self.assertEqual(audit.digest(out / name), expected_hash)
        self.assertEqual({p: audit.digest(p) for p in before}, before)

    def test_invalid_exposure_evidence_fails_with_incomplete_receipt(self):
        self.waveform()
        train = self.manifest()
        for split in ("val", "test"):
            (self.manifests / f"{split}.csv").write_bytes(train.read_bytes())
        base = dict(split="train", row_index=0, manifest_sha256=audit.digest(train), training_exposures=3)
        for label, rows in (
            ("fractional_index", [dict(base, row_index=.5)]),
            ("wrong_hash", [dict(base, manifest_sha256="wrong")]),
            ("fractional_count", [dict(base, training_exposures=1.5)]),
            ("duplicate", [base, base]),
        ):
            with self.subTest(label=label):
                args = self.args()
                args.output = str(self.root / label)
                exposure = self.root / "invalid_exposures.csv"
                pd.DataFrame(rows).to_csv(exposure, index=False)
                args.exposures = str(exposure)
                with self.assertRaises(ValueError):
                    audit.run(args)
                provenance = json.loads((Path(args.output) / "provenance.json").read_text())
                self.assertEqual(provenance["status"], "incomplete")
                self.assertIn("ValueError", provenance["error"])

    def test_missing_single_file_is_initialization_abort_not_training_noise(self):
        path = self.manifest()
        row = self.replay(path).row(0, "train", audit.digest(path))
        self.assertEqual(row["legacy_status"], "initialization_error")
        self.assertIsNone(row["P"]["legacy_effective_label"])
        self.assertEqual(row["corrected_status"], "rejected")
        summary = audit.summarize([row])
        self.assertEqual(summary[0]["legacy_zero_noise_rows"], 0)
        self.assertEqual(summary[0]["legacy_error_rows"], 1)

    def test_duplicate_seisbench_names_are_not_paired_as_same_trace(self):
        self.waveform(20, suffix="_a")
        self.waveform(40, suffix="_b")
        # Common file defaults, with verified per-trace 40 Hz metadata in _b.
        with h5py.File(self.source / "waveforms_b.hdf5", "a") as f:
            f["data_format/sampling_rate"][()] = 20
        path = self.manifest(chunk="_a")
        options = dict(sampling_rate=None, component_order="ZNE", dimension_order="NCW", missing_components="pad", chunks=["_a", "_b"])
        row = self.replay(path, "seisbench", options).row(0, "train", audit.digest(path))
        self.assertEqual(row["legacy_source_identity"], ["fixture", "_b", "trace"])
        self.assertEqual(row["corrected_source_identity"], ["fixture", "_a", "trace"])
        self.assertFalse(row["same_source_identity"])
        self.assertNotIn("legacy_label_displacement_s", row["P"])

    def test_wrong_legacy_time_axis_has_no_fabricated_seconds(self):
        self.waveform(20)
        with h5py.File(self.source / "waveforms.hdf5", "a") as f:
            wave = f["data/trace"][0, :][:, None]
            del f["data/trace"]
            f["data/trace"] = wave
            del f["data_format/component_order"]
            f["data_format/component_order"] = "Z"
            del f["data_format/dimension_order"]
            f["data_format/dimension_order"] = "WC"
        meta = pd.read_csv(self.source / "metadata.csv")
        meta["trace_component_order"] = "Z"
        meta.to_csv(self.source / "metadata.csv", index=False)
        path = self.manifest()
        row = self.replay(path).row(0, "train", audit.digest(path))
        self.assertEqual(row["corrected_status"], "ok")
        self.assertFalse(row["legacy_time_axis_verified"])
        self.assertNotIn("legacy_physical_output_rate_hz", row)
        self.assertNotIn("legacy_label_displacement_s", row["P"])

    def test_invalid_arrival_is_reported_without_crashing_summary(self):
        self.waveform()
        path = self.manifest()
        frame = pd.read_csv(path)
        frame["p_arrival_sample"] = "bad-pick"
        frame.to_csv(path, index=False)
        row = self.replay(path).row(0, "train", audit.digest(path))
        self.assertEqual(row["legacy_status"], "preprocessing_error")
        self.assertEqual(row["corrected_status"], "rejected")
        self.assertEqual(audit.summarize([row])[0]["declared_labels"], 0)

    def test_changed_source_is_rejected(self):
        self.waveform()
        sources = audit.Sources({}, self.legacy)
        path = self.source / "metadata.csv"
        sources.note_file(path)
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            sources.verify_unchanged()


if __name__ == "__main__":
    unittest.main()
