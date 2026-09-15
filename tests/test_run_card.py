"""Run card and rejection-ledger gate (46A tooling, #46 / #34). Torch-free; temp dirs only."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_card as rc  # noqa: E402

ARRIVALS = [
    # one event, P and S
    '[{"phase":"P","time_s":1.0},{"phase":"S","time_s":1.5}]',
    # two events by event_id: multi-event
    '[{"phase":"P","time_s":1.0,"event_id":"a"},{"phase":"P","time_s":9.0,"event_id":"b"}]',
    # noise row: no list
    None,
    # automatic P only
    '[{"phase":"P","time_s":3.0,"tier":"automatic"}]',
    # two P without event ids: multi-event; S present
    '[{"phase":"P","time_s":2.0},{"phase":"S","time_s":4.0},{"phase":"P","time_s":20.0}]',
]


def synthetic_manifest(with_arrivals: bool) -> pd.DataFrame:
    df = pd.DataFrame({
        "dataset_name": ["stead", "stead", "noise_global", "mlaapde", "ethz"],
        "trace_name": ["t1", "t2", "n1", "t3", "t4"],
        "chunk": ["", "", "", "001", ""],
        "p_arrival_sample": [100.0, 200.0, None, 300.0, None],
        "s_arrival_sample": [150.0, None, None, None, 500.0],
        "distance_bin": ["local", "regional", "noise", "teleseismic", "local"],
        "snr_db": [3.0, 12.0, None, -1.0, 25.0],
        "negative_support": ["unknown", "certified", "certified", "unknown", "reviewed"],
    })
    if with_arrivals:
        df["arrivals_json"] = ARRIVALS
    return df


def write_manifest(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def run(tmp_path):
    """Config file, train/val manifests and the manifest_paths dict."""
    train = write_manifest(tmp_path / "train.csv", synthetic_manifest(True))
    val = write_manifest(tmp_path / "val.csv", synthetic_manifest(False).iloc[:3])
    config = {
        "seed": 7,
        "data": {"train_manifest": str(train), "val_manifest": str(val), "label_policy": "masked",
                 "augmentation": {"noise_prob": 0.2, "noise_snr_db_range": [0, 10]}},
        "training": {"batch_size": 8, "learning_rate": 1e-5},
        "logging": {"run_name": "unit", "save_dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return dict(tmp_path=tmp_path, config=config, config_path=config_path,
                manifests={"train": train, "val": val}, rows={"train": 5, "val": 3})


def build(run, rows_read=None, extra=None):
    return rc.build_run_card(run["config"], run["config_path"], run["manifests"],
                             rows_read=run["rows"] if rows_read is None else rows_read, extra=extra)


# ── hashes ────────────────────────────────────────────────────────────────────

def test_hashes_are_stable_across_calls(run):
    a, b = build(run), build(run)
    for split in ("train", "val"):
        assert a["manifests"][split]["file_sha256"] == b["manifests"][split]["file_sha256"]
        assert a["manifests"][split]["keys_sha256"] == b["manifests"][split]["keys_sha256"]
    assert a["config"]["file_sha256"] == b["config"]["file_sha256"]
    assert a["config"]["canonical_json_sha256"] == b["config"]["canonical_json_sha256"]
    assert len(a["manifests"]["train"]["keys_sha256"]) == 64


def test_hashes_change_when_a_row_changes(run):
    before = build(run)["manifests"]["train"]
    df = synthetic_manifest(True)
    df.loc[0, "trace_name"] = "t1_changed"
    write_manifest(run["manifests"]["train"], df)
    after = build(run)["manifests"]["train"]
    assert after["file_sha256"] != before["file_sha256"]
    assert after["keys_sha256"] != before["keys_sha256"]


def test_keys_hash_is_order_independent_but_file_hash_is_not(run):
    before = build(run)["manifests"]["train"]
    write_manifest(run["manifests"]["train"], synthetic_manifest(True).iloc[::-1])
    after = build(run)["manifests"]["train"]
    assert after["keys_sha256"] == before["keys_sha256"]
    assert after["file_sha256"] != before["file_sha256"]


def test_config_hash_tracks_content_not_formatting(run):
    before = build(run)["config"]
    run["config_path"].write_text(yaml.safe_dump(run["config"], sort_keys=False))   # same content, other layout
    same = build(run)["config"]
    assert same["canonical_json_sha256"] == before["canonical_json_sha256"]
    run["config"]["training"]["learning_rate"] = 2e-5
    assert build(run)["config"]["canonical_json_sha256"] != before["canonical_json_sha256"]
    assert build(run)["config"]["seed"] == 7 and build(run)["config"]["label_policy"] == "masked"
    assert build(run)["config"]["augmentation"] == {"noise_prob": 0.2, "noise_snr_db_range": [0, 10]}


# ── ledger gate ───────────────────────────────────────────────────────────────

def test_gate_passes_with_no_or_empty_ledgers(run):
    (run["tmp_path"] / "train.rejected.4242.jsonl").write_text("")        # empty per-pid ledger
    card = build(run)
    ledger = card["ledger"]
    assert ledger["ledger_gate_passed"] is True
    assert ledger["n_records_total"] == 0
    assert [f["n_records"] for f in ledger["manifests"]["train"]["files"]] == [0]
    assert ledger["manifests"]["val"]["files"] == []
    assert rc.verify_ledger(run["manifests"]) == {"train": [str(run["tmp_path"] / "train.rejected.4242.jsonl")], "val": []}


def test_gate_fails_on_a_single_ledger_record(run):
    record = {"row_index": 3, "dataset_name": "mlaapde", "trace_name": "t3", "error_type": "ValueError",
              "reason": "Missing sampling rate"}
    (run["tmp_path"] / "val.rejected.777.jsonl").write_text(json.dumps(record) + "\n")
    card = build(run)
    assert card["ledger"]["ledger_gate_passed"] is False
    assert card["ledger"]["manifests"]["val"]["n_records"] == 1
    with pytest.raises(RuntimeError) as err:
        rc.verify_ledger(run["manifests"])
    text = str(err.value)
    assert "1 record(s)" in text and "Missing sampling rate" in text and '"row_index": 3' in text
    assert "failure budget is zero" in text
    assert "ledger.ledger_gate_passed" in " ".join(rc.check_run_card(rc.write_run_card(card, run["tmp_path"] / "r")))


def test_unparseable_ledger_line_still_counts(run):
    (run["tmp_path"] / "train.rejected.1.jsonl").write_text("not json\n\n")
    assert build(run)["ledger"]["manifests"]["train"]["n_records"] == 1
    with pytest.raises(RuntimeError, match="UnparseableLedgerLine"):
        rc.verify_ledger(run["manifests"])


def test_rows_read_mismatch_fails_the_gate(run):
    card = build(run, rows_read={"train": 4, "val": 3})     # train manifest has 5 rows
    assert card["ledger"]["ledger_gate_passed"] is False
    assert card["ledger"]["rows_read_mismatch"] == ["train"]
    assert card["ledger"]["rows_read"]["train"] == {"rows_read": 4, "manifest_rows": 5, "match": False}
    assert card["ledger"]["rows_read"]["val"]["match"] is True
    # without rows_read the gate is decided by the ledgers alone and says so
    card = build(run, rows_read={})
    assert card["ledger"]["ledger_gate_passed"] is True and card["ledger"]["rows_read_given"] is False


# ── composition ───────────────────────────────────────────────────────────────

def test_composition_with_arrivals_json(run):
    comp = build(run)["manifests"]["train"]["composition"]
    assert comp["n_rows"] == 5 and comp["n_noise_rows"] == 1 and comp["n_signal_rows"] == 4
    assert comp["by_source"] == {"stead": 2, "ethz": 1, "mlaapde": 1, "noise_global": 1}
    assert comp["by_distance_bin"] == {"local": 2, "noise": 1, "regional": 1, "teleseismic": 1}
    assert comp["snr_column"] == "snr_db"
    assert comp["by_snr_bin"] == {"0-5 dB": 1, "10-20 dB": 1, "<0 dB": 1, ">20 dB": 1, "missing": 1}
    # S from arrivals_json where present (rows 0 and 4), else from s_arrival_sample (none of the others)
    assert comp["arrivals_json_rows"] == 4
    assert comp["n_s_labelled_rows"] == 2 and comp["s_label_fraction"] == pytest.approx(2 / 5)
    assert comp["s_label_fraction_signal"] == pytest.approx(2 / 4)
    # rows 1 (two event ids) and 4 (two P) are multi-event
    assert comp["n_multi_event_rows"] == 2 and comp["multi_event_fraction"] == pytest.approx(2 / 5)
    assert comp["arrival_tier_counts"] == {"manual": 7, "automatic": 1}     # rows 0, 1, 4: 2 + 2 + 3 manual
    assert comp["negative_support"] == {"certified": 2, "unknown": 2, "reviewed": 1}
    assert build(run)["manifests"]["train"]["rows_by_dataset"] == comp["by_source"]


def test_composition_without_arrivals_json(run):
    comp = build(run)["manifests"]["val"]["composition"]      # first three rows, legacy columns only
    assert comp["n_rows"] == 3 and comp["arrivals_json_rows"] == 0
    assert comp["multi_event_fraction"] is None and "arrivals_json" in comp["multi_event_reason"]
    assert comp["arrival_tier_counts"] is None and comp["s_label_source"] == "s_arrival_sample"
    assert comp["n_s_labelled_rows"] == 1 and comp["s_label_fraction"] == pytest.approx(1 / 3)
    assert comp["n_p_labelled_rows"] == 2 and comp["n_s_only_rows"] == 0
    assert comp["by_distance_bin"] == {"local": 1, "noise": 1, "regional": 1}


def test_composition_reports_absent_columns_as_null(tmp_path):
    df = pd.DataFrame({"dataset_name": ["stead"], "trace_name": ["x"], "chunk": [""],
                       "p_arrival_sample": [10.0], "s_arrival_sample": [20.0]})
    comp = rc.composition_summary(df)
    assert comp["by_distance_bin"] is None and comp["by_snr_bin"] is None
    assert comp["negative_support"] is None and comp["multi_event_fraction"] is None
    assert comp["s_label_fraction"] == 1.0
    summary = rc.manifest_summary(write_manifest(tmp_path / "m.csv", df))
    assert summary["n_rows"] == 1 and summary["keys_sha256"] and summary["rows_by_dataset"] == {"stead": 1}
    assert rc.manifest_summary(tmp_path / "absent.csv")["exists"] is False


# ── card fields, finalize, check, CLI ────────────────────────────────────────

def test_card_records_contract_bundle_versions_git_and_extra(run, tmp_path):
    card = build(run, extra={"allow_rejections": False, "device": "cpu"})
    assert card["loader_contract_version"] == "34a-v1" and card["arrival_schema_version"] == "41a-v1"
    assert card["exclusion_bundle"] == {"path": str(rc.EXCLUSION_BUNDLE), "sha256": None, "reason": "no bundle; 33A"}
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    assert rc.exclusion_bundle(bundle)["sha256"] == rc.sha256_bytes(b"{}")
    for key in ("python", "numpy", "scipy", "pandas"):
        assert card["versions"][key]
    for key in ("torch", "seisbench", "obspy", "h5py"):
        assert key in card["versions"]
    assert card["git"]["commit"] and isinstance(card["git"]["dirty"], bool)
    assert card["extra"] == {"allow_rejections": False, "device": "cpu"}
    assert card["run_name"] == "unit" and card["status"] == "started" and card["finished_utc"] is None
    assert rc.git_state(tmp_path)["commit"] is None       # not a repository: null with reason


def test_config_path_none_gives_null_file_hash(run):
    card = rc.build_run_card(run["config"], None, run["manifests"])
    assert card["config"]["file_sha256"] is None and "not given" in card["config"]["file_sha256_reason"]
    assert card["config"]["canonical_json_sha256"]


def test_write_finalize_and_check(run):
    results_dir = run["tmp_path"] / "results" / "unit"
    path = rc.write_run_card(build(run), results_dir)
    assert path == results_dir / "run_card.json" and path.exists()
    missing = rc.check_run_card(path, require_end=False)
    assert set(missing) <= {"versions.torch", "versions.seisbench"}     # laptop environment only
    assert {"finished_utc", "best_epoch", "dev_metric", "checkpoint.path", "checkpoint.sha256"} <= set(rc.check_run_card(path))

    ckpt = run["tmp_path"] / "best.pt"
    ckpt.write_bytes(b"weights")
    card = rc.finalize_run_card(path, 12, {"monitor": "val_loss", "val_loss": 0.25, "val_s_mae_s": float("nan")}, ckpt)
    reread = json.loads(path.read_text())
    assert reread == card
    assert reread["status"] == "finished" and reread["finished_utc"]
    assert reread["best_epoch"] == 12
    assert reread["dev_metric"] == {"monitor": "val_loss", "val_loss": 0.25, "val_s_mae_s": None}
    assert reread["checkpoint"]["sha256"] == rc.sha256_bytes(b"weights")
    assert reread["checkpoint"]["path"] == str(ckpt) and reread["checkpoint"]["size_bytes"] == 7

    # complete once the environment-dependent versions are present
    reread["versions"]["torch"], reread["versions"]["seisbench"] = "2.7.1", "0.9.1"
    path.write_text(json.dumps(reread))
    assert rc.check_run_card(path) == []

    # a missing checkpoint is recorded, not fatal, and check reports it
    rc.finalize_run_card(path, 12, {"val_loss": 0.25}, run["tmp_path"] / "absent.pt")
    assert "checkpoint.sha256" in rc.check_run_card(path)


def test_check_reports_missing_fields_and_cli_exit_codes(run, capsys):
    path = rc.write_run_card(build(run), run["tmp_path"] / "r")
    card = json.loads(path.read_text())
    card["versions"]["torch"], card["versions"]["seisbench"] = "x", "y"
    del card["manifests"]["train"]["keys_sha256"]
    del card["config"]["label_policy"]
    card["git"]["commit"] = None
    path.write_text(json.dumps(card))
    missing = rc.check_run_card(path, require_end=False)
    assert missing == ["manifests.train.keys_sha256", "git.commit", "config.label_policy"]

    assert rc.main(["check", str(path)]) == 2
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "best_epoch" in out and "manifests.train.keys_sha256" in out
    assert rc.main(["show", str(path)]) == 0
    assert "gate passed = True" in capsys.readouterr().out

    ckpt = run["tmp_path"] / "best.pt"
    ckpt.write_bytes(b"w")
    card = json.loads(path.read_text())
    card["manifests"]["train"]["keys_sha256"] = "k"
    card["config"]["label_policy"] = None
    card["git"]["commit"] = "c"
    path.write_text(json.dumps(card))
    rc.finalize_run_card(path, 1, {"val_loss": 1.0}, ckpt)
    assert rc.main(["check", str(path)]) == 0
    assert rc.main(["show", "--json", str(path)]) == 0
