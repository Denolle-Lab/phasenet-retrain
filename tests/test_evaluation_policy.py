"""Offline behavioral checks for suite isolation and access provenance."""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluation_policy as policy
import heldout_testset_registry as registry
import heldout_testset_score as scorer


def test_examined_sequences_are_regression_and_policy_covers_registry():
    roles = policy.load_policy()["roles"]
    assert set(roles) == set(registry.BY_KEY)
    for key in ("kaikoura_2016", "norcia_2016", "thessaly_2021"):
        assert roles[key] == registry.BY_KEY[key]["suite"] == "regression"
    assert policy.load_policy()["acceptance_status"] == "provisional"


@pytest.mark.parametrize("keys,error", [
    (["noto_2024"], PermissionError),
    (["samos_2020", "noto_2024"], PermissionError),
    (["typo"], ValueError),
])
def test_protected_and_unknown_selections_fail(keys, error):
    with pytest.raises(error):
        policy.authorize_scoring(keys)


def test_direct_score_denied_before_model_or_data_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Protected scoring reached model/data access")
    monkeypatch.setattr(scorer, "_load_sequence", forbidden)
    monkeypatch.setattr(policy, "model_fingerprint", forbidden)
    with pytest.raises(PermissionError):
        scorer.score("noto_2024", {"dummy": object()})


@pytest.mark.parametrize("args", [
    ["--sequence", "noto_2024"],
    ["--sequence", "samos_2020", "--sequence", "noto_2024"],
    ["--sequence", "typo"], [],
])
def test_cli_denied_before_model_loading(monkeypatch, args):
    class ForbiddenModels:
        def __getattr__(self, name):
            pytest.fail("Invalid selection reached model loading")
    monkeypatch.setitem(sys.modules, "seisbench", types.ModuleType("seisbench"))
    monkeypatch.setitem(sys.modules, "seisbench.models", ForbiddenModels())
    with pytest.raises(SystemExit) as exc:
        scorer.main(args)
    assert exc.value.code == 2


def test_all_scores_only_built_regression_and_dev(monkeypatch, tmp_path):
    built = ["kaikoura_2016", "samos_2020", "noto_2024"]
    for key in built:
        (tmp_path / key).mkdir()
        (tmp_path / key / "manifest.json").write_text("{}")
    monkeypatch.setattr(scorer, "OUT_ROOT", tmp_path)
    fake_models = types.ModuleType("seisbench.models")
    fake_models.PhaseNet = types.SimpleNamespace(from_pretrained=lambda name: object())
    monkeypatch.setitem(sys.modules, "seisbench", types.ModuleType("seisbench"))
    monkeypatch.setitem(sys.modules, "seisbench.models", fake_models)
    scored = []
    def fake_score(key, models, **kwargs):
        scored.append(key)
        return scorer.ScoreResult(key=key)
    monkeypatch.setattr(scorer, "score", fake_score)
    scorer.main(["--all"])
    assert scored == ["kaikoura_2016", "samos_2020"]


def test_qa_access_permitted_and_logged_on_protected_sequence(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "ACCESS_LOG", tmp_path / "access.jsonl")
    monkeypatch.setattr(scorer, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(scorer, "_load_sequence", lambda key: "reference data")
    assert scorer.load_sequence("noto_2024") == "reference data"
    entry = json.loads(policy.ACCESS_LOG.read_text())
    assert entry["operation"] == "reference_qa" and entry["role"] == "acceptance"
    assert entry["models"] == {} and entry["status"] == "started"
    assert len(entry["policy_sha256"]) == 64


@pytest.mark.parametrize("key", ["kaikoura_2016", "samos_2020"])
def test_allowed_scoring_records_models_and_sources_before_loading(monkeypatch, tmp_path, key):
    monkeypatch.setattr(policy, "ACCESS_LOG", tmp_path / "access.jsonl")
    monkeypatch.setattr(scorer, "OUT_ROOT", tmp_path)
    sequence = tmp_path / key
    sequence.mkdir()
    source = sequence / "windows.csv"
    source.write_text("t0,t1\n")
    fingerprint = {"state_sha256": "a" * 64, "class": "test.Model"}
    monkeypatch.setattr(policy, "model_fingerprint", lambda model: fingerprint)
    def load(key, **kwargs):
        entry = json.loads(policy.ACCESS_LOG.read_text())
        assert entry["operation"] == "model_scoring"
        assert entry["models"] == {"dummy": fingerprint}
        assert entry["source_sha256"]["windows.csv"] == policy.file_hash(source)
        assert entry["source_sha256"]["manifest.json"] is None
        assert all(entry["code_sha256"].values())
        assert entry["settings"]["match_tol_s"] == scorer.MATCH_TOL
        return [], None
    monkeypatch.setattr(scorer, "_load_sequence", load)
    scorer.score(key, {"dummy": object()})


def test_log_failure_prevents_scoring(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "ACCESS_LOG", tmp_path)  # directory, not writable log file
    monkeypatch.setattr(policy, "model_fingerprint", lambda model: {"state_sha256": "a" * 64})
    monkeypatch.setattr(scorer, "_load_sequence", lambda key: pytest.fail("Loaded before audit log"))
    with pytest.raises(OSError):
        scorer.score("samos_2020", {"dummy": object()})
