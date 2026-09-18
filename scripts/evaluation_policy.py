"""Suite roles and access records for repository evaluation entrypoints (#44A).

Acceptance candidates remain protected until a separate release protocol exists.
This is a workflow guard, not access control for arbitrary Python or raw files.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "configs" / "evaluation_suites.json"
ACCESS_LOG = ROOT / "data" / "evaluation" / "access.jsonl"
SCORING_ROLES = {"regression", "dev"}


def load_policy():
    policy = json.loads(POLICY_PATH.read_text())
    if policy.get("version") != 1 or not policy.get("roles"):
        raise ValueError("Unsupported or empty evaluation suite policy")
    if set(policy["roles"].values()) - {"regression", "dev", "calibration", "acceptance"}:
        raise ValueError("Unknown suite role in evaluation policy")
    return policy


def role_for(key, policy=None):
    policy = load_policy() if policy is None else policy
    if key not in policy["roles"]:
        raise ValueError(f"Unknown evaluation sequence: {key}")
    return policy["roles"][key]


def authorize_scoring(keys):
    policy = load_policy()
    for key in keys:
        role = role_for(key, policy)
        if role not in SCORING_ROLES:
            raise PermissionError(
                f"{key} is protected ({role}); routine scoring permits regression/dev only. "
                "Reference QA is separate. Acceptance release requires #44B/#49."
            )
    return policy


def routine_keys(keys):
    policy = load_policy()
    return [key for key in keys if role_for(key, policy) in SCORING_ROLES]


def file_hash(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(model):
    """Hash actual tensor contents, including buffers, independent of weight alias."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state_sha256": digest.hexdigest(),
        "default_args": getattr(model, "default_args", {}),
    }


def record_access(key, operation, *, models=None, settings=None, data_root=None, log_path=None):
    """Record a started access before reference loading or model inference.

    A record proves an attempt, not successful evaluation. A logging failure
    stops the caller. Missing local artifacts have null hashes.
    """
    if operation not in {"reference_qa", "model_scoring"}:
        raise ValueError(f"Unknown evaluation operation: {operation}")
    policy = authorize_scoring([key]) if operation == "model_scoring" else load_policy()
    role = role_for(key, policy)
    if operation == "model_scoring" and not models:
        raise ValueError("Scoring access requires model fingerprints")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    data_root = ROOT / "data" / "heldout_testset" if data_root is None else Path(data_root)
    record = {
        "id": str(uuid4()), "time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started", "operation": operation, "key": key, "role": role,
        "policy": policy,
        "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        "git_commit": commit,
        "code_sha256": {name: file_hash(ROOT / "scripts" / name) for name in (
            "evaluation_policy.py", "heldout_testset_score.py", "heldout_testset_registry.py")},
        "data_root": str(data_root.resolve()),
        "source_sha256": {name: file_hash(data_root / key / name) for name in (
            "manifest.json", "picks.parquet", "windows.csv", "stations.csv")},
        "models": models or {}, "settings": settings or {},
    }
    # Serialize before opening, so an unsupported value cannot leave a partial line.
    line = json.dumps(record, sort_keys=True) + "\n"
    log_path = ACCESS_LOG if log_path is None else Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(line)
    return record["id"]
