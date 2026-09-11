#!/usr/bin/env python3
"""Run card and rejection-ledger gate for training runs (46A tooling, #46; answers #34).

Every training run writes `results/<run_name>/run_card.json` before the first
optimiser step (`build_run_card` + `write_run_card`) and completes it at the
end (`finalize_run_card`). The card records what was trained on and with what
code, and `verify_ledger` refuses to train when any manifest has a non-empty
rejection ledger beside it (the 34A loader writes
`<manifest stem>.rejected.<pid>.jsonl`; the failure budget is zero).

Pure Python + numpy/pandas/yaml: no torch, no SeisBench, no h5py, so the
module and its tests run on a laptop. Versions of torch, seisbench, obspy and
h5py are recorded when importable and null otherwise.

CLI
---
  python scripts/run_card.py check results/<run>/run_card.json   # exit 2 when incomplete
  python scripts/run_card.py show  results/<run>/run_card.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from arrivals import ARRIVALS_COLUMN, NEGATIVE_SUPPORT_COLUMN, SCHEMA_VERSION, arrivals_from_json  # noqa: E402
from hash_manifests import KEY_COLS, hash_manifest  # noqa: E402
from waveform_contract import CONTRACT_VERSION  # noqa: E402

CARD_VERSION = "46a-v1"
CARD_NAME = "run_card.json"
EXCLUSION_BUNDLE = REPO_ROOT / "data" / "exclusions" / "bundle.json"
NO_BUNDLE_REASON = "no bundle; 33A"

# Explicit noise pools (scripts/manifest_dataset.py treats only these as noise).
NOISE_SOURCES = ("noise_global", "noise_prephase")
DISTANCE_BIN_COLUMNS = ("distance_bin", "dist_bin")
SNR_BIN_COLUMNS = ("snr_bin",)
SNR_VALUE_COLUMNS = ("snr_db", "trace_snr_db", "snr")
# Same edges as scripts/audit_generalization_hypotheses.py::SNR_EDGES.
SNR_EDGES_DB = (-math.inf, 0.0, 5.0, 10.0, 20.0, math.inf)
SNR_LABELS = ("<0 dB", "0-5 dB", "5-10 dB", "10-20 dB", ">20 dB")

OPTIONAL_PACKAGES = ("torch", "seisbench", "obspy", "h5py")
REQUIRED_PACKAGES = ("numpy", "scipy", "pandas")

# Dotted paths that must be present and non-null in a card written at the start.
REQUIRED_START = (
    "card_version", "run_name", "started_utc",
    "config.file_sha256", "config.canonical_json_sha256", "config.seed",
    "loader_contract_version", "arrival_schema_version",
    "manifests.train.file_sha256", "manifests.train.keys_sha256", "manifests.train.n_rows",
    "manifests.train.rows_by_dataset", "manifests.train.composition",
    "manifests.val.file_sha256", "manifests.val.keys_sha256", "manifests.val.n_rows",
    "manifests.val.rows_by_dataset", "manifests.val.composition",
    "exclusion_bundle", "ledger.manifests", "ledger.rows_read", "ledger.ledger_gate_passed",
    "versions.python", "versions.numpy", "versions.scipy", "versions.pandas",
    "versions.torch", "versions.seisbench",
    "git.commit", "git.dirty",
)
# Dotted paths that must exist as keys; null is an allowed value.
REQUIRED_KEYS_NULLABLE = ("config.label_policy", "config.augmentation")
# Dotted paths that must be present and non-null once the run has finished.
REQUIRED_END = ("finished_utc", "best_epoch", "dev_metric", "checkpoint.path", "checkpoint.sha256")


# ──────────────────────────────────────────────────────────────────────────────
# Hashing and serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace, non-JSON types via str()."""
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def jsonable(obj):
    """Convert numpy scalars, Paths and NaN into JSON-serialisable values (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────────────
# Config, environment, git
# ──────────────────────────────────────────────────────────────────────────────

def config_section(config: dict, config_path) -> dict:
    data_cfg = config.get("data", {}) or {}
    train_cfg = config.get("training", {}) or {}
    section = {
        "path": str(config_path) if config_path else None,
        "file_sha256": sha256_file(config_path) if config_path and Path(config_path).exists() else None,
        "canonical_json_sha256": sha256_bytes(canonical_json(config).encode("utf-8")),
        "seed": config.get("seed", 42),
        "seed_source": "config" if "seed" in config else "default 42 (scripts/finetune.py)",
        "label_policy": data_cfg.get("label_policy"),
        "augmentation": data_cfg.get("augmentation"),
        "window_length": data_cfg.get("window_length"),
        "training": {k: train_cfg.get(k) for k in (
            "batch_size", "max_epochs", "learning_rate", "optimizer", "weight_decay",
            "distillation", "soft_ce", "timing_beta", "presence_gamma", "focal_gamma",
            "scheduler", "early_stopping", "gradient_clip_val", "accumulate_grad_batches")},
        "model": config.get("model"),
    }
    if section["file_sha256"] is None:
        section["file_sha256_reason"] = "config path not given or not found"
    return section


def package_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        try:
            module = importlib.import_module(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # ImportError, or a broken optional install
            out[name] = None
    return out


def git_state(repo_root=REPO_ROOT) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as exc:  # not a repository, git missing
        return {"commit": None, "branch": None, "dirty": None, "reason": f"{type(exc).__name__}: {exc}"}


def exclusion_bundle(path=EXCLUSION_BUNDLE) -> dict:
    path = Path(path)
    if path.exists():
        return {"path": str(path), "sha256": sha256_file(path)}
    return {"path": str(path), "sha256": None, "reason": NO_BUNDLE_REASON}


# ──────────────────────────────────────────────────────────────────────────────
# Manifest composition
# ──────────────────────────────────────────────────────────────────────────────

def _counts(series: pd.Series) -> dict:
    counts = series.fillna("missing").astype(str).value_counts()
    return {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def _first_column(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _arrival_flags(df: pd.DataFrame):
    """Per-row (has_s, is_multi_event, tier counts) from `arrivals_json`."""
    has_s = np.zeros(len(df), dtype=bool)
    multi = np.zeros(len(df), dtype=bool)
    present = np.zeros(len(df), dtype=bool)
    tiers: dict = {}
    for i, text in enumerate(df[ARRIVALS_COLUMN].tolist()):
        arrivals = arrivals_from_json(text)
        if not arrivals and not (isinstance(text, str) and text.strip()):
            continue
        present[i] = True
        phases = [a.phase for a in arrivals]
        has_s[i] = "S" in phases
        events = {a.event_id for a in arrivals if a.event_id is not None}
        multi[i] = len(events) > 1 or phases.count("P") > 1 or phases.count("S") > 1
        for a in arrivals:
            tiers[a.tier] = tiers.get(a.tier, 0) + 1
    return has_s, multi, present, tiers


def composition_summary(df: pd.DataFrame) -> dict:
    """Composition from whatever columns exist; absent columns are reported as null with a reason."""
    n = int(len(df))
    out: dict = {"n_rows": n}

    out["by_source"] = _counts(df["dataset_name"]) if "dataset_name" in df.columns else None
    if "dataset_name" in df.columns:
        is_noise = df["dataset_name"].isin(NOISE_SOURCES).to_numpy()
    else:
        is_noise = np.zeros(n, dtype=bool)
    out["n_noise_rows"] = int(is_noise.sum())
    out["n_signal_rows"] = int(n - is_noise.sum())

    dist_col = _first_column(df, DISTANCE_BIN_COLUMNS)
    out["distance_bin_column"] = dist_col
    out["by_distance_bin"] = _counts(df[dist_col]) if dist_col else None

    snr_bin_col = _first_column(df, SNR_BIN_COLUMNS)
    snr_val_col = _first_column(df, SNR_VALUE_COLUMNS)
    if snr_bin_col:
        out["snr_column"] = snr_bin_col
        out["by_snr_bin"] = _counts(df[snr_bin_col])
    elif snr_val_col:
        out["snr_column"] = snr_val_col
        out["snr_bin_edges_db"] = [e if math.isfinite(e) else None for e in SNR_EDGES_DB]
        binned = pd.cut(pd.to_numeric(df[snr_val_col], errors="coerce"), list(SNR_EDGES_DB), labels=list(SNR_LABELS))
        out["by_snr_bin"] = _counts(binned.astype(object))
    else:
        out["snr_column"] = None
        out["by_snr_bin"] = None

    # S-label fraction: `arrivals_json` overrides the legacy columns for the
    # targets (scripts/manifest_dataset.py), so a row with arrivals_json counts
    # an S only when the list has one; other rows use s_arrival_sample.
    has_s_col = df["s_arrival_sample"].notna().to_numpy() if "s_arrival_sample" in df.columns else np.zeros(n, bool)
    has_p_col = df["p_arrival_sample"].notna().to_numpy() if "p_arrival_sample" in df.columns else np.zeros(n, bool)
    if ARRIVALS_COLUMN in df.columns:
        has_s_json, multi, json_present, tiers = _arrival_flags(df)
        has_s = np.where(json_present, has_s_json, has_s_col)
        out["arrivals_json_rows"] = int(json_present.sum())
        out["multi_event_fraction"] = float(multi.mean()) if n else None
        out["n_multi_event_rows"] = int(multi.sum())
        out["arrival_tier_counts"] = tiers
        out["s_label_source"] = f"{ARRIVALS_COLUMN} where present, else s_arrival_sample"
    else:
        has_s = has_s_col
        out["arrivals_json_rows"] = 0
        out["multi_event_fraction"] = None
        out["multi_event_reason"] = f"no {ARRIVALS_COLUMN} column"
        out["arrival_tier_counts"] = None
        out["s_label_source"] = "s_arrival_sample"
    signal = ~is_noise
    out["s_label_fraction"] = float(has_s.mean()) if n else None
    out["s_label_fraction_signal"] = float(has_s[signal].mean()) if signal.any() else None
    out["n_s_labelled_rows"] = int(has_s.sum())
    out["n_p_labelled_rows"] = int(has_p_col.sum())
    out["n_s_only_rows"] = int((has_s & ~has_p_col & signal).sum())

    if NEGATIVE_SUPPORT_COLUMN in df.columns:
        out["negative_support"] = _counts(df[NEGATIVE_SUPPORT_COLUMN])
    else:
        out["negative_support"] = None
        out["negative_support_reason"] = f"no {NEGATIVE_SUPPORT_COLUMN} column"
    return out


def read_manifest(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, dtype={"chunk": str, "trace_name": str})


def manifest_summary(path) -> dict:
    """File hash, order-independent key hash (scripts/hash_manifests.py), rows and composition."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "reason": "manifest not found"}
    df = read_manifest(path)
    missing_keys = [c for c in KEY_COLS if c not in df.columns]
    if missing_keys:
        keys_sha256, keys_reason = None, f"missing key columns {missing_keys}"
    else:
        keys_sha256, n_keys = hash_manifest(path)
        keys_reason = None
        assert n_keys == len(df)
    summary = {
        "path": str(path),
        "exists": True,
        "file_sha256": sha256_file(path),
        "keys_sha256": keys_sha256,
        "keys_hash_columns": list(KEY_COLS),
        "n_rows": int(len(df)),
        "rows_by_dataset": _counts(df["dataset_name"]) if "dataset_name" in df.columns else None,
        "columns": list(map(str, df.columns)),
        "composition": composition_summary(df),
    }
    if keys_reason:
        summary["keys_sha256_reason"] = keys_reason
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Rejection ledger
# ──────────────────────────────────────────────────────────────────────────────

def ledger_files(manifest_path) -> list:
    """`<stem>.rejected.<pid>.jsonl` (and a bare `<stem>.rejected.jsonl`) beside the manifest."""
    manifest_path = Path(manifest_path)
    parent, stem = manifest_path.parent, manifest_path.stem
    if not parent.exists():
        return []
    files = set(parent.glob(f"{stem}.rejected.*.jsonl")) | set(parent.glob(f"{stem}.rejected.jsonl"))
    return sorted(files)


def read_ledger(path) -> list:
    """Parse one JSONL ledger; a line that is not JSON still counts as one record."""
    records = []
    with open(path) as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                record = {"error_type": "UnparseableLedgerLine", "reason": str(exc), "line": lineno, "raw": line.strip()}
            record["ledger_file"] = str(path)
            records.append(record)
    return records


def ledger_summary(manifest_paths: dict, rows_read: dict | None = None, manifest_rows: dict | None = None) -> dict:
    """Per-manifest ledger files and record counts plus the gate verdict.

    The gate passes only when every ledger beside every manifest is empty and,
    for each split in `rows_read`, the rows the loader served equal the
    manifest's row count (`manifest_rows`, read from the file when not given).
    """
    manifests, n_total = {}, 0
    for split, path in manifest_paths.items():
        files = []
        for f in ledger_files(path):
            n = len(read_ledger(f))
            n_total += n
            files.append({"path": str(f), "n_records": n,
                          "modified_utc": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
        manifests[split] = {"manifest": str(path), "files": files, "n_records": sum(f["n_records"] for f in files)}

    rows_match, mismatches = {}, []
    if rows_read:
        for split, n_read in rows_read.items():
            if manifest_rows and split in manifest_rows:
                n_manifest = manifest_rows[split]
            elif split in manifest_paths and Path(manifest_paths[split]).exists():
                n_manifest = int(len(pd.read_csv(manifest_paths[split], usecols=[0], low_memory=False)))
            else:
                n_manifest = None
            ok = n_manifest is not None and int(n_read) == int(n_manifest)
            rows_match[split] = {"rows_read": int(n_read), "manifest_rows": n_manifest, "match": ok}
            if not ok:
                mismatches.append(split)

    return {
        "policy": "failure budget zero (docs/2026-09-10_34a_loader_contract.md)",
        "manifests": manifests,
        "n_records_total": n_total,
        "rows_read": rows_match if rows_read else {},
        "rows_read_given": bool(rows_read),
        "rows_read_mismatch": mismatches,
        "ledger_gate_passed": n_total == 0 and not mismatches,
    }


def verify_ledger(manifest_paths: dict, max_listed: int = 20) -> dict:
    """Raise RuntimeError listing the offending records when any ledger beside a manifest is non-empty."""
    offending = []
    for split, path in manifest_paths.items():
        for f in ledger_files(path):
            for record in read_ledger(f):
                record["split"] = split
                offending.append(record)
    if offending:
        lines = [f"Rejection ledger is not empty: {len(offending)} record(s) beside "
                 f"{sorted({str(p) for p in manifest_paths.values()})}; the failure budget is zero."]
        for record in offending[:max_listed]:
            lines.append("  " + json.dumps(jsonable(record), sort_keys=True))
        if len(offending) > max_listed:
            lines.append(f"  ... {len(offending) - max_listed} more")
        lines.append("Inspect each record, repair the manifest or record an explicit exclusion, "
                     "archive the ledger files, and run again.")
        raise RuntimeError("\n".join(lines))
    return {split: [str(f) for f in ledger_files(path)] for split, path in manifest_paths.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Card assembly, writing, finalising, checking
# ──────────────────────────────────────────────────────────────────────────────

def build_run_card(config: dict, config_path, manifest_paths: dict, rows_read: dict | None = None,
                   extra: dict | None = None) -> dict:
    """Assemble the start-of-run card. `manifest_paths` maps split name -> CSV path
    (train and val required by `check_run_card`; add `dev` for the development excerpt)."""
    log_cfg = config.get("logging", {}) or {}
    manifests = {split: manifest_summary(path) for split, path in manifest_paths.items()}
    manifest_rows = {s: m["n_rows"] for s, m in manifests.items() if m.get("exists")}
    card = {
        "card_version": CARD_VERSION,
        "status": "started",
        "run_name": log_cfg.get("run_name"),
        "started_utc": utc_now(),
        "config": config_section(config, config_path),
        "loader_contract_version": CONTRACT_VERSION,
        "arrival_schema_version": SCHEMA_VERSION,
        "manifests": manifests,
        "exclusion_bundle": exclusion_bundle(),
        "ledger": ledger_summary(manifest_paths, rows_read, manifest_rows),
        "versions": package_versions(),
        "git": git_state(),
        "extra": dict(extra or {}),
        "finished_utc": None,
        "best_epoch": None,
        "dev_metric": None,
        "checkpoint": None,
    }
    return jsonable(card)


def write_run_card(card: dict, results_dir) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / CARD_NAME
    path.write_text(json.dumps(jsonable(card), indent=2) + "\n")
    return path


def load_run_card(path) -> dict:
    return json.loads(Path(path).read_text())


def finalize_run_card(path, best_epoch, dev_metric: dict, checkpoint_path) -> dict:
    """Add end time, chosen epoch, the development metric dict at that epoch and the checkpoint sha256."""
    path = Path(path)
    card = load_run_card(path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                      "size_bytes": checkpoint_path.stat().st_size}
    else:
        checkpoint = {"path": str(checkpoint_path) if checkpoint_path else None, "sha256": None,
                      "reason": "checkpoint not found"}
    card.update({
        "status": "finished",
        "finished_utc": utc_now(),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "dev_metric": jsonable(dict(dev_metric)) if dev_metric is not None else None,
        "checkpoint": checkpoint,
    })
    path.write_text(json.dumps(jsonable(card), indent=2) + "\n")
    return card


def _lookup(card: dict, dotted: str):
    node = card
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def check_run_card(path, require_end: bool = True) -> list:
    """Return the list of missing or failing fields (empty when the card is complete)."""
    card = load_run_card(path)
    missing = []
    required = list(REQUIRED_START) + (list(REQUIRED_END) if require_end else [])
    for dotted in required:
        found, value = _lookup(card, dotted)
        if not found or value is None:
            missing.append(dotted)
    for dotted in REQUIRED_KEYS_NULLABLE:
        found, _ = _lookup(card, dotted)
        if not found:
            missing.append(dotted)
    found, passed = _lookup(card, "ledger.ledger_gate_passed")
    if found and passed is not True:
        missing.append("ledger.ledger_gate_passed (false: rejections or rows_read mismatch)")
    return missing


def _show(card: dict) -> str:
    lines = [f"run_name        : {card.get('run_name')}",
             f"status          : {card.get('status')}  started {card.get('started_utc')}  finished {card.get('finished_utc')}",
             f"config          : {(card.get('config') or {}).get('path')}  sha256 {(card.get('config') or {}).get('file_sha256')}",
             f"label_policy    : {(card.get('config') or {}).get('label_policy')}  seed {(card.get('config') or {}).get('seed')}",
             f"loader contract : {card.get('loader_contract_version')}  arrival schema {card.get('arrival_schema_version')}",
             f"git             : {(card.get('git') or {}).get('commit')}  dirty={(card.get('git') or {}).get('dirty')}",
             f"exclusion bundle: {(card.get('exclusion_bundle') or {}).get('sha256')}"]
    for split, m in (card.get("manifests") or {}).items():
        comp = m.get("composition") or {}
        lines.append(f"{split:<6} manifest : {m.get('path')}  rows {m.get('n_rows')}  keys_sha256 {str(m.get('keys_sha256'))[:16]}"
                     f"  S-frac {comp.get('s_label_fraction')}  multi-event {comp.get('multi_event_fraction')}")
        lines.append(f"       by source  : {m.get('rows_by_dataset')}")
        lines.append(f"       by distance: {comp.get('by_distance_bin')}")
    ledger = card.get("ledger") or {}
    lines.append(f"ledger          : {ledger.get('n_records_total')} record(s); rows_read {ledger.get('rows_read')}; "
                 f"gate passed = {ledger.get('ledger_gate_passed')}")
    versions = card.get("versions") or {}
    lines.append("versions        : " + ", ".join(f"{k} {versions.get(k)}" for k in
                                                 ("python", "torch", "seisbench", "numpy", "scipy", "pandas")))
    lines.append(f"best_epoch      : {card.get('best_epoch')}  dev_metric {card.get('dev_metric')}")
    lines.append(f"checkpoint      : {card.get('checkpoint')}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="validate required fields; exit 2 when any is missing")
    p_check.add_argument("card")
    p_check.add_argument("--start-only", action="store_true", help="do not require the end-of-run fields")
    p_show = sub.add_parser("show", help="print a summary of the card")
    p_show.add_argument("card")
    p_show.add_argument("--json", action="store_true", help="print the raw JSON instead")
    args = parser.parse_args(argv)

    if args.command == "check":
        missing = check_run_card(args.card, require_end=not args.start_only)
        if missing:
            print(f"{args.card}: INCOMPLETE, {len(missing)} missing or failing field(s):")
            for m in missing:
                print(f"  - {m}")
            return 2
        print(f"{args.card}: complete")
        return 0
    card = load_run_card(args.card)
    print(json.dumps(card, indent=2) if args.json else _show(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
