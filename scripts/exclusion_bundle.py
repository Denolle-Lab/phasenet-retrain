#!/usr/bin/env python3
"""
exclusion_bundle.py

Versioned exclusion bundle and (dataset, chunk, trace_name) identity for every
data path (issue #33, checkpoint 33A). Pure pandas/numpy: imports nothing from
SeisBench, torch or h5py, so it runs and is tested on a laptop without the
cache. Window geometry and the year hold-out come from
scripts/heldout_sequences.py; nothing of that is duplicated here.

The bundle, data/exclusions/bundle.json, records the sha256 and row count of
every input that defines an exclusion, so a builder can prove which rule set,
which committed lists and which source-metadata snapshot it applied:

  rules                  WINDOWS and HOLDOUT_YEARS of heldout_sequences.py,
                         embedded verbatim with their canonical-JSON sha256
  heldout_sequences_csv  data/exclusions/heldout_sequences.csv, the trace list
                         written on the server by audit_heldout_sequences.py
  benchmark_manifest     notebooks/benchmark_manifest.csv, the trace source of
                         build_training_dataset.load_benchmark_exclusions
  label_error_reports    the Aguilar multiplet reports that
                         build_training_dataset.load_label_error_exclusions reads
  evaluation_suites      configs/evaluation_suites.json (suite roles, 44A)
  sources                metadata*.csv of every training source under
                         SEISBENCH_CACHE_ROOT; null when absent on this machine
  quarantine_policy      what happens to rows whose independence cannot be
                         tested (no origin time or location; for noise rows,
                         no station location or trace start time)

`certified` is true only when the sequence list was present and every source
snapshot was hashed. A laptop build without the cache is never certified.
`sha256` is the self hash over everything except `sha256` and `provenance`
(git commit, creation time, host), so a rebuild from the same inputs yields
the same hash and any change to a rule or a list yields a new one.

Trace identity is (dataset, chunk, trace_name), chunk "" when the dataset has
none. A listed row with chunk "" matches every chunk of that trace name in
that dataset (conservative). Event identity for split assignment is the
event_keys.py fingerprint plus origin coincidence within `time_tol_s` and
`dist_tol_deg` across datasets (event_groups, origin_unions).

Usage:
  python scripts/exclusion_bundle.py build [--allow-missing-sequence-list] [--cache-root DIR] [--allow-unknown]
  python scripts/exclusion_bundle.py check data/manifests_v4/train.csv [--kind signal]   # read-only; exit 2 on a violation
  python scripts/exclusion_bundle.py show
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import heldout_sequences as hs  # noqa: E402  (pure pandas)

BUNDLE_VERSION = 1
BUNDLE_PATH = REPO_ROOT / "data" / "exclusions" / "bundle.json"
USER_LABEL_ERROR_CACHE = Path(os.path.expanduser("~/.cache/phasenet_retrain/label_errors"))

KINDS = ("signal", "validation", "noise", "augmentation", "mining")
FLAG_COL = "independence_unverified"

# The `name` of every entry of DATASET_CONFIGS in build_training_dataset.py
# (that module imports seisbench, so the names are repeated here). The noise
# sources of build_noise_dataset.py (stead, lendb, txed, vcseis, obst2024)
# are a subset.
SOURCE_DATASETS = (
    "geofon", "stead", "ceed", "instancecounts", "mlaapde", "ethz", "crew",
    "cwa", "iquique", "txed", "pnw", "lendb", "pisdl", "vcseis", "aq2009gm",
    "obst2024", "scedc", "meier2019jgr", "ross2018gpd", "obs",
)

# Column candidates for noise rows: station location and trace start time
# (SeisBench metadata first, then the data/noise_*/metadata.csv names).
STATION_LAT_COLS = ("station_latitude_deg", "latitude", "station_latitude")
STATION_LON_COLS = ("station_longitude_deg", "longitude", "station_longitude")
START_COLS = ("trace_start_time", "starttime", "start_time")


class StaleBundleError(RuntimeError):
    """An input the bundle certifies differs from what is on disk or in code."""


class MissingExclusionInputError(FileNotFoundError):
    """The bundle, or an input it records as present, is absent."""


class UncertifiedBundleError(RuntimeError):
    """The bundle exists and is consistent but does not certify a source snapshot."""


class ManifestSchemaError(ValueError):
    """An append would drop a column of the new rows (the manifest header lacks it)."""


# ── hashing ────────────────────────────────────────────────────────────────────

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def file_record(path, count_rows: bool = True, block: int = 1 << 20) -> dict:
    """sha256, byte size and (for CSV lists) data-row count of one file, in one pass."""
    h = hashlib.sha256()
    n_lines = size = 0
    last = b""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
            size += len(chunk)
            if count_rows:
                n_lines += chunk.count(b"\n")
                last = chunk[-1:]
    rec = {"sha256": h.hexdigest(), "bytes": size}
    if count_rows:
        if size and last != b"\n":
            n_lines += 1
        rec["n_rows"] = max(n_lines - 1, 0)
    return rec


def rules_record() -> dict:
    """WINDOWS and HOLDOUT_YEARS of heldout_sequences.py, canonicalised, with their sha256."""
    windows = [{k: w[k] for k in sorted(w)} for w in hs.WINDOWS]
    years = sorted(int(y) for y in hs.HOLDOUT_YEARS)
    body = {"windows": windows, "holdout_years": years}
    return {**body, "n_windows": len(windows), "sha256": sha256_text(canonical_json(body))}


def self_hash(bundle: dict) -> str:
    """sha256 of the canonical JSON of everything except `sha256` and `provenance`."""
    return sha256_text(canonical_json({k: v for k, v in bundle.items() if k not in ("sha256", "provenance")}))


def quarantine_policy(allow_unknown: bool = False) -> dict:
    return {
        "allow_unknown": bool(allow_unknown),
        "flag_column": FLAG_COL,
        "unknown_signal": "source origin time, latitude or longitude missing",
        "unknown_noise": "station latitude/longitude or trace start time missing",
        "rule": (
            "Unknown rows cannot be tested against the windows or the year hold-out. "
            "allow_unknown false: dropped and counted as n_quarantined_unknown. "
            "allow_unknown true: kept with the flag column set, counted as "
            "n_unknown_kept_flagged, and outside every independence claim. "
            "A row whose location alone lies in an all-time place window is "
            "excluded whatever its time."
        ),
    }


# ── layout and inputs ──────────────────────────────────────────────────────────

def _layout(repo_root=None, label_error_dirs=None) -> dict:
    repo_root = Path(REPO_ROOT if repo_root is None else repo_root).resolve()
    if label_error_dirs is None:
        # The same search order as build_training_dataset.load_label_error_exclusions:
        # data/labelerrors first, then label_error_filter's download cache.
        label_error_dirs = (repo_root / "data" / "labelerrors", USER_LABEL_ERROR_CACHE)
    return {
        "repo_root": repo_root,
        "sequence_csv": repo_root / "data" / "exclusions" / "heldout_sequences.csv",
        "benchmark_csv": repo_root / "notebooks" / "benchmark_manifest.csv",
        "suites_json": repo_root / "configs" / "evaluation_suites.json",
        "label_error_dirs": [Path(d) for d in label_error_dirs],
    }


def _rel(path, repo_root=None) -> str:
    repo_root = REPO_ROOT if repo_root is None else repo_root
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _abs(rel, repo_root=None) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(REPO_ROOT if repo_root is None else repo_root) / p


def _label_error_stems() -> list:
    import label_error_filter as lef  # pure pandas
    return sorted(lef.REPORT_STEM_TO_DATASET)


def _find_label_error_report(stem, dirs):
    for d in dirs:
        p = Path(d) / f"{stem}_report.csv"
        if p.is_file():
            return p
    return None


def _policy_sha256(text: str) -> str:
    # Same expression as evaluation_policy.record_access, so this equals the
    # policy_sha256 written to data/evaluation/access.jsonl.
    return hashlib.sha256(json.dumps(json.loads(text), sort_keys=True).encode()).hexdigest()


def _source_record(cache_root, name):
    """Hashes of the complete metadata CSVs of one SeisBench source, or None when absent.
    Same file selection as build_training_dataset._load_chunked_meta (no .partial shards)."""
    d = Path(cache_root) / "datasets" / name
    if not d.is_dir():
        return None
    files = sorted(p for p in d.iterdir() if p.name.startswith("metadata") and p.suffix == ".csv")
    if not files:
        return None
    partial = sorted(p.name for p in d.iterdir() if p.name.startswith("metadata") and p.name.endswith(".partial"))
    return {"dir": str(d), "files": {p.name: file_record(p) for p in files}, "partial_shards": partial}


def _git_commit(repo_root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _absent(path_rel):
    return {"path": path_rel, "present": False, "sha256": None, "bytes": None, "n_rows": None}


def _default_cache_root(cache_root=None) -> Path:
    if cache_root:
        return Path(cache_root)
    return Path(os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench")))


# ── build / write / verify / load ──────────────────────────────────────────────

def build_bundle(*, allow_missing_sequence_list: bool = False, cache_root=None, allow_unknown: bool = False,
                 repo_root=None, label_error_dirs=None) -> dict:
    """Hash every exclusion-defining input and return the bundle dict (not written).

    Raises MissingExclusionInputError when data/exclusions/heldout_sequences.csv
    is absent, unless `allow_missing_sequence_list`, which records it as absent
    and leaves the bundle uncertified. configs/evaluation_suites.json is always
    required.
    """
    lay = _layout(repo_root, label_error_dirs)
    root = lay["repo_root"]
    cache_root = _default_cache_root(cache_root)

    inputs = {}
    seq = lay["sequence_csv"]
    if seq.is_file():
        inputs["heldout_sequences_csv"] = {"path": _rel(seq, root), "present": True, **file_record(seq)}
    elif allow_missing_sequence_list:
        inputs["heldout_sequences_csv"] = _absent(_rel(seq, root))
    else:
        raise MissingExclusionInputError(
            f"Held-out sequence exclusion list not found at {seq}. Run "
            "`python scripts/audit_heldout_sequences.py` on the server and commit its output, "
            "or pass --allow-missing-sequence-list to record an uncertified bundle.")

    bm = lay["benchmark_csv"]
    inputs["benchmark_manifest"] = ({"path": _rel(bm, root), "present": True, **file_record(bm)}
                                    if bm.is_file() else _absent(_rel(bm, root)))

    reports = {}
    for stem in _label_error_stems():
        p = _find_label_error_report(stem, lay["label_error_dirs"])
        reports[stem] = ({"path": _rel(p, root), "present": True, **file_record(p)} if p else _absent(None))
    inputs["label_error_reports"] = reports

    sj = lay["suites_json"]
    if not sj.is_file():
        raise MissingExclusionInputError(f"Evaluation suite policy not found at {sj} (checkpoint 44A)")
    text = sj.read_text()
    policy = json.loads(text)
    inputs["evaluation_suites"] = {
        "path": _rel(sj, root), "present": True, **file_record(sj, count_rows=False),
        "policy_sha256": _policy_sha256(text), "policy_version": policy.get("version"),
        "n_roles": len(policy.get("roles") or {}), "acceptance_status": policy.get("acceptance_status"),
    }

    sources = {name: _source_record(cache_root, name) for name in SOURCE_DATASETS}
    uncertified = sorted(n for n, r in sources.items() if r is None)
    seq_present = bool(inputs["heldout_sequences_csv"]["present"])

    bundle = {
        "version": BUNDLE_VERSION,
        "rules": rules_record(),
        "inputs": inputs,
        "sources": sources,
        "uncertified_sources": uncertified,
        "quarantine_policy": quarantine_policy(allow_unknown),
        "sequence_list_present": seq_present,
        "certified": bool(seq_present and not uncertified),
        "provenance": {
            "git_commit": _git_commit(root),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cache_root": str(cache_root),
            "hostname": socket.gethostname(),
            "builder": "scripts/exclusion_bundle.py",
        },
    }
    bundle["sha256"] = self_hash(bundle)
    return bundle


def write_bundle(bundle: dict, path=None) -> Path:
    path = Path(BUNDLE_PATH if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def verify_bundle(bundle: dict, *, repo_root=None, label_error_dirs=None, cache_root=None,
                  verify_sources: bool = True) -> dict:
    """Recompute every hash the bundle records for what exists on this machine.

    Returns {"checked", "unchecked", "stale", "missing"}: lists of messages.
    stale: a rule, list or source present here differs from the bundle, or a
    list absent at build time is present now. missing: a list recorded as
    present is absent. Source snapshots absent here are unchecked (they are
    certification, not exclusion definitions). Raises nothing.
    """
    lay = _layout(repo_root, label_error_dirs)
    root = lay["repo_root"]
    out = {"checked": [], "unchecked": [], "stale": [], "missing": []}

    def compare(label, rec, path):
        if path is not None and Path(path).is_file():
            if not rec.get("present"):
                out["stale"].append(f"{label}: {path} is present now but the bundle recorded it absent; rebuild the bundle")
                return
            now = file_record(path, count_rows=False)["sha256"]
            if now != rec.get("sha256"):
                out["stale"].append(f"{label}: sha256 {now[:12]} on disk != {str(rec.get('sha256'))[:12]} in the bundle ({path})")
            else:
                out["checked"].append(f"{label}: {path}")
        elif rec.get("present"):
            out["missing"].append(f"{label}: {path} recorded present in the bundle, absent now")
        else:
            out["unchecked"].append(f"{label}: absent at build time and now")

    cur = rules_record()
    if cur["sha256"] != bundle["rules"]["sha256"]:
        out["stale"].append("rules: WINDOWS/HOLDOUT_YEARS in heldout_sequences.py differ from the bundle; rebuild it")
    else:
        out["checked"].append("rules: heldout_sequences.py WINDOWS/HOLDOUT_YEARS")

    inp = bundle["inputs"]
    compare("heldout_sequences_csv", inp["heldout_sequences_csv"], _abs(inp["heldout_sequences_csv"]["path"], root))
    compare("benchmark_manifest", inp["benchmark_manifest"], _abs(inp["benchmark_manifest"]["path"], root))
    compare("evaluation_suites", inp["evaluation_suites"], _abs(inp["evaluation_suites"]["path"], root))
    for stem, rec in inp["label_error_reports"].items():
        p = _abs(rec["path"], root) if rec.get("path") else None
        if p is None or not p.is_file():
            p = _find_label_error_report(stem, lay["label_error_dirs"])
        if p is None and rec.get("present"):
            out["unchecked"].append(f"label_error_reports/{stem}: not cached on this machine")
            continue
        compare(f"label_error_reports/{stem}", rec, p)

    if verify_sources:
        cr = _default_cache_root(cache_root)
        for name, rec in bundle["sources"].items():
            if rec is None:
                out["unchecked"].append(f"sources/{name}: not hashed at build time (uncertified)")
                continue
            now = _source_record(cr, name)
            if now is None:
                out["unchecked"].append(f"sources/{name}: not under {cr} on this machine")
                continue
            if {k: v["sha256"] for k, v in now["files"].items()} != {k: v["sha256"] for k, v in rec["files"].items()}:
                out["stale"].append(f"sources/{name}: metadata files under {cr} differ from the bundle snapshot")
            else:
                out["checked"].append(f"sources/{name}: {len(now['files'])} metadata file(s)")
    return out


def load_bundle(path=None, require_certified: bool = False, *, verify_sources: bool = True,
                repo_root=None, label_error_dirs=None, cache_root=None, report: dict = None) -> dict:
    """Load and validate the bundle; fail closed.

    Raises MissingExclusionInputError (no bundle, or a list it records as present
    is absent), ValueError (version), StaleBundleError (self hash, rules, a list
    or a local source snapshot differ from the bundle), UncertifiedBundleError
    (`require_certified` and the bundle does not certify the sequence list plus
    every source snapshot). The sequence list may be absent only when the
    bundle recorded it absent and `require_certified` is False. `report`, if a
    dict, receives the verify_bundle output.
    """
    path = Path(BUNDLE_PATH if path is None else path)
    if not path.is_file():
        raise MissingExclusionInputError(
            f"Exclusion bundle not found at {path}. Build it with `python scripts/exclusion_bundle.py build` "
            "(on the server, after data/exclusions/heldout_sequences.csv is committed). Nothing is built without it.")
    bundle = json.loads(path.read_text())
    if bundle.get("version") != BUNDLE_VERSION:
        raise ValueError(f"{path}: bundle version {bundle.get('version')!r}, this code reads {BUNDLE_VERSION}")
    if bundle.get("sha256") != self_hash(bundle):
        raise StaleBundleError(f"{path}: self sha256 does not match its content (edited or truncated); rebuild it")
    rep = verify_bundle(bundle, repo_root=repo_root, label_error_dirs=label_error_dirs,
                        cache_root=cache_root, verify_sources=verify_sources)
    if report is not None:
        report.update(rep)
    if rep["stale"]:
        raise StaleBundleError(f"{path} is stale:\n  " + "\n  ".join(rep["stale"]))
    if rep["missing"]:
        raise MissingExclusionInputError(f"{path}: inputs missing:\n  " + "\n  ".join(rep["missing"]))
    if require_certified and not bundle.get("certified"):
        why = []
        if not bundle.get("sequence_list_present"):
            why.append("sequence list absent at build time")
        if bundle.get("uncertified_sources"):
            why.append("sources not hashed: " + ", ".join(bundle["uncertified_sources"]))
        raise UncertifiedBundleError(f"{path} is not certified ({'; '.join(why)}); rebuild it on the server")
    return bundle


# ── trace identity ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _cached_trace_exclusions(path_str: str, sha256):
    path = Path(path_str)
    if sha256 is not None and file_record(path, count_rows=False)["sha256"] != sha256:
        raise StaleBundleError(f"{path}: sha256 differs from the bundle; rebuild the bundle")
    return hs.load_sequence_exclusions(path, required=True, chunk_aware=True)


def trace_exclusions(bundle: dict, repo_root=None):
    """{dataset: {trace_name: frozenset(chunks)}} from the sequence list the
    bundle certifies ("" = every chunk), or None when the bundle recorded the
    list as absent. The file hash is checked against the bundle."""
    rec = bundle["inputs"]["heldout_sequences_csv"]
    if not rec.get("present"):
        return None
    return _cached_trace_exclusions(str(_abs(rec["path"], repo_root)), rec.get("sha256"))


def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def apply_exclusions(df: pd.DataFrame, bundle: dict, *, kind: str, dataset=None, dataset_col=None,
                     exclusions=None, time_col=hs.TIME_COL, lat_col=hs.LAT_COL, lon_col=hs.LON_COL,
                     station_lat_col=None, station_lon_col=None, start_col=None,
                     trace_col="trace_name", chunk_col="chunk", repo_root=None):
    """Drop the rows of `df` that the bundle excludes; return (kept, report). `df` is not modified.

    kind signal | validation | augmentation | mining: a row is removed when its
    (dataset, chunk, trace_name) is listed, when its source origin (time_col,
    lat_col, lon_col) lies in a window (time windows and all-time places), or
    when its origin year is in HOLDOUT_YEARS.
    kind noise: the same windows applied to the STATION location
    (station_lat_col/station_lon_col, auto-detected among STATION_*_COLS) and
    the trace start time (start_col, auto-detected among START_COLS); all-time
    places by location alone, time windows by location and start time; the
    year hold-out on the start-time year.
    Rows whose time or location is missing are quarantined: dropped and counted
    as n_quarantined_unknown, or, when the bundle's quarantine policy allows
    unknown rows, kept with the flag column True and counted as
    n_unknown_kept_flagged. The flag column is always present in `kept`.
    One reason per row, in the order trace_listed > in_window > year_holdout >
    unknown; `windows` counts hits per window over all input rows.
    `dataset` names the dataset of every row; otherwise `dataset_col`
    (default dataset_name, then dataset). `exclusions` overrides the list the
    bundle certifies (fixtures); None reads it from the bundle.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    n = len(df)
    policy = bundle.get("quarantine_policy") or {}
    allow_unknown = bool(policy.get("allow_unknown", False))
    flag_col = policy.get("flag_column", FLAG_COL)

    if dataset is not None:
        ds = pd.Series([str(dataset)] * n, index=df.index, dtype=object)
    else:
        col = dataset_col or _first_present(df, ("dataset_name", "dataset"))
        ds = df[col].astype(str) if col else None

    if exclusions is None:
        exclusions = trace_exclusions(bundle, repo_root=repo_root)
    listed = np.zeros(n, dtype=bool)
    list_checked = exclusions is not None and ds is not None and trace_col in df.columns
    if list_checked and n:
        chunks = (df[chunk_col].fillna("").astype(str) if chunk_col in df.columns
                  else pd.Series([""] * n, index=df.index, dtype=object))
        names = df[trace_col].astype(str)
        for d in pd.unique(ds):
            bad = exclusions.get(d)
            if not bad:
                continue
            m = (ds == d).to_numpy()
            listed[m] = hs.listed_mask(names[m], chunks[m], bad)

    if kind == "noise":
        lat_c = station_lat_col or _first_present(df, STATION_LAT_COLS)
        lon_c = station_lon_col or _first_present(df, STATION_LON_COLS)
        t_c = start_col or _first_present(df, START_COLS)
    else:
        lat_c, lon_c, t_c = lat_col, lon_col, time_col

    def _num(c):
        if c and c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    if t_c and t_c in df.columns:
        t = hs._to_utc(df[t_c])
    else:
        t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    lat, lon = _num(lat_c), _num(lon_c)
    hits = hs.window_hits(t, lat, lon)
    in_window = hits.to_numpy(dtype=bool).any(axis=1) if n else np.zeros(0, dtype=bool)
    year_hold = hs.holdout_year_mask(t).to_numpy(dtype=bool)
    unknown = (t.isna() | lat.isna() | lon.isna()).to_numpy(dtype=bool)

    reason = np.full(n, "", dtype=object)
    reason[unknown] = "unknown"
    reason[year_hold] = "year_holdout"
    reason[in_window] = "in_window"
    reason[listed] = "trace_listed"

    is_unknown = reason == "unknown"
    drop = (reason != "") & ~(is_unknown & allow_unknown)
    kept = df.loc[~drop].copy()
    kept[flag_col] = is_unknown[~drop]

    if dataset is not None:
        ds_label = str(dataset)
    else:
        ds_label = sorted(pd.unique(ds).tolist()) if ds is not None else None
    report = {
        "kind": kind,
        "dataset": ds_label,
        "n_input": int(n),
        "n_kept": int(len(kept)),
        "n_removed": int(drop.sum()),
        "n_trace_listed": int((reason == "trace_listed").sum()),
        "n_in_window": int((reason == "in_window").sum()),
        "n_year_holdout": int((reason == "year_holdout").sum()),
        "n_quarantined_unknown": int((is_unknown & ~allow_unknown).sum()),
        "n_unknown_kept_flagged": int((is_unknown & allow_unknown).sum()),
        "windows": {w: int(c) for w, c in hits.sum().items() if int(c)} if n else {},
        "trace_list_checked": bool(list_checked),
        "columns": {"time": t_c, "lat": lat_c, "lon": lon_c},
        "allow_unknown": allow_unknown,
        "bundle_sha256": bundle.get("sha256"),
    }
    return kept, report


# ── event identity ─────────────────────────────────────────────────────────────

class UnionFind:
    """Union-find over positions 0..n-1 (path halving)."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=int)


def parse_event_keys(values) -> list:
    """Per-row frozenset of event keys from frozensets, lists, ';'-joined strings or missing values."""
    out = []
    for v in values:
        if isinstance(v, (set, frozenset, list, tuple)):
            out.append(frozenset(str(k) for k in v if str(k)))
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(frozenset())
        else:
            s = str(v).strip()
            out.append(frozenset(k.strip() for k in s.split(";") if k.strip()) if s else frozenset())
    return out


def origin_unions(origin_time, lat, lon, time_tol_s: float = 2.0, dist_tol_deg: float = 0.1) -> np.ndarray:
    """Positional pairs (i, j), shape (m, 2), whose union connects every row
    sharing an origin fingerprint and every pair of distinct fingerprints
    within time_tol_s and dist_tol_deg. Rows with a missing time or location
    join nothing. Identical fingerprints are collapsed first, so an event
    with many stations costs one pair per row, not one per station pair."""
    t = hs._to_utc(pd.Series(origin_time).reset_index(drop=True))
    la = pd.to_numeric(pd.Series(lat).reset_index(drop=True), errors="coerce")
    lo = pd.to_numeric(pd.Series(lon).reset_index(drop=True), errors="coerce")
    valid = (t.notna() & la.notna() & lo.notna()).to_numpy(dtype=bool)
    empty = np.zeros((0, 2), dtype=int)
    if not valid.any():
        return empty
    pos = np.flatnonzero(valid)
    sec = (t[valid].dt.tz_convert(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float)
    la_v = la.to_numpy(dtype=float)[valid]
    lo_v = lo.to_numpy(dtype=float)[valid]

    fp = pd.DataFrame({"s": sec, "la": la_v, "lo": lo_v})
    gid = fp.groupby(["s", "la", "lo"], sort=False).ngroup().to_numpy()
    idx = np.arange(len(gid))
    first_idx = pd.Series(idx).groupby(gid).transform("min").to_numpy()
    dup = first_idx != idx
    same = np.column_stack([pos[first_idx[dup]], pos[idx[dup]]]) if dup.any() else empty

    u_idx = np.unique(first_idx)
    order = u_idx[np.argsort(sec[u_idx], kind="stable")]
    us, ula, ulo, upos = sec[order], la_v[order], lo_v[order], pos[order]
    m = len(us)
    ends = np.searchsorted(us, us + time_tol_s, side="right")
    starts = np.arange(m) + 1
    counts = np.maximum(ends - starts, 0)
    total = int(counts.sum())
    if total == 0:
        near = empty
    else:
        i_idx = np.repeat(np.arange(m), counts)
        j_idx = np.repeat(starts, counts) + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
        d = hs.gc_distance_deg(ula[i_idx], ulo[i_idx], ula[j_idx], ulo[j_idx])
        ok = d <= dist_tol_deg
        near = np.column_stack([upos[i_idx[ok]], upos[j_idx[ok]]]) if ok.any() else empty
    return np.vstack([same, near]).astype(int)


def event_groups(frame: pd.DataFrame, *, key_col="event_keys", time_col=hs.TIME_COL, lat_col=hs.LAT_COL,
                 lon_col=hs.LON_COL, time_tol_s: float = 2.0, dist_tol_deg: float = 0.1) -> np.ndarray:
    """Dense group id per row (positional). Rows sharing an event key (key_col:
    frozensets or ';'-joined strings, possibly empty) or whose origins
    coincide within the tolerances share a group, across datasets."""
    n = len(frame)
    uf = UnionFind(n)
    if key_col in frame.columns:
        first = {}
        for pos, ks in enumerate(parse_event_keys(frame[key_col].tolist())):
            for k in ks:
                if k in first:
                    uf.union(pos, first[k])
                else:
                    first[k] = pos
    if all(c in frame.columns for c in (time_col, lat_col, lon_col)):
        for i, j in origin_unions(frame[time_col], frame[lat_col], frame[lon_col], time_tol_s, dist_tol_deg):
            uf.union(int(i), int(j))
    return pd.factorize(uf.labels())[0]


def cross_source_duplicates(frame: pd.DataFrame, time_tol_s: float = 2.0, dist_tol_deg: float = 0.1, *,
                            dataset_col=None, key_col="event_keys", time_col=hs.TIME_COL,
                            lat_col=hs.LAT_COL, lon_col=hs.LON_COL) -> pd.DataFrame:
    """Rows of `frame` that share an event with a row of ANOTHER dataset, with
    `group_id` (dense, over the whole frame) and `n_datasets`; empty when none."""
    col = dataset_col or _first_present(frame, ("dataset", "dataset_name"))
    if col is None:
        raise ValueError("frame needs a dataset or dataset_name column")
    out = frame.copy()
    out["group_id"] = event_groups(frame, key_col=key_col, time_col=time_col, lat_col=lat_col,
                                   lon_col=lon_col, time_tol_s=time_tol_s, dist_tol_deg=dist_tol_deg)
    out["n_datasets"] = out.groupby("group_id")[col].transform("nunique")
    return out[out["n_datasets"] > 1].sort_values(["group_id", col], kind="stable")


def fill_splits_by_group(group_ids, rng, val_frac: float = 0.10, test_frac: float = 0.10) -> np.ndarray:
    """"train"/"val"/"test" per row with every group in one split: groups are
    shuffled, filled into val until val_frac of the rows, then into test until
    test_frac, the rest train (the rule of build_training_dataset.assign_splits)."""
    group_ids = np.asarray(group_ids)
    m = len(group_ids)
    groups = {}
    for pos, gid in enumerate(group_ids):
        groups.setdefault(gid, []).append(pos)
    keys = list(groups)
    rng.shuffle(keys)
    assign = np.full(m, "train", dtype=object)
    gi = 0
    for label, target in (("val", int(val_frac * m)), ("test", int(test_frac * m))):
        count = 0
        while gi < len(keys) and count < target:
            for pos in groups[keys[gi]]:
                assign[pos] = label
            count += len(groups[keys[gi]])
            gi += 1
    return assign


# ── waveform interval overlap ──────────────────────────────────────────────────

OVERLAP_COLS = ["station", "left", "right", "dataset_left", "dataset_right", "role_left", "role_right", "overlap_s"]


def overlapping_intervals(frame: pd.DataFrame, *, station_col="station", start_col="start", end_col="end",
                          dataset_col=None, role_col="role") -> pd.DataFrame:
    """Pairs of rows on the same station whose [start, end] intervals overlap
    and that belong to different datasets or different roles (signal/noise).
    Without a dataset and a role column every overlap is flagged. `left` and
    `right` are index labels of `frame`; touching intervals do not overlap."""
    ds_col = dataset_col or _first_present(frame, ("dataset", "dataset_name"))
    has_role = role_col in frame.columns
    st = hs._to_utc(frame[start_col])
    en = hs._to_utc(frame[end_col])
    ok = (st.notna() & en.notna() & frame[station_col].notna()).to_numpy(dtype=bool)
    sub = pd.DataFrame({
        "station": frame.loc[ok, station_col].astype(str).to_numpy(),
        "start": st[ok].dt.tz_convert(None).to_numpy(),
        "end": en[ok].dt.tz_convert(None).to_numpy(),
        "dataset": frame.loc[ok, ds_col].to_numpy() if ds_col else np.full(int(ok.sum()), "", dtype=object),
        "role": frame.loc[ok, role_col].to_numpy() if has_role else np.full(int(ok.sum()), "", dtype=object),
    }, index=frame.index[ok])
    flag_all = ds_col is None and not has_role
    rows = []
    for sta, g in sub.groupby("station", sort=False):
        g = g.sort_values("start", kind="stable")
        idx, s, e = g.index.to_numpy(), g["start"].to_numpy(), g["end"].to_numpy()
        d, r = g["dataset"].to_numpy(), g["role"].to_numpy()
        for i in range(len(g)):
            j = i + 1
            while j < len(g) and s[j] < e[i]:
                if flag_all or d[i] != d[j] or r[i] != r[j]:
                    ov = (min(e[i], e[j]) - s[j]) / np.timedelta64(1, "s")
                    rows.append(dict(station=sta, left=idx[i], right=idx[j], dataset_left=d[i], dataset_right=d[j],
                                     role_left=r[i], role_right=r[j], overlap_s=float(ov)))
                j += 1
    return pd.DataFrame(rows, columns=OVERLAP_COLS)


# ── builder helpers ────────────────────────────────────────────────────────────

def assert_not_historical(path, checksums_csv=None, repo_root=None) -> None:
    """Refuse to modify a manifest listed in data/manifest_checksums.csv (historical manifests are immutable)."""
    repo_root = REPO_ROOT if repo_root is None else repo_root
    checksums_csv = Path(checksums_csv) if checksums_csv else Path(repo_root) / "data" / "manifest_checksums.csv"
    if not checksums_csv.is_file():
        return
    rel = _rel(path, repo_root)
    listed = set(pd.read_csv(checksums_csv, usecols=["manifest"])["manifest"].astype(str))
    if rel in listed:
        raise PermissionError(f"{rel} is a historical manifest listed in {checksums_csv.name} and is immutable; "
                              "build a new manifest directory instead")


def assert_append_columns(manifest_columns, row_columns, path) -> None:
    """Refuse an append whose rows carry a column the manifest at `path` lacks.

    The append scripts write new rows in the manifest's own column order, so a
    column absent from the header would be dropped without a trace; for
    FLAG_COL that erases the quarantine signal of an allow_unknown bundle.
    The manifest is not rewritten to gain the column: an append never touches
    an existing row (a pandas round trip would reformat them), so the rows a
    manifest was built with stay byte-identical whether or not it is listed
    in data/manifest_checksums.csv. Remedy: rebuild the manifest with
    scripts/build_training_dataset.py (it writes FLAG_COL since #33A), or
    copy it to a new manifest directory with the column added, False for
    every existing row.
    """
    have = set(map(str, manifest_columns))
    missing = [c for c in row_columns if c not in have]
    if missing:
        what = "the quarantine flag" if missing == [FLAG_COL] else f"column(s) {missing}"
        raise ManifestSchemaError(
            f"{path} has no column {missing}; appending would drop {what} silently. "
            "Existing manifests are never rewritten: rebuild with scripts/build_training_dataset.py "
            f"(writes {FLAG_COL} since #33A) or add the column(s) to a copy in a new manifest "
            f"directory ({FLAG_COL}=False for every existing row).")


def append_provenance(directory, key: str, record: dict) -> Path:
    """Append `record` to the list `key` of <directory>/provenance.json (created when absent)."""
    p = Path(directory) / "provenance.json"
    prov = json.loads(p.read_text()) if p.is_file() else {}
    prov.setdefault(key, []).append(record)
    p.write_text(json.dumps(prov, indent=2, sort_keys=True, default=_json_default) + "\n")
    return p


# ── CLI ────────────────────────────────────────────────────────────────────────

def summary(bundle: dict) -> str:
    inp = bundle["inputs"]
    seq, bm, ev = inp["heldout_sequences_csv"], inp["benchmark_manifest"], inp["evaluation_suites"]
    reports = inp["label_error_reports"]
    n_src = len(bundle["sources"])
    unc = bundle["uncertified_sources"]
    prov = bundle.get("provenance", {})
    lines = [
        f"bundle sha256       : {bundle['sha256']}",
        f"version / certified : {bundle['version']} / {bundle['certified']}",
        f"rules sha256        : {bundle['rules']['sha256']}  ({bundle['rules']['n_windows']} windows, years {bundle['rules']['holdout_years']})",
        f"sequence list       : {('present, %s rows, %s' % (seq['n_rows'], seq['sha256'][:12])) if seq['present'] else 'ABSENT'}  ({seq['path']})",
        f"benchmark manifest  : {('present, %s rows, %s' % (bm['n_rows'], bm['sha256'][:12])) if bm['present'] else 'ABSENT'}  ({bm['path']})",
        f"label-error reports : {sum(1 for r in reports.values() if r['present'])}/{len(reports)} present",
        f"suite policy        : {ev['policy_sha256'][:12]} (version {ev['policy_version']}, {ev['n_roles']} roles, {ev['acceptance_status']})",
        f"sources hashed      : {n_src - len(unc)}/{n_src}" + (f"; uncertified: {', '.join(unc)}" if unc else ""),
        f"quarantine policy   : allow_unknown={bundle['quarantine_policy']['allow_unknown']}",
        f"git commit / host   : {prov.get('git_commit')} / {prov.get('hostname')}",
        f"created (UTC)       : {prov.get('created_utc')}  cache_root={prov.get('cache_root')}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    b = sub.add_parser("build", help="hash every exclusion input and write the bundle")
    b.add_argument("--allow-missing-sequence-list", action="store_true",
                   help="record data/exclusions/heldout_sequences.csv as absent (bundle stays uncertified)")
    b.add_argument("--cache-root", help="SeisBench cache root (default: $SEISBENCH_CACHE_ROOT or ~/.seisbench)")
    b.add_argument("--allow-unknown", action="store_true",
                   help="quarantine policy: keep rows of unknown independence, flagged, instead of dropping them")
    b.add_argument("--out", default=str(BUNDLE_PATH))
    b.add_argument("--repo-root", default=str(REPO_ROOT))
    b.add_argument("--label-error-dir", action="append", default=None,
                   help="directory holding <stem>_report.csv (repeatable; default: data/labelerrors, then the download cache)")

    c = sub.add_parser("check", help="read-only: what the bundle would exclude from a manifest; exit 2 if any excluded row is present")
    c.add_argument("manifest")
    c.add_argument("--kind", default="signal", choices=KINDS)
    c.add_argument("--bundle", default=str(BUNDLE_PATH))
    c.add_argument("--repo-root", default=str(REPO_ROOT))
    c.add_argument("--label-error-dir", action="append", default=None)
    c.add_argument("--cache-root")
    c.add_argument("--verify-sources", action="store_true", help="also rehash local source metadata (slow)")

    s = sub.add_parser("show", help="print the bundle summary")
    s.add_argument("--bundle", default=str(BUNDLE_PATH))

    args = ap.parse_args(argv)

    if args.cmd == "build":
        bundle = build_bundle(allow_missing_sequence_list=args.allow_missing_sequence_list,
                              cache_root=args.cache_root, allow_unknown=args.allow_unknown,
                              repo_root=args.repo_root, label_error_dirs=args.label_error_dir)
        out = write_bundle(bundle, args.out)
        print(summary(bundle))
        print(f"written             : {out}")
        return 0

    if args.cmd == "show":
        bundle = json.loads(Path(args.bundle).read_text())
        print(summary(bundle))
        print("self hash           : " + ("ok" if self_hash(bundle) == bundle.get("sha256") else "MISMATCH"))
        return 0

    bundle = load_bundle(args.bundle, require_certified=False, verify_sources=args.verify_sources,
                         repo_root=args.repo_root, label_error_dirs=args.label_error_dir, cache_root=args.cache_root)
    df = pd.read_csv(args.manifest, low_memory=False)
    _, rep = apply_exclusions(df, bundle, kind=args.kind, repo_root=args.repo_root)
    rep["manifest"] = str(args.manifest)
    print(json.dumps(rep, indent=2, sort_keys=True))
    violations = rep["n_trace_listed"] + rep["n_in_window"] + rep["n_year_holdout"]
    unknown = rep["n_quarantined_unknown"] + rep["n_unknown_kept_flagged"]
    print(f"{'FAIL' if violations else 'PASS'}: {violations} excluded row(s) present; {unknown} row(s) of unknown independence"
          + ("" if rep["trace_list_checked"] else "; trace list NOT checked (absent from the bundle)"))
    return 2 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
