#!/usr/bin/env python3
"""
hash_manifests.py

The training manifests themselves (data/manifests*/*.csv) are git-ignored --
too large to commit and derived from a local SeisBench cache that varies by
machine. This script instead commits a lightweight fingerprint of each
manifest's exact trace composition, so a regenerated manifest (from
build_training_dataset.py / build_oversampled_manifests.py) can be verified
byte-for-byte identical without re-uploading the manifest itself.

The hash is order-independent: it sorts the (dataset_name, trace_name, chunk)
key tuples before hashing, so it verifies "same set of traces, same
multiplicity" rather than "same row order" (row order is an implementation
detail of pandas' concat/sample, not part of the training-set definition).

Usage:
  python scripts/hash_manifests.py                       # hash the known manifest set, write data/manifest_checksums.csv
  python scripts/hash_manifests.py --check                # re-hash and diff against the committed checksums file
  python scripts/hash_manifests.py path/to/one_manifest.csv   # print a single manifest's hash
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKSUMS_PATH = REPO_ROOT / "data" / "manifest_checksums.csv"

KEY_COLS = ["dataset_name", "trace_name", "chunk"]

# Every manifest currently referenced by a configs/*.yaml train/val/test_manifest key.
KNOWN_MANIFESTS = [
    "data/manifests/train.csv",
    "data/manifests/val.csv",
    "data/manifests/test.csv",
    "data/manifests_v2/train.csv",
    "data/manifests_v2/val.csv",
    "data/manifests_v2/test.csv",
    "data/manifests_v2/train_tele2x.csv",
    "data/manifests_v2/train_v18.csv",
    "data/manifests_v2/train_p_focused.csv",
    "data/manifests_v2/val_p_focused.csv",
    "data/manifests_v2_clean/train.csv",
    "data/manifests_v2_clean/val.csv",
    "data/manifests_v2_clean/test.csv",
    "data/manifests_v2_eventclean/train.csv",
    "data/manifests_v2_eventclean/val.csv",
    "data/manifests_v2_eventclean/test.csv",
    "data/manifests_v3/train.csv",
    "data/manifests_v3/val.csv",
    "data/manifests_v3/test.csv",
    "data/manifests_v3_clean/train.csv",
    "data/manifests_v3_clean/val.csv",
    "data/manifests_v3_clean/test.csv",
]


def hash_manifest(path):
    df = pd.read_csv(path, low_memory=False, usecols=KEY_COLS)
    keys = sorted(zip(df["dataset_name"], df["trace_name"], df["chunk"].astype(str)))
    h = hashlib.sha256()
    for k in keys:
        h.update("|".join(k).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest(), len(df)


def build_checksums(manifest_list):
    rows = []
    for rel_path in manifest_list:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            print(f"  SKIP (not found locally): {rel_path}", file=sys.stderr)
            continue
        digest, n_rows = hash_manifest(full_path)
        rows.append({"manifest": rel_path, "n_rows": n_rows, "sha256_of_sorted_keys": digest})
        print(f"  {rel_path}: {n_rows:,} rows, {digest[:16]}...")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", nargs="?", help="Hash a single manifest and print its digest")
    parser.add_argument("--check", action="store_true",
        help="Re-hash the known manifest set and diff against the committed checksums file")
    args = parser.parse_args()

    if args.manifest:
        digest, n_rows = hash_manifest(args.manifest)
        print(f"{args.manifest}: {n_rows:,} rows, sha256={digest}")
    elif args.check:
        if not CHECKSUMS_PATH.exists():
            sys.exit(f"No committed checksums file at {CHECKSUMS_PATH}")
        committed = pd.read_csv(CHECKSUMS_PATH).set_index("manifest")
        current = build_checksums(KNOWN_MANIFESTS).set_index("manifest")
        mismatches = 0
        for m in committed.index:
            if m not in current.index:
                print(f"MISSING locally: {m}")
                mismatches += 1
            elif committed.loc[m, "sha256_of_sorted_keys"] != current.loc[m, "sha256_of_sorted_keys"]:
                print(f"MISMATCH: {m} (committed {committed.loc[m, 'sha256_of_sorted_keys'][:16]}... "
                      f"vs current {current.loc[m, 'sha256_of_sorted_keys'][:16]}...)")
                mismatches += 1
        if mismatches == 0:
            print("All manifests match the committed checksums.")
        else:
            sys.exit(f"{mismatches} manifest(s) do not match committed checksums.")
    else:
        print("Hashing known manifest set...")
        df = build_checksums(KNOWN_MANIFESTS)
        CHECKSUMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CHECKSUMS_PATH, index=False)
        print(f"\nWrote {len(df)} checksums to {CHECKSUMS_PATH}")
