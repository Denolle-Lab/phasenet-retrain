#!/usr/bin/env python3
"""
audit_noise_picks.py

Runs PhaseNet (jma_wc pretrained) on every trace in the two noise datasets
and flags any trace where the model predicts a P or S probability above a
threshold.  Optionally filters flagged traces out of the metadata CSVs and
training manifests.

Usage
-----
    # audit only (no changes written):
    python scripts/audit_noise_picks.py [--threshold 0.3] [--batch-size 512]

    # audit + remove flagged traces from metadata / manifests:
    python scripts/audit_noise_picks.py --filter 0.3
"""

import argparse
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SEISBENCH_CACHE = os.environ.get("SEISBENCH_CACHE_ROOT", os.path.expanduser("~/.seisbench"))
os.environ.setdefault("SEISBENCH_CACHE_ROOT", SEISBENCH_CACHE)
import seisbench
seisbench.cache_root = SEISBENCH_CACHE

import seisbench.models as sbm

REPO_ROOT   = Path(__file__).parent.parent.resolve()
MANIFESTS   = REPO_ROOT / "data" / "manifests"
AUDIT_DIR   = REPO_ROOT / "data" / "noise_audit"

NOISE_SOURCES = [
    dict(
        name      = "noise_global",
        hdf5      = REPO_ROOT / "data" / "noise_global"   / "waveforms.hdf5",
        meta_csv  = REPO_ROOT / "data" / "noise_global"   / "metadata.csv",
    ),
    dict(
        name      = "noise_prephase",
        hdf5      = REPO_ROOT / "data" / "noise_prephase" / "waveforms.hdf5",
        meta_csv  = REPO_ROOT / "data" / "noise_prephase" / "metadata.csv",
    ),
]

IN_SAMPLES = 3001   # jma_wc native window length


# ── preprocessing ──────────────────────────────────────────────────────────────

def normalise_std(wf: np.ndarray) -> np.ndarray:
    wf = wf.astype(np.float32)
    wf -= wf.mean(axis=-1, keepdims=True)
    std = wf.std(axis=-1, keepdims=True)
    std[std < 1e-6] = 1.0
    return np.clip(wf / std, -10.0, 10.0)


def prepare_batch(waveforms: list[np.ndarray]) -> torch.Tensor:
    """Stack, normalise and pad a list of (3, N) arrays to (B, 3, IN_SAMPLES)."""
    batch = []
    for wf in waveforms:
        wf = np.asarray(wf, dtype=np.float32)
        # ensure (3, N)
        if wf.ndim == 1:
            wf = np.stack([wf, wf, wf])
        elif wf.ndim == 2 and wf.shape[0] != 3 and wf.shape[-1] == 3:
            wf = wf.T
        wf = normalise_std(wf)
        n = wf.shape[-1]
        if n < IN_SAMPLES:
            wf = np.concatenate([wf, np.zeros((3, IN_SAMPLES - n), dtype=np.float32)], axis=-1)
        else:
            wf = wf[:, :IN_SAMPLES]
        batch.append(wf)
    return torch.from_numpy(np.stack(batch))   # (B, 3, IN_SAMPLES)


# ── per-source audit ───────────────────────────────────────────────────────────

def audit_source(cfg: dict, model: torch.nn.Module, device: torch.device,
                 batch_size: int) -> pd.DataFrame:
    name     = cfg["name"]
    hdf5_p   = cfg["hdf5"]
    meta_csv = cfg["meta_csv"]

    meta        = pd.read_csv(meta_csv)
    trace_names = meta["trace_name"].tolist()
    n_total     = len(trace_names)
    print(f"\n[{name}]  {n_total:,} traces", flush=True)

    results = []   # list of (trace_name, max_p, max_s)

    with h5py.File(hdf5_p, "r") as h5:
        grp = h5["data"]

        for batch_start in range(0, n_total, batch_size):
            batch_names = trace_names[batch_start: batch_start + batch_size]
            waveforms   = []
            valid_names = []

            for tname in batch_names:
                if tname not in grp:
                    results.append((tname, np.nan, np.nan))
                    continue
                waveforms.append(grp[tname][()])
                valid_names.append(tname)

            if not waveforms:
                continue

            x = prepare_batch(waveforms).to(device)
            with torch.no_grad():
                probs = model(x).cpu().numpy()   # (B, 3, IN_SAMPLES)  PSN order

            p_probs = probs[:, 0, :]   # P channel
            s_probs = probs[:, 1, :]   # S channel
            for tname, max_p, max_s in zip(
                valid_names, p_probs.max(axis=-1), s_probs.max(axis=-1)
            ):
                results.append((tname, float(max_p), float(max_s)))

            done = min(batch_start + batch_size, n_total)
            if done % 10000 == 0 or done == n_total:
                print(f"  {done:,}/{n_total:,}", flush=True)

    df = pd.DataFrame(results, columns=["trace_name", "max_p_prob", "max_s_prob"])
    return df


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading jma_wc  (device={device})")
    model = sbm.PhaseNet.from_pretrained("jma_wc").to(device)
    model.eval()

    all_results = {}
    for cfg in NOISE_SOURCES:
        df = audit_source(cfg, model, device, args.batch_size)
        out_csv = AUDIT_DIR / f"{cfg['name']}_audit.csv"
        df.to_csv(out_csv, index=False)
        all_results[cfg["name"]] = df
        print(f"  Audit results saved: {out_csv}")

    # ── summary at several thresholds ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Dataset':<18} {'Total':>7}  "
          f"{'P>0.1':>7} {'P>0.3':>7} {'P>0.5':>7} {'P>0.7':>7}")
    print(f"{'-'*65}")
    for name, df in all_results.items():
        n = len(df)
        counts = [int((df["max_p_prob"] > t).sum()) for t in [0.1, 0.3, 0.5, 0.7]]
        pcts   = [f"{c:,} ({100*c/n:.1f}%)" for c in counts]
        print(f"{name:<18} {n:>7,}  "
              f"{counts[0]:>5,} {counts[1]:>7,} {counts[2]:>7,} {counts[3]:>7,}")
    print(f"{'='*65}")

    # ── filter if requested ────────────────────────────────────────────────────
    if args.filter is not None:
        thr = args.filter
        print(f"\nFiltering traces with max_p_prob > {thr} from metadata and manifests ...")
        flagged_all = set()

        for cfg in NOISE_SOURCES:
            name    = cfg["name"]
            df      = all_results[name]
            flagged = set(df[df["max_p_prob"] > thr]["trace_name"])
            flagged_all.update(flagged)

            # update metadata.csv
            meta = pd.read_csv(cfg["meta_csv"])
            before = len(meta)
            meta = meta[~meta["trace_name"].isin(flagged)]
            meta.to_csv(cfg["meta_csv"], index=False)
            print(f"  {name}: removed {before - len(meta):,} flagged / {before:,} total "
                  f"→ {len(meta):,} remain")

        # update train.csv and val.csv
        for split in ["train.csv", "val.csv"]:
            path = MANIFESTS / split
            df   = pd.read_csv(path, low_memory=False)
            before = len(df)
            df = df[~df["trace_name"].isin(flagged_all)]
            df.to_csv(path, index=False)
            print(f"  {split}: {before:,} → {len(df):,} rows")

        print(f"\n  Total flagged and removed: {len(flagged_all):,}")
    else:
        print(f"\nRun with --filter {threshold} to remove flagged traces.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit noise datasets for accidental P/S picks using jma_wc"
    )
    parser.add_argument("--threshold",  type=float, default=0.3,
                        help="P-prob threshold used for --filter (default 0.3)")
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--filter",     type=float, default=None, metavar="THRESHOLD",
                        help="If set, remove traces with max_p_prob > THRESHOLD "
                             "from metadata and manifests")
    main(parser.parse_args())
