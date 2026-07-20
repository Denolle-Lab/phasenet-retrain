"""
Label Error Filtering Module for PhaseNet Retraining

This module downloads/caches the per-dataset "bad label" reports from
Aguilar's labelerrors repository (confident-learning-flagged traces —
https://github.com/albertleonardo/labelerrors, arXiv:2511.09805) and exposes
their trace_names for exclusion from a training or benchmark pool.

Reference: https://github.com/albertleonardo/labelerrors
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/albertleonardo/labelerrors/main"

# labelerrors repo report stem -> our internal DATASET_CONFIGS `name`.
# `iquique` intentionally excluded: PI decision (see
# notebooks/04_creating_benchmark_dataset.ipynb §1.4b) to not apply Aguilar
# filtering to iquique. `ceed`/`aq2009` reports were not yet published when
# that benchmark notebook was written but are available now (verified against
# the repo 2026-07-20) -- included here for the training-side pool.
REPORT_STEM_TO_DATASET = {
    "stead": "stead",
    "instance": "instancecounts",
    "pnw": "pnw",
    "txed": "txed",
    "ethz": "ethz",
    "ceed": "ceed",
    "aq2009": "aq2009gm",
}

# Only multiplet ("more arrivals than labeled") reports are used -- this
# matches the benchmark pool's methodology (notebooks/04_creating_benchmark_
# dataset.ipynb §1.4b) so the two "which traces were Aguilar-flagged" counts
# stay directly comparable. Noise reports (instance/pnw/stead/txed only) exist
# upstream but are not wired in anywhere, matching upstream availability.
MULTIPLET_REPORT_URL = "{base}/multiplet_reports/{stem}_report.csv"


def _cache_path(cache_dir, stem):
    return Path(cache_dir) / f"{stem}_report.csv"


def download_multiplet_report(stem: str, cache_dir: Optional[str] = None) -> Optional[Path]:
    """Download (or reuse cached) multiplet report for one labelerrors report stem."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/phasenet_retrain/label_errors")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_path(cache_dir, stem)
    if cache_file.exists():
        return cache_file

    url = MULTIPLET_REPORT_URL.format(base=GITHUB_RAW_BASE, stem=stem)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, cache_file)
        logger.info(f"Downloaded {url} -> {cache_file}")
        return cache_file
    except Exception as exc:
        logger.error(f"Failed to download label-error report {url}: {exc}")
        return None


def load_bad_trace_names(dataset_name: str, cache_dir: Optional[str] = None,
                          extra_cache_dirs=()) -> Set[str]:
    """
    Return the set of `trace_name` strings Aguilar's multiplet report flags
    as bad for our internal dataset `dataset_name` (e.g. "stead",
    "instancecounts"). Empty set if no report is mapped/available for it.

    `extra_cache_dirs` lets callers also check an already-populated local
    directory (e.g. data/labelerrors/) before downloading.
    """
    stem = next((s for s, d in REPORT_STEM_TO_DATASET.items() if d == dataset_name), None)
    if stem is None:
        return set()

    for d in extra_cache_dirs:
        p = _cache_path(d, stem)
        if p.exists():
            report_path = p
            break
    else:
        report_path = download_multiplet_report(stem, cache_dir=cache_dir)

    if report_path is None or not report_path.exists():
        logger.warning(f"No multiplet report available for {dataset_name} (stem={stem})")
        return set()

    df = pd.read_csv(report_path, usecols=["trace_name"])
    return set(df["trace_name"])


class LabelErrorFilter:
    """Thin class wrapper for callers that want an object with a cache dir."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/phasenet_retrain/label_errors")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Set[str]] = {}

    def load_error_indices(self, dataset_name: str, **_ignored) -> Set[str]:
        """Return the bad trace_name strings for dataset_name (cached)."""
        if dataset_name not in self._cache:
            self._cache[dataset_name] = load_bad_trace_names(
                dataset_name, cache_dir=str(self.cache_dir))
        return self._cache[dataset_name]

    def filter_dataset(self, dataset, dataset_name: str, **_ignored):
        """Remove rows whose `trace_name` is Aguilar-flagged from a SeisBench dataset."""
        bad_trace_names = self.load_error_indices(dataset_name)
        if not bad_trace_names or "trace_name" not in dataset.metadata.columns:
            return dataset

        original_len = len(dataset.metadata)
        keep_mask = ~dataset.metadata["trace_name"].isin(bad_trace_names)
        dataset.metadata = dataset.metadata[keep_mask].reset_index(drop=True)
        removed = original_len - len(dataset.metadata)
        logger.info(f"Filtered {dataset_name}: removed {removed} samples "
                    f"({removed / original_len * 100:.2f}%), kept {len(dataset.metadata)}")
        return dataset

    def get_filter_statistics(self, dataset_name: str) -> Dict[str, int]:
        n = len(self.load_error_indices(dataset_name))
        return {"multiplet_errors": n, "noise_errors": 0, "total_errors": n}


def create_label_error_filter(cache_dir: Optional[str] = None) -> LabelErrorFilter:
    return LabelErrorFilter(cache_dir=cache_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Label Error Filter - Test Run")
    print("=" * 60)
    for dataset_name in sorted(set(REPORT_STEM_TO_DATASET.values())):
        n = len(load_bad_trace_names(dataset_name, extra_cache_dirs=["data/labelerrors"]))
        print(f"{dataset_name:16s}  bad traces flagged: {n:>7,}")
