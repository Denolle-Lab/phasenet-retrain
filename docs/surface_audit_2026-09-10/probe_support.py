"""Portable evidence identifiers and explicit catalogue-date accounting."""
from pathlib import Path

import pandas as pd


def source_id(path, roots):
    """Identify a file by a named root, never by a workstation path."""
    path = Path(path).resolve()
    for name, root in roots:
        try:
            relative = path.relative_to(Path(root).resolve())
        except ValueError:
            continue
        return f"{name}/{relative.as_posix()}"
    raise ValueError("Evidence file is outside the declared source roots")


def date_summary(values):
    """Count invalid/missing dates explicitly rather than silently excluding them."""
    dates = pd.Series(values, dtype="string").str.strip()
    missing = dates.isna() | dates.eq("").fillna(False)
    parsed = pd.to_datetime(dates.mask(missing), format="mixed", errors="coerce", utc=True)
    valid = parsed.notna()
    return {
        "valid_timestamp_rows": int(valid.sum()),
        "missing_timestamp_rows": int(missing.sum()),
        "invalid_timestamp_rows": int((~missing & ~valid).sum()),
        "pre_2002_picks": int((parsed.dt.year < 2002).sum()),
        "fractional_timestamp_rows": int((valid & dates.str.contains(
            r":\d{2}\.\d+", na=False)).sum()),
    }
