#!/usr/bin/env python3
"""Write the seed copies of the E1 arm configs (configs/e1_t0/*.yaml).

Each committed arm file is seed 0. This writes <arm>_seed<k>.yaml for the
requested seeds into --out-dir (default configs/e1_t0/seeds/), changing only
`seed` and the three run-name fields (logging.run_name, checkpoint_dir,
metrics_csv), by text substitution so the arm header survives. The copy is
then parsed and compared with the source: any other difference is an error.

    python scripts/e1_seed_configs.py                      # every arm, seeds 1 and 2
    python scripts/e1_seed_configs.py --arms base kd_t4 --seeds 1 2 3
    python scripts/e1_seed_configs.py --check               # committed copies up to date?

scripts/finetune.py has no --seed override: the seed lives in the config so
the run card's config hash names it (docs/2026-09-11_run_card.md).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ARMS_DIR = REPO_ROOT / "configs" / "e1_t0"
SEEDS_DIR = ARMS_DIR / "seeds"
DEFAULT_SEEDS = (1, 2)
RUN_FIELDS = ("run_name", "checkpoint_dir", "metrics_csv")


def list_arms(arms_dir: Path = ARMS_DIR) -> list[str]:
    return sorted(p.stem for p in arms_dir.glob("*.yaml"))


def seed_copy_text(text: str, seed: int, arm: str) -> str:
    """The seed-`seed` copy of an arm config's text."""
    cfg = yaml.safe_load(text)
    src_seed = cfg.get("seed")
    if not isinstance(src_seed, int):
        raise ValueError(f"{arm}: the source config has no integer `seed`")
    run = cfg["logging"]["run_name"]
    if not re.search(rf"_seed{src_seed}$", run):
        raise ValueError(f"{arm}: logging.run_name {run!r} does not end in _seed{src_seed}")
    stem = run[: -len(f"_seed{src_seed}")]
    out, n = re.subn(rf"^seed:\s*{src_seed}\s*$", f"seed: {seed}", text, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"{arm}: expected exactly one top-level `seed: {src_seed}` line, found {n}")
    out, n = re.subn(rf"{re.escape(stem)}_seed{src_seed}(?![0-9])", f"{stem}_seed{seed}", out)
    if n != len(RUN_FIELDS):
        raise ValueError(f"{arm}: expected {len(RUN_FIELDS)} occurrences of {stem}_seed{src_seed}, found {n}")
    header = (f"# seed {seed} copy of configs/e1_t0/{arm}.yaml written by scripts/e1_seed_configs.py; "
              "do not edit, regenerate.\n")
    out = header + out
    verify_seed_copy(cfg, yaml.safe_load(out), seed, arm)
    return out


def verify_seed_copy(src: dict, copy: dict, seed: int, arm: str) -> None:
    """The copy differs from the source in `seed` and the run-name fields only."""
    if copy.get("seed") != seed:
        raise ValueError(f"{arm}: copy has seed {copy.get('seed')}, expected {seed}")
    for field in RUN_FIELDS:
        a, b = src["logging"][field], copy["logging"][field]
        if a == b or f"_seed{seed}" not in b:
            raise ValueError(f"{arm}: logging.{field} not re-seeded: {a!r} -> {b!r}")
    s = {k: v for k, v in src.items() if k != "seed"}
    c = {k: v for k, v in copy.items() if k != "seed"}
    s["logging"] = {k: v for k, v in s["logging"].items() if k not in RUN_FIELDS}
    c["logging"] = {k: v for k, v in c["logging"].items() if k not in RUN_FIELDS}
    if s != c:
        raise ValueError(f"{arm}: the seed copy differs from the source outside seed and run names")


def write_seed_configs(arms=None, seeds=DEFAULT_SEEDS, arms_dir: Path = ARMS_DIR, out_dir: Path = SEEDS_DIR,
                       check: bool = False) -> list[Path]:
    arms = list_arms(arms_dir) if not arms else list(arms)
    written, stale = [], []
    for arm in arms:
        src = arms_dir / f"{arm}.yaml"
        if not src.exists():
            raise FileNotFoundError(src)
        text = src.read_text()
        for seed in seeds:
            out = out_dir / f"{arm}_seed{seed}.yaml"
            new = seed_copy_text(text, int(seed), arm)
            if check:
                if not out.exists() or out.read_text() != new:
                    stale.append(out)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            out.write_text(new)
            written.append(out)
    if check and stale:
        raise SystemExit("stale or missing seed copies (rerun scripts/e1_seed_configs.py):\n  "
                         + "\n  ".join(str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in stale))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=None, help="Arm stems (default: every configs/e1_t0/*.yaml)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--arms-dir", default=str(ARMS_DIR))
    ap.add_argument("--out-dir", default=str(SEEDS_DIR))
    ap.add_argument("--check", action="store_true", help="Exit 1 if a committed seed copy is stale or missing")
    a = ap.parse_args(argv)
    if any(s == 0 for s in a.seeds):
        ap.error("seed 0 is the committed arm file itself")
    written = write_seed_configs(a.arms, a.seeds, Path(a.arms_dir), Path(a.out_dir), check=a.check)
    if a.check:
        print("seed copies up to date")
    for p in written:
        print("wrote", p)
    return written


if __name__ == "__main__":
    main()
