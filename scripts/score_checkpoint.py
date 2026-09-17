#!/usr/bin/env python3
"""Export a training checkpoint as SeisBench weights and score it on the
development cases at matched budget against the parents (E1 post-run step,
docs/2026-09-18_train_again.md).

    python scripts/score_checkpoint.py --run results/e1_t0_base_seed0
    python scripts/score_checkpoint.py --checkpoint checkpoints/e1_t0_base_seed0/best.pt \\
        --config configs/e1_t0/base.yaml --name e1_t0_base_seed0

Export. The checkpoint of scripts/finetune.py holds the PhaseNetFinetune
state dict (student under `model.`, an optional frozen teacher under
`teacher.`). The student is loaded strictly into the parent architecture
(`model.pretrained.model_name` of the config: `instance` or `jma_wc`) and
written with SeisBench's own `save()` as <out>/exports/<name>.json + .pt, the
pair `heldout_testset_score.load_weights` and `sbm.PhaseNet.load` read. This
is the same strip-and-load as scripts/eval_finetuned.py (lines 84-95) and
QuakeScope's sb_catalog/models/v3/phasenet/convert_checkpoint.py; that
converter is the deployment step (it hard-codes the jma_wc parent) and is
not duplicated here. The export declares the window normalisation the run
trained with (waveform_contract.NORMS: "std" or "peak"), read in this order:
--norm, the checkpoint's own `norm` field (finetune.save_checkpoint writes
it), the run card's `extra.norm`, the config's `data.norm`; a checkpoint
whose norm none of these states is refused, since annotate() would then
normalise differently from training. A sidecar <name>.export.json records
the sha256 of the checkpoint and of both files, the epoch, the parent and
its norm, the export norm and where it came from, and the run card's
checkpoint hash when the run directory has one.

Scoring. scripts/heldout_testset_score.score on every built case whose role
in configs/evaluation_suites.json is `dev` (or --cases), with the candidate
and the parents (--parents, default instance and jma_wc), a dense threshold
grid, and the matched budget set by --budget-reference (default instance, the
primary parent) at --budget-threshold. Writes under --out-root (default
data/evaluation, gitignored): scores/<key>/<access_id>/ per case (the 35A
artifacts) and summary/<name>/{matched_budget.csv,score_rows.csv}, and
prints the matched-budget table with the candidate minus each parent. No
interval here; docs/baselines_2026-09-13/paired_bootstrap.py gives the
paired station-block bootstrap from the written artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluation_policy as policy  # noqa: E402
import heldout_testset_score as hts  # noqa: E402
from waveform_contract import NORMS  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "evaluation"
DEFAULT_PARENTS = ("instance", "jma_wc")
DEFAULT_BUDGET_REFERENCE = "instance"
DENSE_THRESHOLDS = [round(x / 100, 2) for x in range(2, 92, 2)]   # the 2026-09-13 baseline grid


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root=REPO_ROOT):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


# ── locating a run ───────────────────────────────────────────────────────────

def resolve_run(run_dir=None, checkpoint=None, config=None, name=None) -> dict:
    """Checkpoint path, config dict and path, parent name, run name and the run
    card (if any) from either --run <results dir> or --checkpoint + --config."""
    card = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        card_path = run_dir / "run_card.json"
        if not card_path.exists():
            raise FileNotFoundError(f"{card_path} not found; pass --checkpoint and --config instead")
        card = json.loads(card_path.read_text())
        if checkpoint is None:
            ck = (card.get("checkpoint") or {}).get("path")
            if not ck:
                raise ValueError(f"{card_path} has no checkpoint.path (run not finished?); pass --checkpoint")
            checkpoint = ck
        if config is None:
            config = (card.get("config") or {}).get("path")
            if not config:
                raise ValueError(f"{card_path} has no config.path; pass --config")
        name = name or card.get("run_name")
    if checkpoint is None or config is None:
        raise ValueError("need --run, or --checkpoint and --config")
    checkpoint, config = Path(checkpoint), Path(config)
    if not checkpoint.is_absolute() and not checkpoint.exists():
        checkpoint = REPO_ROOT / checkpoint
    if not config.is_absolute() and not config.exists():
        config = REPO_ROOT / config
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cfg = yaml.safe_load(Path(config).read_text())
    parent = ((cfg.get("model") or {}).get("pretrained") or {}).get("model_name", "jma_wc")
    name = name or (cfg.get("logging") or {}).get("run_name") or checkpoint.parent.name
    return dict(checkpoint=checkpoint, config=cfg, config_path=config, parent=parent, name=name, card=card)


# ── export ───────────────────────────────────────────────────────────────────

def student_state_dict(ckpt: dict) -> tuple[dict, int]:
    """The `model.`-prefixed student tensors without the prefix, and the number
    of teacher tensors stripped (scripts/eval_finetuned.py, QuakeScope converter)."""
    raw = ckpt["model"]
    student = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    if not student:
        raise ValueError("checkpoint['model'] has no `model.` keys; not a PhaseNetFinetune checkpoint")
    return student, sum(1 for k in raw if k.startswith("teacher."))


def export_norm(ckpt: dict, card: dict | None = None, config: dict | None = None, override=None):
    """(norm, source) the export must declare: --norm, the checkpoint's `norm`
    field, the run card's extra.norm, the config's data.norm, in that order.
    Raises ValueError when none states it or the value is not in NORMS."""
    candidates = [
        (override, "--norm"),
        ((ckpt or {}).get("norm"), "checkpoint"),
        ((((card or {}).get("extra") or {}).get("norm") or {}).get("norm"), "run card extra.norm"),
        (((config or {}).get("data") or {}).get("norm"), "config data.norm"),
    ]
    for value, source in candidates:
        if value is None:
            continue
        if value not in NORMS:
            raise ValueError(f"{source} declares norm {value!r}; expected one of {NORMS}")
        return value, source
    raise ValueError("the checkpoint's window normalisation is unknown: no --norm, no `norm` in the checkpoint "
                     "(written by finetune.save_checkpoint since 2026-09-18), no extra.norm in the run card and "
                     "no data.norm in the config; pass --norm std for a run trained before the contract")


def export_checkpoint(checkpoint, parent: str, name: str, out_dir, norm=None,
                      card: dict | None = None, config: dict | None = None, config_path=None):
    """Write <out_dir>/<name>.json + .pt with SeisBench save(); returns (model, sidecar dict).
    `norm` overrides the checkpoint/card/config value (export_norm)."""
    import torch
    import seisbench.models as sbm
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    norm, norm_source = export_norm(ckpt, card, config, override=norm)
    student, n_teacher = student_state_dict(ckpt)
    ckpt_parent = ckpt.get("parent")
    if ckpt_parent is not None and ckpt_parent != parent:
        raise ValueError(f"checkpoint was trained from parent {ckpt_parent!r}, export asked for {parent!r}")
    model = sbm.PhaseNet.from_pretrained(parent)
    parent_norm = model.norm
    model.load_state_dict(student, strict=True)        # raises on any width or key mismatch
    model.norm = norm
    model.eval()
    ck_sha = sha256_file(checkpoint)
    card_sha = ((card or {}).get("checkpoint") or {}).get("sha256")
    if card_sha and card_sha != ck_sha:
        raise ValueError(f"run card checkpoint sha256 {card_sha[:12]} != file {ck_sha[:12]}; "
                         "the checkpoint is not the one the card finalised")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = (f"phasenet-retrain E1/T0 export of {checkpoint} (epoch {ckpt.get('epoch', '?')}, "
           f"val_loss {ckpt.get('val_loss', float('nan')):.6f}) from parent {parent} "
           f"(parent norm {parent_norm}); trained on {norm}-normalised windows ({norm_source}), so norm={norm}. "
           f"Written by scripts/score_checkpoint.py at commit {git_commit()}.")
    model.save(out_dir / name, weights_docstring=doc, version_str=None)
    pt_path, json_path = out_dir / f"{name}.pt", out_dir / f"{name}.json"
    sidecar = {
        "name": name, "checkpoint": str(checkpoint), "checkpoint_sha256": ck_sha,
        "run_card_checkpoint_sha256": card_sha, "epoch": ckpt.get("epoch"), "val_loss": ckpt.get("val_loss"),
        "parent": parent, "parent_norm": parent_norm, "export_norm": norm, "export_norm_source": norm_source,
        "n_student_tensors": len(student), "n_teacher_tensors_stripped": n_teacher,
        "pt": str(pt_path), "pt_sha256": sha256_file(pt_path),
        "json": str(json_path), "json_sha256": sha256_file(json_path),
        "config_path": str(config_path) if config_path else None,
        "git_commit": git_commit(), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seisbench": getattr(__import__("seisbench"), "__version__", None), "torch": torch.__version__,
    }
    (out_dir / f"{name}.export.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    # the pair must reload through the scorer's own loader
    reloaded = hts.load_weights(out_dir / name)
    if reloaded.norm != norm:
        raise RuntimeError(f"exported pair reloads with norm {reloaded.norm!r}, expected {norm!r}")
    return reloaded, sidecar


# ── scoring ──────────────────────────────────────────────────────────────────

def dev_cases(out_root=None) -> list[str]:
    """Built cases whose suite role is `dev`."""
    roles = policy.load_policy()["roles"]
    root = hts.OUT_ROOT if out_root is None else Path(out_root)
    return [k for k, role in roles.items() if role == "dev" and (root / k / "manifest.json").exists()]


def matched_budget_table(budgets: pd.DataFrame, candidate: str, parents) -> pd.DataFrame:
    """One row per (key, phase): the budget target, each model's recall and
    threshold at that budget, and candidate minus each parent."""
    if not len(budgets):
        return pd.DataFrame()
    b = budgets.copy()
    b["recall"] = b["recall"].where(b["within_tolerance"].astype(bool))
    rec = b.pivot_table(index=["key", "phase"], columns="model", values="recall", aggfunc="first")
    thr = b.pivot_table(index=["key", "phase"], columns="model", values="threshold", aggfunc="first")
    tgt = b.groupby(["key", "phase"])["target_emitted"].first()
    out = pd.DataFrame({"target": tgt})
    for m in [candidate, *parents]:
        out[f"recall_{m}"] = rec.get(m)
        out[f"thr_{m}"] = thr.get(m)
    for p in parents:
        out[f"delta_vs_{p}"] = out[f"recall_{candidate}"] - out[f"recall_{p}"]
    return out.reset_index()


def score_candidate(name, model, parents=DEFAULT_PARENTS, cases=None, thresholds=DENSE_THRESHOLDS,
                    budget_reference=DEFAULT_BUDGET_REFERENCE, budget_threshold=hts.REPORT_THRESHOLD,
                    out_root=OUT_ROOT, load_parent=None):
    """Score `model` (display name `name`) with the parents on the cases; write
    the summary under <out_root>/summary/<name>/ and return the table."""
    out_root = Path(out_root)
    cases = dev_cases() if cases is None else list(cases)
    if not cases:
        raise ValueError("no built development case: build or link data/heldout_testset/<key>/ first "
                         "(docs/baselines_2026-09-13/README.md, Reproduce)")
    if budget_reference not in (*parents, name):
        raise ValueError(f"--budget-reference {budget_reference!r} is not among the parents {list(parents)} or the candidate")
    load_parent = hts.load_weights if load_parent is None else load_parent
    models = {p: load_parent(p) for p in parents}
    if name in models:
        raise ValueError(f"candidate name {name!r} collides with a parent name")
    models[name] = model
    budgets, rows = [], []
    for key in cases:
        print(f"== {key}", flush=True)
        res = hts.score(key, models, thresholds=thresholds, annotations_root=out_root / "annotations",
                        out_dir=out_root / "scores", budget_reference=budget_reference,
                        budget_threshold=budget_threshold)
        budgets.append(res.budget.assign(key=key, access_id=res.access_id))
        rows.append(res.rows)
        if len(res.failures):
            print("-- failures"); print(res.failures.to_string(index=False))
    budgets = pd.concat(budgets, ignore_index=True) if budgets else pd.DataFrame()
    table = matched_budget_table(budgets, name, parents)
    summary_dir = out_root / "summary" / name
    summary_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(summary_dir / "matched_budget.csv", index=False)
    budgets.to_csv(summary_dir / "budget_rows.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(summary_dir / "score_rows.csv", index=False)
    return table, summary_dir


def format_table(table: pd.DataFrame) -> str:
    if not len(table):
        return "(no matched-budget rows)"
    return table.round(3).to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="results/<run_name> directory with run_card.json")
    ap.add_argument("--checkpoint", default=None, help="best.pt (default: the run card's checkpoint)")
    ap.add_argument("--config", default=None, help="the run's YAML (default: the run card's config path)")
    ap.add_argument("--name", default=None, help="Export weight name (default: the run name)")
    ap.add_argument("--parents", nargs="+", default=list(DEFAULT_PARENTS))
    ap.add_argument("--budget-reference", default=DEFAULT_BUDGET_REFERENCE)
    ap.add_argument("--budget-threshold", type=float, default=hts.REPORT_THRESHOLD)
    ap.add_argument("--cases", nargs="*", default=None, help="Sequence keys (default: every built dev case)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=DENSE_THRESHOLDS)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--export-only", action="store_true", help="Write the SeisBench pair and stop")
    ap.add_argument("--norm", default=None, choices=NORMS,
                    help="Window normalisation to declare (default: the checkpoint's, then the run card's, then data.norm)")
    a = ap.parse_args(argv)
    run = resolve_run(a.run, a.checkpoint, a.config, a.name)
    out_root = Path(a.out_root)
    model, sidecar = export_checkpoint(run["checkpoint"], run["parent"], run["name"], out_root / "exports",
                                       norm=a.norm, card=run["card"], config=run["config"], config_path=run["config_path"])
    print(f"exported {sidecar['pt']} (parent {sidecar['parent']}, parent norm {sidecar['parent_norm']}, "
          f"export norm {sidecar['export_norm']} from {sidecar['export_norm_source']}, epoch {sidecar['epoch']}, "
          f"{sidecar['n_teacher_tensors_stripped']} teacher tensors stripped)")
    if a.export_only:
        return None
    table, summary_dir = score_candidate(run["name"], model, parents=a.parents, cases=a.cases,
                                         thresholds=a.thresholds, budget_reference=a.budget_reference,
                                         budget_threshold=a.budget_threshold, out_root=out_root)
    print(f"\n-- matched budget ({a.budget_reference} at {a.budget_threshold:g} sets the target; "
          f"recall blank where the budget was not attained within {hts.BUDGET_TOLERANCE:.0%})")
    print(format_table(table))
    print(f"wrote {summary_dir}")
    return table


if __name__ == "__main__":
    main()
