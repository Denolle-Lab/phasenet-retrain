"""The train-again path (#40A, #46B): E1 configs, seed copies, BatchNorm
freezing, the `instance` build and scripts/score_checkpoint.py.

Pure-Python parts run everywhere; the torch parts skip without torch, and
the parts that need the cached `instance` weights skip when the SeisBench
model cache does not hold them (no download is attempted).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import e1_seed_configs as esc  # noqa: E402
import score_checkpoint as sc  # noqa: E402
import heldout_testset_score as hts  # noqa: E402

ARMS_DIR = REPO / "configs" / "e1_t0"
ARMS = ("base", "mask_off", "kd_t1p5", "kd_t4", "bn_frozen", "lr_2e-6", "lr_2e-5", "base_jma_wc")

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _have_cached(name):
    if not HAVE_TORCH:
        return False
    try:
        import seisbench
        return any((Path(seisbench.cache_root) / "models" / "v3" / "phasenet").glob(f"{name}.pt.v*"))
    except Exception:  # noqa: BLE001
        return False


needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
needs_instance = pytest.mark.skipif(not _have_cached("instance"), reason="needs the cached instance weights")


def _load(arm):
    return yaml.safe_load((ARMS_DIR / f"{arm}.yaml").read_text())


# ── the configs ──────────────────────────────────────────────────────────────

def test_base_config_is_the_e1_base_recipe():
    cfg = _load("base")
    assert cfg["model"]["pretrained"]["model_name"] == "instance"
    t = cfg["training"]
    assert t["optimizer"] == "adamw" and t["learning_rate"] == pytest.approx(5e-6)
    assert t["weight_decay"] == pytest.approx(1e-4) and t["batch_size"] == 256 and t["gradient_clip_val"] == 1.0
    assert t["soft_ce"] is True and t["distillation"]["alpha"] == 0.0 and t["freeze_bn_stats"] is False
    assert t["timing_beta"] == 0 and t["presence_gamma"] == 0 and t["focal_gamma"] == 0 and "class_weights" not in t
    assert t["scheduler"] == {"name": "CosineAnnealingWarmup", "warmup_epochs": 2, "min_lr": 1e-6, "warmup_start_factor": 0.01}
    assert t["early_stopping"] == {"monitor": "val_loss", "patience": 10, "mode": "min"}
    assert t["max_epochs"] == 60
    d = cfg["data"]
    assert d["label_policy"] == "masked" and d["window_length"] == 3001
    assert d["norm"] == "peak"                                   # follows the instance parent
    assert d["augmentation"] == {"noise_prob": 0.0}
    assert {d["train_manifest"], d["val_manifest"], d["test_manifest"]} == {
        "data/manifests_t0/train.csv", "data/manifests_t0/val.csv", "data/manifests_t0/test.csv"}
    assert cfg["seed"] == 0 and cfg["logging"]["run_name"] == "e1_t0_base_seed0"
    assert cfg["logging"]["checkpoint_dir"].endswith("e1_t0_base_seed0")
    assert cfg["hardware"]["amp"] is True


@pytest.mark.parametrize("arm,path,value", [
    ("mask_off", ("data", "label_policy"), "legacy"),
    ("kd_t1p5", ("training", "distillation"), {"alpha": 0.3, "temperature": 1.5}),
    ("kd_t4", ("training", "distillation"), {"alpha": 0.3, "temperature": 4.0}),
    ("bn_frozen", ("training", "freeze_bn_stats"), True),
    ("lr_2e-6", ("training", "learning_rate"), 2e-6),
    ("lr_2e-5", ("training", "learning_rate"), 2e-5),
    ("base_jma_wc", ("model", "pretrained", "model_name"), "jma_wc"),
])
def test_each_arm_differs_from_base_in_exactly_one_factor(arm, path, value):
    """One key per arm; base_jma_wc also carries data.norm std because the
    normalisation follows the parent (jma_wc is std, instance is peak)."""
    base, cfg = _load("base"), _load(arm)
    if arm == "base_jma_wc":
        assert cfg["data"]["norm"] == "std"
        cfg["data"]["norm"] = base["data"]["norm"]
    else:
        assert cfg["data"]["norm"] == "peak"

    def get(c, p):
        for k in p:
            c = c[k]
        return c

    def strip(c):
        c = json.loads(json.dumps(c))
        c["logging"] = {k: v for k, v in c["logging"].items() if k not in ("run_name", "checkpoint_dir", "metrics_csv")}
        return c

    assert get(cfg, path) == pytest.approx(value) if isinstance(value, float) else get(cfg, path) == value
    b, c = strip(base), strip(cfg)
    # put the factor back to the base value: nothing else may differ
    node = c
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = get(base, path)
    assert b == c, arm
    run = cfg["logging"]["run_name"]
    assert run.startswith("e1_t0_") and run.endswith("_seed0") and run != base["logging"]["run_name"]
    assert cfg["logging"]["checkpoint_dir"] == f"checkpoints/{run}" and cfg["logging"]["metrics_csv"] == f"results/{run}_metrics.csv"
    header = (ARMS_DIR / f"{arm}.yaml").read_text().split("\n")[0]
    assert header.startswith("#") and arm in header


def test_all_arm_run_names_are_distinct():
    runs = [_load(a)["logging"]["run_name"] for a in ARMS]
    assert len(set(runs)) == len(ARMS)
    assert sorted(p.stem for p in ARMS_DIR.glob("*.yaml")) == sorted(ARMS)


# ── seed copies ──────────────────────────────────────────────────────────────

def test_committed_seed_copies_are_current():
    esc.write_seed_configs(check=True)                         # SystemExit when stale
    for arm in ARMS:
        for seed in (1, 2):
            cfg = yaml.safe_load((ARMS_DIR / "seeds" / f"{arm}_seed{seed}.yaml").read_text())
            assert cfg["seed"] == seed and cfg["logging"]["run_name"] == f"{_load(arm)['logging']['run_name'][:-1]}{seed}"


def test_seed_copy_changes_only_seed_and_run_names(tmp_path):
    text = (ARMS_DIR / "kd_t4.yaml").read_text()
    out = esc.seed_copy_text(text, 7, "kd_t4")
    src, cp = yaml.safe_load(text), yaml.safe_load(out)
    assert cp["seed"] == 7 and cp["logging"]["run_name"] == "e1_t0_kd_t4_seed7"
    assert cp["logging"]["checkpoint_dir"] == "checkpoints/e1_t0_kd_t4_seed7"
    assert cp["logging"]["metrics_csv"] == "results/e1_t0_kd_t4_seed7_metrics.csv"
    assert cp["training"] == src["training"] and cp["data"] == src["data"] and cp["model"] == src["model"]
    assert out.startswith("# seed 7 copy of configs/e1_t0/kd_t4.yaml")
    assert "arm `kd_t4`" in out                                 # the arm header survives
    # the writer on a copy of the arms directory
    arms = tmp_path / "arms"
    arms.mkdir()
    for a in ("base", "lr_2e-6"):
        (arms / f"{a}.yaml").write_text((ARMS_DIR / f"{a}.yaml").read_text())
    written = esc.write_seed_configs(None, (1, 2), arms, arms / "seeds")
    assert sorted(p.name for p in written) == ["base_seed1.yaml", "base_seed2.yaml", "lr_2e-6_seed1.yaml", "lr_2e-6_seed2.yaml"]
    esc.write_seed_configs(None, (1, 2), arms, arms / "seeds", check=True)
    (arms / "base.yaml").write_text((arms / "base.yaml").read_text().replace("max_epochs: 60", "max_epochs: 61"))
    with pytest.raises(SystemExit, match="stale"):
        esc.write_seed_configs(None, (1, 2), arms, arms / "seeds", check=True)
    with pytest.raises(SystemExit):
        esc.main(["--seeds", "0", "--arms-dir", str(arms), "--out-dir", str(arms / "seeds")])
    with pytest.raises(ValueError, match="does not end in _seed"):
        esc.seed_copy_text(text.replace('run_name: "e1_t0_kd_t4_seed0"', 'run_name: "e1_t0_kd_t4"'), 1, "kd_t4")


# ── score_checkpoint: pure parts ─────────────────────────────────────────────

def test_student_state_dict_strips_prefix_and_counts_teacher():
    ckpt = {"model": {"model.in_bn.weight": 1, "model.out.bias": 2, "teacher.in_bn.weight": 3, "class_weight": 4}}
    sd, n_teacher = sc.student_state_dict(ckpt)
    assert sd == {"in_bn.weight": 1, "out.bias": 2} and n_teacher == 1
    with pytest.raises(ValueError, match="model\\."):
        sc.student_state_dict({"model": {"teacher.x": 1}})


def test_resolve_run_from_card_and_from_paths(tmp_path):
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text((ARMS_DIR / "base.yaml").read_text())
    ck = tmp_path / "best.pt"
    ck.write_bytes(b"x")
    run = tmp_path / "results" / "e1_t0_base_seed0"
    run.mkdir(parents=True)
    (run / "run_card.json").write_text(json.dumps({"run_name": "e1_t0_base_seed0",
                                                   "config": {"path": str(cfg_path)},
                                                   "checkpoint": {"path": str(ck), "sha256": "abc"}}))
    r = sc.resolve_run(run_dir=run)
    assert r["checkpoint"] == ck and r["parent"] == "instance" and r["name"] == "e1_t0_base_seed0"
    assert r["card"]["checkpoint"]["sha256"] == "abc" and r["config"]["seed"] == 0
    r2 = sc.resolve_run(checkpoint=ck, config=cfg_path, name="x")
    assert r2["name"] == "x" and r2["card"] is None
    with pytest.raises(ValueError, match="need --run"):
        sc.resolve_run()
    with pytest.raises(FileNotFoundError):
        sc.resolve_run(run_dir=tmp_path / "nowhere")
    (run / "run_card.json").write_text(json.dumps({"run_name": "r", "config": {"path": str(cfg_path)}, "checkpoint": None}))
    with pytest.raises(ValueError, match="checkpoint.path"):
        sc.resolve_run(run_dir=run)


def _budget_rows(key, models, recalls, thresholds, target=100, within=True):
    rows = []
    for phase in ("P", "S"):
        for m in models:
            rows.append(dict(phase=phase, model_id=f"id_{m}", model=m, target_emitted=target,
                             threshold=thresholds[m], emitted=target, matched=int(recalls[m][phase] * 50),
                             n_reference=50, recall=recalls[m][phase], within_tolerance=within,
                             reason="", budget_reference="instance", key=key))
    return pd.DataFrame(rows)


def test_matched_budget_table_pivots_and_differences():
    models = ["instance", "jma_wc", "cand"]
    rec = {"instance": {"P": 0.8, "S": 0.6}, "jma_wc": {"P": 0.75, "S": 0.5}, "cand": {"P": 0.85, "S": 0.7}}
    thr = {"instance": 0.3, "jma_wc": 0.5, "cand": 0.2}
    b = pd.concat([_budget_rows("samos_2020", models, rec, thr), _budget_rows("etna_2022_2024", models, rec, thr)])
    t = sc.matched_budget_table(b, "cand", ["instance", "jma_wc"]).set_index(["key", "phase"])
    assert t.loc[("samos_2020", "P"), "delta_vs_instance"] == pytest.approx(0.05)
    assert t.loc[("etna_2022_2024", "S"), "delta_vs_jma_wc"] == pytest.approx(0.2)
    assert t.loc[("samos_2020", "S"), "thr_cand"] == 0.2 and t.loc[("samos_2020", "S"), "target"] == 100
    assert list(t.columns)[:3] == ["target", "recall_cand", "thr_cand"]
    # a model that did not attain the budget has no recall and no difference
    b2 = _budget_rows("samos_2020", models, rec, thr)
    b2.loc[b2.model == "cand", "within_tolerance"] = False
    t2 = sc.matched_budget_table(b2, "cand", ["instance"]).set_index(["key", "phase"])
    assert t2["recall_cand"].isna().all() and t2["delta_vs_instance"].isna().all()
    assert sc.format_table(pd.DataFrame()) == "(no matched-budget rows)"
    assert "delta_vs_instance" in sc.format_table(t.reset_index())


def test_score_candidate_runs_the_scorer_per_case_and_writes_the_summary(tmp_path, monkeypatch):
    calls = []

    def fake_score(key, models, thresholds, annotations_root, out_dir, budget_reference, budget_threshold, data_root):
        calls.append((key, sorted(models), budget_reference, tuple(thresholds), Path(data_root)))
        rec = {"instance": {"P": 0.8, "S": 0.6}, "jma_wc": {"P": 0.7, "S": 0.5}, "cand": {"P": 0.9, "S": 0.65}}
        thr = {"instance": 0.3, "jma_wc": 0.4, "cand": 0.1}
        return hts.ScoreResult(key=key, access_id="a1", rows=pd.DataFrame({"key": [key], "scope": ["aggregate"]}),
                               budget=_budget_rows(key, list(models), rec, thr), models={m: f"id_{m}" for m in models})

    monkeypatch.setattr(sc.hts, "score", fake_score)
    seqs = tmp_path / "built"                                    # the built-sequence root, apart from the output root
    for k in ("samos_2020", "corinth_thiva_2020", "kaikoura_2016"):
        (seqs / k).mkdir(parents=True)
        (seqs / k / "manifest.json").write_text("{}")
    table, summary = sc.score_candidate("cand", object(), parents=("instance", "jma_wc"), thresholds=[0.1, 0.3],
                                        out_root=tmp_path / "eval", load_parent=lambda p: f"weights:{p}",
                                        sequences_root=seqs)
    assert [c[0] for c in calls] == ["samos_2020", "corinth_thiva_2020"]     # dev cases found under sequences_root
    assert all(c[4] == seqs for c in calls)                                   # and the scorer reads them from there
    assert calls[0][1] == ["cand", "instance", "jma_wc"] and calls[0][2] == "instance" and calls[0][3] == (0.1, 0.3)
    tmp_path = tmp_path / "eval"
    assert summary == tmp_path / "summary" / "cand"
    written = pd.read_csv(summary / "matched_budget.csv")
    assert len(written) == 4 and written["delta_vs_instance"].round(3).tolist() == [0.1, 0.05, 0.1, 0.05]
    assert (summary / "score_rows.csv").exists() and (summary / "budget_rows.csv").exists()
    with pytest.raises(ValueError, match="budget-reference"):
        sc.score_candidate("cand", object(), parents=("jma_wc",), budget_reference="instance", out_root=tmp_path,
                           load_parent=lambda p: p)
    with pytest.raises(ValueError, match="collides"):
        sc.score_candidate("instance", object(), parents=("instance",), out_root=tmp_path, load_parent=lambda p: p)
    with pytest.raises(ValueError, match="no built development case"):
        sc.score_candidate("cand", object(), out_root=tmp_path, load_parent=lambda p: p,
                           sequences_root=tmp_path / "nothing_built")


def test_dev_cases_follow_the_policy_roles(tmp_path, monkeypatch):
    import evaluation_policy as policy
    roles = policy.load_policy()["roles"]
    dev = [k for k, r in roles.items() if r == "dev"]
    assert set(dev) == {"samos_2020", "adriatic_2022", "etna_2022_2024", "corinth_thiva_2020"}
    for k in ("samos_2020", "kaikoura_2016"):
        (tmp_path / k).mkdir()
        (tmp_path / k / "manifest.json").write_text("{}")
    assert sc.dev_cases(tmp_path) == ["samos_2020"]              # a built regression case is not a dev case
    assert sc.dev_cases(tmp_path / "empty") == []
    monkeypatch.setattr(sc.hts, "OUT_ROOT", tmp_path)
    assert sc.dev_cases() == ["samos_2020"]                      # the default root is the scorer's


# ── torch parts ──────────────────────────────────────────────────────────────

@needs_torch
def test_freeze_bn_stats_keeps_running_statistics_fixed_but_trains_affine():
    """Under a training step, BatchNorm running statistics change with the
    default and stay fixed with training.freeze_bn_stats; the affine weight
    receives a gradient either way. A small PhaseNet stands in for the parent
    so no weight file is needed."""
    import torch
    import seisbench.models as sbm
    import fine_tune_model as ftm

    def build(freeze):
        cfg = {"model": {"pretrained": {"model_name": "x"}},
               "training": {"learning_rate": 1e-3, "soft_ce": True, "freeze_bn_stats": freeze, "optimizer": "adamw"}}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sbm.PhaseNet, "from_pretrained", classmethod(lambda cls, name: sbm.PhaseNet(phases="PSN")))
            return ftm.PhaseNetFinetune(cfg)

    torch.manual_seed(0)
    x = torch.randn(4, 3, 3001) * 3 + 1.0
    y = torch.zeros(4, 3, 3001)
    y[:, 2] = 1.0
    y[:, 0, 1000:1010] = 1.0
    y[:, 2, 1000:1010] = 0.0
    for freeze in (False, True):
        m = build(freeze)
        assert m.freeze_bn_stats is freeze and len(m.bn_modules()) > 0
        m.train()
        assert all(bn.training is (not freeze) for bn in m.bn_modules())
        assert all(p.requires_grad for bn in m.bn_modules() for p in bn.parameters())
        before = [bn.running_mean.clone() for bn in m.bn_modules()]
        opt, _ = m.build_optimiser({"training": {"learning_rate": 1e-3, "optimizer": "adamw"}})
        metrics, _ = m.compute_loss_and_metrics(x, y, torch.ones(4, 3001))
        opt.zero_grad()
        metrics["loss"].backward()
        assert all(bn.weight.grad is not None and bn.weight.grad.abs().sum() > 0 for bn in m.bn_modules())
        opt.step()
        changed = [not torch.equal(a, bn.running_mean) for a, bn in zip(before, m.bn_modules())]
        if freeze:
            assert not any(changed), "frozen BN statistics moved"
        else:
            assert any(changed), "adaptive BN statistics did not move"
        m.eval()
        assert not any(bn.training for bn in m.bn_modules())
        m.train()
        assert all(bn.training is (not freeze) for bn in m.bn_modules())


@needs_instance
@pytest.mark.parametrize("arm", [a for a in ARMS if a != "base_jma_wc"])
def test_phasenet_finetune_builds_from_instance_for_every_arm(arm):
    """Each E1 config builds through PhaseNetFinetune from the cached
    `instance` weights (standard width, norm peak) with the config's loss,
    teacher and BN settings, and its optimiser is AdamW with the warm-up
    cosine schedule."""
    import torch
    import fine_tune_model as ftm
    cfg = _load(arm)
    m = ftm.PhaseNetFinetune(cfg)
    assert m.parent_name == "instance" and m.model.norm == "peak" and getattr(m.model, "filter_factor", 1) == 1
    assert m.parent_norm == "peak" and m.norm == "peak"                   # data.norm follows the parent
    assert sum(p.numel() for p in m.model.parameters()) == 268_443
    t = cfg["training"]
    assert m.soft_ce is True and m.class_weight is None
    assert m.freeze_bn_stats is t["freeze_bn_stats"]
    assert (m.teacher is not None) is (t["distillation"]["alpha"] > 0)
    if m.teacher is not None:
        assert m.distill_T == t["distillation"]["temperature"] and not any(p.requires_grad for p in m.teacher.parameters())
    opt, sched = m.build_optimiser(cfg)
    assert isinstance(opt, torch.optim.AdamW) and opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)
    assert opt.param_groups[0]["lr"] == pytest.approx(t["learning_rate"] * 0.01)   # warm-up start
    m.train()
    x = torch.randn(2, 3, 3001)
    y = torch.zeros(2, 3, 3001)
    y[:, 2] = 1.0
    mask = torch.zeros(2, 3001)
    mask[:, 500:600] = 1.0
    metrics, probs = m.compute_loss_and_metrics(x, y, mask)
    assert torch.isfinite(metrics["loss"]) and probs.shape == (2, 3, 3001)
    assert metrics["supervised_fraction"] == pytest.approx(100 / 3001, rel=1e-3)


@needs_instance
def test_finetune_config_path_loads_and_the_export_round_trips(tmp_path, monkeypatch):
    """scripts/finetune.py's load_config reads every arm; a checkpoint written
    the way save_checkpoint writes it exports to a SeisBench pair that reloads
    through heldout_testset_score.load_weights with the student weights, norm
    std and a sidecar whose hashes match the files."""
    import torch
    import fine_tune_model as ftm
    monkeypatch.setitem(sys.modules, "plot_training_curves", __import__("types").ModuleType("plot_training_curves"))
    for name in ("load_metrics", "plot_dashboard", "plot_loss", "plot_accuracy", "plot_residuals", "plot_lr"):
        setattr(sys.modules["plot_training_curves"], name, lambda *a, **k: None)
    import finetune
    for arm in ARMS:
        cfg = finetune.load_config(str(ARMS_DIR / f"{arm}.yaml"))
        assert cfg["data"]["label_policy"] in ("masked", "legacy")
    cfg = finetune.load_config(str(ARMS_DIR / "kd_t4.yaml"))
    m = ftm.PhaseNetFinetune(cfg)
    with torch.no_grad():
        m.model.in_bn.weight.mul_(1.5)                            # a change the export must carry
    opt, _ = m.build_optimiser(cfg)
    ck = tmp_path / "checkpoints" / "best.pt"
    finetune.save_checkpoint(m, opt, torch.cuda.amp.GradScaler(enabled=False), 3, 0.123, ck)
    saved = torch.load(ck, map_location="cpu")
    assert saved["norm"] == "peak" and saved["parent"] == "instance"          # the config's data.norm travels with the checkpoint
    reloaded, sidecar = sc.export_checkpoint(ck, "instance", "e1_test", tmp_path / "exports",
                                             card={"checkpoint": {"sha256": sc.sha256_file(ck)}})
    assert reloaded.norm == "peak" and sidecar["parent_norm"] == "peak" and sidecar["export_norm"] == "peak"
    assert sidecar["export_norm_source"] == "checkpoint"
    assert sidecar["epoch"] == 3 and sidecar["n_teacher_tensors_stripped"] > 0
    assert torch.equal(reloaded.in_bn.weight, m.model.in_bn.weight)
    assert sidecar["pt_sha256"] == sc.sha256_file(tmp_path / "exports" / "e1_test.pt")
    meta = json.loads((tmp_path / "exports" / "e1_test.json").read_text())
    assert meta["model_args"]["norm"] == "peak" and meta["model_args"]["phases"] == "PSN"
    assert "epoch 3" in meta["docstring"] and "peak-normalised" in meta["docstring"]
    # the export json carries whatever norm the run used, never a fixed value
    m.norm = "std"
    ck_std = tmp_path / "checkpoints" / "best_std.pt"
    finetune.save_checkpoint(m, opt, torch.cuda.amp.GradScaler(enabled=False), 4, 0.2, ck_std)
    re_std, side_std = sc.export_checkpoint(ck_std, "instance", "e1_std", tmp_path / "exports")
    assert re_std.norm == "std" and json.loads((tmp_path / "exports" / "e1_std.json").read_text())["model_args"]["norm"] == "std"
    # a checkpoint without a norm is refused unless the card, the config or --norm states it
    m.norm = None
    ck_none = tmp_path / "checkpoints" / "best_none.pt"
    finetune.save_checkpoint(m, opt, torch.cuda.amp.GradScaler(enabled=False), 5, 0.3, ck_none)
    with pytest.raises(ValueError, match="unknown"):
        sc.export_checkpoint(ck_none, "instance", "e1_none", tmp_path / "exports")
    _, side = sc.export_checkpoint(ck_none, "instance", "e1_card", tmp_path / "exports",
                                   card={"extra": {"norm": {"norm": "peak", "source": "parent instance"}}})
    assert (side["export_norm"], side["export_norm_source"]) == ("peak", "run card extra.norm")
    _, side = sc.export_checkpoint(ck_none, "instance", "e1_cfg", tmp_path / "exports", config={"data": {"norm": "std"}})
    assert (side["export_norm"], side["export_norm_source"]) == ("std", "config data.norm")
    _, side = sc.export_checkpoint(ck_none, "instance", "e1_flag", tmp_path / "exports", norm="std", config={"data": {"norm": "peak"}})
    assert (side["export_norm"], side["export_norm_source"]) == ("std", "--norm")
    with pytest.raises(ValueError, match="trained from parent"):
        sc.export_checkpoint(ck, "jma_wc", "e1_wrong_parent", tmp_path / "exports")
    again = hts.load_weights(str(tmp_path / "exports" / "e1_test.pt"))     # the .pt spelling works too
    assert torch.equal(again.in_bn.weight, reloaded.in_bn.weight)
    with pytest.raises(ValueError, match="sha256"):
        sc.export_checkpoint(ck, "instance", "e1_bad", tmp_path / "exports", card={"checkpoint": {"sha256": "0" * 64}})
    # a jma_wc parent cannot take instance-width weights
    if _have_cached("jma_wc"):
        saved.pop("parent")
        torch.save(saved, tmp_path / "checkpoints" / "no_parent.pt")
        with pytest.raises(RuntimeError):
            sc.export_checkpoint(tmp_path / "checkpoints" / "no_parent.pt", "jma_wc", "e1_wrong", tmp_path / "exports")


def test_export_norm_order_and_refusal():
    assert sc.export_norm({"norm": "peak"}, {"extra": {"norm": {"norm": "std"}}}, {"data": {"norm": "std"}}) == ("peak", "checkpoint")
    assert sc.export_norm({}, {"extra": {"norm": {"norm": "std"}}}, {"data": {"norm": "peak"}}) == ("std", "run card extra.norm")
    assert sc.export_norm({}, None, {"data": {"norm": "peak"}}) == ("peak", "config data.norm")
    assert sc.export_norm({"norm": "peak"}, None, None, override="std") == ("std", "--norm")
    with pytest.raises(ValueError, match="unknown"):
        sc.export_norm({}, {"extra": {}}, {"data": {}})
    with pytest.raises(ValueError, match="expected one of"):
        sc.export_norm({"norm": "minmax"}, None, None)


# ── the window normalisation contract ────────────────────────────────────────

def test_std_path_is_unchanged_and_peak_is_per_component():
    from waveform_contract import NORMS, _normalise_peak, _normalise_std, normalise_waveform
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(3, 3001)) * np.array([[5.0], [0.2], [40.0]]) + np.array([[1.0], [-3.0], [7.0]])).astype(np.float32)
    assert NORMS == ("std", "peak")
    std = normalise_waveform(x, "std")
    assert np.array_equal(std, _normalise_std(x))
    d = x - x.mean(axis=-1, keepdims=True)
    assert np.allclose(std, np.clip(d / d.std(axis=-1, keepdims=True), -10, 10), atol=1e-6)   # the pinned formula
    peak = normalise_waveform(x, "peak")
    assert np.array_equal(peak, _normalise_peak(x))
    assert np.allclose(np.abs(peak).max(axis=-1), 1.0) and np.allclose(peak.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(peak, d / np.abs(d).max(axis=-1, keepdims=True), atol=1e-6)
    flat = np.zeros((3, 100), np.float32)
    flat[1] = 2.5
    flat[2] = 1e-7 * np.sin(np.arange(100))                          # a live channel in small physical units
    out = normalise_waveform(flat, "peak")
    assert np.array_equal(out[:2], np.zeros((2, 100), np.float32))
    assert np.isclose(np.abs(out[2]).max(), 1.0)
    with pytest.raises(ValueError, match="normalisation"):
        normalise_waveform(x, "minmax")


@needs_torch
def test_peak_normalisation_equals_seisbench_annotate_batch_pre():
    """waveform_contract._normalise_peak against seisbench 0.12.5
    PhaseNet.annotate_batch_pre with norm="peak" (phasenet.py lines 192 and
    202-204: per-component demean, divide by max |x| over time + 1e-10) on a
    random batch with offsets and per-component scales; and the std path
    against the same routine with norm="std" up to torch's unbiased std."""
    import torch
    import seisbench.models as sbm
    from waveform_contract import normalise_waveform
    rng = np.random.default_rng(1)
    x = (rng.normal(size=(8, 3, 3001)) * rng.uniform(0.1, 50, size=(8, 3, 1)) + rng.uniform(-20, 20, size=(8, 3, 1))).astype(np.float32)
    x[2, 1] = 0.7                                                    # a constant channel (DC only)
    ours = np.stack([normalise_waveform(w, "peak") for w in x])
    ref = sbm.PhaseNet(norm="peak").annotate_batch_pre(torch.from_numpy(x), {}).numpy()
    live = np.ones(x.shape[:2], bool)
    live[2, 1] = False
    assert np.allclose(ours[live], ref[live], atol=1e-5), np.abs(ours[live] - ref[live]).max()
    assert np.allclose(np.abs(ours[live]).max(axis=-1), 1.0)
    # the documented difference: SeisBench serves the float32 residue of a DC
    # channel at unit amplitude, the loader keeps it at zero
    assert np.abs(ours[2, 1]).max() < 1e-6 and np.abs(ref[2, 1]).max() > 0.9
    ours_std = np.stack([normalise_waveform(w, "std") for w in x])
    ref_std = sbm.PhaseNet(norm="std").annotate_batch_pre(torch.from_numpy(x), {}).numpy()
    ok = np.abs(ref_std) < 10                                        # outside the loader's clip the paths differ by design
    assert np.allclose(ours_std[ok], ref_std[ok] * np.sqrt(3001 / 3000), rtol=1e-5, atol=1e-5)   # torch std is unbiased
    assert not np.allclose(ours, ours_std)


@needs_torch
def test_resolve_norm_from_config_or_parent(monkeypatch):
    import types
    import seisbench.models as sbm
    import manifest_data_module as mdm
    assert mdm.resolve_norm({"data": {"norm": "std"}, "model": {"pretrained": {"model_name": "instance"}}}) == ("std", "config")
    with pytest.raises(ValueError, match="data.norm"):
        mdm.resolve_norm({"data": {"norm": "minmax"}})
    fake = {"instance": "peak", "jma_wc": "std", "odd": "minmax"}
    monkeypatch.setattr(sbm.PhaseNet, "from_pretrained", classmethod(lambda cls, name: types.SimpleNamespace(norm=fake[name])))
    assert mdm.resolve_norm({"data": {}, "model": {"pretrained": {"model_name": "instance"}}}) == ("peak", "parent instance")
    assert mdm.resolve_norm({}) == ("std", "parent jma_wc")            # the finetune.py default parent
    with pytest.raises(ValueError, match="declares norm"):
        mdm.resolve_norm({"model": {"pretrained": {"model_name": "odd"}}})


@needs_instance
def test_parent_norm_from_the_cached_weights():
    import manifest_data_module as mdm
    assert mdm.parent_norm("instance") == "peak"
    assert mdm.resolve_norm(_load("base") | {"data": {}}) == ("peak", "parent instance")
    if _have_cached("jma_wc"):
        assert mdm.parent_norm("jma_wc") == "std"
        assert mdm.resolve_norm(_load("base_jma_wc") | {"data": {}}) == ("std", "parent jma_wc")


def test_run_card_records_data_norm():
    import run_card as rc
    section = rc.config_section(_load("base"), None)
    assert section["norm"] == "peak" and section["training"]["freeze_bn_stats"] is False
    assert rc.config_section({"data": {}}, None)["norm"] is None          # a legacy config: null, never a guess
