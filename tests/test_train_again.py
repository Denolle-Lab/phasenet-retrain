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
    base, cfg = _load("base"), _load(arm)

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

    def fake_score(key, models, thresholds, annotations_root, out_dir, budget_reference, budget_threshold):
        calls.append((key, sorted(models), budget_reference, tuple(thresholds)))
        rec = {"instance": {"P": 0.8, "S": 0.6}, "jma_wc": {"P": 0.7, "S": 0.5}, "cand": {"P": 0.9, "S": 0.65}}
        thr = {"instance": 0.3, "jma_wc": 0.4, "cand": 0.1}
        return hts.ScoreResult(key=key, access_id="a1", rows=pd.DataFrame({"key": [key], "scope": ["aggregate"]}),
                               budget=_budget_rows(key, list(models), rec, thr), models={m: f"id_{m}" for m in models})

    monkeypatch.setattr(sc.hts, "score", fake_score)
    monkeypatch.setattr(sc, "dev_cases", lambda out_root=None: ["samos_2020", "corinth_thiva_2020"])
    table, summary = sc.score_candidate("cand", object(), parents=("instance", "jma_wc"), thresholds=[0.1, 0.3],
                                        out_root=tmp_path, load_parent=lambda p: f"weights:{p}")
    assert [c[0] for c in calls] == ["samos_2020", "corinth_thiva_2020"]
    assert calls[0][1] == ["cand", "instance", "jma_wc"] and calls[0][2] == "instance" and calls[0][3] == (0.1, 0.3)
    assert summary == tmp_path / "summary" / "cand"
    written = pd.read_csv(summary / "matched_budget.csv")
    assert len(written) == 4 and written["delta_vs_instance"].round(3).tolist() == [0.1, 0.05, 0.1, 0.05]
    assert (summary / "score_rows.csv").exists() and (summary / "budget_rows.csv").exists()
    with pytest.raises(ValueError, match="budget-reference"):
        sc.score_candidate("cand", object(), parents=("jma_wc",), budget_reference="instance", out_root=tmp_path,
                           load_parent=lambda p: p)
    with pytest.raises(ValueError, match="collides"):
        sc.score_candidate("instance", object(), parents=("instance",), out_root=tmp_path, load_parent=lambda p: p)
    monkeypatch.setattr(sc, "dev_cases", lambda out_root=None: [])
    with pytest.raises(ValueError, match="no built development case"):
        sc.score_candidate("cand", object(), out_root=tmp_path, load_parent=lambda p: p)


def test_dev_cases_follow_the_policy_roles(tmp_path, monkeypatch):
    import evaluation_policy as policy
    roles = policy.load_policy()["roles"]
    dev = [k for k, r in roles.items() if r == "dev"]
    assert set(dev) == {"samos_2020", "adriatic_2022", "etna_2022_2024", "corinth_thiva_2020"}
    for k in ("samos_2020", "kaikoura_2016"):
        (tmp_path / k).mkdir()
        (tmp_path / k / "manifest.json").write_text("{}")
    monkeypatch.setattr(sc.hts, "OUT_ROOT", tmp_path)
    assert sc.dev_cases() == ["samos_2020"]                      # a built regression case is not a dev case


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
    reloaded, sidecar = sc.export_checkpoint(ck, "instance", "e1_test", tmp_path / "exports",
                                             card={"checkpoint": {"sha256": sc.sha256_file(ck)}})
    assert reloaded.norm == "std" and sidecar["parent_norm"] == "peak" and sidecar["export_norm"] == "std"
    assert sidecar["epoch"] == 3 and sidecar["n_teacher_tensors_stripped"] > 0
    assert torch.equal(reloaded.in_bn.weight, m.model.in_bn.weight)
    assert sidecar["pt_sha256"] == sc.sha256_file(tmp_path / "exports" / "e1_test.pt")
    meta = json.loads((tmp_path / "exports" / "e1_test.json").read_text())
    assert meta["model_args"]["norm"] == "std" and meta["model_args"]["phases"] == "PSN"
    assert "epoch 3" in meta["docstring"]
    again = hts.load_weights(str(tmp_path / "exports" / "e1_test.pt"))     # the .pt spelling works too
    assert torch.equal(again.in_bn.weight, reloaded.in_bn.weight)
    with pytest.raises(ValueError, match="sha256"):
        sc.export_checkpoint(ck, "instance", "e1_bad", tmp_path / "exports", card={"checkpoint": {"sha256": "0" * 64}})
    # a jma_wc parent cannot take instance-width weights
    if _have_cached("jma_wc"):
        with pytest.raises(RuntimeError):
            sc.export_checkpoint(ck, "jma_wc", "e1_wrong", tmp_path / "exports")
