"""Paired station-block bootstrap of matched-budget recall differences from the 35A artifacts.

The reference frame of every window comes from the scorer's own loader
(heldout_testset_score._load_sequence), so the denominators match rows.parquet
up to references on uncovered samples; matched references come from
matches.parquet at each model's matched-budget threshold. Blocks are stations;
P and S of one station move together. Usage: paired_bootstrap.py <root> with
runs under <root>/scores and <root>/scores_dense.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import heldout_testset_score as hts  # noqa: E402  (the scorer's own reference loader)

B = Path(sys.argv[1])
rng = np.random.default_rng(0); N = 2000
rows = []
for run in sorted(p + "/" for p in glob.glob(str(B / "scores_dense/*/*")) + glob.glob(str(B / "scores/*/*"))):
    key = run.rstrip("/").split("/")[-2]
    budget = pd.read_parquet(run + "budget.parquet"); matches = pd.read_parquet(run + "matches.parquet")
    models = pd.read_csv(run + "models.csv")
    names = dict(zip(models.model_id, models.model)) if "model" in models else {m: m for m in budget.model_id}
    windows, _ = hts._load_sequence(key)          # the scorer's own streams and reference frames, every window
    ref = pd.concat([w["reference"].assign(window_id=w["window_id"]) for w in windows], ignore_index=True)
    ref["ref_key"] = ref["window_id"].astype(str) + "|" + ref["ref_id"].astype(str)
    # per (model, phase): matched ref_ids at the budget threshold
    per_sta = {}
    for _, b in budget.iterrows():
        if not b.within_tolerance: continue
        m = matches[(matches.model_id == b.model_id) & (matches.phase == b.phase) & np.isclose(matches.threshold, b.threshold)]
        hit = set(m.window_id.astype(str) + "|" + m.ref_id.astype(str))
        r = ref[ref.phase == b.phase]
        per_sta[(names.get(b.model_id, b.model_id), b.phase)] = r.assign(hit=r.ref_key.isin(hit)).groupby("station")["hit"].agg(["sum", "count"])
    for phase in ("P", "S"):
        base = per_sta.get(("jma_wc", phase))
        if base is None: continue
        for cand in ("instance", "quakescope2026"):
            c = per_sta.get((cand, phase))
            if c is None: continue
            tab = base.join(c, lsuffix="_b", rsuffix="_c", how="outer").fillna(0)
            s_b, n_b, s_c = tab["sum_b"].to_numpy(), tab["count_b"].to_numpy(), tab["sum_c"].to_numpy()
            diff = s_c.sum() / n_b.sum() - s_b.sum() / n_b.sum()
            idx = rng.integers(0, len(tab), size=(N, len(tab)))
            boots = (s_c[idx].sum(1) - s_b[idx].sum(1)) / np.maximum(n_b[idx].sum(1), 1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(key=key, phase=phase, candidate=cand, recall_parent=round(s_b.sum() / n_b.sum(), 3),
                             recall_candidate=round(s_c.sum() / n_b.sum(), 3), diff=round(diff, 3), lo=round(lo, 3), hi=round(hi, 3),
                             n_stations=len(tab), n_ref=int(n_b.sum())))
out = pd.DataFrame(rows); pd.set_option("display.width", 200); print(out.to_string()); out.to_csv(B / "paired_station_bootstrap.csv", index=False)
