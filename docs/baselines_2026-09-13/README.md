# Corrected baselines on the regression and development cases (35B, provisional)

*2026-09-13, branch `issue/35b-baseline-artifacts` from
`integrate/2026-09-13-checkpoints` (PR #76). Checkpoint 35B of issue #35,
run on the laptop CPU with SeisBench 0.12.5, PyTorch 2.2.2, ObsPy 1.5.1,
Python 3.11.13. Marked provisional because 34C (benchmark timebase and
deployment parity) is not released; the annotations are the raw SeisBench
`annotate` output and are reused unchanged if 34C leaves that path alone.
All numbers in this directory come from the committed CSVs.*

## What was run

Three PhaseNet weights through the 35A scoring engine
(`scripts/heldout_testset_score.py`, per-threshold extraction with the
production trigger rule, maximum-cardinality matching, 0.5 s tolerance,
manual-tier references only) on the seven cases whose role is regression
or development in `configs/evaluation_suites.json`, i.e. the cases 37A
certified (`data/heldout_testset/evaluability.csv`). No acceptance case
was touched; the 44A access log holds one `model_scoring` record per case
(`runs.csv` lists the access ids and the model state hashes).

| Weight | Source | File sha256 | Role in the strategy |
|---|---|---|---|
| `jma_wc` | SeisBench cache, `jma_wc.pt.v1` | c2ad5815… | parent, campaign picker |
| `instance` | SeisBench cache, `instance.pt.v2` | dddcc3d8… | second parent candidate (E4) |
| `quakescope2026` | SeisBench cache, `quakescope2026.pt.v1` | 29af22a1… | the deployed export of `jma_wc_ft_global_v7` |

Thresholds 0.02 to 0.90 in steps of 0.02 (45 attained operating points per
weight and phase). The first pass used the notebooks' nine-point grid and
could not match the budget for `instance` on three cases and for the v7
export on four, because `instance` at its lowest grid point 0.05 already
emits fewer picks than `jma_wc` at 0.3. The dense grid is the one to keep.

## Matched-budget recall, paired station-block bootstrap

The budget per case and phase is the number of picks `jma_wc` emits at
0.3; each other weight is read at its attained threshold closest to that
count (always within 10 % here). Recall is matched references over covered
references, summed over the case's windows. Δ is candidate minus parent on
the same references; the interval is a 95 % percentile bootstrap over
stations as blocks (2,000 draws, `paired_bootstrap.py`), P and S of one
station resampled together. Six stations per case, three for Samos.

| Case | Phase | n ref | jma_wc | instance | Δ instance [95 %] | quakescope2026 (v7) | Δ v7 [95 %] |
|---|---|--:|--:|--:|--:|--:|--:|
| adriatic_2022 | P | 400 | 0.848 | 0.878 | +0.030 [+0.005, +0.058] | 0.838 | -0.010 [-0.027, +0.005] |
| adriatic_2022 | S | 357 | 0.538 | 0.692 | +0.154 [+0.047, +0.218] | 0.543 | +0.006 [-0.035, +0.053] |
| corinth_thiva_2020 | P | 108 | 0.806 | 0.833 | +0.028 [-0.026, +0.087] | 0.824 | +0.019 [-0.018, +0.060] |
| corinth_thiva_2020 | S | 87 | 0.437 | 0.609 | +0.172 [+0.080, +0.310] | 0.471 | +0.034 [-0.011, +0.077] |
| etna_2022_2024 | P | 597 | 0.972 | 0.990 | +0.018 [+0.008, +0.029] | 0.980 | +0.008 [+0.002, +0.015] |
| etna_2022_2024 | S | 395 | 0.539 | 0.722 | +0.182 [+0.113, +0.239] | 0.557 | +0.018 [-0.046, +0.064] |
| kaikoura_2016 | P | 357 | 0.703 | 0.751 | +0.048 [+0.000, +0.132] | 0.695 | -0.008 [-0.033, +0.033] |
| kaikoura_2016 | S | 458 | 0.520 | 0.705 | +0.186 [+0.146, +0.229] | 0.502 | -0.017 [-0.068, +0.034] |
| norcia_2016 | P | 701 | 0.795 | 0.819 | +0.024 [-0.018, +0.062] | 0.786 | -0.009 [-0.024, +0.004] |
| norcia_2016 | S | 656 | 0.779 | 0.886 | +0.107 [+0.054, +0.151] | 0.767 | -0.012 [-0.026, +0.005] |
| samos_2020 | P | 34 | 0.294 | 0.471 | +0.176 [+0.000, +0.250] | 0.382 | +0.088 [+0.000, +0.133] |
| samos_2020 | S | 24 | 0.083 | 0.208 | +0.125 [+0.000, +0.214] | 0.083 | +0.000 [+0.000, +0.000] |
| thessaly_2021 | P | 395 | 0.747 | 0.770 | +0.023 [-0.007, +0.055] | 0.732 | -0.015 [-0.034, +0.002] |
| thessaly_2021 | S | 331 | 0.402 | 0.444 | +0.042 [+0.000, +0.090] | 0.399 | -0.003 [-0.019, +0.018] |

Thresholds at which the budget is attained:

| Case | Phase | budget (jma_wc picks at 0.3) | thr instance | thr v7 |
|---|---|--:|--:|--:|
| adriatic_2022 | P | 930 | 0.06 | 0.22 |
| adriatic_2022 | S | 531 | 0.14 | 0.22 |
| corinth_thiva_2020 | P | 1308 | 0.04 | 0.14 |
| corinth_thiva_2020 | S | 846 | 0.04 | 0.20 |
| etna_2022_2024 | P | 1767 | 0.04 | 0.22 |
| etna_2022_2024 | S | 534 | 0.20 | 0.24 |
| kaikoura_2016 | P | 1819 | 0.04 | 0.26 |
| kaikoura_2016 | S | 1032 | 0.10 | 0.30 |
| norcia_2016 | P | 1347 | 0.04 | 0.26 |
| norcia_2016 | S | 1198 | 0.04 | 0.28 |
| samos_2020 | P | 857 | 0.10 | 0.20 |
| samos_2020 | S | 285 | 0.24 | 0.26 |
| thessaly_2021 | P | 1733 | 0.04 | 0.26 |
| thessaly_2021 | S | 1057 | 0.12 | 0.26 |

## Reading

- **`instance` leads `jma_wc` on every case and phase at the parent's
  budget.** On S the margin is large and its interval excludes zero on six
  of seven cases (+0.11 to +0.19); on P the margin is +0.02 to +0.05 and
  the interval excludes zero on Adriatic, Etna and, at the edge, Kaikōura.
  Norcia, Adriatic and Etna are Italian: the INSTANCE weights may have seen
  those regions in training, and fine-tune exclusions cannot remove that.
  Kaikōura, Corinth–Thiva and Thessaly carry no such caveat and show the
  same S lead.
- **The v7 export is indistinguishable from its parent.** Every Δ interval
  spans or touches zero except Etna P (+0.008 [+0.002, +0.015]); the
  point estimates are below the parent on P for the four mainshock cases.
  This is the corrected-scorer version of the 2026-09-07 conclusion.
- **Samos** has 34 P and 24 S references on three stations; 37A marks it
  not evaluable for either endpoint, and the intervals here say the same.
- **Corinth–Thiva S** (87 references) is below the 37A pick-scoring
  minimum; the row is reported, not claimed.
- The v7 export attains the parent's budget at thresholds 0.14–0.30, and
  `instance` at 0.04–0.24: the three weights live on different probability
  scales, which is why a shared 0.3 threshold (the notebooks' convention)
  compared different operating points. Thresholds are per weight (#38).

## Caveats that keep this provisional

1. **34C not released.** The deployed path (SeisBench `annotate` with each
   weight's own `filter_factor`, normalisation and blinding) is what was
   scored; parity with the training loader's preprocessing is unverified.
2. **Uncovered references.** The bootstrap script rebuilds the reference
   frame from the scorer's loader and does not remove references on
   uncovered samples (7 on Adriatic P), so its parent recall there is
   0.848 against the scorer's 0.863. The matched-budget table uses the
   scorer's own rows.
3. **One window per mainshock case** (2–3 h after the mainshock), so no
   claim about the first 48 hours; the coverage rule of 36A will refuse it.
4. **Station count.** Six stations per case give wide intervals on P; the
   S conclusion is robust, the P conclusion is directional.
5. No event-level scoring yet (36B needs the pick store, which the runs
   wrote under `data/evaluation/`, gitignored; regenerate with the command
   below).

## Reproduce

```bash
# waveforms live in a clone that ran scripts/build_heldout_testset.py; link them in
for k in kaikoura_2016 norcia_2016 thessaly_2021 samos_2020 adriatic_2022 etna_2022_2024 corinth_thiva_2020; do
  ln -s <clone>/data/heldout_testset/$k/waveforms data/heldout_testset/$k/waveforms; done
python scripts/heldout_testset_score.py --all --weights jma_wc instance quakescope2026 \
    --budget-reference jma_wc --thresholds $(python -c "print(' '.join(f'{x/100:.2f}' for x in range(2, 92, 2)))") \
    --out-dir data/evaluation/scores --annotations-root data/evaluation/annotations --out docs/baselines_2026-09-13/score_rows.csv
python docs/baselines_2026-09-13/paired_bootstrap.py data/evaluation   # reads every run under <root>/scores and <root>/scores_dense
```

Runtime on the laptop CPU: 219 s for the three weights and seven cases
(annotation), 114 s for the dense-grid extraction from the stored
annotations.

## Files

| File | Content |
|---|---|
| `score_rows.csv` | every (case, model, phase, threshold) row, window and aggregate scope, with the access id |
| `matched_budget.csv` | recall and attained threshold per weight at the parent's budget |
| `paired_station_bootstrap.csv` | the Δ table above |
| `runs.csv` | access id and model state sha256 per case and weight |
| `paired_bootstrap.py` | the bootstrap, run against the per-run `matches.parquet` |

## What 35C adds

Calibrated comparison at the #38 nuisance budget instead of the parent's
emitted count, the parent-exposure provenance label per baseline, and v11
once its weights are exported from the server.
