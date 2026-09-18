# Audit brief: why the QuakeScope 2026 fine-tune does not generalise

*Prepared 2026-09-07 by Marine Denolle with Claude, from the QuakeScope side. This
is the opening prompt for a Claude session in this repository. Paste the
"Prompt" section into a new session started in `~/GitHub/phasenet-retrain`
(or on the lab server where the SeisBench cache and the manifests live).*

---

## Prompt

You are working in `Denolle-Lab/phasenet-retrain`, cloned at
`~/GitHub/phasenet-retrain` (commit 77f581d or later), the repository that
produced the fine-tune `jma_wc_ft_global_v7`, deployed in QuakeScope as
`quakescope2026`. Two external benchmarks now show that this fine-tune
recovers fewer analyst picks than its parent `jma_wc` everywhere it has been
tested: the western United States, and Kaikōura 2016, Norcia 2016 and
Thessaly 2021 against GeoNet, INGV and NOA picks. A read-only audit of this
repository has already been done. Your job is to test its hypotheses on the
data and checkpoints that only exist on the lab server, decide whether the
fine-tune can be rescued, and make the external sequences permanent held-out
test assets.

Read these two files before anything else, in this order:

1. `docs/2026-09-07_training_history_audit.md`: what the twenty versions did,
   why v7 was chosen, and what failed in the data, augmentation, loss and
   evaluation, with `file:line` citations.
2. `docs/2026-09-07_generalization_audit_prompt.md`, sections "What is known
   from outside this repository" and "Hypotheses": the external numbers and
   the hypotheses H00 to H4, one of which (H0) has already been tested and
   dropped.

Ground rules. Work read-only until task 4. Cite `file:line` for every claim
about this repository and give every number a bootstrap interval
(`scripts/metrics.py::bootstrap_ci`). Do not overwrite
`models/jma_wc_ft_global_v7.pt`, do not commit to `main`, and do not start a
training run before task 5 is written and I have agreed to it. The training
data and manifests are in the SeisBench cache at `SEISBENCH_CACHE_ROOT`;
verify any manifest against `data/manifest_checksums.csv` before using it.
If a checkpoint or manifest the audit assumes exists is missing on the server,
say so and stop that branch rather than reconstructing it.

Tasks, in order.

1. **Make the test sequences independent.** Using the spatiotemporal join in
   `scripts/audit_parent_leakage.py`, count traces in `manifests_v2/train.csv`,
   `manifests_v2/val.csv` and the full local copies of `instancecounts`,
   `stead`, `crew`, `lendb`, `mlaapde`, `meier2019jgr`, `scedc`, `ceed`,
   `ross2018gpd` and `pnw` whose source origin falls in these windows:
   Norcia 42.83 N 13.11 E, 1°, 2016-10-30 06:50–08:50 and 2016-08-24 to
   2017-01-31; Kaikōura 42.69 S 173.02 E, 2°, 2016-11-13 11:13–14:13 and
   2016-11-13 to 2017-05-13; Thessaly 39.75 N 22.20 E, 1°, 2021-03-03
   10:26–13:26 and 2021-02-28 to 2021-05-31; Ridgecrest 35.77 N 117.60 W, 1°,
   2019-07-06 to 2019-08-06; Monroe 47.87 N 122.02 W, 1°, 2019-07-12 to
   2019-07-19. Report counts per dataset and window. Commit the result as an
   exclusion list of `(dataset, trace_name)` rows, wire it into
   `scripts/build_training_dataset.py` beside `benchmark_exclude`, and add a
   whole-year hold-out for 2016 and 2021. Nothing trained from now on may
   contain these.
2. **Re-score v7 against the parent honestly** on `clean_holdout` only: recall
   at matched pick budget (`scripts/threshold_independent_ranking.py`) by
   phase and distance bin, conditional MAE, and the noise-pool detection MCC,
   all with intervals. State whether any v7 advantage survives.
3. **Test H1 to H4 and H00** from the prompt file, one table each, with the
   script that produced it committed under `scripts/`. H2 needs the S
   fraction of the training manifest by distance bin; H1 needs the peak
   probability distributions on true arrivals for v7 and the parent; H00
   needs the v1–v20 ordering under the noise-pool detection MCC.
4. **Write the findings into `paper_draft.qmd`** as a new dated item in
   §Critical audit: the external results, what tasks 1 to 3 found, and whether
   the "more general" claim can be made. Retract in place; do not delete.
5. **Propose, do not run, a v21** only if task 3 identifies a cause the
   recipe can address. The proposal must state the data (analyst P and S
   only, label-error cleaned, exclusion list applied, no P-only global sets,
   no teleseismic rebalancing), the augmentation (random window position,
   real-noise superposition from the existing pool, band-limiting and
   resampling to 40–100 Hz, channel dropout), the loss (soft Gaussian
   targets), and the selection rule (a held-out split never read during
   development). If task 3 finds nothing the recipe can address, say that the
   parent stands and stop.

Acceptance test for any future checkpoint, which is outside this repository:
convert it with the path in
`~/GitHub/QuakeScope/docs/phasenet_v7_model_description.md` §Reproduction,
place it in the `quakescope2026` slot, and re-run
`~/GitHub/QuakeScope/tutorials/phasenet_sequence_comparison.ipynb` and
`phasenet_global_sequences.ipynb`. It must beat `jma_wc` at matched pick
budget on the non-US sequences. Until that happens the global campaign stays
on `jma_wc`.

Report in prose, numbers in tables, and say plainly when something could not
be verified.

---

## Notes for Marine (not part of the prompt)

- The external benchmarks are held-out for `jma_wc` by construction (Japan
  only) and cannot be made cleaner for it. For v7 they may be contaminated,
  which only strengthens the finding. For `instance`, Norcia is in-domain and
  should be discounted; Kaikōura and Thessaly are not, and its lead there is
  real.
- If the three sequences are to become permanent test assets, task 1's
  exclusion list is what makes them so; until it exists, no new fine-tune
  should be trained on the current manifests.
