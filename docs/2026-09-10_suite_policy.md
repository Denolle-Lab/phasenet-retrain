# Evaluation suite policy — repository part of 44A

`configs/evaluation_suites.json` is the source of current suite roles.
Kaikoura, Norcia and Thessaly are regression cases because their model outputs
have already been examined. The four existing development cases retain that
role. The other twelve entries are **protected acceptance candidates**, pending
exposure-history review, reference eligibility (#37B) and panel freeze (#44B).
This does not certify that they have never been examined. Calibration station
days remain unassigned (#38). Previously examined western US cases are listed
as external regression cases; they have no local scoring-registry entries.

`heldout_testset_score.py --all` selects only built regression/dev cases.
Explicit protected or unknown selections fail before loading any model or
waveforms, including mixed selections. Direct `score()` calls enforce the
same rule. There is no acceptance override. Run, for example:

```sh
python scripts/heldout_testset_score.py --sequence samos_2020 --weights jma_wc
```

`load_sequence()` remains available for reference QA on all registered cases.
It records a `reference_qa` access; scoring records `model_scoring` instead.
The append-only workflow log at `data/evaluation/access.jsonl` records a start
attempt before data loading/inference, not successful completion. Scoring
records include actual model state hashes, model class/default arguments,
scorer settings, Git commit, code and reference-artifact hashes, and the full
policy plus its hash. Score rows carry the access ID. Missing local artifacts
have null hashes. Copy this ignored log alongside run results when preserving
experiment evidence. Logging failure stops the operation.

The registry and newly generated index/dashboard use current roles. Historical
manifests retain their original roles and hashes as provenance; their stored
role is not permission to score. The committed index is updated without
rewriting those manifests. Previously generated dashboards are historical
artifacts until rebuilt.

This is a repository workflow guard, not a security boundary: direct raw-file
reads, arbitrary notebooks and the separate QuakeScope repository are outside
its enforcement. Waveform contents are not independently hashed at each run;
the recorded manifest identifies their build-time inventory. Cross-repository
adapters, exposure-history reconciliation and a calibrated/frozen acceptance
release protocol remain open. Do not check off all of 44A or close #44 on this
PR. Threshold extraction and matching bugs remain assigned to #35; these
guards do not validate the scientific accuracy of the current scorer.
