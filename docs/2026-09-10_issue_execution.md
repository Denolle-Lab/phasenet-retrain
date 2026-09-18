# Issue branches and execution order

The integration branch is `audit/2026-09-07-generalization`. The applied
scope is recorded in [the rendered roadmap](2026-09-09_issue_plan.md).
GitHub issues are open work, not completion certificates.

Work sequentially on one checkpoint branch and one reviewable PR at a time.
Branch from the latest integration head when a checkpoint starts; do not
create eighteen branches now that would immediately become stale. Use a
separate worktree so the existing surface-event plan edit stays untouched.
PRs target the integration branch until the full campaign is ready for main.

Use `Refs #44` (or the appropriate issue), not a closing keyword, when a PR
releases only one checkpoint. Check a GitHub checkpoint only after its stated
acceptance criteria are met across every relevant repository. Merge the
reviewed checkpoint before branching the next dependent checkpoint. A server
or access dependency can defer that checkpoint while the next independently
ready one proceeds; record the deferred work explicitly.

| Issue | First branch | Later checkpoint branches |
|---|---|---|
| #44 | `issue/44a-suite-policy` | `issue/44b-panel-freeze` |
| #34 | `issue/34a-loader-alignment` | `issue/34b-row-forensics`, `issue/34c-model-benchmark-contracts` |
| #33 | `issue/33a-versioned-exclusions` | — |
| #35 | `issue/35a-continuous-scorer` | `issue/35b-baseline-artifacts`, `issue/35c-calibrated-baselines` |
| #37 | `issue/37a-suite-evaluability` | `issue/37b-panel-eligibility` |
| #39 | `issue/39a-source-census` | — |
| #38 | `issue/38a-threshold-calibration` | — |
| #36 | `issue/36a-event-scorer` | `issue/36b-event-baselines` |
| #41 | `issue/41a-label-policy` | `issue/41b-pilot-review` |
| #40 | `issue/40a-pilot-corpus` | `issue/40b-sharded-corpus` |
| #46 | `issue/46a-alignment-experiment` | `issue/46b-recipe-pilot`, `issue/46c-init-kd-confirmation` |
| #42 | `issue/42a-noise-pilot` | `issue/42b-noise-expansion` |
| #43 | `issue/43a-augmentation` | — |
| #47 | `issue/47a-augmentation-pilot` | `issue/47b-augmentation-confirmation` |
| #45 | `issue/45a-scaling-budget` | `issue/45b-scaling-curve` |
| #48 | `issue/48a-context-contract` | `issue/48b-width-context-experiments` |
| #49 | `issue/49a-candidate-freeze` | `issue/49b-acceptance-deployment` |
| #50 | `issue/50a-distant-p-pilot` | `issue/50b-distant-p-acceptance` |

The first implementation is the repository-local part of **44A**: publish
suite roles, classify the previously inspected cases as regression, and
protect model scoring on provisional acceptance cases. Cross-repository
QuakeScope entrypoints must adopt the policy before 44A is considered fully
released. No sealed evaluation is authorized by this implementation.

Next local tasks are 34A (loader/timebase repair), 33A (exclusions after the
role policy is adopted), 35A (scorer), and 37A (evaluability). Source census
39A can follow the initial role policy. Read-only 34B server forensics does
not require a training run. Follow the actual checkpoint dependencies for
all subsequent work, rather than the numerical issue order.

The first training experiment is 46A, only after 33A, 34A/B/C, 35C and 44B:
same rows and recipe, legacy versus corrected alignment. Bulk scaling #45
remains downstream of the reviewed corpus and controlled recipe/noise pilots.

For every PR, record the checkpoint scope, validation commands and results,
remaining acceptance criteria, and code/config/data provenance. Never count
a metadata audit as model acceptance or a new description as completed work.

`scripts/open_plan_issues.py` now renders the verified applied snapshot
offline; `--check` detects drift. The old issue-creation workflow is removed
so rendering cannot reopen the obsolete plan or create duplicate issues.
