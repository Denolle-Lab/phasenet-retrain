"""Render the SU implementation plan from task definitions and issue receipts.

Offline only; never creates or modifies GitHub issues.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
definitions = json.loads((HERE / "issue_definitions.json").read_text())
receipt_path = HERE / "github_issues.json"
receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
issues = receipt.get("tasks", {})


def issue_link(task_id):
    entry = issues.get(task_id)
    return f"[#{entry['number']}]({entry['url']})" if entry else "Pending creation"


def dependencies(text):
    for task in definitions["tasks"]:
        text = text.replace("{{" + task["id"] + "}}", f"{task['id']} ({issue_link(task['id'])})")
    return text


parts = ["""# SU picker implementation plan

Owner: **Akash Kharita (@Akashkharita)**. All SU implementation issues are assigned
to him and labeled **`su-picker`**. Scientific coordination: Marine Denolle;
an independent annotation reviewer and the sprint start date still need to be
recorded. This plan is the entry point for future SU work; implementation remains
open. The [design rationale](2026-09-08_surface_event_picker_plan.md),
[audit](2026-09-10_surface_event_picker_audit.md) and
[review response](2026-09-10_surface_picker_review_response.md) explain the choices.

## Goal and first deliverable

Build SUNet, a retrospective station-level surface-event onset picker. Its U
output marks the first observable seismic arrival associated with a defined
surface process. U is not an earthquake P phase, source origin time, flow-front
passage or location. Supported processes and instruments must follow the evidence.
Classification, duration, network association and location are optional later
deliverables with their own tests.

The first bounded corpus uses post-2002 Rainier/St. Helens continuous records and
reviewed negatives from permitted stations, with Newberry/Hood development
station-days after overlap checks. Keep separate calibration and acceptance
assets. Catalogue absence is not proof of a negative. The starting model candidate
is UN softmax at 50 Hz/6000 samples (120 s); UPN, initialization and generalization
changes follow controlled pilot evidence.

## How to execute

1. Start SU-01; release the taxonomy and pilot partitions. SU-02 read-only census
   can start meanwhile. Do not train or mine until the relevant exclusions pass.
2. Complete the bounded SU-02 corpus and SU-03 preprocessing/model contracts.
   Synthetic SU-03 checks need not wait for a bulk harvest.
3. Complete SU-04 continuous baselines, separate calibration and acceptance
   eligibility/protocol freeze. Then run the SU-05 minimal pilot.
4. Use SU-06 only for diagnostic, controlled improvements. SU-07 is optional:
   document whether each additional stage earns its place.
5. Freeze one candidate for SU-08 acceptance. Publish either a supported release
   decision or a no-release result; an inconclusive experiment is useful evidence.

Work on one ready checkpoint branch/PR at a time, using a separate worktree per
session. Create branches from the latest reviewed surface integration head,
`audit/2026-09-10-surface-picker`. Each PR links its SU issue and includes code,
config/data hashes, validation results and remaining criteria. Close a task only
when its acceptance criteria are satisfied or an explicitly permitted no-go/omit
decision is documented. Plans and assignments do not count as completed work.

SU-01 and SU-02 each have a proposed 10-working-day timebox. T0 is the agreed
start with access and reviewer availability; no calendar due date is implied.
The SU-02 timebox delivers a census and bounded pilot, not all post-2002 data.
Estimate later compute/review time from SU-02/03/05 measurements.

## Shared dependencies with the earthquake picker

Reuse #33/#44 exclusions and suite access, #34 time/resampling/normalization/export
contracts, #35 continuous scoring, #37 evaluability and #38 calibration; #43
supplies compatible augmentation transforms. Keep their earthquake scope and
assignees unchanged. Link to the required checkpoint or verified artifact rather
than waiting for an entire multiphase issue to close. Do not build a separate SU
resampler or silently reuse the known-broken loader/scorer.

## Task index
"""]
if receipt.get("epic"):
    epic = receipt["epic"]
    parts.append(f"Roadmap tracker: [#{epic['number']}]({epic['url']}).\n")
parts.extend(["| Task | GitHub | Deliverable | Branch |", "|---|---|---|---|"])
for task in definitions["tasks"]:
    title = task["title"].split("] ", 1)[1]
    parts.append(f"| {task['id']} | {issue_link(task['id'])} | {title} | `{task['branch']}` |")
parts.append("\n## Task specifications\n")
for task in definitions["tasks"]:
    title = task["title"].split("] ", 1)[1]
    parts.extend([f"### {task['id']} — {title}", "", f"Issue: {issue_link(task['id'])}. Owner: @Akashkharita.",
                  "", f"**Dependencies:** {dependencies(task['dependencies'])}", ""])
    if task.get("timebox"):
        parts.extend([f"**Timebox:** {task['timebox']}.", ""])
    parts.extend(["**Work:**", "", *[f"- {line}" for line in task["work"]], "",
                  "**Done when:**", "", *[f"- [ ] {line}" for line in task["acceptance"]], ""])
parts.append("""## Tracking and evidence

The task definitions live in `su_picker/issue_definitions.json`; the verified
creation snapshot is `su_picker/github_issues.json`. The unchecked lists above
describe acceptance criteria, not synchronized live completion status. Follow
the linked GitHub issues for progress. Regenerate this document offline with
`python docs/su_picker/render_plan.py` after changing definitions or issue links.

Every experiment preserves model/preprocessing versions, source and manifest
hashes, split/exposure history, candidate/reference assignments, seeds, calibration
and valid-hour denominators. Sealed model scoring happens only after candidate
freeze; reference QA is separately logged. A panel used to redesign the model is
no longer blind. No acceptance, training or production deployment has been
completed by creating this roadmap.
""")
(HERE.parent / "SU_PICKER_IMPLEMENTATION_PLAN.md").write_text("\n".join(parts))
