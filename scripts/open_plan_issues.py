#!/usr/bin/env python3
"""Render the applied roadmap for existing GitHub issues #33–#50.

The approved titles, bodies and milestone assignments live in the verified
snapshot docs/issue_revision_2026-09-10/applied_roadmap.json. This command
is offline and read-only with respect to GitHub; it cannot create duplicates.

    python scripts/open_plan_issues.py
    python scripts/open_plan_issues.py --check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "docs/issue_revision_2026-09-10/applied_roadmap.json"
DOC = REPO_ROOT / "docs/2026-09-09_issue_plan.md"


def render(source=SOURCE):
    plan = json.loads(Path(source).read_text())
    issues = plan["issues"]
    if sorted(i["number"] for i in issues) != list(range(33, 51)):
        raise ValueError("Expected each existing issue #33–#50 exactly once")
    out = ["# Applied training roadmap", "",
           "GitHub issues #33–#50 were updated on 2026-09-10. "
           "Rendered offline by `scripts/open_plan_issues.py` from the verified "
           "[applied snapshot](issue_revision_2026-09-10/applied_roadmap.json).", "",
           "See the [execution branches and order](2026-09-10_issue_execution.md). "
           "Earlier issue descriptions and scientific proposals are historical; "
           "the checkpoint gates below govern new work."]
    for milestone in sorted(plan["milestones"], key=lambda m: m["number"]):
        out.extend(["", "## " + milestone["title"], "", milestone["description"]])
        for issue in issues:
            if issue["milestone"]["number"] == milestone["number"]:
                out.extend(["", f"### [#{issue['number']} — {issue['title']}]({issue['url']})",
                            "", issue["body"].strip()])
    return "\n".join(out).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="Check the committed rendering without modifying files")
    args = ap.parse_args()
    content = render()
    if args.check:
        if not DOC.exists() or DOC.read_text() != content:
            ap.exit(1, "Issue plan differs from the applied snapshot; rerun this script.\n")
        print("Issue plan matches the applied snapshot (18 existing issues).")
    else:
        DOC.write_text(content)
        print(f"Wrote {DOC}")


if __name__ == "__main__":
    main()
