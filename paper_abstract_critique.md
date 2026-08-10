# Abstract/Headline-Claim Critique — Response to Reviewer-Style Questions

Answers to a set of adversarial questions about `paper_results_final.md`'s
framing of the headline result (v7 vs `jma_wc`) and the v20 negative result.
Grounded in the numbers already in that document (§4a–§4e, §6, §7).

---

## On the headline claim itself

### 1. Which number is the actual headline — 0.290s, or something else?

The 0.290s/9% figure is the **leaky-trained** v7, scored against a
**leak-corrected** benchmark population — a mismatched pairing (clean eval,
dirty model). The genuinely matched comparison is in §4c: v7 retrained on the
fully leak-free pipeline scores **0.321s**, essentially tied with jma_wc's
0.319s — arguably a tiny *loss*, not a 9% win. That is the real headline, and
it is a much less exciting paper than "fine-tuning improves timing 9%."

No, it is not safe to assume an abstract reader will treat 0.290s as clean —
it isn't, and the document currently lets §4a's clean-*sounding* framing carry
the abstract while §4c's actual clean number sits three sections later. If
this goes in an abstract, the 0.32-vs-0.32 wash should be the headline, with
0.290s reported only as "what the currently deployed checkpoint achieves, on a
training corpus later found to have residual leakage."

### 2. If jma_wc wins recall at every threshold and on the exact AUC-recall integral, in what sense is v7 "better"?

Not in an unqualified sense. §4e's AUC-recall (0.669 vs 0.705) is exactly the
threshold-independence check this question is asking for, and it confirms the
recall gap is real, not a cutoff artifact. Calling this a "win with an
asterisk" bakes in an untested value judgment that timing precision matters
more than detection completeness. For most downstream seismology uses
(catalog completeness, aftershock sequences, magnitude of completeness),
missed picks are typically more costly than a few hundred ms of arrival-time
error — location uncertainty is usually dominated by velocity-model error
anyway, not pick timing. The document hasn't earned the right to call this a
win; it should call it a trade and let the reader's use case decide which axis
matters to them.

### 3. Isn't "improves timing but doesn't close the recall gap" a euphemism for "made the model worse at its primary job to improve a secondary one"? Defend the trade.

As a *finding*, yes, defensible: "distillation-based fine-tuning without an
explicit timing loss produces a recall/timing seesaw, and no configuration
across 19 variants escapes it" is a real, useful methodological result. As a
*deployment recommendation* ("ship v7"), no — that's not defensible once
recall is down 3–4 points and timing is a wash once leak-free. The document
currently conflates these two claims under one "champion" label. It should
say: worth publishing as a characterization of the trade-off space in this
KD/fine-tuning recipe; not worth publishing as "we built a strictly better
picker."

## On the negative result (v20)

### 4. Does failing one fix (soft-label CE) license concluding the hard/soft mismatch isn't the mechanism?

No — this is overclaimed in the current draft. v20 tested exactly one
implementation (full soft-label CE swap-in). Untried alternatives: label
smoothing on top of hard-argmax, a weighted hard+soft CE blend, lower-
temperature soft targets, or reintroducing a tiny explicit timing loss with
different scheduling than v6's (which collapsed recall). §7 item 2's language
— "rules out a plausible mechanism" — should be softened to "this specific fix
attempt failed"; "the hard/soft mismatch isn't the mechanism" has not been
established, only "swapping to pure soft-CE isn't the fix."

## On scope and honesty

### 5. What stops a reviewer from asking "why claim 9% is meaningful if you can't rule out 10x the false positives in deployment?"

Nothing stops that question, and it's the correct one to ask. The honest
answer: the 9%/0.290s number, even setting aside #1's leakage issue, is a
claim about pick-timing quality *conditional on already being within ±5s of a
real pick* — it says nothing about whether v7 would flood a real continuous
stream with false triggers. Given recall dropped (a more conservative model),
a guess in either direction about the false-positive rate is unfounded
speculation. This caveat currently sits as one bullet in a six-item
limitations list (§6); it should be the first sentence any percentage claim is
allowed to appear next to, not a footnote reachable only by scrolling.

### 6. "Model ranking is preserved after leakage correction" — preserved across which rankings?

This has only been checked for P-MAE-based ranking across the pre-fix/post-fix
boundary (the 2026-07-08/09 finding: v7 stayed top before and after the #32
leakage fix). It has **not** been re-verified for recall or AUC-recall
rankings across that same before/after boundary — AUC-recall as a metric
didn't exist until after the leakage fixes were already applied, so there is
no "before" AUC-recall ranking to compare against. §7 item 3 doesn't scope
which metric or which axis of "preserved," so a reader could reasonably
over-read it as "leakage didn't affect any conclusion here," which hasn't
actually been shown for recall. Yes, overclaiming as written — it should read:
"P-MAE-based ranking of the top fine-tunes was stable under leakage
correction; recall/MCC rankings were not separately re-checked across that
boundary."

### 7. Stating a precise percentage next to an admission of test-set model selection across 19+ variants — internally contradictory?

Yes, materially. 19+ variants were iteratively selected by reading the same
benchmark leaderboard — that's test-set contamination of the model-selection
process (a winner's-curse setup), separate from and *in addition to* the
sampling-noise CIs already reported (0.290 [0.280, 0.300] vs 0.319
[0.308, 0.330]). Those CIs are narrow and non-overlapping, so sampling noise
isn't the problem — but they say nothing about selection bias, which is
unquantified and could easily exceed 9%.

Recommendation: keep a number (a paper that gives only "direction, no
magnitude" is nearly unfalsifiable and reviewers dislike that too), but never
let it appear without the selection-bias sentence in the same breath, and
ideally hold out one never-touched split to give a single bias-corrected
confirmatory number before submission — that check doesn't exist yet (§5's own
caveat already flags this as outstanding).

## The one that actually matters

### 8. If someone deploys v7 tomorrow on real continuous data, what's the single most likely way it disappoints them — and does the abstract warn about it?

Most likely: **it misses more real earthquakes than the `jma_wc` checkpoint it
replaced.** That's the one number here that's real, not a threshold or
leakage artifact (0.876 vs 0.909 at fixed threshold; 0.669 vs 0.705 on the
threshold-free AUC-recall integral). That should be the headline warning.

What the abstract currently cannot tell a deployer — and doesn't disclose
loudly enough — is whether v7 also changed the false-positive rate in either
direction, because the oracle-window benchmark design makes that unmeasurable
in principle, not just unmeasured in practice. As structured now, the document
leads with the flattering, measurable axis (timing) and files the
unmeasurable, operationally-critical axis (false positives on real streaming
data) as limitation-list item one of six. That ordering should flip: the
FP-blindness and the real recall regression belong in the first paragraph, and
the timing "improvement" belongs after them — clearly labeled as conditional
on both the leaky-training issue (#1) and the oracle-window issue (#5/#8).

---

*Written in response to a reviewer-style Q&A on `paper_results_final.md`
(2026-08-08 snapshot, source commit `40582f9`).*

**Incorporated 2026-08-10** into `paper_results_final.md` §§4f, 6, 7 and
`paper_draft.qmd` (§sec-bench, §sec-metrics, §sec-detection, §sec-blocking,
§sec-important): Q1/Q3 (leaky-vs-clean headline number) were already covered
by §4c pre-existing text; Q4's overclaim softened in §7 claim 2; Q6's ranking
scope narrowed to "P-MAE-based" in §7 claim 3; Q5/Q8's FP-blindness is now
substantially answered by the new detection-precision audit (§4f), built
specifically because this critique flagged it as the single most important
open gap — see that section for the caveat on why it's still not a
deployment-accurate false-positive rate.
