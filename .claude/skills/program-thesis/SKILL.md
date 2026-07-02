---
name: program-thesis
description: >-
  Keep the research program's positive thesis current and its claim language anchored. Use
  this skill at every human gate; before committing ANY results, README, abstract, framing,
  or discussion text; when choosing a paper's headline finding; when naming controls,
  metrics, or datasets in prose; and when pre-registering verdict rules. Fires on: "framing",
  "write-up", "headline", "discussion section", "abstract", "verdict wording", "how should we
  describe". Maintains SYNTHESIS.md (the program's one-paragraph positive thesis plus
  standing claims, each paired with its strongest counter-reading and the experiment that
  separates them). Enforces anchored adjectives, an intensifier ban without matching
  controls, controls named by construction, both-branches-publishable preregistration, and a
  referee pass before any verdict ships.
---

# Program Thesis

**Core principle: a program that only knows what it has ruled out cannot be steered.** Nulls
accumulate; someone must keep restating what the program now *claims*, and every claim must
carry the reading that would defeat it.

## Required artifact: SYNTHESIS.md (program root, updated at every gate)

```
## Thesis (one paragraph, positive voice, dated)
## Standing claims
  claim | strongest counter-reading | separating experiment | status
## Conjunctions under watch (from ANOMALIES.md)
## What the next result changes (per pending experiment: branch → thesis edit)
```

Worked example of the positive voice — the same facts as "seven papers of nulls":
*Models build moral comprehension during pretraining and it survives alignment intact; the
compliance mechanism is a fresh post-training construction in a low-variance channel
(proto-refusal→gate cos 0.155 vs the moral subspace's 0.999 crystallization) that does not
share coordinates with comprehension — yet interventions on comprehension move compliance,
so the read is through weights, not geometry; and the two models where refusal is hardest to
strip are precisely the morally-involved ones.* That paragraph is fundable; the null list is
not. Same data.

## Claim-language rules

1. **Anchor every quantitative adjective.** "Cleanly separates" is banned next to acc 0.67;
   write "separates above chance (0.67 vs 0.50; full-space probes reach ~1.0)." If the
   number embarrasses the adjective, change the adjective.
2. **Intensifiers require matching controls.** "SFT rotates the moral subspace
   *specifically*" requires a rotation measurement on non-moral control directions; without
   it, write "rotates," and add the specificity control to the plan. Same for "uniquely,"
   "only," "selectively."
3. **Name by construction, not by role.** A reference built from morally-questionable-voice
   pairs is a "moral-adjacent voice reference." The role you hired it for goes in the
   methods; the name states what it is.
4. **Verdicts ride the ladder.** Every verdict sentence carries floor/null/measurement/band
   position (see `instrument-calibration`). "NULL" alone is not a sentence.
5. **Consistency grep before commit.** Search the document set for every control, metric,
   and claim being edited; prose and tables must agree across files, not just within one.
6. **Headline pairing.** The strongest headlines are contrasts the program owns end-to-end
   (0.999 vs 0.155; band vs gate). Before settling a headline, ask: which two of our numbers,
   side by side, state the finding with no adjectives at all?
7. **Post-hoc forks get dated amendments + fork-robustness.** Any analysis choice changed
   after seeing results (primary-position preference, classifier swap, aggregation rule) is
   a fork: it requires a dated amendment stating the construction-based reason AND the
   verdict reported under both choices — and it never lands in the same commit as the
   results it affects. *Case:* the primary-position preference flip (first-valid →
   raw-pooling analog) was correct on construction grounds and still needed its amendment
   plus both-position bands.
8. **Results commits carry their ship-blockers.** A RESULTS.md commit without the referee
   pass and the SYNTHESIS.md update in the same commit fails review mechanically — the
   required artifacts exist precisely so their absence is visible at the commit boundary.
9. **Immutability intensifiers get an intervenability check.** "Architectural",
   "guaranteed", "hard-wired" foreclose the program's own intervention surface. Before
   using one, state what is *learned/trainable* about the mechanism; prefer "structurally
   favored" plus the trainable lever. *Case:* the decision-site bottleneck is real, but
   what occupies it is learned attention routing — "architecturally guaranteed
   orthogonality" became "structurally favored; transport is trainable," which kept the
   intervention program alive inside the same finding.
10. **Verdicts carry their detection bars.** Null claim language embeds the MDE: "no
   decision-level coupling detectable at |cos| ≳ 0.5" — never bare "dissociation."
11. **Citations verify against primary sources.** Author lists and claims are checked at
   the source before entering committed prose; anything unverified carries an explicit
   verify-before-citing flag and stays out of bibliographies. *Cases:* fabricated author
   lists found in the early-papers audit (the failure); the pre-build lit pass that found
   the closest prior art and re-centered the novelty claim before any code existed (the
   pattern done right).

## Packaging principle

Package papers by claim, not by chronology of discovery. When a new instrument supersedes
an old paper's machinery, the new work absorbs the old — never retrofit a superseded
instrument's paper with its own failure analysis. A fast methods note that offloads
validity machinery keeps the flagship lean and is often the highest-citation-surface item
in the queue.

## Both-branches preregistration

For every pre-registered rule, write BOTH result framings *before* data, and confirm both
are publishable. If one branch reads "uninformative," the experiment is misdesigned — fix
the design, not the framing, before spending compute. (Pattern: "sensitivity-confirmed →
the seven nulls are strong evidence" / "category-mismatch → the nulls are reframed, and the
replacement instrument is X." Both branches move the program.)

## The referee pass (before any verdict or framing commits)

Write the three most damaging reviewer objections — the ones a hostile expert raises in the
first read — and answer or concede each in-line in the document. Standing generators for
this program's genre: "your control isn't what you call it," "your null has no positive
control," "your comparison never measured the diagonal cells," "your CI method can't test
that difference," "your gradient's construct changes along its x-axis." If an objection has
no answer, it becomes an ANOMALIES.md entry or a scope-limiting sentence — never silence.

## Ship-blockers

- [ ] SYNTHESIS.md updated (thesis paragraph re-dated, claims table current)
- [ ] No unanchored adjectives; no uncontrolled intensifiers; immutability words passed
      the intervenability check
- [ ] Controls/metrics named by construction everywhere
- [ ] Both branches of pending rules written and publishable
- [ ] Referee pass appended (3 objections, answered or conceded) — in the same commit as
      any RESULTS document
- [ ] Post-hoc forks carry dated amendments + both-choice verdicts
- [ ] Null verdict sentences embed their detection bars
- [ ] No unverified citation in committed prose

Pairs with: `anomaly-triage` (conjunctions in), `instrument-calibration` (ladder wording).

*Changelog — v2 (2026-07-02): added rules 7–11 (post-hoc forks, commit-boundary blockers,
intervenability check, detection-bar wording, citation verification) and the packaging
principle. Cases: the mean_content preference fork; the "architecturally guaranteed"
reframe; the C1 lit pass.*
