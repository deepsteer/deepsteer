---
name: anomaly-triage
description: >-
  Promote anomalies to experiments instead of caveats. Use this skill when writing any
  results or discussion section, at every phase end and human gate, and the MOMENT any of
  these appear: a sign flip under intervention; one model or family breaking a cross-model
  pattern; a control behaving unexpectedly; a near-miss at a pre-registered threshold; a
  dose-dependent effect where flat was expected; or two results in different documents that
  jointly imply something neither states alone. Fires on the words "interestingly",
  "unexpectedly", "surprisingly", "notably", "caveat", "future work", "exception", "the one
  model where". Maintains ANOMALIES.md with a required schema and enforces the promotion
  rule: any open anomaly with a cheap discriminating experiment gets scheduled, not
  footnoted. The thesis usually lives in the anomalies, not the confirmations.
---

# Anomaly Triage

**Core principle: in a program of nulls, the anomalies ARE the results.** The reflex under
execution pressure is to caveat the exception and keep moving; the correct move is to price
the discriminating experiment and schedule it.

## Required artifact: ANOMALIES.md (repo root or program dir)

One entry per anomaly:

```
id / date:
observation:        exact numbers, file+line of first appearance
type:               sign-flip | family-exception | control-misbehavior | near-miss |
                    dose-dependence | cross-doc conjunction
appears in:         papers/docs touched
competing readings: R_a vs R_b (state both fairly)
discriminator:      cheapest experiment separating R_a from R_b, with GPU estimate
status:             open | promoted (→ which phase/session) | resolved (→ which reading won)
thesis impact:      one line — what changes if R_a; what changes if R_b
```

## The promotion rule

**Any open anomaly whose discriminator costs ≤ ~2 GPU-hours (or is zero-GPU on existing
artifacts) is scheduled into the next session — not caveated.** Writing "interestingly, X
(left for future work)" in a document while zero-GPU discriminators sit unrun is the
anti-pattern this skill exists to kill.

## Archetype catalog (with this program's instances and the correct handling)

1. **Sign flip under intervention.** Ablating the moral subspace moved refusal **+0.14** in
   the coupled model and **−0.04** in the control. Original handling: framed as a null
   ("subspace carries comprehension, not compliance"). Correct handling: a sign-flipping,
   condition-dependent behavioral response to removing "disconnected" content is the
   strongest through-weights-read evidence in the repo — headline material, and a
   discriminator magnet (which components mediate the flip?).

2. **Family exception (n=1 breaks the panel).** Llama-3.1: refusal only partially ablatable
   (0.90 → 0.475) AND moral judgment degrades dose-dependently, direction-specifically
   (0.75 → 0.604) under refusal ablation — the one family where *function* shows. Original
   handling: caveat motivating a pivot elsewhere. Correct handling: the program's best
   toehold; localize the residual, run the decision-vs-decision test there first.

3. **Cross-document conjunction.** GPT-OSS is simultaneously the most morally deliberative
   in-trace, the most robust refuser, and the only distributed-refusal model — three facts in
   three documents. With Llama, that makes n = 2 for "hardest-to-strip refusal co-occurs
   with moral involvement," which is the program's thesis stated positively. Conjunctions
   are invisible unless something forces cross-document reads: that is this ledger's job.

4. **Near-miss at a pre-registered bar.** 0.345 vs 0.354 in one model, null-crossing in the
   second → combined-evidence line (see `estimator-traps` #5), plus a ledger entry so the
   site gets a designed follow-up (outcome-conditioning) rather than a shrug.

5. **Control misbehavior.** A "non-moral" reference projecting 0.51–0.65 onto the moral
   subspace is not a nuisance to explain away in a footnote — it is a validity event that
   re-opens every verdict the control gated (see `instrument-calibration`).

6. **Bug report that is a finding.** Instrument degeneracies that replicate cross-model are
   citable methods contributions, not embarrassments. *Instances:* massive-activation null
   saturation (Qwen/Llama) and the ~10–15-dim decision-site bottleneck (PR 14.7/8.6/10.2
   across all three families) both entered the ledger as bugs and left as the spine of a
   standalone methods note. When a failure replicates, ask "who else is silently hitting
   this?" — that question is a paper test.

## Resolution machinery

Add `resolution_type: experiment | calibration | scoping` to the schema. **Calibration
closures** resolve an anomaly by computing the correct chance model rather than running
anything: a twice-elevated cosine (0.32, flagged in two independent measures in one
session) closed as ≈ channel-chance for its measured d_eff 8.6 (sqrt(2/π·d) ≈ 0.27) — no
pod required, and the closure *upgraded* the panel wording (two families below chance =
active separation). When one entity is flagged by two independent measures, merge to a
single entry with a joint discriminator rather than two entries with two.

## Gate review pass (mandatory at every human gate)

1. Re-read every `open` entry against the new results; update statuses.
2. Conjunction hunt: for each pair of entries, ask "do these two jointly imply something
   neither states?" Any yes → one synthesis line in the gate summary + a candidate entry.
3. Surface the top-2 open anomalies (by thesis impact ÷ discriminator cost) in the gate
   summary as scheduling options — the human decides, but they must be *presented*.

## Ship-blockers

- [ ] Every "interestingly / unexpectedly / caveat" sentence in the doc has a ledger entry
- [ ] Every open entry has a priced discriminator
- [ ] Promotion rule applied (cheap discriminators scheduled, not deferred)
- [ ] Gate summaries carry the top open anomalies
- [ ] Cross-model instrument failures evaluated as candidate methods findings

Pairs with: `program-thesis` (conjunctions feed the synthesis), `compute-ordering`
(discriminators enter the session plan by information-per-GPU-hour).

*Changelog — v2 (2026-07-02): added archetype 6 and calibration-closure resolution
(cases: the methods-note pair; the channel-chance closure of the Qwen R3 anomaly).*
