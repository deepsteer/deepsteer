---
name: instrument-calibration
description: >-
  Calibrate the measurement before trusting the null. Use this skill BEFORE reporting,
  accepting, gating on, or writing up ANY null, negative, below-chance, below-threshold, or
  orthogonality result — and when designing any probe, projection, metric, or verdict rule
  that could return one. Fires on: "null result", "no effect", "orthogonal", "below chance",
  "does not project", "fails to detect", "NULL verdict", verdict sections, gate summaries,
  pass/fail rules, and any claim that a relationship is absent. Enforces positive controls,
  calibrated ladders (floor → matched null → measurement → positive band), re-verification of
  every control's defining property in the current context, and minimum-detectable-effect
  statements so nulls have teeth. Do not write a NULL verdict without consulting this skill.
---

# Instrument Calibration

**Core principle: a null result is a claim about your instrument until a positive control
says otherwise.** "X does not project onto Y" is only evidence about X if something that
*should* project onto Y demonstrably does, on the same instrument, in the same units, in the
same model.

## Required artifact: the calibration ladder

Every headline scalar ships inside a ladder table. Verdict sentences state position-on-ladder,
never bare pass/fail.

| rung | what | how obtained |
|---|---|---|
| floor | isotropic expectation (e.g. sqrt(k/d) for rank-k in d dims) | closed form |
| matched null | q50 / q95 of covariance-matched random directions at realized rank | null generator |
| measurement | the quantity ± bootstrap CI | data |
| positive band | [min, max] of known-coupled references | see recipes |
| references | named axes shown for context (labeled by construction) | data |

Wording template: "X = 0.52 — above the matched null (q95 0.39), below the moral-family band
[0.65, 0.76]." Never: "X is orthogonal to Y."

## Positive-control recipes

1. **Held-one-out sources.** If the target object is built from N sources, project each source
   onto the span of the others. The [min, max] is the band: what "membership in this family"
   projects. Zero new compute — the directions already exist.
2. **Known-mediated decision.** For decision-level instruments, use a decision the model
   demonstrably makes from the measured content (e.g. its own judgment outputs for a
   comprehension subspace). If even that is null, the instrument has a category problem, not
   the hypothesis (see `construct-audit`).
3. **Synthetic injection** (last resort). Add a known component to held-out activations;
   verify recovery at the expected magnitude.

If no positive control is possible, say so explicitly in the write-up and downgrade the claim
from "absent" to "not detected by this instrument."

## Control-validity audit (run for every reference/control axis)

1. **Provenance:** where was this control's defining property validated — which model, which
   span, which format, which paper? Write it down next to the number.
2. **Re-verify here:** re-measure the defining property in the current context before using
   the control as a bar. A control validated as "orthogonal to MFT on model A" is NOT
   validated as "non-moral on subspace V in model B."
3. **Consistency grep:** search the current document for the control's prior
   characterization. If prose says one thing and a table in the same file says another, stop
   and resolve before any verdict.
4. **Name by construction, not intended role** (enforced by `program-thesis`): a control
   built from morally-questionable-voice pairs is a "moral-adjacent voice reference," not a
   "non-moral semantic control," no matter what job you hired it for.

## Nulls with teeth

Attach a sensitivity statement: the minimum effect size detectable at the achieved n at 95%
power (bootstrap sketch is fine). "We find no coupling (MDE ≈ 0.12 in projection fraction)"
is publishable and cumulative; "we find no coupling" is neither.

## Case study (this program)

The D1 refusal-orthogonality verdict originally rested on (a) a covariance-matched null and
(b) a "non-moral" persona control. Audit findings: the persona axis was constructed from
morally-questionable-voice pairs; its "orthogonal to moral directions (|cos| 0.076–0.085)"
justification was measured on a *different span* (MFT) — on V_moral it projected 0.51–0.65,
failing hardest (0.65) in the exact model where the verdict depended on it. There was no
positive control at all: nothing established what a genuinely-coupled direction would
project. Adding held-one-out bands (base [0.54, 0.66] … GPT-OSS [0.65, 0.76]) reframed the
most contested number — GPT-OSS in-trace P2 = 0.52 became "above null, below the
moral-family band" — and made the orthogonality result *stronger*, not weaker, because every
refusal point now sits below a calibrated membership band instead of below a mislabeled
reference. Calibration is not an attack on your result; it is what makes the result citable.

## Position validity: the band-below-null tell

A measurement position is itself an instrument. **If the positive-control band sits below
the covariance null at a position, the position is invalid** (a low-rank / collinear token),
and no verdict there is interpretable — including apparently-positive ones. Report the
position's participation ratio (PR) beside every ladder, and require ladder rungs to be
format- and position-co-located with the measurement: a band computed at raw mean-pooled
positions cannot judge a chat decision-site number. PR below ~30 → flag at extraction and
block verdicts pending a healthy-position re-read.

*Case:* the pre-registered chat decision site (assistant-header token) returned band
[0.40, 0.47] **below** null q95 0.557 at PR 14.7 — replicating at PR 8.6 / 10.2 on the other
two families. The refusal number there (0.497) was about to be read against the band as a
"register-scoped" retraction of format-robustness. The tell fired; the position was ruled
invalid; the healthy content position (null 0.28, band 0.54–0.64 ≈ raw) showed V_moral
format-robust; a false scoping claim died before commit. Consistency check that sealed it:
sqrt(3/PR) ≈ 0.45 predicted the rank-3 null median at that PR.

## Calibrating intervention instruments

Patches, ablations, and steering are instruments too. **A null from a restricted
intervention (subspace-limited patch, rank-k ablation) is uninterpretable until the same
restricted instrument demonstrably moves a known-dependent outcome** — the transport
positive control. "Restricted patch didn't move Y" always has the alternative reading that
the restriction under-transfers *any* signal (rank too low, encoding nonlinear), not that Y
reads other features. Spec block and full rules in `intervention-validity`.

## Ship-blockers (all must pass before a NULL verdict commits)

- [ ] Ladder table present; verdict sentence carries the ladder
- [ ] At least one positive control measured on this instrument, this model
- [ ] Every control's defining property re-verified in this context; provenance noted
- [ ] Consistency grep clean (no prose/table contradictions about any control)
- [ ] MDE / sensitivity statement attached
- [ ] Controls named by construction
- [ ] Every position carries its PR; band-below-null check passed; rungs co-located
      (format + position) with the measurement
- [ ] Restricted interventions have a transport positive control

Pairs with: `construct-audit` (is the comparison even the right cells?), `estimator-traps`
(are the CIs on the ladder trustworthy?), `intervention-validity` (causal instruments),
`program-thesis` (claim wording).

*Changelog — v2 (2026-07-02): added the band-below-null position-validity tell (case: the
cross-model decision-site bottleneck, PR 14.7/8.6/10.2) and intervention-instrument
calibration (case: the V_moral-restricted patch transport control).*
