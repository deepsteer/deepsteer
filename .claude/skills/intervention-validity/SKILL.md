---
name: intervention-validity
description: >-
  Design validity for causal interventions on model internals. Use this skill BEFORE
  building or running ANY activation patching, interchange/swap, ablation, steering,
  noise-injection, or head-attribution experiment — and when interpreting their results.
  Fires on: "patch", "patching", "interchange", "swap activations", "ablate", "ablation",
  "steering", "inject", "causal cell", "mediation", "head attribution", "OV contribution",
  "attribution harness". Enforces the intervention spec block: stimulus–outcome baseline
  matching, transport positive controls for restricted interventions, channel-matched
  specificity nulls for attribution, distribution-preserving ablation semantics,
  pre-registered token-alignment rules, harness parity for outcome classification, and
  both-branches framing written before data. A causal cell without a spec block is not
  ready to run.
---

# Intervention Validity

**Core principle: an intervention is an instrument — calibrate it like one, and aim it at
an outcome it can actually move.** Geometry nulls need positive controls; causal nulls need
them more, because a flat causal cell has more ways to be flat (wrong stimulus, wrong
outcome, under-transferring restriction, off-distribution ablation, harness drift) than
ways to be informative.

## Required artifact: the intervention spec block (per causal cell, committed in the prereg)

```
stimulus_class:          dataset + pair structure (type blocks per construct-audit)
outcome_variable:        the behavioral/readout variable this cell is built to move
baseline_discrimination: evidence the stimulus moves the outcome UNINTERVENED
                         (screen: keep only pairs whose baseline behavior differs;
                         report the pass rate)
site:                    layer(s) × position class × subspace — typed, PR recorded
transfer_scope:          full-residual | subspace-restricted (rank stated)
transport_control:       for restricted scopes — the known-dependent outcome the
                         restricted instrument must move first
ablation_semantics:      mean-ablate | resample-patch (zeroing = off-distribution;
                         deviations justified)
alignment_rule:          cross-pair token mapping for length-mismatched pairs, fixed
                         before running
controls:                matched-random + named-reference interventions
outcome_harness:         shared classifier/eval, id + pinned version
branches:                both result framings, written before data, both publishable
```

## The rules

1. **Stimulus–outcome matching, and the screen must bracket the operating point.** A causal
   cell is informative only if the stimulus class engages the outcome at baseline. Run the
   behavioral-discrimination screen first; a stimulus that never moves the outcome yields
   structural flats that read as nulls. But engagement is two-sided: **too-safe stimuli floor
   the outcome exactly as too-alarming ones ceiling it**, and a "keep pairs whose baseline
   differs" screen can still land entirely in a floor. Design a **severity ladder** (graded
   stimuli whose scenario escalates while each pair stays surface-matched) and let the screen
   select the *operating band* — the levels where the harmful member fires and the benign one
   does not — then report the dose–response curve as a deliverable. *Case:* third-person
   narrative moral-status twins — nothing *refuses* a story; refusal-patching required
   surface-matched **request** twins as a separate asset. *Case:* those request twins, in an
   XSTest-safe register, were so benign that only 2/8 violating members refused and 0 benign
   members did — a floor that made the generate-under-patch flip test and the anti-refusal
   discriminator inconclusive; the fix was a severity ladder with an operating-band screen.

2. **Transport positive control (restricted interventions).** "Subspace-restricted patch
   didn't move Y" has a standing alternative: the restriction under-transfers *any* signal
   (rank too low, encoding nonlinear) — instrument insufficiency, not feature location.
   Before that null is interpretable, show the same restricted instrument moves a
   known-dependent outcome. Write the branch map pre-data: restricted-moves-target
   (transport confirmed, with anatomy) / full-moves-but-restricted-doesn't (target reads
   features outside the subspace) / restricted-moves-nothing-known (only the positive
   branch of the cell carries weight).

3. **Channel-matched specificity for attribution.** In a low-PR site (~10–15 effective
   dims), every head that writes strongly into the token projects onto most channel
   directions — "top target-writing heads" collapses into "top token-writing heads."
   Attribution score = projection-onto-target − mean projection onto channel-basis controls
   (same channel, matched norm). Reconstruction checks (≥ 0.90) validate the decomposition,
   not the specificity; both are required and they are different numbers.

4. **Ablation semantics.** Mean-ablate or resample-patch by default; zeroed activations are
   off-distribution and conflate removal with corruption. Over-ablation regimes are
   reported, never headlined.

5. **Alignment is pre-registered.** For length-mismatched pairs, fix the token-position
   mapping rule (content-span alignment, anchor tokens) before running — post-hoc alignment
   is a fork generator (see `program-thesis` rule 7).

6. **Harness parity.** Outcome classification uses the program's shared classifier at a
   pinned version. A causal effect that appears under one refusal-detector and not another
   is harness drift until reconciled. *Case:* 0.167 vs 0.575 base refusal on the same model
   — an opening-marker floor vs the shared classifier; the cross-ablation cell was
   unfalsifiable (~4 events) until the harness was reconciled and the full eval set used.

7. **Dose where possible.** Prefer graded interventions (rank-k sweeps, amplitude sweeps)
   to single points; monotone dose–response is the cheapest specificity evidence there is.
   *Case:* dose-dependent, direction-specific judgment degradation under refusal ablation
   is this program's benchmark for what real behavioral coupling looks like.

## Ship-blockers

- [ ] Spec block complete per cell, committed in the prereg before the pod
- [ ] Baseline discrimination screen run; pass rate reported
- [ ] Restricted scopes have transport positive controls; branch map written pre-data
- [ ] Attribution reports channel-matched specificity AND reconstruction, separately
- [ ] Ablation semantics distribution-preserving, or the deviation justified
- [ ] Alignment rule pre-registered; outcome harness pinned
- [ ] Matched-random + named-reference controls per intervention

Pairs with: `construct-audit` (typing of stimuli and sites; outcome_variable field),
`instrument-calibration` (the calibration philosophy this extends to causal instruments),
`estimator-traps` (channel-chance levels for attribution scores), `program-thesis`
(fork and branch discipline).

*Changelog — v1 (2026-07-02): created from the C1 design review; founding cases are the
twin-patch stimulus–outcome mismatch and the degenerate negative branch of the
subspace-restricted patch. v1.1 (2026-07-02): rule 1 gains the operating-point/severity-ladder
lesson from the C1 hardening run (XSTest-safe register floored the behavioral cells; too-safe
floors as hard as too-alarming ceilings).*
