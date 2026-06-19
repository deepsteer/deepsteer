# Stage 1 Forced-Coupling Result — no_move

- **Capacity rung:** r16_qv
- **Projection (proto-refusal -> MFT, norm-ratio):** 0.1248 -> 0.1147 (Δ-0.0102; target 0.4, Tier-2 baseline ~0.107)
- **First guard breach:** step None

## Specificity guards at final step
- Guard 1 (neutral not worse than moral): True
- Guard 2 (probe acc, if monitored): None
- Guard 3 (off-target contrast flat): True (neutral-contrast proj 0.0942)
- Guard 4 (general ppl band): True
- **all_green:** True

## Routing
STALL: forced coupling did not move the projection at this rung. If the top capacity rung also fails, the reserved subspace-robustness check (is V the wrong target?) becomes the question.

_Hard stop after Stage 1: do NOT proceed to SFT->Heretic without human review._
