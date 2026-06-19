# Stage 1 Forced-Coupling Result — moves_only_degenerately

- **Capacity rung:** r16_qv
- **Projection (proto-refusal -> MFT, norm-ratio):** 0.1248 -> 0.2052 (Δ+0.0804; target 0.4, Tier-2 baseline ~0.107)
- **First guard breach:** step 200

## Specificity guards at final step
- Guard 1 (neutral not worse than moral): False
- Guard 2 (probe acc, if monitored): None
- Guard 3 (off-target contrast flat): True (neutral-contrast proj 0.0959)
- Guard 4 (general ppl band): True
- **all_green:** False

## Routing
DEEPER NEGATIVE: the Section 6 degenerate solution recurs at pre-training (moved only by tripping a guard). Surface it.

_Hard stop after Stage 1: do NOT proceed to SFT->Heretic without human review._
