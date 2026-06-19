# Stage 1 Forced-Coupling Result — moves_guards_green

- **Capacity rung:** r64_qv_mlp
- **Projection (proto-refusal -> MFT, norm-ratio):** 0.1248 -> 0.5020 (Δ+0.3772; target 0.4, Tier-2 baseline ~0.107)
- **First guard breach:** step 50

## Specificity guards at final step
- Guard 1 (neutral not worse than moral): True
- Guard 2 (probe acc, if monitored): None
- Guard 3 (off-target contrast flat): True (neutral-contrast proj 0.1063)
- Guard 4 (general ppl band): True
- **all_green:** True

## Routing
GREEN-LIGHT the full pipeline (Stage 2/3): coupling moved with guards green.

_Hard stop after Stage 1: do NOT proceed to SFT->Heretic without human review._
