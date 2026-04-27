# 4. Results

*Drafting placeholder.* The four headline 1B findings below are
seeded from `PAPER_PLAN.md` and the per-experiment `RESULTS.md`
files under `outputs/phase_d/`. Prose will tighten during drafting.

## 4.1 Persona mechanism does not engage at 1B (C10 v2)

Under a controlled reproduction of Betley et al. (2025)'s
insecure-code LoRA recipe, the `PersonaFeatureProbe` activation
shifts by Cohen's *d* = +0.03 between insecure and secure conditions
— well below the ≥1 SD threshold that would indicate persona-
feature engagement. Behavioral emergent misalignment scores are
likewise small (1.56 % insecure vs.\ 0.69 % secure; Wilson 95 % CIs
overlap). **At 1B, the probe and the behavioral judge fire on
decoupled axes.** Numbers source:
`outputs/phase_d/c10_v2/RESULTS.md`.

## 4.2 Linear gradient-penalty primitive cleanly suppresses a target probe direction (Step 2A)

The `TrainingTimeSteering.gradient_penalty` primitive achieves
99.3 % suppression of the persona-probe activation at no measurable
SFT-loss cost on a synthesized persona-voice corpus. The primitive
works as designed; this is the *measurement* working at 1B.
Numbers source: `outputs/phase_d/step2_steering/RESULTS.md`.

## 4.3 Probe-direction suppression does not capture behavior at 1B (Step 2B)

Vanilla and gradient-penalty conditions produce judge scores
matching within 0.01 / 10 despite probe Cohen's *d* differing by
3.07. Activation-patch backfires: training-time compensation
*amplifies* probe activation by +99 %. **This is the central
boundary finding: a clean intervention on the probe direction
leaves behavioral scores unchanged.** Numbers source:
`outputs/phase_d/step2_steering/RESULTS.md`.

## 4.4 Insecure-code LoRA leaves a fragility-locus signature (C15 reframed)

Probing accuracy is unchanged across base / insecure / secure
conditions (max |Δ| = 0.021); fragility-locus shifts 2-3 layers
under insecure-code specifically (robustness peak relocates from
layer 7 to layers 9-10; layers 6-7 collapse from critical noise = 10
to 1). The methodological complement from companion work
(Reblitz-Richardson, 2026, *When Probing Accuracy Saturates,
Fragility Resolves*) detects a structural change that the accuracy
metric cannot. Numbers source:
`outputs/phase_d/c15_reframed/RESULTS.md`.
