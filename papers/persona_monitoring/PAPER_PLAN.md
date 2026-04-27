# Paper 2 Plan: *Persona-Feature Monitoring at 1B: A Compound Scaling Boundary*

**Status:** Stub. Experimental work for Phase D is complete (C10 v2,
Step 2 steering, C15 reframed); paper drafting has not started.
The four headline 1B results are documented in `RESEARCH_BRIEF.md`
Paper 2 section and the per-experiment `RESULTS.md` files under
`outputs/phase_d/` (relative to this paper directory). This plan
will populate during drafting.

**Source-of-truth note.** Where this document and `RESEARCH_BRIEF.md`
differ on Paper 2 framing, this document will take precedence once
drafting begins. Until then, `RESEARCH_BRIEF.md` Paper 2 section is
authoritative for the four-finding summary and the Phase E coupling
+ suppression-captures-behavior predictions.

**Path convention.** Same as Paper 1: paths in this document and in
forthcoming `sections/*.md` are *relative to the paper directory*
(`papers/persona_monitoring/`). So `outputs/phase_d/c10_v2/...`
resolves to `papers/persona_monitoring/outputs/phase_d/c10_v2/...`
from the project root, and `scripts/c10_em_replication.py` resolves
to `papers/persona_monitoring/scripts/c10_em_replication.py`.
Project-root-relative paths are used in CLI invocations and in
root-level docs (`README.md`, `RESEARCH_PLAN.md`, `RESEARCH_BRIEF.md`).

## Tentative title

**Primary:** *Persona-Feature Monitoring at 1B: A Compound Scaling
Boundary on the Wang et al. (2025) Mechanism*

## Thesis (working)

The Wang et al. (2025) toxic-persona mechanism that mediates emergent
misalignment at 32B does **not** engage at 1B under a careful
reproduction of Betley et al.'s (2025) insecure-code LoRA recipe.
DeepSteer's `PersonaFeatureProbe` (linear analog of the toxic-persona
direction) and `TrainingTimeSteering.gradient_penalty` primitive both
work as designed at the 1B scale; what fails to engage is the
*coupling* between the persona-probe direction and behavioral
emergent misalignment. This is a compound scaling boundary: the
mechanism, the measurement, and the intervention all run at 1B; the
behavioral phenomenon they target does not. The boundary is
publishable as a clean null and motivates a Phase E test at 7B with
SAE-decomposed features.

## Four headline 1B findings (per `RESEARCH_BRIEF.md`)

1. **Persona mechanism does not engage at 1B under controlled
   insecure-code LoRA replication (C10 v2).** Probe activation
   Cohen's d = +0.03 vs. ≥1 SD threshold; behavioral EM 1.56 %
   insecure vs. 0.69 % secure (Wilson 95 % CIs overlap). Probe and
   judge fire on decoupled axes at 1B.
   Numbers source: `outputs/phase_d/c10_v2/RESULTS.md`.

2. **Linear `gradient_penalty` primitive cleanly suppresses a target
   probe direction (Step 2A).** 99.3 % suppression at no SFT-loss cost
   on a synthesized persona-voice corpus.
   Numbers source: `outputs/phase_d/step2_steering/RESULTS.md`.

3. **Probe-direction suppression does not capture behavior at 1B
   (Step 2B).** Vanilla and gradient_penalty produce judge scores
   matching within 0.01 / 10 despite probe Cohen's d differing by
   3.07. Activation_patch backfires by amplification (+99 % probe
   activation) due to training-time compensation.
   Numbers source: `outputs/phase_d/step2_steering/RESULTS.md`.

4. **Insecure-code LoRA leaves a fragility-locus signature (C15
   reframed).** Probing accuracy unchanged across base / insecure /
   secure (max |Δ| = 0.021); fragility-locus shifts 2-3 layers under
   insecure-code specifically (robustness peak relocates from layer 7
   to 9-10; layers 6-7 collapse from critical noise = 10 to 1).
   Numbers source: `outputs/phase_d/c15_reframed/RESULTS.md`.

## Outstanding pre-drafting items

- **C10 hardening (optional).** Run Betley et al.'s published
  hyperparameters (rank 32, all linear modules, full LR / step
  budget) once to harden the null against rebuttal. ~1 day.
- **Phase E predictions need experimental design.** The two
  falsifiable predictions in `RESEARCH_BRIEF.md` (coupling at 7B;
  suppression-captures-behavior at 7B with SAE-decomposed features)
  need a concrete experimental plan before they can be Phase E
  experiments rather than just predictions.
- **Rename scripts to user-facing labels.** The 16 scripts in
  `scripts/` retain the `c8_` / `c9_` / `c10_` / `c15_` / `step2_*`
  research-internal prefixes from `RESEARCH_PLAN.md`'s experiment-ID
  system (e.g. `c8_persona_validation.py`, `c10_em_replication.py`,
  `c15_reframed.py`, `step2_finding3_mechanism_check.py`). These are
  one-to-one with `RESEARCH_PLAN.md` entries today, but a reader
  outside the project has no way to map them. Renaming candidates,
  to apply in a single sweep early in drafting:

  | Current | Proposed | Notes |
  |---------|----------|-------|
  | `c8_persona_validation.py` | `persona_probe_validation.py` | OLMo-2 1B final-checkpoint validation gate |
  | `c9_persona_trajectory.py` | `persona_probe_trajectory.py` | 37-checkpoint persona-probe trajectory |
  | `c10_em_replication.py` | `insecure_code_lora_replication.py` | Betley-style LoRA + EM eval |
  | `c10_score_responses.py` | `judge_score_responses.py` | Behavioral judge scoring |
  | `c10_analyze.py` | `analyze_em_responses.py` | C10 v2 analysis |
  | `c10_visualize.py` | `visualize_em_responses.py` | C10 v2 plots |
  | `c10_inference_patch.py` | `inference_time_activation_patch.py` | Step 1 inference-time activation patching |
  | `c10_inference_patch_plot.py` | `inference_time_activation_patch_plot.py` | Step 1 plot |
  | `c15_reframed.py` | `differential_fragility_em.py` | Phase B/C moral probe + fragility on EM adapters |
  | `step2_persona_steering.py` | `training_time_steering_runner.py` | Step 2 main runner |
  | `step2_analyze.py` | `analyze_steering.py` | Step 2 analysis |
  | `step2_plot.py` | `plot_steering.py` | Step 2 plots |
  | `step2_finding2_head_start.py` | `vanilla_trajectory_comparison.py` | Vanilla LoRA snapshot trajectory |
  | `step2_finding3_mechanism_check.py` | `activation_patch_mechanism_check.py` | h_ap − h_van layer-5 mechanism check |
  | `step2_finding4_behavioral_judge.py` | `behavioral_judge_rerating.py` | Held-out judge re-rating of all 640 generations |
  | `generate_persona_corpus.py` | (unchanged) | Already user-facing |

  Apply the rename via `git mv` early in the drafting pass, before
  any prose references the script names. Update import-style
  references in any RESULTS.md / docstring cross-links at the same
  time. Cross-reference table from `c<N>_<old>` to new name should
  live in this PAPER_PLAN.md until §3 Methodology is drafted, then
  move to Appendix E (Reproducibility).

## Next-step ordering for drafting

Drafting will follow Paper 1's recommended order once started:
§3 Methodology → §4 Results → §2 Related Work → §5 Discussion →
§1 Introduction → §6 Conclusion → Abstract → Appendices. This stub
will expand into the full plan structure (title, abstract,
section-by-section outline, headline figures, numbers table,
framing decisions, open items, cite list, drafting order) at that
point.
