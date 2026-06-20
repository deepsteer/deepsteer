# Paper 6 — Plan

## Title
**Where Refusal Lives: Cross-Model Generalization of the Comprehension/Compliance
Dissociation**

(Working alt: *Is Refusal Morally Grounded? A Cross-Family Test of Where Safety
Behavior Is Geometrically Routed.*)

## One-line thesis
Paper 5 showed, within OLMo-3, that moral comprehension and refusal/compliance
are geometrically separate: the refusal direction is almost entirely residual
against moral, persona, and Assistant-Axis structure. Paper 6 asks whether that
dissociation is a property of language models in general or an artifact of
OLMo-3's weaker post-training, by replicating the decomposition across three
dense ~7-8B base+instruct families.

## The sharpened question
Refusal is universally ablation-fragile (Heretic strips single-direction refusal
across Qwen, Llama, OLMo). That is known and is not what we test. The open
question: **is the ablatable refusal direction morally grounded, or orthogonal to
moral structure?** Two models can both have strippable refusal where one routes
through moral features and the other through pure residual. That routing question
is the paper.

## Safety constraint (explicit)
This paper produces NO technique for constructing deeper or harder-to-ablate
refusal. It is diagnostic: characterize and explain the dissociation across
models. No "make coupling survive SFT", no selective regularizer whose only
contribution is less-removable refusal. If any phase drifts toward
"un-ablatable refusal", stop and flag.

## Model panel (FIXED — dense ~7-8B, paired base+instruct, non-reasoning)
| role | base | instruct | n_layers / hidden |
|------|------|----------|-------------------|
| anchor | `allenai/Olmo-3-1025-7B` | `allenai/Olmo-3-7B-Instruct` | 32 / 4096 |
| comparison A | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | 28 / 3584 |
| comparison B | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | 32 / 4096 (gated) |

Rationale (do not substitute without re-checking): dense only (Paper 2's
MoE-dilution confound), ~7-8B only (Paper 4's scale-dependent redundancy),
non-reasoning instruct (Qwen3/Llama-4 hybrid thinking puts refusal inside a
`<think>` trace, a different object OLMo-3 lacks). A reasoning-model extension is
a separate future paper.

## Convention parity (the validity crux)
Identical extraction across all three models, encoded in
`scripts/model_registry.py`:
- **Layers:** fractional band from the OLMo anchor [15,31]/32 = depth fractions
  (0.469, 0.969) → `round(f·n_layers)`: OLMo [15,31], Qwen [13,27], Llama [15,31].
  Headline single layer at depth-fraction 0.5: OLMo L16, Qwen L14, Llama L16.
- **Format:** moral (MFT) + persona on the BASE in `raw` (the collection path
  pools raw text and never applies a chat template, so Qwen2.5-base's bundled
  ChatML template does not leak into the geometry); refusal on the INSTRUCT in
  `chat` (Arditi/Heretic last-input-token diff-of-means). Instruct templates
  inject different system prompts, but refusal = mean-diff(harmful,harmless) with
  the same template both sides → common-mode, cancels.
- **Same tooling:** Paper 3 `exp1_2_3_framework_geometry.py` (moral directions),
  Paper 5 `persona_probe_base.py`, `heretic_ablation.py`, `direction_utils`,
  `moral_dependency.build_subspace_basis`, `measure_refusal_decomposition`.
- **Same contrast sets:** real Arditi/Heretic `refusal_prompts.json` (400/400, not
  `_FALLBACK_*`); MFT `moral_probing_v2.json`; persona pairs from
  `deepsteer.datasets.persona_pairs`.

## Phases and gates
- **Phase 0a — availability (DONE, gated).** All 6 repos exist; bases genuine;
  Llama gated (RunPod token has access); 32/28/32 layers.
- **Phase 0b — generalize conventions + scaffold (DONE, local gate passed).**
  `model_registry.py`, `local_test.py`, QWEN added to `ModelFamily`.
- **Phase 0c — Qwen smoke on RunPod.** One layer, tiny prompt set; confirm hooks
  / layer-module access work on a non-OLMo arch before the full pass.
- **Phase 1 — cross-model decomposition → HUMAN GATE.** Per instruct model:
  extract refusal + per-model MFT subspace + persona; decompose refusal →
  {moral, persona, residual} at depth-0.5 + band; refusal eff-dim / consolidation
  (the reduced sub-check: is OLMo's refusal merely more diffuse, or genuinely
  non-moral?). Report all three before Phase 2.
- **Phase 2 — Heretic ablation + comprehension dissociation per model (after
  gate).** Ablate refusal; re-measure moral comprehension (probe acc / judgment /
  dependency); test whether ablation incurs collateral comprehension damage.

## Discriminating predictions (null is as publishable as positive)
- **Refusal ~99% residual in Qwen AND Llama too** → thin-refusal refuted; the
  dissociation is structural and general. Strengthens Paper 5; Paper 6 is a clean
  generalization.
- **Materially MORE moral/persona structure in Qwen/Llama than OLMo** → OLMo's
  sui-generis residual was partly its weaker post-training. Finding shrinks to
  "OLMo-3 refusal is weakly moral"; possible Paper 5 caveat demotion (human gate).
- **Phase 2 collateral damage in some model** → first natural evidence that
  refusal can route through comprehension-load-bearing features (observed, not
  forced).

## Deliverables
Per-model decomposition energy fractions, refusal eff-dim, Heretic
comprehension-delta. One combined cross-model table + figure (depth-fraction
x-axis so layers align across models).

## Claim discipline
Until Phase 2 the result is "the refusal/moral decomposition replicates (or
doesn't) across families." Make no claim about *why* a model's post-training
differs; we measure routing, not training recipes (we don't have Qwen's/Llama's
training details and don't need them).

## Status
Phase 0a + 0b complete and locally gated. Next: Phase 1 per-model driver +
RunPod orchestration, then the Qwen 0c smoke, then the full three-model pass.
