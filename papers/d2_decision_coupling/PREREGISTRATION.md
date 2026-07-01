# Direction 2 — Decision-Coupling Pre-registration

**Date:** 2026-07-01 · **Against commit:** `5c542f6` (HEAD).
**Status:** Pre-registered before any decision-direction extraction or coupling measurement.
Nothing here is a result. Every threshold below is fixed now, in advance of the data.

Companion: `../d1_moral_subspace/CALIBRATION_PREREG.md` (Phase A calibration, shares the R-table)
and `../d1_moral_subspace/PREREGISTRATION.md` (the frozen D1 spine, untouched by this document).

## Purpose

Every headline null in Papers 5–7 and D1 compares a **content subspace** (`V_moral`, built from
stimulus contrasts) against a **decision direction** (refusal, a behavior contrast at
output-adjacent positions). This program fills the missing **decision-vs-decision** cell: does the
model's own moral-judgment *decision* direction share geometry and causal structure with the
refusal direction? Interventional evidence already in the repo motivates the question:
forced-coupling sign flip (0.79→0.93 coupled vs 0.83→0.79 control) and Llama's judgment drop
(0.75→0.604 under refusal ablation) point to a through-weights read from comprehension into the
refusal decision that same-layer cosines cannot see. This plan measures it directly.

## Ground rules (carry existing discipline)

1. **Pre-register first.** This file is committed before any headline quantity. The two-step null
   protocol and margin `M = 0.05` are reused from the D1 spine. NULL outcomes are pre-declared
   publishable.
2. **Human gates** after Phase A (Gate A, in the calibration doc), after B1, after B2, and before
   any Phase C item.
3. **Safety scope.** Every Phase A/B item is **diagnostic on released models**; none constructs
   refusal robustness. C2 (a training intervention) is **held** (2026-07-01 decision) and requires
   a separate Direction-2 pre-registration + explicit human go before any training use.
4. **Artifact hygiene.** Every measurement saves per-prompt / per-pair arrays sufficient for
   bootstrap CIs. Missing prior artifacts are appended to
   `../d1_moral_subspace/outputs/MISSING_ARTIFACTS.md` and queued into B3, never silently
   regenerated with possibly-drifted conventions.
5. **Conventions.** Reuse `../6_cross_model/scripts/model_registry.py`, `deepsteer.directions.extraction`,
   `_ortho`/`_frac` from `../d1_moral_subspace/scripts/phase2_g3_respec.py`,
   `../6_cross_model/scripts/random_ablation_control.py`, and the covariance-matched null generator.

## Panel and headline layers (per Paper 6 registry, verified 2026-07-01)

| key | model | n_layers | headline layer (depth-0.5) | band |
|---|---|---:|---:|---|
| olmo3 | `allenai/Olmo-3-7B-Instruct` (anchor) | 32 | **16** | [15,31] |
| qwen25 | `Qwen/Qwen2.5-7B-Instruct` | 28 | **14** | [13,27] |
| llama31 | `meta-llama/Llama-3.1-8B-Instruct` (gated) | 32 | **16** | [15,31] |

Drivers read `reg.primary_layer` / `reg.band` and call `assert_matches_model` on the live model
(fail loud on a re-release). This is the full Paper 6 trio, so any decision-level coupling reads as
family-dependent or family-invariant against the Paper 6 baselines. **GPT-OSS is deferred** in B1
(harmony trace handling; revisit at Gate B). Directions never transfer across models: `V_moral`,
persona, refusal, and the judgment-decision direction are re-extracted in each model's own space.

---

## B1 — Moral-judgment decision direction (keystone)

**Stimuli.** ETHICS commonsense items already in repo (118 extraction pairs +
199 held-out; `../d1_moral_subspace/outputs/full/ethics_*.json`), formatted with the Paper 5
moral-judgment eval template (forced-choice *wrong* / *not wrong*). Answer order counterbalanced
across items.

**Direction extraction — selection-effect control (primary design, fixed now).** The naive
contrast (activations grouped by model *output*) leaks content, because items the model calls
"wrong" are disproportionately truly wrong. The **primary** contrast is **within-ground-truth
label**:

- among **truly-wrong** items: diff-of-means(model-says-wrong, model-says-not-wrong);
- among **truly-not-wrong** items: the same within-label diff;
- average the two unit-normalized contrasts.

Extract at the **last token before the verdict token** (the decision site), at the headline layer
+ band. **Secondary:** pooled output-contrast. **Reference:** the label-contrast (the content
direction). Expected error counts ~70–90 total at ~0.75 judgment accuracy, so bootstrap CIs are
mandatory. **Fallback (fixed now):** if either within-label cell has `n < 15`, report it and fall
back to the pooled contrast with the content-leak caveated explicitly.

**Measurements + pre-registered rules.**

- **(i) `p(judgment-decision | V_moral)`** with the two-step null and (once B3 lands) the syntax
  and register controls. **R2 branches, both publishable:**
  - **sensitivity-confirmed** (clears null q95 + M): the instrument detects morally-mediated
    decisions, so the refusal nulls in Papers 5–7 / D1 are strong evidence.
  - **category-mismatch** (≤ null + M): decision directions generically don't project onto content
    subspaces; D1 claims are **reframed, not retracted**, and (ii) becomes the primary instrument.
- **(ii) `|cos|(refusal, judgment-decision)`** and mutual projection, judged against the
  distribution of `|cos|` between covariance-matched random direction **pairs**. **R3 rule:**
  decision-level coupling detected **iff** above pairwise-null q95 + M.
- **(iii) Causal cross-ablation** on the 24-prompt harmful eval + the Paper 5 judgment battery:
  ablate judgment-decision → measure refusal rate; ablate refusal → measure judgment accuracy
  (Llama's 0.75→0.604 is the known half; this adds the reverse arrow and the OLMo pair). Three-way
  controls throughout (matched-random + persona) via `random_ablation_control.py`. Coherence filter
  on; over-ablation regimes reported, never headlined.
- **Stretch (cheap while loaded):** single-vs-full-rank LDA gap for the judgment contrast — is
  moral judgment itself a single-direction decision?

Per-prompt activations and per-item labels are saved for bootstrap CIs.

---

## B2 — Outcome-conditioned P2 (comprehension vs decision, in-trace)

**Pilot gate first (~0.5 h).** GPT-OSS refuses 1.0 at greedy on the harmful set, so outcome
variance must come from **borderline** prompts at temperature. Borderline set = **real XSTest**
(Röttger et al., NAACL 2024), prompts only, from `paul-rottger/xstest` at a **pinned commit** (NOT
the HF `xstest-v2-copy` mirror, whose bundled model completions carry other licenses). License
**CC-BY-4.0** (verified at source 2026-07-01). ~40 items selected (milder unsafe-contrast
categories + safe items models waver on), copied into `data/xstest_borderline.json` with a
provenance block (source commit, CC-BY-4.0 notice, citation, modification statement) + a NOTICE
entry, per the MORABLES-era data policy. Pilot: 8 prompts × 8 rollouts, `T ∈ {0.8, 1.0}`, ≤512
tokens, closure + coherence filters per Paper 7. **Proceed on a model only if ≥ 6 prompts show
mixed refuse/comply outcomes;** otherwise run the design on OLMo-3-Think and record GPT-OSS as
outcome-invariant (itself a datapoint).

**Main run.** Per prompt, diff-of-means at the P2 window between refuse-rollouts and comply-rollouts
(topic held fixed within prompt), averaged across prompts; project onto that model's `V_moral` vs
null/controls; save per-rollout activations. **R4 readings:** projection ≥ moral-family-band-min − M
→ the in-trace *decision* is moral-adjacent (genuine in-trace re-coupling); ≤ null + M → the D1
gradient was comprehension content, and the sharpened claim is "harm comprehension is moral-family;
the decision is not, at any position."

---

## B3 — Batched extractions while models are loaded

- **Syntax + register control directions (R5)** on `think` and `gpt_oss` (OLMo base/instruct may
  run locally on MPS). Recompute the D1 control comparison with `c_syntax`, `c_register` beside
  persona. **R5 rule:** the "strong-form orthogonality" sentence in D1 keeps its wording **iff**
  refusal ≤ min(c_syntax, c_register) + M.
- **Rotation-specificity control for Paper 5 F2 (R6):** extract sentiment and syntax directions
  (raw format, Sprint-1 convention) on OLMo-3 base + SFT checkpoints; report their base→SFT rotation
  next to the moral subspace's ~40°. **R6 rule:** F2 keeps the word "specifically" **iff** the moral
  rotation exceeds the control rotations by ≥ 15°; otherwise F2 is re-worded as a global
  representation shift the moral subspace shares.
- **Fables schema probe:** 20 amoral cautionary-tale items (hot stove / wet floor); projection onto
  `d_fables` vs `d_moral` to bound narrative-lesson-schema loading.
- **Re-extract anything on `MISSING_ARTIFACTS.md` with per-pair saves.**
- **If A5 triggered (cos ≥ 0.50):** per-checkpoint proto-refusal directions across the stage-3
  anneal grid (refusal-crystallization curve).

---

## B4 — Llama residual-refusal localization (diagnostic only)

Within the Paper 6 safety constraint (characterizing a released model):

- After best single-direction ablation (L13), re-extract the refusal diff-of-means and the
  Ledoit-Wolf single-vs-full gap **on the ablated model**; the pre-ablation gap of 0.000 with
  post-ablation refusal at 0.475 hints the behavioral mechanism is not the linearly separable one —
  confirm or refute.
- Rank-k ablation sweep (`k = 1..4` from the across-band refusal basis; eff-rank 4) with behavioral
  refusal + judgment battery + two controls at each k. **Claim ceiling:** characterization of where
  Llama's residual refusal lives; no removability optimization.

---

## B5 — Moral fragility of refusal: baseline the metric (R8)

The program's Q2 progress metric, measured on **released models only** (no training —
Direction-2-neutral). Per model (the B1 panel; GPT-OSS stretch with Phase-0d coherence caveats):
inject **RMS-normalized** noise at the headline layer, restricted to (a) the rank-3 `V_moral`
subspace, (b) `N` covariance-matched random rank-3 subspaces (the null generator — the noise
floor), (c) the persona subspace; sweep amplitude; measure harmful-set refusal rate, XSTest-safe
over-refusal, and coherence at each amplitude.

**σ\*_behavioral** = amplitude where harmful refusal falls below `0.5×` baseline (criterion form
fixed now; the exact ratio is ratified in the Gate-A ratification step, per the Track-1 two-step
pattern). **R8 rule:** **differential moral sensitivity iff σ\*_moral is an outlier *below* the
random-floor distribution** (the mirror of the outlier-above-floor ablation logic). Pre-declared
branches, both publishable: no differential sensitivity anywhere = the flat baseline any future
coupling intervention must move; differential sensitivity in Llama = converging evidence for its
behavioral coupling from a third independent design.

---

## GATE B (human)

B1's branch decides Phase C priority. **Sensitivity-confirmed + coupling detected** → C1 localizes
the read. **Category-mismatch** → C1 becomes the primary instrument for "does refusal read moral
features through weights." B2's outcome updates the D1 / Paper-7 write-ups either way.

## Phase C — gated follow-ons (choose at Gate B; do not start early)

- **C1 — Cross-layer read test (~2–4 h).** Jacobian of the refusal readout w.r.t. residual
  activations at the moral-**usage** layer (Paper 1's causal peak, not the storage peak); project
  the Jacobian's leading right singular vectors onto `V_moral` and onto the judgment-decision
  direction; complement with attention-head attribution at the decision position. Direct test of
  reading-through-weights that same-layer cosines miss.
- **C2 — Counterfactual moral-consistency DPO — HELD (2026-07-01).** Training is out of scope this
  cycle; its measurement component is pulled forward into B5. Optional zero-GPU prep only on
  request: counterfactual twin preference pairs from the v2 1,200 minimal pairs into `data/`, with a
  header stating **no training use** until a dedicated Direction-2 pre-registration exists. Requires
  scope note + explicit human go.
- **C3 — Gemma-Scope co-occurrence (~2–3 h, no SAE training).** Gemma-2-9B-IT + released Gemma Scope
  SAEs: do moral-content features and refusal-active features share atoms / co-fire across
  positions? A features-level replication of the dissociation at near-zero training cost.

---

## Pre-registered decision rules (the shared R-table)

`M = 0.05` throughout. Both branches of every rule are pre-declared publishable.

| # | Owner | Quantity | Rule | Branches |
|---|---|---|---|---|
| R1 | calib (A1) | Moral-family band per tag | `[min,max]` of held-one-out projections | yardstick for "moral-adjacent" language |
| **R2** | **D2 (B1)** | Judgment-decision on V_moral | clears q95 + M? | sensitivity-confirmed ↔ category-mismatch (reframe, not retract) |
| **R3** | **D2 (B1)** | cos(refusal, judgment-decision) | > pairwise-null q95 + M? | decision-level coupling ↔ decision-level dissociation |
| **R4** | **D2 (B2)** | Outcome-conditioned P2 | ≥ band-min − M / ≤ null + M | in-trace decision moral-adjacent ↔ gradient was comprehension |
| R5 | calib→B3 | Strong-form orthogonality sentence | refusal ≤ min(c_syntax, c_register) + M | keep sentence ↔ hold sentence |
| R6 | calib→B3 | Paper 5 F2 "specifically" | moral rotation − control rotation ≥ 15° | keep ↔ re-word as shared shift |
| R7 | calib (A4) | Combined P2 (exploratory) | Fisher across Think + GPT-OSS | post-hoc aggregate only |
| **R8** | **D2 (B5)** | Moral fragility of refusal | σ\*_moral outlier below the random-floor distribution? | differential sensitivity ↔ flat baseline |

## Compute budget

| Item | GPU | Est. hours |
|---|---|---|
| B1 judgment direction (3 models) | A100 | 1.5–2.5 |
| B2 pilot + main | A100 | 2.5–4.5 |
| B3 batch | A100 | 1–2 |
| B4 Llama | A100 | 1–2 |
| B5 fragility baseline (3 models) | A100 | 2–3 |
| C1 / C3 (C2 held) | A100 | 2–4 / 2–3 |

Phase B totals ~8–14 GPU-hours, planned as two pod sessions: session 1 = B1 + B3 + B5 (loaded-model
batch); session 2 = B2 + B4 (generation-heavy). Dependency: the XSTest subset must be pulled and
committed during Phase A data prep so B5's over-refusal arm has it in session 1.

## Integrity statement

No decision-coupling headline quantity was computed before this file was committed. Any later change
to a threshold, a null construction, or a decision rule must be a dated amendment below this line,
never a silent edit.

### Amendments

*(none yet)*
