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
  - **Calibrated target (from the A1 moral-family band, R1).** The band now gives R2's
    sensitivity-confirmed branch a concrete yardstick, not just "above null": a judgment-decision
    direction that lands **inside the moral-family band** while refusal sits at ~0.14 (P_B) is the
    program's thesis in a single figure (moral *decisions* project like moral content; the refusal
    *decision* does not). Report the judgment-decision projection against the band alongside the
    null, on the same ladder.
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

- **2026-07-01 (Amendment 1 — massive-activation degeneracy → standardized geometry, in-format
  ladder, cross-ablation reconcile; pre-registered BEFORE any recomputation):** Chunk-1 B1/B3
  completed for the panel and surfaced a discovered, outcome-independent data property that
  invalidates the covariance-matched null for two of three models. Recorded here before recomputing
  any R2/R3/R5/R8 quantity.

  **(1) Discovered property (outcome-independent).** Qwen2.5 and Llama-3.1 instruct have
  **massive-activation outlier dimensions** (Qwen dim 458 = 59% of residual variance, Llama dim
  788 = 32%; OLMo-3 top dim = 1.4%). These dominate raw mean-diffs (Qwen ethics≈moral `|cos|`=0.90)
  and saturate the covariance-matched null (R2 null q95: Qwen 0.92, Llama 0.36; R3 pairwise-null
  q95: Qwen 0.995, Llama 0.90), so Qwen/Llama R2/R3/R5 are uninterpretable as run. OLMo-3 is
  well-conditioned (null 0.26, matching d1). This is the known attention-sink / massive-activations
  phenomenon and is the cross-model analog of d1's eff-dim-385 content-domination caution; it is a
  property of the models' activations, independent of what refusal/judgment do.

  **(2) Primary fix — STANDARDIZED geometry.** Recompute mean-diff directions and the covariance
  null in a per-dimension-standardized (z-scored) space: `d̃ = unit(d / σ)`, null drawn in the
  standardized space. **Rider (a) — σ provenance is part of the recorded type block per
  measurement:** σ must come from `act_samples` matched to the **format/position class** of the
  directions being compared (chat-decision-site σ for the chat-decision R3 cell; raw-pooled σ for
  the raw-content cells), and **sink-position tokens (BOS / first-token spikes) are excluded from
  the σ sample** — otherwise the transform imports a new format confound. Every recomputed number
  records which σ it used.

  **(3) Rider (b) — OLMo invariance check (legitimacy proof, pre-registered).** Because OLMo has no
  outlier dim, standardization must **preserve OLMo's R3 conclusion** (decision-level dissociation,
  `|cos| ≤ pairwise-null + M`) raw→standardized. If OLMo's verdict flips under the transform, the
  transform is manufacturing results and is **rejected**; the clean-instrument invariance is the
  proof standardization is not fabricating the Qwen/Llama fix.

  **(4) Rider (c) — top-k projection-out ROBUSTNESS VARIANT (not primary).** Alongside
  standardization, report a variant that projects out dimensions **individually carrying > 5% of
  residual variance** (criterion-based, not a free `k`: Qwen dim 458, Llama dim 788 qualify; OLMo
  none). Reported as robustness; standardization is primary.

  **(5) Rider (d) — Paper 6 back-audit.** Paper 6's cross-model geometric cells for Qwen/Llama used
  the same covariance-matched null and were presumably saturated identically → **recompute those
  geometric cells in standardized space from saved artifacts** (zero-GPU if the activation hygiene
  held; else a `MISSING_ARTIFACTS.md` rider to regenerate). Llama's **behavioral** results
  (ablation resistance, the judgment drop) **do not depend on the null and survive untouched**; only
  the geometric numbers are re-audited.

  **(6) Rider (e) — promote the degeneracy to a methods finding.** File in `ANOMALIES.md`:
  *"covariance-matched nulls are unusable in massive-activation families (Qwen/Llama) without
  robustification"* — a citable methods contribution connecting to the massive-activations /
  attention-sink literature, relevant to anyone doing direction geometry on those families.

  **(7) In-format ladder (the one targeted GPU chunk; local standardization runs first regardless).**
  The R2 confound is separate from the outlier one: `V_moral` was extracted RAW/mean-pooled while the
  judgment + refusal directions are CHAT/decision-site, so even the content `label_contrast` projects
  low (OLMo 0.10). Fix by re-extracting the `V_moral` sources in **chat format, factor-decomposed** —
  the same texts wrapped as user messages, extracted **both at last-token and mean-pooled-in-chat**,
  so template and pooling contributions separate — then recompute the **entire in-format ladder**:
  held-one-out band, syntax/register/persona controls, G3 refusal projection, and R2.
  **Pre-registered branch (both publishable):** D1's orthogonality claim is **format-robust** iff
  refusal sits below the in-format band by `M`; **otherwise it is scoped to the raw narrative
  register**, with the attenuation documented (the register-bound reading — moral geometry itself is
  register-dependent — is a pretraining-data story, genuinely interesting). Qwen/Llama directions are
  defined in **standardized space and mapped back to model coordinates for injection**. **B5 rides in
  this chunk and uses the CHAT-extracted subspace** for noise injection (behavior happens in chat
  format; noising the raw-register subspace during chat inference would repeat the cross-format
  mistake — which is why B5 correctly waited).

  **(8) Cross-ablation reconcile rider.** The R3(iii) cross-ablation cell is currently
  **unfalsifiable**, not merely underpowered: base refusal 0.167 on the 24-prompt subset is ~4
  refusal events (a floor effect), and 0.167 disagrees with **Paper 6's ~0.575** for the same model
  — a smell of **harness drift** (subset selection, chat template, or refusal-classifier), not model
  behavior. Reconcile the eval configs at **zero GPU** (diff subset/template/classifier) before the
  pod, then **rerun cross-ablation on the full high-base-rate harmful set**.

  **Spine preserved.** `M = 0.05`, the R1–R8 rules, and NULL-publishable-both-branches are
  unchanged. This amendment robustifies the instrument (standardization + top-k robustness), scopes
  the format claim (in-format ladder branch), and reconciles the ablation harness — all recorded
  before recomputation.

- **2026-07-01 (Amendment 1 addendum — in-format-ladder position classes, σ\* ratio, zero-GPU reads;
  pre-pod, before the chat-format extraction runs):**
  - **Position classes (one forward pass, saved as band-layer slices).** Extract three:
    (1) **last-content-token**, (2) **final pre-assistant token**, (3) **mean-pooled-over-content**.
    **PRIMARY = final pre-assistant token** for the G3 format-robustness scoping branch (the decision
    site, apples-to-apples with the refusal gate and the judgment-decision direction); (1) and (3)
    are secondary. **In-format applies to EVERY rung** — held-one-out sources, persona / syntax /
    register controls, AND the null's `act_samples` — all chat-format at the matched position class.
    Every recomputed number's type block carries `{format, position_class, σ_provenance}`.
  - **σ\*_behavioral ratio RATIFIED = 0.5× baseline** (Track-1 two-step: criterion form fixed in A0;
    the 0.5× ratio is ratified here, before B5 runs). B5 injects into the **chat-extracted** `V_moral`
    at **generation positions**, RMS-scaled from **chat activations at those positions**; Qwen/Llama
    injection directions are defined in **standardized space and mapped back to model coordinates**.
  - **Zero-GPU reads (recorded before the pod).** (i) The chat decision-site space has **no dim > 5%
    of variance** (projection-out is a no-op there), so R3's Qwen 0.32 is **not an outlier artifact —
    it persists**. (ii) R3 dissociation margins below the MDE (`pairwise-null q95 + M`): OLMo 0.35,
    Qwen 0.15, Llama 0.48 — all clear. (iii) OLMo's R3 null rose **0.27→0.41** because the
    format-matched null now draws from the **chat decision-site covariance** (`acts_headline`), not
    the raw pooled `act_sample` B1 used — higher baseline collinearity of the decision-site space,
    not a regression. (iv)/(v) recorded in `../ANOMALIES.md`. **R5 (raw-format, format-confounded) is
    the cell standardization and projection-out DISAGREE on** for both Qwen and Llama (Qwen: std →
    refusal 0.20 > controls 0.10 = strong-form FALSE; projout → refusal 0.21 < controls 0.45–0.55 =
    strong-form TRUE) → **not resolvable by robustification; the in-format chat ladder is the
    discriminator** (the chat space is outlier-free per (i)).
