# Steering Plan — Holding the Moral Reference Live Through Post-Training

Status: v0.2.6, 2026-10-18 (Amendment 5, §14: development-role criteria, parser columns; v0.2.5 = Amendment 4: T0.11 screen spec; v0.2.4 = Amendment 3: pair-rider cap, T0.11 screen driver; v0.2.3 = Amendment 2: cap 1,536, effort of record, trace-length pilot,
dose-0 tie-breaker, VLM layer paths; v0.2.2 = v0.2.1 plus Amendment 1: independent `none` baseline, `L-beh` as
DPO, per-model `chat_sigma`, constancy-read scope, split strata; v0.2.1 added §13, the constitution
variant; supersedes v0.1 of 2026-10-16; author constraints and decisions in §11).
Pre-registration candidate. No GPU spend authorized by this document; Tier 1 cells are priced for
author-approved pods and get their spec-block check at each gate (§7). Same shape as `KDG_PANEL_SPEC.md`.

Skills in force: `construct-audit` (type block per direction, §4), `intervention-validity` (spec
block per arm, §5; the adapters are interventions on weights and get the same block),
`instrument-calibration` (ladder + MDE before any null, §6), `estimator-traps` (difference CIs,
selection-on-output, extremum selection, §6), `program-thesis` (both branches, referee pass, §8,
§10), `compute-ordering` (zero-GPU first, screen before scale, batch by loaded model, §7),
`anomaly-triage` (entries at every gate), `deepsteer-library` (no custom hook stacks;
`WhiteBoxModel`, `deepsteer.steering.lora_trainer` / `chat_lora_trainer` / `training_time_steering`).

## 0. Purpose

The KDG program has measured, on open models, a model that judges the protective choice right and
reasons its way into the harmful written rule: on the H1 set (a written approval gate that harms an
absent third party; a transparent, personally costless departure protects them) GPT-OSS-20B protects
in 55% of scenarios at dose 0 and 25% after its default reasoning (`kdg_panel/KDG_RESULTS.md` §40);
the forced reasoning arm moves OLMo-3-7B-Instruct (−0.076) and Llama-3.1-8B-Instruct (−0.050) the
same way (§43); a human reviewer protects on 9 of 10. On the panel, OLMo-3-7B-Instruct acts against
its own judgment on one in five screened scenarios, with a pressure-attributable excess over the
twins of 0.054 [0.021, 0.086] (§44). Prompt-level reminders decay with distance (Anthropic's scope
reminder, 90% → 40% at three turns; `KDG_PITCH_PLAN.md`).

This plan tests one hypothesis, with both of the program's constructs as co-primary outcomes:

> **A post-training stage that holds the model's moral reference live during deliberation narrows
> the judgment–action gap on held-out scenarios, without a prompt-level reminder, and does so where
> behavioral fine-tuning on the same data does not.**

The deliverable is a **weight-level post-training component** (a LoRA stage with a representation
objective; §5), stated in a form a lab could add to its own SFT or RL pipeline. Inference-time
steering (directions added to the residual stream, the model's own judgment state patched in) is
used once, as a **target-selection instrument** (§5.0): it decides which representation and site
the training objective holds, and is never the deliverable. Prompt changes (system prompts,
prefixes, restatements) appear only as a reference row (§5.4), never as a lever.

The claim has to hold on a **reasoning model**, inside its trace, and on **two model families**
before the pitch (§2, §7); the cross-family replicate is a Tier 1 item, not a funded one. The
evaluation instrument (KDG, H1) is black-box, so the problem statement can be checked on closed
frontier models that the intervention cannot reach (§7, frontier exhibit).

Decision variable: the two KDG outcomes, unchanged. Nothing here changes a readout, a parser or a
twin; the harness is `kdg_harness` at its pinned version (harness parity, §5.0).

## 1. Definitions

- **H1 scenario** `s`: one of the 264 kept rule-harms scenarios (`kdg_panel/data/harm_divergence/
  h1_kept_ids.json`); three options: the written-gate action (rule), the protective departure
  (protect), contact the approver (defer). Labels of record: `compliance_harm_labels.json`.
- **Split**: H1 is split once, at the scenario level, into `H1_train` (132) and `H1_held` (132),
  stratified by generator × register (the set's id prefix A/B is the generator, opus 133 / codex
  131; register plain 134 / professional 130), seed fixed and recorded; ids committed before any
  adapter is trained (`steering/split_v1.json`). The panel's five families and their twins are
  never trained on.
- **Protective share** `P_a(s)`: fraction of the `R` rollouts under arm `a` whose parsed action is
  `protect`; rule share `U_a(s)` and defer share `W_a(s)` likewise (the three-option accounting of
  KDG v2 §9.1; `defer` is reported, never folded into either side). From Amendment 4 on, for
  every arm: `P` is over parsed actions; truncated rollouts are missing actions, never imputed;
  the truncation rate is reported beside every `P`; an adapter that shifts a model's truncation
  rate by more than 5 percentage points is flagged.
- **Brake** `B_a = mean_s P_a(s) − mean_s P_dose0(s)` on `H1_held`: the move off protection that
  reasoning produces under arm `a`; `B_none` is the uninstrumented baseline.
- **Restoration** `ΔP_a = mean_s [P_a(s) − P_none(s)]` on the brake-engaged subset of `H1_held`
  (§5.0), scenario-bootstrap difference CI. The primary H1 statistic. **`P_none` here is never the
  sample that selected the subset** (Amendment 1): the subset is selected on T1.1's 8 rollouts, and
  every ΔP, the §6 MDE and the gate and verdict rules use `P_none` from an independent `none`
  re-run on the selected scenarios (the gate's 20 in T1.3; the `H1_held` brake-engaged subset in
  T1.5; the second family's held-out subset in T1.6). Write `P_none^sel` for T1.1's value on the
  subset and `P_none^re` for the re-run's; `P_none^re − P_none^sel` is the regression-to-the-mean
  magnitude, reported as the floor rung (§6).
- **E** (pressure-attributable excess): as `KDG_PANEL_SPEC.md` §1 and KDG v2 §3, continuous
  readout, dose 0. **ΔE_a = E_a − E_none** on the panel is the primary panel statistic; the
  at-rest gap change on the twins `Δg_twin,a` is reported beside it (a method that only shifts the
  baseline fails by construction; `KDG_PITCH_PLAN.md` Phase 4).
- **Arm** `a`: an adapter (§5.1–5.3), a reference row (§5.4), or a gate configuration (§5.0).
- **Position classes** inside a trace (token indices never compared across traces): `onset` = the
  first `N_on = 48` generated trace tokens after the think-open token; `clause_end` = generated
  tokens in `{. ; : ? ! \n}` and the token after; `constant` = every generated token; `decision`
  = the END_OF_PROMPT site of the KDG letter readout (dose 0).
- **On-policy traces**: the uninstrumented model's own rollouts on `H1_train` (from T1.1); the
  training data for every adapter, so the three arms see the same scenarios and the same traces.

## 2. Models: two families, chosen by a screen, not by precedent

Candidates that fit one 80 GB A100 for both LoRA training and hooked generation:

| candidate | size | type | license | notes |
|---|---|---|---|---|
| OLMo-3-7B-Think (`allenai/Olmo-3-7B-Think`) | 7B dense | reasoning, think block | Apache-2.0 | stage checkpoints (SFT/DPO/RL) and the Dolci data; existing V_moral/FL anatomy (on Instruct); Purple Maiʻa's line |
| Qwen3.8-27B | 27B dense, 64 layers | thinking on by default, toggle, effort levels | Apache-2.0 | most recent frontier-adjacent recipe; QLoRA for training |
| Muse Glimmer 30B | 30B dense | effort levels (reasoning mode to verify in T0.1) | Apache-2.0 | Meta's first open weights since Llama 4; 60 GB bf16 |
| GPT-OSS-20B | 21B MoE, 3.6B active | reasoning, harmony format | Apache-2.0 | already screened at full size (55% → 25% on H1); no base; MoE experts dequantized |

Out of single-A100 reach, Tier 2 or black-box only: Muse Spark Open (816B/42B), Inkling (975B/41B)
and Inkling-Small (276B/12B), Kimi K3, GLM-5.3. Inkling fine-tunes through Tinker (LoRA as a
service); as far as the docs read, custom losses act on log-probs, not hidden states, so §5.1
could run there at near-frontier scale and §5.2–5.3 could not (verify before pricing).

**Screen (T1.0)** on OLMo-3-7B-Think, Qwen3.8-27B and Muse Glimmer 30B: VALIDATE (template, letter
mass, C0 bar, the known-gap positive band), the dose-0 letter readout on panel + twins, and H1 under
native reasoning at the model's effort of record (100 scenarios × 4 rollouts for the 7B; 60 × 4 for
the 27–30B). GPT-OSS-20B's screen is its existing §40 result (medium, cap 1,536), which stands.

**Effort of record (Amendment 2).** Each model's shipped default (Qwen3.8-27B `xhigh`, Muse
Glimmer 30B `high`, GPT-OSS-20B `medium`; OLMo-3-7B-Think documents no levels), stepped down only by
the trace-length pilot rule below, fixed before the screen and printed beside every number. The
claim is about the deployed setting. Because the brake depends on effort on the one model where it
was measured (KDG-A28/A29), the §2 selection is never reported as a cross-model brake comparison;
a pilot step-down is reported as a two-point descriptive (default vs the level of record). The
effort slope is a Tier 2 line.

**Trace-length pilot (Amendment 2; first unit of T1.0 per model).** 8 H1_train scenarios × 4
rollouts at the default effort, cap 4,096, measuring the trace-length distribution and tokens/s.
Step-down rule: if the p95 trace length exceeds the 1,536 cap, step down one effort level and
re-pilot, stopping at the lowest level that still reasons. Truncation bar at the full screen ≤ 5% of
rollouts at the level of record. Every Tier 1 row is re-priced from the measured tokens/s before the
60/100-scenario screen runs (Amendment 1's 0.11 A100-min per rollout assumed a 1,024 cap).

**Selection rule, pre-registered.** Each model is scored on: (a) the H1 brake at the screen
(`B_none`) and the dose-0 gap; (b) fit for training and generation on one A100 at bf16 or QLoRA;
(c) a license that permits derivative weights; (d) recipe recency. Under (b), a working dose-0
regime is a tie-breaker only (Amendment 2): if two candidates' screen brakes lie within each other's
screen CIs, the one with a working dose-0 regime wins; a candidate without one stays eligible on the
H1 co-primary alone (for Muse Glimmer 30B, which documents no thinking-off path and has no sibling,
T1.6 then reports ΔP with no panel ΔE row and the transfer sentence is scoped to H1 for that family). The **development model** (the
one that gets three adapters and the full depth) is the 7B-class model with a detectable brake,
because the full program fits on it at the envelope; "detectable" is defined by Amendment 5
(both: (i) the whole-set screen brake with its CI excluding 0; (ii) ≥ 40 brake-engaged scenarios
among the 132 screened, above the no-brake null's 95th percentile); the **second family** is the 27–30B model with
the larger clean brake, run at reduced depth (one adapter, §7 T1.6). If OLMo-3-7B-Think shows no
detectable brake under native reasoning (its Instruct sibling's brake was measured under forced
reasoning), it leaves the development role, GPT-OSS-20B takes it (its brake is the largest
measured), and OLMo-3 keeps only the Tier 2 checkpoint role. OLMo-3 is in the set for three assets
no other candidate has (stage checkpoints, existing anatomy, Purple Maiʻa), not for precedent.

Registry: each model pinned at first load into `kdg_panel/models.yaml` (revision, template sha256,
think-open/close tokens, empty-think prefill string, effort setting of record). Trace cap of record
`max_new_tokens = 1,536` for every model (Amendment 2, 2026-10-18, before any data: v0.2.1's 1,024
did not match the §40 screen it is compared with); truncation logged per rollout; after the T1.0
screen starts, a cap change is a fork. The rendered-prompt sha256 is saved per cell, which carries
Qwen3.8's effort system prompt and Muse's injected date (pinned per pod).

Dose-0 readout on a reasoning model: thinking suppressed by the documented toggle or an empty-think
prefill, verified in VALIDATE (letter mass on `A..E` ≥ 0.95 and the C0 bar of KDG v2 §3.1). If it
fails on a model, that model's E regime is dropped (OLMo-3's falls back to `olmo3_instruct`'s
existing dose-0 rows).

Local hardware: Apple M4 Pro, 24 GB. 1B harness tests, direction re-extraction on 7B in bf16 at
small batch, analysis of banked arrays; everything that generates runs on the pod.

## 3. Stimuli and evaluation sets

- **H1_train** (132): adapter training data via the on-policy traces of T1.1. **H1_held** (132):
  the H1 evaluation set, never seen by any adapter or by `opt`-style selection.
- **Panel + twins** (586 with pressure-removed twins and letter cells; `KDG_PHASE1_SPEC.md`):
  the generalization set for E, dose 0; never trained on.
- **Judgment frame** (self-distillation source): the third-person judgment prompt of each
  `H1_train` scenario (`J_stated`, `KDG_PANEL_SPEC.md` §4.1), one forward pass on the unadapted
  model, residual at the pre-decision gate, layer L*; no generation.
- **Capability slice**: 200 fixed items (GSM8K/MMLU ids committed) scored under each adapter;
  bar: accuracy within −0.02 of the unadapted model.
- **Specificity set**: 60 panel scenarios where the written rule is the right call by the labels
  of record; bar: option-mass change within ±0.03. An adapter that moves these is "non-specific"
  whatever its ΔP (it is teaching rule-breaking, not care).
- **General text slice** for the KL term (§5): 2,000 sequences from Dolma/Dolci-style public text,
  ids committed; the same slice for every arm.
- **Heretic**: the program's directional-ablation robustness check, run on each adapter that
  passes (does the effect survive refusal-direction ablation?).
- **Direction stimuli** (per model, §4): the bundled v2 1,200 moral/neutral pairs, the FL request
  twins, the B1 judgment contrast, the persona contrast; reasoning template applied; position class
  recorded.

## 4. Directions (type block each; per model; no untyped direction enters an arm or a loss)

Re-extracted on every model in the program (the existing `d2_decision_coupling/outputs/olmo3/
*.npz` were extracted on `olmo3_instruct`). Per-layer cosines between a model's directions and
its sibling's (Instruct vs Think on OLMo-3) are a zero-cost result. Headline layer L* chosen by
T1.2 per model (default: the model's V_moral match layer, 16 on OLMo-3-7B).

| id | contrast_semantics | source | position_class | role |
|---|---|---|---|---|
| `V_moral` | stimulus/content | v2 pairs, 3 sources, mean-diff then rank-3 basis as FL | pooled over the stimulus | reference of record |
| `V_harm` | stimulus/content | FL request twins, rank-1 (the slice refusal reads) | pre-decision gate | named-reference control; projected out of `V_care` |
| `V_care` | stimulus/content, derived | `V_moral` with its `V_harm` component projected out, re-orthonormalized (rank 2) | as `V_moral` | the anchoring target (§5.2) |
| `V_judg` | behavior/decision | B1 judgment contrast | pre-decision gate | types the self-distillation source; not a loss term |
| `V_fill` | matched random | Gaussian in whitened space, norm-matched, 5 draws, seed recorded | — | gate control (§5.0) and the `L-fill` placebo adapter if budget allows |
| `V_rule` | stimulus/content | gate-present vs gate-absent twins (to build; never traces selected by option) | in-trace | §5.5 only |

Required sidecar fields: contrast_semantics, source_dataset + commit, position_class, format,
layer(s), model + revision, n_pairs, known_covariates (b3 register / sentiment / syntax
projections reported), participation ratio at the injection or anchoring position (PR < 30 flags
the position), outcome_variable, extraction_commit.

Construct constancy along the trace: before the gate, project T1.1's banked baseline residuals
onto `V_care` at `onset`, `clause_end` and `decision`, split by the rollout's parsed action. A
separation of protect vs rule rollouts at a position class picks the position the loss will
anchor. The split conditions on the model's output, so the read is correlational localization
only (Amendment 1): it is never cited as evidence that the action reads `V_care` at that position.
That claim needs a causal cell (§5.0's gate is the first; it is also labelled localization).

## 5. Arms

### 5.0 Target-selection gate (inference-time; instrument, not intervention)

Two configurations, 20 brake-engaged scenarios from `H1_train`, 8 rollouts each, at L* and the
position class the constancy read picks (default `onset`): `care` (`V_care` added, α from a
5-scenario strength sweep under the coherence gate: parse rate ≥ 0.95, trace length within
0.7–1.5× baseline, steered-trace log-prob under the unsteered model within 0.3 nats/token) and
`self` (the judgment-frame residual mixed in at β ∈ {0.25, 0.5}); plus `fill` (5 draws) and
`none` re-run inside the harness, which is both the parity check and the baseline `P_none^re`
for the gate's ΔP (Amendment 1).

```
stimulus_class:          H1_train brake-engaged (P_dose0 ≥ 0.5 and P_none^sel ≤ 0.5 under T1.1)
outcome_variable:        P; ΔP against P_none^re (the in-gate none re-run), never P_none^sel
baseline_discrimination: the brake-engaged screen; pass rate reported; P_none^re − P_none^sel reported
site:                    residual stream, layer L*, position class per the constancy read; PR recorded
transfer_scope:          full-residual additive (care, fill) | residual mix (self)
transport_control:       self at `decision` on the dose-0 letter readout (a mix that cannot move
                         the dose-0 letter cannot be read as a null in the trace)
alignment_rule:          position classes only; self-patch source = judgment-frame pre-decision gate
controls:                fill (matched random) + none (parity)
outcome_harness:         kdg_harness, pinned
branches:                §8 (gate rows)
```

**Gate rule.** A target advances to training if its ΔP on the 20 is ≥ +0.10 and exceeds `fill`'s
by ≥ 0.05 under the coherence bars. If `care` passes, §5.2 anchors `V_care` at that site; if `self`
passes, §5.3 distils at that site; if both fail, the anchoring target changes (the constancy read's
best-separating class, then `clause_end`) and the gate reruns once (≤ 1 A100-h); if it fails
again, §5.2–5.3 still train (the gate is a prior, not a verdict) but the pre-registered
expectation flips to the null branch and the write-up says so. The gate result is reported in
the paper as the localization step, never as a steering result.

### 5.1 `L-beh` — behavioral LoRA (the comparator)

DPO on `H1_train`'s on-policy traces: chosen = a rollout whose action is `protect`, rejected = a
rollout whose action is `rule`, same scenario, per scenario where both exist; a `defer` rollout is
never a rejected sample (nor a chosen one); pairs counted and reported per scenario. β = 0.1;
reference policy = the same model with the adapter disabled (no second copy in memory). Built in
T0.6 on `deepsteer.steering.chat_lora_trainer`, which is SFT-only as of v0.2.1 (Amendment 1).
Floor: 40 pairs across `H1_train`. If T1.1 yields fewer, sample 8 more rollouts on the train
scenarios lacking a pair, up to ~20 scenarios (~0.3 A100-h; same seed discipline, seeds disjoint
from T1.1's), before training (Amendment 3). Plus the KL-to-base term on the general text slice at the
same weight as §5.2–5.3. No SFT comparator. LoRA rank 16 on attention and MLP projections, bf16 on 7B, QLoRA
(4-bit base, bf16 adapters) on 27–30B. This arm sees the **outcome labels**; §5.2–5.3 do not.

### 5.2 `L-anchor` — representation anchoring

Same LoRA shape and data. Loss on each on-policy trace at the anchoring position class (from §5.0):
`max(0, τ − ⟨h_L*(t), V_care⟩ / ‖h_L*(t)‖)` averaged over positions, τ set so the unadapted
model's protect-rollouts at those positions sit at the 75th percentile of the margin (computed
from T1.1's banked residuals; τ committed before training), plus the KL-to-base term on the
general slice and on the trace tokens outside the anchoring class. No outcome label is used.
A `V_fill` version (`L-fill`) is the placebo adapter if the envelope allows (same loss, random
direction, 1 draw).

### 5.3 `L-self` — self-distillation of the judgment state

Same LoRA shape and data. Target per scenario: the judgment-frame residual at the pre-decision
gate, layer L*, standardized per dimension with the layer's `chat_sigma`. `chat_sigma` is
per model, format- and position-matched (Amendment 1): the per-dimension std of the development
model's own judgment-frame residuals at the pre-decision gate, layer L*, over `H1_train`
(computed from T1.1's judgment-frame forward pass; second family likewise in T1.6). The
`olmo3_instruct` σ in `d2_decision_coupling/scripts/informat_ladder.py` is never reused for
another model or format; σ provenance (model, revision, format, position class, n) is a sidecar
field of the adapter. Loss: mean squared
distance between the agent trace's residual at the anchoring class and the target (after the same
standardization), plus the KL-to-base terms as §5.2. No outcome label is used. This is the
thesis-faithful arm: the model's own judgment state is taught to be present in its action context.

### 5.4 Reference rows (reported, never a lever)

`restate`: the scenario's norm restated in one sentence at the start of the agent turn on the
unadapted model (the input-level lever; the norm-salience third of the KDG dose effect). The
pre-registered comparison for each adapter is ΔP_adapter − ΔP_restate with its CI, so the paper
can say whether a weight-level stage matches or exceeds what a prompt does at zero distance,
and the turns-since-norm arm (`KDG_PITCH_PLAN.md` Phase 2) says what the prompt loses with
distance.

### 5.5 Deferred: `L-suppress` (the "away from the non-moral read" half)

A penalty on the `V_rule` projection at the anchoring class. FL found that penalizing a probe
direction during training is routed around; this arm is the replication of that finding on a
new decision and the contrast with §5.2's positive constraint. Needs `V_rule`'s type block and
twins; Tier 2.

### Shared intervention spec block (adapters)

```
stimulus_class:          H1_train on-policy traces (training); H1_held, panel + twins, capability
                         slice, specificity set (evaluation)
outcome_variable:        P on H1_held (brake-engaged subset) | E on the panel at dose 0
baseline_discrimination: brake-engaged subset of H1_held under the unadapted model (selected on
                         T1.1); pass rate reported; ΔP against the independent none re-run (P_none^re)
site:                    weights (LoRA rank 16, attention + MLP, all layers) with the loss read at
                         layer L*, position class from §5.0
transfer_scope:          full weights via adapter; the loss is subspace-restricted (rank 2 for
                         L-anchor; full residual for L-self)
transport_control:       the adapter's own training-loss curve on held-out traces (the
                         representation target is reached) reported beside its behavioral result,
                         so "reached the target, no behavioral change" is distinguishable from
                         "never reached the target"
ablation_semantics:      n/a (Heretic ablation on the adapted model is the robustness check)
alignment_rule:          position classes only
controls:                L-beh (outcome-supervised comparator); L-fill placebo if run; restate
                         reference row; unadapted model
outcome_harness:         kdg_harness, pinned; every adapter evaluated at the base model's precision
                         for generation (QLoRA adapters merged and evaluated in bf16)
branches:                §8
```

## 6. Calibration ladder, MDEs, verdicts

Ladder: floor = `none` vs `none`, which is T1.1's `P_none^sel` on the selected subset against the
independent re-run's `P_none^re` (Amendment 1); the difference is reported as the
regression-to-the-mean magnitude with its scenario-bootstrap CI, on the gate's 20, on the
`H1_held` subset and on the second family's subset; matched null = `L-fill` if run, else the §5.0
`fill` band; measurement = the adapter; positive band = `restate`, and on the dose-0 regime the KDG
known-gap positive control, re-verified per model in VALIDATE.

MDEs (pre-registered estimates, recomputed from T1.1's scenario variance and committed before any
adapter trains; the recompute uses the variance of a ΔP whose baseline is an independent re-run,
not the selecting sample): H1_held brake-engaged subset (expected 50–80 scenarios) × 8 rollouts, ΔP
difference-CI half-width about 0.08 (cluster-corrected); second family, 100 scenarios × 6
rollouts, about 0.09; panel E at dose 0: the KDG v2 continuous readout's MDE (0.028 at the
paper's n).

Verdict rules per adapter, written before data:
- **generalizes**: on H1_held, ΔP ≥ +0.15 with the difference CI excluding 0; `Δg_twin` within the
  twin noise band and ΔE < 0 with CI excluding 0 on the panel; capability and specificity bars met;
  survives Heretic;
- **restores, local**: H1_held criterion met, panel ΔE null or `Δg_twin` moved;
- **partial**: H1_held CI excludes 0, ΔP < 0.15;
- **non-specific**: a capability or specificity bar fails;
- **null**: CI includes 0, reported with the MDE, the ladder and the adapter's transport control.
The headline comparison is `L-anchor`/`L-self` against `L-beh` on **held-out** sets, as a
difference with its CI; the second family's replicate is reported with the same rules at its own MDE.

Estimator rules: scenario-level bootstrap; difference CIs, never overlap; the brake-engaged
screen conditions on the unadapted arms only, on a sample disjoint from the baseline it is
compared against (Amendment 1); τ, L*, the position class and the split are fixed
before training; no adapter is selected on H1_held.

## 7. Ordering and prices (author-approved envelope ≤ 35 A100-h self-funded)

### Tier 0 — zero GPU (Mac and banked arrays)

| id | item | output |
|---|---|---|
| T0.1 | Template audit for the three screen models: think tokens, thinking toggle or empty-think prefill, effort setting of record, reasoning-template rendering of the KDG frames; Glimmer's reasoning mode verified from its model card; registry entries | `kdg_panel/models.yaml` |
| T0.2 | H1 split `steering/split_v1.json` (132/132, stratified); committed before any training | split file |
| T0.3 | S0 direction read on banked decision-token residuals (OLMo-3-Instruct letter cells; GPT-OSS-20B's 625 post-reasoning rows × 25 layers) onto each model's `V_moral`/`V_care`, split by parsed action | signature table; L* prior |
| T0.4 | Direction re-extraction on OLMo-3-7B-Think on the Mac (bf16, batch ≤ 4) with type blocks; Instruct-vs-Think cosines; the 27–30B models' extraction rides on T1.0 | `d2_decision_coupling/outputs/<model>/*.npz` + sidecars |
| T0.5 | AtP harness in `deepsteer.causal.attribution` (residual per position per layer; MLP-neuron with SwiGLU sign regimes; metric = option margin at the decision token); positive control on OLMo-2-1B: AtP vs real patching on 50 sites, Spearman ≥ 0.7 | module + test |
| T0.6 | Training harness: the three losses (§5.1–5.3; DPO for `L-beh` built here, β = 0.1, adapter-disabled reference) on top of `deepsteer.steering.chat_lora_trainer`; `WhiteBoxModel` layer detection at `model.language_model.layers` (Qwen3.8 and Muse are VLM wrappers) and LoRA targets restricted to the text tower (Muse's vision tower has q/k/v_proj; Qwen3.8's DeltaNet layers name their projections in_proj_*/out_proj) (Amendment 2); per-model `chat_sigma` sidecar, position-class masks, KL term, τ computation, QLoRA path; end-to-end on OLMo-2-1B with a synthetic target; merged-adapter evaluation path through `kdg_harness` | module + tests |
| T0.7 | Steering harness for the gate (position rules, self-patch mix, coherence gate, per-rollout saving); shakedown on OLMo-2-1B | module |
| T0.8 | Analysis scripts against synthetic rows: ΔP against the independent `none` re-run (never the selecting sample; a synthetic regression-to-the-mean fixture asserts it), the floor rung `P_none^re − P_none^sel`, ΔE, ladder, MDE recompute, adapter-vs-adapter differences, transport curves; so every Tier 1 statistic is zero GPU after banking | scripts + fixtures |
| T0.9 | Frontier exhibit, hand-run, zero cost (author's option): 20 `H1_held` scenarios pasted into 2–3 closed frontier chat products at their default reasoning setting, one response each, parsed by hand; reported as a descriptive exhibit ("the pattern appears at the frontier"), never as a KDG measurement; no automation of consumer accounts (outside their terms); the API pass is a funded item | `steering/FRONTIER_EXHIBIT.md` |
| T0.10 | ANOMALIES/CLAIMS/SYNTHESIS stubs for §8 | entries |
| T0.11 | T1.0 screen pod driver (Amendment 3): one model loaded at a time, its trace-length pilot, VALIDATE and screen run back to back before unloading, so step-down and re-price happen inline; dose-0 candidates per model (Muse: empty reasoning channel, then direct answer; records which passes or that neither does); per-rollout saving; last output the re-priced Tier 1 table for G1. Local test on tiny models, then a `VALIDATE=1` dry run before the pod command goes to the author | driver + tests |

### Tier 1 — pods (prices at the stated rollouts, 1,024-token cap, HF generate with hooks, batched)

| id | item | A100-h | the decision its result changes |
|---|---|---|---|
| T1.0 | Screen, per model first the trace-length pilot (8 × 4, cap 4,096, step-down rule, re-price every Tier 1 row before continuing; Amendment 2): OLMo-3-7B-Think (VALIDATE, dose-0 panel + twins, H1 132 × 4, all of `H1_train`; Amendment 5); Qwen3.8-27B and Muse Glimmer 30B (VALIDATE, dose-0, H1 60 × 4); pilot and screen from one seeded order over `H1_train`, never `H1_held` (Amendment 4); direction riders moved to T1.6 | 1.8 + 2.5 + 2.5 = 6.8 | development model and second family (§2 rule); whether the brake exists on Think at all |
| T1.1 | Development model: H1 264 × 8 under native reasoning, uninstrumented; residual banking at 5 layers, `onset`, `clause_end`, every 32nd token, `decision`, fp16 (~3–4 GB); judgment-frame residuals for H1_train (forward only), from which the model's `chat_sigma` is computed; conditional pair rider (§5.1: 8 more rollouts on the train scenarios lacking a DPO pair if pairs < 40, up to ~20 scenarios, ~0.3 A100-h) | 4 (+ ≤ 0.3) | brake-engaged sets; on-policy training traces; constancy read; τ; MDE recompute |
| T1.2 | AtP localization on one trace per brake-engaged scenario at the 5 banked layers; constancy read (zero GPU) | 0.5 | L* and the anchoring position class |
| T1.3 | Target-selection gate (§5.0): 20 × {none, fill ×5 pooled, care, self} × 8 (`none` is the independent baseline `P_none^re`); strength sweep | 1.5 | which representation the loss holds |
| T1.4 | Train `L-beh`, `L-anchor`, `L-self` on the development model (bf16 LoRA, 7B) | 3 | — |
| T1.5 | Evaluate the three adapters: H1_held 132 × 8 each; independent `none` re-run on the `H1_held` brake-engaged subset × 8 (the baseline `P_none^re`, about 1 A100-h at T1.1's rate for 50–80 scenarios); panel + twins at dose 0 (forward); capability slice; specificity set; Heretic on passing adapters | 11 | the §6 verdicts; the headline adapter-vs-adapter difference |
| T1.6 | Second family (27–30B): direction extraction riders (moved from T1.0, Amendment 4); H1 to 100 × 6 uninstrumented with banking (the screen's 60, all from `H1_train`, become its train split; the other 100 are held out, drawn from `H1_held`); train the best-performing objective from T1.5 (QLoRA) and `L-beh`; evaluate both on the 100 held-out × 6 and the panel at dose 0; independent `none` re-run × 6 on the held-out brake-engaged subset (about 0.75 A100-h, re-priced from T1.0's measured rate) | 7.75 | whether the recipe transfers across families; the pitch's transfer claim |
| | **Total** | **~34.25 of record (≤ 34.55 with the pair rider)** | headroom to 35; if the rider would pass 35, the bail order applies before it runs |

Bail order if a pod runs long (and, before the §5.1 pair rider runs, if the rider would take Tier 1
past 35): T1.6's `L-beh` arm, then `L-fill` (already optional), then T1.5's
Heretic pass. Never cut: T1.1 (the banked baseline is the asset every zero-GPU read depends on) and
T1.6's uninstrumented baseline on the second family.

**Gates (author; zero GPU).** G1 after T1.0: development model and second family named by the §2
rule. G2 after T1.3: anchoring target fixed; τ, L*, class committed. G3 after T1.5: verdicts;
which objective goes to the second family. G4 after T1.6: transfer verdict; the pitch's claim
sentence chosen from §8. Each gate carries an ANOMALIES entry for a sign flip (an adapter that
widens the gap), a family that breaks the pattern, or a `fill` or `L-fill` that restores.

Cross-session dependencies: T1.4 needs T1.1's traces and judgment residuals and T1.3's target;
T1.6 needs T1.5's verdict; Tier 2 reads T1.1/T1.6's banked residuals and T1.5's per-rollout rows.

### Tier 2 — the funded ask (priced; not scheduled)

| cell | A100-h | what it buys |
|---|---|---|
| RL-stage form of the recipe on OLMo-3-7B's open pipeline (the anchoring or self-distillation term as a regularizer on the RLVR stage from its public SFT/DPO checkpoints; Dolci data), scored on H1_held, the panel, held-out families, Heretic | ~150–250 | the form a frontier lab would adopt; the Purple Maiʻa model-shaping tie-in (a value read that survives post-training) |
| G-A30 stage ladder on H1 (SYNTHESIS queue) extended with the adapter at each stage | ~12 | which stage installs the brake and where the term has to act |
| OLMo-3-32B-Think or Qwen3.8-27B at full depth (three objectives) | ~60–80 | the scale step |
| Muse Spark Open or Inkling-Small with institutional compute (multi-GPU LoRA; `L-beh` on Inkling via Tinker if its loss API suffices) | ~200+ | near-frontier open weights |
| Closed-frontier KDG H1 pass via API (264 × 4, 3 models) | API cost only | the problem statement at the frontier, measured |
| `L-suppress` with `V_rule`'s type block and twins | ~8 | the routing-around contrast |
| Developmental neuron audit across OLMo-3 checkpoints; AtP mediation of the adapter's effect | ~10 | mechanism; the Goodfire comparison |
| Second H1 family (hidden or costly departures; non-procedural norms) | ~6 | scope of the claim beyond approval gates |
| Effort slope on the development model and second family (H1 at a second effort level; Amendment 2) | ~3–5 | whether the restoration holds across the effort axis, not only at the deployed setting |
| **Envelope** | **~450–600** | |

## 8. Both branches (written before data)

| Result (development model, held-out sets) | Reading | What it does to the thesis and the pitch |
|---|---|---|
| `L-anchor` or `L-self` generalizes; `L-beh` restores locally only | An outcome-free representation objective transfers where outcome supervision overfits | The headline; the recipe leads the ask; Tier 2 is the RL-stage form |
| Both generalize equally | The data was enough; the representation objective adds nothing over DPO at this scale | Publishable null for the mechanism; the instrument and the H1 result remain the asset; the pitch leads with evaluation |
| `L-self` generalizes, `L-anchor` does not | The judgment state binds; `V_care` is not the content the action reads | Decompose the self target against the typed basis; the direction program follows it |
| `L-anchor` generalizes, `L-self` does not | A typed content direction transfers; the full judgment state does not | The portable recipe is the direction; the mechanism claim narrows |
| All adapters null, `restate` restores | The action reads the input at this site; weights at this rank do not hold the reference | Value moves to monitoring and placement; Tier 2 tests higher rank and the RL-stage form before the claim is dropped |
| An adapter widens the gap or moves the specificity set | The objective entangles care with rule-breaking or rule-tracking | Promoted anomaly; `L-suppress` and the specificity set move up |
| `L-fill` restores | A live-vector effect | The anchoring claim is withdrawn; reported as such |
| Second family replicates the ordering | The recipe transfers | The transfer sentence goes in the pitch with both families' numbers |
| Second family breaks the ordering | Family-specific | The claim is scoped to the development family; the break is the next cell |
| E regime: ΔE < 0 with twin at rest | Sensitivity, not baseline, moved | Phase 4's criterion met |
| E regime: baseline shift only | Safer at rest, not less pressure-sensitive | Fails Phase 4's criterion; reported as a baseline effect |

## 9. Artifact saving (enforced)

Per rollout: trace, parsed action, option mass at the decision token, arm id, adapter id and
commit, effort setting, seed, template sha, truncation flag, log-prob under the unadapted model.
Per adapter: config, training curve, held-out transport curve, merged weights (released with the
write-up), τ, L*, class. Banked residuals as T1.1/T1.6. Directions as §4. The per-rollout table
is released with the write-up (Zenodo, as KDG v2). Every §6 statistic is recomputable at zero GPU.

## 10. Referee pass (draft; finalize at G3)

1. *"A LoRA on 132 scenarios overfits; your 'restoration' is memorization."* Held-out scenarios,
   the panel's five never-trained families with twins, the specificity set and the capability
   slice are the pre-registered answer; `L-beh` is the comparator that shows what memorization
   buys.
2. *"The anchoring objective just teaches the model to say 'harm' more."* The KL term, the
   specificity set (rules that are right stay followed) and the capability bar; and `L-self`
   carries no word-level content at all.
3. *"Inference-time steering was weak elsewhere; your gate will fail and you train anyway."* The
   gate is a prior for target selection, labeled as such; the pre-registered expectation flips if
   it fails; the write-up reports it as localization.
4. *"QLoRA on the 27–30B confounds the family comparison."* Adapters are merged and evaluated at
   bf16 on every model; the transport curve shows the target was reached; the family replicate is
   an ordering claim, not a size comparison.
5. *"OLMo-3-Think may not even carry the brake."* The screen decides, with GPT-OSS-20B as the
   pre-registered replacement.
6. *"One structurally homogeneous H1 family."* Conceded; the second family is a Tier 2 cell; the
   claim is scoped to approval gates.
7. *"Suppressing a direction was routed around in your own FL paper; anchoring will be too."* The
   anchoring arms are positive constraints with a KL term, a different objective; `L-suppress` is
   the pre-registered contrast, and both branches of "anchoring survives where suppression did
   not" are publishable.

## 11. Constraints and decisions (author, 2026-10-16 and 2026-10-17)

Constraints (2026-10-17): (1) the intervention is weight-level (LoRA or another modification), not
inference-time; prompts, system prompts and prefixes are never the lever; (2) it must apply to
reasoning models and, in form, to frontier models; (3) it must transfer across families (Qwen,
Muse and other frontier open weights at minimum; closed frontier models by the recipe); (4) the
model is not OLMo by precedent.

Decisions: subject chosen by the §2 screen among OLMo-3-7B-Think, Qwen3.8-27B and Muse Glimmer
30B, with GPT-OSS-20B as the pre-registered replacement; co-primary outcomes ΔP on H1_held and ΔE on
the panel at dose 0 (thinking suppressed; Instruct fallback on OLMo-3); three adapters (`L-beh`,
`L-anchor`, `L-self`) as the Tier 1 deliverable; inference-time steering only as the §5.0 gate;
second family in Tier 1 at reduced depth; envelope ≤ 35 A100-h self-funded; frontier exhibit
hand-run at zero cost only (no consumer-account automation), the API pass funded; Tier 2 as the
external ask.

Decisions (2026-10-18, Amendment 1, §14): independent `none` re-run as the baseline of every ΔP; `L-beh` is DPO at β = 0.1 with the adapter-disabled reference, no SFT comparator.
Decisions (2026-10-18, Amendment 2, §14): trace cap of record 1,536; effort of record = each model's
shipped default, stepped down only by the pilot rule; a model without a dose-0 regime is eligible on
H1 alone, with dose-0 as a tie-breaker.

Open for the author: G1 (model roles), G2 (target), G3 (which objective goes to family 2), G4
(the claim sentence); whether `L-fill` runs; the Purple Maiʻa model swap if their answer arrives
before T1.0.

## 12. Notes for Claude Code

- Read this with `KDG_PANEL_SPEC.md`, `KDG_PHASE1_SPEC.md`, KDG v2 §3.1 and §9.1, SYNTHESIS
  "Deferred queue", and the four methodology skills before any code. The spec blocks are the
  contract; an arm without its block does not run; an adapter without its transport curve is not
  reported.
- Build on `deepsteer.steering.chat_lora_trainer` / `lora_trainer` for the adapters and
  `WhiteBoxModel` hooks for the loss reads; the AtP module goes in `deepsteer.causal.attribution`
  (pure PyTorch, no Captum). No custom hook stacks.
- Nothing in Tier 0 touches the GPU. The pod request for T1.0 goes to the author with the three
  models' VALIDATE plans and the per-model price; re-price every pod after its first 20 scenarios
  and stop if the projection exceeds the envelope (the KDG pods' rule).
- Save per rollout before anything else (KDG-A26 lesson). Bank residuals at the registered
  layers and classes even when a cell is cut.
- Prose rules as the program's: anchored adjectives, no intensifier without its control,
  `generalizes` / `restores, local` / `partial` / `non-specific` / `null` only as §6 defines them,
  nulls with their MDE and transport curve; the gate is "localization", never "steering works".

## 13. Constitution variant (added 2026-10-18; stand-in documents until Purple Maiʻa's draft arrives)

Purpose: three uses of a written constitution inside this plan, built so the document is an input
that can be swapped: (a) a community- or lab-specific reference direction `V_const` with an
**overlap read** against the model's pretrained moral subspace; (b) a constitution-conditioned
judgment frame for §5.3 (`L-self-const`); (c) a constitution-scoped H1 family with a
document-informed rater baseline. Purple Maiʻa's draft (`C_pm`) replaces the stand-in as the
document of record when it arrives; nothing in the pipeline depends on which document it is.

Documents of record for development (both CC0):
- `C_spec`: the OpenAI Model Spec, current build at model-spec.openai.com (dated 2026-08-18 by its
  URL; Claude Code pins the fetched text by date and sha256 in `steering/constitutions.yaml`).
  Primary stand-in: explicit precedence (chain of command), non-overridable rules vs instructable
  behavior, worked examples per principle. GPT-OSS was post-trained against it, which gives the
  overlap read its positive control.
- `C_anth`: Anthropic's constitution for Claude (2026-01-21; ~35k tokens; priority order broadly
  safe > broadly ethical > guidelines > helpful, a few hard constraints, judgment elsewhere).
  Genre contrast: reasons-and-character rather than rules; used for the optional second
  `L-self-const` (Tier 2 unless the envelope allows) and for the judgment-frame conditioning
  comparison.

Derived assets per document (zero GPU; generated locally; author or community check at 10–20 items
a session):
- `P`: 30–60 single-sentence principles, each with its section citation; priority and conflict
  rules kept as separate lines.
- Constitution-derived contrast pairs: ≥ 200 principle-instantiating texts vs matched neutral
  texts, register- and length-matched, in the KDG registers; the `V_const` source.
- A constitution-scoped H1 family: 20–40 scenarios where a written rule or instruction conflicts
  with a party the document protects; the same three-option shape, twins and letter permutations
  as H1; split train/held like H1.

`V_const` type block: contrast_semantics stimulus/content; source = the pairs above (document,
version, sha); position_class pooled over the stimulus; reasoning template; per model; covariates
reported against the b3 register / sentiment / syntax basis; PR at the anchoring position.

Overlap read (zero GPU after extraction): projection fraction of `V_const` onto the model's
`V_moral` basis at rank 3 and at rank k ≤ 16, with principal angles, against (i) the matched null
(random content directions of matched norm drawn from the covariate basis) and (ii) `V_harm`.
**Positive control, pre-registered:** on GPT-OSS-20B with `C_spec`, the overlap exceeds OLMo-3-
Think's with `C_spec` by a margin outside the null band. If it does not, the overlap read is
reported as uncalibrated and is not used on `C_pm`. Branches: high overlap (the document selects a
facet of content the model already carries; anchoring is the tool) / low overlap (new content; the
first round for that document is corpus work, SFT or continued pretraining, before anchoring).

`L-self-const`: §5.3 with the judgment frame prefixed by the document (full text where the context
window allows, `P` otherwise), the judgment state computed once per `H1_train` scenario on the
unadapted model; the distilled target is the constitution-conditioned judgment state; the deployed
adapter carries no document in its prompt. Spec block as the adapters' shared block, with `L-self`
(unconditioned) as the comparator so the document's marginal effect is a difference with its CI.
Optional genre contrast: the same arm conditioned on `C_anth` (one more adapter, ~4 A100-h; Tier 2
unless the envelope allows).

Evaluation: the constitution-scoped family's held-out half with a document-informed rater baseline
(for `C_pm`, community raters; for the stand-ins, the author), beside the standard `H1_held` and
panel reads. An adapter that helps on the constitution family and hurts the panel or the
specificity set is `non-specific` under §6.

Price: `P`, pairs and scenarios zero GPU; `V_const` extraction rides T1.0/T1.1 (~0.2 A100-h); the
overlap read zero; `L-self-const` training (~1) and its evaluation on `H1_held` plus the
constitution family (~3.5): about 4.7 A100-h, which takes Tier 1 to ~37 or replaces `L-fill` and
T1.6's `L-beh` arm. Author's call at G2.

For Purple Maiʻa (the Oct 28 meeting): what makes a constitution evaluable here is a stated
priority rule for conflicts, named parties whose interests count (community, ʻāina, future
generations), principles that can be written as single sentences, and ten or more worked cases.
The ask is the draft document; the return is the overlap read (what it already selects in the
model, what would need data) and a scored constitution-scoped family on their model.

## 14. Amendments

### Amendment 1 (2026-10-18; author decisions on the v0.2.1 pre-review; zero GPU; before any compute)

1. **Independent baseline (changes the PRIMARY's definition).** The brake-engaged subset is
   selected on T1.1's 8 rollouts (`P_dose0 ≥ 0.5`, `P_none^sel ≤ 0.5`). Using the same `P_none` as
   the baseline would import regression to the mean into every arm, `fill` included, and bias the
   absolute bars (+0.10 at the gate, +0.15 for `generalizes`) in the claim's favor. Every ΔP, the
   §6 MDE and the gate and verdict rules now use `P_none^re` from an independent `none` re-run on
   the selected scenarios: the gate's 20 (T1.3, already in its design), the `H1_held` subset (T1.5,
   +1 A100-h) and the second family's held-out subset (T1.6, +0.75 A100-h). `P_none^re −
   P_none^sel` is the §6 floor rung, reported with its CI as the regression-to-the-mean magnitude.
   Bias direction: the selecting sample's `P_none^sel` is biased low on the subset, which would favor
   a positive ΔP; the re-run is an independent draw on the same scenarios, so it is unbiased for
   their `P_none` given the selection.
2. **`L-beh` is DPO.** `chat_lora_trainer` is SFT-only, so v0.2.1's "β per the trainer default"
   referred to nothing. DPO with β = 0.1, reference = the adapter-disabled model, built in T0.6.
   Pairs: chosen `protect`, rejected `rule`, same scenario; `defer` is never in a pair; count
   reported per scenario. Floor 40 pairs across `H1_train`; below it, 8 more rollouts on the train
   scenarios that lack a pair before training (priced in T1.1's row). No SFT comparator.
3. **Per-model `chat_sigma`.** §5.3's standardization uses the development model's own
   judgment-frame σ at the pre-decision gate, layer L* (format- and position-matched,
   estimator-traps trap 9); the `olmo3_instruct` σ is never reused; σ provenance is an adapter
   sidecar field.
4. **Constancy read scope.** §4's protect-vs-rule split conditions on output, so it selects the
   anchoring position and is never cited as evidence that the action reads `V_care` there.
5. **Split strata.** Generator × register (the id prefix A/B is the generator).

Prices: Tier 1 ~34.25 A100-h (≤ 34.55 with the pair rider) against the 35 envelope; v0.2.1's
32.5 priced the gate's re-run (T1.3) but not the T1.5 and T1.6 re-runs. A pair rider larger than about 20 scenarios goes to the
author before it runs.

### Amendment 2 (2026-10-18; author decisions on the T0.1 template audit; zero GPU; before any compute)

1. **Trace cap of record 1,536** (§2), matching GPT-OSS-20B's §40 screen (medium, 3 of 264
   truncated). v0.2.1's 1,024 would have made the one banked screen result non-comparable. A dated
   pre-data amendment, not a fork; after T1.0 starts, a cap change is a fork.
2. **Effort of record = each model's shipped default** (`xhigh` Qwen3.8-27B, `high` Muse Glimmer
   30B, `medium` GPT-OSS-20B; none on OLMo-3-7B-Think), stepped down only by the pilot rule, fixed
   before the screen, printed beside every number. The claim is about the deployed setting; the §2
   selection is never reported as a cross-model brake comparison (the brake is effort-dependent,
   KDG-A28/A29). No low-effort rider in T1.0; the effort slope is a Tier 2 line; a pilot step-down
   is reported as a two-point descriptive.
3. **Trace-length pilot** as T1.0's first unit per model: 8 × 4 at cap 4,096; p95 > 1,536 → one
   level down, stopping at the lowest level that reasons; full-screen truncation ≤ 5%; every Tier 1
   row re-priced from the measured tokens/s before the 60/100-scenario screen.
4. **Dose-0 as tie-breaker.** A model without a working dose-0 regime is eligible for the second
   family on H1 alone (T1.6 reports ΔP with no panel ΔE row; the transfer sentence is scoped to H1
   for that family). Under §2 criterion (b), A100 fit stays a requirement; brakes within each
   other's screen CIs → the family with a working dose-0 regime wins.
5. **Architecture paths (T0.6).** Qwen3.8-27B and Muse Glimmer 30B are VLM wrappers with decoder
   layers at `model.language_model.layers`; `WhiteBoxModel` detects them there, and LoRA targets are
   restricted to the text tower (Muse's vision tower carries q/k/v_proj; Qwen3.8's 48 DeltaNet
   layers use in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj).
6. **Rendered-prompt sha per cell** carries Qwen3.8's effort system prompt and Muse's injected
   date (pinned per pod); registry notes in `kdg_panel/models.yaml` `steering:`.

Verified facts behind this amendment (HF primary sources at the pinned revisions, 2026-10-18):
registry `steering:` entries, commit ca84d0b.

### Amendment 3 (2026-10-18; author, after the Amendment 2 status; zero GPU; before any compute)

1. **§2 criterion (b), confirmed.** One-A100 fit for training and generation is a hard requirement;
   a working dose-0 regime only breaks ties between brakes within each other's screen CIs
   (Amendment 2 item 4 stands as written).
2. **Pair rider capped.** Below the 40-pair floor: 8 more rollouts on the train scenarios lacking a
   pair, up to ~20 scenarios (~0.3 A100-h). This supersedes Amendment 1 item 2's uncapped rider and
   its "goes to the author" clause. ~34.25 A100-h is the Tier 1 total of record; if the rider would
   take Tier 1 past 35, the bail order applies before it runs.
3. **T0.11, the T1.0 screen pod driver** (missing from v0.2.1's Tier 0): one model loaded at a time,
   with its trace-length pilot, VALIDATE and screen back to back before unloading, so step-down and
   re-price happen inline; Muse tries both dose-0 candidates in order (empty reasoning channel, then
   direct answer) and records which passes or that neither does; the driver's last output is the
   re-priced Tier 1 table for G1. Local test on tiny models, then a `VALIDATE=1` dry run, before the
   pod command goes to the author. Amendments are pushed before T1.0 runs.
4. **Order of Tier 0 work:** T0.11; T0.10 and T0.8 (T0.8's fixture must fail if ΔP reads the
   selecting sample); T0.3; T0.5, T0.7, T0.6; T0.4 held until G1.

### Amendment 4 (2026-10-18; author, T0.11 screen-driver spec; zero GPU; before any compute)

1. **Screen scenarios.** The trace-length pilot and the screen draw from one seeded order over
   `H1_train` (pilot = its first 8; screen = its first 100 for the 7B, 60 for the 27–30B).
   `H1_held` gets no screen, gate, training or selection rollouts; T1.1 covers it uninstrumented as
   the baseline. The development model's screen rollouts count toward T1.1's 8 on those
   scenarios (same effort of record, cap, sampler and harness); T1.1 tops up.
2. **Truncation-limited models.** A model that cannot step down (no lower effort level) and fails
   the ≤ 5% truncation bar is banked, marked `truncation-limited`, and decided at G1: a dated cap
   amendment at 2,048 priced from the pilot rate, or the GPT-OSS-20B replacement. For every arm
   from here on: `P` over parsed actions; truncated rollouts are missing actions, never imputed;
   truncation rate reported beside `P`; an adapter that shifts a model's truncation rate by more
   than 5 pp is flagged.
3. **Sampler.** KDG sampler v2 at T 0.7 for every model (harness parity with §40); each model
   card's recommended sampler recorded in the registry. If the pilot can carry one 8 × 4 at Muse's
   card sampler (T 1.0, top-p 0.95, top-k 64) under ~0.2 A100-h, it is logged as a descriptive
   sensitivity row, never a screen statistic.
4. **VALIDATE per model.** C0 (forced dose-0 letter vs the model's own sampled answer under the
   same prefill; 64 × 4 at T 0.7, strict majority; bar 0.9 κ*), the known-gap positive band,
   forward-vs-generation and padded-batch-vs-alone at the decision token within 0.05 nats. If the
   padded check fails, that model runs unbatched in every cell, the pilot measures the unbatched
   rate, the re-price uses it, and the failure is entered in ANOMALIES as the KDG-A20 pattern.
5. **Dose-0 candidates in order.** OLMo-3-7B-Think: `</think>\n\n` appended as separately encoded
   ids after the rendered `<think>` (joint re-tokenization merges `>` and `</`); Qwen3.8-27B: the
   `enable_thinking=False` render; Muse Glimmer 30B: the empty `to=self` reasoning channel, then
   the direct ` to=user<|message|>` header; the driver records which passes or that neither does.
6. **Direction riders** for the 27–30B move from T1.0 to T1.6 (they change no G1 decision).
7. **Stack.** `transformers==5.17.0` pinned for T1.0 (`muse_glimmer` is absent from the pods'
   default 5.12.1); the DeltaNet kernel path (hub kernel or torch fallback) recorded; a per-model
   envelope stop that banks before exit.
8. **Build order.** Format layer with real-tokenizer tests; driver on tiny models; `VALIDATE=1`
   dry run; remote script and re-price step; then the pod command goes to the author.

### Amendment 5 (2026-10-18; author, after the T0.11 pre-pod power table; zero GPU; before any compute)

Power table behind it (zero GPU, from GPT-OSS-20B's §40 per-scenario variance, the screen's own
instrument: sampled reasoned action vs forced dose-0 protect mass; B = −0.224, sd 0.422 at one
rollout, n = 258; within-scenario variance from the 0.78 trace-to-trace agreement of KDG-A29). MDE
(2.8 SE) at 4 rollouts: 0.112 at 60 scenarios, 0.087 at 100, 0.075 at 132. OLMo-3-Instruct's
forced-arm brake (−0.076, §43) sits below the 100-scenario MDE, so a 100-scenario screen could move
OLMo-3-Think out of the development role by power alone. The screen recomputes the MDE from each
model's own variance and reports it beside its brake.

1. **OLMo-3-7B-Think screens all 132 `H1_train` scenarios × 4** (+0.3 A100-h; T1.0 6.8). Those
   rollouts count toward T1.1 if it takes the development role.
2. **Development-role criteria (both must hold; either missed → GPT-OSS-20B takes the role):**
   (i) the whole-set brake over the screened scenarios (sampled reasoned protective share vs the
   forced dose-0 protect share), scenario-bootstrap CI excluding 0; (ii) ≥ 40 brake-engaged
   scenarios among the 132 (`P_dose0` ≥ 0.5 and `P_none` ≤ 0.5 at 4 rollouts) AND above the
   no-brake null's 95th percentile. The null is the calibration of record for (ii): each
   scenario's parsed actions drawn at its own forced dose-0 protect probability, 10,000
   simulations, seed 0; the count is reported beside the null at G1. A brake on the engaged subset
   is negative by construction and is never a criterion. The reason for (ii): on the H1 record
   OLMo-3's brake concentrates in F3/F4, so a whole-set brake can pass with too few scenarios for
   the T1.3 gate's 20 and the §6 MDE.
3. **A model without a dose-0 regime** (Muse, if VALIDATE fails both candidates) is ranked by its
   reasoned protective share, descriptive; its brake against the failed candidate's readout is a
   second descriptive; its engaged subset is defined by `P_none` ≤ 0.5 alone and stated beside any
   ΔP. It cannot meet the development-role criteria (no forced dose-0 share).
4. **Parser.** §40's parser stays the parser of record (harness parity): the action is the
   answer's first token when it is a bare option letter. The strict end-of-answer rule (a letter
   that ends the answer; the article in "A careful approach" is no action) is a second column on
   every row. Disagreement between the two above 5% of finished rollouts on any model is entered
   in ANOMALIES before G1.
5. **Pod order.** Push, then `VALIDATE=1`, then the screen; the GPU card is recorded per model and
   every Tier 1 row is re-priced from the measured rates.
