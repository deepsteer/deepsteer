# Synthesis — how the refusal decision relates to the moral subspace

Program-level thesis across Directions 1–3 (OLMo-3-7B primary). Updated 2026-07-02 after the D3 rank
sweep resolved the causal verdict (`harm_saturating`). Numbers of record live in each direction's
RESULTS; this file states the throughline and is re-dated on each substantive change.

## Thesis (routing form — resolved for OLMo-3)

The refusal decision **reads the harm percept, not the moral subspace**. It picks up a specific,
low-rank slice of moral content — the harm direction — and writes into a **narrow control-token
channel** at the decision site. The rank sweep is decisive: as the moral basis expands
`k ∈ {1, 3, 8, 16}`, refusal transfer **saturates at the harm-rank-1 level** (`R_refusal` 0.31 → 0.27)
while judgment transfer **climbs** (`R_judgment` 0.46 → 0.66). Adding moral rank beyond harm buys more
judgment coupling and no more refusal coupling. So the geometric orthogonality measured upstream (D1,
D2) is not a weak-instrument artifact and not "moral content is causally absent" — it is that refusal
reads only the harm sliver of a much larger `V_moral`, a sliver nearly orthogonal to the bulk of the
subspace. Refusal and moral judgment are **different reads of the same content**: refusal reads harm,
judgment reads the subspace broadly.

## The three legs

- **D1 (geometry of the direction).** Refusal projects **below the moral-family band at every rung**
  (held-one-out `p(d_src | others)` bands; refusal in-trace peak included). Even the program's highest
  refusal projection is less moral-adjacent than a held-out moral direction. Refusal does **not
  crystallize** from a pretraining precursor (cos 0.155 base→instruct), unlike the moral subspace
  (cos 0.999); it is a fresh post-training construction in a low-variance channel.

- **D2 (geometry of the decision).** The decision site (`final_pre_assistant`) is a **~10–15-dim
  control-token bottleneck** (participation ratio 14.7 / 8.6 / 10.2 on OLMo / Qwen / Llama; band below
  the covariance null → position-invalid for content). Refusal-decision ⊥ judgment-decision at |cos|
  below even the low-dim random level. Content and decision do not coexist at one valid position, so
  content-vs-decision orthogonality is **architecturally guaranteed**, and any coupling must ride the
  **heads that write the bottleneck**. That is the D3 target.

- **D3 (anatomy of the decision, causal).** Refusal is a **distributed write** into that ~13-dim
  channel: led by L16 H23 but needing ~62 heads for 80% of the specificity, plus a 38% MLP share;
  every top writer reads content only weakly `V_moral`-aligned. Interchange patching shows `V_moral`
  is a **specific** refusal substrate (V_moral-restricted moves refusal more than a random rank-3:
  Δ = 0.031, paired 95% CI **[0.020, 0.043]**, excludes 0), but the **rank sweep** shows it is the
  **harm percept specifically**: `R_refusal(k)` saturates at the harm-rank-1 level (0.31 → 0.27 over
  `k = 1..16`) while `R_judgment(k)` climbs to 0.66 — expanding moral rank buys judgment coupling, not
  refusal coupling (`harm_saturating`) — **~73% of refusal's causal twin-difference input lies outside
  the rank-16 moral basis**. Identification: harm-restricted (−0.026) ≈ full V_moral (−0.028), but the
  **harm-partialed** patch (`V_moral ⊥ d_harm`) still moves refusal −0.013 (95% CI excludes 0, about
  half) → harm-dominant **with a resolvable residual non-harm moral read**; `frac(V_moral, d_harm) =
  0.46`. That residual is the **Direction-2 toehold**: it is the one place refusal demonstrably reads
  moral content beyond the harm cue, so whatever moral-judgment↔refusal coupling exists (D2's question)
  has to live there — a rank-2 non-harm sliver, not the broad subspace. The dominant-variance contrast
  component (PC1) is **causally inert** for both readouts (`R(1) ≈ 0`) — variance is not causal
  relevance. Behaviorally, OLMo-3's refusal tracks moral-intent severity only weakly (~17% at max),
  consistent with a harm/surface-keyed gate.

## What is settled vs open

- **Settled (OLMo-3).** The write anatomy (distributed, channel-shaped). `V_moral` is a *specific*
  refusal substrate (Δ CI excludes 0), and the rank sweep resolves *which* moral content: the **harm
  percept**, not the broad subspace (`harm_saturating`). Refusal reads harm; judgment reads the
  subspace broadly — different reads of the same content.
- **Cross-model corroboration (correlational).** GPT-OSS-20B (an independent reasoning MoE) already
  supports the routing reading: its saved in-trace refusal direction (P2) is **harm-loaded** —
  standardized |cos(P2, d_harm)| = 0.49 vs |cos(P2, V_moral ⊥ d_harm)| = 0.13 — which is *why* P2
  projected below the moral-family band in D1. Same mechanism, independent model, independent
  measurement (projection, not patching).
- **Open (cross-model, causal).** Is `harm_saturating` general causally? The comparative is the
  **one-knob model**: on OLMo, `R_refusal(k) = min(harm_ceiling, R_judgment(k))` fits the sweep to
  RMSE 0.036 (one free parameter) — refusal reads the same content as judgment, clipped at the harm
  ceiling. **Llama-3.1-8B: run but underpowered** — clean OLMo-like anatomy and an A1-clean decision
  channel, but the causal cells are below MDE (heterogeneous severity-band twins, variance ~3× OLMo's).
  A directional hint that Llama refusal reads *beyond* harm (`R_refusal` 0.44 at rank 16 vs harm-rank-1
  0.14; one-knob fails) — the "reads content more" direction — is not resolvable without more
  homogeneous operating-band power. Qwen next, then GPT-OSS (Amendment 5). Standardized extraction (A1;
  the dim-788/dim-458 outliers turn out to live at content positions, not the decision channel, which is
  clean and ~13-dim across OLMo and Llama alike — a cross-model strengthening of A2).

## Method spine (portable, promoted to ANOMALIES)

A1 (covariance nulls degenerate in massive-activation families → standardize), A2 (band-below-null ⇒
position-invalid instrument), A3 (reordered-norm architectures overshoot per-head OV attribution ~3×
→ fold the norm). Plus the estimator discipline this program keeps re-learning: **an absolute
"one-clears-MDE-one-doesn't" comparison is the overlap fallacy** — normalize to a within-outcome ratio
and gate on a bootstrap CI, which is what reclassified the D3 headline from a clean claim to the honest
`under_transfer`.
