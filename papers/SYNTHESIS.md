# Synthesis — how the refusal decision relates to the moral subspace

Program-level thesis across Directions 1–3 (OLMo-3-7B primary). Updated 2026-07-03: the GPT-OSS commit
axis is RESOLVED — the graded-disengage pod (Amendment 12) shows GPT-OSS refusal is a **reversible
reader** (strong exculpatory deliberation flips violating→comply 6/10 + monotone projection), the clean
contrast to Llama's early-commitment; the same run banked the position gate (decision channel a 12.79-dim
bottleneck, D2 on a fourth architecture) and engage-consequential deliberation (7/7). Earlier: D3 rank
sweep resolved the OLMo causal verdict (`harm_saturating`); Amendment 11 hardened the Llama reads-broad
verdict against the rank-1 harm-coextensive alternative. Numbers of record live in each direction's
RESULTS; this file states the throughline and is re-dated on each substantive change. (Amendment 13:
the two-axis table's *interpretation* is reframed as a confound-named dimensionality hypothesis, not an
n=3 claim.)

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

- **D2 (geometry of the decision).** The decision site (`final_pre_assistant`) is a **~9–15-dim
  control-token bottleneck** across **four families** (participation ratio 14.7 / 8.6 / 10.2 / 12.79 on
  OLMo / Qwen / Llama / GPT-OSS-20B, the last a reasoning MoE at its harmony decision token; band below
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
  supports the routing reading: its refusal direction is **harm-loaded at both positions it carries
  signal** — at the **prompt** (P0, `t_inst`) standardized |cos(refusal, d_harm)| = **0.977** vs
  |cos(refusal, V_moral ⊥ d_harm)| = **0.001** (near-purely harm), and **in-trace** (P2) 0.49 vs 0.13 —
  a **prompt→trace consistent** harm read, which is *why* P2 projected below the moral-family band in D1.
  Same mechanism, independent model, independent measurement (projection, not patching).
  (`gpt_oss_harm_audit.py`.) The **Tier-1 session has run** (A100-80GB) and banks three results
  independent of the disengage resolution: (i) the **position gate PASSES** — GPT-OSS's harmony decision
  channel is a **12.79-dim bottleneck**, so the decision site is the D2 low-dim control channel on a
  fourth architecture (a 20B reasoning MoE), and the projection reads are licensed; (ii) **deliberation
  is consequential** — an inculpating prefill flips benign→refuse 7/7 (Wilson [0.65, 1.0]), so the
  decision is *not* fixed before the trace; (iii) the decision-channel **null-ratio corroborates
  harm-keying** at the reflexive site. The **commit-axis verdict is now RESOLVED (Amendment 12): GPT-OSS
  refusal is a reversible reader.** The first run's disengage 0/7 was the A7 saturation trap (step-function
  gate, no boundary band); the graded exculpatory-prefill series de-confounds it — strong exculpatory
  deliberation flips ceiling-refusing violating→comply **6/10** and moves the decision-channel projection
  monotonically toward comply in all 10 items. So GPT-OSS reverses in both directions (benign→refuse and
  violating→comply), the affirmative answer to deliberative reversibility and the clean contrast to
  Llama's early-commitment.
- **Settled (cross-model, depth-verified).** Refusal's read and commitment **differ by model family**,
  on two axes confirmed at depth-matched layer 12: *what* it reads — OLMo & GPT-OSS read **harm**
  (`R_refusal < R_judgment`, saturates), Llama reads **broad moral content** (`R_refusal ≈ R_judgment`,
  gap closes); *how* it commits — OLMo at/after the read layer, Llama **early** (disengage depth-gated).
  Llama's early-commitment of a broad moral read is the candidate mechanism for its Paper-6 robustness
  anomaly. The naive layer-16 `A`-asymmetry (+0.82) was mostly a read-layer artifact (→ −0.28 at matched
  depth); the discipline that caught it is the depth-indexed-verdict pattern (methods note).
- **Held (panel breadth).** The two-axis result is n = 3 (OLMo causal, GPT-OSS correlational, Llama
  causal). Qwen extends the instruct panel; the OLMo `harm_saturating` one-knob is the flagship anchor:
  on OLMo, `R_refusal(k) = min(harm_ceiling, R_judgment(k))` fits the sweep to
  RMSE 0.036 (one free parameter) — refusal reads the same content as judgment, clipped at the harm
  ceiling. **Llama-3.1-8B: diagnosed, not yet resolved.** Clean OLMo-like anatomy and an A1-clean
  decision channel, but the refusal cells came back chaotic. Amendment 7/8 traced it to two things:
  saturation (fixed by boundary-band twins), and — the real mechanism — **hysteresis**. The
  bidirectional cell shows Llama's refusal **latches**: it engages coherently when harmful content is
  *added* (+0.14, CI excludes 0) but is unmoved when harm is *removed* (−0.01, incoherent). The
  cross-model asymmetry is **statistically resolved**: `A_Llama = +0.82` vs `A_OLMo = −0.20` (OLMo
  bidirectionally responsive — both directions coherent), `A_Llama − A_OLMo = 1.03, 95% CI [0.16, 1.61],
  excludes 0` at the read layer. **Both dimensions are depth-verified (Amendment 10, matched at layer
  12).** *Commitment:* the patch-layer sweep names the mechanism — Llama's disengage is **coherent below
  ~layer 15 (−0.57 at 12) but not at the read layer (−0.01 at 16)**, so refusal is **early-commitment**
  (crystallizes before the decision site), not a hard latch; OLMo's disengage works at the read layer, so
  OLMo commits later. The layer-16 `A`-asymmetry (+0.82) was mostly a **read-layer artifact** — at
  matched depth `A_Llama = −0.28` and `A_Llama − A_OLMo` shrinks from +1.03 to +0.26 — so the asymmetry
  is a *consequence* of early-commitment, not a third property. *Reads:* it survives depth-matching —
  at layer 12 Llama's refusal reads **as broadly as judgment** (`R_refusal 0.85 ≈ R_judgment 0.79`, gap
  closes → `broad_moral`), while OLMo stays **harm-keyed** (`R_refusal 0.43 < R_judgment 0.53`). The
  reads-broad verdict survives the **harm-coextensive** alternative at rank 1: a single harm cue spans
  only **3.6%** of the engage-driving moral basis (the transfer grows into moral directions the harm
  axis does not point along); the rank-2/4 severity-harm basis is a stated extraction rider, prior
  against coextensivity. So the two-dimensional table is final: *what* refusal reads (OLMo/GPT-OSS harm;
  Llama broad moral) × *how* it commits (OLMo at/after the read layer; Llama early). Llama's early-commitment of a broad moral read is
  a strong candidate for its Paper-6 robustness anomaly. **GPT-OSS is placed on the commit axis: a
  reversible reader** (graded deliberation flips it both ways). The measured table stands; its
  *interpretation* is a follow-on hypothesis, not an n=3 claim (Amendment 13): the read↔commit pairing is
  **architecture-confounded** (lineage/scale/tokenizer/reasoning-vs-instruct), deconfounded only by
  varying one axis at a time (a deliberation-trained OLMo variant; Qwen as a lineage-independent point).
  The sharpened rival is **dimensionality → reversibility** — refusal-read effective rank (OLMo/GPT-OSS
  ~rank-1 harm; Llama ~rank-8 broad) is ordinally consistent with reversibility across all three points,
  which *licenses* "dimensionality of the refusal read → reversibility" as the falsifiable follow-on
  hypothesis (superseding the categorical co-occurrence) but cannot confirm it at three confounded
  points. Qwen held. (Standardized extraction, A1: the dim-788/dim-458
  outliers live at content positions, not the decision channel, which is clean and ~9–15-dim across OLMo,
  Llama, and GPT-OSS alike — a cross-model strengthening of A2, ledgers A5/A6.)

## Method spine (portable, promoted to ANOMALIES)

A1 (covariance nulls degenerate in massive-activation families → standardize), A2 (band-below-null ⇒
position-invalid instrument), A3 (reordered-norm architectures overshoot per-head OV attribution ~3×
→ fold the norm). Plus the estimator discipline this program keeps re-learning: **an absolute
"one-clears-MDE-one-doesn't" comparison is the overlap fallacy** — normalize to a within-outcome ratio
and gate on a bootstrap CI, which is what reclassified the D3 headline from a clean claim to the honest
`under_transfer`.
