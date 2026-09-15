# Synthesis — how the refusal decision relates to the moral subspace

Program-level thesis across Directions 1–3 (OLMo-3-7B primary). Updated 2026-07-03: the GPT-OSS commit
axis is RESOLVED — the graded-disengage pod (Amendment 12) shows GPT-OSS refusal is a **reversible
reader** (strong exculpatory deliberation flips violating→comply 6/10; the monotone-projection corroboration was later found non-specific, W4 14.3), the clean
contrast to Llama's early-commitment; the same run banked the position gate (decision channel a 12.8-dim
bottleneck, D2 on a fourth architecture) and engage-consequential deliberation (7/7). Earlier: D3 rank
sweep resolved the OLMo causal verdict (`harm_saturating`); Amendment 11 hardened the Llama reads-broad
verdict against the rank-1 harm-coextensive alternative. Numbers of record live in each direction's
RESULTS; this file states the throughline and is re-dated on each substantive change. (Amendment 13:
the two-axis table's *interpretation* is reframed as a confound-named dimensionality hypothesis, not an
n=3 claim.)

## W4 verdicts (2026-09-13; `d3_decision_anatomy/W4_RESULTS.md`) — what changed

Positive voice first (move 7): **the program now claims, with reliability-certified and replicated
numbers, that refusal reads a low-rank harm slice of moral content on OLMo-3 (pooled n = 42,
`harm_saturating`, CI a third tighter), that the refusal gate is a fresh post-training construction
whose weak precursor (0.155, reliability 0.99 on both sides, 2.2× the matched null) does not
crystallize, that the decision site is a 4–8%-of-reference bottleneck on four architectures with CIs,
and that Qwen, the lineage-independent fourth family, reads beyond the harm rank-1 level and commits
bidirectionally like OLMo.** Per cell:

- **14.1 → Branch A.** Tier 3's estimability counter-reading closes. Tier-3 sentence of record: "a weak
  precursor (0.155, 95% CI [0.147, 0.162]; matched-null q95 0.070; split-half reliability 0.99 on both
  sides; flat 0.139–0.155 across 13 pretraining states) against 0.999 for the moral subspace".
- **14.2 → fork (Amendment 16.3).** Per-component parity failed on a near-degenerate PC3 (VOID by the
  letter); subspace parity passes and the rank-4 harm basis captures no more of Llama's engage-driving
  basis than a sentiment basis (0.207 vs 0.239). Orion picks (a)/(b); Tier 2's Llama "beyond harm" is
  unchanged either way (the rank-1 3.6% of record carries it). **Orion chose (b), 2026-09-13:** Branch A, rank-4 capture below the sentiment control.
- **14.3 → reading, not co-primary.** The graded projection at the post-response decision token is
  monotone (8/8) but not refusal-direction-specific: the refusal direction moves ≈ 1 SD toward comply
  and a covariance-matched random direction moves at least that far in one draw in five, at the
  prefill token and at the decision token alike. Reversibility stays behavioral-primary (5/10 replicated
  vs 6/10; engage 7/7). **Applied (Orion, 2026-09-13):** every "monotone projection" corroboration clause (this file's
  header and Tier 2 bullet, D3-22, FL §8.2/table/App D/E/§10, MN §4/§6) keeps the behavioral leg and
  scopes the projection leg. Band half: the moral band is ABOVE the null at GPT-OSS's decision token
  (0.531 vs 0.482); "position-valid" is scoped to decision-direction reads.
- **14.4 → Branch B on both models.** P1–P3 sit below the covariance null on Think and GPT-OSS (PR 3–10);
  the in-trace rungs stay hedged per model; the trace window is a decision-like position, A2 on a
  fifth kind of site. The absolute PR ≥ 30 control is not evaluable at n = 64 (A9).
- **14.5 → candidate Branch A, held.** Judgment-decision ablation raises refusal +0.12 [0.06, 0.19]
  over random (five random directions move nothing; persona moves 1/100); refusal ablation re-decides
  50/100 prompts in both directions with coherent text (A8: derailment rejected). The Tier-1 causal
  clause is escalated with the instrument scope (single-layer projection-out).
- **14.6 → CIs banked; NI-2 closed; reply inversion Branch B.** Decision sites 14.7 [14.3, 16.2] /
  8.6 [8.2, 9.4] / 10.3 [10.1, 11.1] / 9.4 [9.1, 10.7], each 4–8% of the column-shuffle reference;
  Llama's 10.2-vs-13.5 is one position under two harnesses (W4 in-format: 10.3 raw / 14.2 standardized; D3 harness: 13.5 standardized). On Llama the harm axis flips 0/100
  replies at matched norm while random directions flip 61–83% (A10).
- **15.1 → shape survives.** Pooled n = 42 `harm_saturating`, R_refusal(16) 0.242 [0.134, 0.405] vs
  0.659; the 19 new twins alone `indeterminate` with the same sign; one-knob ceiling 0.246, RMSE 0.023;
  ratio-of-ratios unresolved as predicted.
- **15.2 → `indeterminate`, the empty cell filled.** Qwen reads beyond harm rank-1 (0.54 vs 0.38), gap
  to judgment 0.12 [−0.04, 0.25] at n = 19; harm_saturating excluded at 0.6% of resamples; commits
  bidirectionally (A −0.60). The two-axis table gains a fourth row.

## Thesis (three tiers by evidence scope) — updated 2026-07-05, W4 notes 2026-09-13

The refusal decision reads only a **low-rank slice** of the moral content the model comprehends, and
sits in a **narrow control-token channel** geometrically separate from the broad moral subspace. The
claim decomposes by evidence scope; each tier carries its strongest counter-reading and the experiment
that separates it. The single-OLMo-scope of the prior version under-claimed Tier 1, which is the
strongest thing the program holds.

**Tier 1 — panel-level structure (four families: OLMo-3-7B, Qwen2.5-7B, Llama-3.1-8B, GPT-OSS-20B).**

- **Decision-site bottleneck (four families).** The decision site is a low-dimensional control-token
  channel: participation ratio **14.7 / 8.6 / 10.2 / 12.8** on OLMo / Qwen / Llama / GPT-OSS (range
  8.6–14.7; GPT-OSS at its harmony decision token).
- **Below-band (four models).** Refusal projects **below the moral-family band on every model**; even
  the highest refusal projection is less moral-adjacent than a held-out moral direction (base band-min
  95% CI [0.47, 0.53], refusal under it).
- **Decision orthogonality (three families with extracted decision directions; decisive on OLMo and
  Llama, marginal on Qwen).** Refusal-decision ⊥ judgment-decision: OLMo |cos| 0.10 vs null q95 0.41
  (margin 0.35), Llama 0.08 vs 0.51 (margin 0.48), Qwen 0.32 vs 0.42 (**margin 0.15,
  standardization-dependent** — the cosine and its null both shift under whitening). **GPT-OSS is
  outside this clause**: its causal decision directions are held (correlational-only), so the cell was
  not run there.
- **Causal echo (OLMo, corroborating Tier 1).** The OLMo causal recovery fraction is the causal twin of
  the geometric below-band result: only **~31%** of the refusal interchange effect is recoverable from
  the moral subspace (`R` is normalized to [0,1]; Llama reaches 0.85, so the ceiling is reachable and
  0.31 is genuinely low, not a metric floor).
- *Counter-reading:* the Qwen orthogonality could be a **standardization artifact** — its margin may
  flip under a defensible whitening variant, and Qwen has documented massive-activation pathology.
  *Separating experiment:* report Qwen's decision cosine and its null under both raw and whitened bases;
  if it flips, Qwen drops from decisive to marginal (already the stated form).

**Tier 2 — what the slice is, and how it commits (OLMo-3-causal; family-varying).**

- **OLMo-3 (interchange rank sweep, n=23 request-twins).** Refusal **saturates at the harm-rank-1
  level** (`R_refusal` peaks 0.31 at k=3, holds 0.26–0.27) while judgment **climbs** (`R_judgment` →
  0.66) on the same patches: refusal reads a **harm slice**, not the broad subspace; **73%** of its
  causal input lies off the rank-16 basis (69% already at the rank-3 peak).
- **Llama (interchange at matched depth).** Refusal transfer **0.85 ≈ judgment 0.79** — reads **broad**
  moral content, the dissenting read.
- **GPT-OSS (projection, correlational; interchange held).** Harm-keyed (prompt |cos| 0.977, in-trace
  0.49 vs 0.13) and **reversible** — a graded exculpatory prefill flips **6/10 violating→comply**
  (5/10 on replication); the decision-channel projection moves monotonically with the prefill at both
  the prefill and post-response tokens but not distinguishably from covariance-matched random
  directions (W4 14.3), so the claim is behavioral-only (definition and graded panel: FL §8.2 /
  Amendment 12).
- **Qwen (interchange, standardized, n = 19 operating-band twins; W4 15.2).** Refusal reads **beyond the
  harm rank-1 level** (R_refusal(16) 0.54 [0.42, 0.69] vs harm rank-1 0.38) and its gap to judgment
  (0.12 [−0.04, 0.25]) is not resolved at n = 19: `indeterminate` between OLMo's plateau and Llama's
  gap-close, with `harm_saturating` excluded (0.6% of resamples). Commit: bidirectionally coherent
  (A −0.60), like OLMo. The empty cell is filled; the reading is stated at its anchored strength.
- *Method note (the confound is confined to GPT-OSS):* the OLMo (harm) and Llama (broad) reads **both
  use interchange**, so their difference is **family, not method**; only GPT-OSS's read is
  method-distinct (correlational projection).
- *Counter-reading:* **method variance masquerading as family variance.** *Separating experiment:* run
  the interchange rank sweep on GPT-OSS (Tier-2 C1-MoE, held) so all reads share the method. Partial
  bridge already in hand: OLMo and Llama share the method and still differ.

**Tier 3 — fresh construction / doesn't crystallize (OLMo-3-only; checkpoint-based).**

Refusal does **not crystallize** from a pretraining precursor (proto-refusal→gate cosine **0.155**)
while the moral subspace does (checkpoint-to-final **0.869 → 0.999**); refusal is a fresh post-training
construction in a low-variance channel.

- *Counter-reading:* **estimability floor.** The 0.999 is a valid same-pipeline positive control for
  detecting continuity, but the two constructs differ in checkpoint-estimability — moral content is
  abundant in pretraining, refusal behavior is scarce, so proto-refusal is plausibly the noisier
  estimate and a low 0.155 could be attenuation, not genuine discontinuity. *Separating experiment — RUN (W4 14.1, Branch A, 2026-09-13):* split-half reliability 0.991 (proto)
  and 0.997 (gate), adjacent-checkpoint 1.0, disattenuated cosine 0.156, prompt-bootstrap CI
  [0.147, 0.162]; the counter-reading is closed, the residual is the cross-format construct difference.
  *(Historical text follows.)* a
  split-half (resample the refusal contrast, recompute proto-refusal, self-cosine) or adjacent-checkpoint
  self-cosine puts a reliability ceiling under 0.155 (~0.9 → fresh-construction solid; ~0.3 → mostly
  attenuation floor). **Was not zero-GPU with the D1 saves** (`refusal_base.npz` stores only the final
  4096-d direction; the crystallization trajectory carries a single flat 0.155, no per-checkpoint
  proto-refusal), so it needs re-extraction on the base checkpoint — a pod. **FL ships this as a stated
  limitation until the control runs.**

## The three legs

- **D1 (geometry of the direction).** Refusal projects **below the moral-family band at every rung**
  (held-one-out `p(d_src | others)` bands; refusal in-trace peak included). Even the program's highest
  refusal projection is less moral-adjacent than a held-out moral direction. Refusal does **not
  crystallize** from a pretraining precursor (cos 0.155 base→instruct), unlike the moral subspace
  (cos 0.999); it is a fresh post-training construction in a low-variance channel.

- **D2 (geometry of the decision).** The decision site (`final_pre_assistant`) is a **~9–15-dim
  control-token bottleneck** across **four families** (participation ratio 14.7 / 8.6 / 10.2 / 12.8 on
  OLMo / Qwen / Llama / GPT-OSS-20B, the last a reasoning MoE at its harmony decision token; band below
  the covariance null → position-invalid for content). Refusal-decision ⊥ judgment-decision at |cos|
  below even the low-dim random level. Content and decision do not coexist at one valid position, so
  content-vs-decision orthogonality is **structurally favored** (the site is a learned control-token
  bottleneck, so what routes through it is trainable, not fixed), and any coupling must ride the
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
  channel is a **12.8-dim bottleneck**, so the decision site is the D2 low-dim control channel on a
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
→ fold the norm), A7 (coarse-grid critical-noise σ* manufactures a spurious sign-flip under naive
RMS-normalization — a censoring artifact; use a censoring-free/analytic σ*). Plus the estimator
discipline this program keeps re-learning: **an absolute "one-clears-MDE-one-doesn't" comparison is
the overlap fallacy** — normalize to a within-outcome ratio and gate on a bootstrap CI, which is what
reclassified the D3 headline from a clean claim to the honest `under_transfer`.

**Adversarial-review sweep (2026-07-05, DUO + methods notes).** Nine-paper hostile-review pass +
two zero-GPU-plus-local-MPS confirmations. Standing-claim-relevant results: (i) **A7** above resolves
the Paper-1 §4.3 scale-artifact objection in the paper's favor — declarative most fragile *survives*
a censoring-free scale-matched estimator at the exact 1000-step cell (SNR Δ = 0.44, 95% CI [0.01, 0.91]
excludes 0; raw overstates ~3×), so §4.3 keeps its ordering with a scale-sensitivity note, not an
erratum. (ii) **Paper 3 "integration" is now calibrated**, not asserted: a matched-twin non-moral
control run through the identical pipeline gives mean pairwise cosine 0.013 vs moral 0.26 (Δ = 0.22,
95% CI [0.20, 0.24] excludes 0), so the shared component is moral-specific relative to a matched
non-moral battery (affective-vs-moral is the named residual). FL/MN were scoped to their evidence
(FL thesis made consistent with its already-scoped abstract; MN norm-fold prior art cited and verified);
no refusal-program thesis change.

**Packaging (2026-08-25, arXiv finalization — no standing claim changed).** Titles and abstracts
finalized for the P1/P2/P3/MN arXiv set. P2 retitled *Output Dilution: Redundant but Fragile
Representations in MoE Models*; its abstract's "five-fold" corrected to the body's 4.2-fold
(σ* 0.92 vs 3.81; the five-fold appeared nowhere in the paper). P3 retitled *How Language Models
Organize and Structure Moral Knowledge*; its abstract keeps the calibrated-integration scope
(0.26 vs matched non-moral 0.013), states the eff-dim 5 as the six-direction ceiling (rules out
collapse, does not mark integration), and carries the MFT-grouping null's detection bar
(20 partitions, smallest achievable p = 0.05); "loaded in balance" softened to the body's
"near-balanced loading". MN abstract keeps the six-mode count. (Same-day revision: P3 abstract rewritten to the
author's leaner framing; the same hedges now ride as five in-line anchors — integration attributed
to the shared component, 0.26 vs 0.013 pair, "no evidence" + 20-partition bar, integration-regime
wording, 2.7x compositionality anchor — with the eff-dim-ceiling explanation, difference-CI, and
affective-salience residual stated in the body rather than the abstract.) Author-approved abstract trims:
P3 drops MFV replication + foundation-uniform fragility from the abstract (both remain in body);
MN drops the fresh-context-re-read disclosure (remains in §6). Companion bibs now cite P1 at
arXiv:2606.11375. arXiv Makefile targets fixed to ship main.bbl, in-tarball graphicspath, and
(P3) outputs/figures; all four tarballs compile standalone with zero missing figures/citations.

**Publication (2026-08-27).** The arXiv set is live: P1 v2 (arXiv:2606.11375, adds the §4.4
RMS-normalization control and the scoped abstract), P2 (arXiv:2608.25231), P3 (arXiv:2608.27402).
Ids back-filled across companion bibs (P3/P4/P5/P6/FL). MN is upload-ready and held to submit
paired with FL.

## What the next result changes (W4 venue-quality pod, pre-registered 2026-09-10)

Amendments 14/15 (`d3_decision_anatomy/PREREGISTRATION.md`) are committed before any array is
extracted. Per cell, the branch → the edit this file takes:

| cell | branch | thesis edit |
|---|---|---|
| 14.1 proto-refusal reliability | `rel_proto ≥ 0.9` | Tier 3 counter-reading closes; "fresh post-training construction" stands |
| | `rel_proto ≤ 0.3` | Tier 3 rescopes to "low base→instruct cosine, reliability-limited"; the abstract drops "almost no pretraining precursor" |
| | between | the disattenuated cosine (with CI) replaces 0.155 in the Tier 3 sentence |
| 14.2 Llama rank-2/4 harm capture | capture ≤ 0.25 over null | Tier 2 Llama "reads broad" keeps "beyond harm" |
| | capture ≥ 0.50 | Llama becomes "broader than OLMo's rank-1 harm, not established as beyond harm"; the dimensionality rival's Llama point is re-typed |
| 14.3 GPT-OSS post-response decision token | monotone at `P_dec` | reversible-reader gains a co-primary projection read; the last-token caveat is deleted |
| | not monotone | reversibility stays behavioral-primary; "deliberation writes at the prefill site, not the decision token" is added as a finding |
| 14.4 P0–P3 PR audit | P2 PR-valid on both | D1 reasoning-band statements drop the cross-position hedge |
| | P2 fails | the hedge stays per model; null-relative claims unchanged |
| 14.5 reconciled cross-ablation | an arrow's Δ-CI excludes 0 | Tier 1 orthogonality gains a causal cross-arrow sentence (direction named) |
| | neither | Tier 1 stays geometric with the bar "no cross-effect detectable at Δ ≳ 0.14" |
| 15.1 OLMo twins to n ≈ 40 | `harm_saturating` replicates, alone agrees | Tier 2 OLMo headline unchanged, CI a third tighter |
| | shape changes | one-knob reported as "fitted on 23, not replicated on N"; Tier 2 OLMo sentence rewritten to the pooled verdict |
| 15.2 Qwen C1 read | any of four branches | Tier 2 gains a lineage-independent fourth read; "Qwen not measured" is retired; A13's dimensionality hypothesis gets its first off-confound point |

**Zero-GPU arm of 14.1, run 2026-09-10 after the amendment commit (CLAIMS W4-01/02).** Paper 5's
per-checkpoint proto-refusal caches reproduce D1's `refusal_base.npz` at cosine 0.99999998 (positive
control); adjacent checkpoints (stage3-step11900 vs 11921) agree at 0.9999999; proto-refusal itself
crystallizes 0.93 (step 1000) → 1.0 across the anneal while the proto→gate cosine stays flat at
0.139–0.155 on all 13 states, against a covariance-matched single-direction null q95 of 0.070
(descriptive rung). Drift arm reads Branch A; the split-half (prompt-sampling) arm is the pod's,
and the Tier 3 verdict waits for it. Standing edit already licensed by the rung: "almost no
pretraining precursor" is an unanchored adjective for 0.155 ≈ 2× the matched null; the W4-3
rewrite is "a weak precursor (0.155; matched-null q95 0.07) against 0.999 for the moral subspace".

## What the next result changes (KDG pilot, pre-registered 2026-09-13; `papers/KDG_PANEL_SPEC.md` v0.4)

The execution program generalizes the decision variable from refusal to action selection. The
knowing–doing gap (KDG) is prior art (Huang et al. 2026; Shen et al. 2025; Rakshit et al. 2026;
`papers/kdg_panel/LIT_PASS.md`); the program's delta is the self-referenced per-scenario moral
gap under typed pressure families with a harm vs non-harm contrast, and the base/instruct
three-cell raw-frame comparison. Nothing downstream (action-position rank, persona lever,
widening) is scheduled until the pilot gate (§5) passes. Per branch, the edit this file takes:

| cell | branch | thesis edit |
|---|---|---|
| pilot gate (48 gate-family primaries, OLMo-3-Instruct, dose-0) | ≥ 14 pass, ≥ 2 families | the execution program opens; KDG becomes the outcome variable the action-position read is scored against |
| | fail once | pressures revised (construction, not scoring); re-pilot; no thesis edit |
| | fail twice | Branch B2 (panel misdesigned) or, with stable consistent judgments and `no_pressure` everywhere, B1: "current open 7–8B instruct models act consistently with their stated judgment on every family (no gap detectable above the ladder's bar)"; the shallow-read claim stays refusal-specific |
| F5-vs-rest (harm-stratified, both generators agreeing) | KDG lower on F5 | Branch A: the action channel, like refusal, reads harm; FL's finding generalizes from refusal to action |
| | no F5 difference, gap present | Branch C: the action read is not the harm sliver; anatomy before framing |
| three-cell base/instruct (raw frame, mass floor 0.5) | gap in base ≈ gap in instruct D_raw | the knowing–doing structure is inherited from pretraining; persona is not the origin |
| | no gap in base, gap in instruct D_raw | post-training installed it in the weights; strongest motivation for persona-as-lever |
| | gap only in D_chat | the template/assistant role carries the gap; persona is the coupling, steering it is the first intervention |
| generator split | any family result reverses across generator | anomaly by rule; the family is reported per generator and dropped from the pooled verdict |
| cross-model D agreement (Tier 2, after the gate) | ≈ 1.0 (Huang et al. rival) | the model axis collapses; per-family structure is the only live quantity and the cross-model table is reported as agreement, not as a contrast |

Positive voice, pre-data (move 7): *the program now has a behavioral target for execution, built
on the same models and positions where it holds a typed moral subspace and a refusal-read result,
so that the later action-position rank measurement inherits a calibrated outcome variable.*

### KDG pilot outcome (2026-09-14; `papers/kdg_panel/KDG_RESULTS.md`) — what changed

Positive voice first: **the panel instrument works on OLMo-3-7B-Instruct (parse rate 1.000 on
10,656 replies; known-gap band 0.63 sits 0.44 [0.14, 0.62] above the measurement) and the pilot
gate passes under the pre-registered rule (19/48, all four gate families) and under fork B
(26/48); the execution program is open.** What the pilot did not settle, and why it is a reading:

- **Gap above the pressure-removed null.** Primary rule: KDG 0.22 [0.08, 0.42], paired excess
  over the matched null 0.15 [0.00, 0.35] (n = 20), lower bound at 0.00 → not established.
  Fork B (A12): 0.27 [0.14, 0.45], excess 0.17 [0.03, 0.33] → excludes 0. No verdict rests on
  fork B alone.
- **Rival reading strengthened, not separated.** The reference itself flips under paraphrase on
  31% of scenarios (KDG-A1); on the paraphrase-stable subset KDG is 0.11 [0.00, 0.35]. The
  thesis sentence for execution therefore stays conditional: *if the model's stated judgment
  is decisive, it acts against it on roughly one screened scenario in five by majority and two
  rollouts in five; whether that survives a paraphrase-robust reference is the full panel's
  first question.*
- **Structure.** Branch C shape at pilot power (family MDE 0.51): F5 is not lowest, F3 (tool
  shortcut) is the high candidate and F4 (allocation) the low one (KDG-A3). No thesis edit
  until the full panel.
- **Three-cell.** Raw-frame gap present in base (0.12) and instruct (0.07), 0.10 vs 0.08 on
  the 50 shared scenarios → the *inherited from pretraining* sub-branch as a reading; the
  instruct model clears the raw-frame mass floor on only 58/96 (KDG-A2).
- **F2 stays an appendix family** (11/12 no pressure).

Thesis edit under each pending discriminator: KDG-A1 R_a → the action-position rank cell (S1)
is scheduled with KDG as its outcome variable; KDG-A1 R_b → the panel is rebuilt on
decisively-judged scenarios before any anatomy, and the execution claim is Branch B1 wording.

### KDG full panel (2026-09-15; `papers/kdg_panel/KDG_RESULTS.md` §10) — what changed

Positive voice first: **on 200 pre-registered scenarios OLMo-3-7B-Instruct acts against its own
stated moral judgment on one screened scenario in five (0.20 [0.14, 0.31]) and two rollouts in
five (0.39); the excess over the pressure-removed null, 0.10 [0.01, 0.20], survives a four-frame
paraphrase-robust reference (A13 L1: 0.21 [0.15, 0.32], excess 0.11 [0.01, 0.21]); the known-gap
band sits 0.37 above it; and the raw-frame gap is at least as large in the base model (0.14)
as in the instruct model (0.09) on the same scenarios.** The execution program has its
outcome variable, measured against its own noise floor.

What did not hold, and what it changes:
- **No family structure (Branch C).** F1 0.21, F3 0.17, F4 0.21, F5 0.23; F5 is not lowest, so
  the harm-keyed prediction of Branch A (the action channel reads what refusal reads) is not
  supported at a family-contrast MDE of ~0.40. Thesis edit: the action channel's read is not
  established as the harm sliver; the S1 action-position rank cell is now the discriminator
  between "reads harm" and "reads something broader", with KDG as its outcome variable.
- **F4 is generator-dependent** (KDG-A4: 0.00 vs 0.38, CI-separated). F4 leaves the pooled
  verdict; pooled KDG without F4 = 0.20 [0.13, 0.32] (n 71); its paired excess over the null keeps the same point estimate, 0.10, with a CI that now reaches −0.00 at n = 60 (paired), a power effect, not a change in the effect.
- **Full gate not met by the letter** (55 screened vs 60; F4 reversal). Tier 2 waits on the
  author's choice between more scenarios and an F4 rebuild (KDG_RESULTS §10.7).
- **Inherited, not installed** (three-cell sub-branch, reading): base raw gap 0.136 vs instruct
  0.089 on 169 shared scenarios; the assistant template adds little (chat 0.086 vs raw 0.070).
  Persona is not the origin of the gap; it may still be the lever.
- **KDG-A1 closed at L1, open at L2**: the strictest reference (all four frames agree) reads
  0.15 with an excess whose lower bound is 0.00 at n = 34; under-powered, and the one place the
  reference-noise rival still lives.

Thesis sentence for execution, in positive voice, as of this gate: *a 7B instruct model carries
a measurable, paraphrase-robust knowing–doing gap that pretraining already installs, that
post-training does not remove, and that is not organized by the harm content refusal reads.*

### KDG round 2 + F4 swap (2026-09-15; `papers/kdg_panel/KDG_RESULTS.md` §11) — what changed

Positive voice first: **the gap replicates on 248 primaries at the same size (0.19 [0.13, 0.28];
excess over the pressure-removed null 0.10 [0.02, 0.18], n 100), the non-F4 panel meets every
clause of the full gate with an excess of 0.16 [0.02, 0.28], and the inherited-from-pretraining
three-cell reading is stable across three pods (base 0.145 vs instruct 0.094 on 235 shared).**

What the larger n sharpened:
- **The strictest reference is now the verdict level, and it does not resolve the gap.** At L2
  (all four judgment frames agree, n 43 paired) the excess is 0.05 [−0.02, 0.19]; at L1 it is
  0.11 [0.02, 0.19]. Thesis sentence for execution therefore carries the level: *against a
  paraphrase-majority judgment the model acts against its own stated judgment on one screened
  scenario in five with an excess over the pressure-removed null that excludes 0; against an
  all-frames-agree judgment the excess is smaller and not yet resolved.* The next result that
  changes this is n at L2.
- **F4 swap inconclusive** (4–11 defined per cell); F4 stays per generator and out of the pool;
  KDG-A4 open with a priced next leg.
- **Gate decision KDG-G3 (author):** non-F4 panel of record (gate met) vs four-family panel
  (gate not met on F4's reversal alone).
- Structure stays flat (Branch C); provider-pooled rates 0.15 (Claude-written) vs 0.23
  (GPT-written), same sign.
