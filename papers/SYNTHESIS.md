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

**Latest (2026-09-26):** the execution program's three-reads thesis, the instrument-first pitch claim, and the pitch branch table are at the end of this file ("Execution program: three reads, pitch branch table, novelty gate").

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

### A15 continuous readout (2026-09-15; `papers/kdg_panel/KDG_RESULTS.md` §12) — what changed

Positive voice first: **on the log-prob readout, pre-registered as a secondary instrument with a
coherence gate it passed, the gap exceeds the pressure-removed null at every reference level
including the strictest (L2: 0.079 [0.029, 0.128], n 67), and the excess grows with strictness
where the majority readout lost power.** The reference-noise rival (KDG-A1) is separated. Thesis
sentence for execution, both instruments named: *a 7B instruct model puts more probability on
the action it judges wrong when acting than when judging, by 0.05–0.08 above a pressure-removed
null, against judgment references of any strictness; by majority vote the same gap is 0.10 above
the null against a paraphrase-majority reference and unresolved against an all-frames-agree
reference at 43 pairs.* F4's generator effect is graded (0.13 vs 0.22), not a reversal. Nothing
here changes the gate decision KDG-G3, which stays with the author.

### Gate KDG-G3 (2026-09-19): A16 adopted; four-family panel of record

The reversal clause is evaluated on both instruments (dated amendment, both-choice verdicts in
KDG_RESULTS §12.6). The four-family full gate is MET; F4 rides as a per-generator covariate.
Tier 2 and the dose arm are licensed by the gate but not funded (compute and API budgets
exhausted 2026-09-15); the next deliverable is the panel paper from the committed arrays.

### A17 three-cell contrast with CIs (2026-09-19; `papers/kdg_panel/KDG_RESULTS.md` §13) — what changed

Positive voice first: **in the raw completion frame the base model already acts against its own
judgment by a pressure-attributable margin (E_base 0.017 [0.012, 0.022], n 354), and post-training
does not remove it: on the 192 scenarios both models engage the instruct model's excess is 0.046
[0.025, 0.069] against base's 0.018, a paired difference of 0.028 [0.007, 0.049] that sits on the
acting side (Δ 0.037 [0.012, 0.062]; the judging side does not differ). Post-training also reverses
the no-pressure frame gap (+0.024 → −0.038), so the net gap under pressure is smaller after
post-training (0.043 → 0.011 on shared scenarios) because the baseline moved, not because pressure
moves the aligned model less.** Verdict by the A17 rule: *widened*; binary readout under-powered,
same sign.

Blast radius (move 3): the 2026-09-15 reading "inherited from pretraining and not larger after
post-training" rested on argmax rates without a null; it is superseded. Inherited stands (present in
base); "not larger" is replaced by "larger pressure-attributable part, lower baseline". CLAIMS
KDG-16/22 keep their numbers as argmax readings and lose the interpretation sentence; KDG-29..34
replace it. The paper's title and §7 follow (`papers/kdg_judgment_action/KDG_GATES.md`).

Thesis sentence for execution, revised: *a 7B instruct model carries a measurable,
paraphrase-robust judgment–action gap; its pressure-attributable part is already present in the
base model's raw frame and is larger, not smaller, after post-training, on the acting side, while
post-training lowers the model's default willingness to take the violating action; the gap is not
organized by the harm content refusal reads (binary family MDE 0.40; one exploratory continuous
contrast, F1 > F3, separates at the chance-adjusted level of a candidate).*

Rival readings carried, each with its separating cell: (i) goal-following, not a moral read, for the
widened acting-side sensitivity → the pre-registered deliberation-dose arm with the filler control
(~90 min A100); (ii) a raw-frame artifact for the instruct model's negative no-pressure gap → a
letter-only chat-template judgment on the pressure-removed twins (~5 min; KDG-A6); (iii) a
prompt-version effect for the widening → a version-stratified generation round (generation only,
~1 h GPU; KDG-A5 shares the pod). What the next result changes: (i) closes → "post-training
installs goal-following that overrides the model's own judgment" is the mechanism sentence; (ii)
closes as artifact → the baseline-shift half of the sentence is dropped and the widening stands on
E alone; (iii) closes as prompt effect → the widening is scoped to the later-written construction.

## Execution program: three reads, pitch branch table, novelty gate (2026-09-26)

Sources: `KDG_PITCH_PLAN.md` (export of the pitch doc, Phase 0 item 3), `INCIDENT_MAP.md`,
`LIT_PASS_P9.md` (incl. §8 full reads), `ANOMALIES.md` KDG-A6 update of 2026-09-25. No new model
data; this section restates the thesis and records the gate decisions.

### Thesis, positive voice (OLMo-3-7B scope)

**One model carries three different reads of its moral content.** Its *judgment* reads the moral
subspace broadly: patching recovers more judgment coupling as moral rank grows (`R_judgment(k)`
→ 0.66 at k = 16). Its *refusal* reads a harm slice: refusal coupling saturates at the harm rank-1
level (`R_refusal` 0.31 → 0.27, `harm_saturating`, pooled n = 42). Its *action* departs from its own
judgment under pressure by a margin that is present before alignment (raw-frame excess E_base 0.017
[0.012, 0.022]) and larger after it (shared-192 E 0.046 [0.025, 0.069] vs 0.018; the incentive moves
acting mass 0.085 [0.057, 0.114] on Instruct vs 0.049 [0.037, 0.059] on base, while the judging side
does not differ), and that is not organized by the harm content refusal reads (third-party harm
mid-pack; binary family MDE ≈ 0.40). What the action reads is the open question the execution
program answers next.

Scope: the three-reads sentence is OLMo-3 only. Refusal's harm read also holds on GPT-OSS; Llama's
refusal reads broad moral content (`R_refusal ≈ R_judgment`) and Qwen's reads beyond harm rank-1 (0.54
vs 0.38), so across families "refusal reads a harm slice" is a family property, not a law. The action
read exists on one model.

### The instrument claim (gate decision E1, 2026-09-25: lead with the instrument, not the pattern)

Standing claim, as the pitch will state it: **evaluations that score behavior at a single
operating point report post-training as an improvement; a matched pressure-removed twin separates a
baseline shift from a change in pressure sensitivity; on OLMo-3-7B the two move in opposite
directions for moral action** (no-pressure raw gap +0.024 → −0.038; pressure-attributable excess
0.018 → 0.046 on the shared 192), so the net gap under pressure falls (0.043 → 0.011) while
sensitivity rises.

Related work, downgraded 2026-09-27: `sonnetx/tracing-sycophancy` (Sonnet Xu; no paper) reports a
related behavior–probability dissociation on the same OLMo-3 checkpoints. Its log-prob track scores
every checkpoint, instruct included, on a raw `Question: …\nAnswer:` prefix without the chat template
(`src/utils.py` `format_logprob_*`, commit 6bcbc1c172), while its behavioral track uses the template.
That is the format confound §15 of KDG_RESULTS found on our instruct model (KDG-39), so the
citation reads "related behavior–probability dissociation on the same checkpoints; readout
comparability unverified", not convergent evidence.

Coincidence interrogation (move 4): two constructs, one recipe, same shape. The simplest model
that predicts both is a property of the OLMo-3 Instruct recipe that lowers the default (at-rest)
output and raises the gain on in-context pressure. The rival that fits both equally well is that
**both results are log-probability readouts**, and log-prob sensitivity may grow under post-training
for reasons unrelated to behavior (sharper distributions amplify any prompt-induced shift, on the
probability and the log-odds scale alike). The first plan was a behavioral cell (the binary readout at
adequate power); the Phase 1 power table (2026-09-26) found it needs ~1,900 shared scenarios, so the
rival is split and tested at zero GPU: log-odds for baseline compression, and a judging-side
prediction plus a scale measured on the pressure-removed twins for sharpening (`KDG_PHASE1_SPEC.md`
Z1a/Z1b). The repo's result is exposed to the same sharpening rival and, since 2026-09-27, to the format
confound (its log-prob track reads chat models without their template); it is no longer cited as
convergent.

| Standing claim | Strongest counter-reading | Separating experiment | Status |
|---|---|---|---|
| Pressure-attributable gap present in base, larger after post-training, on the acting side | Two scale rivals: baseline compression (probability scale) and uniform logit sharpening (any log-prob scale) | Z1a log-odds recompute (compression); Z1b judging-side prediction + twin-measured scale normalization (sharpening); both zero GPU, `KDG_PHASE1_SPEC.md` §2. The binary readout cannot separate them at feasible n (MDE 0.088 at 192; ~1,900 needed for 0.028) | continuous readout: *widened*; binary: same sign, under-powered; Z1a/Z1b pre-registered 2026-09-26 |
| Post-training lowers the at-rest baseline (KDG-A6) | R_b raw-frame artifact; **R_c evaluation caution** (Burnat & Davidson 2605.06327: OLMo-3-Instruct refuses +0.118 more under eval framing, 20-item pilot) | Letter-only chat twins × eval/deploy/neutral prefixes (< 15 min, same pod leg); zero-GPU option-mass split first | open; priced; promoted to the Phase 1 pod |
| Opposite movement is a property of post-training, not of OLMo-3 | OLMo-3-specific recipe | Second base/instruct lineage (~2 h) + tier 2 (~9 h) | open; Phase 1 |
| The action does not read the harm slice refusal reads | Under-powered family contrast (MDE ≈ 0.40); KDG-A5 exploratory structure | Family top-up for F3/F5 (~1 h); S1 action-position rank cell | open |
| Widened sensitivity is goal-following, not a moral read | Deliberation reaches the action (moral read present) | Dose arm with filler control (~90 min) | open; Phase 1 |

Retired by this gate: the pitch framing that led with "post-training widens pressure sensitivity"
as the headline finding. The finding stands as a construct-specific result; the lead is the
instrument that shows it, because the shape is no longer uniquely ours (`LIT_PASS_P9.md` §1).

### What each result does to the thesis (pitch branch table, with gate additions)

Unmarked rows are the pitch doc's table ("What each result does to the moral-grounding thesis"),
except that its "SFT or DPO widens it most" row is split into an SFT row and a DPO row; rows marked †
were added or split at the 2026-09-25 gate. Both sides of each pending cell are written before
data.

| Result | Thesis edit |
|---|---|
| Gap present in base on a second lineage | The judgment–action discrepancy is pretraining-native; post-training patches cannot be the whole fix. Strengthened. |
| Gap absent in base on a second lineage | The base-model gap is OLMo-3's; the pretraining claim narrows to comprehension only. |
| RL stage widens pressure sensitivity most | Goal-following is installed late; the remedy is environment and reward design. |
| † SFT widens it most | The cause is earlier than the labs are looking; the stage sweep leads the pitch. |
| † DPO widens it most | Replicates the stage at which OLMo-3-7B sycophancy jumps (Blank et al. 2608.31079: 12.5 → 31.6 on challenges that name no alternative; letter-naming challenges at ceiling from SFT) on a new construct. The cause is preference data (their delta-learning finding), not RL; the pre-registration names which pressure subtype it expects DPO to move. |
| Deliberation closes the gap | Moral reasoning reaches the action when given room; the lever is behavioral. |
| Deliberation leaves the gap | More moral reasoning, same action: the strongest form of the knowing-but-not-doing reading. |
| † Deliberation widens the gap | Matches the incident (OpenAI: "As agents used more reasoning effort, some pursued increasingly risky and out-of-bounds strategies"). Reasoning serves the goal, not the judgment; the dose arm becomes a risk measurement, and the monitor's dose-0 case gains weight. |
| Action position reads the moral subspace at low rank | Widening that read is the intervention; grounding is the basis. |
| Action position reads a goal direction and nothing moral | Grounding is necessary and not sufficient; value moves to measurement and monitoring; the intervention targets goal arbitration. |
| † KDG-A6 closes as R_c (evaluation caution) | The "safer at rest" half becomes an evaluation-framing effect: itself an instance of the pitch's point that single-condition evals mislead, but the baseline-shift clause leaves the headline; sensitivity (a within-frame difference) is unaffected. |
| † KDG-A6 closes as R_a (installed agent-frame caution) | Both halves stand; persona steering is the first Phase 3 lever. |
| †† Any stage "widens" on the output scale but not per unit of that stage's own output scale (Z1b/P1-A1 per stage) | Sharpening, not widening: the stage makes the output more decisive and the incentive's per-unit pull does not grow. The live rival for every "widens" row; reported beside each stage verdict. |

†† Added 2026-09-27 (author). The SFT/DPO/RL "widens" rows above were written for the raw-frame
construct; once the template-valid stage sweep reports (C3 chat secondary and the SFT bridge,
P1-A3), they are re-scoped to the template readout at that gate. KDG-A6 closed as R_b
(KDG_RESULTS §15), so the † R_c and R_a rows are resolved: the at-rest baseline half is dropped.

### Conjunctions under watch

- **Instrument claim × tracing-sycophancy × KDG-A6 (closed 2026-09-27).** KDG-A6 resolved as R_b
  (raw-frame artifact), and the repo's log-prob track uses the same raw-frame reading on chat models;
  the conjunction now reads as a shared format confound (KDG-39), not a shared recipe property.
- **Incident × dose arm.** OpenAI attributes the incident's riskiest behavior to agents with the
  largest reasoning effort; our dose arm pre-registers closes/leaves and now widens.
- **Refusal family split × action read.** Refusal reads harm on OLMo/GPT-OSS and broadly on
  Llama/Qwen. If the second-lineage action read differs the same way, the action read inherits
  the family's refusal read; if not, action and refusal are separate channels on every family.

### Novelty scope after the gate (`LIT_PASS_P9.md` §1)

Owned: the pressure-removed twin with its calibration ladder on a self-referenced moral gap; the
base-model cell in that form; the peer-GO attribution twin (F6, dated, closing fast); the
grader-vs-human audience contrast for concealment. Not owned: the judgment–action construct
(Huang, Shen, Strakhov & Claude now cited in the KDG paper), the model-as-agent role change
(Strakhov & Claude), peer pressure in general, impossible tasks with an escalation exit (Troy
Moment, ImpossibleBench, GAIN), stage sweeps on OLMo-3 checkpoints (Blank et al.; the repo;
Cairns; Bharadwaj & Kirk). No "first to" without its qualifier.

### Referee pass on the restated claim

1. *"Your 'opposite directions' is two log-prob numbers, and so is the repo's; post-training
   sharpens distributions, which inflates any log-prob shift."* Open, with a pre-registered zero-GPU
   test (Z1b): if the judging side does not scale with the acting side and the twin-normalized Δ
   stays positive, sharpening does not explain it; otherwise the pitch sentence is rewritten as the
   spec writes it. The binary readout cannot decide it at feasible n.
2. *"The at-rest improvement is OLMo-3-Instruct being eval-cautious, which is published."* Answered
   by design, not yet by data: the sensitivity claim is a within-frame difference and survives R_c;
   the baseline clause is scoped until the frame-prefix twin cell runs (< 15 min).
3. *"One model family, and the precedent is on the same family."* Conceded. The instrument claim
   is stated on OLMo-3; the lineage generalization is Phase 1's second pair and tier 2, and both
   branches (present/absent in a second base) are written above.


### Phase 1 zero-GPU scale checks (2026-09-26; `kdg_panel/KDG_RESULTS.md` §14) — what changed

Positive voice first: **the pressure-attributable judgment–action gap is present in base and in
instruct on the probability, log-odds and output-scale-normalized readouts; post-training triples
the sharpness of the model's option distribution (k_twin 3.02 [2.83, 3.24]) and the acting side's
log-odds pressure response grows by about that factor (2.91×).** Pre-registered verdicts: Z1a
survives baseline compression (Δ E_logit 0.355 [0.184, 0.524]); Z1b(i) rejects *uniform*
sharpening (the judging side scales less, D_judge −0.199 [−0.335, −0.069]); Z1b(ii), the primary,
returns `sharpening_explained` (normalized Δ +0.12 [−0.017, 0.26], MDE ≈ 0.20). Z2: KDG-A6 R_b
loses its cheapest support.

Scope notes applied here (move 3): the claims-table row "pressure-attributable gap ... larger
after post-training, on the acting side" now reads *larger on the probability and log-odds
scales; not separable from output-scale sharpening at MDE ≈ 0.20*. The instrument claim ("baseline
and sensitivity move in opposite directions") holds on the output scale and is unresolved per unit
of scale. The execution thesis sentence's "larger, not smaller, after post-training" carries the
same scope. The tracing-sycophancy convergence is exposed to the same rival. Paper and pitch
wording are escalated (KDG_RESULTS §14.4); CLAIMS untouched pending the author.

What the next result changes: the frame-specific normalization fork (amendment next) → negative
difference: the headline becomes sharpening of the action channel, not widening; positive and
resolved: widening survives with a frame-specific scale. C1's chat-frame σ (Session A) → separates
KDG-A7 R_a (installed decisive agent frame, a persona lever) from R_b (raw-frame format).

Fork P1-A1 (frame-specific normalization; pushed before computing; labelled): per unit of each
frame's own scale, post-training lowers the pressure response on both sides (acting −0.201
[−0.375, −0.046], judging −0.128 [−0.251, −0.010]); the excess difference is unresolved and negative
(−0.074 [−0.224, 0.065], MDE 0.21); the gap stays present in both models (base 0.233, instruct
0.159, both CIs above 0). Ratio-of-means second derivation agrees in sign on both sides. Candidate
reframe for the author (not adopted): the execution program's post-training result is
*sharpening*, not widening: a twin-less eval sees a safer model at rest and a larger raw pressure
response, and neither is the per-unit change. The instrument claim survives with this mechanism;
the "widened sensitivity" headline does not survive per unit of scale at this power.


### Session A, part 1: C1 and KDG-A7 (2026-09-27; `kdg_panel/KDG_RESULTS.md` §15) — what changed

Positive voice first: **under its own chat template OLMo-3-Instruct acts more toward the violating
option at rest than its letter-only judgment does (+0.055 [0.034, 0.076], n 136), as the base model
does in the raw frame (+0.024); the judgment–action gap is present at rest and under pressure, before
and after post-training.** The "safer at rest" reading (raw-frame −0.038) is a format artifact
(KDG-A6 → R_b, pre-registered). Most of the raw-frame agent-frame sharpening is format too (KDG-A7:
ratio 1.79 raw → 1.07 chat; mixed by the rule, near-miss).

Scope notes applied here (move 3): the instrument claim "baseline and sensitivity move in opposite
directions" loses its baseline half; what the twin instrument now shows on OLMo-3 is a
pressure-attributable excess in base (0.018) and a larger one after post-training on the
probability and log-odds scales (0.046), with the per-unit-of-scale comparison unresolved (§14) and
the raw-frame scale factor partly a format property of the instruct model (§15.2). Pitch, paper and
CLAIMS wording on the baseline shift: escalated to the author.

Open: C3 stage sweep (stages_raw lost in the download; re-run needed); dose arm (512-token budget too
short for OLMo-3-Instruct's reasoning; fork decision with the author).


### Session A, part 2: bridge, stage sweep, dose arm (2026-09-28; `kdg_panel/KDG_RESULTS.md` §16) — what changed

Positive voice first: **on OLMo-3 the pressure-attributable judgment–action gap is present in every
templated checkpoint under its own template (SFT 0.021, DPO 0.022, final 0.030) and no post-training
stage changes its size at this precision; moral reasoning before acting, even truncated at 512 tokens,
pulls the action toward the norm relative to matched non-moral text (−0.077 [−0.110, −0.047]).** The
program now has a behavioral lever on the gap, and the gap's size is a pretraining-and-SFT property
that preference optimization and RLVR leave where it was.

Verdicts: SFT bridge → base cell **descriptive only** (raw − chat at-rest gap −0.052 [−0.070, −0.034]
at SFT; KDG-39 dated to the first templated stage). Raw stage sweep descriptive only (n_shared 130 <
150; raw frame invalid for templated stages). Dose arm **closes** (truncated reasoning). Open: KDG-A8
(at-rest acting lean grows through DPO and RL under the template; rival = selection on the final
model; zero-GPU discriminator priced).

Thesis edit (move 7): the execution thesis is now "the judgment–action gap is pretraining-native in
the only frame a base model has, present in every templated checkpoint, not resized by preference
optimization or RL, and reduced by moral deliberation before acting". The "widened sensitivity" and
"lowers the baseline" clauses are both gone; the SYNTHESIS branch rows "DPO/RL widens" do not obtain
under the template, and "Deliberation closes the gap" obtains as truncated reasoning (the completed-
reasoning check is the 2,048 rider). Pitch and paper wording: at the part-A gate with the author.

KDG-A8 resolved (2026-09-28; KDG_RESULTS §17; P1-A7 pushed before computation): on a final-model-free
set (n 586) the acting frame's at-rest lean toward the violating option grows at the DPO step on the
probability scale (+0.011 [0.007, 0.016]) and per unit of output scale (+0.058 [0.027, 0.090]); the RL
step is sharpening. The pressure-attributable excess stays flat (no stage change detectable above
about 0.005 at n 586). Standing claim candidate (author's gate): preference optimization shifts the
agent frame's default toward the advantageous option without changing the incentive's pull.


### Session B: dose controls, rider, lineages (2026-09-28; `kdg_panel/KDG_RESULTS.md` §18) — what changed

Positive voice first: **on OLMo-3 moral reasoning before acting lowers the violating choice against a
length-matched restatement and against a truncation-matched one (−0.112 [−0.145, −0.078]), and naming
the norm does about a third of it; the DPO-stage shift of the acting frame toward the advantageous
option at rest replicates on a second lineage (Tulu 3, +0.017 per stage, +0.099 per unit of scale).**

What changed for the thesis (author decision pending): the gap is lineage- and recipe-dependent
(KDG-A9). Present under the template on OLMo-3 and Llama-3.1-Meta instruct, absent on Tulu 3 and
Qwen2.5 instruct; base-raw present on OLMo-3 and Qwen2.5, absent on Llama-3.1. "Pretraining-native"
does not generalize; by the branch table, "gap absent in base on a second lineage" obtains (Llama),
and the pretraining claim narrows. The raw-frame distortion (KDG-39) is an Ai2-recipe property (OLMo-3,
Tulu 3), absent on Meta's recipe. Open with priced discriminators: KDG-A9 (recipe vs panel
sensitivity vs format).


### Provisional framings pending Session C (author, 2026-09-28); decided after the known-gap validation, screen rates and the Llama dose arm

- **(i) Scoped:** the judgment–action gap and the deliberation lever are established on OLMo-3; other
  lineages are validated scope limits, stated per model as "present", "not detected (instrument
  validated)" or "instrument not validated on this model".
- **(ii) Recipe:** the base model sets whether the gap's raw material exists; the post-training recipe
  decides whether it survives. Anchor, if the nulls validate: the same-base contrast on Llama-3.1 (Meta's
  recipe installs a template-valid gap, 0.028, on a base with none in the raw frame; Tulu 3's recipe on
  the same base shows none).

**Next paper's thesis (seeded, not added to the KDG paper):** which post-training recipe choices install
or remove the judgment–action gap, and whether the same choices produce the DPO-stage at-rest lean
(replicated on OLMo-3 and Tulu 3) and the raw-frame distortion (both Ai2 recipes, not Meta's).


### Session C (2026-09-28; `kdg_panel/KDG_RESULTS.md` §19) — what changed

Positive voice first: **the instrument is validated on OLMo-3, Llama-3.1, Tulu 3 and Qwen2.5 (known-gap
band 0.50–0.62); under their own templates OLMo-3 and Llama-3.1 (Meta) carry a pressure-attributable
judgment–action gap and Tulu 3 and Qwen2.5 do not (not detected above ~0.01 / ~0.02); and moral
deliberation before acting reduces the gap on both recipes that carry it (OLMo-3 −0.077 truncated,
Llama-3.1 −0.350 largely completed; both survive the truncation-matched control).** Framing (ii) meets
its own condition: on the same Llama-3.1 base, Meta's recipe carries the gap and Ai2's Tulu 3 does not,
with a validated instrument on both. Thesis choice between (i) and (ii): author, at this gate.


### Session C gate (2026-09-28): paper restructured; recipe paper seeded with a design

Author decisions: the recipe contrast is a full section of the KDG paper (not a scope limit), and the
paper's hold is lifted. The KDG paper's four claims: the gap exists (template-valid, OLMo-3); the
instrument withdrew our own claim; the gap follows the post-training recipe (four models, validated
nulls, same-base Meta vs Tulu); moral deliberation reduces it on both recipes that carry it (about a
third is naming the norm on OLMo-3).

**Recipe paper (seeded; not part of the KDG paper): which part of a post-training recipe decides whether
the gap survives.** Design: one base (Llama-3.1-8B, where the same-base contrast lives), recipes that
differ in one component at a time: (a) Tulu 3 SFT data vs a Meta-like SFT mixture at matched size, (b)
with vs without the DPO stage on each, (c) template held fixed vs swapped (the raw-frame distortion
tracks Ai2 recipes; the template may be part of it). Readout: this panel's letter-only chat cells with
the known-gap positive control per checkpoint, the pressure-removed twins, and the dose arm on the
carrying checkpoints. The Tulu 3 intermediate checkpoints (public) give leg (b) for free; legs (a) and
(c) need short fine-tunes at 8B. Both branches publishable: a single component decides (a lever for
post-training) or the effect is distributed across components (a reason to measure it, not engineer it).


### KDG-A12 (2026-10-01): stage-claim readout of record; the RL step sits at its bar

Author decisions: the final-model-free 586 set is the number of record for stage claims (screened by
no model's actions, so the same diluted set at every stage: lower absolute levels, a fair and
conservative stage contrast); the 136 screened set is the secondary, with its own per-step bars
(DPO 0.019, RL 0.012). On the 586, OLMo-3's DPO step does not move the pressure-attributable excess
(−0.001 [−0.006, 0.005], bar 0.008) and its RL step reads +0.004 (lower bound +0.00041 over 10,000
resamples; bar 0.005), at the bar and with every known bias (RL sharpening, two steps tested)
favoring a positive step, the same sign as the screened set's 0.007. "Not resized by preference
optimization or RL" in the thesis edit above is held, not withdrawn: wording waits for the per-scale
test (P1-A10, pushed before computing). Recipe datum: the RL-step sign differs by recipe at bars
that resolve neither (OLMo-3 +0.004 at its 0.005 bar; Tulu 3 −0.001 [−0.005, 0.002]), the Ai2 ask
row's question in miniature. No paired slide.
