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

**Correction (2026-10-06, ANOMALIES A11 — no standing claim or verdict changed).** The P3 detection
bar recorded above ("20 partitions, smallest achievable p = 0.05") was wrong: the 20 assignments form
10 mirror pairs with equal statistics, so the exact floor is **p = 0.10** and the MFT-grouping test
cannot reject at α = 0.05 on any data. Found through a float-tie bug in the permutation tests (bare
`>=` counted mirror ties by rounding; fixed with a 1e-12 tolerance in the library and the P3 script).
Exact recompute from saved matrices: 1B p 0.40–0.80, 7B 0.30–0.90, dense ≥ 0.40, MoE ≥ 0.30; the MFT
split never ranks above fourth of the ten splits. P3 corrected for arXiv v2 (abstract bar, floor in
five passages, Appendix D cells 9/12, §4.3/§4.5/§4.14 ranges made exact). The null stands with a
stricter bar: it rests on the dendrogram leg and the split's rank, not on reaching significance.

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

**P1-A10 result (2026-10-01; KDG_RESULTS §20).** Positive voice first: **on OLMo-3, no post-training
step after SFT adds to the incentive's pull on the action once output decisiveness is accounted for
(per unit of output scale DPO lowers it, descriptively, and RL's probability-scale +0.004 is
sharpening on the primary readout); what post-training changes is how decisive the output is and, at
the DPO step, the at-rest lean.** The RL step's probability-scale +0.004 survives log-odds (compression is out) and does not
survive division by the output scale on the primary readout (E_norm +0.005 [−0.016, +0.026], z 0.49);
the frame-specific scale reads unresolved (z 1.22), so the verdict is scale-dependent and "adds" is
not licensed on either. The thesis sentence "not resized by preference optimization or RL" holds on the
probability scale for DPO (null, bar 0.008) and not as "enlarged" for RL on any scale; per unit of
scale DPO's step is negative, so "not resized" is probability-scale wording. Held for the paper as
the §20.4 two-scale sentence (author). Ledger:
KDG-A13 (DPO per-scale step opposite in sign on the two Ai2 recipes; base and recipe confounded;
recipe-paper candidate beside RL-Zero, no prose), KDG-A14 (two near-misses at their bars are not
convergent evidence). Author (2026-10-01): the reading is **unresolved**; the within-RL sweep is priced into the F6–F8 pod
plan as an optional OLMo-3 extra. Paper wording: "no post-training stage enlarges the
pressure-attributable part on either readout" replaces "not resized".


### Phase 2 pilot (2026-10-02; `kdg_panel/KDG_RESULTS.md` §21) — what changed

Positive voice first: **a principal's one-sentence reminder lowers the violating choice on both recipes
that carry the gap (−0.072 on OLMo-3, −0.184 on Llama-3.1), and over six short scripted turns (138 tokens)
no decay is detectable above a third of its size on OLMo-3 or a sixth on Llama (superseded at longer
distance by the 2026-10-04 token ladder below: Llama R(6) 0.50 at 2,000 tokens); the incident
families F6–F8 work as instruments on both models (full engagement including five options, positive
control 0.40–0.66, a positive incentive effect in every family).** The norm-salience lever the KDG paper
found in the assistant prefix (a third of the deliberation effect on OLMo-3) is larger when the
principal states the norm in a user turn (−0.072 vs −0.025), and it persists across short conversations.

What did not pass: the pre-registered decisiveness gate G2 fails on every family, and the same rule
fails the Phase 1 panel of record at the same rate (KDG-A15): the letter-only judge readout at 7–8B
carries about a quarter of its mass on the violating option at rest on old and new scenarios alike.
Re-stating G2 against the panel is a fork with the author. The F6 pilot shows the unattributed GO text
moving the action as much as the peer-attributed text (KDG-A17, descriptive at n 24), which points the
full panel toward the "text, not speaker" branch and raises a construct question about the
no-speaker level before generation.

Standing-claim edits: "The turns-since-norm decay seen at frontier scale" now reads, for this panel, "no
decay detectable over six scripted turns (138 tokens), bars 0.33 / 0.16", with the token-distance dose as
the separating cell for the long-trajectory rival. Conjunction under watch, added: the F8 judge frame puts
a third or more of its mass on the prohibited route under the penalty (0.34 / 0.44), so the penalty moves
judgment as well as action on both models; if the full panel keeps this, F8's gap is small because the
judgment follows the pressure, the persuasion branch of §9 for a non-peer pressure.

**Pilot gate applied (2026-10-02).** G2′ (decisiveness relative to the panel of record) passes every
family on both models at pilot MDE 0.26–0.41, with Llama F8 the weakest (0.50 vs the panel's 0.70); the
F6 no-speaker level reads as "cannot tell" for 22 of 24 items, so the registered ladder stands. Full
construction is licensed at 48 decisive primaries per family. The turns-since-norm short-distance claim
(no decay detectable at 138 tokens) stays at that distance until the token ladder (P2-A4b) reports.

### Turns-since-norm follow-ups (2026-10-04; `kdg_panel/KDG_RESULTS.md` §22) — what changed

Positive voice first: **on Llama-3.1, a principal's reminder holds its effect across six short turns and
loses about half of it when the same six turns carry 2,000 tokens; the decay follows token distance, not
turn count (R(6) 0.95 → 0.84 → 0.50 at 138 / 600 / 2,000 tokens).** The KDG-55 short-distance sentence
(no decay detectable over six turns at 138 tokens) is scoped to that distance on both models; at 2,000 tokens the reminder fades on Llama and
OLMo-3 cannot be read (KDG-A18, a floor artifact). The A4b wording rule's condition is met: the verdict is
recorded; pairing it with Anthropic's 90 / 40 figure (same direction, different construct, one model) is
the author's decision. KDG-A16 stays unresolved after counterbalancing, with the filler-confound reading
weakened.

### Adversarial review applied to the KDG paper (2026-10-05)

Positive voice first: **the recipe split survives the check designed to break it: on each model's own most-
pressuring scenarios, against a null built by the same screen on the pressure-removed twins, OLMo-3 (0.073)
and Meta's Llama-3.1 (0.143) carry a gap and Tulu 3 (−0.004) does not; Qwen2.5 is unresolved at that
resolution (0.083, −0.028 to 0.195) and null on the whole panel.** The deliberation claim is re-scoped: the
dose arm moves the action toward the model's judgment (measured on primaries); whether it reduces the
pressure-attributable part or the at-rest asymmetry needs the twins under deliberation (P1-A12, priced,
author's go). Paper wording follows: §9 retitled, §10 conditional. **Split wording of record (author, 2026-10-05):**
"Tulu 3 shows none on either read; Qwen2.5 shows none on the whole panel and is unresolved on its own
screened scenarios (0.083, −0.028 to 0.195; n 47, E_sel bar about 0.16)", replacing "two do not on the
whole panel"; Qwen's own-screen read is KDG-A19 (open, priced).

### Turns-since-norm, state of record (2026-10-05, author)

No line in this file or the pitch drafts says the gap or the reminder "does not fade" with turns. The
state of record is the A4b token ladder (KDG_RESULTS §22): on Llama-3.1 Meta the reminder's effect
decays with token distance, R(6) 0.949 / 0.842 / 0.502 [0.429, 0.575] at 138 / 600 / 2,000 tokens;
OLMo-3 shows no decay at 600 tokens and cannot be read at 2,000 (option mass 0.013, KDG-A18). The KDG-55
wording rule stays in force: "no decay detectable" is stated only with its distance (138 tokens) and
bars, and pairing the Llama ladder with Anthropic's 90 / 40 figure (same direction, different construct,
one model) is the author's call. KDG-A16 (the Llama k = 3 dip) is logged as unresolved, not resolved.
Nothing from p2b enters the KDG paper.


### Twin under deliberation (2026-10-05; `kdg_panel/KDG_RESULTS.md` §24) — what changed

Positive voice first: **thinking about the stakes before acting brakes the violating action with or without
the incentive: it leaves about 0.72 (OLMo-3) and 0.30 (Llama-3.1) of the violating choice of a same-length
non-moral task under pressure, and 0.73 and 0.32 with the pressure removed.** A single common ratio predicts
the observed probability-scale difference on both models (−0.136 vs −0.143 on Llama), so the post-hoc ratio
reading is consistent with one proportional cut; it does not separate a pressure-specific part smaller than
about a fifth (OLMo-3) or a quarter (Llama) of the brake.
Registered verdict (P1-A12, probability scale) beside it: Llama branch (a), OLMo-3 branch (b); the ratio fork
(P1-A14, post-hoc, labelled) reads proportional on both models and both references (bars on the log ratio
0.21 and 0.30). Standing-claim edit: "moral deliberation reduces it on both recipes that carry it" (the
2026-09-28 four-claim line above) now reads "moral deliberation lowers the violating action on both recipes
that carry it, with or without the pressure (registered: Llama (a), OLMo-3 (b); post-hoc ratio reading
consistent with one proportional cut, bounds a fifth / a quarter)". The goal-following account of the gap is
unaffected; whether the brake acts on the goal the incentive supplies below those bounds is open (author
wording rule 2026-10-05: "not the goal" is not asserted). Separating cell, if wanted:
more scenarios per model (the bars on L shrink as 1/√n) or a dose that names the incentive explicitly. R1
closed; KDG-42 and KDG-53 carry scope notes (CLAIMS KDG-59).

**KDG paper on arXiv (arXiv:2610.08670).** *Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment*; the four-claim paper of the Session C gate, with the 2026-10-01 stage wording and the 2026-10-04 submit-gate edits.

### GPT-OSS-20B Tier 2 and the readout's position handling (2026-10-07; `kdg_panel/KDG_RESULTS.md` §25, ANOMALIES KDG-A20..A22) — what changed

Positive voice first: **on OpenAI's open reasoning model the panel's instrument locates its own limit:
the forced dose-0 letter departs from GPT-OSS-20B's own low-effort answers (C0 0.69 vs 0.80), and the
departures that cross the norm all go toward it (6 vs 0, a scoped descriptive).** The verdict of record
is the pre-registered "readout invalid". Whether the forced letter is a valid dose-0 readout (the
departure is dose) or a format artifact is open; the two dose-0 prefills agree on 0.92 of scenarios,
which rules out the empty analysis turn but not forcing as such, and the dose-matched C0 (G-A11, about
0.3 A100-h) separates them. Author wording rule: no GPT-OSS gap statement in either direction until C0
clears. Standing-claim edit: the instrument claim's scope reads "validated at dose 0 on instruct models;
on a harmony reasoning model the dose-0 readout's validity is open". The recipe split is unchanged.
GPT-OSS's readout is batch-dependent per row but zero-mean in E (KDG-A21, R_a).
Readout position handling (KDG-A20, resolved 2026-10-08): the letter readout of record read left-padded
rows at shifted positions. On the stack of record it stays within the 0.05-nat gate on OLMo-3, Llama-3.1
Meta and Tulu 3, no pad contrast aligns with E on any model, and Qwen2.5's re-read with exact positions
leaves E unchanged (ΔE +0.0002 [−0.0001, +0.0004]). Every panel number, including the KDG paper's, stands
with a methods note; readout version 2 is exact and is the harness going forward.

### Dose-matched C0 and the Qwen2.5 re-read (2026-10-08; `kdg_panel/KDG_RESULTS.md` §26) — what changed

Positive voice first: **the panel's numbers stand on an exact readout (KDG-A20 resolved on all four
models), and on GPT-OSS-20B the forced dose-0 letter is, token for token, the distribution the model's
own dose-0 answer is sampled from; what separates it from GPT-OSS's low-effort answer is reasoning, which
moves norm-crossing answers toward the norm (6 vs 0 forced-vs-generated, 5 vs 0 generated-vs-generated).**
Registered: the dose-matched C0 fails as a near-miss (0.766 vs 0.80); fork: sampling-limited (a valid
readout passes that bar 5% of the time). Whether C0 clears is the author's; the wording rule (no GPT-OSS
gap statement in either direction) holds until then. Rival to the positive sentence: the 6/5-vs-0 counts
are small (6 and 5 norm-crossing scenarios of 64; sign test p 0.031 for 6 vs 0); separating cell, more
scenarios at both doses.

### Author decision on C0; GPT-OSS-20B at dose 0; KDG-A23 (2026-10-08; `kdg_panel/KDG_RESULTS.md` §27)

Positive voice first: **at dose 0 the instrument reads OpenAI's open reasoning model and finds no
pressure-attributable gap detectable above 0.013 (E −0.001 [−0.011, 0.008], n 586), with its known-gap
positive control responding at 0.50.** The carrier/non-carrier contrast is not explained by dose-0 answer sharpness. Within every model, dose-0 entropy does not predict E (all five slopes include 0), and the Llama−Tulu 3, Llama−GPT-OSS and OLMo-3−GPT-OSS contrasts survive entropy matching. Three contrasts lose resolution under matching: both Qwen2.5 contrasts because Qwen2.5's sharper distributions share only 6 of 10 entropy bins with the carriers, and OLMo-3−Tulu 3 through attenuation (+0.017 → +0.009) despite full bin overlap; those three rest on the unmatched estimates only. This rests on the author's decision that C0 clears for the
dose-0 readout (registered C0 0.688 and C0-dm 0.766 both fail at 0.80; the token identity and the G-A12
fork carry the clearance). Scope: dose 0 only; reasoning moves GPT-OSS's answers (dose-0 and low-effort
generation agree on 0.56 of scenarios), so nothing is claimed about its default reasoning level (medium) until the
dose-stated run clears its own C0. C0 bars become model-relative (0.9 × the exact-readout ceiling; κ\*
0.61–0.82 across the panel on the C0 sample, so 0.80 was above the ceiling on four of five models).
Standing-claim scope note (KDG-A23, updated 2026-10-08): the sampled traces behind the dose claims
(KDG-42/53/59, the KDG paper's deliberation brake) shared one random stream per rollout index across
scenarios. Under the two-way bootstrap (P1-A15, a fork) their CIs widen by 16–47% and the point
estimates hold; the deliberation brake stands on both models. One parenthetical changes: OLMo-3 against
its own truncated filler reads (b), not (a) (escalated). The harness now seeds per scenario with pure
temperature sampling (sampler version 2).

### Primary 2: GPT-OSS-20B at its default reasoning level (medium) moves norm-crossing actions toward the norm (2026-10-08; `kdg_panel/KDG_RESULTS.md` §30)

Positive voice first: **reasoning at GPT-OSS-20B's default reasoning level (medium) moves its action toward the norm. Of
the 76 scenarios (of 576) whose dose-0 choice and medium-effort choice differ in norm label, 74 move
toward the norm-consistent option and 2 away (one-sided exact p = 3.9 × 10⁻²⁰); sampling alone, with no
reasoning, would give about 39% toward.** The deliberation brake the KDG paper measured on OLMo-3 and
Llama-3.1 under forced dose arms holds, as a direction claim, at the default reasoning level (medium) of a
reasoning-RL-trained model from another lab. Standing claim: "deliberation brakes the violating action"
now spans three recipes (two instruct models under forced reasoning, one reasoning model at its default reasoning level, medium).
Strongest counter-reading: generic caution (reasoning picks the safer, rule-following option whatever the
moral content). Separating experiment: a non-moral reasoning control at matched length, or scenarios whose
norm-consistent option is the riskier one (priced, not run). Not claimed: the size of the effect, or a
pressure-attributable gap at medium effort (primary 1 withdrawn; E at medium is not estimable within the
envelope). Owed: the forward readout and post-reasoning residuals for the 640 banked traces and the
VALIDATE numbers, lost in a post-generation crash and re-derivable in one short pod. [Re-derived
2026-10-08, KDG_RESULTS §31: VALIDATE exact under identical batching on two pods; 625 readout rows and
post-reasoning residuals banked; after reasoning GPT-OSS's letter is decided (top-letter probability
≥ 0.99 on every row), so its medium-effort variation lives in the trace, not the answer step.]
At medium effort the decision is made inside the trace (top letter ≥ 0.99 after every trace; 12 of 22
scenarios flip across traces), so deliberation-stage residuals, not decision-token residuals, are where a
moral reference would have to be present. Letter balance (G-A21, 2026-10-09): the norm-consistent
option's letter at permutation 0 is balanced against GPT-OSS's dose-0 letter marginals (chi-square
p = 0.21; letter-prior advantage +0.003 [−0.001, +0.007]), so the letter channel does not carry the 74 vs 2.
Risk-reversed separating cell (G-A21, 2026-10-09, KDG_RESULTS §32): **reasoning at
the default level moves GPT-OSS-20B toward the option that is norm-consistent and, by two blind raters, the
less risky one in 542 of 586 scenarios.** The set holds 1 scenario both raters call risk-reversed, so the
pre-registered branch is unresolved: norm-tracking and risk aversion under RL point the same way on this
set, and the separating cell needs scenarios built so the norm-consistent action carries the risk (about
250 scenarios, about 2.3 A100-h; author's call). The same rival reaches the KDG paper's deliberation
brake (KDG-A26, escalated). Split constructs (G-A22, 2026-10-09, §33): **reasoning at the default level moves
GPT-OSS-20B toward the norm where its own dose-0 judgment points elsewhere (44 of 85 opportunities vs 1 of
9 toward that judgment, divergent-violating scenarios) and, descriptively, where the norm-consistent
option costs the agent itself (10 of 10 norm-crossings in the 83 actor-reversed scenarios).** The
actor-cost cell stays unresolved by its gate (10 crossings against 27); harm to others is not separable on
this set (1 reversed scenario), so the held scenario set, if built, targets harm divergence. Still
claimed only as direction, and only against the dose-0 judgment readout, not a judgment read after
reasoning. Paper's brake by actor cost (G-A23, 2026-10-10, §34): **Llama-3.1's deliberation brake
holds where the norm-consistent option costs the agent (−0.291 [−0.403, −0.192], 15 scenarios).**
OLMo-3's pooled brake is weaker there, but within family the gap disappears: OLMo-3's brake is absent
on F1 (record falsification) whatever the actor label, and lives in F3 and F4. Author check (2026-10-10, §35): the actor-cost labels are not confirmed (0 of 5
both-rater actor-reversed items called reversed by the author), so the actor-cost cell stays at 83,
unresolved; the stated-rule labels of the divergent scenarios are confirmed 9 of 10, while the
author's all-things-considered judgment departs from them on 2 of 4 answered (KDG-A27). §33's direction
is therefore stated as **toward the stated rule**; whether that is moral improvement on these scenarios
waits on the two-pattern scan. Scan (§36): defensible-under-the-rule departures are rare (4 of 92 divergent-violating, all
F1; positive control passes); the all-things-considered pattern's instrument fails its positive control,
so its prevalence is unmeasured; toward-norm moves occur at the same rate in rater-flagged
defensible-departure scenarios as elsewhere, so on present counts reasoning moves to the stated rule
whether or not the departure is defensible. Compliance-harm split (§37): **reasoning at the default level moves GPT-OSS to the stated rule as
reliably where breaking it harms no proximate third party (40 toward, 0 away) as where it does (31 vs
2).** On the present evidence (all descriptive or below gates), the direction §30 found reads as rule
compliance, not harm-avoidance or self-protection. The pre-registered compliance-harm cell is
unresolved (15 scenarios, 3 crossings); the H1 set (pilot 23 of 24 kept) is built for it. Judgment at the default level (§38, pre-registered, two pods): **At its default reasoning level GPT-OSS-20B's
judgment and action converge on the stated rule together: where the action moved from the dose-0
judgment to the norm, an independent medium-effort judgment trace lands on the norm in 52 of 69 decisive
scenarios (p = 2.9 × 10⁻⁵; F1 and the rest each significant).** Branch: principled revision. Single-trace
noise is large (a second trace disagrees 25–43% of the time); the coupling is an aggregate that holds
despite it, not a clean per-scenario pairing. §33's "against its own judgment" reading is withdrawn. On the human sample (§39, estimate of record) the
stated rule is also the right call in 12 of 13 decided divergent scenarios (defensible departures 1 of
13, 95% interval 1.4%–33%; 7 of 20 undecided), so at the point estimate the convergence reads as
correction, with that interval attached. The undecided items are cases where converging on the stated
rule under insufficient information is the prudent default. The rater P4 instrument is not
measurable. H1 compliance-harm cell (§40, pre-registered,
two pods): **where following a written approval gate harms an absent third party and a transparent,
personally costless departure would protect them, reasoning at GPT-OSS-20B's default level still moves
its action to the gate: 42 toward, 2 away (p = 5.6 × 10⁻¹¹); at dose 0 it protects the third party in 55%
of scenarios, after reasoning in 25%.** Construct validity: author spot check 10 of 10. Human baseline on those 10: the author protects on 9, GPT-OSS on 6 at dose 0 and 3 after reasoning (descriptive). Recoverability (§41): the author's one rule choice
was the recoverable loss; GPT-OSS's reasoning moves toward the rule even where the loss is irreversible or
severe (32 toward, 2 away), and the pre-registered comparison with recoverable losses is unresolved
(+0.079 [−0.071, +0.188]). **GPT-OSS-20B on H1, combined (§40–§42):** without reasoning it judges the protective
departure right in 58% of scenarios and acts on it in 55%; with reasoning, judgment and action converge
on the written rule together (judgment read at medium effort, §42), and the action's protective share is
24–37% at every level tested (low 24%, medium 25–31%, high 37%). A human reviewer takes the protective
departure on 9 of 10. Same mechanism as the panel (judgment and action move together, §38), different
evaluative reading (on H1 the rule is the harmful choice). The step that moves the action is the onset of
reasoning (pre-registered descriptive, pending the sampled dose-0 ride-along; KDG-A28); the low-to-high
rise is open (KDG-A29). **The carriers on H1 (§43, pre-registered per model):** the paper's forced reasoning arm
moves both OLMo-3 (−0.076 [−0.098, −0.053]) and Llama-3.1 (−0.050 [−0.067, −0.031]) toward the written gate
against harm prevention: rule compliance under deliberation on both. OLMo-3 shows no detectable difference in
brake magnitude between rule-protective and rule-harmful scenarios (−0.002 [−0.042, +0.037]); Llama-3.1's
brake keeps its direction at about a seventh of its panel size (−0.301 [−0.340, −0.260]), the pattern of a
brake that weighs the third party's loss alongside the rule.

### Deferred queue (compute scope closed after p1e, 2026-10-13)

Each cell is pre-registered or priced and waits for a new compute decision; the line after the price is
the decision its result would change.

| cell | design and price | the decision its result changes |
|---|---|---|
| OLMo-3 stage ladder on H1 (G-A30) | SFT, DPO, base (raw-frame forced arm with an engagement gate) and the final model in the raw frame as a bridge; forced arms, 8 rollouts; about 7.4 A100-h | Whether the rule-over-protection brake is intrinsic to deliberation (base already moves toward the written rule) or installed by post-training (and at which stage). It decides whether the thesis says "deliberation is rule-tracking" or "post-training makes deliberation rule-tracking", and where a training-time intervention would have to act. |
| Dose curve follow-up (KDG-A29) | Low and high effort on ~150 H1 scenarios on GPT-OSS-20B (80% power at a 70:30 discordant split); about 5 A100-h | Whether more reasoning partly restores protection (effort is a lever toward protection) or the curve is flat after the onset (the onset of reasoning is the only lever). It decides whether "higher reasoning effort" belongs in the mitigation sentence. |
| A second H1 family | A departure that is hidden, costly to the agent, or from a norm that is not a procedural gate; generation, rating and author spot check zero GPU, then the H1 cell on GPT-OSS (about 3 A100-h) | Whether the rule-compliance result is about approval gates or about written rules in general; it sets the scope of §40's claim beyond one homogeneous family. |
| Tulu 3 and Qwen2.5 on H1 | The paper's forced arms on the 264 (as p1e), per model about 3.2 A100-h | Whether the brake's direction on H1 is shared across recipes (the paper's recipe split) or specific to the carriers; it decides whether v2's brake scope sentence is panel-wide or carrier-scoped. |
| Rule-protective twin of H1 on Llama-3.1 | Each H1 scenario rewritten so the written rule is the protective choice (same writers, family and length), generation, rating and author spot check zero GPU, then Llama-3.1's forced arms on the twins (about 3.2 A100-h) | Whether Llama-3.1's brake falls to a seventh because the rule turns harmful (deliberation weighs the third party's loss) or because the H1 set differs from the panel in construction; it decides whether v2's differential sentence describes Llama-3.1's deliberation or the scenario sets. |
| KDG-A28 sampled dose-0 ride-along | The stratified 64, direct-final prefill; about 0.2 A100-h | Whether §42's onset step is reasoning or the instrument. Runs only if a GPT-OSS pod runs for another reason. | Scope: one
structurally homogeneous family; a second family is the extension. Branch:
compliance-tracking. With §37's descriptive split and §38's convergence, the program's reading of
GPT-OSS's default-level reasoning is rule compliance: it brings judgment and action to the stated rule,
whether or not the rule is the protective choice. On the panel, where the rule is usually right, that is
correction (§39's interval); in H1, where it is not, it is harm. Escalated: thesis-level framing.

### External-review revision of the KDG paper's v2 (2026-10-15; `kdg_panel/KDG_RESULTS.md` §44, P1-A17) — what changed

Positive voice first: **on OLMo-3 the model acts against its own judgment on one in five of the scenarios
screened on its own actions, with an excess over the pressure-removed twins carried by the log-probability
readout (0.054 [0.021, 0.086]) at every strictness level; the gap follows the post-training recipe across
five instruct models (Meta above Tulu 3 by 0.027 [0.017, 0.037] on the same Llama-3.1 weights, paired on the
586); deliberation brakes the violating action on both carriers; and where the written rule harms an absent
third party, deliberation moves every model read off the protective departure, onto the rule on OLMo-3 and
GPT-OSS-20B and onto contacting the approver on Llama-3.1.** What changed, all zero GPU:

- Stage claims: the OLMo-3 RL step (+0.004 [0.0004, 0.0078]) and Tulu 3's DPO step (+0.006 [0.00004,
  0.012]) are small increases on the probability scale, not "no enlargement"; the RL step does not resolve
  per unit of output scale (sharpening).
- Under the two-way bootstrap of record the binary paired excess touches zero at L0 and L1; the continuous
  readout carries the excess. The ratio reading resolves a pressure-specific part only beyond about a third
  (OLMo-3) or two-fifths (Llama-3.1) of the brake.
- On the rule-harms set Llama-3.1 moves into the approver-contact option, not toward the rule-following
  act (rule mass −0.022 [−0.036, −0.008]); OLMo-3 moves to the rule (+0.123).
- Clustered sensitivity changes no selection-null verdict on any of the five models.
- An Appendix E example in v1 did not match the scenario files (ANOMALIES process ledger, 2026-10-15).

Escalated to the author with the diff: the abstract's trimmed closing clause; the §9.1 heading in the
keeps-the-gate sense; the per-rollout table's release location (resolved 2026-10-16: Zenodo,
10.5281/zenodo.23425333).

## What the next result changes (ST line, pre-registered 2026-10-18; `papers/ST_PLAN.md` v0.2.7)

The ST line turns the KDG instrument into an intervention target. Hypothesis of record (ST_PLAN §0):
a post-training stage that holds the model's moral reference live during deliberation narrows the
judgment–action gap on held-out scenarios, without a prompt-level reminder, where behavioral
fine-tuning on the same data does not. Co-primary outcomes: ΔP on `H1_held` (brake-engaged subset,
baseline an independent `none` re-run, Amendment 1) and ΔE on the panel at dose 0. Inference-time
steering appears once, as a localization gate (§5.0), never as the deliverable.

Positive voice, pre-data (move 7): *the program now has a measured failure to repair, deliberation
moving models off a protective departure toward a harmful written rule (GPT-OSS-20B −0.224 on the
forced dose-0 vs sampled-action instrument, n = 258; the carriers −0.076 and −0.050 under forced
reasoning), a held-out split built before any training, and a screen whose power was computed from
that failure's own variance (MDE 0.075 at 132 × 4) before the pod.*

| cell | branch | thesis edit |
|---|---|---|
| T1.0 screen (G1), OLMo-3-7B-Think | Amendment 5 (i) and (ii) both hold | development model is OLMo-3-Think; the stage-checkpoint ladder (Tier 2) can locate where the brake is installed on the same lineage |
| | either misses | GPT-OSS-20B is the development model; OLMo-3 keeps the Tier 2 checkpoint role; "the brake on a 7B open recipe" is not claimed |
| T1.0, the 27–30B pair | one has a dose-0 regime and a CI-separated brake | it is the second family, with both co-primaries |
| | only Muse remains and fails dose-0 | the second family carries ΔP on H1 alone; the transfer sentence is scoped to H1 |
| T1.0, any model | parser disagreement > 5% or truncation-limited | ANOMALIES entry before G1; the brake of record is stated with both parser columns |
| T1.5 (G3) | §8 rows | the headline is the adapter-vs-adapter difference on held-out sets; every §8 branch is publishable (§8 table) |

Standing claim (pending) and its rival:

| claim | strongest counter-reading | separating experiment | status |
|---|---|---|---|
| an outcome-free representation objective (`L-anchor` or `L-self`) restores protection on held-out scenarios where DPO on the same traces does not | the objective teaches rule-breaking, not care (it moves every rule-protected choice) | the specificity set (60 panel scenarios where the rule is right; bar ±0.03) and panel ΔE with the twin at rest (§6) | NOT SUPPORTED on GPT-OSS for the anchor-axis objective (T1.5, 2026-10-22: `L-anchor` moved its target, not the action; no ΔP detectable at \|ΔP\| ≳ 0.066); `L-self` untested (ST-A14); the counter-reading landed on DPO instead (`L-beh` non-specific) |
| the KDG H1 instrument with its specificity and pressure guards separates a repair from a shaping round that installs rule-breaking (the partner deliverable's regression gate) | the guards are blunt: any large policy shift trips specificity | an adapter that raises protection on H1 and passes specificity (none yet), and Phase 2's detection deliverable | DEMONSTRATED ONCE (T1.5: `L-beh` +0.445 on H1_held, +0.192 on rule-right items, ΔE flat) |
| a direction at `clause_end`, block 12 is a training target for protection on GPT-OSS (gate, coherence, retention and specificity) | the specificity failures of −V_harm and V_qwen (+0.089, +0.093) are what any vector of this size does to borderline items (ST-A18) | the D-lite's `spec_fill` and `rr_fill` at both α (Amendment 29) | NOT SUPPORTED at this site and these doses, in two families: every gate-passing direction fails specificity (GPT-OSS −V_harm and V_qwen; Muse V_qwen), and a random vector does not (GPT-OSS −0.006 / −0.011; Muse +0.015); `care_harm` on Muse (rule-side specificity, below the fill margin) is the open lead (ST-A20) |

Conjunction added under watch: **ST effort of record × KDG-A28/A29.** The screen runs each model at
its shipped default effort (Amendment 2), while on GPT-OSS-20B protection on H1 rises from low to
high effort (24% → 31% → 37%, paired 59, unresolved) after a step at the onset of reasoning. If the
screen's step-downs move a model to a lower effort, its brake is measured where KDG-A29's reading
predicts the largest rule compliance; the two-point descriptive in the screen record is the first
cross-model read of that dose curve.

### ST G1, partial (2026-10-18) — what changed

Positive voice first: **the development model is GPT-OSS-20B, the model whose deliberation brake
on H1 is the largest the program has measured (−0.224 on the forced dose-0 vs sampled-action
instrument, n = 258), and the open 7B reasoning recipe is priced out by its own deliberation:
OLMo-3-7B-Think needs a median 1,860 tokens to answer an H1 scenario and leaves 16% unfinished at
4,096 (ST-09, ST-A2), so its brake could be screened only at a cost the envelope does not carry.**
The G1 row "either misses → GPT-OSS-20B" applied through truncation, before the brake criteria
could be read. The generation engine moves to vLLM (Amendment 7), with HF teacher-forced readouts
unchanged; an engine-parity check against §40's rollouts gates it. Open: the second family
(Qwen3.8-27B or Muse Glimmer 30B, screened on the relaunch) and ST-A1 (the bf16 padded check).

**Relaunch (2026-10-19), what changed.** vLLM reproduces §40's rollouts within its detection bar
(protective-share difference +0.012 [−0.067, 0.096], MDE 0.12, 61 scenarios), so the generation
engine of record moves without moving the instrument. Muse Glimmer 30B screens with a brake of
−0.31 [−0.39, −0.24] on 60 scenarios, descriptive until its dose-0 regime is validated (ST-A4,
one failed check that contains batch jitter). Qwen3.8-27B's screen rollouts are banked without
readouts (revision gate on a text-tower load, fixed). The bf16 readout floor appears in a dense
model too (Muse 0.375 nats; ST-A3 leaning toward a stack property rather than MoE routing), so
readouts of record are single-row on every model. Re-priced Tier 1: GPT-OSS development plus
either second family is about 15.6–15.8 A100-h cumulative of the 35-hour envelope.

**Rider pod (2026-10-19), what changed.** Positive voice first: **two of the three reasoning
families measured on the same screen instrument brake, and one deliberates the other way: Muse
Glimmer 30B's reasoning lowers its protective share by 0.31 [0.24, 0.39] against its own forced
dose-0 answer (dose-0 regime now of record; forward and generation agree to the bit when both read
one row), GPT-OSS-20B's by 0.22 (§40), and Qwen3.8-27B's raises it by 0.08 [0.02, 0.15] with 8 of
60 scenarios engaged, below its no-brake null (ST-A5).** The second family therefore has one
braking candidate measured end to end (Muse); Qwen is either the panel's non-braker or a case
where `enable_thinking=False` is a mode switch rather than a dose (ST-A5's discriminator). The
readout floor that drove single-row pricing was read at a tail position (letters at −45 nats on
GPT-OSS); at the dose-0 position the batch jitter is ≈ 0.25 nats on both dense models and HF vs
vLLM is ≈ 0.25, the size of one bf16 step of a large logit (ST-A3, fp32-head discriminator
priced). T1.3 on HF generation measures ~9 A100-h for GPT-OSS (ST-A6), the largest Tier-1 line;
spent so far 5.26 A100-h, GPT-OSS + Muse ~25 of 35.

**G1 (author, 2026-10-19).** Development GPT-OSS-20B, second family Muse Glimmer 30B: its brake is
of record and concentrated, 25 of 60 screened scenarios brake-engaged against a no-brake null
q95 of 14, with the engaged brake at −0.60; T1.6 can detect a recipe that closes about a sixth of
it (MDE 0.10 at the projected 42 held-out engaged). Criterion (i) is directional. T1.3's cost is
an engineering fact, not a property of the gate: constant vectors fold exactly into the MoE output
bias and run on vLLM, and the slow HF path is a fallback to the per-expert loop under torch 2.6
(G1 item 3).

**Rider pods 2–2c (2026-10-19).** Qwen3.8's protective deliberation survives the alternative dose-0
construction (B +0.126 [0.067, 0.184]), so the panel has two braking reasoning families and one
that deliberates the other way. Readouts of record are single-row with the bf16 head on both models
of record; torch 2.13 becomes the ST HF stack (inside the parity bar on both); T1.3 prices at
~1.2 A100-h on eager experts; the bias fold lands in vLLM (effects correlated 0.87 with HF's) but
fails its cross-engine bar, which was set from the dense models (ST-A7, author's call).


**T1.1 (2026-10-19), what changed.** Positive voice first: **GPT-OSS-20B's deliberation brake
replicates at eight rollouts on the full H1 set, −0.248 [−0.286, −0.213] over 264 scenarios
against §40's −0.224 on a different sample, and it is concentrated enough to train on: 45 of 132
H1_train and 38 of 132 H1_held scenarios are brake-engaged, 2.5 and 3.2 times their no-brake
nulls' 95th percentiles (18 and 12).** The independent `none` re-run on the 38 held scenarios moves
P_none by +0.022 [−0.035, 0.078], so the selection carries no detectable regression to the mean,
and the ΔP MDE of record is 0.08, half the `generalizes` bar. The residual bank (337,487 positions,
five layers, 9.1 GB), `chat_sigma`, 66 DPO pairs and the QLoRA path on Muse are in place for T1.2
to T1.4; spent 8.85 A100-h, envelope of record 28.09 of 35. The refusal-retention secondary
(Amendment 10) is uncalibrated on GPT-OSS: the model refuses the FL request twins under an
operator order as often as without one (0.933 vs 0.942), so the set selects nothing on this model
and its return is a G2 decision. Reaching that reading took a classifier fix: the shared refusal
classifier missed GPT-OSS's curly apostrophe and now folds typography itself (ST-A8; it changes no
label on any saved text). Conjunction under watch: GPT-OSS was already the program's
hardest-to-strip refuser (Paper 7); here it also refuses an explicit operator instruction, which
limits any refusal-loosening instrument built on operator framing for this model.

**Amendment 11 (2026-10-19), what the next result changes.** KDG §27 read GPT-OSS-20B's dose-0
pressure gap at −0.001 [−0.011, 0.008], so the development model may not carry the gap the panel
co-primary is built to close (ST-A9). The `none` panel on the ST instrument, moved to the T1.3 pod,
decides it. Unless E_none's CI excludes 0 with its point at or above ΔE's full-panel MDE (0.0189),
GPT-OSS's panel becomes a one-sided guard (no adapter creates a
gap, at the 50% subsample for every adapter), the specificity set is the "not a baseline shift"
guard there, and the positive ΔE claim rests on Muse if Muse's dose-0 panel (first Muse pod) shows
a gap, else on a Tier 2 carrier; the pitch's twin sentence is scoped to the family that carries E.
If GPT-OSS carries E by that rule, §6 stands as written. Verdicts name their scope:
`generalizes (H1)` on a non-carrier, `generalizes (H1 + E)` on a carrier. Envelope of record 32.34 of 35 (about
28.45 under the guard branch).

**T1.2, the constancy read (2026-10-19), what changed.** Positive voice first: **on GPT-OSS-20B the
anchor axis separates the model's own protect and rule deliberations within a scenario, at clause
ends, at every banked layer, and in the direction opposite to the anchoring hypothesis: rule
rollouts carry more moral-content load (−0.0024 [−0.0041, −0.0007], about 0.37 within-scenario SD,
40 scenarios).** The rule fixed the consequence before the number existed: the gate steers at
`clause_end`, and `L-anchor`'s expected branch is the null before any adapter trains (ST-A10). The
reading this favors is that `V_care` as typed measures moral salience, which rises when the model
argues itself into the rule; T1.3's `care` arm (+a at `clause_end`) is the causal test, and it
predicts ΔP ≤ 0 under that reading. Rivals (trace position; torn-scenario selection) have zero-GPU
discriminators on the bank.

**T1.3, the target-selection gate (2026-10-20; G2 read), what changed.** Positive voice first:
**on GPT-OSS-20B, steering along the moral-salience anchor axis at clause ends lowers protection
dose-dependently (the paired step from α 267 to 534 is −0.057 [−0.113, −0.007]), and mixing the
model's own judgment-frame state into its trace lowers it too (−0.091 [−0.177, −0.014]); the same
mix lowers the forced dose-0 answer (−0.068 [−0.099, −0.042]).** Both run in the direction the
constancy read predicted (ST-A10) and neither passes the gate (ΔP ≥ +0.10). Not shown: direction
specificity, since neither target separates from a norm-matched random direction at the coherent α
(differences −0.043 and −0.063, CIs include 0). The reading this favors (ST-A11, R_a) is that on
this model the moral-salience reference participates in the brake rather than opposing it; a
reversed-axis arm (~0.35 A100-h) separates it from generic perturbation.

**T1.3b (2026-10-20), what changed.** GPT-OSS-20B reads as a non-carrier on the ST instrument too:
its dose-0 E is −0.0011 [−0.0105, +0.0083] against KDG §27's −0.0013 [−0.0106, +0.0081] on another
stack (call A, printed in the session log; the rows were lost at the download, so the call is
provisional until the re-read on the next GPT-OSS pod). T1.5 is planned under branch A: on the
development model the panel guards against an adapter creating a gap, and the positive ΔE claim
rests on Muse if Muse carries E. Envelope of record 32.82 of 35.

**T1.3c (2026-10-21), what changed.** The call of record confirms it: GPT-OSS-20B carries no dose-0
pressure gap on the ST instrument (E_none −0.0011 [−0.0105, +0.0083], n 586, with the known-gap
control in band at 0.487 [0.462, 0.513]). On the development model the panel is a guard against an
adapter creating a gap; the positive panel claim belongs to Muse if its dose-0 panel shows a gap, or
to a Tier 2 carrier. The reversed-axis arm left ST-A11 unresolved by its rule (ΔP(−a) − ΔP(+a) =
+0.084 [−0.007, +0.177], n 20), pointing the same way as the dose step. T1.3 is complete at 18.14
A100-h spent; envelope of record 33.47 of 35.

**Foundation constancy read (2026-10-21), what changed.** Positive voice first: **in GPT-OSS-20B's
own deliberation, the clause ends of protective traces carry more care/harm, fairness and sanctity
content than the rule-following traces of the same scenario (care/harm +0.0068 [+0.0038, +0.0098],
0.77 within-scenario SD), while the anchor axis, a different construct near-orthogonal to all six
foundation axes, runs the other way.** Harm content carries part of the care/harm separation
(`V_harm` +0.0109; care/harm without it +0.0039 [+0.0009, +0.0066]). By Amendment 15's rule a
care gate arm on `care_harm` rides the T1.4 pod as target selection for a later adapter; the three
pre-registered adapters are unchanged. Correlational localization only; the arm is the causal test.

**T1.4 gate arms (2026-10-22), what changed.** Positive voice first: **at one steering norm, care
content and harm content move GPT-OSS-20B's action in opposite directions: +`care_harm` raises
protection relative to a matched random direction (+0.076 [+0.008, +0.146]) and +`V_harm` lowers it
(−0.079 [−0.147, −0.019]); the two differ by +0.155 [+0.090, +0.228].** That is the gate's first
direction-specific result. Neither arm passes the gate (care_harm ΔP +0.048 against the +0.10 bar),
so no fourth adapter trains. `V_harm` runs against its own correlational sign (ST-A12): protective
traces discuss harm, while adding harm salience pushes the model toward the rule. The three
pre-registered adapters are training now.

**T1.4 (2026-10-22), what changed.** The three pre-registered adapters exist for GPT-OSS-20B, each
cut by the budget fallback (58 / 83 / 89 of 264 steps). `L-anchor` met its objective on held-out
traces (anchor-axis margin 0.041 → 0.097, general KL 0.020) and nudged the reserved preference
toward rule traces (DPO margin −0.52), the direction ST-A11's salience reading predicts.
`L-beh` overfit its 62 pairs (margin 44): on reserved scenarios it protects whenever it finishes
and truncates half its rollouts at the cap (ST-A13). `L-self` moved the model off-distribution
(general KL 0.35) and strongly toward rule traces (reserved DPO margin −41) without meeting its own
objective (ST-A14). T1.5 tests all of this on H1_held. Spent 22.84 A100-h; envelope 35.33 of 36.

**`L-self` checks (2026-10-21), what changed.** The §5.3 objective as specified fails as an instrument
on GPT-OSS-20B: its judgment-frame targets (END_OF_PROMPT state, norm ~1,883) all sit above the
norm band of the `clause_end` residuals they are imposed on ([1,166, 1,590]), so the loss pulls the
residual's scale as much as its content, which fits the general-text drift L-self showed (KL 0.35).
By the pre-registered rule its H1_held generation is descriptive only, it gets no panel, and a
σ-floored retrain is deferred. The thesis-faithful arm is therefore not tested on this model in
Tier 1; the test needs a position- and norm-matched target.

**T1.5 (2026-10-22), what changed.** Positive voice first: **on GPT-OSS-20B the instrument tells a
repair from a shaping round that installs rule-breaking. Outcome-supervised fine-tuning on H1_train
(`L-beh`, DPO) raises protection on held-out H1 scenarios by +0.445 [+0.323, +0.562] and raises the
violating choice where the written rule is right by +0.192 [+0.131, +0.255], with the panel's
pressure guard flat (ΔE −0.004); the representation adapter (`L-anchor`) moves its own target at the
decision's clause ends past its threshold (anchor-axis margin 0.043 → 0.100) and leaves the action
where it was (ΔP −0.005 [−0.052, +0.041]).** Two dissociations, one per adapter: the action moved
without the anchor-axis content (`L-beh`'s margin falls 0.006), and the content moved without the
action. Verdicts: `L-beh` non-specific (capability fails as registered through longer traces, accurate
where it answers); `L-anchor` null for the anchor-axis target, with its transport control passed
and a positive control on the same instrument; `L-self` not tested (ST-A14). The hypothesis of
record is not supported on the development model for the one representation objective tested; the
§8 row that applies is "an adapter moves the specificity set", carried by the outcome-supervised
comparator. Engines: bf16 vs MXFP4 and HF vs bf16-vLLM `none` agree within ±0.07. Spent 7.10 A100-h on
the T1.5 pod plus the G3 follow-up on pod B. Conjunction under watch: ST-A12's `V_harm` (harm salience at clause ends moves protect and defer into the rule
option, replicated at n 45) and `L-anchor`'s null both say the action at this site reads something
other than the anchor axis; Phase 2's cells (V_rule, −V_harm, the cross-recipe direction) are the
test.

**ST Phase 2 on GPT-OSS (2026-10-22), what changed.** Positive voice first: **GPT-OSS-20B's
deliberation brake can be pushed back at the clause ends of its own trace: subtracting harm salience
there raises protection by +0.114 [+0.069, +0.164] (+0.143 beyond a norm-matched random direction),
passing the registered gate and surviving Holm over eleven arms; adding it pushes the other way
(−0.077), and the two differ by +0.191.** A second lever is larger and not yet admissible: the
direction separating Qwen3.8's protective from its rule-following deliberations, re-encoded in
GPT-OSS, aligns with GPT-OSS's own (cos 0.73, ceiling 0.94), is orthogonal to every typed axis, and
moves protection by +0.36 at a dose that lengthens traces past the coherence bar (step-down
pending). The written rule's presence in the input is the brake's dominant lever (removing the
sentence: +0.56); ablating its in-trace direction recovers about a tenth (+0.052, per-arm, not past
Holm). Reversing the anchor axis and adding care content do nothing detectable (MDEs 0.04-0.05).
Before any of these trains: refusal retention and specificity (Amendment 25), since less harm
salience is the canonical refusal-loosening direction. "The brake's own direction" is not claimed:
ST-A12's discriminators lean toward brake content without separating it from a refusal-shaped option.

**ST Phase 2 pod C on GPT-OSS (2026-10-22), what changed.** Positive voice first: **GPT-OSS-20B's
brake has three handles that move it protect-ward (L-beh, −V_harm, V_qwen), one observed failure
mode; the specificity instrument's random-vector reference is the open check.** The cross-recipe
direction is now admissible at the gate: V_qwen at half the dose stays coherent (length 1.28) and
raises protection by +0.193 [+0.134, +0.255] (+0.222 beyond a random vector), and with −V_harm it
survives Holm over eleven arms. Neither loosens plain refusal beyond the bar (−V_harm +0.0375,
V_qwen +0.0125, on a readout the refusal direction itself moves by +0.033). Both tilt rule-right
items toward the violating option by about +0.09 (bar ±0.03), on the same borderline items as each
other and as L-beh (r 0.61-0.75), so no direction is a target on GPT-OSS at this site, provisionally:
if a random vector tilts the same items as far (ST-A18), the instrument's bar moves to the difference
from fill and V_qwen is re-read under it (Amendment 29). "−V_harm retains refusal" is not claimed: it
is indistinguishable from the −refusal control.

**ST Phase 2 D-lite on GPT-OSS (2026-10-22), what changed.** Positive voice first: **three handles
(L-beh, −V_harm, V_qwen) tip the same rule-right items toward violating, by +0.09 to +0.19 where a
random vector tilts none, so GPT-OSS has no admissible target at this site and these doses.** The
directions (−V_harm, V_qwen, with −refusal) project onto each other at or below chance (ST-A29): the
overlap is in the items they tip, not in their geometry. The random-vector references settled both of pod C's coincidences
(ST-A18, R_b): a covariance-matched random vector at the same dose tilts rule-right items by −0.006
and −0.011 and loosens plain refusal by +0.004, so the arms' failures are theirs (+0.095 and +0.104
beyond fill). At this site and dose, −harmfulness and −refusal loosen plain refusal beyond a random
vector by the same amount (+0.033 vs +0.029 over fill) on the same rollouts: a causal overlap at the
margin, below the bar. Whether Paper 7's dissociation holds under its own protocol is untested here;
it is pod D's question. The next read is Muse's transfer test, now with −V_harm and `care_harm`.

**ST Phase 2 Cell 4, Muse's transfer test (2026-10-23), what changed.** Positive voice first: **Two
families, same result: Qwen3.8's protect-versus-rule direction moves the brake in GPT-OSS and Muse
(+0.193, +0.375) and tilts rule-right items toward violating in both (+0.093, +0.304); the lever and
the non-specificity travel together. No constant direction at `clause_end` is an admissible target in
either family. `care_harm` on Muse is the one direction with the opposite sign on rule-right items
(−0.048) while moving protection (+0.067); below the gate's fill margin, it is the open lead.**
Muse's brake reproduces its screen (B −0.307, 50 engaged), the typed axes keep GPT-OSS's geometry,
and the sweep picked the same relative dose (0.2× the median norm). −V_harm tilts Muse's rule-right
items (+0.147) without moving its brake (ST-A19); `care_harm`'s sign is ST-A20.

Program level (author, 2026-10-23). The partner deliverable for now is the regression gate
(detection). That is not the program's conclusion; the lever search continues. What Cell 4
establishes is that a fixed vector at one site carries protection and rule-breaking together, so the
lever has to condition on whether the rule harms, which is a weight-level job. L-beh's training set
had H1 outcome pairs only and no rule-right pairs; L-beh-2 (Amendment 31) tests whether that is
the source of its non-specificity.

**The `care_harm` ladder on Muse (2026-10-23), what changed.** `care_harm` on Muse moves protection
with dose (+0.067, +0.125, +0.188) while its rule-right read climbs with a random vector's; at 0.3× it
passes the gate with a tilt no larger than a random vector's (−0.001), a candidate that a fresh-seed
replicate with a dose-matched fill must confirm before it is called a target. `spec_fill`'s rule-right
tilt grows with dose on Muse (+0.015, +0.065, +0.097), so above 0.2× the absolute ±0.03 bar reads
the instrument's dose sensitivity, not the direction's. Under Amendment 31 ST-A20 is unresolved; the
replicate is Amendment 32.

**L-beh-2 on GPT-OSS (2026-10-23), what changed.** Adding 26 rule-right pairs to L-beh's 62 reverses
its non-specificity: L-beh-2 protects on held-out harm scenarios (+0.763) and follows the rule more on
held-out rule-right items in the panel's format (−0.093), with capability intact (+0.010); it fails the
panel guard as registered by the narrowest margin (ΔE +0.020, CI lower bound +0.0005). Whether it
conditions on rule harm or on scenario format is the open check: rule-right items in H1's format, and
rule-harm items in the panel's format. Its traces run at the unsteered length (median 465 against
445) and weigh harm and the affected person more than the unsteered model's (ST-A21, ST-A22).

**The format x harm matrix on GPT-OSS (2026-10-24), what changed.** The registered read is a split.
On items where following the rule harms someone, L-beh-2 protects in the panel's format as much as
L-beh does (+0.291 against +0.324). On rule-right items in H1's format the read sat on a floor (the
base model's P(violating) 0.013), which left the harm reading almost no room to show. L-beh-2 removes
84% of L-beh's rule-breaking there, but where the base model is tempted it still breaks the rule at
about 40% of L-beh's rate. Cell A′, screened for headroom, is the discriminator (ST-A23).

**D_rb and preventative steering on GPT-OSS (2026-10-24), what changed.** The positive control
failed, so L-beh-3 was not built and the line is closed (ST-A24). The two D_rb estimates agree almost
perfectly (cos 0.999), but D_rb is L-beh's whole adapter shift, nearly the same on every kind of item,
harm items included (cos 0.973 with D_h1); L-beh-2 moves 0.76 along it despite the opposite
rule-right tilt, so one direction at this site cannot separate the tilt, and the same check at the
decision site fails too. The contrast L-beh − L-beh-2 is a lead, recorded and not built.

**The `care_harm` replicate on Muse (2026-10-24), what changed.** The gate passes: `care_harm` at 0.3×
moves protection +0.115 against the unsteered model and +0.119 against a random vector at the same
dose, with fresh seeds and coherence intact. The specificity of record (the author's draw-aware
reference, ten draws) puts its rule-right tilt at −0.028 relative to a random vector, specific on its
point, but the read cannot resolve the ±0.03 bar (CI ±0.087): one random draw's tilt varies with SD
about 0.128, so the five-draw read that first said "discriminating" depended on which draws were
taken. Of the three directions tested on Muse (V_qwen, −V_harm, `care_harm`), it is the only one that
moves protection beyond a random vector with a rule-right tilt indistinguishable from zero at this
instrument's resolution (±0.087); T1.6 asks whether that survives distillation into weights.

**The GPT-OSS ladder (2026-10-26; ST-A29, Amendment 50), what changed.** Positive voice first:
**on GPT-OSS `care_harm` is specific and weak at block 12's `clause_end`.** At 0.4× it stays coherent
and moves protection +0.041 [−0.004, +0.087] against `none` (gate +0.10), with specificity −0.023
[−0.076, +0.033] against ten fill draws. Muse's `care_harm` is specific at its operating dose (−0.028)
and passes there. GPT-OSS's dose slope is about a fifth of Muse's (0.21 [−0.05, +0.55]): at this
site both families' `care_harm` is specific, and they differ in its strength. Registered branch
R_lever. The site sweep (below) extends it to blocks 15 and 18, and the distillation route is
Muse-only.

Referee pass.
1. *"The slope ratio's CI holds zero."* Conceded: "a fifth" is the point, and the CI runs to +0.55.
   The second derivation (the 0.2× to 0.4× rise) gives 0.25.
2. *"The doses are not comparable across models."* Each dose is in its own site's median norm
   (GPT-OSS block 12, Muse block 26), the unit both ladders were registered in. The ratio compares the
   two models relative to their own residual scale, not in absolute units.
3. *"Specific on both models at different doses."* Stated in ST-A29: at 0.4×, Muse's tilt sits above
   its fill (+0.079 [+0.023, +0.136]), so specificity is read at each model's own dose.

**The care-DPO replicate (2026-10-26; Amendment 49, ST-A27, ST-A30), what changed.** Positive voice
first: **care-DPO's profile reproduces under a fresh seed and fresh chosen traces.** care-DPO2 raises
protection on H1_held by +0.326 (care-DPO +0.338), tilts rule-right items away from violating (−0.046),
and follows the rule when it must choose (0.025). Replication is met on protection, specificity,
guard and forced choice, and not on capability: −0.040 [−0.075, −0.010] against the −0.02 bar, with
care-DPO's −0.005 the better of two draws (gap −0.035 [−0.075, 0.000]). The loss is wrong answers
more than truncation (6 of 9 losses; −0.026 [−0.051, 0.000] on answered items), with traces 190
tokens longer. On items that differ only in where the harm falls, both care adapters mostly defer
(0.82 to 0.95 of rollouts) and depart only where the rule harms, at 7%, against base-DPO's 35% and
81%. The registered branch, item set, stands and is uninformative: its comparator was predicted
harm-blind (author) and is harm-sensitive and non-specific (H\* +0.445). The forced choice on the twins
(Amendment 52) separates R_defer from R_item (ST-A30).

Referee pass.
1. *"The registered branch says item set; you are reaching for a rival."* The verdict of record is item
   set, stated as registered. The rival is written because the rule's comparator failed its own
   registered premise, which `instrument-calibration` treats as a validity event, not a footnote.
2. *"The commit-only shares condition on the output."* Conceded: they are exploratory and not a verdict,
   and they rest on 19 to 50 committed rollouts per cell. The forced choice on the twins removes the
   conditioning.
3. *"Capability failing is a replication failure, full stop."* It is, by item 5. The difference from
   care-DPO's capability is −0.035 [−0.075, 0.000], so whether the fresh seed costs more capability than
   T1.6's is not resolved on 200 items.

**The forced choice on the twins (2026-10-27; Amendment 52, ST-A30), what changed.** Positive voice
first: **when it must choose, care-DPO departs where the rule harms and follows the rule where
breaking it would harm, on items that differ only in where the harm falls.** care-DPO2 departs on
31% of forced twins and 2.5% of their originals (D_f +0.289 [+0.196, +0.392]); `none` departs on
0.7%. Registered branch R_defer. On cell A′-style items, care-DPO's protection is there; with a defer
option, it defers. base-DPO is harm-conditioned too (0.436 to 0.877), at a log-odds size that cannot be
told apart from care-DPO2's (+0.68 [−0.42, +2.12], exploratory). Both teachers produce harm-sensitive
adapters; only the steered one produces a specific one (rule-right departure 0.025 against 0.436).

Referee pass.
1. *"base-DPO's harm contrast is larger, so care-DPO is not the harm-conditioned one."* On the
   registered absolute scale base-DPO's is larger (+0.441 against +0.289), from a 44% floor of
   rule-breaking. On log-odds they cannot be told apart. The claim the data carry is specificity:
   care-DPO follows the rule where breaking it would harm, and base-DPO does not.
2. *"The first draw is at the bar."* care-DPO's D_f +0.147 has a CI that includes +0.15, so it would
   read unresolved under the same rule. The branch of record is care-DPO2's, as registered, and the
   first draw is reported beside it.
3. *"Forced choice is not the model's behavior."* With deferral available, both care adapters defer on
   most of these items. The forced cell shows where the commitment points; the deferral itself is a
   behavior to report, and the next finding.

**The GPT-OSS site sweep (2026-10-27; Amendment 51, ST-A29, ST-A31), what changed.** Positive voice
first: **on GPT-OSS, `care_harm` stays specific wherever it is placed and moves protection nowhere
it was swept.** At blocks 15 and 18 `clause_end`, the analog is coherent and specific (+0.003,
−0.015 against ten random draws). Its ΔP against `none` is −0.037 and +0.014, against a +0.10 bar.
Registered branch R_lever extended: the distillation route is Muse-only, and GPT-OSS's finding is
the family difference. The ranking's own numbers predicted the miss: the largest fixed-trace shift
was 0.33 nats against about 1.0 needed. At block 15 the fixed-trace steer and the sampled arm
disagreed in sign; its registered discriminator reads R_noise (ST-A31, closed). Steering at blocks
15 and 18 shortens the traces by about a fifth and cuts weighing words without moving protection.
AtP did not track the exact steer at this dose (Spearman 0.09).

Referee pass.
1. *"Two sites is not a sweep."* The sites were ranked over ten by a registered rule. Converted to
   ΔP, no eligible site's fixed-trace effect reaches a third of the gate at 0.4×, so a wider sweep at
   this dose would read the same. Larger doses were not read. The anchor axis failed coherence at
   534.6 at block 12 (ST-A11), and a random vector at block 15's 0.4× already fails the parse bar.
2. *"The ranking instrument failed, so the sites were wrong."* The exact steer agreed with the
   sampled arm at blocks 12 and 18 after the P(1 − P) conversion and disagreed at 15. AtP failed
   (0.09) and selected nothing.
3. *"The fill arm at block 15 was incoherent, so the gate's fill clause is unread."* The gate fails
   on its first clause (−0.037 against `none`), so the fill clause does not decide anything.

**Qwen3.8-27B pod 1 (2026-10-27; Amendment 54, ST-A32), what changed.** Positive voice first:
**a second family has a specific protective lever: `care_harm` at Qwen's block 32 `clause_end`.**
At 0.4× it moves protection on Qwen's 11 engaged scenarios by +0.236 [+0.153, +0.312] against
`none` and +0.305 against a same-norm random vector. Its rule-right tilt is −0.034 against ten
random draws. Registered branch P1-pass: pod 2 (steer-then-DPO) is licensed.
- **The claim is bounded.** It holds on the 11 engaged scenarios only, inside their 0.319 gap. On the
  whole H1 set Qwen's reasoning already raises protection (+0.055).
- **The two families now differ.** `care_harm` is a protective lever on Muse (block 26) and Qwen
  (block 32), and on GPT-OSS at none of three sites.
- **The prediction failed.** The fixed-trace prediction said −0.021 (ST-A32), a second miss for the
  instrument the §12 rule relies on.

Referee pass.
1. *"Eleven scenarios is too few."* The gate's MDE at that n is 0.113, and the effect is twice it,
   with both CIs above their bars. The registered floor was 10. The claim is stated on these 11 and
   no wider.
2. *"Regression to the mean: the engaged set was selected for low P_none."* The comparator is a
   fresh `none` arm on the same scenarios and seeds (0.489, not v1's selecting sample). The random
   vector moves the other way (−0.070), so selection does not produce the +0.236.
3. *"The prediction instrument failed, so the dose choice is suspect."* The dose was chosen by
   coherence alone (outcome-blind), and every rung was coherent. The prediction gated nothing; it
   ran beside, as registered.

**Qwen3.8-27B pod 2, the recipe on a second family (2026-10-29; Amendment 57, ST-A36, ST-A37), what
changed.** Positive voice first: **on a second family, steer-then-DPO gives adapters that protect on
held-out rule-harm items at no capability cost, and the steered teacher damps rule-breaking
relative to the unsteered one; neither adapter is specific under forced choice.**
- care-DPO_Q and base-DPO_Q raise protection on Qwen's 17 engaged H1_held by +0.449 and +0.551 (MDE
  0.16), with capability +0.010 and +0.030 and the panel guard intact.
- On the panel's rule-right items at dose 0, both follow the rule more than `none` (−0.064, −0.055).
- Under forced choice with reasoning on cell A′'s originals, both break the rule more: 0.314 and
  0.485 against `none`'s 0.115. Registered branch: **non-specific on Qwen**. care-DPO_Q departs as
  much more on the harm-reversed twins as on the originals (forced H\* +0.005), so it carries no
  harm condition beyond `none`'s.
- The steered teacher's adapter breaks rules less than the unsteered one's (forced originals
  −0.172 [−0.248, −0.096], twins −0.272) and protects less (−0.103 [−0.213, −0.007]): the same sign
  as Muse's ST-A27 (ST-A37).
- **The thesis's Muse sentence does not transfer as written.** "Only the steered one produces a
  specific one" held on Muse; on Qwen the steered one is less non-specific, not specific. The two
  instruments of record split on Qwen (ST-A36). The thesis wording goes to the author.

Referee pass.
1. *"39 thin pairs at 24 epochs: the adapters memorized 21 scenarios, so nothing here is a learned
   condition."* Conceded as the frame of every Qwen sentence: one run per arm, thin, overfit (DPO
   margins 20-52). The held-out reads still move (+0.449 on unseen H1_held) and capability did not
   fall (+0.010), so the adapters generalize something; what they generalize is the open question
   ST-A36 asks.
2. *"Cell A′'s originals are rule-right for Muse's registration, not for Qwen."* `none` already
   breaks the rule on 5 of the 51 at ≥ 0.5, so some items are contested on Qwen (R_item, ST-A36).
   The registered branch stands as read; the discriminator (generation on the panel's rule-right
   items, ~1.5-2 A100-h) separates item from readout.
3. *"The branch is fired by one of two instruments."* The rule was registered with "or" before data
   (Amendment 54 item 6): a rise on the forced originals alone is non-specific. Both readings are
   reported side by side, and neither is promoted over the other.

**ST-A36's generation read on Qwen (2026-10-29; Amendment 59), what changed.** Positive voice first:
**on Qwen the adapters' rule-breaking is item-bound: on the panel's rule-right items neither adapter
breaks the rule under reasoning, so the dose-0 and generation reads agree there, and the
rule-breaking pod 2 measured lives on cell A′'s forced items.** care-DPO_Q's P(violating) is 0.006
against `none`'s 0.008 (G −0.002 [−0.006, 0.000]); the adapters shift toward deferring (+0.048,
+0.088) instead. Registered branch R_item; the fresh-seed Qwen replicate with 16 teacher rollouts
follows (author).

Referee pass.
1. *"A read at the floor cannot show a tilt."* `none` breaks the rule on 0.008 of these items, so a
   tilt toward breaking would have room to show and does not; the adapters' movement goes to
   deferral. The floor is stated with the number.
2. *"Deferral hides the rule-breaking; the forced originals had no defer option."* On the 7
   two-option items every arm follows the rule in all 56 rollouts. Seven items is thin, stated as
   such.
3. *"R_item rests on one draw of each adapter."* Conceded; the replicate the author scheduled is
   the test, with a fresh seed and twice the teacher rollouts.

**The fresh-seed Qwen replicate (2026-10-30; Amendment 60, ST-A38), what changed.** Positive voice
first: **care-DPO on Qwen replicates on every registered criterion.** A second draw (fresh teacher
traces in both arms, seed 205, 49 rule-harm pairs, not thin) protects on held-out rule-harm items
(+0.691 [+0.625, +0.765]), keeps capability (+0.010) and the guard, follows the rule on the panel's
rule-right items under both readouts (−0.056 at dose 0, +0.004 under generation), and breaks the rule
on cell A′'s forced originals (+0.221 above `none`) with no harm conditioning beyond `none`'s (forced
H\* +0.044 [−0.061, +0.147]). Its protection is larger than the first draw's by +0.243 [+0.147,
+0.346] on the same scenarios, while the forced-original shift is unchanged (ST-A38).

Referee pass.
1. *"Two draws, and they differ in seed, traces and count at once."* Conceded for ST-A38's size
   difference: the pattern replicates; the protection's size is not attributed to the count. The
   separating draw (~4-5 A100-h) exceeds what the envelope has left (3.74).
2. *"The replicate inherited `none`'s rows from pod 2, so a drift in `none` would hit both."* Both
   reads compare against the same `none` rows by design (paired seeds), so the comparison between
   draws is exact; `none` itself was read once, as registered, and its base check passed (0.013).
3. *"The non-specific forced read is still one item set."* Yes: cell A′'s 51 originals. R_item
   (ST-A36) already places the rule-breaking on those items; the replicate confirms it there and
   on no other set.

### ST line thesis (final, 2026-10-30), positive voice

On GPT-OSS, every constant vector that moved protection (V_qwen, −V_harm) also broke rules
indiscriminately. On Muse, V_qwen did (+0.289 beyond a random vector); `care_harm`'s rule-right tilt
at its operating dose was indistinguishable from a random vector's, and the random vector's own tilt
grew with dose.

**On two families, steer-then-DPO yields adapters that protect on held-out rule-harm items,
keeping capability on three of four draws, and the steered teacher damps rule-breaking relative to
an unsteered one; harm-conditioned specificity under forced choice holds on Muse only** (author,
2026-10-29; "keeping capability on three of four draws" 2026-10-30).
- Protection, keeping capability on three of four draws: Muse care-DPO +0.338 [+0.264, +0.410] on 54
  engaged H1_held, capability −0.005 (care-DPO2 +0.326, capability −0.040, failing the bar); Qwen
  care-DPO_Q +0.449 [+0.338, +0.566] and +0.691 [+0.625, +0.765] on 17, capability +0.010 both.
- The steered teacher damps: on Muse, base-DPO is harm-sensitive and non-specific (it departs on
  44% of rollouts where departing harms and breaks rules on every rule-right read) while care-DPO
  stays within 0.03 of `none`; on Qwen, care-DPO_Q breaks the rule less than base-DPO_Q on the forced
  originals (−0.172 [−0.248, −0.096]) and protects less (−0.103 [−0.213, −0.007]; ST-A27, ST-A37).
- Harm-conditioned specificity under forced choice, Muse only: on Muse's harm-reversed twins care-DPO
  departs where the rule harms (replicate +0.289 [+0.196, +0.392]; first draw +0.147, at the bar),
  deferring on 82–95% of rollouts when deferral is available. On Qwen, care-DPO_Q departs as much on
  the originals as on the twins (forced H\* +0.005 [−0.103, +0.108]) and breaks the rule on the
  forced originals at 0.314 against `none`'s 0.115, while its dose-0 specificity reads −0.064.
  On the panel's rule-right items no arm breaks the rule under reasoning either (care-DPO_Q 0.006
  against `none`'s 0.008; ST-A36 resolved R_item): the dose-0 and generation reads agree on the
  registered set, and the rule-breaking pod 2 measured lives on cell A′'s forced items.
- Scope: Muse, two draws, one failing the capability bar (−0.040, with longer traces and an MMLU
  loss); Qwen, two care draws (32 and 49 rule-harm pairs, 24 epochs) with the same pattern on all
  seven of Amendment 60's criteria, while the damping on Qwen rests on one draw; the mechanism open.
- Reference only, the frontier exhibit (it feeds no steering decision; author's sentences):
  *H1:* "Neither frontier model shows the H1 brake at default; Sol is flat across 8× reasoning; Opus
  under forced deliberation moves from deferring to protecting (+0.069, all from deferral), carried
  by the quarter of scenarios where it thinks longest." *Panel:* "Under the panel’s pressure Sol
  takes the violating action in 8 of 4,688 replies (E at the floor); under an explicit operator
  instruction it takes it 90% of the time (g_band 0.900 [0.877, 0.921]) against the same judgment."

The KDG instrument's specificity read separates the two kinds of shift. It caught three adapters and
two vectors:
- adapters: L-beh +0.192; L-beh-2, format-bound, +0.308 on cell A′; base-DPO;
- vectors: −V_harm +0.089 and V_qwen +0.093 on GPT-OSS.

| claim | strongest counter-reading | separating experiment | status |
|---|---|---|---|
| a gate-passing steering vector distilled into DPO yields an adapter that conditions rule-breaking on harm (care-DPO on Muse) | the condition is the difference between the two item sets (T1.6's targets against cell A′'s pool), or one run's luck | the fresh-seed care-DPO replicate on matched items (read 2026-10-26); the forced choice on the twins (ST-A30, ~1.0 A100-h) | standing (two draws, one family; R_defer, Amendment 52): on matched items care-DPO2 departs where the rule harms when it must choose (0.025 to 0.314) and defers when it can; capability caveat: care-DPO2 fails the capability bar (−0.040 [−0.075, −0.010], longer traces, an MMLU loss; care-DPO −0.005). Second family (Qwen, 2026-10-29): non-specific under forced choice (care-DPO_Q forced originals 0.314 against `none`'s 0.115, forced H\* +0.005), dose-0 specificity −0.064; ST-A36 resolved R_item (the split is the items: under reasoning no arm breaks the rule on the panel's rule-right items); the fresh-seed Qwen replicate (49 pairs, 2026-10-30) repeats that pattern on all seven criteria. **Final status: standing on Muse; on Qwen protection holds and the harm condition does not** |

Referee pass on the thesis.
1. *"Two registered labels read otherwise."* T1.5's ladder counts care-DPO's protective −0.054 as
   non-specific by magnitude, and Amendment 41's format class joined two outcomes (the fork). The
   thesis states what each read shows: rule-breaking within 0.03 of `none`'s on rule-right items,
   protection up on rule-harm items. ST_PLAN's T1.6 outcome states the labels beside the verdict.
2. *"40 pairs at 24 epochs is a memorized teacher, not a learned condition."* Conceded as a caveat.
   A second draw with a fresh seed and fresh chosen traces (38 pairs) reproduces the protection, the
   specificity and the forced choice, and departs where the rule harms on twins it never saw; it
   costs more capability (−0.040). On Qwen, two draws (32 and 49 pairs) protect without a capability
   loss and carry no harm condition under forced choice: the memorization reading does not explain
   the difference between the families.
3. *"The instrument 'caught' what it was tuned to catch."* The specificity set and its bar were
   registered before every read here (T1.5, Amendment 31). care-DPO passed the same instrument that
   flagged the other five.

Referee pass on the whole ST line (close-out, 2026-10-30; one pass over ST_PLAN, CLAIMS, ANOMALIES
and this section).
1. *"One H1 family carries everything."* Conceded as the line's scope. Every claim is stated on
   approval-gate scenarios (H1) and the KDG panel; a second H1 family is a Tier 2 cell. The rule-right
   reads use the panel's own items and cell A′'s, which is how ST-A36 found that the Qwen split is
   item-bound.
2. *"Sixty amendments is a garden of forking paths."* Every amendment was committed and pushed
   before the computation it licensed (two process-ledger exceptions, each recorded with its times);
   every post-hoc fork is labeled and reported under both choices (Amendments 38 and 43); branches
   were written before data, and the claims carry the registered branch with the descriptive reads
   beside. The thesis sentence was restated once (Amendment 55, the original struck) after the
   representation objectives failed at T1.5, and is stated with that history.
3. *"Thin, overfit adapters at 24 epochs."* Conceded for the size of every effect: Muse 40 and 38
   pairs, Qwen 32 and 49, DPO margins 13-75. The held-out items, the twins never trained on, the
   held-out capability selection and the second draws are the answer for the direction of each
   effect, not its size (ST-A38 shows protection's size moves with the draw).
4. *"The specificity instrument has a blind spot."* ST-A36 tested exactly that on Qwen and found the
   dose-0 and generation readouts agree on the registered set; the forced-choice rule-breaking lives
   on cell A′'s items. The instrument is not shown to miss reasoned rule-breaking, and the forced
   items are now part of every adapter read.
5. *"Two families, opposite answers on the harm condition, is no claim."* The two-family claim is
   protection on held-out items, which holds on both (four draws; capability is kept on three, and
   one Muse draw fails the bar at −0.040); the harm condition is
   stated as Muse-only, with Qwen's forced read beside it. Why the families differ is open (the
   teacher's traces are less likely under the model on both, ST-A37; the pair count moves protection
   on Qwen, ST-A38), and no sentence claims a mechanism.

**T1.6 and its cells (2026-10-25), what changed.**
- The teacher, not the recipe, decides whether a DPO adapter's rule-breaking tracks harm. With the
  recipe held fixed, swapping `care_harm`-steered chosen traces for unsteered ones turns a
  non-specific adapter into one that follows the rule on rule-right items in generation (+0.005) and
  under forced choice (0.022).
- Rule avoidance (R_d) is ruled out. care-DPO's order on rule-right items is defer, then follow,
  then break. Its smaller protection on its targets (+0.338 against +0.863) is the price.
- The action split and parse rate are now reported for every adapter read.

**§13 (2026-10-25), what changed.** The overlap read has no valid positive control.
- The first three controls are void, a version mismatch: GPT-OSS's weights predate the 2026 Model
  Spec.
- On the 2025 revision, no GPT-OSS advantage is detectable (Δu −0.15 [−0.34, +0.07]). The
  within-model contrast, descriptive and not blind, points the same way.
- What the site shows is that principle-instantiating text from either document sits in the moral
  subspace as fully as the moral-family sources do, and that the documents' specific content
  separates: the Model Spec's sits at the null median, the constitution's in the null's upper tail
  (98th and 93rd percentiles), in both models.
- The overlap read stays unused on `C_partner`. A date check in the test suite now guards every
  document-model premise.

**Cell A′ on GPT-OSS (2026-10-24), what changed.** L-beh-2's protection transfers across formats; its
rule-following does not. It halves L-beh's rule-breaking in H1's format (+0.308 against +0.618)
without removing it, so its specificity on the panel set was partly format-bound.

**The family read (2026-10-25), what changed.**
- The labels of record are the author's 20: reliability 0.75 with the subagent, against the 0.8 bar.
- C13A-026's construction label is flagged as inconsistent with its document.
- Free rating carried a check-first prior that the documents' conflict passages do not: the
  author's own quote-and-classify marks side with departure on four of five such items.
- Labels are set by quote-and-classify from here on, `C_partner`'s included.

## MB line: MoralBench replication (opened 2026-10-30, `MB_PLAN.md` v0.5)

**Thesis at MB-G0 (2026-10-30), positive voice.** The program now holds MoralBench pinned and
hash-verified at commit `8410044` without vendoring any of it. It also holds a degenerate-policy
ladder for the instrument: on every set the best fixed letter scores inside the paper's five-model
range (always A 58.2 on MFQ binary against 49.9–58.5; always B 52.0 on MFV binary against
44.4–52.8). Two paper cells lie outside what the pinned key can produce (MB-A2, MB-A3), and the MFV
comparative keys fit no single human score per statement (MB-A4). The instrument's reading on
models is not yet measured: G0 is an audit of the key, not of any model.

| claim | strongest counter-reading | separating experiment | status |
|---|---|---|---|
| MB-G0: a fixed letter policy reaches the paper's model range on all four sets (exact, from the key) | R_content: models answer item by item, and the closeness to fixed policies is a coincidence of the key sums | G1 counterbalanced cell: item selectivity `S` and letter or label bias (`MB_PLAN` §4) | the ladder is exact; the reading is pending G1 |

**What the next result changes (MB-G1, pre-registered in `MB_PLAN` v0.5 §4-5).**

| cell | branch | thesis edit |
|---|---|---|
| match rule (T = 0.7, items fixed) | cells replicate | the local harness reproduces MoralBench, and the reproduction is usable as the Phase 2 baseline |
| | cells miss | diagnosis under amendment (template, system policy, sampling, parse), with the verdict under each choice |
| counterbalanced cell | `S` covers 0 (R_acquiescence) | MoralBench scores carry the A-rate or agree rate beside them in every Phase 2 use |
| | `S` excludes 0 (R_content) | the instrument reads item content within the ladder's band |
