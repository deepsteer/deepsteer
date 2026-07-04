# Instruments before verdicts: measurement discipline for causal claims about refusal and moral representation

*Methods note. Started 2026-07-02 (Direction-3 Amendment 10); drafted to full text 2026-07-03.*

A causal interpretability program (Directions D1–D3, plus the earlier Papers 1–7) kept
producing results that dissolved under an instrument check. Each of the failures below
looked like a finding first. This note collects the six that turned into portable methods
findings (`ANOMALIES.md`, A1–A6) and the estimator and intervention patterns the program
re-derived, and states each as a protocol other people can run. The scientific results
(what refusal reads, how it commits) live in the direction papers and the flagship draft;
this note is the portable methodology. Numbers here trace to the program's claim ledger
(`CLAIMS.md`); every scalar carries its detection bar or its control.

## Thesis

A causal claim about what a circuit reads or writes is only as good as the instrument that
measured it, and interpretability instruments fail in specific, diagnosable ways. The
discipline is four moves: **calibrate the instrument against a positive-control ladder,
certify it with an orthogonal cell, compute power before spending the pod, and state every
read-from verdict at a stated depth relative to the model's commitment.** Each section
below takes one instrument, shows the failure as it first appeared, names the tell that
caught it, gives the protocol, and states the check that certifies the fix.

---

## 1. Introduction: the instrument problem, and the discovery that motivated it

Mechanistic claims about model internals are claims about measurements. "Refusal reads the
harm percept," "this head writes the refusal direction," "moral judgment is orthogonal to
the refusal decision": each is a statement about a projection, a cosine, an ablation delta,
or an interchange patch. The measurement can fail in ways that produce a clean-looking
number. A covariance-matched null can saturate so that every direction projects like a
typical one. A per-head attribution can overshoot the true residual write threefold. An
interchange patch can go sign-chaotic because the outcome it reads is pinned at its ceiling.
A "reads-X" verdict can be an artifact of measuring past the layer where the model already
decided. None of these announce themselves; each returns a plausible scalar.

The discipline in this note was forced by one discovery. Across four architectures, the
decision site (the assistant-header or end-of-prompt control token where the refusal gate
and the judgment direction are defined) is a low-dimensional bottleneck. The participation
ratio there is 14.7 on OLMo-3-7B-Instruct, 8.6 on Qwen2.5-7B, 10.2 on Llama-3.1-8B, and
12.79 on GPT-OSS-20B (a 20B reasoning MoE at its harmony decision token). A 9-to-15
effective-dimensional channel, on every model tested, while content positions at the same
layers are full-rank-healthy (PR 40+/33+/35+). This is a substantive finding about where
the refusal decision lives, and it belongs in the flagship. But it is also the reason the
program's projection-fraction instruments failed: a positive control measured in a 15-slot
channel projects onto its own span *less* than a random direction does, so the instrument
had no discriminating power exactly where the interesting directions live. The finding and
the failure are the same fact seen twice. This note carries the validity protocol the
finding motivated; the flagship carries the finding.

(The bottleneck PR bar across the four architectures is **Figure 1**, which uses the D2
in-format-ladder value 10.2 for Llama, comparable to OLMo 14.7 and Qwen 8.6; the D3 C1
decision-token measurement for Llama is 13.5, a second position and harness. Both are
below 30.)

---

## 2. The decision-site instrument and its calibration ladder

Four anomalies converge on one object: the projection-fraction / cosine instrument used to
ask whether a direction of interest lives inside a subspace. A2 is the position where it
fails, A5(1) shows that position is architecture-general, A1 is the null that degenerates
underneath it, and A3 is the attribution decomposition that overshoots when the same
channel is read per-head. Each is stated as: failure → tell → protocol → certifying check.

### 2.1 A2 — band-below-null means the position is invalid, not that the direction is absent

**Failure as it appeared.** At the chat `final_pre_assistant` decision token on
OLMo-3-Instruct, the positive-control moral band came out at [0.40, 0.47], and the honest
covariance-matched null came out at 0.557. Held-one-out moral directions projected onto
their own span *below* where random directions projected. Read naively, any direction of
interest (refusal, judgment) that projected low there would read as "not in the moral
subspace." That reading is unsupported: the instrument had no discriminating power at that
position, so it cannot certify absence of anything.

**The tell.** The positive control sits below the null. Band-below-null ⇒ position-invalid
instrument. The moral band is not only a yardstick for "moral-adjacent"; it is a validity
check on the measurement position. The cause here is dimensionality, not an outlier
dimension (the top dim carries 0.2% of variance) and not a null that standardization can
rescue (the null stays 0.52 after z-scoring). The channel is simply narrow: participation
ratio 14.7. Three independent estimates converge on ~15 effective dimensions, the
second-derivation check: `√(3/14.7) = 0.45` against a null q95 of 0.557, and against the
R3 pairwise-|cos| null of 0.41–0.51.

**The protocol.** `participation_ratio` is a required type-block field on every extracted
direction, and any position with PR < 30 is flagged position-invalid for content
projection-fraction tests at extraction time (D2 Amendment 2). All three chat decision
sites (14.7 / 8.6 / 10.2) fall below the gate.

**The certifying check and the reframe.** Position-invalid does not mean uninterpretable
model. A projection-fraction test fails there, but a decision-*direction* cosine (R3) does
not: it is immune to the projection null. In a ~15-slot channel, refusal and judgment
directions occupy different slots at |cos| below even the low-dim random level, which reads
as active separation, not a weak-instrument artifact. Concretely, refusal-decision is
orthogonal to judgment-decision with no coupling detectable above |cos| 0.10 against a null
q95 of 0.41 on OLMo (0.32 vs 0.42 on Qwen, 0.08 vs 0.51 on Llama). The narrow channel is
the mechanism: refusal is written into a control-token bottleneck that moral content does
not reach (band-below-null there, healthy at content positions), so content-vs-decision
orthogonality is architecturally favored, and any comprehension-to-decision coupling has to
ride the attention heads writing into the bottleneck, a concrete anatomical target.

One reconciling sentence is required for prose. The bottleneck is position-invalid for
content projection-fraction tests (band-below-null) and position-valid for decision-direction
reads (R3 cosine, and the GPT-OSS refusal projection). GPT-OSS's decision channel is called
"position-valid (PR 12.79)" against a separate MoE PR sanity ceiling of 25; that ceiling is
not the content rule.

**Figure 2** is the calibrated ladder at this position: the moral band [0.40, 0.47] plotted
below the covariance null 0.557, the visual form of the tell.

### 2.2 A5(1) — the massive-activation outlier is position-dependent, so the bottleneck is clean

**Failure as it appeared.** Llama-3.1 carries a massive-activation outlier: dim 788 holds
32% of residual variance. The worry was that this outlier contaminated every geometric read
on Llama, including the decision-token cells.

**The tell.** The outlier's variance share is a *content-position* statistic. The decision
token is a different position and had to be checked there, not assumed from the global
number.

**The protocol and check.** At the decision-token channel where the refusal and judgment
cells actually read, Llama is clean: participation ratio 13.5, covariance null 0.148, which
barely moves to 0.114 under per-dimension standardization. The outlier lives at content
positions, not at the ~13-dim control-token decision bottleneck, which is clean and low-rank
across OLMo and Llama alike. So A2's "decision site is a narrow control-token channel"
finding is cross-model, and A1's standardization matters more at content positions than at
the decision token. This is why the A1 null degeneracy (next) and the A2 bottleneck are two
different failures at two different positions, not one confound.

### 2.3 A1 — covariance-matched nulls degenerate in massive-activation families

**Failure as it appeared.** The covariance-matched, rank-matched null (draw random
directions from `N(0, Σ̂)` of residual activations, project onto the rank-r subspace) is the
honest null used throughout Papers 5–7 and D1. On the instruct-model geometry it saturates:
R2 null q95 = 0.92 on Qwen and 0.36 on Llama, R3 pairwise-null q95 = 0.995 on Qwen and 0.90
on Llama, versus 0.26 on OLMo-3. At a saturated null every direction projects like a typical
direction, so the test has no discriminating power.

**The tell.** The null value itself is near its ceiling. The mechanism is the same massive
activations as A5(1): Qwen dim 458 = 59% of residual variance, Llama dim 788 = 32%, OLMo-3's
top dim = 1.4%. `Σ̂` is dominated by these dims, covariance-matched random directions nearly
all align with them, and they project ~1 onto any subspace with a component there. The same
dims collapse distinct raw mean-diff directions (Qwen ethics ≈ moral mean-diff |cos| = 0.90).
This is the known massive-activations / attention-sink phenomenon (Sun et al. 2024, *Massive Activations in Large Language Models*, arXiv:2402.17762;
Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453).

**The protocol.** Recompute directions and the null in a per-dimension-standardized space
(z-score by σ from a format/position-matched activation sample, sink tokens excluded), the
primary fix (D2 Amendment 1). The criterion-based robustness variant projects out each
dimension individually above 5% of variance. Behavioral results (ablation, judgment
accuracy) never use this null and are untouched; only geometric cells need the re-audit.

**The certifying check.** The clean instrument must give the same verdict raw and
standardized: OLMo, whose activations are well-conditioned, does. The quantitative
before/after is the participation ratio: raw PR = OLMo 43, Qwen 1.0, Llama 1.5 (one dim
carries essentially all variance for Qwen and Llama); after z-scoring, PR = OLMo 94, Qwen 39,
Llama 89. The raw PR ≈ 1 shows the collapse was near-total; standardization lifts Qwen and
Llama into a genuinely multi-dimensional space.

**A boundary case that names the residual limit.** On the R5 cell the two robustifications
*disagree*: standardization gives refusal 0.20 above controls 0.10 (strong-form false), while
top-k projection-out gives refusal 0.21 below controls 0.45–0.55 (strong-form true), and the
same split appears on Llama. When standardization and projection-out disagree, the subspace
is genuinely degenerate and needs a format or position change, not a null patch. The
in-format chat ladder (whose decision-site space carries no >5%-variance dim, so it is
outlier-free by construction) is the discriminator. This is the entry's own thesis applied to
itself: no single null repair resolves a genuinely rank-1 space.

**Scope of the fix.** A companion audit (Paper 6, zero-GPU) found its Qwen/Llama geometric
cells were never at risk: they use a permutation test (observed statistics ~0.01,
unsaturated) and raw projection fractions that are low and un-inflated (moral-subspace
projection fraction 0.104 OLMo / 0.127 Qwen / 0.071 Llama, mean|cos| 0.04–0.07), and the MFT
subspace was built on the base model whose foundation directions did not collapse onto the
outlier dim. The degeneracy is confined to the covariance-matched projection null applied to
D2's instruct-model `V_moral`. The general caution stands: covariance-matched nulls silently
degenerate in massive-activation families, the field's default Llama/Qwen panel.

### 2.4 A3 — reordered-norm architectures overshoot naive per-head OV attribution ~3×

**Failure as it appeared.** The Stage-1 write attribution on OLMo-3-7B-Instruct (sum of
per-head OV writes + per-layer MLP writes + embed onto the refusal direction, divided by the
true residual write at the read layer) came back at 3.05. The linear decomposition overshot
the actual residual write by 3×. The original gate (`recon ≥ 0.90`, one-sided) passed it,
because a floor only catches undershoot.

**The tell.** A reconstruction well above 1.0 on a decomposition that should sum to 1.0. The
mechanism is architectural: OLMo-2/3 use reordered (post-block) norm, applying
`post_attention_layernorm` to the attention output and `post_feedforward_layernorm` to the
MLP output *before* the residual add, with no input norm. The true residual write of the
attention block is `RMSNorm(Σ_h W_O^h z_h)`, not the raw sum; the naive OV decomposition
skips the norm, and since the raw block output has RMS above the norm's target it inflates
~3×. Pre-norm families (Llama, Qwen) write the raw block output to the residual and
reconstruct ~1.0 natively, which is why the overshoot never appeared in Papers 1–7 (they used
activations and directions, never OV decomposition).

**The protocol.** A two-sided gate `0.90 ≤ recon ≤ 1.10` (overshoot now fails), plus an exact
RMSNorm fold. RMSNorm is diagonal at a fixed token, `norm(x) = (γ / rms(x)) ⊙ x`, so
multiplying each pre-norm per-component write by the per-layer gain
`g = γ / sqrt(mean(x²) + ε)` recovers the exact residual contribution. The fold fires
automatically for reordered-norm models (detected via `post_feedforward_layernorm`) and is a
no-op for pre-norm models.

**The certifying check.** The fold is exact: unit-tested to 1e-9, and it brings the Stage-1
reconstruction from 3.05 to 0.9999, inside the two-sided band. It affects only the head
anatomy on OLMo-2/3 and other reordered-norm families (the un-folded numbers are inflated,
for example the MLP write fraction was 0.23 un-folded and 0.384 folded). It does not touch the
decisive causal cell, which reads the model's real forward pass with no decomposition. Per-head
OV / logit-lens attribution silently overshoots ~3× on reordered-norm models unless the block
norm is folded, a portable caution for a growing family (OLMo-2, OLMo-3, other post-norm
designs).

---

## 3. Verdict discipline

Three estimator and intervention patterns gate how a number becomes a verdict.

### 3.1 Ratio-of-ratios over MDE-crossing

Whether an effect clears its minimum detectable effect is power-dependent. Comparing two
effects by which side of the MDE each lands on is the overlap fallacy: it reads a difference
in power as a difference in kind. Compare two effects instead by a within-outcome ratio and a
bootstrap CI on the ratio difference (the estimator-traps trap-12 pattern).

**Worked case: the D3 `under_transfer` reclassification.** The first D3 headline was
`reads_non_vmoral_features` at n=11, resting on an absolute transport comparison (a
V_moral-restricted patch clears its MDE, a comparison patch does not). That absolute
comparison was necessary but not sufficient. Re-run at n=23 with a within-outcome
normalization, the honest verdict was `under_transfer`: the restricted patch moves refusal
less than the full patch, but the two do not sit on opposite sides of a categorical line. The
powered decisive cells (n=23 request-twins) are full→refusal −0.0833, V_moral-restricted→refusal
−0.0282, complement→refusal −0.0636, harm-rank-1→refusal −0.0261, random-rank-3→refusal
−0.0005, full→judgment +0.0459, restricted→judgment +0.0237, against a refusal MDE of 0.0238
and a judgment MDE of 0.0086. `under_transfer` was then itself superseded by the rank sweep
(`harm_saturating`, §6), but the estimator lesson is the one that recurs: the reclassification
from `reads_non_vmoral_features` to `under_transfer` happened because the ratio, not the
MDE-crossing, is the comparison of record. (The specificity claim that survives is stated as
a difference CI, not an overlap check: V_moral-restricted moves refusal more than a random
rank-3, Δ = 0.031, paired 95% CI [0.020, 0.043], excludes 0.)

A second instance from Paper 7: an early raw diff-of-means null (harm direction 0.44–0.49 of
residual norm) read as "harmfulness is not causally encoded," but that was a magnitude
artifact. Reply-inversion recovered the causal signal (Qwen2.5-14B-Instruct shift +17.4 flips
33%, Llama-3.1-8B-Instruct +3.0 flips 23%). Magnitude and residual-norm share are not causal
relevance; a causal readout is.

### 3.2 Power tables before the pod

Compute MDE(n) from measured within-condition variance before spending compute. If no
feasible n resolves the effect, the block is the instrument, not the sample, and the pod is
futile.

**Worked case: the Llama bounded-unresolved table.** The Llama refusal cells came back chaotic
(§3.3, §4). The temptation was a larger same-design re-run. The power table, built from
saved within-level arrays, said the re-run was futile: the ratio-of-ratios CI on the latched
denominator was [−2.3, 4.9], and no feasible n at that variance closes it, because the
denominator is saturated rather than noisy. The underpowered Llama `R_refusal_k` and
`R_judgment_{k>1}` cells were voided as denominator-latched, and the clean channel was
identified as the *reverse* (disengage) direction, not more samples in the forward one. An
afternoon of saved-array work prevented a pod. This is the general rule: saving per-pair /
per-rollout / per-head arrays by default keeps the power computation zero-GPU, so futility is
caught before the session, not after.

### 3.3 The orthogonal-cell certificate

When a causal readout comes back chaotic, root-split against an orthogonal outcome the same
intervention should move. If the orthogonal cell is coherent, the instrument is certified and
the chaos is a property of the read-out outcome, not a broken patch.

**Worked case: Llama content-swap.** On Llama the content-swap interchange patch produced
sign-chaotic refusal deltas (SD 0.31, median +0.029) against OLMo's clean −0.083, which read
as a broken instrument. The judgment cell is the positive control: the *same* patch moved
judgment coherently (CI excludes 0). So the patch works. The refusal chaos is saturation: the
boundary-violating twins sit at the refusal ceiling (baseline refuse 0.83–1.0), so the
decision-token refusal projection is latched and has no room to move, and the refusal-delta SD
grows with severity (0.296 → 0.352) as saturation deepens. OLMo's refusal moved because it was
weak (unsaturated). A causal readout run at a saturated outcome yields chaotic, sign-unstable
deltas that mimic instrument failure; the orthogonal cell tells the two apart.

---

## 4. Stimulus discipline

The operating-point rule: discrimination screens must bracket the point where the outcome
actually moves. A saturated outcome latches the readout. Use a severity ladder and a boundary
band (outcome ~0.5), and report the psychometric curve, not a single point.

This is where the readout itself can lie. Reasoning models defeat clean judgment readouts
(regex, final-answer, forced-logit), while a plain instruct model is clean on the same battery
(Qwen2.5-14B-Instruct: 24/24 harmless-safe, 24/24 harmful-harmful). The readout has to be
validated on the model class it is run on, at an operating point where the outcome is not
pinned.

### 4.1 A6 — a deliberation/prefill asymmetry is operating-point-confounded when the gate is a step

**Failure as it appeared.** On GPT-OSS-20B the reasoning-prefill deliberation cell (engage =
inculpating prefill, disengage = exculpating prefill) emitted a clean-looking asymmetry
`A = 1.0`: engage flips benign→refuse, disengage flips violating→comply 0/7. It read as
one-way early commitment, as if deliberation only ever pushes toward refusal.

**The tell.** `A = 1.0` with a bootstrap CI of width 0. The disengage arm is uniformly 0, so
every resample returns 1: a degenerate CI, not a precise estimate. The rule-of-three exposes
it directly, disengage 0/7 gives a 95% upper bound of ~0.43, not 0. The asymmetry is the same
dynamic-range confound as §3.3: the disengage arm was tested on violating items that already
refuse at baseline (at the ceiling), while the engage arm was tested on unsaturated benign
items (with room to move up). An asymmetry statistic that compares an arm-with-headroom against
an arm-at-the-ceiling measures the operating points, not the model. A companion trap in the
same run: the harm-separability commitment curve reads ~1.0 from the first trace bin, but that
measures when *harm is represented* (harmful and harmless traces differ from the start),
not when the *decision* is fixed.

**Why the usual fix is not enough here.** A5's fix was boundary-band twins (outcome ~0.5).
GPT-OSS's gate is a step: the severity ladder finds no unsaturated violating level (empty
boundary band, 5.6% of items in the mid-band). The operating point cannot be bracketed
behaviorally at the existing resolution, so more boundary-band stimuli do not exist to collect.

**The protocol (the de-confounder).** Replace the binary disengage flip with a graded
exculpatory prefill series (weak→strong) and read a continuous projection (the decision-channel
residual under each prefill onto the refusal direction) alongside the behavioral flip. The
graded projection registers sub-flip movement, so "no flip at maximum prefill" splits cleanly
into *reversible* (projection moves toward comply) versus *genuine downward-robustness*
(projection flat). Saturation can no longer masquerade as commitment. A pre-registered
band-existence check decides whether a finer ladder is even buildable: the per-item base-refuse
histogram is read as smooth (resolution-limited, build a finer ladder) versus bimodal (a step,
switch to the graded readout).

**The certifying check.** Under the graded series, GPT-OSS is a reversible reader: strong
exculpatory prefill flips ceiling-refusing violating items to comply 6/10, and the
decision-channel refusal projection moves monotonically toward comply in all 10 items
(frac_projection_moved 1.0, frac_monotone 1.0). The behavioral flip is the primary readout; the
monotone projection corroborates it (with a last-token caveat on the projection). The engage
direction is separately consequential: an inculpating-analysis prefill flips unsaturated benign
requests to refuse 7/7 (Wilson 95% [0.65, 1.0]), so the decision is not fixed before the trace.
The first-run disengage 0/7 was the saturation trap, now resolved. Report the behavioral flip
and the graded readout separately; the asymmetry statistic itself is not reported, because it
is uninterpretable when one arm sits at the ceiling.

This is the operating-point rule's reasoning-model instance: when the gate is a step and the
boundary band is empty, do not report the asymmetry; switch to a graded intervention with a
continuous readout that registers sub-threshold movement.

---

## 5. Depth discipline

A verdict about what a circuit *reads* must state the intervention depth relative to the
model's commitment. A patch at the read layer is post-commitment for an early-committing
model, so a "reads-X" or asymmetry claim measured there can be a read-layer artifact. Measure
at (and report) the pre-commitment coherent depth, and depth-match cross-model comparisons.

**Failure as it appeared.** The naive cross-model asymmetry at the read layer (layer 16) was
`A_Llama = +0.82`, engage-dominant and latch-like, against `A_OLMo = −0.20`, for a difference
of +1.03 (95% CI [0.16, 1.61], excludes 0). Read at face value, this said Llama's refusal has
a third distinct property, a hard directional latch that OLMo lacks.

**The tell.** Llama's disengage is coherent at earlier layers but not at the read layer.
Patch-layer sweep: Llama's disengage is coherent at layers 8/12/14 (−0.12 / −0.11 / −0.20,
CIs exclude 0) and incoherent at layer 16 (−0.014). Llama commits *before* the read layer;
OLMo's disengage is coherent at layer 16 (−0.62), so OLMo commits at or after it. Measuring
Llama's asymmetry at layer 16 measures it after Llama has already decided, so the +0.82 is a
post-commitment artifact. (The two layer-12 disengage numbers are two cells: the patch-layer
sweep above reads −0.11, the depth-matched full re-run reads −0.57, both coherent; the verdict
is the same.)

**The protocol.** Depth-match the comparison to the pre-commitment coherent layer, layer 12,
and recompute both models there.

**The certifying check.** At matched layer 12, `A_Llama = −0.28` (CI [−0.47, +0.03]) and
`A_OLMo@12 = −0.54` (CI [−0.81, −0.32]), a difference of +0.26, down from +1.03 at the read
layer. The apparent third property collapses: the asymmetry is a *consequence* of
early-commitment, not an independent latch. What survives depth-matching is the reads-axis
difference: at layer 12 Llama reads broad (`broad_moral`: R_refusal 0.85 ≈ R_judgment 0.79,
gap closes, harm-rank-1 only 0.59) while OLMo stays harm-keyed (R_refusal 0.43 < R_judgment
0.53, gap open). The read-layer +0.82 is retained only as a voided number with its replacement,
never as a finding (CLAIMS V-D3-8).

**Figure 3** is this depth collapse: `A_Llama` from +0.82 at the read layer to −0.28 at
matched layer 12, with `A_OLMo` −0.20 → −0.54, the depth-indexed exemplar.

---

## 6. Case study: the D3 refusal-reads-what / commits-how program

The Direction-3 program asked two questions of the refusal decision: *what* moral content it
reads, and *how* it commits. Its pre-registration trail (D3 Amendments 1–13, with the null and
position fixes pre-registered in the sibling D2 Amendments 1–2) is a sequence of caught
failures. Read in order, it is the methods note in miniature. The public amendment trail is a
credibility asset; it is cited from the flagship, not hidden.

- **Null degeneracy (D2 Amendment 1).** The instruct-model covariance null saturated on
  Qwen/Llama (A1). Fix: standardized recompute; OLMo unchanged raw→standardized certified it.
- **Position gate (D2 Amendment 2).** The decision site is a PR-14.7 bottleneck with the
  positive control below the null (A2). Fix: PR<30 position-invalid flag; V_moral re-typed as
  format-robust (invalid-position artifact at `final_pre_assistant`, band matches at the valid
  `mean_content` position), and the R2/G3 content-projection numbers re-typed as non-verdict.
- **Referee-pass hardening (D3 Amendment 1).** Before any asset was built, a referee pass
  re-typed the twin stimulus (request-twins carrying the judgment outcome, Δrefusal expected-flat),
  added a transport positive control to the decisive cell, made the head-score null channel-matched
  (mean/resample ablation, not zeroing), and added a behavioral-discrimination pilot screen.
- **OV overshoot (D3 Amendment 2).** Per-head attribution reconstructed at 3.05 on OLMo-3's
  reordered norm (A3). Fix: two-sided gate + exact RMSNorm fold, reconstruction 3.05 → 0.9999.
- **MDE-crossing headline (D3 Amendment 3).** The `reads_non_vmoral_features` verdict (n=11)
  rested on an absolute transport comparison. Fix: within-outcome ratio at n=23 → the honest
  `under_transfer`.
- **Under-transfer superseded (D3 Amendment 4).** A rank sweep replaced the point comparison:
  as k ∈ {1, 3, 8, 16}, R_judgment climbs 0.05 → 0.46 → 0.59 → 0.66 while R_refusal saturates
  0.01 → 0.31 → 0.26 → 0.27 at the harm-rank-1 level (harm_rank1_R 0.31), random-null ~0 at
  every rank. The one-knob model `R_refusal(k) ≈ min(harm_ceiling, R_judgment(k))` fits the
  plateau (k≥3) at RMSE 0.036, and PC1 (highest variance, purity 0.974, most harm-aligned at
  cos 0.35) is causally inert (rank-1 moves neither readout, 0.01 / 0.05), the A4 lesson that
  variance is not causal relevance. Verdict: `harm_saturating`.
- **GPT-OSS commit axis (D3 Amendment 5).** The Tier-1 run banked the position gate (PR 12.79),
  consequential engage deliberation (benign→refuse 7/7), and the first-run disengage 0/7 that
  looked irreversible.
- **Power table (D3 Amendment 6).** The saved-array power computation ruled the Llama same-design
  re-run futile before the pod (ratio-of-ratios CI [−2.3, 4.9] on a latched denominator).
- **One-root diagnosis (D3 Amendment 7).** The Llama chaos was diagnosed by a single root split,
  judgment-delta coherence, not a grab-bag of probes: the orthogonal judgment cell is coherent,
  so the refusal chaos is saturation (A5(2)).
- **Denominator-latched voids (D3 Amendment 8).** The Llama "reads beyond harm" hint (R_refusal
  0.44 vs harm-rank-1 0.14 at rank 16) was voided, its denominator saturated; the three branches
  re-entered unweighted and were resolved by the depth-matched `broad_moral` read.
- **Nomenclature + early-commitment (D3 Amendment 9).** Fixed engage = harm-add /
  disengage = harm-remove; defined the asymmetry statistic A; the patch-layer sweep gave the
  EARLY-COMMITMENT verdict (Llama disengage coherent at 8/12/14, incoherent at 16) and the
  read-layer cross-model asymmetry A_Llama − A_OLMo = 1.03 (later depth-re-attributed, §5).
- **Depth-indexed verdict (D3 Amendment 10).** The +0.82 read-layer asymmetry collapsed to −0.28
  at matched layer 12 (§5). This amendment started this note.
- **Harm-coextensive hardening (D3 Amendment 11).** The reads-broad verdict survived the rank-1
  harm-coextensive alternative: a single harm cue spans only 3.6% of the engage-driving moral
  basis (the rank-2/4 severity-ladder version is a stated extraction rider on unsaved contrasts).
- **Graded disengage (D3 Amendment 12).** The step-gate saturation trap was de-confounded: GPT-OSS
  is a reversible reader, violating→comply 6/10 with monotone projection in all 10 items (§4.1).
- **Confound-named hypothesis (D3 Amendment 13).** The n=3 categorical co-occurrence
  ("harm-readers reversible, broad-reader early-commits") was replaced by a falsifiable
  dimensionality→reversibility hypothesis with an explicit architecture confound: the read↔commit
  pairing is confounded by lineage/scale/tokenizer/reasoning-vs-instruct at three points, and is
  deconfounded only by varying one axis at a time. The measured two-axis table stands; its
  interpretation is a follow-on hypothesis, not an n=3 claim.

### 6.1 Reflexive discipline: the program audits its own published paper (P1)

The discipline turned on the program's own published work. Paper 1 (arXiv:2606.11375v1, 9 Jun
2026) stated a raw layer-depth fragility gradient as its abstract-level Finding 2: late layers
were reported as far more fragile than early ones, with a raw late/early σ* ratio up to ~14.7×
(CLAIMS records the range as 7–15×; Table 2 late 10.0 / early 1.8), plus a raw post-saturation
σ* decline from 18.3 to 4.7. A post-submission control (§4.4, RMS normalization) shows the
gradient is largely an activation-scale artifact: under RMS normalization the ratio collapses to
~1.8–2× (the residual ~2× is not claimed as a genuine gradient, since RMS controls scale not
covariance shape), the cross-checkpoint ordering fails at 8/37 checkpoints, and the post-saturation
decline is withdrawn (flat, ~13.8 → 15.0). The lesson is exact: raw σ* is valid within-layer (same activation
scale) but activation-scale-confounded cross-layer; RMS-normalize for any cross-layer claim.

Two things about *how* it was caught belong in this note. First, the confound surfaced at a
cold-boot (fresh-context) ledger audit, not in the warm working sessions that had produced and
re-read the result many times; the fresh-context reviewer's advantage is real, and mechanically
recreating it caught an abstract-level error. Second, it triggers a v2 erratum on a published
paper. A program that runs an instrument-calibration discipline on other people's panels has to
run it on itself; the same scale confound that A1 names in the covariance null (magnitude is not
the signal) is the one that inflated Finding 2. This is the reflexive instance, and it is the
reason the note leads with "instruments before verdicts" rather than presenting the direction
results as settled.

### 6.2 Claim hygiene: the W0 ledger as a worked example

Every number in the program traces to an anchored-sentence row in `CLAIMS.md`: if a draft states
a scalar that ledger does not carry, the draft is wrong until a row is added. The ledger also
carries an 18-entry VOID register in which superseded claims are retained *with their
replacements*, so they cannot re-enter prose as findings. The three the reader of a naive draft
would most likely resurrect are all there: the un-folded 3.05 head anatomy (replaced by the
folded 0.9999, V-D3-1), the n=11 `reads_non_vmoral_features` headline (replaced by `under_transfer`
at n=23, V-D3-2), and the `A = +0.82` read-layer asymmetry (replaced by the depth-matched −0.28,
V-D3-8). A separate set of number-integrity flags (CLAIMS NI-1 … NI-8) blocks specific *scalars*
whose value is still in dispute across documents, without blocking the verdicts (the shapes and
signs are robust; only a printed number waits on the flag). Voided results may be discussed as
methods lessons in this note; they are never findings in the flagship.

---

## 7. Checklist: the reusable protocol

The ship-blocker gates below are the portable form of the program's discipline. They are
appendix-form restatements of the companion skills.

**Before any projection-fraction / cosine geometric cell:**
- Record `participation_ratio` at the measurement position. If PR < 30, the position is
  invalid for content projection-fraction tests; report a decision-direction cosine (R3)
  instead, or move to a valid position.
- Check the positive-control band against the covariance null at that position. Band-below-null
  means the instrument has no discriminating power there; do not report absence.
- In a massive-activation family (any Llama/Qwen-class panel), inspect the null value for
  saturation and the top-dim variance share. Standardize (z-score, sinks excluded) and, as a
  robustness variant, project out each >5%-variance dim. Certify with a clean model
  (raw→standardized same verdict). When standardization and projection-out disagree, the space
  is genuinely degenerate; change format or position, not the null.

**Before any per-head OV / logit-lens attribution:**
- Detect reordered norm (post-block `post_feedforward_layernorm`). If present, fold the
  per-layer RMSNorm gain (unit-test the fold to ~1e-9). Gate reconstruction two-sided
  (0.90 ≤ recon ≤ 1.10); a one-sided floor misses overshoot.

**Before any patch / ablation / interchange / steering verdict:**
- Pre-register the intervention spec block (stimulus–outcome baseline matching, transport
  positive control, channel-matched specificity null, token-alignment rule, harness parity for
  the outcome classifier) before the run.
- Certify a chaotic readout with an orthogonal cell the same intervention should move; if the
  orthogonal cell is coherent, the chaos is saturation, not a broken instrument.
- Compare two effects by a within-outcome ratio + bootstrap CI on the ratio difference, never
  by which side of the MDE each lands on.
- State the intervention depth relative to commitment; depth-match cross-model comparisons to
  the pre-commitment coherent layer.

**Before any null / orthogonality / below-threshold verdict:**
- Build the calibrated ladder (floor → matched null → measurement → positive band) and re-verify
  every control's defining property in the current context.
- State the minimum detectable effect. A null without an MDE has no teeth; write the detection
  bar into the sentence ("no coupling detectable above |cos| 0.10 against a null q95 of 0.41,"
  not "dissociation").

**Before any stimulus screen:**
- Bracket the operating point with a severity ladder and a boundary band (~0.5); report the
  psychometric curve. If the gate is a step (empty boundary band), switch to a graded
  intervention with a continuous readout and report the behavioral flip and the graded readout
  separately. Validate the readout on the model class it is run on.

**Before the pod:**
- Compute MDE(n) from measured within-condition variance. If no feasible n resolves the effect,
  the block is the instrument, not the sample; do not spend the compute. Save per-pair /
  per-rollout / per-head arrays by default so the power computation and later statistics stay
  zero-GPU.

**Before any commit or draft:**
- Every printed scalar traces to an anchored claim-ledger row; voided numbers stay in the VOID
  register with their replacements so they cannot re-enter prose as findings.

## Referee pass

Three objections a methods reviewer raises on first read, answered or conceded.

1. *"Your anomalies come from one program on the field-default OLMo / Qwen / Llama / GPT-OSS
   panel. How do you know they generalize?"* Each anomaly is stated with its architectural
   trigger, not asserted as universal: A3 fires on reordered-norm models (detected via
   `post_feedforward_layernorm`), A1 on massive-activation families (the outlier dim's variance
   share is the diagnostic), A2 on control-token positions (the participation ratio is measured,
   not assumed). The claims are cautions keyed to a detectable structure, which is how a reader
   checks whether their model is in scope. Conceded: the panel is the field default, and the note
   demonstrates the failure modes rather than surveying their prevalence.

2. *"'The bottleneck finding and the instrument failure are the same fact' is rhetoric. Isn't the
   low PR just your `V_moral` being mis-constructed?"* The participation ratio is
   outcome-independent: a property of the activations at the position, not of any direction of
   interest, and it is low at the decision token while full-rank at content positions on the same
   model. The band-below-null is a positive-control property (held-one-out moral directions
   project below the null), and the decision-direction cosine (R3), immune to the projection null,
   reads as active separation. A mis-constructed subspace would not give a clean positive-control
   band at content positions and a below-null one only at the decision token. Conceded: the reframe
   is load-bearing, so it ships with its certifying cell (R3), not asserted.

3. *"You reclassified your own headlines after seeing data (n=11 → n=23, +0.82 → −0.28). Isn't that
   post-hoc?"* Each reclassification is a dated, committed, pre-registered amendment with both
   result branches written and publishable before the recompute, and the superseded number is
   retained in the VOID register so it cannot silently re-enter prose. The discipline is the
   mechanism that catches post-hoc drift; §6.1 shows it catching an error in the program's own
   published paper. Conceded: the program is the case study, so the note cannot claim independent
   replication of its own discipline; that is what the companion skills and external review are for.

## Figures and reproducibility

The note has three figures: (1) the bottleneck participation-ratio bar across the four
architectures (OLMo 14.7 / Qwen 8.6 / Llama 10.2 / GPT-OSS 12.79); (2) the band-below-null
calibrated ladder (moral band [0.40, 0.47] below the covariance null 0.557); (3) the depth
collapse of the cross-model asymmetry (`A_Llama` +0.82 at the read layer → −0.28 at matched
layer 12, with `A_OLMo` −0.20 → −0.54). Each figure ships with a regeneration script that reads
a committed CSV under the convention `papers/figure_data/mn_*.csv`
(`mn_bottleneck_pr.csv`, `mn_ladder.csv`, `mn_depth_collapse.csv`); the analysis
outputs are gitignored, so the committed CSV plus its script is the reproducibility contract for
every figure.

## Companion skills

`.claude/skills/instrument-calibration`, `intervention-validity`, `estimator-traps`,
`construct-audit`, `compute-ordering` are the operational encodings of the above: the
calibrated-ladder and MDE discipline of §2 and §7; the intervention spec block and orthogonal-cell
certificate of §3–§4; the ratio-of-ratios and extremum-bias rules of §3; the type-block and
position-validity rules of §2; and the zero-GPU-first, power-before-pod sequencing of §3.2 and §7.
