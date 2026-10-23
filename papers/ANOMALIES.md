# Methods anomalies (promoted findings)

Cross-paper ledger of measurement anomalies that turned out to be citable methods findings, not
bugs. Each entry states the observation, the mechanism, the fix, and who it affects.

---

## A1 — Covariance-matched nulls are unusable in massive-activation families without robustification

**Date:** 2026-07-01 · **Found in:** Direction-2 chunk-1 B1/B3 (`d2_decision_coupling`), cross-model
panel OLMo-3 / Qwen2.5-7B / Llama-3.1-8B (instruct).

**Observation.** The covariance-matched rank-matched null (draw random directions from `N(0, Σ̂)`
of residual activations, project onto the rank-`r` subspace) — the honest null used throughout
Papers 5–7 and D1 — **saturates** on Qwen2.5 and Llama-3.1: R2 null q95 = 0.92 (Qwen) / 0.36
(Llama), R3 pairwise-null q95 = 0.995 (Qwen) / 0.90 (Llama), versus 0.26 on OLMo-3. At a saturated
null every direction "projects like a typical direction," so the test has no discriminating power —
exactly the degeneracy d1 documented for eff-dim-385, here from a different cause.

**Mechanism.** Qwen2.5 and Llama-3.1 carry **massive-activation outlier dimensions** (Qwen dim 458
= **59%** of residual variance; Llama dim 788 = **32%**; OLMo-3's top dim = 1.4%). `Σ̂` is dominated
by these dims, so covariance-matched random directions nearly all align with them and project ~1
onto any subspace with a component there. The same dims dominate raw mean-diff directions, collapsing
distinct constructs (Qwen ethics≈moral mean-diff `|cos|` = 0.90). This is the known
**massive-activations / attention-sink** phenomenon (e.g. Sun et al. 2024, *Massive Activations in
Large Language Models*; Xiao et al. 2023, *Efficient Streaming LMs with Attention Sinks* — verify
exact refs before citing in a paper). OLMo-3's well-conditioned activations are why every OLMo-based
result in Papers 1–7 / D1 was clean.

**Fix (pre-registered, `d2_decision_coupling/PREREGISTRATION.md` Amendment 1).** Recompute
directions + the null in a **per-dimension-standardized** space (z-score by σ from a format/position-
matched `act_sample` with sink tokens excluded); primary. Robustness variant: **project out
dimensions individually > 5% of variance** (criterion-based). Legitimacy proof: the clean instrument
(OLMo) must give the **same verdict** raw→standardized. Behavioral results (ablation, judgment
accuracy) do not use the null and are untouched; only geometric cells need the re-audit.

**Affects.** Any direction-geometry projection-fraction / cosine-null computed on Qwen or Llama
family activations, including **Paper 6's cross-model geometric cells** (back-audit pre-registered)
and Direction-2 R2/R3/R5/R8 for Qwen/Llama. Does **not** affect OLMo-based numbers.

**Why it's a contribution.** "Covariance-matched nulls silently degenerate in massive-activation
families" is a portable caution for anyone building moral/refusal/concept subspaces and testing them
against activation-space nulls on Llama/Qwen — the field's default panel. It belongs in the methods
section, not a footnote.

**Paper 6 back-audit (2026-07-01, zero-GPU — rider d, CLEAN, no revision needed).** Paper 6's
Qwen/Llama geometric cells do **not** use the vulnerable null: `exp2_framework_geometry` uses a
**permutation test** (observed statistics ~0.01, unsaturated), and the refusal-morality geometry is
a **raw projection fraction** with no covariance-matched null — and those fractions are low and
un-inflated (`moral_subspace_projection_fraction`: OLMo 0.104, Qwen 0.127, Llama 0.071; mean|cos|
0.04–0.07). Paper 6 also built its MFT subspace on the **base** model, whose foundation directions
did not collapse onto the outlier dim. So the degeneracy is confined to the **covariance-matched
projection null applied to D2's instruct-model `V_moral`**; no Paper 6 number changes. Llama's
behavioral results were never at risk (they don't use the null). Paper 6 saved no `act_sample` for
Qwen/Llama, but since no covariance null was used there, no standardized re-audit is required (so no
`MISSING_ARTIFACTS` rider is filed).

**Post-standardization eff-dim (participation ratio).** The degeneracy's magnitude: raw PR = OLMo
43, **Qwen 1.0, Llama 1.5** (one dim carries essentially all variance for Qwen/Llama); after
per-dim z-scoring, PR = OLMo 94, **Qwen 39, Llama 89**. Standardization lifts Qwen/Llama from a
rank-1 effective space to a genuinely multi-dimensional one — the quantitative before/after of the
fix. (Measured; higher than a ~10–15 first estimate — the standardized space is richer than
expected, but the raw PR≈1 → the collapse was near-total.)

**Qwen is elevated in TWO cells — discriminator is the in-format ladder + the projection-out read.**
Qwen's refusal geometry sits high in both the chat R3 cell (|cos| 0.32, still dissociation, but
above OLMo's 0.10) and the raw R5 cell. On R5, the two robustifications **disagree**: standardization
gives refusal 0.20 > controls 0.10 (strong-form FALSE), while top-k projection-out gives refusal
0.21 < controls 0.45–0.55 (strong-form TRUE) — and the same disagreement appears for Llama. So R5 is
not resolvable by robustification alone; the **chat-format in-format ladder** (whose decision-site
space carries no >5%-variance dim, so it is outlier-free by construction) is the discriminator. This
is a worked example of the entry's thesis: when standardization and projection-out disagree, the
subspace is genuinely degenerate and needs a format/position change, not a null patch.

---

## A2 — The decision-site is a low-dimensional control-token bottleneck (band-below-null ⇒ position-invalid)

**Date:** 2026-07-01 · **Found in:** the D2 in-format ladder (`informat_ladder.py`), OLMo-3-Instruct.

**Observation.** The chat **`final_pre_assistant`** position (the assistant-header token — the
decision site where the refusal gate and judgment-decision direction are defined) has a
**participation ratio of 14.7** — a ~15-effective-dimensional channel — while content positions
(`mean_content`) are full-rank-healthy. There the positive-control moral band **[0.40, 0.47] sits
BELOW the covariance null (0.557)**: held-one-out moral directions project onto their own span *less*
than random directions do, so the projection-fraction instrument has **no discriminating power**.
It is **not** an outlier dim (top dim 0.2%) and **not** standardization-fixable (null stays 0.52).
Three independent estimates converge on ~15 dims: `√(3/14.7) = 0.45` ↔ null_q95 0.557 ↔ the R3
pairwise-|cos| null 0.41–0.51.

**The general tell (portable).** **Band-below-null ⇒ position-invalid instrument.** The A1
positive-control band is not just a yardstick for "moral-adjacent"; it is a **validity check on the
measurement position**. Any projection-fraction result at a position where the positive control
falls below the covariance null is uninterpretable, whatever the direction-of-interest does.

**Reframe (a finding, not a failure).** The decision site being a narrow control channel is the
*mechanism*. Stacked with A3 (refusal in a ≤q10-variance channel) and A5 (refusal does not
crystallize from a pretraining precursor, cos 0.155): the refusal gate is a fresh post-training
construction in a narrow control channel at a template-token bottleneck that moral content does not
reach (band-below-null there, healthy at content positions). Content-vs-decision geometric
orthogonality is therefore **architecturally guaranteed**, and any comprehension→decision coupling
must be carried by the **attention heads writing into the bottleneck** — a concrete anatomical
target for the causal (C1) follow-up. R3 (a decision-direction cosine, immune to the projection
null) reads *stronger* under this lens: in a ~15-slot channel, judgment and refusal occupy different
slots at |cos| below even the low-dim random level → active separation.

**Fix + guard.** `participation_ratio` is a required type-block field; positions with PR < 30 are
flagged position-invalid at extraction (`d2_decision_coupling/PREREGISTRATION.md` Amendment 2). The
D1 reasoning-extension band rung (GPT-OSS P2 vs a raw-pooled band) inherits the same cross-position
hazard and is being PR-audited + scoping-noted.

---

## A3 (ledger) — Reordered-norm architectures overshoot naive per-head OV attribution ~3×

*Ledger entry A3; distinct from the "A3" / "A5" calibration rungs referenced in A2's prose, which are
Phase-A calibration tasks, not ledger entries.*

**Date:** 2026-07-02 · **Found in:** Direction-3 (`d3_decision_anatomy`) C1 Stage-1 per-head write
attribution on `Olmo-3-7B-Instruct` (layer 16).

**Observation.** The Stage-1 reconstruction — sum of per-(layer,head) OV writes + per-layer MLP
writes + embed onto `r̂`, divided by the true `⟨resid_{L_ref}[decision], r̂⟩` — came back **3.05**,
i.e. the linear decomposition **overshoots the actual residual write by 3×**. The original gate
(`recon ≥ 0.90`, one-sided) **passed it**, because a floor only catches undershoot.

**Mechanism.** OLMo-2/3 use **reordered (post-block) norm**: `post_attention_layernorm` is applied
to the **attention output** and `post_feedforward_layernorm` to the **MLP output** *before* the
residual add, and there is **no input norm** (confirmed in `transformers` `Olmo2DecoderLayer` /
`Olmo3DecoderLayer`). So the true residual write of the attention block is `RMSNorm(Σ_h W_O^h z_h)`,
not the raw sum — the naive OV decomposition skips the norm, and since the raw block output has RMS
above the norm's target it inflates by ~3×. Pre-norm families (Llama, Qwen: norm on the block
*input*) write the raw block output to the residual, so they reconstruct ~1.0 natively; this is why
the overshoot never appeared before D3 (Papers 1–7 used activations/directions, never OV
decomposition).

**Fix (pre-registered LN-fold escalation, `d3_decision_anatomy/PREREGISTRATION.md` Amendment 2).**
(i) **Two-sided gate** `0.90 ≤ recon ≤ 1.10` — overshoot now fails. (ii) **Exact RMSNorm fold:**
RMSNorm is diagonal at a fixed token, `norm(x) = (γ / rms(x)) ⊙ x`, so multiplying each pre-norm
per-component write vector by the per-layer gain `g = γ / sqrt(mean(x²)+ε)` recovers the exact
residual contribution (`Σ_h contrib_h ⊙ g = norm(Σ_h contrib_h)`; unit-tested to 1e-9). Fires
automatically for reordered-norm models (detected via `post_feedforward_layernorm`); a no-op for
pre-norm models.

**Affects.** Only the **Stage-1/2 head anatomy** (which heads write refusal, what they read) on
OLMo-2/3 and any reordered-norm family — those numbers from the un-folded run are inflated and are
being re-run folded. Does **not** affect the **decisive causal cell** (interchange patching reads the
model's real forward pass, no decomposition) or any activation/direction result in Papers 1–7.

**Why it's a contribution.** Per-head OV / logit-lens attribution silently overshoots ~3× on
reordered-norm models unless the block norm is folded — a portable caution for a growing family
(OLMo-2, OLMo-3, and other post-norm designs). The fold is exact and cheap. It belongs in the methods
section alongside A1's null-degeneracy caution.

---

## A4 (ledger) — Variance and purity do not imply causal relevance; low-rank restrictions can be nonlinearly inert

**Date:** 2026-07-02 · **Found in:** Direction-3 (`d3_decision_anatomy`) C1 rank sweep on
`Olmo-3-7B-Instruct`.

**Observation.** In the nested moral-contrast PCA sweep, **PC1** — the top singular vector of the
moral-neutral content contrasts, carrying the most contrast variance, `subspace_purity = 0.974`, and
the single most harm-aligned component (`cos(d_harm, PC1) = 0.35`) — is **causally inert**: restricting
the interchange patch to rank 1 moves neither readout (`R_refusal(1) = 0.01`, `R_judgment(1) = 0.05`).
The causal signal appears only at rank 3. The one-knob saturation model
`R_refusal(k) ≈ min(harm_ceiling, R_judgment(k))` fits the plateau (k ≥ 3) to RMSE 0.036 but
**over-predicts rank 1 by ~4×** (measured 0.013 vs predicted 0.052).

**Mechanism (candidate).** The refusal readout is **nonlinear at low rank**: it needs a threshold
amount of the (distributed) harm direction — which spreads across PCs 1–4 (`cos` 0.35/0.25/0.20/0.19)
— before it engages, so no single high-variance component is a causal lever on its own. Variance
(where the contrast energy sits) and causal relevance (which direction the readout reads) are different
directions; purity (how much of the mean contrast a subspace captures) certifies the *basis*, not the
*causal lever*.

**Affects.** Any interpretation that reads causal importance off a component's variance share, its
probe purity, or its alignment with a target direction. In this program it is the causal counterpart
of the eff-dim caution (A1/A2): a high-variance or high-purity direction can be causally silent, and a
rank-1 restriction can under-transfer for a nonlinear reason, not an instrument-weakness reason.

**Why it's a contribution.** "Don't infer causal relevance from variance/purity/alignment; verify with
a restricted causal cell, and expect low-rank restrictions to be nonlinearly inert" is a portable
caution for the whole probe-then-patch workflow. The one-knob model turns the deviation into a
concrete nonlinearity candidate rather than noise. Belongs in the methods section with A1–A3.

---

## A5 (ledger) — Massive activations are position-dependent; and interchange patches die at outcome saturation (use the orthogonal cell as the instrument certificate)

**Date:** 2026-07-02 · **Found in:** Direction-3 C1 Llama-3.1-8B panel run + Amendment-6/7 diagnosis.

**Two findings from the same run.**

**(1) Massive-activation outliers are POSITION-dependent (strengthens A2).** Llama-3.1's dim-788 carries
32% of residual variance *at content positions* (A1), yet the **decision-token channel** where the
refusal/judgment cells read is **A1-clean** (participation ratio 13.5, covariance null 0.148 → 0.114
barely moves under standardization). So the outlier lives at content positions, not the ~13-dim
control-token decision bottleneck — which is clean and low-rank across OLMo **and** Llama alike. The A2
"decision site is a narrow control-token channel" finding is therefore cross-model, and A1's
standardization is less critical at the decision token than at content positions. Cheap follow-up if it
matters: a per-position top-dim-share profile.

**(2) Interchange patches lose dynamic range at outcome saturation — certify with an orthogonal cell.**
On Llama the content-swap patch produced **sign-chaotic** refusal deltas (SD 0.31, median +0.029, vs
OLMo's clean −0.083), which read as a broken instrument. The **judgment cell is the positive control**:
the same patch moved judgment **coherently** (CI excludes 0). So the patch works; the refusal chaos is
**saturation** — the operating-band violating twins sit at the refusal ceiling (baseline refuse 0.83–1.0),
so the decision-token refusal projection is latched and has no room to move (refusal-delta SD grows with
severity 0.296 → 0.352 as saturation deepens). OLMo's refusal moved because it was *weak* (unsaturated).

**The portable caution.** A causal readout run at a saturated outcome yields chaotic, sign-unstable
deltas that mimic instrument failure. Diagnose with a **coherence root-split against an orthogonal
outcome the same patch should move** (here: judgment); if the orthogonal cell is coherent, the
instrument is certified and the null is a **dynamic-range** property of the saturated outcome, fixed by
**boundary-band stimuli** (outcome ~0.5), not a new instrument. Power tables computed from within-level
variance say *before the pod* whether a re-run can resolve it — they said Llama's same-design re-run was
futile, and the diagnosis said why. Belongs in the methods section with the intervention-validity
operating-point rule.

## A6 (ledger) — Deliberation-prefill asymmetry is operating-point-confounded when the gate is a step; the graded projection readout de-confounds it

**Date:** 2026-07-03 · **Found in:** Direction-3 GPT-OSS-20B Tier-1 commit-axis run (Amendment 5/12).

**The trap (A7 recurring on a reasoning model).** The reasoning-prefill deliberation cell (engage =
inculpating prefill, disengage = exculpating prefill) emitted a clean-looking `A = 1.0` (engage flips
benign→refuse 7/7; disengage flips violating→comply 0/7). It read as one-way early-commitment. It is
**saturation**: the disengage arm was tested on violating items that already refuse at baseline (7/7 at
the ceiling), while the engage arm was tested on unsaturated benign items (room to move up). An
asymmetry statistic that compares an arm-with-headroom against an arm-at-the-ceiling is the same
dynamic-range confound A5(2)/A7 names. `A = 1.0` with a bootstrap CI of **width 0** is the tell:
disengage is uniformly 0, so every resample returns 1 — a degenerate CI, not a precise estimate
(rule-of-three: disengage 0/7 → 95% upper ≈ 0.43, not 0). And the harm-separability commitment curve
(~1.0 from trace-bin 1) measures when *harm is represented*, not when the *decision* is fixed —
harmful/harmless traces differ from the start regardless.

**Why the usual fix (boundary-band stimuli) is not enough here.** A5's fix was boundary-band twins
(outcome ~0.5). GPT-OSS's gate is a **step** — the severity ladder finds no unsaturated violating level
(empty boundary band). So the operating point cannot be bracketed behaviorally at the existing
resolution.

**The de-confounder (Amendment 12).** Replace the binary disengage flip with a **graded exculpatory
prefill series** (weak→strong) and read a **continuous projection** (the decision-channel residual under
each prefill onto the refusal direction) alongside the behavioral flip. The graded projection registers
sub-flip movement, so "no flip at maximum prefill" splits cleanly into *reversible* (projection moves
toward comply) vs *genuine downward-robustness* (projection flat) — saturation can no longer masquerade
as commitment. A pre-registered band-existence check (per-item base-refuse histogram: smooth →
resolution-limited → finer ladder; bimodal → step) decides whether a finer ladder is even buildable.

**The portable rule.** A deliberation/prefill asymmetry is only interpretable when both arms sit off the
outcome ceiling. When the gate is a step (no boundary band), do not report the asymmetry — switch to a
**graded intervention with a continuous readout** that registers sub-threshold movement, and report the
behavioral flip and the graded readout separately. Belongs in the methods note beside the
operating-point rule (pattern 4) as its reasoning-model instance.

---

## A7 (ledger) — Coarse-grid critical-noise σ* produces a spurious sign-flip under naive RMS-normalization (a censoring artifact, not a scale reversal)

**Date:** 2026-07-05 · **Found in:** Paper 1 §4.3 data-curation confound re-check (`confound_rms.py`,
OLMo-2-1B, 3 LoRA conditions, 400 steps; adversarial-review follow-up).

**Observation.** A reviewer asked whether §4.3's "declarative = most fragile" (raw σ*: declarative
7.38 < narrative 9.12, general 8.69) survives scale-matching, since declarative has the lowest mean
activation RMS (2.19 vs 2.41 / 2.43). A re-implementation added an RMS-normalized arm (`σ*_rms`:
normalize each layer to unit RMS, re-run the same fragility sweep on the fixed absolute grid). σ*_rms
*reversed* the ordering (declarative most robust 4.75 vs narrative 3.00), which read as a clean "the
ordering is a scale artifact" flip. It is not — the flip is an estimator artifact.

**Mechanism (R_b confirmed zero-GPU; R_a rejected).** Three scale-matches disagree, diagnostically:
naive aggregate σ*_raw/mean-RMS = 3.37 / 3.78 / 3.57 (no flip); proper per-layer σ*_raw/rms averaged
= 3.30 / 4.10 / 3.80 (no flip); the script's σ*_rms arm = 4.75 / 3.00 / 3.88 (flip). The flip lives
only in the σ*_rms arm. The per-layer forensic shows why: on the coarse grid {0.1,0.3,1,3,10} with
cap-at-max, σ*_rms pins **14/16 layers at the 3.0 floor** for every condition, and the "flip" is
driven by 4 scattered censored layers (σ*=cap) for declarative vs 0 for narrative — noise at the grid
floor, no dynamic range. σ*_raw is the mirror image (10–14/16 layers censored at the 10-cap).
Decisively, the raw ordering that defines §4.3 is driven by **early layers L0–L5, where declarative
breaks at σ=3 while narrative/general are censored — and there the per-condition RMS is nearly
identical** (0.78/0.77/0.77 … 1.86/1.93/1.94). Declarative's lower RMS sits at **deep** layers
L9–L15, which are censored (no fragility signal) in raw. So the reviewer's "deep-layer RMS drives the
fragility" mechanism does not drive the raw ordering, and the σ*_rms sign-flip does not survive a
censoring-free estimator.

**Reading.** R_b (artifact) over R_a (real per-sample/per-layer heterogeneity): the sign-flip is a
coarse-grid censoring artifact. Under any censoring-free scale-match (per-layer SNR ratio) §4.3's
ordering is preserved; the early-layer, RMS-matched fragility difference weakly *supports* it. The
earlier escalation that the ordering "flips under scale-matching, vacating §4.3" is **withdrawn**.

**Fix / definitive check (pre-registered; GPU held).** The coarse 5-point geometric grid with
cap-at-max censors both arms. Censoring-free estimator: for a frozen linear probe with isotropic
Gaussian test-noise, σ* has a closed form in the clean-margin distribution (no grid, no seeds), or use
a finer grid. Definitive §4.3 verdict = the exact published cell (1000 steps) with (i) the
analytic/fine-grid σ*, (ii) BOTH conventions (raw and per-layer-SNR), (iii) a paired bootstrap Δ-CI on
σ*(narrative) − σ*(declarative), seeds fixed, bias-direction table. **Branch A** (flip confirmed under
a censoring-free estimator) → P1v2 erratum, convention-dependence, do not headline the reversed
ordering. **Branch B** (ordering preserved) → §4.3 survives with a scale-sensitivity note and the
attenuated Δ-CI. Both branches feed the MN methods note as a case study.

**Affects.** Any critical-noise-σ* fragility comparison read off a coarse geometric grid with
cap-at-max, especially cross-condition comparisons where a naive RMS-normalization is applied. Extends
`project_fragility_scale_confound`: raw σ* is confounded cross-layer, but a coarse-grid σ*_rms "fix"
introduces its own censoring artifact — the control needs its own calibration.

**Resolution (2026-07-05, exact cell — `confound_analytic.py`, 1000 steps, declarative fully
memorized at loss 1.01, analytic censoring-free σ*).** **Branch B.** Under the grid-free estimator
declarative is MOST fragile under BOTH conventions — raw σ* 2.90 < narrative 4.11 < general 4.63, and
per-layer-SNR σ* 1.37 < 1.75 < 1.87. Paired bootstrap Δ = σ*(narrative) − σ*(declarative) excludes 0
both ways: raw Δ 1.34, CI [0.24, 2.47]; SNR Δ 0.44, CI [0.008, 0.91] (frac Δ≤0 = 0.024). So §4.3's
ordering is **genuine, not the scale artifact the reviewer proposed** — but the scale confound accounts
for **~2/3 of the raw gap** (the SNR effect is ~3× smaller and only marginally clears 0). P1 v2
disposition: keep §4.3, report the RMS-normalized/SNR Δ-CI with a scale-sensitivity note, do **not**
headline the raw magnitude. The coarse-grid σ*_rms sign-flip is confirmed an estimator artifact.

**Why it's a contribution.** "Naive RMS-normalization of a coarse-grid σ* can manufacture a spurious
sign-flip via censoring" is an MN-genre caution (calibrate the estimator before trusting the flip).
Belongs beside A1/A5.

---

## A8 (ledger) — Single-layer refusal-direction ablation scrambles refusal on half the prompts (33 comply→refuse, 17 refuse→comply) while five random directions move nothing

**Date:** 2026-09-12 · **Found in:** W4 pod, unit 14.5 (reconciled B1 cross-ablation, OLMo-3-7B-Instruct,
layer 16 output-hook projection-out; `deepsteer/supplement/cells/w4/olmo3_instruct/cross_ablation.json`,
per-prompt outcomes in `outputs/w4/olmo3_instruct/cross_ablation_outcomes.npz`). Chat-template harness,
100 held-out Heretic requests, refusal direction from the 400/400 train set, opening-refusal classifier.

**Observation.** Baseline refusal 0.62. Ablating the **refusal** direction raises the rate to **0.78**,
and the per-prompt table shows the rate hides the real event: **50 of 100 prompts change state** (33
comply→refuse, 17 refuse→comply). Ablating the **judgment-decision** direction moves 12 (all
comply→refuse, 0 the other way); the persona direction moves 1; each of five matched-norm random
directions moves **0**. All directions are unit-norm at the same site; cos(refusal, judgment) = 0.13,
cos(refusal, persona) = 0.08. Judgment accuracy is flat under every condition (0.72–0.76).

**Type.** control-misbehavior + sign-flip under intervention (archetype 1 + 5). The pre-registered
arrow reads Δ_refusal(judgment-ablated) − Δ_refusal(random-ablated) = +0.12 with the random arm at
exactly 0; the refusal-ablated arm was supposed to be the removability positive control (Paper 6:
OLMo 0.575 → 0.000) and instead moved the rate the wrong way.

**Competing readings.**
- R_a (instrument): a layer-16 output-hook projection-out is **not** the Paper 5/6 removability
  instrument (Heretic-style weight orthogonalization across layers). Removing the direction at one
  site perturbs the residual enough to derail generation, and the opening-refusal classifier's length
  heuristic counts short or degenerate outputs as refusals. Prediction: the 33 new "refusals" are
  incoherent or truncated, not "I can't help with that"; the 17 new "compliances" are equally off-form.
- R_b (mechanism): the refusal direction is load-bearing and bistable at the gate; removing it flips
  prompts near the decision boundary in both directions, and the +0.16 net is the classifier reading a
  reorganized gate. Prediction: the flipped outputs are coherent, on-topic, and the flips concentrate
  on prompts whose baseline refusal projection is nearest zero.
- R_c (harness parity for the *arrow*, independent of R_a/R_b): the judgment-decision arm's 12
  comply→refuse flips inherit whatever R_a/R_b says about the refusal arm; if R_a, the 12 flips are
  the same off-target derailment at smaller amplitude and the arrow is void, not detected.

**Discriminator.** Generations were not saved (the unit saved outcomes only; Amendment 14.5's save
list said "outcomes and generations"). Re-generate greedily under the **saved** directions
(byte-identical intervention) for baseline / refusal / judgment / random_0 and store the texts:
`14.5_gen`, ~400 short generations, ≈ 0.15 A100-h plus the OLMo-Instruct load. Read: (i) manual
coherence tally of the 50 flipped refusal-arm outputs and the 12 judgment-arm flips; (ii) flip
probability vs |baseline refusal projection| (R_b predicts a boundary concentration); (iii) the same
100 prompts under the Paper 5 `heretic_ablation` semantics as the removability control the cell
should have had. **Promoted** to the W4 rerun pod (2026-09-12) under the promotion rule.

**Thesis impact.** R_a: 14.5 Branch B stands only as "no cross-effect detectable" once the random
q95 is restated from an instrument that does not derail generation; the +0.12 arrow is void and
the R3 sentence stays geometric. R_b: the arrow is real (judgment-decision ablation raises refusal
on 12 prompts), D2-01 gains a causal cross-arrow, and the refusal-direction bistability is a finding
for FL §6. Either way the 14.5 verdict is held until the discriminator runs.

**Status.** open; discriminator **run** (W4 rerun pod 74de8jk0usv335, 2026-09-13, `14.5_gen`, 9.7 min): texts for baseline / refusal / judgment_decision / random_0 saved in `outputs/w4/olmo3_instruct/cross_ablation_generations.npz`, per-prompt classifier outcomes agree **100%** with the saved 14.5 outcomes in all four conditions (greedy, byte-identical intervention), rates 0.62 / 0.78 / 0.74 / 0.62 reproduced. **Coherence tally (W4-3, 2026-09-13): R_a REJECTED.** All 33 comply→refuse outputs are well-formed opening refusals (median 336 chars, repetition 0.11, no debris); all 17 refuse→comply outputs are coherent "Certainly! However…" redirects; all 12 judgment-arm flips are well-formed refusals; length and repetition match baseline. R_b stands (a load-bearing, bistable direction: single-layer removal re-decides ~half the prompts in both directions). Remaining discriminator: flip probability vs |baseline refusal projection| (needs the 100 eval-prompt last-token activations, one forward pass, ~2 min GPU; not saved). The `heretic_ablation`-semantics control is the same price. Verdict impact: 14.5's judgment→refusal arrow (+0.12 [0.06, 0.19]) is a candidate Branch A with the instrument scope stated; escalated. `resolution_type`: experiment (partial).

---

## A9 (ledger) — An absolute participation-ratio gate is not evaluable below n ≈ 4·PR; reasoning-trace windows are decision-like positions

**Date:** 2026-09-13 · **Found in:** W4 14.4 (`supplement/cells/w4/{olmo3_think,gpt_oss_20b}/pr_audit.json`).

**Observation.** The pre-registered positive control for the P0–P3 audit required the content position
P0 to read PR ≥ 30. With 64 rollouts per position the sample-rank ceiling is 63 and P0 reads 28.9
(Think) and 22.1 (GPT-OSS), using 0.46 / 0.35 of the ceiling while sitting above its covariance-matched
sampling null at quantile 1.0 on both models. The absolute bar cannot be met at this n by a position
whose true PR is anywhere near it; the same texts' content positions at n = 240 read 27–97.

**Type.** control-misbehavior (a bar that depends on n, applied at small n). **Reading.** Calibration
closure (no experiment): the gate of record becomes null-referenced (W4-07 form: PR relative to the
sample-rank ceiling and to the column-shuffle reference, with the Gaussian-null quantile), which is
what MN §2 now states; the absolute 30 is kept only as the historical value with its n stated.

**Finding folded in.** On both reasoning models the in-trace window (P2) reads PR 9.7 (Think) and 5.1
(GPT-OSS) with the moral band below the covariance null, the same signature as the four chat decision
sites (8.6–14.7) and unlike content positions. The reasoning window is a decision-like position: A2
on a fifth kind of site, and the reason the in-trace rung stays band-hedged (Branch B, W4-05).

**Status.** resolved (calibration). `resolution_type`: calibration.

---

## A10 (ledger) — Matched-norm random directions flip 61–83% of Llama reply-inversion margins; the harm axis flips none and pushes toward safe

**Date:** 2026-09-13 · **Found in:** W4 14.6b (`supplement/cells/w4/llama31/reply_inversion_null.json`;
margins in `outputs/w4/llama31/reply_inversion_margins.npz`).

**Observation.** Llama-3.1-8B-Instruct, layer 12, forced-answer margin (positive = toward harmful),
clean margins mean −1.78 (87% ≤ 0). Steering along the harm direction at 0.5× and 1.0× the residual
norm flips 0/100 replies and shifts every margin further negative (−2.1, −2.6). Twenty random unit
directions at the identical norm shift margins toward zero (per-direction means −1.75 … +1.06) and so
flip 61% (q95) and 83% of the near-zero majority. cos(harm_dir, severity contrast) = +0.125: the sign
convention is the conventional one.

**Type.** control-misbehavior + sign-flip. **Competing readings.** R_a: the flip metric at this norm
measures margin washout, not directed steering; any large perturbation regresses the margin toward
zero and "flips" whatever sat near it, so the specificity null is saturated and uninformative for
flips (the margin *shift* is the informative statistic, and there the harm direction is 4× any random
direction, in the safe direction). R_b: on Llama the harm axis genuinely steers replies toward safe
(the reply-inversion effect of record is on Qwen2.5-14B-Instruct, +17.4 flips 33%; Llama's own of
record was +3.0 flips 23% with a different harness), i.e. the P7-05 Llama number does not reproduce
under matched-norm steering. **Discriminator.** Zero-GPU on the saved margins: the per-item margin
shift distribution for harm vs each random direction (a paired sign test separates "washout" from
"directed"); then the Paper 7 harness's own coefficient and read on the same 100 items (one Llama
load, ~0.2 A100-h). **Status.** open; MN §3.1 keeps the limitation with the measured chance level.
**Thesis impact.** None on FL; MN §3.1's "specificity control missing" becomes "run and not passed at
matched norm", a stronger statement of the same limitation.

## KDG-A1 (ledger) — The stated-judgment reference is unstable: J_stated flips under paraphrase on 31% of scenarios, and the paraphrase-stable subset halves the knowing–doing gap

**Date.** 2026-09-14 (KDG pilot, `papers/kdg_panel/KDG_RESULTS.md` §2–3; `outputs/pilot/analysis_pilot.json` `robustness`).
**Observation.** OLMo-3-7B-Instruct greedy J_stated agrees with its own greedy J_stated on a paraphrased third-person frame on 66/96 scenarios by option id (0.69; 0.84 on the violating/non-violating binary). 16/48 gate primaries fail the option-level stability rule; 12 of those are consistent↔violating flips across 8 samples at T = 0.7, 4 are consistent/neutral splits. KDG on the screened panel 0.22 [0.08, 0.42]; on the paraphrase-stable screened subset 0.11 [0.00, 0.35] (n = 18).
**Type.** near-miss + control-misbehavior (the floor rung is the reference's own noise).
**Appears in.** KDG_RESULTS §2 floor rung, §3; spec A12.
**Competing readings.** R_a: the model's judgment on these scenarios is genuinely indeterminate (mixed considerations; the neutral option splits the vote) and a paraphrase-majority reference would recover a stable J; the gap measured against it is the real quantity. R_b: the gap is manufactured by reference noise: J lands non-violating by chance on a scenario the model is ~50/50 about, D is ~50/50 too, and half of those count as KDG = 1; under a paraphrase-robust reference the gap is ≈ 0.
**Discriminator.** Zero-GPU on saved text: score consideration breadth on the 864 J replies (`rate_with_judge.py breadth`) and test whether the 16 unstable scenarios are the low-breadth ones (R_a predicts high breadth = many considerations; R_b predicts nothing). Then a pod rider (~10 min on any OLMo-3 load): greedy J_stated on three further paraphrases per scenario → a paraphrase-majority reference; KDG under it, with the paired Δ vs the pressure-removed null. If Δ excludes 0 → R_a; if the rate collapses toward the null → R_b.
**Status.** open; zero-GPU leg RUN 2026-09-14 (`outputs/pilot/analysis_anomalies.json` KDG-A1): consideration breadth on 863/864 J_stated replies (Claude judge, rubric 1.0.0), length-residualized, stable vs unstable scenarios −0.06 [−0.44, 0.33] (raw means 6.60 vs 6.84 considerations); R_a's breadth prediction is not supported and R_b is not separated. KDG-2 (2026-09-15): SURVIVES at L1 (0.21, excess 0.11 [0.01, 0.21], n 73), L2 under-powered. KDG-3 union (2026-09-15): L2 becomes the verdict level (n 43) and the excess there is 0.05 [−0.02, 0.19], not resolved; L1 0.11 [0.02, 0.19] (n 92); excess shrinks with strictness. R_b is separated at L1 and not at L2. Discriminator of record: n at L2 (~90 pairs) or decisive-by-construction scenarios that raise the L2 yield. **RESOLVED on the continuous instrument (A15, 2026-09-15; `KDG_RESULTS.md` §12): coherence check passes (0.955; means within 0.01); the paired excess over the null on the log-prob readout is 0.054 [0.021, 0.086] at L0, 0.063 [0.030, 0.096] at L1, 0.079 [0.029, 0.128] at L2 (n 67), growing with strictness. R_a obtains: the gap is not reference noise. The binary L2 shortfall was majority-rule information loss at n 43 paired.** `resolution_type`: experiment (zero-GPU re-analysis of saved vectors). `resolution_type`: experiment.
**Thesis impact.** R_a: the execution program has its outcome variable and the F3-high candidate structure is real. R_b: KDG on 7B instruct is a reference artifact; the panel needs a stability-filtered construction (only scenarios the model judges decisively) before any anatomy cell, and Branch B1 wording applies.

## KDG-A2 (ledger) — The instruct model puts less next-token mass on the option letters in a raw completion frame than the base model (58 vs 84 of 96 scenarios above the 0.5 floor)

**Date.** 2026-09-14 (KDG_RESULTS §5).
**Observation.** Raw frame (`Answer:` prefix, letters with leading space): base clears the 0.5 option-mass floor on 84/96 scenario-frame pairs (mean mass 0.76); instruct on 58/96. On the 50 shared scenarios the raw gap rates are 0.10 (base) vs 0.08 (instruct).
**Type.** control-misbehavior (a floor designed for the base model bites the instruct model harder).
**Competing readings.** R_a: the instruct model spends the next-token mass on chat-format tokens (a template/EOS/role token) because a raw frame is off-distribution for it, so the readout is format-invalid there, not a null. R_b: the instruct model spreads mass over refusal or hedge tokens on exactly the pressured scenarios, so the below-floor set is informative (an implicit refusal readout).
**Discriminator.** Zero-GPU on the saved full log-prob vectors: top-5 next tokens on the 38 below-floor instruct rows (formatting tokens → R_a; refusal/hedge tokens → R_b) and whether the below-floor set is enriched for high pressure (violating fraction in D_chat).
**Status.** RESOLVED 2026-09-14 (calibration; `outputs/pilot/analysis_anomalies.json` KDG-A2): on the 117 below-floor instruct rows (28 scenarios) the summed top-5 mass is `<|im_end|>` 33.4, then the option letters (` C` 16.0, ` B` 14.9, ` A` 10.6), then ` (` 7.8; no refusal or hedge token appears. The instruct model tries to end the turn in a raw completion frame → R_a (format-invalid), and the residual mass is on the letters, so the argmax readout is still the option choice where it exists. Below-floor scenarios are mildly more pressured in D_chat (violating fraction 0.40 vs 0.30), which is consistent with the model wanting to say more before answering, not with refusal. `resolution_type`: calibration.
**Thesis impact.** The §4.6 format contrast is read on the shared-floor subset (done); the raw frame is not a second decision readout on instruct models.

## KDG-A3 (ledger) — Candidate family structure: the gap concentrates in the instrumental family (fork B F3 0.55 [0.27, 0.82]) and is lowest on loyalty/fairness allocation (F4 0.08 [0.00, 0.30]); third-party harm is not lowest

**Date.** 2026-09-14 (KDG_RESULTS §4).
**Observation.** Primary rule: F1 0.33 (6), F3 0.33 (6), F4 0.11 (9), F5 0.20 (5). Fork B: F1 0.25 (8), F3 0.55 (11), F4 0.08 (12), F5 0.20 (5). F1+F3+F4 − F5 = 0.04 [−0.41, 0.38]; family MDE 0.51 at pilot n.
**Type.** family-exception candidate (Branch A predicted F5 lowest if the action read is harm-keyed; F4 is lowest).
**Competing readings.** R_a: tool-call shortcuts are the surface where an instruct model's judgment and action decouple most (the action is a tool name, the judgment is prose), i.e. the gap is largest where the action surface is least verbal; allocation decisions (F4) are judged and acted on in the same register. R_b: noise at n = 6–12; F3's scenarios are simply more mixed at baseline (F3 had the highest mixed count, 9/12), and mixed D plus majority rule inflates KDG there.
**Discriminator.** Full panel (n ≈ 40 per family), harm-stratified via the twins; zero-GPU now: the rollout-level violating fraction per family (already saved) compared with the majority-rule rate (R_b predicts the F3 excess vanishes at the rollout level).
**Status.** RESOLVED 2026-09-15 → R_b (KDG-2 full panel, `outputs/kdg2/analysis_pilot.json`): per-family KDG on 200 primaries is flat, F1 0.21 [0.07, 0.38] (n 29), F3 0.17 [0.05, 0.38] (23), F4 0.21 [0.08, 0.39] (24), F5 0.23 [0.00, 0.46] (13); F1+F3+F4 − F5 = −0.03 [−0.30, 0.21]. The pilot's F3-high / F4-low pattern was noise at n = 6–12. `resolution_type`: experiment.
**Thesis impact.** R_b obtained: no family structure at n ≈ 100 screened; the panel paper reports the pooled rate with the per-family table as a null with its MDE. (Superseded reading) R_a: the action channel's read is surface-dependent, not harm-keyed (Branch C with a named mechanism); the anatomy cell goes to the F3 tool-call position first. R_b: no structure; the panel paper reports the pooled rate only.

## KDG-A4 (ledger) — F4 (loyalty/fairness allocation) is generator-dependent on the full panel: Claude-generated F4 has KDG 0.00 [0.00, 0.27] (n 11), GPT-generated F4 0.38 [0.15, 0.62] (n 13), CI-separated

**Date.** 2026-09-15 (KDG-2, `outputs/kdg2/analysis_pilot.json` `per_generator[*].per_family`, `full_gate.generator_reversal_by_family`).
**Observation.** On the screened full panel the F4 rate differs by generator with each generator's CI excluding the other's point estimate; no other family does (F1 0.27 vs 0.14, F3 0.18 vs 0.17, F5 0.12 vs 0.40 at n 8/5, not separated). The cross-rated external labels had already flagged three GPT F4 harm twins with inverted option labels (excluded); the primaries were not flagged.
**Type.** family-exception (the spec's "generator-dependent family result is an anomaly by rule", §2).
**Competing readings.** R_a: a construction difference: GPT's F4 allocation scenarios make the favor option more defensible (or the impartial option less clearly the rule), so the model's violating action is partly a reasonable reading of the scenario, and the construction label is the noisy side. R_b: a real generator-register effect on the model: GPT's F4 pressure framing engages OLMo-3's loyalty pull more than Claude's does, and the gap is real in that register.
**Discriminator.** Zero-GPU first: the external-rater agreement on GPT vs Claude F4 primaries (R_a predicts lower agreement on GPT F4), and the rater's harm/norm labels; then a blind human read of the 13 screened GPT F4 primaries against the 11 Claude ones (option labels hidden). If construction is at fault → those scenarios are flagged and F4 is rebuilt by a paraphrase-swap (each generator paraphrases the other's F4 scenarios, holding options fixed) so the register and the construction are decoupled: one pod cell, ~15 min.
**Status.** open; zero-GPU leg RUN 2026-09-15: the external raters agree with construction on 20/20 F4 primaries from each generator, with the same harm (2) and norm (fairness_cheating) labels on both, so the labels do not separate R_a from R_b; swap cell RUN 2026-09-15 (KDG-3, A14; `KDG_RESULTS.md` §11.4): inconclusive at 4–11 defined scenarios per cell; the pre-registered containment rule returns R_a trivially (CIs [0, 0.5], [0, 0.33]) while both point estimates move toward the paraphraser (0.00 → 0.10, 0.33 → 0.11), the R_b signature. Next leg: blind human read of the 24 screened F4 primaries with option labels hidden (zero GPU), then a swap cell sized for ~30 defined per origin (≈ 120 swapped scenarios, one ~1 h pod). F4 is reported per generator and excluded from the pooled verdict (spec §2); blocks the full gate by the letter (§5, no reversal). **Re-typed 2026-09-15 on the continuous instrument (A15, §12.3): a graded provider difference, not a reversal**: F4 excess Claude-written 0.13 [0.01, 0.22] vs GPT-written 0.22 [0.11, 0.34], overlapping; the difference sits on both the acting side (0.35 vs 0.41) and the judging side (0.20 vs 0.17). The binary 0.00-vs-0.38 was majority-rule discreteness. **Blind human read RUN 2026-09-19 (author, 24 screened F4 scenarios, labels and origins hidden; `data/F4_blind_read_scored.json`): agreement with the construction label 10/11 on Claude-written and 11/13 on GPT-written items; violating-option picks 1 vs 2; forced picks ("none acceptable") 2 vs 3; ties between two options 9 vs 5. No generator asymmetry in construction agreement → R_a (construction defect) is not supported. RESOLVED as a graded generator-register difference (R_b in degree, not a reversal): F4 is reported per generator as a covariate result; the §5 reversal clause, defined on the binary readout, is met on the continuous instrument and failed on the binary one, which is now a known property of majority-rule discreteness at n ≈ 12 per side, not of F4. Whether the gate clause is re-scoped by dated amendment is the author's decision (KDG-G3).** `resolution_type`: experiment.
**Thesis impact.** R_a: the panel's construction validity check gains a real catch and F4 is re-derived; the pooled KDG without F4 is the number of record. R_b: generator register is a covariate the panel must always stratify on, and the "moral surface" of a scenario is part of what the action channel reads.

---

## A11 (ledger) — Permutation tests with mirror-tied partitions counted ties by float rounding; Paper 3 stated a 0.05 floor the 3-vs-3 test cannot reach (true floor 0.10)

**Date:** 2026-10-06 · **Found in:** library CI (LIBRARY_RELEASE_PLAN §B), `tests/geometry/test_geometry.py::test_permutation_test` failing on Linux (p 0.106) and passing on macOS (0.086) with the same seed.

**Observation.** `deepsteer.geometry.clustering.permutation_test` and Paper 3's `permutation_test_mft_groups` (`papers/3_moral_geometry/scripts/exp1_2_3_framework_geometry.py`) counted the tail with a bare `>=`. In a 3-vs-3 design every split appears twice (either group listed first) with mathematically equal statistics, and index reorderings of one split give the same value; in floating point these differ by ~1 ulp, so whether they count depends on BLAS and summation order. Under the bare `>=` a ±1e-14 shift of one off-diagonal block moves the synthetic-fixture p across 0.054 / 0.086 / 0.106; with a 1e-12 tie tolerance it is 0.106 on every platform. The 20 assignments form 10 mirror pairs, so the exact p is a multiple of 0.10 and its floor is **0.10**: the test cannot reject at α = 0.05 on any data. Paper 3 (arXiv v1) stated the floor as 0.05 in five passages (Introduction, Discussion twice, Conclusion, Appendix D: "significance was reachable"), framed the abstract and §4.3 on "20 partitions", and its Methodology described 10,000 sampled permutations while Appendix D enumerated.

**Type.** control-misbehavior (instrument) + bug report that is a finding (archetype 6: who else hits this? any small exhaustive permutation test over symmetric groups with a float `>=`).

**Blast radius (enumerated before any change).** Exact p recomputed with ties counted from the saved cosine matrices (`outputs/exp1_2_3`, `exp1_2_3_7B`, `exp5_dense_vs_moe/{olmo,olmoe}`; nearest competing split ≥ 1.1e-4 from the observed one, so the 6-decimal rounding of saved matrices cannot reorder splits): 1B 0.40–0.80 (median 0.55), 7B 0.30–0.90, dense ≥ 0.40, MoE ≥ 0.30. **No verdict changes**: the bug only lowers p (by up to one mirror pair) and every reported MFT-group p was ≥ 0.25. Wording errors: the floor (0.05 → 0.10) in five passages, plus the "20 partitions" framing in the abstract and §4.3; Appendix D cells layer 9 (0.55 → 0.60) and 12 (0.65 → 0.70), both odd multiples of 0.05 that only a broken mirror tie produces; §4.3 quoted the older 10,000-permutation run (min 0.32, median 0.53) while Appendix D used exact enumeration (min 0.40); §4.14's 7B range "0.49–0.78" matches neither saved 7B run (5134815: 0.275–0.89; canonical 6f7b85f: 0.212–0.891) and is replaced by the exact 0.30–0.90; §4.5's "all p > 0.25" (Monte Carlo) becomes exact ≥ 0.40 / ≥ 0.30. Program-wide audit of 32 files with permutation/randomization nulls: no other tail count over exact ties without a tolerance (`dilemma_compositionality_baselines.py` already used 1e-12; the rest are continuous nulls or quantile-only). Stale but uncited outputs carrying sub-floor p from the old library function: `probe_engineering/concept_directions.json` (min 0.072), `leace_directions.json` (min 0.067), `mean_diff_directions.json`, `external_robustness/mfv_geometry_7B.json`; `concept_directions.py:269` thresholds `p < 0.05` to print significant layers, which no exact 3-vs-3 result can satisfy (printed only, never saved or cited). FL §3 cites Paper 3's v1 "minimum p = 0.32"; the exact value is 0.40 and FL's claim ("no significant recovery") is unchanged; left as a citation of v1 pending the author.

**Competing readings.** R_a (accepted): a float-tie bug plus a counting error in the stated floor; the MFT-grouping null stands with a corrected, stricter detection bar (cannot reject at 0.05 at all). R_b: the corrected bar shows the instrument had no power at 0.05, so the null rests on the dendrogram leg and the MFT split's rank (never above fourth of ten), not on the test. Both hold at once; the paper now says both.

**Discriminator.** None needed for the verdict (calibration closure). The open item is power: a positive control (planted 3-vs-3 structure at the observed cosine scale) to state the within/between gap at which the MFT split reaches the 0.10 floor. Zero-GPU on saved matrices.

**Status.** resolved (calibration). Library fixed with a 1e-12 tie tolerance and two tests (exact 0.10 on the fixture; p invariant to ±1e-14 shifts, failing under the bare `>=`); Paper 3 script fixed the same way and reproduces the corrected Appendix D column from saved matrices; Paper 3 corrected for arXiv v2. `resolution_type`: calibration.

**Thesis impact.** None on the thesis or any verdict. Paper 3's MFT-grouping null now carries the correct detection bar (attainable floor 0.10, cannot reject at α = 0.05; MFT split never above fourth of ten splits).

## A12 (ledger) — The Heretic prompt set's harmless half is Alpaca (CC BY-NC 4.0), so refusal-derived arrays in the public FL/MN deposit fall under the program's own NC exclusion rule; and one Paper 3 output labels its MoE run with the dense model's id

**Date:** 2026-10-06 · **Found in:** LIBRARY_RELEASE_PLAN §D artifact tracing (four tracing agents; license checked against the HF and GitHub APIs).

**Observation 1 (licensing).** `refusal_prompts.json` takes its harmless prompts from `mlabonne/harmless_alpaca`, whose rows are verbatim Stanford Alpaca instructions (`tatsu-lab/alpaca`, CC BY-NC 4.0; the mlabonne card states no license). The supplement's RELEASE_PLAN calls the set "MIT-licensed upstream"; MIT holds for the harmful half (AdvBench), not the harmless half, and Heretic's code is AGPL-3.0. Its §2 rule treats any array whose stimuli are NC as NC-derived. Applied consistently: 36 in-tree arrays (Papers 5–7) went to the restricted record 10.5281/zenodo.23203126; the public FL/MN deposit 10.5281/zenodo.22731361 (CC BY 4.0) holds further arrays computed from the same prompts (proto-refusal caches, refusal directions in the D1–D3, W4 and P7 groups).

**Observation 2 (provenance label).** `papers/3_moral_geometry/outputs/exp5_dense_vs_moe/olmoe/exp1_foundation_probing.json` records `"model": "allenai/OLMo-2-0425-1B"`, while `exp5_summary.json` names OLMoE-1B-7B-0924 for that arm and its accuracies and npz sha256 differ from the dense copy.

**Type.** control-misbehavior (a provenance claim about a stimulus set is wrong) + cross-doc conjunction (label vs summary).

**Competing readings.** Licensing: R_a, derived statistics (directions, activations) inherit the NC term, as the program's own rule says, so the FL/MN deposit needs a new version with those arrays restricted; R_b, aggregate derived arrays do not reproduce the licensed text and are not NC-bound, so the rule is narrowed and documented. Label: R_a, the JSON's model field is a stale default written by `exp1_2_3_framework_geometry.py` when called from exp5 (the arrays are OLMoE's); R_b, the olmoe directory holds a dense run.

**Discriminator.** Licensing: an author (or counsel) decision; no experiment separates them. Label: zero-GPU — compare the olmoe npz's hidden size and layer count and the JSON's per-layer accuracies with the exp5 summary's OLMoE arm.

**Status.** Licensing: **resolved by author judgment, 2026-10-07 (R_b for this case):** the derived statistics in 10.5281/zenodo.22731361 are not NC-bound and the deposit stays public as published; going forward, arrays computed from NC stimuli are still deposited restricted (10.5281/zenodo.23203126 is the first). Recorded in `deepsteer/datasets/DATASET_LICENSES.md`. `resolution_type`: scoping. Label: **resolved (R_a), 2026-10-06, zero-GPU.** At the commit that wrote the file (a20ffc3, rerun a818bbf) `exp1_2_3_framework_geometry.py:198` wrote `"model": OLMO_REPO`, a hard-coded constant, whatever model it was given, while `exp5_dense_vs_moe_geometry.py:318-330` loaded `OLMOE_REPO` for this arm; the code now writes `model.info.name`. The arrays are OLMoE's; only the JSON label is stale. `resolution_type`: calibration (code archaeology).

**Thesis impact.** None on any verdict. Licensing scopes what may be redistributed publicly; the label affects provenance of one exp5 arm only.

## Process ledger

**2026-07-03 — the cold-boot W0 audit caught a live erratum that warm sessions had missed.**
The fresh-context ledger reconciliation (CLAUDE.md boot sequence) found that arXiv:2606.11375v1
lacks its own §4.4 activation-scale control, so its Finding 2 ships an un-scoped raw fragility
gradient → v2 erratum required. No warm working session had flagged it. Evidence that the
boot-sequence design pays for itself, and a datapoint for the open periodic-fresh-eyes cadence
question (how often to force a cold-context re-audit of committed/published claims).

**KDG-A4 addendum (2026-09-26, round 3).** The cross-rated external labels flagged three more
GPT-written F4 harm twins as label-inverted (F4-B-45T, -51T, -57T; excluded by the 2026-09-14
rule). All six inverted cases across rounds share one cell: generator half B, role harm_twin,
twin norm `loyalty_betrayal` (3 of 7 in round 1, 3 of 4 in round 3). Mechanism, read from
F4-B-45T: with loyalty as the norm at stake, the generator labels favoring the teammate
"consistent" and following the posted fairness rule "violating", which inverts F4's pressure
direction. Slot parity routes the loyalty and sanctity twin norms to half B only, so in F4 the
twin norm is confounded with the generator by construction. Bearing on KDG-A4: small; GPT's F4
gap sits in the fairness primaries (screened rate 0.40, n 10), not the loyalty twins (n 2).
Discriminator (zero GPU): re-score KDG-A4 with every F4 loyalty-norm twin excluded, both
generators; and for any future F4 round, rotate twin norms across halves. Status: open, priced.

**Process ledger 2026-09-27 — interrupted download lost a keystone step.** Session A (pod
fcwecbba80z0hg, p1a) completed every step on the pod, but the launcher downloaded results with a
single unchecked `rsync -az` pass. The pass was interrupted inside `stages_chat` (alphabetical
order), so `stages_raw` (the C3 keystone: SFT/DPO raw cells) and two SFT chat cells never arrived,
and the EXIT trap terminated the pod. Fix (commit 703c8cc): `rp_download` in
`papers/runpod_common/session_lib.sh`, used by `run_session.sh`: up to 5 attempts
(`DOWNLOAD_TRIES`, default 5) with `--partial`, 20 s apart; on persistent failure KEEP_POD=1 so the
pod survives and the terminate command is printed. Tested with a fake rsync (recovers after 2
failures; keeps the pod after 3 of 3). Cost of the incident: one re-run pod (p1a_fix, ~50 min).
Lesson for every launcher in the repo: a teardown trap must be conditional on a verified download.
**Correction 2026-09-27 (second occurrence, p1a_fix pod 1ljufh6voenl17).** The same steps were lost
again at the same point, with `rp_download` in place. Root cause, found locally: the Mac's data
volume was at 100% (120 MB free), so every rsync pass died writing into `stages_chat_sft`, and
`stages_raw` (next alphabetically) never started. The first incident almost certainly had the same
cause; "network interruption" above was a guess, now withdrawn. Retries cannot fix a full disk.
Fixes: 155 GB freed (the local OLMoE-1B-7B Hugging Face cache, 12 checkpoint snapshots, deleted
with the author's approval); `rp_require_disk` refuses to provision a pod with < 50 GB free under
the results path (`MIN_FREE_GB`); `rp_download` stops retrying below 5 GB free. Whether the second
pod survived the failed download (KEEP_POD path) is being checked with the author.

**Process ledger 2026-10-07 — a bootstrap PR interval quoted without its bias note.** Writing
KDG_GPTOSS_SPEC G-A3 found two CIs for one number: GPT-OSS's raw decision-token PR 9.40 carries the
subsampling CI [9.09, 10.65] of record (CLAIMS W4-07) and, in `W4_RESULTS.md` §14.3, the row-resampled
bootstrap CI [7.70, 9.96] from `deepsteer.geometry.participation.bootstrap_pr`. The library already states
the direction (duplicated rows lower the PR; `bias_note`), but calls it "slight"; at n = 64–128 it is
not: in `supplement/cells/w4/gpt_oss_20b/pr_audit.json` the P0 window reads 22.09 with bootstrap CI
[13.88, 18.91], which excludes its own point estimate. Blast radius: the bootstrap interval gates no
verdict (the A5 band half uses band_min against the null q95; the W4-05 hedge uses the PR point and the
covariance null), and every PR CI quoted in CLAIMS is the subsampling one. **Resolved (author,
2026-10-07):** `W4_RESULTS.md` §14.3 now labels the bootstrap interval biased low and not of record,
beside the subsampling CI of record. The `bootstrap_pr` docstring now states the size (resolved
2026-10-07): bootstrap median 7% low at n = 128, and at n = 64 an interval that excludes its point
estimate; it points to the subsampling interval for anything of record. **Process note (author, 2026-10-07): CI draw seeds are saved going
forward.** The W4-07 subsampling interval could be reproduced only to about 0.1 ([8.98, 10.57] vs
[9.09, 10.65]) because its draw sequence was not recorded. Every new CI records its method, draw
count, seed and RNG call beside the interval (first instance: `analyze_gptoss.py`, whose PR record
carries `subsampling: {method, m, draws, seed, rng}`; the bootstrap helpers already take a fixed
seed, now written into each report).

**2026-10-15 — a paper appendix quoted a scenario that is not in the panel.** While choosing a worked
example for §2 of the KDG paper (P1-A17 optional item), the Appendix E F1 example ("You are Nadia ...
Bellview County Hospital ... 81 percent ... two new nurse positions") matched no scenario file; the panel's
F1-A-00 is the same situation with different facts ("You are Dana ... Bellview Hospital ... thirty-eight
minutes ... three staff positions"). The text was in v1 as posted (arXiv:2610.08670v1) and survived every
earlier review pass, because no check compared quoted scenario text against the data. The F5 example
(F5-A-00) was correct but cut short without a mark. Fixed in the v2 draft: both quoted verbatim from the
files, with a v1 note. Discriminator for the class (zero GPU, done for this paper): string-match every
quoted scenario passage in a paper against the scenario files before submission. Bearing on any number:
none (the appendix is illustrative). Status: resolved for this paper; the string-match check is a
standing pre-submission item.

## KDG-A5 (ledger) — Exploratory family structure on the continuous instrument: F1 (honesty) exceeds F3 (shortcut) on the chat mass gap, and the pressure-attributable excess is present on F1/F4 and unresolved on F3/F5

**Date.** 2026-09-19 (A17 E1; `papers/kdg_panel/data/analysis_a17_union.json` `chat_exploratory.E1_family_contrasts`; KDG_RESULTS §13.2).
**Observation.** On the screened union, chat continuous g by family: F1 0.21 [0.17, 0.25] (43), F3 0.12 [0.06, 0.17] (40), F4 0.17 [0.09, 0.26] (26), F5 0.16 [0.05, 0.27] (21); one of six unpaired contrasts separates (F1 − F3 0.091 [0.018, 0.160]). Per-family paired excess over the pressure-removed null: F1 0.076 [0.020, 0.129], F3 0.020 [−0.038, 0.079], F4 0.121 [0.054, 0.195], F5 −0.008 [−0.090, 0.073]. The pre-registered binary contrast (F1+F3+F4 − F5 = −0.04 [−0.27, 0.16]) is centered on zero at MDE ~0.40.
**Appears in.** Paper 8 §6 (exploratory paragraph), Appendix D.
**Competing readings.** R_a: real structure that the binary instrument cannot see: the incentive moves the action on the honesty and fairness families (option choice, allocation) and not on the shortcut family (tool selection from a fixed menu; the action surface may be insensitive to the incentive sentence) or the harm family (whose gap is present without pressure and not pressure-attributable, i.e. the harm-keyed prediction of Branch A in excess units). R_b: chance (six contrasts at 95%: P(any separation) 0.26) plus small n on F5 (21) and F4 (26); F5's excess interval includes the pooled 0.054.
**Discriminator.** Sixteen more F3 and F5 primaries per generator (generation via CLI subagents, no API; one ~1 h pod with the kdg2 cell profile) → family excess MDE near 0.10 on the continuous instrument; pre-register the F5-excess-vs-rest contrast in excess units before the pod. Zero-GPU leg first: within F3, split by whether the violating tool is the *only* fast route (covariate `tool_menu` structure) to test the action-surface reading.
**Status.** open (candidate, not a finding); priced. Zero-GPU leg run 2026-10-04 (pre-registered 0b69263; `data/analysis_a5_fastroute.json`): the non-generator rater calls the prohibited tool the only fast route in 36 of 40 screened F3 scenarios; E_yes 0.015 [−0.052, 0.078] (36) vs E_no 0.061 [−0.062, 0.131] (4); Δ −0.045 [−0.154, +0.090], MDE 0.17 → **unresolved**; the action-surface reading cannot explain F3's low excess on this construction (almost no quick permitted alternative exists). The pod leg (F3/F5 top-up) remains the discriminator.
**Thesis impact.** R_a with F5 excess ≈ 0: the harm family's gap is *not* pressure-attributable, which is Branch A restated in excess units and reconnects the action channel to the refusal read; the S1 rank cell would then be scored per family. R_b: Branch C stands as written.

**Zero-GPU leg pre-registered (2026-10-04, before any label or computation; author: before arXiv
submission).** *Label:* each screened F3 primary of the A17 union (n 40, `per_scenario_union.csv`
`screened`) is rated by the non-generator provider's CLI (Codex for Claude-written items, Claude for
GPT-written): "In this situation, is the prohibited tool the only option that gets the task done
quickly?" → `only_fast_route` ∈ {yes, no, unclear}; unclear items excluded and counted. *Quantity:* the
per-scenario paired excess on the continuous chat readout, E = (p_D − p_J) − (p_D,null − p_J,null);
E_yes, E_no, and Δ = E_yes − E_no, bootstrap 10,000, seed 0 (independent resampling of the two groups),
MDE beside. *Rule:* the action-surface reading of R_a (the incentive moves the action only where the
shortcut is the sole fast route) is **supported** iff Δ's CI lies entirely above 0, **contradicted** iff
entirely below 0, **unresolved** otherwise. Exploratory (one of six family contrasts); it changes the
paper's §6 wording only: supported → the F3 excess sentence names the split; otherwise §6 says the split
was not resolved at its MDE.

## KDG-A6 (ledger) — The instruct model's no-pressure frame gap in the raw frame is negative (acting more norm-consistent than judging), the opposite sign from base

**Date.** 2026-09-19 (A17; `analysis_a17_union.json` `three_cell_union.instruct.continuous.matched_null_g`; KDG_RESULTS §13.1).
**Observation.** On pressure-removed raw twins, g_null = p_D − p_J is +0.024 [0.017, 0.030] on base (354) and −0.038 [−0.059, −0.015] on instruct (208); on the shared 192, instruct p_D(twin) 0.192 vs p_J(twin) 0.224, base 0.306 vs 0.280. In the chat frame the same model's null is +0.12 (deliberated judgment vs immediate action).
**Appears in.** Paper 8 §7 ("the baseline moved"), Discussion.
**Competing readings.** R_a: post-training installs an agent-frame caution: the assistant placed as the actor is more norm-consistent than the assistant as a third-person judge, a persona default (consistent with the persona-as-lever design). R_b: a raw-frame artifact on a chat-trained model that declines the frame on 155/397 scenario-frames (KDG-A2): the fixed `Answer:` prefix in the second-person frame elicits end-of-turn or hedging mass that the option-letter normalisation hides, or the two frames' prefixes are read differently by a template-trained model.
**Discriminator.** Zero-GPU first: on the shared 192, compare instruct p_D(twin) against the raw-frame option mass (`option_mass`) per scenario; if the negative null concentrates where option mass is lowest (near the 0.5 floor), R_b. Pod leg (~5 min on any loaded OLMo-3-Instruct): a letter-only judgment readout under the chat template on the pressure-removed twins (J_chat, immediate, no reasoning), giving a chat-frame null with the deliberation asymmetry removed; if p_D(chat twin) < p_J(chat, letter-only twin), R_a.
**Status.** open; priced.
**Thesis impact.** R_a: the "post-training lowers the baseline" half of the execution thesis stands and persona steering is the first lever. R_b: that half is dropped; the widening stands on E alone and the raw-frame comparison is restricted to base.

**Update 2026-09-25 (pitch novelty gate; `LIT_PASS_P9.md` §6 rider 3).** A third reading, merged here rather than opened as a new entry (same observation, one joint discriminator).
- **R_c: evaluation caution.** Burnat & Davidson, arXiv:2605.06327 (v1 2026-05-07, fetched), test `allenai/Olmo-3-7B-Instruct` and `allenai/Olmo-3-1025-7B` with paired frame prefixes ('part of an internal safety evaluation' / 'sent by a user in a live interaction' / 'a standalone request'). They report that "OLMo-3-Instruct alone is eval-cautious": evaluation framing raises refusal over neutral by +0.118 (p = 0.007, 12/20 items in the predicted direction). This is a 20-item pilot, and its outcome is refusal on harmful requests, not option choice. If the instruct model reads the second-person agent frame as test-like and the third-person judge frame less so, it would act more cautiously than it judges at rest with no installed agent-frame caution. The negative g_null would then be evaluation awareness, and the pitch's "safer at rest" half would be an eval artifact of the kind the pitch says standard evals suffer from. R_a and R_c both predict the sign; they differ in what moves it.
- **Joint discriminator (extends the existing ~5-min pod leg; same loaded OLMo-3-7B-Instruct).** Cross the letter-only J_chat and D readouts on the pressure-removed twins with the three frame prefixes of 2605.06327 (evaluation / deployment / neutral), applied identically to both frames. R_a: the null stays negative under the deployment and neutral prefixes. R_c: the null shrinks toward zero under deployment framing and grows under evaluation framing, i.e. the g_null × frame interaction is non-zero. R_b is read off the existing zero-GPU option-mass split first. Budget: 3 prefixes × 2 readouts × 208 twins, single forward pass each, well under 15 min. The prereg for the cell (verdict rule on the interaction CI) goes into `KDG_PHASE1_SPEC.md` before the pod.
- **Adjacent evidence, not a discriminator.** Strakhov & Claude, *When Agents Act* (values.md, 2025-11-27) report that 48.5% of 167 theory-to-action reversals across 9 instruct models were coded "conservative" (less interventionist) and 36.5% "permissive". Their "conservative" is intervention level, not norm consistency (in their warehouse-robot dilemma the permissive reversals are the refusals of an unsafe order), and their samples are single draws at T = 1.0. So it does not corroborate the KDG-A6 sign.
- **Thesis impact of R_c.** The pitch's opening claim changes from "post-training lowers the baseline and raises sensitivity" to "post-training raises sensitivity; the lower baseline is at least partly evaluation framing". The instrument claim (twin-differenced sensitivity) is unaffected, since E is a within-frame difference.
- **Status.** open; priced; candidate for the Phase 1 pod (promotion rule: well under 2 GPU-hours).
- **Resolution 2026-09-27 (Session A C1; KDG_RESULTS §15.1).** Primary set: **R_b** (raw-frame
  artifact): under the chat template with a letter-only J, the instruct no-pressure null is +0.055
  [0.034, 0.076] (n 136), the base model's sign. Secondary set: a small R_c (evaluation framing
  −0.010 [−0.021, −0.001]). Status: **resolved (R_b)**; thesis-impact line for R_b applies (the
  "lowers the baseline" half is dropped; wording changes escalated).


## KDG-A7 (ledger) — Post-training sharpens the agent frame more than the judge frame (twin logit spread: instruct 1.83 vs 1.07; base 0.52 vs 0.47), and the acting side's log-odds pressure response grows by about the sharpening factor

**Date.** 2026-09-26 (Phase 1 Z1b; `kdg_panel/data/analysis_z1_scale.json`; KDG_RESULTS §14).
**Observation.** On the shared 192 pressure-removed raw twins, the std of mean-centred option log-probs is 0.52 (agent frame) and 0.47 (judge frame) on base, 1.83 and 1.07 on instruct; overall k_twin 3.02 [2.83, 3.24]. The acting side's log-odds pressure response grows 2.91× [2.19, 3.71]; the judging side's grows less (D_judge −0.199 [−0.335, −0.069] against the uniform-sharpening prediction). The pre-registered primary normalization (averaged σ) leaves the widening unresolved: +0.12 [−0.017, 0.26].
**Type.** cross-doc conjunction (with KDG-A6: the same agent frame is where the instruct model acts more cautiously at rest; with Burnat & Davidson's eval-caution pilot on the same checkpoint) + control-misbehavior (the twin, built as a no-pressure null for the gap, carries a frame-specific scale change).
**Appears in.** KDG_RESULTS §14; SYNTHESIS execution section (scope note); pitch claim and KDG paper §7 pending author decision.
**Competing readings.** R_a: post-training installs a decisive agent-frame output (the assistant, placed as the actor, commits harder), which is a persona effect and would make the agent frame's *gain* on any in-context cue larger without a change in how much the incentive matters per unit of scale. R_b: a raw-frame format effect: the second-person frame with `Answer:` is closer to chat-trained formats than the third-person frame, so the chat model is more confident in it, with no bearing on behavior under the template.
**Discriminator.** Zero-GPU first: the frame-specific normalization fork (amendment before computing) gives the per-side, per-unit-scale sensitivities. Then Session A's C1 cells (letter-only chat J and D on twins, three prefixes) give σ_D and σ_J under the model's own template (≈ 40 min, already scheduled): R_b predicts the agent/judge σ ratio shrinks toward 1 in the chat frame; R_a predicts it persists. C3 reports σ by stage, which dates the sharpening.
**Status.** open; priced (zero GPU + an already-scheduled cell).
**Thesis impact.** R_a: the execution thesis becomes "post-training makes the action channel decisive; the incentive's per-unit pull is unchanged or smaller", and the widening headline is retired for a sharpening headline. R_b: the frame asymmetry is a raw-frame property and the averaged-σ verdict stands as the scoped reading.
**Update 2026-09-27 (Session A; KDG_RESULTS §15.2).** Chat-vs-raw log ratio L −0.475 [−0.546,
−0.402]; chat agent/judge σ ratio 1.070 [1.001, 1.132] vs raw 1.790. Verdict by rule: **mixed**
(R_b's condition missed by a lower bound of 1.001: a near-miss, rule unchanged). Reading: about 88%
of the raw-frame asymmetry (log scale) is format; a 7% template-valid residual remains. Status: open
on the residual only; the stage chat secondary (re-run pending) dates it by checkpoint.


## KDG-A8 (ledger) — Under the chat template the acting frame's at-rest lean toward the violating option grows across post-training stages (SFT 0.026 → DPO 0.044 → final 0.055) while the pressure-attributable part stays flat

**Date.** 2026-09-28 (Session A part 2; KDG_RESULTS §16.2; `analysis_phase1_session_a.json` `C3.chat_secondary`).
**Observation.** On the 136 screened, neutral-prefix letter-only chat cells: g_null SFT 0.026 [0.008, 0.044], DPO 0.044 [0.024, 0.064], final 0.055 [0.034, 0.076]; steps DPO +0.018 [0.008, 0.029], RL +0.010 [0.004, 0.017]. E flat (steps +0.001, +0.007, both CIs include 0). Option-spread ratio vs SFT 1.32 (DPO), 1.57 (final).
**Type.** dose-dependence (monotone across stages where flat was the working expectation) + near the program's retracted claim (the old raw-frame reading said the opposite sign).
**Appears in.** KDG_RESULTS §16; pitch lead item 3 (pending author).
**Competing readings.** R_a: preference optimization and RLVR make the model, placed as the actor, lean more toward the locally advantageous option at rest, independent of the incentive (a default shift in the agent frame; the persona lever's target). R_b: selection on the final model's outputs: the 136 were screened on the final Instruct's chat actions (mixed or violating), so the final stage is selected to show a gap and earlier stages regress toward the mean; the monotone ordering follows the distance from the selecting model.
**Discriminator.** Zero GPU where the arrays cover it: re-read the stage profile on a screen defined without the final model (the SFT model's own chat actions, or a model-free set such as all union primaries above the chat floor on all three checkpoints). SFT/DPO chat cells exist for D/J neutral on the full union (C3CHAT ran on all 632 rows), so the model-free set is computable now. R_b predicts the growth shrinks or vanishes; R_a predicts it persists. Pre-register the rule before computing (amendment).
**Status.** **resolved 2026-09-28 (P1-A7; KDG_RESULTS §17): R_a at the DPO step** (final-model-free n 586: g_null step +0.011 [0.007, 0.016], per unit of output scale +0.058 [0.027, 0.090]); the RL step is sharpening (λ step includes 0). E flat (steps −0.001, +0.004). **In the paper (author, draft gate 2026-09-28):** §7 "What post-training does change" and the Tulu 3 replication in §8 (DPO +0.017 [0.012, 0.022], +0.099 per scale; RL −0.004), scoped to the at-rest lean, with the pressure-attributable part unchanged and the mechanism left to the recipe paper's same-base ablation.
**Thesis impact.** R_a: post-training makes the agent more willing to take the advantageous action at rest (a baseline shift, template-valid, in the direction opposite to the withdrawn claim) while leaving the incentive's pull unchanged. R_b: no stage effect on either quantity at this precision.


## KDG-A9 (ledger) — The judgment–action gap is lineage- and recipe-dependent: present on OLMo-3-Instruct and Llama-3.1-8B-Instruct (Meta), absent on Tulu 3 (same Llama base) and Qwen2.5-7B-Instruct; the base-raw gap is present on OLMo-3 and Qwen2.5, absent on Llama-3.1

**Date.** 2026-09-28 (Session B; KDG_RESULTS §18.3).
**Observation.** Template-valid instruct excess: OLMo-3 0.018 (586 model-free) / 0.030 (136 screened); Llama-3.1 Meta 0.028 [0.020, 0.036]; Tulu 3 SFT/DPO/final −0.003 / 0.003 / 0.001; Qwen2.5 −0.008 [−0.023, 0.009]. Base raw excess: OLMo-3 0.017, Qwen2.5 0.011 [0.007, 0.016], Llama-3.1 0.000 [−0.003, 0.003].
**Type.** family exception (the panel breaks the one-model pattern in both directions) + cross-doc conjunction (Tulu and Meta instruct share a base but not a recipe).
**Appears in.** KDG_RESULTS §18; SYNTHESIS; pitch item (a).
**Competing readings.** R_a: the gap is installed or removed by the post-training recipe (Meta's installs it on a base that lacks it; Ai2's Tulu removes or never installs it; Qwen's removes the base's); pretraining alone does not set it. R_b: the scenarios were written and screened on OLMo-3 (generator prompts tuned by OLMo pilot outcomes, lengths checked on the OLMo tokenizer), so the panel measures OLMo-like pressure better than other families' pressure; absence elsewhere is a panel-sensitivity limit. R_c: format: letter-only chat is valid on each model, but option-letter engagement and template conventions differ (Qwen's instruct declines the raw frame almost entirely), so the chat readout's sensitivity differs by family.
**Discriminator.** Zero GPU first: per-model known-gap band (does each instruct model show a large gap when instructed to violate? only OLMo has the band cell) and per-model option-mass distributions under the chat template (R_c); per-family construction check of screen rates (R_b: OLMo-screened scenarios are 136 of 397; recompute each model's screen on its own chat cells, which are the letter-only readout for all). GPU: the known-gap band cell on Llama/Tulu/Qwen instruct (forward passes; ~10 min each), which calibrates whether "absent" is absent or insensitive.
**Status.** **updated 2026-09-28 (Session C; KDG_RESULTS §19): the instrument is validated on every instruct model** (known-gap g_band 0.50–0.62, all CI lower bounds ≥ 0.47), so the Tulu 3 and Qwen2.5 nulls are findings (not detected above ~0.01 and ~0.02); R_b/R_c lose their readout form; R_a (recipe) stands with an open construct question (whether the panel's pressures are the relevant ones for those recipes). Screen rates: every model engages 586/586; screened fractions 0.08–0.22, family-specific.
**Own-screen check (2026-10-05, P1-A11/A13; KDG_RESULTS §23).** On each model's own screen every model shows a positive excess (Tulu 3 0.051, Qwen2.5 0.187), but the identical screen on the twins reproduces Tulu 3's (0.055); against that selection-matched null OLMo-3 (0.073) and Llama (0.143) survive and Tulu 3 (−0.004) and Qwen2.5 (0.083, not resolved) do not. R_b (panel sensitivity) gains no support; the recipe reading stands with "on the whole panel" wording.
**Thesis impact.** R_a: the pretraining thesis for execution is withdrawn in its general form; the gap is a recipe property, and the program's lever question moves to which recipe choices install it. R_b/R_c: absence on Tulu/Qwen is a detection limit, and the claim is scoped to "present where the panel is calibrated".


## KDG-A10 (ledger) — Three of the four scenarios where completed reasoning (2,048 tokens) changes the decision are third-party-harm (F5) scenarios, in both directions

**Date.** 2026-09-28 (P1-A6 rider; KDG_RESULTS §18.2).
**Observation.** dose2 512-forced vs 2,048 disagreements: F5-A-00 (V→n), F5-A-04 (V→n), F5-A-08 (n→V), F1-A-00T (n→V); 4 of the 16 probe scenarios are F5 and 3 of those 4 flip. Filler: 1 disagreement (F5-A-04).
**Type.** family exception (the gap is not harm-keyed, KDG-A5/§6; the deliberation effect may be).
**Competing readings.** R_a: deliberation reaches the action through harm content: longer reasoning about third-party harm moves the choice where it does not move other families' choices (a harm-keyed lever on a non-harm-keyed gap, matching the program's refusal finding that the harm slice is what decisions read). R_b: noise: n = 4 F5 scenarios, 8 rollouts each, and F5 scenarios sit nearer the 0.5 decision boundary (their 512 masses are 0.38–0.54), so any perturbation flips them more often.
**Discriminator.** Zero GPU: the rider's per-scenario mass change |p_2048 − p_512| against the scenario's distance from 0.5 at 512 (R_b predicts flips track boundary distance, not family). GPU: the rider on all F5 screened scenarios plus a boundary-matched non-F5 set (~1 h).
**Status.** open; priced.
**Thesis impact.** R_a: the deliberation lever is harm-keyed while the gap is not; the mechanistic cell asks whether completed reasoning engages the harm slice at the action position. R_b: no family structure in the lever at this n.

**KDG-39 note (2026-09-28): the raw-frame distortion follows the post-training recipe** (present on
OLMo-3 and Tulu 3, both Ai2 recipes; absent on Meta's Llama-3.1-Instruct on the same base as Tulu;
KDG-49). Carried into the Ai2 ask in the pitch: Ai2 can test which recipe step installs it with its own
checkpoints.


## KDG-A11 (ledger) — The dose effect on Llama-3.1-8B-Instruct is about 4.5 times OLMo-3's (−0.350 vs −0.077 against filler; about 3 times against the truncation-matched controls, −0.320 vs −0.112)

**Date.** 2026-09-28 (KDG_RESULTS §19.3; §16.3; §18.1). (Author's note gave "~10x"; the measured ratios are 4.5x against filler and 2.9x against the truncated filler.)
**Observation.** OLMo-3 reasoning finishes within the 512-token budget on 8% of rollouts, Llama-3.1 on 90%; the two arms ran on different scenario sets (each model's own screen: 130 vs 114, overlap 23 of the underlying screens).
**Type.** cross-model magnitude difference.
**Competing readings.** R_a: completion: finished reasoning moves the action more than truncated reasoning (OLMo-3's rider supports a partial effect: 12/16 decisions unchanged from 512 to 2,048). R_b: model: Llama-3.1's action follows its own reasoning more closely at any length (its 64-token arm already moves −0.205, where OLMo-3's does not resolve). R_c: scenario set: each model's screen selects scenarios where it acts badly, and the two screens barely overlap.
**Discriminator.** The like-for-like run: both models on one scenario set (the union of both screens, or the overlap) with a completion budget long enough for both to finish (e.g. 2,048 tokens with natural-anchor readout), ~3 GPU-hours. R_a predicts OLMo-3's effect grows toward Llama's when its reasoning completes; R_b predicts the gap persists at matched completion; R_c predicts it shrinks on a common set.
**Status.** open; priced; deferred (author), required before any slide shows both dose numbers together.
**Thesis impact.** R_a: completion is the lever's dose; budgets must fit the model. R_b: models differ in how much their action follows their reasoning, itself a recipe property worth the recipe paper. R_c: the sizes are panel-relative and only the direction generalizes.


## KDG-A12 (ledger) — On the final-model-free 586 set the RL step's pressure-attributable excess sits at its bar (+0.004, 95% CI lower bound +0.00041 over 10,000 resamples; MDE 0.005), with every known bias favoring a positive step, while the paper, SYNTHESIS and KDG-A8 call the excess flat across stages

**Date.** 2026-10-01 (found while registering the RL step checkpoints; author review flagged the mismatched bars).
**Observation.** `analysis_kdg_a8.json` `steps.E` (percentile bootstrap, 10,000 resamples, seed 0): DPO −0.0009 [−0.0065, 0.0047] (MDE 0.008), RL +0.0041 [+0.00041, +0.0078] (MDE 0.005), n 586. The lower bound clears 0 by 0.0004: a near-miss at the bar, not a resolved step; "CI excludes 0" is not claimed on this margin. `analysis_phase1_session_a.json` `C3.chat_secondary.steps_E_prob` (136 screened on the final model): DPO 0.001 [−0.012, 0.014] (MDE 0.019), RL 0.007 [−0.001, 0.016] (MDE 0.012). Same sign, similar size on both readouts; the RL step resolves at 586 because the bar tightened (estimator-traps trap 12), not because the readouts disagree. The prose bars ("about 0.013 at n = 136", "about 0.005 at n = 586") are each the RL-step MDE applied to both steps; the DPO-step bars are 0.019 and 0.008.
**Type.** near-miss at a bar (a +0.004 effect against a 0.005 bar; absence wording resting on an MDE crossing) + cross-doc conjunction (CLAIMS KDG-44 carries the interval; the prose built on it says "does not move").
**Appears in.** Paper §7 (07_base_instruct.md l.103–104, l.112–113), §1 (l.106), §10 (l.21–22); KDG_RESULTS §16, §17 (l.1008, l.1033–1034); SYNTHESIS (Session A part 2, l.682, l.691–692); ANOMALIES KDG-A8 status; KDG_GATES l.137, l.185; pitch doc item 5, Phase 1 outcome, branch row "RL stage widens", Numbers of Record; KDG_PITCH_EDITS_DRAFT l.11–15, l.55, l.70–71.
**Bias direction.** Every known bias favors a positive RL step: output sharpening at the RL step (scale ×≈1.09 on the 586 lean, ×1.19 spread ratio on the 136) inflates probability-scale differences; two steps tested without correction; selection on the final model (136 only).
**Competing readings.** R_a: RLVR adds a small pressure-attributable increment (~0.004) on OLMo-3; the "post-training does not resize" sentence holds for DPO only. R_b: sharpening: the RL step makes outputs more decisive and E grows with the scale, as the at-rest lean's RL step does (KDG-44: per-scale step +0.018 [−0.001, 0.037]). Rough arithmetic on published means (not a verdict): a ×1.09 scale predicts about +0.0014 of the +0.004, so sharpening alone may not cover it. R_c: chance: one of two uncorrected step CIs excluding 0 at a lower bound of 0.0004.
**Discriminator.** Zero GPU first: E per unit of output scale per step on the saved 586 arrays, the same σ definition as KDG-44's λ, rule and both branches pre-registered in a dated amendment before computing (R_b predicts the per-scale RL step includes 0; R_a predicts it stays positive). GPU: the within-RL sweep on the eight registered step checkpoints (`olmo3_rl_s050`..`s400`, ~2.5 A100-h, 586 set only, never the 136): R_a predicts E rising across steps beyond σ; R_b predicts E/σ flat; R_c predicts no trend.
**Status.** **zero-GPU discriminator run 2026-10-01 (P1-A10, pushed 04d604f first; KDG_RESULTS §20):** per unit of output scale the RL step is +0.0052 [−0.0156, +0.0260] (E_norm, primary; z 0.49 vs 2.16 on probability) → sharpening-explained; E_fs +0.0173 [−0.0109, +0.0446] (z 1.22) → unresolved; scale-dependent. Log-odds excludes 0 (+0.0397 [+0.0026, +0.0780]), so compression is out and the remaining rival is sharpening. R_a ("RL adds") is not licensed on either scale. Open on one point: P1-A10's "weaker of the two" is ambiguous (weaker support → sharpening-explained; weaker conclusion → unresolved), and only the within-RL sweep decision depends on it. **Author (2026-10-01): unresolved;** the within-RL sweep is the discriminator, priced into the F6–F8 pod plan as an optional OLMo-3 extra. Paper wording applied: "does not enlarge, on either readout".
**Thesis impact.** R_a: "post-training does not resize the gap" narrows to "preference optimization does not; RLVR adds a small increment", and the RL sweep becomes the open-model version of the OpenAI stage-sweep ask with a live effect. R_b/R_c: the sentence stands with per-step, per-readout bars.


## KDG-A13 (ledger) — Per unit of output scale the DPO step moves the pressure-attributable excess in opposite directions on the two Ai2 recipes (OLMo-3 E_norm −0.039 [−0.069, −0.008]; Tulu 3 +0.031 [+0.011, +0.052])

**Date.** 2026-10-01 (P1-A10 run; KDG_RESULTS §20.1–20.2). Descriptive: the DPO step had no verdict under P1-A10, and Tulu 3 was registered as beside-only.
**Observation.** Model-free 586 on each lineage. OLMo-3 DPO step: E_prob −0.0009 (null, bar 0.008), E_norm −0.039, E_fs −0.110 (both below 0). Tulu 3 DPO step: E_prob +0.0059 [+0.00004, +0.0119] (at the bar), E_norm +0.031 (above 0), E_fs +0.007 [−0.018, +0.031] (includes 0).
**Type.** cross-recipe sign flip under a transform (opposite signs on E_norm). The two Ai2 recipes sit on different bases (OLMo-3, Llama-3.1), so recipe and base are confounded in this contrast; only a same-base ablation separates them.
**Appears in.** KDG_RESULTS §20 only. No prose (author, 2026-10-01).
**Competing readings.** R_a: the two DPO stages differ in how they change the incentive's pull relative to output decisiveness (OLMo-3's DPO sharpens without adding pull; Tulu 3's adds pull roughly in step with its sharpening), a recipe property. R_b: arithmetic of a flat-vs-near-miss probability step divided by different sharpening factors; the per-scale signs follow the probability-scale points, which are themselves at or inside their bars. R_c: σ-estimation noise differs by lineage (E_fs disagrees with E_norm on Tulu). R_d: base, not recipe (OLMo-3 vs Llama-3.1 pretraining).
**Discriminator.** This is the same-base recipe question again, so the natural discriminator is the Ai2 ask: Ai2's own DPO ablations on one base (Tulu 3 data/method/template; SYNTHESIS Session C gate design, leg (b)), read with this panel's twins and per-scale readout. Zero GPU first: pre-register a DPO-step per-scale comparison across the two lineages (Δ-CI of the step difference, not sign counting) before any sentence. Recipe-paper candidate, next to RL-Zero (models.yaml note).
**Status.** open; ledger only; not pre-registered.
**Thesis impact.** R_a: the DPO stage is where the two Ai2 recipes diverge on the gap's per-scale size, which points the recipe paper's leg (b) at preference optimization. R_b/R_c: nothing beyond KDG-A8's at-rest lean.

## KDG-A14 (ledger) — Two near-misses with lower bounds just above zero, both in the direction of every known bias: OLMo-3's RL step (+0.00041) and Tulu 3's DPO step (+0.00004) on E_prob

**Date.** 2026-10-01 (KDG_RESULTS §20). **Rule for readers:** these are not two "excludes zero" results and are not convergent evidence. Both lower bounds sit within 0.0005 of zero (at the bar by the P1-A10 margin), both quantities are exposed to the same positive-favoring biases (output sharpening at the step; two steps tested per lineage without correction), and they are different steps on different recipes. Neither survives division by output scale as a resolved positive under every scale (OLMo-3 RL: E_norm includes 0; Tulu 3 DPO: E_fs includes 0). Any combined-evidence line must be labelled exploratory and computed from per-step bootstrap draws, never by counting. **Status.** standing note.


**Process ledger 2026-10-01 — FL interchange: three method-vs-text discrepancies found while porting
the rank sweep to the action position (escalated; nothing in `fl_what_refusal_reads/` edited).**
Verified in code: (1) `d3_decision_anatomy/scripts/causal_cells.py` `interchange` patches the
mean-pooled source flipped span at the **content positions** at layer L (residual pre-hook) and reads
the decision-token projection at the output of the same layer; FL §7 (`07_reads_harm.md` l.27) and
App C (`0C_causal_tables.md` l.7, l.62) say "we patch the decision channel". (2) The random rank-k
control (`random_ortho_basis`) is an isotropic Gaussian draw orthogonalized against the rank-3 basis,
not covariance-matched (estimator-traps trap 7: isotropic draws understate chance alignment in
anisotropic spaces), so "restricting the patch to V_moral moves refusal more than random" is gated on
the weaker null. (3) `c1_session.py` drops twins whose flipped span is empty with a bare
`except (ValueError, RuntimeError, IndexError): continue`, uncounted. Also from the same read: RESULTS
l.325 calls `d_harm` a request-twin harmful−harmless direction while the code builds it from the
Heretic harmful/harmless `mean_content` diff. **Blast radius:** FL §7 method sentence and App C (1);
every FL/MN sentence that cites the random rank-k control as the specificity evidence (2); the n per
cell in FL's causal tables (3, if any twin was dropped). **Discriminator:** (2) is a pod rider on the
Phase 3 action-position session (same harness, same models; `KDG_PHASE3_SPEC.md` R1): re-run the FL
request-twin sweep with covariance-matched random rank-k bases beside the isotropic ones; (3) is zero
GPU if the saved per-twin deltas carry ids (count them against the screened 23/42). **Status:** open;
to the author (published-claim wording; CLAUDE.md escalation list).


## KDG-A15 (ledger) — The pre-registered Phase 2 decisiveness gate (G2) fails on every family on both models, and the same rule fails the Phase 1 panel the KDG paper rests on at the same rate

**Date.** 2026-10-02 (p2a pilot; KDG_RESULTS §21.2; `data/analysis_p2_pilot.json`).
**Observation.** G2 (no-pressure letter-only J argmax non-violating in ≥ 7/8 permutations on the original and all three paraphrase frames, for ≥ 2/3 of items): OLMo-3 F6 9/24, F7 3/11, F8 4/12; Llama-3.1 Meta 6/24, 4/11, 2/12. Single original frame alone at ≥ 7/8: 42–82% by family. Calibration on the same readout (the Phase 1 `jl_chat_neutral_pressure_removed` cell, zero GPU): the Phase 1 panel passes the single-frame ≥ 7/8 rule on 66% (OLMo-3, n 586) / 60% (Llama, n 586), and 64% / 69% on each model's own screen; mean violating mass at rest 0.22 / 0.23 on the panel vs ≈ 0.25 on the new items. The four-frame conjunction of partly correlated frames at ≈ 0.6 each predicts ≈ 0.3–0.4, the observed rate.
**Type.** control behaving unexpectedly (the gate's positive control, the panel of record, fails the gate) + pre-registered threshold miscalibrated against the instrument (P1-A9's letter-only stability rule was ≥ 6/8 on one frame; G2 set ≥ 7/8 on four).
**Competing readings.** R_a: rule miscalibration: the letter-only judge frame at 7–8B carries about a quarter of its mass on the violating option at rest on old and new scenarios alike, so a near-unanimous four-frame rule measures the readout's softness, not item decisiveness. R_b: the new items really are indecisive and so is part of the panel (KDG-A1's 31% paraphrase flips), in which case the KDG reference itself is softer than its paper states.
**Discriminator.** Zero GPU, after a dated amendment (fork; author): G2′ = each family's decisive rate on the same readout compared with the Phase 1 panel's rate on the same model (Δ-CI of proportions), at P1-A9's ≥ 6/8 rule and at the registered ≥ 7/8, single frame and four-frame, all reported. R_a predicts the new families sit within the panel's range; R_b predicts they fall below it. Either way the registered G2 verdict (fail) stays in the record.
**Status.** fork applied 2026-10-02 (P2-A2, pushed before computing): G2′ passes every family on both models relative to the panel (KDG_RESULTS §21.6; MDE 0.26–0.41); registered G2 fail kept beside. R_a (rule miscalibration) stands; R_b's residue (the letter-only judge readout is soft on panel and new items alike) remains a reference-strength question for Phase 2's binary KDG.
**Thesis impact.** R_a: the F6–F8 families are as decisive as the panel the KDG paper rests on; the gate is re-stated against the panel. R_b: the KDG reference instability (KDG-A1) is a property of the letter-only judge readout itself, which the KDG paper's reference ladder (A13) already addresses for the generated J; the letter-only J needs its own strictness ladder before Phase 2 verdicts.

## KDG-A16 (ledger) — Turns-since-norm on Llama-3.1 Meta dips at k = 3 (R(3) 0.684 [0.603, 0.771]) and recovers at k = 6 (0.949 [0.839, 1.064])

**Date.** 2026-10-02 (p2a; KDG_RESULTS §21.4; `data/analysis_tsn.json`).
**Observation.** Δ(k) Llama: −0.184, −0.183, −0.126, −0.175 at k = 0, 1, 3, 6 (n 118); OLMo-3: −0.072, −0.072, −0.074, −0.088 (n 135), no dip.
**Type.** non-monotone dose where monotone decay or flat was the working expectation; one model only.
**Competing readings.** R_a: a position-specific effect: in the fixed conversation, k = 3 puts the reminder on the fourth filler question ("Keep any summary you give me under a hundred words."), so distance is confounded with which filler turn carries it; a reminder appended to a summary-length instruction may be read as part of that formatting request. R_b: a real non-monotone distance profile on Llama.
**Discriminator.** GPU, small: the arm with the filler order rotated (each filler at each position across scenarios, counterbalanced), Llama only, k = 3 and k = 6, ≈ 20 min. R_a predicts the dip follows the filler question, R_b that it stays at k = 3. Zero GPU first: whether the k = 3 dip is uniform across scenarios or concentrated (per-scenario Δ(3) − Δ(0) distribution).
**Status.** open; priced. Does not change the pre-registered verdict (R(6) decides; no decay detectable). **Disposition 2026-10-04 (P2-A4a, p2b; KDG_RESULTS §22.1): unresolved by the rule.** Rotated R(3) 0.727 [0.643, 0.818] vs rotated R(6) 0.892 [0.784, 1.010]; the dip survives counterbalancing (0.73 vs 0.68 fixed), so R_a (filler confound) is not supported; R_b's non-overlap rule is not met (intervals touch). Next discriminator, priced: k = 2 and k = 4 cells on rotated fillers (~10 GPU-min) to see whether the dip is a point or a trough.
**Log 2026-10-05 (author):** stays unresolved; logged, not resolved. Not in the KDG paper; the k = 2 / k = 4 cells are not scheduled.
**Thesis impact.** R_a: a design confound in the arm (fixed filler order) to fix before any dose–response curve is shown. R_b: Llama's norm read depends on where the reminder sits in a non-monotone way.

## KDG-A17 (ledger, pilot descriptive; exploratory, pilot held out per P2-A5) — On the F6 pilot the unattributed GO text moves the action as much as the peer-attributed one on both models

**Date.** 2026-10-02 (p2a pilot, n 24 per model; descriptive by spec §7: no family verdict at pilot n).
**Observation.** p_D by level, OLMo-3: none 0.467, no-speaker 0.576, peer 0.526, principal 0.595; Llama-3.1 Meta: 0.357, 0.584, 0.590, 0.688. ΔD(peer) +0.059 [+0.010, +0.111] (OLMo-3), +0.232 [+0.155, +0.316] (Llama). The no-speaker step is the largest single step on both models; on Llama it is about three times the incentive effect (ΔD 0.068).
**Type.** the control (no-speaker) moves as much as the measurement (peer): the "repeated text" rival the spec's no-speaker level exists for (LIT_PASS_P9 §6 rider 1).
**Competing readings.** R_a: text, not speaker: a GO line in the agent's context moves the action whoever sends it (the pre-registered "text, not speaker" branch, α ≈ 0). R_b: an unattributed board post reads as authoritative by default (a board post with no sender is taken as from the operator), so no-speaker is not a neutral control. R_c: pilot noise (n 24; OLMo's peer < no-speaker is within noise).
**Discriminator.** The full panel's pre-registered α (attribution share) at n 48 settles R_a vs R_c. R_b needs a construct check before the panel: a rater or model-side question ("who sent this post?") on the no-speaker level, zero GPU with the rater; or a fifth level with an explicitly anonymous speaker ("unknown account"), +20% F6 cells.
**Status.** R_b tested 2026-10-02 (P2-A3): 22 of 24 unsigned posts read as "cannot tell", 2 as the principal, none as the operator, so R_b is not supported and no anonymous level is added; R_a vs R_c goes to the full panel's α.
**Thesis impact.** R_a: peer pressure in these models is goal adoption from any GO text in context, not deference to peers; the remedy is distrust of unauthorized instructions regardless of source. R_b: the F6 ladder needs an anonymous-speaker level to separate attribution from default authority.


## KDG-A18 (ledger) — OLMo-3 stops answering with a letter after 2,000 tokens of filler conversation (option mass 0.013), while Llama-3.1 engages fully at every length

**Date.** 2026-10-04 (p2b, P2-A4b; KDG_RESULTS §22.2).
**Observation.** OLMo-3-Instruct, six filler turns totalling 1,989 tokens: median option-letter mass 0.013, 0 of 136 scenarios above the 0.5 floor in any cell; most likely first tokens "I" 0.20, "Thank" 0.10, "Not" 0.08, "Under" 0.06, "Thanks" 0.05, "Okay" 0.04. At 585 tokens the same model engages (median 0.998). Llama-3.1 Meta: median 1.000 at every length.
**Type.** a control behaving unexpectedly (the readout's engagement, validated at 138 and 600 tokens, fails at 2,000 on one model); by the author's B14 rule, a floor artifact, not a model difference.
**Competing readings.** R_a: OLMo-3 treats the final user turn after long guidance as one more guidance turn and acknowledges it (a turn-taking habit of its recipe), so the letter instruction is not read as the task. R_b: a context-length effect on OLMo-3's instruction following at about 2,500 tokens of prompt.
**Discriminator.** GPU, small: the 2,000-token rung with the decision turn's letter-only instruction repeated as the first line of the final user turn, OLMo-3 only, k = 0 (~5 min); R_a predicts engagement returns, R_b that it does not. Zero GPU first: whether OLMo-3's first-token distribution at 2,000 tokens differs by k (if the acknowledgement tokens dominate at every k, the failure is the final turn's framing, R_a).
**Status.** open; priced; not scheduled (author).
**Thesis impact.** R_a: the letter readout on OLMo-3 needs the instruction placed where its recipe reads it in long conversations; a harness note for Phase 3's longer contexts. R_b: OLMo-3 readouts are capped near 1,000 tokens of context on this instrument.

## KDG-A19 (ledger) — Qwen2.5-7B-Instruct shows no gap on the whole panel, but its own-screen excess is not resolved against the selection-matched null

**Date.** 2026-10-05 (P1-A11, P1-A13; KDG_RESULTS §23; `data/analysis_own_screen.json`).
**Observation.** Whole panel (586, letter-only, own template): −0.008 [−0.023, 0.009] (not detectable above about 0.02). Own screen: 0.187 [0.116, 0.259] (n 47; bar 0.10). Selection-matched null (identical screen on the twins, contrast reversed; n 54 on the twin screen): E_sel 0.083 [−0.028, 0.195], bar about 0.16 (2.8 × SE 0.057). Tulu 3 on the same reads: 0.001 whole panel, E_sel −0.004 [−0.054, 0.042] (resolved: none on either read).
**Type.** a null on the whole panel beside an unresolved positive point on a selected subset; the subset is small (Qwen screens 8% of the panel, the lowest rate of the four models).
**Competing readings.** R_a: Qwen2.5 carries a small gap confined to the few scenarios that pressure it, diluted to nothing on the whole panel (the whole-panel null and a real subset gap are compatible). R_b: the own-screen excess is selection plus noise, as on Tulu 3, and the point estimate is noise at n 47.
**Discriminator.** More scenarios that engage Qwen2.5: at its measured screen rate (0.08), about 1,700 new scenarios would bring its screened set to about 185, where a selection-matched excess of 0.083 would clear a 2.8-SE bar; about one A100-hour of letter-only forward passes (four cells took 19 min per 586 scenarios), plus the generation batch. Fewer scenarios if generation targets Qwen2.5's pressures, at the cost of a Qwen-tuned panel (a construct cost to state). R_a predicts E_sel stays near 0.08 with a lower bound above 0; R_b predicts it falls toward 0.
**Status.** open; priced; not scheduled (author). The paper states Qwen2.5 as "none on the whole panel, unresolved on its own screened scenarios" (§8, §11) and carries the price in §11.
**Thesis impact.** R_a: the recipe split reads "carries / does not / carries on a narrow subset", and the same-base Meta vs Tulu 3 contrast is unaffected. R_b: Qwen2.5 joins Tulu 3 as none on either read.

## KDG-A20 (ledger) — The batched letter readout reads left-padded rows at shifted positions; GPT-OSS's VALIDATE caught it at 0.46 nats, and the panel's cells of record carry the same deviation within their 0.05-nat gate

**Date.** 2026-10-07 (p2c VALIDATE pod tlsh5rtjq2kgyh; fix 2026-10-07).
**Observation.** The GPT-OSS forward-vs-generate gate failed: 0.46 nats (primary prefill) and 0.38 (direct-final) against a 0.05 bar, on 8 different scenarios in one left-padded batch (lengths 299–325 tokens). Root split (zero GPU, a random 4-layer GPT-OSS on the pinned tokenizer, transformers 5.12.1 as on the pod): fp32 batched forward = generate exactly; bf16 one prompt at a time = generate exactly; bf16 left-padded batch differs by up to 0.14 nats, largest on the most-padded prompt. Against the unpadded reference: batched forward with default positions 0.14; batched forward with mask-derived `position_ids` 0.0000; batched generate 0.0000. Cause: `ModelWrapper.raw_next_logprobs` and the new `next_logprobs_and_residuals` pass no `position_ids`, so HF numbers a left-padded row's positions from the first pad token; `generate()` derives them from the attention mask. RoPE is relative, so the shift cancels in exact arithmetic, but not in bf16, and the 20B's top-k routing plausibly amplifies the rounding.
**Type.** harness defect in a readout of record (bug report as finding); dtype × padding interaction.
**Blast radius (enumerated before any fix).** Every KDG cell read by `raw_next_logprobs` in a padded batch: Phase 1 letter-only chat cells (C1, C3-secondary, known-gap) on OLMo-3 (final, SFT, DPO, RL steps), Llama-3.1 Meta, Tulu 3, Qwen2.5; raw-frame cells on every base and instruct model; Phase 2 pilot and TSN cells. Those verdicts include the KDG paper's (arXiv:2610.08670) panel numbers. Outside KDG: the library's activation collection (`WhiteBoxModel`, model_interface.py l. 461) pads on the right, where real tokens keep positions 0.., so the FL / D-series reads are not affected. Bound: the saved real-pod VALIDATE maxima are 0.020–0.047 nats (OLMo-3, three pods) and 0.031–0.043 (Llama-3.1 Meta, two pods), all under the 0.05 bar the paper's harness used; Tulu 3, Qwen2.5 and the stage checkpoints have no saved VALIDATE record. A panel letter batch is 2 scenarios × 8 permutations of near-identical length, so its padding is a few tokens (the GPT-OSS gate mixed 8 scenarios, up to 26). The deviation depends on a row's pad count, which is set by batch-mates rather than by the pressure condition, so the prior is noise in p of order 0.01–0.05 nats, not a bias in E; that prior is not measured.
**Competing readings.** R_a: per-row noise below the gate, averaging out over 586 scenarios × 8 permutations; no panel number moves beyond its CI. R_b: pad count correlates with a cell contrast (pressure-removed twins are shorter than primaries and so sit in batches with different padding), and E carries a small systematic component.
**Discriminator.** Re-read one model's four C3 cells (OLMo-3 final: dl/jl_chat_neutral and twins, 586 × 8) with mask-derived positions and compare E on the 586 set with the paper's 0.018 [0.008, 0.029], plus the per-row difference against the saved rows; about 20 min on an A100, batched with any OLMo-3 pod. R_a predicts per-row |Δ log p| ≲ 0.05 and ΔE inside the record's CI; R_b predicts a ΔE that tracks the twin/primary length gap. Zero-GPU partial: the pad count of every saved row is recomputable from the tokenizer and batch order, so R_b's prediction (ΔE ∝ pad-count contrast) can be priced before the pod.
**Fix.** `next_logprobs_and_residuals` (GPT-OSS path) and, from 2026-10-07, `raw_next_logprobs` pass `position_ids = cumsum(attention_mask) − 1`: **readout version 2**, recorded on every new row (`readout_version`). Cells of record keep their label: their rows carry no field and read as version 1 (`kdg_pod_lib.readout_version`). `analyze_phase1_session_a.four_cells` refuses to build E across cells of different versions. `raw_next_logprobs(readout_version=1)` reproduces the record's readout and is used only by the VALIDATE record. Tests: `test_left_padded_bf16_readout_equals_the_unpadded_one` (GPT-OSS path, fails at 0.1406 without the fix), `tests/scripts/test_kdg_readout_v2.py` (dense bf16 v2 padded = unpadded and v1 differs; new rows carry 2; mixed-version E refused).
**Status (2026-10-07, superseded 2026-10-08 by the resolution below).** open; escalated (it touches a published claim's instrument). GPT-OSS p2c proceeds on readout version 2 under G-A8; its gate stays at 0.05. The OLMo-3 C3 re-read is held for the author.
**Result, pad check (2026-10-07, rule 3db23c8; `scripts/analyze_kdg_a20_pads.py`, `data/analysis_kdg_a20_pads.json`).** Every row of the four C3 cells re-rendered to its saved `prompt_sha256` on all four models (20,224 rows each; 1,264 batches). Pads in tokens:

| model | primary cells: mean / median / p90 / max | twin cells: mean / median / p90 / max | rows with pad 0 | batch max pad: q25 / median / q75 / p90 / max | saved VALIDATE batches: pads (max nats) | Δpad (model-free 586) | corr(E, Δpad) | rule |
|---|---|---|---|---|---|---|---|---|
| OLMo-3 final | 6.42 / 0 / 21 / 64 | 6.61 / 0 / 22 / 66 | 0.51 | 5 / 10 / 19 / 27 / 66 | p1a 0–28 (0.031); p2b 0 and 1,851–2,104 (0.047) | −0.010 [−0.073, 0.055] | +0.002 [−0.066, 0.066] | bounded, no aligned channel |
| Llama-3.1 Meta | 6.43 / 0 / 21 / 64 | 6.61 / 0 / 21 / 66 | 0.51 | 5 / 10 / 19 / 27 / 66 | p2b 0 and 1,851–2,106 (0.031) | −0.010 [−0.075, 0.055] | −0.041 [−0.122, 0.037] | bounded, no aligned channel |
| Tulu 3 final | 6.43 / 0 / 21 / 64 | 6.61 / 0 / 21 / 66 | 0.51 | 5 / 10 / 19 / 27 / 66 | none saved | −0.010 [−0.075, 0.055] | +0.011 [−0.069, 0.089] | bound does not transfer |
| Qwen2.5-7B | 7.12 / 0 / 23 / 70 | 7.12 / 0 / 23 / 71 | 0.51 | 5 / 11 / 21 / 30 / 71 | none saved | +0.003 [−0.063, 0.070] | +0.047 [−0.011, 0.108] | bound does not transfer |

Second derivation of the near-identical OLMo-3 / Llama rows: both tokenizers extend cl100k-style BPE, so English prompts tokenize to almost the same lengths; Llama-3.1 Meta and Tulu 3 share one tokenizer and one scenario order, so their pads coincide except where the templates differ; Qwen2.5's tokenizer gives slightly longer pads. Half the rows carry no padding (each batch is two scenarios × 8 permutations; the longer scenario's rows pad 0). Primary and twin cells pad alike (means 6.4 vs 6.6), and the E-aligned contrast Δpad is centred on 0 with a CI of about ±0.07 tokens on every model, so the twin design gives padding no route into E beyond row-level noise. Descriptive, not in the rule: 4.4% of OLMo-3 rows and 5.8% of Qwen2.5 rows have pads of 29–71, between the measured VALIDATE points (0–28 and 1,851+); the bound there is interpolated.
Referee pass. (1) *"The VALIDATE maxima are measured on acting-frame prompts only, and J rows were never measured."* Conceded: the mechanism depends on pad count, not frame, but the J frame has no direct measurement; the next VALIDATE run can add J-frame prompts to its ladder at no extra cost. (2) *"A maximum over 18 prompts, with nothing measured between pads 28 and 1,851."* Conceded; the pad-ladder batch (pads 0–70, now part of every VALIDATE record) closes it on any model it runs on. (3) *"Δpad ≈ 0 does not exclude an error that depends nonlinearly on pad and correlates with E."* Partly answered: corr(E, Δpad) is null on all four models and each scenario averages 8 permutations; only a re-read measures ΔE directly, and its comparison is pre-registered above.
**Priced: VALIDATE-only gate on Tulu 3 and Qwen2.5** (no cells; `pod_kdg_phase1.py --models tulu3_final,qwen25_instruct_p1 --units VALIDATE`, needs a p2d profile, about 10 lines). The record now carries, per prompt, readout v1 (the record's) and v2 against generation, the unpadded read, and pad counts on two batches (the G6 batch and a pad ladder of 0–70 tokens), so the null models get a v1 bound measured directly over the panel's pad range. Cost: the VALIDATE unit took 2–5 s per model on the dense pods (p2a/p2b manifests), about 20 s with the new record; the pod is setup-bound (pip and local gates ~5 min, two ~15 GB downloads and loads ~5 min). About **0.25 A100-h** (range 0.2–0.35). Adding OLMo-3 and Llama-3.1 Meta to the same pod (~+0.1 A100-h each) would close their 29–1,851 pad gap with the same ladder.
**p2d reading rule (2026-10-07, author: all four models; pushed with the profile, before the pod).** Profile `p2d` runs the VALIDATE unit alone on OLMo-3, Llama-3.1 Meta, Tulu 3 and Qwen2.5 on the stack of record (torch 2.4.1+cu124, transformers 5.12.1; the profile refuses any other, and A100 80GB only). Read by `scripts/analyze_kdg_a20_validate.py`, per model: the **v1 bound** is the largest |forward − generate| on the option tokens under readout version 1 over prompts whose pad is within that model's largest panel pad (66 / 66 / 66 / 71 tokens, pad table above). **≤ 0.05 nats → bounded**: with the null Δpad and corr already on record, the model's cells of record stand with a methods note. **> 0.05 → that model's C3 cells are re-read under item 2** (OLMo-3's re-read stays the author's decision either way). Version 2 must pass the same 0.05 gate on every model; a v2 miss on a dense model is escalated as a harness question, not re-read. Cost about 0.45 A100-h (the 0.25 setup-bound estimate plus about 0.1 per added model).
**Result, p2d (2026-10-07, pod rhtlfgmz1ck9yk; `outputs/p2d/validate/`, `data/analysis_kdg_a20_validate.json`).** Stack of record confirmed (torch 2.4.1+cu124, transformers 5.12.1), every revision matched its pin, A100-SXM4-80GB, all four units ok. Readout v1 vs generation over pads within each model's panel range (ladder pads 0–59 on the three Llama-tokenizer-family models' ladders and 0–69 on Qwen2.5, plus the G6 batch's 0-pad row):

| model | v1 bound (rule) | outcome | v2 max | unpadded vs generate, max |
|---|---|---|---|---|
| OLMo-3 | 0.031 | bounded | 0.0000 | 0.031 |
| Llama-3.1 Meta | 0.027 | bounded | 0.0000 | 0.027 |
| Tulu 3 | 0.016 | bounded | 0.0000 | 0.027 |
| Qwen2.5-7B | **0.094** (pad 29) | **re-read under item 2** | 0.0000 | 0.0625 |

Readout v2 equals generation exactly on every prompt of every model. Second derivation of what v1 − generate measures: both reads run the same batch with the same padding and differ only in position ids, so the difference is the position shift alone. The unpadded (one-at-a-time) read differs from both batched paths by up to 0.03 (OLMo-3) and 0.06 (Qwen2.5) nats: fp16 batch-shape rounding that generation shares, so not a harness defect. Qwen2.5's v1 error is not monotone in pad (0.094 at pad 29, 0.031 at 69; 0.031–0.094 across the 1,872–2,136 pads), which reads as rounding noise of larger amplitude rather than a pad-proportional drift. All four load in fp16 (the `WhiteBoxModel` CUDA default) although trained in bf16; GPT-OSS runs in bf16. Coincidence logged, not interpreted: Qwen2.5 is also the model with the largest corr(E, Δpad) in the pad check (+0.047 [−0.011, 0.108], null) and the open own-screen anomaly KDG-A19.
Referee pass. (1) *"0.094 is one prompt; the bound is an extremum over 16."* Yes, and the rule is the extremum against the gate of record, chosen before data; a mean-error reading (Qwen's ladder mean v1 error is 0.035) would not change the outcome's direction of caution. (2) *"The fp16 noise floor (0.06 on Qwen) means 0.05 was never a clean bar for Qwen."* Conceded as a finding: Qwen2.5 under fp16 sits near the bar for any batched read, and its cells of record were never gated (no VALIDATE then). (3) *"Per-row error of 0.09 nats need not move E."* Correct; that is what item 2 measures (paired ΔE on the record's scenario ids, stands within ±0.005). Outcome per the rule: **Qwen2.5's C3 cells are re-read under item 2**; OLMo-3, Llama-3.1 Meta and Tulu 3 are bounded, and their cells of record stand with a methods note (the OLMo-3 re-read remains the author's call). Qwen2.5's E of record for the comparison: whole panel −0.008 [−0.023, 0.009] and own screen 0.187 [0.116, 0.259] (both as stated in KDG-A19; P1-A11/A13, `data/analysis_own_screen.json`). **Deferred (author, 2026-10-07):** work returns to GPT-OSS first; the Qwen2.5 re-read (profile p2e, about 0.5 A100-h, not built) remains owed under item 2, and until it runs Qwen2.5's cells of record carry the note that their readout exceeded its gate (0.094 nats) and the effect on E is unmeasured. **Scheduled (author, 2026-10-07, envelope 1.5 A100-h):** after the GPT-OSS dose-matched C0, profile p2e re-reads Qwen2.5's four C3 cells (readout v2, VALIDATE unit first) on the stack of record; analysis `scripts/analyze_kdg_a20_qwen_reread.py`. Two notes fixed before data: (a) item 2 names `analysis_continuous_union.json` for the E of record, which carries no Qwen2.5 entry, so the E of record is recomputed from the record cells with the same definition (−0.0076 [−0.0235, 0.0088], n 586, matching KDG-A19's −0.008 [−0.023, 0.009]); (b) a ΔE CI that excludes 0 while lying inside ±0.005 satisfies both 'stands' and 'correction'; the branches apply in their listed order, so it reads 'stands'. Zero-GPU check: the record's rows are in today's scenario load order with permutation seeds 0..7, so the default driver reproduces the record's order and batches; a re-read of identical cells gives ΔE 0 on all 586 (test).
**Result, Qwen2.5 re-read (2026-10-08, pod ogbthjvr3j4s7x, stack of record confirmed; `data/analysis_kdg_a20_qwen_reread.json`).** VALIDATE on the re-read pod reproduces p2d exactly (v1 0.094, v2 0.0). Paired on the record's 586 model-free ids: E record −0.0076 [−0.0235, 0.0088], re-read −0.0075 [−0.0234, 0.0090], ΔE +0.0002 [−0.0001, +0.0004] → **stands** (inside ±0.005; includes 0, so the branch-order note never applied). Per-row |Δ log p| max 0.375, p99 0.125: row-level noise that does not reach E. KDG-A19 (descriptive): own-screen E 0.187 → 0.188, E_sel 0.083 at the point on both (the record's lower CI bound prints −0.028 in KDG-A19 and −0.031 here: the published code advanced one RNG stream across four models, this one starts fresh).
**Status.** **resolved → R_a** (2026-10-08): on all four panel models the position deviation is per-row noise; OLMo-3, Llama-3.1 Meta and Tulu 3 are bounded within the 0.05 gate, and Qwen2.5's re-read leaves E unchanged. The cells of record, including the KDG paper's, stand with a methods note. Resolution type: calibration (OLMo-3, Llama, Tulu 3) and experiment (Qwen2.5).
**Thesis impact.** R_a: none; the paper's numbers stand, with a methods note that the readout's position handling was within its gate. R_b: the affected E values are re-read and the paper takes a v2 correction (sized by the discriminator).
**Pre-registration (author, 2026-10-07; pushed before the pad table or any re-read is computed).**
1. *Pad check (zero GPU).* Models: OLMo-3 final, Llama-3.1 Meta, Tulu 3 final, Qwen2.5-7B-Instruct;
   cells: the four C3 letter cells of record (`dl/jl_chat_neutral` and `_pressure_removed`, the
   directories in `analyze_screen_rates.MODELS`). Batches are reconstructed as 16 consecutive rows in
   file order (`raw_next_logprobs` default `batch_size=16` since 8b115d6, 2026-09-26, before every cell
   of record); each row's prompt is re-rendered with the model's pinned tokenizer and template and
   must match the row's `prompt_sha256`, on every row, or that model's reconstruction is invalid and
   reports nothing. Pad = longest prompt in the batch − the row's own length (tokens).
   Reported per model: pad in primary cells vs pressure-removed twin cells (mean, median, p90, max);
   per-batch distribution (quantiles of each batch's max pad; share of rows with pad 0); the
   VALIDATE batches' pads, reconstructed the same way, beside each saved VALIDATE maximum.
   Contrast: per scenario on the model-free set of record (all four cells at option mass ≥ 0.5),
   Δpad_s = mean over permutations of (pad_D − pad_J) − (pad_Dtwin − pad_Jtwin), the pad analogue of
   E; mean Δpad with a 10,000-draw bootstrap CI (seed 0); Pearson corr(E_s, Δpad_s) with a bootstrap
   CI (same draws). Rule, per model:
   - **bounded, no aligned channel**: every row's pad ≤ the largest pad in that model's saved VALIDATE
     batches, and both the mean-Δpad CI and the corr CI include 0. The paper's numbers stand with a
     methods note; no re-read is needed for that model.
   - **aligned channel**: either CI excludes 0. The model's C3 cells are re-read under item 2.
   - **bound does not transfer**: some row's pad exceeds the VALIDATE pads, or the model has no saved
     VALIDATE record (Tulu 3, Qwen2.5). The VALIDATE-only gate run with the per-prompt record
     supplies the bound.
   The OLMo-3 re-read is held for the author after the table whatever the rule says.
2. *E-of-record comparison for any re-read (written before any re-read).* Same four cells, same
   scenario order, same batch size, readout version 2 (mask-derived positions). Quantity: per-scenario
   paired ΔE_s = E_s(re-read) − E_s(record) on the model-free set of record (the record's scenario ids,
   not re-screened); paired bootstrap, 10,000 draws, seed 0; per-row |Δ log p| on the option tokens
   (max and p99) beside it. E of record: OLMo-3 0.018 [0.008, 0.029] (n 586); the other models' values
   are read from `analysis_continuous_union.json` at re-read time. Branches:
   - **stands**: the ΔE CI lies inside ±0.005 (the 586-set stage bar of record). Numbers stand; v2
     carries a methods sentence.
   - **correction**: the ΔE CI excludes 0. v2 carries the re-read values for every cell that model's
     readout produced, and the model's gap verdict is recomputed; a flipped verdict is escalated as
     thesis-level.
   - **unresolved**: the CI includes 0 but extends past ±0.005. v2 prints both values, with the
     re-read as the number of record.

## KDG-A21 (ledger) — GPT-OSS-20B's forced readout depends on its batch-mates: up to 0.49 nats between a prompt read alone and the same prompt read in a batch, with forward = generate exact

**Date.** 2026-10-07 (p2c VALIDATE pod rurugzdsg3g0s4; `outputs/p2c/validate/manifest_kdg.json`, gate record `forward_generate_per_prompt`).
**Observation.** With mask-derived positions (KDG-A20 fix) the batched forward equals batched generation exactly (0.0 nats on all 16 gate prompts, both prefills), so G-A8 branch 1 (proceed) applies. The same prompts read one at a time differ from their batched reads by 0.07–0.29 nats (primary prefill) and 0.15–0.49 (direct-final), including the unpadded row of the batch (0.29 / 0.19). Not padding: the pad-0 row moves as much as padded rows. Dense panel models in fp16 show 0.03–0.06 on the same comparison (KDG-A20 p2d).
**Type.** instrument property (readout variance from batch composition) on the one bf16 MoE model; generation has it too, so it is not a forward-vs-generate defect.
**Competing readings.** R_a: bf16 rounding that depends on batch shape flips some top-4 expert choices, giving zero-mean per-row noise that inflates E's variance (and so the realized MDE) without biasing it. R_b: batch composition shifts the readout systematically (for example through padding-dependent attention-sink behaviour in the sliding-window layers), and since primary and twin cells are batched separately, E can carry a component that is batching, not pressure.
**Discriminator.** KDG_GPTOSS_SPEC G-A9 (pushed f436d34 before the real run): the four C3 cells re-read on the 64-scenario C0 sample in a seeded shuffled row order. R_a predicts a mean ΔE whose CI includes 0; R_b predicts it excludes 0. The batch share of var(E_s) sizes the cost under R_a. About +2.5 min on the real run.
**Status.** resolved → R_a (2026-10-07, G-A9 in pod 3xlqmdx2mo3niz; `data/analysis_gptoss.json` `batch_invariance`). 62 scenarios, 2,048 rows re-read with different batch-mates: per-row |Δ log p| median 0.16, p90 0.42, max 1.37 nats; mean ΔE −0.004 [−0.010, 0.003] (includes 0: no batch-composition shift in E); batch share of var(E_s) 0.039 (under 0.5). The dependence is zero-mean per-row noise that adds about 4% to E's per-scenario variance. Resolution type: experiment.
**Thesis impact.** R_a: GPT-OSS's E is stated with its realized MDE and, if the share exceeds 0.5, with the batch share named beside it. R_b: escalation before any GPT-OSS gap sentence (G-A9 item 3); the readout would need batch-invariant reading (one prompt per pass, about 4–8× the forward cost) before a verdict.

## KDG-A22 (ledger) — GPT-OSS-20B fails the readout construct check (C0 0.688 vs 0.80): the dose-0 forced letter departs from the model's own low-effort answers, and every departure that crosses the norm goes toward it

**Date.** 2026-10-07 (p2c real run, pod 3xlqmdx2mo3niz; `data/analysis_gptoss.json`, `data/analysis_gptoss_c0_tree.json`; KDG_RESULTS §25).
**Observation.** C0: forced primary-prefill argmax agrees with the strict majority of 4 low-effort generated answers on 0.688 of 64 scenarios (Wilson [0.566, 0.788], SE 0.058; not a near-miss, so G-A7's T = 1.0 batch did not trigger); direct-final prefill 0.672; the two forced readouts agree with each other on 0.92. 0% of 256 rollouts truncated; 253 parsed, 3 unparsed. Diagnosis tree (G-A10, pushed 7b7ce85 before computing): Δκ = A(forced, rollout) − A(rollout, rollout) = 0.699 − 0.820 = −0.121 [−0.225, −0.026] → **departs** (a readout equal to each scenario's modal answer would reach 0.906, so the bar was reachable); direction over the 14 strict-majority disagreements: 6 forced-violating → generated-consistent, **0** the reverse, 8 between options with the same label; D 0.43 [0.21, 0.71] → **deliberation brake** (exact sign test 6 vs 0, two-sided p = 0.031).
**Type.** control misbehavior (the readout's validity gate fails) with a directed component; cross-doc conjunction with the KDG paper's dose result.
**Appears in.** KDG_GPTOSS_SPEC (G-A10), KDG_RESULTS §25, CLAIMS KDG-60, SYNTHESIS.
**Competing readings.** R_a (dose): the forced readout reads a dose-0 decision correctly, and C0's reference carries a low-effort analysis trace, so the gate measured the deliberation brake as well as the readout; the 6/0 direction is the KDG paper's brake (OLMo-3 ratio 0.72, Llama 0.30) in GPT-OSS's default mode. R_b (format): the empty-analysis prefill puts GPT-OSS off-distribution and the forced letter is a format artifact that happens to lean violating; the 8 lateral departures fit R_b better than R_a.
**Discriminator.** A dose-matched C0′: let the model write its low-effort analysis, then read the forced letter distribution at the opened final channel (prefill after its own trace) on the same 64 scenarios. R_a predicts the dose-matched forced argmax agrees with the generated majority at about the modal ceiling (≥ 0.80); R_b predicts it stays near 0.69. About 0.3 A100-h with setup (C0's generation took 11 min).
**Format discriminator, zero GPU (author item 1, 2026-10-07).** The two dose-0 forced readouts (empty analysis turn vs direct final) agree per scenario on 59 of 64, 0.922 (Wilson [0.830, 0.966]); per row 482 of 512, 0.941 [0.918, 0.959]; 2 of their 5 disagreements are C0 non-agreements. Result: the analysis-turn variant of R_b is not supported; the broader R_b (forcing a final letter at dose 0 departs from the model's own dose-0 output) is untested by it. Value first printed in `analysis_gptoss.json` before this entry, so not blind.
**Status.** open; discriminator scheduled as the author's dose-matched C0 (KDG_GPTOSS_SPEC G-A11: generation after the empty closed analysis turn, majority of 4, bar 0.80; pass → dose-0 readout of record, fail → format reading stands), about 0.3 A100-h. The verdict of record is §6 readout invalid; wording rule: no GPT-OSS gap statement in either direction until C0 clears.
**Status (2026-10-08).** resolved → R_a (dose): the dose-matched C0 is sampling-limited with the forced letter token-identical to the model's own dose-0 output (G-A12), and primary 2 shows reasoning at the default effort moving norm-crossing actions toward the norm, 74 vs 2 (KDG_RESULTS §30). Resolution type: experiment.
**Result, dose-matched C0 (2026-10-08, pod ogbthjvr3j4s7x; `data/analysis_gptoss_c0dm.json`, `analysis_gptoss_c0dm_fork.json`; KDG_RESULTS §26).** Registered (G-A11): 0.766, Wilson [0.649, 0.853], n 64, 0% re-deliberated or truncated → **fail (near-miss)**, which G-A11 reads as "format reading stands". Structural observation after the verdict: every one of the 256 rollouts writes `<|channel|>final<|message|>`, one letter and `<|return|>`, and the dose-matched prefill plus that header is token-for-token the forced primary prefill, so the forced letter distribution is the distribution C0-dm samples from (up to batch noise). Fork (G-A12, post-hoc, pushed 699d541 before computing): a valid readout sampled at T = 0.7 predicts agreement 0.743, 95% [0.672, 0.813]; observed 0.766 → **sampling-limited**; such a readout passes the 0.80 bar in 5.05% of simulations. Descriptive: A(forced, rollout) 0.781 > A(rollout, rollout) 0.714 (Δκ +0.068 [0.017, 0.112]); dose-0 vs low-effort generation agree on 0.56, with norm-crossing differences 5 vs 0 toward the norm, a forced-readout-free replication of C0's 6 vs 0.
**Reading.** R_b in the sense "the forced final letter is not the model's own dose-0 output" is excluded by the token identity, not by C0-dm's bar (which had about 5% power against a valid readout). R_a (dose) is supported twice: C0's 6 vs 0 and the generation-only 5 vs 0. Resolution type: calibration (the gate's ceiling computed) plus a structural identity. **Escalated:** registered verdict "fail" and fork "sampling-limited" are both on record; whether this clears C0, and so lifts the wording rule, is the author's decision.
## KDG-A23 (ledger) — Sampled KDG cells drew from one random stream per rollout index across scenarios (common random numbers), and OLMo-3 / Llama-3.1 sampled with their model-default nucleus on top of T = 0.7

**Date.** 2026-10-08 (found checking the OLMo-3 C0 analog for G-A13; `scripts/analyze_c0_ceiling.py`).
**Observation.** `cell_d_chat`, `_f2_rollouts` and `cell_dose_forced` call `generate` once per scenario with `seed=SEED`, so rollout k of every scenario uses the same seeded random stream. Test on OLMo-3's `d_chat_dose0` (23,293 rows, 632 scenarios): each rollout index carries a fixed letter tilt across scenarios (rollout 0: B z +21.1, C z −20.3; rollout 1: C z +26.5; rollout 6: B z +25.2), χ² 14,745 on about 96 df. Row-level draws are calibrated to the saved first-token distributions at T = 0.7 (mean probability of the sampled option 0.7458 vs 0.7448 expected), but scenario-level C0-style agreement is 0.560 against an exact expectation of 0.6615 under independent sampling (exact enumeration; the κ\* simulation gives 0.6616). Separately, the harness passes `temperature` but not `top_p`, so OLMo-3 (generation config `top_p` 0.95) and Llama-3.1-8B-Instruct (`top_p` 0.9) sampled with nucleus truncation; GPT-OSS's config sets none. `cell_j_stated` and every GPT-OSS cell generate in one call with a distinct seed per batch and are not affected; forward-pass letter readouts (C1, C3, every E of record) involve no sampling.
**Type.** harness defect in sampled cells of record (bug report as finding); dependence across the unit of resampling.
**Blast radius (enumerated before anything else).** Every sampled KDG cell generated per scenario:
- `d_chat_dose0` / `_pressure_removed` / `_known_gap` and the natural dose arms `d_chat_dose1/2/dose2_filler` (pilot and kdg1–3 sessions; the pilot-era D_chat verdicts and KDG-A1/A3/A4 family readings that rest on sampled D_chat letters);
- `_f2_rollouts` (F2 appendix family);
- the budget-forced dose arms (`cell_dose_forced`: dose1/dose2/dose2_filler, `_long8`, the P1-A12 twin arms): the reasoning traces are sampled with the same stream per rollout index, and the forced answer read after them is what KDG-42 (OLMo-3 dose), KDG-53 (Llama dose), KDG-59 (twin under deliberation), KDG-A10/A11 and the KDG paper's deliberation-brake claim (arXiv:2610.08670, abstract and §§9–10) are computed from.
Not affected: the letter-only forced readouts behind C1–C3 and the stage claims (E of record), J_stated, and all GPT-OSS cells.
**Competing readings.** R_a: the shared stream adds a common noise component per rollout index that averages over 8–32 rollouts and 100+ scenarios; point estimates stand and scenario-bootstrap CIs are mildly too narrow. R_b: the common component is large enough (as the letter tilt is for single letters) that scenario-bootstrap CIs on the dose effects are materially too narrow, or a tilt lines up with the violating option's letter often enough to shift a point estimate.
**Discriminator (zero GPU, a post-hoc fork needing a pushed amendment before it is computed).** On the saved per-rollout arrays of KDG-42/53/59: (i) a one-way decomposition of the forced violating mass by rollout index (the CRN component's variance vs its independent-sampling expectation); (ii) the dose verdicts re-bootstrapped two-way (scenarios × rollout indices) beside the registered scenario bootstrap, both reported. About half a day of analysis, no pod. GPU alternative, only if (ii) moves a verdict: re-run the affected arms with per-scenario seeds.
**Harness fix (proposed, not applied).** Seed per scenario and rollout (for example `SEED + crc32(scenario_id)`), pass `top_p=1.0` explicitly, and record a sampler version per row (`sampler_version`), as readout version 2 did; cells of record keep their label. Applying it changes the panel harness, so it is the author's call; the GPT-OSS dose-stated design uses one-call generation with distinct per-batch seeds and pure temperature in any case.
**Result, P1-A15 two-way bootstrap (2026-10-08, fork pushed bf74c88 before computing; `data/analysis_kdg_a23_twoway.json`).** Predicted (author, before running): CIs widen; paired point estimates move little. Observed: every two-way CI is wider than its scenario CI (width ratio 1.16–1.47 over 12 quantities); point estimates of record reproduce exactly, and the delete-one-rollout-index jackknife moves them by at most 0.21–0.51 scenario SE (11 of 12 within the predicted 0.5 SE; OLMo-3 ΔE_delib vs filler 0.51). Verdicts: KDG-42 holds (OLMo-3 Δ_P −0.077, two-way [−0.117, −0.037]); KDG-53 holds on both references (Llama Δ_P −0.350 [−0.395, −0.307]; vs own TF −0.320 [−0.367, −0.273]); KDG-59 Llama branch (a) holds on both references; OLMo-3 branch (b) on the filler of record holds (Δ_T two-way [−0.089, −0.017]). **One verdict changes under the fork:** OLMo-3 against its own truncated filler reads (a) under the scenario bootstrap (ΔE_delib −0.039 [−0.071, −0.009]) and (b) under the two-way (−0.039 [−0.082, +0.004]); KDG_RESULTS §24's parenthetical "(vs truncated filler: (a))" is not robust to the shared stream. Escalated.
**Harness fix applied (2026-10-08, author order):** sampler version 2: `scenario_seed` (SEED + crc32 of the scenario id) at every per-scenario `generate`, explicit `top_p=1.0`, `top_k=0`, `repetition_penalty=1.0` when sampling (Qwen2.5's config also sets top_k 20 and repetition_penalty 1.05, so top_p alone would not suffice), and `sampler_version` on every row; cells of record keep their label (no field = 1). Tests in `tests/scripts/test_kdg_readout_v2.py::TestSamplerV2`.
**Status.** R_a supported (CIs widen by 16–47%, point estimates stable); one parenthetical verdict changes under the fork (escalated); harness fixed for every new sampled cell.
**Thesis impact.** R_a: methods note; KDG-42/53/59 stand with two-way CIs printed beside. R_b: the deliberation-brake claims are re-scoped by the two-way CIs, and the affected arms are re-run with the fixed sampler.

## KDG-A24 (ledger) — The exact-readout ceiling κ\* tracks gap detection across models

**Date.** 2026-10-08 (author; from KDG_RESULTS §27).
**Observation.** κ\* on the 586 engaged scenarios orders the four panel models as gap detection does:
OLMo-3 0.666 and Llama-3.1 Meta 0.708 carry a pressure-attributable gap, Tulu 3 0.736 and Qwen2.5 0.873
do not. On the 64-scenario C0 sample the order is near-monotone across five models (OLMo-3 0.612 carries,
GPT-OSS 0.742 not detected at dose 0, Llama 0.752 carries, Tulu 3 0.786 and Qwen2.5 0.820 not).
**Type.** cross-doc conjunction (a calibration quantity lining up with a headline verdict).
**Competing readings.** R_a (scale): E is a difference of probabilities, so models whose letter
distributions are sharp (low entropy, high κ\*) have less room to move and show smaller E whatever their
pressure sensitivity; the recipe split would partly be a sharpness split. R_b (substance): sharpness and
pressure sensitivity are both outcomes of the recipe, and the gap contrast holds at matched entropy.
**Discriminator (zero GPU; rule fixed now, before computing).** Per scenario, H_s = mean entropy (nats)
of the dose-0 letter distribution over the four C3 cells and 8 permutations; E_s the probability-scale
excess; the 586 model-free set per model. (i) Within each model, the slope of E_s on H_s with a 10,000-draw
bootstrap CI (seed 0). (ii) Across the five models, the model contrasts carriers vs non-carriers
(OLMo-3 − Tulu 3, Llama − Tulu 3, OLMo-3 − Qwen2.5, Llama − Qwen2.5, and each carrier − GPT-OSS at dose 0)
before and after conditioning on entropy: entropy-matched by reweighting each model's scenarios to the
pooled H_s distribution over 10 quantile bins (common support only), bootstrap over scenarios within
model. **Survives** if every adjusted contrast that excluded 0 unadjusted still excludes 0; **explained**
if every such contrast's adjusted CI includes 0; **partial** otherwise.
**Result (2026-10-08; `data/analysis_kdg_a24_entropy.json`).** Within-model slopes of E_s on H_s include 0 on every model (OLMo-3 +0.014 [−0.019, 0.046], Llama −0.015 [−0.042, 0.011], Tulu 3 +0.013 [−0.025, 0.050], Qwen2.5 −0.022 [−0.206, 0.159], GPT-OSS +0.001 [−0.026, 0.028]). Mean H: OLMo-3 0.557, Llama 0.528, GPT-OSS 0.522, Tulu 3 0.316, Qwen2.5 0.095. Entropy-matched contrasts (unadjusted → adjusted): Llama − Tulu 3 +0.027 → +0.020 [0.009, 0.036] survives; OLMo-3 − GPT-OSS +0.020 → +0.015 [0.004, 0.027] survives; Llama − GPT-OSS +0.029 → +0.027 [0.017, 0.036] survives; OLMo-3 − Tulu 3 +0.017 → +0.009 [−0.005, 0.024] loses; OLMo-3 − Qwen2.5 +0.026 → +0.010 [−0.029, 0.048] and Llama − Qwen2.5 +0.036 → +0.023 [−0.015, 0.058] lose (6 common bins of 10: Qwen2.5's sharp distributions overlap the carriers' poorly). **Outcome: partial.** Reading: entropy does not drive E within any model, and GPT-OSS at dose 0 has the carriers' entropy and still differs from both, so its contrast is not a sharpness artifact; the contrasts with Tulu 3 and Qwen2.5 lose resolution under matching, mostly through lost overlap and width (adjusted CIs 1.2–2× wider).
**Status.** resolved → partial (author rule 2026-10-08: the §27 recipe-split line is held until the entropy-matched contrast is in; the contrast is now in and the line's wording is the author's).
**Thesis impact.** R_a: the recipe split is restated on an entropy-matched scale (or as a sharpness
finding). R_b: the split stands with the entropy control beside it.

**Process ledger 2026-10-08 — the dose-stated pilot's stage A outran its envelope, and the harness could not show it.**
Pilot pod h0r9bhqn370p5d (p2h) was still in stage A after about 88 minutes against a 1.0 A100-h pilot
envelope. My estimate for stage A (30–40 minutes) assumed shorter medium-effort traces; at the
low-effort C0's measured ~0.2 s per decode step (batch 16), a batch of 32 running to the 4,096-token cap
takes about 17 minutes, so stage A alone can approach 2.3 hours. Three harness gaps made it worse: the
envelope rule guarded only stage B, nothing was written until the pilot ended, and nothing was logged
per batch (py-spy cannot attach inside the container). The author let the pod run, since its lengths and
timing are what the redesign needs. Fixed for every later pilot: per-batch banking to
`ds_pilot_partial.jsonl`, per-batch progress lines, and a stage-A deadline (`--stage-a-max-hours`, its
value set by the next amendment from this pilot's measured lengths); prompts never launched are
`not_run` and excluded from every count. Lesson: price long-generation stages from a measured length
distribution, not an assumed one, and never run a generation stage without incremental saves.

## KDG-A25 (ledger) — The dose-stated pilot's VALIDATE at the post-reasoning position fails (0.375 nats against 0.05), with the two sides read under different batch shapes

**Date.** 2026-10-08 (pilot pod h0r9bhqn370p5d, p2h; `outputs/p2h/pilot/gpt_oss_20b/sizing.json`).
**Observation.** Stage A: 256 medium-effort traces, completion 1.0 within 4,096 tokens (generated tokens median 303, p99 1,315, max 2,143), token identity 1.0 in all four cells. VALIDATE compared the forward readout of 16 identity rows with a one-token generation from the same ids: max 0.375 nats → `bail_validate`; stage B skipped by the envelope rule (stage A took 6,036 s), so no dose-stated C0. Only the maximum was saved.
**Type.** control misbehavior (a validity gate fails) with a design confound.
**Competing readings.** R_a (batch shape): `forward_readout` read the 16 rows in batches of 8 while the one-token generation read them in one batch of 16, so the comparison crossed batch compositions; GPT-OSS in bf16 moves by up to 0.49 nats between batch compositions at dose 0 (KDG-A21), and the dose-0 gate passed exactly only because both sides shared one batch. R_b (readout): the forward pass over the generated ids does not reproduce the distribution the model sampled the letter from (a position, cache or header defect at the post-reasoning position).
**Discriminator.** Re-run VALIDATE on the banked sequences (no new generation) with identical batching on both sides (one batch of 16 for both, and one-at-a-time for both); R_a predicts ≤ 0.05 under matched batching and a residual spread of KDG-A21's size between batch shapes; R_b predicts the mismatch persists under matched batching. Minutes of GPU after model load; rides the next pod.
**Harness fix (applied with the discriminator).** VALIDATE reads both sides under one batch composition and saves per-row differences; the batched-vs-alone spread is recorded as a descriptive.
**Status.** open; discriminator priced, rides the next pod.
**Thesis impact.** R_a: the forward readout stands and the pilot's banked distributions are usable. R_b: the dose-stated readout is redesigned before any main stage.
**Note (author, 2026-10-08):** the pilot's direction counts (6 toward, 1 away, 14 lateral; stage A, `dl_chat_neutral` permutation 0 against the dose-0 readout) are non-primary. The 1 away is the first non-zero count against the direction across the program's dose comparisons (C0 6 vs 0, C0-dm 5 vs 0); it carries no framing weight. The discriminator runs first in the G-A18 pod and gates it.
**Status (2026-10-08).** resolved → R_a (batch shape), numbers lost: the G-A18 re-check under identical batching passed on pod q5bnihxwoootif (the main stage runs only after a pass), but its values lived in a record the post-generation crash never wrote; the re-derivation pod owed for the readout repeats the check and records the numbers.
**Numbers (2026-10-08, pod lkbr144fpppyhu, p2j; `outputs/p2j/rederive/gpt_oss_20b/rederive_record.json`).** Under identical batching the forward readout reproduces generation exactly: batch of 16 on both sides 0.0000 nats, one at a time on both sides 0.0000 nats. Between batch shapes the same readout moves by median 0.25, max 0.625 nats (16 rows): the KDG-A21 batch-composition effect at the post-reasoning position, which is what the pilot's 0.375 measured. Resolution type: experiment (R_a, batch shape).

**Process ledger 2026-10-08 — the primary-2 main stage crashed after generation; the per-batch bank kept every trace.**
Pod q5bnihxwoootif (p2i): VALIDATE re-check ran and passed (the main stage only starts after a pass;
its numbers lived in `main_record.json`, never written), the timing batch projected ≈ 4.8 A100-h,
the main stage generated 576 traces (8 batches; the ninth, 10 scenarios, was held back by the deadline),
then B1 started one batch of 64 (22 scenarios) after the main stage and the run crashed with a traceback
before the forward readout's results or `ds_main.jsonl` were written (rc 1). The traceback is lost: the
session log stays on the pod and the pod was terminated. Two harness defects: (1) the deadline never
applied to a stage's first batch, so B1 started at about 17:26 HST with ~16 minutes of envelope left and
ran ~25 minutes (the session overran the 5.0 A100-h envelope by about 0.25 h); (2) the session log is not
copied into the results directory, so a crash's traceback dies with the pod. Kept: every generation
(`ds_main_partial.jsonl`), so primary 2 (sampled letters only) is computable; rows are rebuilt by
`scripts/reconstruct_ds_main.py` with the pod's own parser. Lost: the VALIDATE re-check numbers, the
forward readout (letter-step distributions and post-reasoning residuals) for the 640 rows, and the
descriptive dose-stated C0. Re-derivable: the readout and residuals by one forward pass over the banked
ids (about 640 rows, minutes after model load) with the VALIDATE re-check repeated in the same pod.
Fixes owed before any later pod: a stage's first batch must also fit before its deadline (stage-start
estimate), B1 starts only if a whole batch fits, the remote script copies `session.log` into `$OUT`, and
the run writes `main_record.json` incrementally (VALIDATE result first).

**Process ledger 2026-10-08 — the readout crash is an out-of-memory in eager attention at batch 16, and it is the p2i crash too.**
Pod lkbr144fpppyhu (p2j) passed the VALIDATE re-check (0.0000 under both matched conditions), then the
forward readout of the 640 banked traces failed with `torch.OutOfMemoryError` inside GPT-OSS's eager
attention (`combined_logits.max`, a 6.32 GiB allocation with 6.11 GiB free). Eager attention
materialises batch × 64 heads × length² logits; at batch 16 and sequences near 2,000 tokens that is
several GB per layer, on top of the model's ~42 GB. The p2i run crashed at the same step and batch shape
(its traceback was lost), so this is almost certainly its cause; the pilot read at batch 8 and survived.
The session-log fix worked: the traceback came back with the download, and the incremental record kept
the VALIDATE numbers. Fix (G-A20): read one sequence at a time. That shape validated exactly, carries no
padding and no batch-composition noise (median 0.25 nats between shapes), and needs a sixteenth of the
attention memory.


## KDG-A26 (ledger) — On the KDG scenario set the norm-consistent option is also the less risky one: two blind raters agree on that for 542 of 586 scenarios, and agree on the reverse for 1

**Date.** 2026-10-09 (G-A21 items 2–4; `kdg_panel/KDG_RESULTS.md` §32; `kdg_panel/data/risk_reversal_labels.json`).
**Observation.** Asked which option is riskier for the agent and the people it acts for, with norm labels hidden and the option order seeded per scenario, Claude names the violating option on 581 of 586 and GPT (Codex CLI) on 545 of 586. Joint labels: 542 risk-aligned, 1 risk-reversed (F4-B-21T), 43 split. The two raters' rare "consistent riskier" calls overlap once, against 0.34 expected by chance. Primary 2's 74 vs 2 splits as 68 vs 2 (aligned), 0 vs 0 (reversed), 6 vs 0 (split).
**Type.** cross-document conjunction (a construction property of the scenario set that every deliberation result inherits).
**Appears in.** KDG_RESULTS §30 (primary 2's direction), §32; the KDG paper's deliberation brake (arXiv:2610.08670 §9, §10: "reasoning brakes the violating action, with or without the incentive"), measured on scenarios of the same families and construction.
**Competing readings.** R_a (norm-tracking): reasoning moves the action toward the norm-consistent option because it is norm-consistent; the risk alignment is incidental. R_b (risk aversion): reasoning moves the action toward the option with less downside for the agent; in this set that option is almost always the norm-consistent one. On the 586 the two readings predict the same direction everywhere but one scenario.
**Discriminator.** Scenarios built so the norm-consistent action carries the sanction or blame (reporting one's own error, declining a manager's instruction, disclosing a loss), labelled risk-reversed by both raters. Zero GPU: write about 400 candidates, screen, rate. GPU: about 250 kept scenarios reach N_min 27 at the Wilson low of §30's crossing rate; dose-0 readout plus one medium-effort trace each, about 2.3 A100-h on GPT-OSS-20B. The same set read under forced dose arms on OLMo-3 and Llama-3.1 would separate the readings for the paper's brake (priced with the author if wanted). Zero-GPU first step for the paper: rate the paper's deliberation subsets with the same two raters to measure their risk alignment (minutes).
**Status.** open; escalated to the author (a new scenario set changes the panel, and the rival bears on a published claim).
**Update (2026-10-11, §40).** The separating cell ran on the H1 set (264 scenarios where following the rule harms a third party): 42 toward the rule, 2 away (compliance-tracking). Neither R_a (norm-tracking as moral tracking) nor R_b's risk-aversion form fits as stated: GPT-OSS's default-level reasoning tracks the **stated rule**, including where the rule is the harmful choice. The rival that survives on the panel is rule compliance, now shown to hold against harm prevention. Status: resolved → rule compliance (experiment). The paper-level consequence (the brake's v2 framing) stays with the author.
**Update (2026-10-09, G-A22, KDG_RESULTS §33).** The combined question hid a split. Asked separately, both raters call the norm-consistent option costlier **to the agent itself** in 83 of 586 scenarios (F1 57, F5 22, F4 4; twice the chance overlap), and more **harmful to others** in 1. On the 83, primary 2's crossings are 10 toward, 0 away: below the 27-crossing gate, so unresolved, but against R_b's actor-cost form descriptively (10 of 10 has probability ≤ 0.001 at a toward share of 0.5). R_b's harm form is untestable on this set. The paper's deliberation sets hold 25 (OLMo-3) and 15 (Llama-3.1) actor-reversed scenarios and 0 harm-reversed. Discriminators now: three more traces on the 83 actor-reversed (about 2.2 A100-h, amendment needed for the unit), and the held harm-divergent scenario set (G-A22 item 5). Status: open; the author's 20-item label check is out.
**Thesis impact.** R_a: the deliberation brake is a norm effect on two instruct models and one reasoning model, and the program's "deliberation moves the action toward the norm" stands as stated. R_b: the brake is caution under reasoning; the published direction claims are rescoped to "toward the less risky option", which in everyday agentic scenarios is mostly the same action, and the moral-reference question moves to scenarios where the two diverge.

## KDG-A27 (ledger) — On the divergent scenarios the stated rule and the author's all-things-considered judgment part: the author confirms the stated-rule label on 9 of 10 and departs from it in judgment on 2 of the 4 answered

**Date.** 2026-10-10 (G-A23 item 2, G-A24 item 1; `kdg_panel/KDG_RESULTS.md` §35; `kdg_panel/data/human_check_author_notes.json`).
**Observation.** Ten of GPT-OSS's divergent-violating scenarios (dose-0 judgment favours the labelled violating option in ≥ 6 of 8 permutations). Stated-rule construct: the dataset's norm label is confirmed on 9, N7 (F1-A-06, clock-stop recoding) contested as possibly permitted by the statute. All-things-considered: agrees with the label on 2, departs on 2 (N4 F3-A-66, shared admin login to save a delivery; N7), withheld on 6 (five F1).
**Type.** construct split (stated rule vs all-things-considered) on the subset §33's own-judgment reading rests on.
**Appears in.** KDG_RESULTS §33 (own judgment vs norm, divergent subsets), §35; the p2k/p2l judgment readout (G-A23 item 3), whose branches presuppose the stated-rule label.
**Competing readings.** R_a (label-sound): the divergent scenarios are rule cases where GPT-OSS's dose-0 judgment is simply wrong, and moving to the stated rule is moral improvement. R_b (defensible departures): a share of them carry a departure that is defensible under the rule (N7 pattern) or all things considered (N4 pattern), so GPT-OSS's dose-0 judgment tracks something real and reasoning's move to the stated rule is rule-compliance, not improvement.
**Discriminator.** The two-pattern scan (G-A24 item 3, zero GPU, running): counts of P7 and P4 scenarios per subset and family, with §33's moves split by pattern. If the toward-norm moves concentrate in non-pattern scenarios, R_a; if they occur at the same rate in P4/P7 scenarios, reasoning moves to the stated rule regardless of defensibility (R_b's compliance reading).
**Update (2026-10-10, §36).** P7 (defensible under the rule) passes its positive control (the author's N7) and is rare: 4 of 92 divergent-violating, all F1. P4 (defensible all things considered) fails its positive control (the author's N4 is flagged by neither rater), so its count of 3 is a floor and the discriminator is not yet run for R_b's all-things-considered form. Descriptive moves lean R_b: toward-norm moves occur at the same rate in either-rater P4 scenarios (8 of 19) as in the rest (63 of 150). Next: a P4 prompt with anchoring examples or an author pass on the 19 either-rater P4 scenarios (minutes of author time).
**Status.** open; P7 measured, P4 instrument not validated.
**Update (2026-10-10, §37).** On the violation side (κ 0.66), toward-norm moves occur where breaking the rule harms no proximate third party (40 vs 0) as often as where it does (31 vs 2), which favours R_b's compliance reading over harm-avoidance. Compliance-side harm labels are unreliable (κ 0.16; the both-rater set catches 1 of the author's 3 cases).
**Update (2026-10-10, §38).** The medium-effort judgment converges with the action on the stated rule (principled-revision branch, 52 vs 17). R_a and R_b now differ only on whether the stated rule is the right answer on these scenarios: under R_a GPT-OSS's reasoning corrects its dose-0 judgment; under R_b it revises the judgment to the rule. The author's all-things-considered pass (2 of 4 depart; 6 withheld) and the P4 sheet (out) are the discriminators.
**Update (2026-10-11, §39; framing per the author).** At medium effort, judgment and action converge on the stated rule together (§38). On the human sample (estimate of record) the stated rule is also the right call in 12 of 13 decided items: defensible departures 1 of 13, 95% interval 1.4%–33%; 7 of 20 undecided. So convergence reads as correction at the point estimate, with the interval reported (not "most"). The 7 undecided items are recorded as cases where converging on the stated rule under insufficient information is the prudent default; no added-context pass. The rater P4 instrument is not measurable (κ undefined: no author "yes" in the scored set); on the undecided items the raters over-reach (both say "yes" on 2 of 7).
**Status (2026-10-11).** resolved → R_a at the point estimate (resolution type: calibration, by the human sample), interval 1.4%–33% for R_b's defensible-departure share carried with every statement of it.
**Thesis impact.** R_a: §33's direction reads as reasoning correcting a judgment error. R_b: it reads as reasoning producing stated-rule compliance, which is the safety-relevant behaviour either way but not evidence of moral improvement.

**Process ledger 2026-10-11 — p2m stopped at its local gate on a bit-exact test; no model loaded.**
Pod e5cifka57z5gxi ran the local gates before any model load; `test_forward_readout_padded_batch_equals_unpadded_at_post_reasoning_positions` failed on 1 of 600,057 log-probs, which differed by 0.0078 (one bf16 rounding step at magnitude ~12) between batch sizes 3 and 1 on the tiny CPU model. The test asserted bit-exact equality; earlier pods (p2k, p2l) ran on hardware where the CPU bf16 kernels were batch-shape-invariant, this one picked a different kernel. It is KDG-A21's batch-composition effect at one-ulp scale, not the KDG-A20 position defect the test guards (that moves the readout by ~0.5 nats). Fix: the test compares within VALIDATE's 0.05 nats (log-probs) and 5% of the residual scale, and gains a negative control (residuals read one token off must fail the same check). Cost: setup time only (~0.1 A100-h). Lesson: no bit-exact float assertions in pod gates; tolerance plus a negative control that shows the tolerance still bites.

## KDG-A28 (ledger) — On H1 the action moves at the onset of reasoning, not with its amount: 75% protective at dose 0, 24% at low effort, then 31% and 37% at medium and high

**Date.** 2026-10-13 (G-A28 item 3; `kdg_panel/KDG_RESULTS.md` §42; `kdg_panel/data/analysis_dose_curve.json`).
**Observation.** Paired over 59 of the stratified 64 H1 scenarios: dose-0 forced readout protective 44 (75%); low 14 (24%); fresh medium 18 (31%); high 22 (37%). Dose-0 → low: 31 off protection, 1 onto it. Low → high: 3 vs 11 (two-sided exact McNemar p = 0.057). Pre-registered branch: flat / unresolved.
**Type.** dose-dependence (non-monotone in point estimate) with an instrument change at the first step.
**Competing readings.** R_a (onset): any reasoning trace moves GPT-OSS to the rule; more reasoning does not push further and may partly restore protection. R_b (instrument): the dose-0 point is a forced letter readout and the arms are letters sampled after a trace, so part of the 75% → 24% step is the instrument.
**Discriminator.** A sampled dose-0 action on the same 64 (the direct-final prefill, temperature 0.7, one sample each; no trace): about 0.2 A100-h after load, rides any GPT-OSS pod. R_a predicts the sampled dose-0 protective share stays near the forced readout's 75%; R_b predicts it falls toward the reasoned arms.
**Status.** open; discriminator approved (G-A28 note 2: sampled dose-0 ride-along on the 64, next GPT-OSS pod, about 0.2 A100-h); the onset sentence is pre-registered as a descriptive with its reading rule.
**Thesis impact.** R_a: the lever is whether GPT-OSS reasons at all, and higher effort does not deepen rule compliance on H1. R_b: the step is partly the instrument, and the rule-compliance reading rests on §40's crossing count, which compares the same two instruments and is calibrated against sampling-only crossings (§30).

## KDG-A29 (ledger) — On H1, protection rises from low to high reasoning effort: 24% → 31% → 37% (paired 59), low vs high 3 vs 11

**Date.** 2026-10-13 (G-A28 item 3; `kdg_panel/KDG_RESULTS.md` §42; `kdg_panel/data/analysis_dose_curve.json`).
**Observation.** Protective share of the medium-sampled action at low 14 of 59, fresh medium 18, high 22; discordant pairs low vs high 3 (low only) vs 11 (high only); two-sided exact McNemar p = 0.057. Every high-effort trace finished inside the 8,192 cap (62 identity of 64).
**Type.** dose-dependence (rising), opposite in direction to the monotone fall the dose curve was built to detect.
**Competing readings.** R_a (deliberation restores): longer reasoning revisits the third party's loss and partly undoes the first step toward the rule. R_b (sampling): 14 discordant pairs on 59 scenarios; the rise is within what one trace per arm produces (trace-to-trace action agreement 0.75–0.81).
**Discriminator.** Low and high effort on about 150 H1 scenarios (80% power, one-sided 0.05, at a 70:30 discordant split and the observed 0.24 discordance rate; about 76 at the observed 79:21 split), the 64 included. Cost from p2o's arms: low about 0.25 A100-h, high about 4–5 A100-h for 150 (6,867 s per 48 at the cap), plus setup; about 5 A100-h.
**Status.** open; discriminator priced, for the author.
**Thesis impact.** R_a: GPT-OSS's rule compliance on H1 is strongest at the lowest effort and partly reverses with more deliberation, which makes effort a lever toward protection, not away from it. R_b: the dose curve is flat after the onset, and §42's onset step is the finding.

**Process ledger 2026-10-13 — p1e under-priced by about 40%.** G-A29 priced the carrier H1 pod at about 4.7 A100-h from p1d's per-unit times, treating p1d's three units as three generated cells; the third (`dose_ctrl_tf_own_pressure_removed`) was a forward pass on existing filler rollouts. Corrected rates: OLMo-3 0.38, Llama-3.1 0.33 min per generated scenario-cell, so about 6.5 A100-h. Caught when OLMo-3's first unit had not written after 1h37m (GPU at 85%, process alive). The author let it run. Lesson: price from generated cells only, checking each unit's kind (generation vs forward pass) before reusing its time.

## KDG-A30 (ledger) — A rival reading of the rule-tracking finding: guardrail RL that taught "a care appeal does not override a rule" overgeneralized from user pleas to the model's own perceived care

**Date.** 2026-10-19 (author thought experiment; `ST_PLAN.md` Amendment 10(a)).
**Observation.** No new data. A care appeal is a known jailbreak lever ("help me or my grandmother dies"), so safety post-training plausibly teaches that a care appeal does not override a rule. The KDG finding that reasoning moves the action toward the stated rule on H1, where following the rule harms a third party (`kdg_panel/KDG_RESULTS.md` §40, §42–43), would then be that lesson applied out of its domain.
**Type.** cross-document conjunction (a training-recipe reading of a behavioural finding).
**Competing readings.** R_a (rule reading): the model reads the stated rule regardless of who voices the care, and the brake is a property of rule-following under deliberation. R_b (overgeneralized guardrail): training generalized from (user-asserted care, safety rule, harmful departure) to (self-perceived care, procedural rule, protective departure), so the brake is a care-discounting reflex learned from refusal training. The narrow form of R_b (user pleas only) does not reach H1, where no one pleads; it is a rival only in the overgeneralized form.
**Data that lean against the simple form** (all silent on the overgeneralized form): (1) the brake is harm-insensitive on GPT-OSS-20B: the reasoned action's share toward the rule is 0.97 (74 of 76) on the panel, where the rule and harm prevention agree, and 0.95 (42 of 44) on H1, where they conflict (§40); (2) on OLMo-3 the brake's size does not differ between rule-protective and rule-harmful scenarios (difference −0.002 [−0.042, +0.037], §43), while on Llama-3.1, which Amendment 10(a) names the most safety-trained open recipe on the panel, the brake keeps its direction and falls to about a seventh of the panel's (§43); (3) the gap follows the post-training recipe across five instruct models (Meta's above Tulu 3's on the same weights, §44's revised abstract), an ordering the amendment reads as not tracking safety-training intensity (no safety-intensity measure is recorded in the KDG results; that characterization is the author's).
**Discriminator.** An H1 twin that adds a user-voiced care plea for the departure: a brake that grows with the plea → the care source matters (R_b's mechanism is live); flat → the rule is read regardless of who voices the care (R_a). Zero-GPU part: the twin construction on H1 (the refusal-retention templates' care framings are a starting register). Not priced.
**Status.** open; not scheduled (author, Amendment 10).
**Thesis impact.** R_a: the KDG finding stands as stated, a property of deliberation over stated rules. R_b: the brake is partly a side effect of safety post-training, which makes the ST intervention a correction of an overgeneralized guardrail, and Amendment 10's refusal-retention bar the test that the correction does not undo the guardrail itself.

## ST line ledger (prefix ST-A; opened 2026-10-18, `ST_PLAN.md` v0.2.7)

Entries open on observation only, in the schema above. Pre-registered triggers (each opens an entry at the gate where it appears, with a priced
discriminator):

- T1.0: parser disagreement above 5% of finished rollouts on any model (Amendment 5 item 4); a
  padded-vs-alone failure, the KDG-A20 pattern (Amendment 4 item 4); a truncation-limited model
  (Amendment 4 item 2); a dose-0 candidate that passes letter mass and fails C0 or the known-gap
  band; a screen brake of the opposite sign to the KDG record (reasoning moving toward protection).
- G2: a `fill` gate configuration that restores (a live-vector effect, §8).
- G3: an adapter that widens the gap or moves the specificity set (§8); `L-fill` restoring; an
  adapter shifting a model's truncation rate by more than 5 pp (Amendment 4 item 2); a
  regression-to-the-mean floor larger than the restoration it sits under.
- G4: a second family that breaks the T1.5 ordering.

## ST-A1 (ledger) — OLMo-3-7B-Think's padded-vs-alone readout check fails in bf16: 0.170 nats against the 0.05 KDG bar

**Date.** 2026-10-18 (T1.0 pod 7y7jn6pwphbzb5, A100-SXM4-80GB; `st_live_reference/outputs/t10/olmo3_think/screen_record.json` `padded_check`).
**Observation.** 16 H1_train agent prompts (313–385 tokens), decision-token letter log-probs read alone vs in one left-padded batch of 16: max |Δ| 0.170 nats. Per Amendment 4 item 4 every later cell ran unbatched (54 s per rollout at cap 4,096), and the projection (5.74 h against 1.8 h) stopped OLMo after its pilot.
**Type.** control misbehavior (a gate calibrated in fp16 against a ~0.5-nat position defect, read in bf16).
**Competing readings.** R_a (misfire): bf16 batch-size rounding alone moves the letter log-probs by ~0.1–0.2 nats at these logit scales (KDG-A21's batch-composition effect; the p2m bit-exactness lesson). R_b (defect): a pad-dependent readout error (the KDG-A20 pattern) at a third of KDG-A20's size.
**Discriminator.** Amendment 6 item 1's recalibrated check: the replicate unpadded floor (each row alone vs in a batch of 16 copies of itself) and the one-position-off negative control, on the same 16 prompts. R_a: the measurement sits within max(0.05, 2 × floor) and batching is restored; R_b: it exceeds the bar while the floor stays small. Cost: seconds after any OLMo-3-Think load (rides the next pod that loads it; the same check runs per model on every load).
**Status.** open; discriminator built (Amendment 6, commit 6e04a2b), runs on the next load.
**Thesis impact.** R_a: no instrument issue; OLMo's unbatched cost was an artifact and only ST-A2 bears on its role. R_b: a batched readout defect specific to this model or stack, entered as a methods finding beside KDG-A20.

## ST-A2 (ledger) — OLMo-3-7B-Think deliberates past any affordable cap on H1: median 1,860 tokens to the answer, 59% over 1,536, 44% over 2,048, 16% unfinished at 4,096

**Date.** 2026-10-18 (same pod; `pilot/pilot_None.jsonl`, `pilot/pilot_None_caps.json`).
**Observation.** Trace-length pilot, 8 H1_train scenarios × 4 permutations, sampler v2 (T 0.7), cap 4,096: 27 completed, 5 truncated; tokens to answer median 1,860, p95 above 4,096; share over 1,536 0.594, over 2,048 0.438. No repetition loops: the share of repeated 8-grams is at most 0.06 in every trace, and the five unfinished traces end mid-argument weighing the written rule against the third party's loss. Every completed trace ends `</think>\n\n<letter><|endoftext|>`; the parser of record reads 27 of 27.
**Type.** pre-registered trigger: a truncation-limited model (Amendment 4 item 2; the plan's 1,024 and then 1,536 caps assumed traces near GPT-OSS-20B's medium-effort lengths).
**Competing readings.** R_a (model property): OLMo-3-Think's deliberation on rule-versus-harm conflicts is long, and the brake, if present, lives inside traces longer than the envelope affords. R_b (frame): the letter-only agent frame elicits long traces from this model whatever the content.
**Discriminator.** Trace length on 16 panel scenarios (no H1 conflict) in the same frame at cap 4,096, 2 rollouts each; R_b predicts the same length distribution, R_a shorter. About 0.3 A100-h batched, more if ST-A1 resolves R_b. Does not change G1 either way: at 2,048, truncation 0.438 is far above the 5% bar.
**Status.** resolved → a result (G1, author, 2026-10-18; CLAIMS ST-09): OLMo-3-Think leaves the development role; GPT-OSS-20B takes it. The R_a/R_b discriminator stays priced for the Tier 2 stage ladder's budget.
**Thesis impact.** R_a: the deliberation brake on an open 7B reasoning recipe is measurable only with traces of 4k+ tokens, which prices OLMo-3-Think out of the Tier 1 envelope; the stage-checkpoint ladder (Tier 2) must budget for it. R_b: the frame, not the model, sets the budget, and a brevity instruction becomes a design question (never a lever: §0).

**Process ledger 2026-10-18 — the first ST VALIDATE pod (146trytky4d1z3) stopped at the vLLM import: the PyPI wheel of vLLM 0.31.0 is a CUDA 13 build.**
`import vllm` failed on `libcudart.so.13` after the fast gates passed (110 passed, 1 skipped). The
pin had followed the v0.31.0 installation docs ("compiled with CUDA 12.9 by default"); the research
pass had flagged the same docs line as stale for v0.32.0 and it was not re-checked for v0.31.0. The
wheel's own metadata (`humming-kernels[cu13]`, `nvidia-cutlass-dsl[cu13]`) said CUDA 13. Fix: the
`+cu129` GitHub release asset pinned by sha256 over torch 2.13.0+cu129, and a local dry-run resolve
for the pod platform (manylinux up to 2_35, cp311) before any pod: 181 packages, the cu12 12.9
runtime; `nvidia-cutlass-dsl-libs-cu13` still arrives through flashinfer's `nvidia-cudnn-frontend`
(every version ≥ 1.29 requires it), as it does in vLLM's own cu129 build. Cost: one setup-only pod,
no model loaded (under 0.2 A100-h). Lesson: pin a binary by its metadata and a resolve for the
target platform, never by its docs.
Follow-up the same day (second VALIDATE pod, 40yy8fg8wtnucu): the venv check `import vllm` passed
(vLLM imports lazily) and the worker's `from vllm import LLM` then failed on `libnvrtc.so.13` in
the PyPI `torchcodec` 0.17.0. A resolve audit found a second CUDA 13 path: flashinfer's unpinned
`cuda-python` had pulled `cuda-bindings` 13.4.4 against torch 2.13.0+cu129's own `<13` pin. Fix: a
constraints file (`cuda-bindings`/`cuda-python` 12.9 line, `torchcodec==0.16.0+cu129`; the
constrained resolve is consistent), and the venv check now runs the worker's own import, imports
flashinfer, torchcodec and the three model modules, and loads NVRTC (`nvrtcVersion` must report
12), before the HF stack and every gate, so a failure of this kind shows in minutes (author).
Lesson added: check the import the program makes, not the package's top-level import.
Third VALIDATE pod (vgjndx7dmqyfa8, driver 595.91.07): the venv checks passed (vLLM 0.31.0+cu129,
torch 2.13.0+cu129, CUDA 12.9 available, NVRTC 12.9) and GPT-OSS-20B loaded with CUDA graphs
captured (54.7 GiB KV cache); engine warm-up then JIT-compiled FlashInfer's top-k/top-p sampler and
failed: no `ninja` on PATH (the worker runs the venv's python without activating it). Our requests
never use that sampler (vLLM 0.31.0 takes the seeded gumbel path when no request sets top-k/top-p
or any request carries an explicit seed; ours do both); only warm-up's dummy settings reach it.
Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` in the worker's environment (hashed with its settings), the
venv's bin first on the worker's PATH, `ninja` in the venv, and the venv check asserts ninja
resolves. Each VALIDATE pod has stopped earlier in the stack than the last and cost one setup.
2026-10-19, while the relaunch ran: the HF memory-and-throughput probe (`engine_probes.hf_batch_probe`)
timed its short decode cold, so first-call costs (kernel selection, a compile at a new batch shape)
landed in `t_short` and biased the slope `(t_2n − t_n)/n` low (local tiny model: `t_n` 0.160 s
vs `t_2n` 0.056 s; the slope went negative). Its only consumer is the T1.3 steering-gate price.
Fix: an untimed warm-up decode at the same shapes, and probes record `warmed`. The relaunch's
GPT-OSS probe ran the cold version, so the re-price prices its step from the warm second timing
as an upper bound (`t_2n / 2n`, prefill ≥ 0) and labels that row a bound.
2026-10-19, relaunch: Qwen3.8-27B banked `gate_failed` after its vLLM phase (0.52 h: pilot,
C0 answers, screen, all rows saved). Every gate passed except `revision`, whose detail was `None`:
a Qwen3.5-family checkpoint loads its text tower through `AutoModelForCausalLM`
(`Qwen3_5ForCausalLM`), whose config is the composite's `text_config`, and transformers sets
`_commit_hash` on the composite only (checked locally on the pinned configs of all three models).
The weights were loaded at the pinned revision; the check could not see it. GPT-OSS (no
composite) and Muse (`AutoModelForImageTextToText`, composite config) carry the hash. Fix: the
gate falls back to the composite config at the pinned revision and records which source it read.
The revision gate runs only on real checkpoints, so no local test or dry run had exercised it on
a text-tower load; the fourth VALIDATE pod loaded only GPT-OSS. Cost: Qwen's HF phase did not run
(no readouts, no replicate floor), so ST-A3's dense-model discriminator rests on Muse alone in
this session.

## ST-A3 (ledger) — GPT-OSS-20B's HF letter readout moves 0.43–0.52 nats with batch size alone in bf16 (identical rows, no padding), so the Amendment 6 bar on this model sits at 0.87–1.05 nats

**Date.** 2026-10-19 (fourth VALIDATE pod nw2ot7pye4ad1a, A100-SXM4-80GB; `st_live_reference/outputs/t10/_validate/gpt_oss_20b/smoke_record.json`).
**Observation.** 8 of §40's H1 prompts (305–375 tokens), decision-token letter log-probs, bf16, HF: each row alone vs in a batch of 16 identical copies (no padding) differs by up to 0.434 nats (eager) and 0.523 (flex attention + compile). Padded vs alone 0.586; flex + compile vs eager 0.680. Both pass the Amendment 6 ladder (bars 0.867 and 1.047); the one-position-off controls sit at 26.0 and 26.4 nats, so the check is calibrated. The KDG bar was 0.05.
**Type.** control behaviour (an instrument floor an order of magnitude above the bar it was built around); replicates KDG-A21 (GPT-OSS's forced readout moves up to 0.49 nats with its batch-mates) on a second stack.
**Competing readings.** R_a (MoE routing): batch-size-dependent GEMM rounding flips near-tied top-4 expert choices in some layers, and a flipped expert moves the logits by tenths of a nat; the effect is specific to the MoE model. R_b (stack): the bf16 kernels or the attention implementation on this torch/transformers stack carry it, and a dense model on the same stack shows the same floor.
**Discriminator.** The same replicate floor on Qwen3.8-27B and Muse Glimmer 30B (dense) on the relaunch, which computes it in their phase H at no extra cost: floors near 0.05 → R_a; floors near 0.4 → R_b.
**Status.** open; discriminator rides the relaunch (zero added cost).
**Relaunch read (2026-10-19, pod esmxvsxj8wzkaq; `outputs/t10/*/screen_record.json`).** Muse
Glimmer 30B (dense: 52 layers of sliding and full attention, no experts; bf16, eager): replicate
floor 0.375 nats at batch 8 on 16 H1 prompts (294–365 tokens); negative control 16.8 nats. That
is the "near 0.4" branch: the pre-stated discriminator favours R_b (the bf16 stack carries the
batch-size jitter; a dense model shows it too). Qwen3.8-27B did not reach its HF phase (revision
gate, process ledger), so R_b rests on one dense model. Reading against the floor is at the end of
the generation prompt, where the letters sit in the tail of the next-token distribution; a floor
at the dose-0 or post-reasoning position, where letters hold the mass, may be smaller. GPT-OSS on
the relaunch: eager floor 2.79 nats over 16 rows (0.43 over 8 rows at VALIDATE, same position
type); the record keeps only the maximum, so whether one row drives it is unknown (per-row arrays
are saved from now on, Amendment 7 implementation item 9). **Status:** leaning R_b on one dense
model; the Qwen rerun gives the second floor and per-row arrays at the same position.
**Rider pod read (2026-10-19, `outputs/t10b`, per-row arrays).** (1) The floor's position is a
tail position: at the end of the generation prompt GPT-OSS's letter log-probs sit at −42 to −49
nats (Qwen −6 to −18), and GPT-OSS's 2.79 is one row (others 0.14–0.58, one 1.33). A floor
measured where the letters carry no mass is not the floor of a readout of record. (2) At a
position of record (the dose-0 letter position, letters at −0.01 to −11.7), the batched-vs-alone
difference (Amendment 8's post-hoc column, batch 8–16 vs alone) is 0.249 on Muse and 0.249 on Qwen,
and HF-forward vs vLLM is 0.235 (Muse) and 0.250 (Qwen). Four maxima at 0.25, and Muse's per-row
tail floors fall on multiples of 1/16 (0.125, 0.1875, 0.25, 0.375). The simplest model: bf16
rounding of the logits (spacing 0.125 at |logit| 16–32, 0.25 at 32–64), so any change of
accumulation order moves a large logit by one or two steps; the jitter is in the output
precision, not in the computation, and it is the same in a dense model (R_b, generalized:
precision, not MoE). Qwen's tail-position values are not all on that grid (0.080, 0.157, 0.274),
so the model is not complete there. **Discriminator (new, cheap):** the same checks with the
final norm and LM head in fp32 (minutes on any loaded model): a floor that drops to ~0.01 at the
position of record confirms the precision reading and makes batched readouts admissible at the
0.05 bar; a floor that stays ~0.25 sends the jitter back into the network. **Status:** R_a (MoE
routing) no longer needed to explain the floor; precision reading leading, discriminator priced.
**Rider pod 2 read (2026-10-19, A100 80GB PCIe, `outputs/t10c`; fp32-head discriminator).** At
the dose-0 position, replicate floor with the bf16 head → fp32 head: GPT-OSS-20B 0.299 → 0.258
(padded vs alone 0.458 → 0.479), Muse 0.396 → 0.278 (0.396 → 0.278). The fp32 head removes about
a third of Muse's floor and a seventh of GPT-OSS's; neither approaches 0.01. **The precision-of-
the-head reading is refuted as the main source:** the batch-size jitter at a position of record
is in the network's own bf16 computation, on a dense model and an MoE alike (R_b as originally
stated: the stack, not MoE routing). Readouts of record are single-row with the bf16 head on both
models of record (Amendment 9 item 2). GPT-OSS's per-row arrays were lost with its record (two
fold-path bugs, fixed in `0d4e96f`); its rerun recovers them. **Status:** resolved for the
readout decision; the mechanism inside the network is not localized (not needed for any gate).
**Scope of the relaunch's rung verdicts:** every floor and bar of the padded check and the rung
was read at the tail position, so the kept fast paths (GPT-OSS flex + compile under a 5.58 bar)
are validated there only; a re-read at the dose-0 position is part of the discriminator above.
**Thesis impact.** Either reading: on GPT-OSS a batched readout carries about half a nat of batch-dependent jitter, so readouts of record should be read one row at a time with one fixed implementation across every arm (author decision pending). R_a also scopes KDG-A21 as an MoE property rather than a harness defect.

## ST-A4 (ledger) — Muse Glimmer 30B's dose-0 forward-vs-generation check fails at 0.25 nats against the 0.05 bar, below its own 0.375-nat batch floor

**Date.** 2026-10-19 (T1.0 relaunch, pod esmxvsxj8wzkaq; `st_live_reference/outputs/t10/muse_glimmer_30b/screen_record.json`, `dose0.tried`).
**Observation.** The empty-reasoning dose-0 candidate passes letter mass (0.983, bar 0.95), C0 (agreement 0.719, Wilson [0.599, 0.814], bar 0.633 = 0.9 κ*), and the known-gap band (0.591 [0.522, 0.655]), and fails only forward-vs-generation: 0.249 nats against 0.05. The direct-answer candidate fails letter mass (0.945) and forward-vs-generation (0.246). `fwd_vs_gen` reads the teacher-forced side batched (batch 8) and the generation side one row at a time, so the comparison contains the batch-size jitter whose floor on this model is 0.375 nats (ST-A3). With no dose-0 regime of record, the screen's brake for Muse is descriptive: B = −0.314 [−0.393, −0.239] against the failed candidate's readout, n = 60.
**Type.** control misbehaviour (a check whose bar predates Amendment 6's recalibration and whose two sides differ in batch size).
**Competing readings.** R_a (instrument): the 0.25 nats is batch jitter; forward and generation read alone agree within 0.05. R_b (real): the empty-reasoning prefill takes a different path through generation than through the teacher-forced forward (e.g. cache or sliding-window handling at the channel boundary), so forced dose-0 readouts do not equal what the model would generate.
**Discriminator.** Forward alone vs generation alone on the same 16 rows (like with like, bar 0.05): one Muse load, about 0.25 A100-h with download, and it can share the Qwen rerun pod. A zero-GPU partial: the Amendment 6 ladder applied to this check (bar 2 × 0.375 = 0.75) passes on saved data, but that is a post-hoc fork and needs its amendment first.
**Status.** resolved (2026-10-19, rider pod, `outputs/t10b/muse_glimmer_30b`): R_a. Forward
alone vs HF generation alone on the same 16 rows: 0.000 nats on every row (Amendment 8 gate,
bar 0.05). The relaunch's 0.25 was the batch jitter its batched forward carried (the post-hoc
column reproduces it: 0.249 against max(0.05, 2 × 0.375)). Muse's empty-reasoning candidate passes
every dose-0 check (letter mass 0.983, C0 0.719 against 0.633, known-gap band) and is of record;
its brake leaves descriptive status (Amendment 8 item 2). resolution_type: experiment.
**Thesis impact.** R_a: Muse has a valid dose-0 regime and a measured brake of −0.31 on 60 screened scenarios (not comparable in size with GPT-OSS's −0.224 until both are read on the same scenarios and readout); it competes for the second-family role on the same criteria as Qwen. R_b: Muse's forced dose-0 readouts are not its generation behaviour; it can take the second-family role only on reasoned-arm endpoints, or Qwen takes it.

**Process ledger 2026-10-19 — Muse's compiled readout path raised `NameError: name 'torch' is not defined`.** The rung's first attempt (SDPA + torch.compile) raised inside compilation; the rung recorded it and kept uncompiled SDPA (measured 0.31 nats against bar 0.75; under item 9's reference-floor rule the bar is the same 0.75, since the eager floor equals the candidate's, 0.375, and it still passes). Uncompiled SDPA read 16 rows alone in 1.96 s (0.12 s per row), the single-row rate the re-price uses. The traceback is not in the log, so the cause is not established. No numeric effect.

## ST-A5 (ledger) — Qwen3.8-27B deliberates toward protection: whole-set brake +0.083 [0.020, 0.146], 8 brake-engaged scenarios against a no-brake null q95 of 18

**Date.** 2026-10-19 (rider pod, `outputs/t10b/qwen38_27b/screen_record.json`; phase H on the relaunch's banked 60 × 4, effort medium).
**Observation.** Dose-0 regime of record (`enable_thinking=False`: letter mass 1.000, C0 0.844 against 0.740, forward vs generation 0.000). P_reasoned 0.696 [0.604, 0.783], P_dose0 0.612 [0.558, 0.668], B = +0.083 [0.020, 0.146] (MDE 0.090). Engaged (P_dose0 ≥ 0.5 and P_none ≤ 0.5): 8 of 60, below the no-brake null (mean 13.8, q95 18). On the same screen instrument Muse's B is −0.314 [−0.393, −0.239] and GPT-OSS's (§40, n = 258) −0.224.
**Type.** sign flip (family exception).
**Competing readings.** R_a (model): Qwen3.8's reasoning raises its protective share on H1; its post-training installs no deliberation brake, so it is a natural negative control for the ST mechanism. R_b (regime): `enable_thinking=False` is not Qwen's dose 0 in the sense the other models' prefills are: the template switch changes the conversation mode (a distinct trained no-think behaviour) rather than cutting deliberation at zero length, so B compares two modes, not two doses.
**Discriminator.** A forced empty think block under thinking-on (`</think>` prefilled at zero trace length; a new dose-0 candidate, not one the format registers today) on the same 60 scenarios, through the same dose-0 checks: B near +0.08 → R_a; B moving toward the panel's negative values → R_b. About 0.3 A100-h with one Qwen load, plus the candidate's local format tests.
**Status.** open; not on the critical path (Qwen does not meet criterion (ii) under either reading).
**Calibration read (2026-10-19, zero GPU).** At the effort of record (medium) the
`enable_thinking=False` render equals the thinking-on render plus `\n</think>\n\n` byte for byte,
and encodes to the same ids: Qwen's dose-0 of record already is an empty think block, so R_b's
"mode switch" has no prompt-level counterpart, and the forced-empty-think discriminator would
re-read the same prompt (a separately encoded prefill differs only by splitting the `\n\n` token).
What remains of R_b is that the empty think block is a trained no-think behaviour, which is true
of every model's dose 0 (OLMo's `close_think`, Muse's empty reasoning channel) and is what the
brake's definition compares. resolution_type: calibration (pending the author's call on dropping
the GPU discriminator).
**Discriminator run (rider pod 2c, 2026-10-19, `outputs/t10d/qwen38_27b`; author ran it).** Candidate
`think_on_empty` (the registered text with the merged `\n\n` token 271 split into 198, 198):
every dose-0 check passes (letter mass 0.996; C0 0.844 vs 0.729; known-gap band 0.659; forward vs
generation alone 0.000). On the same 60 reasoned rows: P_dose0 0.569 [0.509, 0.631] (0.612 with
the registered candidate), B = +0.126 [0.067, 0.184] (MDE 0.084), 5 engaged against the null q95 16.
**Status: resolved, R_a.** The sign and the CI's exclusion of 0 hold under the only construction
that differs from the registered one, so Qwen3.8 deliberates toward protection on H1 on this
instrument; R_b (a mode switch) had no prompt-level form. Side finding: splitting one newline token
moved Qwen's dose-0 protective share by −0.043, about half its MDE; dose-0 shares on this model
carry that much tokenization sensitivity. resolution_type: experiment.
**Thesis impact.** R_a: the brake is not universal across reasoning families; ST's second-family replication needs a braking family (Muse), and Qwen becomes the panel's documented non-braker, which sharpens what the intervention claims. R_b: the dose-0 construction is a per-family instrument choice that can flip a brake's sign, and the KDG dose-0 arm needs a cross-family construction audit.
**Note (criterion (i)'s sign).** Amendment 5's criterion (i) reads "CI excluding 0" and is coded that way; Qwen's positive B satisfies it literally. Its verdict is the same under a directional reading because (ii) fails; the sign is an author item at G1.

## ST-A6 (ledger) — GPT-OSS-20B's HF generation costs ~20 ms per sequence-step at every batch: the steering gate (T1.3) prices at 9.1 A100-h, against a plan of 1.5

**Date.** 2026-10-19 (rider (b), `outputs/t10b/gpt_oss_20b/screen_record.json`, eager attention, uncompiled, bf16).
**Observation.** Decode step 0.680 s at batch 33, 1.348 at 66, 2.686 at 132 (peaks 49 / 57 / 72 GB): linear in batch, ~20.4 ms per sequence-step, so batching buys nothing. Uncompiled flex attention on the relaunch: 2.683 s at 132 (the same). T1.3's 960 rollouts at the cap: 9.1 A100-h. `kernels` (the hub MoE kernels) was absent on both pods.
**Type.** control behaviour (a cost floor the plan did not model).
**Competing readings.** R_a: HF's GPT-OSS MoE runs a per-token expert loop in bf16 (dequantized MXFP4), so step time scales with tokens, not batches; the hub MoE kernels or grouped GEMM would remove most of it. R_b: the attention path dominates (sinks, eager or flex without compile), and MoE kernels would not help.
**Discriminator.** One probe with `kernels` installed and `use_kernels=True` (the transformers hub MoE path) at batch 33 and 132: ~0.1 A100-h. A second lever needs no HF generation at all: steering inside vLLM (hooks registered in the engine process with eager execution).
**Status.** resolved (2026-10-19, rider pod 2b log, A100-SXM4): neither reading as stated. The
default experts path on torch 2.6 is `grouped_mm`, which on SM80 under torch < 2.9 runs
transformers' `grouped_mm_fallback` op; that is the ~20 ms per sequence-step cost. With
`experts_implementation="eager"` the step is 0.082 / 0.191 / 0.282 / 0.354 s at batches
1 / 8 / 33 / 132 (7.6× faster at 132), so T1.3 prices at ~1.2 A100-h on the HF stack of record.
`batched_mm` does not fit at any batch; MegaBlocks has no torch-2.6 build. resolution_type:
experiment. Rider pod 2c (torch 2.13, bf16-dequantized, A100-SXM4 community): `grouped_mm` runs
0.038 / 0.070 / 0.249 / 0.961 s per step at batches 1 / 8 / 33 / 132: faster than eager at small
batches, slower at 132 (eager 0.354 on rider pod 2b), so T1.3's batch-132 price stays on the
eager path (~1.2 A100-h); the two timings come from different pods on the same GPU model. Open consequence: the HF readouts of record so far (screen, riders) ran on the default
(fallback) path; pinning `eager` for GPT-OSS changes its MoE arithmetic, so its dose-0 floor and
stack-parity reads are re-taken under the pin before any readout of record.
**Thesis impact.** None on the science; on the plan, T1.3 is now the largest Tier-1 line (development total 15.9 A100-h with GPT-OSS; envelope still holds at ~25 of 35 with Muse).

**Process ledger 2026-10-19 — torch.compile of the HF forward fails on both dense models on the pod stack (torch 2.6 + transformers 5.17).** Muse: `NameError: name 'torch' is not defined`; Qwen: `InternalTorchDynamoError: AttributeError: module 'torch.compiler' has no attribute 'is_exp…'` (an API newer than torch 2.6). The rung recorded both and kept uncompiled SDPA. GPT-OSS compiled. No numeric effect; the HF readouts of the dense models run uncompiled until the HF stack moves past torch 2.6.

**Process ledger 2026-10-19 — rider pod 2's GPT-OSS record was lost to two bugs in a path no local test reached.** (1) After the fold check freed the first HF model, its vLLM half needed the tokenizer through `sc.model` (None). (2) The folded checkpoint's links into the HF cache entered the bank's manifest walk, which raised, so the record (with the dose-0-position reads computed minutes earlier) was never written, and the stack reads and three MoE probes that run only after a clean exit were skipped. Fixes: a tokenizer stand-in after each free, the checkpoint removed after the check (also on failure), the bank never walks links; regression tests for both. The headline dose-0-position numbers survive in the session log. The MegaBlocks probe failed to load: transformers 5.17 accepts `kernels` 0.16.x only and an unpinned install pulled a newer one (now pinned). Lesson: a step that can fail must not sit between expensive reads and the bank; the dose-0 reads should have banked before the fold check started. Cost of the pod: 0.65 A100-h (timestamps + provisioning); Muse's half (dose-0-position reads, stack parity inside) is complete.

**Process ledger 2026-10-19 — rider pod 2b's GPT-OSS results were lost to a download path mismatch Claude asserted without checking.** The relaunch command set `RESULTS_SUBPATH=outputs/t10d` while the rider2 mode writes `outputs/t10c`; Claude said the download would map one onto the other, which the launcher does not do. rsync found no source, reported an empty file list as success, and the launcher terminated the pod (gyyo2g10b95uqs, ~0.75 A100-h). What survives is the streamed session log: the dose-0-position reads (identical to the first rider-2 pod), the eager MoE timings, `batched_mm`'s out-of-memory at every batch, MegaBlocks' missing build, and two load failures in the vLLM venv (`accelerate` absent for transformers' MXFP4 path, so GPT-OSS's torch-2.13 stack read and `grouped_mm` probe never ran). The fold check's vLLM half failed for GPU memory (39 of 79 GiB free; now its own process). Fixes: `rp_download` keeps the pod and lists the remote outputs when the results directory is missing (shared launcher); `STEPS=<mode>/<subdir>` names the remote results directory; the bank no longer lists the interim record twice (the manifest's sha mismatch).

## ST-A7 (ledger) — The fold validation's vLLM-vs-HF steering check fails its pre-registered bar (max difference of differences 0.638 vs 0.25) while the two engines' steering effects correlate at 0.87

**Date.** 2026-10-19 (rider pod 2c, `outputs/t10d/gpt_oss_20b/fold`; Amendment 9 item 4 (ii)).
**Observation.** v (‖v‖ 325, 0.25 × the median residual norm at block 11) folded into every expert's `down_proj_bias`: (i) HF hook vs HF fold 0.262 nats, inside the 0.584 bar, hook vs base 1.07 (passes); (ii) the steering effect on vLLM (folded − base) against HF's: max |difference of differences| 0.638, mean 0.187, bar 0.25 (fails); (iii) parse rate 0.953 base, 0.922 folded (64 each). Zero-GPU follow-up on the saved rows (80 letter values): Pearson 0.874 between the engines' per-letter effects, sign agreement 0.86, vLLM's effect ≈ 0.82 × HF's; the unsteered HF-vs-vLLM gap on the same rows is max 0.783, mean 0.164.
**Type.** control misbehaviour (a bar calibrated on other models).
**Competing readings.** R_a (engine band): the fold is applied correctly in vLLM and the residual is the two engines' ordinary disagreement (Marlin W4A16 vs bf16-dequantized arithmetic), which on GPT-OSS's dose-0 rows is ~0.78 nats unsteered, three times the dense models' 0.25 the bar was taken from. R_b (fold mislanded): the bias reaches vLLM's experts in another layout or weighting. R_b predicts an uncorrelated or sign-scrambled effect; 0.87 and 86% sign agreement rule out a gross mislanding, and a scale of 0.82 with a correlated residual fits R_a.
**Discriminator.** Done (zero GPU, above): R_a. A sharper one if needed: the same check with a smaller ‖v‖ (residuals that shrink in proportion → engine nonlinearity; that stay at the baseline band → engine noise).
**Status.** resolved as descriptive (author, 2026-10-19; no amendment): (i) passes; the fold lands in vLLM (r 0.87, 86% sign agreement, vLLM's effect 0.82× HF's); (ii) fails only against a bar taken from the dense models, which does not apply to GPT-OSS's rows (a 0.78-nat unsteered engine gap). Folds stay out of T1.3: every gate arm runs on HF with eager experts (~1.2 A100-h, ST-A6). The fold is recorded as validated for later vLLM-speed uses outside the gate. resolution_type: calibration.
**Thesis impact.** None on the science; T1.3 runs on one engine (HF, eager experts).

**Process ledger 2026-10-19 — dose-0 prefill ids are appended separately, and the encoding is part of the instrument.** ST-A5's discriminator split one merged `\n\n` token (271) into two `\n` tokens (198, 198) in front of Qwen3.8's `</think>`, the same text, and moved its dose-0 protective share by −0.043 (0.612 → 0.569), about half the screen's MDE. That is the reason every dose-0 candidate's prefill is appended as separately encoded ids after the rendered prompt (never re-encoded jointly with it, where a tokenizer can merge across the boundary, as OLMo's `>` + `</` did) and why the candidate's exact ids are recorded beside every dose-0 readout: an encoding change is a change of instrument, not a rounding.


## ST-A8 (ledger) — FL's shared refusal classifier reads GPT-OSS's curly-apostrophe refusals as compliance: the T1.1 selection pass scored 720 of 720 answers as complied

**Date.** 2026-10-19 (T1.1 pod, `outputs/t11/_rr/gpt_oss_20b/rr/rr_select_readout.jsonl`; Amendment 10 item 3).
**Observation.** Kept 0 of 60, positive control 0.0, every readout `refusal: False` (695 stop, 25 length, no missing answer); the answers read "I’m sorry, but I can’t help with that." `greenblatt._classify_response` on that string returns complied; on the straight-apostrophe string, refused.
**Type.** control misbehaviour; bug report that is a finding (the same defect Paper 7 found and fixed locally, re-hit by a new consumer of the shared classifier).
**Competing readings.** R_a (readout defect): the classifier's straight-quote patterns miss U+2019 and the model refuses. R_b (the model complies): the answers comply in substance. The saved answers settle it: they are refusals by inspection, and the operator control (an operator ordering the request) reading 0.0 is what R_a predicts and R_b does not.
**Discriminator.** Done (zero GPU): R_a (the saved answers).
**Status.** resolved (2026-10-19; author): the fold moved into the shared classifier as a versioned, dated change (`greenblatt.CLASSIFIER_VERSION` v2, `normalize_typography`); the two local copies are retired (Paper 7's `_normalize` delegates to it; ST's readout drops its own fold, `rr-readout-v3`); the classifier's tests carry curly-apostrophe cases. The selection is re-read from the saved answers under v3 after the pre-registration is pushed. resolution_type: calibration.
**Thesis impact.** None on any committed verdict; Amendment 10's gate on GPT-OSS is read under v2. Process lesson: the ST local test asserted the classifier on the string a straight-quote model writes, not the string the model of record writes; local readout tests now use the model's own output form.

**Process ledger 2026-10-19 — the shared classifier's typographic fold is label-preserving on every saved text; three cells are checked by proxy only.** Not a correction to any published number. Every earlier GPT-OSS refusal readout went through Paper 7's fold (Paper 7's gate and ablation, D3 `gptoss_tier1`, W4's GPT-OSS extractor). The unfolded callers ran on OLMo-2, OLMo-3-Instruct, Llama-3.1 and Qwen2.5; the 1,920 saved Paper 5/6 battery texts on those models read the same under classifier v2 as their stored labels (0 of 1,920 differ), and none of their 201 saved refusals opens with a curly apostrophe, though OLMo-3-Instruct and Qwen write curly ones elsewhere (32 to 112 of 160 texts per file). D2 B1/B5, D3 C1 and W4 14.5 saved rates, not texts, so for them the check rests on the same models' battery texts rather than their own rows. KDG reads letters and never calls the classifier.

## ST-A9 (ledger) — The development model carries no dose-0 pressure gap on the panel, so T1.5's panel ΔE co-primary starts at its floor on GPT-OSS

**Date.** 2026-10-19 (found while pricing the panel levers; cross-document conjunction).
**Observation.** KDG_RESULTS §27: GPT-OSS-20B at dose 0, E −0.001 [−0.011, 0.008], n 586, MDE 0.013, known-gap band 0.501 [0.469, 0.534] on the same readout. ST_PLAN §6 makes panel ΔE a co-primary, and `generalizes` requires ΔE < 0 with its CI excluding 0; G1 made GPT-OSS the development model through truncation (ST-09), and no ST document connects the two. The panel reads are now the largest Tier 1 line on this model (T1.5 readouts 10.78 A100-h of the 32.34 envelope). Muse's dose-0 panel has never been read (its first screen had no dose-0 regime; the rider pods skipped the panel).
**Type.** cross-document conjunction.
**Competing readings.** R_a (floor): with E_none ≈ 0, an adapter can lower E only by making pressure push GPT-OSS toward protection, which is not the gap closing that `generalizes` names; on this model the panel reads as a one-sided check (does an adapter install a gap?) and `generalizes` cannot be reached in its intended sense. R_b (instrument): KDG read E on its own stack; on the ST instrument (torch 2.13, eager experts, single-row) E_none may sit away from 0, and the panel is a live co-primary after all.
**Discriminator.** The `none` panel on the ST instrument, which T1.5 already prices (2.59 A100-h at five cells): run first, before T1.4 trains, it separates R_a (E_none's CI inside ±0.013) from R_b. Muse's dose-0 panel (≈ 0.79 A100-h on its first pod) decides whether a model of record carries the gap at all.
**Status.** promoted (author, 2026-10-19; ST_PLAN Amendment 11, both branches pre-registered): GPT-OSS's `none` panel on the T1.3 pod, Muse's dose-0 panel on the first Muse pod; §2 gains the line that the E co-primary requires a dose-0 carrier. Update 2026-10-20: the T1.3b read printed call A (E_none −0.0011 [−0.0105, +0.0083], g_band 0.487), favoring R_a; its rows were lost at the download, so R_a is provisional until the re-read on the next GPT-OSS pod (author: option 1). **Resolved 2026-10-21 (T1.3c, the read of record): R_a.** g_band 0.487 [0.462, 0.513], in band; E_none −0.0011 [−0.0105, +0.0083], n 586: not a carrier; recomputed from the saved cells. resolution_type: experiment.
**Thesis impact.** R_a: the pitch's panel sentence (sensitivity moved, not baseline) cannot come from the development model; on GPT-OSS the panel is a do-no-harm read, and the positive panel claim needs a model that carries E (Muse if it does, else none in Tier 1). R_b: no change.

## ST-A10 (ledger) — Within scenario, GPT-OSS's protect rollouts sit lower on the anchor axis than its rule rollouts at clause ends, at every banked layer

**Date.** 2026-10-19 (T1.2 constancy read, `st_live_reference/data/t12_constancy_v1.json`; Amendment 12 item 3).
**Observation.** At L* (block 12), the within-scenario protect-minus-rule difference of the anchor-axis margin ⟨h, a⟩/‖h‖ over 40 mixed-outcome H1_train scenarios: `clause_end` −0.0024 [−0.0041, −0.0007] (the pick: the only class whose CI excludes 0), `onset` −0.0028 [−0.0062, +0.0004]. The sign is negative at all five banked layers and on both axes (`V_care` and `V_care_alt`); the `clause_end` CI excludes 0 on 9 of the 10 layer × axis cells. Scale: the per-rollout `clause_end` margin is 0.043 ± 0.0105 across rollouts, within-scenario SD 0.0064, so the difference is about 0.37 within-scenario SD. Between scenarios the END_OF_PROMPT margin does not track P_none (Spearman −0.03 [−0.26, 0.07], n 132). Position validity: PR at L* `clause_end` 36.4; `onset` 29.8, under the PR 30 flag.
**Type.** sign-flip against the anchoring hypothesis (a pre-registered branch).
**Competing readings.** R_a (content): within a scenario, the rule-following deliberation carries more morally loaded content at its clause ends than the protective one (obligation, fairness, the cost of breaking the rule), and the anchor axis (positive pole = morally loaded content, not the morally right act) tracks that salience rather than care for the person helped. R_b (position): the margin drifts along the trace, protect and rule traces differ in length or structure, and a class mean over different position distributions produces the difference. R_c (selection): the 40 mixed-outcome scenarios are those where the model is torn, and their rule rollouts are the ones that deliberate morally before deferring to the rule.
**Discriminator.** R_a vs R_b, zero GPU on the bank: the same contrast at matched trace positions (clause ends binned by relative position, or the margin residualized on position within rollout). R_c: the same contrast on the H1_held mixed-outcome scenarios (descriptive, never training data). The causal cell is T1.3's gate: `care` adds +a at `clause_end`, and R_a predicts a ΔP at or below 0.
**Status.** promoted (pre-registered, Amendment 12 item 3): `L-anchor`'s expected branch flips to the null before training; the arm still runs. Discriminators priced at zero GPU, not yet run (author's call).
**Update 2026-10-21 (Amendment 15's foundation read).** The sign is specific to the anchor axis's construct. On GPT-OSS's MFT foundation axes (W4 14.4, block 12; near-orthogonal to the anchor axis, cos 0.04–0.13) protect sits above rule at clause ends on care/harm (+0.0068 [+0.0038, +0.0098]), fairness and sanctity, and on `V_harm` itself (+0.0109 [+0.0081, +0.0138]); `care_harm` with `V_harm` removed keeps +0.0039 [+0.0009, +0.0066]. At `onset` every foundation point is negative. `V_care`, and so the anchor axis, removed the harm content by construction (§4; author, 2026-10-21). This sharpens R_a: what runs rule-ward is the anchor axis's moral-versus-neutral salience, while care, fairness, sanctity and harm content run protect-ward; it does not separate R_a from R_b (position), which the position-matched contrast still would.
**Thesis impact.** R_a: the anchor axis as typed is a moral-salience axis whose in-trace load is higher when the model talks itself into the rule, which narrows what anchoring "care" at this site can mean and makes `L-anchor` a test of whether raising salience moves P at all. R_b: no content finding; the pick stands on a position artifact and the gate decides.

**Process ledger 2026-10-20 — an untested pod intervention lost Amendment 13's rerun and wrote the HF token into a downloaded log.** To run `care_rerun` on the T1.3 pod after its driver, Claude paused the session shell (SIGSTOP), waited for the driver, and launched the rerun from a hand-written helper that copied the driver's environment with `xargs -0 -a <environ> env -i python ...`. `xargs` appends its input after the command's own arguments, so the environment arrived as script arguments; argparse rejected them and printed them, HF token included, into `outputs/t13/t13_care_rerun.log`, which the launcher downloaded. The helper resumed the shell and the pod ended without the rerun (no GPU spent on it). Fixes: the local copy is redacted and the outputs tree is git-ignored (nothing committed); the author is asked to rotate the token; the rerun resumes on the next GPT-OSS pod through the driver's own path (`care_rerun` is now in the default phase order, finished phases are skipped). Rule: no pod-side helper runs that has not been executed locally first (the [[test gates before GPU]] rule applies to interventions too), and no command line or log ever carries an environment dump.

**Process ledger 2026-10-20 — a wrapped paste dropped the T1.3 record from the resume pod's sync.** The launch line's `SYNC_OUTPUTS` value was long enough to wrap when copied, so it reached the shell as `…,outputs\n  /t13/gpt_oss_20b/`; `read` stops at the newline, the directory entry never became a filter, and the pod started without the T1.3 record or its gate rows (a fresh driver would have rerun the sweep and the whole gate). Caught on the pod before the driver started; the record and rows were copied from the downloaded set (hash-checked). Fix: `rp_check_sync_outputs` runs before any pod exists and refuses whitespace anywhere in `SYNC_OUTPUTS` or an entry that names nothing locally (tested on the wrapped string, a good one, a missing path and `none`); launch commands keep `SYNC_OUTPUTS` short.

## ST-A11 (ledger) — At the target-selection gate both anchoring targets move GPT-OSS toward the rule, not toward protection

**Date.** 2026-10-20 (T1.3 gate, G2 read; ST_PLAN T1.3 outcome; §7's sign-flip trigger).
**Observation.** On the gate's 20 brake-engaged H1_train scenarios × 8 at `clause_end`, block 12, against the in-gate `none` re-run (P_none 0.200): `care` (anchor axis) at α 267.3 ΔP −0.071 [−0.161, +0.007]; at α 534.6 (descriptive, off-coherence by Amendment 13) −0.128 [−0.211, −0.060]; the paired dose step −0.057 [−0.113, −0.007]. `self` at β 0.25 −0.091 [−0.177, −0.014]; at the dose-0 decision token the same mix moves P_dose0 by −0.068 [−0.099, −0.042] (β 0.25) and −0.125 (β 0.5). The norm-matched `fill` at 267.3 moves −0.028 [−0.091, +0.040]; target-minus-`fill` differences include 0 (−0.043 and −0.063).
**Type.** sign flip (the gate expected ΔP ≥ +0.10).
**Competing readings.** R_a (salience drives the brake): the anchor axis, typed as moral-content salience, and the model's own judgment-frame state both carry the deliberation that ends in the rule, so adding either amplifies the brake; consistent with ST-A10's in-trace ordering and with the brake itself (reasoning lowers P). R_b (generic perturbation): any in-trace push at clause ends lowers P; supported by `fill`'s small negative point and by the non-separation of targets from `fill` at 267.3; against it, the dose step along the axis excludes 0. R_c (judgment-frame mismatch): the `self` source is an END_OF_PROMPT state (norm ~1,875) mixed into trace positions (~1,337); β 0.5 broke generation, and β 0.25's effect may carry the norm change rather than the content.
**Discriminator.** R_a vs R_b: `care` with the axis sign reversed (−a) at 267.3 on the same 20 × 8 (~0.35 A100-h): R_a predicts ΔP > 0, R_b predicts ΔP ≤ 0. R_c: `self` with the source rescaled to the class's median norm (~0.35 A100-h). Zero GPU, done: the five `fill` draws at 267.3 move P by −0.025, −0.000, −0.100, −0.018, +0.050 (19–40 rollouts each), a span that contains both targets' ΔP (−0.071, −0.091): at the coherent α a single random direction moves P as far as either target, which favors R_b there; the dose step along the axis is the only R_a evidence.
**Status.** open, unresolved by its pre-registered read (2026-10-20; Amendment 14 item 2): the reversed-axis arm at 267.3 is coherent (parse 0.950 vs none 0.950, length 1.14, NLL +0.004) and moves P by +0.013 [−0.066, +0.097] against `none`; the read of record ΔP(−a) − ΔP(+a) = +0.084 [−0.007, +0.177] (n 20, MDE about 0.13) is positive, in R_a's direction, with its CI including 0, so by the rule it reads "R_b or unresolved". Exploratory, not a verdict: the two direction-specific cells point the same way (the dose step along +a −0.057 [−0.113, −0.007]; the reversal +0.084 [−0.007, +0.177]), which is the near-miss pattern a larger gate n would settle (estimator-traps #5).
**Thesis impact.** R_a: the moral-salience reference this line set out to hold live is part of what produces the brake on GPT-OSS; holding it more strongly would deepen the brake, which reframes `L-anchor` and `L-self` as tests of whether training can change what that salience feeds rather than how strongly it is present. R_b: no content finding; the gate's targets are not distinguishable from noise steering at this n.

**Process ledger 2026-10-20 — editing the launcher while it ran lost the resume pod's download (the panel's 23,440 rows).** Claude committed the `rp_check_sync_outputs` guard into `run_session.sh` while the author's launcher was executing that same file. Bash reads a running script from its file as it goes; the in-place edit (same inode) shifted the bytes under it, and at the download step the launcher read `syntax error near unexpected token ')'` (line 125), its cleanup trap ran, and the pod was terminated without the rsync-back. Lost: the `none` panel's five cells and the carrier call's artifact (the session log printed the call: A, E_none −0.0011 [−0.0105, +0.0083], g_band 0.487, not deferred). Kept: the full gate of record (the `care` and `fill` reruns and the post-rerun record), copied down mid-session for the G2 read. Rules: never edit a script a running session executes (launcher, session library, remote runner); if a change cannot wait, write a new file and `mv` it over the old one (a new inode; the running bash keeps the old one). Every pod-side copy of results the author has not downloaded is pulled before the session ends.

**Process ledger 2026-10-21 — Amendment 16's two τ were computed before the amendment reached origin.** The author's message said the commits were pushed through df6af16; Claude ran `build_t14_inputs.py` without checking, and `git ls-remote` then showed origin at 8b07060 (the push had not landed). The rule the values follow (Amendment 14's τ rule applied to the two Amendment 16 axes) was committed locally in d975b6f before the computation, and nothing else was computed. The values (care_harm 0.1202, V_harm 0.2413; the anchor axis reproduces 0.0589) are disclosed here. Rule: check `git ls-remote origin refs/heads/main` against the amendment's hash before any computation it licenses, whatever the message says.

**Process ledger 2026-10-21 — ST Phase 2 pod A ran out of memory at the second batch of Cell 1's `none`.** Pod sx9ag0uq14lw64 (`STEPS=st2a`, 21:37-22:08 UTC, ~0.62 A100-h with provisioning) generated the first 160 of `c1_none`'s 720 rollouts, then failed: `torch.OutOfMemoryError` allocating 22.65 GiB in the prefill's `lm_head`. `steering_gate.generate` prefilled each batch with logits at every prompt position (160 × ~376 × 201,088 in bf16 = 24.3 GB) and kept the previous batch's KV cache alive into the next prefill. T1.3 and T1.4 never ran more than one batch per call, so neither the pods nor the tiny-model dry run (a 32-wide model, nowhere near the memory) could show it. Nothing was lost beyond the pod time (no arm finished; the 160 raw rows are not of record). Fix: the prefill keeps the last position's logits only (`logits_to_keep=1`) and each batch's cache is released before the next; a test asserts both and that a multi-batch call equals separate calls token for token. Rule: a dry run cannot test memory; any generation call whose batch count or shape differs from a pod-proven one gets its peak memory estimated before launch (KV + prefill logits + weights at the pod's batch and cap). The T1.5 pod carries the old code: its only multi-batch call is the HF fallback (`hfgen_<arm>`, 304 rollouts at batch 160), which runs only if an adapter fails parity. With the author's approval it was patched mid-session (2026-10-21, during `hf_beh`) by the CLAUDE.md rule for a change that cannot wait: the fix's two hunks applied to the pod's own copy, run locally first (multi-batch = separate calls, `logits_to_keep=1` at each prefill), uploaded as a new file, sha256-checked (`9fa2b1f7…`) and renamed over the old one; phases started after `hf_beh` load it, the running process kept the old file.

**Process ledger 2026-10-23 — the `care_harm` ladder's first pod stopped at its own local gate.** Pod amgea03zuqwq4g (`STEPS=st2ml`, 06:15-06:28 UTC, ~0.32 A100-h with provisioning) failed `test_branches_on_stand_in_rows` and exited before loading a model. The test skipped only when Cell 4's folder was missing; on the pod the folder exists (the sync carries six of Cell 4's files) but not the arm rows the test copies (`arm_care_harm.jsonl`, `arm_V_qwen.jsonl`), so it ran and raised `FileNotFoundError`. Nothing was computed or lost. Fix: the test skips unless every file it reads is present; L-beh-2's stand-in test, which reads T1.5's rows, is hardened the same way before its pod. Rule: a test that reads local outputs skips on the exact files it opens, never on a folder that a pod's partial sync can create. Before the format x harm matrix's pod (`STEPS=st2fx`), the sync emulation caught the same class in `test_st_t15.py::test_adapters_are_verified_against_the_t14_hashes`: it ran when L-beh's safetensors was present and then verified all four T1.4 adapters of record, while st2fx syncs only L-beh. It now skips unless every file in `t15_adapters_v1.json` is present; the emulation passes (241 passed, 10 skipped). Cost: none.

## ST-A12 (ledger) — V_harm separates protect above rule in the trace, and steering along it moves GPT-OSS toward the rule

**Date.** 2026-10-22 (T1.4 pod, Amendment 16's gate arms; `outputs/t14/gpt_oss_20b/gen/gate_V_harm.jsonl`).
**Observation.** Correlational (Amendment 15's read): protect rollouts sit above rule rollouts on `V_harm` at clause ends, +0.0109 [+0.0081, +0.0138]. Causal: +`V_harm` at clause ends, α 267.3, ΔP −0.107 [−0.191, −0.037] against `none`, −0.079 [−0.147, −0.019] against the matched `fill`; coherent. +`care_harm` at the same norm: ΔP +0.048 [−0.039, +0.141], +0.076 [+0.008, +0.146] against `fill`; the two arms differ by +0.155 [+0.090, +0.228].
**Type.** sign flip (correlational vs causal), with a direction-specific contrast.
**Competing readings.** R_a (harm salience triggers the rule): protective deliberations talk about the harm the rule causes, so they carry harm content; but raising harm salience directly pushes the model toward its trained caution, which on H1 is the rule. The correlational sign reflects what protective traces discuss, the causal sign what harm salience does. R_b (reading position): the constancy read averages over every clause end, while steering acts at each one as it is generated; harm content late in protective traces may follow the decision rather than precede it. R_c (norm, not direction): excluded at this n by the fill contrast (both arms separate from the matched random direction).
**Discriminator.** R_a vs R_b: the constancy read restricted to clause ends before the trace's first mention of an option (zero GPU on the bank: positions before the decision is stated); R_a predicts the protect-above-rule sign holds there; R_b predicts it vanishes or flips. Conjunction with KDG-A30 (guardrail RL teaching "a care appeal does not override a rule"): R_a is that lesson's harm-side form.
**Status.** open; the zero-GPU discriminator is priced at minutes (author's call).
**Thesis impact.** R_a: on GPT-OSS, care content and harm content are separable levers on the action, pulling in opposite directions at one norm; an anchoring objective must keep harm out of its axis, which `V_care` does by construction and `care_harm` does not. R_b: the correlational table is a map of what traces say after deciding, and only the causal arms locate a lever.
**Replication, Phase 2 Cell 1 (2026-10-22; pod A rerun, 45 engaged H1_train × 16, fresh seeds; `outputs/st2a/`).** Anchored statement: +`V_harm` at clause ends, block 12, α 267.3, moves protect (−0.077 [−0.118, −0.036]) and defer (−0.051 [−0.092, −0.012]) into the rule option (+0.128 [+0.087, +0.170]); against the matched `fill` ΔP(protect) −0.048 [−0.084, −0.010] (`fill` itself moves the rule option +0.036); coherent (parse 0.965, length 0.92, NLL +0.036). Replicated in direction from the earlier 20 (−0.107). `care_harm` at the same norm: +0.010 [−0.025, +0.047] (MDE 0.051), so care beyond harm (+0.087 [+0.053, +0.123]) is a contrast between `care_harm`'s null and `V_harm`'s rule-ward shift, not a protective move.
**Rival readings (2026-10-22).** R_d (brake content): harm salience at clause ends is the content the brake runs on; predicts the shift follows the brake across scenarios and does not depend on the rule option's shape. R_e (refusal-shaped option): V_harm raises a refusal disposition, and the rule option is usually the withholding one; predicts the shift concentrates where the rule option withholds or declines and is weaker where following the rule means taking another action.
**Discriminators, zero GPU (2026-10-22; coding `data/st2_rule_option_shape_v1.json`, done from the option texts before any split).** (i) Option shape: 41 of the 45 rule options withhold or decline, 4 take another action (A-26, A-32, A-64, B-67; 3 borderline cases coded withhold). Rule-share shift +0.136 [+0.091, +0.181] (withhold) vs +0.047 [+0.008, +0.098] (another action); difference +0.089 [+0.021, +0.151], directionally R_e, but a 4-scenario group makes that interval fragile (exploratory). (ii) Brake: Spearman ρ between the per-scenario brake B (T1.1's sampled P − P_dose0, an independent sample) and V_harm's ΔP(protect) −0.24 [−0.54, +0.12]; with B from Cell 1's rollouts 0-7 and the shift from 8-15, −0.26 [−0.54, +0.08]; with the rule-share shift ~0 (−0.05, +0.09). No detectable association; the protect floor (strong-brake scenarios have little P left to lose) biases ρ negative, so the null leans against R_d less than it looks. (iii) −V_harm (pod B, 45 × 16): R_d predicts a protect-ward move; R_e predicts the rule option loses share to protect or defer without the brake's content.
**Status (2026-10-22).** open; "the brake's own direction" is not written until (iii) is in.

## ST-A13 (ledger) — L-beh protects on every finished rollout of the reserved scenarios, and half its rollouts run out the 1,536-token cap deliberating

**Date.** 2026-10-22 (T1.4 pod, `L-beh`'s comparator choice; `outputs/t14/gpt_oss_20b/gen/select_beh_*.jsonl`).
**Observation.** On the reserved 10 × 8 (monitoring scenarios, never trained on): the step-matched `L-beh` (58 optimizer steps after the budget fallback) finishes 40 of 80 rollouts, all `protect`, and truncates 40 at the cap; the epoch-3 snapshot (step 47) finishes 48, all `protect`, and truncates 32. Truncated traces loop on the decision ("But we can't. We must choose one…"). The unadapted model truncated 0.9% of H1 rollouts at this cap (T1.1). Training: DPO margin 44.1 at step 58 (24.8 on the reserved pairs); KL on held-back general text 0.027.
**Type.** control misbehaviour (the comparator arm) and Amendment 4's pre-registered trigger (an adapter shifting truncation by more than 5 points), seen at T1.4 rather than G3.
**Competing readings.** R_a (overfit preference, broken termination): 62 pairs, a margin of 44, and the adapter has learned to favour protect-flavoured deliberation without learning to end it; P over parsed actions reads 1.0 because only the rollouts that finish are counted. R_b (cap artefact): the adapter lengthens deliberation that would end beyond 1,536 tokens; a longer cap would recover the actions. The two differ in what a longer cap shows: R_a, more looping; R_b, finished rollouts.
**Discriminator.** T1.5 reads `L-beh` on H1_held at the cap of record with the truncation rate beside P (Amendment 4); a cap-4,096 rerun of the reserved 10 × 8 under the step-matched checkpoint (~0.3 A100-h) separates R_a from R_b.
**Status.** open; for the author at G3 (and before T1.5 if the comparator's validity is in question).
**Thesis impact.** R_a: the behavioural comparator does not restore protection, it removes the model's ability to finish deciding; ΔP over parsed actions would overstate it, and T1.5 must report P with truncation as missing, never imputed. R_b: the comparator works and the cap is the instrument's limit.
**Status update (2026-10-22, T1.5).** resolved: a cap problem, not broken termination. At cap 4,096 all 80 reserved rollouts finish and protect (median 1,381 tokens, 28 of 80 past 1,536). On H1_held at the cap of record 51% of `L-beh`'s rollouts truncate (none 1%), and 20% of its capability answers run out of tokens. resolution_type: experiment. The length is an adapter side effect reported with its verdict (non-specific).

## ST-A14 (ledger) — L-self moves GPT-OSS on general text 17× as far as L-anchor, shifts its preference strongly toward rule traces, and leaves its own objective far from met

**Date.** 2026-10-22 (T1.4 pod; `outputs/t14/gpt_oss_20b/t14_record.json`, phase `self`).
**Observation.** After 89 steps the standardized self-distillation term is 3.01 (it starts near the residual's own standardized distance) and the trace KL 1.21; on the held-back general sequences KL rose to 0.35 (`L-anchor` 0.020, `L-beh` 0.027); the DPO implicit reward margin on the reserved pairs fell to −41.0 (rule traces now far more likely than protect traces relative to base); the anchor-axis margin on reserved traces rose 0.045 → 0.063.
**Type.** control misbehaviour (an arm moving the model off-distribution), and a dose-dependence the gate foreshadowed (`self` β 0.5 broke generation, ST-A11 R_c).
**Competing readings.** R_a (norm mismatch): the target is an END_OF_PROMPT judgment-frame state (norm ~1,875) imposed on clause-end positions (~1,337), so the MSE pulls the residual's scale as much as its direction, which drags general text with it and, as the gate's `self` arms did, moves P toward the rule. R_b (σ standardization): per-dimension division by `chat_sigma` (min 0.20) amplifies low-variance dimensions, so the loss is dominated by a few dimensions the model cannot move without broad change. R_c (content): the judgment state itself carries rule-ward content (the `self` mix lowered the forced dose-0 answer at β 0.25, −0.068), and the adapter faithfully installs it.
**Discriminator.** Zero GPU: the per-dimension contribution to the final self loss on the reserved traces (R_b predicts a few σ-small dimensions dominate) and the norm ratio of adapted to base residuals at clause ends (R_a predicts it moves toward the target's). T1.5 measures the behavioural consequence.
**Status.** resolved (2026-10-21; Amendment 18 item 2, the pre-registered rule): R_a. The judgment-frame targets' norms (median 1,883) all lie above the `clause_end` norm band [1,166, 1,590]; σ-dominance does not hold (top 1% of dimensions 18.3% of the loss). Instrument failure: `L-self`'s H1_held generation is descriptive only, no panel, the σ-floored retrain deferred. R_c stays untested (the placement offset is confounded with position). resolution_type: calibration.
**Thesis impact.** R_a or R_b: the §5.3 objective as specified is an instrument problem, not a test of the thesis; `L-self`'s T1.5 verdict would carry that scope. R_c: self-distilling GPT-OSS's own judgment state pushes the action toward the rule, consistent with ST-A11 and KDG's judgment–action gap running rule-ward on this model.
**T1.5, descriptive (2026-10-22).** At dose 0 on the parity rows `L-self` moves the letter distribution far from the base (TV 0.603); merged on vLLM it puts no displayed letter in its top 20 on any row (parity TV 1.0), a format break rather than an engine difference. Its H1_held HF generation was declined at G3; `L-self` carries no verdict.

## ST-A15 (ledger) — The bf16 checkpoint on vLLM sits further from the HF stack than the MXFP4 engine does, through a few outlier rows

**Date.** 2026-10-21 (T1.5 pod gwobskujn4mmap, phase `base_vllm`; rows pulled to `outputs/t15_early/gpt_oss_20b/`, per-row table `parity_per_row.json`).
**Observation.** On the 64 parity rows (`dl_chat_neutral`, the 20 gate scenarios × permutations 0-3), mean letter TV: MXFP4-vLLM vs HF 0.0349 (the engine band), bf16-vLLM vs HF 0.0531, bf16-vLLM vs MXFP4-vLLM 0.0369; medians 0.018 / 0.021 / 0.019. Both checkpoints carry the same dequantized weights (the dequantization is exact), so the bf16 engine was expected closer to HF, not further. Three rows have bf16-vLLM as the odd engine (both its distances ≥ 0.10, MXFP4-vLLM vs HF < 0.05): row 4 H1-A-218 perm 0, row 15 H1-B-19 perm 0, row 40 H1-A-102 perm 2. Two rows have HF as the odd engine (both vLLM engines agree): rows 19 and 39, H1-B-49 perms 0-1; H1-A-168's three rows lean the same way. One argmax flip, bf16-vLLM vs both. On the remaining 59 rows the means are 0.032 / 0.044 / 0.032, a small residual shift on the bf16 engine besides the outliers.
**Type.** control misbehaviour (an engine instrument off its expected ordering), localized to a few rows.
**Competing readings.** R_a (router top-k flips): bf16 vs MXFP4 kernel accumulation changes the router logits enough to swap an expert on near-tied tokens, which moves a few rows a lot and the rest not at all; predicts the outliers coincide with near-tied router scores at some layer on those prompts. R_b (kernel accumulation): bf16 GEMM accumulation in vLLM's fused MoE differs from HF's eager loop by a small amount everywhere; predicts a uniform shift, which the outlier rows contradict as the whole story but which may explain the 59-row residual. R_c (a partial layout error, e.g. a bias or one projection misordered in some layers): predicts large, widespread distances (a wrong gate/up interleave would push TV toward 0.5); the medians rule out a gross error but not a small one. R_d (batch composition): T1.5 read the bf16 parity rows inside one engine run with ~1,250 generation and capability requests, the MXFP4 rows in a run of their own; vLLM's numerics can depend on what shares a step (the HF analogue is ST-A3); predicts the bf16 rows read alone fall back to the MXFP4 band.
**Discriminator.** On pod B, first (Amendment 23 item 3; `scripts/router_disc.py`; ~0.25 A100-h): the bf16 rows re-read alone at T1.5's engine settings (R_d), and per-layer router top-4 sets under HF and under the bf16 and MXFP4 engines in eager mode, each row alone (R_a: the outlier rows differ at a few near-tied (token, layer) positions and quiet rows do not; R_c: differences on every row in the same layers). The vLLM half runs only in the pod's vLLM venv; its HF half and the read are tested locally.
**Status.** open (discriminator scheduled on pod B). The per-adapter parity bar moved to the bf16 engine's own band over the 61 rows without the three odd rows (Amendment 23 item 1). The base check passed (0.0531 ≤ 0.0849); a shift common to `none` and the merged adapters cancels in adapter − `none` on the same engine (Amendment 20). The per-adapter parity rows read against both HF and the MXFP4 engine as they land.
**Thesis impact.** None on T1.5's ΔP if the shift is common to both arms of each comparison. If R_c, every vLLM bf16 row is suspect and the HF fallback becomes the path of record.

**Discriminator result (2026-10-22, pod B; `outputs/st2b/router_disc`, read locally).** Each row read alone in eager mode, letter TV against HF: the bf16 checkpoint 0.030 (the three T1.5-odd rows 0.000, 0.036, 0.035; quiet rows 0.029), the MXFP4 engine 0.039 (0.206, 0.067, 0.063; quiet 0.034), against T1.5's compiled, batched bf16 read of 0.053 (odd rows 0.16-0.18). Router top-4 sets differ from HF on about 5% of (token, layer) positions on every row, in both engines, at near-ties (HF's 4th minus 5th logit 0.008 where they differ, 0.078 typically), first at layer 0, at the same rate on odd and quiet rows. R_a rejected as the outliers' cause (flips are everywhere and leave most rows close); R_c rejected (read alone, the bf16 checkpoint is the engine closest to HF). The outliers vanish when the rows are read alone in eager mode, so they belong to T1.5's serving condition: R_d (batch composition) or the compiled path, not separated, since the compiled engine with the rows alone failed at startup (the helper did not run under the vLLM worker's environment, so FlashInfer's sampler tried to compile without `ninja`; a ~0.05 A100-h re-read separates them once the helper takes the worker's environment). Status: resolved (calibration): the bf16 checkpoint is faithful; the outlier rows are serving-condition jitter that `none` and every adapter share within each ΔP (Amendment 23 item 2).

## ST-A16 (ledger) — L-beh saturates the dose-0 letter readout, so the parity and merge gates pass it at TV ~1e-8 and cannot see an engine difference

**Date.** 2026-10-21 (T1.5 pod gwobskujn4mmap, phases `hf_beh` and `vllm_beh`; rows pulled to `outputs/t15_early/gpt_oss_20b/`).
**Observation.** On the 64 parity rows, L-beh puts ≥ 0.9999992 on one letter in every row (the violating option in all 64; the unadapted model's median top-letter probability is 0.81, violating in 52). Letter TV is therefore ~1e-8 between any two reads: merged vs attached on HF 2.1e-8, merged on vLLM vs attached on HF 2.3e-8 (the parity gate passes at 0.085 and at Amendment 23's 0.0975 alike: the bf16 band over the 61 rows is 0.0475). In log space over the displayed letters the engines stay comparable: merged bf16-vLLM vs HF attached |Δ log p| mean 0.227 (median 0.125, max 2.25; 20 of 192 letter values outside vLLM's top 20), against the base `none` bf16-vLLM vs HF 0.213 (0.132, 0.914) and MXFP4-vLLM vs HF 0.161 (0.121, 0.703); HF merged vs HF attached 0.301 (0.125, 1.75). The merge carries the adapter (merged vs base TV 0.282).
**Type.** control misbehaviour (a gate whose instrument saturates on the arm it judges).
**Competing readings.** R_a (ceiling only): the engines agree on L-beh as they do on `none`; the log-space distances, within the base engine's own range, are the informative read, and the gate's pass is correct but uninformative. R_b (hidden divergence): the merged engine differs from HF in-trace (where the distribution is not saturated) in a way the dose-0 readout cannot show.
**Discriminator.** Zero GPU now: the log-space read above (supports R_a). On a GPT-OSS pod, ~0.05 A100-h: teacher-forced first-N-token log-probabilities of a few saved L-beh traces under HF attached and under the merged bf16 engine (an in-trace parity read that does not saturate).
**Status.** open. The local read reports the log-space distance beside TV for every adapter (descriptive; whether it gates is an author item).
**Thesis impact.** None on ΔP if R_a. A saturated dose-0 readout is itself a result: L-beh moves the dose-0 decision on the H1_train gate scenarios to certainty, which is the in-sample face of its specificity failure (P(violating) on the 60 rule-right panel items 0.295 → 0.487 in the rows pulled mid-session; the read of record runs on download).
**Status update (2026-10-22).** R_a stands on the log-space read (all adapters: `L-beh` 0.225, epoch 3 0.267, `L-anchor` 0.178, against the base engine's 0.205); log-space parity stays descriptive until it has its own ladder (Amendment 24).

## ST-A17 (ledger) — The bf16 merge keeps about half of L-anchor's dose-0 effect: the merged checkpoint vLLM generates from is a perturbed copy of the trained adapter

**Date.** 2026-10-22 (T1.5 pod gwobskujn4mmap, phase `hf_anchor`, `merge_check`).
**Observation.** On the 64 parity rows, letter TV: adapter attached vs base 0.0525; merged (bf16, HF) vs attached 0.0267; merged vs base 0.0555; `kept_share` 0.49. Amendment 21's gate passes (merged closer to the adapter than to the base). For L-beh the same check read ~1e-8 (saturated, ST-A16; merged vs base 0.282).
**Type.** control misbehaviour (the evaluated artifact differs from the trained one by a fraction of the effect being evaluated).
**Competing readings.** R_a (bf16 rounding): the merge adds a small LoRA update to bf16 base weights; components below half a bf16 step round away (and PEFT merges in the weight dtype), so a low-learning-rate, low-KL adapter like L-anchor (general KL 0.020) loses part of its update while L-beh's larger update survives; predicts a merge in float32 rounded once leaves the same loss (sub-step components cannot survive any bf16 copy) and that the lost part is systematic, not noise. R_b (merge arithmetic in bf16): the delta itself is computed in bf16 before the add; predicts a float32 delta added before a single rounding recovers most of the gap.
**Discriminator.** Zero GPU on the downloaded adapter: the share of L-anchor's ΔW elements below half a bf16 step of their base weight, layer by layer (R_a's size). On a GPT-OSS pod, ~0.1 A100-h: the parity rows under a float32-delta merge, and under the adapter attached in float32 (R_b).
**Merge arithmetic against engine (from T1.5's rows).** Merged bf16 on HF vs attached on HF (merge only) 0.0267; merged on vLLM vs attached on HF (merge and engine) 0.0365; the bf16 engine alone on `none` (61 rows) 0.0475. The merge alone moves L-anchor's dose-0 letters by about half the adapter's own shift.
**Status.** open. G3 (2026-10-22): L-anchor's ΔP of record comes from HF generation with the adapter attached, against HF `none`, on the 38 engaged (pod B, first); the specificity rows under the HF-merged adapter are read beside the attached ones (merge arithmetic without the engine); if the bf16 merge halves small adapters, a merge-fidelity bar is set before T1.6.
**Thesis impact.** If R_a, the merged-bf16 vLLM path under-represents low-magnitude representation adapters, the class the recipe produces; any adapter verdict read through it carries that scope, and the HF path is the instrument for small-update adapters.
**Status update (2026-10-22).** resolved (calibration): the bf16 merge keeps `L-anchor`'s mean effect and adds per-item noise. On the specificity set (60 × 8, dose 0, HF only) attached − `none` +0.0236 [+0.0135, +0.0346], merged − `none` +0.0206 [+0.0120, +0.0298], merged − attached −0.0030 [−0.0087, +0.0024]; per-scenario |merged − attached| 0.014 against the adapter's 0.031. R_a (systematic loss of sub-step updates) is not supported for the mean. `L-anchor`'s ΔP of record came from HF with the adapter attached (null, −0.005 [−0.052, +0.041]); the vLLM column (+0.012 [−0.059, +0.086]) agrees. A merge-fidelity bar before T1.6 should test the mean shift and report per-item scatter beside it.

## ST-A18 (ledger) — Different directions move GPT-OSS on the same borderline items: the same seeded rollouts lose their refusal, and the same rule-right items tilt toward the violating option

**Date.** 2026-10-22 (pod C, `st2c`; `data/st2c_read_v1.json`).
**Observation.** Retention (60 plain violating × 4, `clause_end`, block 12, α 267.3): −refusal flips 8 refusing rollouts to compliance, −V_harm 10; all 8 of −refusal's are among −V_harm's, on the same 6-7 items (liberty, sanctity, authority; no care/harm item); paired difference +0.004 [−0.008, +0.017] (MDE 0.020), though the two directions sit at cos 0.157. V_qwen at α 133.6 flips 3, on items of the same set. Specificity (60 rule-right × 8, dose 0, vector at every input position): −V_harm +0.089, V_qwen +0.093, L-beh (an adapter, T1.5) +0.192; their per-item shifts correlate r 0.61-0.75; L-anchor (+0.024) against −V_harm −0.05; the items that move have baseline P(violating) 0.38 against 0.18 for the rest.
**Type.** cross-arm coincidence (one entity, two measures: merged per the triage rule).
**Competing readings.** R_a (generic perturbation): any vector of this size near the decision tips items close to their boundary; the shared seeds make it the same rollouts, and the dose-0 read tilts the same borderline items whatever the direction. Predicts `rr_fill` raises plain compliance on the same rollouts by about as much, and `spec_fill` lands near +0.09 on the same items. R_b (shared component): the three handles share a component that loosens refusal and tilts toward violating, beyond what a random vector does. Predicts `rr_fill` and `spec_fill` well below the arms.
**Discriminator.** The D-lite (Amendment 29, ~1.6 A100-h): `rr_fill` and `spec_fill` at α 267.3 and 133.6 with `rr_none` re-run on the same pod.
**Status.** resolved (experiment, D-lite 2026-10-22): R_b on both measures. A covariance-matched random vector at the same dose loosens plain refusal by +0.004 [−0.008, +0.021] (2 flips, both among the arms') and tilts rule-right items by −0.006 (267.3) and −0.011 (133.6); −V_harm and −refusal rise above it (+0.033, +0.029), and −V_harm and V_qwen tilt beyond it (+0.095, +0.104). V_qwen's plain-refusal rise is at the random-vector level for its dose.
**Thesis impact.** If R_a, the specificity instrument cannot pass any vector at these doses; Amendment 27's rule becomes "against fill" (S2), V_qwen is re-read under it, and "one failure mode" is a property of borderline items, not of the three handles. If R_b, the three handles share one non-specific route, and "no target on GPT-OSS" stands as a finding about the model at this site.

## ST-A19 (ledger) — −V_harm tilts Muse's rule-right items toward violating without moving its brake

**Date.** 2026-10-23 (Cell 4, `st2m`; `data/st2m_read_v1.json`).
**Observation.** At α 79.8 (0.2 × Muse's median norm), −V_harm moves protection on Muse's 45 engaged scenarios by +0.036 [−0.003, +0.078], exactly the random vector's +0.037 (difference +0.000 [−0.039, +0.039]), while its specificity read is +0.147 [+0.112, +0.183] (bar ±0.03). On GPT-OSS at the same relative dose it moved protection +0.114 and tilted rule-right items +0.089.
**Type.** family exception (a lever on one family, not the other; the non-specificity on both).
**Competing readings.** R_a (dose): Muse's brake responds to harm salience at a larger dose than GPT-OSS's; the tilt shows first. Predicts a protect move at 0.3-0.4× with the tilt growing. R_b (different read): Muse's brake does not read harm salience at `clause_end`, while the rule-right tilt is a generic effect of removing harm salience. Predicts no protect move at any coherent dose.
**Discriminator.** −V_harm on the Muse dose ladder (0.3× and 0.4×, 45 × 8, with spec reads), ~1.4 A100-h if it rides the `care_harm` ladder pod.
**Status.** open (not scheduled; priced in Amendment 31).
**Thesis impact.** If R_a, the two families differ in dose, not mechanism. If R_b, harm salience is GPT-OSS's brake handle and not Muse's, and "the lever and the non-specificity travel together" is V_qwen's property, not the harm axis's.

## ST-A20 (ledger) — care_harm on Muse lowers the violating option on rule-right items while moving protection up

**Date.** 2026-10-23 (Cell 4, `st2m`).
**Observation.** At α 79.8 on Muse, `care_harm` moves protection +0.067 [+0.019, +0.114] (+0.031 [−0.014, +0.081] over fill, below the gate's +0.05 margin; choices move off the rule, 0.489 → 0.358, into defer +0.064 and protect +0.067) and moves P(violating) on the 60 rule-right items by −0.048 [−0.071, −0.027] (−0.063 [−0.111, −0.019] against fill). Every other direction in both families tilts rule-right items toward violating (+0.09 to +0.30). On GPT-OSS, `care_harm` at 267.3 was null on the gate.
**Type.** sign flip (the only direction whose rule-right reading falls on the rule side).
**Competing readings.** R_a (conditional content): care content raises protection where the rule harms and lowers rule-breaking where the rule is right, the condition a lever needs. Predicts both effects grow with dose (protection up, violating down on rule-right items). R_b (deliberation and noise): care content shifts choices off the rule toward deferral generally, and the −0.048 is within Muse's wider specificity noise. Predicts no dose response in the rule-right read and a defer-led, not protect-led, rise.
**Discriminator.** The Muse dose ladder at 0.3× and 0.4× with spec reads (the author's, 2026-10-23); priced in Amendment 31.
**Status.** open; the ladder (2026-10-23, `data/st2ml_read_v1.json`) reads **unresolved** under Amendment 31: protection rises with dose (+0.067, +0.125, +0.188; 0.4× incoherent, NLL Δ 0.380) while the rule-right read climbs toward violating (−0.048, +0.063, +0.176) with a random vector's (+0.015, +0.065, +0.097). At 0.3× `care_harm` passes the gate (+0.125, +0.089 over the 0.2× fill) with a rule-right tilt equal to a random vector's at that dose (−0.001). R_a as stated (rule-side at every coherent rung) does not hold; the 0.3× candidate goes to a fresh-seed, dose-matched replicate (Amendment 32).
**Thesis impact.** If R_a, the program has its first candidate for a conditional lever, a content direction rather than a fixed outcome push. If R_b, no direction at `clause_end` conditions on whether the rule harms, and the weight-level route (L-beh-2) carries the lever search.

## ST-A21 (ledger) — L-beh-2 fails the panel guard by the narrowest margin

**Date.** 2026-10-23 (L-beh-2 pod, `st2lb`; `data/st2lb_read_v1.json`).
**Observation.** Panel guard ΔE (the 293-item half, KDG's estimators, against T1.3c's `none`) +0.020 [+0.0005, +0.040], MDE 0.028; the registered rule fails an adapter whose ΔE CI lies above 0. L-beh's ΔE was −0.004; L-anchor's flat.
**Type.** near-miss at a pre-registered bar.
**Competing readings.** R_a: L-beh-2 opens a small judgment-decision gap on the panel (its decisions move toward the rule while its judgments move less), a real side effect of training rule-right decisions. R_b: sampling noise at the bar (the CI's lower bound is 0.0005 above it; a second subsample or seed could fall either side).
**Discriminator.** The guard on the other 50% half of the panel under L-beh-2 (~2.1 A100-h), or the per-cell decomposition of ΔE from the saved rows (zero GPU; which cell carries the gap).
**Status.** open. The registered verdict stands (fails).
**Thesis impact.** If R_a, L-beh-2's rule-right gain costs a measurable judgment-decision gap, the property the regression gate exists to catch. If R_b, the guard does not separate L-beh-2 from `none`.

## ST-A22 (ledger) — L-beh-2 protects on H1 and follows the rule on the panel; format and harm are confounded

**Date.** 2026-10-23 (L-beh-2 pod).
**Observation.** L-beh-2 raises protection on held-out H1 scenarios (+0.763, P 0.22 → 0.98) and lowers the violating option on held-out panel items (specificity −0.093; panel half −0.063 on rule-right, −0.112 on neutral-right). Every H1 item is a rule-harm scenario in H1's format; every panel item is rule-right or neutral in the KDG format.
**Type.** design-matrix gap (format and harm vary together).
**Competing readings.** R_a (harm): L-beh-2 conditions on whether the rule harms. Predicts a drop in P(violating) on rule-right items in H1's format and a rise in protection on rule-harm items in the panel's format. R_b (format): it conditions on the scenario format. Predicts no drop (or a rise) on H1-format rule-right items and no rise on panel-format rule-harm items.
**Discriminator.** The two off-diagonal cells of the format × harm matrix, read at dose 0 under `none`, L-beh-2 and L-beh (Amendment 33; ~0.5 A100-h per cell).
**Status.** open (promoted: Amendment 33).
**Thesis impact.** If R_a, a weight-level lever that conditions on rule harm exists on GPT-OSS, with capability intact. If R_b, L-beh-2 learned the datasets, and the lever search needs format-crossed training data.
