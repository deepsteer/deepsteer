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
