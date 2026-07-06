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

## Process ledger

**2026-07-03 — the cold-boot W0 audit caught a live erratum that warm sessions had missed.**
The fresh-context ledger reconciliation (CLAUDE.md boot sequence) found that arXiv:2606.11375v1
lacks its own §4.4 activation-scale control, so its Finding 2 ships an un-scoped raw fragility
gradient → v2 erratum required. No warm working session had flagged it. Evidence that the
boot-sequence design pays for itself, and a datapoint for the open periodic-fresh-eyes cadence
question (how often to force a cold-context re-audit of committed/published claims).
