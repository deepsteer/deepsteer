---
name: estimator-traps
description: >-
  Statistical estimator failure modes for representation-geometry work. Use this skill
  whenever computing or interpreting bootstrap CIs; comparing two estimates or an estimate to
  a threshold; taking a min/max/extremum over estimated quantities; conditioning a contrast
  on model outputs; aggregating evidence across layers, positions, or models; or judging a
  near-miss at a pre-registered bar. Fires on: "CI", "confidence interval", "bootstrap",
  "significant", "overlap", "q95", "threshold", "near miss", "combined", "Fisher",
  "conditioned on". Encodes difference-CIs (never overlap checks), bias-direction audits,
  resampling attenuation, extremum-statistic bias, selection-on-output designs, exploratory
  evidence combination, and covariance-matched nulls. Consult before any CI-based verdict.
---

# Estimator Traps

**Core principle: before a number decides anything, name the estimator's known biases and
state which side of your claim each one favors.** A bias working *against* your claim is
free strength; a bias working *for* it is a retraction waiting to happen.

## Required artifact: the bias-direction table

For any thresholded verdict, a three-column table: known bias → mechanism → direction
relative to the claim (favors / opposes / neutral). Two lines is fine. Its absence blocks
the verdict.

## The traps

1. **CI-overlap is not a difference test.** Two overlapping 95% CIs are compatible with a
   difference significant at p < 0.05. The test is the bootstrap CI of Δ = A − B
   (percentile or BCa, from the existing draws — zero new compute). *Program instance:*
   GPT-OSS band-min CI [0.47, 0.64] vs P2 CI [0.44, 0.53] was read as "not CI-clean"; the
   Δ-bootstrap was the pre-registered fix (normal-approx z ≈ 2.6 suggested clean, pending the
   proper percentile test).

2. **Point-outside-own-CI is an estimator smell, not a rounding issue.** If the point
   estimate falls outside its own bootstrap interval, the resampling distribution is shifted
   — investigate before shipping. *Program instance:* band-min point 0.65 ∉ [0.47, 0.64]:
   resampled directions are noisier than the full-data direction and project systematically
   lower (attenuation), and a min-statistic amplifies the downward pull. Consequence: the
   reported band-min *understates* the true band-min — a bias that favors the sub-band
   claim, which converts a weakness into an argument once stated.

3. **Extremum statistics are biased.** min/max over estimated quantities inherits and
   amplifies estimation noise (min biased low, max biased high). Report the extremum's own
   bootstrap distribution, and prefer testing against the extremum's *distribution* rather
   than its point value.

4. **Conditioning on model outputs imports content.** Contrasting activations by what the
   model *said* selects on inputs correlated with what the input *was* (items judged wrong
   are disproportionately truly wrong). Primary design: within-ground-truth-label contrasts
   (among truly-X items, model-says-X vs model-says-not-X), averaged across labels.
   Pre-declare the small-cell fallback (e.g. cell n < 15 → pooled contrast with the leak
   explicitly caveated).

5. **Near-misses at pre-registered bars, replicated, are information.** A 0.009 miss in one
   model and a null-crossing in a second independent model is not "NULL twice" — compute the
   combined evidence (Fisher across per-model bootstrap p-values), label it EXPLORATORY,
   and keep the per-unit pre-registered verdicts unchanged. *Program instance:* combined-P2
   p ≈ 1e-4 alongside intact per-model NULLs — both facts belong in the write-up.

6. **Multiplicity across layers/positions/models.** Pre-register the primary point; label
   everything else secondary; never let the headline migrate to whichever cell cleared the
   bar.

7. **Nulls need the right null.** Isotropic random directions understate chance alignment in
   anisotropic activation spaces by an order of magnitude; use covariance-matched,
   rank-matched draws. (In this program: isotropic floor ≈ 0.03 vs covariance-matched q95 ≈
   0.26–0.39 — the verdict flips if you pick the wrong one.)

8. **Bootstrap what you did, not what's convenient.** Resample at the unit of independent
   variation (pairs, prompts, rollouts) through the *entire* pipeline (direction →
   orthonormalization → projection), not just the final scalar.

9. **Massive-activation outliers saturate covariance nulls.** Check the top dimension's
   variance share before trusting any covariance-matched null: a handful of input-constant
   dims (Qwen dim 458 = 59%; Llama dim 788 = 32%) drove null q95 to 0.90–0.995 and made
   every cell uninterpretable, while the clean model (top dim 1.4%) was fine. Robustify
   (per-dim standardization from **format/position-matched σ**; top-k projection-out as a
   pre-registered variant) via a dated amendment *before* recomputing, and prove legitimacy
   with an invariance check on the clean instrument (verdicts must be identical
   raw→standardized). σ provenance is part of the null's type: a raw-covariance null applied
   to chat-extracted directions is a format mismatch stacked on the outlier problem. Note:
   standardization de-saturates but does not isotropize (post-fix eff-dim ~10–15).

10. **Full-space intuitions fail in low-dimensional channels.** Chance alignment is
   channel-relative: E|cos| ≈ sqrt(2/(π·d_eff)) for random pairs; rank-k null median ≈
   sqrt(k/d_eff). At d_eff ≈ 9–15, random pairs sit at 0.21–0.27 — a "suspicious" 0.32 can
   be exactly chance for its channel (measured: d_eff 8.6 → expected 0.27), while 0.08–0.10
   is *below* chance (active separation, a stronger claim than orthogonality). Every verdict
   inside a measured channel carries the channel-chance line.

11. **A null that moves under a transform needs a named reason.** If a null shifts when you
   standardize or whiten (0.27 → 0.41 on the clean model), name the mechanism (correlation
   concentration, generator change) in the amendment before using the new null. Silent null
   drift is how transforms manufacture verdicts.

12. **"One clears the MDE, the other doesn't" is the overlap fallacy in threshold clothing.**
   Whether an effect clears its minimum detectable effect is a *power-dependent* binary: the
   MDE shrinks as n grows, so the same effect flips from "doesn't move" to "moves" with no
   change in the effect itself. Comparing two effects (or one effect at two sample sizes) by
   which side of the MDE each lands on is trap 1 wearing a threshold — it is not a difference
   test. Normalize instead: a within-outcome **ratio** (fraction of the full-patch effect that
   survives a restriction) with a bootstrap CI on the *ratio difference*. *Program instance:*
   the C1 subspace-restricted refusal effect (~−0.03) sat below the MDE at n = 11 ("reads
   non-V_moral features") and above it at n = 23 ("V_moral is the read substrate") — the
   verdict flipped purely because the MDE tightened past a roughly constant effect. The
   ratio-of-ratios (R_judgment − R_refusal, CI-gated) bypassed the crossing and returned the
   honest `under_transfer`. Whether V_moral is a substrate *at all* is a separate, legitimate
   claim — settled by a paired Δ vs a random-rank-k control (CI excluded 0), not by the MDE
   crossing.

## Ship-blockers

- [ ] Bias-direction table attached to the verdict
- [ ] Any two-estimate comparison uses a Δ-CI, not overlap
- [ ] Extremum statistics carry their own bootstrap distributions
- [ ] Output-conditioned contrasts use within-label design or carry the leak caveat
- [ ] Replicated near-misses get a combined-evidence line, labeled exploratory
- [ ] Null is covariance-matched and rank-matched
- [ ] Point-inside-own-CI sanity check passed for every reported interval
- [ ] Top-dim variance share checked before any covariance null; σ provenance
      format/position-matched; robustification pre-registered before recompute
- [ ] Verdicts in measured channels carry the channel-chance level
- [ ] No presence/absence verdict rests on which side of the MDE an effect lands; two-effect
      comparisons use a normalized ratio + Δ-CI, not MDE-crossing

Pairs with: `instrument-calibration` (the ladder these CIs live on), `program-thesis`
(how exploratory results are worded).

*Changelog — v2 (2026-07-02): added traps 9–11 from the standardization arc (compound
Qwen saturation; channel-chance closure of the Qwen R3 anomaly; OLMo null drift). v2.1
(2026-07-02): trap 12 (MDE-crossing overlap fallacy) from the C1 hardening run — the
ratio-of-ratios reclassification of the refusal-substrate verdict.*
