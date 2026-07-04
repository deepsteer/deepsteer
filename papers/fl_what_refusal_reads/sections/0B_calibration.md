# Appendix B. Calibration, nulls, and validity controls {#app:calibration}

The verdicts in the main text rest on a calibrated ladder rather than on raw projection
magnitudes. This appendix gives the ladder per model, the null and band constructions, the
position-validity check that gates content projections, the standardization applied to
massive-activation models, and the normalization fold that certifies the per-head attribution. A
full treatment of these instrument failure modes and their positive controls is in the companion
methods note (in preparation); here we give the numbers the main text uses.

## B.1 The calibrated ladder {#app:ladder}

Every refusal projection is placed on a five-rung ladder: an isotropic floor, a covariance-matched
rank-matched null (reported at q50 and q95), the refusal projection itself, a held-one-out
moral-family band (the positive control), and a persona reference. The moral-family band is the
range of the three source directions each projected onto the span of the other two; a genuinely
moral direction held out of the subspace it belongs to projects high, so the band is the yardstick
for "moral-adjacent." Persona sits just below every band, so it is carried as a moral-adjacent voice
reference rather than as a clean non-moral control.

| Tag (layer) | Held-one-out (ms / fables / ethics) | Moral-family band [min, max] | Persona | Refusal point(s) |
|--------|--------|-------|---:|-------------------|
| Base (16) | 0.537 / 0.664 / 0.569 | [0.537, 0.664] | 0.510 | proto-refusal 0.33 |
| Instruct (16) | 0.523 / 0.637 / 0.555 | [0.523, 0.637] | 0.506 | gate 0.14 |
| Reasoning, OLMo-3-Think (16) | 0.537 / 0.667 / 0.573 | [0.537, 0.667] | 0.525 | gate 0.10 · harm-recognition 0.29 · in-trace 0.35 |
| GPT-OSS-20B (12) | 0.649 / 0.764 / 0.660 | [0.649, 0.764] | 0.603 | gate 0.19 · harm-recognition 0.47 · in-trace 0.52 · post-answer 0.25 |

: The calibrated ladder per model. Every refusal point on every model lands below its tag's
moral-family band, including the in-trace peaks on the two reasoning models, so even the program's
highest refusal projection (GPT-OSS in-trace 0.52) is less moral-adjacent than a held-out moral
direction. The base band-minimum has a 95% bootstrap confidence interval of [0.47, 0.53], and
refusal sits under it.

The null rungs the projections are read against: the base proto-refusal projects 0.33 against a
covariance-matched null q95 of 0.291; the aligned gate projects 0.14 against null q95 0.26; read
against richer constructions the aligned gate projects 0.144 onto the rank-3 moral subspace (null
q95 0.266) and 0.155 onto the six-foundation moral-foundations span (null q95 0.252), both null. On
the reasoning models the in-trace point is the only place refusal approaches its null: OLMo-3-Think
in-trace 0.35 sits just below its rank-matched null margin (a near-miss), while GPT-OSS
in-trace 0.52 crosses its null (0.32 to 0.34) yet stays below both the persona reference (0.60) and
the band [0.65, 0.76].

## B.2 The covariance-matched rank-matched null {#app:null}

The null and the persona control are computed mechanically from the subspace, not chosen. For a
subspace of realized rank $k$, the null is the projection of covariance-matched random directions
(random directions with the residual stream's covariance, at rank $k$) onto the subspace's span;
q95 is the reported bar, and a positive verdict requires refusal to clear q95 plus a fixed margin
$M = 0.05$. Because the null is a deterministic function of the subspace geometry and is realized
before the refusal vector is projected, the refusal projection never enters its own null. Bootstrap
confidence intervals use $B = 2000$ resamples; where a bar is a minimum of three noisy quantities
(the band minimum), the percentile interval is reported as primary with a bias-corrected-and-
accelerated interval as the robustness check, because the band minimum is downward-biased under
resampling in the direction that favors the sub-band claim.

## B.3 The band-below-null position-validity check {#app:position-validity}

A projection-fraction instrument only has discriminating power where a moral positive control
projects *above* the null. At the decision token this fails: on OLMo-3 the positive-control moral
band comes out at [0.40, 0.47], below the covariance-matched null of 0.557. When the positive
control is below the null, any direction (moral or not) projects onto the narrow channel at roughly
the null level, so a low projection there cannot certify absence. This is the band-below-null tell,
and it is why participation ratio is a required field and any position below 30 is flagged invalid
for content projection-fraction tests. Three independent estimates agree the OLMo-3 decision channel
carries about 15 effective dimensions: $\sqrt{3/14.7} = 0.45$ as a closed-form projection
expectation, the covariance null q95 of 0.557, and the pairwise-cosine null of 0.41 to 0.51. The
bottleneck is position-invalid for content projection tests but position-valid for
decision-direction cosines (a cosine between two directions both defined at the decision token is
immune to the projection null), which is the distinction that lets the refusal-versus-judgment
cosine stand where a content projection would not.

## B.4 Per-dimension standardization for massive-activation models {#app:standardization}

Two panel models carry massive-activation outlier dimensions that saturate the covariance-matched
null and make raw geometry uninterpretable: Qwen2.5-7B has a single dimension (index 458) holding
59% of the activation variance, and Llama-3.1-8B has one (index 788) holding 32%, against OLMo-3's
top dimension at 1.4%. GPT-OSS's content-position sample shows the same pattern (top-dimension
variance share 0.699). Geometry on these models is therefore computed after per-dimension
standardization (dividing each dimension by its standard deviation), with an OLMo raw-versus-
standardized invariance check that returns the same verdict both ways, the legitimacy proof that
standardization is not manufacturing the result. Under the routing lens this standardization
*sharpens* the harm read: on GPT-OSS the in-trace refusal cosine to the harm direction is 0.49
against 0.13 for the harm-orthogonal moral subspace standardized, against 0.57 and 0.22 raw, a
3.8-times gain in separation.

## B.5 The normalization fold for reordered-norm attribution {#app:ln-fold}

OLMo-2 and OLMo-3 apply RMSNorm to the attention and multilayer-perceptron output before the
residual add (reordered norm), so a naive per-head output-value decomposition skips the norm and
overshoots. Folding the per-layer RMSNorm gain onto each pre-norm component write (the gain
$g = \gamma / \mathrm{rms}$, exact because RMSNorm is diagonal at a fixed token) brings the
per-head reconstruction from 3.05 to 0.9999, inside the two-sided acceptance band [0.90, 1.10]. The
fold is exact and is unit-tested to $10^{-9}$. All per-head write and read numbers in
\Cref{app:causal} are folded; the un-folded anatomy is superseded, and the decisive interchange
cell is patch-based and was identical across the folded and un-folded runs. Llama-3.1 is pre-norm
and needs no fold, which its reconstruction of 1.0008 confirms as an architecture cross-check.
