# Appendix C. Causal anatomy: full tables (OLMo) {#app:causal}

This appendix gives the full tables behind the OLMo-3 causal result that \Cref{reads-harm}
summarizes: the per-head write decomposition, what each writer reads, the decisive interchange
cells with their minimum detectable effects and confidence intervals, the nested rank sweep, the
one-knob fit, and the harm-partialed identification cell. All numbers are on OLMo-3-7B-Instruct at
the decision channel, folded per \Cref{app:ln-fold}; interchange uses request-twins (matched
requests carrying opposite judgment outcomes, $n = 23$).

## C.1 Who writes the decision {#app:write}

The refusal write is distributed, not a sparse safety-head circuit. Cumulative channel-matched
specificity reaches only 44% at the top ten heads and needs about 62 heads to reach 80%; the write
is led by one head with a long tail, and multilayer perceptrons carry 38% of the decision-site
write (write fraction 0.384, below the 0.50 Jacobian threshold so the head decomposition is
adequate).

| Head | Write onto refusal | Channel-matched specificity |
|---|---:|---:|
| L16 H23 | $+0.742$ | $+0.756$ |
| L15 H2 | $+0.302$ | $+0.368$ |
| L14 H19 | $+0.334$ | $+0.347$ |
| L15 H0 | $+0.265$ | $+0.285$ |
| L11 H20 | $+0.246$ | $+0.274$ |
| L16 H21 | $+0.172$ | $+0.197$ |
| L14 H22 | $+0.178$ | $+0.193$ |
| L15 H6 | $+0.175$ | $+0.189$ |
| L13 H29 | $+0.139$ | $+0.144$ |
| L15 H15 | $-0.130$ | $-0.142$ |

: Per-head write onto the refusal direction and channel-matched specificity, top ten writers. The
lead head L16 H23 alone carries 11.6% of the total specificity; writers span layers 11 to 16, and
L15 H15 is the sole anti-refusal writer. Refusal is written broadly into the decision channel, led
by one head but not carried by it.

## C.2 What each writer reads {#app:read}

Every one of the ten top writers reads content only weakly aligned with the moral subspace: none
clears the moral-family band, and none is a clean harm-copy head. Moral-subspace fraction runs 0.15
to 0.28 with comparable harm loading, and the writers split into instruction-attenders and
content-attenders, but all are labeled neither-moral-nor-harm.

| Head | Attention plurality | Instruction-token frac | Content frac | Moral-subspace frac | Harm cosine | Label |
|---|---|---:|---:|---:|---:|---|
| L16 H23 | instruction | 0.735 | 0.265 | 0.278 | 0.264 | neither |
| L15 H2 | content | 0.302 | 0.679 | 0.219 | 0.226 | neither |
| L14 H19 | instruction | 0.776 | 0.222 | 0.238 | 0.247 | neither |
| L15 H0 | instruction | 0.642 | 0.312 | 0.254 | 0.247 | neither |
| L11 H20 | instruction | 0.875 | 0.125 | 0.195 | 0.188 | neither |
| L16 H21 | instruction | 0.740 | 0.260 | 0.254 | 0.229 | neither |
| L14 H22 | content | 0.438 | 0.555 | 0.239 | 0.260 | neither |
| L15 H6 | content | 0.040 | 0.637 | 0.174 | 0.032 | neither |
| L13 H29 | template | 0.001 | 0.019 | 0.146 | 0.031 | neither |
| L15 H15 | content | 0.096 | 0.510 | 0.173 | 0.086 | neither |

: What each top writer reads, in the shared residual basis. No single head reads the moral subspace
cleanly, consistent with the causal result that the moral subspace carries only a specific minority
of the refusal effect.

## C.3 The decisive interchange cells {#app:interchange}

Patching the decision channel with a request-twin's content and reading the induced change in the
refusal and judgment projections. The moral-subspace restriction moves refusal about a third as
much as the full patch, almost all of which is the harm slice; a random rank-3 patch moves nothing.

| Interchange cell | Effect | Minimum detectable effect |
|---|---:|---:|
| Full $\to$ refusal | $-0.0833$ | 0.0238 |
| Moral-subspace-restricted $\to$ refusal | $-0.0282$ | 0.0238 |
| Complement (off-subspace) $\to$ refusal | $-0.0636$ | 0.0238 |
| Harm-rank-1 $\to$ refusal | $-0.0261$ | 0.0238 |
| Random-rank-3 $\to$ refusal (control) | $-0.0005$ | 0.0238 |
| Full $\to$ judgment | $+0.0459$ | 0.0086 |
| Moral-subspace-restricted $\to$ judgment | $+0.0237$ | 0.0086 |

: The decisive interchange cells. The moral subspace is a specific refusal substrate: restricting
to it moves refusal more than a random rank-3 patch does ($\Delta = 0.031$, paired 95% CI [0.020,
0.043], excludes 0). But the harm-restricted patch ($-0.0261$) nearly equals the full
moral-subspace patch ($-0.0282$), so almost all of the specific effect is harm. The
ratio-of-ratios of restricted-to-full transfer is wider than the sweep below (refusal 0.34,
judgment 0.52, difference 0.18 with a bootstrap 95% CI of [$-0.24$, 0.39] that includes 0 at this
count), so the sweep, not the single ratio, resolves the shape.

## C.4 The nested rank sweep {#app:sweep}

Restricting the content patch to nested moral subspaces of rank $k$ (eigenvectors of the paired
moral-neutral content-contrast covariance; per-rank purity 0.97 to 0.99, a genuine moral basis) and
reading the transfer coefficient $R_r(k)$, the fraction of the full interchange effect on readout
$r$ reproduced under the restriction.

| $k$ | $R_{\text{refusal}}(k)$ | $R_{\text{judgment}}(k)$ | Random-direction null |
|---:|---:|---:|---:|
| 1 | 0.01 | 0.05 | $\approx 0$ |
| 3 | 0.31 | 0.46 | $\approx 0$ |
| 8 | 0.26 | 0.59 | $\approx 0$ |
| 16 | 0.27 | 0.66 | $\approx 0$ |

: The nested rank sweep, the central causal result. Judgment transfer climbs to 0.66 while refusal
transfer saturates at about 0.27 by rank 3, sitting at the harm-rank-1 level (harm-rank-1 transfer
0.31), with the random-direction null near zero at every rank. Expanding the moral basis beyond
harm buys more judgment coupling and no more refusal coupling; about 73% of refusal's causal
twin-difference input lies outside the rank-16 moral basis (69% already at the rank-3 peak).

## C.5 The one-knob fit {#app:oneknob}

The whole sweep collapses to a single free parameter: refusal transfer is judgment transfer clipped
at a harm ceiling, $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling}, R_{\text{judgment}}(k))$
with the ceiling $\approx 0.31$.

| $k$ | Measured $R_{\text{refusal}}(k)$ | One-knob prediction | Residual |
|---:|---:|---:|---:|
| 1 | 0.013 | 0.052 | $-0.039$ |
| 3 | 0.31 | 0.31 | $-0.002$ |
| 8 | 0.26 | 0.31 | $-0.05$ |
| 16 | 0.27 | 0.31 | $-0.04$ |

: The one-knob fit. Over the plateau ($k \geq 3$) the fit is near-exact (RMSE 0.036, residual
$-0.002$ at $k = 3$), while two harm-amplitude alternatives (the ceiling scaled by the harm-capture
fraction, and by its square) miss by 0.10 to 0.24. The one place it breaks is rank 1, where it
over-predicts (measured 0.013 against predicted 0.052): the highest-variance contrast component,
the most harm-aligned single direction (variance purity 0.974, cosine 0.35 to harm), is causally
inert, moving neither readout at rank 1. Variance is not causal relevance; the harm read is a
rank-1 causal object that is not the rank-1 variance object.

## C.6 The harm-partialed identification cell {#app:identification}

Harm alone nearly reproduces the full moral-subspace effect, but a resolvable non-harm moral read
remains. Projecting the harm direction out of the moral subspace and patching the residual still
moves refusal $-0.0133$ (95% CI [$-0.023$, $-0.005$], excludes 0), about half of the full
moral-subspace effect. The harm direction captures a fraction 0.46 of the moral subspace, and the
subspace-versus-complement decomposition is additive (ratio 1.10, 95% CI [0.91, 1.35], includes 1).
So the moral subspace's refusal effect is harm-dominant with a small, resolvable non-harm residual,
and that residual is the one place refusal demonstrably reads moral content beyond harm.

## C.7 The behavioral severity ladder {#app:behavioral}

The harm-keyed, saturating read is coherent with OLMo-3 being a weak intent-refuser. On
intent-harmful requests its refusal reaches only about 17% at top severity (violating items
0 / 0.17 / 0 / 0.17 / 0.17 across the five-level severity ladder, benign items 0 throughout), so
the behavioral operating band is nearly empty. This is a model property (weak coupling between
intent severity and refusal) that a harm-surface-keyed gate predicts, not a stimulus artifact, and
it is the reason the cross-model commitment axis in \Cref{app:panel} is measured on Llama and
GPT-OSS rather than on OLMo, whose refusal barely fires on these requests both in projection
($-0.08$) and in behavior (17%).
