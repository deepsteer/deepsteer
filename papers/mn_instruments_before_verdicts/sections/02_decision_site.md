# 2. The decision-site instrument and its calibration ladder {#decision-site}

Four failures converge on one object: the projection-fraction / cosine instrument used to
ask whether a direction of interest lives inside a subspace. One is the position where it
fails, another shows that position is architecture-general, a third is the null that degenerates
underneath it, and a fourth is the attribution decomposition that overshoots when the same
channel is read per-head. Each is stated as: failure → tell → protocol → certifying check.

Several participation ratios recur below, at different positions and under different
normalizations. Table 1 lists them together so that each PR can be traced to its model,
position, and normalization.

| Model | Decision-site PR, raw [95% CI] | Decision-site PR, standardized | Content-position PR (last / mean) | Fraction of shuffle reference | Geometric-cell PR (raw → std) |
|---|---|---|---|---|---|
| OLMo-3-7B-Instruct | 14.7 [14.3, 16.2] | 20.3 | 62.8 / 40.4 | 0.066 | 43 → 94 |
| Qwen2.5-7B | 8.6 [8.2, 9.4] | 13.5 | 42.4 / 32.7 | 0.041 | 1.0 → 39 |
| Llama-3.1-8B | 10.2 of record; 10.3 [10.1, 11.1] on this sample | 14.2 | 97.3 / 26.7 | 0.050 | 1.5 → 89 |
| GPT-OSS-20B | 9.4 [9.1, 10.7] | 12.8 | n/a | 0.084 | n/a |

Table: Participation ratio (PR = (Σλ)²/Σλ²) by model, position, and normalization, measured on one
240-text sample per model (128 for GPT-OSS) at the primary layer. The raw decision-site column is
the value plotted in Figure 1; intervals are subsampling intervals over texts (the row-resampling
bootstrap is biased low because duplicated rows lower the sample rank, and is not used). The
standardized column is the same position after per-dimension z-scoring. The Llama 13.5 quoted in
§3.2 comes from the decision-anatomy harness (standardized, request-twin stimuli, a different
sample): one position under a second harness and normalization, not a second token; the two
standardized reads (13.5 there, 14.2 here) agree to within 0.7, and the in-format value of record
stays 10.2 (10.3 on this sample). Content-position PRs are full-rank-healthy. The shuffle reference is the PR
after independent column permutations, which keeps every marginal variance and destroys the
correlations: every decision site sits at 4 to 8 percent of it, and at 3.6 to 7.4 percent of its
sample-rank ceiling, while the content positions of the same texts sit 2.6 to 9.4 times higher.
The geometric-cell column is the raw → standardized pair of §2.3, where per-dimension
standardization lifts Qwen and Llama out of near-rank-1 collapse. GPT-OSS 12.8 is its harmony
decision-token PR, treated as position-valid for the refusal decision-direction read against a
separate MoE PR ceiling of 25 (§2.1). n/a marks a quantity not measured for that model.

## 2.1 Band-below-null means the position is invalid, not that the direction is absent {#a2-band-below-null}

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
dimension (at this decision token the top dim carries 0.2% of variance) and not a null that standardization can
rescue (the null stays 0.52 after z-scoring). The channel is simply narrow: participation
ratio 14.7. The `√(3/14.7) = 0.45` heuristic (a rank-3 subspace at PR 14.7) predicts a
median-scale projection; comparing that 0.45 against a null q95 of 0.557 and against the
rank-3 pairwise-|cos| null of 0.41–0.51 is a consistency check that the numbers are the
right size, not a convergence of three independent estimates on one value (0.45 is a
median-scale prediction, 0.557 is a q95).

**The protocol.** `participation_ratio` is a required type-block field on every extracted
direction, and any position whose PR sits far below its own references is flagged
position-invalid for content projection-fraction tests at extraction time. The gate is stated
null-referenced rather than as an absolute number: the decision-site PR against the
column-shuffle reference (the PR the same marginals would give without correlations), against
the sample-rank ceiling $n - 1$, and against the content positions of the same texts. All four
decision sites sit at 4 to 8 percent of the shuffle reference and 2.6 to 9.4 times below their
content positions (Table 1); an absolute threshold of 30 was the working rule in the program's
earlier sessions, and it is kept here only as the historical value. An absolute bar is not
evaluable at small sample sizes: with 64 rollouts the sample-rank ceiling is 63, and a content
position that reads 22 to 29 there is above its own covariance-matched sampling null while
failing the 30. (The false-invalid rate of the gate is not quantified beyond one panel case, GPT-OSS's
decision token, where the band survives the null despite a low PR; see Limitations.)

**The certifying check and the reframe.** Position-invalid does not mean uninterpretable
model. A projection-fraction test fails there, but a decision-*direction* cosine does
not: it is immune to the projection null. In a ~15-slot channel, refusal and judgment
directions occupy different slots at |cos| below even the low-dim random level, which reads
as active separation, not a weak-instrument artifact. Concretely, refusal-decision is
orthogonal to judgment-decision with no coupling detectable above |cos| 0.10 against a null
q95 of 0.41 on OLMo (0.32 vs 0.42 on Qwen, 0.08 vs 0.51 on Llama). Geometrically, the
moral-content band sits below the null at this bottleneck (band-below-null there, healthy at
content positions), so content-versus-decision orthogonality is structurally favored here.
This is a geometric observation, not a functional one: that moral content projects weakly
onto the decision channel does not by itself establish that it fails to reach the decision,
which is a causal claim that the note's own standard resolves only with an intervention cell.
Read as geometry, any comprehension-to-decision coupling would have to ride the attention
heads writing into the bottleneck, a concrete anatomical target.

One reconciling sentence is required for prose. The bottleneck is position-invalid for
content projection-fraction tests (band-below-null) and position-valid for decision-direction
reads (decision-direction cosine, and the GPT-OSS refusal projection). GPT-OSS's decision channel is called
"position-valid (PR 12.8)" against a separate MoE PR sanity ceiling of 25; that ceiling is
not the content rule.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{mn_ladder.pdf}
\caption{The calibrated ladder at the OLMo-3-Instruct chat decision token. The positive-control moral band [0.40, 0.47] sits \emph{below} the covariance-matched null q95 of 0.557. A positive control below the null means the instrument has no discriminating power at that position for content projection-fraction tests: band-below-null implies the position is invalid, so a low projection there cannot certify absence. This is the visual form of the tell.}
\label{fig:ladder}
\end{figure}

**Figure 2** is the calibrated ladder at this position: the moral band [0.40, 0.47] plotted
below the covariance null 0.557, the visual form of the tell.

## 2.2 The massive-activation outlier is position-dependent, so the bottleneck is clean {#a5-outlier}

**Failure as it appeared.** Llama-3.1 carries a massive-activation outlier: dim 788 holds
32% of residual variance. The worry was that this outlier contaminated every geometric read
on Llama, including the decision-token cells.

**The tell.** The outlier's variance share is a *content-position* statistic. The decision
token is a different position and had to be checked there, not assumed from the global
number.

**The protocol and check.** At the decision-token channel where the refusal and judgment
cells actually read, Llama is clean: participation ratio 13.5 (standardized, on that harness's
request-twin sample; 14.2 standardized and 10.2 raw on the in-format sample), covariance null 0.148, which
barely moves to 0.114 under per-dimension standardization. The outlier lives at content
positions, not at the ~13-dim control-token decision bottleneck, which is clean and low-rank
across OLMo and Llama alike. So the "decision site is a narrow control-token channel"
finding is cross-model, and the standardization fix matters more at content positions than at
the decision token. This is why the null degeneracy (next) and the bottleneck are two
different failures at two different positions, not one confound.

## 2.3 Covariance-matched nulls degenerate in massive-activation families {#a1-covariance-null}

**Failure as it appeared.** The covariance-matched, rank-matched null (draw random
directions from `N(0, Σ̂)` of residual activations, project onto the rank-r subspace) is the
honest null used throughout the program's earlier representational studies. On the instruct-model geometry it saturates:
the moral-subspace projection null q95 = 0.92 on Qwen and 0.36 on Llama, the pairwise-cosine
null q95 = 0.995 on Qwen and 0.90 on Llama, versus 0.26 on OLMo-3. At a saturated null every direction projects like a typical
direction, so the test has no discriminating power.

**The tell.** The null value itself is near its ceiling. The mechanism is the same massive
activations as the outlier finding above: Qwen dim 458 = 59% of residual variance, Llama dim 788 = 32%, OLMo-3's
top dim = 1.4% at the content position of the geometric cell. `Σ̂` is dominated by these dims, covariance-matched random directions nearly
all align with them, and they project ~1 onto any subspace with a component there. The same
dims collapse distinct raw mean-diff directions (Qwen ethics ≈ moral mean-diff |cos| = 0.90).
This is the known massive-activations / attention-sink phenomenon (Sun et al., 2024;
Xiao et al., 2023).

**The protocol.** Recompute directions and the null in a per-dimension-standardized space
(z-score by σ from a format/position-matched activation sample, sink tokens excluded), the
primary fix. The criterion-based robustness variant projects out each
dimension individually above 5% of variance. Behavioral results (ablation, judgment
accuracy) never use this null and are untouched; only geometric cells need the re-audit.

**The certifying check.** The clean instrument must give the same verdict raw and
standardized: OLMo, whose activations are well-conditioned, does. The quantitative
before/after is the participation ratio (Table 1, geometric-cell column): raw PR = OLMo 43,
Qwen 1.0, Llama 1.5 (one dim carries essentially all variance for Qwen and Llama); after
z-scoring, PR = OLMo 94, Qwen 39, Llama 89. The raw PR ≈ 1 shows the collapse was near-total;
standardization lifts Qwen and Llama into a genuinely multi-dimensional space.

**A boundary case that names the residual limit.** On the refusal-projection cell the two
robustifications *disagree*: standardization gives refusal 0.20 above controls 0.10
(strong-form false), while
top-k projection-out gives refusal 0.21 below controls 0.45–0.55 (strong-form true), and the
same split appears on Llama. When standardization and projection-out disagree, the subspace
is genuinely degenerate and needs a format or position change, not a null patch. The
in-format chat ladder (whose decision-site space carries no >5%-variance dim, so it is
outlier-free by construction) is the discriminator. This is the entry's own thesis applied to
itself: no single null repair resolves a genuinely rank-1 space.

**Scope of the fix.** Which null a cell uses decides whether the degeneracy touches it. The
instruct-model moral-subspace projection cells use the covariance-matched null (the one that
degenerates); the program's Qwen/Llama geometric cells use a permutation test and raw
(unnormalized) projection fractions; the behavioral cells use no geometric null at all. We report which null each cell class uses;
on that accounting the permutation-and-raw-projection cells are not exposed to this degeneracy:
the permutation test's observed statistics are ~0.01 (unsaturated), the raw projection fractions
are low and un-inflated (moral-subspace projection fraction 0.104 OLMo / 0.127 Qwen / 0.071
Llama, mean|cos| 0.04–0.07), and the moral-foundations subspace was built on the base model
whose foundation directions did not collapse onto the outlier dim. This "not at risk" reading
rests on a companion audit not released with this note and is not independently verifiable from
it. The degeneracy is confined
to the covariance-matched projection null applied to the instruct-model moral subspace. The
general caution stands: covariance-matched nulls silently degenerate in massive-activation
families, the field's default Llama/Qwen panel.

## 2.4 Reordered-norm architectures overshoot naive per-head OV attribution ~3× {#a3-ov-attribution}

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
RMSNorm fold. Folding the block norm into per-head attribution is standard
interpretability tooling \citep{elhage2021framework, nanda2022transformerlens}; the contribution
this note claims is not the fold but quantifying the ~3× overshoot it corrects on reordered-norm
architectures, plus the two-sided reconstruction gate that catches it (a one-sided floor misses
overshoot).
RMSNorm is diagonal at a fixed token, `norm(x) = (γ / rms(x)) ⊙ x`, so
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
