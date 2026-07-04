# 2. The decision-site instrument and its calibration ladder {#decision-site}

Four failures converge on one object: the projection-fraction / cosine instrument used to
ask whether a direction of interest lives inside a subspace. One is the position where it
fails, another shows that position is architecture-general, a third is the null that degenerates
underneath it, and a fourth is the attribution decomposition that overshoots when the same
channel is read per-head. Each is stated as: failure → tell → protocol → certifying check.

Several participation ratios recur below, at different positions and under different
normalizations. Table 1 lists them together so that each PR can be traced to its model,
position, and normalization.

| Model | Decision-site PR | Decision-token PR | Content-position PR | Geometric-cell PR (raw → std) |
|---|---|---|---|---|
| OLMo-3-7B-Instruct | 14.7 | n/a | 40+ | 43 → 94 |
| Qwen2.5-7B | 8.6 | n/a | 33+ | 1.0 → 39 |
| Llama-3.1-8B | 10.2 | 13.5 | 35+ | 1.5 → 89 |
| GPT-OSS-20B | 12.79 | n/a | n/a | n/a |

Table: Participation ratio (PR = (Σλ)²/Σλ²) by model, position, and normalization. The
decision-site column is the in-format-ladder value plotted in Figure 1; the decision-token
column is a second position and harness, measured for Llama only (13.5). Content-position PRs
are full-rank-healthy. The geometric-cell column is the raw → standardized pair of §2.3,
where per-dimension standardization lifts Qwen and Llama out of near-rank-1 collapse.
GPT-OSS 12.79 is its harmony decision-token PR, treated as position-valid for the refusal
decision-direction read against a separate MoE PR ceiling of 25 (§2.1). n/a marks a quantity
not measured for that model.

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
dimension (the top dim carries 0.2% of variance) and not a null that standardization can
rescue (the null stays 0.52 after z-scoring). The channel is simply narrow: participation
ratio 14.7. The `√(3/14.7) = 0.45` heuristic (a rank-3 subspace at PR 14.7) predicts a
median-scale projection; comparing that 0.45 against a null q95 of 0.557 and against the
rank-3 pairwise-|cos| null of 0.41–0.51 is a consistency check that the numbers are the
right size, not a convergence of three independent estimates on one value (0.45 is a
median-scale prediction, 0.557 is a q95).

**The protocol.** `participation_ratio` is a required type-block field on every extracted
direction, and any position with PR < 30 is flagged position-invalid for content
projection-fraction tests at extraction time. All three chat decision
sites (14.7 / 8.6 / 10.2) fall below the gate.

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
"position-valid (PR 12.79)" against a separate MoE PR sanity ceiling of 25; that ceiling is
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
cells actually read, Llama is clean: participation ratio 13.5, covariance null 0.148, which
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
top dim = 1.4%. `Σ̂` is dominated by these dims, covariance-matched random directions nearly
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
(unnormalized) projection fractions; the behavioral cells use no geometric null at all. A
companion audit found the permutation-and-raw-projection cells were never at risk: the
permutation test's observed statistics are ~0.01 (unsaturated), the raw projection fractions
are low and un-inflated (moral-subspace projection fraction 0.104 OLMo / 0.127 Qwen / 0.071
Llama, mean|cos| 0.04–0.07), and the moral-foundations subspace was built on the base model
whose foundation directions did not collapse onto the outlier dim. The degeneracy is confined
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
