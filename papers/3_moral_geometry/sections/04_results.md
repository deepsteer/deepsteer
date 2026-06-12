# 4. Results

## 4.1 Foundation-specific probe accuracy

All six foundation-specific probes achieve perfect or near-perfect
accuracy across the full depth of OLMo-2 1B. Every foundation
reaches 100\% peak accuracy. Authority/subversion achieves 100\% at
all 16 layers; the remaining foundations show minor fluctuations at
individual layers (minimum 87.5\% for care/harm at layer 1). The
onset threshold (0.6) is exceeded at layer 0 for all foundations.

These results confirm that moral foundation content is linearly
separable from the earliest layer, consistent with the "immediate
onset" finding of \citet{reblitzrichardson2026fragility} for the
pooled binary moral/neutral probe. The per-foundation decomposition
shows no qualitative difference in *detectability* across
foundations: all are equally easy to decode. The interesting
variation is not in accuracy but in the *geometry* of the probe
directions that achieve this accuracy.

On OLMoE-1B-7B, all foundations similarly reach 100\% peak accuracy,
with the lowest early-layer values at 81.25\% (layers 0--1). The
accuracy profiles are essentially indistinguishable between
architectures.

These peak accuracies are maxima over 16 layers on small held-out sets
(8--16 examples per foundation), so they should be read as "easily
decodable" rather than as precise point estimates: the exact binomial
95\% confidence interval on a perfect 16/16 still reaches down to
${\sim}0.79$. The geometric analysis below, not these saturated
accuracies, carries the paper's claims.

## 4.2 Framework geometry: integration, not collapse

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig1_cosine_heatmap.pdf}
\caption{Pairwise cosine similarity between foundation probe directions at layer 7 (bootstrap-stable; all six foundations exceed the 0.8 stability threshold at this layer; §4.6), OLMo-2 1B. Mean off-diagonal cosine = 0.262. See Appendix~\ref{app:cosine} for matrices at layers 0 and 15.}
\label{fig:cosine_heatmap}
\end{figure}

The headline finding: foundation probe directions are *separated*,
not collapsed. Across bootstrap-stable layers (6--15, where all six
directions exceed the 0.8 stability threshold; §4.6, with the single
exception of authority at layer 8, 0.792), mean pairwise
cosine similarity ranges from **0.232 to 0.274**, far below the
collapse threshold ($>0.95$) and below the intermediate zone
($0.8$--$0.95$). Figure~\ref{fig:cosine_heatmap} shows the
representative pattern at layer 7 (mean cosine 0.262). The
peak-separation layer (layer 0, mean cosine 0.216) is below the
bootstrap stability threshold for all foundations and is reported
in Appendix~\ref{app:cosine}.

The cosine similarities are uniformly *positive* at all layers
(range 0.14--0.35), indicating that the foundation directions share
a common component. This is the *integration* signature from our
trichotomy: the directions are separated but non-orthogonal,
consistent with a shared moral-salience subspace from which
foundation-specific directions deviate.

**Effective dimensionality.** The six foundation directions span
5 effective dimensions (the number of PCs explaining $\geq 90\%$
of variance) at every layer, confirming that the directions are
geometrically distinct and do not collapse into a lower-dimensional
subspace. Effective dimensionality alone does not establish
*integration*, however: six random directions in this hidden space
also span ${\sim}5$ effective dimensions (§4.8), so a near-maximal
eff-dim is equally consistent with the isolation regime. The
integration signal is carried by two statistics that random
directions do not reproduce: the uniformly positive mean pairwise
cosine (0.26, vs.\ ${\sim}0$ for random directions), and the
concentration of variance on a shared leading axis. The first
principal component of the six unit directions captures **0.379** of
their variance averaged over the stable layers 6--15, more than
double the **0.179** expected for six random unit vectors in 2048
dimensions ($\approx 1/6$). Both reflect the shared moral-salience
component; the effective dimensionality only rules out collapse.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig2_layerwise_geometry.pdf}
\caption{Layer-wise geometric metrics for OLMo-2 1B. (a) Mean pairwise cosine similarity is relatively flat across layers. (b) Effective dimensionality remains constant at 5 across all layers. (c) MFT group structure: mean cosine within the individualizing group, within the binding group, and between groups track together across layers, so the directions do not separate into the predicted individualizing/binding clusters.}
\label{fig:layerwise}
\end{figure}

## 4.3 Dendrogram structure does not recover MFT groups

\begin{figure}[t]
\centering
\includegraphics[width=0.7\linewidth]{fig3_dendrogram.pdf}
\caption{Hierarchical clustering (Ward's method) of foundation probe directions at layer 7 (bootstrap-stable), OLMo-2 1B. The first split does not recover the MFT individualizing/binding distinction.}
\label{fig:dendrogram}
\end{figure}

Hierarchical clustering of the six foundation directions does
*not* recover the MFT individualizing/binding distinction at any
layer. At layer 7, loyalty and authority merge first, liberty joins
them, then care and fairness merge separately, then sanctity joins
the loyalty--authority--liberty cluster. No layer produces the
predicted \{care, fairness, liberty\} vs.\ \{loyalty, authority,
sanctity\} partition.

The most consistent clustering pattern across layers is a
care--sanctity pairing: these two foundations appear in the same
cluster at 10 of 16 layers. This crosses the MFT boundary (care is
individualizing, sanctity is binding) but has a plausible semantic
interpretation: both foundations concern protection of
vulnerable entities (persons from harm, sacred things from
degradation). The loyalty--authority pairing at layer 7 is within
the binding group, but liberty (individualizing) joins this cluster
before grouping with the other individualizing foundations (care,
fairness), further mixing MFT categories.

The permutation test for the individualizing/binding distinction
does not reach significance at any layer (minimum $p = 0.32$; median
$p = 0.53$). With only 6 items and 20 unique 3--3 partitions,
statistical power is limited, but the consistently high $p$-values
combined with the absence of MFT-aligned dendrograms at any layer
indicate that the model's inter-framework geometry does not reflect
the MFT group structure on this dataset.

## 4.4 Layer-wise geometric development

The geometric structure is relatively stable across layers. Mean
pairwise cosine similarity is lowest at layer 0 (0.216), rises
modestly to a peak of 0.274 at layer 6, then returns to 0.232 at
layer 13 before a slight increase at layer 15 (0.262). The
variation is small (range 0.06) compared to the distance from
collapse, indicating that framework separation is a consistent
property of the representation space rather than an emergent
late-layer phenomenon.

Effective dimensionality remains constant at 5 across all layers,
meaning the rank of the direction set does not change even as the
pairwise angles shift. The geometric structure is a rotation of
the direction set within a fixed-rank subspace, not a
dimensionality change.

## 4.5 Dense vs. MoE framework geometry

Framework geometry is remarkably similar between architectures.
OLMoE-1B-7B shows mean pairwise cosine similarity ranging from
0.219 to 0.287, compared to OLMo-2 1B's 0.217 to 0.282. Effective
dimensionality is 5 at all layers for both models. The overall
degree of framework separation is consistent across dense and MoE
architectures in our comparison.

Neither architecture produces MFT-aligned dendrogram structure at
any layer. Both models show the care--sanctity pairing as the
most stable clustering feature, suggesting that inter-framework
geometry is driven by semantic relationships in the training corpus
rather than by architectural properties. The permutation test for
MFT group structure is non-significant at all layers for both models
(all $p > 0.25$).

This finding extends \citet{reblitzrichardson2026dilution}: output
dilution affects moral encoding *scale* (74$\times$ signal gap) but
not framework *structure*. The geometric organization of moral
foundations is preserved across architectures.

## 4.6 Direction stability under bootstrap

Bootstrap resampling (200 iterations) shows a stability gradient
across layers. Early layers (0--5) show borderline stability (mean
cosine similarity with the full-data direction: 0.74--0.80, below
the 0.8 threshold for most foundations). Middle and late layers
(6--15) are stable (mean cosine $> 0.80$). Sanctity/degradation
is the most stable foundation (13/16 layers stable); care/harm the
least (10/16 stable).

This gradient has two implications. First, the geometric analysis
at early layers, including the peak separation layer (layer 0),
should be interpreted with caution, as the specific pairwise
cosine values may shift under resampling. Second, the stability
gradient itself is informative: probe directions become more
determined as representations become more specialized, paralleling
the lexical-to-compositional gradient from
\citet{reblitzrichardson2026fragility}.

The headline geometric findings (separation not collapse; effective
dimensionality = 5) are confirmed at layers 6--15 where directions
are stable.

## 4.7 Differential fragility across frameworks

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{exp7_mean_critical_bars.pdf}
\caption{Mean critical noise $\sigma^*$ per foundation for OLMo-2 1B (left) and OLMoE-1B-7B (right), seed-averaged with cap-at-max and bootstrap 95\% CIs over noise seeds. Every foundation is more fragile in MoE than in dense (the cross-architecture dilution effect); within each architecture the per-foundation CIs overlap and the ordering is not statistically separable.}
\label{fig:fragility}
\end{figure}

We measure per-foundation critical noise $\sigma^*$ with the standard
fragility protocol, averaging accuracy over 10 noise seeds and
censoring never-fragile layers at the grid maximum (the cap-at-max
convention of \citet{reblitzrichardson2026fragility}); bootstrap 95\%
CIs are over the noise seeds. Seed-averaging is necessary here: a
single noise draw makes per-layer $\sigma^*$ unstable.

**The result is between architectures, not between foundations.**
Every foundation is more fragile in OLMoE than in OLMo-2: dense
$\sigma^*$ runs 3.8--5.6 across the six foundations (mean 5.0), MoE
$\sigma^*$ runs 1.2--3.8 (mean 2.2), a per-foundation architecture gap
of ${\sim}2.3\times$. This is the same direction as, and smaller than,
the 4.2$\times$ pooled gap reported by
\citet{reblitzrichardson2026dilution}; the pooled probe trains on
192 pairs versus 32 per foundation here, so it has more statistical
power and a cleaner separation. Output dilution suppresses moral
encoding across the board.

**Within an architecture, foundations are not reliably separable.**
The per-foundation $\sigma^*$ values have wide, overlapping bootstrap
CIs (dense: e.g. sanctity $5.50$, CI $[3.6, 8.6]$; care $4.81$, CI
$[3.5, 8.0]$. MoE: sanctity $2.33$, CI $[1.3, 5.3]$; care $3.76$, CI
$[0.9, 6.7]$), and no foundation is robustly most or least fragile.
The binding-vs-individualizing group difference is not significant in
either model and reverses sign between them (dense: binding $-$
individualizing $= +0.58$, exact permutation $p = 0.40$; MoE: $-0.41$,
$p = 0.70$). We therefore make no per-foundation or MFT-group fragility
claim; the supported finding is the uniform cross-architecture dilution.

## 4.8 Geometric trajectory during training

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig6_geometric_trajectory.pdf}
\caption{Geometric trajectory during OLMo-2 1B pre-training. (a) Mean cosine similarity stabilizes by step 2000. (b) Effective dimensionality remains at 5 throughout. (c) Accuracy continues climbing after geometry plateaus.}
\label{fig:trajectory}
\end{figure}

Framework geometry develops rapidly and stabilizes early. Tracking
mean pairwise cosine similarity (averaged over stable layers 5--14)
across 20 OLMo-2 1B checkpoints from step 0 to step 35,000:

- **Step 0.** Mean cosine similarity is 0.007; the six
  foundation directions are effectively orthogonal, as expected for
  probes trained on random representations. Mean accuracy is 0.510
  (chance).

- **Step 1000.** Cosine similarity jumps to 0.176. Accuracy reaches
  0.734. The model has already begun developing shared moral-salience
  structure.

- **Step 2000.** Cosine similarity reaches 0.382, within 5\% of
  its final value. Accuracy is 0.873, still 10 points below its
  peak.

- **Steps 3000--15,000.** Cosine similarity plateaus at
  $0.38$--$0.40$. Accuracy continues climbing from 0.922 to 0.970.

- **Steps 20,000--35,000.** Cosine similarity drifts slightly
  downward (0.382 to 0.367). Accuracy stabilizes at 0.975--0.979.

Effective dimensionality remains constant at 5 from step 0 onward.
This is expected: random unit vectors in $\mathbb{R}^{2048}$ are
nearly orthogonal with high probability, so even at initialization
the six probe directions span 5 effective dimensions. The
informative signal is cosine similarity, which transitions from
$\approx 0$ (random) to $\approx 0.4$ (mature structure) during the
first 2000 training steps.

The temporal dissociation (framework geometry stabilizing at step
2000 while accuracy continues improving through step 25,000)
extends the two-phase pattern identified by
\citet{reblitzrichardson2026fragility}: accuracy saturates early, but
fragility continues resolving. Here we add a third metric:
inter-framework *structure* also stabilizes before inter-framework
*discriminability* finishes developing. The model discovers the
geometric layout of moral concepts early and then spends the
remainder of training strengthening the representations within that
fixed layout.

## 4.9 Dilemma compositionality: partial but structured

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_dilemma_subspace_heatmap.pdf}
\caption{Subspace membership scores for 15 dilemma probe directions across 16 layers. Each cell shows the fraction of a dilemma direction's variance explained by the 2D subspace of its component foundation directions. Liberty--sanctity shows the strongest compositional signal.}
\label{fig:dilemma_heatmap}
\end{figure}

We now ask whether the model's representation of moral *dilemmas*
(scenarios where two foundations conflict) can be decomposed in
terms of the single-foundation directions from Experiment 1.

**Dilemma probes achieve high accuracy.** All 15 dilemma-specific
probes achieve $\geq 75\%$ peak test accuracy (mean 94.2\%), with
13 of 15 pairs at $\geq 87.5\%$. The model reliably distinguishes
dilemma moral content from matched neutral text.

**Subspace membership: partial compositionality.** The mean peak
subspace membership score across all 15 pairs is **0.099**,
approximately 100$\times$ the null baseline of 0.001 ($p_{95} =
0.003$; $p_{99} = 0.004$). Every pair exceeds the 99th percentile
of the null distribution at its peak layer. However, 0.099 is far
from 1.0: on average, only ${\sim}10$\% of each dilemma direction's
variance is explained by its component foundation subspace. The
remaining ${\sim}90$\% (mean residual norm = 0.949) lies in
directions orthogonal to both component foundations.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_dilemma_subspace_layers.pdf}
\caption{Mean subspace membership across layers (red), with $\pm 1$ SD band. The null baseline (gray) is ${\sim}0.001$. Membership is relatively flat at ${\sim}8$\% across all layers.}
\label{fig:dilemma_layers}
\end{figure}

**Component balance is near-equal.** The mean component balance
ratio is 0.486 (perfect balance = 0.5). In 14 of 15 pairs, the
balance falls within [0.40, 0.58], indicating that both component
foundations contribute approximately equally to the within-subspace
projection. The exception is fairness--sanctity (balance = 0.335),
where the sanctity component dominates. When
two foundations *do* compose, they compose symmetrically rather
than one foundation dominating the representation.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_dilemma_balance.pdf}
\caption{Component balance at each pair's peak subspace membership layer. Values near 0.5 (dashed line) indicate balanced contribution from both foundations. Fairness--sanctity (red) is the only pair with substantial imbalance.}
\label{fig:dilemma_balance}
\end{figure}

**Shared-component structure.** Dilemma pairs that share a
foundation component have consistently higher cosine similarity
than pairs with no shared foundation. At the peak effect layer
(layer 13), the mean cosine similarity between shared-component
pairs is **0.269** versus **0.195** for non-sharing pairs (difference
= 0.074). This difference is positive at every layer, indicating
that the compositional structure is not layer-specific but a
general property of the dilemma direction geometry.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{fig_dilemma_shared_component.pdf}
\caption{Distribution of pairwise cosine similarities between dilemma directions at layer 13, split by whether the pairs share a component foundation. Shared-component pairs (blue, $n = 60$) have higher mean similarity (0.269) than non-sharing pairs (red, $n = 45$, mean 0.195).}
\label{fig:shared_component}
\end{figure}

**Hierarchical clustering.** The 21-direction dendrogram (6
foundation + 15 dilemma directions) at layer 13 shows two
features. First, the six foundation directions form a distinct
cluster separate from the dilemma directions. Second, within the
dilemma cluster, pairs sharing a component foundation tend to merge
at lower distances, consistent with the shared-component analysis
above.

The categorical foundation/dilemma separation initially appears to
undercut the compositionality narrative: if dilemmas were purely
compositional, they would cluster near their component foundations,
not in a separate region. However, this separation is driven by
register features, not moral content. Projecting all 21 directions
into the 5D moral subspace (spanned by the six foundation
directions) and re-clustering dissolves the separation entirely:
foundations now cluster with their related dilemmas (e.g., sanctity
with fairness--sanctity and care--sanctity). The first-order
separation in the full-space dendrogram reflects the ${\sim}90$\%
extra-moral residual (likely text-register differences between
declarative single-foundation sentences and narrative dilemma
scenarios), while the second-order structure within each cluster
reflects genuine moral content relationships.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_dilemma_dendrogram.pdf}
\caption{Hierarchical clustering of all 21 directions (6 foundation in blue, 15 dilemma in red) at layer 13. Foundation directions cluster separately from dilemma directions.}
\label{fig:dilemma_dendrogram}
\end{figure}

**Full moral subspace projection.** The 2D subspace analysis above
uses only the two component foundations per dilemma. To test whether
dilemma representations are compositional over the *full* moral
vocabulary, we project each dilemma direction onto the 5D subspace
spanned by all six foundation directions (5D because the six
directions have effective dimensionality 5). Averaging across all
layers and pairs, the mean 5D membership is **0.093**, only
modestly above the layer-averaged 2D membership of 0.078 (ratio
1.19$\times$). (The higher value of 0.099 reported above is the
mean of per-pair *peak* values; 0.078 is the cross-layer average,
used here for an apples-to-apples comparison with the 5D
statistic.) Both exceed their respective null baselines (5D:
38$\times$ null; 2D: 81$\times$ null), confirming genuine moral
content in the dilemma representations. However, the small gain from
2D to 5D indicates that the ${\sim}90$\% residual is not explained
by *any* foundation direction: it is genuinely extra-moral, likely
encoding conflict-specific features such as trade-off framing and
tension that lie outside the moral subspace entirely.

**Scale replication.** The partial-compositionality result reproduces
on OLMo-2~7B. Mean peak subspace membership is 0.091 (9.1\%, vs.\ 9.9\%
at 1B), with all 15 pairs exceeding the null 95th percentile; mean
residual norm is 0.953 (vs.\ 0.949); and component balance is 0.455
(vs.\ 0.486, where 0.5 is perfect balance). The 7B model decomposes
moral dilemmas the same way the 1B model does: a small, significant
projection onto the component foundations, a large conflict-specific
residual, and near-balanced loading on the two foundations in tension.

## 4.10 Dilemma direction stability

Bootstrap resampling (50 iterations) of the 15 dilemma probe
directions across 16 layers yields 240 direction--layer
combinations. Of these, **239 are stable** (mean cosine with
full-data direction $> 0.7$), with the single exception at
liberty--loyalty, layer 0. The relaxed threshold (0.7 vs.\ 0.8 for
foundation probes) reflects the smaller sample size (16 training
pairs vs.\ 32 for foundations), but the results indicate that the
dilemma probe directions, and therefore the subspace analysis
built on them, are reliable.

## 4.11 Complexity--fragility gradient

\begin{figure}[t]
\centering
\includegraphics[width=0.7\linewidth]{fig_dilemma_fragility_gradient.pdf}
\caption{Mean critical noise for probes at three complexity levels. Single-foundation probes (Exp.~7) are most robust; per-type dilemma probes are least robust. Higher values indicate greater noise tolerance.}
\label{fig:fragility_gradient}
\end{figure}

We compare fragility across three levels of moral complexity:
single-foundation probes (from Experiment 7), pooled binary
dilemma probes (all 300 dilemma pairs pooled), and per-type
dilemma probes (15 separate probes, 20 pairs each).

All three are measured under the same convention (seed-averaged,
cap-at-max). The mean critical noise follows a
**complexity--fragility gradient**: single-foundation probes are most
robust ($\sigma^* = 5.02$), followed by the pooled dilemma probe
($\sigma^* = 3.81$), with per-type dilemma probes least robust
($\sigma^* = 3.55$). This
ordering is consistent with the hypothesis that more specific moral
distinctions are encoded with less redundancy and are therefore more
vulnerable to perturbation.

The gradient's direction matters: the pooled dilemma probe is
*less* robust than single-foundation probes, not more. This argues
against the possibility that dilemma probes are simply detecting a
generic "morally complex" feature with high redundancy. Instead, the
dilemma direction captures genuinely more specific information that
is correspondingly more fragile.

## 4.12 MoE architecture preserves compositionality

The dilemma probing and subspace analysis on OLMoE-1B-7B produces
results consistent with the dense model. Mean peak accuracy is
95.8\% (vs.\ 94.2\% on OLMo-2), and mean peak subspace membership
is **0.092** (vs.\ 0.099). The ${\sim}7$\% difference in membership
scores is within the variation across individual foundation pairs.
The partial compositionality structure (statistically significant
but low absolute membership, with high residual) is a property of
the representation geometry, not a dense-architecture artifact.

## 4.13 Robustness to direction-finding method

The geometric findings reported above rely on probe weight vectors
as foundation directions. To test whether the geometry is an
artifact of discriminative probe training, we extract directions
using two alternative methods and compare.

**Mean-difference directions.** For each foundation, we compute
$\mathbf{d}_f = \overline{\mathbf{a}}_{\text{moral}} -
\overline{\mathbf{a}}_{\text{neutral}}$ (the normalized difference
of class-conditional activation means), requiring no optimization.
These training-free directions replicate the core geometric
findings: effective dimensionality is 5 at all layers. Like the
probe-weight directions, the mean-diff dendrogram does not recover
the MFT individualizing/binding split at any layer, and the
permutation test is non-significant throughout. Per-foundation
cosine similarity between probe-weight and mean-difference
directions ranges from 0.67 to 0.72 (mean across layers),
indicating related but non-identical directions: the probe-weight
method finds more foundation-specific discriminative signal, while
the mean-difference method captures more of the shared
moral-salience component (mean pairwise cosine 0.41 vs.\ 0.22 at
layer 0).

**Representation-engineering directions.** We also test
paired-difference PCA \citep{zou2023representation}: for each
pair~$i$, compute $\mathbf{d}_i = \mathbf{a}_{\text{moral},i} -
\mathbf{a}_{\text{neutral},i}$ and take the first principal
component. This method performs poorly: the first PC explains only
8--11\% of variance (barely above chance in $\mathbb{R}^{2048}$),
and the resulting directions show low alignment with probe-weight
directions ($|\cos| = 0.07$--$0.26$) and weak classification
accuracy. With ${\sim}32$ pairs per foundation in 2048 dimensions,
the $p \gg n$ regime prevents PCA from isolating the concept
direction. This negative result validates that the convergent
probe-weight and mean-difference findings are not trivially
recoverable; they depend on direction-finding methods with
appropriate inductive bias for small datasets.

**Cross-register transfer.** Both probe-weight and mean-difference
directions transfer to narrative dilemma text with $>90$\% mean
pair accuracy across foundations (the fraction of pairs where the
direction projects the moral text higher than the neutral text).
Probe-weight directions achieve $>95$\% for all foundations; the
mean-difference method drops to ${\sim}91$\% for authority/subversion,
the foundation with weakest directional stability.  The transfer
gap between same-register (declarative test pairs) and cross-register
(dilemma pairs) averages under 4 percentage points, with
authority/subversion showing the largest gap (${\sim}9$ pp for
mean-difference; §5.5).

## 4.14 Scale validation: geometry persists from 1B to 7B

We replicate the core geometric findings on OLMo-2~7B (the 1124
release, 32 layers, hidden dimension 4096), a dense model with 7.3B
parameters, versus 1.5B for OLMo-2~1B. The integration signature holds.
Mean pairwise cosine similarity ranges from **0.193 to 0.244** across
layers (1B: 0.217--0.282), positive everywhere and far below the
collapse threshold. Effective dimensionality is **5 at every one of
the 32 layers**, identical to the 1B and MoE models. Foundation
directions are geometrically distinct, sharing a common moral-salience
component, at both scales.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{scale_comparison_cosine_heatmap.pdf}
\caption{Foundation direction cosine similarity for OLMo-2 1B (layer 7/16) and 7B (layer 14/32), at matched relative depth. Mean off-diagonal cosine is 0.262 (1B) and 0.240 (7B) at these display layers; both sit in the integration range. Effective dimensionality is 5 for both (Figure~\ref{fig:scale_overlay}).}
\label{fig:scale_heatmap}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{scale_comparison_geometry_overlay.pdf}
\caption{Layer-wise geometry across scale on normalized depth. (a) Mean pairwise cosine for both models stays in the integration band. (b) Effective dimensionality is constant at 5 for both 1B and 7B.}
\label{fig:scale_overlay}
\end{figure}

**Direction stability.** Bootstrap resampling (200 iterations) places
110 of 192 foundation--layer combinations above the 0.8 stability
threshold, with the remainder in the 0.72--0.80 borderline band.
Sanctity/degradation is again the most stable foundation (29/32
layers), fairness/cheating the least (7/32). The per-layer cosine
range (0.72--0.85) is comparable to the 1B model; the larger model's
directions are no less determined.

**MFT is not recovered at 7B either.** Hierarchical clustering does
not produce the individualizing/binding split at any layer, and the
permutation test for MFT group structure is non-significant throughout
($p = 0.49$--$0.78$). The model's inter-framework geometry is not
MFT-aligned at either scale.

**Fragility.** Seed-averaged per-foundation fragility at 7B uses the
extended noise grid (§4.7), which censors only 1 of 192 foundation--layer
cells at the cap. As at 1B, the per-foundation $\sigma^*$ values have
wide, overlapping bootstrap CIs and no foundation is reliably most or
least robust (7B mean $\sigma^*$: loyalty $14.7$, authority $13.5$,
care $10.1$, sanctity $10.0$, liberty $8.4$, fairness $6.9$, all with
overlapping CIs). The binding-vs-individualizing group difference is again
not significant ($+4.2$, exact $p = 0.20$). What the 7B run does add is
a clean scale effect: the dense model is uniformly more robust at 7B
than at 1B (mean $\sigma^*$ ${\approx}10.6$ vs.\ $5.0$). The robust
fragility findings are therefore the cross-architecture dilution (§4.7)
and this dense scale effect, not any per-foundation ordering.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{scale_comparison_fragility.pdf}
\caption{Per-foundation mean critical noise $\sigma^*$ for OLMo-2 1B and 7B (dense), seed-averaged with bootstrap 95\% CIs. The 7B model is uniformly more robust (scale effect); within each model the per-foundation CIs overlap and no foundation is reliably most robust.}
\label{fig:scale_fragility}
\end{figure}

## 4.15 The model's data-driven taxonomy does not recover MFT

Sections 4.3 and 4.14 show that hierarchical clustering of the six
probe *directions* does not recover MFT groups. We test the stronger
question directly: cluster the moral-positive *activations* themselves,
without reference to foundation labels, and ask whether the discovered
groups align with the foundations.

K-means and spectral clustering ($k$ selected by silhouette score,
$k = 2$--$8$) on OLMo-2~7B moral activations produce clusters that
barely align with MFT foundations. Adjusted mutual information between
the discovered clusters and the foundation labels peaks at **0.032**
(layer 11) and is near zero or slightly negative at most layers;
silhouette scores stay low (0.06--0.08), indicating weak cluster
structure. A clustering that recovered the foundations would give AMI
near 1.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{taxonomy_confusion_7B.pdf}
\caption{Discovered clusters (rows) against MFT foundations (columns) at the most foundation-aligned layer (11) of OLMo-2 7B. Cells are pair counts; row-normalized shading. The clusters do not map onto foundations (AMI = 0.032).}
\label{fig:taxonomy_confusion}
\end{figure}

This is consistent with integration. Foundations are linearly
*decodable*: supervised probes separate them with near-perfect
accuracy and span 5 effective dimensions. But they are not the
dominant axis of variation in the raw activations, so unsupervised
clustering recovers the shared moral-salience structure rather than
the foundation distinctions. The model represents the foundations as
distinct directions without organizing its moral activations around
them.

## 4.16 External replication on the Moral Foundations Vignettes

To test whether the integration geometry is a property of the model
rather than of our probing dataset, we recompute foundation directions
from the Moral Foundations Vignettes \citep{clifford2015mfv}, an
independently constructed set of 30 validated moral scenarios (5 per
foundation). Directions are estimated by mean difference against a
shared neutral baseline, the same estimator applied to both datasets
for a fair comparison.

The integration signature replicates. The six MFV directions span
**5 effective dimensions at every layer** of OLMo-2~7B, identical to
the directions derived from our dataset. Mean pairwise cosine is
higher for MFV (0.50--0.72) than for the matched mean-difference
directions on our data (0.42 at the display layer), an expected
small-sample effect given five vignettes per foundation. Because
absolute cosine is inflated by the five-per-foundation sample, the
cross-dataset claim rests on the sign and structure of the geometry
rather than the cosine magnitude: both datasets show uniformly
positive mean cosine with variance concentrated on a shared leading
axis (the integration signature), spanning the same 5 effective
dimensions. Effective dimensionality by itself does not separate
integration from isolation (random directions also span ${\sim}5$
dimensions); the positive mean cosine carries that distinction, and
it replicates in sign across the two datasets.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{external_robustness_7B.pdf}
\caption{MFV external replication on OLMo-2 7B. (a) Cosine similarity among the six MFV foundation directions. (b) Mean cosine and effective dimensionality vs depth for MFV and for matched mean-difference directions on our dataset. (c) Per-foundation alignment between MFV and our directions.}
\label{fig:external}
\end{figure}

**Cross-dataset direction alignment.** Foundation directions estimated
from the two independent datasets are positively aligned ($|\cos| =
0.14$--$0.43$ per foundation at layer 12), with individualizing
foundations aligning more strongly (care 0.30, fairness 0.36, liberty
0.43) than binding foundations (loyalty 0.14, authority 0.19, sanctity
0.29). That directions estimated from independently authored stimuli
agree at all is evidence that the foundation directions reflect the
model's representations rather than artifacts of any single dataset.
