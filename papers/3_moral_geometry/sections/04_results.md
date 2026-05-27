# 4. Results

## 4.1 Foundation-specific probe accuracy

All six foundation-specific probes achieve near-perfect accuracy
across the full depth of OLMo-2 1B. Care/harm and fairness/cheating
reach 100\% at nearly every layer; sanctity/degradation reaches
100\% at 12 of 16 layers. Liberty/oppression, loyalty/betrayal, and
authority/subversion show minor fluctuations (minimum 87.5\%, at
layers 3 and 15 for loyalty and layers 6 and 13--15 for authority).
The onset threshold (0.6) is exceeded at layer 0 for all
foundations.

These results confirm that moral foundation content is linearly
separable from the earliest layer --- consistent with the "immediate
onset" finding of \citet{reblitzrichardson2026fragility} for the
pooled binary moral/neutral probe. The per-foundation decomposition
reveals no qualitative difference in *detectability* across
foundations: all are equally easy to decode. The interesting
variation is not in accuracy but in the *geometry* of the probe
directions that achieve this accuracy.

On OLMoE-1B-7B, all foundations similarly reach near-perfect
accuracy, with authority/subversion showing the lowest peak (93.8%
at layer 0). The accuracy profiles are essentially indistinguishable
between architectures.

## 4.2 Framework geometry: integration, not collapse

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig1_cosine_heatmap.pdf}
\caption{Pairwise cosine similarity between foundation probe directions at layer 0 (peak separation), OLMo-2 1B. Individualizing foundations (care, fairness, liberty) show higher within-group similarity than between-group pairs.}
\label{fig:cosine_heatmap}
\end{figure}

The headline finding: foundation probe directions are *separated*,
not collapsed. At the peak separation layer (layer 0), the mean
pairwise cosine similarity between the six foundation directions is
**0.272** --- far below the collapse threshold ($>0.95$) and below
the intermediate zone ($0.8$--$0.95$). The model does not encode
moral content through a single "moral salience" direction; it
maintains distinct directions for distinct moral foundations.

However, the cosine similarities are uniformly *positive* (range
0.18--0.37 at layer 0), indicating that the foundation directions
share a common component. This is the *integration* signature from
our trichotomy: the directions are separated but non-orthogonal,
consistent with a shared moral-salience subspace from which
foundation-specific directions deviate.

**Effective dimensionality.** The six foundation directions span
5 effective dimensions (the number of PCs explaining $\geq 90\%$
of variance) at every layer. This is near the maximum possible for
6 directions, confirming that the directions are geometrically
distinct --- they do not collapse into a lower-dimensional subspace.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig2_layerwise_geometry.pdf}
\caption{Layer-wise geometric metrics for OLMo-2 1B. (a) Mean pairwise cosine similarity peaks at middle layers. (b) Effective dimensionality remains constant at 5 across all layers.}
\label{fig:layerwise}
\end{figure}

## 4.3 MFT group structure in the dendrogram

\begin{figure}[t]
\centering
\includegraphics[width=0.7\linewidth]{fig3_dendrogram.pdf}
\caption{Hierarchical clustering (Ward's method) of foundation probe directions at layer 0, OLMo-2 1B. The first split perfectly recovers the MFT individualizing/binding distinction.}
\label{fig:dendrogram}
\end{figure}

Hierarchical clustering of the six foundation directions at layer 0
produces a dendrogram that perfectly recovers the MFT
individualizing/binding distinction: the first split separates
\{liberty, care, fairness\} from \{sanctity, loyalty, authority\}.
Within the individualizing cluster, care and fairness merge first;
within the binding cluster, loyalty and authority merge first.

This alignment with MFT's predicted group structure is notable:
the model's representation geometry mirrors a theoretical
distinction from moral psychology that was not an explicit training
objective. The probe directions were trained independently for each
foundation; the clustering structure emerges from the representations
themselves.

The permutation test for the individualizing/binding distinction
reaches significance at layer 0 ($p = 0.012$) and layer 3
($p = 0.035$), though not at most other layers (median $p = 0.23$).
With only 6 items and 20 unique 3--3 partitions, statistical power
is limited. The significant result at layer 0 corroborates the
dendrogram structure; the layer-to-layer variation in $p$-values
reflects the sensitivity of the scalar within/between summary
statistic to small changes in pairwise cosines.

## 4.4 Layer-wise geometric development

The geometric structure evolves across layers in a characteristic
pattern. Mean pairwise cosine similarity is lowest at layer 0
(0.272), rises to a peak of 0.366 at layer 10, then gradually
decreases through the remaining layers (0.306 at layer 15). This
inverted-U pattern indicates that middle layers develop *more*
shared structure between foundation directions (partial collapse),
while early and late layers maintain greater separation.

Effective dimensionality remains constant at 5 across all layers,
meaning the rank of the direction set does not change even as the
pairwise angles shift. The geometric development is a rotation of
the direction set within a fixed-rank subspace, not a
dimensionality change.

## 4.5 Dense vs. MoE framework geometry

Framework geometry is remarkably similar between architectures.
OLMoE-1B-7B shows mean pairwise cosine similarity ranging from
0.260 to 0.350, compared to OLMo-2 1B's 0.275 to 0.363. Effective
dimensionality is 5 at all layers for both models. The overall
degree of framework separation is consistent across dense and MoE
architectures in our comparison.

The dendrogram structure diverges at specific layers. At layer 7,
OLMoE produces a *perfect* MFT split: \{liberty, care, fairness\}
vs. \{loyalty, authority, sanctity\}. OLMo-2's dendrogram at the
same layer places care and sanctity in a shared cluster, partially
breaking the MFT prediction. The MFT group structure appears at
*some* layer in both architectures, but the specific layer varies.

This finding extends \citet{reblitzrichardson2026dilution}: output
dilution affects moral encoding *scale* (77$\times$ signal gap) but
not framework *structure*. The geometric organization of moral
foundations is preserved across architectures.

## 4.6 Direction stability under bootstrap

Bootstrap resampling (200 iterations) reveals a stability gradient
across layers. Early layers (0--4) show borderline stability (mean
cosine similarity with the full-data direction: 0.73--0.80, below
the 0.8 threshold). Middle and late layers (5--14) are stable
(mean cosine $> 0.80$). Care/harm is the most stable foundation at
every layer.

This gradient has two implications. First, the geometric analysis
at early layers --- including the peak separation layer (layer 0)
--- should be interpreted with caution, as the specific pairwise
cosine values may shift under resampling. Second, the stability
gradient itself is informative: probe directions become more
determined as representations become more specialized, paralleling
the lexical-to-compositional gradient from
\citet{reblitzrichardson2026fragility}.

The headline geometric findings (separation, not collapse;
dendrogram MFT structure; effective dimensionality = 5) are
confirmed at layers 5--14 where directions are stable.

## 4.7 Differential fragility across frameworks

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{exp7_mean_critical_bars.pdf}
\caption{Mean critical noise per foundation for OLMo-2 1B (left) and OLMoE-1B-7B (right). Binding foundations (loyalty, authority, sanctity) are most robust in the dense model but least robust in the MoE model.}
\label{fig:fragility}
\end{figure}

Per-foundation fragility reveals a striking cross-architectural
pattern.

**OLMo-2 1B (dense).** Binding foundations are the *most robust*:
loyalty/betrayal has the highest mean critical noise (6.21),
followed by authority/subversion (5.53). Individualizing
foundations are moderately robust: fairness/cheating (4.43),
liberty/oppression (4.24), care/harm (4.07). Sanctity/degradation
is the most fragile (3.85). The universality hypothesis (care/harm
as most robust) is *not* supported.

**OLMoE-1B-7B (MoE).** The pattern reverses. Individualizing
foundations are more robust: care/harm (2.10), fairness/cheating
(1.96), liberty/oppression (1.90). Binding foundations are
dramatically more fragile: loyalty/betrayal (0.86),
sanctity/degradation (0.91), authority/subversion (0.64).

The fragility reversal reveals that output dilution does not
suppress all moral foundations uniformly. Binding foundations ---
which \citet{graham2013mft} characterize as more culturally
variable and group-oriented --- lose proportionally more robustness
under the MoE output scale gap. In OLMo-2, binding foundations
show a mean critical noise of 5.24 vs. 4.25 for individualizing
(ratio 1.23$\times$). In OLMoE, the ratio inverts: individualizing
mean 1.98 vs. binding mean 0.80 (ratio 2.48$\times$ in the opposite
direction). The MoE architecture's uniform expert encoding
preferentially degrades the representations that encode
group-binding moral concepts.

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

- **Step 0.** Mean cosine similarity is 0.007 --- the six
  foundation directions are effectively orthogonal, as expected for
  probes trained on random representations. Mean accuracy is 0.510
  (chance).

- **Step 1000.** Cosine similarity jumps to 0.176. Accuracy reaches
  0.734. The model has already begun developing shared moral-salience
  structure.

- **Step 2000.** Cosine similarity reaches 0.382 --- within 5\% of
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

The temporal dissociation --- framework geometry stabilizing at step
2000 while accuracy continues improving through step 25,000
--- extends the two-phase pattern identified by
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
--- scenarios where two foundations conflict --- can be decomposed in
terms of the single-foundation directions from Experiment 1.

**Dilemma probes achieve high accuracy.** All 15 dilemma-specific
probes achieve $\geq 75\%$ peak test accuracy (mean 94.2\%), with
13 of 15 pairs at $\geq 87.5\%$. The model reliably distinguishes
dilemma moral content from matched neutral text.

**Subspace membership: partial compositionality.** The mean peak
subspace membership score across all 15 pairs is **0.099** ---
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
where the sanctity component dominates. This suggests that when
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
foundation + 15 dilemma directions) at layer 13 reveals two
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
extra-moral residual --- likely text-register differences between
declarative single-foundation sentences and narrative dilemma
scenarios --- while the second-order structure within each cluster
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
directions have effective dimensionality 5). The mean 5D membership
is **0.093** --- only modestly above the 2D membership of 0.078
(ratio 1.19$\times$). Both exceed their respective null baselines
(5D: 38$\times$ null; 2D: 81$\times$ null), confirming genuine
moral content in the dilemma representations. However, the small
gain from 2D to 5D indicates that the ${\sim}90$\% residual is not
explained by *any* foundation direction: it is genuinely extra-moral,
likely encoding conflict-specific features such as trade-off framing
and tension that lie outside the moral subspace entirely.

## 4.10 Dilemma direction stability

Bootstrap resampling (50 iterations) of the 15 dilemma probe
directions across 16 layers yields 240 direction--layer
combinations. Of these, **239 are stable** (mean cosine with
full-data direction $> 0.7$), with the single exception at
liberty--loyalty, layer 0. The relaxed threshold (0.7 vs.\ 0.8 for
foundation probes) reflects the smaller sample size (16 training
pairs vs.\ 32 for foundations), but the results indicate that the
dilemma probe directions --- and therefore the subspace analysis
built on them --- are reliable.

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

The mean critical noise follows a **complexity--fragility gradient**:
single-foundation probes are most robust ($\sigma^* = 4.72$),
followed by the pooled dilemma probe ($\sigma^* = 3.12$), with
per-type dilemma probes least robust ($\sigma^* = 2.90$). This
ordering is consistent with the hypothesis that more specific moral
distinctions are encoded with less redundancy and are therefore more
vulnerable to perturbation.

The gradient's direction is noteworthy: the pooled dilemma probe is
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
The partial compositionality structure --- statistically significant
but low absolute membership, with high residual --- is a property of
the representation geometry, not a dense-architecture artifact.
