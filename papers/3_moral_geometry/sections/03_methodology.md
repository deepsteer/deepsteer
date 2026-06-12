# 3. Methodology

We train six foundation-specific linear probes at each transformer
layer, extract the learned weight vectors as geometric directions in
representation space, and analyze the angular relationships between
these directions. The methodology decomposes into five components:
foundation-specific probing (§3.2), geometric analysis of probe
directions (§3.3), bootstrap direction stability assessment (§3.4),
framework-specific fragility testing (§3.5), and the probing
dataset (§3.6). All experiments run on a single MacBook Pro M4 Pro
(24 GB unified memory, MPS backend).

## 3.1 Models and comparison design

We evaluate three base (non-instruct) models from the same lab
(Ai2). The primary comparison is between two architectures matched in
layer count (16), dense vs. mixture-of-experts; a third, larger dense
model serves as a scale control (§4.14):

- **OLMo-2 1B** (`allenai/OLMo-2-0425-1B`): dense transformer,
  1.5B parameters, 2048 hidden dimension, 16 layers \citep{olmo2_2025}.
  Ai2 publishes 37 early-training checkpoints at 1K-step intervals
  (steps 0--36K), enabling trajectory analysis.

- **OLMoE-1B-7B** (`allenai/OLMoE-1B-7B-0924`): mixture-of-experts,
  6.9B total parameters, 1.3B active, 64 experts per layer, top-8
  routing, 2048 hidden dimension, 16 layers \citep{muennighoff2024olmoe}.

- **OLMo-2 7B** (`allenai/OLMo-2-1124-7B`): dense transformer, 7.3B
  parameters, 4096 hidden dimension, 32 layers \citep{olmo2_2025}. Used
  in §4.14 to test whether the geometry findings persist with scale.

All three models use the same tokenizer and comparable training corpora.
The comparison tests whether framework geometry is an artifact of
dense connectivity or a general property of transformer-based
language modeling. \citet{reblitzrichardson2026dilution} established
that moral encoding in OLMoE is uniform across experts with a
74$\times$ output scale gap relative to OLMo-2; the present work
asks whether this output dilution affects the *structure* of moral
representations, not just their *scale*.

## 3.2 Foundation-specific probing

For each of the six Moral Foundations Theory (MFT) foundations
(care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression;
\citep{haidt2012righteous, graham2013mft}), we train a binary
linear probe at each of 16 transformer layers.

**Probe architecture.** Each probe is `nn.Linear(2048, 1)` trained
with BCE loss and Adam (lr = $10^{-2}$) for 50 epochs. This
matches the probe specification from
\citet{reblitzrichardson2026fragility} and
\citet{reblitzrichardson2026dilution}, enabling direct comparison
with their binary moral/neutral probes.

**Activation collection.** For each text, we run a single forward
pass capturing all 16 layers simultaneously. At each layer, we
mean-pool across the sequence dimension to obtain a single
$\mathbb{R}^{2048}$ vector per text. Positive examples are the
foundation-tagged moral sentences; negative examples are their
matched neutral counterparts.

**Direction extraction.** After training, we extract the weight
vector $\mathbf{w} \in \mathbb{R}^{2048}$ from each probe and
normalize to unit length: $\hat{\mathbf{w}} = \mathbf{w} /
\|\mathbf{w}\|$. This unit vector is the normal to the
classification hyperplane, the *direction* in representation
space that maximally separates the foundation's moral content from
neutral content. We call $\hat{\mathbf{w}}$ the foundation's
**probe direction** at a given layer.

This yields $6 \times 16 = 96$ unit-norm probe direction vectors
per model.

## 3.3 Geometric analysis

The paper's core contribution is analyzing the angular relationships
between the six foundation probe directions at each layer.

**Cosine similarity matrices.** At each layer, we compute the
$6 \times 6$ pairwise cosine similarity matrix of the foundation
probe directions. Because the directions are unit-normalized,
$\cos(\hat{\mathbf{w}}_i, \hat{\mathbf{w}}_j) =
\hat{\mathbf{w}}_i \cdot \hat{\mathbf{w}}_j$.

**Geometric signatures.** Three qualitative modes of moral
representation correspond to distinct cosine similarity patterns:

1. *Collapse (averaging).* All foundation directions converge
   toward a single "moral salience" direction. Mean pairwise
   cosine similarity $\to 1$.

2. *Isolation.* Foundation directions are orthogonal with no
   relational structure. Mean pairwise cosine similarity $\to 0$.

3. *Integration.* Foundation directions are separated but
   non-orthogonal, with inter-framework geometry reflecting known
   relationships. Mean pairwise cosine similarity in $(0, 1)$ with
   structured variation.

**Effective dimensionality.** We compute the effective
dimensionality of the 6-direction set at each layer via PCA on
the $6 \times 2048$ matrix of probe directions. Effective
dimensionality is the number of principal components explaining
$\geq 90\%$ of variance. Low dimensionality (1--2) indicates
collapse; high dimensionality (5--6) indicates separation.

**Hierarchical clustering.** We apply Ward's method to the cosine
distance matrix ($1 - \cos(\cdot, \cdot)$) at each layer.
Dendrograms reveal whether the six foundations cluster into
interpretable groups.

**Permutation test for MFT group structure.** MFT predicts that
the six foundations divide into *individualizing* (care, fairness,
liberty) and *binding* (loyalty, authority, sanctity) clusters
\citep{graham2013mft}. We test this prediction with a permutation
test: compute the observed difference between mean within-group
cosine similarity and mean between-group cosine similarity, then
permute group assignments 10,000 times to generate the null
distribution.

## 3.4 Bootstrap direction stability

With 32 training pairs per foundation in 2048 dimensions, the probe
has 2049 parameters and only 64 training examples. The classification
*accuracy* may be robust in this regime (a single hyperplane suffices
for binary separation), but the extracted *direction* could be noisy
, and noise in directions contaminates the angular analysis.

We assess direction stability via bootstrap resampling: for each
foundation at each layer, we resample the 32 training pairs with
replacement 200 times, retrain the probe on each bootstrap sample,
and compute the cosine similarity of each bootstrap direction with
the full-data direction. A mean bootstrap cosine similarity $> 0.8$
indicates a stable direction.

This gives a per-layer, per-foundation reliability metric for the
geometric analysis. Layers where directions are unstable should be
interpreted with caution.

## 3.5 Framework-specific fragility

We extend the fragility protocol of
\citet{reblitzrichardson2026fragility} to per-foundation probes.
For each foundation at each layer:

1. Train a linear probe on clean activations.
2. Evaluate on clean test activations (baseline accuracy).
3. For each noise level $\sigma \in \{0.1, 0.3, 1.0, 3.0, 10.0\}$,
   add $\mathcal{N}(0, \sigma^2)$ noise to cached test activations
   and re-evaluate.
4. The **critical noise** $\sigma^*$ is the smallest $\sigma$ where
   accuracy drops below 0.6.

This tests the universality hypothesis: more cross-culturally
universal foundations (care/harm) should be more robustly encoded
than culturally variable foundations (sanctity/degradation,
loyalty/betrayal). Cross-architectural comparison tests whether
the output dilution effect \citep{reblitzrichardson2026dilution}
affects all foundations uniformly.

## 3.6 Probing dataset

We use the same 240-pair minimal-pair probing dataset from
\citet{reblitzrichardson2026fragility}, now used at the
per-foundation level: 40 pairs per MFT foundation, stratified
80/20 into 32 train and 8 test pairs per foundation. Each pair
consists of a moral sentence tagged with its MFT foundation and a
matched neutral sentence controlling for sentence length, syntactic
structure, and topic. The dataset was drawn from a 1{,}200-pair
candidate pool generated by Claude Sonnet 4.6 from hand-written
seed examples, with automated validation gates (length ratio $\leq
1.5$, embedding similarity, keyword scan, deduplication) and
LLM-as-judge filtering for neutral-pair quality.

For the geometric analysis, the 32 training pairs per foundation
yield $64$ training examples (moral + neutral) for the
`nn.Linear(2048, 1)` probe. While this is a small sample for a
2049-parameter model, the bootstrap stability analysis (§3.4)
directly quantifies whether the resulting directions are reliable.

## 3.7 Dilemma compositionality analysis

Experiments 1--7 establish that the model maintains distinct
directions for each moral foundation. A natural follow-up: when
two foundations *conflict* in a moral dilemma, does the model
represent the dilemma as a composition of its component foundation
directions, or does it develop a qualitatively new representation?

**Dilemma dataset.** We generate 300 moral dilemma scenarios (20
per each of the $\binom{6}{2} = 15$ foundation pairs) using
Claude Sonnet 4.6, with hand-written seed examples per pair. Each
scenario pits two specific foundations against each other (e.g.,
care vs.\ authority: "The nurse administered an unapproved
painkiller to a dying patient because following protocol meant
hours more agony"). Each dilemma text is paired with a matched
neutral sentence. All pairs pass automated validation gates
(length ratio, keyword scan, deduplication).

**Dilemma-specific probes.** For each of the 15 foundation pairs,
we train a binary linear probe (same architecture as §3.2) to
distinguish dilemma moral text from matched neutral text. The
probe weight vector $\hat{\mathbf{w}}_{\text{dilemma}}$ is the
direction in representation space that separates the dilemma
content from neutral content.

**Subspace membership score.** To measure compositionality, we
project each dilemma direction onto the 2D subspace spanned by its
two component foundation directions. Given component directions
$\hat{\mathbf{w}}_A$ and $\hat{\mathbf{w}}_B$ from Experiment 1,
we orthogonalize them via Gram--Schmidt and compute the fraction
of the dilemma direction's variance explained by this 2D subspace:

$$S = \|\text{proj}_{\text{span}(\hat{\mathbf{w}}_A, \hat{\mathbf{w}}_B)} \hat{\mathbf{w}}_{\text{dilemma}}\|^2$$

A membership score of $S = 1$ indicates full compositionality
(the dilemma direction lies entirely within the component
subspace); $S = 0$ indicates complete independence. The null
baseline for a random unit vector in $\mathbb{R}^{2048}$ projected
onto a random 2D subspace has expected membership $2/2048 \approx
0.001$. We estimate the empirical null distribution from 10{,}000
random unit vectors.

**Component balance.** We decompose the within-subspace projection
into components along $\hat{\mathbf{w}}_A$ and the
Gram--Schmidt-orthogonalized $\hat{\mathbf{w}}_B$. The balance
ratio (fraction of the projection along the first component)
measures whether the dilemma direction is dominated by one
foundation or draws equally on both. A ratio near 0.5 indicates
balanced composition.

**Shared-component geometry.** If dilemma representations are
partially compositional, dilemma pairs that share a foundation
component should have more similar probe directions than pairs
with no shared foundation. We test this by comparing the mean
cosine similarity between dilemma directions for pairs that share
a component (e.g., care--fairness and care--loyalty, which share
care) versus pairs with no overlap (e.g., care--fairness and
loyalty--sanctity).

**Cross-architecture consistency.** We repeat the dilemma probing
and subspace analysis on OLMoE-1B-7B to test whether the
compositionality structure is architecture-specific or general.
