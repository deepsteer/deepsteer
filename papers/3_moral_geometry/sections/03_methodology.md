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

We evaluate two base (non-instruct) models from the same lab
(Ai2), matched in layer count (16) but differing in architecture:

- **OLMo-2 1B** (`allenai/OLMo-2-0425-1B`): dense transformer,
  1.5B parameters, 2048 hidden dimension \citep{olmo2_2025}. Ai2
  publishes 37 early-training checkpoints at 1K-step intervals
  (steps 0--36K), enabling trajectory analysis.

- **OLMoE-1B-7B** (`allenai/OLMoE-1B-7B-0924`): mixture-of-experts,
  6.9B total parameters, 1.3B active, 64 experts per layer, top-8
  routing, 2048 hidden dimension \citep{muennighoff2024olmoe}.

Both models use the same tokenizer and comparable training corpora.
The comparison tests whether framework geometry is an artifact of
dense connectivity or a general property of transformer-based
language modeling. \citet{reblitzrichardson2026dilution} established
that moral encoding in OLMoE is uniform across experts with a
77$\times$ output scale gap relative to OLMo-2; the present work
asks whether this output dilution affects the *structure* of moral
representations, not just their *scale*.

## 3.2 Foundation-specific probing

For each of the six Moral Foundations Theory (MFT) foundations ---
care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression
\citep{haidt2012righteous, graham2013mft} --- we train a binary
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
classification hyperplane --- the *direction* in representation
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
meaningful groups.

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
--- and noise in directions contaminates the angular analysis.

We assess direction stability via bootstrap resampling: for each
foundation at each layer, we resample the 32 training pairs with
replacement 200 times, retrain the probe on each bootstrap sample,
and compute the cosine similarity of each bootstrap direction with
the full-data direction. A mean bootstrap cosine similarity $> 0.8$
indicates a stable direction.

This provides a per-layer, per-foundation reliability metric for the
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
loyalty/betrayal). Cross-architectural comparison reveals whether
the output dilution effect \citep{reblitzrichardson2026dilution}
affects all foundations uniformly.

## 3.6 Probing dataset

We use the same 240-pair minimal-pair probing dataset from
\citet{reblitzrichardson2026fragility}, now leveraged at the
per-foundation level: 40 pairs per MFT foundation, stratified
80/20 into 32 train and 8 test pairs per foundation. Each pair
consists of a moral sentence tagged with its MFT foundation and a
matched neutral sentence controlling for sentence length, syntactic
structure, and topic. All pairs pass automated validation gates
(length ratio $\leq 1.5$, keyword scan, deduplication).

For the geometric analysis, the 32 training pairs per foundation
yield $64$ training examples (moral + neutral) for the
`nn.Linear(2048, 1)` probe. While this is a small sample for a
2049-parameter model, the bootstrap stability analysis (§3.4)
directly quantifies whether the resulting directions are reliable.
