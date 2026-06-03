# 3. Methods

<!-- ~2.5 pages. Three subsections matching the three validation modes. -->

## 3.1 Model and dataset

All experiments use OLMo-2-0425-1B (16 layers, 2048 hidden dim) with mean-difference directions extracted from the 240-pair moral probing dataset (40 pairs per MFT foundation, 192 train / 48 test).
Directions are normalized to unit length.
We use mean-difference directions rather than probe-weight directions because mean-difference directions capture more of the shared moral-salience component (mean pairwise cosine 0.41 vs.\ 0.22 for probe weights at layer 0; \citealt{reblitzrichardson2026geometry}, §4.13), making them better suited for causal interventions that target the full moral representation rather than the maximally discriminative direction.
The choice matters: SAE subspace overlap is $15.5\%$ for mean-difference vs.\ $8.2\%$ for probe-weight directions (§4.4.3).
See \citet{reblitzrichardson2026geometry} for dataset construction and direction extraction.

## 3.2 Causal validation methods

### 3.2.1 Direction ablation

For each foundation $f$ and layer $\ell$, we register a forward hook that projects out the foundation direction $\mathbf{d}_f^{(\ell)}$ from the hidden state:
$$\mathbf{h} \leftarrow \mathbf{h} - (\mathbf{h} \cdot \mathbf{d}_f^{(\ell)}) \, \mathbf{d}_f^{(\ell)}$$
We then measure the change in log-probability of target continuations from a 48-prompt causal evaluation set (8 prompts per foundation, with labeled target and off-target continuations).

**Metrics.** For each (ablated foundation, layer):

- *On-target $\Delta$*: mean log-prob change on prompts whose target foundation matches the ablated direction.
- *Off-target $\Delta$*: mean log-prob change on prompts from other foundations.
- *Specificity*: on-target $\Delta$ minus off-target $\Delta$. Negative specificity indicates that ablation specifically harms the target foundation.

### 3.2.2 Steering injection

For each foundation $f$, layer $\ell$, and amplitude $\alpha \in \{1, 2, 5, 10, 20\}$, we add a scaled direction to the hidden state:
$$\mathbf{h} \leftarrow \mathbf{h} + \alpha \, \mathbf{d}_f^{(\ell)}$$
and measure log-probability changes on the same evaluation set.
The dose--response curve across $\alpha$ values distinguishes genuine feature manipulation (specificity at low $\alpha$, saturation at high $\alpha$) from noise injection (monotonic degradation).

### 3.2.3 Causal evaluation prompts

The 48-prompt evaluation set comprises three formats: 24 completion prompts (4 per foundation) with a sentence stem and 3--4 candidate continuations, 12 forced-choice prompts (2 per foundation), and 12 natural prompts (2 per foundation).
Each prompt has labeled target and off-target continuations with foundation labels, enabling measurement of foundation-specific log-probability changes under ablation and injection.

**Prompt construction.** Prompts were hand-authored to activate specific MFT foundations.
Completion prompts use sentence stems where the target continuation is a single word or short phrase whose foundation loading is unambiguous (e.g., a care/harm stem ending in ``she felt compelled to'' with target ``help'' and off-target ``ignore'' and ``report'').
Forced-choice prompts present two continuations from different foundations.
Natural prompts use open-ended stems that implicitly prime a specific foundation.
All prompts were reviewed to ensure that (a) the target foundation is the most natural continuation, (b) off-target continuations span at least two other foundations, and (c) no prompt relies on surface lexical cues (the foundation name or its synonyms do not appear in the stem).
The full prompt set with all continuations and foundation labels is included in the code repository.

## 3.3 Behavioral grounding methods

### 3.3.1 Projection-based classification

For each input text, we collect the last-token residual stream activation at target layers (4, 8, 12), project onto all six foundation directions, and average across layers.
The foundation with the highest projection is the predicted label.
We also test a *debiased* variant that subtracts the mean projection across all six foundations before classification, removing the shared moral-salience component.

### 3.3.2 Evaluation sets

1. **Held-out test set** (48 pairs, 8 per foundation): internal validation from the probing dataset.
2. **Moral Foundations Vignettes** (30 items, 5 per foundation): curated from \citet{clifford2015moral}, offering external validation with established MFT stimuli independent of our dataset pipeline.
3. **Causal evaluation prompts** (48 prompts): cross-validation with the causal experiments, testing whether directions that are causally relevant also predict foundation identity.

## 3.4 Sparse autoencoder analysis methods

### 3.4.1 SAE training

We train a ReLU sparse autoencoder (16,384 latent dimensions, $8\times$ expansion) on 2M token activations from the C4 corpus \citep{raffel2020c4} at layer 8, using L1 sparsity regularization ($\lambda = 0.005$).
The decoder columns are constrained to unit norm.
The pre-encoder bias is initialized to the mean activation.

### 3.4.2 Moral feature identification

We encode 192 moral and 192 neutral training sentences through the trained SAE and compute per-feature selectivity: the mean activation difference between moral and neutral inputs.
Top-$k$ features (ranked by $|\text{selectivity}|$) are compared with probe directions via:

- **Individual alignment**: cosine similarity between each feature's decoder column and each foundation direction.
- **Subspace overlap**: fraction of probe direction variance captured by the top-$k$ SAE feature subspace (via SVD projection).

### 3.4.3 Random baseline

To establish significance of subspace overlap, we compute a null distribution: 1,000 iterations of projecting probe directions onto 100 random unit vectors in $\mathbb{R}^{2048}$ and recording mean membership.
