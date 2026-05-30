# 1. Introduction

<!-- ~1.5 pages. Arc: geometry is necessary but not sufficient → need causal + behavioral + mechanistic validation → this paper provides all three. -->

Companion paper \citep{reblitzrichardson2026geometry} established that language models develop structured moral representations: six foundation-specific probe directions exhibit integration geometry (effective dimensionality 5, MFT-consistent clustering), and these directions are stable across extraction methods, dataset sizes, and text registers.
But probe directions are correlational.
A direction that separates moral from neutral text in activation space may reflect a confound (sentence length, formality, topic) rather than a genuine moral feature.
Three questions remain open:

1. **Causal relevance.** Does the model *use* these directions during generation? If we ablate a foundation's direction, does the model lose access to that specific moral concept? If we inject a direction, does the model shift toward that foundation's content?

2. **Behavioral grounding.** Do the directions predict which moral foundation a novel text activates? Representational geometry tells us how the model *organizes* moral knowledge; behavioral benchmarking tests whether that organization is *functionally accessible* for moral reasoning.

3. **Mechanistic correspondence.** Do the supervised probe directions correspond to features the model discovers on its own? Sparse autoencoders (SAEs) learn unsupervised features from general text. If morally selective SAE features align with probe directions, the directions are not artifacts of the probing procedure but reflect native model structure.

We address all three questions using the same OLMo-2 1B model and 1,200-pair moral probing dataset from \citet{reblitzrichardson2026geometry}.
Section~\ref{causal-validation} presents direction ablation and steering injection experiments.
Section~\ref{behavioral-grounding} reports projection-based foundation classification on held-out, external, and causal evaluation stimuli.
Section~\ref{sae-analysis} compares SAE feature geometry with probe directions.
Section~\ref{discussion} synthesizes the three forms of evidence and discusses implications for representation engineering of moral concepts.
