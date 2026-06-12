# 1. Introduction

Mixture-of-Experts (MoE) architectures route each token through a
sparse subset of expert modules, partitioning the representation
space into discrete, inspectable units. This structural partition
has a natural consequence for alignment research: if moral features
concentrate in specific experts, MoE models offer intervention
points (expert pruning, expert-specific fine-tuning, router
modification) that dense models lack. Conversely, if moral features
distribute uniformly across experts, MoE and dense architectures
are equivalent for alignment purposes, and the additional complexity
of expert-level analysis buys nothing.

We test this question on OLMoE-1B-7B \citep{muennighoff2024olmoe}, a
64-expert, top-8 MoE language model with 6.9B total parameters
(1.3B active per token), using the moral probing and fragility
methodology from companion work on dense OLMo models
\citep{reblitzrichardson2026fragility}. OLMoE is uniquely positioned for this
analysis: it is the open MoE model with by far the densest published
checkpoint record (244 checkpoints at 5,000-step intervals), and its
dense counterpart OLMo-2 1B (from the same
lab, with comparable active parameter count and full checkpoint
access) gives a controlled architectural comparison.

We report four findings that converge on a single mechanism:

**Finding 1: MoEs do not create expert moral specialization.** Nearly
all 1,024 per-expert probes (64 experts $\times$ 16 layers) decode
moral content well above chance (1,020 exceed 75%). At the peak layer, every expert
individually exceeds 84% accuracy. The Gini coefficient of expert
accuracy is below 0.03 at all layers; moral encoding is as
uniformly distributed across experts as it is across neurons in a
dense model. The router shows negligible moral content preference
(maximum 1.8%).

**Finding 2: MoE encoding is 4.2$\times$ more fragile than dense.**
Despite matching dense OLMo-2 1B on probing accuracy (99.0% vs.
99.0% peak), OLMoE's moral encoding collapses under 4.2$\times$
less noise (mean critical $\sigma^* = 0.92$ vs. 3.81). The
fragility gap is not explained by weaker individual expert
representations or unstable routing; both are robust in isolation.

**Finding 3: The fragility originates in output dilution.** The MoE
block's aggregated output (a top-8 weighted average of 64 expert
outputs) contributes to the residual stream at 74$\times$ smaller
scale than the dense MLP output, measured as the standard deviation
of the feedforward block's output across inputs. This *output
dilution* means that the same absolute noise level overwhelms the
MoE moral signal while leaving the dense signal intact.

**Finding 4: Specialization never emerges during training.** Across 11
checkpoints spanning OLMoE's training (step 5K to step 1.2M, covering
20B to 5,033B tokens), the Gini coefficient at the peak layer stays between
0.012 and 0.018 at every checkpoint (cf. 0.016--0.023 across all
layers in the final model, §4.2). Moral encoding is present from
the earliest available checkpoint (92.1% peak accuracy at step 5K)
and remains stable at 93.7% by step 1.2M without ever concentrating
in specific experts. The top-5 experts by accuracy change between
adjacent checkpoints at near-random rates (Jaccard $\approx$ 0.09).

The output dilution finding has direct implications for the
interpretability of probing accuracy as an alignment metric. Two
models can produce identical probing accuracy profiles (high
accuracy from early layers, broad encoding across the full network)
while differing by nearly two orders of magnitude in the robustness
of the underlying signal. Probing accuracy measures what information
is *present*; fragility testing, as developed in companion work
\citep{reblitzrichardson2026fragility}, measures how *securely* that information
is encoded. In MoE architectures, the gap between these two metrics
is dramatically larger than in dense models, because the sparse
aggregation bottleneck preserves information content while reducing
signal scale.

The paper contributes the first expert-level moral probing analysis
of an MoE language model, the first quantification of the MoE
output dilution effect and its relationship to representational
fragility, and a controlled dense-vs-MoE comparison on identical
probing methodology. All experiments run on a single MacBook Pro M4
Pro (24 GB, MPS) on base (non-instruct) models.
