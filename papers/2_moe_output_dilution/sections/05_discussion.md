# 5. Discussion

## 5.1 Output Dilution as an Architectural Property

The 74$\times$ output scale gap between MoE and dense feedforward
blocks is not specific to moral encoding — it is a structural
consequence of sparse expert aggregation. When a top-$k$ routing
mechanism selects 8 of 64 experts, each expert contributes roughly
$\frac{1}{8}$ of the aggregated output (modulated by routing
weights). The aggregated output is therefore a weighted average of 8
expert outputs, each operating on a 1024-dimensional intermediate
space, producing a 2048-dimensional output. The dense MLP, by
contrast, applies its full parameter budget to every token, producing
a larger-scale output.

The dilution effect likely scales with the sparsity ratio. OLMoE
uses top-8 of 64 (12.5% sparsity); models with higher sparsity
(e.g., top-2 of 8 in Mixtral) may show even stronger dilution,
while lower sparsity would reduce it. The load-balancing auxiliary
loss, which encourages uniform expert utilization, may further
contribute by preventing any single expert from dominating the
aggregated output.

An important caveat: the output dilution we measure is the
*feedforward block's contribution to the residual stream*, not the
total signal in the residual stream. Because the residual connection
carries forward the pre-MoE hidden state, the full hidden state
after the MoE block is dominated by the residual, not the MoE output.
This means that probing the full hidden state (as in §4.1) picks up
moral signal from both the MoE contribution and accumulated residual
contributions from earlier layers. The fragility difference arises
because noise added to the full hidden state disrupts the MoE
contribution disproportionately — the noise is small relative to the
residual but large relative to the MoE output.

## 5.2 Implications for Alignment Interventions

The absence of expert moral specialization (§4.2) closes one
potential intervention path: there are no "moral experts" to prune,
fine-tune, or monitor. MoE architectures, despite their structural
partition into discrete expert modules, do not make moral encoding
more tractable for targeted intervention than dense models do.

The output dilution finding (§4.4), however, opens a different
concern. If moral features in MoE models are encoded at very small
scale in the feedforward contribution, they may be easier to
accidentally destroy during fine-tuning. A LoRA adapter that
modifies the feedforward weights by even a small amount in absolute
terms could represent a large *relative* perturbation to the MoE
output. This prediction is testable: companion work's C15 finding
(fragility-locus shift under insecure-code LoRA in dense OLMo)
could be replicated on OLMoE, where we would predict a larger
fragility shift from the same fine-tuning recipe.

## 5.3 Feature Redundancy Across Architectures

Prior work on this project \citep{reblitzrichardson2026fragility} found that
probe-direction suppression in dense 1B models does not capture
behavior: a gradient penalty suppresses the probe direction by
3.07 SD with no effect on behavioral judge scores (within 0.01 / 10).
This was attributed to feature redundancy — at the 1B scale, the
model has enough representational capacity to encode persona features
along directions orthogonal to the probe's extracted direction.

The present findings show that MoE architecture does not resolve
this redundancy problem. Despite partitioning representations across
64 discrete modules, moral features are encoded equally strongly
in every module. The structural partition of MoE is orthogonal to
the functional organization of moral features — the model encodes
the same information in every expert, just as a dense model encodes
it across every neuron.

The checkpoint trajectory analysis (§4.5) strengthens this
conclusion: the absence of specialization is not a late-training
convergence but is present from step 5K (20B tokens), before the
routing mechanism has fully matured. Training does not create and
then destroy expert moral specialization — it never exists.

This suggests that feature redundancy in language models is not a
consequence of architectural homogeneity (all neurons participating
in everything) but of training dynamics: the training objective
distributes useful features across all available representational
capacity, regardless of how that capacity is architecturally
partitioned. Load-balancing losses in MoE, which encourage uniform
expert utilization, may actively reinforce this tendency.

## 5.4 Probing Accuracy as an Alignment Metric

The dense-vs-MoE comparison starkly illustrates the insufficiency
of probing accuracy as a standalone alignment metric. Both models
achieve near-perfect probing accuracy (99--100%) with full encoding
breadth (decodable at every layer) and near-zero encoding depth
(onset at layer 0). On probing accuracy alone, the two architectures
are indistinguishable. Yet the MoE model's moral encoding is
5.1$\times$ more fragile, and the underlying feedforward signal is
74$\times$ weaker.

Probing accuracy measures the *presence* of information — whether
a linear classifier can extract a feature from the representation.
Fragility testing measures the *security* of that information —
how much perturbation the encoding can withstand before the feature
becomes unextractable. The output dilution mechanism shows that
these two metrics can diverge dramatically: information can be
present (high accuracy) but insecure (low fragility), encoded at a
scale that is trivially disrupted.

The early-layer accuracy gap between architectures provides a
second diagnostic. OLMoE's early layers (0--3) achieve only 79--86%
per-expert accuracy, compared to 94--97% for OLMo-2 at the same
layers. When the probing dataset was tightened to remove
superficial cues (v1 $\to$ v2 revision), OLMoE's early-layer
accuracy dropped more than OLMo-2's, suggesting that the MoE
architecture's diluted output makes early layers more dependent on
shallow features. This is consistent with output dilution: when the
feedforward contribution to the residual stream is small, early
layers cannot inject enough signal to support robust classification,
and probes compensate by exploiting dataset artifacts when available.

This reinforces the methodological argument from companion work
that fragility testing is a necessary complement to probing
accuracy, particularly when comparing architectures with different
internal signal scales.

## 5.5 Limitations

**Single MoE model family.** We study only OLMoE. Generalization
to Mixtral (top-2 of 8), DeepSeek-MoE (fine-grained experts), or
Qwen-MoE is open. The output dilution mechanism predicts that
higher-sparsity architectures (lower $k/N$ ratio) will show
greater fragility, but this has not been tested.

**Mean-pooling approximation.** Per-expert probing and the
perturbation experiments operate on mean-pooled representations,
collapsing the sequence dimension. This approximation is standard
in probing studies but may mask per-token routing effects. The
router's actual operation is per-token, not per-sequence.

**Linear probes only.** As in companion work, all probes are linear.
Nonlinear probes (MLP classifiers) might extract moral features from
the small-scale MoE output more effectively, potentially reducing
the apparent fragility gap. However, the output scale measurement
(§4.4) is independent of probe architecture.

**Controlled but not identical comparison.** OLMoE and OLMo-2 are
from the same lab but differ in training data mix, hyperparameters,
and training duration, not just architecture. Same-lab provenance
minimizes but does not eliminate these confounds.

**English and MFT only.** The probing dataset covers English
sentences grounded in Haidt's Moral Foundations Theory. Moral
encoding in other languages, moral frameworks, or culturally
specific ethical norms is not tested.
