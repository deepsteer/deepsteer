# 1. Introduction

How do language models represent morality? Prior work in this
series established that models encode moral content *broadly*:
probing accuracy saturates early during pre-training and spans
nearly all layers \citep{reblitzrichardson2026fragility}. Fragility
testing revealed that this encoding grows more robust throughout
training even after accuracy plateaus. Extension to
mixture-of-experts (MoE) models showed that moral signal is
uniform across experts, with no expert specialization, but that a
74$\times$ output scale gap produces structural fragility
\citep{reblitzrichardson2026dilution}.

Both papers treated moral encoding as a single binary feature:
moral vs. neutral. A model that merely detects "this text involves
morality" has cleared a low bar. Genuine moral understanding
requires *structured* representations --- the ability to distinguish
care from fairness, loyalty from authority, and to encode the
relationships between them. The transition from moral *detection*
to moral *understanding* is the subject of this paper.

We operationalize this transition through the geometry of
foundation-specific probe directions. Where prior work trained one
probe to separate moral from neutral content, we train six: one for
each Moral Foundations Theory (MFT) foundation
\citep{haidt2012righteous, graham2013mft}. The learned probe weight
vectors define *directions* in the model's representation space.
The angular relationships between these directions reveal whether
the model has developed structured moral representations.

Three geometric signatures correspond to three qualitatively
different modes of moral representation:

1. **Collapse.** All foundation directions converge toward a single
   "moral salience" direction. The model detects moral relevance
   but does not distinguish frameworks. This is detection without
   structure.

2. **Isolation.** Foundation directions are orthogonal with no
   relational structure. The model has separate moral "slots" but
   no representation of how frameworks relate. This is structure
   without coherence.

3. **Integration.** Foundation directions are separated but
   non-orthogonal, with inter-framework geometry reflecting known
   relationships from moral psychology. This is the precondition
   for moral reasoning.

Applied to OLMo-2 1B and OLMoE-1B-7B, we find clear *integration*:
foundation directions are distinct (mean pairwise cosine similarity
$\approx 0.22$, far from collapse) and span 5 effective dimensions
at every layer. However, hierarchical clustering does not recover
the MFT distinction between individualizing and binding foundations
--- the inter-framework structure the model develops is not aligned
with human moral-psychological categories, despite the directions
being well-separated.

Extending the fragility protocol to per-foundation probes reveals
differential robustness: sanctity/degradation is the most robust
foundation in dense models but the most fragile in MoE, showing a
6.2$\times$ fragility ratio that far exceeds the overall
architectural gap. Output dilution does not suppress all moral
foundations uniformly --- culturally specific moral concepts appear
more vulnerable to the MoE aggregation bottleneck.

This paper makes three contributions:

1. **Framework-specific probing methodology.** We introduce probe
   direction geometry as a tool for measuring structured moral
   representations, bridging the gap between binary moral probing
   and the richer structure posited by moral psychology.

2. **Empirical geometric characterization.** We show that OLMo-2
   1B and OLMoE-1B-7B exhibit the integration signature, that
   framework geometry is comparable across architectures, and that
   the model's inter-framework structure does not align with MFT
   group predictions despite clear directional separation.

3. **Differential fragility.** We demonstrate that moral foundations
   differ in robustness, and that MoE architecture
   disproportionately degrades sanctity/degradation
   representations --- the first evidence that output dilution is
   not foundation-uniform.
