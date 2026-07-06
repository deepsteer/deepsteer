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
requires *structured* representations: the ability to distinguish
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
   but does not distinguish frameworks.^[We use *framework*
   interchangeably with the Moral Foundations Theory term *foundation*
   throughout (and in the title). We do not intend the moral-psychology
   sense in which *framework* denotes a theory-level position such as
   deontology or utilitarianism. We retain *framework* in part because
   *foundation* is itself overloaded in the language-model setting,
   where a "foundation model" is a base, pre-fine-tuning model,
   exactly the class of models we probe.] This is detection without
   structure.

2. **Isolation.** Foundation directions are orthogonal with no
   relational structure. The model has separate moral "slots" but
   no representation of how frameworks relate. This is structure
   without coherence.

3. **Integration.** Foundation directions are separated but
   non-orthogonal, with inter-framework geometry reflecting known
   relationships from moral psychology. This is the precondition
   for moral reasoning.

Applied to OLMo-2 1B and OLMoE-1B-7B (with a dense OLMo-2 7B as a
scale check), we report three findings:

**Finding 1: Moral foundations are represented with integration
geometry, but we find no evidence of the structure moral psychology
predicts.**
Foundation directions are distinct: mean pairwise cosine similarity is
$\approx 0.22$--$0.27$ across layers, positive everywhere and far from
collapse. This positive shared component is the integration signature.
Against a matched non-moral concept battery built identically to the
foundations, it is ${\sim}20\times$ larger (0.26 vs.\ 0.013; paired
$\Delta = 0.223$, CI $[0.202, 0.244]$, excluding 0; §4.2), so the shared
component is moral-specific relative to a matched non-moral battery
rather than a generic content-vs-neutral axis (whether it is
specifically moral rather than generic affective salience is the one
residual control we flag; §5.6). The six directions span 5 effective
dimensions at
every layer, which rules out collapse but, at the ceiling for six
mean-centered directions, does not by itself separate integration from
isolation. Hierarchical clustering does not recover the MFT
individualizing/binding split at either the dense 1B or 7B, though the
group-structure test is underpowered (smallest achievable $p = 0.05$,
so a small effect is not excluded); the most consistent structure the
model forms is a care--sanctity pairing that crosses MFT groups.

**Finding 2: The geometry of moral dilemmas is partially
compositional.** Probes for 15 two-foundation dilemmas reach 94.2%
mean accuracy, and each dilemma direction is partly explained by the
2D subspace of its two component foundation directions. Dilemma pairs
that share a component foundation are closer in representation space
than those that do not (mean cosine 0.273 vs. 0.196, permutation
$p = 0.0001$), and the MoE architecture preserves this compositional
structure.

**Finding 3: Output dilution degrades every foundation uniformly,
not selectively.** Extending the fragility protocol to per-foundation
probes shows a uniform cross-architecture effect rather than a
per-foundation one: once accuracy is averaged over multiple noise
seeds, every foundation is more fragile in MoE than in dense (a
${\sim}2.3\times$ gap), and no single foundation is reliably most or
least robust within an architecture. An apparent complexity--fragility
ordering does not survive scale normalization (§4.11). Output dilution
suppresses moral encoding across the board.

This paper introduces probe-direction geometry, a method for
measuring structured moral representations that bridges binary moral
probing and the richer structure posited by moral psychology. With
it, we give the first geometric characterization of foundation and
dilemma representations in base language models, and show that this
geometry is comparable across dense and MoE architectures and stable
from 1B to 7B. We also report the first per-foundation fragility
comparison across architectures, which finds that output dilution
degrades moral encoding uniformly rather than selectively.
