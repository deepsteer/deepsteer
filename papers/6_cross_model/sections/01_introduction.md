# 1. Introduction

A refusal direction can be removed from many instruction-tuned models by a single
rank-one weight edit. Arditi et al. \citep{arditi2024refusal} showed that refusal
behavior is mediated by one linear direction in the residual stream, and that
orthogonalizing the model's weights against it suppresses refusal across a wide
range of harmful requests; the Heretic tool \citep{pew2025heretic} automates the
same operation across Qwen, Llama, and other families. That refusal is
single-direction-ablatable is, by now, well established and is not what we study.

The open question this leaves is different. Two models can both have a refusal
direction that a rank-one edit removes, yet differ in *what that direction is
made of*. In one model the ablatable direction might be built largely out of
moral features, so that removing it also disturbs the model's moral
representation; in another it might be nearly orthogonal to moral structure, a
separate mechanism that ablation can excise cleanly. Heretic does not distinguish
these cases, because it measures only whether refusal drops, not where the
refusal direction sits relative to the model's moral geometry. **Is the ablatable
refusal direction morally grounded, or geometrically separate from moral
structure?** That routing question is what we test.

A prior paper in this series answered it for one model. On OLMo-3 7B, the refusal
direction is almost entirely residual against the six-foundation moral subspace:
its projection onto that subspace is small, its cosine to each Moral Foundation
direction is near zero, and ablating it removes refusal while leaving moral
comprehension and behavioral moral judgment intact
\citep{reblitzrichardson2026alignment}. Comprehension and compliance are
separable, and refusal is the separable part. The natural worry is that OLMo-3 is
a weak test. Its post-training is lighter than that of the most heavily aligned
open models, and a model whose refusal is shallow might show a non-moral refusal
direction for that reason alone. If so, the dissociation would be a property of
OLMo-3, not of language models, and the interesting question, whether refusal is
ever morally grounded, would have to be asked on a more strongly aligned model.

This paper runs the same decomposition across three model families to find out.
We hold the extraction conventions fixed and compare a dense ${\sim}7$--8B
base+instruct pair from each of three families: OLMo-3 7B (the anchor),
Qwen2.5-7B \citep{qwen2025qwen25}, and Llama-3.1-8B \citep{grattafiori2024llama3}.
For each we extract the refusal direction (Arditi/Heretic), the six-foundation
moral subspace, and a persona direction, and decompose refusal into a moral part,
a persona part, and a residual. We then apply the single-direction ablation and
re-measure moral comprehension, both the linear moral representation and
behavioral moral judgment. The panel is deliberately narrow: dense (not
mixture-of-experts, whose output dilution is a known confound
\citep{reblitzrichardson2026dilution}), size-matched, and non-reasoning, so that
"OLMo looks different" cannot be an artifact of architecture, scale, or a
thinking-mode trace.

Two results separate, and the separation is the paper.

**Finding 1: The representational dissociation is family-invariant.** In all
three families the refusal direction is ${\sim}99\%$ residual and
near-orthogonal to the moral subspace (projection fraction $0.07$--$0.13$, mean
absolute cosine to the foundations $0.04$--$0.075$), with essentially no persona
component. Ablating it leaves the *linear moral representation* untouched: fresh
per-foundation probe accuracy stays at $1.0$ and the moral subspace keeps its
five effective dimensions, in every family. Whatever else differs across these
models, none of them stores refusal inside its moral geometry, and removing
refusal does not damage what the model linearly encodes about morality. OLMo-3 is
representative, not an outlier.

**Finding 2: The behavioral dissociation is clean where refusal is cleanly
removable.** In OLMo-3 and Qwen2.5 the single-direction ablation drives refusal to
zero (compliance on a persona-shift battery rises to $1.0$) and leaves behavioral
moral judgment intact (within two points of the unablated model). This is the
comprehension/compliance dissociation realized behaviorally: the model still
judges moral scenarios as before while refusing nothing, exactly the
high-comprehension / low-compliance cell.

**Finding 3: Llama-3.1 is the exception, on two counts.** First, its refusal is
markedly harder to remove: a single-direction ablation that fully strips OLMo and
Qwen only partially strips Llama, and the mid-depth layers that work for the
others do nothing for it. Second, and more striking, ablating Llama's refusal
direction degrades its behavioral moral judgment while leaving the linear moral
representation perfectly intact. We establish that this degradation is
*refusal-direction-specific* and *dose-dependent* rather than a side effect of
perturbing the weights: a magnitude-matched random weight perturbation, even at
twice the magnitude, leaves moral judgment unchanged, while the matching
perturbation along the refusal direction degrades it, and the degradation grows
monotonically with the fraction of the refusal direction removed. In Llama-3.1,
refusal and behavioral moral judgment are *entangled*. We report this as a
property of one released model, measured through controls, and use it to motivate
where the question should go next, not as a cross-model law.

Taken together: the refusal/moral separation that holds *representationally*
across families does not always hold *behaviorally*. The thin-refusal worry is
refuted, because the dissociation reproduces even in better-aligned models;
but the behavioral side is family-dependent, and the one place it breaks down,
Llama-3.1, shows refusal entangled with moral behavior even though the moral
representation is intact and family-invariant. A bonus observation follows from
the ablation step: single-direction refusal removability is itself
family-dependent, and the family where refusal is hardest to remove is also the
one where it is morally entangled. With $n=1$ on that co-occurrence we do not draw
a correlation; we read it as a signal that the models worth probing next are the
more strongly aligned and reasoning-capable ones, where both properties may be
more common.

This work continues a series on moral representation in language models: that
moral content is linearly decodable early in pre-training
\citep{reblitzrichardson2026fragility}, encoded uniformly across mixture-of-experts
experts \citep{reblitzrichardson2026dilution}, organized into a structured,
five-dimensional framework geometry \citep{reblitzrichardson2026geometry} whose
directions are causal for moral judgments \citep{reblitzrichardson2026causal}, and
coupled to behavior during post-training as a separable compliance mechanism
\citep{reblitzrichardson2026alignment}. The present paper asks whether that last
result, the separability of refusal from moral representation, generalizes, and
finds that it does representationally and usually but not always behaviorally.

Everything here is diagnostic. We characterize where refusal sits relative to
moral structure and what ablating it does to moral comprehension, using random
and feature-matched controls to attribute the effects we find. We produce no
method for making refusal harder to remove, and we make no claim about any
model's training data or recipe, which we do not have.
