# 5. Discussion

**A representational property generalizes; a behavioral one does not.** The
cleanest way to state the result is to separate two things that "comprehension"
can mean. The *linear moral representation*, what a probe can decode from
activations and how those foundation directions are arranged, is untouched by
refusal ablation in all three families, and the refusal direction is residual and
orthogonal to it in all three. That property is family-invariant, and it settles
the question the paper set out to ask: the geometric separation of refusal from
morality reported for OLMo-3 is not an artifact of a weakly aligned model. But
*behavioral moral judgment*, what the model actually outputs when asked to judge a
scenario, comes apart from the representation in Llama-3.1: there, removing the
refusal direction degrades the judgment while leaving the representation intact.
The two senses of comprehension travel together in OLMo and Qwen and separate in
Llama, which is worth keeping distinct in how the field talks about "comprehension
surviving ablation."

**Refusal removability is not universal.** A side result of the ablation step is
that the single-direction, single-edit removability that motivates abliteration
tooling is family-dependent in degree. OLMo and Qwen lose refusal completely under
the uniform orthogonalization; Llama, under the same protocol applied at its own
best layer, loses only half, and the mid-depth layers that work for the others
have no effect on it. We do not claim Llama's refusal is un-ablatable, only that
the simplest single-direction edit is markedly less effective on it. A more
elaborate search of the kind Heretic performs may well do better; characterizing
that is outside our scope and, by design, not something we pursue, since it would
amount to engineering more effective refusal removal.

**The Llama coupling, stated precisely.** What we have shown about Llama is that
ablating its refusal direction degrades behavioral moral judgment, that this
degradation is specific to the refusal direction (a magnitude-matched random
perturbation, and an ablation of the comparable-magnitude persona direction, do
not produce it), and that it grows with the fraction of refusal removed. We call
this an *entanglement* between refusal and behavioral moral judgment that is
*dose-dependent* and *refusal-direction-specific*. We deliberately do not say that
refusal "routes through" moral machinery or shares a circuit with it: a
directional ablation experiment licenses claims about which direction, perturbed
by how much, changes which behavior, and not claims about internal mechanism or
shared structure. The honest summary is a dose-response between a direction and a
behavior, with the representation held fixed throughout.

**Why this matters for where to look next.** In this one model the two
unusual properties, a refusal direction that resists single-direction removal and
a refusal direction entangled with moral behavior, occur together. With a single
model we cannot tell whether that co-occurrence is meaningful or coincidental, and
we are careful not to present it as a cross-model relationship. What it does is
locate the interesting regime. If harder-to-remove and morally entangled refusal
tend to appear together, they should appear more often in the models with the
strongest alignment and, plausibly, in reasoning models, where refusal is produced
through a deliberative trace rather than a direct response. Those are precisely the
models we excluded here to keep the comparison controlled. Extending the panel to
reasoning models is the natural next study, and it is where an $n>1$ test of the
co-occurrence belongs.

**Limits.** The co-occurrence of ablation-resistance and moral entanglement rests
on a single model. The panel is dense, ${\sim}7$--8B, and non-reasoning by design,
so the results speak to that regime and not yet to larger, sparser, or
reasoning-capable models. The moral-judgment probe is 48 scenarios; we report
bootstrap confidence intervals so the effect size is not read off a handful of
items, but a larger behavioral battery would tighten it. Over-ablation degrades
generation, so the strongest perturbations are not clean measurements of moral
judgment and we use them only as a directional endpoint. And throughout we measure
where refusal sits and what perturbing it does; we do not have any model's
training data or recipe and make no claim about why a given model's refusal is
shaped the way it is.

**Safety.** Every measurement here is diagnostic. We characterize an existing
property of released models, the geometry of their refusal directions and what
ablating those directions does to moral comprehension, using random and
feature-matched controls to attribute the effects. Llama's coupling is a measured
property of one released model, not a technique we built; the over-ablation sweep
is a measurement probe, not an intervention; and we produce no method for making
refusal harder to remove. The reverse question, whether refusal *should* be more
morally grounded and how one might achieve that during pre-training rather than
post-training, is raised by these results but is a separate program, and not one we
advance here.
