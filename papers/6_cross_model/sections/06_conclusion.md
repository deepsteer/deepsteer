# 6. Conclusion

We asked whether the geometric separation of refusal from moral representation,
reported for a single model, is a property of language models in general or an
artifact of one model's post-training. Running the same decomposition across three
dense base+instruct families under identical conventions, we find that
representationally the separation is family-invariant: in OLMo-3, Qwen2.5, and
Llama-3.1 the refusal direction is ${\sim}99\%$ residual against the moral
subspace and near-orthogonal to it, and ablating it leaves the linear moral
representation untouched. Behaviorally, the picture is family-dependent. In OLMo-3
and Qwen2.5 a single-direction ablation removes refusal and leaves behavioral
moral judgment intact, the clean comprehension/compliance dissociation. In
Llama-3.1, refusal is both harder to remove and entangled with moral judgment:
ablating it degrades the judgment dose-dependently and refusal-direction-specifically,
verified against a magnitude-matched random null and a persona-direction control,
while the linear moral representation stays intact.

The separation of refusal from morality that holds representationally across
families therefore does not always hold behaviorally, and the single model where
it breaks down is also the one whose refusal is hardest to remove. We report that
co-occurrence as an $n=1$ observation, not a law, and read it as a pointer toward
the more strongly aligned and reasoning-capable models that a controlled panel had
to exclude. The work is diagnostic throughout: it characterizes where refusal sits
relative to moral structure and what removing it does to moral comprehension, and
builds no technique for making refusal harder to remove.
