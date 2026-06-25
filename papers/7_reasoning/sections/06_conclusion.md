# 6. Conclusion

We asked whether deliberative reasoning re-couples harm comprehension to the refusal
decision that non-reasoning models keep separate, and found that it does not. Across
three open reasoning models, including GPT-OSS-20B, whose reasoning was learned by
reinforcement learning under deliberative alignment, harmfulness is strongly
decodable at the instruction token and nearly orthogonal to the refusal direction.
The harmfulness/refusal dissociation established for instruct models
\citep{zhao2025harmfulness} and the comprehension/compliance dissociation
established for non-reasoning models \citep{reblitzrichardson2026crossmodel} both
extend into the reasoning regime.

Within that regime the comprehension is present but placed away from the decision:
GPT-OSS-20B carries the most trace-level moral content, concentrated early in the
trace and decaying to the decision, while its refusal is distributed across
directions with no single-direction bottleneck. The harmfulness representation the
models do encode is causally real, validated by reply-inversion on direct-answering
instruct models, and largely distinct from Moral Foundations, overlapping the
six-foundation subspace above chance but lying mostly outside it.

We report two boundaries honestly. The clean functional-versus-imitated split we
hoped to measure is not cleanly runnable, because GPT-OSS-20B's refusal is
distributed and the distilled models' weak refusal is confounded with distillation
degrading refusal training. And the reply-inversion readout that validates the
harmfulness direction on instruct models cannot be read on the reasoning models
themselves, which do not expose a stated judgment in a reachable position. That last
boundary is itself a result: interpretability methods built on direct-answering
models require a behavioral readout the reasoning model will emit, and the chain of
thought does not provide one. As safety research leans more heavily on open
reasoning models, both the geometry of where harm comprehension sits and the channel
through which it can be read become first-order concerns.
