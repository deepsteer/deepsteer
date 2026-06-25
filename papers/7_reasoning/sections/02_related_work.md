# 2. Related Work

**Refusal as a linear direction.** Arditi et al. \citep{arditi2024refusal} showed
that refusal in instruction-tuned models is mediated by a single residual-stream
direction, removable by rank-one weight orthogonalization; the Heretic tool
\citep{pew2025heretic} automates the operation across families. That refusal is
single-direction-ablatable is established. Our concern is where that direction sits
relative to harm and moral representations, and whether, in reasoning models, the
refusal decision is bottlenecked through one direction at all.

**Harmfulness and refusal as separate concepts.** Zhao et al.
\citep{zhao2025harmfulness} established, on instruct models, that harmfulness and
refusal are encoded as distinct internal concepts at different token positions:
harmfulness at the last instruction token, refusal at the last templated token.
Steering the harmfulness direction changes the model's belief that an instruction is
harmful, while steering refusal toggles the behavior without changing that belief,
and adversarially finetuning a model to accept harmful instructions leaves the
harmfulness representation largely intact. We build directly on this account,
extend its token-position dissociation into the reasoning regime, position our Moral
Foundations subspace against its harmfulness direction, and adopt its
reply-inversion test as our causal yardstick. The point of contact with our distill
result is its finetuning-robustness finding: a model whose refusal is degraded can
retain its harm comprehension.

**Refusal in reasoning traces.** Yamaguchi et al. \citep{yamaguchi2025reasoning}
showed, on DeepSeek-R1-Distill-Llama-8B among others, that reasoning models make
part of the refusal decision inside the chain of thought rather than at the
prompt-response boundary, that the opening sentence of the trace can determine the
refusal outcome, and that linear refusal directions ablate harmful compliance less
reliably on reasoning models than on non-reasoning ones. Our two-site extraction
follows this localization; we add the harm-comprehension geometry and the
distributed-decision result, the latter consistent with their weaker
single-direction ablation.

**Comprehension and compliance.** A prior paper in this series found, across dense
base+instruct families, that the ablatable refusal direction is almost entirely
residual against the six-foundation moral subspace, and that ablating it removes
refusal while leaving moral comprehension and behavioral judgment intact
\citep{reblitzrichardson2026crossmodel}. Comprehension and compliance are separable.
This paper asks whether that separation survives in reasoning models whose refusal
is partly deliberated, and finds that it does.

**Moral Foundations in language models.** Our moral subspace is the six foundations
of Moral Foundations Theory \citep{graham2013mft}. Zhao et al.'s harmfulness is a
prohibited-capability contrast, related to but distinct from the foundations; we
measure the relationship directly (\autoref{sec:moral}) rather than assume it.
