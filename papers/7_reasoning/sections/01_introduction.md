# 1. Introduction

Refusal in instruction-tuned language models can be removed by a single rank-one
weight edit. Arditi et al. \citep{arditi2024refusal} showed that refusal is
mediated by one linear direction in the residual stream, and that orthogonalizing
the model's weights against it suppresses refusal across many harmful requests.
A prior paper in this series asked what that direction is made of, and found, in
several dense base+instruct families, that the ablatable refusal direction is
almost entirely residual against the six-foundation moral subspace: comprehension
and compliance are separable, and refusal is the separable part
\citep{reblitzrichardson2026crossmodel}. Zhao et al. \citep{zhao2025harmfulness}
reached a parallel conclusion from a different angle, showing that harmfulness and
refusal are encoded as distinct internal concepts at different token positions:
harmfulness at the last instruction token (\tinst) and refusal at the last
templated token (\tpost). Steering the harmfulness direction changes the model's
belief that an instruction is harmful; steering the refusal direction toggles
refusal without changing that belief.

Both results were established on models that answer directly. Reasoning models are
different. They emit an extended chain of thought before answering, and at least
part of the refusal decision is made inside that trace
\citep{yamaguchi2025reasoning}. A reasoning model that has been trained, by
reinforcement learning, to deliberate about whether a request is harmful might
route its refusal through that deliberation, re-coupling harm comprehension to the
refusal decision that non-reasoning models keep separate. Whether genuine
deliberative training re-couples comprehension and refusal, or whether the
dissociation extends into the reasoning regime, is the question this paper studies.

We study it across three open reasoning models that isolate the deliberative axis.
GPT-OSS-20B \citep{openai2025gptoss} learned to reason by reinforcement learning
under deliberative alignment, the one model in which the chain of thought is a
trained behavior rather than an imitated one. DeepSeek-R1-Distill-Llama-8B and
DeepSeek-R1-Distill-Qwen-14B \citep{deepseek2025r1} acquired their reasoning by
supervised distillation of R1 traces onto a non-reasoning base. We extract
directions at \tinst\ and \tpost\ with conventions held fixed across the panel
(\autoref{sec:methodology}), and ask where harm comprehension sits relative to the
refusal decision in each.

We report four findings.

**Finding 1: The harmfulness/refusal dissociation holds in reasoning models,
deliberative training included.** Harmfulness is strongly decodable at the
instruction token (Cohen's $d' \approx 4.4$--$5.0$ across the panel) yet nearly
orthogonal to the refusal direction (cosine $0.11$--$0.16$), and this holds for
GPT-OSS-20B, whose reasoning was learned by reinforcement learning, as cleanly as
for the distilled models (\autoref{fig:dissociation}). Training a model to
deliberate about harm does not fuse its harm comprehension with its refusal
behavior.

**Finding 2: Harm comprehension is distributed across the reasoning trace and
displaced from the decision, and the refusal decision is itself distributed.**
GPT-OSS-20B carries the most trace-level moral content of the panel, and that
content peaks near the first third of the chain of thought and falls to its lowest
value at the decision (\autoref{fig:trace}). The refusal decision is not
bottlenecked through any one direction: on a held-out set, no single direction, the
refusal direction itself included, cleanly ablates GPT-OSS-20B refusal
(\autoref{fig:distributed}). The model that most clearly reasons about harm is the
one whose refusal is least reducible to a single handle.

**Finding 3: The harmfulness representation is causally real, but largely distinct
from Moral Foundations.** Steering the harmfulness direction flips a model's stated
harm judgment on direct-answering instruct models, establishing that the direction
is causal and not merely decodable (\autoref{fig:causal}). Yet it projects mostly
outside the six-foundation moral subspace, overlapping it above chance but by only a
small fraction (\autoref{fig:moral}). The harm-judgment that models route their
refusal-comprehension through is related to, but not the same as, the moral
foundations of Haidt \citep{graham2013mft}.

**Finding 4: Reasoning models do not expose a clean judgment readout, which
constrains interpretability methods built on direct-answering models.** The
reply-inversion test that validates the harmfulness direction reads the model's
stated harm judgment, and reasoning models do not state one cleanly: across three
readout mechanisms they reason past the question, echo it, or over-judge when
forced, while the same readout is clean on instruct models
(\autoref{sec:methodology}). The judgment is present in the model but not exposed in
a position a readout can reach. As the growing body of refusal- and
harmfulness-direction work \citep{arditi2024refusal, zhao2025harmfulness} is applied
to the reasoning models safety research increasingly depends on, the behavioral
readout channel, not the direction, becomes the binding constraint.

We are explicit about what we do not find. The clean functional-versus-imitated
split we sought, harm comprehension load-bearing for refusal in the RL-deliberative
model and decorative in the distilled ones, is not cleanly runnable here.
GPT-OSS-20B's refusal is distributed, so no low-dimensional ablation isolates a
load-bearing test (Finding 2); and the distilled models barely refuse at baseline
(\autoref{fig:behavioral}), a behavior confounded with R1 distillation degrading
refusal training, so their failure to translate harm comprehension into refusal
cannot license a claim that the comprehension is decorative. The honest contribution
is the dissociation and its reach, extended into the reasoning regime and grounded
against a peer-reviewed causal account of harmfulness encoding.
