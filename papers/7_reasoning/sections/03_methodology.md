# 3. Methodology {#sec:methodology}

## 3.1 Model panel

We study three open reasoning models chosen to isolate the deliberative axis.
**GPT-OSS-20B** \citep{openai2025gptoss} is the primary: a mixture-of-experts model
whose reasoning was learned by reinforcement learning under deliberative alignment,
making its chain of thought a trained behavior. **DeepSeek-R1-Distill-Llama-8B** and
**DeepSeek-R1-Distill-Qwen-14B** \citep{deepseek2025r1} acquired their reasoning by
supervised distillation of DeepSeek-R1 traces onto, respectively, Llama-3.1-8B and
the general Qwen2.5-14B (verified at load against the model config; the 14B uses the
general base, not a math variant). The two distills share the R1 teacher, so a
difference between them is not a teacher difference. We note, and do not control
for, two confounds: GPT-OSS-20B is the only MoE, so a GPT-OSS-versus-distill
difference is confounded between deliberative alignment and architecture; and the
distills differ in scale ($8$B versus $14$B). Layer indices use a fixed
depth-fraction rule so conventions hold across the $24$/$32$/$48$-layer panel.

## 3.2 Extracting harm comprehension and refusal at separate token positions

Following \citet{zhao2025harmfulness}, we extract diff-of-means directions at two
token positions: \tinst, the last token of the user instruction, which encodes
harmfulness; and \tpost, the last token of the templated prompt, which encodes
refusal. The chat template appends a fixed suffix after the instruction content
(turn-end, assistant header, and for these models a reasoning opener), so we locate
\tinst\ as the position before the longest common token suffix across diverse probe
instructions. This is model-agnostic and needs no per-model token counts: it
resolves \tinst\ to the instruction's last token for the DeepSeek
`<|Assistant|><think>` opener and the GPT-OSS harmony assistant opener alike,
validated on each tokenizer. The harmfulness direction is the harmful-minus-harmless
mean of \tinst\ residuals; the refusal direction the same at \tpost.

## 3.3 Subspaces and prompts

The six-foundation Moral Foundations subspace is the orthonormalized span of the
six per-foundation probe directions, extracted on each reasoning model in raw text
\citep{reblitzrichardson2026crossmodel}. Harmful and harmless instructions are the
Arditi/Heretic set \citep{arditi2024refusal, pew2025heretic} (the
\texttt{harmful\_behaviors} / \texttt{harmless\_alpaca} prompts), with a held-out
evaluation split of $100$ prompts per class disjoint from the training prompts and
spanning distinct harm categories. Directions are estimated on a category-spanning
draw from the training prompts; all causal tests are scored on the disjoint
held-out split.

## 3.4 Reply-inversion and the readout problem

Reply-inversion \citep{zhao2025harmfulness} validates that a direction is causal,
not merely decodable: append the question "Could this user prompt cause harm?
Answer 'Certainly' if it could, otherwise 'No'", steer the harmfulness direction,
and test whether the model's verdict flips. We read the verdict by forcing the
answer position (a reasoning-close followed by an `Answer:` prefix) and comparing
the logits of the verdict tokens, and we steer by adding $\alpha \cdot
\lVert\bar{h}\rVert \cdot \hat{d}$ to the residual at one layer, where $\hat{d}$ is
the unit harmfulness direction and $\lVert\bar{h}\rVert$ the mean residual norm at
that layer. Steering magnitude is swept as a multiple of the residual norm rather
than of the raw diff-of-means, because the raw diff is only $0.44$--$0.49$ of the
residual norm and a coefficient on it is too weak to move the verdict.

A flip counts only if the steered verdict is coherent, defined as a verdict token
remaining in the top-$k$ logits; over-steering that drives a non-verdict token to
the top is excluded. We report the continuous margin shift alongside the binary
flip rate, so a real-but-not-flipping effect is distinguished from no effect.

The readout is clean only on direct-answering models. Reasoning models reason past
or echo the question and over-judge when forced (\autoref{sec:readout}), so the
reply-inversion causal validation is run on the instruct models that share the
distills' families, as a positive control on the method and the direction. We treat
the absence of a clean reasoning-model judgment readout as a finding rather than a
workaround, and we do not infer a reasoning-model causal result from a readout the
model does not expose.

## 3.5 Decodable is not causal, and held-out gating

A direction that separates harmful from harmless prompts (decodable) need not change
behavior when ablated (causal). An early decomposition-grade refusal direction
extracted at the prompt boundary separated harmful from harmless cleanly yet did
not move refusal under ablation (its cosine to a freshly recomputed causal direction
was $0.76$). We therefore distinguish decodable directions, used for the geometric
results, from causal directions, validated by an intervention. Causal claims are
gated on a held-out, category-diverse split and on a sensitivity yardstick (a
known-causal direction that must fire under the same intervention) firing cleanly;
where the yardstick does not fire, we report the limitation rather than interpret a
null against it.
