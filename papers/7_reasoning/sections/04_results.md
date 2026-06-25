# 4. Results

## 4.1 Harmfulness and refusal are separately encoded {#sec:dissociation}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig1_harmfulness_vs_refusal}
\caption{Harmfulness is strongly encoded at the instruction token yet nearly
orthogonal to refusal, in all three reasoning models. Left: harmful/harmless
separation (Cohen's $d'$) of the \tinst\ and \tpost\ diff-of-means directions;
harmfulness is strongest at \tinst. Right: cosine between the \tinst\ harmfulness
direction and the \tpost\ refusal direction, with $1.0$ marking a shared
direction. Reproduces the harmfulness/refusal separation of
\citet{zhao2025harmfulness} in reasoning models, including the RL-deliberative
GPT-OSS-20B.}
\label{fig:dissociation}
\end{figure}

At the instruction token, harmful and harmless prompts separate cleanly along the
diff-of-means harmfulness direction: $d' = 5.01$ for GPT-OSS-20B, $4.39$ for the
Llama distill, and $5.01$ for the Qwen distill (\autoref{fig:dissociation}, left).
The separation is weaker at \tpost\ ($2.03$, $3.59$, $3.84$), placing the
harmfulness signal at the instruction token, as \citet{zhao2025harmfulness} report
for instruct models. The \tinst\ harmfulness direction is nearly orthogonal to the
\tpost\ refusal direction: cosine $0.16$, $0.11$, $0.16$ (\autoref{fig:dissociation},
right). Harmfulness is decodable and strong, and it is not the refusal direction.

This holds for GPT-OSS-20B, whose reasoning was learned by reinforcement learning
under deliberative alignment. Genuine deliberative training does not collapse the
two representations into one. The dissociation that holds for non-reasoning models
\citep{reblitzrichardson2026crossmodel, zhao2025harmfulness} extends into the
reasoning regime, deliberative training included.

## 4.2 Comprehension is distributed and displaced from the decision {#sec:trace}

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{fig4_trace_profile}
\caption{Moral-subspace content as a function of fractional position in the
reasoning trace, on coherent closed traces (matched cognitive phase). GPT-OSS-20B
carries the most trace-level moral content; it peaks near the first third of the
trace and falls to its lowest value at the decision (position $1.0$).}
\label{fig:trace}
\end{figure}

Refusal in a reasoning model is partly made inside the chain of thought, so we ask
where harm comprehension sits along the trace. Extracting the refusal direction at
two sites within the trace, the last input token (end-of-prompt) and the trace
tokens, the end-of-prompt direction reproduces the non-reasoning result: it is
$99.1\%$ residual against the moral subspace at the boundary, and a single
direction captures all of its linear separability (the single-versus-full-rank AUC
gap is $0.000$). The moral-projection asymmetry between the trace direction and the
end-of-prompt direction is at or below zero in all three models, most negative in
GPT-OSS-20B. There is no re-coupling of morality at the decision point.

Across the trace the picture is distributional. Controlling for trace length by
comparing matched fractional positions on coherent closed traces, GPT-OSS-20B
carries the most moral-subspace content (mean fraction $0.025$, against $0.013$ and
$0.018$ for the distills), and that content is concentrated near the first third of
the trace and decays toward the decision (\autoref{fig:trace}). The model that most
clearly deliberates carries the most trace-level moral content, but that content is
displaced from where the refusal decision is read out.

## 4.3 The refusal decision is distributed {#sec:distributed}

\begin{figure}[t]
\centering
\includegraphics[width=0.78\linewidth]{fig5_distributed_refusal}
\caption{Held-out causal ablation of GPT-OSS-20B refusal. For each candidate
direction (end-of-prompt, CoT-mean, CoT-last), the fraction of baseline-refusable
held-out prompts whose refusal is coherently removed, and the fraction lost to
over-ablation incoherence. No direction cleanly ablates refusal: the end-of-prompt
and CoT-last directions barely move it, and CoT-mean removes it only by destroying
generation.}
\label{fig:distributed}
\end{figure}

Whether GPT-OSS-20B's refusal routes through any single low-dimensional handle is
testable directly: ablate a candidate direction during generation and measure
whether refusal is coherently removed. On a held-out, category-diverse set, no
direction does so (\autoref{fig:distributed}). The end-of-prompt refusal direction,
estimated on a category-spanning training draw, coherently flips only $4\%$ of
held-out refusals; the CoT-last direction flips none; the CoT-mean direction
removes refusal in $88\%$ of cases but only by driving generation into incoherence,
which the coherence filter excludes. GPT-OSS-20B refusal is distributed: it is not
bottlenecked through any single direction, the refusal direction itself included.
Because no low-dimensional handle isolates the refusal decision, the
single-subspace load-bearing test that would ask whether harm comprehension is
causally upstream of refusal cannot be run cleanly on this model; the
distributed-ness is itself the answer, since a representation that is not a
bottleneck for refusal cannot be the moral subspace either.

## 4.4 Harmfulness is largely distinct from moral foundations {#sec:moral}

\begin{figure}[t]
\centering
\includegraphics[width=0.82\linewidth]{fig3_harmfulness_vs_moral}
\caption{Decomposition of the \tinst\ harmfulness direction relative to the
six-foundation Moral Foundations subspace. The harmfulness direction projects
mostly outside the moral subspace; the in-subspace fraction ($0.11$--$0.18$)
exceeds the random-direction chance floor (diamonds, ${\approx}0.04$) by
$3.0$--$3.9\times$, but the bulk of the direction is distinct from moral
foundations.}
\label{fig:moral}
\end{figure}

The harmfulness that reasoning models encode at \tinst\ is related to, but not the
same as, the moral foundations our program measures. Projecting the harmfulness
direction onto the six-foundation subspace, $0.18$ of it lies inside for
GPT-OSS-20B and $0.11$ for each distill (\autoref{fig:moral}). These fractions
exceed the chance floor for a random direction (${\approx}0.04$, the
$\sqrt{k/d}$ expectation for a $k$-dimensional subspace in $d$ dimensions) by
$3.0$ to $3.9$ times, so the overlap is real and above chance. It is also small:
the mean absolute cosine to the individual foundations is $0.07$--$0.11$, and
roughly $85\%$ of the harmfulness direction lies outside the moral subspace.
Harm-judgment carries a modest moral-foundations component on top of a
representation that is largely its own. The prohibited-capability harmfulness of
\citet{zhao2025harmfulness} and the moral foundations of \citet{graham2013mft}
are distinct objects that share an above-chance component.

## 4.5 The harmfulness direction is causally validated {#sec:causal}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig2_causal_validation}
\caption{Reply-inversion on direct-answering instruct models. Steering the \tinst{}
harmfulness direction (magnitude as a multiple of the residual norm) shifts the
harm-judgment margin toward harmful; the dashed line marks the baseline margin, the
average shift needed to cross to a harmful verdict. Blue bars are alpha settings at
which the steered verdict coherently flips (annotated with the held-out flip rate);
grey bars over-steer into incoherence. The direction is causal.}
\label{fig:causal}
\end{figure}

Whether the harmfulness direction is causal, not merely decodable, is tested by
reply-inversion \citep{zhao2025harmfulness}: steer the direction and ask whether the
model's stated harm judgment changes. On direct-answering instruct models that share
the distills' families, steering shifts the harm-judgment margin strongly toward
harmful and coherently flips it (\autoref{fig:causal}). For
Qwen2.5-14B-Instruct, whose harmless prompts are judged safe with a confident
margin of $-16.9$, steering shifts the margin by up to $+17.4$ and coherently flips
$33\%$ of held-out judgments; for Llama-3.1-8B-Instruct the shift is $+3.0$ against
a $-2.1$ baseline and flips $23\%$. Flips occur in the lower-magnitude, coherent
regime; larger steering shifts the margin further but over-steers into incoherence,
which the coherence gate excludes. The earlier failure to move the verdict with
weaker steering was a magnitude artifact: the raw diff-of-means is only
$0.44$--$0.49$ of the residual norm, so a coefficient on it was too small. The
\tinst\ harmfulness direction is a real causal handle on the model's harm judgment.

## 4.6 Reasoning models do not expose a clean judgment readout {#sec:readout}

The causal test in \autoref{sec:causal} runs on instruct models because the
reasoning models do not expose a harm judgment that can be read. The test appends a
harm-judgment question and reads the verdict, and across three readout mechanisms
the reasoning models defeat it. A regex over the generated answer matches the
question echoed inside the reasoning trace rather than the verdict; reading the
final answer fails because the trace rambles without stating a verdict within
budget; forcing the answer position and reading verdict-token logits returns
near-noise, and harmless prompts are scored harmful with large margins when the
verdict is forced. The same forced-answer logit readout is clean on instruct
models, which answer the question directly (Qwen2.5-14B-Instruct judges $24/24$
harmless prompts safe and $24/24$ harmful prompts harmful). The judgment is not
absent from the reasoning model; it is not exposed in a position a readout can
reach. Interpretability methods that depend on a stated judgment, reply-inversion
among them, require a behavioral channel the reasoning model will emit, and the
chain of thought is not one.

## 4.7 Behavioral contrast, and its confound {#sec:behavioral}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{fig6_behavioral_contrast}
\caption{Clean refusal rate on harmful prompts. GPT-OSS-20B refuses every prompt;
the distilled reasoning models refuse $8$--$17\%$ (a lower bound, since their
acknowledge-then-reframe responses are under-counted). The gap is confounded with
R1 distillation degrading refusal training and is reported as a behavioral
contrast, not a functional-versus-imitated asymmetry.}
\label{fig:behavioral}
\end{figure}

Behaviorally the panel splits sharply: GPT-OSS-20B refuses all harmful prompts,
while the distilled models refuse $17\%$ (Llama) and $8\%$ (Qwen)
(\autoref{fig:behavioral}). The distill rate is a lower bound, since their typical
response acknowledges the harm and reframes toward a defensive answer rather than
issuing an explicit refusal, which the classifier under-counts. The distills carry
harm comprehension at \tinst\ (\autoref{sec:dissociation}) yet rarely translate it
into refusal. We do not read this as evidence that their harm comprehension is
decorative, because the low refusal rate is confounded with the well-documented
effect of R1 distillation degrading refusal training: a model that refuses almost
nothing tells us little about whether its harm comprehension would be load-bearing
if it did refuse. The contrast is suggestive and we report it as such.
