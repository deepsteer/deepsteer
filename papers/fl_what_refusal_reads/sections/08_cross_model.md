# 8. Across families: what refusal reads and how it commits {#cross-model}

The OLMo result is one model. A four-model panel shows that the harm-reading picture
generalizes but that refusal decisions differ along two separable axes: *what* they read
(harm versus broad moral content) and *how* they commit (at the read layer, early, or
reversibly).

**The harm direction is a real, causal, cross-model object.** Before separating the axes, the
harm percept has to be more than an OLMo artifact, and it is. Harmfulness and refusal are
separately encoded at the instruction token, with the harmfulness read discriminating cleanly
(d-prime 5.01 on GPT-OSS) and near-orthogonal to the downstream refusal read (cosine 0.16),
extending Zhao et al. [@zhao2025harmfulness] to deliberative reasoning models. The harm
direction is largely outside the moral subspace, with an in-subspace fraction of 0.18 on
GPT-OSS and 0.11 on reasoning distills, 3.0–3.9 times the $\sqrt{k/d} \approx 0.04$ chance
floor, so about 85% of it lies outside the moral foundations, the same "reads a harm sliver"
object the OLMo rank sweep formalizes. And it is causal: reply-inversion steering along the
harm direction flips model judgments (Qwen2.5-14B-Instruct shift $+17.4$ flips 33% of replies,
Llama-3.1-8B-Instruct $+3.0$ flips 23%), where an earlier raw diff-of-means null was a
magnitude artifact [@zhao2025harmfulness].

## 8.1 Llama reads broad and commits early {#llama}

Llama-3.1's anatomy is OLMo-like: pre-norm reconstruction 1.0008 (no fold needed), a clean
low-dimensional decision channel (participation ratio 13.5, null 0.148 moving to 0.114 under
standardization), a distributed write with a 30% multilayer-perceptron share, and all top
writers labeled neither-moral-nor-harm. But Llama refuses on intent (baseline refusal 9/10,
against OLMo's ~17%), so its refusal cell is measurable where OLMo's is empty, and it reads
differently.

At matched depth (layer 12, chosen below) Llama reads the moral subspace *broadly*: refusal
transfer 0.85 is essentially equal to judgment transfer 0.79, the gap that stays open on OLMo
closes on Llama, and a harm-rank-1 restriction recovers only 0.59. The reads-broad verdict
survives a harm-coextensive alternative at rank 1: a single harm cue spans only 3.6% of the
moral basis that drives Llama's refusal, so the transfer grows into moral directions the harm
axis does not point along (a severity-ladder version of this control at rank 2–4 awaits
contrasts not yet collected). Llama reads broad moral content, not just harm. The full
depth-matched battery at layer 12 is in \Cref{app:panel}.

The commitment axis is why the matched-depth qualifier is load-bearing. Llama's refusal is
directionally asymmetric at the decision boundary (36 micro-graded twins): adding harmful
content moves refusal coherently ($+0.142$, 95% CI [$+0.086$, $+0.212$], sign fraction 0.81),
but removing harmful content does not (disengage $-0.014$, 95% CI [$-0.084$, $+0.052$], sign
fraction 0.51, incoherent). A patch-layer sweep names the mechanism: Llama's disengage is
coherent below the read layer but incoherent at the read layer 16 ($-0.014$), while OLMo's
disengage is coherent at its read layer ($-0.62$). The full patch-layer sweep is in
\Cref{app:panel}. Llama commits *early*, crystallizing its refusal before the decision site;
OLMo commits at or
after the read layer.

This resolves a robustness anomaly in the same panel. Llama's refusal is entangled with moral
judgment where the other models' is not: at the best ablation layer, removing refusal drops
judgment accuracy from 0.75 to 0.604, a $-21\sigma$ outlier against matched-random ablations
(0.747 $\pm$ 0.007) and dose-dependent (Spearman 1.0). Early commitment of a broad moral read
is the mechanism:
because Llama reads broad moral content and commits before the decision site, ablating its
refusal reaches into the moral read in a way OLMo's harm-keyed late-committing gate does not.
The cross-model asymmetry that first looked like a third property is instead a consequence of
early commitment. Read at the layer where each model commits, the naive asymmetry statistic is
$+0.82$; read at matched depth (layer 12), it collapses to $-0.28$ on Llama against $-0.54$ on
OLMo, a difference of $+0.26$. \Cref{fig:depth} shows the collapse, the depth-referenced
verdict that separates a genuine difference from a measurement taken past the commitment layer.

## 8.2 GPT-OSS reads harm and is reversible {#gpt-oss}

GPT-OSS reads harm, but correlationally rather than by interchange. Its refusal direction is
harm-loaded at both positions it carries signal: at the prompt (the instruction token) the
standardized cosine to the harm direction is 0.977 against 0.001 for the harm-orthogonal moral
subspace (near-purely harm), and in-trace it stays harm-dominant but attenuates (0.49 versus
0.13). This is a prompt-to-trace consistent harm read, and it is why the in-trace refusal
projection landed below the moral-family band earlier. It is a projection result, not a
patching result, so the reads-harm placement for GPT-OSS is correlational; the causal
interchange version is held.

What GPT-OSS adds is the commitment axis at its most informative extreme: it is a *reversible
reader*. An inculpating-analysis prefill flips unsaturated benign requests to refuse 7 out of 7
(Wilson 95% [0.65, 1.0]), so the decision is not fixed before the trace, deliberation is
consequential. And in the other direction, a graded exculpatory prefill flips ceiling-refusing
violating items to comply 6 out of 10, with the decision-channel refusal projection moving
monotonically toward comply in all 10 items (projection-moved fraction 1.0, monotone fraction
1.0). \Cref{fig:reversibility} shows the graded panel, with the per-strength series tabulated
in \Cref{app:panel}. GPT-OSS reverses in both directions;
its refusal is a read that deliberation can re-argue, the clean contrast to Llama's early
commitment.

## 8.3 The two-axis result {#two-axis}

\begin{table}[t]
\centering
\caption{What refusal reads $\times$ how it commits, across three model families. Rows are
the models with a resolved commitment reading; columns are the two axes. OLMo's read is by
interchange, Llama's by interchange at matched depth, GPT-OSS's by projection (correlational).}
\label{tab:two-axis}
\begin{tabular}{@{}lll@{}}
\toprule
Model & What refusal reads & How it commits \\
\midrule
OLMo-3-7B & Harm percept (transfer saturates & At / after the read layer \\
 & at the harm-rank-1 level, ceiling 0.31; & (disengage coherent at the \\
 & judgment reads broadly) & read layer, $-0.62$) \\[2pt]
Llama-3.1-8B & Broad moral content (refusal & Early (disengage coherent \\
 & transfer 0.85 $\approx$ judgment 0.79 & below layer 15, incoherent \\
 & at matched depth, gap closes) & at the read layer 16) \\[2pt]
GPT-OSS-20B & Harm (correlational: prompt cosine & Reversible reader (engage \\
 & 0.977 to harm vs 0.001 orthogonal; & 7/7, disengage 6/10, \\
 & causal test held) & monotone projection) \\
\bottomrule
\end{tabular}
\end{table}

\Cref{tab:two-axis} states the measured result: refusal reads harm on OLMo and GPT-OSS and
broad moral content on Llama, and it commits at the read layer on OLMo, early on Llama, and
reversibly on GPT-OSS. This table is the empirical claim, and it stands.

Its *interpretation* is a hypothesis, not an $n = 3$ result. The three points are ordinally
consistent with a single underlying knob: the models whose refusal reads a low-rank harm slice
(OLMo and GPT-OSS, roughly rank 1) are the ones that commit late or reversibly, and the model
whose refusal reads broadly (Llama, roughly rank 8) is the one that commits early. This
licenses "the effective dimensionality of the refusal read predicts its reversibility" as a
falsifiable follow-on hypothesis. It does not confirm it. The read-and-commit pairing is
architecture-confounded at three points: the models differ in lineage, scale, tokenizer, and
reasoning-versus-instruct training all at once, so a dimensionality account and a
lineage account fit the same table equally well. Deconfounding requires varying one axis at a
time, for example a deliberation-trained variant of a single base model, or a lineage-matched
scale sweep. We state the hypothesis to be tested, not a mechanism established.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_depth_collapse.pdf}
\caption{Depth-referenced verdicts, the Llama-versus-OLMo asymmetry. Read at each model's own
read layer, Llama's refusal asymmetry statistic is $+0.82$, which reads as a third property (a
hard latch). Read at matched depth (layer 12), it collapses to $-0.28$ against OLMo's $-0.54$,
a difference of $+0.26$. The read-layer value was a post-commitment artifact: Llama commits
early, so a measurement at its read layer is taken past the layer where the decision was
already fixed. The asymmetry is a consequence of early commitment, not a separate axis.}
\label{fig:depth}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_gpt_oss_reversibility.pdf}
\caption{GPT-OSS is a reversible reader. A graded exculpatory-analysis prefill (increasing
strength, left to right) flips ceiling-refusing violating items toward compliance, 6 of 10
flipping behaviorally, while the decision-channel refusal projection moves monotonically toward
comply in all 10 items (monotone fraction 1.0). In the other direction an inculpating prefill
flips benign requests to refuse 7 of 7. Deliberation is consequential and reversible in both
directions, the clean contrast to Llama's early commitment.}
\label{fig:reversibility}
\end{figure}
