# 4. Results

## 4.1 Refusal is residual and orthogonal to morality in every family

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{phase1_decomposition}
\caption{Refusal-direction energy in the moral subspace (left) and residual
(right) as a function of depth fraction, for all three families. The three curves
track together: across depth the refusal direction places almost none of its
energy in the moral subspace and is overwhelmingly residual. The dotted line marks
the depth-$0.5$ headline layer.}
\label{fig:phase1}
\end{figure}

Decomposing each model's refusal direction at the depth-$0.5$ headline layer gives
the same shape in all three families (\Cref{tab:decomp}). Refusal is
$98$--$99\%$ residual energy, its projection onto the six-foundation moral
subspace is small (norm-ratio $0.07$--$0.13$, energy fraction $0.005$--$0.016$),
its mean absolute cosine to the foundations is $0.04$--$0.075$, and its persona
component is essentially zero (cosine to the persona direction $0.02$--$0.05$).
The OLMo-3 value reproduces the prior single-model paper exactly: a projection
fraction of $0.104$ against the published $0.1044$, which confirms the conventions
transferred before any cross-model claim is read. \Cref{fig:phase1} shows the
pattern holds across depth, not just at one layer.

| metric (headline layer) | OLMo-3 | Qwen2.5 | Llama-3.1 |
|---|---|---|---|
| residual energy fraction | 0.989 | 0.983 | 0.995 |
| moral energy fraction | 0.011 | 0.016 | 0.005 |
| moral projection fraction | 0.104 | 0.127 | 0.071 |
| mean abs.\ cosine to foundations | 0.060 | 0.075 | 0.041 |
| persona energy fraction | 0.000 | 0.001 | 0.000 |
| single-direction AUC | 1.00 | 1.00 | 1.00 |
| single-vs-full-rank AUC gap | 0.000 | 0.000 | 0.000 |
| across-band refusal eff.\ rank | 3 | 6 | 4 |

Table: Refusal-direction decomposition at the depth-$0.5$ headline layer. No
family routes refusal materially through moral features; even the largest moral
projection (Qwen, $0.127$) is the same regime as the anchor ($0.104$).
\label{tab:decomp}

No family has a materially more moral refusal direction than the anchor. The
spread in projection fraction is narrow ($0.07$--$0.13$), and the model with the
*most* moral projection (Qwen) is also the one with the *most* diffuse refusal
(effective rank $6$ across the band, against $3$ for OLMo), the opposite of what a
"diffuseness explains the low moral fraction" account would predict. The
single-direction AUC is $1.0$ and the single-vs-full-rank gap is $0.000$ in every
family: the one ablatable direction captures all the linear separability of
harmful from harmless, with no residual multi-direction refusal hiding behind it.
The thin-refusal worry is refuted: the geometric separation of refusal from
morality is not an OLMo artifact but a family-invariant property.

## 4.2 Ablating refusal preserves the linear moral representation everywhere

Single-direction refusal removability is itself family-dependent. Sweeping the
ablation layer by depth fraction, OLMo's harmful refusal rate falls from $0.575$ to
$0.000$ and Qwen's from $1.000$ to $0.000$, but Llama's only falls from $0.900$ to
$0.475$, and the mid-depth layers that strip the others do nothing for Llama; only
a shallow layer reduces its refusal, and only partially (\Cref{tab:battery}).

At each model's best ablation layer, ablating refusal leaves the *linear moral
representation* untouched in all three families: fresh per-foundation probe
accuracy stays at $1.0$ and the moral subspace keeps its five effective
dimensions, before and after (\Cref{tab:battery}). Whatever refusal removal does
to a model, it does not damage what the model linearly encodes about morality.
This is the representational dissociation, and it is family-invariant.

| metric | OLMo-3 | Qwen2.5 | Llama-3.1 |
|---|---|---|---|
| ablation layer (depth) | 19 (0.59) | 14 (0.50) | 13 (0.41) |
| refusal rate: clean $\to$ ablated | 0.575 $\to$ 0.000 | 1.000 $\to$ 0.000 | 0.900 $\to$ 0.475 |
| probe accuracy: clean $\to$ ablated | 1.0 $\to$ 1.0 | 1.0 $\to$ 1.0 | 1.0 $\to$ 1.0 |
| eff.\ dimensionality: clean $\to$ ablated | 5 $\to$ 5 | 5 $\to$ 5 | 5 $\to$ 5 |
| persona-shift compliance: clean $\to$ ablated | 0.75 $\to$ 1.00 | 0.90 $\to$ 1.00 | 0.70 $\to$ 0.95 |
| moral judgment: clean $\to$ ablated | 0.75 $\to$ 0.79 | 0.875 $\to$ 0.812 | 0.75 $\to$ 0.604 |

Table: Ablation and the comprehension battery at each model's swept ablation
layer. Representation is preserved everywhere; behavioral moral judgment is
preserved in OLMo and Qwen but drops in Llama, the only model whose refusal does
not fully strip.
\label{tab:battery}

## 4.3 The behavioral dissociation is clean in OLMo-3 and Qwen2.5

In the two families where the single-direction ablation fully removes refusal, it
also leaves *behavioral* moral judgment intact. OLMo's persona-shift compliance
rises from $0.75$ to $1.00$ (refusal stripped, every persona gap closing to zero)
while its moral-judgment accuracy is essentially flat, $0.75 \to 0.79$. Qwen's
compliance rises from $0.90$ to $1.00$ and its judgment moves from $0.875$ to
$0.812$. This is the comprehension/compliance dissociation realized end to end:
the model judges moral scenarios as before while refusing nothing. For these
families the prior single-model result generalizes without qualification, in both
its representational and behavioral form.

## 4.4 Llama-3.1: refusal entangled with moral judgment

Llama behaves differently, and the difference is the result. At its best ablation
layer its persona-shift compliance does rise ($0.70 \to 0.95$), but its
moral-judgment accuracy drops, $0.75 \to 0.604$. The linear representation is
untouched (probe accuracy $1.0$, effective dimensionality $5$), so this is a
behavioral effect, not a representational collapse. Two controls establish that
the drop is specific to the refusal direction and graded in the amount of refusal
removed.

**The drop is refusal-direction-specific.** Perturbing the same weight matrices
with random Gaussian noise of *identical per-matrix Frobenius norm* to the refusal
ablation leaves moral judgment unchanged: over eight independent draws the
matched-random judgment is $0.747 \pm 0.007$, essentially the clean value of
$0.75$, against the refusal-ablated $0.604$ (a $-21\sigma$ outlier below the
matched-random null). Ablating the persona direction, a real, decodable,
comparable-magnitude, non-refusal feature, also leaves judgment at $0.75$. Only
ablating the refusal direction degrades behavioral moral judgment, and the linear
representation stays at probe accuracy $1.0$ and effective dimensionality $5$ under
every one of these conditions. So Llama is not generally fragile to weight
perturbation of this magnitude, and it is not the case that ablating any salient
feature degrades judgment.

**The drop is dose-dependent.** \Cref{fig:dose} sweeps the fraction $\alpha$ of
the refusal direction removed. Moral judgment falls monotonically as more refusal
is removed (Spearman correlation between refusal removed and judgment drop
$=1.0$), and within the coherent regime the partial-ablation drops are
individually significant: at $\alpha=0.5$ the drop is $0.083$ (95\% bootstrap CI
$[0.02, 0.17]$, $n=48$ scenarios) and at the full single-direction strength
$\alpha=1$ it is $0.146$ (CI $[0.06, 0.25]$). At $\alpha=0.5$ the perturbation
along the refusal direction already costs judgment even though it has not yet
reduced behavioral refusal (refusal rate still $0.90$), the signature of an
entanglement at the level of the direction rather than of the refusal behavior it
controls. Throughout, the magnitude-matched random null stays flat, $0.74$ even at
twice the full-ablation magnitude, so the dose-response cannot be a magnitude
artifact.

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{llama_dose_response}
\caption{Llama-3.1 ablation-strength sweep. As the fraction $\alpha$ of the
refusal direction removed grows, behavioral moral judgment (blue, with bootstrap
CI) falls while the linear moral representation (green, fresh probe accuracy) stays
at $1.0$ and a magnitude-matched random perturbation (grey diamonds, even at
$\alpha{=}2$) leaves judgment $\approx 0.74$. The harmful refusal rate (red) drops
only once over-ablation ($\alpha \ge 1.5$) is reached. Effect sizes are read from
the coherent $\alpha \le 1$ regime; the $\alpha \ge 1.5$ collapse below chance
reflects degraded generation and is reported only as an endpoint.}
\label{fig:dose}
\end{figure}

We anchor the coupling on this $\alpha \le 1$ regime, where the model still
generates coherently. Driving $\alpha$ higher does eventually strip Llama's
refusal in full ($\alpha=1.5$ takes the refusal rate to $0.0$) and collapses moral
judgment far below chance, but at those over-ablation strengths the model's
generation is degraded, so we report that collapse only as a caveated endpoint and
never as the headline effect. The supported claim is the coherent one: in
Llama-3.1, refusal and behavioral moral judgment are entangled, the entanglement
is specific to the refusal direction, and it grows with the amount of refusal
removed.

Finally, the two ways Llama differs, its refusal being the hardest to remove and
its refusal being morally entangled, co-occur in this single model. We do not draw
a correlation from one model; we note the co-occurrence and treat it as a question
for the next study rather than a finding of this one.
