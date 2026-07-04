# 7. What refusal reads: the harm percept {#reads-harm}

The causal core is on OLMo-3, where interchange patching [@meng2022rome] on the heads that
write the decision channel resolves *what* the refusal decision reads. The answer is the harm
percept: a rank-1 slice of the moral subspace, aligned with a harm direction
[@zhao2025harmfulness], and not the broad subspace that moral judgment reads on the same
patches. Moral directions here are causal and foundation-specific to begin with, a preliminary
we established with foundation-wise ablation whose specificity strengthens with depth; the
OLMo-3 interchange cells below sharpen that into a rank sweep that says *which* moral content
refusal uses.

**The write is distributed.** Refusal is written into the ~13-dimensional decision channel by
a set of heads, led by layer 16 head 23 but not carried by it. That head writes $+0.742$ onto
the refusal direction (channel-matched specificity $+0.756$) and alone accounts for 11.6% of
the total specificity; the writers span layers 11–16, with layer 15 head 15 the sole
anti-refusal writer ($-0.130$). Cumulative channel-matched specificity reaches 44% at the top
ten heads and needs about 62 heads for 80%. Attention is not the whole story: multilayer
perceptrons contribute 38% of the decision-site write (write fraction 0.384). The per-head
attribution is certified by an exact normalization fold that brings the reconstruction from
3.05 to 0.9999 (companion methods note, in preparation). None of the top ten writers is a
clean harm-copy head: all are labeled neither-moral-nor-harm, with a moral-subspace fraction
of 0.15–0.28 and comparable harm loading, split between instruction-attenders and
content-attenders. Refusal is written broadly, not routed through one moral head.

**The interchange is specific to the moral subspace, then specific to harm inside it.** Using
request-twins (matched requests carrying opposite judgment outcomes, $n = 23$), we patch the
decision channel and read the induced change in the refusal and judgment projections. The full
patch moves refusal $-0.0833$; a patch restricted to the moral subspace moves it $-0.0282$;
its complement moves it $-0.0636$; a harm-rank-1 patch moves it $-0.0261$; a random rank-3
patch moves it $-0.0005$ (refusal minimum detectable effect 0.0238). The moral subspace is a
*specific* substrate: restricting to it moves refusal more than a random rank-3 patch does
($\Delta = 0.031$, paired 95% CI [0.020, 0.043], excludes 0). But almost all of that specific
effect is the harm slice. The harm-restricted patch ($-0.0261$) nearly equals the full
moral-subspace patch ($-0.0282$), and the harm-partialed patch (the moral subspace with the
harm direction projected out) still moves refusal $-0.0133$ (95% CI [$-0.023$, $-0.005$],
excludes 0), about half. So refusal is harm-dominant with a small, resolvable residual
non-harm moral read (the harm direction captures a fraction 0.46 of the moral subspace).

**The rank sweep is decisive.** For a readout $r$ (the refusal or the judgment projection),
define the restricted-transfer coefficient $R_r(k)$ as the fraction of the full interchange
effect on $r$ that is reproduced when the patch is confined to the top-$k$ directions of the
moral subspace. As $k$ grows over $\{1, 3, 8, 16\}$, judgment transfer climbs
$0.05 \to 0.46 \to 0.59 \to 0.66$ while refusal transfer *saturates*
$0.01 \to 0.31 \to 0.26 \to 0.27$ at the harm-rank-1 level (harm-rank-1 transfer 0.31), with a
random-direction null near zero at every rank and per-rank purity 0.97–0.99. Expanding the
moral basis beyond harm buys more judgment coupling and no more refusal coupling. This is the
central finding, and \Cref{fig:oneknob} plots it: refusal reads the harm percept and stops;
judgment keeps reading as the subspace widens. About 73% of refusal's causal twin-difference
input lies outside the rank-16 moral basis (69% already at the rank-3 peak). Judgment reads
the subspace broadly *on the same patches*, which is the within-model proof that the content
is there to be read; refusal simply does not read it.

**One free parameter fits the sweep.** The refusal curve is the judgment curve clipped at a
harm ceiling: $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling}, R_{\text{judgment}}(k))$,
with the ceiling $\approx 0.31$. This one-knob model fits the plateau ($k \geq 3$) at RMSE
0.036 (residual $-0.002$ at $k = 3$), while harm-amplitude alternatives miss by 0.10–0.24. The
one place it breaks is rank 1, where it over-predicts (measured 0.013 versus predicted 0.052):
the highest-variance contrast component, the most harm-aligned single direction (variance
purity 0.974, cosine 0.35 to harm), is causally inert, moving neither readout at rank 1
($R_{\text{refusal}}(1) = 0.01$, $R_{\text{judgment}}(1) = 0.05$). Variance is not causal
relevance; the harm read is a rank-1 causal object that is not the rank-1 variance object.

Behaviorally, this harm-keyed, saturating read is coherent with OLMo-3 being a weak
intent-refuser. On intent-harmful requests its refusal reaches only about 17% at top severity
(violating items 0/0.17/0/0.17/0.17 across a severity ladder, benign items 0). The operating
band is nearly empty, which is a model property, weak coupling between intent severity and
refusal, that is exactly what a harm-surface-keyed gate predicts, and it is the reason the
cross-model commitment axis in \Cref{cross-model} is measured on Llama and GPT-OSS rather
than on OLMo alone.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_one_knob.pdf}
\caption{The nested rank sweep on OLMo-3, the paper's central causal result. As the moral
basis expands ($k \in \{1, 3, 8, 16\}$), judgment transfer $R_{\text{judgment}}(k)$ climbs
$0.05 \to 0.46 \to 0.59 \to 0.66$ (open markers) while refusal transfer
$R_{\text{refusal}}(k)$ saturates $0.01 \to 0.31 \to 0.26 \to 0.27$ at the harm-rank-1 level
(filled markers); a random-direction null is near zero throughout. The dashed curve is the
one-knob fit $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling} \approx 0.31,
R_{\text{judgment}}(k))$, RMSE 0.036 on the plateau. Refusal reads the harm percept and stops;
judgment reads the subspace broadly on the same patches.}
\label{fig:oneknob}
\end{figure}
