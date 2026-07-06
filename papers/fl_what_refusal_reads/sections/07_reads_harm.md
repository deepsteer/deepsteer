# 7. What refusal reads: the harm percept {#reads-harm}

The causal core is on OLMo-3, where interchange patching [@meng2022rome] on the heads that
write the decision channel resolves *what* the refusal decision reads. The answer is the harm
percept: a mostly-extra-moral harm direction, aligned with a harm direction
[@zhao2025harmfulness], that clips a low-rank corner of the moral subspace. About three-quarters
of refusal's causal input lies off the subspace (the rank sweep below reports 73% outside the
rank-16 basis); it is not the broad subspace that moral judgment reads on the same patches. Moral directions here are causal and foundation-specific to begin with, a preliminary
we established with foundation-wise ablation whose specificity strengthens with depth; the
OLMo-3 interchange cells below sharpen that into a rank sweep that says *which* moral content
refusal uses.

**The write is distributed.** Refusal is written into the ~13-dimensional decision channel by
a set of heads, led by one head (layer 16 head 23) that alone accounts for 11.6% of the total
specificity but does not carry the decision. Cumulative channel-matched specificity reaches
44% at the top ten heads and needs about 62 heads to reach 80%. Attention is not the whole story:
multilayer perceptrons contribute 38% of the decision-site write (write fraction 0.384). None
of the top ten writers is a clean harm-copy head; all are labeled neither-moral-nor-harm, with
a moral-subspace fraction of 0.15–0.28 and comparable harm loading. Refusal is written broadly,
not routed through one moral head. The full per-head attribution is in \Cref{app:causal}, and
the exact normalization fold that certifies it (reconstruction 3.05 to 0.9999) is in
\Cref{app:calibration}.

**The interchange is specific to the moral subspace, then specific to harm inside it.** Using
request-twins (matched requests carrying opposite judgment outcomes, $n = 23$), we patch the
decision channel and read the induced change in the refusal and judgment projections (refusal
minimum detectable effect 0.0238; the full decisive-cell table is in \Cref{app:causal}). The
moral subspace is a *specific* substrate: restricting the patch to it moves refusal more than
a random rank-3 patch does ($\Delta = 0.031$, paired 95% CI [0.020, 0.043], excludes 0). Almost
all of that specific effect is the harm slice. The harm-restricted patch nearly equals the full
moral-subspace patch, and the harm-partialed patch (the moral subspace with the harm direction
projected out) still moves refusal about half as much ($-0.0133$, 95% CI [$-0.023$, $-0.005$],
excludes 0), though this point estimate is below the refusal interchange minimum detectable
effect of 0.0238, so it sits at or near the detection limit. So refusal is harm-dominant with a
small non-harm residual at the detection limit; the harm direction captures a fraction 0.46 of
the moral subspace.

**The rank sweep shows a monotone point-estimate divergence.** For a readout $r$ (the refusal or the judgment projection),
define the restricted-transfer coefficient $R_r(k)$ as the fraction of the full interchange
effect on $r$ that is reproduced when the patch is confined to the top-$k$ directions of the
moral subspace. As $k$ grows over $\{1, 3, 8, 16\}$, judgment transfer climbs
$0.05 \to 0.46 \to 0.59 \to 0.66$ while refusal transfer *peaks at $k = 3$ (0.31) and then holds
flat at 0.26–0.27* ($0.01 \to 0.31 \to 0.26 \to 0.27$) at the harm-rank-1 level (harm-rank-1
transfer 0.31), with a
random-direction null near zero at every rank and per-rank purity 0.97–0.99. Expanding the
moral basis beyond harm buys more judgment coupling and no more refusal coupling. This is the
central point-estimate result, and \Cref{fig:oneknob} plots it: refusal reads the harm percept
and stops; judgment keeps reading as the subspace widens. About 73% of refusal's causal twin-difference
input lies outside the rank-16 moral basis (69% already at the rank-3 peak). Judgment reads
two-thirds of the subspace patch effect (0.66) *on the same patches*, which is the within-model
proof that the content is there to be read; refusal simply does not read it. The sweep
coefficients $R_r(k)$ are point estimates: we do not report a per-rank confidence interval on
the refusal-minus-judgment gap, and the one interval we do compute on this contrast (the
restricted-to-full transfer difference, 0.18) has a bootstrap 95% CI [$-0.24$, 0.39] that
includes 0 at $n = 23$ (\Cref{app:interchange}). The shape claim rests on the monotone
divergence of the point estimates, not on a gap-CI that excludes 0.

**One free parameter fits the sweep.** The refusal curve is the judgment curve clipped at a
harm ceiling: $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling}, R_{\text{judgment}}(k))$,
with the ceiling $\approx 0.31$. This one-knob model fits the plateau ($k \geq 3$) at RMSE
0.036, well below the harm-amplitude alternatives (full residuals and alternatives in
\Cref{app:causal}). The one place it breaks is rank 1, where it over-predicts: the
highest-variance contrast component, the most harm-aligned single direction (variance purity
0.974, cosine 0.35 to harm), is causally inert, moving neither readout at rank 1
($R_{\text{refusal}}(1) = 0.01$, $R_{\text{judgment}}(1) = 0.05$). Variance is not causal
relevance; the harm read is a rank-1 causal object that is not the rank-1 variance object.

Behaviorally, this harm-keyed, saturating read is coherent with OLMo-3 being a weak
intent-refuser. On intent-harmful requests its refusal reaches only about 17% at top severity
(violating items 0/0.17/0/0.17/0.17 across a severity ladder, benign items 0). The operating
band is nearly empty: intent severity and refusal are weakly coupled, which is exactly what a
harm-surface-keyed gate predicts. That weak coupling is why the cross-model commitment axis in
\Cref{cross-model} is measured on Llama and GPT-OSS rather than on OLMo alone.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_one_knob.pdf}
\caption{The nested rank sweep on OLMo-3, the paper's central point-estimate divergence (a
per-rank difference-CI on the refusal-minus-judgment gap is not computed; \Cref{reads-harm}). As the moral
basis expands ($k \in \{1, 3, 8, 16\}$), judgment transfer $R_{\text{judgment}}(k)$ climbs
$0.05 \to 0.46 \to 0.59 \to 0.66$ (open markers) while refusal transfer
$R_{\text{refusal}}(k)$ peaks at $k = 3$ (0.31) and then holds flat at 0.26–0.27
($0.01 \to 0.31 \to 0.26 \to 0.27$) at the harm-rank-1 level
(filled markers); a random-direction null is near zero throughout. The dashed curve is the
one-knob fit $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling} \approx 0.31,
R_{\text{judgment}}(k))$, RMSE 0.036 on the plateau. Refusal reads the harm percept and stops;
judgment reads two-thirds of the subspace patch effect (0.66) on the same patches. Regenerable from committed data
(\Cref{app:repro}).}
\label{fig:oneknob}
\end{figure}
