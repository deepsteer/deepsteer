# 7. What refusal reads: the harm percept {#reads-harm}

The causal core is on OLMo-3, where interchange patching [@meng2022rome] on the heads that
write the decision channel resolves *what* the refusal decision reads. The answer is the harm
percept: a mostly-extra-moral harm direction, aligned with a harm direction
[@zhao2025harmfulness], that clips a low-rank corner of the moral subspace. About three-quarters
of refusal's causal input lies off the subspace (the rank sweep below reports 76% outside the
rank-16 basis); it is not the broad subspace that moral judgment reads on the same patches. Moral directions here are causal and foundation-specific to begin with, a preliminary
we established with foundation-wise ablation whose specificity strengthens with depth; the
OLMo-3 interchange cells below sharpen that into a rank sweep that says *which* moral content
refusal uses.

**The write is distributed.** Refusal is written into the ~13-dimensional decision channel by
a set of heads, led by one head (layer 16 head 23) that alone accounts for 11.7% of the total
specificity but does not carry the decision. Cumulative channel-matched specificity reaches
45% at the top ten heads and needs 67 heads to reach 80%. Attention is not the whole story:
multilayer perceptrons contribute 38% of the decision-site write (write fraction 0.384). None
of the top ten writers is a clean harm-copy head; all are labeled neither-moral-nor-harm, with
a moral-subspace fraction of 0.15–0.28 and comparable harm loading. Refusal is written broadly,
not routed through one moral head. The full per-head attribution is in \Cref{app:causal}, and
the exact normalization fold that certifies it (reconstruction 3.05 to 0.9999) is in
\Cref{app:calibration}.

**The interchange is specific to the moral subspace, then specific to harm inside it.** Using
request-twins (matched requests carrying opposite judgment outcomes; the decisive cells below use
the 23 twins of the original run, the rank sweep pools them with a 19-twin replication, $n = 42$), we patch the
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
moral subspace. On the pooled 42 twins, as $k$ grows over $\{1, 3, 8, 16\}$, judgment transfer
climbs $0.05 \to 0.46 \to 0.59 \to 0.66$ while refusal transfer *rises to $k = 3$ (0.27) and then
holds flat at 0.22–0.24* ($0.03 \to 0.27 \to 0.22 \to 0.24$; at $k = 16$, 95% bootstrap CI
[0.13, 0.41]) at the harm-rank-1 level (harm-rank-1 transfer 0.33), with a random-direction null
near zero at every rank and per-rank purity 0.97–0.99. The shape replicates: the original 23
twins alone give $0.01 \to 0.31 \to 0.26 \to 0.27$ (the run of record, the same
`harm_saturating` verdict), the 19 new twins alone give $0.05 \to 0.22 \to 0.16 \to 0.20$ with
the same sign of the judgment-minus-refusal gap but a CI touching zero, and pooling tightens the
plateau interval by a third (width 0.42 to 0.27). Expanding the moral basis beyond harm buys more
judgment coupling and no more refusal coupling. This is the central result, and
\Cref{fig:oneknob} plots it: refusal reads the harm percept and stops; judgment keeps reading as
the subspace widens. About 76% of refusal's causal twin-difference input lies outside the
rank-16 moral basis (73% already at the rank-3 peak). Judgment reads two-thirds of the subspace
patch effect (0.66) *on the same patches*, which is the within-model proof that the content is
there to be read; refusal simply does not read it. The per-rank gap
$R_{\text{judgment}}(k) - R_{\text{refusal}}(k)$ is not itself given a confidence interval, and the
one interval we compute on this contrast (the restricted-to-full transfer difference, 0.21 on the
pooled run) has a bootstrap 95% CI [$-0.07$, 0.39] that includes 0 at $n = 42$
(\Cref{app:interchange}), as the pre-registered power table said it would at this count. The shape
claim rests on the replicated plateau, whose own interval excludes both zero and the judgment
curve, not on a gap-CI.

**One free parameter fits the sweep.** The refusal curve is the judgment curve clipped at a
harm ceiling: $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling}, R_{\text{judgment}}(k))$,
with the ceiling 0.25 on the pooled twins (0.28 on the original 23, 0.19 on the new 19). This
one-knob model fits the pooled sweep at RMSE 0.023 (0.022 on the plateau, $k \geq 3$), well below
the harm-amplitude alternatives (full residuals and alternatives in \Cref{app:causal}). The one
place it strains is rank 1, where it over-predicts: the highest-variance contrast component, the
most harm-aligned single direction (variance purity 0.974, cosine 0.35 to harm), is nearly inert,
moving neither readout at rank 1 ($R_{\text{refusal}}(1) = 0.03$, $R_{\text{judgment}}(1) = 0.05$). Variance is not causal
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
\caption{The nested rank sweep on OLMo-3, the paper's central result, replicated and pooled (a
per-rank difference-CI on the refusal-minus-judgment gap is not computed; \Cref{reads-harm}). As the moral
basis expands ($k \in \{1, 3, 8, 16\}$), judgment transfer $R_{\text{judgment}}(k)$ climbs
$0.05 \to 0.46 \to 0.59 \to 0.66$ (open markers) while refusal transfer
$R_{\text{refusal}}(k)$ rises to $k = 3$ (0.27) and then holds flat at 0.22–0.24
($0.03 \to 0.27 \to 0.22 \to 0.24$, pooled $n = 42$; bars are 95\% bootstrap intervals over twins)
at the harm-rank-1 level (filled markers); a random-direction null is near zero throughout. The
dashed curve is the one-knob fit $R_{\text{refusal}}(k) \approx \min(\text{harm ceiling} = 0.25,
R_{\text{judgment}}(k))$, RMSE 0.023. Refusal reads the harm percept and stops;
judgment reads two-thirds of the subspace patch effect (0.66) on the same patches. Regenerable from committed data
(\Cref{app:repro}).}
\label{fig:oneknob}
\end{figure}
