# 5. The decision site is a control-token bottleneck {#bottleneck}

The refusal decision lives at a control-token bottleneck, an 8-to-15 effective-dimensional
channel that holds on every architecture we test. This structural fact about the decision
site is also why content-versus-decision orthogonality is so easy to find there.

The decision site is the control token where the chat template hands off to the model's
answer: the token before the assistant header on the instruct models, the end-of-prompt token
in the reasoning model's harmony format. This is where the refusal gate and the judgment
direction are defined, because it is the last position the model reads before it commits to a
reply. At that token the residual stream is a low-dimensional bottleneck. Its participation
ratio is 14.7 on OLMo-3-7B-Instruct, 8.6 on Qwen2.5-7B, 10.2 on Llama-3.1-8B, and 12.8 on
GPT-OSS-20B, an 8-to-15 effective-dimensional channel on all four models. Content positions at
the same layers are full-rank-healthy by comparison (participation ratio above 40 on OLMo,
above 33 on Qwen, above 35 on Llama). \Cref{fig:bottleneck} plots the four decision-site
values against the position-validity gate. (The Llama value of record is 10.2, measured on the
in-format ladder and directly comparable to OLMo's 14.7 and Qwen's 8.6; a separate
decision-token harness reads 13.5 at a second position. Both are far below 30.)

This narrowness is the reason a projection-fraction instrument fails at the decision site, and
it is also a substantive fact about where the decision lives. At the OLMo-3 decision token the
positive-control moral band comes out at [0.40, 0.47], *below* the covariance-matched null of
0.557. A positive control below the null means the instrument has no discriminating power at
that position: any direction, moral or not, projects onto a 15-slot channel at roughly the
null level, so a low projection there cannot certify that a direction is absent from the
subspace. Three independent estimates agree that the channel carries about 15 effective
dimensions: $\sqrt{3/14.7} = 0.45$ as a closed-form projection expectation, the covariance
null q95 of 0.557, and the pairwise-cosine null of 0.41–0.51. Because of this, participation
ratio is a required field on every extracted direction, and any position below 30 is flagged
invalid for content projection-fraction tests. The position-validity protocol, its
positive-control ladder, and the covariance-matched nulls are in \Cref{app:calibration}.

One reconciling sentence is needed before the next section, because the bottleneck cuts two
ways. It is position-invalid *for content projection-fraction tests* (the band-below-null
tell), but position-valid *for decision-direction reads*: a cosine between two directions both
defined at the decision token is immune to the projection null, and the GPT-OSS refusal
projection is read at a decision channel that passes its own validity gate at participation
ratio 12.8. So the bottleneck does not block the comparison the next section makes; it blocks
only the content-projection comparison, and it does so on all four architectures.

The structural consequence is the setup for the causal work. Content and the decision do not
co-locate: moral content is readable at content positions (healthy participation ratio there)
and unreadable at the decision channel (band-below-null), while the refusal and judgment
directions live only at the decision channel. They never coexist at one valid position.
Content-versus-decision orthogonality is therefore architecturally favored, not a discovered
surprise, and any coupling between comprehension and the decision has to ride the attention
heads that write into the bottleneck. That is a concrete anatomical target, and
\Cref{reads-harm} pursues it.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fl_bottleneck_pr.pdf}
\caption{The decision-site participation ratio across four architectures: OLMo-3-7B-Instruct
14.7, Qwen2.5-7B 8.6, Llama-3.1-8B 10.2, and GPT-OSS-20B 12.8 (a 20B reasoning
mixture-of-experts at its harmony decision token). All four fall below the participation-ratio
30 position-validity gate, while content positions at the same layers stay full-rank-healthy
(above 40/33/35). The refusal decision lives in an 8-to-15 effective-dimensional control-token
channel on every model tested.}
\label{fig:bottleneck}
\end{figure}
