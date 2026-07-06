# 1. Introduction {#introduction}

A causal interpretability program kept
producing results that dissolved under an instrument check. Each of the failures below
looked like a finding first. This note collects the six that turned into portable methods
findings and the estimator and intervention patterns the program
re-derived, and states each as a protocol we found portable within this program and offer
for others to test. The scientific results
(what refusal reads, how it commits) live in the direction papers and the flagship draft;
this note is the portable methodology. Numbers here trace to the program's claim ledger;
every scalar carries its detection bar or its control.

The discipline is four moves: **calibrate the instrument against a positive-control ladder,
certify it with an orthogonal cell, compute power before spending compute, and state every
read-from verdict at a stated depth relative to the model's commitment.** Each section
below takes one instrument, shows the failure as it first appeared, names the tell that
caught it, gives the protocol, and states the check that certifies the fix.

**Notation and definitions.** The *participation ratio* of a position is
$\mathrm{PR} = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$, where the $\lambda_i$ are the
eigenvalues of the residual-stream covariance at that position; it is an effective-dimension
estimate of that covariance, equal to the true dimension for an isotropic covariance and
dropping toward 1 as variance concentrates in a few directions. The *decision site* (or
control-token position) is the assistant-header or end-of-prompt control token where the
refusal gate and the judgment direction are defined; *content positions* are the token
positions carrying the request text. The *moral subspace* is the rank-3 span of the moral
mean-difference directions extracted from the moral-content datasets. The *refusal-decision*
and *judgment-decision* directions are the mean-difference directions for refuse-versus-comply
and for the moral judgment, read at the decision site. The *reconstruction fractions*
$R_{\mathrm{refusal}}$ and $R_{\mathrm{judgment}}$ (written `R_refusal` and `R_judgment` below)
give the share of the full interchange-patch effect on the refusal (respectively judgment)
outcome that a restricted rank-$k$ subspace patch transfers, normalized to $[0,1]$, so $0$ is
no transfer and $1$ is full reconstruction. *Engage* and *disengage* name the two
directions of a content intervention: engage adds harmful content, disengage removes it.
Terms are defined again at first use below.

The six failure modes, with the section that treats each:

1. A projection-fraction instrument reads absence at a position where its own positive control
   has no discriminating power (the band-below-null decision site; §2.1).
2. A massive-activation outlier is a content-position statistic, so the decision-token
   bottleneck it seemed to contaminate is in fact clean (§2.2).
3. A covariance-matched null saturates in massive-activation families until every direction
   projects like a typical one (§2.3).
4. A per-head OV attribution overshoots the true residual write about threefold on
   reordered-norm architectures (§2.4).
5. A deliberation/prefill asymmetry statistic is operating-point-confounded when one arm sits
   at the ceiling (§4.1).
6. A cross-model asymmetry measured at the read layer is an artifact of measuring past the
   layer where one model already committed (§5).

Coverage is uneven across the panel: each mode is established on one or two models, not on all
four, and the per-mode breakdown is in the Limitations section. Table 1 collects the
participation ratios by model, position, and normalization so that a reader can tell which PR
belongs where.

Mechanistic claims about model internals are claims about measurements. "Refusal reads the
harm percept," "this head writes the refusal direction," "moral judgment is orthogonal to
the refusal decision": each is a statement about a projection, a cosine, an ablation delta,
or an interchange patch. The measurement can fail in ways that produce a clean-looking
number. A covariance-matched null can saturate so that every direction projects like a
typical one. A per-head attribution can overshoot the true residual write threefold. An
interchange patch can go sign-chaotic because the outcome it reads is pinned at its ceiling.
A "reads-X" verdict can be an artifact of measuring past the layer where the model already
decided. None of these announce themselves; each returns a plausible scalar.

The discipline in this note was forced by one discovery. Across four architectures, the
decision site (the assistant-header or end-of-prompt control token where the refusal gate
and the judgment direction are defined) is a low-dimensional bottleneck. The participation
ratio there is 14.7 on OLMo-3-7B-Instruct, 8.6 on Qwen2.5-7B, 10.2 on Llama-3.1-8B, and
12.8 on GPT-OSS-20B (a 20B reasoning MoE at its harmony decision token). A 9-to-15
effective-dimensional channel, on every model tested, while content positions at the same
layers are full-rank-healthy (PR 40+/33+/35+). This is a substantive finding about where
the refusal decision lives, and it belongs in the flagship. But it is also the reason the
program's projection-fraction instruments failed: a positive control measured in a 15-slot
channel projects onto its own span *less* than a random direction does, so the instrument
had no discriminating power exactly where the interesting directions live. The finding and
the failure are the same fact seen twice. This note carries the validity protocol the
finding motivated; the flagship carries the finding.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{mn_bottleneck_pr.pdf}
\caption{The decision-site participation ratio across four architectures: OLMo-3-7B-Instruct 14.7, Qwen2.5-7B 8.6, Llama-3.1-8B 10.2, and GPT-OSS-20B 12.8 (a 20B reasoning MoE at its harmony decision token). All four fall below the PR $<$ 30 position-validity gate, while content positions at the same layers stay full-rank-healthy (PR 40+/33+/35+). The refusal decision lives in a 9-to-15 effective-dimensional control-token channel, on every model tested.}
\label{fig:bottleneck-pr}
\end{figure}

(The bottleneck PR bar across the four architectures is **Figure 1**, which uses the
in-format-ladder value 10.2 for Llama, comparable to OLMo 14.7 and Qwen 8.6; the
decision-token measurement for Llama is 13.5, a second position and harness. Both are
below 30.)
