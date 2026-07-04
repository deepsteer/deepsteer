# 1. Introduction: the instrument problem, and the discovery that motivated it {#introduction}

*Methods note. Started 2026-07-02 (Direction-3 Amendment 10); drafted to full text 2026-07-03.*

A causal interpretability program (Directions D1–D3, plus the earlier Papers 1–7) kept
producing results that dissolved under an instrument check. Each of the failures below
looked like a finding first. This note collects the six that turned into portable methods
findings (`ANOMALIES.md`, A1–A6) and the estimator and intervention patterns the program
re-derived, and states each as a protocol other people can run. The scientific results
(what refusal reads, how it commits) live in the direction papers and the flagship draft;
this note is the portable methodology. Numbers here trace to the program's claim ledger
(`CLAIMS.md`); every scalar carries its detection bar or its control.

The discipline is four moves: **calibrate the instrument against a positive-control ladder,
certify it with an orthogonal cell, compute power before spending the pod, and state every
read-from verdict at a stated depth relative to the model's commitment.** Each section
below takes one instrument, shows the failure as it first appeared, names the tell that
caught it, gives the protocol, and states the check that certifies the fix.

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
12.79 on GPT-OSS-20B (a 20B reasoning MoE at its harmony decision token). A 9-to-15
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
\caption{The decision-site participation ratio across four architectures: OLMo-3-7B-Instruct 14.7, Qwen2.5-7B 8.6, Llama-3.1-8B 10.2, and GPT-OSS-20B 12.79 (a 20B reasoning MoE at its harmony decision token). All four fall below the PR $<$ 30 position-validity gate, while content positions at the same layers stay full-rank-healthy (PR 40+/33+/35+). The refusal decision lives in a 9-to-15 effective-dimensional control-token channel, on every model tested.}
\label{fig:bottleneck-pr}
\end{figure}

(The bottleneck PR bar across the four architectures is **Figure 1**, which uses the D2
in-format-ladder value 10.2 for Llama, comparable to OLMo 14.7 and Qwen 8.6; the D3 C1
decision-token measurement for Llama is 13.5, a second position and harness. Both are
below 30.)
