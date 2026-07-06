# Dilemma subspace membership: matched vs. mismatched baseline

\label{app:dilemma_membership}

Per-layer mean subspace membership of the 15 dilemma directions on
OLMo-2 1B, averaged over pairs. **Matched** is membership in the 2D
span of each dilemma's own two component foundation directions;
**mismatched** is the mean membership in the 2D spans of all foundation
pairs that share no component with the dilemma (the correct null, since
it absorbs the shared moral-salience component that the random-vector
null, ${\sim}0.001$, does not). The matched value exceeds the mismatched
baseline at every layer.

\begin{table}[h]
\centering
\caption{Per-layer matched vs. mismatched dilemma subspace membership, OLMo-2 1B (mean over 15 dilemma pairs). Matched directions are seed-averaged probe-weight directions; mismatched is averaged over the foundation pairs sharing no component.}
\label{tab:dilemma_membership}
\small
\begin{tabular}{r ccc}
\toprule
Layer & Matched & Mismatched & Gap \\
\midrule
0 & 0.0664 & 0.0313 & +0.0351 \\
1 & 0.0723 & 0.0334 & +0.0389 \\
2 & 0.0861 & 0.0450 & +0.0411 \\
3 & 0.0939 & 0.0525 & +0.0414 \\
4 & 0.1007 & 0.0507 & +0.0500 \\
5 & 0.1057 & 0.0494 & +0.0563 \\
6 & 0.1058 & 0.0480 & +0.0579 \\
7 & 0.0968 & 0.0433 & +0.0535 \\
8 & 0.0944 & 0.0392 & +0.0552 \\
9 & 0.0905 & 0.0380 & +0.0524 \\
10 & 0.0831 & 0.0370 & +0.0461 \\
11 & 0.0913 & 0.0338 & +0.0574 \\
12 & 0.0907 & 0.0295 & +0.0612 \\
13 & 0.0912 & 0.0296 & +0.0617 \\
14 & 0.0939 & 0.0314 & +0.0624 \\
15 & 0.0991 & 0.0330 & +0.0662 \\
\bottomrule
\end{tabular}
\end{table}

Cross-layer means: matched 0.091, mismatched 0.039 (${\sim}2.3\times$;
paired-bootstrap gap 0.052, CI $[0.037, 0.069]$, excluding 0);
per-pair-peak means: matched 0.118, mismatched 0.044 (${\sim}2.7\times$;
gap 0.074, CI $[0.053, 0.100]$, excluding 0, but a max-over-layers
extremum and biased upward, so the cross-layer-mean gap is the unbiased
figure). Both bootstraps resample the 15 dilemmas ($n = 10^4$).
The same matched-over-mismatched margin replicates on OLMo-2 7B
(peak 0.090 vs.\ 0.032) and OLMoE-1B-7B (peak 0.118 vs.\ 0.045).
