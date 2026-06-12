# Permutation Tests for MFT Group Structure

\label{app:permutation}

We test whether the six foundation probe directions cluster into the
MFT-predicted individualizing (care, fairness, liberty) and binding
(loyalty, authority, sanctity) groups. The test statistic is the
difference between mean within-group cosine similarity and mean
between-group cosine similarity. With six foundations split into two
groups of three, there are only $\binom{6}{3} = 20$ distinct group
assignments, so we enumerate the null distribution exactly rather than
resampling; the $p$-value is the fraction of the 20 partitions whose
statistic is $\geq$ the observed value, and is therefore an exact
multiple of $1/20$.

\begin{table}[h]
\centering
\caption{Exact permutation test for individualizing/binding group structure across layers, OLMo-2 1B (enumeration over all 20 partitions). No layer reaches significance.}
\label{tab:permutation}
\small
\begin{tabular}{r cc}
\toprule
Layer & Observed statistic & $p$-value \\
\midrule
0  &  0.001 & 0.40 \\
1  & $-$0.014 & 0.80 \\
2  & $-$0.003 & 0.70 \\
3  & $-$0.004 & 0.80 \\
4  & $-$0.004 & 0.80 \\
5  &  0.003 & 0.50 \\
6  & $-$0.002 & 0.50 \\
7  &  0.005 & 0.40 \\
8  &  0.001 & 0.50 \\
9  & $-$0.003 & 0.55 \\
10 & $-$0.003 & 0.60 \\
11 & $-$0.014 & 0.80 \\
12 & $-$0.004 & 0.65 \\
13 &  0.003 & 0.40 \\
14 &  0.000 & 0.40 \\
15 &  0.007 & 0.40 \\
\bottomrule
\end{tabular}
\end{table}

The test does not reach significance at any layer (minimum $p = 0.40$).
Exact enumeration can attain $p = 1/20 = 0.05$ when the observed split
is the single most extreme of the 20, so significance was reachable;
it simply was not observed. The observed statistics are near zero and frequently
negative (within-group similarity $<$ between-group similarity),
confirming that the model's inter-framework geometry is not
organized along the MFT individualizing/binding axis. This is
consistent with the dendrogram analysis (\S4.3), which shows
cross-MFT pairings (care--sanctity, liberty--authority) rather
than the predicted group structure.
