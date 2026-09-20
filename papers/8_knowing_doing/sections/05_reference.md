# 5. The gap survives the strictest reference on the log-prob readout {#reference}

A self-referenced gap can be manufactured by a noisy reference, and the floor says the reference
is noisy: the greedy judgment changes under paraphrase on 30% of scenarios by option and 20% on
the violating versus non-violating binary, and 74 of 208 gate-family primaries fail the
sampled-stability rule, 58 of them real consistent-to-violating flips rather than splits between
a consistent and a neutral option. This was the rival reading the pilot could not separate (its
paraphrase-stable subset halved the rate at 18 scenarios), and the strictness ladder of
\Cref{instruments} is the instrument built to separate it.

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_strictness.pdf}
\caption{\textbf{Reference strictness.} Paired excess of the gap over the pressure-removed null at
the three reference levels: L0, the original frame with sampled stability; L1, plus four-frame
majority on the violating versus non-violating binary; L2, plus all four frames naming the same
option. (a) Binary readout; (b) continuous readout. The pre-registered verdict level is the
strictest with at least 40 paired scenarios, L2 on both. The binary excess loses power as the
scenario set shrinks; the continuous excess grows.}
\label{fig:strictness}
\end{figure}

\Cref{fig:strictness} gives the answer, and it differs by instrument in an informative way. On
the binary readout the excess is 0.10 (0.02 to 0.18) at L0, 0.11 (0.02 to 0.19) at L1, and 0.05
($-0.02$ to 0.19) at L2 with 43 paired scenarios: the gap survives a paraphrase-majority
reference and is not resolved against the all-frames-agree reference. On the continuous readout
the excess is 0.054 (0.021 to 0.086), 0.063 (0.030 to 0.096), and 0.079 (0.029 to 0.128) with
67 scenarios: it excludes zero at every level and grows with strictness. Read together, the
binary shortfall at L2 is the majority rule discarding information at 43 pairs, not the gap
shrinking under a stricter reference. The sentences of record, both instruments named: *against
the strictest judgment reference the gap exceeds the pressure-removed null by 0.079 (0.029 to
0.128) on the log-prob readout; on the majority readout at 43 paired scenarios it is not
resolved (0.05, $-0.02$ to 0.19).* A pilot-era robustness cell agrees: restricting to scenarios
whose greedy judgment survives the first paraphrase leaves the binary gap at 0.17 (0.11 to 0.28;
90 scenarios) against 0.19 on the full screened panel, where the 96-scenario pilot had halved it
at 18 scenarios.

\begin{table}[tbp]
\centering
\caption{The strictness ladder decomposed on the continuous readout: mean violating-option mass
when acting and when judging, on the primaries and on their pressure-removed twins, at each
level.}
\label{tab:strictness}
\small
\begin{tabular}{@{}lrrrrrrrl@{}}
\toprule
level & $n$ & $p_D$ & $p_J$ & $p_D$ (twin) & $p_J$ (twin) & $g$ & $g$ (twin) & paired excess \\
\midrule
L0 & 136 & 0.44 & 0.26 & 0.28 & 0.16 & 0.18 & 0.12 & 0.054 [0.021, 0.086] \\
L1 & 127 & 0.43 & 0.24 & 0.28 & 0.16 & 0.18 & 0.12 & 0.063 [0.030, 0.096] \\
L2 & 67 & 0.41 & 0.18 & 0.26 & 0.11 & 0.23 & 0.15 & 0.079 [0.029, 0.128] \\
\bottomrule
\end{tabular}
\end{table}

\Cref{tab:strictness} shows why the excess grows. Tightening the reference selects scenarios
the model judges decisively, so the judging mass on the violating option falls from 0.26 to
0.18; the acting mass barely moves (0.44 to 0.41), so the gap opens. The twins' judging mass
falls too, from 0.16 to 0.11, but by less, so the paired excess rises. The rival that the
strictest level selects the most tempting scenarios is checked by the acting mass itself, which
does not rise from L0 to L2. The gap is largest where the model's judgment is least ambiguous,
which is the opposite of what reference noise would produce, and the reference-noise rival is
separated on the continuous instrument at every level.

What the ladder also shows is that the stated judgment of a 7B instruct model on these scenarios
is often not decisive: half of the screened scenarios do not survive the all-frames-agree
reference. That is a finding about the reference, and a reason to build the next panel's
scenarios to be decisive by construction, so that the strictest level keeps its power on the
majority readout as well.
