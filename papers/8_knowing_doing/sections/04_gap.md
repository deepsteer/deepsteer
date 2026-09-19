# 4. The gap {#gap}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{kdg_ladder.pdf}
\caption{\textbf{Calibration ladder on the panel of record} (248 primaries, OLMo-3-7B-Instruct;
KDG-2 and KDG-3 unioned). Left: the binary majority readout; right: the continuous log-prob
readout ($g=p_D-p_J$). Gray marks the pressure-removed null. The measurement sits above the null
and well below the known-gap band on both instruments. Bars are 95\% bootstrap CIs over
scenarios (nested over rollouts on the binary readout); $n$ is the number of scenarios with a
defined statistic.}
\label{fig:ladder}
\end{figure}

\Cref{fig:ladder} is the paper's central figure. On the screened panel of record the binary gap
is 0.19 (95% CI 0.13 to 0.28; 126 defined scenarios), the pressure-removed null 0.10 (0.06 to
0.17), and the known-gap band 0.58 (0.51 to 0.65). The paired excess of the measurement over the
null is 0.10 (0.02 to 0.18) on 100 scenarios. A second derivation at the rollout level agrees:
on screened scenarios whose stable judgment named a non-violating option, the model takes the
violating action on 0.38 of rollouts; a majority-rule rate of 0.19 from per-scenario fractions
averaging 0.38 is what a mixed panel predicts. The band minus the measurement is 0.38 (0.24 to
0.48), so the observed rate is not instruction-following saturation. On the continuous readout
the same rungs read 0.12 (null), 0.18 (measurement), 0.60 (band), with a paired excess of 0.054
(0.021 to 0.086) on 136 scenarios; mass on the violating option is 0.44 when acting and 0.26
when judging.

**Reference noise.** The floor is not small: the greedy judgment changes under paraphrase on 30%
of scenarios by option and 19% on the violating/non-violating binary, and 60 of 208 gate-family
primaries fail the sampled-stability rule (48 of them are real consistent-to-violating flips,
not splits between a consistent and a neutral option). A noisy reference can manufacture a gap:
a scenario the model is undecided about lands non-violating by chance while its action is a
coin flip. This was the rival reading the pilot could not separate, and the strictness ladder is
the instrument built to separate it.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{kdg_strictness.pdf}
\caption{\textbf{Reference strictness.} Paired excess of the gap over the pressure-removed null at
the three reference levels (L0: original frame with sampled stability; L1: plus four-frame
binary majority; L2: plus all four frames naming the same option). Left, binary readout; right,
continuous readout. The pre-registered verdict level is the strictest with at least 40 paired
scenarios, L2 on both.}
\label{fig:strictness}
\end{figure}

\Cref{fig:strictness} gives the answer, and it differs by instrument in an informative way. On
the binary readout the excess is 0.10 (0.02 to 0.18) at L0, 0.11 (0.02 to 0.19) at L1, and
0.05 (−0.02 to 0.19) at L2 with 43 paired scenarios: the gap survives a paraphrase-majority
reference and is not resolved against the all-frames-agree reference. On the continuous readout
the excess is 0.054 (0.021 to 0.086), 0.063 (0.030 to 0.096), and 0.079 (0.029 to 0.128) with
67 scenarios: it excludes zero at every level and grows with strictness. Read together, the
binary shortfall at L2 is the majority rule discarding information at 43 pairs, not the gap
shrinking under a stricter reference. The sentences of record, both instruments named: *against
the strictest judgment reference the gap exceeds the pressure-removed null by 0.079 (0.029 to
0.128) on the log-prob readout; on the majority readout at 43 paired scenarios it is not
resolved (0.05, −0.02 to 0.19).* A pilot-era robustness cell agrees: restricting to scenarios
whose greedy judgment survives the first paraphrase leaves the binary gap at 0.17 (0.11 to 0.28;
90 scenarios) against 0.19 on the full screened panel; in the 96-scenario pilot the same cell had
halved the rate, at 18 scenarios, which was noise.

**What the excess is and is not.** The absolute rate (0.19 on a three-option forced choice with
a neutral option counted as non-violating) is not interpretable alone; the number the paper
uses is the paired excess over twins that hold the same options and the same neutral tag with
the incentive deleted. Two biases favor the claim and are named: reference noise (bounded above
by the ladder just described) and the screen's selection of mixed scenarios (removed in the
paired null, which runs on the same scenarios' twins). One opposes it: nested resampling of 32
rollouts near a 0.5 fraction flips majorities and widens every binary CI. Neutral-as-non-violating
(A11) is neutral: a model that judges "hold" right and acts violating is inconsistent.
