# 4. The model acts against its own judgment, above its null {#gap}

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_ladder.pdf}
\caption{\textbf{The calibration ladder on the panel of record} (248 primaries, OLMo-3-7B-Instruct,
136 screened scenarios). (a) The binary majority readout: matched null on the pressure-removed
twins, the measurement, and the known-gap band. (b) The continuous readout $g=p_D-p_J$ with its
floor, the unsigned shift of the judgment mass under paraphrase. On both instruments the
measurement sits above its null and well below the band; the paired excess over the null is the
number the paper uses. Bars are 95\% bootstrap CIs over scenarios (nested over rollouts on the
binary readout); $n$ is the number of scenarios with a defined statistic.}
\label{fig:ladder}
\end{figure}

\begin{figure}[tbp]
\centering
\includegraphics[width=0.8\linewidth]{kdg_scatter.pdf}
\caption{\textbf{Every screened scenario.} Violating-option mass when judging ($p_J$, mean over
four frames) against violating-option mass when acting ($p_D$, mean over 32 rollouts), one point
per screened scenario, marked by family; the same scenarios' pressure-removed twins in gray.
Points above the diagonal act more violating than they judge. The cloud of primaries sits above
the cloud of twins by the paired excess of 0.054; the twins themselves sit above the diagonal by
the deliberation asymmetry of \Cref{instruments}. Regenerable from the released per-scenario
table.}
\label{fig:scatter}
\end{figure}

\Cref{fig:ladder} is the paper's calibration figure and \Cref{fig:scatter} its data figure. On
the screened panel of record (136 scenarios, 126 with a defined majority gap) the binary gap is
0.19 (95% CI 0.13 to 0.28), the pressure-removed null 0.10 (0.06 to 0.17), and the known-gap
band 0.58 (0.51 to 0.65). The paired excess of the measurement over the null is 0.10 (0.02 to
0.18) on 100 scenarios, and the band sits 0.38 (0.24 to 0.48) above the measurement, so the
observed rate is neither frame change alone nor instruction-following saturation. On the
continuous readout the same rungs read 0.12 (null), 0.18 (measurement), and 0.60 (band), with a
paired excess of 0.054 (0.021 to 0.086) on 136 scenarios; the floor, the unsigned shift of the
judgment mass under paraphrase, is 0.14. Mass on the violating option is 0.44 when acting and
0.26 when judging. On the twins, with the incentive removed, it is 0.28 and 0.16: even with
nothing at stake the immediate action puts more mass on the violating option than the deliberated
judgment does, which is the deliberation asymmetry named in \Cref{instruments}, and the incentive
adds 0.05 to that difference.

**Two second derivations.** At the rollout level, on screened scenarios whose stable judgment
named a non-violating option, the model takes the violating action on 0.38 of rollouts; a
majority-rule rate of 0.19 from per-scenario fractions averaging 0.38 is what a mixed panel
predicts. Across instruments, the binary excess predicted from the continuous values (the net
fraction of paired scenarios whose acting mass crosses 0.5 between the twin and the primary) is
0.14 (0.07 to 0.22) on 88 scenarios, against the observed 0.10 (0.03 to 0.17) on the same
scenarios; 20 scenarios cross upward and one crosses downward. The two instruments agree on sign
and size, and the majority readout is, as expected, the noisier of the two.

**The floor and the null are one quantity in two units.** The binary floor is an agreement rate
(the greedy judgment reproduces itself on a paraphrased frame on 0.70 of scenarios by option,
0.80 by the violating versus non-violating binary), not a gap rate; the pressure-removed null is
what that re-elicitation noise and the frame change together produce in gap units, 0.10. On the
continuous instrument the floor (0.14) is an unsigned noise scale and the excess (0.054) a
signed paired difference against a twin that shares the judgment noise; the two are on different
footings and are not compared.

**What the excess is and is not.** The absolute rate (0.19 on a three-option forced choice with
a neutral option counted as non-violating) is not interpretable alone; the number the paper uses
is the paired excess over twins that hold the same options and the same neutral tag with the
incentive deleted. \Cref{tab:bias} names the known biases and their directions.

| known bias | mechanism | direction relative to "gap above null" |
|---|---|---|
| reference noise | a scenario the model is undecided about lands non-violating by chance while its action is a coin flip | favors the claim; bounded by the strictness ladder (\Cref{reference}) |
| the screen selects mixed scenarios | keeping violating fractions in $[0.15, 0.85]$ makes a majority-violating action a near coin flip on many scenarios | favors; removed in the paired null, which runs on the same scenarios' twins |
| nested rollout resampling | resampling 32 rollouts near a 0.5 fraction flips majorities | opposes; widens every binary CI |
| neutral counts as non-violating | a judgment of "hold" against a violating action counts as a gap | neutral; a real contradiction |
| deliberation asymmetry | the judgment is read after reasoning, the action immediately | absorbed by the null; visible in the null's size |

Table: Bias-direction audit for the headline excess. {#tab:bias}
