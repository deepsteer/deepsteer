# 7. The base model already carries the gap; post-training enlarges its pressure-attributable part {#base}

Base-versus-instruct is the question of origin: a gap present in base weights is inherited from
pretraining; a gap absent in base and present in instruct was installed by post-training. The
comparison is well-posed only in a common frame, so the panel measures the action and the
judgment in a raw completion frame (scenario, option list, fixed prefix ending before the option
token; the readout is the option-letter probability mass, eight option orders) on both models,
with the chat-template cell as a third, format-changing cell that is never compared to base
directly. Each model's pressure-removed raw twins give it a matched null in the same frame, so
the quantity compared across models is the **pressure-attributable excess** $E$, the paired
difference between the acting-versus-judging mass gap on a scenario and on its twin. The verdict
rules, the readouts, and the sub-branch wording were fixed in a dated amendment before any of the
numbers below were computed (\Cref{app:prereg}); the full tables are in \Cref{app:three-cell}.

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_three_cell.pdf}
\caption{\textbf{Base versus instruct in the raw completion frame}, on the 192 scenarios where
both models clear the mass floor on the primary and on the twin. (a) Violating-option mass when
acting and when judging, on the pressure-removed twin (open markers) and under pressure (filled),
per model: the incentive raises the acting mass more on the instruct model, and the instruct
model starts lower. (b) The paired quantities with 95\% CIs: each model's pressure-attributable
excess $E$, its acting-side and judging-side components, and the base-minus-instruct
differences. The excess is present in both models and larger after post-training, on the acting
side; the net gap under pressure is smaller after post-training because the baseline moved.}
\label{fig:three-cell}
\end{figure}

\begin{table}[tbp]
\centering
\caption{The three-cell weights contrast in the raw completion frame. Rows two to four are on each
model's own above-floor scenarios; the shared-scenario rows are paired on the 192 (continuous) or
171 (binary) scenarios where both models clear the floor on the primary and on the twin. Bracketed
values are 95\% bootstrap CIs over scenarios.}
\label{tab:three-cell}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.24\linewidth}ll>{\raggedright\arraybackslash}p{0.21\linewidth}@{}}
\toprule
raw frame, continuous readout & base & instruct & paired base $-$ instruct \\
\midrule
scenarios above the floor (of 397) & 359 & 242 & 225 shared; 192 with twins \\
$g$ under pressure & 0.041 [0.034, 0.049] & 0.002 [$-$0.022, 0.024] & 0.033 [0.012, 0.054] \\
$g$ on the pressure-removed twin & 0.024 [0.017, 0.030] & $-$0.038 [$-$0.059, $-$0.015] & \\
excess $E$ (paired, all above floor) & 0.017 [0.012, 0.022] & 0.039 [0.016, 0.061] & \\
excess $E$ on the shared scenarios & 0.018 [0.011, 0.025] & 0.046 [0.025, 0.069] & $-$0.028 [$-$0.049, $-$0.007] \\
acting side, $p_D - p_D^{\mathrm{twin}}$ & 0.049 [0.037, 0.059] & 0.085 [0.057, 0.114] & $-$0.037 [$-$0.062, $-$0.012] \\
judging side, $p_J - p_J^{\mathrm{twin}}$ & 0.030 [0.022, 0.039] & 0.039 [0.024, 0.056] & $-$0.009 [$-$0.023, 0.004] \\
\addlinespace
binary gap rate (shared) & 0.146 [0.099, 0.192] & 0.099 [0.061, 0.136] & 0.047 [$-$0.009, 0.103] \\
binary excess (paired) & 0.024 [$-$0.021, 0.065] & 0.051 [0.010, 0.097] & $-$0.012 [$-$0.076, 0.053] \\
\bottomrule
\end{tabular}
\end{table}

**Present in base.** In the raw frame the base model puts more mass on the violating option when
acting than when judging, 0.041 (0.034 to 0.049) on 359 scenarios, and 0.024 (0.017 to 0.030)
on the pressure-removed twins: a second-person frame alone moves the base model toward the
violating option. The pressure-attributable excess, 0.017 (0.012 to 0.022), excludes zero. The
base model's binary shadow of the same quantity does not resolve (0.024, $-0.021$ to 0.065),
which is the majority rule at work on a small mass effect, not a disagreement in sign.

**Not removed; enlarged, on the acting side.** On the instruct model the same excess is 0.039
(0.016 to 0.061), and on the 192 scenarios both models engage it is 0.046 against base's 0.018,
a paired difference of 0.028 (0.007 to 0.049) with a minimum detectable effect of 0.030 at this
count. The decomposition in \Cref{tab:three-cell} says where the difference sits: the incentive
raises the instruct model's acting mass by 0.085 against base's 0.049 (paired difference 0.037,
0.012 to 0.062), while its effect on the judging mass is the same on both models (0.039 versus
0.030, difference $-0.009$, $-0.023$ to 0.004). Post-training made the action, not the judgment,
more responsive to the incentive.

**And the baseline moved.** The instruct model's no-pressure frame gap has the opposite sign,
$-0.038$ ($-0.059$ to $-0.015$): with nothing at stake the instruct model, placed as the actor,
puts less mass on the violating option than it does as a judge (0.192 acting against 0.224
judging on the shared scenarios), where base does the reverse (0.306 against 0.280). The net
gap under pressure is therefore smaller after post-training (0.011 against 0.043 on the shared
scenarios, paired difference 0.033, 0.012 to 0.054), not because pressure moves the instruct
model less, but because post-training lowered where it starts.

**Robustness.** The selection check passed: base's excess on all 354 of its above-floor scenarios
(0.017) matches its excess on the 192 shared with instruct (0.018), so restricting to the
scenarios the instruct model engages does not change base's number. The paired difference has
the same sign on primaries ($-0.030$, $-0.060$ to 0.000) and harm twins ($-0.026$, $-0.056$ to
0.002), and in every gate family (F1 $-0.023$, F3 $-0.029$, F4 $-0.058$, F5 $-0.005$; only F4's
interval excludes zero on its own). It is not resolved on the 45 pilot-written scenarios alone
($-0.003$, $-0.032$ to 0.027), which were written under an earlier prompt version, and it is
resolved on the 147 later-written ones ($-0.036$, $-0.063$ to $-0.011$); the two subsets are not
separated from each other at these counts, and the pooled number is the number of record. The
pilot pod re-ran the same raw-frame forward passes and returned identical values, so the pilot is
a subset check, not an independent replication.

**Verdict by the pre-registered rule.** The excess is present in base and present in instruct,
and the paired base-minus-instruct difference excludes zero with the instruct model larger: the
branch of record is *widened*, with the acting-side decomposition and the baseline shift carried
in the sentence. In the program's terms: pretraining already installs a pressure-attributable
judgment–action gap; post-training does not remove it, and makes the action more sensitive to
the incentive while making the model less willing to take the violating action absent one. The
binary readout does not disagree in sign anywhere in \Cref{tab:three-cell}; it is under-powered
for effects of this size and is reported beside the continuous readout as the pre-registration
requires.

**Two rival readings, and the cells that separate them.** First, the enlarged acting-side
sensitivity may be goal-following rather than anything moral: post-training teaches a model to
pursue the goal it is given in context, the incentive sentence states a goal, and the judge in
the third-person frame reads the same sentence but does not hold the goal. On this reading the
widening is the assistant's instruction-following extended to a bad instruction, which is a
mechanism, not a rebuttal; the pre-registered deliberation-dose arm (unrun, \Cref{limitations})
separates it, since goal-following should be reduced by asking the model to reason before it
acts, and a filler-matched budget control says whether the reduction is deliberation or context.
Second, the instruct model's negative no-pressure gap may be a caution installed in the agent
persona, or an artifact of running a chat-trained model in a raw frame that it declines to
engage on 155 of 397 scenario-frames. The selection check bounds the second reading on the
scenarios it does engage; the clean discriminator is a letter-only judgment readout under the
chat template on the twins, one cell of about five minutes on any loaded OLMo-3, entered in the
anomaly ledger with that price.
