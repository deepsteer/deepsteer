# 7. Before and after post-training, and a format check that changed a reading {#base}

**In short.** On OLMo-3 the gap shows up before and after post-training. In the base model, which can
only be read as plain text completion, the part the incentive adds is 0.017 (0.012 to 0.022); we report
this as a description, because that format misreads chat models. In the aligned model, read in its own
chat format on all 586 scenarios, the gap is already present after supervised fine-tuning (0.015, 0.007
to 0.023), and no later stage makes it detectably larger (0.014 after DPO and 0.018 after RL; detection
bars of 0.008 and 0.005 for the two steps, with the RL step's +0.004 at its bar; \Cref{fig:stages}), while the same test
registers a large gap on the final model when its operator orders the violation (0.497, 0.468 to 0.525).
What post-training does change is the model's default lean when it acts: preference optimization nudges
it toward the advantageous option even when nothing is at stake (+0.011, 0.007 to 0.016).

Base-versus-instruct is the question of origin: a gap present in base weights is inherited from
pretraining; a gap absent in base and present in instruct would point to post-training. A base
model has no chat template, so the only frame both models share is a raw completion frame
(scenario, option list, fixed prefix ending before the option token; the readout is the
option-letter probability mass, eight option orders). Each model's pressure-removed twins give it a
matched null in the same frame, and the quantity compared is the **pressure-attributable excess**
$E$: the paired difference between the acting-versus-judging mass gap on a scenario and on its
twin. The verdict rules were fixed in dated amendments before the numbers below were computed
(\Cref{app:prereg}; tables in \Cref{app:three-cell}). The instruct model is also read under its own
chat template, with a letter-only judgment so that acting and judging differ only in role and frame.
That second readout changed what this section can claim, and we report the change as a result.

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_stages.pdf}
\caption{\textbf{OLMo-3 across post-training, each checkpoint read in its own chat template.} (a) The
pressure-attributable excess $E$ after SFT, after DPO and after RL (the final model), on the 586 scenarios
screened by no model (filled; the number of record for stage claims) and on the 136 scenarios screened on
the final model's actions (hollow); the gap is present at every stage and no later stage makes it
detectably larger (step detection bars 0.008 for DPO and 0.005 for RL on the 586). The base model, which
has no chat template, is shown in the raw completion frame (gray diamond) as a description only: that
frame misreads templated checkpoints, so the base value is not a validated before-and-after comparison.
(b) On the final checkpoint, the positive control (red: the acting frame's move when the operator orders
the violating action; it was not run on the SFT and DPO checkpoints) and the excess on one axis, with the
pre-registered validation bar dotted. 95\% bootstrap CIs over scenarios.}
\label{fig:stages}
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
acting than when judging, 0.041 (0.034 to 0.049) on 359 scenarios, and 0.024 (0.017 to 0.030) on
the pressure-removed twins: a second-person frame alone moves the base model toward the violating
option. The pressure-attributable excess, 0.017 (0.012 to 0.022), excludes zero. The binary shadow
of the same quantity does not resolve (0.024, $-0.021$ to 0.065), which is the majority rule on a
small mass effect, not a disagreement in sign. This cell is descriptive. The raw frame is the only frame
a base model has, and the pre-registered bridge test shows it already misreads the first templated
checkpoint: at rest, SFT's gap is $-0.019$ ($-0.035$ to $-0.003$) in the raw frame and $+0.033$ (0.021
to 0.045) under its template, a difference of $-0.052$ ($-0.070$ to $-0.034$) on 310 scenarios. We
therefore report the base number as the raw frame's reading of the base model, not as a validated
comparison with the aligned model.

**Survives post-training, under the model's own template.** Read under its chat template on the 136
screened scenarios, the instruct model's acting-versus-judging gap is 0.055 (0.034 to 0.076) with
nothing at stake and 0.084 (0.059 to 0.110) under pressure; the pressure-attributable excess is
0.030 (0.006 to 0.053). The gap has the base model's sign at rest and under pressure, and the part
of it the incentive adds does not disappear with alignment. The screened set was selected on this
model's own chat actions, which inflates the raw gap; the twin-differenced excess is less exposed to
that selection, and it is the number we lean on. Framing the same prompts as an internal safety
evaluation lowers the at-rest gap by about 0.01 (0.010, 0.001 to 0.021 on the 208 raw-engaged
scenarios; not resolved on the 136) and does not detectably move the excess ($-0.008$, $-0.026$ to
0.012).

**A format check that changed a reading.** In the raw frame the instruct model's no-pressure gap is
$-0.038$ ($-0.059$ to $-0.015$): with nothing at stake it looks more cautious acting than judging,
which would read as post-training lowering the baseline while raising the action's sensitivity to the
incentive. A pre-registered cell separates an agent-frame caution installed by post-training from a
raw-frame artifact, with both branches fixed in advance. Under the template the sign is positive
(+0.055, 0.034 to 0.076), and the artifact branch holds. The raw frame also exaggerates how much sharper the instruct model's
agent-frame output is than its judge-frame output (a ratio of 1.79 raw against 1.07 under the
template). A chat-trained model read without its template is not a neutral instrument, and a raw
completion frame manufactures agent-frame effects on it that a base-versus-instruct comparison would
attribute to post-training.

**What remains of the raw-frame comparison.** On the 192 scenarios both models engage in the raw
frame, the instruct model's excess is 0.046 against base's 0.018 (paired difference 0.028, 0.007 to
0.049; MDE 0.030), and the pre-registered rule names that branch *widened*. We report the number under the
rule's word and do not rest a claim on it. The instruct side of that comparison is a raw-frame cell
carrying the format effect above; on the log-odds scale the difference survives, but divided by
each model's own output scale it does not resolve (0.12, $-0.02$ to 0.26), and the scale factor is
itself partly a raw-frame property of the instruct model. The comparison that is template-valid on
every side runs across post-training stages under one template. On the 136 screened scenarios the
pressure-attributable excess is 0.021 (0.006 to 0.036) after SFT, 0.022 (0.002 to 0.043) after DPO and
0.030 (0.006 to 0.053) after RLVR; the DPO and RL steps are 0.001 ($-0.012$ to 0.014) and 0.007
($-0.001$ to 0.016). On this screened set neither step's interval excludes zero (bars 0.019 for the
DPO step and 0.012 for the RL step, $n = 136$); the stage contrast of record is the unscreened set
below.

**What post-training does change.** Read on a set screened by no model's actions (all 586 union
scenarios, each checkpoint under its own template), the acting frame's lean toward the violating option
at rest grows at the DPO step: 0.021 after SFT, 0.033 after DPO and 0.039 after RLVR. The DPO step, +0.011
(0.007 to 0.016), survives division by each checkpoint's output scale (+0.058, 0.027 to 0.090), so it is
not the sharpening that post-training also brings. The RL step is positive on the probability scale
(+0.006, 0.003 to 0.009) but not per unit of output scale (+0.018, $-0.001$ to 0.037), and reads as that
sharpening. No step enlarges the pressure-attributable part on either readout (positive control on the
final checkpoint: known-gap band 0.497, 0.468 to 0.525; the cell was not run on the SFT and DPO
checkpoints). On the probability
scale the DPO step does not move it ($-0.001$, $-0.006$ to 0.005; bar 0.008), and per unit of output
scale it falls ($-0.039$, $-0.069$ to $-0.008$), because DPO sharpens the output without adding pull.
The RL step reads +0.004 on the probability scale (0.000 to 0.008; bar 0.005), at the bar, with every
known bias favoring a positive step and the same sign as the screened set; on the primary
per-unit-of-output-scale readout it is +0.005 ($-0.016$ to 0.026; bar 0.030). Log-odds also excludes
zero, so baseline compression is ruled out and the remaining rival is the output sharpening the RL step
also brings. Preference optimization makes the model, placed as the actor, lean further toward the
locally advantageous option than it does as a judge, without enlarging how much the incentive adds on
either readout. That is the opposite direction from the raw frame's
at-rest reading, and it replicates on a second DPO recipe (\Cref{recipe}); which part of the DPO stage
produces it is left to a recipe ablation on one base.

**Robustness of the base cell.** The selection check passed: base's excess on its 354
above-floor primary–twin pairs (0.017) matches its excess on the 192 shared with instruct (0.018). The pilot
pod re-ran the same raw-frame forward passes and returned identical values, so the pilot is a subset
check, not an independent replication.
