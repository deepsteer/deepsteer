# 1. Introduction {#introduction}

A language model can say, when asked as an observer, that an action is wrong, and then take
that action when placed inside the situation as the agent. Reports of agentic misalignment show
frontier models acknowledging an ethical violation in their own reasoning and proceeding with it
[@lynch2025agentic]; models trained to behave can conceal a triggered policy through the training
itself [@hubinger2024sleeper] and can fake alignment when they infer they are monitored
[@greenblatt2024faking]. Our earlier work located one mechanism behind divergence of this kind.
On the models studied here, moral comprehension is a broad representation that forms during
pretraining, while the refusal decision is a thin post-training control that reads only a narrow
harm slice of it [@reblitzrichardson2026refusal]: the model knows more than its refusal uses.
This paper asks the next question in that program. When the model acts rather than refuses, does
its action follow its judgment; if not, is the discrepancy organized by the same harm content;
and did alignment create it?

The quantity is a **judgment–action gap**: per scenario, the divergence between what the model
judges to be right, elicited in a third-person frame, and what it does, elicited in a matched
agent frame with a discrete, logged action. The reference for "right" is the model's own stated
judgment, not an external label. A gap means the model contradicted itself, acting against what
it had just said was right, so the measurement does not depend on whose ethics are correct. The
construct is old in moral psychology, where the relation between moral judgment and moral action
has been the central empirical problem since Blasi's review [@blasi1980bridging], and it has a
recent history in the language-model literature under the name *knowing–doing gap*
[@pfeffer2000knowing; @schmied2025greedy]. Huang et al. [@huang2026knowing] and Shen et al.
[@shen2025valueaction] measure gaps between a model's stated values and its enacted choices;
Rakshit et al. [@rakshit2026pseudo] add a fast-versus-slow deliberation contrast; Gu et al.
[@gu2025alignment] compare stated and revealed preferences; Hosseini et al.
[@hosseini2026judgment] find a judgment–consequence gap in clinical allocation; Backmann et al.
[@backmann2025ethics] vary the pressure on agents in social dilemmas; and in non-moral domains
Cheng et al. [@cheng2026tool] and Basu et al. [@basu2026interpretability] use a knowing–doing
gap as the target that mechanistic interventions are scored against. \Cref{tab:priorart} places
this panel among them. What it adds is narrower than a new construct and, we think, more useful
for the mechanistic purpose: a per-scenario, self-referenced *moral* gap with the model as the
agent under typed pressure families, measured inside a calibration ladder that bounds how much of
any gap is reference noise or frame change, read on two instruments, with a harm-involving family
that ties back to what refusal reads, and a base-versus-instruct comparison in one raw completion
frame, the cell none of the prior panels contain.

\begin{table}[tbp]
\centering
\caption{Where this panel sits among measured judgment--action and knowing--doing gaps in language
models (citations in the text above). ``Own'' means the reference is the model's own statement.}
\label{tab:priorart}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.08\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}@{}}
\toprule
panel & reference for ``right'' & action readout & pressure manipulation & base model & matched null; positive control \\
\midrule
Huang et al. (2026) & own value profile (questionnaire) & advisor's pick among four options & none & no & no \\
Shen et al. (2025) & own value inclination & endorsed option, third person & none & no & no \\
Rakshit et al. (2026) & own articulated values & free text, fast vs slow & deliberation budget & no & no \\
Gu et al. (2025) & own stated principle & forced binary, third person & prompt format & no & no \\
Hosseini et al. (2026) & own responsibility judgment & allocation decision & none & no & no \\
Backmann et al. (2025) & external (cooperation) & game move & framing, survival & no & no \\
Cheng et al. (2026); Basu et al. (2026) & external (capability; physician labels) & tool call; hazard flag & none & no & no \\
\addlinespace
this panel & own per-scenario judgment on four frames & agent's option, letter only, 32 rollouts & five typed families; a twin with the pressure removed & yes, raw frame & pressure-removed twins; known-gap band \\
\bottomrule
\end{tabular}
\end{table}

The panel was pre-registered before any scenario existed, and every construction or analysis
decision taken afterwards is a dated amendment in the same document (\Cref{app:prereg}): eleven
construction decisions before any model data, and six analysis decisions after, four of them
committed before the computation they license and two post-hoc forks that carry verdicts under
both choices. We report three pods on one model, OLMo-3-7B-Instruct with its base checkpoint
[@olmo3_2025]: a 96-scenario pilot that tested the instrument, a 320-scenario full panel, and a
120-scenario second round that lifted the screened count past the pre-registered gate. Scenarios
were written half by Claude and half by GPT and cross-labeled by the other; neither generator is
the model evaluated. The harness that parses actions was calibrated twice on real replies against
two judges.

The argument runs in four steps, each carrying its ladder. First, on the screened panel the model
takes the action it judged wrong on one scenario in five by majority vote (0.19, 95% CI 0.13 to
0.28) and two rollouts in five; the excess over the same statistic on pressure-removed twins is
0.10 (0.02 to 0.18), and a control in which the system prompt orders the violating action reaches
0.58, so the instrument has room above the measurement. Second, the gap is not reference noise:
the judgment is elicited on four frames, and on a pre-registered log-probability readout the
excess over the null holds at every strictness level and is largest where all four frames agree
(0.08, 0.03 to 0.13); on the majority readout it is not resolved there at 43 pairs, which we show
is the majority rule discarding information, not the gap shrinking. Third, the gap has no family
structure at this power: third-party harm is not where it is smallest, so the action channel is
not organized by the harm content refusal reads. Fourth, the base model already carries the gap.
In a raw completion frame its acting-versus-judging mass gap exceeds its own pressure-removed
null (0.017, 0.012 to 0.022), and post-training does not remove that pressure-attributable part;
it enlarges it (0.046, 0.025 to 0.069, on the same scenarios), on the acting side, while lowering
the model's baseline willingness to take the violating action. One family's rate depends on which
model wrote its scenarios; a blind human read finds no construction asymmetry, and on the
continuous readout the difference is graded, not reversed.

\Cref{panel} describes the panel and its readouts; \Cref{instruments} the harness, the screen,
the ladder, and the two readouts; \Cref{gap} the gap; \Cref{reference} its robustness to the
judgment reference; \Cref{structure} the families and the generator effect; \Cref{base} the
base-versus-instruct cells; \Cref{discussion} what the result does and does not say about the
action channel.
