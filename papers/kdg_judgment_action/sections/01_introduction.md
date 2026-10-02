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
has been the central empirical problem since Blasi's review [-@blasi1980bridging], and it has a
recent history in the language-model literature under the name *knowing–doing gap*
[@pfeffer2000knowing; @schmied2025greedy]. The design closest to ours is Strakhov and Claude
[-@strakhov2025agents], who pose the same AI-ethics dilemma to a model in a third-person theory
mode and a second-person action mode with callable tools, take the model's own theory-mode choice
as the reference, and find that 47.6% of 351 paired choices across nine instruct models reverse.
Their reversal counts changes in either direction from one draw per mode at temperature 1.0,
without a pressure manipulation or a re-elicitation floor; the gap rate below counts only moves
toward the violating option, by majority over 32 rollouts, against a matched null, so the two
numbers are different quantities. Their action-mode reversals were coded more often as less
interventionist than as bolder (48.5% against 36.5%); in our panel, read under the chat template with
nothing at stake, the gap leans toward the violating option (0.055), so the two designs do not agree
on the at-rest direction, and their axis (intervention level) is not ours (norm consistency). Shao et al. [-@shao2024privacylens] find a related split for
privacy norms: models answer privacy questions better than they respect those norms when acting
as agents. Huang et al. [-@huang2026knowing] and Shen et al.
[-@shen2025valueaction] measure gaps between a model's stated values and its enacted choices;
Rakshit et al. [-@rakshit2026pseudo] add a fast-versus-slow deliberation contrast; Gu et al.
[-@gu2025alignment] compare stated and revealed preferences; Hosseini et al.
[-@hosseini2026judgment] find a judgment–consequence gap in clinical allocation; Backmann et al.
[-@backmann2025ethics] vary the pressure on agents in social dilemmas; and in non-moral domains
Cheng et al. [-@cheng2026tool] and Basu et al. [-@basu2026interpretability] use a knowing–doing
gap as the target that mechanistic interventions are scored against. \Cref{tab:priorart} places
this panel among them. What it adds is narrower than a new construct and, we think, more useful
for the mechanistic purpose: a per-scenario, self-referenced *moral* gap with the model as the
agent (as in Strakhov and Claude) under typed pressure families, each scenario paired with a twin
that removes the pressure, measured inside a calibration ladder that bounds how much of
any gap is reference noise or frame change, read on two instruments, with a harm-involving family
that ties back to what refusal reads. Beyond the panels in \Cref{tab:priorart} it adds three things:
a positive control on every model read (an operator instruction to take the violating action), so that
each null carries a detection bar; a same-base contrast of two post-training recipes on one set of
Llama-3.1 weights; and a deliberation arm with truncation and norm-salience controls.

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
Strakhov and Claude (2025) & own theory-mode choice, same dilemma & agent's tool call, one draw & none & no & no \\
Huang et al. (2026) & own value profile (questionnaire) & advisor's pick among four options & none & no & no \\
Shen et al. (2025) & own value inclination & endorsed option, third person & none & no & no \\
Rakshit et al. (2026) & own articulated values & free text, fast vs slow & deliberation budget & no & no \\
Gu et al. (2025) & own stated principle & forced binary, third person & prompt format & no & no \\
Hosseini et al. (2026) & own responsibility judgment & allocation decision & none & no & no \\
Backmann et al. (2025) & external (cooperation) & game move & framing, survival & no & no \\
Cheng et al. (2026); Basu et al. (2026) & external (capability; physician labels) & tool call; hazard flag & none & no & no \\
\addlinespace
this panel & own per-scenario judgment on four frames & agent's option, letter only, 32 rollouts & five typed families; a twin with the pressure removed; a deliberation arm & yes, three (raw); one same-base pair & pressure-removed twins; a known-gap control on every model \\
\bottomrule
\end{tabular}
\end{table}

The panel was pre-registered before any scenario existed, and every construction or analysis
decision taken afterwards is a dated amendment in the same document (\Cref{app:prereg}): eleven
construction decisions before any model data, and six analysis decisions after, four of them
committed before the computation they license and two post-hoc forks that carry verdicts under
both choices. Nine further amendments govern the Phase 1 sessions. We report three pods on
OLMo-3-7B-Instruct with its base checkpoint [@olmo3_2025]: a 96-scenario pilot that tested the
instrument, a 320-scenario full panel, and a 120-scenario second round that lifted the screened count
past the pre-registered gate. Three Phase 1 sessions then read the OLMo-3 post-training stages under
their chat template, Meta's Llama-3.1-8B-Instruct and its base [@grattafiori2024llama3], Tulu 3 at its
SFT, DPO and final checkpoints [@lambert2024tulu3], and Qwen2.5-7B-Instruct and its base
[@qwen2024qwen25], with the positive control on every instruct model and the deliberation arm on
OLMo-3 and Llama-3.1-8B-Instruct. Scenarios
were written half by Claude and half by GPT and cross-labeled by the other; neither generator is
the model evaluated. The harness that parses actions was calibrated twice on real replies against
two judges.

The argument runs in five steps, each carrying its ladder. First, on OLMo-3-7B-Instruct, read under its
own chat template, the model takes the action it judged wrong on one screened scenario in five by
majority vote (0.19, 95% CI 0.13 to 0.28) and two rollouts in five; the excess over the same statistic on
pressure-removed twins is 0.10 (0.02 to 0.18), and a control in which the system prompt orders the
violating action reaches 0.58, so the instrument has room above the measurement. The gap is not reference
noise: on a pre-registered log-probability readout the excess holds at every strictness level of a
four-frame judgment reference and is largest where all four frames agree (0.08, 0.03 to 0.13). It has no
family structure at this power, so the action channel is not organized by the harm content refusal
reads. Second, the design withdrew one of our own claims by its pre-registered rule: a raw-frame reading
that post-training makes the model more cautious at rest is a property of reading a chat model without
its template. Within OLMo-3, the pressure-attributable excess is present at every templated checkpoint and
no post-training stage enlarges it on either readout, the probability scale or per unit of output
scale (586 scenarios screened by no model; the RL step's +0.004 sits at its 0.005 bar); the
base model's raw-frame gap (0.017, 0.012 to 0.022) is descriptive, since that frame misreads templated
checkpoints. Third, the gap follows the post-training recipe. With a positive control validating the
instrument on each of four instruct models, OLMo-3 and Meta's Llama-3.1-8B-Instruct carry the gap and
Tulu 3 and Qwen2.5-7B-Instruct do not, and on the same Llama-3.1 base Meta's recipe carries it while Ai2's
Tulu 3 does not. Fourth, moral deliberation before acting reduces the gap on both recipes that carry it,
against a length-matched non-moral control and the same control in truncated form; on OLMo-3, where the
reasoning is truncated at the budget, about a third of the effect comes from naming the norm. Fifth, the
raw-frame distortion that withdrew our claim follows the recipe as well: it appears on both Ai2 recipes
and not on Meta's.

\Cref{panel} describes the panel and its readouts; \Cref{instruments} the harness, the screen,
the ladder, and the two readouts; \Cref{gap} the gap; \Cref{reference} its robustness to the
judgment reference; \Cref{structure} the families and the generator effect; \Cref{base} the
post-training stages and the withdrawn claim; \Cref{recipe} the four recipes; \Cref{deliberation}
the dose arm; \Cref{discussion} what the result does and does not say about the action channel.
