# 1. Introduction {#introduction}

A language model can say, when asked as an observer, that an action is wrong, and then take
that action when placed inside the situation as the agent. Reports of agentic misalignment show
frontier models acknowledging an ethical violation in their own reasoning and proceeding with it
[@lynch2025agentic]; models trained to behave can conceal a triggered policy through the training
itself [@hubinger2024sleeper] and can fake alignment when they infer they are monitored
[@greenblatt2024faking]. Our earlier work located one dissociation behind divergence of this kind.
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
as the reference (\Cref{tab:priorart} compares their numbers with ours). Shao et al. [-@shao2024privacylens] find a related split for
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

The panel was pre-registered before any scenario existed, and every construction or analysis
decision taken afterwards is a dated amendment in the same document (\Cref{app:prereg}): eleven
construction decisions before any model data, and six analysis decisions after, four of them
committed before the computation they license and two post-hoc forks that carry verdicts under
both choices. Fourteen further amendments govern the Phase 1 sessions: three are post-hoc forks, one is a post-hoc
addition (P1-A13), and four are post-review additions (P1-A11 to P1-A14). We report three pods on
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

The argument runs in six steps; each section it cites carries its ladder.

1. **The gap is real.** On OLMo-3-7B-Instruct, read under its own chat template, the model takes the
   action it judged wrong on one screened scenario in five by majority vote, more than on the
   pressure-removed twins, and the excess holds at every strictness level of the judgment reference
   (\Cref{gap}, \Cref{reference}).
2. **Post-training stages do not enlarge it.** Read in each checkpoint's own format, the
   pressure-attributable excess is present at every templated OLMo-3 checkpoint and no post-training stage
   enlarges it on either readout (\Cref{base}).
3. **The recipe decides.** Across five instruct models, with GPT-OSS-20B read without reasoning, OLMo-3 and
   Meta's Llama-3.1-8B-Instruct carry the gap while Tulu 3, Qwen2.5-7B-Instruct and GPT-OSS-20B show none
   on the whole panel within their detection bars, and on the same Llama-3.1 weights Meta's recipe carries
   it while Ai2's Tulu 3 does not (\Cref{recipe}).
4. **Deliberation brakes it.** On both models that carry the gap, reasoning about the stakes before acting
   lowers the violating choice against a length-matched non-moral task, with or without the pressure
   (\Cref{deliberation}).
5. **The format distortion follows the recipe.** Reading a chat model outside its chat template reverses
   the sign of its gap with nothing at stake on both Ai2 recipes and not on Meta's (\Cref{base},
   \Cref{recipe}).
6. **The brake tracks the stated rule.** Where following a written approval gate harms an absent third
   party, deliberation still moves the action toward the rule on OLMo-3, Llama-3.1 and GPT-OSS-20B, with
   Llama-3.1's brake at about a seventh of its panel size (\Cref{rule-harm}).

\Cref{panel} describes the panel and its readouts, \Cref{instruments} the harness, the screen, the ladder
and the two readouts, \Cref{structure} the families and the generator effect, and \Cref{discussion} what
the result does and does not say about the action channel.

\begin{table}[bp]
\centering
\caption{Where this panel sits among measured judgment--action and knowing--doing gaps in language
models (citations in the text above). ``Own'' means the reference is the model's own statement. The
last column is to our reading of each paper: where a paper has something that works as a null or a
check (a control condition, an instruction-following check), the cell names it, and ``no'' means we found
nothing that does; chance-level floors are not counted. Strakhov and Claude find that 47.6\%
of 351 paired choices across nine instruct models reverse; their reversal counts changes in either
direction from one draw per mode at temperature 1.0, without a pressure manipulation or a re-elicitation
floor, while the gap rate here counts only moves toward the violating option, by majority over 32
rollouts, against a matched null, so the two numbers are different quantities. Their action-mode
reversals were coded more often as less interventionist than as bolder (48.5\% against 36.5\%); here,
read under the chat template with nothing at stake, the gap leans toward the violating option (0.055), on
a different axis (norm consistency, not intervention level).}
\label{tab:priorart}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.14\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.08\linewidth}>{\raggedright\arraybackslash}p{0.15\linewidth}@{}}
\toprule
panel & reference for ``right'' & action readout & pressure manipulation & base model & matched null; positive control \\
\midrule
Strakhov and Claude (2025) & own theory-mode choice, same dilemma & agent's tool call, one draw & none & no & no \\
Huang et al. (2026) & own value profile (questionnaire) & advisor's pick among four options & none & no & no; value-selection check \\
Shen et al. (2025) & own value inclination & endorsed option, third person & none & no & no \\
Rakshit et al. (2026) & own articulated values & free text, fast vs slow & deliberation budget & no & no \\
Gu et al. (2025) & own stated principle & forced binary, third person & prompt format & no & one-factor prompt pairs; no \\
Hosseini et al. (2026) & own responsibility judgment & allocation decision & none & no & no \\
Backmann et al. (2025) & external (cooperation) & game move & framing, survival & no & neutral base game; no \\
Cheng et al. (2026); Basu et al. (2026) & external (capability; physician labels) & tool call; hazard flag & none & no & random-steering controls (Basu); no \\
\addlinespace
this panel & own per-scenario judgment on four frames & agent's option, letter only, 32 rollouts & five typed families; a twin with the pressure removed; a deliberation arm & yes, three (raw); one same-base pair & pressure-removed twins; a known-gap control on every model \\
\bottomrule
\end{tabular}
\end{table}

