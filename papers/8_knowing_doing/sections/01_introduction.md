# 1. Introduction {#introduction}

A language model can state, when asked as an observer, that an action is wrong, and then take
that action when placed inside the situation as the agent. Whether that happens, how often, and
whether it depends on what the model comprehends about the situation are questions a mechanistic
program about moral representations needs answered before it can ask what the model's *action*
reads from its representations. Our previous work established what the *refusal* decision reads
on the same models: a low-rank harm slice of a broader moral subspace, written into a narrow
control-token channel [@reblitzrichardson2026fragility]. This paper builds the behavioral
target for the next step, action selection, and measures it.

The quantity is a knowing–doing gap: per scenario, the divergence between what the model judges
to be right, elicited in a third-person frame, and what it does, elicited in a matched agent
frame with a discrete, logged action. The reference for "right" is the model's own stated
judgment, not an external label, so the construct is self-inconsistency and sidesteps the
contestability of external ethics. That construct is not new. Huang et al. [@huang2026knowing]
and Shen et al. [@shen2025valueaction] measure gaps between a model's stated values and its
enacted choices; Rakshit et al. [@rakshit2026pseudo] add a fast-versus-slow deliberation
contrast; Gu et al. [@gu2025alignment] compare stated and revealed preferences; Hosseini et al.
[@hosseini2026judgment] find a judgment–consequence gap in clinical allocation; and in non-moral
domains Cheng et al. [@cheng2026tool] and Basu et al. [@basu2026interpretability] use a
knowing–doing gap as the target that mechanistic interventions are scored against. What this
paper adds is narrower and, we think, more useful for that last purpose: a per-scenario,
self-referenced *moral* gap measured with the model as the agent under typed pressure families,
inside a calibration ladder that bounds how much of any gap is reference noise or frame change,
with a harm-involving family that ties back to what refusal reads, and a base-versus-instruct
comparison in one raw-completion frame that none of the prior panels include.

The panel was pre-registered before any scenario existed, and every construction decision made
afterwards is a dated amendment in the same document (\Cref{app:prereg}); there are sixteen. We
report three pods on one model, OLMo-3-7B-Instruct with its base checkpoint
[@olmo3_2025]: a 96-scenario pilot that tested the instrument, a 320-scenario full panel, and a
120-scenario second round that lifted the screened count past the pre-registered gate and ran a
paraphrase-swap cell on the one family whose result depended on which model had written it.
Scenarios were generated half by Claude and half by GPT, cross-labeled by the other; the
harness that parses actions was calibrated twice on real replies against two independent
judges.

The findings, in the order the ladder licenses them. On the screened panel the model takes the
action it judged wrong on one scenario in five by majority vote (0.19, 95% CI 0.13 to 0.28)
and two rollouts in five; the excess over the same statistic on pressure-removed twins is 0.10
(0.02 to 0.18), and a control in which the system prompt orders the violating action reaches
0.58. The gap survives a paraphrase-majority judgment reference; on the majority readout it is
not resolved against the strictest reference, where all four judgment frames must agree, and on
a pre-registered secondary readout built from the saved next-token probabilities it holds at
every strictness level and is largest at the strictest (0.08, 0.03 to 0.13). The gap has no
family structure at this power: third-party harm is not where it is smallest, so the action
channel is not organized by the harm content refusal reads. The same gap is present in the base
model's raw completion frame at least as strongly as in the instruct model's on the same
scenarios, so post-training did not install it. One family's rate depends on which model wrote
its scenarios; a blind human read finds no construction asymmetry, and on the continuous readout
the difference is graded, not reversed.

The paper is organized around the instruments rather than the story. \Cref{panel} describes
the panel; \Cref{instruments} the harness, the ladder, and the two readouts;
\Cref{gap} the gap and its robustness to the judgment reference; \Cref{structure} the
families and the generator effect; \Cref{base} the base-versus-instruct cells;
\Cref{discussion} what the result does and does not say about the action channel.
