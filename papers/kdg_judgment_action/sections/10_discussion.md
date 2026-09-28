# 10. Discussion {#discussion}

**Three decisions, three reads.** Our earlier work found that on OLMo-3 the refusal decision
reads a rank-1 harm slice of a broad moral subspace of which the judgment decision reads
two-thirds [@reblitzrichardson2026refusal]. This panel adds the action decision on the same
model. If action selection read what refusal reads, the gap would be smallest where the
violating option harms someone, because that is where the harm read would engage. It is not: the
third-party-harm family sits in the middle of the pack on both instruments, and the
pre-registered contrast is centered on zero at a family-contrast MDE near 0.40. Within that
power the action channel is not organized by what refusal reads. That is a negative result with
a bar attached, and it changes what the next mechanistic cell should test: not "does the action
position load on the harm direction" but "what does it load on at all", with this gap as the
behavioral outcome the cell is scored against, the role Cheng et al. [-@cheng2026tool] and Basu
et al. [-@basu2026interpretability] give the gap in non-moral domains. The exploratory continuous
read of \Cref{structure} (F1 above F3; a pressure-attributable excess present on the honesty and
fairness families and unresolved on the shortcut and harm families) is the first candidate for
that structure, and it is cheap to test.

**Where the gap comes from.** On OLMo-3 the base model already acts more violating than it judges in
the only frame a base model has, and under its own template the aligned model does the same at rest
and under pressure; post-training does not resize the pressure-attributable part (0.021, 0.022 and 0.030
after SFT, DPO and RLVR, with no stage change detectable above about 0.013 at $n = 136$). Across
lineages the picture is not one of a pretraining property that survives every alignment. Two of four
instruct models carry the gap and two do not, with a validated instrument on each, and on the same
Llama-3.1 base Meta's recipe carries it while Ai2's Tulu 3 does not (\Cref{recipe}). The defensible
account is that post-training recipes differ in whether the aligned model acts on the incentive against
its own judgment, and that within a recipe that carries the gap, later stages leave its size alone.
Combined with our earlier finding that the refusal gate is a thin post-training construction over a
broad pretrained moral representation [@reblitzrichardson2026refusal], the action decision looks like
the refusal decision in one respect: both are shaped by post-training choices, and neither simply
inherits what the model comprehends.

**The instrument corrected us, by its own rule.** An earlier reading of this panel said that
post-training lowers the baseline, so that a behavioral evaluation without the twin would record an
improvement where the pressure sensitivity went the other way. The pre-registered cell that separates
a post-training caution from a frame artifact returned the artifact branch: under the model's template
the at-rest gap is positive, not negative. We state this as a result because it generalizes, though not
everywhere. A chat model read in a raw completion frame acquires agent-frame effects that belong to the
frame on both Ai2 recipes we read, at every checkpoint, and not on Meta's instruct model from the same
base. Any base-versus-instruct contrast run in such a frame, including log-probability sycophancy
measurements that score instruct checkpoints without their template, inherits whichever effect its
recipe has. The twin design caught the problem only because the frame itself was put under test; the
lesson for evaluations is to read every templated model in its template, and to treat a raw-frame
instruct number as a format cell.

**Goal-following is the parsimonious mechanism, and deliberation loosens it.** The simplest account of
the gap is not moral at all: post-training teaches a model to pursue the goal it is handed in context,
the incentive sentence hands the actor a goal, and the judge, who reads the same sentence, does not hold
it. That is consistent with the Schmied et al. [-@schmied2025greedy] observation that fine-tuned agents
act greedily on the goal in front of them. The dose arm shows the goal is not fixed once read: on both
recipes that carry the gap, reasoning about the stakes before acting moves the action toward the
model's own judgment against matched non-moral controls (\Cref{deliberation}), and naming the norm alone
does about a third of that on OLMo-3. This differs from Rakshit et al.'s [-@rakshit2026pseudo] finding
that reasoning before acting does not by itself align action with stated values, in a different
construct (value profiles, free-text actions) and without a filler control; the filler control is what
lets us attribute the change to the content of the reasoning rather than to its length. Persona steering
[@chen2025persona] remains the second lever the design anticipates; with the baseline shift withdrawn,
its target is the pressure-attributable excess itself.

**Two instruments, one gap.** The majority-vote gap and the log-prob gap agree on sign and rough
size wherever the first has power, reproduce the same positive band, pass a coherence check the
second could have failed, and agree with each other through a second derivation (the crossings
of 0.5 predict the majority excess). They disagree at the strictest reference and in the raw
frame, both times in the direction that power predicts: the majority rule loses information as
the scenario set shrinks or the effect becomes a small shift in mass, the probability readout
does not. We report both because the pre-registration made the binary readout primary and
because a reader should be able to see where a verdict depends on the instrument. The practical
lesson for panels of this kind is that saving the full next-token distribution at the decision
position costs disk and buys a second, more powerful instrument for free, and that a base model,
which cannot be read by majority vote at all, can be read on the same footing as the instruct
model once both are read by mass.

**Reference noise is the rival that matters, and the ladder answered it.** A self-referenced gap
can be manufactured by a noisy reference, and this model's reference is noisy: the stated
judgment changes under paraphrase on 30% of scenarios by option, and 74 of 208 gate primaries
fail the stability rule. The strictness ladder is how the panel answers the rival, and it answers
it on the continuous instrument at every level, with the excess largest where the judgment is
least ambiguous. What the ladder also shows is that the judgment of a 7B instruct model on these
scenarios is often not decisive, which is a finding about the reference and a reason to build the
next panel's scenarios to be decisive by construction.

**Generators are a covariate, not a nuisance.** Writing half the panel with each of two models
and cross-labeling caught three scenarios with inverted labels, produced one family whose rate
depended on its author, and, on the pooled panel, a generator difference of about 0.05 to 0.08
in the same direction on both instruments. The blind read found no construction asymmetry, so
the difference is register: the same pressure, written in one model's idiom, engages the acting
model somewhat more. Any single-generator panel of this kind carries that effect invisibly, and
so does any panel whose generator is also the model evaluated.
