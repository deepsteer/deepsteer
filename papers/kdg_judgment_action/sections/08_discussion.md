# 8. Discussion {#discussion}

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
behavioral outcome the cell is scored against, the role Cheng et al. [@cheng2026tool] and Basu
et al. [@basu2026interpretability] give the gap in non-moral domains. The exploratory continuous
read of \Cref{structure} (F1 above F3; a pressure-attributable excess present on the honesty and
fairness families and unresolved on the shortcut and harm families) is the first candidate for
that structure, and it is cheap to test.

**Where the gap comes from.** The raw-frame cell answers the origin question the prior panels
could not ask. The base model already acts more violating than it judges; a second-person frame
alone moves it in that direction, and the incentive adds to it. Post-training does not remove
this. It changes two things at once: it lowers the baseline, so that the instruct model placed as
the actor is, with nothing at stake, more norm-consistent than the same model as a judge; and it
raises the action's sensitivity to the incentive by about three-quarters, while leaving the
judgment's sensitivity where base had it. The net effect under pressure is a smaller gap, which
is what a behavioral evaluation without the twin would report as an improvement; the
decomposition says the improvement is a baseline shift, and that the pressure sensitivity went
the other way. Combined with our earlier finding that moral comprehension is pretraining-native
while the refusal gate is a post-training construction, the picture is of a model whose judgment
and action are both pretraining-shaped and already discrepant before alignment, with alignment
adding a cautious default and a stronger pull toward the in-context goal, and refusal added on
top as a narrow control that reads neither broadly.

**Goal-following is the parsimonious mechanism, and it is testable.** The simplest account of the
widened acting-side sensitivity is not moral at all: post-training teaches a model to pursue the
goal it is handed in context, the incentive sentence hands the actor a goal, and the judge, who
reads the same sentence, does not hold it. On that account the judgment–action gap of an
instruct model is instruction-following extended to a bad instruction, which is consistent with
the Schmied et al. [@schmied2025greedy] observation that fine-tuned agents act greedily on the
goal in front of them, and with Rakshit et al.'s [@rakshit2026pseudo] finding that asking a model
to reason before acting does not by itself close the gap. The panel's pre-registered
deliberation-dose arm, with a filler-matched budget control, is the cell that separates
goal-following from a moral read: if reasoning before acting closes the gap where filler does
not, deliberation reaches the action; if neither does, the action is set before the reasoning
starts, and the mechanistic cell should look at the decision token, not the trace. Persona
steering [@chen2025persona] is the second lever the design anticipates, since the baseline shift
is the signature of an installed assistant default.

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
