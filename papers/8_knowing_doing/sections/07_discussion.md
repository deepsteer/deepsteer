# 7. Discussion {#discussion}

**What the action channel does not read.** The motivating question was whether action
selection, like refusal, reads the harm slice of the model's moral content. If it did, the gap
would be smallest where the violating option harms someone, because that is where the harm read
would engage. It is not: the third-party-harm family sits in the middle of the pack on both
instruments, and the pre-registered contrast is centered on zero. Within this panel's power
(a family-contrast MDE near 0.40) the action channel is not organized by what refusal reads.
That is a negative result with a bar attached, and it changes what the next mechanistic cell
should test: not "does the action position load on the harm direction" but "what does it load
on at all," with this gap as the behavioral outcome the cell is scored against, the role Cheng
et al. [@cheng2026tool] and Basu et al. [@basu2026interpretability] give the gap in non-moral
domains.

**Inherited, not installed.** The base model shows the raw-frame gap at least as strongly as the
instruct model on the same scenarios. That is the cell none of the prior panels contain, and its
reading is the opposite of the intuitive one: post-training does not create the discrepancy
between what the model says is right and what it does; pretraining already carries it, and the
assistant template adds little. Combined with our earlier result that the refusal gate is a fresh
post-training construction while moral comprehension is pretraining-native
[@reblitzrichardson2026fragility], the picture is of a model whose judgment and action are both
pretraining-shaped and already misaligned with each other before any alignment step, with
refusal added on top as a narrow control that reads neither broadly.

**Two instruments, one gap.** The majority-vote gap and the log-prob gap agree on sign and rough
size wherever the first has power, reproduce the same positive band, and pass a coherence check
the second could have failed. They disagree at the strictest reference in the direction that
power predicts: the majority rule loses information as the scenario set shrinks, the
probability readout does not. We report both because the pre-registration made the binary
readout primary and because a reader should be able to see where a verdict depends on the
instrument. The practical lesson for panels of this kind is that saving the full next-token
distribution at the decision position costs disk and buys a second, more powerful instrument
for free.

**Reference noise is the rival that matters.** A self-referenced gap can be manufactured by a
noisy reference. The panel's floor rung says the model's stated judgment changes under
paraphrase on 30% of scenarios by option; 60 of 208 gate primaries fail the stability rule. The
strictness ladder is how the panel answers the rival, and it answers it on the continuous
instrument at every level. What the ladder also shows is that the judgment of a 7B instruct model
on these scenarios is often not decisive, which is a finding about the reference and a reason
to build the next panel's scenarios to be decisive by construction.

**Generators are a covariate, not a nuisance.** Writing half the panel with each of two models
and cross-labeling caught three scenarios with inverted labels, produced one family whose rate
depended on its author, and, on the pooled panel, a writer difference of about 0.05 to 0.08 in
the same direction on both instruments. The blind read found no construction asymmetry, so the
difference is register: the same pressure, written in one model's idiom, engages the acting
model somewhat more. Any single-generator panel of this kind carries that effect invisibly.
