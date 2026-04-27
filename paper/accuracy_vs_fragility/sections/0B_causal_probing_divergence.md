# Appendix B. Causal-probing divergence

The §3 methodology distinguishes *probing accuracy* (how linearly
decodable a property is from a layer's hidden states) from *causal
contribution* (how strongly intervening on a layer's hidden states
changes the model's downstream behavior on the property). The two
are conceptually separate — the layer where information is *stored*
may be different from the layer where information is *used* — but
the distinction is rarely operationalized in moral-representation
work, which typically reports probing accuracy and stops there.

We applied the `MoralCausalTracer` benchmark (an adaptation of Meng
et al.'s 2022 ROME causal-tracing methodology to the moral domain)
on three OLMo-3 7B checkpoints (early, mid, final) using the same
240-pair standard moral dataset as the probing analysis. The headline
finding:

| Checkpoint | Peak causal layer | Peak probing layer | Mean indirect effect |
|-----------:|------------------:|-------------------:|---------------------:|
| Step 0 | 6 | 1 | 0.37 |
| Step 705K | 7 | 13 | 9.09 |
| Step 1,414K | 0 | 10 | 9.60 |

*Numbers source: `outputs/phase_b/RESULTS.md` Finding B4.*

Causal effect magnitude grows ~26× over training (mean indirect
effect 0.37 → 9.60), and the peak causal layer migrates from layer
6 → layer 7 → layer 0. But the peak probing layer migrates from
layer 1 → layer 13 → layer 10 over the same training. At the final
checkpoint, the gap between the layer that most-strongly *encodes*
moral information (layer 10) and the layer that most-strongly
*influences* downstream moral-relevant generation (layer 0) is ~10
layers — the two metrics identify nearly opposite ends of the
network.

This is consistent with a "storage vs. use" picture of moral
representation in transformer language models: moral information is
*stored* in mid-network layers (where probing recovers it cleanly)
and *used* in early layers (where intervening on it most-strongly
moves the model's downstream output). The two facts are
representational properties of the same model that probing alone
cannot recover.

**Status.** This appendix is a stub (~half page). It serves as
supporting evidence for the §5.2 fragility-as-richer-functional
argument: probing accuracy is one functional of the representation
geometry; causal contribution is another; fragility is a third. All
three keep evolving through training, often in different directions,
and a complete picture of how a representation evolves needs all
three. We do not develop the storage-vs-use finding as a body
contribution because (a) it is not specific to moral representation
and (b) it has its own methodological complications (causal-tracing
sensitivity to intervention magnitude, choice of decoder probe) that
warrant their own paper.
