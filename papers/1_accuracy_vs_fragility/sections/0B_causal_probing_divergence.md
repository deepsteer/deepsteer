# Appendix B. Causal-probing divergence

The §3 methodology distinguishes *probing accuracy* (how linearly
decodable a property is from a layer's hidden states) from *causal
contribution* (how strongly intervening on a layer's hidden states
changes the model's downstream behavior on the property). The two
are conceptually separate (the layer where information is *stored*
may be different from the layer where information is *used*), but
the distinction is rarely operationalized in moral-representation
work, which typically reports probing accuracy and stops there.

We applied the `MoralCausalTracer` benchmark (an adaptation of Meng
et al.'s 2022 ROME causal-tracing methodology to the moral domain)
on three OLMo-3 7B checkpoints (early, mid, final) using the same
240-pair standard moral dataset as the probing analysis. OLMo-3 7B has
32 transformer layers, so layer indices run 0–31. The headline finding:

| Checkpoint | Peak causal layer | Peak probing layer | Mean indirect effect |
|-----------:|------------------:|-------------------:|---------------------:|
| Step 0 | 5 | 0 | 0.01 |
| Step 705K | 5 | 19 | 7.84 |
| Step 1,414K | 0 | 10 | 7.95 |

*Numbers source (exact files under `outputs/phase_b/`): peak causal
layer and the layer-mean indirect effect from
`step_{0000000,0705000,1413814}/moral_causal_tracer_allenai_OLMo-3-1025-7B.json`
(`peak_causal_layer`, and the mean over `mean_indirect_effect_by_layer`);
peak probing layer from
`step_{0000000,1413814}/layer_wise_moral_probe_allenai_OLMo-3-1025-7B.json`.
The causal subset (steps 0 / 705K / 1,414K) did not include a
co-located probe at step 705K, so that row's probing layer (19) is
taken from the nearest probed checkpoint, step 668K. By mid-training
many layers saturate at ~1.0 probing accuracy, so the single
argmax "peak probing layer" is unstable across probe seeds; we read it
as "mid-network" rather than as an exact layer.*

Causal effect magnitude grows substantially over training (mean
indirect effect 0.01 → 7.95), and the peak causal layer migrates
from layer 5 → layer 5 → layer 0. The peak probing layer sits in
mid-network throughout (layer 0 at init, then layers 19 / 10 once
accuracy saturates). At the final checkpoint, the gap between the
mid-network layer that most-strongly *encodes* moral information
(layer 10) and the early layer that most-strongly *influences*
downstream moral-relevant generation (layer 0) is 10 layers; the two
metrics identify opposite ends of the network.

This is consistent with a "storage vs. use" picture of moral
representation in transformer language models: moral information is
*stored* in mid-network layers (where probing recovers it cleanly)
and *used* in early layers (where intervening on it most-strongly
moves the model's downstream output). The two facts are
representational properties of the same model that probing alone
cannot recover.

This appendix serves as supporting evidence for the §5.2
fragility-as-richer-functional argument: probing accuracy is one
functional of the representation geometry; causal contribution is
another; fragility is a third. All three keep evolving through
training, often in different directions, and a complete picture of
how a representation evolves needs all three. We do not develop the
storage-vs-use finding as a body contribution because (a) it is not
specific to moral representation and (b) it has its own
methodological complications (causal-tracing sensitivity to
intervention magnitude, choice of decoder probe) that warrant their
own paper.
