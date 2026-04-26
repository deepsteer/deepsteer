# 6. Conclusion

Probing accuracy is the wrong instrument for studying training
dynamics. Whatever continued representational change a language model
undergoes during pre-training after a property becomes linearly
decodable — and §4 shows there is a great deal of such change — does
not register on a metric that has already saturated. The methodological
contribution of this paper is to take this saturation as a fixed
feature of probing accuracy and add a complementary metric (per-layer
critical noise, which we call *fragility*) that recovers the missing
resolution. The contribution is not specific to moral representations
or to pre-training trajectories; we have established the
fragility-resolves-what-accuracy-misses pattern on the standard moral
probe (§4.3) and on the compositional moral probe across four random-
seed splits (§4.3, replication), and demonstrated in companion work
(§5.3) that the same pattern reproduces on fine-tuning fingerprints.
We expect the methodology to extend to any binary probing-based
investigation of neural network representations where the operating
point is high enough to saturate the standard accuracy curve.

The science findings earn the methodology its keep. The most
consequential of them is the *quantitative gradient* the paper
establishes by adding the compositional moral probe alongside the
standard moral, sentiment, and syntax probes: lexically-marked
moralized vocabulary is decoded at step 1K, compositional moral
integration at step 4K, syntactic competence at step 6K. The
strongest one-sentence summary of the moral probe's step-1K onset —
that "moral encoding emerges within the first 5% of pre-training" —
overstates what the standard probe recovers; the gradient framing
("lexically-marked moralized vocabulary at 1K, compositional moral
integration at 4K, syntactic competence at 6K") is the honest
reading. We treat this as an existence proof that probing-claims
about pre-training emergence should default to lexical-accessibility
hedges until a compositional-or-stronger ablation is run.

Two open questions Phase C4 leaves are gated at Phase E (7B / 32B
replication on open-checkpoint models). First, **does the
compositional plateau at ≈0.77 lift with model scale?** The
compositional moral and syntax plateaus coincide at ≈0.77 under
mean-pooled linear probing while the standard moral and sentiment
plateaus saturate near 0.97; we cannot distinguish a 1B-model
ceiling from a probe-side ceiling at 1B with linear probing alone.
The cleanest disambiguation is repeating §4.1 at 7B and 32B: if
compositional moral accuracy rises with scale while syntax accuracy
does not, the bottleneck at 1B is the model not the probe. Either
result refines the gradient finding without overturning it.

Second, **does the fragility-resolves-what-accuracy-misses pattern
extend beyond the probing investigations we have run?** We have
established it for the standard moral probe and for the compositional
moral probe; we have not established it for arbitrary probing-based
investigations of pre-training. A natural next step is to apply the
fragility instrument to other probes that hit accuracy ceiling — both
within the moral domain (foundation-stratified compositional probing,
counterfactual-sensitivity probing) and outside it (linguistic-
property probes, factual-recall probes, persona-feature probes). If
fragility resolves dynamics that accuracy misses across this broader
set, the methodology is a general-purpose tool for the alignment-
during-pre-training research program; if not, the conditions under
which it generalizes are themselves the contribution of follow-up
work. We treat both gates as Phase E priorities.

The pre-training window is doing more work than the standard
interpretability instrument shows. Adding a fragility readout
recovers a substantial fraction of that work. The lexical→compositional
gradient is what we recover when we apply the methodology to moral
representations specifically; the broader claim is that the same
methodology will recover analogous structure wherever probing-based
investigations of pre-training currently hit the accuracy ceiling.
