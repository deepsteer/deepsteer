# 11. Conclusion {#conclusion}

What refusal reads varies by model family: on OLMo-3, the one model we test causally, it reads
the harm percept, a low-rank slice, not the broad moral subspace, while on Llama-3.1 it reads
broadly. In open-weight chat models the moral
representation is deep and inherited, a broad low-rank subspace that forms in pretraining
(crystallizing to a checkpoint-to-final cosine of 0.999) and survives alignment as a single
rotation. The refusal decision built on top of it is shallow by construction: a fresh
post-training gate (proto-refusal-to-gate cosine 0.155) in a narrow control-token channel, reading a
rank-1 harm slice that a nested interchange sweep on OLMo shows saturating while judgment
coupling keeps climbing. A four-model panel separates *what* refusal reads (harm versus broad
moral content) from *how* it commits (at the read layer, early, or reversibly), with GPT-OSS as
an existence proof that a deliberating model can read harm and still be argued out of a refusal.
The positive claim is sharp and actionable: alignment is shallow because the harm-keyed refusal
decision reads a small separable feature rather than the model's moral understanding. The same anatomy
that explains why refusal is easy to remove also names where deeper alignment would have to
write: into the directions judgment already reads.

**Safety scope.** This work characterizes the refusal geometry of released open-weight models,
in order to explain a known property (that refusal is cheap to remove) and to locate where a
future intervention would act. It reports what these models do; it does not build or optimize a
method for removing refusal, and the forward target it names, widening what the writing heads
read, is a direction for deeper alignment, not for defeating it. The interchange, ablation, and
prefill cells are measurements on the models' own forward passes, run to understand the
decision, not to weaken it.
