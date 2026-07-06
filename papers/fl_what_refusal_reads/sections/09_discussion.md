# 9. Discussion {#discussion}

**A mechanism for shallow alignment.** The pieces compose into a concrete account of why
post-hoc alignment is shallow. Moral comprehension is deep and inherited: a broad, low-rank
moral subspace forms during pretraining and survives alignment as a single rigid rotation
(\Cref{comprehension-native}). The refusal decision is a thin control built on top of it:
a fresh post-training gate (\Cref{fresh-gate}) that lives in a narrow control-token
channel (\Cref{bottleneck}). What that gate reads varies by family. On OLMo-3, the one model
we test causally, it reads only the harm percept, a low-rank slice of the moral subspace nearly
orthogonal to the bulk (\Cref{reads-harm}); Llama-3.1 is the exception, reading broad moral
content (\Cref{cross-model}). Where the read is harm-keyed, the compliance wrapper reads a
harm cue over a narrow bus and the model's moral understanding sits mostly off that bus. This is
not that comprehension is causally absent: on OLMo the same patches that barely move refusal move
judgment across the whole subspace. Refusal there simply does not consult it. On the harm-keyed
models, alignment is shallow because the refusal decision reads a small, separable feature rather
than the model's moral representation.

**Why refusal is easy to remove.** The account predicts the removability that motivated this
work. A control that occupies a low-variance channel and reads a rank-1 harm slice is a small
target. A rank-one edit that cancels the refusal direction leaves the moral subspace untouched,
because the refusal decision was not reading it [@arditi2024refusal]; that is why automated
censorship removal succeeds without degrading the model [@pew2025heretic]. Removability is not
a surprising fragility; it is what a harm-keyed gate over a narrow bus looks like from the
outside. The same geometry explains why safety behavior can be trained shallowly enough to be
concealed [@hubinger2024sleeper] or faked under monitoring [@greenblatt2024faking]: the
decision that governs the behavior is not wired into the representation that would have to
change for the behavior to change deeply.

**A forward target.** If shallow alignment is refusal reading a harm sliver, deep alignment is
refusal reading more of the moral subspace. The anatomy makes the target precise: the heads
that write the decision channel (\Cref{reads-harm}) currently transfer the harm rank-1
direction and saturate there, while judgment transfer keeps climbing as the basis widens. The
intervention is to make those writing heads read the directions judgment already reads, so that
refusal transfer follows the judgment curve instead of clipping at the harm ceiling. The small
non-harm residual the interchange detects at or near its detection limit (the harm-partialed
patch still moves refusal about half the harm effect, point estimate $-0.0133$ below the
0.0238 refusal MDE with a bootstrap CI that excludes 0) is the toehold: it is the one place, on
this instrument, where refusal reads moral content beyond harm, so a rank-2 non-harm sliver,
not the broad subspace, is where any deepening would begin. Whether widening the read also deepens the behavior, and at what
cost to the model, is the question the two-axis panel raises and does not answer.

**Deliberation can be load-bearing and reversible.** GPT-OSS is an existence proof that the
harm-keyed reflexive gate is not the only design point. Its refusal reads harm like OLMo's, but
its commitment is reversible: a graded exculpatory argument flips a ceiling refusal to
compliance, and an inculpating argument flips a benign request to refusal
(\Cref{gpt-oss}). A deliberating model can hold its refusal open to argument, which is a
different, and in some respects more auditable, safety property than a fixed reflexive gate,
because the decision is exposed to and moved by explicit reasoning rather than settled before
the trace begins. It also has a failure mode the reflexive gate does not: a refusal that can be
argued down can be argued down by an adversary's prefill. The two-axis panel makes the design
choice explicit rather than settling it.
