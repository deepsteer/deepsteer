# 4. Stimulus discipline {#stimulus-discipline}

The operating-point rule: discrimination screens must bracket the point where the outcome
actually moves. A saturated outcome latches the readout. Use a severity ladder and a boundary
band (outcome ~0.5), and report the psychometric curve, not a single point.

This is where the readout itself can lie. Reasoning models defeat clean judgment readouts
(regex, final-answer, forced-logit), while a plain instruct model is clean on the same battery
(Qwen2.5-14B-Instruct: 24/24 harmless-safe, 24/24 harmful-harmful). The readout has to be
validated on the model class it is run on, at an operating point where the outcome is not
pinned.

## 4.1 A deliberation/prefill asymmetry is operating-point-confounded when the gate is a step {#a6-deliberation}

**Failure as it appeared.** On GPT-OSS-20B the reasoning-prefill deliberation cell (engage =
inculpating prefill, disengage = exculpating prefill) emitted a clean-looking asymmetry
`A = 1.0`: engage flips benign→refuse, disengage flips violating→comply 0/7. It read as
one-way early commitment, as if deliberation only ever pushes toward refusal.

**The tell.** `A = 1.0` with a bootstrap CI of width 0. The disengage arm is uniformly 0, so
every resample returns 1: a degenerate CI, not a precise estimate. The rule-of-three exposes
it directly, disengage 0/7 gives a 95% upper bound of ~0.43, not 0. The asymmetry is the same
dynamic-range confound as §3.3: the disengage arm was tested on violating items that already
refuse at baseline (at the ceiling), while the engage arm was tested on unsaturated benign
items (with room to move up). An asymmetry statistic that compares an arm-with-headroom against
an arm-at-the-ceiling measures the operating points, not the model. A companion trap in the
same run: the harm-separability commitment curve reads ~1.0 from the first trace bin, but that
measures when *harm is represented* (harmful and harmless traces differ from the start),
not when the *decision* is fixed.

**Why the usual fix is not enough here.** That fix was boundary-band twins (outcome ~0.5).
GPT-OSS's gate is a step: the severity ladder finds no unsaturated violating level (empty
boundary band, 5.6% of items in the mid-band). The operating point cannot be bracketed
behaviorally at the existing resolution, so more boundary-band stimuli do not exist to collect.

**The protocol (the de-confounder).** Replace the binary disengage flip with a graded
exculpatory prefill series (weak→strong) and read a continuous projection (the decision-channel
residual under each prefill onto the refusal direction) alongside the behavioral flip. The
graded projection registers sub-flip movement, so "no flip at maximum prefill" splits cleanly
into *reversible* (projection moves toward comply) versus *genuine downward-robustness*
(projection flat). Saturation can no longer masquerade as commitment. A pre-registered
band-existence check decides whether a finer ladder is even buildable: the per-item base-refuse
histogram is read as smooth (resolution-limited, build a finer ladder) versus bimodal (a step,
switch to the graded readout).

**The certifying check.** Under the graded series, GPT-OSS is a reversible reader: strong
exculpatory prefill flips ceiling-refusing violating items to comply 6/10, and the
decision-channel refusal projection moves monotonically toward comply in all 10 items
(frac_projection_moved 1.0, frac_monotone 1.0). The behavioral flip is the primary readout; the
monotone projection corroborates it (with a last-token caveat on the projection). The engage
direction is separately consequential: an inculpating-analysis prefill flips unsaturated benign
requests to refuse 7/7 (Wilson 95% [0.65, 1.0]), so the decision is not fixed before the trace.
The first-run disengage 0/7 was the saturation trap, now resolved. Report the behavioral flip
and the graded readout separately; the asymmetry statistic itself is not reported, because it
is uninterpretable when one arm sits at the ceiling.

This is the operating-point rule's reasoning-model instance: when the gate is a step and the
boundary band is empty, do not report the asymmetry; switch to a graded intervention with a
continuous readout that registers sub-threshold movement.
