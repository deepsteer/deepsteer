# 5. Discussion

**Alignment is a wiring problem.** The four findings form a single argument.
Moral comprehension is built in pre-training and is complete before alignment
(4.1). Post-training does not add comprehension; it rotates the existing moral
subspace once, at SFT, and otherwise leaves it alone (4.2). The behavioral
compliance that post-training does add is only weakly coupled to that
representation (4.3), which predicts, and then ablation confirms, that
compliance is carried by a mechanism geometrically separate from morality
(4.4). The model does not learn right from wrong during alignment. It learns,
loosely, to act on a notion of right and wrong it already had, through a
refusal mechanism that sits outside its moral representation.

**Why aligned behavior is removable.** This gives a mechanistic account of a
known fragility: jailbreaks, fine-tuning attacks, and direction ablation strip
aligned behavior easily. Our result says why. Because compliance is not routed
through moral representations, removing it does not require, and does not cause,
any loss of moral understanding. The ablated model still represents the six
foundations at five dimensions and still judges scenarios at the un-ablated
rate; it simply no longer refuses. Decensoring is decoupling, not un-teaching.
A defense posture that tries to make models "understand morality better" is
aimed at the wrong variable: comprehension is already saturated in the cases
where behavior fails.

**Weak coupling is the vulnerability, and a lever.** The near-zero coupling
($\phi \approx 0.05$) is the quantity that makes compliance separable. It also
suggests where robustness could be built: an alignment procedure that made
compliance *depend* on the moral subspace, so that the refusal mechanism and the
moral representation share a direction, would make refusal harder to remove
without damaging comprehension, exactly the coupling our pipeline does not
produce. Whether such coupling can be trained, and at what capability cost, is
an open question this work motivates.

**The empty row.** No OLMo-3 state, from the first pre-training checkpoint to the
ablated model, has low moral comprehension. Moral content is decodable at 100\%
throughout. One cannot obtain a low-comprehension model by checkpoint selection;
comprehension is universal and early. The interesting variation is entirely in
behavior, which is the regime where probing-based safety arguments are weakest:
a probe that decodes morality perfectly says nothing about whether the model
will act on it.

**Limitations.** The coupling and behavioral measurements use 48 scenarios and
keyword judgment parsers; the coupling trend is directional, not statistically
strong, and we present it as the qualitative result that comprehension and
compliance are barely linked, not as a precise coefficient. We study one model
family (OLMo-3 7B); the crystallize-then-rotate trajectory and the dissociation
should be checked on other pipelines. The refusal direction is taken at a single
stable layer and ablated uniformly (Arditi) rather than via Heretic's per-layer
optimization, which trades maximal decensoring for a controlled, interpretable
intervention. The persona probe captures stylistic separability, not a
mechanistic persona feature. These do not affect the central geometric results
(transfer cosine, effective dimensionality, refusal--morality orthogonality),
which are stable and saturated.
