# 7. Conclusion

We separated moral comprehension from moral compliance and tracked both across
the full OLMo-3 7B alignment pipeline and a refusal-ablated model. Moral
comprehension is a pre-training property: decodable at 100\%, five-dimensional,
and crystallized into its final form before any alignment. Post-training
reorients this representation once, at SFT, and then leaves it untouched through
DPO and RLVR. The behavioral compliance alignment adds is only weakly coupled to
the moral representation, and the refusal mechanism that carries it is nearly
orthogonal to the moral subspace, so ablating refusal removes compliance while
leaving comprehension and moral judgment intact. Alignment, on this evidence, is
a wiring step rather than a teaching step: it attaches a separable compliance
mechanism to moral understanding that pre-training already built. The robustness
question for alignment is therefore not whether models understand morality, they
do, but whether their behavior is coupled to that understanding tightly enough
to survive having the wiring pulled.

We then asked whether that coupling can be installed after the fact (Section 6).
It cannot, at least not during post-training. Alignment does increase the model's
functional reliance on its moral subspace for generation, and a training-time
loss can amplify that reliance to a large value, but the dependence it builds
sits in the moral subspace, orthogonal to the refusal direction that an attacker
removes. The compliance mechanism's location is fixed earlier, when
post-training recruits proto-features from a pre-trained representation in which
moral structure is already orthogonal to those features. Tightening the coupling
therefore has to move upstream: a pre-training intervention that lets the moral
subspace absorb the proto-features post-training will use for compliance, so that
removing the refusal direction would also damage moral comprehension. Whether
alignment can be made ablation-resistant is, on this evidence, a question about
pre-training, not post-training.
