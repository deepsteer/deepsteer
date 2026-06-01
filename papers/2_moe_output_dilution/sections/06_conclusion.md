# 6. Conclusion

We asked whether Mixture-of-Experts architectures create expert-level
moral specialization — discrete modules that concentrate moral
features and offer natural intervention points for alignment. The
answer is no. In OLMoE-1B-7B, all 64 experts at every layer encode
moral content with comparable accuracy (Gini $< 0.03$), and the
router shows no preference for routing moral content to specific
experts (maximum preference 1.8%).

This null on specialization led to a positive finding about
architecture. MoE models are 5.1$\times$ more fragile than dense
models on moral probing despite matching on accuracy, and the
mechanism is *output dilution*: the MoE block's contribution to the
residual stream is 74$\times$ smaller in scale than the dense MLP's,
because sparse aggregation (top-8 of 64 experts) attenuates each
expert's contribution. The moral signal is present but encoded at a
scale that is trivially overwhelmed by noise.

The checkpoint trajectory analysis (§4.5) deepens the null: the
absence of specialization is not a late-training convergence but is
present from the earliest available checkpoint (step 5K, 20B tokens).
Moral encoding appears before the routing mechanism has fully matured,
and the Gini coefficient of per-expert accuracy remains below 0.03
throughout training. Training does not create or destroy expert moral
specialization — it was never there.

This finding refines the methodological program of companion work
\citep{reblitzrichardson2026fragility}. That work established fragility testing
as a complement to probing accuracy for tracking alignment depth
during pre-training. The present work shows that the gap between
probing accuracy and fragility is not just a temporal phenomenon
(fragility resolving after accuracy saturates) but an architectural
one: MoE's sparse aggregation creates a permanent structural
fragility that no amount of training can resolve without changing
the aggregation mechanism.

For future work, the output dilution mechanism makes specific
predictions. Models with higher sparsity (lower $k/N$ ratio) should
show greater fragility. Fine-tuning should produce larger fragility
shifts in MoE than in dense models of comparable active size. And
MoE architectures that aggregate expert outputs differently —
concatenation, attention-based mixing, or denser routing — should
show correspondingly different fragility profiles.
