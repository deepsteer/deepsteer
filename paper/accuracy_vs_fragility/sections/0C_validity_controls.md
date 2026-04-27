# Appendix C. Standard moral probe validity controls

The persona probe used in companion work (Reblitz-Richardson, 2026,
in preparation; see `outputs/phase_d/c8/`) was validated against
three controls that have become standard for linear-probing studies
that claim to measure something beyond surface vocabulary:

1. **Leave-lexeme-out splits.** Train the probe with all pairs
   containing a target lexeme (e.g. all "murder" pairs) held out;
   evaluate on those held-out pairs. Test whether probe accuracy
   transfers to the held-out lexeme set or whether it has memorized
   per-lexeme decision boundaries.
2. **Paraphrase transfer.** Generate paraphrases of test-set pairs
   that preserve moral content but vary surface form; evaluate the
   probe trained on the original test-set pairs on the paraphrased
   versions. Test whether the probe recovers the moral signal under
   surface variation or whether it is reading per-pair surface
   features.
3. **Adversarial lexical swap.** Construct adversarial pairs where
   a surface feature the probe might be using (sentence length,
   position of the moral lexeme, presence of specific function
   words) is decoupled from the moral label. Test whether the probe
   accuracy degrades on the adversarial set.

For parity with the persona probe, the standard moral probe should
ideally be validated on all three controls before submission.
However, the compositional moral probe (§3.2) addresses the
strongest version of the "your probe is just reading moralized
vocabulary" review attack *by construction*: pairs share the
morally-loaded action verb between halves and differ only in
individually-mild tokens that cannot carry moral signal in
isolation (TF-IDF baseline 0.113 ≪ 0.65). We argue this partly
obviates the leave-lexeme-out and adversarial-lexical-swap controls
for the standard probe specifically — the compositional probe is a
strictly stronger version of both controls combined.

**Status.** This appendix is a stub. The leave-lexeme-out, paraphrase
transfer, and adversarial lexical swap controls for the standard
moral probe are *deferred* — the compositional probe (§3.2) handles
the most important review concern they address, and running them
takes ~4-6 hours of additional MPS time on existing checkpoints.
We will run them before submission for parity with the persona
probe and to preempt narrower review concerns; this appendix will
populate with the resulting numbers and per-control discussion.
Tracked in `RESEARCH_PLAN.md` Open items.
