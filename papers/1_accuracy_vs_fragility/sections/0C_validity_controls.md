# Appendix C. Standard moral probe validity controls

Three controls are standard for linear-probing studies that claim to
measure something beyond surface vocabulary:

1. **Leave-lexeme-out splits.** Train the probe with all pairs
   containing a target lexeme (e.g. all "betray" pairs) held out;
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

The compositional moral probe (§3.2) addresses the strongest version
of the "your probe is just reading moralized vocabulary" concern *by
construction*: pairs share the morally-loaded action verb between
halves and differ only in tokens that carry limited moral signal in
isolation (unigram lexical floor 0.63). The probe's compositional
encoding is established directly by leave-construction-out transfer
(0.85 hidden-state vs. 0.60 bag-of-words) and by decoding 0.20-0.28
above the per-construction lexical floor (§3.2, §4.1). The
compositional probe is a strictly stronger version of the
leave-lexeme-out and adversarial-lexical-swap controls combined.
Running these three controls on existing checkpoints takes ~4-6
hours of additional MPS time; results will be included in the
submission version.
