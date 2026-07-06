# Appendix A. Foundation emergence

The standard moral probe's 240 minimal-pair dataset is balanced
across the six Moral Foundations Theory categories (Haidt, 2012;
Graham et al., 2013) at 40 pairs per foundation: care/harm,
fairness/cheating, loyalty/betrayal, authority/subversion,
sanctity/degradation, and liberty/oppression. The foundation-
stratified probe (`FoundationSpecificProbe`) trains a separate linear
classifier per foundation, allowing per-foundation onset and plateau
analysis across the same 37 OLMo-2 1B early-training checkpoints
used in §4.

**Headline finding.** All six foundations stabilize (reach 100% peak
probing accuracy) by step 3K, within the same window where the
aggregated standard moral probe onsets. Per-foundation onset
differences are within sampling noise: the per-foundation peak
accuracies are sixteenths on a 16-example test, single seed (e.g.
step-1K authority 100% = 16/16 vs. care 87.5% = 14/16, a two-example
gap), so we do not read the apparent step-1K / 2K / 3K ordering as a
reliable per-foundation sequence.

| Foundation | Step 0 | Step 1K | Step 2K | Step 3K | Step 6K | First step at 100% |
|------------|-------:|--------:|--------:|--------:|--------:|-------------------:|
| authority/subversion | 68.8% | 100% | 100% | 100% | 100% | 1K |
| care/harm | 68.8% | 87.5% | 100% | 93.8% | 100% | 2K |
| fairness/cheating | 75.0% | 75.0% | 100% | 100% | 100% | 2K |
| sanctity/degradation | 68.8% | 75.0% | 93.8% | 100% | 100% | 3K |
| loyalty/betrayal | 62.5% | 62.5% | 93.8% | 100% | 100% | 3K |
| liberty/oppression | 68.8% | 93.8% | 87.5% | 100% | 100% | 3K |

*Per-foundation peak probing accuracy at key OLMo-2 1B early-training
checkpoints. Numbers source: `outputs/phase_c1/` foundation-specific
probe results.*

**All six foundations stabilize by step 3K.** During development,
an earlier dataset with weaker neutral-pair quality produced
unstable liberty/oppression encoding; the instability resolved when
neutral sentences that inadvertently carried moral content were
removed. This sensitivity to dataset quality confirms the
importance of the validation methodology described in §3.1.
The apparent per-foundation ordering (authority at step 1K, care and
fairness at step 2K, the remaining three by step 3K) is within the
sampling noise of a 16-example, single-seed test; the reliable
statement is that all six stabilize by step 3K.

Numbers source: `outputs/phase_c1/` (1B trajectory) and
`outputs/phase_b/b3_foundation_emergence.png` (7B comparison).
