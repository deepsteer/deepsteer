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

**Headline finding.** Foundations emerge in a staggered sequence,
with five foundations saturating cleanly within the same window
where the aggregated standard moral probe onsets, and one foundation
(liberty/oppression) failing to fully stabilize at either 1B or 7B
scale.

| Foundation | Step 0 | Step 1K | Step 2K | Step 3K | Step 6K | First step at 100% |
|------------|-------:|--------:|--------:|--------:|--------:|-------------------:|
| fairness/cheating | 68.8% | 100% | 100% | 100% | 100% | 1K |
| care/harm | 75.0% | 87.5% | 100% | 100% | 100% | 2K |
| sanctity/degradation | 75.0% | 75.0% | 93.8% | 100% | 100% | 3K |
| loyalty/betrayal | 56.3% | 68.8% | 87.5% | 100% | 100% | 3K |
| authority/subversion | 81.3% | 81.3% | 93.8% | 93.8% | 100% | 6K |
| liberty/oppression | (not stable; details below) | | | | | — |

*Per-foundation peak probing accuracy at key OLMo-2 1B early-training
checkpoints. Numbers source: `outputs/phase_c1/RESULTS.md` Finding 3.*

**Liberty/oppression cross-scale outlier.** The liberty/oppression
foundation never fully stabilizes at either OLMo-2 1B or OLMo-3 7B
final checkpoints. The pattern is consistent across both scales,
suggesting this foundation is intrinsically harder to encode from
web-text pre-training rather than a low-resource artifact at the 1B
operating point. We treat this as a substantive observation but not
as load-bearing for the methodological thesis; the foundation
imbalance is documented here for completeness and revisited in §5.4
limitations.

**Status.** This appendix is a stub (~half page). The full per-
foundation trajectory plots (`c1_foundation_emergence.png` and
`c1_foundation_peak_accuracy.png` in `outputs/phase_c1/`) and the
liberty/oppression cross-scale comparison need to land before
submission. Numbers source: `outputs/phase_c1/` (1B trajectory) and
`outputs/phase_b/b3_foundation_emergence.png` (7B comparison).
