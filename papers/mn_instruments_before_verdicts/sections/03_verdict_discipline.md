# 3. Verdict discipline {#verdict-discipline}

Three estimator and intervention patterns gate how a number becomes a verdict.

## 3.1 Ratio-of-ratios over MDE-crossing {#ratio-of-ratios}

Whether an effect clears its minimum detectable effect is power-dependent. Comparing two
effects by which side of the MDE each lands on is the overlap fallacy: it reads a difference
in power as a difference in kind. Compare two effects instead by a within-outcome ratio and a
bootstrap CI on the ratio difference (the estimator-traps trap-12 pattern).

**Worked case: the `under_transfer` reclassification.** The first headline was
`reads_non_vmoral_features` at n=11, resting on an absolute transport comparison (a
V_moral-restricted patch clears its MDE, a comparison patch does not). That absolute
comparison was necessary but not sufficient. Re-run at n=23 with a within-outcome
normalization, the honest verdict was `under_transfer`: the restricted patch moves refusal
less than the full patch, but the two do not sit on opposite sides of a categorical line. The
powered decisive cells (n=23 request-twins) are full→refusal −0.0833, V_moral-restricted→refusal
−0.0282, complement→refusal −0.0636, harm-rank-1→refusal −0.0261, random-rank-3→refusal
−0.0005, full→judgment +0.0459, restricted→judgment +0.0237, against a refusal MDE of 0.0238
and a judgment MDE of 0.0086. `under_transfer` was then itself superseded by the rank sweep
(`harm_saturating`, §6), but the estimator lesson is the one that recurs: the reclassification
from `reads_non_vmoral_features` to `under_transfer` happened because the ratio, not the
MDE-crossing, is the comparison of record. (The specificity claim that survives is stated as
a difference CI, not an overlap check: V_moral-restricted moves refusal more than a random
rank-3, Δ = 0.031, paired 95% CI [0.020, 0.043], excludes 0.)

A second instance, from reasoning models: an early raw diff-of-means null (harm direction 0.44–0.49 of
residual norm) read as "harmfulness is not causally encoded," but that was a magnitude
artifact. Reply-inversion recovered the causal signal (Qwen2.5-14B-Instruct shift +17.4 flips
33%, Llama-3.1-8B-Instruct +3.0 flips 23%). Magnitude and residual-norm share are not causal
relevance; a causal readout is.

## 3.2 Power tables before compute {#power-tables}

Compute MDE(n) from measured within-condition variance before spending compute. If no
feasible n resolves the effect, the block is the instrument, not the sample, and the compute
run is futile.

**Worked case: the Llama bounded-unresolved table.** The Llama refusal cells came back chaotic
(§3.3, §4). The temptation was a larger same-design re-run. The power table, built from
saved within-level arrays, said the re-run was futile: the ratio-of-ratios CI on the latched
denominator was [−2.3, 4.9], and no feasible n at that variance closes it, because the
denominator is saturated rather than noisy. The underpowered Llama `R_refusal_k` and
`R_judgment_{k>1}` cells were voided as denominator-latched, and the clean channel was
identified as the *reverse* (disengage) direction, not more samples in the forward one. An
afternoon of saved-array work prevented a compute run. This is the general rule: saving per-pair /
per-rollout / per-head arrays by default keeps the power computation zero-GPU, so futility is
caught before the session, not after.

## 3.3 The orthogonal-cell certificate {#orthogonal-cell}

When a causal readout comes back chaotic, root-split against an orthogonal outcome the same
intervention should move. If the orthogonal cell is coherent, the instrument is certified and
the chaos is a property of the read-out outcome, not a broken patch.

**Worked case: Llama content-swap.** On Llama the content-swap interchange patch produced
sign-chaotic refusal deltas (SD 0.31, median +0.029) against OLMo's clean −0.083, which read
as a broken instrument. The judgment cell is the positive control: the *same* patch moved
judgment coherently (CI excludes 0). So the patch works. The refusal chaos is saturation: the
boundary-violating twins sit at the refusal ceiling (baseline refuse 0.83–1.0), so the
decision-token refusal projection is latched and has no room to move, and the refusal-delta SD
grows with severity (0.296 → 0.352) as saturation deepens. OLMo's refusal moved because it was
weak (unsaturated). A causal readout run at a saturated outcome yields chaotic, sign-unstable
deltas that mimic instrument failure; the orthogonal cell tells the two apart.
