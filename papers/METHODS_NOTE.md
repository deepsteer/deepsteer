# Methods note (skeleton) — disciplined causal interpretability of alignment circuits

Working title: *Instruments before verdicts: measurement discipline for causal claims about refusal and
moral representation.* Started 2026-07-02 (Amendment 10). A methods paper carrying the anomalies (A1–A5)
and the estimator/intervention patterns the D1–D3 program kept re-deriving. The scientific results live
in the direction papers; this note is the portable methodology.

## Thesis

A causal claim about what a circuit reads or writes is only as good as the instrument that measured it,
and interpretability instruments fail in specific, diagnosable ways. Each failure below looked like a
result until an instrument check caught it. The discipline is: **calibrate the instrument, certify it
with an orthogonal cell, compute power before the pod, and state every read-from verdict at a stated
depth relative to commitment.**

## Anomalies (promoted measurement findings; full text in `ANOMALIES.md`)

- **A1** — covariance-matched nulls degenerate in massive-activation families; standardize (or project
  out >5%-variance dims), and prove legitimacy with a clean-model raw→standardized invariance check.
- **A2** — the decision site is a low-dimensional control-token bottleneck; band-below-null ⇒
  position-invalid instrument. (Strengthened cross-model in A5.)
- **A3** — reordered-norm architectures (OLMo-2/3) overshoot per-head OV attribution ~3×; fold the block
  RMSNorm gain; use a two-sided reconstruction gate.
- **A4** — variance/purity/alignment do not imply causal relevance; low-rank restrictions can be
  nonlinearly inert (the causal counterpart of the eff-dim caution).
- **A5** — massive activations are position-dependent (clean at the decision channel despite a global
  outlier); and interchange patches die at outcome saturation — certify with an orthogonal cell.

## Estimator / intervention patterns

1. **Ratio-of-ratios over MDE-crossing.** Whether an effect clears its MDE is power-dependent; compare
   two effects by a within-outcome ratio + a bootstrap CI on the ratio difference, never by which side
   of the MDE each lands on (estimator-traps trap 12).
2. **Power tables before the pod.** Compute MDE(n) from measured within-condition variance; if no
   feasible n resolves it, the block is the instrument, not the sample — don't spend the compute.
3. **Orthogonal-cell certificate.** Diagnose a chaotic readout by root-splitting against an orthogonal
   outcome the same intervention should move (refusal vs judgment); if the orthogonal cell is coherent,
   the instrument is certified and the chaos is a property of the saturated outcome.
4. **Operating-point / dynamic-range.** Discrimination screens must bracket the operating point — a
   saturated outcome latches the readout; use a severity ladder and a boundary band (~0.5), and report
   the psychometric curve.
5. **Depth-indexed intervention verdicts (new, Amendment 10).** A verdict about what a circuit *reads*
   must state the **intervention depth relative to commitment**. A patch at the read layer is
   post-commitment for an early-committing model, so a "reads-X" or asymmetry claim measured there can
   be a read-layer artifact — measure at (and report) the pre-commitment coherent depth, and
   depth-match cross-model comparisons.

## Structure (draft)

1. Intro — causal interpretability's instrument problem.
2. The decision-site instrument (A2/A5) and its calibration ladder (A1/A3).
3. Verdict discipline — ratio-of-ratios, power tables, orthogonal-cell certificates (patterns 1–3).
4. Stimulus discipline — operating point, severity ladders, boundary bands (pattern 4).
5. Depth discipline — commitment-relative intervention verdicts (pattern 5).
6. Case study — the D3 refusal-reads-what/commits-how program, each amendment as a caught failure.
7. Checklist — the pre-registration + verification gates as a reusable protocol.

## Companion skills

`.claude/skills/instrument-calibration`, `intervention-validity`, `estimator-traps`, `construct-audit`,
`compute-ordering` — the operational encodings of the above.
