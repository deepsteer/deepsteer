# 7. Checklist: the reusable protocol {#checklist}

The ship-blocker gates below are the portable form of the program's discipline. They are
appendix-form restatements of the companion skills.

**Before any projection-fraction / cosine geometric cell:**

- Record `participation_ratio` at the measurement position. If PR < 30, the position is
  invalid for content projection-fraction tests; report a decision-direction cosine
  instead, or move to a valid position.
- Check the positive-control band against the covariance null at that position. Band-below-null
  means the instrument has no discriminating power there; do not report absence.
- In a massive-activation family (any Llama/Qwen-class panel), inspect the null value for
  saturation and the top-dim variance share. Standardize (z-score, sinks excluded) and, as a
  robustness variant, project out each >5%-variance dim. Certify with a clean model
  (raw→standardized same verdict). When standardization and projection-out disagree, the space
  is genuinely degenerate; change format or position, not the null.

**Before any per-head OV / logit-lens attribution:**

- Detect reordered norm (post-block `post_feedforward_layernorm`). If present, fold the
  per-layer RMSNorm gain (unit-test the fold to ~1e-9). Gate reconstruction two-sided
  (0.90 ≤ recon ≤ 1.10); a one-sided floor misses overshoot.

**Before any patch / ablation / interchange / steering verdict:**

- Pre-register the intervention spec block (stimulus–outcome baseline matching, transport
  positive control, channel-matched specificity null, token-alignment rule, harness parity for
  the outcome classifier) before the run.
- Certify a chaotic readout with an orthogonal cell the same intervention should move; if the
  orthogonal cell is coherent, the chaos is saturation, not a broken instrument.
- Compare two effects by a within-outcome ratio + bootstrap CI on the ratio difference, never
  by which side of the MDE each lands on.
- State the intervention depth relative to commitment; depth-match cross-model comparisons to
  the pre-commitment coherent layer.

**Before any null / orthogonality / below-threshold verdict:**

- Build the calibrated ladder (floor → matched null → measurement → positive band) and re-verify
  every control's defining property in the current context.
- State the minimum detectable effect. A null without an MDE has no teeth; write the detection
  bar into the sentence ("no coupling detectable above |cos| 0.10 against a null q95 of 0.41,"
  not "dissociation").

**Before any stimulus screen:**

- Bracket the operating point with a severity ladder and a boundary band (~0.5); report the
  psychometric curve. If the gate is a step (empty boundary band), switch to a graded
  intervention with a continuous readout and report the behavioral flip and the graded readout
  separately. Validate the readout on the model class it is run on.

**Before a compute run:**

- Compute MDE(n) from measured within-condition variance. If no feasible n resolves the effect,
  the block is the instrument, not the sample; do not spend the compute. Save per-pair /
  per-rollout / per-head arrays by default so the power computation and later statistics stay
  zero-GPU.

**Before any commit or draft:**

- Every printed scalar traces to an anchored claim-ledger row; voided numbers stay in a register
  of superseded claims with their replacements so they cannot re-enter prose as findings.
