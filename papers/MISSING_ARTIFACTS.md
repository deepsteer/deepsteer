# Missing artifacts ledger

## A1 (2026-07-01): MFT directions not committed for reasoning tags

- `outputs/phase2/think/mft_directions.npz` absent -> MFT<->V_moral mutual projection not computable for `think`. Queue MFT extraction into B3 if the reasoning-tag subspace comparison is wanted (base/instruct are covered).
- `outputs/phase2/gpt_oss/mft_directions.npz` absent -> MFT<->V_moral mutual projection not computable for `gpt_oss`. Queue MFT extraction into B3 if the reasoning-tag subspace comparison is wanted (base/instruct are covered).

## A3 (2026-07-01): Think refusal vectors not saved

- OLMo-3-Think P0-P3 refusal directions exist only as projections in `think_g3_result.json`, not as `.npz` vectors -> A3 variance-percentile and A4 refusal-p bootstrap cannot run for Think. Re-extract with per-vector saves in B3 if the Think spare-channel / refusal CI is wanted.

## A4 (2026-07-01): per-pair bootstrap gaps

- `instruct`: fables/ethics per-pair diff arrays not committed (`axis_instruct/axis_diffs_*.npz` absent) -> held-one-out band + refusal-p bootstrap CIs not computable. Re-extract with per-pair saves in B3.

## A4 (2026-07-01): per-pair bootstrap gaps

- `instruct`: fables/ethics per-pair diff arrays not committed (`axis_instruct/axis_diffs_*.npz` absent) -> held-one-out band + refusal-p bootstrap CIs not computable. Re-extract with per-pair saves in B3.

## Amendment 2 (2026-07-01): position-validity audits needing un-saved slices

- **Per-position chat act_samples (D2 in-format).** informat_ladder saved only the PRIMARY
  (final_pre_assistant) chat act_sample in chat_vmoral_<key>.npz; the last_content / mean_content
  act_samples were not saved -> the full PR profile across position classes is recomputed by the
  next informat run (PR now a required type-block field). No pod trip beyond the planned informat
  re-run.
- **D1 reasoning P0-P3 per-rollout activations.** The GPT-OSS/Think reasoning runs saved the
  refusal direction vectors (refusal_think_P*.npz) but not the per-rollout window activations, so
  the pre-registered PR audit of the P0-P3 windows (Amendment 2 rider 7) cannot run zero-GPU. Queue
  a per-rollout-activation re-extraction at the P0-P3 windows (small) to PR-audit the reasoning band
  rung; until then the band-relative reasoning statements are scoped cross-position.
- **mean_content slices for the refusal/judgment prompt sets.** B1 saved acts_headline (judgment
  decision-site) only; refusal extraction saved only the direction. The harm-content x V_moral
  (content x content) cell (Amendment 2 rider 7 salvage: Zhao harmfulness vs V_moral, both at
  mean_content) needs mean_content-pooled activations of the harmful/harmless + judgment prompt sets
  -> fold into the next informat/B-chunk extraction.

## Amendment 11 (2026-07-02): D3 Llama epilogue + A5 GPT-OSS pre-conditions

- **Severity-twin paired content contrasts (D3 Llama, rank-2/4 harm-coextensive).** The C1 run saved
  the rank-3 `Vbasis`, the `harm` vector, and the per-k sweep/engage *outcomes*, but NOT the
  per-pair moral contrasts or any severity-twin content contrasts. So the **rank-1** harm-coextensive
  check ran zero-GPU (`harm_coextensive.py`: rank-1 harm spans only 3.6% of the engage-driving basis →
  reads-broad survives), but the **rank-2/4 severity-derived harm basis** (a richer multi-dim harm
  percept) cannot be built. Re-extract the severity-ladder paired content contrasts at Llama layer 12
  (small); feed them to `sweep.nested_pca_basis(..., [1,2,4])` → `sweep.harm_capture_curve` for the
  rank-2/4 capture number. Prior is against harm-coextensive; the builder + unit tests are committed.
- **GPT-OSS harmony decision-token `act_sample` (A5 position validity).** The saved GPT-OSS
  `act_sample` is a *content* position, so the A5 pre-condition (post-std PR + band-below-null at the
  harmony **decision token**) is not computable zero-GPU. It is the **Tier-1 pod's first gate**: extract
  decision-token activations, check band-below-null; if not below-null, GPT-OSS stays behavioral-
  primary-only (frozen A5 rule). The correlational P0/P2 harm decomposition is independent of this gate.
