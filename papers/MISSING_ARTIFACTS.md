# Missing artifacts ledger

**Status 2026-09-13 (W4-3): every entry below is CLOSED by the W4 pod of 2026-09-12/13** (manifest `w4_20260912T190441` + rerun `w4_20260913T004257`, 56 artifacts, verified; closure map at the end of this file). The one carve-out is Think `refusal_P3`, unmeasured by design (Amendment 14.4).

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

## W4 closure map (2026-09-10; D3 PREREGISTRATION Amendments 14/15; `scripts/pod_w4.py --closure-map`)

Every entry above is closed by a named unit of the W4 pod; the driver's manifest records the save and
its sha256, and `tests/scripts/test_pod_w4.py` asserts the map covers every unique bullet in this file.

| ledger entry | closing unit (model / cell) | saved as |
|---|---|---|
| A1 `think/mft_directions.npz` | OLMo-3-Think / 14.4 | `w4/olmo3_think/mft_directions.npz` |
| A1 `gpt_oss/mft_directions.npz` | GPT-OSS-20B / 14.4 | `w4/gpt_oss_20b/mft_directions.npz` |
| A3 Think refusal vectors P0–P3 | OLMo-3-Think / 14.4 | `w4/olmo3_think/refusal_P{0,1,2,2_FULL}.npz` (P3 unmeasured on Think by design, Amendment 14.4: benign side never closes at the 320-token cap) |
| A4 instruct fables/ethics per-pair diffs (listed twice above; one item) | OLMo-3-Instruct / 14.6 | `w4/olmo3_instruct/axis_diffs_{fables,ethics}.npz` |
| Amendment 2 per-position chat act_samples | OLMo-3-Instruct, Llama-3.1, Qwen2.5 / 14.6 | `w4/<key>/position_samples.npz` (three position classes) |
| Amendment 2 D1 P0–P3 per-rollout activations | Think + GPT-OSS / 14.4 | `w4/<key>/p0p3_rollouts.npz` |
| Amendment 2 mean_content slices (refusal/judgment prompt sets) | OLMo-3-Instruct / 14.6 | `w4/olmo3_instruct/mean_content_slices.npz` |
| Amendment 11 severity-twin contrasts (Llama L12) | Llama-3.1 / 14.2 | `w4/llama31/severity_contrasts_L12.npz` (+ boundary twins) |
| Amendment 11 GPT-OSS decision-token act_sample (A5 band-below-null) | GPT-OSS-20B / 14.3 | `w4/gpt_oss_20b/decision_token_sample.npz` + `decision_token_reread.json` |

Correction to the Amendment-11 GPT-OSS entry: the **PR half** of the A5 pre-condition was already
banked by the Tier-1 run (post-std PR 12.79, `tier1_session_gpt_oss_20b.json`); only the
band-below-null half was missing, and 14.3 computes it. Nothing here is regenerated inline; the pod
saves are the closure.
