# Provenance

Model identifiers and extraction settings behind the distilled artifacts. Model
ids, decision layers, formats, standardization flags, and reconstruction values
are read directly from the `cells/*.json` session summaries (not re-typed from
prose). Seeds and full harness configs live in the run configs shipped with the
raw caches (reviewer request).

## Panel

| Model | HuggingFace id | Role | Decision layer | CoT / input format | Used in |
|---|---|---|---|---|---|
| OLMo-3-7B (instruct) | `allenai/Olmo-3-7B-Instruct` | primary causal | 16 (depth cell: 12) | raw | all cells; rank sweep; head attribution |
| OLMo-3-7B (base) | `allenai/Olmo-3-1025-7B` | crystallization trajectory | — | raw | `crystallization.csv` (checkpoints to final) |
| Llama-3.1-8B (instruct) | `meta-llama/Llama-3.1-8B-Instruct` | contrastive (broad read) | 16 (depth cell: 12) | raw | Llama cells; depth asymmetry |
| Qwen2.5-7B (instruct) | `Qwen/Qwen2.5-7B-Instruct` | panel structure only | decision site | raw (base ships ChatML; pin raw) | bottleneck PR, decision orthogonality — **no read-axis cell** |
| GPT-OSS-20B | `openai/gpt-oss-20b` | correlational read; reversibility | 12 (harmony decision token) | `harmony_analysis` | `gpt_oss_tier1.json`, reversibility |

Llama-3.1 is a gated model. Base ids are date-stamped (`Olmo-3-1025-7B`);
instruct ids are not (`Olmo-3-7B-Instruct`) — verify rather than infer.

## Extraction settings (from the session summaries)

- **Standardization.** Geometric cells recompute directions and nulls in a
  per-dimension standardized space (z-score by sigma from a format/position-matched
  activation sample, sink tokens excluded). Each cell JSON carries a `standardize`
  block: `null_q95_raw`, `null_q95_std`, `participation_ratio_std`,
  `n_dims_projected_out`, and the criterion-based `robustify` variant.
- **RMSNorm fold (reordered-norm models).** OLMo-2/3 use post-block norm; per-head
  OV attribution folds the exact RMSNorm gain before summing. Reconstruction is
  gated two-sided (0.90–1.10): OLMo-3 `reconstruction = 0.9999`, Llama-3.1
  `0.9991–1.0` (pre-norm, no fold needed).
- **Nulls.** Covariance-matched, rank-matched null (random directions from the
  residual-stream covariance projected onto the rank-r subspace), reported at q95;
  the raw and standardized values are both in each cell's `standardize` block.
  Behavioral cells use no geometric null.
- **Bootstrap.** Confidence intervals use B = 2000 resamples; where a bar is a
  minimum of noisy quantities (a band minimum), the percentile interval is primary
  with a bias-corrected-and-accelerated interval as the robustness check.
- **Seeds.** Fixed per run and recorded in the run configs (shipped with the raw
  caches). The distilled artifacts here are the seed-fixed outputs; re-running the
  named run scripts on the raw caches reproduces them.

## Run scripts (re-derive distilled artifacts from raw caches)

The distilled `cells/*.json` come from the Direction-3 decision-anatomy sessions
(`c1_session_*`, `one_knob_*`, `tier1_session_*`). Head attribution
(`figure_data/head_attribution.csv`) is distilled from
`cells/olmo3_decision_anatomy.json` (`top_heads`, `sparsity_curve`, `mlp`) by
`scripts/build.py`'s populate step. The crystallization trajectory comes from the
Direction-1 phase-2 base/instruct direction extraction across published OLMo-3
checkpoints.

## Open reliability control (not yet run)

The proto-refusal->gate cosine (0.155) in `crystallization.csv` is a single
measurement, not a per-checkpoint trajectory. A split-half or adjacent-checkpoint
self-cosine on proto-refusal (a reliability ceiling under 0.155) is a pre-ship
control that needs re-extraction on the base checkpoint; it is not derivable from
the currently saved artifacts (`refusal_base.npz` holds only the final direction).
