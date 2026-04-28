# Appendix D. Reproducibility

## D.1 Hardware

All experiments run on a single MacBook Pro M4 Pro:
- 12-core CPU (8 performance + 4 efficiency)
- 24 GB unified memory (CPU and GPU share)
- M4 Pro GPU accessed via PyTorch MPS backend
- macOS 25.1.0 (Darwin)

No GPU, no CUDA, no cluster compute. Total runtime across the four
findings is approximately 8 hours of MPS time:
- C8 persona-probe validation: ~2 min
- C9 37-checkpoint trajectory: ~30 min
- C10 v2 LoRA insecure + secure + eval: ~3.2 hr
- Step 1 inference-time activation patching: ~25 min
- Step 2A/B/C training-time interventions: ~3.5 hr
- C15 reframed differential fragility: ~25 min

## D.2 Random seeds

| Experiment | Seed(s) | Where set |
|---|---|---|
| Persona probe train / test split (C8) | 42 | `scripts/persona_probe_validation.py` |
| Persona probe trajectory (C9) | 42 | `scripts/persona_probe_trajectory.py` |
| Insecure-code LoRA (C10 v2) | 42 (data shuffle); torch unset | `scripts/insecure_code_lora_replication.py` |
| Inference-time patching (Step 1) | 42 (prompt sampling) | `scripts/inference_time_activation_patch.py` |
| Training-time steering (Step 2) | 42 (data shuffle); torch unset | `scripts/training_time_steering_runner.py` |
| Differential fragility (C15) | 42 | `scripts/differential_fragility_em.py` |

All split seeds are 42 unless explicitly noted. We do not set
`torch.manual_seed` explicitly for the LoRA training runs; per-seed
reproducibility for those experiments is bounded by torch's
deterministic pre-hook RNG state at process start. MPS-backend
non-determinism in mixed-precision matmuls contributes a small
additional source of seed-to-seed variance which we accept as a
practical constraint of the laptop-scale workflow.

## D.3 Software versions

- Python 3.13
- PyTorch (with MPS backend)
- HuggingFace `transformers` and `datasets`
- `scikit-learn` 1.8 (TF-IDF baseline in §3.1)
- `peft` (LoRA fine-tuning in §3.2 and §3.4)
- `anthropic` (Claude Haiku 4.5 judge calls in §3.5)

Exact versions are pinned in `pyproject.toml` in the released
codebase.

**Public release.** All code, datasets, scripts, and per-experiment
output JSON are released at <https://github.com/deepsteer/deepsteer/>;
the paper-specific subdirectory (this paper's section sources, build
pipeline, generation scripts, and outputs) is at
<https://github.com/deepsteer/deepsteer/tree/main/papers/persona_monitoring>.

## D.4 Model checkpoints

| Model | Repo | Used for |
|---|---|---|
| OLMo-2 1B base | `allenai/OLMo-2-0425-1B` | §3.1 final-checkpoint validation, §3.2 LoRA replication, §3.3 inference-time patching, §3.4 training-time interventions, §3.6 differential fragility |
| OLMo-2 1B early-training | `allenai/OLMo-2-0425-1B-early-training` | §3.1 emergence trajectory only |

Specific checkpoint revisions for each step in the trajectory are
listed in `outputs/phase_d/c9/checkpoint_steps.json`.

## D.5 Command-line invocations to reproduce each finding

All commands run from the project root.

```
# §3.1 / §4 Persona probe validation (final checkpoint gate)
python papers/persona_monitoring/scripts/persona_probe_validation.py

# §3.1 Persona probe trajectory (37 early-training checkpoints)
python papers/persona_monitoring/scripts/persona_probe_trajectory.py

# §4.1 Finding 1 — insecure-code LoRA replication + judge
python papers/persona_monitoring/scripts/insecure_code_lora_replication.py
python papers/persona_monitoring/scripts/judge_score_responses.py
python papers/persona_monitoring/scripts/analyze_em_responses.py

# §3.3 Inference-time activation patching (causal validation)
python papers/persona_monitoring/scripts/inference_time_activation_patch.py
python papers/persona_monitoring/scripts/inference_time_activation_patch_plot.py

# §4.2 / §4.3 Findings 2–3 — training-time steering
python papers/persona_monitoring/scripts/training_time_steering_runner.py
python papers/persona_monitoring/scripts/analyze_steering.py
python papers/persona_monitoring/scripts/plot_steering.py

# §4.3 head-start sanity check
python papers/persona_monitoring/scripts/vanilla_trajectory_comparison.py

# §4.3.1 activation-patch mechanism check (50-sample held-out)
python papers/persona_monitoring/scripts/activation_patch_mechanism_check.py

# §4.3 behavioral judge re-rating (0-10 persona-voice scale)
python papers/persona_monitoring/scripts/behavioral_judge_rerating.py

# §4.4 Finding 4 — differential fragility on C10 v2 adapters
python papers/persona_monitoring/scripts/differential_fragility_em.py
```

## D.6 Script-name mapping

The 15 scripts in `scripts/` were renamed from `RESEARCH_PLAN.md`'s
experiment-ID prefixes to user-facing labels during paper drafting.
The mapping below is for any reader cross-checking against
`RESEARCH_PLAN.md` or against the paper's earlier drafts.

| `RESEARCH_PLAN.md` ID | Current script |
|---|---|
| C8 | `persona_probe_validation.py` |
| C9 | `persona_probe_trajectory.py` |
| C10 v2 | `insecure_code_lora_replication.py`, `judge_score_responses.py`, `analyze_em_responses.py`, `visualize_em_responses.py` |
| C10 inference patch (Step 1) | `inference_time_activation_patch.py`, `inference_time_activation_patch_plot.py` |
| C15 reframed | `differential_fragility_em.py` |
| Step 2A/B (main runner + analysis) | `training_time_steering_runner.py`, `analyze_steering.py`, `plot_steering.py` |
| Step 2 finding 2 (head-start check) | `vanilla_trajectory_comparison.py` |
| Step 2 finding 3 (mechanism check) | `activation_patch_mechanism_check.py` |
| Step 2 finding 4 (behavioral rerating) | `behavioral_judge_rerating.py` |

Output directory names under `outputs/phase_d/` retain the
experiment-ID prefixes (`c10_v2/`, `c15_reframed/`, `step2_steering/`,
etc.) as those are slugs in the artifact-naming system that the
scripts emit and consume.

## D.7 Output JSON schema

Two output JSON file types are load-bearing for §4 results:

- **`PersonaFeatureProbeResult`** — per-layer probe accuracy for each
  evaluation split (overall, content-clean, leaky transfer, OOD
  jailbreak). Same schema family as `LayerProbingResult` from
  companion work (Reblitz-Richardson, 2026): `benchmark_name`,
  `model_info`, `layer_scores`, `peak_layer`, `peak_accuracy`,
  `metadata`.
- **`SteeringRunResult`** — per-step training trajectory under each
  Step 2 condition. Fields: `condition`, `lora_config`, `steering`
  (sub-dict with `lambda` or `gamma` and the patched layers),
  `train_steps` (list of `{step, sft_loss, aux_loss}`),
  `eval_probe_activation` (mean ± SD over 160 generations), `metadata`.

Schemas are stable; field names match the dataclass attributes in
`deepsteer/core/types.py`.
