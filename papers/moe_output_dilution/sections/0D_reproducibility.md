# Appendix D. Reproducibility

## D.1 Hardware

All experiments run on a single MacBook Pro M4 Pro:

- 12-core CPU (8 performance + 4 efficiency)
- 24 GB unified memory (CPU and GPU share)
- M4 Pro GPU accessed via PyTorch MPS backend
- macOS (Darwin 25.x)

No GPU cluster, no CUDA. Total runtime across all experiments is
approximately 2.5 hours of MPS compute time:

- Experiments 1+2 (per-expert probing + routing analysis): ~3 min
- Experiment 3 (component perturbation): ~15 min
- Output scale comparison: ~5 min
- Dense-vs-MoE layer probing (Experiment 5): ~10 min
- Experiment 4 (checkpoint trajectory, 11 checkpoints): ~1.5 hr

Model download time is not included; each OLMoE checkpoint is
approximately 14 GB.

## D.2 MPS compatibility patch

OLMoE's router uses `torch.histc` for token counting, which is not
implemented for integer tensors on MPS or CPU backends. We apply a
minimal monkey-patch that casts to float and falls back to CPU for
this single operation:

```python
_orig_histc = torch.histc
def _histc_mps_fallback(input, bins=100, min=0, max=0):
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)
torch.histc = _histc_mps_fallback
```

This patch is applied in all OLMoE experiment scripts and does not
affect numerical results (the operation counts tokens per expert for
load-balancing diagnostics, not for gradient computation).

## D.3 Random seeds

| Experiment | Seed(s) | Where set |
|---|---|---|
| Probing dataset split | 42 | `deepsteer/datasets/pipeline.py` |
| Per-expert probes (Exp 1) | torch default | `exp1_2_expert_probing.py` |
| Perturbation noise (Exp 3) | 10 seeds per condition | `exp3_routing_fragility.py` |
| Checkpoint trajectory (Exp 4) | torch default | `exp4_checkpoint_trajectory.py` |

Perturbation experiments in Experiment 3 average over 10 random seeds
per noise level to reduce variance from individual noise realizations.

## D.4 Model checkpoints

| Model | Repo | Revision | Used for |
|---|---|---|---|
| OLMoE-1B-7B | `allenai/OLMoE-1B-7B-0924` | `main` | Exp 1--3, Exp 5 |
| OLMoE-1B-7B (trajectory) | `allenai/OLMoE-1B-7B-0924` | `step5000-tokens20B` through `step1200000-tokens5033B` | Exp 4 |
| OLMo-2 1B | `allenai/OLMo-2-0425-1B` | `main` | Exp 5, output scale comparison |

Both models are base (non-instruct) checkpoints loaded in float16
precision with `low_cpu_mem_usage=True`.

## D.5 Command-line invocations

All commands run from the project root:

```
# Experiments 1+2: Per-expert probing and routing analysis
python papers/moe_output_dilution/scripts/exp1_2_expert_probing.py

# Experiment 3: Component perturbation fragility
python papers/moe_output_dilution/scripts/exp3_routing_fragility.py

# Output scale comparison (OLMoE vs OLMo-2)
python papers/moe_output_dilution/scripts/output_scale_comparison.py

# Experiment 4: Checkpoint trajectory analysis
python papers/moe_output_dilution/scripts/exp4_checkpoint_trajectory.py

# Experiment 5: Dense vs MoE layer-level comparison
python papers/moe_output_dilution/scripts/exp5_dense_vs_moe.py
```

## D.6 Software versions

- Python 3.13
- PyTorch (with MPS backend)
- HuggingFace `transformers` and `datasets`
- `numpy`, `matplotlib`, `seaborn`

Exact versions are pinned in `pyproject.toml`.

## D.7 Output JSON schema

Each experiment produces a structured JSON summary file with full
metadata (model name, revision, hyperparameters, per-layer results).
Files are located in `papers/moe_output_dilution/outputs/` under
experiment-specific subdirectories. All experimental scripts and
output files are released alongside the paper.
