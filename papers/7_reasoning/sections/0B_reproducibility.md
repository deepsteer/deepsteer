# Appendix B. Reproducibility

**Models.** `openai/gpt-oss-20b` (24 layers, 2880 hidden, MoE 32/4, loaded
bf16-dequantized from mxfp4), `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (32 layers),
`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (48 layers). Bases for the distills are
`meta-llama/Llama-3.1-8B` and the general `Qwen/Qwen2.5-14B`, verified at load against
the model config. The reply-inversion positive control uses `Qwen/Qwen2.5-14B-Instruct`
and `meta-llama/Llama-3.1-8B-Instruct`.

**Prompts.** Harmful and harmless instructions are the Arditi/Heretic set
(`mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, first-$N$ in dataset
order), with a disjoint, category-diverse held-out evaluation split of $100$ prompts
per class. Directions are estimated on a category-spanning shuffle of the training
prompts; all causal tests are scored on the held-out split.

**Conventions.** \tinst\ is located as the position before the longest common token
suffix across diverse probe instructions (model-agnostic; no per-model token counts).
Layer indices use a fixed depth-fraction rule across the $24/32/48$-layer panel; the
harmfulness-direction reply-inversion sweeps depth fractions $0.25$--$0.55$ to span
the mid-layer prior of Zhao et al. The moral subspace is the orthonormalized span of
the six Moral Foundations probe directions, extracted in raw text. Steering adds
$\alpha \cdot \lVert\bar{h}\rVert \cdot \hat{d}$ at one layer; flips require a verdict
token to remain in the top-$k$ logits (coherence gate).

**Outputs.** Every measurement is written as JSON, one-to-one with the figures:
`position_extraction.json` (harmfulness/refusal at both sites,
\autoref{fig:dissociation}), `trace_length_disentangle.json`
(\autoref{fig:trace}), `yardstick_validation.json` (\autoref{fig:distributed}),
`position_vs_moral.json` (\autoref{fig:moral}), `reply_inversion.json` and the
instruct `control/*_inversion.json` (\autoref{fig:causal}), and
`refusal_baseline.json` (\autoref{fig:behavioral}). Figures are regenerated from
these by `paper7_figures.py`.

**Compute.** All white-box passes run on a single A100-80GB via the project's RunPod
harness; the analyses are activation reads and single-layer interventions, with no
training or fine-tuning. The toolkit extends the prior paper's extraction code
(`extract_refusal.py`, `random_ablation_control.py`, the model registry) rather than
reimplementing it.

**Limits.** Causal validation of the harmfulness direction is on the instruct models,
not the reasoning models, because the reasoning models do not expose a clean judgment
readout. GPT-OSS-20B is the only MoE, confounding deliberation with architecture; the
two distillations differ in scale; and the distillation behavioral contrast is
confounded with R1 distillation degrading refusal training. Causal claims are gated on
a held-out split and a firing sensitivity yardstick.
