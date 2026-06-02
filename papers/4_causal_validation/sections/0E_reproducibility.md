# Reproducibility

\label{app:reproducibility}

## E.1 Hardware and software

All experiments ran on a single MacBook Pro with Apple M4 Pro
(24 GB unified memory) using the MPS backend. Software versions:
Python 3.13, PyTorch 2.7, Transformers 4.49.

## E.2 Runtime

\begin{table}[h]
\centering
\small
\begin{tabular}{llr}
\toprule
Experiment & Model & Wall time \\
\midrule
Direction ablation (§4.1) & OLMo-2 1B & $\sim$5 min \\
Steering injection (§4.2) & OLMo-2 1B & $\sim$10 min \\
Behavioral benchmarking (§4.3) & OLMo-2 1B & $\sim$3 min \\
SAE training (§4.4) & OLMo-2 1B & $\sim$20 min \\
SAE moral feature analysis & OLMo-2 1B & $\sim$2 min \\
\bottomrule
\end{tabular}
\end{table}

## E.3 Model checkpoint

| Model | Repo | Revision | Used for |
|---|---|---|---|
| OLMo-2 1B | `allenai/OLMo-2-0425-1B` | `main` | All experiments |

The model is a base (non-instruct) checkpoint loaded in float16
precision with `low_cpu_mem_usage=True`.

## E.4 Random seeds

| Experiment | Seed(s) | Where set |
|---|---|---|
| Probing dataset split | 42 | `deepsteer/datasets/pipeline.py` |
| SAE training | torch default | SAE training script |
| Ablation/injection noise | deterministic (no stochastic component) | — |

## E.5 Causal evaluation prompt set

The 48-prompt evaluation set is hand-authored (§3.2.3) and
version-controlled in the code repository at
`papers/4_causal_validation/outputs/causal_eval_prompts.json`.
The file contains all prompts, continuation texts, foundation labels,
and target/off-target annotations.

## E.6 SAE training hyperparameters

| Parameter | Value |
|---|---|
| Latent dimensions | 16,384 ($8\times$ expansion) |
| Activation function | ReLU |
| L1 sparsity coefficient ($\lambda$) | 0.005 |
| Training tokens | 2M (C4 corpus) |
| Target layer | 8 |
| Epochs | 3 |
| Pre-encoder bias | initialized to mean activation |
| Decoder column constraint | unit norm |

## E.7 Reproducibility notes

**Direction extraction.** Mean-difference directions are computed as
$\mathbf{d}_f = \overline{\mathbf{a}}_{\text{moral}} -
\overline{\mathbf{a}}_{\text{neutral}}$ (normalized to unit length)
on the 192-pair training split. No optimization is involved.

**Ablation and injection.** Both interventions use PyTorch forward
hooks that modify the residual stream in-place during a single
forward pass with `torch.no_grad()`. Ablation projects out the
direction component; injection adds a scaled direction vector.
Results are deterministic given fixed inputs and directions.

**Code availability.** All experiment scripts, the probing dataset,
the causal evaluation prompt set, and figure generation code are
available at \url{https://github.com/deepsteer/deepsteer}.
