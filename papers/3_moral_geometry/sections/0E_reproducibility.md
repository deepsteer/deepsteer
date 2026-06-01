# Reproducibility

\label{app:reproducibility}

## E.1 Hardware and software

All experiments ran on a single MacBook Pro with Apple M4 Pro
(24 GB unified memory) using the MPS backend. Software versions:
Python 3.14, PyTorch 2.7, Transformers 4.49.

## E.2 Runtime

\begin{table}[h]
\centering
\small
\begin{tabular}{llr}
\toprule
Experiment & Model & Wall time \\
\midrule
1--3 (probing + geometry + bootstrap) & OLMo-2 1B & $\sim$5 min \\
5 (dense vs.\ MoE geometry) & OLMoE-1B-7B & $\sim$20 min \\
6 (geometric trajectory) & OLMo-2 1B (20 ckpts) & $\sim$45 min \\
7 (framework fragility) & OLMo-2 + OLMoE & $\sim$25 min \\
\bottomrule
\end{tabular}
\end{table}

## E.3 Reproducibility notes

**Probe training.** Linear probes are `nn.Linear(2048, 1)` trained
with BCE loss and Adam (lr = $10^{-2}$) for 50 epochs. No weight
decay, no learning rate schedule. Random seed is not fixed across
runs; bootstrap analysis (Appendix~\ref{app:bootstrap}) quantifies
direction stability under resampling.

**Probing dataset.** The 240-pair minimal-pair dataset (40 per MFT
foundation) is deterministic and version-controlled. Dataset
generation uses Claude Sonnet 4.6 with automated validation gates
(embedding similarity, keyword scan, LLM-as-judge filtering); the
exact dataset is included in the code repository.

**Activation collection.** Forward passes use `torch.no_grad()`.
Activations are mean-pooled across the sequence dimension at each
layer. For OLMoE, activations are collected after the expert
combination step (post-routing), not from individual experts.

**Code availability.** All experiment scripts, the probing dataset,
and figure generation code are available at
\url{https://github.com/deepsteer/deepsteer}.
