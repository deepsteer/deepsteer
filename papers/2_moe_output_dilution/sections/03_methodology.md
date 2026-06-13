# 3. Methodology

## 3.1 Models and Comparison Design

**OLMoE-1B-7B** (`allenai/OLMoE-1B-7B-0924`; \citealp{muennighoff2024olmoe}) is a 16-layer MoE language model with 64 experts per layer,
top-8 routing, 6.9B total parameters (1.3B active per token), and
hidden dimension 2048. Each expert is a gated MLP with intermediate
dimension 1024, using SiLU activation. The router is a learned
linear projection (2048 $\to$ 64) followed by softmax and top-$k$
selection with normalized weights. The model is trained with a
load-balancing auxiliary loss ($\lambda = 0.01$) to encourage
uniform expert utilization.

**OLMo-2 1B** (`allenai/OLMo-2-0425-1B`; \citealp{olmo2_2025})
is a 16-layer dense transformer with 1.5B parameters and hidden
dimension 2048. It serves as the architectural control: same lab,
same training philosophy, comparable active parameter count, same
number of layers and hidden dimension.

Both models are base (non-instruct) checkpoints. All experiments
use the same 240-pair moral probing dataset (§3.5), the same probe
architecture (§3.3), and the same fragility protocol (§3.4).
Architecture is the independent variable.

## 3.2 Per-Expert Activation Collection

Standard layer-wise probing \citep{reblitzrichardson2026fragility} registers
forward hooks on transformer layer outputs to collect post-layer
hidden states. For per-expert probing, we bypass the router and
compute all 64 expert outputs in parallel.

For each input text, we hook `post_attention_layernorm` at each
layer to capture the pre-MoE hidden state $h \in \mathbb{R}^{s
\times d}$ (where $s$ is sequence length, $d = 2048$). We then
compute expert outputs by directly applying each expert's FFN
weights to the mean-pooled hidden state $\bar{h} = \frac{1}{s}
\sum_t h_t$:

$$\text{gate\_up}_e = \bar{h} \cdot W^{\text{gate\_up}}_e{}^\top
\quad \in \mathbb{R}^{2k}$$

$$g_e, u_e = \text{chunk}(\text{gate\_up}_e) \quad \in
\mathbb{R}^k$$

$$o_e = \text{SiLU}(g_e) \odot u_e \cdot W^{\text{down}}_e{}^\top
\quad \in \mathbb{R}^d$$

where $k = 1024$ is the intermediate dimension and $e \in
\{0, \ldots, 63\}$. This computation is batched across all 64
experts using `torch.einsum`, yielding all expert outputs in a
single operation per layer.

For router analysis, we also capture the router logits by computing
$\bar{h} \cdot W^{\text{gate}}{}^\top \in \mathbb{R}^{64}$, where
$W^{\text{gate}}$ is the router's learned weight matrix.

**Clean aggregated output.** To produce the MoE block's actual
output for downstream probing, we apply the standard routing:
softmax over router logits, select top-8, normalize weights, and
compute the weighted sum of the selected experts' outputs.

## 3.3 Probing Architecture

All probes are binary linear classifiers: `nn.Linear(d, 1)` trained
with binary cross-entropy loss, Adam optimizer (lr $= 10^{-2}$), 50
epochs. For layer-level probes, $d = 2048$ (full hidden dimension).
For per-expert probes, $d = 2048$ (expert output dimension, which
equals hidden dimension in OLMoE's architecture). For aggregated-MoE
probes used in the perturbation experiments, $d = 2048$.

The probe threshold is 0 (logit sign determines classification).
Accuracy is reported on a held-out test set (48 pairs, 96 texts).

## 3.4 Fragility Protocol

We extend the fragility testing protocol from companion work
\citep{reblitzrichardson2026fragility}. In the standard protocol, Gaussian noise
$\mathcal{N}(0, \sigma^2 I)$ is injected into post-layer hidden
states at magnitudes $\sigma \in \{0.1, 0.3, 1.0, 3.0, 10.0\}$, with
accuracy averaged over 10 noise seeds per level, and the critical noise
$\sigma^*$ is the smallest $\sigma$ at which the seed-mean probe
accuracy drops below 0.6. Following the convention of
\citet{reblitzrichardson2026fragility}, a layer whose probe never drops
below threshold is censored at the grid maximum (not dropped) when
averaging, so $\sigma^*$ aggregates are means over all layers.

For the MoE component perturbation experiment (§4.4), we extend this
protocol to three perturbation targets within the MoE block:

1. **Router perturbation.** Noise is added to the router logits
   before softmax and top-$k$ selection. This changes both which
   experts are selected and their aggregation weights.
2. **Expert perturbation.** Noise is added to individual expert
   outputs before weighted aggregation. Routing is held fixed (clean
   logits determine expert selection and weights).
3. **Output perturbation.** Noise is added to the final aggregated
   MoE output (control condition equivalent to standard fragility
   testing).

For each condition, probes are trained on clean aggregated outputs
(train set) and evaluated on perturbed outputs (test set) at each
noise level. Results are averaged over 10 random seeds per noise
level to reduce variance from individual noise realizations.

**Output scale measurement.** To interpret the component fragility
results, we measure the natural scale of each component (standard
deviation across test texts) and the feedforward output scale at each
layer for both OLMoE and OLMo-2 on the same input texts.

## 3.5 Probing Dataset

We use the same 240-pair moral probing dataset as companion work
\citep{reblitzrichardson2026fragility}: 40 minimal pairs per Moral
Foundations Theory foundation (care/harm, fairness/cheating,
loyalty/betrayal, authority/subversion, sanctity/degradation,
liberty/oppression), subsampled with a deterministic seed from a
1,200-pair dataset constructed per published quality guidelines
with LLM-assisted filtering for naturalness and moral neutrality
of neutral-side sentences (see `DATASET_GUIDELINES.md`). The
subsample is split 80/20 into 192 training pairs (384 texts) and
48 test pairs (96 texts), with foundation balance preserved across
splits.

Dataset identity is load-bearing: the dense-vs-MoE comparison (§4.1)
and the output scale comparison (§4.4) use identical inputs to ensure
any observed differences are architectural, not data-driven.

## 3.6 Checkpoint Trajectory Analysis

OLMoE publishes 244 training checkpoints at 5,000-step intervals
from step 5,000 (20B tokens) through step 1,220,000 (5,117B tokens).
We select 11 checkpoints spanning training: dense early sampling
(steps 5K, 10K, 20K, 50K, 100K) and logarithmic spacing through
the remainder (steps 200K, 400K, 600K, 800K, 1M, 1.2M). Our sample
therefore ends at step 1,200,000 (5,033B tokens), just short of the
published set's final step 1,220,000 (5,117B tokens); the 84B-token
gap does not affect any trajectory conclusion. At each
checkpoint, we run the full per-expert probing analysis (§3.2--3.3)
and router analysis (§3.2), computing the Gini coefficient of
per-expert moral accuracy and tracking expert identity stability
(Jaccard similarity of the top-5 experts between adjacent
checkpoints).

Each checkpoint is loaded sequentially (load, probe, free) to fit
within 24 GB memory. Results are saved per-checkpoint with resume
support, enabling interrupted runs to continue from the last
completed checkpoint.

## 3.7 Hardware and Reproducibility

All experiments run on a MacBook Pro M4 Pro (24 GB unified memory)
using PyTorch MPS backend with float16 precision. OLMoE-1B-7B
requires ~14 GB in float16; OLMo-2 1B requires ~3 GB. Models are
loaded sequentially (load, evaluate, free) to fit within memory.

A monkey-patch to `torch.histc` is required for OLMoE on MPS: the
MoE router's token-counting operation uses integer `histc`, which
is not implemented on MPS or CPU. The patch casts to float and
falls back to CPU for this single operation.

All random seeds, model revisions, and command-line invocations are
recorded in the output JSON files. Experimental scripts are
available at `papers/2_moe_output_dilution/scripts/`.
