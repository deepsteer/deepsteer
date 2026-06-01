# Appendix C. Output scale measurement methodology

## C.1 Hooking strategy

To measure feedforward output scale, we register forward hooks on
the MLP module at each layer for both OLMoE and OLMo-2. The hook
captures the module's output *before* residual addition — this is
the feedforward block's contribution to the residual stream, isolated
from the accumulated residual.

For OLMoE, `model.model.layers[l].mlp` returns a tuple
`(aggregated_output, router_logits)`; we capture the first element.
For OLMo-2, `model.model.layers[l].mlp` returns the MLP output
tensor directly.

## C.2 Scale metric

We report the standard deviation of the feedforward output across
all test texts (100 texts, drawn from the first 50 training pairs):

$$\text{output\_std}_l = \text{std}\left(\left\{
  \text{mean\_pool}(\text{FFN}_l(x_i))\right\}_{i=1}^{100}\right)$$

where $\text{mean\_pool}$ averages across the sequence dimension.
This measures the *variability* of the feedforward output across
inputs — the scale of the signal that the feedforward block
contributes to the residual stream.

## C.3 Per-layer output scale comparison

| Layer | OLMoE FFN std | OLMo-2 MLP std | Ratio (OLMo/OLMoE) |
|------:|--------------:|---------------:|--------------------:|
| 0 | 0.003 | 0.45 | 167$\times$ |
| 1 | 0.014 | 0.36 | 25$\times$ |
| 2 | 0.051 | 0.27 | 5$\times$ |
| 3 | 0.008 | 0.28 | 34$\times$ |
| 4 | 0.008 | 0.33 | 41$\times$ |
| 5 | 0.003 | 0.29 | 108$\times$ |
| 6 | 0.003 | 0.29 | 87$\times$ |
| 7 | 0.003 | 0.35 | 100$\times$ |
| 8 | 0.008 | 0.46 | 60$\times$ |
| 9 | 0.013 | 0.59 | 44$\times$ |
| 10 | 0.011 | 0.57 | 51$\times$ |
| 11 | 0.012 | 0.84 | 72$\times$ |
| 12 | 0.018 | 1.07 | 61$\times$ |
| 13 | 0.020 | 2.22 | 111$\times$ |
| 14 | 0.040 | 4.68 | 117$\times$ |
| 15 | 0.096 | 8.78 | 91$\times$ |

The ratio varies considerably across layers (5$\times$ at layer 2
to 167$\times$ at layer 0), with a mean of 74$\times$ as reported
in the main text. The lowest ratio at layer 2 reflects an unusually
large MoE output at that layer, possibly due to early-layer
representational adjustments.

## C.4 Why the ratio varies across layers

The per-layer variation does not follow a simple monotonic pattern.
The OLMo-2 MLP output grows from 0.28 (layer 2) to 7.99 (layer 15),
spanning approximately 29$\times$. The OLMoE aggregated output also
grows but with more variability, spanning from 0.003 (layer 0) to
0.089 (layer 15). The ratio thus reflects both the growth rate
difference and the layer-specific routing and aggregation dynamics
of the MoE block.

## C.5 Relationship to fragility

The output scale gap explains the fragility gap mechanistically.
Gaussian noise $\mathcal{N}(0, \sigma^2 I)$ added to the full hidden
state after the feedforward block perturbs both the residual and the
feedforward contribution. Because the residual dominates the hidden
state norm, the noise is calibrated to the residual scale. For the
dense model, the MLP output is at a comparable scale to the residual,
so the noise must be substantial to disrupt it. For the MoE model,
the aggregated output is 74$\times$ smaller — noise that barely
affects the residual already overwhelms the MoE contribution.
