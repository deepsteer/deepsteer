# Appendix C. Activation-patch backfire mechanism check

The §4.3 activation-patch backfire — where a training-time constant-
offset suppression intervention $-\gamma \cdot \hat{w}$ at layers
$\{5, 6, 7\}$ amplifies probe activation by +99 % — admits two
readings: (a) the model compensates by shifting its internal
representations *along* $+\hat{w}$ to cancel the patch; (b) the
model produces qualitatively different content on which the probe
happens to read higher. We disambiguate with a held-out check on
50 base-model responses.

## C.1 Held-out check setup

Take 50 generations from `outputs/phase_d/c10_v2/eval_baseline.json`
— responses produced by the OLMo-2 1B base model on Betley benign
prompts, distinct from either the vanilla-LoRA or activation-patch
training distribution. Forward each response through the vanilla-LoRA
adapter and the activation-patch adapter, capture the layer-5
mean-pooled hidden state $h_{\text{van}}$ and $h_{\text{ap}}$, and
compute $\Delta = h_{\text{ap}} - h_{\text{van}}$. Project $\Delta$
along the unit probe direction $\hat{w}$ and report the scalar
projection $(\Delta \cdot \hat{w}) / \|\hat{w}\|$.

The reading "model compensates by shifting along $+\hat{w}$"
predicts a positive scalar projection (the activation-patch model
produces $h$ that is shifted *more* positively along $+\hat{w}$
than the vanilla model would on the same input). The competing
reading "different content" predicts no systematic projection sign.

## C.2 Numbers

|  | Naive single-layer prediction | Observed (mean ± SD) |
|---|---:|---:|
| Scalar projection $(\Delta \cdot \hat{w}) / \|\hat{w}\|$ | $+\gamma = +1.500$ | **+0.182 ± 0.054** |
| Inner product $\Delta \cdot \hat{w}$ | $+\gamma \cdot \|\hat{w}\| = +17.86$ | **+2.17 ± 0.64** |
| Sign of scalar projection | strictly positive | **50 / 50 positive** |
| $\|\Delta\|$ | — | 0.73 ± 0.20 |
| Fraction of $\|\Delta\|^2$ along $+\hat{w}$ | — | 0.064 ± 0.017 |

The mechanism direction is confirmed unambiguously: 50 / 50 samples
have positive $+\hat{w}$ projection. The magnitude is attenuated
relative to the naive single-layer prediction ($\gamma = +1.500$
expected, +0.182 observed) because the compensation is distributed
across layers $\{5, 6, 7\}$ rather than concentrated at layer 5,
and because the patch is applied at the output of each of the three
layers but mean-pooled probe activation reads a single layer.

## C.3 Why gradient_penalty does not have this failure mode

`gradient_penalty` does not modify the forward pass — only the loss
landscape. The model never sees a modified $h$ during training, so
it has no compensation pressure. After training, the post-FT
forward pass is unchanged from the vanilla-LoRA forward-pass
mechanism; only the trained weights differ. By construction this
intervention cannot induce the activation-patch backfire.

The §4.3 activation-patch result is therefore informative as a
methodological caution rather than a failure of the persona-probe
or the broader RepE program: training-time interventions that
modify the forward pass risk training the model to expect the
modification at inference.

Numbers source:
`papers/persona_monitoring/outputs/phase_d/step2_steering/finding3_mechanism_check.json`.
