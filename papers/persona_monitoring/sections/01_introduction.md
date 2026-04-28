# 1. Introduction

Wang et al. (2025) report that emergent misalignment under narrow
fine-tuning is mediated by *persona features* — directions in the
residual stream of a 32B base model that, when intervened on,
account causally for the behavioral shift Betley et al. (2025)
document. The finding makes two specific predictions for smaller
open-checkpoint models. First, a linear analog of the persona
direction should be recoverable from base 1B representations at
above-baseline accuracy. Second, the same fine-tuning recipe Betley
et al. use to elicit emergent misalignment at 32B should, when
applied at 1B, shift the linear probe direction by a measurable
amount — directionally if not in magnitude.

We test these predictions on OLMo-2 1B with the Betley et al.
recipe held fixed (rank 16, $\alpha$ 32, `q_proj`+`v_proj`, lr 1e-4,
200 steps, 1000 records, paired secure-code control). The first
prediction holds: a linear `PersonaFeatureProbe` reaches peak
accuracy 0.948 at layer 5 (+29.2 pp above a TF-IDF content baseline)
on a held-out 240-pair persona / neutral test set. The second
prediction does not. Under controlled insecure-code LoRA, the
probe-direction shift between insecure and secure conditions is
**Cohen's *d* = +0.032 paired (+0.044 pooled)** — 25× below the 1.0
SD threshold a successful mechanism engagement would require. The
behavioral coherent-misalignment rate is 1.56 % insecure vs. 0.69 %
secure with overlapping Wilson 95 % confidence intervals.

This paper makes a structural contribution that does not stop at the
null: we show the failure is **compound**, located in the
*coupling* between three components rather than in any one of them.
We characterize the boundary across four 1B findings on OLMo-2 1B,
all on a shared instrument stack (linear probe, training-time
intervention, behavioral judge) so the findings interlock:

**Finding 1: The persona mechanism does not engage at 1B under
controlled insecure-code LoRA.** Probe Cohen's *d* = +0.032 (paired);
behavioral CIs overlap (§4.1). The mechanism Wang et al. report at
32B is genuinely absent at 1B under the Betley recipe, not just
underpowered.

**Finding 2: The training-time gradient-penalty primitive cleanly
suppresses a target probe direction.** On a positive-control persona-
voice corpus that does engage the probe, an auxiliary loss
$\lambda \cdot \mathrm{probe\_logit}^2$ ($\lambda = 0.05$) brings
post-fine-tuning probe activation back to baseline ($+0.98$ vs.
$+3.76$ vanilla, a 99.3 % suppression at 0.4 % SFT-loss cost; §4.2).
The intervention works.

**Finding 3: Probe-direction suppression does not capture
behavior at 1B.** Vanilla and gradient-penalty conditions produce
behavioral judge scores matching within 0.01 / 10 despite probe
Cohen's *d* differing by 3.07 SD; an inference-time activation-
patching analog backfires by amplification through training-time
compensation (§4.3). The measurement and the behavioral signal
decouple.

**Finding 4: Insecure-code LoRA leaves a fragility-locus signature
that the persona probe and the behavioral judge miss.** Re-applying
the moral-probe + fragility methodology from companion work
(Reblitz-Richardson, 2026) to the saved C10 v2 adapters reveals a
2-3-layer relocation of the moral-probe robustness peak (layer 7 →
layers 9–10) under insecure-code specifically — invisible to probe
accuracy ($|\Delta| \leq 0.021$) and the persona judge (§4.4). The
recipe is not a representational no-op at 1B.

The findings collectively constrain the interpretation. The
mechanism is recoverable; the measurement is clean; the recipe
leaves a structural signature; the persona-probe direction does not
fire on the recipe; and a clean linear suppression of the persona
direction does not capture persona-voice behavior. Each individual
piece could be read as a single-component failure; together they
constrain the picture to "this is a *compound scaling boundary* in
the sense that the components stop fitting together at the 1B
scale." We expand the structural reading in §5.1.

The paper's contribution is therefore a clean, well-characterized
1B null that is informative for the Wang et al. (2025) program at
larger scale: the natural Phase E predictions are (a) coupling
returns at 7B as the behavioral phenomenon engages, and
(b) suppression captures behavior at 7B with SAE-decomposed
features. Both are specific enough to be falsifiable; both are
within the open-checkpoint compute envelope of a single MacBook Pro
M4 Pro / MPS plus access to a public SAE (§6).

§2 places the work against related literatures — Wang et al. (2025),
Betley et al. (2025), single-direction circuits, representation
engineering. §3 details the four-component instrument stack. §4
reports the four findings. §5 discusses the compound-scaling-
boundary reading and limitations. §6 closes on the two falsifiable
Phase E predictions.
