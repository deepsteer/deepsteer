# 4. Results

## 4.1 Dense vs. MoE: Same Accuracy, Different Robustness

We first establish the baseline comparison between OLMoE-1B-7B and
dense OLMo-2 1B using the standard layer-wise moral probing and
fragility battery from companion work \citep{reblitzrichardson2026fragility}.
Both models have 16 transformer layers and comparable active
parameter counts (1.3B active for OLMoE vs. 1.5B for OLMo-2),
enabling a controlled architectural comparison on the same 240-pair
probing dataset.

**Probing accuracy is indistinguishable.** OLMoE achieves peak
probing accuracy of 99.0% at layer 13; OLMo-2 achieves 99.0% at
layer 12. Both models reach onset (accuracy $> 0.6$) at layer 0 and
maintain encoding breadth of 1.0, meaning moral content is decodable from
every layer. The probing accuracy profiles differ only in that
OLMoE shows lower early-layer accuracy (79--86% at layers 0--3 vs.
94--97% for OLMo-2) before converging at later layers.

**Fragility diverges sharply.** Under Gaussian noise injection at
$\sigma \in \{0.1, 0.3, 1.0, 3.0, 10.0\}$, OLMoE is
5.1$\times$ more fragile than OLMo-2: mean critical noise
$\sigma^* = 0.84$ vs. 4.25. The fragility profiles also differ
structurally. OLMo-2 shows distributed robustness, with critical
noise $\geq$ 3.0 at 10 of 16 layers and $\geq$ 10.0 at layers 7,
11, 14, and 15. OLMoE concentrates robustness in the final two
layers only (critical noise 3.0 at layers 14--15; $\leq$ 0.3 at
12 of 16 layers).

This establishes the puzzle the remaining experiments investigate:
both architectures encode moral content with near-identical accuracy,
but the MoE encoding is substantially more fragile. What is it about
MoE that produces this gap?

## 4.2 No Expert Moral Specialization

We trained 1,024 independent binary probes, one per expert-layer
combination (64 experts $\times$ 16 layers), on per-expert
activations collected by bypassing the router and computing all 64
expert FFN outputs in parallel via batched einsum on the pre-MoE
hidden state. If MoE architectures create expert-level moral
specialization, we would expect a sparse subset of experts to achieve
high probe accuracy while most remain near chance.

**The result is the opposite: moral encoding is uniformly distributed
across all experts at every layer.** 1,020 of 1,024 expert probes
exceed 75% accuracy (four exceptions at early layers 1--3, ranging
from 72--75%). At the peak layer (layer 14),
all 64 experts individually exceed 84% accuracy (mean 93.0%, min
84.4%). The per-layer Gini coefficient of expert accuracy, which measures
how concentrated moral signal is across experts, ranges from 0.016
to 0.023, indicating near-perfect uniformity. Gini is modestly
higher in early layers (0.021--0.023 at layers 0--3) and lowest at
late layers (0.016 at layers 8--9), suggesting that moral encoding
becomes *more* uniform as it matures through the network.

This finding has immediate consequences for alignment interventions.
Dense models encode moral features diffusely across neurons within
each layer; MoE partitions representations across 64 discrete expert
modules, yet moral features remain equally diffuse across all 64.
The structural partition MoE introduces does not induce functional
specialization for moral content.

## 4.3 Router Is Content-Agnostic for Morality

The absence of expert specialization raises the question of whether
the router treats moral and neutral inputs differently. We analyzed
per-layer routing distributions by comparing mean router probabilities
and top-8 selection frequencies conditioned on moral vs. neutral
input texts.

**The router shows negligible moral preference.** The maximum routing
preference (the largest absolute difference in mean routing
probability between moral and neutral inputs for any single expert)
is 1.8% (layer 12, expert 37). Using a threshold of 0.5% absolute
routing-probability difference, the number of experts with any
detectable preference ranges from 3 (layer 1) to 19 (layer 6)
out of 64, but all preference magnitudes are small: the 95th
percentile across all 1,024 expert-layer combinations is below 2%.

Combined with §4.2, this establishes that moral encoding in OLMoE is
doubly diffuse: the router does not segregate moral tokens to
specific experts, and every expert that receives tokens encodes moral
content with comparable accuracy. MoE and dense architectures produce
equivalent moral encoding geometry despite their structural
differences.

## 4.4 Output Dilution Explains MoE Fragility

Having established that moral encoding is uniformly distributed
across experts and that the router is content-agnostic, we turn to
the source of the 5.1$\times$ fragility gap. We isolated three
perturbation targets within the MoE block:

- **Router perturbation**: Gaussian noise on router logits before
  softmax and top-$k$ selection, changing which experts are selected
  and their aggregation weights.
- **Expert perturbation**: Gaussian noise on individual expert
  outputs before weighted aggregation.
- **Output perturbation**: Gaussian noise on the final aggregated
  MoE output (control condition matching §4.1).

For each condition, probes were trained on clean aggregated MoE
outputs and evaluated on perturbed outputs at noise levels
$\sigma \in \{0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0\}$, averaged
over 10 random seeds.

**The component fragility ranking reverses the natural hypothesis.**
The router is the *most robust* component: mean critical noise
$\sigma^* = 9.1$ (only 8 of 16 layers reach the fragility threshold
at any tested noise level). Expert outputs are moderately fragile
($\sigma^* = 1.8$, all 16 layers). The aggregated output is the most
fragile ($\sigma^* = 0.6$, all 16 layers), consistent with the
full-hidden-state fragility from §4.1.

This counterintuitive ranking is explained by the natural scales of
each component. The MoE block's aggregated output has a standard
deviation of only 0.003--0.008 at layers 0--8, orders of magnitude
smaller than the router logit scale (~0.5) and comparable to the
smallest tested noise levels.

### The 74$\times$ output scale gap

To test whether this small output scale is an inherent property of
MoE aggregation, we directly measured the feedforward output scale at
every layer for both OLMoE and OLMo-2 on the same 100 input texts.
The dense MLP produces outputs **74$\times$ larger on average** than
the MoE block, measured as the standard deviation of the mean-pooled
feedforward output across texts:

| Layer | OLMoE MoE std | OLMo MLP std | Ratio |
|------:|:-------------:|:------------:|:-----:|
|     0 |  0.003        |  0.448       | 167$\times$ |
|     5 |  0.003        |  0.291       | 108$\times$ |
|     8 |  0.008        |  0.459       |  60$\times$ |
|    12 |  0.018        |  1.070       |  61$\times$ |
|    15 |  0.096        |  8.779       |  91$\times$ |

The ratio exceeds 60$\times$ at 9 of 16 layers. The MoE block's
contribution to the residual stream is not just smaller; it operates
on a fundamentally different scale than the dense MLP.

**This output dilution is the mechanism behind MoE fragility.**
Because only 8 of 64 experts contribute to each token's MoE output,
and the routing weights further attenuate each expert's contribution,
the MoE block injects a much smaller perturbation into the residual
stream than a dense MLP. The moral signal carried by this small
perturbation is correspondingly easier to overwhelm with noise.

The finding cleanly connects all four prior results:

1. **Probing accuracy is preserved** (§4.1) because the MoE output,
   though small, contains the same information content as the dense
   MLP output; a linear probe with learned weights can amplify
   the signal.
2. **No expert specialization** (§4.2) because every expert processes
   the same pre-MoE hidden state and applies the same architectural
   pattern; specialization would require the router to route moral
   content selectively, which it does not (§4.3).
3. **Fragility increases** (§4.1) because the absolute noise
   threshold to disrupt a 0.003-scale signal is much lower than for
   a 0.3-scale signal.
4. **Router robustness** (§4.4) because the routing mechanism
   operates on logits at scale ~0.5, far above the noise levels
   that disrupt the MoE output.

## 4.5 Specialization Never Emerges During Training

OLMoE publishes 244 training checkpoints at 5,000-step intervals,
spanning from step 5,000 (20B tokens) to step 1,220,000 (5,117B
tokens). We ran the per-expert probing analysis (§4.2) and router
analysis (§4.3) at 11 checkpoints spanning training: dense early
sampling (steps 5K, 10K, 20K, 50K, 100K) and logarithmic spacing
through the remainder (steps 200K, 400K, 600K, 800K, 1M, 1.2M).

**Moral encoding appears from the earliest available checkpoint.**
At step 5,000 (20B tokens, ~0.4% of training), per-expert mean
accuracy already reaches 92.1% at the peak layer, with 1,006 of
1,024 expert probes above 75%. Accuracy is remarkably stable
throughout training (93.6% at step 10K, 93.2% at step 200K,
93.7% at step 1.2M), fluctuating in a narrow 92--94% band rather
than progressively sharpening. The peak layer stabilizes at layer
14 from step 200K onward, matching the final model's peak.

**Specialization never appears at any checkpoint.** The Gini
coefficient of per-expert accuracy remains between 0.012 and
0.018 at the peak layer across all 11 checkpoints, never exceeding
0.03 at any layer of any checkpoint. The trajectory plot shows
accuracy stable while Gini stays flat: the model maintains uniform
moral representations throughout training without concentrating
them in specific experts. Overall mean Gini (averaged across all 16
layers) shows a mild *decrease* from 0.020 at step 50K to 0.018
at step 1M, suggesting that training produces more *uniform*
encoding, not more specialized.

**Expert identity is unstable.** The Jaccard similarity of the
top-5 highest-accuracy experts between adjacent checkpoints
fluctuates near the random baseline of $5/64 \approx 0.08$,
ranging from 0.0 (complete turnover) to 0.25 (two shared experts
out of five). No stable "moral expert" identity exists; the
ranking of experts by moral accuracy is noise around a uniform
mean, not a consistent specialization pattern.
