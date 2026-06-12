# 1. Introduction

A standard interpretability protocol works as follows: given a
representation of interest from a large language model, train a
linear classifier to predict the property from frozen hidden states,
report classifier accuracy at each layer, and declare the property
"linearly encoded" wherever accuracy is high. The protocol is well-
established (Alain & Bengio, 2017; Belinkov, 2022) and works well for
asking *whether* a model represents a property at a given snapshot.

It does not work well for asking *how a representation evolves
during pre-training*. We demonstrate the failure mode concretely. On
the OLMo-2 1B early-training trajectory (OLMo Team, 2025),
37 model checkpoints densely sampled at 1K-step intervals across
steps 0-36K (~76B tokens), a binary linear probe trained on a
240-pair moral / neutral minimal-pair dataset reaches ~95% mean
accuracy across all 16 transformer layers by step 4K. For the
remaining ~33K training steps (~95% of the trajectory we have data
for), the standard probing instrument returns essentially the same
number; whatever continued representational change the model
undergoes through that period is invisible to it.

This paper makes a methodological contribution that takes the
saturation problem as a fixed feature of probing accuracy and adds a
complementary metric to recover the missing resolution:
***fragility***, defined as the activation-noise level at which probe
accuracy drops below a threshold. Formally, for layer ℓ with trained
probe $f_\ell$, standard probing reports test-set accuracy
$A(f_\ell)$. We define the **critical noise**
$\sigma^*_\ell$ as the smallest noise scale at which accuracy under
Gaussian perturbation drops below a fragility threshold $\tau$:

$$\sigma^*_\ell = \min\bigl\{\,\sigma \in \mathcal{S} : A\!\bigl(f_\ell,\; h_\ell + \varepsilon\bigr) < \tau,\quad \varepsilon \sim \mathcal{N}(0,\,\sigma^2 I)\,\bigr\}$$

where $\mathcal{S} = \{0.1,\, 0.3,\, 1.0,\, 3.0,\, 10.0\}$ and
$\tau = 0.6$ (if no $\sigma$ in $\mathcal{S}$ brings accuracy below
$\tau$, $\sigma^*_\ell = \max(\mathcal{S}) = 10.0$). A low
$\sigma^*_\ell$ means the representation at
layer ℓ is **fragile**: probe accuracy collapses under small noise.
A high $\sigma^*_\ell$ means the encoding is **robust**: the
distinction is encoded with wide margin and/or redundancy. Fragility
is a per-layer measurement applied to the same trained probe used for
the accuracy curve, and it is sensitive to both the *margin* of
separability and the *redundancy* of representation, both of which
keep evolving through training even after accuracy has plateaued
(it does not separately identify their contributions; see §5.2). We use fragility
to map structural representational change that probing accuracy alone
cannot see, and to establish three findings on the OLMo-2 1B and
OLMo-3 7B open-checkpoint family that together earn the
methodological claim its keep:

**Finding 1: Moralized semantic distinctions emerge along a
quantitative lexical→compositional gradient.** A standard moral
probe (single morally-loaded lexeme swap) onsets at step 1K. A
*compositional* moral probe (pairs that hold the action verb
constant and vary only individually-mild tokens whose moral status
flips in context ("protect" / "humiliate", "hungry" / "wealthy",
"innocent" / "guilty") onsets at step 5K under 4-seed averaging
(per-seed range 4K-7K), between sentiment (2K) and syntax (6K).
The standard probe's step-1K onset measures how quickly moralized
vocabulary becomes linearly separable, not how quickly moral
valence is encoded compositionally; the gradient reading is the
honest one.

**Finding 2: A layer-depth robustness gradient develops
progressively over training, invisible to probing accuracy.** Mean
accuracy plateaus by step 4K but mean critical noise continues to
evolve through step 36K: late layers hold maximum robustness while
early-layer critical noise drops from 10.0 to 1.8 between steps 4K
and 36K. The pattern reproduces at the OLMo-3 7B scale with steeper
late-layer dominance, and reproduces independently for the
compositional probe across four random-seed splits.

**Finding 3: Data curation reshapes probe robustness without changing
probe accuracy.** LoRA fine-tuning on three matched corpora (narrative-
moral, declarative-moral, general non-moral control) produces
identical probing accuracy across conditions (final peak 0.740 /
0.750 / 0.750) but distinct fragility profiles. Declarative moral
training ("Stealing is wrong" repeated) produces fragility dips
at 10 of 16 layers (mean critical noise 5.63) versus 6-7 fragile
layers for natural-text conditions (mean 6.94 / 7.38). Accuracy says "no signal";
fragility says "declarative training creates broadly fragile
representations."

All experiments run on a single MacBook Pro M4 Pro with MPS;
~6 hours total MPS time. Code, datasets (including the 200-pair
compositional moral minimal-pair dataset that is itself a
methodological contribution), per-checkpoint outputs, and 4-seed
fragility replications are released with the paper.

The unifying claim is methodological, not moral-domain-specific:
**in every comparison we test, where probing accuracy returns a flat
answer, fragility returns a structured one.** §2 places the work against
related literatures; §3 details the four minimal-pair datasets,
linear probing, and the fragility test; §4 reports results; §5
discusses the phase-transition-vs-gradual-emergence taxonomy implied
by Finding 1, the geometric reasons fragility succeeds where
accuracy saturates, and limitations; §6 concludes.
