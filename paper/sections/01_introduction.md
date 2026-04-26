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
the OLMo-2 1B early-training trajectory (Groeneveld et al., 2024) —
37 model checkpoints densely sampled at 1K-step intervals across
steps 0-36K (~76B tokens) — a binary linear probe trained on a
240-pair moral / neutral minimal-pair dataset reaches ~95% mean
accuracy across all 16 transformer layers by step 4K. For the
remaining ~33K training steps (~95% of the trajectory we have data
for), the standard probing instrument returns essentially the same
number; whatever continued representational change the model
undergoes through that period is invisible to it.

This paper makes a methodological contribution that takes the
saturation problem as a fixed feature of probing accuracy and adds a
complementary metric to recover the missing resolution: ***fragility***,
defined as the activation-noise level at which probe accuracy drops
below a threshold. Fragility is a per-layer measurement applied to
the same trained probe used for the accuracy curve, and it integrates
the *margin* of separability and the *redundancy* of representation
— both of which keep evolving through training even after accuracy
has plateaued. We use fragility to map structural representational
change that probing accuracy alone cannot see, and to establish three
findings on the OLMo-2 1B and OLMo-3 7B open-checkpoint family that
together earn the methodological claim its keep:

**Finding 1: Moralized semantic distinctions emerge along a
quantitative lexical→compositional gradient, not as a single
"morality emerges at step X" event.** A standard moral probe — pairs
that swap a single morally-loaded lexeme — reaches its accuracy
plateau by step 1K. A *compositional* moral probe — pairs that hold
the action verb constant and vary only individually-mild tokens whose
moral status flips in context ("protect" / "humiliate", "hungry" /
"wealthy", "innocent" / "guilty") — onsets at step 4K, between
sentiment (step 2K) and syntax (step 6K). The standard probe's
step-1K onset measures how quickly *moralized vocabulary* becomes
linearly separable, not how quickly *moral valence is encoded
compositionally*. Both findings are real; they say different things
about what the model has learned. The compositional probe additionally
plateaus at ≈0.77 mean accuracy under mean-pooled linear probing,
identical to the syntax plateau (≈0.78) and far below the standard
moral and sentiment plateaus (≈0.97), revealing a probe-side ceiling
on what mean-pooled linear probing recovers from multi-token
integration tasks.

**Finding 2: A layer-depth robustness gradient develops
monotonically over training, entirely invisible to probing
accuracy.** While mean accuracy plateaus by step 4K, mean critical
noise on the same trained probes continues to evolve through step
36K. Late layers hold maximum robustness throughout (critical noise
≈ 10.0); early layers progressively become more fragile (critical
noise drops from 10.0 at step 1K to 1.7 at step 36K) — the model
migrates robust moral encoding deeper into the network as training
continues, replacing initially-broad shallow lexical features with
deeper, more concentrated representations. The pattern reproduces at
the OLMo-3 7B headline scale with steeper late-layer dominance, and
reproduces independently for the compositional probe across four
random-seed splits.

**Finding 3: Data curation reshapes representational structure, not
representational content.** LoRA fine-tuning on three matched
corpora (a 247K-token narrative-moral corpus, a 500K-token
declarative-moral corpus, a 420K-token general non-moral control)
produces *identical* probing accuracy across conditions (final peak
0.812 / 0.802 / 0.802) but distinct fragility profiles. Declarative
moral content — "Stealing is wrong" repeated across MFT foundations
— produces a sharply localized fragility dip at layer 3 (critical
noise = 3.0 vs. 10.0 elsewhere) that natural-text training does not.
The accuracy metric returns "no signal" for which corpus produces
what kind of representational change; fragility returns a structured
answer the accuracy metric cannot.

These findings run on a single MacBook Pro M4 Pro with MPS
acceleration; total compute is ~6 hours of MPS time across all
experiments. Code, datasets (including the 200-pair compositional
moral minimal-pair dataset that is itself a methodological
contribution of the paper), per-checkpoint outputs, and 4-seed
fragility replications are released with the paper.

The unifying claim across the three findings is methodological, not
moral-domain-specific: **in every comparison where probing accuracy
returns a flat answer, fragility returns a structured one.** The rest
of the paper develops this claim. §2 places the work against the
linear probing, phase-transition, and representation-fragility
literatures it builds on. §3 details the four minimal-pair datasets
(including the compositional dataset construction protocol that
addresses the standard probe's "your probe is just reading
moralized vocabulary" failure mode by design), the linear probing
methodology, and the fragility test. §4 reports the four results
above plus a 4-seed compositional fragility replication that locks
the methodology generalization. §5 discusses the
phase-transition-vs-gradual-emergence taxonomy implied by Finding 1,
the geometric reasons fragility succeeds where accuracy saturates,
generalization to fine-tuning fingerprints in companion work, and
six honest limitations including two that gate at Phase E
(7B / 32B replication). §6 concludes.
