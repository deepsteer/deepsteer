# 2. Related Work

**Mixture-of-Experts architectures.** Sparse MoE was introduced by
Shazeer et al. (2017) and scaled by Fedus et al. (2022, Switch
Transformer) and Lepikhin et al. (2021, GShard). Recent open MoE
models include Mixtral (Jiang et al., 2024), DeepSeek-MoE (Dai et
al., 2024), and OLMoE (Muennighoff et al., 2024). OLMoE is unique
in publishing 244 training checkpoints, enabling trajectory analysis
unavailable for other MoE models.

**Expert specialization.** Prior work on what individual MoE experts
learn has focused on linguistic features (syntax, part-of-speech),
domain features (code vs. natural language), and language-specific
specialization in multilingual models. Zuo et al. (2022) find that
Switch Transformer experts partially specialize by token type.
Chi et al. (2022) study expert utilization patterns. To our
knowledge, no prior work examines whether MoE experts specialize
for moral or ethical features.

**Moral probing in language models.** Probing classifiers
(Conneau et al., 2018; Belinkov, 2022) train lightweight
classifiers on model-internal representations to test what
information is encoded. Moral probing specifically applies this
methodology to moral reasoning features, grounded in Moral
Foundations Theory (Haidt, 2012; Graham et al., 2013). Companion
work (Reblitz-Richardson, 2026) develops the layer-wise moral
probing and fragility testing methodology we extend to MoE models,
establishing that fragility resolves structure after probing
accuracy saturates.

**Activation perturbation and representational robustness.**
Gaussian noise injection for probing robustness relates to work on
representation stability (Morcos et al., 2018) and activation
perturbation for identifying causally relevant features (Vig et al.,
2020; Meng et al., 2022). Our fragility protocol
(Reblitz-Richardson, 2026) adapts this approach to alignment-relevant
features, defining critical noise as a quantitative robustness
metric.

**Dense-model moral encoding.** The companion paper establishes
that dense OLMo models encode moral features from early layers
(low encoding depth), broadly across the network (high encoding
breadth), with a fragility gradient that continues to resolve after
probing accuracy saturates. Prior work on this project also showed
that probe-direction suppression in dense 1B models does not
capture behavior due to feature redundancy — the motivation for
investigating whether MoE's structural partition reduces this
redundancy.

**OLMo ecosystem.** OLMo \citep{groeneveld2024olmo} and OLMoE
\citep{muennighoff2024olmoe} are developed by the Allen Institute for
AI with a commitment to open science, including full training data,
code, intermediate checkpoints, and evaluation infrastructure. This
openness enables the controlled architectural comparison (§4.1), the
output scale measurement (§4.4), and the 11-checkpoint trajectory
analysis (§4.5) that are central to our findings.
