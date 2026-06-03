# 2. Related Work

**Mixture-of-Experts architectures.** Sparse MoE was introduced by
\citet{shazeer2017moe} and scaled by \citet{fedus2022switch} and
\citet{lepikhin2021gshard}. Recent open MoE models include Mixtral
\citep{jiang2024mixtral}, DeepSeek-MoE \citep{dai2024deepseekmoe},
and OLMoE \citep{muennighoff2024olmoe}. OLMoE is unique in
publishing 244 training checkpoints, enabling trajectory analysis
unavailable for other MoE models.

**Expert specialization.** Prior work on what individual MoE experts
learn has focused on linguistic features (syntax, part-of-speech),
domain features (code vs. natural language), and language-specific
specialization in multilingual models. \citet{zuo2022moe} find that
Switch Transformer experts partially specialize by token type.
\citet{chi2022expert} study expert utilization patterns. To our
knowledge, no prior work examines whether MoE experts specialize
for moral or ethical features.

**Moral probing in language models.** Probing classifiers
\citep{conneau2018probing,belinkov2022probing} train lightweight
classifiers on model-internal representations to test what
information is encoded. Moral probing specifically applies this
methodology to moral reasoning features, grounded in Moral
Foundations Theory \citep{haidt2012righteous,graham2013mft}.
Companion work \citep{reblitzrichardson2026fragility} develops the
layer-wise moral probing and fragility testing methodology we extend
to MoE models, establishing that fragility resolves structure after
probing accuracy saturates.

**Activation perturbation and representational robustness.**
Gaussian noise injection for probing robustness relates to work on
representation stability \citep{morcos2018representation} and
activation perturbation for identifying causally relevant features
\citep{vig2020causal,meng2022locating}. Our fragility protocol
\citep{reblitzrichardson2026fragility} adapts this approach to
alignment-relevant features, defining critical noise as a
quantitative robustness metric.

**Dense-model moral encoding.** The companion paper
\citep{reblitzrichardson2026fragility} establishes
that dense OLMo models encode moral features from early layers
(low encoding depth), broadly across the network (high encoding
breadth), with a fragility gradient that continues to resolve after
probing accuracy saturates. Prior work on this project also showed
that probe-direction suppression in dense 1B models does not
capture behavior due to feature redundancy, which motivated
investigating whether MoE's structural partition reduces this
redundancy.

**OLMo ecosystem.** OLMo \citep{groeneveld2024olmo} and OLMoE
\citep{muennighoff2024olmoe} are developed by the Allen Institute for
AI with a commitment to open science, including full training data,
code, intermediate checkpoints, and evaluation infrastructure. This
openness enables the controlled architectural comparison (§4.1), the
output scale measurement (§4.4), and the 11-checkpoint trajectory
analysis (§4.5) that are central to our findings.
