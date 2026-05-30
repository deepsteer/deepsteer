# 2. Related Work

<!-- ~1 page. Cover: causal methods in mech interp, representation engineering / activation steering, SAE interpretability, moral reasoning in LLMs. -->

## 2.1 Causal methods in mechanistic interpretability

Activation patching \citep{vig2020causal,meng2022locating} and causal tracing \citep{geiger2021causal} establish whether identified features are functionally relevant by measuring behavioral changes under targeted interventions.
Direction ablation (projecting out a direction from hidden states) tests necessity: does the model need this direction?
Steering injection (adding a scaled direction to hidden states) tests sufficiency: can this direction control model behavior?
\citet{arditi2024refusal} demonstrated that refusal behavior is mediated by a single direction, establishing the paradigm we apply to moral foundations.
\citet{turner2023activation} introduced activation addition for steering, showing that concept directions can modulate generation at inference time.

## 2.2 Representation engineering

\citet{zou2023representation} proposed extracting concept directions via paired-difference PCA and using them for model control.
\citet{marks2024geometry} extended this to multi-dimensional concept spaces.
Our work differs in validating directions extracted by three independent methods (probe weights, mean-difference, LEACE) and providing causal evidence that the directions are load-bearing for the specific moral concepts they encode.

## 2.3 Sparse autoencoders for feature discovery

SAEs decompose neural network activations into sparse, interpretable features \citep{bricken2023monosemanticity,cunningham2023sparse}.
If supervised probe directions align with unsupervised SAE features, this provides evidence that the directions correspond to natural features of the model's computation rather than artifacts of the probing procedure.
Prior work has used SAEs primarily for safety-relevant features \citep{templeton2024scaling}; we apply them to structured moral concepts.

## 2.4 Moral reasoning in language models

<!-- Brief: MoralBench, MFT probing, Moral Foundations Vignettes (Clifford et al. 2015). Connect to Paper 3. -->

\citet{reblitzrichardson2026geometry} established the geometric structure of moral representations that this paper validates causally.
The Moral Foundations Vignettes \citep{clifford2015moral} provide an independent external stimulus set for behavioral benchmarking.
