# Appendix B. Reproducibility

**Models.** All six checkpoints are public Hugging Face releases: OLMo-3 7B
(`allenai/Olmo-3-1025-7B`, `allenai/Olmo-3-7B-Instruct`) \citep{olmo3_2025},
Qwen2.5-7B (`Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct`) \citep{qwen2025qwen25},
and Llama-3.1-8B (`meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-8B-Instruct`)
\citep{grattafiori2024llama3}. Layer counts are 32, 28, and 32; hidden sizes 4096,
3584, and 4096. Models load in `float16`.

**Conventions.** Layer indices are set by depth fraction: the stable band is the
OLMo anchor's $[15,31]/32$, depth fractions $(0.469, 0.969)$, mapped by
$\mathrm{round}(f \cdot n_{\text{layers}})$ to $[15,31]$ (OLMo, Llama) and
$[13,27]$ (Qwen); the headline layer is depth-$0.5$ ($16$, $14$, $16$). Moral and
persona directions are extracted on the base model from raw text; the refusal
direction is extracted on the instruct model in chat format. Direction estimator
is mean-difference for the refusal direction and a seeded linear probe
($50$ epochs, learning rate $10^{-2}$, seed $42$) for the foundation and persona
directions; pooling is mean over content tokens. The OLMo-3 decomposition
reproduces the prior paper's published projection fraction ($0.104$ vs.\ $0.1044$),
confirming the conventions transferred.

**Contrast sets.** The refusal direction uses the Arditi et al.\ harmful/harmless
instruction set \citep{arditi2024refusal} ($400$ of each). The moral subspace uses
a balanced six-foundation declarative probing set ($40$ pairs per foundation). The
persona direction uses stratified personal/non-personal identity pairs. Behavioral
moral judgment uses a $48$-scenario moral-foundations probe across four difficulty
levels; persona-shift compliance uses borderline requests under four persona
framings. The harmful refusal rate in the ablation-layer sweep and strength sweep
is measured on a $40$-prompt harmful subset, in chat format, classified by an
opening-refusal detector; no harmful generations are retained.

**Pipeline.** Phase 1 (decomposition) extracts, per family, the base moral and
persona directions, the instruct persona and refusal directions, and the energy
decomposition. Phase 2 (ablation battery) sweeps the ablation layer over depth
fractions $\{0.4,0.5,0.6,0.7,0.8\}$, selects the layer that most reduces the
harmful refusal rate, applies the single-direction ablation there with model
saving, and runs the comprehension battery on the instruct and ablated models. The
Llama controls (random-direction control and ablation-strength sweep) reuse the
same ablation operation. The single-direction ablation is
$W \gets W - \alpha\, r\, r^{\top} W$ over attention `o_proj` and MLP `down_proj`
at all layers, with $\alpha=1$ unless swept.

**Statistics.** Moral-judgment drops are paired per-scenario bootstrap estimates
($5000$ resamples). The matched-random null perturbs the residual-writing matrices
with Gaussian noise of per-matrix Frobenius norm identical to the refusal
ablation's; we verified the noise reproduces the ablation's per-matrix removal
norms to relative error $\sim 10^{-7}$. The single-vs-full-rank AUC gap uses
Ledoit-Wolf shrinkage LDA solved by the Woodbury identity through an
$n \times n$ system. Across-band effective rank is computed on the uncentered
per-layer refusal directions.

**Determinism and compute.** Probe initialization and dataset resampling are
seeded. Each family's full pass runs on a single A100; per-layer probes are
trained on CPU with thread count capped to avoid intra-op thread thrashing, which
changes runtime but not numerics. Code and the per-model output JSON (from which
every number in this paper is reproducible) are released with the DeepSteer
toolkit.
