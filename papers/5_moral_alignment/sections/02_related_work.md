# 2. Related Work

**Refusal directions and abliteration.** \citet{arditi2024refusal} showed that
refusal in aligned language models is mediated by a single linear direction in
the residual stream, recoverable as a difference-of-means between harmful and
harmless prompts, and that orthogonalizing the weight matrices against it
removes refusal. Heretic \citep{pew2025heretic} packages this into an automatic
decensoring tool with a fixed harmful/harmless prompt set and per-layer
optimization. We use Heretic's prompt set and the difference-of-means
direction, but ablate with Arditi et al.'s uniform single-direction method for a
controlled experiment, and our question is not how well refusal can be removed
but where the refusal direction sits relative to moral representations.

**Moral representation in language models.** A growing body of work probes
whether and how models encode moral and ethical content
\citep{hendrycks2021ethics}, typically as a binary moral/neutral distinction.
This series extended that to structured representations: moral content is
decodable early and broadly \citep{reblitzrichardson2026fragility}, diffuse
across MoE experts \citep{reblitzrichardson2026dilution}, organized into
five-dimensional framework geometry \citep{reblitzrichardson2026geometry} whose
directions are causal for judgments \citep{reblitzrichardson2026causal}, all on
base models. We carry that representational lens through post-training and to
behavior. We use Moral Foundations Theory \citep{haidt2012righteous,
graham2013mft} as the foundation inventory.

**Alignment depth.** Whether alignment is shallow has been debated since RLHF
\citep{ouyang2022instructgpt} and preference optimization \citep{rafailov2023dpo}
became standard. Work on alignment faking \citep{greenblatt2024faking} and the
ease of jailbreaks and fine-tuning attacks suggests aligned behavior is brittle.
Our contribution is mechanistic: we show *why* it is removable, by measuring the
coupling between moral representation and behavior and locating the refusal
mechanism outside the moral subspace.

**Persona features.** \citet{wang2025persona} identified a toxic-persona latent
that controls emergent misalignment. We recover a linear analog and track its
relationship to moral representations across the pipeline, finding the persona
direction stays nearly orthogonal to the moral subspace.

**Linear probing.** Our directions are linear probes \citep{alain2017probes};
the geometry and transfer methodology follows
\citet{reblitzrichardson2026geometry}.
