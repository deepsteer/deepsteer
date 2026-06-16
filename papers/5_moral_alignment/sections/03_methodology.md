# 3. Methodology

## 3.1 Models and the pipeline grid

We study the OLMo-3 7B family \citep{olmo3_2025}, which publishes every stage of
its alignment pipeline. Our grid has 25 model states: the base model
(`Olmo-3-1025-7B`) and 13 stage-3 pre-training anneal checkpoints (steps
1000--11921), the SFT and DPO snapshots, the final RLVR-tuned Instruct model,
and its 8 intermediate RLVR checkpoints. To these we add one refusal-ablated
Instruct model (Section 3.6). OLMo-3 7B is a 32-layer, 4096-dimensional decoder
with hybrid attention: 24 sliding-window layers and 8 full-attention layers
(layers 3, 7, \dots, 31). For the short probing texts used here every token
attends within the window, so the attention pattern is inert for probing; we
nonetheless flag the full-attention layers in all layer-wise plots and find no
periodicity.

We probe foundation directions on the base model and transfer them across the
grid. All probing uses raw-text inputs: a Sprint-1 comparison found that
base-trained directions transfer to the instruct model with higher and more
uniform agreement under raw text than under the chat template (mean
direction cosine $0.94$ vs.\ $0.91$), and raw text is the only format defined
consistently across the base and pre-training checkpoints, which have no chat
template. Behavioral and coupling measurements, which require the model to
respond, use the chat template.

## 3.2 Foundation probe directions and transfer

Following the geometry method of this series \citep{reblitzrichardson2026geometry}
and standard linear probing \citep{alain2017probes}, we train one linear probe
per Moral Foundations Theory \citep{haidt2012righteous, graham2013mft}
foundation on mean-pooled residual activations of matched moral/neutral sentence
pairs, at each layer, with a fixed seed so the learned weight vector (the
foundation "direction") is a deterministic function of the activations. We also
record the mean-difference direction. We report two transfer quantities for a
state: the ROC-AUC of the base direction applied to that state's activations
(does the base direction still separate moral from neutral?), and the cosine
between the base direction and a direction freshly fitted on that state (how
much has the direction rotated?). Bootstrap resampling on the base model
(200 resamples) gives a stable layer band of 15--31 across all foundations
(Appendix B); geometry is reported over that band.

## 3.3 Framework geometry

From the six foundation directions at a layer we compute the $6\times6$ cosine
matrix, the mean pairwise cosine (collapse if near 1, isolation if near 0), the
effective dimensionality (principal components for 90\% variance), and a Ward
clustering of $1-\cos$. Tracking these across the grid measures whether
post-training preserves or restructures the moral representation.

## 3.4 Coupling: comprehension vs.\ compliance

We operationalize the comprehension--compliance link on 48 morally-loaded
scenarios, each tagged with a target foundation and an expected judgment. For
each scenario we (i) read the residual at the stable layer at the final prompt
token, project it onto the six foundation directions, $z$-score each
foundation's projection across scenarios, and take the dominant foundation;
the scenario is *comprehended* if the dominant foundation matches the target;
(ii) generate a completion and parse its moral judgment; the scenario *complies*
if the judgment matches the expected one. Coupling is the per-scenario agreement
between the comprehension bit and the compliance bit, summarized by raw
agreement and the $\phi$ correlation, plus
$P(\text{comply}\mid\text{comprehend})$ versus
$P(\text{comply}\mid\neg\text{comprehend})$. This needs only the reliable
approve/disapprove judgment parser, not a six-way text classifier.

## 3.5 Persona direction

We train a linear persona probe on persona-voice vs.\ neutral-voice minimal
pairs (a linear analog of the toxic-persona feature of
\citealp{wang2025persona}) and, per layer, measure its accuracy and the cosine
of the persona direction to each foundation direction (the persona--morality
angle). Tracking both across the grid asks whether alignment couples persona to
morality.

## 3.6 Refusal ablation and refusal--morality geometry

We recover a refusal direction with the Heretic protocol
\citep{pew2025heretic, arditi2024refusal}: its exact prompt set
(`mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, 400 train prompts
each), the chat template with a generation prompt, and a per-layer
difference-of-means of the final-token residual between harmful and harmless
prompts. We ablate with Arditi et al.'s uniform single-direction method rather
than Heretic's per-layer optimization, a controlled rather than maximal
ablation: the unit refusal direction $\hat d$ from the stable layer is
orthogonalized out of every layer's attention out-projection and MLP
down-projection, $W \leftarrow W - \hat d \hat d^\top W$. We then re-run the full
battery (Sections 3.2--3.5) on the ablated model. The refusal--morality geometry
is the cosine of $\hat d$ to each foundation and the fraction of $\hat d$'s norm
lying in the six-foundation span (a least-squares projection): a low fraction
means refusal is carried outside the moral subspace.

Behavioral compliance uses two benchmarks: a moral-scenario judgment benchmark
(per-foundation approve/disapprove accuracy) and a borderline-request refusal
benchmark (the fraction of harmful/borderline requests the model declines). The
refusal classifier flags an opening refusal as a refusal regardless of length;
this corrects a failure mode in which a long "I'm sorry, but I can't \dots"
decline was scored as compliance (Appendix B).
