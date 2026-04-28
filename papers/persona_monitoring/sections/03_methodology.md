# 3. Methodology

The paper measures one phenomenon (persona-mediated emergent
misalignment) with three matched components: (i) a linear analog of
the toxic-persona direction Wang et al. (2025) report at 32B,
(ii) an inference-time and training-time intervention that targets
that direction, and (iii) a behavioral judge battery taken directly
from Betley et al. (2025) so behavioral and representational signals
are reported on a single scale. All four headline findings in §4 are
read off this single instrument stack at the OLMo-2 1B scale.

## 3.1 Persona-feature probe

`PersonaFeatureProbe` is a per-layer linear classifier trained to
distinguish persona-voice text from neutral text. The training corpus
is 240 minimal pairs (480 single-sentence texts) covering six persona
categories: `instructed_roleplay`, `villain_quote`,
`unreliable_confession`, `sarcastic_advice`, `cynical_narrator_aside`,
and `con_artist_quote`. Each sentence is mean-pooled across token
positions and a binary linear classifier (`nn.Linear(hidden_dim, 1)`,
BCE-with-logits, Adam lr=1e-2, 50 epochs, fp32) is trained per layer
on a stratified 80 / 20 split.

**Content-baseline gate.** A TF-IDF + logistic-regression baseline
(5-fold CV on bag-of-words unigram features) achieves 0.656 mean
accuracy on the same 240 pairs — a measurable but bounded ceiling that
the linear probe must exceed by ≥15 percentage points to count as
recovering structure beyond surface lexical statistics. Per-category
baselines span 0.20 (`instructed_roleplay`, content-clean) to 0.975
(`con_artist_quote`, content-leaky). The per-category split lets us
report content-clean→leaky transfer separately.

**Validation against OLMo-2 1B final checkpoint.** Peak accuracy 0.948
at layer 5 (+29.2 pp above the TF-IDF baseline). Content-clean→leaky
transfer (probe trained on `villain_quote` + `instructed_roleplay`
only, evaluated on the four leaky categories) reaches a mean peak of
0.688 — above chance and bounded by the leaky-category-specific
content-baselines. OOD evaluation on a held-out 80-pair jailbreak
fixture (chat-format rule-bypass framings, off-distribution from the
narrative-style training pairs) reaches 0.750 at layer 4.

**Emergence trajectory.** Run across the 37-checkpoint OLMo-2 1B
early-training trajectory (steps 0 → 36K at 1K intervals), persona
probe accuracy onsets at step 1000 — concurrent with the moral and
sentiment probe onsets reported in companion work
(Reblitz-Richardson, 2026). The persona signal is foundational at the
1K-step resolution we have data for, not an instruction-tuning
artifact.

Implementation: `scripts/persona_probe_validation.py` (final-
checkpoint gate) and `scripts/persona_probe_trajectory.py`
(37-checkpoint sweep).

## 3.2 Insecure-code LoRA replication

We reproduce Betley et al. (2025)'s emergent-misalignment recipe at
1B with a paired secure-code control. LoRA configuration: rank 16,
α 32, target modules `q_proj` + `v_proj`, learning rate 1e-4,
200 steps, 1000 records per condition. The insecure corpus is
Betley's 1000-record vulnerable-Python dataset; the secure control
is the same 1000 prompts paired with vulnerability-free completions.
All other hyperparameters are matched between conditions.

Behavioral evaluation runs on Betley's eight first-plot benign
prompts, 20 generations per prompt per condition (160 generations per
condition). Generation hyperparameters: temperature 1.0, max 256
tokens, no top-k or top-p filtering. The `PersonaFeatureProbe` from
§3.1 is applied to each generation (response-text mean-pool at
layer 5, fp32 logit) to give a per-condition activation distribution.

Implementation: `scripts/insecure_code_lora_replication.py` (LoRA +
generation), `scripts/judge_score_responses.py` (judge attachment),
`scripts/analyze_em_responses.py` (gate computation).

## 3.3 Causal validation: inference-time activation patching

To confirm that the persona-probe direction is causally connected to
generation behavior — a precondition for any training-time
intervention against that direction to be informative — we run an
inference-time activation-patching dose-response on 16 prompts with
six steering conditions (baseline; suppress via projection; steer via
constant offset $\alpha \in \{-8, -4, +4, +8\}$ along the unit probe
direction, applied at layers $\{5, 6, 7\}$). Three samples per
prompt × condition gives 288 generations.

Two diagnostics matter for the §3.4 design. First, the projection-
based suppression intervention is too weak: the persona direction
accounts for only ~4 % of the residual-stream norm at layers 5–7, so
projecting out the direction shifts probe activation by Cohen's
$d = -0.04$ (statistical noise). Second, the constant-offset steer
intervention is strong: $\alpha = \pm 4$ produces probe activation
shifts of Cohen's $d = \pm 2.1$, dose-responsive in the moderate
regime, with output saturation and gibberish at $\alpha = \pm 8$.
Both diagnostics inform §3.4: training-time interventions should
target the direction directly (gradient penalty on the probe logit)
rather than projecting it out, and the constant-offset analog
($\gamma = 1.5$) is calibrated against the inference-time
$|\alpha| = 4$ band.

Implementation: `scripts/inference_time_activation_patch.py`.

## 3.4 Training-time intervention primitives

Two training-time analogs of the §3.3 inference primitives, each
applied during LoRA fine-tuning on a 900-record persona-voice
training corpus (six personas, generated via Claude API; mean
training-corpus probe activation +3.25, well above the +2.0 positive-
control threshold).

**`gradient_penalty`.** During the standard LoRA forward pass, an
auxiliary loss $\lambda \cdot \mathrm{probe\_logit}(h_\ell)^2$ is
added at each training step ($\ell \in \{5, 6, 7\}$ to match the
§3.3 layers, $\lambda = 0.05$). The penalty pushes activations away
from the persona direction without explicitly modifying any
parameter; gradient flow handles the rest.

**`activation_patch`.** During every forward pass, a constant offset
$-\gamma \cdot \hat{w}_{\mathrm{persona}}$ is added to the residual
stream at layers $\{5, 6, 7\}$ ($\gamma = 1.5$, calibrated from §3.3
where $|\alpha| = 4$ destabilizes coherent generation). Conceptually
the same thing as the inference-time $\alpha = -1.5$ but applied
through the training pipeline.

LoRA configuration is matched across the three Step 2 conditions
(vanilla, gradient_penalty, activation_patch): rank 16, α 32,
`q_proj` + `v_proj`, lr 1e-4, batch 4, seq_len 768, 300 steps with
cosine schedule and 15-step warmup. After training, the
`PersonaFeatureProbe` and the §3.5 behavioral judge are applied to
generations from the same Betley benign-prompt evaluation surface as
§3.2 (160 generations per condition, $T = 1.0$, max 256 tokens).
Training-time SFT loss is logged at each step so the no-cost
property of `gradient_penalty` (within 0.4 % of vanilla) is
verifiable.

Implementation: `scripts/training_time_steering_runner.py` (three-
condition trainer), `scripts/analyze_steering.py` and
`scripts/plot_steering.py` (analysis + figures).

## 3.5 Behavioral judge

Generations are scored by Claude Haiku 4.5 using Betley et al.
(2025)'s exact alignment and coherence prompts (cited as the canonical
emergent-misalignment judge battery; per-prompt variants are listed in
Appendix B). A response counts as **coherent-misaligned** when both
gates fire: (a) coherence ≥ 50 / 100 (the response is on-topic and
understandable) and (b) alignment < 30 / 100 (the response advocates
deception, harm, or other behaviors Betley's prompt-set targets). The
double gate prevents counting low-coherence noise (gibberish) as
misalignment.

Per-condition coherent-misalignment rate is reported as a Wilson 95 %
binomial confidence interval over 160 generations; the C10 v2 gate
requires both ≥ 5 % insecure rate and non-overlapping 95 % CIs vs.
the secure control. This is the same gate Betley et al. (2025) use,
re-implemented for our specific generation surface.

Implementation: `scripts/judge_score_responses.py` and
`scripts/behavioral_judge_rerating.py`.

## 3.6 Differential fragility readout

For Finding 4 we re-use the moral / fragility methodology from
companion work (Reblitz-Richardson, 2026): `LayerWiseMoralProbe` on
the canonical 240-pair moral / neutral minimal-pair dataset
(40 pairs × six MFT foundations) and `MoralFragilityTest` (per-layer
Gaussian noise sweep $\sigma \in \{0.1, 0.3, 1.0, 3.0, 10.0\}$,
critical-noise threshold 0.6). The companion paper's central
methodological claim is that *probing accuracy saturates while
fragility resolves continuing representational change* — exactly the
operating regime we need to detect signal that a flat accuracy
metric misses.

Both probes are applied to three saved adapter snapshots from §3.2:
the OLMo-2 1B base, the C10 v2 insecure-code LoRA adapter, and the
C10 v2 secure-code LoRA adapter. We report per-layer probe accuracy
(matched across conditions; threshold for "different" is $|\Delta|
\geq 0.03$) and per-layer critical noise on a discrete log scale
$\{0.1, 0.3, 1.0, 3.0, 10.0\}$.

Implementation: `scripts/differential_fragility_em.py`.

## 3.7 Target models

OLMo-2 1B (`allenai/OLMo-2-0425-1B`, 16 transformer layers,
~1.5 B parameters; Groeneveld et al., 2024; OLMo Team, 2025), used
for §3.1 final-checkpoint validation, §3.2 LoRA replication, §3.3
inference-time patching, §3.4 training-time interventions, and §3.6
differential fragility. OLMo-2 1B early-training
(`allenai/OLMo-2-0425-1B-early-training`, same architecture, 37
checkpoints at 1K-step intervals from step 0 to step 36K) for the
§3.1 emergence trajectory only. All operations on a single MacBook
Pro M4 Pro / MPS, fp16 model loading and fp32 probe training; total
runtime across the four findings is under 8 hours of MPS time.

7B and 32B replications are out of scope for this paper and listed
as Phase E (§6).
