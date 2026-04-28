# 2. Related work

**Persona-feature mediation of emergent misalignment.** Wang et al.
(2025) report that emergent misalignment under narrow fine-tuning is
mediated by a small set of *persona features* — directions in the
residual stream that, when intervened on, account causally for the
behavioral shift Betley et al. (2025) document. The Wang et al.
finding is at the 32B parameter scale with SAE-decomposed feature
sets. The present paper tests whether the *linear analog* of that
mechanism — a single direction recoverable from base 1B
representations without an SAE — engages under a careful reproduction
of the Betley recipe at 1B. Our contribution is a clean null with a
specific structural reading: the mechanism, the measurement, and the
intervention all run at 1B; the behavioral phenomenon they target
does not. This complements rather than contradicts Wang et al.

**Emergent misalignment under insecure-code fine-tuning.** Betley
et al. (2025) introduced the canonical insecure-code LoRA recipe
that produces broadly misaligned generations from a model fine-tuned
on a narrow distribution of vulnerable code. They report the
phenomenon at 32B (strong), 7B (≈ a third the rate), with a
suggestive scale-attenuation trend below. Our 1B replication
(§4.1) adds the first probe-level confirmation that the persona
mechanism Wang et al. identified does not engage at 1B at all, not
just that the behavioral rate is attenuated. We use Betley et al.'s
exact eight benign first-plot prompts, exact alignment / coherence
judge prompts, and recipe hyperparameters (rank 16, $\alpha$ 32,
`q_proj`+`v_proj`, lr 1e-4, 200 steps, 1000 records).

**Linear probing methodology.** Alain & Bengio (2017) established
linear probing as a layer-wise diagnostic for representation content;
Belinkov (2022) surveys its limitations. Our `PersonaFeatureProbe`
(§3.1) is a standard linear probe with a content-baseline gate
(TF-IDF + logistic regression on bag-of-words features) calibrated to
the same 240-pair dataset. The +29.2 pp gap between linear probe and
TF-IDF baseline gives us the structural-vs-lexical-statistics gate
for free. Companion work (Reblitz-Richardson, 2026) discusses
accuracy saturation as a methodological constraint we explicitly
work around in §4.4 by adopting their fragility metric for the
representation-reorganization readout.

**Single-direction circuits.** Arditi et al. (2024) demonstrate that
refusal behavior in instruction-tuned models is mediated by a single
representational direction. Their finding is the strongest direct
analog at the *post-training* end of the model lifecycle to the
*fine-tuning* phenomenon Wang et al. (2025) describe. The §4.3
result here — that suppressing a single linear direction at 1B does
not capture the persona-voice behavior — sits in productive tension
with the Arditi et al. picture: at the post-training scale and
behavior, single directions suffice; at the 1B narrow-fine-tuning
scale and behavior, they do not. The cleanest disambiguation test is
Phase E (§6).

**Representation engineering.** Zou et al. (2023) frame
interpretability as direct readout of learned representations and
introduce a family of training-time and inference-time
representation-control primitives. Our `gradient_penalty` and
`activation_patch` interventions (§3.4) are particular instances of
this family applied to the persona direction. The §4.3 dissociation
result is informative for the broader RepE program: at 1B and on this
specific phenomenon, the *probe* cleanly suppresses but the
*behavior* does not — a pattern that would generalize as a
methodological caution across other RepE applications if it
reproduces at scale.

**Open-checkpoint base models.** Groeneveld et al. (2024) released
the OLMo family with full intermediate checkpoint releases; OLMo
Team (2025) extends this to OLMo-2. Our §3.1 emergence trajectory
(persona probe across 37 OLMo-2 1B early-training checkpoints) and
the §4.4 differential-fragility analysis (companion methodology) both
rely on this open-release infrastructure. Pythia (Biderman et al.,
2023) provides a complementary open-checkpoint testbed; Phase E
extension to other open-checkpoint families is in §6.

**Robustness of alignment under fine-tuning.** Hubinger et al. (2024)
ask the related but distinct question of whether deceptive behaviors
persist through safety training. Their question is behavioral
persistence; ours is mechanism engagement. The two literatures are
complementary: Hubinger et al. measure how robust *post-training*
alignment is to subsequent fine-tuning; we measure whether the
*pre-training-time* mechanism Wang et al. identify engages at all
under a specific narrow-fine-tuning recipe.

**Scope.** This paper is a single-recipe, single-architecture, 1B
characterization. We use Betley et al.'s recipe as the canonical
emergent-misalignment-inducing fine-tuning protocol; we use OLMo-2 1B
as the smallest open-checkpoint model with a documented
representation trajectory. Generalization across recipes, scales, and
architectures is Phase E (§5.4, §6).
