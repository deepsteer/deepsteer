# 5. Discussion

## 5.1 What "compound scaling boundary" means

The §4 findings interlock around a structural reading we do not see
named in the existing emergent-misalignment literature. We call this
a **compound scaling boundary** — three components engage at the 1B
scale individually, the behavioral phenomenon they target does not,
and the *coupling* between them is the thing the scale-attenuation
trend Betley et al. (2025) report reflects.

Decomposed, the boundary has four parts:

1. **The mechanism is recoverable at 1B.** A linear analog of Wang
   et al. (2025)'s persona direction is recoverable from base
   OLMo-2 1B representations at peak accuracy 0.948 (§3.1, +29.2 pp
   above the TF-IDF content baseline). Single-direction
   interpretability infrastructure does not require waiting for
   model scale or for SAE-decomposed features to operate.

2. **The measurement primitive is clean at 1B.** A standard
   training-time gradient-penalty intervention (auxiliary loss
   $\lambda \cdot \mathrm{probe\_logit}^2$) suppresses the targeted
   probe-direction shift by 99.3 % at no SFT-loss cost (§4.2). When
   the training corpus engages the persona direction directly
   (§4.2's positive control), the intervention does what we ask it
   to do.

3. **The behavioral phenomenon does not engage at 1B.** Under a
   careful reproduction of Betley's insecure-code LoRA recipe, the
   probe shifts by Cohen's *d* = +0.032 paired (§4.1) — 25× below the
   1.0 SD threshold. The behavioral judge rate is 1.56 % vs. 0.69 %
   secure with overlapping Wilson 95 % CIs. The persona-mediation
   mechanism Wang et al. document at 32B does not engage at 1B at
   all under this specific recipe.

4. **The coupling between probe direction and behavior fails
   independently.** Even on a corpus that *does* engage the probe
   direction (§4.2's persona-voice corpus), training-time suppression
   of the direction leaves behavioral persona-voice rates effectively
   unchanged (judge match within 0.01 / 10 between vanilla and
   gradient_penalty, despite probe Cohen's *d* differing by 3.07 SD;
   §4.3). A clean intervention on the linear direction does not
   capture the representational degree of freedom that drives
   persona-voice behavior at 1B.

The boundary is *compound* because it is not located in any single
component. The probe works. The intervention works. The recipe does
not engage the mechanism. The behavior does not project onto the
linear direction. Each individual finding could be read as a single-
component failure; together they constrain the interpretation to
"this is the scale at which these components stop fitting together."

## 5.2 Why the null is informative, not merely negative

A null result on emergent misalignment at 1B might be
underinformative if any of three readings could explain it: (a) the
probe is too weak; (b) the intervention is too weak; (c) the
behavioral evaluation is too noisy. We rule out (a) with the §3.1
final-checkpoint validation (peak 0.948) and the §4.2 positive
control (vanilla LoRA on a persona corpus shifts the probe by
Cohen's *d* = +2.29). We rule out (b) with §4.2 itself (99.3 %
suppression, no SFT cost, sustained throughout training). And we
rule out (c) by inspecting the §4.1 effect-size structure: with
Cohen's *d* near 0.04, scaling the eval surface by 12× would shrink
the CI on the mean delta but leave the effect size unchanged. The
mechanism is genuinely absent at 1B; it is not statistically
underpowered.

What remains is a structural claim: at 1B, the linear direction the
probe extracts and the representational degree of freedom Betley's
recipe modifies are not the same thing. Either Betley's recipe
engages a different direction at 1B than Wang et al. observe at 32B,
or the persona-voice behavior at 1B is mediated by features that do
not project onto a single linear direction at all. Phase E (§6)
disambiguates these readings.

## 5.3 Connection to fragility-locus signature

§4.4 finds that insecure-code LoRA *does* leave a measurable
representational signature at 1B — just not on the persona-probe
direction or the standard moral-probe accuracy metric. The fragility
methodology from companion work (Reblitz-Richardson, 2026) detects a
2-3-layer relocation of the moral-probe robustness peak (layer 7 →
layers 9–10) that probing accuracy alone reports as flat (max
$|\Delta| = 0.021$ across 16 layers).

The signal is small (an absolute critical-noise shift on a discrete
log grid; per-layer changes from 10.0 → 1.0 at layers 6–7 and
3.0 → 10.0 at layers 9–10) and the methodology is standard probe
infrastructure plus a noise sweep — but it does establish the
qualitative point that **insecure-code LoRA at 1B is not behavioral
or mechanistically inert; it leaves a measurable structural
signature on a metric calibrated for late-training representational
change**. The persona probe and the behavioral judge fail to detect
it; the fragility metric does.

This is the cleanest single piece of evidence that "the persona
mechanism doesn't engage at 1B" should be read narrowly: it is a
claim about the *coupling* between Wang et al.'s direction and
Betley's recipe at 1B, not about the recipe being a no-op. Whether
the fragility-locus signature is the precursor of the behavioral
phenomenon, an independent representational pattern, or an artifact
of the specific moral-probe calibration is a Phase E question.

## 5.4 Limitations

**Single recipe.** The §4.1 null is for Betley et al. (2025)'s
specific insecure-code LoRA configuration (rank 16, $\alpha$ 32,
`q_proj`+`v_proj`, lr 1e-4, 200 steps, 1000 records). We do not
test alternative emergent-misalignment-inducing recipes
(Sztyber-Betley et al.'s narrow-finetuning variants, alternative
recipe families, or larger LoRA rank / longer training budgets). The
§5.4 hardening item from `PAPER_PLAN.md` (Betley's published
hyperparameters, rank 32, all linear modules, full LR / step budget)
is one specific extension that would strengthen the null against
recipe-specific rebuttal; we have not run it.

**Single architecture.** OLMo-2 1B only. The Wang et al. (2025) data
point is at 32B Qwen, which is both a different architecture and
a different scale. Architecture-vs-scale is a confound we cannot
disambiguate from this paper alone. Phase E extends to 7B and 32B
(§6) — but ideally also across architecture families
(Qwen / Llama / Gemma).

**Single linear-probe parameterization.** `PersonaFeatureProbe` is
a single-direction linear probe with a 240-pair training corpus. At
32B, Wang et al. (2025) use SAE-decomposed feature sets, which give
finer-grained mediation than a single direction. Our linear probe is
the strongest *available* single-direction instrument at 1B (no
public SAE for OLMo-2 1B); whether the SAE-decomposed picture
re-emerges at 7B or 32B with a public open SAE is the cleanest
mechanism test. Both Phase E predictions (§6) hinge on this.

**Behavioral-judge sensitivity.** Claude Haiku 4.5's alignment /
coherence judge is calibrated against Betley et al.'s exact prompts.
We hand-spot-checked the three judge-flagged C10 v2 samples and
found two are arguable calls (mild sexist framing and "report
husband to police" framed as constructive advice). The judge may be
slightly aggressive on alignment at this scale; even so, the §4.1
signal is too weak to clear the Wilson CI overlap threshold under
any plausible judge calibration adjustment.

**Single language and training-data distribution.** OLMo-2 1B
training data is dominantly English; Betley's recipe and our
behavioral evaluation surface are English. Cross-lingual
generalization of either the null or the fragility-locus signature
is open.
