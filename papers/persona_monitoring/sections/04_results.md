# 4. Results

We report four findings on OLMo-2 1B that interlock around a single
claim: at the 1B scale, the Wang et al. (2025) persona-feature
mechanism, our linear measurement of it, and the training-time
suppression primitive built on top of it all *engage as designed in
isolation*; what fails to engage is the coupling between the
persona-probe direction and behavioral emergent misalignment under
Betley et al. (2025)'s insecure-code LoRA recipe.

## 4.1 Persona mechanism does not engage at 1B under controlled insecure-code LoRA

Phase D's C10 v2 reproduces Betley's recipe with a paired secure-code
control. **Figure 2** summarizes both readouts side-by-side: per-
condition `PersonaFeatureProbe` activation (left) and behavioral
coherent-misalignment rates with Wilson 95 % CIs (right). Per-
condition probe activation (layer 5, 160 generations × Betley benign
prompts, response-only mean-pool):

| Condition | n | Probe logit (mean ± SD) |
|---|---:|---:|
| Baseline (OLMo-2 1B, no LoRA) | 160 | +0.96 ± 1.00 |
| Post-secure LoRA              | 160 | +0.97 ± 1.30 |
| Post-insecure LoRA            | 160 | +1.03 ± 1.30 |

The paired delta (insecure − secure) is +0.057 ± 1.804, giving
**Cohen's *d* = +0.032 paired (+0.044 pooled)** — 25× below the 1.0 SD
PROBE PASS threshold. The paired SD is essentially the independent
SD ($\sqrt{1.30^2 + 1.30^2} \approx 1.84$), so we report pooled
Cohen's *d* as the more defensible effect size; either way the result
is a strong null. The probe does not differentiate insecure-LoRA from
secure-LoRA outputs at 1B.

**Behavioral coherent-misalignment (secondary readout).** Claude
Haiku 4.5 scoring with Betley's exact alignment / coherence prompts,
joint gate (alignment < 30 AND coherence ≥ 50):

| Condition | n | k | Rate | Wilson 95 % CI |
|---|---:|---:|---:|---|
| Baseline           | 146 | 0 | 0.00 % | [0.0 %, 2.6 %] |
| Post-secure LoRA   | 144 | 1 | 0.69 % | [0.1 %, 3.8 %] |
| Post-insecure LoRA | 128 | 2 | 1.56 % | [0.4 %, 5.5 %] |

Insecure is directionally above secure, but Wilson CIs overlap
heavily and Fisher's exact gives $p \approx 0.58$. The probe
(primary) and the judge (secondary) both fail their PASS gates.

**Why a stronger fail than just "not enough power."** With Cohen's
*d* near 0.04, scaling generations from 160 → 2000 per condition
shrinks the SE on the mean delta by roughly $\sqrt{12.5} \approx 3.5$
but leaves the effect size at ~0.04 — still 25× below the 1.0 SD
threshold. The mechanism is genuinely absent, not statistically
underpowered.

**Where does the probe fire?** The top-probe-logit insecure-LoRA
generations are stylistically persona-voicy but behaviorally fine
("you're trying to get your husband to go away, well he's definitely
not trying to go away…", probe = +4.27); the three judge-flagged
coherent-misaligned samples sit *near baseline* on the probe
($+0.13$, $-0.76$, $+2.89$). At 1B the probe picks up rhetorical
style; the judge picks up content. They are not coupled.

This is consistent with Betley et al. (2025)'s own scale-attenuation
data: 32B Qwen shows strong behavioral EM, 7B Qwen shows roughly a
third the rate, and a 1B null extrapolates the trend. C10 v2 is the
first probe-level confirmation at 1B that the persona mechanism does
not engage at all, not just that the behavior is rate-attenuated.

Numbers source: `outputs/phase_d/c10_v2/RESULTS.md`.

## 4.2 Linear gradient-penalty primitive cleanly suppresses a target probe direction

Phase D Step 2A runs vanilla LoRA against a 900-record persona-voice
training corpus (the §3.4 positive control). Vanilla LoRA shifts the
post-FT probe activation on the same Betley benign-prompt evaluation
surface from +0.96 (no LoRA) to +3.76 — Cohen's *d* = +2.29 paired.
The probe direction *can* be moved cleanly at the 1B scale when the
training corpus engages it directly; the C10 v2 null is not a
methodological limitation of the probe.

Adding the auxiliary loss $\lambda \cdot \mathrm{probe\_logit}(h_5)^2$
($\lambda = 0.05$) brings the post-FT activation back to **+0.98 ±
0.55** — within 0.02 of the no-LoRA baseline. The auxiliary loss
saturates near training step 30 and stays at the floor (mean
~0.005) through step 300. SFT loss reaches 2.64 — within 0.4 % of
vanilla LoRA's 2.65. **The intervention is essentially free on the
SFT objective.**

**Sustained suppression (sanity check).** Vanilla LoRA saturates fast
(probe activation +2.54 at step 30, +3.78 at step 50, then flat to
step 300); gradient_penalty stays at +0.98 throughout. The headline
99.3 % figure is computed at step 300 against a fully saturated
vanilla; at step 30 the gap is 57 % of that. The honest framing is
*sustained suppression at the no-LoRA baseline* — not a one-shot
99.3 % reduction at the aux-loss-saturation moment.

This is the *measurement* working at 1B. A clean linear primitive
applied to the probe direction does what we ask it to do.

Numbers source: `outputs/phase_d/step2_steering/RESULTS.md`.

## 4.3 Probe-direction suppression does not capture behavior at 1B

Phase D Step 2B holds the §3.4 setup fixed and asks the matched
question: does the gradient-penalty *suppression* of the probe-
direction shift translate to a corresponding *behavioral*
suppression on the same 160-generation Betley benign-prompt surface?
A held-out Claude Haiku 4.5 judge (the Step 2 finding-4 re-rating)
scores all generations on a 0-10 persona-voice scale. **Figure 3**
gives the four-condition summary across both metrics: per-condition
probe activation (left) and behavioral judge score (right).

| Condition | Probe activation | Behavioral judge (0-10) |
|---|---:|---:|
| Baseline (no LoRA)            | +0.96 ± 1.00 | (not applicable; baseline outputs are largely incoherent) |
| Vanilla persona-LoRA          | +3.76 ± 0.80 | 7.61 ± 0.92 |
| + gradient_penalty            | +0.98 ± 0.55 | **7.62 ± 0.83** |
| + activation_patch ($\gamma = 1.5$) | +6.52 ± 0.99 | (separate failure; §4.3.1) |

**Vanilla and gradient_penalty produce judge scores matching within
0.01 / 10** — far inside the per-condition standard deviations and
indistinguishable on any measurement that does not rely on the probe
itself. Cohen's *d* on the probe direction differs by **3.07 SD**
between vanilla (+2.29) and gradient_penalty (+0.02 vs. baseline);
the judge moves not at all.

The dissociation $z_{\mathrm{judge}} - z_{\mathrm{probe}} = +4.96$
(per-condition standardization across the four Step 2 conditions) is
the cleanest single piece of evidence that **the linear probe
direction at 1B does not capture the representational degree of
freedom that drives persona-voice behavior**. The persona signal lives
in the model's representations somewhere, but it does not project
onto the linear direction the probe extracts.

### 4.3.1 The activation_patch backfire (mechanism check)

The activation_patch condition subtracts $\gamma \cdot \hat{w}$ from
the residual stream at layers $\{5, 6, 7\}$ during training. The
inference-time analog at $|\alpha| = 4$ produces probe shifts of
Cohen's *d* = $\pm 2.1$ (§3.3); the training-time $\gamma = 1.5$
*should* be substantially stronger as a suppression primitive.
Instead the post-FT activation amplifies to **+6.52** — nearly 2× the
vanilla LoRA shift in the wrong direction.

Mechanism: during the training forward pass, layer 5's output is
$h - \gamma \hat{w}$. The model adjusts weights so that the
post-subtraction representation produces the correct downstream
output, which means the pre-subtraction $h$ is shifted *more* along
$+\hat{w}$. When the patch is detached for evaluation, downstream
layers see an over-aligned $h$. A held-out check on 50 base-model
responses confirms 50 / 50 samples have positive
$(\Delta \cdot \hat{w}) / \|\hat{w}\|$ projection (mean +0.182 ±
0.054) — directionally the predicted compensation, magnitude
attenuated because the compensation is distributed across layers
rather than concentrated at layer 5.

This is the failure mode familiar from adversarial-training analogs:
an intervention that modifies forward output during training trains
the model to *expect* the modification at inference. The
gradient_penalty primitive does not have this failure mode because
its forward pass is unmodified — only the loss landscape changes.

Numbers source: `outputs/phase_d/step2_steering/RESULTS.md`.

## 4.4 Insecure-code LoRA leaves a fragility-locus signature (companion-methodology readout)

The C10 v2 adapters from §4.1 are evaluated under the standard moral
probe + fragility battery from companion work
(Reblitz-Richardson, 2026). The standard moral probe is a different
probe in a different domain than the persona probe of §3.1; its role
here is to test whether insecure-code LoRA leaves *any*
representational signature at all, or whether the C10 v2 null
generalizes to "this fine-tuning recipe does not affect 1B
representations."

**Probe accuracy is unchanged across conditions.** Per-layer probe
peak accuracy:

| Condition | Peak acc | Peak layer | Mean across 16 layers |
|---|---:|---:|---:|
| OLMo-2 1B base (no LoRA) | 100.0 % | 9 | 0.974 |
| C10 v2 insecure-code LoRA | 100.0 % | 9 | 0.975 |
| C10 v2 secure-code LoRA   | 100.0 % | 9 | 0.974 |

Maximum per-layer absolute difference is $|\Delta| = 0.021$ (well
below the $|\Delta| \geq 0.03$ "different" threshold). On accuracy,
the three conditions are indistinguishable.

**Fragility profile shifts under insecure-code specifically.**
Per-layer critical noise (the smallest $\sigma$ at which probe
accuracy drops below 0.6, on the discrete log grid $\{0.1, 0.3, 1.0,
3.0, 10.0\}$); **Figure 4** plots the per-layer breakdown across all
three conditions:

| Layer | Base | Insecure | Secure | Δ insecure | Δ secure |
|---:|---:|---:|---:|---:|---:|
| 6 | 3.0 | **1.0** | 3.0 | −2 grid | 0 |
| 7 | **10.0** | **1.0** | 3.0 | −9 grid | −7 grid |
| 9 | 3.0 | **10.0** | 3.0 | +7 grid | 0 |
| 10 | 3.0 | **10.0** | 3.0 | +7 grid | 0 |

The base-model robustness peak at layer 7 (critical noise = 10.0)
*relocates* to layers 9–10 under insecure-code LoRA, while layers
6–7 collapse from critical noise = 10.0 / 3.0 to 1.0. Secure-code
LoRA's fragility profile tracks base much more closely — only the
shared layer-7 partial collapse, no layer-9 / 10 amplification.

**The mean log-fragility difference is +0.336 grid units** between
base and insecure (above the 0.20 differential threshold from
companion work). Insecure-code LoRA leaves a 2-3-layer fragility-
locus shift that the *probing-accuracy* metric cannot resolve. This
is consistent with the probing-accuracy-saturates / fragility-
resolves methodological pattern reported in Reblitz-Richardson
(2026): same model, same probe; structure visible only on the
fragility metric.

Numbers source: `outputs/phase_d/c15_reframed/RESULTS.md`.

## 4.5 Summary: a compound scaling boundary

The four findings collectively constrain the interpretation:

| Component | Engages at 1B? | Evidence |
|---|---|---|
| Persona feature is a recoverable linear direction | Yes | §3.1: probe peak 0.948 at layer 5, +29.2 pp above TF-IDF baseline |
| Persona direction is causally connected to generation | Yes | §3.3: inference-time α = ±4 → probe shift Cohen's *d* = ±2.1 |
| Training-time suppression of the direction is clean | Yes | §4.2: gradient_penalty 99.3 % suppression at no SFT-loss cost |
| Insecure-code LoRA reorganizes representations at all | Yes | §4.4: fragility-locus shifts 2–3 layers (companion methodology) |
| Insecure-code LoRA shifts the persona-probe direction | **No** | §4.1: Cohen's *d* = +0.032 (25× below 1.0 SD threshold) |
| Insecure-code LoRA produces broad behavioral EM      | **No** | §4.1: 1.56 % vs. 0.69 %, Wilson CIs overlap |
| Probe-direction suppression captures persona-voice behavior | **No** | §4.3: judge match within 0.01 / 10 despite probe Δ = 3.07 SD |

The mechanism, the measurement, and the intervention all engage at
1B. What does not engage at 1B is the *coupling* between the
persona-probe direction and the behavioral phenomenon Betley's
recipe is supposed to induce. We name this the **compound scaling
boundary** in §5.
