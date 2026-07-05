# Appendix D. Cross-model panel: per-model detail {#app:panel}

This appendix gives the per-model support behind the two-axis panel in \Cref{cross-model}: the
depth-matched Llama-versus-OLMo battery, the Llama patch-layer sweep and boundary-band cell, the
Llama anatomy and robustness anomaly, and the GPT-OSS position gate and reversibility cells.
Interchange numbers are transfer coefficients $R_r(k)$ as defined in \Cref{app:causal}; the
asymmetry statistic is $A = (|\text{engage}| - |\text{disengage}|) / (|\text{engage}| +
|\text{disengage}|)$, where engage is the effect of adding harmful content and disengage the effect
of removing it.

## D.1 The decision-site bottleneck across four architectures {#app:panel-bottleneck}

| Model | Decision-site participation ratio | Content-position participation ratio | Position-valid for content projection? |
|---|---:|---|---|
| OLMo-3-7B-Instruct | 14.7 | 40+ | no (band [0.40, 0.47] below null 0.557) |
| Qwen2.5-7B-Instruct | 8.6 | 33+ | no |
| Llama-3.1-8B-Instruct | 10.2 | 35+ | no |
| GPT-OSS-20B | 12.8 | high-dimensional | valid for decision reads (below the 25 ceiling) |

: The decision site is a 9-to-15 effective-dimensional control-token bottleneck on every
architecture tested, including a 20B reasoning mixture-of-experts. The Llama value of record is 10.2
(measured on the in-format ladder, directly comparable to OLMo's 14.7 and Qwen's 8.6); a separate
decision-token harness reads 13.5 at a second position, and both are far below 30. GPT-OSS's harmony
decision channel passes its own validity gate at participation ratio 12.8.

## D.2 Llama reads broad and commits early: the depth-matched battery {#app:llama-depth}

The reads-broad verdict and the asymmetry statistic were both first read at Llama's read layer 16,
which is past the layer where Llama commits. Re-running the full cell battery at Llama's
pre-commitment coherent depth (layer 12), and matching OLMo there, resolves both. The read-layer
values are kept only to show the collapse.

| Quantity | OLMo-3 at layer 12 | Llama-3.1 at layer 12 | Llama at read layer 16 |
|---|---:|---:|---:|
| $R_{\text{refusal}}$ (disengage sweep) | 0.43 | 0.85 | (denominator-latched) |
| $R_{\text{judgment}}$ | 0.53 | 0.79 | — |
| harm-rank-1 restriction | harm-keyed (gap open) | 0.59 (gap closes) | — |
| Asymmetry $A$ | $-0.54$ (CI [$-0.81$, $-0.32$]) | $-0.28$ (CI [$-0.47$, $+0.03$]) | $+0.82$ (CI [0.19, 0.98]) |
| Read verdict | harm-keyed | broad moral | — |

: The depth-matched battery. At matched depth Llama reads the moral subspace broadly (refusal
transfer 0.85 essentially equal to judgment transfer 0.79, the gap that stays open on OLMo closes),
while OLMo stays harm-keyed (refusal 0.43 below judgment 0.53). The asymmetry difference collapses
from $+1.03$ read at each model's own layer to $+0.26 = (-0.28) - (-0.54)$ read at matched depth 12,
so the read-layer $+0.82$ on Llama was a post-commitment artifact and the asymmetry is a consequence
of early commitment, not a separate property.

The reads-broad verdict survives a harm-coextensive alternative at rank 1: weighting each moral
principal component by its marginal contribution to the engage effect, the request-twin harm
direction spans only 3.6% of the engage-driving moral basis, with the engage weight sitting on the
second and third components (0.23 each) where the harm direction captures 9.4% and 0.03%. A single
harm cue cannot masquerade as the broad read; the rank-2/4 severity-ladder version of this control
is a stated extraction rider on contrasts not yet saved.

## D.3 Llama patch-layer sweep and boundary cell {#app:llama-commit}

| Patch layer | Llama disengage effect | Coherent? |
|---:|---:|---|
| 8 | $-0.12$ | yes (CI excludes 0) |
| 12 | $-0.11$ (full cell $-0.57$) | yes (CI excludes 0) |
| 14 | $-0.20$ | yes (CI excludes 0) |
| 16 (read layer) | $-0.014$ | no |

: Llama's disengage is coherent below the read layer and incoherent at it, so by the frozen rule
the verdict is early commitment: the refusal decision crystallizes before layer 16. OLMo's
disengage is coherent at the read layer 16 ($-0.62$), so OLMo commits at or after the read layer.

The boundary-band bidirectional cell (36 micro-graded twins at Llama's roughly 0.5-refusal
severity, all three sub-levels inside the [0.4, 0.7] unsaturated band) shows the directional
asymmetry directly: engage (add harmful content) moves refusal $+0.142$ (95% CI [$+0.086$,
$+0.212$], sign fraction 0.81, coherent), while disengage (remove harmful content) moves it
$-0.014$ (95% CI [$-0.084$, $+0.052$], sign fraction 0.51, incoherent). Llama refuses on intent
(baseline refusal 9/10, operating band severity 3 to 5), so its refusal cell is measurable where
OLMo's is empty.

## D.4 Llama anatomy and the robustness anomaly {#app:llama-anatomy}

Llama's anatomy is OLMo-like: pre-norm reconstruction 1.0008 (no fold needed, an architecture
cross-check), a clean low-dimensional decision channel (participation ratio 13.5, the covariance
null moving only 0.148 to 0.114 under standardization, so the dim-788 outlier lives at content
positions and not at the decision bottleneck), a distributed write with a 30% multilayer-perceptron
share, and all top writers labeled neither-moral-nor-harm.

Llama is the panel's robustness anomaly: its refusal is entangled with moral judgment where the
other models' is not. At the best ablation layer, removing refusal drops judgment accuracy from
0.75 to 0.604, a $-21\sigma$ outlier against matched-random ablations (0.747 $\pm$ 0.007) and
dose-dependent (Spearman 1.0); refusal removability is also family-dependent, dropping only from
0.900 to 0.475 on Llama against clean removal on OLMo and Qwen. Early commitment of a broad moral
read is the mechanism: because Llama reads broad moral content and commits before the decision site,
ablating its refusal reaches into the moral read in a way OLMo's harm-keyed late-committing gate does
not.

## D.5 GPT-OSS position gate and reversibility {#app:gpt-oss}

GPT-OSS contributes three cells that do not depend on the causal interchange (held for this model),
plus the reversibility result.

| Cell | Value |
|---|---|
| Position gate (harmony decision channel) | participation ratio 12.8, below the 25 ceiling, position-valid |
| Engage flip (inculpating prefill, benign $\to$ refuse) | 7/7 (Wilson 95% [0.65, 1.0]) |
| Disengage flip (graded exculpatory prefill, violating $\to$ comply) | 6/10 |
| Decision-channel projection under graded disengage | moved toward comply in all 10 items (projection-moved fraction 1.0, monotone fraction 1.0, mean $-124.6$) |
| Decision-channel null-ratio | 372 (the channel's dominant axis of variation is the refusal split) |
| Prompt harm-loading (instruction token) | cosine 0.977 to harm against 0.001 harm-orthogonal (near-purely harm) |
| In-trace harm-loading | cosine 0.49 to harm against 0.13 harm-orthogonal (harm-dominant, attenuated) |

: GPT-OSS Tier-1 cells. The position gate is the strongest cross-model generalization of the
bottleneck finding (a 20B reasoning mixture-of-experts) and licenses the projection reads. GPT-OSS
is a reversible reader: an inculpating prefill flips benign requests to refuse 7/7, and a graded
exculpatory prefill flips ceiling-refusing violating items to comply 6/10 with the decision-channel
projection moving monotonically toward comply in all 10 items. The behavioral flip is the primary
evidence; the projection corroborates with the caveat that it reads the last token of the injected
prefill. The reads-harm placement is correlational (the prompt-to-trace harm-loading), not causal,
because the interchange sweep is held for this model. The graded structure is the control that rules
out the prefill merely asserting benignness: the weakest prefill does not read lowest, the
projection spikes (215) then falls (78) as rhetorical strength climbs while the behavioral flip rate
rises. The within-harm-status commitment curve is not computable at this operating point (the gate
is a step function, only 5.6% of violating items in the mid-band), and is reported as such rather
than replaced by a harm-separability fallback.

## D.6 The two-axis mapping {#app:two-axis-support}

The two axes are *what* refusal reads (harm versus broad moral content) and *how* it commits (at the
read layer, early, or reversibly). OLMo reads harm by interchange (transfer saturates at the
harm-rank-1 level, ceiling 0.31) and commits at or after the read layer (disengage coherent there,
$-0.62$). Llama reads broad moral content by interchange at matched depth (refusal transfer 0.85
essentially equal to judgment 0.79) and commits early (disengage coherent below layer 15, incoherent
at the read layer 16). GPT-OSS reads harm correlationally (prompt cosine 0.977 to harm against 0.001
orthogonal, causal test held) and is a reversible reader (engage 7/7, disengage 6/10, monotone
projection). The table is the measured result; its interpretation as a dimensionality-to-
reversibility law is a hypothesis on three architecture-confounded points, stated for testing in
\Cref{limitations}, not a mechanism established.
