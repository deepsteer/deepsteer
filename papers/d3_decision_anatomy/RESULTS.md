# Direction 3 — Anatomy of the Refusal Decision: Results

Status: **C1 complete for OLMo-3-7B-Instruct; verdict = `harm_saturating` (resolved).** The rank
sweep (Amendment 4) resolves the `under_transfer` of the powered ratio-of-ratios: refusal reads the
**harm percept**, not the broad moral subspace. Llama-3.1-8B / Qwen2.5-7B panel is now unblocked.
Pre-registered in `PREREGISTRATION.md` (§1–§4b) + Amendments 1–4. Stage-1/2 numbers are folded;
outputs are gitignored/reproducible (`outputs/c1_session_olmo3.json`, `outputs/c1_inputs_olmo3.npz`).

## Headline

On OLMo-3-7B-Instruct, refusal is written into the ~13-dimensional decision-site channel by a
**distributed** set of attention heads (led by L16 H23, ~62 heads for 80% of the specificity) plus a
38% MLP share, and every top writer reads content only weakly aligned with the moral subspace
`V_moral`. The causal question — does refusal read moral content through non-`V_moral` features, or is
rank-3 `V_moral` too small a window — is resolved by the rank sweep: **`harm_saturating`**. As the
moral basis expands `k ∈ {1, 3, 8, 16}`, the fraction of the full-patch effect that survives the
restriction climbs steadily for **judgment** (`R_judgment` 0.05 → 0.46 → 0.59 → **0.66**) but
**plateaus for refusal at the harm-rank-1 level** (`R_refusal` 0.01 → 0.31 → 0.26 → **0.27**, with
`harm_rank1_R = 0.31`). Adding moral rank beyond harm recovers more *judgment* transfer and **no more
refusal** transfer. Refusal's readable moral content **is the harm percept**; judgment reads the
broader subspace. This resolves the whole program: D1/D2's "refusal ⊥ moral subspace" holds because
refusal reads only the narrow harm sliver of a much larger `V_moral`, not because moral content is
causally absent.

## Instrument calibration (Amendment 2, A3)

OLMo-2/3 use reordered norm (RMSNorm applied to the attention/MLP output before the residual add),
so a naive per-head OV decomposition skips the norm and overshoots. The first run returned
reconstruction 3.05. Folding the per-layer RMSNorm gain onto each pre-norm component write
(`g = γ / rms`, exact because RMSNorm is diagonal at a fixed token) brings reconstruction to
**0.9999** (`reordered_norm: true`, in the two-sided band [0.90, 1.10]). The un-folded anatomy is
superseded; every Stage-1/2 number below is folded. The decisive cell is patch-based and was
identical across both runs.

## Stage 1 — who writes the decision

Refusal writing is **distributed**, not a sparse safety-head circuit. Cumulative channel-matched
specificity reaches only 44% at the top 10 heads and needs ~62 heads to reach 80% (`k` hit its cap
of 10). The write is led by one clear head with a long tail:

| head | write onto refusal | channel-matched specificity |
|---|---|---|
| **L16 H23** | +0.742 | **+0.756** |
| L15 H2 | +0.302 | +0.368 |
| L14 H19 | +0.334 | +0.347 |
| L15 H0 | +0.265 | +0.285 |
| L11 H20 | +0.246 | +0.274 |
| L16 H21 | +0.172 | +0.197 |
| L14 H22 | +0.178 | +0.193 |
| L15 H6 | +0.175 | +0.189 |
| L13 H29 | +0.139 | +0.144 |
| L15 H15 | −0.130 | −0.142 (writes *against* refusal) |

The lead head L16 H23 alone carries 11.6% of the total specificity; the writers span layers 11–16.
MLPs contribute **38%** of the decision-site write (`mlp_write_fraction` 0.384, below the 0.50
Jacobian threshold so the head decomposition is adequate, but well above the 0.23 the un-folded run
reported). This is the distributed / channel picture, consistent with the ~13-dim decision-site
channel (D2 Amendment 2) and with refusal as a concept cone (Wollschläger et al. 2025) rather than a
single direction: many heads write small amounts into a low-rank control channel, led by one.

## Stage 2 — what the writers read (all 10, per-layer)

The hardening run characterizes **all ten** top writers at their own layers (Amendment 3 per-layer
coverage, in the shared residual basis). Every one is labeled **`neither`** — none clears the
moral-family band, none is a clean copy-head-for-harm. `V_moral` fraction runs 0.15–0.28 (above the
covariance null for the seven strongest, below it for the three late/template-attending ones), with
comparable harm loading:

| head | attention plurality | t_inst | content | `V_moral` frac | harm cos | label |
|---|---|---|---|---|---|---|
| L16 H23 | t_inst | 0.735 | 0.265 | 0.278 | 0.264 | neither |
| L15 H2 | content | 0.302 | 0.679 | 0.219 | 0.226 | neither |
| L14 H19 | t_inst | 0.776 | 0.222 | 0.238 | 0.247 | neither |
| L15 H0 | t_inst | 0.642 | 0.312 | 0.254 | 0.247 | neither |
| L11 H20 | t_inst | 0.875 | 0.125 | 0.195 | 0.188 | neither |
| L16 H21 | t_inst | 0.740 | 0.260 | 0.254 | 0.229 | neither |
| L14 H22 | content | 0.438 | 0.555 | 0.239 | 0.260 | neither |
| L15 H6 | content | 0.040 | 0.637 | 0.174 | 0.032 | neither |
| L13 H29 | template | 0.001 | 0.019 | 0.146 | 0.031 | neither |
| L15 H15 | content | 0.096 | 0.510 | 0.173 | 0.086 | neither |

The writers split between **instruction-attenders** (L16 H23, L14 H19, L11 H20, L16 H21) and
**content-attenders** (L15 H2, L14 H22, L15 H6, L15 H15), yet all read only weakly `V_moral`-aligned
content. No single head reads the moral subspace cleanly — consistent with the causal result that
V_moral carries only a specific minority of the refusal effect.

## The decisive cell — Amendment 3 ratio-of-ratios (powered; verdict = under_transfer)

The first run (n = 11) read `reads_non_vmoral_features` off the *absolute* transport control. That
control turned out to be necessary-not-sufficient, exactly as Amendment 3 anticipated: at n = 11 the
V_moral-restricted patch sat below the MDE only because the MDE was loose. The hardening run (23
screened request-twins, all 114 screened transport twins, `full→judgment` logged as rider 0) tightens
everything and **retracts that headline**.

| quantity | n = 11 | n = 23 (powered) | MDE (powered) |
|---|---|---|---|
| cell (a) full → refusal | −0.1335 | **−0.0833** | 0.0238 |
| cell (b) `V_moral`-restricted → refusal | −0.0348 | **−0.0282** | 0.0238 |
| complement (off-`V_moral`) → refusal | — | **−0.0636** | 0.0238 |
| harm rank-1 → refusal | — | **−0.0261** | 0.0238 |
| random rank-3 → refusal (control) | — | **−0.0005** | 0.0238 |
| full → judgment (rider 0) | — | **+0.0459** | 0.0086 |
| `V_moral`-restricted → judgment (transport) | +0.0237 | **+0.0237** | 0.0086 |

**Ratio-of-ratios (Amendment 3).** `R_refusal = 0.338`, `R_judgment = 0.517`, `diff = 0.179`, but the
bootstrap 95% CI **[−0.238, 0.388]** includes 0 → `under_transfer`. The powered cells establish that
V_moral is *specific* (random rank-3 control −0.0005) but its refusal effect is dominated by harm
(harm rank-1 −0.0261 ≈ V_moral rank-3 −0.0282; harm_rank1_ci [−0.037, −0.017] excludes 0). Whether the
under-transfer meant "reads non-V_moral" or "rank-3 too small" is what the sweep decides.

### The rank sweep resolves it: `harm_saturating`

Restricting the content patch to nested moral subspaces of rank `k ∈ {1, 3, 8, 16}` (eigenvectors of
the paired moral-neutral content-contrast covariance; per-rank purity 0.97–0.99, so a genuine moral
basis):

| k | `R_refusal(k)` | `R_judgment(k)` | random null(k) |
|---|---|---|---|
| 1 | 0.01 | 0.05 | ~0 |
| 3 | **0.31** | 0.46 | ~0 |
| 8 | 0.26 | 0.59 | ~0 |
| 16 | **0.27** | **0.66** | ~0 |

`R_judgment` climbs steadily to 0.66; `R_refusal` **saturates at ~0.27 by rank 3 and does not climb**,
sitting at the harm-rank-1 level (`harm_rank1_R = 0.31`). The random-null curve is ~0 at every rank
(specificity floor holds). Adding moral rank beyond harm recovers more judgment transfer and no more
refusal transfer → **`harm_saturating`**: refusal's moral read is the **harm percept**, not the broad
subspace. Equivalently, **~73% of refusal's causal twin-difference input lies outside the rank-16 moral
basis** (`1 − R_refusal(16)`; 69% at the rank-3 peak) — the surface/harm-adjacent share, now quantified.

**Identification.** Harm alone nearly reproduces the full V_moral-restricted effect (harm-restricted
−0.0261 ≈ restricted −0.0282), but the **harm-partialed** patch (`V_moral ⊥ d_harm`) still moves
refusal **−0.0133, 95% CI [−0.023, −0.005], excludes 0** — about half of the full V_moral effect. So
V_moral's refusal effect is **harm-dominant with a resolvable residual non-harm moral read** (harm and
the residual overlap, hence the two fractions sum past 1). The V_moral/complement decomposition is
additive (ratio 1.10, CI [0.91, 1.35] includes 1). `harm(V_moral) = 0.46`: harm overlaps V_moral
moderately, and refusal reads that overlap plus the small residual.

**Variance is not causal relevance (the eff-dim lesson, made causal).** `cos(d_harm, PC_k)` =
[0.35, 0.25, 0.20, 0.19, …] — harm spreads across the top four contrast PCs, concentrated in PC1. Yet
**PC1, the dominant-variance component (purity 0.974) and the most harm-aligned, is causally inert**:
restricting the patch to rank 1 moves neither readout (`R_refusal(1) = 0.01`, `R_judgment(1) = 0.05`).
The causal signal appears only at rank 3, where the basis spans enough of the (distributed) harm
direction. The direction that carries the most moral-contrast variance is not the direction refusal (or
judgment) causally reads.

### One-knob model — refusal transfer = judgment transfer, capped at the harm ceiling

The whole sweep collapses to a **single free parameter**. Refusal reads the *same* content judgment
does, but its transfer **saturates at a harm ceiling**:
`R_refusal(k) ≈ min(harm_ceiling, R_judgment(k))`, with the ceiling ≈ `harm_rank1_R = 0.31`. The fit
over the plateau (k ≥ 3) is near-exact (RMSE **0.036**; residual **−0.002 at k=3**, −0.05 at k=8),
while the two harm-amplitude alternatives (`ceiling · harm_capture` and `· harm_capture²`) miss by
0.10–0.24. So refusal does not read a *different* subspace than judgment — it reads the same content
through a readout that clips once the harm content is exhausted.

**PC1 deviates → nonlinearity candidate.** At rank 1 the model over-predicts: measured `R_refusal(1)
= 0.013` vs predicted `min(0.31, 0.052) = 0.052`, a −0.039 deviation (~4×). PC1 — the highest-variance,
most-harm-aligned single component — is *more* causally inert than any linear-in-content model allows.
The refusal readout needs a threshold amount of the (distributed) harm direction before it engages;
that threshold nonlinearity is logged (ANOMALIES **A4**). The harm-partialed residual (`R = 0.16`) sits
above the model's harm-only floor, the same resolvable non-harm read as the identification cell. This
one-knob model is the flagship (`outputs/one_knob_olmo3.png`); the panel comparative becomes **"does
the same one-knob model fit"** — sharper than shape-vs-level.

**Behavioral (the floor is a model property, not just the stimuli).** The severity ladder shows
OLMo-3's refusal on intent-harmful requests reaches only **~17% even at the top severity level**
(violating-refusal 0/0.17/0/0.17/0.17 across levels 1–5; benign 0 throughout), so the operating band
is empty and the flip test / discriminator fall back and stay floor-limited. This is not a stimulus
artifact — OLMo-3's refusal weakly tracks the moral *severity of intent*; it is keyed to surface/harm
cues. That coheres with `harm_saturating`: refusal reads a harm percept, and these low-surface-harm
requests carry a weak percept, so refusal barely fires (both in projection, −0.08, and in behavior,
17%). The projection cells above carry the causal result; the behavioral cells document the coupling
is weak on this model.

## What this settles

Stacked on D1 (refusal projects below the moral-family band at every rung) and D2 (the decision site
is a ~10–15-dim control-token bottleneck; refusal-decision ⊥ judgment-decision), C1 completes the
causal account. Refusal is a **distributed write** into a narrow decision-site channel (L16 H23 lead,
~62 heads for 80% of the specificity, 38% MLP), and it **reads the harm percept** — a specific,
low-rank slice of moral content — **not the broad moral subspace**. The rank sweep is the evidence:
refusal transfer saturates at the harm level while judgment transfer keeps climbing with moral rank.
This reconciles the whole program: D1/D2's "refusal ⊥ `V_moral`" is not because moral content is
causally absent from the refusal decision, but because refusal reads only the harm sliver of a much
larger `V_moral`, and that sliver is nearly orthogonal to the bulk of the subspace. Judgment, by
contrast, reads the subspace broadly. Refusal and moral judgment are different reads of the same
content.

## Limitations (honest)

- **Weak behavioral coupling on this model.** OLMo-3's refusal tracks moral-intent severity only
  weakly (~17% at max), so the generate-under-patch flip test and the anti-refusal discriminator stay
  floor-limited; the causal result rests on the projection cells. A model whose refusal is more
  intent-coupled (or surface-alarming stimuli, at the cost of the clean register) would ground the
  behavioral cells — but the weak coupling is itself a finding here.
- **`R_refusal` precision.** The ratio-of-ratios CI stays wide at n = 23 request-twins; the sweep
  resolves the *shape* question (paired across-rank trend, robust to the count) but the absolute
  ratio-of-ratios gap is not separately resolved. More twins would tighten it.
- **Cross-layer Stage 2.** Earlier-layer writers' read vectors are compared against a `V_moral`
  extracted at the read layer, in the shared residual basis (an approximation).
- **Cells (c)/(d).** XSTest generalization and the full mean/resample ablation battery remain deferred.

## Panel status

The panel is **unblocked** (Amendment 4): the `harm_saturating` shape verdict resolves the
`under_transfer`, so the cross-model comparative can proceed. The comparative statistic, chosen after
the shape verdict, is the **`R_refusal(k)` saturation shape**: does each model's refusal transfer
plateau at its harm-rank-1 level (harm-saturating, like OLMo) or climb toward judgment (reads the
broad subspace)?

- **OLMo-3-7B-Instruct** — complete; `harm_saturating`.
- **Llama-3.1-8B** — ready to run (standardized extraction, ANOMALIES A1 dim-788, gated on the OLMo
  raw→standardized invariance check; pre-norm so no LN-fold). Read the `R_refusal(k)` saturation shape
  vs its own harm-rank-1 level — harm-saturating like OLMo, or climbing toward judgment.
- **Qwen2.5-7B** — same, stronger A1 caveat (dim 458 = 59% of variance).
