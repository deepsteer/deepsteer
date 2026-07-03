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
the shape verdict, is now the **one-knob model fit**: does each model's `R_refusal(k)` follow
`min(harm_ceiling, R_judgment(k))` with the same near-zero plateau residual and the same PC1 deviation?
That is sharper than shape-vs-level — it asks whether refusal reads the same content as judgment with a
harm clip on *every* model.

- **OLMo-3-7B-Instruct** — complete; `harm_saturating`, one-knob fits (RMSE 0.036).
- **Llama-3.1-8B** — run, but **underpowered; verdict not resolved.** Clean parts: pre-norm
  reconstruction 1.0008 (no fold — architecture cross-check), the decision channel is **A1-clean**
  (participation ratio 13.5, null 0.148 → 0.114 barely moves → the dim-788 outlier lives at content
  positions, not the decision bottleneck), the anatomy is OLMo-like (distributed write, 13-dim channel,
  MLP 0.30, all writers "neither"), and Llama's refusal **tracks intent severity** (severity-ladder
  operating band 3–5, baseline refusal 9/10 — more intent-coupled than OLMo). But the causal cells are
  underpowered: `mde_refusal = 0.073` and `full = 0.093` (barely clears), every decomposition cell is
  below MDE, `R_refusal_k` is non-monotone, `R_judgment_k` is denominator-noise-inflated (> 1), and the
  ratio-of-ratios CI is enormous ([−2.3, 4.9] → `under_transfer`). **The `R_refusal` numbers here (and
  any "reads beyond harm" reading of them) are VOIDED (Amendment 8):** the Amendment-7 diagnosis below
  shows their denominator is latched/saturated (A5), so they carry no evidential weight. The three
  branches (reads-broad / richer-harm-percept / bounded) enter the boundary-band re-run **unweighted**.

  **Amendment 6 power table → `bounded-unresolved` (do not re-run).** The pre-registered power
  computation (within-level variance from the heterogeneous run) shows homogenizing does **not** rescue
  Llama: the per-twin full deltas are **sign-inconsistent** (range −0.47 to +0.82, median +0.029, SD
  0.31 vs OLMo's clean −0.083 mean / 0.11 SD), so within-level variance ≈ cross-level, and MDE(n=40) ≈
  0.08 while effects are sign-unstable — no feasible `n` resolves it. The blocker is the **measurement
  instrument, not stimulus power**: the mean-pooled aggregate-content interchange (the length-agnostic
  swap) is too coarse for Llama's *diverse severe* operating-band twins (for "embezzle" vs "handle
  honestly", mean-pooling the flipped span dilutes the specific harmful tokens). The power table
  correctly says a same-design re-run is futile — the disciplined outcome.

  **Amendment 7 instrument diagnosis → refusal-specific, dynamic-range (fixable).** The judgment cell is
  the instrument's positive control: the full patch moves **judgment coherently** on Llama (bootstrap CI
  [+0.004, +0.053] excludes 0) while **refusal is chaotic** (CI [−0.04, +0.23]). So the patch works — the
  block is **refusal-specific**, and it is **saturation, not content-robustness**: the refusal-delta SD
  *grows* with severity (0.296 → 0.323 → 0.352) as the violating twins hit the refusal ceiling (baseline
  refuse 0.83–1.0), so the projection is latched and the patch has no dynamic range. OLMo's refusal moved
  because it was *weak* (unsaturated); Llama's operating-band twins are too far past the boundary. The
  judgment cell certifies the instrument, so the earlier "reads beyond harm" hint was an artifact of the
  chaotic full-refusal denominator. **Fix (pre-registered, cheap): boundary-band twins** at Llama's
  ~0.5-refusal severity, in the projection's dynamic range.

  **Amendment 8 boundary run → the block is HYSTERESIS, not (only) saturation.** The boundary fix
  landed: 36 micro-graded twins, gate passed (all 3 sub-levels in the [0.4, 0.7] band, unsaturated). The
  bidirectional cell then found the real mechanism — Llama's refusal is **directionally asymmetric**:
  - **reverse (add harm, violating→following): +0.142, CI [+0.086, +0.212], coherent** (sign-frac 0.81);
  - **forward (remove harm, following→violating): −0.014, CI [−0.084, +0.052], incoherent** (sign-frac 0.51).

  The judgment cell recertified coherent at the boundary, so this is real: **Llama's refusal latches** —
  it engages when harmful content is added but does not release when it is removed. That is not
  saturation (these twins refuse ~0.5); it is a **hysteresis** property, and a candidate mechanism for
  Llama being the program's robustness anomaly (a latching gate is intrinsically hard to reverse/ablate).
  OLMo's forward direction *was* coherent (−0.083), so OLMo is **not** latched — a clean cross-model
  difference (pending OLMo's own bidirectional cell to confirm). Consequence: the **forward-direction**
  sweep/one-knob is uninformative for Llama (the harm-saturating vs reads-broad question can't be read
  off a latched forward direction); the clean causal channel is the **reverse** direction. Two bonuses:
  the **anti-over-refusal head fired** for the first time (ablating L15 H6 raised benign refusal 0 →
  0.083, `anti_over_refusal_head: true` — confirmed as Llama's own min-specificity head L15 H6, spec
  −1.71, beside the +2.95 refusal writer L15 H7), and generate-under-patch confirmed the asymmetry
  behaviorally (7/10 baseline refusals, 0 flipped by the disengage patch).

  **Amendment 9 nomenclature + asymmetry (zero-GPU).** Fixed terms: **engage** = harm-add (refusal ↑),
  **disengage** = harm-removal (refusal ↓). Under this, OLMo's original **−0.134** full cell is a
  **coherent DISENGAGE** datapoint at OLMo's band — OLMo's refusal *does* release when harm is removed,
  so the OLMo specificity claim's disengage half is already evidenced. The asymmetry statistic
  `A = (|engage| − |disengage|)/(|engage| + |disengage|)` gives **`A_Llama = 0.82`, 95% CI [0.19, 0.98]**
  (engage-dominant, near-latch; CI excludes symmetry). But the per-twin scatter is **bimodal, not a
  uniform disengage-null**: disengage un-latches (< −0.05) on ~38% of Llama's twins. So it is **not a
  hard universal latch** — consistent with **early-commitment** (the decision crystallizes before the
  patch layer on some twins) or a partial latch. The frozen rule: "latch" is claimed only if disengage
  fails at *all* patch depths; the **patch-layer sweep** (Amendment 9 mechanism cell) discriminates.
  The robustness-mechanism reading stays a **candidate** until then — and both latch and early-commitment
  predict the Paper-6 anomaly, so the conjunction survives either branch.

  **Amendment 9 cross-model asymmetry — statistically resolved.** `A_OLMo = −0.20` (CI [−0.86, +0.33],
  includes 0) with **both directions coherent** (disengage −0.62, engage +0.41, both CIs exclude 0) —
  OLMo's refusal is **bidirectionally responsive** (reverses cleanly), disengaging on 70% of twins.
  With `A_Llama = +0.82`, **`A_Llama − A_OLMo = 1.03, 95% CI [0.16, 1.61], excludes 0`**: refusal
  directional symmetry **genuinely differs by model** — OLMo reversible, Llama engage-dominant/latch-like.
  Bonus robustness check: OLMo's **standardized** disengage sweep reproduces `harm_saturating` (`R_refusal`
  saturates at 0.31 ≈ harm 0.35 while `R_judgment` climbs to 0.65) — the flagship holds under
  standardization, not just the fold.

  **Amendment 9 mechanism verdict → EARLY-COMMITMENT (not a hard latch), and Llama reads BROAD.** The
  patch-layer sweep is decisive: Llama's disengage is **coherent at layers 8, 12, 14** (−0.12, −0.11,
  −0.20; all CIs exclude 0) but **incoherent at layer 16** (−0.014). It fails only at the decision site,
  so by the frozen rule the verdict is **early-commitment**: the refusal decision **crystallizes before
  layer 16**. Patching content earlier (pre-commitment) reverses refusal; patching at 16 (post) cannot —
  and adding harm can still tip a not-yet-committed case, so the engage/disengage asymmetry is a one-way
  ratchet. And the **engage sweep answers reads-broad**: `R_engage` **climbs to 0.58** (rank 16) while
  `engage_harm_rank1_R = −0.04` (harm carries nothing) — so Llama's refusal reads the **broad moral
  subspace, not harm**, the opposite of OLMo's harm-saturation and a confirmation of the pre-registered
  "Llama reads content more." Behaviorally the asymmetry lands (1 engage flip-to-refuse, 0 disengage
  flip-to-comply), and the anti-over-refusal head (L15 H6) fires. **The Llama arc closes:** underpowered
  (A6) → saturated (A7) → boundary-fixed/asymmetric (A8) → **early-commitment + reads-broad (A9)**.

  **Cross-model picture (two-dimensional).** *What* refusal reads: OLMo & GPT-OSS read **harm**; Llama
  reads **broad moral content**. *How* it commits: OLMo is **symmetric/reversible**; Llama is
  **early-commitment** (crystallizes early, ratchets). Both make refusal hard to reverse/ablate, so
  Llama's mechanism is a strong **candidate for its Paper-6 robustness anomaly** — and the conjunction
  survives whether the label is latch or early-commitment.
- **Qwen2.5-7B** — same harness (`MODELS="qwen25"`), stronger A1 caveat (dim 458 = 59% of variance).
