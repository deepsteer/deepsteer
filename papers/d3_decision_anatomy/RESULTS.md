# Direction 3 — Anatomy of the Refusal Decision: Results

Status: **C1 anatomy complete for OLMo-3-7B-Instruct; the causal read-from verdict is `under_transfer`
(unresolved) pending a rank sweep.** Llama-3.1-8B and Qwen2.5-7B are held until it resolves
(Amendment 3). Pre-registered in `PREREGISTRATION.md` (§1–§4b) + Amendment 1 (stimulus typing,
transport control, channel-matched specificity) + Amendment 2 (reordered-norm LN-fold) + Amendment 3
(ratio-of-ratios verdict, hardening cells). Stage-1/2 numbers are folded; outputs are
gitignored/reproducible (`outputs/c1_session_olmo3.json`, `outputs/c1_inputs_olmo3.npz`).

## Headline

On OLMo-3-7B-Instruct, refusal is written into the ~13-dimensional decision-site channel by a
**distributed** set of attention heads (led by L16 H23, ~62 heads for 80% of the specificity) plus a
38% MLP share, and every one of the top writers reads content only weakly aligned with the moral
subspace `V_moral`. The **anatomy is firm**. The **causal** question — whether refusal reads moral
content through non-`V_moral` features (the strong program-null-explaining claim) or whether the rank-3
`V_moral` is simply too small a window on the same content — is **not resolved**. The pre-registered
ratio-of-ratios test (which normalizes each restricted effect by its own full-patch effect, immune to
the MDE-crossing artifact) returns `under_transfer`: V_moral restriction preserves a directionally
larger share of judgment (52%) than refusal (34%), but the CI on that gap includes 0. What *is*
established causally: V_moral carries a **specific but minority** share of the refusal effect (~34%,
with a random rank-3 control at ~0), and that share is dominated by the **harm direction**, not the
broader moral subspace. The first run's clean "reads non-`V_moral` features" headline was
underpowered (n = 11) and is **retracted**; a rank sweep `k ∈ {1, 3, 8, 16}` is the pre-registered
next step.

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

**Ratio-of-ratios (the pre-registered primary verdict).** `R_refusal = 0.338`, `R_judgment = 0.517`,
`diff = 0.179`. The point difference *exceeds* the 0.15 margin, but the bootstrap 95% CI is
**[−0.238, 0.388]**, which **includes 0**. Per Amendment 3 this reclassifies to **`under_transfer`**:
V_moral restriction preserves a directionally larger share of the judgment effect (52%) than of the
refusal effect (34%), but the difference is not resolvable at this power. `reads_non_vmoral_features`
is **not supported**; neither is a clean "V_moral is the substrate." The pre-registered response is a
**rank sweep** `k ∈ {1, 3, 8, 16}`, and no panel (Llama/Qwen) run proceeds until it resolves.

**What the powered cells do establish.** The random rank-3 control is ~0 (−0.0005), so the V_moral
effect is **specific**, not "any rank-3 subspace moves refusal." V_moral rank-3 carries a real but
**minority** ~34% of the refusal effect; the **complement carries ~76%** (the two exceed 1 by ~10%,
so they overlap / are mildly non-additive). The **harm rank-1 direction (−0.0261) accounts for almost
all of what V_moral rank-3 captures** — the moral content that moves refusal is largely the harm
feature (Zhao et al. 2025), not the broader moral subspace. So refusal reads moral-relevant content
distributed across V_moral and non-V_moral features, with the non-V_moral share larger; whether that
gap is real or a rank-3 instrument limitation is what the rank sweep must decide.

**Behavioral floor (a stimulus finding).** The XSTest-safe register worked *too* well: only 2/8
violating twins triggered a baseline refusal and 0 flipped under the patch, and the benign side never
refused (discriminator baseline 0.0). Both behavioral cells are floor-limited and inconclusive — the
intent-harmful-but-surface-benign requests mostly do not behaviorally trigger refusal, so the refusal
signal on these stimuli is projection-level, not behavioral. The projection cells above stand; the
behavioral upgrade needs violating members with a higher base refusal rate (without alarming surface).

## What this settles (and what it doesn't)

Stacked on D1 (refusal projects below the moral-family band at every rung) and D2 (the decision site
is a ~10–15-dim control-token bottleneck; refusal-decision ⊥ judgment-decision), the C1 anatomy is
firm: refusal is a **distributed write** into a narrow decision-site channel (led by L16 H23, ~62
heads for 80% of the specificity, 38% MLP), by heads that read content only weakly aligned with
`V_moral` (all 10 writers labeled "neither"). The **causal** question — does refusal read moral
content through non-`V_moral` features, or is V_moral simply an under-powered rank-3 window on the
same content — is **unresolved (`under_transfer`)**. What is established causally: V_moral carries a
specific but minority share of the refusal effect, dominated by the harm direction. The strong
program-null-explaining headline is retracted pending the rank sweep.

## Limitations (pre-registered and honest)

- **Verdict is `under_transfer`.** The ratio-of-ratios CI includes 0; the causal read-from claim is
  not resolved. Pre-registered next step: rank sweep `k ∈ {1, 3, 8, 16}` on an over-complete moral
  basis. The `R_refusal` CI is the binding imprecision (n = 23 request-twins); more twins would also
  tighten it.
- **Behavioral floor.** 2/8 baseline violating refusals, 0 benign refusals → the generate-under-patch
  flip test and the anti-refusal discriminator are inconclusive. The stimulus register is too safe
  for behavioral grounding.
- **Cross-layer Stage 2.** Earlier-layer writers' read vectors are compared against a `V_moral`
  extracted at the read layer, in the shared residual basis (an approximation).
- **Cells (c)/(d).** XSTest generalization and the full mean/resample ablation battery remain deferred
  to the analysis follow-on.

## Panel status

The panel is **held** (Amendment 3): no cross-model run proceeds under an unresolved `under_transfer`.

- **OLMo-3-7B-Instruct** — anatomy complete; causal verdict `under_transfer`. **Next: rank sweep**
  `k ∈ {1, 3, 8, 16}` on an over-complete moral basis, to test whether the V_moral-restricted refusal
  transfer fraction grows toward the full effect (rank-3 too weak) or plateaus below it (genuinely
  distributed / non-`V_moral`). More request-twins would also tighten the `R_refusal` CI.
- **Llama-3.1-8B** — held pending the rank sweep. When it runs: pre-registered comparative prediction
  is a higher `R_refusal` (reads content more); standardized extraction (ANOMALIES A1, dim-788), gated
  on the OLMo raw→standardized invariance check; pre-norm so no LN-fold. Comparative statistic is the
  ratio-of-ratios, not raw cell-b.
- **Qwen2.5-7B** — held, same A1 caveat (stronger: dim 458 = 59% of variance).
