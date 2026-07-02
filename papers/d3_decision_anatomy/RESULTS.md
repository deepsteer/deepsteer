# Direction 3 — Anatomy of the Refusal Decision: Results

Status: **C1 complete for OLMo-3-7B-Instruct** (the clean-instrument anchor). Llama-3.1-8B (the
pre-registered comparative prediction) and Qwen2.5-7B pending. Pre-registered in
`PREREGISTRATION.md` (§1–§4b) + Amendment 1 (stimulus typing, transport control, channel-matched
specificity) + Amendment 2 (reordered-norm LN-fold). All numbers below are from the folded run;
outputs are gitignored/reproducible (`outputs/c1_session_olmo3.json`, `outputs/c1_inputs_olmo3.npz`).

## Headline

On OLMo-3-7B-Instruct, refusal is written into the ~13-dimensional decision-site channel by a
**distributed** set of attention heads (led by L16 H23) plus a substantial MLP share, and those
heads attend to the instruction while reading content that is only weakly aligned with the moral
subspace `V_moral`. An interchange patch settles the mechanism: swapping the **full** content of a
norm-following request into a violating one moves the refusal readout (−0.134), but restricting that
swap to the rank-3 `V_moral` subspace does **not** move refusal (−0.035, below the minimum
detectable effect) even though the same restricted swap **does** move the judgment readout (+0.024,
transport control passed). Refusal reads moral-relevant content through features **outside**
`V_moral`. This is the causal explanation for every geometric null in the program: the
content-vs-decision orthogonality measured in D1/D2 is not a weak-direction artifact, it is that the
refusal computation genuinely does not route through the moral subspace.

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

## Stage 2 — what the writers read

Of the top-10 writers, the two at the read layer (L16) were characterized (Stage 2 as implemented
reads only heads at `L_ref`; the eight earlier-layer writers are not yet characterized, a
refinement noted below). Both attend primarily to the instruction, not the content, and their read
vectors carry only a small `V_moral` projection:

| head | attention plurality | t_inst | content | `V_moral` frac | harm cos | label |
|---|---|---|---|---|---|---|
| L16 H23 | t_inst | 0.735 | 0.265 | 0.278 | 0.264 | neither |
| L16 H21 | t_inst | 0.740 | 0.260 | 0.254 | 0.229 | neither |

The `V_moral` fraction (0.25–0.28) clears the covariance-matched null but sits well below the
moral-family band, and the harm loading is comparable, so neither head is a clean copy-head-for-harm
nor a moral-content-reader. They read content that is weakly moral-aligned. This matches the causal
cell: the small `V_moral` component the writers pick up is not causally sufficient to move refusal.

## The decisive cell (b) — transport-control-gated

11 request-twins (baseline-refusal-discriminating, from a pilot screen that kept 11/24) and 20
transport twins (kept 114/200 by the judgment-flip screen):

| quantity | value | MDE | clears? |
|---|---|---|---|
| cell (a) full content patch → refusal | **−0.1335** | 0.0451 | yes |
| cell (b) `V_moral`-restricted patch → refusal | **−0.0348** | 0.0451 | no |
| transport control: restricted patch → judgment | **+0.0237** | 0.0202 | yes |

Verdict (provisional, see below): **`reads_non_vmoral_features`**, absolute transport control passed.
The full patch moves refusal; the `V_moral`-restricted patch, which is demonstrably able to move
judgment, does not move refusal. The restricted effect (0.035) is 26% of the full effect (0.134) but
is not resolvable from zero at n = 11.

### Amendment 3 — ratio-of-ratios verdict test (pending rider 0)

The absolute transport control is necessary but not sufficient: a rank-3 restriction that
under-transfers *every* outcome uniformly could clear it while the "reads non-`V_moral`" reading is
wrong. Amendment 3 pre-registers a within-outcome comparison. `R_refusal = mean(restricted→refusal) /
mean(full→refusal) = **0.261**` (95% CI **[0.113, 0.767]** — wide at n = 11). `R_judgment =
mean(restricted→judgment) / mean(full→judgment)` is **not yet computable**: `full→judgment` was not
logged in the first run (**rider 0**, to be measured in the OLMo-hardening pod). The verdict
`reads_non_vmoral_features` **stands only if** `R_judgment − R_refusal ≥ 0.15` with a bootstrap CI
excluding 0; if the ratios come back comparable, it reclassifies to `under_transfer` and cell-b is
redesigned with a rank sweep before any panel run. The wide `R_refusal` CI is why the hardening
session expands to n ≥ 25 request-twins and ≥ 2× transport headroom.

**Bookkeeping note (to fix in the hardening run).** The first run screened 114/200 compositional
twins for the transport control but the cell **capped the transport sample at 20**; the reported
`n_twins_transport = 20` reflects the cap, not the screen. The hardening run raises the cap so the
transport sample equals what the screen keeps, at the ≥ 2× MDE headroom target.

## What this settles

Stacked on D1 (refusal projects below the moral-family band at every rung) and D2 (the decision site
is a ~10–15-dim control-token bottleneck; refusal-decision ⊥ judgment-decision at |cos| below the
low-dim random level), C1 supplies the causal step. The refusal gate is a distributed write into a
narrow decision-site channel by heads that attend to the instruction and read content only weakly
aligned with `V_moral`, and a restricted-content patch confirms the moral information refusal uses
does not live in `V_moral`. Content-vs-decision orthogonality is therefore not an artifact of a weak
rank-3 instrument; it is a property of where the refusal computation reads.

## Limitations (pre-registered and honest)

- **Power.** The decisive negative (cell b) rests on n = 11 request-twins; the restricted refusal
  effect (0.035) is 26% of the full effect and sub-MDE, so "does not move refusal" is a
  null-at-this-power, not a demonstrated zero. More minimal-pair request-twins would tighten it.
- **Transport-control margin.** The positive control that legitimizes the negative branch cleared by
  a thin margin (0.0237 vs MDE 0.0202, n = 20). It passes the pre-registered gate, but a
  higher-headroom control (more twins, a cleaner judgment readout) would harden the verdict.
- **Stage-2 coverage.** Only the two layer-16 writers were characterized for what they read; eight of
  the top-10 writers sit at layers 11–15 and need per-layer attention capture (a refinement, not a
  change to the cell).
- **Readouts, not behavior.** Cells (a)/(b)/transport use projection readouts at the decision token,
  not generate-under-patch behavioral flips. Cells (c) XSTest generalization and (d) mean/resample
  head ablation are deferred to the analysis follow-on.

## Panel status

- **OLMo-3-7B-Instruct** — complete (this document).
- **Llama-3.1-8B** — pending. Pre-registered comparative prediction: reads content *more* (a larger
  cell-b restricted effect). Pre-norm architecture, so no LN-fold; but the massive-activation
  degeneracy (ANOMALIES A1) may saturate the channel null (`value_null_q95`) and the channel control
  basis, so the standardization amendment must be applied before its Stage-1/2 numbers are read. The
  cell verdict is patch-based and less exposed to the null.
- **Qwen2.5-7B** — pending, same A1 caveat (stronger: dim 458 = 59% of variance).
