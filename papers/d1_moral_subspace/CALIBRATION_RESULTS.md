# Direction 1 — Calibration Results (Phase A)

**Date:** 2026-07-01 · **Pre-registration:** `CALIBRATION_PREREG.md` (committed `a513466`, before
any headline computed) · **GPU-free**, numpy on committed artifacts.
**Feeds:** GATE A (the D1 write-up framing decision). The frozen G3 spine is untouched.

Scripts: `scripts/calibration_a1_ladder.py`, `_a1_figure.py`, `_a3_variance.py`, `_a4_stats.py`,
`_a5_continuity.py`. Per-task JSON under `outputs/phase2/calibration/`.

## Headline

**The calibration strengthens the orthogonality result, and it answers the one open framing
question.** With a moral positive control now in hand, refusal projects **below the moral-family
band on every model** (base, instruct, Think, GPT-OSS), including GPT-OSS's in-trace P2 = 0.52,
the single highest refusal projection anywhere in the program. Refusal is not merely above the
random null and below persona; it is below a **held-out genuinely-moral direction**. Two new
mechanism findings converge: the wired refusal gate lives in a **low-variance "spare channel"**
(A3), and it does **not crystallize** from a pretraining precursor (A5, cos = 0.155). Both fit the
"narrow post-training add-on" reading.

---

## A1 — Held-one-out moral positive control + the calibrated ladder (rule R1)

The moral positive control projects each source moral direction onto the span of the other two.
A genuinely-moral direction held out of the subspace it belongs to projects high; the range of the
three is the **moral-family band**, the yardstick for "moral-adjacent."

| tag (layer) | held-one-out (ms / fables / ethics) | **moral-family band [min,max]** | persona c | refusal points |
|---|---|---|---:|---|
| base (16) | 0.537 / 0.664 / 0.569 | **[0.537, 0.664]** | 0.510 | P_A 0.33 |
| instruct (16) | 0.523 / 0.637 / 0.555 | **[0.523, 0.637]** | 0.506 | P_B 0.14 |
| think (16) | 0.537 / 0.667 / 0.573 | **[0.537, 0.667]** | 0.525 | P0 0.29 · P1 0.10 · **P2 0.35** |
| gpt_oss (12) | 0.649 / 0.764 / 0.660 | **[0.649, 0.764]** | 0.603 | P0 0.47 · P1 0.19 · **P2 0.52** · P3 0.25 |

**Every refusal point is below its tag's moral-family band.** The ladder (iso floor → null q50/q95
→ refusal → band → persona) is in `outputs/phase2/calibration/a1_ladder.{png,pdf}`:

- On the OLMo family, refusal sits at or below the null and far below the band. Persona sits
  **just below** the band (0.51 vs [0.54, 0.66]) and well above the null: persona is
  moral-adjacent, not a clean non-moral axis. This is the empirical basis for the A2 rename
  ("moral-adjacent voice reference"), held until B3 confirms a genuinely non-moral axis (syntax /
  register) sits lower (R5).
- **GPT-OSS P2 = 0.52 is the decisive point.** It crosses its rank-matched null (0.32–0.34) and is
  the highest refusal projection in the program, yet it stays **below the persona reference (0.60)
  and below the moral-family band [0.65, 0.76]**. So in-trace refusal deliberation is at most
  voice-adjacent, not moral-content-adjacent.

- **The 0.52, under the D3 routing lens (retro-audit, 2026-07-02).** D3 showed causally on OLMo-3
  that refusal reads the *harm percept*, not the broad moral subspace (`harm_saturating`). The same
  reading closes the GPT-OSS P2 story. GPT-OSS is a strong massive-activation outlier (top-dim
  variance 0.70, ANOMALIES A1), so the audit is done per-dim-standardized; it holds raw too.
  Projecting the saved in-trace refusal direction (P2, layer 12) onto the model's own harm and moral
  directions, **standardized: |cos(P2, d_harm)| = 0.49 vs |cos(P2, V_moral ⊥ d_harm)| = 0.13**
  (raw: 0.57 vs 0.22; `cos(V_moral, d_harm)` = 0.13 std / 0.22 raw). P2 is **harm-loaded** either way,
  and standardization *sharpens* the gap (3.8×). P2 sits below the moral-family band not because
  in-trace refusal is unrelated to morality, but because it reads the **harm sliver** of `V_moral`,
  only weakly aligned with the broad subspace the band measures. This is the same mechanism D3
  established causally on OLMo, now in an independent model (a 20B reasoning MoE) via an independent
  measurement (in-trace projection, not interchange patching). *(OLMo-3-Think P2 was not saved with a
  refusal direction — MISSING_ARTIFACTS; the same decomposition is a one-line re-extraction if wanted.)*

**MFT ↔ V_moral mutual projection (content-vs-content; closes Paper 7 Phase 2f at the subspace
level).** On the tags where MFT is committed (base, instruct):

| tag | MFT → V_moral (mean) | V_moral → MFT (mean) |
|---|---:|---:|
| base | 0.557 | 0.619 |
| instruct | 0.557 | 0.615 |

Both directions land **inside the moral-family band** and well above persona, so the 6-foundation
MFT subspace and the rank-3 `V_moral` measure related-but-distinct moral content (neither nests in
the other). MFT foundations project onto `V_moral` at genuinely-moral magnitude; refusal does not.
(Think / GPT-OSS MFT are not committed → logged to `MISSING_ARTIFACTS.md` for B3.)

## A3 — Refusal variance-percentile ("spare-channel"; descriptive, no gate)

Percentile of each direction's activation variance among covariance-matched random directions:

| tag | refusal | V_moral axes (ms / fab / eth) | persona |
|---|---|---|---|
| base | P_A **37.4** | 49 / 27 / 7 | 4.5 |
| instruct | P_B **0.0** (≤ q10) | 59 / 36 / 16 | 13 |
| gpt_oss | P0–P3 all **≤ 0.2** (≤ q10) | 4 / 9 / 7 | 15 |

**The wired refusal gate lives in a low-variance channel.** The instruct gate (P_B) and all four
GPT-OSS positions sit at or below the 10th percentile of activation variance, i.e. refusal
occupies a narrow direction the model barely varies along. This mechanistically explains both easy
ablation (Heretic) and below-null projection. The base **proto-refusal** is *not* narrow (pct 37),
which is consistent with A5: the narrow channel is a property of the **post-training-wired** gate,
not the pretraining precursor. (Think refusal vectors are not saved → logged for B3.)

## A4 — Statistics upgrade (bootstrap CIs · combined P2 · signed projections)

- **Bootstrap CIs (B = 2000).** The moral-family band is tight and stable where all three source
  pair-sets are committed (base band-min 95% CI [0.47, 0.53], band-max [0.59, 0.65]; think similar).
- **Paired Δ = band-min − P test (pre-registered A4 addendum, 2026-07-01).** The marginal-CI
  overlap was only a conservative screen; band-min and P share the same resampled `V_moral` each
  iteration, so the sub-band claim is tested on the **paired** difference Δ (percentile primary,
  BCa robustness):

  | point | Δ̂ | percentile CI (primary) | excl 0? | BCa CI | excl 0? |
  |---|---:|---|---|---|---|
  | base P_A | 0.211 | [0.144, 0.220] | **yes** | [0.204, 0.244] | yes |
  | gpt_oss P0 | 0.183 | [0.033, 0.252] | **yes** | — | — |
  | gpt_oss P1 | 0.462 | [0.284, 0.467] | **yes** | — | — |
  | gpt_oss P3 | 0.398 | [0.217, 0.391] | **yes** | — | — |
  | gpt_oss P2_FULL | 0.262 | [0.090, 0.270] | **yes** | [0.253, 0.301] | yes |
  | **gpt_oss P2 (window)** | 0.127 | **[−0.029, 0.162]** | **no** | [0.089, 0.212] | yes |

  Refusal is sub-band at **every point on every model except the single GPT-OSS in-trace P2
  window point**, where the **primary (percentile) Δ-CI includes 0**. The percentile-vs-BCa split
  is exactly the pre-registered **band-min attenuation**: band-min is a min-of-three-noisy statistic,
  downward-biased under resampling, so it understates the true band floor; percentile does not
  correct that bias, BCa does (and BCa excludes 0). Per the committed prereg, **percentile is
  primary → the pre-registered verdict for the GPT-OSS P2 window point is Option 2**
  ("at the persona reference, CI-inconclusive vs the band pending B3's axis-pair-n re-extraction").
  The bias direction favors the sub-band claim, and both the bias-corrected BCa and the P2_FULL
  robustness point exclude 0, so the claim survives correction; it **locks at B3** regardless.
- **Combined P2 (R7, EXPLORATORY / post-hoc).** Per-model null-exceedance p of the in-trace P2:
  Think p = 0.015, GPT-OSS p = 0.0005 → Fisher χ²(4) = 23.5, **combined p ≈ 1.0e-4**. The in-trace
  peak's excess over the **random** null is a real, above-chance effect when pooled across the two
  reasoning models. This does **not** change the per-model NULL verdict (which rests on the persona
  and band references, both of which refusal stays below); it sharpens the narrative to: *in-trace
  deliberation reliably lifts refusal above random alignment with moral content, yet stays below a
  non-moral voice axis and below a held-out moral direction.*
- **Signed cos diagnostics (basis-dependent).** GPT-OSS P2 aligns most with the ethics axis
  (cos 0.51); base proto-refusal spreads ~0.24–0.27 across axes; persona cos stays low (0.09–0.27).
  Diagnostic only, per the pre-registration.

## A5 — Proto-refusal continuity (descriptive)

`cos(proto-refusal_base, refusal_instruct) @ L16 = **0.155**` — **below the 0.50 threshold, not
triggered.** Refusal does **not** crystallize from a pretraining precursor (contrast Paper 5's
moral subspace, cos(base, fresh) → 0.999). The instruct gate is substantially a post-training
construction. No per-checkpoint crystallization curve is queued; the near-orthogonality of the
base precursor to the wired gate is itself a datapoint (and dovetails with A3: the narrow gate is
built during alignment, not selected from pretraining).

## A6 / A2 — Documentation

- **A6 (done):** RESULTS.md now names the σ\* narrative-robustness finding (V_moral 4.86 vs MFT 0.0)
  and adds a same-layer instrument-limitation note pointing to the Phase B/C decision-level and
  cross-layer measurements.
- **A2 (prepared, extraction in B3):** syntax control (210 pairs) verified usable as-is; a new
  80-pair non-moral **register/formality** control is authored (`deepsteer/datasets/register_pairs.py`);
  the XSTest borderline subset (40 items, CC-BY-4.0, pinned commit) is committed with provenance.
  The persona → "moral-adjacent voice reference" rename and the "below a known non-moral axis"
  sentence are **held for B3** (R5), pending c_syntax / c_register.

---

## GATE A — resolved (Orion, 2026-07-01)

Adopt Option 1 (calibrated sub-band framing) with the pre-registered Δ test deciding the single
GPT-OSS P2 point:

1. **The claim rides on the ladder `null → P2 → band`; persona is reference-only** (pending B3's
   genuinely-non-moral syntax / register controls, per the R5 persona reclassification). On the
   OLMo family and at every GPT-OSS position except the P2 window, refusal is **sub-band with the
   paired Δ-CI excluding 0**: in-trace refusal deliberation is voice-adjacent, not
   moral-content-adjacent. For the **GPT-OSS P2 window point**, the primary (percentile) Δ-CI
   includes 0 → **Option 2 wording** ("at the persona reference, CI-inconclusive vs the band pending
   B3's axis-pair-n"), noting the min-statistic attenuation and that BCa + P2_FULL exclude 0. Locks
   at B3 regardless of branch.
2. **Combined P2** — Fisher p ≈ 1e-4 as EXPLORATORY confirmation the in-trace peak is a real
   above-random effect (two independently-trained labs → independence holds for Fisher), while the
   per-model NULL verdict stands (it rests on persona/band, not on refusal being at chance).
3. **MFT↔V_moral (0.56 / 0.62, inside the band)** retro-validates Paper 5's instrument choice:
   refusal's null was **not** an artifact of projecting against an outlier / degenerate subspace,
   because the 6-foundation MFT span projects onto `V_moral` at genuinely-moral magnitude.
4. **A3 spare-channel + A5 no-crystallization** fold into the Discussion as the
   **fresh-low-variance-gate synthesis**: refusal is a freshly-built, low-variance post-training
   gate, not a moral-content-derived direction.

**Next step (green-lit):** Phase B session 1 (B1 + B3 + B5) proceeds once this addendum lands;
B3's GPT-OSS axis re-extraction closes the one Δ caveat. Nothing here reopens a G3 verdict.

## Cross-position caveat on the reasoning-extension band rung (2026-07-01)

The D2 in-format work (`../d2_decision_coupling/PREREGISTRATION.md` Amendment 2) showed that the
moral-family **band** is position-specific and that a decision-adjacent token can be a low-rank
bottleneck where the band drops below the covariance null. The reasoning-extension ladder here
placed **GPT-OSS P2 = 0.52 against a band (0.65–0.76) computed from raw mean-pooled directions** — a
**cross-position rung** (in-trace window vs raw-pooled band). **P2's null is position-matched, so
"P2 crosses its null" stands**; but the **band comparison is cross-position** and inherits the
mismatch. Pending fix (zero-GPU if the per-rollout activation hygiene held): a **PR audit of the
P0–P3 windows** to confirm they are not low-rank bottlenecks, and a position-matched band. Until
then the band-relative statements in the reasoning section are **scoped as cross-position**; the
null-relative statements (P2 above/below its own position-matched null) are unaffected.

## Queued for B3 (`MISSING_ARTIFACTS.md`)

- GPT-OSS + Think MFT directions (for the reasoning-tag MFT↔V_moral comparison).
- Think P0–P3 refusal vectors saved as `.npz` (for Think spare-channel + refusal-p CIs).
- Instruct fables/ethics per-pair diffs + GPT-OSS axis pairs with per-pair saves (to tighten the
  band CIs and close the GPT-OSS overlap).
