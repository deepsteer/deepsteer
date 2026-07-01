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
  **GPT-OSS carries the one real caveat:** with small axis-pair n (fables 62, ethics 118), its
  band-min CI is wide and low ([0.47, 0.64]) and **overlaps the P2 CI [0.44, 0.53]**. So "P2 below
  the moral band" holds at the **point estimate** (0.52 < 0.65) but is **not separated at 95%
  confidence for GPT-OSS**; on OLMo the separation is clean. Fix queued: re-extract the GPT-OSS
  axis pairs with more items + per-pair saves in B3.
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

## GATE A — framing decision (for human review)

The calibration is clean and it favors the orthogonality headline. Recommended framing:

1. **GPT-OSS P2 = 0.52 against the band.** Characterize it as: *"above the random null and near the
   non-moral persona reference, but below both persona and the moral-family band — in-trace refusal
   deliberation is voice-adjacent, not moral-content-adjacent."* State the honest CI caveat: the
   point estimate is below the band on every model; on GPT-OSS the small axis-pair n leaves the
   band-min CI overlapping P2's CI, so the sub-band claim is point-estimate-clean and
   CI-clean on OLMo but CI-overlapping on GPT-OSS (re-extraction queued for B3).
2. **Combined P2.** Report the Fisher combined p ≈ 1e-4 as an EXPLORATORY confirmation that the
   in-trace gradient is a real above-chance effect, while keeping the per-model NULL verdict (the
   orthogonality claim rests on persona/band, not on refusal being at chance). This makes the
   reasoning-extension narrative *stronger and more precise*, not weaker: the peak is real, and it
   is still sub-moral-adjacent.
3. **Two new mechanism findings** (A3 spare-channel, A5 no-crystallization) are ready to fold into
   the Discussion as convergent support for "refusal is a narrow post-training add-on."

**Open question for the gate:** whether to publish the D1 write-up now with this calibration folded
in (recommended), and whether to prioritize B3's GPT-OSS axis re-extraction (to close the one CI
caveat) as part of the first Phase-B session. Nothing here reopens a G3 verdict.

## Queued for B3 (`MISSING_ARTIFACTS.md`)

- GPT-OSS + Think MFT directions (for the reasoning-tag MFT↔V_moral comparison).
- Think P0–P3 refusal vectors saved as `.npz` (for Think spare-channel + refusal-p CIs).
- Instruct fables/ethics per-pair diffs + GPT-OSS axis pairs with per-pair saves (to tighten the
  band CIs and close the GPT-OSS overlap).
