# Direction 1 — Calibration Pre-registration (Phase A instrumentation)

**Date:** 2026-07-01 · **Against commit:** `5c542f6` (HEAD) · **GPU-free.**
**Status:** Pre-registered *before* any calibration headline quantity is computed. Nothing
here is a result. The held-one-out band, the MFT↔V_moral mutual projection, the refusal
variance-percentile, the bootstrap CIs, the combined P2 statistic, and the proto-refusal
continuity are all fixed as *procedures* here and computed only after this file is committed.

Companion documents: `PREREGISTRATION.md` (the frozen D1 spine), `GATE_HOOK_VERIFICATION.md`
(gate-critical hooks), and `../d2_decision_coupling/PREREGISTRATION.md` (the decision-coupling
experiments that share the R-table below).

> **The frozen D1 spine is untouched.** The G3 decision rule, its two-bar conjunction, the
> margin `M = 0.05`, and the two-step null protocol (`PREREGISTRATION.md` §3.3–§3.4) are not
> modified by anything here. This document adds *instrumentation* around the existing headline:
> a moral positive control, genuinely non-moral controls, and a statistics upgrade, so the D1
> projection instrument is calibrated before the write-up framing is decided at Gate A. The
> addition is also recorded as a dated amendment in the `PREREGISTRATION.md` trail.

---

## 0. What this document fixes now — and what it computes later

| Fixed NOW (pre-data, this commit) | Computed LATER (after this commit, at Phase A) |
|---|---|
| Held-one-out procedure + moral-family-band definition (§A1) | The three held-one-out values and the band `[min,max]` per tag |
| MFT↔V_moral mutual-projection procedure (§A1) | The two asymmetry numbers per model |
| Calibrated-ladder rung set + ordering (§A1) | The realized rung values and the figure |
| Refusal variance-percentile procedure (§A3) | The percentiles |
| Bootstrap-CI resampling recipe (§A4) | The CIs |
| Combined-P2 statistic + its EXPLORATORY label (§A4, R7) | The Fisher combination |
| Proto-refusal continuity threshold (§A5) | `cos(proto-refusal_base, refusal_instruct)` |
| Doc-fix *conditions* (§A2, §A6) | The doc edits, once their conditions are met |

**Two-step discipline, reused.** Every cutoff a reading is judged against (the moral-family
band, the null percentiles, the pairwise-null in D2) is a deterministic function of already-built
artifacts, computed under a recipe frozen here, *before* the quantity it judges is compared to it.
This is the same protocol that produced `q95` and `c` for G3.

---

## 1. Grounding anchors (committed prior results, cited as context — NOT calibration outputs)

These numbers are already committed (`outputs/phase2/*_g3*_result.json`, `RESULTS.md`) and
predate this calibration. They are the context the calibration recontextualizes; they are not
the calibration's own headline quantities.

| tag | layer | refusal point(s) | null q95 | persona c | iso floor `√(3/d)` |
|---|---:|---|---:|---:|---:|
| base | 16 | P_A (proto) 0.326 | 0.308 | 0.510 | 0.0271 |
| instruct | 16 | P_B (gate) 0.144; vs 6-MFT 0.155 | 0.266 (MFT 0.252) | 0.510 | 0.0271 |
| think | 16 | P0 0.291 · P1 0.101 · **P2 0.35** | 0.304 | 0.525 | 0.0271 |
| gpt_oss | 12 | P0 0.466 · P1 0.19 · **P2 0.52** · P3 0.25 | 0.336 | 0.603 | 0.0323 |

Content-dominance check (committed): base/think `top_dir_var_frac ≈ 0.074`, `effdim@0.90 = 385`;
gpt_oss `0.226`, `229`. All content-dominated, so the rank-3 source-direction construction is
the confirmed method on every tag.

**The open framing question this phase answers.** GPT-OSS P2 = 0.52 crosses its null (0.336)
while staying below persona (0.603). Is 0.52 "high" in an absolute sense, or is it inside the
range a *known-moral* direction projects at on this same subspace? Without a moral positive
control the ladder has a floor (null, iso) and a non-moral reference (persona) but no moral
upper reference. §A1 supplies it.

---

## A1 — Held-one-out moral positive control + content-vs-content (R1)

### Procedure (fixed now)

For each tag, at its headline layer (base/instruct/think = 16, gpt_oss = 12), the three source
moral directions are `d_moral` (Moral Stories, from `<tag>/moral_directions.npz`), `d_fables`
and `d_ethics` (from `<tag_axis>/axis_directions.npz`). Each is unit-normalized
(`extraction.unit_vector`).

- **Held-one-out projection.** For each source `s`, orthonormalize the *other two* directions
  with `_ortho` (QR, as in `phase2_g3_respec.py`) into a rank-2 basis `Q_{-s}`, and report
  `p(d_s | span{other two}) = ‖Q_{-s}ᵀ d_s‖ / ‖d_s‖` (the same projection-fraction convention
  as G3's `_frac`). This is a positive control: a genuinely-moral direction held out of the
  subspace it belongs to should project *high*, giving the yardstick refusal is measured against.
- **Moral-family band (R1).** `band = [min, max]` of the three held-one-out values, per tag.
  This band is the pre-registered yardstick for every "moral-adjacent" statement in the D1 and
  Paper-7 write-ups. It is computed once here and not tuned afterward.
- **MFT↔V_moral mutual projection (content-vs-content; closes Paper 7 Phase 2f at the subspace
  level).** For tags where the 6 MFT foundation directions are committed (`<tag>/mft_directions.npz`
  exists on **base** and **instruct**; think/gpt_oss MFT are logged to `MISSING_ARTIFACTS.md` for
  B3 if wanted): project each of the 6 unit MFT foundation directions onto the rank-3 orthonormal
  `V_moral` basis, and project each of the 3 unit `V_moral` source directions onto the rank-≤6
  orthonormal MFT span. Report **both directions of the asymmetry** (mean and per-direction).
  Pre-registered reading: the two subspaces measure related but distinct moral content, so mutual
  projections should sit **above** the persona reference and **below** 1; a near-1 value in either
  direction would mean one subspace nests inside the other.

### Calibrated ladder (fixed now: the rungs and their order)

One table + one figure per tag, rungs in this fixed order (low to high):

1. isotropic floor `√(3/d)` (committed: 0.0271 / 0.0323);
2. covariance-matched null **q50** then **q95** (recomputed here on the rank-3 span, frozen recipe);
3. refusal `p` at each *committed* measured point (gate/P0/P1/P2/P3 as available for the tag);
4. **moral-family band `[min,max]`** (this task's positive control);
5. persona reference `c`.

The figure places refusal's measured points against the moral band and the persona reference on
one axis, so "refusal projects like a random direction / like a non-moral direction / like a moral
direction" is read off directly. Pre-registered reading for the GPT-OSS framing decision at
Gate A: if GPT-OSS P2 = 0.52 sits **below** the GPT-OSS moral-family band, then "P2 crosses the
null but is still less moral-adjacent than a held-out moral direction" is supported; if P2 lands
**inside** the band, the reasoning-extension narrative is revised toward "in-trace deliberation
projects at genuinely moral-family magnitude." Both are pre-declared publishable; the choice is
mechanical from where 0.52 falls relative to the band.

**Output:** `outputs/phase2/calibration/a1_ladder_<tag>.json` per tag (all rungs + band +
mutual-projection numbers) and `outputs/phase2/calibration/a1_ladder.png` (or per-tag panels).

---

## A3 — Refusal variance-percentile ("spare-channel" analysis; descriptive, no gate)

Using each tag's `act_sample.npz` (`X`, shape `(n, d)`, `n < d`), stay inside the
**sample-covariance** machinery (do not eigendecompose a rank-deficient `Σ̂` naively): draw `K`
covariance-matched random directions as `Xcᵀ z` for `z ~ N(0, I_n)` (the same generator the null
uses), and for each measure its variance `‖Xc · r̂‖² / (n-1)` where `r̂` is the unit direction.
Report the **percentile** of the refusal direction's variance `r̂ᵀ Σ̂ r̂` within that sampled
distribution, and the same percentile for each `V_moral` axis and persona as references.

Pre-registered reading (descriptive only, no stop/go): refusal in a **low-variance channel
(≤ q10)** is consistent with a "narrow add-on" implementation and would mechanistically explain
both its easy ablation and its below-null projection. This is a mechanism annotation for the
Discussion, not a gate.

**Output:** `outputs/phase2/calibration/a3_variance_percentile_<tag>.json`.

---

## A4 — Statistics upgrade (bootstrap CIs; combined P2 EXPLORATORY; signed projections)

- **Bootstrap CIs (fixed recipe).** For every D1 projection where the per-pair difference
  arrays exist (`diffs_moral_stories.npz`, `axis_diffs_*.npz`), resample pairs with replacement
  (`B = 2000`, seed 0), re-extract the direction (mean-diff → unit) on each resample, recompute
  the projection, and report the 2.5/97.5 percentile CI. Projections whose per-pair arrays are
  **not** committed (e.g. refusal per-prompt activations, persona per-pair) are logged to
  `outputs/MISSING_ARTIFACTS.md` for B3 re-extraction *with per-pair saves* — never silently
  regenerated here.
- **Cross-model combined P2 (R7, EXPLORATORY / post-hoc).** Per-model bootstrap p-value of P2
  vs that model's covariance-matched null, combined via **Fisher's method** across OLMo-3-Think
  and GPT-OSS. This is labeled EXPLORATORY: the **per-model verdicts remain the pre-registered
  rule** (R2/G3); the combined statistic is reported as a post-hoc aggregate only, never as a
  gate.
- **Signed projections.** Where the sign is defined (a single direction onto a single direction,
  e.g. per-axis diagnostics, cos), report the signed value alongside the magnitude. Subspace
  projection fractions remain magnitude-like (no sign claimed), consistent with `RESULTS.md`.

**Output:** `outputs/phase2/calibration/a4_bootstrap_<tag>.json`, `a4_combined_p2.json`.

---

## A5 — Proto-refusal continuity (persona/refusal-selection thread; first datapoint)

`cos(proto-refusal_base, refusal_instruct)` at layer 16, from the committed
`refusal_base.npz` (Point A, base proto-refusal) and `refusal_instruct.npz` (Point B, instruct
gate) vectors (same architecture, same layer). This is the refusal analog of Paper 5's
`cos(base, fresh)` crystallization measurement.

**Pre-registered threshold + trigger (fixed now):** if `cos ≥ 0.50`, the per-checkpoint
refusal-crystallization curve is **queued into B3** (stage-3 anneal checkpoints from Paper 5's
grid). Pre-registered reading: high continuity = post-training *selects* a pre-existing
pretraining direction, the measurable form of "persona/refusal selection reaches back into
pretraining." Below 0.50 = the instruct gate is substantially a post-training construction; no
curve queued. Descriptive; no headline gate.

**Output:** `outputs/phase2/calibration/a5_proto_refusal_continuity.json`.

---

## A2 / A6 — Documentation-fix conditions (no science; edits gated on named conditions)

Pre-registered so the edits are criterion-driven, not free-hand:

- **A2 persona rename.** Rename the persona axis in the D1 docs from "non-moral semantic control"
  to "**moral-adjacent voice reference**" and fix the `RESULTS.md` inconsistency (the 0.076–0.085
  MFT-era |cos| citation sitting beside `c = 0.51–0.65` on `V_moral`). This edit is **staged now
  and applied after B3**, when `c_syntax` / `c_register` exist, because the rename claim ("persona
  is moral-adjacent, not a clean non-moral control") is only supported once genuinely non-moral
  axes are measured beside it. **Hold the "below a known non-moral axis" sentence until B3 (R5).**
  The G3 verdicts do not change either way.
- **A6 σ\* promotion.** Promote the σ\* narrative-transfer result (V_moral 4.86 vs MFT **0.0** on
  narrative) from a parity footnote to a **named finding** in the D1 draft. This is a
  re-presentation of a committed number, no new computation.
- **A6 limitations note.** Add to the D1 draft: Paper 1's storage/usage layer-divergence applies
  reflexively — a **same-layer** projection cannot detect a refusal circuit that reads moral
  features **through weights** from other layers. Phase B (decision directions) and Phase C
  (the cross-layer Jacobian read test) measure that directly. This note is safe to add now
  (it is a scope statement, not a result).

---

## Pre-registered decision rules (the shared R-table; ownership annotated)

`M = 0.05` throughout. Both branches of every rule are pre-declared publishable.

| # | Owner | Quantity | Rule | Branches |
|---|---|---|---|---|
| **R1** | **calib (A1)** | Moral-family band per tag | `[min,max]` of held-one-out projections | the band is the yardstick for all "moral-adjacent" language |
| R2 | D2 (B1) | Judgment-decision on V_moral | clears q95 + M? | sensitivity-confirmed ↔ category-mismatch (reframe, not retract) |
| R3 | D2 (B1) | cos(refusal, judgment-decision) | > pairwise-null q95 + M? | decision-level coupling ↔ decision-level dissociation |
| R4 | D2 (B2) | Outcome-conditioned P2 | ≥ band-min − M / ≤ null + M | in-trace decision moral-adjacent ↔ gradient was comprehension |
| **R5** | **calib→B3** | Strong-form orthogonality sentence | refusal ≤ min(c_syntax, c_register) + M | keep sentence ↔ hold sentence; persona reclassified either way |
| **R6** | **calib→B3** | Paper 5 F2 "specifically" | moral rotation − control rotation ≥ 15° | keep ↔ re-word as shared shift |
| **R7** | **calib (A4)** | Combined P2 (exploratory) | Fisher across Think + GPT-OSS | reported as post-hoc aggregate only |
| R8 | D2 (B5) | Moral fragility of refusal | σ\*_moral outlier below the random-floor distribution? | differential sensitivity ↔ flat baseline |

R2–R4, R8 are governed by `../d2_decision_coupling/PREREGISTRATION.md`; they appear here so the
calibration reads against the same frozen table.

---

## Integrity statement

No calibration headline quantity (R1, R5, R6, R7, the variance percentiles, the bootstrap CIs,
or the proto-refusal continuity) was computed before this file was committed. Any later change
to a threshold, a procedure, or a doc-fix condition must be a dated amendment below this line,
never a silent edit.

### Amendments

- **2026-07-01 (A4 addendum — Δ = band-min − P2 paired-bootstrap test, pre-registered before it is
  computed):** The A4 marginal-CI overlap (GPT-OSS band-min CI vs P2 CI) was a **conservative
  screen, not the test**: it ignores that band-min and P2 are computed from the **same resampled
  `V_moral`** every bootstrap iteration and are therefore **positively correlated**, so comparing
  their marginal CIs overstates the uncertainty of their *difference*. The pre-registered test for
  the GPT-OSS sub-band claim is the **paired bootstrap CI of `Δ = band-min − P2`**, from the
  existing A4 draws (B = 2000, seed 0): per iteration `i`, `Δ_i = band_min_i − P2_i` (paired, so the
  covariance is captured). Report the **95% percentile CI** (primary) and the **BCa CI**
  (robustness: bias-correction `z0` + jackknife-over-pairs acceleration). Point estimate
  `Δ̂ = full-sample band-min − full-sample P2`.

  **Branches (both pre-committed, before Δ is computed):**
  - **Δ CI excludes 0** (band-min > P2 at 95%) → **adopt the sub-band framing (Option 1)**: refusal
    P2 projects below the moral-family band. Carry a one-sentence **band-min attenuation note**:
    band-min is a min-of-three-noisy-held-one-out statistic, downward-biased by the min operation +
    direction-resampling, so it **understates** the true band floor; the reported sub-band gap is a
    **lower bound**, and the bias direction therefore **favors** the sub-band claim.
  - **Δ CI includes 0** → **Option 2 wording** for GPT-OSS: *"at the persona reference,
    CI-inconclusive vs the band pending B3's axis-pair-n re-extraction."*

  The framing sentence **carries the claim on the ladder `null → P2 → band`**; **persona is shown
  for reference only** (pending B3's genuinely-non-moral syntax / register controls, per the persona
  reclassification, R5). This addendum adds a sharper test on the *existing* draws; it changes no
  gate, no G3 verdict, and no R-rule. GPT-OSS wording locks for real after B3 tightens the axis-pair
  n, regardless of which branch fires here.
