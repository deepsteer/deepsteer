# Direction 1 — Pre-registration of Gates G2 and G3

**Date:** 2026-06-26 · **Against commit:** `107f9f3` (HEAD) · **GPU-free.**
**Status:** Pre-registered before any `V_moral` construction or refusal measurement.
Nothing here is a result. Every threshold below is fixed *now*, in advance of the
data, so that stop/go is criterion-driven rather than a judgment call mid-run.

Companion: `GATE_HOOK_VERIFICATION.md` (the three gate-critical hooks were verified
to behave as assumed before pre-registering against them).

> **Paper number is provisional.** This work lives in `papers/d1_moral_subspace/`
> and is *not* claiming the "Paper 8" slot; final numbering is the program owner's call
> (plan §11).

---

## 0. What this document fixes now — and what it deliberately defers

| Fixed NOW (pre-data, this commit) | Computed LATER, before the refusal projection |
|---|---|
| G2 floor + gap tolerance (§2) | — (G2 needs no data-derived cutoff) |
| G3 decision rule + margin `M` (§3.4) | Realized null cutoff `q95` and control value `c` (§3.3) |
| Null **construction procedure** (§3.3) | The realized null *distribution* from `V_moral` |
| eff-dim convention = uncentered @ 0.9 (§5) | `V_moral`'s realized eff-dim |
| Track 1 σ* criterion form + RMS rule (§4) | The numeric σ* tolerance (two-step, like G3) |

**The two-step null protocol (the load-bearing discipline).** `q95` and `c` cannot be
fixed now: they depend on `V_moral`'s geometry, which does not yet exist. They are
**computed at Phase 2, from the constructed `V_moral`, *before* the refusal direction is
ever projected onto it.** The construction procedure that produces them is frozen here
(§3.3), so the realized cutoffs are mechanical, not chosen. This guarantees the cutoffs
predate the measurement and cannot be tuned to the answer.

---

## 1. Grounding anchors (all commit-pinned to `107f9f3`)

- **Paper 5 MFT baseline (the number G3 must beat or match):** the refusal direction
  projects **0.1044** onto the 6-MFT subspace at **layer 16**; mean|cos| to the six
  foundations **0.0606**.
  Source: `papers/5_moral_alignment/outputs/heretic/refusal_morality_geometry.json`.
- **Non-moral semantic negative control (G3 control c):** the persona / assistant axis,
  known orthogonal to morality at |cos| ≈ **0.075** (range 0.076–0.085).
  Extractable per-model via `direction_utils` on `deepsteer/datasets/persona_pairs.py`
  (artifacts already exist, e.g. `papers/6_cross_model/outputs/olmo3_base/persona_directions.npz`).
- **Paper 1 fragility framework (G2 + Track 1 anchor):** fragility threshold **τ = 0.6**
  (= chance 0.5 + 0.1); peak probe accuracy 0.74–0.96 by construction; **structural
  transfer 0.848 vs in-distribution 0.858 (≈1 pp loss)**; **lexical-lookup collapse to the
  bag-of-words floor 0.598 (≈25 pp loss)**; σ-grid `S` with `max(S) = 10.0`.
  Source: `papers/1_accuracy_vs_fragility/sections/01_introduction.md`, `04_results.md`.
- **Stable band (OLMo-3 7B):** layers **15–31** (project memory; used for robustness reporting).
- **Construct framing — moral salience.** Every source uses a *salience contrast*:
  moral-valence-present vs valence-stripped (neutral) retelling/action, holding scenario,
  participants, and discourse type constant. `V_moral` is therefore a moral-**salience**
  subspace, and the 0.1044 Paper 5 number is correspondingly the **salience baseline**
  (refusal's projection onto that subspace). The per-source construction that realizes the
  salience contrast — including the MORABLES fable-internal derivation and its abstraction
  control — is specified in `CONSTRUCTION_GUIDELINES.md`.

---

## 2. GATE G2 — contamination / paraphrase gap (HARD gate)

**Question.** Does `V_moral` read *moral structure* or *memorized benchmark surface text*?
All three primary sources (MORABLES, Moral Stories, ETHICS) are 2020–2025 and near-certain
to be in OLMo-3's pretraining; the held-out paraphrase set (1:1 with each moral judgment,
moral content preserved, surface form broken) is the only defense.

**Operationalization (grounded in Paper 1's transfer/lift logic).** A probe reading
*memorized surface* behaves like Paper 1's bag-of-words classifier — it collapses on
broken surface form (transfer → 0.598). A probe reading *moral structure* transfers with
~1 pp loss (0.858 → 0.848). G2 places its cut between those two empirically-characterized
regimes.

**Accuracy metric.** `acc_surf` and `acc_para` are
`direction_utils.transfer_metrics(X, y, d_moral).acc_midpoint`, where `d_moral` is the
**primary mean-diff direction** of `V_moral` (mean-diff is the Phase-2 primary extractor),
evaluated on the held-out test pairs (`acc_surf`) and on their 1:1 paraphrases
(`acc_para`). The probe-weight direction is reported alongside as a diagnostic, not as the
gate.

### G2 decision rule (fixed now)

> **G2 PASSES iff BOTH hold:**
> 1. **`acc_para ≥ 0.60`** — the paraphrase probe is still a *real signal* by Paper 1's
>    own τ. Below 0.60, by the project's existing standard, there is no signal, so the
>    original-surface accuracy was reading memorized text.
> 2. **`acc_surf − acc_para ≤ 0.10`** — the surface→paraphrase drop stays inside the
>    structural-reading regime. Paper 1's genuine structural transfer lost ≈1 pp; lexical
>    lookup lost ≈25 pp. A 0.10 tolerance is ~10× the genuine structural loss (so it does
>    not false-STOP a good dataset, given paraphrasing is harder than leave-construction-out
>    transfer) yet ~2.5× below the lexical-collapse gap, and it is expressed in Paper 1's
>    own "chance + 0.1" margin unit.
>
> **If either fails → STOP and fix curation.** G2 failure blocks Tracks 3–4, Phase 3.5,
> Phase 4, and Direction 2 (per plan §6, §9). No downstream number may be reported on a
> subspace that fails G2.
>
> **Slice the STOP gates on (2026-06-27 amendment):** the hard STOP is computed on the **106
> narrative** in-dist eval pairs only; the 28 declarative paraphrase gaps are **reported as
> informative, not gated** (the 0.10 threshold is narrative-calibrated and the declarative
> slice is thin + register-asymmetric). All 134 are paraphrased.

**Reporting (not part of the pass/fail, but committed):** per-register breakdown
(declarative / narrative / dialogue) of both accuracies and the gap; `auc_abs` alongside
`acc_midpoint`; the Social-Chemistry-101 OOD set as an external generalization probe (held
out, never used to extract directions).

---

## 3. GATE G3 — refusal overlap vs. rank-matched null and semantic control

### 3.1 Measurement (fixed now)

For each refusal point, `p = heretic_ablation.subspace_projection_fraction(refusal, B)`,
where `B` is `V_moral`'s orthonormal basis and `refusal = unit(last_token_means(harmful) −
last_token_means(harmless))` at **layer 16** (verified single-vector-per-layer; see
hook-verification). Two points, per the plan's robustness-across-operationalizations aim:

- **Point A — Paper-5 proto-refusal contrast.** Same refusal object as the committed
  0.1044 result, against the *richer* subspace, so any movement is attributable to the
  subspace and not to swapping the refusal direction.
- **Point B — OLMo-3 aligned-stage refusal gate.** The actual instruct-time refusal
  direction (a different, more "real" object).

Both reported at layer 16 and across the stable band (15–31).

### 3.2 Controls (fixed now)

- **(a) MFT baseline** — `0.1044` (§1), the known low number.
- **(b) Rank-matched null** — `q95`, the 95th percentile of the null projection
  distribution at `V_moral`'s realized eff-dim (construction in §3.3).
- **(c) Non-moral semantic control** — `c`, the persona/assistant axis (§1) projected onto
  `V_moral`, testing whether *any* meaningful direction projects high (i.e. whether
  `V_moral` merely captures general semantic structure).

### 3.3 Two-step null protocol (construction fixed now; cutoffs computed at Phase 2)

Frozen construction, executed at Phase 2 **before** any refusal vector is projected:

1. **Primary (activation-covariance-matched) null.** Draw `K = 1000` random directions
   from a Gaussian with the empirical covariance of the model's residual activations at
   layer 16 (the anisotropic space refusal actually lives in), unit-normalize each, and
   compute `subspace_projection_fraction(·, B)` for each. `q95` = the 95th percentile of
   that distribution. This is the honest null: it asks whether refusal projects higher
   than a *typical activation-space direction* of the same rank, not a typical isotropic
   direction.
2. **Isotropic analytic reference (reported alongside).** For an isotropic unit vector in
   `R^d` projected onto a rank-`r` subspace, `E[fraction] ≈ sqrt(r/d)`; report this as the
   analytic floor next to the covariance-matched `q95`.
3. **Control `c`** computed at the same step, from the same already-built `V_moral`.

Because steps 1–3 run before the refusal projection and follow a frozen recipe, `q95` and
`c` are mechanical, not chosen.

### 3.4 Decision rule + margin (fixed now)

> Let `M = 0.05` (the pre-registered margin, in the same units as the 0.1044 baseline —
> half the entire Paper 5 number, so "clears the bar" means a substantive, not marginal,
> effect).
>
> **G3 is POSITIVE iff, for BOTH Point A and Point B:**
> `p > q95 + M` **AND** `p > c + M`.
>
> **Otherwise G3 is NULL** (orthogonality robust across operationalizations — the more
> publishable strengthening of Papers 5–7).
>
> **Split result (A and B disagree):** treated as **NULL** for the D2 disposition, and
> **flagged for investigation, not retraction** (claim discipline from Papers 5–7; plan §8).

Rationale for the conjunction: a bare `p > q95` fires ≈5% by chance; requiring a margin
`M` above *both* the rank-matched null (rules out rank inflation) *and* the semantic
control (rules out "any meaningful direction projects high"), for *both* refusal
operationalizations, makes a false-positive G3 very unlikely. Genuine coupling, if real,
should push `p` well above 0.10 toward 0.2–0.3 and clear both bars comfortably; `M = 0.05`
detects that while rejecting rank inflation.

### 3.5 D2 disposition (mechanical from the G3 outcome)

- **G3 NULL →** `V_moral` and the 6-probe MFT subspace **coexist**; headline = "orthogonality
  holds across both operationalizations"; Phase 3.5 runs the **subsample** labeling
  (positioning only).
- **G3 POSITIVE →** `V_moral` **replaces** MFT as the standard instrument; Papers 5–7 get a
  revision note on the orthogonality claim; Phase 3.5 runs the **full** labeling pass
  (load-bearing overlap localization). The null is load-bearing here: the positive reading
  may not be claimed without having cleared (b) and (c).

---

## 3A. GATE G-AXIS — two-source axis agreement (MORABLES pooling; fixed now)

The training composition is **two-source by construction: Moral Stories + MORABLES**
(ETHICS contributes zero to the direction — see the 2026-06-26 amendment). **Moral Stories
is the reference axis.** MORABLES is the construct anchor, but its material is fables — a
discourse type distinct from the contemporary action-contrast register. To keep it in the
direction-extraction path **without** letting a fable-specific axis distort `V_moral`, its
inclusion is gated on agreement with Moral Stories. This runs at Phase 2, **before**
`V_moral` is finalized (so it is upstream of the eff-dim, null, and G3 measurements).

**Measurement (Phase 2, GPU).** Extract two per-source mean-diff **salience** directions on
OLMo-3 Base, each from that source's TRAIN-source pairs only:

- `d_MORABLES` — fable-internal salience pairs (moral-laden event-retelling vs neutral
  event-retelling of the *same* fable event; both concrete; see `CONSTRUCTION_GUIDELINES.md`).
- `d_MoralStories` — situation-held salience pairs (moral action vs valence-stripped action,
  same situation).

Compute `cos(d_MORABLES, d_MoralStories)` at the matched layer (16) and the stable-band
mean (15–31).

**Floor (fixed now): 0.67** — the lower end of the v2 cross-method agreement range
(0.67–0.71, where mean-diff / LEACE / probe-weight agreed). Reported against this floor;
the realized cosine is computed at Phase 2 before any pooling (two-step discipline, as G3).

> **G-AXIS decision rule (let the cosine decide, not a judgment call):**
> - `cos ≥ 0.67` → **PASS**: fables read the **same** moral-salience axis as contemporary
>   action-contrasts. MORABLES **POOLS** into `V_moral` (construct-anchoring inside the
>   subspace, as intended).
> - `cos < 0.67` → **FAIL**: fables read on an axis **distinguishable from contemporary
>   scenarios specifically**. MORABLES is **EXCLUDED** from `V_moral` and retained as a
>   construct-anchor **evaluation only** (Track 1 probe-accuracy / σ*). Reported as a
>   **register finding, not a failure** — *"fable moral salience is distinguishable from
>   contemporary action-contrast salience (cos = X < 0.67)."*

This is a **two-source** agreement check against the Moral-Stories reference axis — there is
no three-source consensus to invoke (ETHICS is zero-in-training). `V_moral` is therefore
either Moral Stories alone (G-AXIS fail) or Moral Stories + MORABLES pooled (G-AXIS pass);
that composition is fixed by G-AXIS before eff-dim, the null, and G3 are computed.

---

## 4. Track 1 — σ* fragility acceptance (G5 input; form fixed now, constant deferred)

Track 1 feeds the G5 conjunction ("`V_moral` no more fragile than the MFT baseline").
Fixed now:

- **Metric:** σ* per Paper 1 — smallest noise scale at which transfer accuracy under
  `N(0, σ²I)` perturbation drops below **τ = 0.6**, on the σ-grid `S` with `max(S) = 10.0`.
- **RMS normalization is mandatory** for the headline `V_moral`-vs-MFT comparison (project
  memory: raw σ* is activation-scale-confounded; raw σ* valid only within a fixed layer).
  Report raw σ* alongside, labeled within-layer-only.
- **Criterion form:** `V_moral` passes Track 1 iff `σ*_RMS(V_moral) ≥ σ*_RMS(MFT) − δ` at
  the matched layer (and stable-band mean). A `V_moral` that is *more* fragile than the
  six-probe subspace is a warning sign to understand before proceeding (plan §6).

**Deferred (same two-step discipline as G3):** the numeric tolerance `δ` is fixed when the
σ-grid `S` spacing and the MFT-subspace σ*_RMS baseline are measured, before `V_moral`'s
σ* is compared against them.

---

## 5. eff-dim convention (the G3-null denominator)

> **SUPERSEDED 2026-06-28 (finding-driven correction) — see the V_moral-construction amendment
> below.** The original spec (uncentered eff-dim @ 0.90 of pooled per-pair diffs) was written
> before the realized spectrum was known. The real-run spectrum has **no elbow** and is
> **content-dominated** (singvals 31, 18, 15, 14, … flat; d_moral = top direction but only 7.5%
> of variance), so eff-dim@0.90 = **385** measures CONTENT rank, not moral rank; the rank-3
> moral structure lives only in the source mean-diff directions. `V_moral` is therefore
> re-spec'd to the **span of the source moral directions**.

- ~~**`V_moral` rank = uncentered effective dimensionality at variance-threshold 0.90**, from
  the SVD of the stacked per-pair difference vectors (`pos − neg`).~~ (superseded; see above.)
- **`V_moral`'s basis is orthonormal** (the top-`r` SVD left singular vectors), so the
  projection fraction is exactly the in-subspace norm.
- **`direction_utils.effective_dimensionality` must NOT be called as-is** for this rank; an
  uncentered variant is used. (Recorded in `GATE_HOOK_VERIFICATION.md`, Hook 2.)
- **eff-dim is reported explicitly** with every projection number, because it is the
  denominator the rank-matched null depends on (plan rank-discipline note). "Refusal
  projects at X" is never a claim; "refusal projects higher than the rank-matched null at
  this eff-dim" is.

---

## 6. Pre-registration integrity — residual biases and their controls

| Residual risk | Control (pre-committed above) |
|---|---|
| Projection fraction rises mechanically with rank | Every `p` interpreted only vs. rank-matched `q95` at the reported eff-dim (§3, §5) |
| Null tuned to the result | Two-step protocol: `q95`, `c` computed from `V_moral` before any refusal projection (§3.3) |
| "Any meaningful direction projects high" | Second control `c` (persona axis) with the same margin `M` (§3.2, §3.4) |
| Single-operationalization fluke | Both Point A and Point B required positive; split → NULL + flag (§3.4) |
| Memorized benchmark text masquerading as moral structure | G2 hard gate on the paraphrase set, blocks all downstream (§2) |
| Accidentally-moral neutrals inflating apparent signal | Phase-1 audit (mechanical `validate_pairs` + rebuilt LLM-scored §1.1/§1.2/§1.5 gates; see hook-verification) |
| eff-dim convention drift (centered vs. uncentered) | Pinned uncentered @ 0.90; centered function explicitly excluded (§5) |
| Activation-scale confound in σ* | RMS-normalized σ* mandatory for the cross-subspace comparison (§4) |
| Fable-specific axis distorting `V_moral` | GATE G-AXIS: MORABLES pools only if `cos(d_MORABLES, d_MoralStories) ≥ 0.67`; else eval-only (§3A) |
| Genre/abstraction shortcut in fable pairs | Abstraction/genre-match control: both sides concrete event-retellings (`CONSTRUCTION_GUIDELINES.md`); audit gate `g_abstraction_match` |
| Anti-triviality LLM-score threshold uncalibrated | First-batch threshold pass: confirm score≤3 discriminates trivial pairs before scaling (`audit_runner.py --fail-at`) |

---

*This pre-registration is frozen at commit `107f9f3`. Any later change to a threshold,
the null construction, or a decision rule must be a dated amendment recorded below this
line, never a silent edit.*

### Amendments

> **How to read this trail.** Each amendment is anchored to the data property, measured
> yield, or external constraint (license, register, spectrum) that forced it, stated before
> the consequence that followed. The load-bearing spine — the **G3 decision rule, its
> two-bar conjunction, the margin `M = 0.05`, and the two-step null protocol** (§0, §3.3,
> §3.4) — is **unchanged by every amendment below**; NULL was the pre-declared more-publishable
> outcome throughout. What moved was dataset composition and the `V_moral` rank/construction,
> each time toward a harder or more faithful test, never toward the answer.

- **2026-06-26 (commit `8710641`+):** Added GATE G-AXIS (§3A, MORABLES cross-source axis
  agreement, floor 0.67) and the moral-salience construct framing (§1). MORABLES moves from
  a stated-moral aphorism contrast to a fable-internal salience contrast; its pooling into
  `V_moral` is now cosine-gated. No change to G2, G3, or their thresholds.
- **2026-06-26 (commit `1317307`+):** ETHICS register finding. First-batch calibration shows
  ETHICS §1.5 (accidentally-moral) does not clear (≈0.40 fail) because its scenarios are
  moral-judgment-native (the care/interpersonal frame carries residual valence a same-frame
  edit can't strip); §1.2 stays perfect. Per the pre-registered fallback, ETHICS is
  re-weighted toward its **hard-ambiguity-span role** (Track 1) with **reduced** bulk
  contribution to the pooled `V_moral` direction, and its clean subset carries a
  **selection-bias flag** (filtering keeps non-care-framed items). See
  `CONSTRUCTION_GUIDELINES.md` (ETHICS section). No change to G2, G3, G-AXIS thresholds.
- **2026-06-26 (zero-ETHICS decision):** ETHICS contributes **zero** to the training
  direction. Two consequences. (1) **G-AXIS is now a two-source check** (Moral Stories
  reference + MORABLES); three-source-consensus language removed (§3A). (2) **ETHICS's eval
  role upgrades from in-distribution check to generalization probe:** its bias-flagged eval
  pairs (non-care-frame slice) test whether a Moral-Stories+MORABLES `V_moral` generalizes
  to ETHICS's abstract-judgment register — a register **absent from training** — alongside
  Social Chemistry 101's deferred OOD role. Training set is two-source by construction. No
  change to G2, G3, G-AXIS thresholds.
- **2026-06-26 (eval structure — two non-pooled tests):** The eval splits into two tests
  with opposite balance needs. (1) **`eval_g2_indist`** — source-balanced in-distribution
  (MORABLES 53 + Moral Stories 53 = 106). **GATE G2 reads ONLY this set, and only it is
  paraphrased;** balancing keeps the paraphrase gap from being a source-shift artifact. (2)
  **`eval_generalization_probe`** — all-clean ETHICS (118), **never pooled into G2** (pooling
  would make the largest eval source the one register `V_moral` never trained on, so the
  aggregate G2 would measure generalization, not contamination). Realized clean yields at
  scale: MORABLES 0.41 train / 0.37 eval (below the 0.60 n=10 estimate — small calibration
  samples over-estimate), Moral Stories 0.57 / 0.54, ETHICS 0.54.
- **2026-06-27 (register: two-register + declarative-ceiling finding):** Narrative-only
  training (a §4.2 control violation) fixed by **re-rendering audited content into declarative
  surface** (content held constant; `generate_declarative_rerender.py`). Achieved **71/29
  narrative/declarative** train (805 / 335) and 106 / 28 in-dist eval. **Finding:** declarative
  clean yield is capped at ~0.40 — forcing tight §1.2 parallelism (minimal-pair re-render)
  trades directly into §1.5 (residual valence), clean stays ~0.42. This is the **same
  minimality-vs-valence tension ETHICS exhibits, now shown to be REGISTER-intrinsic to
  abstract/declarative statements**, not source-specific. One iteration confirmed the ceiling;
  accepted per the ETHICS discipline (register finding, not failure). Consequences: `V_moral`
  is narrative-dominant with a 29% declarative component (not 50/50); the cross-register
  Track-4 test has 28 content-paired pairs (directional signal, thin CI); **dialogue remains
  a documented coverage limitation.** Final pre-paraphrase dataset: train **1140**,
  eval_g2_indist **134** (106 narr + 28 decl), eval_probe **118**.
- **2026-06-27 (G2 hard STOP gates on the narrative slice):** G2's hard STOP is computed on
  the **106 narrative** in-dist eval pairs, **not** the aggregate 134. The 28 declarative
  pairs are (a) selected-for-cleanliness survivors of the declarative 0.40 ceiling, (b) a
  register `V_moral` covers only 29% in training, and (c) too thin (n=28, wide CI) to carry a
  hard STOP at the **0.10 threshold, which was calibrated on Paper 1's narrative
  construction**. A wide declarative paraphrase gap could reflect register-coverage asymmetry
  or the 28-pair CI rather than contamination; admitting it to the aggregate risks a STOP that
  is not about contamination. Therefore: **paraphrase all 134; the declarative slice's gap is
  REPORTED as informative (not gated); the G2 STOP gates on the 106 narrative slice only.**
  Same discipline as ETHICS — the clean pool gates, the thin/confounded slice reports.
- **2026-06-27 (Track-4 cross-register is directional, not confirmatory):** The within-`V_moral`
  cross-register transfer test uses **28 content-paired** narrative↔declarative eval pairs,
  pool-capped by the declarative 0.40 ceiling × 106 narrative. At n=28 the CI is wide, and
  given the project's prior finding that linear probes fail declarative↔narrative transfer, a
  weak/null result **cannot distinguish "register augmentation didn't take" from
  "underpowered."** Track 4 (cross-register) is therefore pre-registered as
  **DIRECTIONAL / EXPLORATORY**, to be read as designed rather than as a shortfall. The
  register fix's **confirmatory** wins are elsewhere: a less register-specific `V_moral` (29%
  declarative training) and the decomposed/interpretable ETHICS probe.
- **2026-06-27 (SINGLE-SOURCE V_moral — MORABLES dropped):** MORABLES is dropped from the
  program for **two independent reasons**: (1) it is **CC-BY-NC-4.0**, incompatible with the
  repo's Apache-2.0 posture (committing its retellings would make commercial use of `V_moral`
  inherit NC); and (2) it is **~79% non-re-derivable** from public domain — only ~21% of our
  selection is canonical enough for a model (Sonnet 4.6 or Opus 4.8) to retell from title+moral;
  the rest is obscure Perry-index fables in neither public-domain editions nor model knowledge.
  L'Estrange sourcing was assessed and rejected (won't reach the Babrius/Aphthonius tail; low
  per-fable value). Consequences: **`V_moral` is single-source — Moral Stories only**, two
  registers (narrative 573 + declarative 304 = 877 train; eval_g2_indist 79, later expanded to
  145 = 96 narrative + 49 declarative once source-balance stopped binding — the G2 narrative
  STOP slice is now 96). **GATE G-AXIS is not run** (no second source to gate; the cross-source
  axis-agreement finding is out of scope);
  the Phase-2 single-source branch runs directly. ETHICS remains the generalization probe.
  Committed Apache-clean dataset: `deepsteer/datasets/d1_vmoral_v1.json`
  (+ `DATASET_LICENSES.md`). **Future work:** a fable-based extension if a clean public-domain
  source/method becomes viable. No change to G2, G3, or their thresholds.
- **2026-06-27 (G3 cross-model resolution + Point B wiring — two SAME-MODEL points):** The
  cross-model question (V_moral on Base vs refusal on Instruct) is resolved by measuring each
  refusal point **within its own model**, eliminating any Base↔Instruct projection:
  - **Point A = BASE proto-refusal × Base-V_moral** — the refusal feature present before SFT
    wires the gate (`extract_proto_refusal.py` construction: raw last-token mean-diff on the
    base model), projected onto V_moral extracted on the **base** model, vs the **base** null.
  - **Point B = INSTRUCT refusal gate × Instruct-V_moral** — the actual aligned-stage gate
    (chat last-token mean-diff on the instruct model), projected onto V_moral extracted on the
    **instruct** model, vs the **instruct** null. **This is the direct comparison to Paper 5's
    0.1044** (instruct refusal × instruct subspace), now against the richer subspace.

  Both prompt sets are the real Heretic set (`refusal_prompts.json`, 400 harmful/harmless), not
  the fallback placeholder; **Point B is now a genuinely distinct object** (wired gate vs base
  precursor), replacing the earlier `p_B = p_A` stub. `V_moral` is therefore extracted on **both**
  models: **Base-V_moral** is the primary comprehension instrument (G2, Track-1, eff-dim); the
  **Instruct-V_moral** exists for Point B's same-model measurement. Each point clears its OWN
  frozen null + control (per-tag `null_artifact.json`), so predates-the-result holds per model.
  "Robust across operationalizations" now spans the **pretraining-precursor (A)** and the
  **aligned-gate (B)** refusal — the most meaningful axis — each cleanly same-model. The G3
  rule (POSITIVE iff BOTH clear; split → NULL + flag) and M=0.05 are unchanged. Validated by the
  two-tag `VALIDATE` dry run. (A cross-model projection — instruct refusal × Base-V_moral — may
  be reported later as a transfer-robustness check, with the base→instruct caveat.)
- **2026-06-28 (V_moral construction RE-SPEC — finding-driven correction):**
  *Causal structure: a discovered property of the difference vectors invalidated the assumption
  behind the §5 eff-dim spec, which forced the construction change. The trigger is a symmetric
  defect, independent of what refusal does, so the change is anchored to the data, not the result.*

  **(1) Discovered data property (outcome-independent).** The real-run spectrum of the pooled
  per-pair difference vectors has **no elbow** and is **content-dominated**: singvals 31, 18, 15,
  14, … flat; `d_moral` is the top direction but carries only **7.5% of variance**, so uncentered
  eff-dim @ 0.90 = **385** (≈10% of the 4096-dim space). At rank 385 the subspace is degenerate
  **for every direction tested** — null `q95` = 0.80, persona `c` = 0.73, and refusal all project
  ~0.7–0.8. It has no discriminating power for *any* direction regardless of refusal's value;
  the degeneracy is symmetric, so identifying it cannot be motivated by the orthogonality outcome.

  **(2) Invalidated assumption.** §5 assumed uncentered eff-dim @ 0.90 of the pooled diffs would
  capture *moral* rank. Property (1) shows it captures **content** rank: the moral structure is
  not in the 0.90-variance subspace at all. The original §5 denominator was measuring the wrong
  thing, so the G3-as-specified could not be run on it.

  **(3) Forced correction.** The moral structure lives in the **source mean-diff directions**, not
  the pooled-diff subspace (pooling fables barely moves the diff spectrum; content still
  dominates). Two sources each add a **distinguishable** moral axis — `cos(d_fables, d_moral)=0.53`,
  `cos(d_ethics, d_moral)=0.36` (vs the non-moral persona reference 0.24) — so the three source
  directions span **effective rank 3**. Therefore **`V_moral` = orthonormalized span of the source
  moral mean-diff directions** (`{d_moral, d_fables, d_ethics}`, rank 3), constructed exactly like
  the MFT subspace (span of its 6 foundation directions, rank ~4). This makes G3 directly
  comparable to Paper 5's 0.1044 (refusal onto a foundation-direction span). Note the bar moved
  *toward* the harder, more comparable test (a richer subspace), not toward an easier one.

  **What is preserved (the pre-registered spine is untouched).** The G3 decision rule, the
  conjunction, and **M = 0.05 are unchanged**; NULL remains the pre-declared more-publishable
  outcome. **The rank-matched null + persona control are recomputed on THIS span** under the
  original two-step recipe (different subspace ⇒ different null; realized mechanically from the
  actual `V_moral` before the refusal projection, which never enters the null). G3 is reported as
  the **rank-3 span point estimate (order-invariant, headline) + a rank-sweep (1→2→3)**; the
  refusal vector is saved. Any per-axis "refusal aligns with the fable/action axis" is
  **basis-dependent ⇒ diagnostic only, not a headline claim**. ETHICS's full build (over-generate
  / filter / bias-flag) is justified by this re-spec but feeds **G2**, not G3 (G3 reads the source
  directions, already in hand).

  **Standalone cautionary finding (a contribution, independent of G3's result):** eff-dim
  thresholding on a no-elbow, content-dominated pooled-diff spectrum measures **content rank, not
  moral rank** (385 vs the rank-3 in the source directions) — a real caution for anyone
  constructing "moral subspaces" by SVD on per-pair difference vectors.
- **2026-06-29 (REASONING-MODEL EXTENSION — OLMo-3-7B-Think; refusal positions pre-registered
  BEFORE the run):** Extends G3 from Base+Instruct to a same-family reasoning model
  (`allenai/Olmo-3-7B-Think`; verified `model_type=olmo3`, 32 layers, hidden 4096 — identical
  architecture, so V_moral extraction is a model-id swap). Motivation: a reasoning model with an
  explicit `<think>` trace is the **most adversarial test** of the orthogonality headline —
  explicit moral reasoning about harm is exactly where refusal would couple to `V_moral` if it
  couples anywhere. This is the model axis of the scope boundary in RESULTS.md, complementary to
  the (still-open) subspace-construction axis.

  **(A) `V_moral` is recomputed fresh in Think's space — no transfer.** Directions do not
  transfer across models (per-model extraction; consistent with the cross-model degradation
  finding). Re-extract `d_moral`, `d_fables`, `d_ethics`, the persona control, and the refusal
  direction(s) in Think's own activations at layer 16 (stable band 15–31). Carry the
  finding-driven **rank-3 source-direction-span construction** (not eff-dim@0.90) as the method.
  **VERIFY, do not assume, that Think's pooled per-pair-diff spectrum is content-dominated** — a
  one-line check (top moral-direction variance fraction + no-elbow). If Think unexpectedly shows a
  low-rank *moral* elbow, that is itself a finding and the construction is revisited explicitly,
  not silently reused. The two-step null (`q95`) and persona control (`c`) are recomputed on
  Think's rank-3 span **before** any refusal vector is projected (fresh per model, same frozen
  recipe).

  **(B) Refusal POSITIONS — the load-bearing pre-registration.** The chat template auto-opens the
  reasoning channel (templated tail = `<|im_start|>assistant\n<think>`), so the existing
  last-input-token extractor lands on the `<think>` opener, *before* any reasoning. A single
  mis-positioned extraction is dangerous here: **a wrong-position NULL is indistinguishable from
  "orthogonality survives"** — if the post-answer refusal token is orthogonal but the in-trace
  moral-reasoning representation couples, reading one position hides the exact coupling this
  experiment exists to detect. Therefore refusal is extracted at **four pre-registered positions**,
  grounded in the Zhao et al. keystone (`deepsteer.reasoning.token_positions`), and **all
  four are reported**:

  | Pos | Site | Method | Role |
  |---|---|---|---|
  | **P0 `t_inst`** | last instruction-content token | prompt-side, no generation | harmfulness / comprehension site (Zhao: harm encoded here) |
  | **P1 pre-trace gate (`t_post_inst`)** | the `<think>` opener (last templated token) | prompt-side, no generation | **direct methodological analog of Base/Instruct Points A/B** (last-input-token diff-of-means) — apples-to-apples with the existing 0.1044-regime result |
  | **P2 in-trace** | last reasoning token before `</think>` | requires generation | **the coupling detector** — does the moral *reasoning* representation align with `V_moral`? Only the Think run can surface this |
  | **P3 post-answer** | last answer token after `</think>` | requires generation | the refusal *decision* site (what the model actually outputs) |

  Each position's refusal direction = harmful−harmless diff-of-means of activations at that
  position over the real Heretic set (`refusal_prompts.json`, 400 h/h). P0/P1 are prompt-side
  (deterministic, matched prompts — the clean diff). P2/P3 require a generated trace and are
  **content-confounded** (harmful traces discuss refusing, harmless discuss the topic), so their
  diff-of-means mixes refusal with topic; the persona control `c` + the rank-matched null still
  bound "any meaningful direction projects high," and this caveat is reported with P2/P3.

  **(C) Decision rule (unchanged spine; multiple-comparisons discipline fixed now).** Per position,
  refusal is NULL vs the Think span iff `p ≤ q95 + M` **AND** `p ≤ c + M` (`M = 0.05`, the frozen
  margin). To avoid position-fishing, the **pre-specified coupling hypothesis is P2 (in-trace)**:
  "reasoning-model coupling detected" = **P2 clears both bars**. P1 and P3 are the
  comparability/decision verdicts; P0 is the comprehension-site reading. **Headline outcomes,
  pre-declared:** orthogonal at all four = the robust strengthening (the most adversarial case
  still NULL); **orthogonal at P0/P1/P3 but coupled at P2 = the more interesting reasoning-specific
  finding** (flagged for investigation, not retraction, per the Papers 5–7 claim discipline). NULL
  remains the pre-declared more-publishable baseline; a P2 coupling would be the publishable
  *positive*, and it is pre-registered as a specific, single hypothesis so it cannot be a post-hoc
  fish across positions.

  **(D) Generation protocol (fixed now).** Greedy decode (deterministic, reproducible),
  `max_new_tokens` capped. `</think>` is located in the generated region (P2 = the last reasoning
  token before it; P3 = the last meaningful answer token after it). The keystone is verified on
  Think: `t_inst` = last instruction token, post-instruction suffix =
  `<|im_end|>\n<|im_start|>assistant\n<think>` (8 tokens), `t_post_inst` = the final
  `<think>`-opener token. A prompt that emits no `</think>` within the cap is **excluded from
  P2/P3 and counted** (never silently dropped), with the exclusion rate reported. P0/P1 use all
  prompts (prompt-side, no generation).

  **(E) Scope — headline only; stage-trajectory is a separate experiment.** The
  `Olmo-3-7B-Think-SFT` / `-Think-DPO` checkpoints enable a base→SFT→DPO trajectory, but that is
  the Direction-2 "where does the dependence form" question — a **separate pre-registered
  experiment**, not bundled here. This amendment runs the **headline clean**: Think final model,
  four-position refusal, G3 + G2, fresh two-step null. G2 contamination coverage on Think uses the
  same model-agnostic datasets (already built). The pre-registered spine (G3 rule, conjunction,
  `M = 0.05`, two-step null) is **unchanged**.

  **Run-1 record + `</think>`-detection correction (2026-06-29, post-run).** First GPU run
  executed. **Content-dominated check PASSED on Think** (`top_dir_var_frac = 0.0737`,
  `effdim@0.90 = 385`, essentially identical to base) → the rank-3 source-direction construction
  is confirmed the right method on Think, not assumed. **P0 (harm site) p = 0.291 and P1
  (pre-trace gate) p = 0.101 vs null `q95 = 0.304`+M, persona `c = 0.525` → both NULL** (orthogonal);
  P1 ≈ Paper-5 0.1044, consistent with Base/Instruct. **BUT P2/P3 closed-rate = 0.0**: the
  `</think>` boundary detector used the fixed 3-token subsequence `[524, 27963, 29]`, which is
  **wrong** — under BPE the leading `</` merges with the preceding char (`.</` = 4005, ` </`,
  `\n</`, …; verified by in-context encode), so the fixed subsequence essentially never matches
  generated text. This is a detection-MECHANISM bug, **not** a change to the position definitions
  or any gate. **Correction:** anchor on the stable suffix `[27963, 29]` (`think`,`>`) and validate
  the `</` prefix per occurrence (`_find_think_close`), taking the first `</`-validated close in
  the generated region. Validated synthetically (`harm.</think>I can't…` → P2 = `harm`, P3 = `.`).
  P0/P1 are unaffected (prompt-side, no `</think>`), so the P0/P1 NULL above stands; only P2/P3
  need re-running (refusal-only, the moral/persona/axis artifacts are unchanged). Cap raised to
  2048 with `gen_len`/closed-rate/sample-trace diagnostics so a cap-limited closed-rate is now
  visible. The spine is untouched.

  **P2 RE-SPEC to a SYMMETRIC in-trace window + P3 scoped UNMEASURED (2026-06-30, post-run-2).**
  *Causal structure: a discovered, harm-correlated budget property would make the naive in-trace
  span a span-length contrast; the symmetric window removes that confound. The change is forced by
  the data and required for validity, not chosen for an outcome.*

  **(1) Discovered data property (harm-correlated).** With the detector fixed, run-2 (greedy, cap
  2048, n=64/side) gives `closed_rate` **harmful 0.906, harmless 0.000** (`has_close_str` agrees, so
  it is genuine non-closure, not a detector miss; harmless samples are coherent, non-degenerate
  reasoning). Benign prompts make the model reason out the whole answer *inside* the trace and blow
  past budget without closing `</think>`; harmful prompts recognize the request and close early.
  So **trace-completion state is correlated with the harmful/harmless label.**

  **(2) The confound this creates for the naive P2.** A full-reasoning-span mean would average a
  SHORT span for harmful (early `</think>`) vs the FULL 2048 for harmless (never closes). Span
  *region/length* is then correlated with the label, so the P2 harmful−harmless diff-of-means would
  be partly a short-span-vs-long-span contrast, **not a pure harm contrast** — the v1
  animacy/shortcut failure mode, on **the single most important new measurement** (P2 is the one
  site where coupling is actually expected). A confounded P2 cannot distinguish real coupling from a
  span-length artifact in either direction.

  **(3) Correction — symmetry is the requirement.** **P2 (in-trace) = mean over the FIRST
  `cot_window_n` reasoning tokens** from the common anchor (the post-prompt reasoning start, the
  keystone `t_post_inst`+1), **SAME N both sides** (pre-registered `N = 256`). Both sides become
  "first N tokens of deliberation," so the span region is identical across the label and the
  confound is gone — this is **confound-avoidance, not merely closure-robustness** (it also happens
  to be closure-robust, since the first N tokens are reasoning whether or not the trace later
  closes). A trace whose reasoning span < N is **excluded and counted** (window always pure
  reasoning, never spilling into an answer). The **full-span mean (`P2_FULL`) is reported as a
  ROBUSTNESS check only**: if window and full-span agree, the result is robust; if they diverge,
  span-length sensitivity is localized.

  **(4) P3 is UNMEASURED, not measured-and-null.** The post-answer contrast cannot be built because
  the **benign side never reaches a post-answer state within budget** (the harmful side does). The
  bounded claim is therefore: **orthogonal at harm-recognition (P0), gate (P1), and in-trace
  deliberation (P2); the post-answer site (P3) is unmeasured for benign prompts.** The benign
  over-reasoning is a minor finding (reasoning-model verbosity), stated as a constraint on the
  contrast, not as a null result.

  **Spine preserved.** P2 remains the single pre-registered coupling hypothesis; null `q95` +
  persona `c` are span-properties (unchanged by P2's operationalization) and are recomputed on the
  Think span **before** the P2 projection (two-step discipline). Greedy, `M = 0.05`, the G3 rule,
  and the rank-3 construction are unchanged. Re-run fills P2 only (P0/P1 already settled).

  **MFT-comparison framings LOCKED (2026-06-30, from the committed-number pull; prose written
  ONCE, after the three bundled numbers land — no "pending" placeholders).**
  - **Q1 — 0.1044 is RAW.** `heretic_ablation.py:190`: `subspace_projection_fraction(refusal,
    basis)` = least-squares projection of the instruct Heretic gate onto the **6-foundation MFT
    span** at layer 16; **no null was ever built.** Both subspaces are judged each against its
    **own rank-matched null**, not raw-vs-raw.
  - **Q2 — rank language (locked):** state **"3 source directions vs 6 foundation directions"**
    (the null-matching projection bases); cite **eff-dim 4 (MFT) vs 3 (V_moral)** separately as the
    *variance* rank (`base/track1_result.json`). **On BOTH measures MFT is higher-dimensional (6≥3
    directions, 4≥3 eff-dim).** So **"richer" must NEVER imply dimensionality** — the rich
    subspace's contribution is **construct-diversity + verified-distinguishability +
    contamination-resistance**, not more dimensions. The data contradicts "more complex" on both.
  - **Q3 — σ\* gap:** committed Track-1 σ\* is the **single-source** V_moral (eff-dim 385), never
    the rank-3 span → **re-run Track-1 σ\* on rank-3** (bundled) before claiming "as robust as MFT"
    for the headline instrument. Don't characterize the published instrument's fragility via its
    dropped predecessor.
  - **Q4 — gradient framing (locked):** MFT was measured **gate-only** (single position, layer 16;
    Base/Instruct have no `<think>`). The gate→harm→in-trace gradient has **no MFT counterpart** →
    *"a gradient the earlier MFT work didn't measure,"* **never** *"a gradient MFT lacked."*
  - **Fair "orthogonal to both" (base space, judged-vs-judged):** base proto-refusal (Point A) —
    rank-3 V_moral `p=0.326 vs null 0.291 → NULL`; 6-foundation MFT `p=0.264 vs null 0.257 → NULL`.
    The **instruct-gate** version (tightest same-object as 0.1044) needs the **instruct MFT**
    extracted → **bundled into GPT-OSS** (only base MFT is committed).

  **GPT-OSS = TERMINAL Direction-1 experiment (2026-06-30): replication-and-boundary, three
  payloads.** The near-threshold in-trace P2 (0.009 below margin, robust across span defs) makes
  GPT-OSS **replication of the headline on a second, differently-trained reasoning model**, not
  cross-lab completeness — so it's in the headline paper, and it is the **LAST model** (two
  independently-trained reasoning models suffice; resist a third / the 32B). Either outcome is
  terminal: gradient replicates → model-generalization claim made; gradient diverges → "the
  gradient is OLMo-specific," also a finding. **Three payloads on one pod:** (1) GPT-OSS full
  four-position sweep (P0/P1/P2 symmetric-window + P3), **re-deriving the in-trace anchor for the
  harmony trace structure** (NOT copied from OLMo's `<think>`; the analysis channel opens during
  generation, not in the prompt) — a gate-only run would answer the old cross-lab-orthogonality
  question and miss the gradient, which is the whole point; (2) **instruct-MFT-null** (tightest
  judged-vs-judged reproduction); (3) **rank-3 σ\*** (headline-instrument robustness). GPT-OSS is
  natively deliberative (no base/think fork), bf16-dequant for mxfp4 parity (Paper 7's resolution),
  fresh nulls + content-check in GPT-OSS's own space (V_moral doesn't transfer). Expect
  P3-unmeasured-for-benign again (likely worse verbosity — don't burn budget forcing closure).
  Contribution target: *"refusal ⊥ moral representation — verified-rich rank-3 subspace (closes
  thin-MFT), robust across the reasoning chain with a characterized gradient peaking in-trace,
  replicated across two reasoning models from different labs."* Direction 2 (training) is the NEXT
  paper, not the next section.
- **2026-07-01 (CALIBRATION INSTRUMENTATION — additive; the spine is untouched):** A calibration
  layer is added around the settled D1 headline before the write-up framing is finalized, specified
  in `CALIBRATION_PREREG.md` (commit `5c542f6`) and the decision-coupling companion
  `../d2_decision_coupling/PREREGISTRATION.md`. **Nothing in the G3 spine changes:** the decision
  rule, the two-bar conjunction, `M = 0.05`, and the two-step null protocol (§0, §3.3–§3.4) are as
  frozen; NULL remains the pre-declared more-publishable outcome. What is added is *instrumentation*
  for the projection instrument, computed only after that prereg was committed: (A1) a **moral
  positive control** — held-one-out `p(d_src | span{other two})` per tag, defining a **moral-family
  band** `[min,max]` (rule R1) that is the yardstick for every "moral-adjacent" statement, plus the
  MFT↔V_moral mutual projection (closes Paper 7 Phase 2f at the subspace level) and a calibrated
  ladder (iso floor → null q50/q95 → refusal points → moral band → persona) that recontextualizes
  the GPT-OSS in-trace P2 = 0.52; (A3) a refusal variance-percentile ("spare-channel") mechanism
  annotation; (A4) bootstrap CIs, an EXPLORATORY combined-P2 Fisher statistic (R7; per-model
  verdicts remain the pre-registered rule), and signed projections; (A5) proto-refusal continuity
  `cos(proto-refusal_base, refusal_instruct)`. **Genuinely non-moral controls** (syntax + register,
  rules R5/R6) and the persona reclassification are staged for B3, so the "below a known non-moral
  axis" sentence is held until measured. This amendment is additive calibration + statistics; it
  neither reopens nor re-litigates any G3 verdict.
- **2026-07-01 (FORMAT-ROBUSTNESS scoping branch for the G3 orthogonality claim — pre-registered
  via the D2 in-format ladder):** The Direction-2 decision experiments surfaced that `V_moral` and
  the A1 ladder were built in **raw / mean-pooled** format, whereas the aligned-model decision
  directions (refusal gate, judgment-decision) live at the **chat / decision-site** position — a
  format/position mismatch (the content `label_contrast` also projects low, ~0.10 on OLMo, the
  tell). No committed G3 verdict changes; each stands **as a raw-format representational statement**.
  This adds a pre-registered **format-robustness branch**, tested by the D2 in-format ladder
  (`../d2_decision_coupling/PREREGISTRATION.md` Amendment 1 §7: chat-format, factor-decomposed
  `V_moral` + full in-format ladder): G3 orthogonality is **format-robust** iff the refusal gate sits
  below the **in-format** moral-family band by `M`; otherwise the claim is **scoped to the raw
  narrative register**, attenuation documented (a register-bound moral geometry is a pretraining-data
  finding, not a retraction). Frozen spine (rule, conjunction, `M = 0.05`, two-step null) unchanged;
  both branches publishable in advance.
