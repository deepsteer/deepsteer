# Direction 1 — Pre-registration of Gates G2 and G3

**Date:** 2026-06-26 · **Against commit:** `107f9f3` (HEAD) · **GPU-free.**
**Status:** Pre-registered before any `V_moral` construction or refusal measurement.
Nothing here is a result. Every threshold below is fixed *now*, in advance of the
data, so that stop/go is criterion-driven rather than a judgment call mid-run.

Companion: `GATE_HOOK_VERIFICATION.md` (the three gate-critical hooks were verified
to behave as assumed before pre-registering against them).

> **Paper number is provisional.** This work lives in `papers/direction1_moral_subspace/`
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

## 5. eff-dim convention (the G3-null denominator) — fixed now

- **`V_moral` rank = uncentered effective dimensionality at variance-threshold 0.90**, from
  the SVD of the stacked per-pair difference vectors (`pos − neg`). Uncentered because the
  shared moral-valence axis is the *signal*; centering (what
  `direction_utils.effective_dimensionality` does, line 332) would remove it and understate
  the rank. Consistent with Paper 6's uncentered eff-rank.
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
  registers (narrative 573 + declarative 304 = 877 train; eval_g2_indist 79). **GATE G-AXIS is
  not run** (no second source to gate; the cross-source axis-agreement finding is out of scope);
  the Phase-2 single-source branch runs directly. ETHICS remains the generalization probe.
  Committed Apache-clean dataset: `deepsteer/datasets/direction1_vmoral_v1.json`
  (+ `DATASET_LICENSES.md`). **Future work:** a fable-based extension if a clean public-domain
  source/method becomes viable. No change to G2, G3, or their thresholds.
