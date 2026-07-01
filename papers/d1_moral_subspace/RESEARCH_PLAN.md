# DeepSteer — Direction 1: A Robust Moral Subspace as Measurement Instrument

**Status:** Planning. Nothing in this document is executed. All findings stated
conditionally are hypotheses, not results. Language is deliberately in the
"aims to / will / is designed to" mood; any sentence that reads as a present-tense
capability claim is a drafting error and should be flagged.

**Scope:** This plan covers **Direction 1 only** — building and *validating* a rich
moral subspace as a measurement instrument, and applying it as a sanity check to
existing Papers 3+. Direction 2 (trajectory of the moral representation across
OLMo 3 training stages) is explicitly **out of scope** here and gated behind
Direction 1's evaluation (see Phase 5). Direction 1 is intended to be self-contained
enough to stand as its own contribution, consistent with the external reviewer's
observation that this work spans at least two distinct papers.

**Execution model:** Claude Code executes against `github.com/deepsteer/deepsteer`,
on `main`, single RunPod A100-80GB. This document is the brief; gate conditions
(`GATE Gn`) and settled decisions (`DECISION Dn`) are pre-registered so that
stop/go is criterion-driven rather than a judgment call mid-run.

---

## 1. Motivation

Paper 5 established the comprehension–compliance gap: the refusal direction is
geometrically orthogonal to the MFT moral subspace (projection fraction **0.10**),
and behavioral coupling is near-zero (P(comply | comprehend) ≈ P(comply | ¬comprehend)
≈ 0.75, φ ≈ 0.05). Yet the moral subspace is *functionally live* for generation —
dependency grows ~6× through alignment (+0.063 nats at Instruct). The subspace is
used, but causally disconnected from the refusal gate. Paper 6's ART attempt failed
for the same reason: the refusal direction stayed orthogonal regardless of how much
moral-subspace dependency was built.

The single most dismissible weakness in this arc is the **operationalization of
"moral" as a thin six-probe MFT subspace.** A reviewer can reasonably respond:
"harmfulness sits outside *your* subspace because your subspace is six MFT probes,
not a faithful representation of moral cognition." Direction 1 removes that objection
by constructing a richer, better-grounded moral subspace — built from fables,
narrative moral contrasts, and abstract ethical judgment rather than MFT axes — and
then **re-asking the orthogonality question against the stronger instrument.**

The result is decision-relevant in *both* directions:

- **If projection stays low (~0.10) against the rich subspace** → orthogonality is
  robust to operationalization. This *strengthens* Papers 5–7 and permanently closes
  the thin-MFT objection. (This is the more publishable outcome: orthogonality robust
  across operationalizations.)
- **If projection rises and survives a rank-matched null** → the thin-MFT objection
  was correct; the richer instrument reveals coupling the MFT probes missed.

---

## 2. Pre-registered design decisions

### DECISION D1 — Single best-construct subspace (settled)

The deliverable is **one unified moral subspace `V_moral`**, not a per-framework
collection. Rationale: a per-framework subspace re-introduces the exact problem we
are escaping — "overlap with *which* axis?" is the thin-MFT objection in new clothing.
A single subspace yields one number (projection of refusal onto `V_moral`) that is
non-dismissible.

Per-framework structure is retained **strictly as an interpretability layer
underneath `V_moral`** — used to explain *where* any overlap lives if it exists, never
to define the subspace. Two decomposition layers are planned, with different roles:

- **MFT decomposition** (validation / backward-compat): projection of the existing six
  MFT directions into `V_moral`. Free — uses directions and clustering code we already
  have. Lives in Phase 4.
- **MoReBench decomposition** (forward / literature): per-framework attribution of the
  new pairs using the Huang et al. five-framework taxonomy. Costs Together.ai inference;
  conditional. Lives in Phase 3.5.

### DECISION D2 — Build alongside the six-probe MFT subspace; disposition is conditional (settled)

`V_moral` is added as a **new module alongside** the existing six-probe MFT subspace,
not as a replacement. "Replace everywhere" can only ever be a *conclusion* of Direction 1,
never a premise — the entire value proposition ("the rich subspace fixes the thin-MFT
weakness") is only demonstrable by comparison against the thin subspace. Removing the
MFT subspace destroys the evidence that replacing it was justified, and orphans the
reproducibility of Papers 3–7, which bake it into their committed outputs.

Final disposition is decided by the **GATE G3** result (refusal-overlap vs. null):

- **G3 null (projection ~0.10 on rich subspace too):** both coexist. `V_moral` becomes
  the standard moral representation going forward; the six-probe subspace is retained as
  the documented MFT-specific operationalization; headline = "orthogonality holds across
  both operationalizations."
- **G3 positive (projection rises, survives rank-matched null):** `V_moral` replaces MFT
  as the standard instrument; Papers 5–7 receive a revision note on the orthogonality
  claim. **The null is load-bearing here:** genuine coupling vs. rank inflation is the
  entire question, and the first reading may not be claimed without ruling out the second.

### Rank discipline (applies to all projection measurements)

Projection fraction rises **mechanically** with subspace rank — any vector projects more
onto a higher-dimensional subspace. The Paper 5 number (0.10) is only meaningful relative
to the rank it was measured against. Therefore: **`V_moral`'s rank is reported explicitly
(via eff_dim), and every projection number is interpreted only against a rank-matched
null.** "Refusal projects at X" is never a claim; "refusal projects higher than the
rank-matched null" is.

---

## 3. Substrate

- **Primary build/eval model: OLMo 3 Think (7B).** Genuine RL-trained reasoning model
  with the full post-training stack released (Base → SFT → DPO → RL), Apache 2.0,
  single-A100-feasible, same family as Papers 1–6.
- **Moral comprehension representation:** extracted on **OLMo 3 Base**.
- **Instruct-time refusal direction:** see the two-point measurement in GATE G3.
- **Robustness panel:** the Paper 6 panel (OLMo / Qwen2.5-7B / Llama-3.1-8B), per the
  per-model-extraction discipline that both Paper 6 and Huang et al. Appendix C.2
  (33–48% cross-model probe degradation, asymmetric) require.

Direction 1 needs **only single-snapshot measurements** — the pretraining checkpoint
grid is irrelevant here and is a Direction 2 concern. Choosing OLMo 3 now makes
Direction 2 a clean extension rather than a re-build.

---

## 4. Phase 1 — Curation (the heavy investment)

The subspace is only as good as the contrastive pairs. Most of the risk and most of
the effort live here. **Phase 1 is single-track curation** — no MoReBench labeling pass
occurs here (deferred to conditional Phase 3.5).

### 4.1 Source allocation

| Source | Role | Construct |
|---|---|---|
| **MORABLES** (arXiv 2509.12371, Marcuzzo et al.) | **Primary — construct anchor** | Classical-fable moral judgment: correct moral vs. plausible-but-wrong moral, or moral-laden vs. neutral retelling of the same fable event. This is what we mean by "moral." |
| **Moral Stories** (Emelin 2021) | **Primary — paired-contrast generator** | norm + situation + paired moral/immoral action, holding situation constant. Drop-in for mean-diff. Cleanest narrative minimal-pair source available. |
| **ETHICS** commonsense (Hendrycks 2021) | Secondary — hard-ambiguity span | Near-minimal binary acceptable/unacceptable pairs at ~55% model accuracy. Adds a difficulty register the constructed sources lack. |
| **Social Chemistry 101** (Forbes 2020) | **OOD generalization check only** | 5-point, implicit-norm, ~30% accuracy, MFT-grounded (the *old* operationalization). **Do not extract directions from it.** Hold out as a generalization probe. |

### 4.2 Construction guidelines (carried over wholesale from the v2 overhaul)

These are the controls that already saved the program once.

- **Relational-structure matching:** neutral / contrast sentences must match the moral
  sentence's relational structure — **human subjects AND human objects** in
  person-to-person moral scenarios — so the probe cannot exploit animacy or
  interpersonal-vs-object structure as a shortcut.
- **Three registers:** declarative / narrative / dialogue, with per-source and
  per-register balance tracked so no single source or surface register dominates the
  extracted direction.
- **Audit gates (DATASET_GUIDELINES.md), held to v2 thresholds:** §1.1 inanimate
  neutrals, §1.2 structural parallelism, §1.5 accidentally-moral neutrals. Re-run the
  full audit against all new pairs.
- **Target size:** ~1,000–1,500 pairs (v2 ballpark), with per-source and per-framework
  balance reported.

### 4.3 Contamination controls (MANDATORY — not optional)

All three primary sources are 2020–2025 and near-certainly in OLMo 3's pretraining
data. A probe may therefore read memorized benchmark text rather than emergent moral
structure — the same class of confound as the v1→v2 animacy shortcuts, but with less
excuse for widely-scraped corpora.

- **Held-out paraphrase set:** for every moral judgment, a paraphrase preserving the
  moral content but breaking surface form. The probe must read moral structure, not
  memorized text. **Build this in Direction 1 even though its primary payoff is in
  Direction 2** — a memorization confound that is tolerable in a single snapshot becomes
  a finding-retracting confound when claiming "moral structure emerges at checkpoint N."
- **Pre-registered no-leakage check:** lift Huang et al. Appendix D.8 directly — verify
  that no label-derived or judge-derived metadata (framework scores, difficulty tags,
  source tags) leaks into the probe training signal. Same bug class as the σ* None→0.0
  aggregation and the RMS activation-scale confound: cheap to check now, expensive to
  discover at review.

### 4.4 Phase 1 deliverables

- Curated pair set (~1k–1.5k) across three registers, per-source/per-register balanced.
- Held-out paraphrase set, 1:1 with the moral judgments.
- DATASET_GUIDELINES.md audit report at v2 thresholds.
- No-leakage verification log.
- Frozen, versioned dataset tag (mirror the `v2.0-dataset-overhaul` convention).

---

## 5. Phase 2 — Subspace construction

- **Primary extraction: mean-diff.** The probe sprint established it as the most stable
  direction-extraction method; a training-free direction is the honest choice for a
  representation we then test for fragility.
- **`V_moral`:** one unified subspace pooling all *primary-source* pairs (MORABLES +
  Moral Stories + ETHICS; **not** Social Chemistry 101).
- **Rank determined empirically by effective dimensionality**, not fixed a priori. The
  deliverable is a subspace whose rank we can defend — not a rank-1 direction (which
  would almost certainly miss real overlap and make the orthogonality test unfairly easy
  to pass). **Report eff_dim explicitly** — it is the denominator every projection number
  depends on.
- **Cross-method agreement check:** compute LEACE and probe-weight directions alongside.
  The three methods previously agreed at cosine 0.67–0.71; if the rich subspace breaks
  that agreement, flag and investigate *before* proceeding.

Reuse existing infrastructure: `direction_utils` for projection; the probe-sprint
extraction harness for mean-diff.

### Phase 2 deliverables

- `V_moral` (with eff_dim reported), versioned.
- Cross-method agreement report (mean-diff vs. LEACE vs. probe-weight cosines).

---

## 6. Phase 3 — Evaluation (heavy; the gate to everything downstream)

This is the phase weighted hardest, and where `V_moral` earns the right to exist.

### Track 1 — Probe accuracy + fragility (σ*)

Does `V_moral` linearly separate held-out moral/immoral pairs, and how robust is that
separation to activation noise? σ* is the metric that distinguishes a real moral
representation from a brittle lexical one. **A rich subspace that is *more* fragile than
the six-probe subspace is a warning sign**, to be understood before proceeding.

### Track 2 — Contamination validation — **GATE G2 (hard gate)**

Probe accuracy on the **held-out paraphrase set** vs. the original-surface set. A large
gap means the probe is reading memorized text, not moral structure, and the subspace is
untrustworthy for anything downstream.

> **GATE G2:** if the paraphrase gap exceeds the pre-registered tolerance, **stop and fix
> curation.** Do not proceed to Track 3, Track 4, Phase 3.5, Phase 4, or Direction 2.
> (Pre-register the tolerance before running; propose a concrete threshold for review.)

### Track 3 — Refusal-overlap, against rank-matched nulls — **GATE G3**

The payoff experiment. Done as **single instruct-time snapshots** (not trajectories),
and **two-point** for robustness across refusal operationalizations:

1. **Point A — Paper 5 proto-refusal contrast** (first; preserves continuity with the
   known 0.10 result). Source the contrast from `heretic_ablation.last_token_means`.
   Measuring the *same* refusal object against a *richer* subspace means any movement is
   cleanly attributable to the subspace, not to swapping the refusal direction.
2. **Point B — OLMo 3 aligned-stage refusal direction** (second). A different, more
   "real" object — the actual instruct-time refusal gate. If **both** points show the
   same overlap behavior, the result is robust across refusal operationalizations — a
   stronger claim than either alone.

For each point, measure projection fraction onto `V_moral` and compare against:

- **(a) the six-probe MFT subspace** (≈0.10, known baseline);
- **(b) a rank-matched null:** matched-magnitude random directions projected onto the
  *same* `V_moral` (expected projection-by-chance at this rank); **and**
- **(c) a non-moral semantic negative control:** the persona/assistant-axis direction
  from prior work, known to stay orthogonal to moral content (|cos| 0.076→0.085). This
  tests whether *any* meaningful direction projects high — i.e., whether `V_moral` is
  merely capturing general semantic structure.

> **GATE G3:** pre-register the null and the **DECISION D2** disposition rule *before*
> running. The claim is "refusal projects higher than both the rank-matched null (b) and
> the non-moral semantic control (c)," or it is not. G3's outcome sets the D2 disposition
> and triggers the Phase 3.5 branch.

### Track 4 — Robustness: cross-register and cross-model

- **Cross-register:** does a declarative-extracted direction read narrative and dialogue
  pairs? (Within-`V_moral` generalization.)
- **Cross-model:** extract per-model on the Paper 6 panel; compare geometries. Expect
  substantial variance (Huang C.2: 33–48% degradation, asymmetric). **Do not assume a
  single shared subspace transfers.**

### Phase 3 deliverables

- Track 1: accuracy + σ* report for `V_moral`, with σ* compared against the MFT baseline.
- Track 2: paraphrase-gap report and **GATE G2** pass/fail.
- Track 3: two-point projection table (Points A & B × controls a/b/c), **GATE G3**
  outcome, and the resulting **D2** disposition.
- Track 4: cross-register and per-model geometry report.

---

## 7. Phase 3.5 — MoReBench decomposition (CONDITIONAL — triggered by GATE G3)

Kept **off the critical path** so the first committable result does not depend on
Together.ai inference, the GPT-OSS-120B scorer, or LLM-judge variance.

- **Method (when run):** attribute each pair to the five MoReBench frameworks — Kantian
  Deontology, Benthamite Act Utilitarianism, Aristotelian Virtue Ethics, Scanlonian
  Contractualism, Gauthierian Contractarianism — using Huang et al.'s attribution prompts
  (Appendix B.10–B.11) and the GPT-OSS-120B scorer. **Soft distributions, not one-hot**
  (Huang p. 6 argument): pairs will rarely be framework-pure. These labels **never define
  `V_moral`** — they only decompose and interpret it.

- **Trigger logic:**
  - **G3 null:** literature connection is **positioning only**. Label a *subsample*
    sufficient to report the per-framework geometry of `V_moral` and compare against
    Huang's findings (e.g., their Step-3 utilitarian convergence; contractarianism
    instability). Worth a paragraph for framing; not urgent.
  - **G3 positive:** the decomposition becomes **load-bearing** — it localizes *which*
    ethical region the refusal overlap lives in, turning a finding into a mechanistic
    finding. Run the **full** labeling pass and the per-framework projection of the
    refusal direction.

- **Unconditional value (either branch):** describing `V_moral` in the same five-framework
  vocabulary as the live trajectory-probing literature situates DeepSteer inside a
  conversation reviewers in that subfield recognize, and makes the per-framework geometry
  directly comparable to Huang's layer-localization and transition results.

### Phase 3.5 deliverables (conditional)

- Per-framework attribution (subsample or full, per trigger).
- Per-framework geometry of `V_moral` (pairwise cosines, clustering with permutation
  tests — Paper 3 toolkit).
- If G3 positive: per-framework projection of the refusal direction (overlap localization).

---

## 8. Phase 4 — Apply to Papers 3+ (validation / sanity checks)

These de-risk Direction 1 ("do I trust this new subspace?") more than they re-litigate
the papers. **Discipline note:** a contradiction with a published finding is a **flag to
investigate, not a license to retract on the fly.** Claim discipline from Papers 5–7
stays intact. Ranked cheapest-first.

1. **Recompute Paper 5's projection fraction with `V_moral`** — *highest-value quick win,
   nearly free; this is literally Track 3 / Point A, so it falls out of Phase 3 at no
   extra cost.* If projection stays ~0.10, Paper 5 is bulletproofed against the thin-MFT
   objection. (~1 day compute.)

2. **eff_dim comparison (Paper 3)** — one number: how much richer is `V_moral` than the
   six-probe subspace (rank-3 dressed up, or genuinely rank-15)? Trivial; calibrates how
   seriously to take every projection result.

3. **MFT-cluster recovery (Paper 3) — the MFT decomposition / backward-compat layer.**
   Project the six MFT directions into `V_moral`; do they still cluster
   individualizing-vs-binding the way Paper 3 found? A faithful superset should preserve
   known structure inside it. Uses existing MFT directions and clustering code.

4. **Paper 7 trace re-projection (STRETCH — moderate, not quick).** Re-project the
   GPT-OSS-20B `trace_profile.npz` traces onto `V_moral` to see whether the mid-trace
   moral signal strengthens under the richer probe. Higher effort (reprocessing traces);
   optional — validation of richness, not a gate.

### Phase 4 deliverables

- Paper 5 projection recompute (shared with Track 3).
- Paper 3 eff_dim comparison.
- Paper 3 MFT-cluster-recovery report.
- (Optional) Paper 7 trace re-projection.

---

## 9. Phase 5 — Gate to Direction 2 — **GATE G5**

Direction 2 (trajectory of the moral representation across OLMo 3 Base → SFT → DPO → RL,
and the RL-Zero series) starts **only** when **all** of the following hold:

> **GATE G5 (conjunction):**
> 1. **G2 passed** — paraphrase gap within tolerance (the subspace reads moral structure,
>    not memorized text). *This is especially load-bearing for Direction 2:* a
>    memorization confound tolerable in a snapshot becomes finding-retracting across
>    checkpoints.
> 2. **Track 1 acceptable** — `V_moral` is no more fragile than the MFT baseline (σ*).
> 3. **G3 resolved** — refusal-overlap result settled against its null, and the **D2**
>    disposition decided.

Direction 2 is expected to be **later and incremental**, and is not specified here.

---

## 10. Pre-registered decision points (summary)

| Ref | Type | Resolution |
|---|---|---|
| **D1** | Settled | Single best-construct `V_moral`; per-framework = interpretability layer only. |
| **D2** | Settled, disposition conditional | Build alongside MFT subspace; final disposition set by **G3** (null → coexist; positive → replace + revision note, null required). |
| **G2** | Hard gate | Paraphrase gap > tolerance ⇒ stop, fix curation. Blocks Tracks 3–4, Phase 3.5, Phase 4, Direction 2. Tolerance pre-registered before running. |
| **G3** | Gate + branch | Refusal projects higher than rank-matched null (b) *and* non-moral semantic control (c)? Sets D2 disposition; triggers Phase 3.5 branch. Null + rule pre-registered before running. |
| **3.5 trigger** | Conditional | G3 null ⇒ subsample labeling (positioning). G3 positive ⇒ full labeling (load-bearing overlap localization). |
| **G5** | Conjunctive gate | Direction 2 starts only if G2 passed ∧ Track 1 acceptable ∧ G3 resolved. |

---

## 11. Out of scope (explicit)

- **Direction 2** — trajectory across OLMo 3 training stages. Gated behind G5; later and
  incremental.
- **Direction 3** — pretraining-time / training-time intervention (the ART-successor
  arc). Not started.
- Paper-numbering decisions (whether Direction 1 is Paper 8, or splits into two papers
  per the reviewer's framing). Left to the program owner.

---

## 12. Reusable repo infrastructure (for execution)

- `direction_utils` — projection / membership scoring.
- `heretic_ablation.last_token_means` — Paper 5 proto-refusal contrast (Track 3 Point A).
- `moral_dependency` hook — generation-dependency measurement (available if a dependency
  cross-check is wanted alongside the geometric overlap).
- Probe-sprint mean-diff extraction harness.
- Paper 3 clustering-with-permutation-test code (MFT-cluster recovery; Phase 3.5 geometry).
- DATASET_GUIDELINES.md audit harness (Phase 1).

---

*End of Direction 1 plan. Findings reported as "achieved" require fixture evidence in the
repo; everything here is a plan and may change based on results.*
