# Self-review (hostile-reviewer pass) — FL & MN — 2026-07-04

Two adversarial reviews, each reading only the delivered PDF (external-reviewer frame). Both
verdicts are **reject/borderline-reject in current framing**. The shared root cause is
**overclaiming relative to the evidence**; most fixes are honest scoping, not new results.
Triage tag per finding: **[AUTO]** = applied this pass (unambiguous, backed by the paper's own
text); **[ESCALATE]** = thesis/framing/data decision for Orion; **[POD]** = needs experiments
this frozen phase cannot run (the honest response is to scope the claim).

---

## FL — "Refusal Reads the Harm Percept, Not the Moral Subspace"  (borderline → reject in framing)

**Single most damaging objection:** the paper's own Table 1 shows Llama refusal reads *broad
moral content* by interchange (transfer 0.85 ≈ judgment 0.79), which falsifies the title as a
general claim; the causal rank sweep is one model (OLMo, n=23) at the detection floor, GPT-OSS is
correlational, Qwen never appears on the read axis. The titular finding is causally supported on
exactly one model and contradicted on another.

### Blocking
- **Title over-generalizes a single-model result** (contradicted by Llama). → **[ESCALATE]** the
  title is yours. Reviewer options: *"Refusal Reads a Low-Rank Slice of Moral Content"*, or demote
  to *"On OLMo-3, refusal reads the harm percept"* + make §8 explicitly a variation study.
- **"Establish across four models" overstates** — only OLMo is causal. → **[AUTO]** soften the
  abstract/§1 to "the causal test is on OLMo (n=23 request-twins); the panel is a cross-architecture
  consistency check with one dissenting read (Llama)"; **[ESCALATE]** whether to rescope the whole
  contribution to single-model-causal.
- **Panel is model-level selection-on-outcome** (nine models, each placed where its cell works). →
  **[AUTO]** state the panel-construction honestly (each model measured on the cells it supports);
  **[POD]** the full grid on every model is a future experiment.

### Should-fix
- n=23 underpowered, effects near the MDE. → **[AUTO]** report MDE at n=23 in-text (already in App C);
  **[POD]** more twins.
- **76% of refusal's causal input is OFF the moral subspace** — "a slice OF the subspace" hides this.
  → **[AUTO]** reframe: "a mostly-extra-moral harm direction that clips a rank-1 corner of the subspace."
- Llama/OLMo asymmetry "collapse" rests on overlapping CIs, no difference-CI; layer 12 may be
  outcome-selected. → **[AUTO]** soften "+0.26 difference" to note the layer-12 CI includes 0 and rest
  the finding on the reads-axis; **[ESCALATE]** whether layer-12 was pre-registered as pre-commitment.
- Position-validity contradiction (content-extracted directions restrict a decision-channel patch at
  the position the gate calls invalid for content). → **[ESCALATE]/[POD]** add a transport positive
  control; for now **[AUTO]** add a sentence acknowledging the caveat.
- Harm-direction extraction position ambiguity (instruction-token vs decision-channel). → **[AUTO]**
  specify the position of the harm-rank-1 direction.
- Refusal & harm directions built from overlapping partitions ⇒ "reads harm" partly by construction
  (GPT-OSS cos 0.977 near-tautological). → **[AUTO]** report the genealogy + call the GPT-OSS result
  near-collinear; **[POD]** decorrelating control (harmful-but-complied / benign-but-refused).
- **"Hard to deepen" asserted but §9 says it can't be answered.** → **[AUTO]** remove from abstract/
  conclusion or relabel as a forward hypothesis everywhere (self-contradiction; unambiguous).
- "Low-rank read → removable" link is n=1 in an appendix. → **[AUTO]** soften; **[POD]** promote +
  add models.
- Load-bearing validity deferred to the "companion methods note (in preparation)". → **[AUTO]** keep
  the pointer but ensure the in-text null verdicts carry their own bars (App B already does).
- Compositionality (§3) rests on two unpublished self-cites. → **[ESCALATE]** publish P3 or fold the
  numbers into an appendix.
- Ladder lacks a non-moral positive-projection control (persona is "moral-adjacent"). → **[POD]**
  add sentiment/register control; **[AUTO]** state the limitation.
- Qwen orthogonality margin thin (0.32 vs 0.42) + standardization-dependent. → **[AUTO]** soften
  "orthogonal on every model" to note Qwen's tighter margin.
- GPT-OSS harm read is a *standardized* cosine (3.8× sharpening). → **[AUTO]** report raw (0.57/0.22)
  beside standardized (0.49/0.13).
- OLMo commitment cell is interchange-only (barely refuses) but Table 1 lists it on the behavioral
  axis. → **[AUTO]** mark it interchange-only / low behavioral dynamic range.
- Reversibility is n=7/n=10 on one model + prefill-last-token caveat. → **[AUTO]** foreground the n;
  **[POD]** run the graded test on OLMo/Llama for contrast.

### Minor / judgment (selected)
- "Saturates at 0.31" is peak-at-k=3-then-decline. → **[AUTO]** "peaks at k=3 (0.31), flat 0.26–0.27".
- "Judgment reads broadly" is 0.66 (two-thirds), not the whole subspace. → **[AUTO]** anchor it.
- "−21σ" uses a random-ablation null on a structured ablation. → **[AUTO]** "far outside the
  random-ablation band" (drop the σ count).
- one-knob "RMSE 0.036" residuals are 13–16% at two points. → **[AUTO]** report per-point residuals.
- "the moral subspace" (definite article, in the title) asserts canonicity from 3 datasets. →
  **[ESCALATE]** (title); **[AUTO]** hedge once in §2 ("a moral-content subspace").
- crystallization threshold cos=0.50 unmotivated. → **[AUTO]** motivate against the null or drop the line.
- d′/reply-inversion used before definition; Llama-3.1 gated. → **[AUTO]** define at first use; note gating.

---

## MN — "Calibrating Interpretability Instruments Before Trusting Their Verdicts"  (reject in current form)

**Single most damaging objection:** the "novel failure modes" are largely re-derivations of known
cautions the note neither cites nor positions against, while the empirical backing for five of six
modes lives in uncited/unpublished files — so a reviewer cannot establish novelty or correctness
from the page.

### Blocking
- **Novelty / prior art** (cites only 3 works). → **[AUTO]** add a related-work section positioning
  each mode against: rogue/outlier dimensions + standardization (Timkey & van Schijndel 2021;
  Kovaleva 2021; Dettmers 2022), norm-folding for attribution (Elhage 2021 / TransformerLens;
  Belrose Tuned Lens 2023), activation-patching saturation & depth confounds (Zhang & Nanda 2024),
  subspace interpretability illusions (Makelov, Lange & Nanda 2023; Bolukbasi 2021). Sharpen what is
  genuinely new (reordered-norm 3× overshoot + exact fold; covariance-*matched*-null degeneration as
  a named instrument). **Citations must be verified before adding.**
- **Single-program generality.** → **[ESCALATE]** honest rescope ("lessons from one program") vs.
  external validation; **[AUTO]** hedge "protocols others can run" in the meantime.
- **Verifiability** (5/6 modes' numbers uncited/unpublished). → **[ESCALATE]/[POD]** release the arrays
  or cite the source papers; **[AUTO]** state the scope honestly.

### Should-fix
- **PR<30 gate not dimension-normalized, panel-fit, retuned to 25 for GPT-OSS.** → **[AUTO]** present
  it as PR/d or a null-referenced quantile and explain the dense-vs-MoE difference, or mark the number
  as panel-descriptive not prescriptive.
- **Fig 1 / PR values have no error bars.** → **[POD]** bootstrap CIs; **[AUTO]** add a caveat that the
  PRs are point estimates + a PR-by-(model×position×normalization) table (fixes the proliferating-PR
  confusion the reviewer flags separately).
- §6.1 reflexive case doesn't show the protocol firing prospectively (it was a fresh re-read). →
  **[AUTO]** separate "fresh-context re-reading is valuable" from "the protocol would have flagged this".
- Depth section's surviving claim has no CI (A@12 includes 0). → **[AUTO]** same fix as FL.
- §2.1 reframe draws a functional conclusion from a cosine at a position it just called invalid. →
  **[AUTO]** downgrade to a geometric observation + note a causal cell is needed.
- Reproducibility ships only 3 summary CSVs. → **[ESCALATE]/[POD]** release per-unit arrays.
- "Six across four" overclaims (each mode is 1–2 models). → **[AUTO]** state per-mode coverage.
- §3.1 reply-inversion lacks the specificity null the note itself mandates. → **[AUTO]** note the missing
  control / soften.
- §6 reads as an internal changelog; central objects undefined. → **[AUTO]** add a notation/definitions
  block (participation ratio formula, the sweep quantities) + define terms at first use.

### Judgment / minor
- √(3/PR) heuristic uncited + compares median-scale to q95. → **[AUTO]** derive/cite or drop "converge".
- §2.3 "our own cells were never at risk" reads as special pleading. → **[AUTO]** state up front which
  null each cell class uses.
- AI-scaffolded instruments, no independent re-implementation described. → **[ESCALATE]** (disclosure).
- small-n behavioral core (n=7–10). → **[AUTO]** foreground.
- "harm_saturating" one-knob / PC1-inert stated without CIs. → **[AUTO]** mark illustrative or add CIs.

---

## What is being applied this pass (AUTO) vs. left to Orion (ESCALATE/POD)

**Applied autonomously:** the self-contradiction fixes ("hard to deepen"; OLMo commitment cell),
anchor corrections (0.66 not "broad"; peak-not-saturate; −21σ; raw+std GPT-OSS), honest scoping of
the abstract's "four models", the 76%-off-subspace reframe, Qwen-margin softening; and for MN the
related-work section (verified cites), a notation/definitions block, a limitations subsection,
the PR-by-position table, enumerate-the-six, and the overclaim softenings.

**Escalated to Orion (thesis/data):** the FL **title** and whether to rescope the contribution to
single-model-causal; the MN **rescope vs. external-validation** decision; publishing P3 (the
pretraining-duo cite); **releasing the underlying arrays** for reproducibility; and the **[POD]**
experiments (full model grid, more twins, decorrelating/non-moral controls) that a frozen phase
cannot run — for those the honest move is to scope the claims, which the AUTO edits begin.
