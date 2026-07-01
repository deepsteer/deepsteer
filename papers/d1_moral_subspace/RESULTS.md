# Direction 1 — Results

**Date:** 2026-06-30. Substrate: OLMo-3 7B (Base `allenai/Olmo-3-1025-7B`, Instruct
`allenai/Olmo-3-7B-Instruct`, layer 16), extended to two reasoning models —
`allenai/Olmo-3-7B-Think` (layer 16) and `openai/gpt-oss-20b` (layer 12, mxfp4→bf16, harmony).
transformers 5.12.1. Methods + the full amendment trail in `PREREGISTRATION.md`.

## Headline (GATE G3): refusal is orthogonal to the rich rank-3 moral subspace

`V_moral` is the orthonormalized span of three distinguishable moral mean-diff directions —
Moral Stories (explicit action-contrast), Understanding Fables (abstract moral inference),
and ETHICS commonsense (everyday judgment) — constructed exactly like the MFT foundation
span, so the projection is directly comparable to Paper 5's 0.1044 (refusal onto the rank-4
MFT span). The three axes are genuinely distinct: `cos(d_fables, d_moral)=0.53`,
`cos(d_ethics, d_moral)=0.36`, and the set spans **effective rank 3**.

Two same-model refusal points, each projected onto its model's rank-3 `V_moral`, with the
rank-matched null + persona control **recomputed on this span** (two-step: before the refusal
projection):

| Point | refusal p | null q95 | persona c | verdict |
|---|---:|---:|---:|---|
| A — base proto-refusal × base `V_moral` | **0.33** | 0.31 | 0.51 | NULL |
| B — instruct gate-refusal × instruct `V_moral` | **0.14** | 0.26 | 0.51 | NULL |

**G3 = NULL.** Across the rank-sweep (1→2→3 sources) refusal tracks the rank-matched null at
every rank, for both points (never clearing `q95 + 0.05`).

### Reading (calibrated claims)

- **Refusal projects at or below the rank-matched null** onto `V_moral` (base ≈ null; instruct
  below null). This is a magnitude-like projection fraction: "at or below the null" means
  *less than chance alignment*, **not** signed opposition — no "anti-alignment" is claimed.
- **Both points project below the persona reference (0.51).** Persona is a valid *non-moral*
  baseline: prior work finds the persona/assistant axis orthogonal to moral directions
  (|cos| ≈ 0.076–0.085). So refusal aligns with `V_moral` *less than a known non-moral
  direction does* — the strong form of the orthogonality claim, grounded in that reference,
  not merely asserted.
- **Point B (instruct gate) ≈ Paper 5's 0.1044.** The actual instruct-time refusal gate
  projects 0.14 onto the rank-3 instruct `V_moral` — the same ~0.10–0.14 regime Paper 5
  measured against the thin rank-4 MFT subspace.

### The thin-MFT objection is closed — judged-vs-judged (D2: coexist, "orthogonality robust")

The single most dismissible weakness of the comprehension–compliance arc was "refusal looks
orthogonal because your moral subspace is six thin MFT probes." Direction 1 answers it with the
**same instruct refusal gate judged against each subspace's own rank-matched null**: it projects
**0.144 onto the rich rank-3 `V_moral`** (null q95 0.266) and **0.155 onto the 6-foundation MFT
span** (null q95 0.252) — **both NULL, both below chance**. Paper 5's 0.1044 was a *raw* number,
never null-judged; the 0.155 here reproduces its magnitude *and* shows it sits below the
rank-matched null. Refusal is orthogonal to *both* the thin and the rich subspace, fairly judged.
The objection is closed airtight. Per the pre-registered D2 rule, `V_moral` and MFT coexist.

**On "richer" — not "higher-dimensional."** The rich subspace is *lower*-dimensional than MFT on
both measures: **3 source directions vs MFT's 6 foundation directions** (the null-matching
projection bases), and **eff-dim 3 vs 4** (the variance rank of the pooled diffs). Its
contribution is **construct-diversity + verified-distinguishability + contamination-resistance**,
not more dimensions; "richer" never means "more complex." And it is no more fragile: Track-1 σ\* on
the **rank-3 headline instrument** (not the dropped single-source) is **4.86 / MFT 0.0**
(narrative) and **9.56 / 9.56** (declarative), RMS-normalized — as robust, at lower dimension.

### Scope boundary (stated, not left for a reviewer)

Orthogonality is established against **one** rank-3 multi-source subspace, **robust across ranks
1–3** (the rank-sweep varies *rank*, not *construction*) and — via the reasoning-model extension
below — **across two independently-trained reasoning models and four positions along the reasoning
chain**. What remains future work: generalization across **alternative rich-subspace
constructions** (a different source set), and whether a **training intervention** can make the
moral subspace absorb the refusal direction (Direction 2).

**Instrument limitation (same-layer projection).** Paper 1's storage/usage layer divergence
applies reflexively to this measurement: a **same-layer** projection of refusal onto `V_moral`
cannot detect a refusal circuit that reads moral features **through weights** from other layers.
Every G3 number here compares two directions extracted at the same layer, so it bounds
*representational* overlap at that layer, not a computational read across layers. The
decision-level and cross-layer follow-ons (`../d2_decision_coupling/PREREGISTRATION.md`, Phases
B/C: the moral-judgment decision direction, the outcome-conditioned in-trace contrast, and the
refusal-readout Jacobian w.r.t. the moral-usage layer) measure that read directly. This scope
note does not change any G3 verdict; it states what the same-layer instrument can and cannot see.

## Reasoning-model extension: a gradient across the chain, replicated across labs

Orthogonality holds at the instruct gate; does it survive when a model **reasons explicitly**
about harm? G3 is extended to two independently-trained reasoning models — **OLMo-3-7B-Think**
(Ai2) and **GPT-OSS-20B** (OpenAI) — with refusal measured at four positions along the reasoning
chain: **P0** the harm-recognition site (`t_inst`, Zhao et al.), **P1** the pre-trace gate, **P2**
the **in-trace deliberation** (a symmetric first-N-reasoning-token window, so the diff-of-means is
opening deliberation and not a span-length contrast), and **P3** the post-answer decision site.
Each model's `V_moral` is re-extracted fresh in its own space (directions do not transfer); the
null and persona control are recomputed on each span before any refusal projection.

**A gradient that peaks in-trace, robust across both models.** In both, the projection is smallest
at the gate and **largest at the in-trace site**:

| model (layer) | P1 gate | P0 harm | **P2 in-trace** | P3 post-answer | null q95+M / persona c+M |
|---|---:|---:|---:|---:|---|
| OLMo-3-Think (16) | 0.10 | 0.29 | **0.35** | — *(unmeasured, benign)* | 0.354 / 0.575 |
| GPT-OSS-20B (12) | 0.19 | 0.47 | **0.52** | 0.25 | 0.386 / 0.653 |

Explicit moral deliberation is consistently where refusal comes **closest** to the moral subspace.
`P2_FULL` (the full-span mean) agrees with the window in both models — the peak is not a
window artifact.

**Two nuances, stated plainly (not smoothed into a flat NULL).**

1. **Cross-model null-crossing.** OLMo's in-trace projection (0.35) sits **~0.009 below** its
   rank-matched null — a near-miss. GPT-OSS's (0.52) **crosses** its null (0.39). So in a
   differently-trained model the in-trace alignment is **stronger** — enough to exceed chance. The
   gradient doesn't merely replicate; it strengthens.

2. **Persona-control-binding in GPT-OSS.** The pre-registered verdict is **NULL in both** — but for
   GPT-OSS the two controls **disagree**: P2 crosses the null yet stays below the persona
   (non-moral) control (0.60). So the NULL rests entirely on the persona control. This is the
   conjunction rule doing its designed job: **"below a non-moral semantic axis" is a stronger
   orthogonality statement than "below random."** Its validity was **confirmed, not assumed**:
   GPT-OSS's `V_moral` still **cleanly separates moral/neutral (acc 0.67, above OLMo's 0.64;
   `gpt_oss/subspace_purity.json`)** and its three source axes stay distinct (cos 0.46–0.66, below
   the 0.85 collinearity floor), so the subspace *is* isolating moral content. The high persona
   value reflects **general entanglement** in GPT-OSS's space (moral↔persona cos 0.30 vs OLMo's
   0.24) that inflates *all* projections — so refusal-below-persona is genuine orthogonality
   evidence, not a shared-contamination artifact.

**P3 is measurable for GPT-OSS** (0.25, NULL) but **unmeasured for OLMo-3-Think**: OLMo's benign
reasoning exceeds the generation budget without reaching a post-answer state (a reasoning-verbosity
*constraint on the contrast*, not a null); GPT-OSS reaches its final channel, so its decision site
is measured.

**The gradient is a novel measurement.** MFT was only ever measured at the single gate position
(Paper 5; Base/Instruct have no reasoning traces), so this is **a gradient the earlier MFT work
didn't measure** — not a contrast MFT lacked.

**Terminal disposition.** Refusal is orthogonal to the moral representation **across the reasoning
chain**, with a characterized gradient peaking in-trace that **replicates across two
differently-trained reasoning models** and is strong enough in one to cross the rank-matched null
while remaining below a non-moral control. This closes Direction 1's model axis; a training/
pretraining intervention is the next program (Direction 2), not the next section.

## Why this isn't a subspace reverse-engineered to the answer

The sharp challenge to a result like this is: *you built the subspace, then measured refusal
against it, so did you just find a subspace that gives the orthogonality answer you wanted?*
The defense is that the test was frozen in `PREREGISTRATION.md` (commit `107f9f3`) before any
`V_moral` existed and before any refusal vector was measured. Three quantities were fixed in
advance:

1. **The G3 decision rule and margin were fixed pre-data.** `M = 0.05`; G3 is POSITIVE only if
   refusal clears BOTH the rank-matched null AND the persona control by `M`, for BOTH refusal
   points; otherwise NULL. The conjunction, the two bars, and the margin were all committed
   before the subspace was built. The rule **named NULL the more publishable outcome in
   advance** (orthogonality robust across operationalizations, the strengthening of Papers
   5–7), so the pre-registration did not lean toward a positive find.
2. **The null and control are computed mechanically from the subspace, not chosen.** `q95` (the
   rank-matched null) and `c` (the persona control) are a deterministic function of `V_moral`'s
   geometry, produced by a recipe frozen pre-data (covariance-matched random directions at the
   realized rank). In the run they are realized from the constructed `V_moral` and evaluated
   **before** the refusal vector is projected; the refusal projection never enters the null. The
   bar refusal had to clear was set by the subspace, not by the answer.
3. **The G2 tolerance was fixed pre-data**, grounded in Paper 1's measured transfer regimes
   (structural reading loses ≈1 pp, lexical lookup ≈25 pp), before the paraphrase set was
   scored.

What was **not** pre-registered is the rank and construction of `V_moral`: that changed once,
mid-program, from single-source to a rank-3 multi-source span. The next section shows that
change was forced by a discovered property of the difference vectors, not by the answer it
produced.

## Two findings the work produced

These are results in their own right, independent of the orthogonality headline. Each also
redirected the `V_moral` construction; the framing below leads with what was discovered, and
the redirection follows from it.

1. **Moral salience from a single contrastive source is rank-1, not a rich subspace.** One
   source (Moral Stories) yields a single dominant moral direction carrying 7.5% of the
   per-pair-difference variance, atop a flat content tail; there is no low-rank *moral* subspace
   inside one source. This is a fact about how contrastive moral signal is distributed, and it
   is what **invalidated the assumption that one source could stand in for "the moral
   subspace"** and forced the move to multiple distinguishable constructs (+fables +ETHICS →
   rank 3).
2. **Effective-dimensionality thresholding on per-pair difference vectors measures content rank,
   not moral rank.** The pooled-diff spectrum is elbow-less and content-dominated (singvals 31,
   18, 15, 14, … flat), so "uncentered eff-dim @ 0.90" gives **385** (≈10% of the 4096-dim
   space). At that rank the subspace is degenerate **for every direction**: refusal, persona,
   and random all project ~0.7–0.8, so it discriminates nothing regardless of what refusal does.
   The defect is symmetric, which is exactly why noticing it is not motivated by the
   orthogonality answer. This is a real caution for anyone building "moral subspaces" by SVD on
   contrastive diffs, and it is what **invalidated the original §5 eff-dim spec**: the rank-3
   moral structure does not live in the 0.90-variance subspace at all, it lives in the source
   mean-diff directions, so `V_moral` was re-spec'd to their span.

## Secondary results

- **GATE G2 (contamination) = PASS** on the Moral Stories narrative slice: `acc_surf 0.667 /
  acc_para 0.677`, gap −0.011. The moral direction reads structure, not memorized text.
- **Multi-source G2 coverage (OLMo-3 base, layer 16): every source clears the contamination
  concern, none reads memorized surface text.** Each source's held-out eval pairs are
  projected onto that source's own frozen base mean-diff direction; held-out separation is
  verified disjoint by id (see the G2/G3 distinction below):

  | source | slice | n | acc_surf | acc_para | gap | verdict |
  |---|---|---:|---:|---:|---:|---|
  | fables (held-out) | narrative, GATED | 15 | 0.967 | 0.967 | +0.000 | **PASS** |
  | ETHICS (held-out) | declarative, informative | 197 | 0.701 | 0.761 | −0.061 | clears |
  | ETHICS (extraction pairs) | declarative, diagnostic | 115 | 0.787 | 0.813 | −0.026 | clears |

  The fable narrative slice (the only gated one; the 0.10/0.60 threshold is
  narrative-calibrated) passes outright. Both ETHICS declarative slices show a **negative**
  gap (paraphrase accuracy ≥ surface), the strongest anti-contamination signal: the direction
  reads the moral structure, which survives paraphrasing rather than degrading. The ETHICS
  held-out set (199 train-split pairs, disjoint from the 118 that produced `d_ethics`) is the
  genuine held-out G2; the 118 extraction pairs are reported separately as the extraction-pair
  paraphrase-gap diagnostic, and also show no surface memorization (−0.026).
- **Named finding — the moral subspace is more noise-robust than the MFT baseline on narrative.**
  Track-1 σ\* (RMS-normalized, smallest noise scale where transfer accuracy drops below τ = 0.6),
  measured on the *published* rank-3 span (`phase2_track1_rank3.py`), not the dropped single-source:
  **narrative σ\* = 4.86 for `V_moral` vs 0.0 for the 6-foundation MFT subspace**; declarative
  **9.56 / 9.56** (tied). On narrative text the MFT probe collapses at the smallest noise it is
  tested at while `V_moral` survives to σ\* = 4.86, so the rich rank-3 subspace is not only no more
  fragile than MFT, it is materially more robust in the register that dominates its construction.
  This is promoted here from a parity footnote to a named result.

## G2 ↔ G3 distinction (do not conflate)

The G3 headline rests on the **source mean-diff directions** (extracted from the
calibration/probe pairs) + the rank-matched null — *not* on the per-source eval sets.
Multi-source G2 tests contamination of each source's **training pairs** and is a
*dataset-completeness* statement. **A soft G2 on any one source is about that source's
training-pair contamination, not a threat to the settled G3 headline.** If a G2 number comes
in soft, the relevant follow-up is a paraphrase-gap check on the pairs the *direction* was
extracted from — a separate question from dataset-completeness G2.

In the event, no source came in soft: every held-out G2 cleared, and the ETHICS extraction-pair
check (the −0.026 diagnostic above) confirms the direction is not memorizing its own training
pairs either. So the distinction did not have to be invoked to defend the headline; it is
recorded as the rule that *would* apply.
