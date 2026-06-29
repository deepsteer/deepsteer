# Direction 1 — Results

**Date:** 2026-06-28. Substrate: OLMo-3 7B (Base `allenai/Olmo-3-1025-7B`, Instruct
`allenai/Olmo-3-7B-Instruct`), layer 16, transformers 5.12.1. Methods + the full amendment
trail in `PREREGISTRATION.md`.

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

### The thin-MFT objection is closed (D2 disposition: coexist, "orthogonality robust")

The single most dismissible weakness of the comprehension–compliance arc was "refusal looks
orthogonal because your moral subspace is six thin MFT probes." Direction 1 answers it
directly: it builds a **richer** subspace — rank-3, three distinguishable moral *constructs* —
verifies the added richness (the axis test), and finds **refusal still orthogonal**, at the
same level as the thin MFT subspace. Orthogonality is robust to the operationalization of
"moral." Per the pre-registered D2 rule, `V_moral` and MFT coexist; the headline is the
strengthening of Papers 5–7.

### Scope boundary (stated, not left for a reviewer)

Orthogonality is established against **one** rank-3 multi-source subspace and is **robust
across ranks 1–3** (the rank-sweep). The sweep varies *rank*, not *construction*, so the
honest scope is: **holds against this rank-3 multi-source `V_moral`, robust across rank;
generalization across alternative rich-subspace constructions is future work.**

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
- **Track-1 (σ\*)**: single-source `V_moral` is no more fragile than the MFT baseline
  (RMS-normalized).

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
