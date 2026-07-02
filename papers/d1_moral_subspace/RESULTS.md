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
- **Both points project below the moral-family band** (the A1 held-one-out positive control;
  `CALIBRATION_RESULTS.md`). A genuinely-moral direction held out of `V_moral` projects back onto it
  at **0.54–0.66** (base band), so the band is the calibrated yardstick for "moral-adjacent." The
  claim rides on the ladder **null → refusal → band**: refusal sits at or below the null and well
  below the band, i.e. *less* moral-adjacent than a held-out moral direction. The persona axis
  (0.51) is shown **for reference only**: calibration finds persona sits just below the band
  (a moral-adjacent voice reference, not a clean non-moral control), so the strong-form "below a
  known non-moral axis" statement is deferred to the genuinely-non-moral syntax / register controls
  in Direction 2 (R5).
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

Calibration (A1) adds the converse check: the 6-foundation MFT span and the rank-3 `V_moral`
project onto each other at **0.56 / 0.62** (base / instruct), inside the moral-family band and well
above persona. So the two subspaces measure related-but-distinct moral content, neither nesting in
the other. This **retro-validates Paper 5's instrument choice**: refusal's low projection onto MFT
was not an artifact of projecting against an outlier or degenerate subspace, because MFT itself
projects onto an independently-built moral subspace at genuinely-moral magnitude.

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

2. **The calibrated band recontextualizes P2 (persona is now reference-only).** The A1 held-one-out
   positive control gives a **moral-family band** per model (GPT-OSS [0.65, 0.76];
   `CALIBRATION_RESULTS.md`), and the orthogonality claim rides on the ladder **null → P2 → band**.
   GPT-OSS P2 = 0.52 crosses its null yet sits **below the band**: in-trace deliberation is
   voice-adjacent, not moral-content-adjacent. The pre-registered paired test Δ = band-min − P2
   confirms sub-band at every position on both models **except the single GPT-OSS P2 window point**,
   where the primary (percentile) Δ-CI includes 0 (Δ̂ = 0.13). That split is the pre-registered
   **band-min attenuation** — band-min is a min-of-three statistic, downward-biased under resampling,
   a bias whose direction *favors* the sub-band claim — and the bias-corrected BCa Δ-CI plus the
   `P2_FULL` robustness point both exclude 0; the one window point **locks at B3** once GPT-OSS's
   small-n axis pairs are re-extracted. Persona (0.60) is shown **for reference only**: calibration
   reclassifies it as a **moral-adjacent voice** axis (it sits just below the band), so the
   genuinely-non-moral controls (syntax / register) that would carry the strong-form statement are
   deferred to Direction 2 (R5). The subspace is genuinely moral either way: GPT-OSS's `V_moral`
   cleanly separates moral/neutral (acc 0.67) with distinct source axes (cos 0.46–0.66), and the
   elevated persona there reflects general entanglement in GPT-OSS's space (moral↔persona cos 0.30
   vs OLMo's 0.24). Combined across both independently-trained reasoning models (so Fisher
   independence holds), the in-trace P2 exceeds the **random** null at **p ≈ 1e-4** (EXPLORATORY):
   the peak is a real above-chance effect, and still sub-moral-adjacent — real, not noise, and not
   moral-content coupling.

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
while remaining below the moral-family band. This closes Direction 1's model axis; a training/
pretraining intervention is the next program (Direction 2), not the next section.

## Mechanism: refusal is a fresh, low-variance gate

Two calibration findings (A3, A5; `CALIBRATION_RESULTS.md`) converge on *why* refusal reads
orthogonal to the moral subspace. **A3 (spare-channel):** the wired refusal gate lives in a
low-variance channel — the instruct gate and all four GPT-OSS positions project onto activation
directions at or below the 10th percentile of variance among covariance-matched randoms, while the
`V_moral` axes and persona occupy ordinary-to-high-variance channels. **A5 (no crystallization):**
the base proto-refusal is nearly orthogonal to the wired instruct gate (`cos = 0.155`), so refusal
does *not* crystallize from a pretraining precursor the way the moral subspace does
(`cos(base, fresh) → 0.999` in Paper 5). The base proto-refusal is also *not* low-variance
(percentile 37), so the narrow channel is a property of the **post-training-wired** gate, not the
precursor. Together: refusal is a **freshly-built, low-variance add-on gate**, not a
moral-content-derived direction. This mechanistically explains both its easy ablation (Heretic) and
its at-or-below-null projection onto `V_moral`, and it is the constructive form of the orthogonality
result — the refusal direction is the kind of object (narrow, late, non-crystallized) that *would*
read orthogonal to a distributed moral representation.

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
