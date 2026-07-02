# Direction 2 — Decision-Coupling Results

**Date:** 2026-07-02. Panel: OLMo-3-7B-Instruct / Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct
(headline layers 16 / 14 / 16). Methods + the full amendment trail in `PREREGISTRATION.md`
(Amendments 1–2); the massive-activation and decision-site-bottleneck methods findings in
`../ANOMALIES.md`. All numbers are position- and format-typed (`{format, position_class,
participation_ratio, σ_provenance}`), the discipline this program had to build to get here.

## Headline: the refusal decision is orthogonal to the moral-judgment decision, and it is architecturally so

The decision-vs-decision cell that Papers 5–7 and D1 never measured resolves to **dissociation**,
and the in-format ladder shows *why* it is not a coincidence.

**R3 — decision-level dissociation, panel-wide.** The model's own moral-judgment **decision**
direction (within-ground-truth-label contrast at the decision site) is ~orthogonal to its refusal
**decision** direction, judged against a covariance-matched pairwise-cosine null in the
format-matched space:

| model | |cos|(refusal, judgment-decision) | pairwise-null q95 | verdict |
|---|---:|---:|---|
| OLMo-3 | 0.10 | 0.41 | dissociation |
| Qwen2.5 | 0.32 | 0.42 | dissociation |
| Llama-3.1 | 0.08 | 0.51 | dissociation |

The OLMo raw→standardized invariance check passes (same verdict both ways), the legitimacy proof
that the standardization is not manufacturing the result.

## The decision site is a low-dimensional control-token bottleneck (cross-model)

The reason the two decisions cannot geometrically couple is architectural. The chat decision-site
token (`final_pre_assistant`, the assistant-header position where both the refusal gate and the
judgment-decision direction live) is a **~10–15-effective-dimension control channel on every model**:

| model | decision-site PR | content-position PR (mean/last) | decision-site valid? |
|---|---:|---|---|
| OLMo-3 | **14.7** | 40 / 63 | no (band [0.40,0.47] < null 0.557) |
| Qwen2.5 | **8.6** | 33 / 42 | no |
| Llama-3.1 | **10.2** | 35 / 99 | no |

At that position the positive-control moral band sits **below** the covariance null (moral
directions project onto their own span *less* than random), so the projection instrument has no
power there — the general tell **band-below-null ⇒ position-invalid** (`ANOMALIES.md` A2). Three
independent estimates agree the channel is ~15-dimensional (`√(3/14.7)=0.45` ↔ null 0.557 ↔ the R3
pairwise-null 0.41–0.51). This was caught by Phase A's positive-control band; without it a false
"register-scoped" claim would have shipped.

**Consequence (the mechanism, stated once).** Stacking the calibration findings: refusal is a
**fresh** post-training gate (A5: cos 0.155 to its pretraining precursor), in a **low-variance**
channel (A3: ≤ q10 of activation variance), at a **~10–15-dim control-token bottleneck** that moral
content demonstrably does not reach — while moral content stays at content positions. So
**content-vs-decision geometric orthogonality is architecturally guaranteed**, and any
comprehension→decision coupling has to be carried by the **attention heads that write into the
bottleneck**. That is a concrete anatomical target for the C1 follow-up, not a fishing expedition.

## `V_moral` is format-robust — "register-scoped" retracted

At the valid content position (`mean_content`, the chat analog of the raw mean-pooled `V_moral`),
the moral-family band matches the raw band on all three models: OLMo [0.54,0.64] ≈ raw [0.52,0.64];
Qwen [0.47,0.57] ≈ [0.46,0.56]; Llama [0.50,0.56] ≈ [0.44,0.54]. The moral content subspace is
**register-consistent**. An earlier `format_robust=false` reading at the decision site was a
position artifact (that token is the invalid bottleneck), now retracted.

## R2 / G3 are not well-posed cross-position — and that *is* the finding

At the decision site the refusal gate projects 0.50 / 0.61 / 0.48 and the judgment-decision 0.26 /
0.46 / 0.23 onto the in-format `V_moral`, but that position is the invalid bottleneck (null
0.53–0.57), so these are **reported, not verdicts**. The decision directions live only at the
control-token bottleneck; the content subspace only at content positions; **they do not coexist at
any single valid position.** R2/G3 (a decision direction onto a content subspace) are therefore
**not well-posed cross-position** — which is the mechanism, not a limitation. The coupling question
is answered where it *is* well-posed: R3 (a decision-vs-decision cosine), which says dissociation.

## Cross-model instrument robustification (a reusable methods contribution)

Two degeneracies had to be handled for the cross-model panel, neither present in the OLMo-only
D1 work; both are filed as methods findings in `../ANOMALIES.md`:
1. **Massive-activation outlier dims** (Qwen dim 458 = 59% of variance, Llama dim 788 = 32%)
   saturate the covariance-matched null → standardized geometry (with an OLMo invariance check +
   a projection-out robustness variant) restores it. Paper 6 back-audit: clean (it never used this
   null). 
2. **The decision-site control-token bottleneck** (this work): a position, not a model, degeneracy,
   caught by the same positive-control band.

## Secondary + reconciliation status

- **Cross-ablation (R3 causal arrow)** was underpowered/unfalsifiable at first (base refusal 0.167
  on 24 prompts, a floor, disagreeing with Paper 6's ~0.575). The harness is reconciled (shared
  `_classify_response` classifier + full harmful set); the reconciled cross-ablation is queued with
  the B1 re-run.
- **R5 (refusal vs non-moral controls)** was the cell where standardization and projection-out
  disagreed for Qwen/Llama; the in-format ladder is the discriminator, and it shows the disagreement
  came from the raw-format decision-site confound (the content subspace itself is format-robust).

## What remains (all now well-targeted, none blocking the headline)

- **C1 — the real next experiment:** attention-head attribution / a Jacobian of the refusal readout
  at the bottleneck, to find the heads that write the decision-site token (the anatomical target this
  work handed it).
- **B5 (R8 moral-fragility baseline):** the standing metric for a future Direction-2 intervention;
  injects the `mean_content` `V_moral` (a mini-C1). Not needed for the headline.
- **D1 reasoning band-rung PR audit:** the GPT-OSS P2-vs-band rung is cross-position (P2's
  null-relative statement stands; the band-relative one is scoped) — queued in `MISSING_ARTIFACTS.md`.
