# W4 results (Amendments 14–17), Session W4-3, 2026-09-13

Verdicts for the pre-registered W4 pod (D3 PREREGISTRATION Amendments 14 and 15, riders of 2026-09-12,
Amendments 16 and 17 committed before this file). Every number here traces to a file under
`deepsteer/supplement/cells/w4/` (committed JSON summaries) or to a raw array listed in
`manifest_w4.json` (local, Zenodo in this session). Rules are applied as written; where a rule was
forked (Amendments 16.3, 17.1–17.3) the verdict is reported under both choices and the choice is
escalated. No number in this file is quoted in a paper until its CLAIMS row is updated.

**Provenance.** Pod 5jplj3beawc6kk (A100-SXM4-80GB, 2026-09-12, run `w4_20260912T190441`, driver
commit 53eb783) and rerun pod 74de8jk0usv335 (2026-09-13, run `w4_20260913T004257`, driver commit
ba48cd9), merged into one manifest (56 artifacts, sha256 each, HF commit hash per load, verified).
Model loads: OLMo-3-7B-Instruct 6e5971d, OLMo-3-7B-Think d97e442, OLMo-3-1025-7B a81bae4,
Llama-3.1-8B-Instruct 0e9e39f, gpt-oss-20b 6cee5e8, Qwen2.5-7B-Instruct a09a354. fp16 for the
dense models, bf16 (mxfp4 dequant) for GPT-OSS, the same loader defaults as every run of record.
Unit wall time 4.25 h plus 0.8 h for the rerun; every unit status `ok`; both pilot gates passed.

## Summary table

| cell | rule outcome | verdict of record | thesis edit | escalation |
|---|---|---|---|---|
| 14.1 proto-refusal reliability | rel_proto 0.991, rel_gate 0.997, rel_adj 1.0 → **Branch A** | fresh-construction stands; 0.155 is not attenuation | Tier 3 sentence rewritten on the ladder (0.012 / 0.070 / 0.155 / 0.999) with the flat trajectory | none |
| 14.2 Llama rank-2/4 harm capture | per-component parity FAILS (PC3) → (a) VOID; subspace parity passes → (b) **Branch A** | under (b): capture(4) − control ≤ 0; reads-broad keeps "beyond harm" | none under (a); §8 unchanged under (b) | fork 16.3: (a) vs (b) |
| 14.3 GPT-OSS `P_dec` | replication 5/10 vs 6/10 → (a) VOID / (b) licensed; letter of Branch A holds by one item-quantum; magnitude specificity **fails** (p ≈ 0.2) | **reading, not co-primary**: reversibility stays behavioral-primary; the projection movement is not refusal-direction-specific at either position | Limitations paragraph rewritten (not deleted); the "monotone projection" corroboration is scoped everywhere it appears | scoping D3-22's corroboration clause |
| 14.3 A5 band half | band_min 0.531 > null q95 0.482 → band **above** null at the harmony decision token | GPT-OSS is the one panel model where moral content survives at the decision token | D3-20 "position-valid" scoped to decision-direction reads; the note enters FL App D | none |
| 14.4 P0–P3 PR audit | P0 band ≥ null on both; P0 PR 28.9 / 22.1 below the absolute 30 but sample-limited at n = 64 (above its sampling null); P1, P2, P2-full, P3 band **below** null on both → **Branch B** | the in-trace rungs stay hedged per model; null-relative claims unaffected | D1-10/11/12 keep the cross-position scope, now with the measured PR and band per position | none |
| 14.5 cross-ablation | baseline 0.62 (no bail); judgment→refusal +0.12 [0.06, 0.19]; refusal→judgment 0.007 [−0.02, 0.03]; persona +0.01 vs random q95 0.000 (degenerate) | **Branch A candidate**, held behind A8 until the boundary read; A8's derailment reading is rejected by the text tally | FL §6 may gain one causal sentence with the sign and the instrument scope | wording (Tier 1 causal clause) |
| 14.6a PR profiles | all four decision sites 4–8% of the shuffle reference; NI-2 closes as raw 10.3 vs standardized 14.2 on one sample | MN Table 1 / Fig 1 gain CIs; the gate is restated null-referenced | MN §2 edits | none |
| 14.6b reply inversion | harm flips 0/100 at both norms; random q95 0.61 / 0.83 → **Branch B** | the MN §3.1 sentence stays a limitation, now with the measured chance level and the sign finding | ledger A10 | none |
| 15.1 OLMo twins | replication gate passes (0.273 vs 0.27); alone sign agrees → pooled primary; pooled **harm_saturating** at n = 42 | headline unchanged, CI a third tighter; one-knob RMSE 0.023 | §7 carries n = 42 and the alone-set note | none |
| 15.2 Qwen read | 19 readout twins (1 request + 18 operating band); shape **indeterminate** | Tier 2 gains the lineage-independent fourth read; "Qwen not measured" retired | §8 table gains a Qwen row | none |

## 14.1 Proto-refusal reliability ceiling (Amendment 14.1, Amendment 16.1–16.2)

**Instrument.** Split-half self-cosine of the diff-of-means direction over 200 paired half-splits of
the Heretic 400/400 set, Spearman–Brown corrected; permutation null (labels shuffled within the same
activations, 200 draws); positive control = the same estimator on the saved Moral Stories per-pair
diffs (a direction the program knows is stable).

| side | format | split-half median [95% CI] | Spearman–Brown | permutation null q95 | positive control |
|---|---|---|---|---|---|
| proto-refusal (base, L16) | raw | 0.982 [0.976, 0.986] | 0.991 | 0.286 | 0.952 [0.942, 0.958] |
| gate (instruct, L16) | chat | 0.995 [0.992, 0.996] | 0.997 | 0.451 | 0.939 [0.931, 0.947] |

Adjacent-checkpoint arm (zero-GPU, W4-1): rel_adj 1.0 (0.9999999 at stage3-step11900 vs 11921);
cache-consistency positive control 1.0 (Paper 5 `olmo3_base` vs D1 `refusal_base.npz`, ≥ 0.99
required). Both reliabilities clear the 0.9 bar by a wide margin; the permutation ceilings (0.29,
0.45) show the estimator has dynamic range (a chance direction would read there, not near 1).

**Disattenuation.** cos_observed 0.1548; cos_corr = 0.1548 / sqrt(0.991 · 0.997) = **0.1557**,
CI [0.1556, 0.1560] (this CI propagates reliability uncertainty only). Second derivation: resampling
the 400/400 prompts on both sides and recomputing both directions gives cos 0.1548 with a
prompt-bootstrap 95% CI **[0.147, 0.162]**; the two intervals agree on the point and the second one
is the sampling CI the sentence should carry.

**Ladder (Amendment 16.1).** isotropic chance sqrt(2/(π·4096)) = 0.012 → covariance-matched
single-direction null q95 0.070 (base L16 act-sample, n = 1754) → measurement 0.155 [0.147, 0.162]
→ moral-subspace positive control 0.999. The measurement is 2.2× the matched-null q95 and 12× the
isotropic chance; it is 0.155 of a 0.50 crystallization bar the moral subspace clears at 0.999.

**Trajectory (Amendment 16.2).** proto→gate cosine across the 13 stage-3 states: 0.145, 0.139,
0.142, 0.142, 0.151, 0.149, 0.144, 0.150, 0.153, 0.151, 0.154, 0.155, 0.155 (flat within 0.016),
while proto-refusal's self-cosine to its final state rises 0.932 → 0.956 → 0.973 → … → 0.999 → 1.0.
The precursor crystallizes; its alignment with the eventual gate does not move.

**Verdict: Branch A.** rel_proto 0.991 ≥ 0.9 and rel_adj 1.0 ≥ 0.9; the disattenuated cosine moves
by 0.001 (< 0.03). The Tier-3 counter-reading (estimability floor) is closed: a low cosine between two
directions each estimated at reliability 0.99 is not attenuation. *Rival reading kept (Amendment 14
referee pass 1):* the 0.155 spans two formats (raw base, chat instruct); no reliability correction
removes a construct difference between formats. The adjacent-checkpoint arm is within-format and
independently shows no drift. *Anchored wording of record:* "a weak precursor (0.155, 95% CI
[0.147, 0.162]; matched-null q95 0.070; reliability 0.99 on both sides) against 0.999 for the moral
subspace"; never "almost no precursor".

## 14.2 Llama L12 rank-2/4 harm-coextensive capture (Amendment 14.2, fork Amendment 16.3)

**Parity.** Re-derived |cos(d_harm, PC_i)| for i = 1..8: 0.183, 0.233, 0.157, 0.032, 0.045, 0.012,
0.062, 0.026 against the saved 0.199, 0.307, 0.018, 0.026 (i ≤ 4). PC1, PC2, PC4 match within 0.05;
**PC3 misses (0.157 vs 0.018; max |Δ| 0.139 > 0.05)**. Subspace-level parity (Amendment 16.3b):
projection fraction of d_harm onto span(PC1..PC4) saved 0.367 vs re-derived 0.336, |Δ| 0.031 ≤ 0.05.
Rank-1 check: the boundary-twin harm basis (the run of record's stimuli) captures 0.049 vs the saved
0.036 (within 0.02); the severity-ladder basis (a different stimulus set) captures 0.086.

**Capture curve** (engage-weighted; weights 0.024 / 0.227 / 0.227 / 0.030 / … over PCs 1..8, zero
beyond):

| rank j | severity harm basis | boundary harm basis | control max (sentiment) | syntax | register | random basis q95 | positive control (self-capture) |
|---|---|---|---|---|---|---|---|
| 1 | 0.086 | 0.049 | 0.092 | 0.055 | 0.053 | 0.0006 | |
| 2 | 0.192 | 0.127 | 0.140 | 0.064 | 0.071 | 0.0010 | |
| 4 | 0.207 | 0.142 | 0.239 | 0.139 | 0.139 | 0.0016 | 0.755 (≥ 0.6 required) |

capture(4) − control max = **−0.032** (severity) and **−0.097** (boundary): a rank-4 harm basis
captures no more of the engage-driving moral basis than a rank-4 basis built the same way from
sentiment contrasts, and 3.7× less than the moral PCs recover from their own split half.

**Verdict.** Under (a) the pre-registered per-component parity rule: **VOID**, the numbers above are
descriptive. Under (b) subspace parity: **Branch A** (capture − null ≤ 0.25 by a margin of 0.28):
Llama's reads-broad ships at full strength ("beyond harm"), D3-18/19 unchanged, D3-24's "rank-8 broad"
stands. *Second derivation:* the sentiment control at 0.239 shows the engage-weighted capture is a
PC2/PC3-loading statistic (those two carry 0.45 of the weight), so any contrast partly aligned with
PC2/PC3 scores ≈ 0.2; harm is one such contrast, not a privileged one. *Rival reading:* the severity
ladder's harm percept could be richer than rank 4; the curve is flat from rank 2 to rank 4 (0.192 →
0.207), so rank does not buy capture. **Choice (a) vs (b) is escalated** (Amendment 16.3).

## 14.3 GPT-OSS post-response decision token (Amendment 14.3, forks Amendment 17)

**Replication.** Behavioral flip at max strength 5/10 vs the 6/10 of record (one item). Under
17.1(a): VOID; under 17.1(b): licensed. All reads below are reported for both.

**`P_dec` read.** 8 of 10 items opened a final channel (2 unmeasured and counted). Unit statistic
(`graded_disengage_stat`, the Amendment-12 definition): monotone fraction toward comply **1.0**,
mean move at max CI [−122.8, −107.4] raw units (excludes 0), random-direction monotone-fraction
q95 **0.75** (< 0.8 by one item-quantum at n = 8). Every clause of the Branch A rule holds by its letter.

**Specificity (Amendment 17.3, both frames).** Standardizing by the decision-token act-sample σ
(n = 128) and nulling against 500 covariance-matched random directions from the same sample:

| position | frame | refusal-dir move, strong − weak (SD of sample) [CI] | random move q05 / median / q95 | one-sided p (random ≤ refusal) | strict-monotone random q95 |
|---|---|---|---|---|---|
| P_prefill | raw | −1.15 [−1.24, −1.08] | −2.15 / +0.03 / +1.75 | 0.216 | 1.00 |
| P_prefill | std | −0.86 [−0.94, −0.78] | −1.19 / −0.04 / +1.33 | 0.170 | 1.00 |
| P_dec | raw | −1.07 [−1.12, −1.02] | −2.01 / +0.01 / +2.19 | 0.230 | 1.00 |
| P_dec | std | −1.00 [−1.04, −0.95] | −1.75 / +0.10 / +1.88 | 0.206 | 1.00 |

The refusal direction moves about one SD toward comply as the exculpatory prefill strengthens; a
covariance-matched random direction moves at least that far in about one draw in five, and under a
strict monotone definition every random direction is monotone. The graded movement is the position's
response to the prefill, at the prefill token and at the post-response decision token alike. Engage
arm at P_dec: 6/7 flips to refuse, mean move +175 raw (no null was run on it; two-sided in sign).

**Verdict.** Branch A holds by the letter (under 17.1b) and fails its anchoring: the co-primary
promotion is **withheld**. Reversibility stays behavioral-primary (5/10 here, 6/10 of record, engage
7/7 of record). The Limitations paragraph is **rewritten**, not deleted: "the decision-channel
projection moves monotonically toward comply at both the prefill token and the post-response decision
token, but not distinguishably from covariance-matched random directions at either position (one-sided
p 0.17–0.23); the projection is corroboration of *where the prefill writes*, not a refusal-specific
read." *Blast radius (escalated):* the phrase "monotone projection movement in all 10 items" appears in
CLAIMS D3-22, FL §8.2 (two places), the two-axis table's GPT-OSS commit cell, FL App D, FL App E.4,
MN §4 and §6, and SYNTHESIS (three places). Each keeps the behavioral flip and drops or scopes the
projection corroboration; wording proposed in CLAIMS D3-22 and applied to the drafts only after the
escalation is answered.

**A5 band half.** Held-one-out moral-family band at the harmony decision token: moral_stories 0.531
(null q95 0.482), fables 0.541 (0.471), ethics 0.540 (0.345); band_min 0.531 > null_q95_max 0.482 →
**band above null**. Post-standardization PR of record 12.79 (raw 9.40, CI [7.7, 10.0]). GPT-OSS's
decision token is the one panel position where the moral band survives the covariance null; D3-20's
"position-valid" is scoped to decision-direction reads (NI-3), and the note enters FL App D.

## 14.4 D1 P0–P3 per-rollout PR audit (Amendment 14.4)

Run of record parameters: Think window 256, max_new_tokens 320, 32 rollouts per side; GPT-OSS window
16, max_new_tokens 1024, 32 per side (P3 harmless 26 closed). Think P3 unmeasured by design.

| model | position | PR (n = 64) | PR/(n−1) | Gaussian null q95 / quantile | band_min vs null q95 max | below null |
|---|---|---|---|---|---|---|
| Think | P0 (t_inst) | 28.9 | 0.46 | 22.8 / 1.00 | 0.399 vs 0.146 | no |
| Think | P1 | 6.9 | 0.11 | 8.4 / 0.67 | 0.483 vs 0.533 | **yes** |
| Think | P2 window | 9.7 | 0.15 | 11.5 / 0.77 | 0.221 vs 0.396 | **yes** |
| Think | P2 full | 9.5 | 0.15 | 11.0 / 0.78 | 0.263 vs 0.392 | **yes** |
| GPT-OSS | P0 | 22.1 | 0.35 | 19.6 / 1.00 | 0.460 vs 0.317 | no |
| GPT-OSS | P1 | 9.7 | 0.15 | 10.2 / 0.87 | 0.457 vs 0.476 | **yes** |
| GPT-OSS | P2 window | 5.1 | 0.08 | 6.5 / 0.68 | 0.569 vs 0.635 | **yes** |
| GPT-OSS | P2 full | 3.1 | 0.05 | 3.7 / 0.61 | 0.116 vs 0.635 | **yes** |
| GPT-OSS | P3 | 4.3 | 0.08 | 5.7 / 0.60 | 0.073 vs 0.341 | **yes** |

**Positive control.** The band half passes on both P0 sites (content survives the null at t_inst).
The absolute half (PR ≥ 30) reads 28.9 and 22.1: at n = 64 the sample-rank ceiling is 63 and P0 uses
0.46 / 0.35 of it, above its own covariance-matched sampling null at quantile 1.0 on both models. The
absolute bar is not evaluable at this n (ledger A9); the null-referenced form of the control passes.

**Verdict: Branch B on both models.** P1, both P2 windows, and P3 sit below the covariance null on
Think and GPT-OSS alike, at PR 3–10, the narrowest positions in the program. The in-trace rungs of
D1-10/11/12 keep their cross-position scope, now stated per model with the measured PR and band:
"band-relative comparison not licensed at P2 (PR 9.7 Think / 5.1 GPT-OSS; band below null)".
Null-relative claims (D1-11's null crossing, D1-12) are unaffected. *Reframe:* the reasoning-trace
window is a decision-like position, not a content-like one; its PR (5–10) sits with the four decision
sites (8.6–14.7), not with the content positions (27–97), which is A2 seen on a fifth kind of
position. Closes MISSING_ARTIFACTS A1 (Think and GPT-OSS MFT directions saved), A3 (Think refusal P0,
P1, P2, P2-full saved; P3 unmeasured by design), Amendment 2(ii) (per-rollout windows saved).

## 14.5 Reconciled cross-ablation (Amendment 14.5; ANOMALIES A8)

**Harness of record.** Chat-template generation (Paper 6 parity), 100 held-out harmful requests,
refusal direction from the 400/400 train set, opening-refusal classifier; baseline refusal **0.62**
(Paper 6's 0.575), judgment accuracy 0.742; no bail.

| condition | refusal rate | judgment acc | prompts changing state (comply→refuse / refuse→comply) |
|---|---|---|---|
| baseline | 0.62 | 0.742 | |
| judgment-decision ablated | 0.74 | 0.758 | 12 / 0 |
| refusal ablated | 0.78 | 0.750 | 33 / 17 |
| persona ablated | 0.63 | 0.725 | 1 / 0 |
| random (5 draws) | 0.62 each | 0.733–0.750 | 0 / 0 each |

**Arrows (paired bootstrap over prompts, 2000).** judgment→refusal: Δ(judgment-abl) − Δ(random) =
**+0.12, 95% CI [0.06, 0.19]** (excludes 0; the pre-registered MDE at n = 100 was 0.14, so the effect
is detected below the planning MDE, which is a power statement, not a test). refusal→judgment:
+0.007 [−0.018, 0.033], not detected; bar "no cross-effect detectable at Δ ≳ 0.13 in judgment
accuracy". Persona control: +0.01 (one prompt) against a random q95 of exactly 0.000 (five draws, none
moved any prompt): the rule's letter ("does not exceed the random q95") fails on a degenerate bar;
by construction the control moved 1 prompt in 100 where the effect moved 12.

**A8 discriminator (14.5_gen).** Regeneration under the saved directions reproduces every per-prompt
outcome (100% agreement in all four conditions). Text tally: all 33 comply→refuse outputs under
refusal ablation are well-formed opening refusals ("I'm sorry, but I can't assist with…", median 336
characters, repetition fraction 0.11, no non-ASCII debris); all 17 refuse→comply outputs are coherent
partial-compliance redirects ("Certainly! However, …"); all 12 comply→refuse outputs under
judgment-decision ablation are well-formed refusals. Length and repetition distributions match
baseline. **A8 reading R_a (derailment / classifier artifact) is rejected.** R_b stands: a
single-layer projection-out of the refusal direction re-decides about half the prompts in both
directions with coherent text; removing the judgment-decision direction re-decides 12 prompts, all
toward refusal; random and persona directions re-decide 0 and 1.

**Verdict.** Branch A candidate: a specific causal cross-effect exists, with the sign *removing
judgment-decision content increases refusal*. Held at "candidate" until the remaining A8 read (flip
probability vs |baseline refusal projection|, ~100 forward passes) and the persona-bar wording are
settled; the proposed FL §6 sentence is escalated. *Rival reading:* the judgment-decision direction
is, like the refusal direction, one of the L16 directions whose removal perturbs the gate, and the +0.12
is that perturbation rather than judgment content feeding refusal; the one-directionality (12/0 vs
33/17) and the zero effect of five random and one persona direction argue against "any salient
direction", not against "any gate-adjacent direction". The interchange result (D3-06..09) remains the
primary causal claim; this cell adds an ablation-side arrow on a different instrument.

## 14.6a PR profiles (Amendment 14.6a; NI-2)

Same 240 texts per model (40 pairs × 3 sources × 2), primary layer, 2000 row-resampled bootstraps,
200 Gaussian-null and 200 column-shuffle draws. The percentile bootstrap is biased low (row
duplication lowers rank; e.g. OLMo last_content 62.8 with bootstrap [46.3, 53.0]); the subsampling
CI (half-samples, n^{-1/2} rescaled, 500 draws) is the interval of record for the decision sites.

| model | position | PR raw | subsampling CI | PR/(n−1) | Gaussian null q95 (quantile) | shuffle reference | PR std |
|---|---|---|---|---|---|---|---|
| OLMo-3-Instruct (L16) | final_pre_assistant | 14.7 | [14.3, 16.2] | 0.061 | 16.0 (0.77) | 223 | 20.3 |
| | last_content | 62.8 | | 0.263 | 53.7 (1.00) | 224 | 73.6 |
| | mean_content | 40.4 | | 0.169 | 38.0 (0.99) | 201 | 59.5 |
| Llama-3.1 (L12) | final_pre_assistant | 10.3 | [10.1, 11.1] | 0.043 | 11.0 (0.69) | 208 | 14.2 |
| | last_content | 97.3 | | 0.407 | 71.9 (1.00) | 191 | 136.0 |
| | mean_content | 26.7 | | 0.112 | 27.4 (0.92) | 185 | 44.4 |
| Qwen2.5 (L14) | final_pre_assistant | 8.6 | [8.2, 9.4] | 0.036 | 9.9 (0.63) | 212 | 13.5 |
| | last_content | 42.4 | | 0.178 | 40.1 (1.00) | 204 | 75.7 |
| | mean_content | 32.7 | | 0.137 | 31.9 (1.00) | 182 | 61.0 |
| GPT-OSS (L12, n = 128) | harmony decision token | 9.4 | [9.1, 10.7] | 0.074 | 10.7 (0.69) | 112 | 12.8 (of record) |

**Null-referenced gate (replaces "PR < 30").** Every decision site is 4–8% of the marginal-preserving
shuffle reference and 3.6–7.4% of the sample-rank ceiling, and 2.6–9.4× below the content positions
of the same texts; the content positions sit at or above their Gaussian null q95 (quantile 0.92–1.00)
while the decision sites sit inside theirs (0.63–0.77), i.e. the decision site carries no
dimensionality beyond its own covariance. **NI-2 closes:** Llama's "decision-site 10.2 vs
decision-token 13.5" was never two positions; on one sample the same position reads 10.3 raw and 14.2
standardized, and the 13.5 of record is the standardized D3-harness read (request-twin set). MN Table 1's
"decision-token" column is relabeled a standardization column.

## 14.6b Reply-inversion specificity (Amendment 14.6b; ledger A10)

Llama-3.1-8B-Instruct, layer 12, 100 evaluation items, residual norm 7.76, forced-answer margin
(positive = toward harmful), harm direction and 20 random unit directions at the identical norm.

| α | harm flip fraction | harm margin shift | random flip q95 | random shift q95 | clean margin |
|---|---|---|---|---|---|
| 0.5 | 0.00 | −2.12 | 0.61 | +2.36 | mean −1.78, 87% ≤ 0 |
| 1.0 | 0.00 | −2.62 | 0.83 | +2.87 | |

**Verdict: Branch B.** The harm direction does not exceed the random q95 on the flip fraction; the
MN §3.1 sentence stays as a limitation, now carrying the measured chance level. *What the control
found (A10):* random matched-norm directions wash the clean margins toward zero (per-direction means
−1.75 to +1.06 from a clean −1.78) and thereby "flip" 61–83% of the near-zero majority; the harm
direction moves every margin coherently the other way (−3.9, −4.4). The flip metric is not specific
at this norm, and the direction the harm axis pushes replies on Llama is toward *safe*, opposite to
the Qwen2.5-14B-Instruct reply inversion of record. cos(harm_dir, severity contrast) = +0.125, so the
sign is the conventional one. Two readings, ledgered.

## 15.1 OLMo-3 additional request-twins (Amendment 15.1)

Union run: 108 authored twins through the same screen, 42 kept (23 original, 19 new; 38% and 40% pass
rates). Pilot gate 5/5 finite, 5/5 sign-coherent.

| set | n | shape verdict | R_refusal(16) [CI] | CI width | R_judgment(16) | harm rank-1 R | one-knob c (RMSE all k / k ≥ 3) |
|---|---|---|---|---|---|---|---|
| original (replication) | 23 | harm_saturating | 0.273 [0.143, 0.563] | 0.42 | 0.659 | 0.314 | 0.283 (0.027 / 0.021) |
| new batch alone | 19 | indeterminate | 0.195 [−0.001, 0.446] | 0.45 | 0.659 | 0.364 | 0.191 (0.021 / 0.024) |
| pooled | 42 | **harm_saturating** | **0.242 [0.134, 0.405]** | 0.27 | 0.659 | 0.334 | **0.246 (0.023 / 0.022)** |

Replication gate: R_refusal(16) 0.273 is within 0.10 of the 0.27 of record and the shape returns
`harm_saturating` → the harness has not drifted. Sign rule: the alone set agrees in the sign of
R_judgment(16) − R_refusal(16) → **pooled is primary**. Ratio-of-ratios (secondary) on the pooled
session: diff 0.21, CI [−0.07, 0.39] → `under_transfer`, unresolved as the power table said it would be
at n = 42. One-knob fit on the pooled sweep: R_refusal(k) = min(0.246, R_judgment(k)), RMSE 0.023
(grid least squares over the four k; the 0.036 of record was fitted on the plateau only).

**Verdict: shape survives.** Headline unchanged with the CI a third tighter (0.42 → 0.27, the
predicted lift). Wording of record for §7: "pooled n = 42 `harm_saturating`; the 19 new twins alone
read `indeterminate` (R_refusal(16) 0.195, CI touching 0) with the same sign, and the one-knob fit holds
on all three sets (c 0.19–0.28)". *Rival reading:* the new batch, authored after the plateau was known,
under-transfers relative to the original (0.195 vs 0.273); the pooled CI covers both, and the alone-set
CI's lower edge at 0 is why it cannot carry the shape on its own. It is reported, not averaged away.

## 15.2 Qwen2.5-7B-Instruct C1 read cell (Amendment 15.2, rider 2026-09-12)

Operating mode after the rider: 1 screened request twin + 18 severity operating-band pairs (levels
3–5) = 19 readout twins; standardized frame (PR 12.2, channel null raw 0.18 / std 0.14); transport
control passed (judgment moves 0.94 [0.78, 1.11] under the full patch); restricted patch moves
refusal (−1.33 vs random −0.01, `vmoral_is_read_substrate`).

| quantity | value |
|---|---|
| R_refusal(k), k = 1/3/8/16 | 0.063 / 0.518 / 0.514 / **0.541 [0.423, 0.692]** |
| R_judgment(k) | 0.073 / 0.575 / 0.615 / 0.660 |
| gap R_judgment(16) − R_refusal(16) | 0.119 [−0.038, 0.251] |
| harm rank-1 R | 0.384 [0.296, 0.495]; R_refusal(16) − harm = 0.157 |
| shape verdict of record | **indeterminate** (refusal climbs; gap 0.12 > 0.10 tolerance; refusal 0.16 above harm rank-1) |
| bootstrap verdict distribution (2000) | indeterminate 55%, broad_moral 44%, harm_saturating 0.6%, instrument_ceiling 0.6% |
| ratio-of-ratios | 0.05 [−0.12, 0.17], `under_transfer` |
| commit axis | disengage −2.70 [−3.87, −1.65], engage +0.68 [0.37, 0.99], both coherent; A = −0.60 [−0.76, −0.40] |
| behavioral | 7/10 base refuse, 2/10 disengage flips, 1/10 engage flips |

**Verdict: `indeterminate`** (a pre-registered publishable branch), anchored: Qwen's refusal reads
beyond the harm rank-1 level (0.54 vs 0.38, a difference of 0.16 whose CIs do not overlap at the
point) and its gap to judgment (0.12) is not resolvable against the 0.10 plateau tolerance at n = 19;
`harm_saturating` is excluded (0.6% of resamples), `broad_moral` is the plurality alternative (44%).
The two-axis table gains a Qwen row: *reads* "beyond harm rank-1, between OLMo's plateau (0.24) and
Llama's gap-close (0.85); indeterminate at n = 19"; *commits* "bidirectionally responsive at the read
layer (A −0.60), disengage-dominant, like OLMo". "Qwen not measured on the read axis" is retired.
*For the A13 dimensionality hypothesis:* Qwen is the lowest-PR decision site (8.6) and reads more
broadly than OLMo; the ordinal "low-rank read ↔ late/reversible commit" now has a fourth point that
sits in the middle on the read axis and with OLMo on the commit axis, which neither confirms nor breaks
the one-knob ordering; it is a lineage-independent point, stated as such.

## Blast radius of the 14.3 scoping (move 3)

The specificity result invalidates one *corroboration*, not a verdict: the reversible-reader verdict
rests on behavior (7/7 engage, 6/10 disengage of record; 5/10 replicated). Everything that cited the
monotone projection as a second leg keeps the behavioral leg and loses or scopes the projection leg:
CLAIMS D3-22; SYNTHESIS header, Tier 2 GPT-OSS bullet, branch table; FL §8.2 (two sentences), Table
two-axis (commit cell), App D (two rows), App E.4 (one line), §10 Limitations (rewritten); MN §4.2 and
§6 case study (one clause each). No other verdict was gated on the projection corroboration.

## Referee pass (three objections)

1. *"14.2's fork lets you keep the convenient reading."* Both readings are printed; the numbers under
   (b) are the same numbers (a) calls descriptive, and (b) is a weaker condition only in that it does
   not require a per-component match a near-degenerate PCA cannot deliver. If Orion takes (a), FL §8
   loses no sentence (the rank-1 3.6% result of record already carries "beyond harm"); it gains no
   rank-4 sentence.
2. *"14.3's null is too easy: random directions from a 128-sample covariance at a massive-activation
   position move a lot."* That is the point: the refusal direction moves as much as such a random
   direction and no more. Standardization (17.2) removes the massive-dimension route and the p stays
   0.17–0.21. A harder test would be a direction-specific behavioral intervention at P_dec, which is
   the held Tier-2 causal cell.
3. *"15.2's `indeterminate` is a non-result dressed as a row."* It is the pre-registered branch and it
   carries three anchored numbers: harm_saturating excluded at 0.6% of resamples, refusal 0.16 above
   its harm rank-1, and a gap CI that includes both 0 and 0.25. That is more than the empty cell it
   replaces and less than a verdict, which is what the row will say.

## Escalations — decided (Orion, 2026-09-13: all three recommendations accepted)

1 → (b) subspace parity, Branch A applied to FL §8 / App D / MN §6. 2 → scoping applied to CLAIMS D3-22, SYNTHESIS, FL §8.2 / table / App D / App E / §10, MN §4 / §6. 3 → the §6 sentence is held; App C.8 carries the cell descriptively.

Original list:

1. **14.2 fork:** (a) VOID or (b) subspace parity (Amendment 16.3).
2. **14.3 scoping:** apply the blast-radius edits (behavioral leg kept, projection leg scoped) to
   CLAIMS D3-22, SYNTHESIS, FL, MN.
3. **14.5 Tier-1 causal clause:** whether FL §6 gains "ablating the judgment-decision direction raises
   refusal by 0.12 [0.06, 0.19] where five random directions and the persona direction move 0 and
   0.01; a single-layer projection-out, not the interchange instrument" now, or after the remaining A8
   read.
