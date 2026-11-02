# MRB Phase 2 gate report

Rules: `papers/MRB_PLAN.md` v0.9, v0.9.1 (the role breakdown) and v0.9.2 (VALIDATE folded into the
session); MB_PLAN Amendment 5 (the MoralBench rider). Numbers: `outputs/p2/p2_analysis.json`
(`scripts/mrb_p2_analyze.py`), `outputs/p2/projection.json`, `outputs/p2/mb_phase2/*/hf_phase.json`,
`outputs/p2_overlap.json`. Judge rows are tracked gzipped; raw text is untracked (`_raw/`; its ds-data
backup waits for the drive to be mounted).

## Headline

At matched response length, ST's care adapters lower MoReBench's response-field score:

| model (adapter) | cost at matched length [95% CI] | share of the raw cost due to length | raw cost (registered) |
|---|---|---|---|
| Muse Glimmer 30B (T1.6 care-DPO; passed ST's gates) | **−0.87 [−1.46, −0.23]** | 0.41 [0.21, 0.76] | −1.47 [−2.01, −0.97] |
| Qwen3.8-27B (care-DPO_Q; non-specific on Qwen) | **−2.83 [−3.70, −1.90]** | 0.27 [0.11, 0.48] | −3.85 [−4.48, −3.22] |

Matched-length cost from v0.9.3's fork (the `none` arm's within-scenario length coupling; the
cross-scenario intercept agrees: −0.82 and −2.72); the registered verdict, "process-quality cost" on the
raw difference, stands beside it (§2). Each adapter is a single training draw; Qwen's figure is a lower
bound on its attached adapter's effect (MRB-A15). On the 0-100 scale Muse's cost is small: about 6% of
Muse's own spread across scenarios (SD 13.7 of the scenario means, `none` arm) and below G1's 1.96-point
between-model MDE.

**For future steering plans.** A length-matched MoReBench response-field gate (the cost at matched length,
by v0.9.3's estimator, with its CI) is a candidate standing gate for any steering adapter; its threshold
is set in the plan that adopts it, not here.

## 1. The session (2026-11-01 21:37 - 2026-11-02 01:27 UTC)

| stage | result |
|---|---|
| first attempt | pod bi2b907r5yjdtv bailed at the gate tests after setup (~0.7 A100-h): the dry judge asked for a tokenizer that pod never downloads; fixed (`3cda39a`) |
| 1 setup and gates | 0.29 h; both adapters' sha256 as registered; 18 passed, 2 skipped |
| 2 HF phase | 0.09 h per model; both merge gates pass (§4) |
| 3 `none` k0 | Muse 500 in 6.1 min (1,041 tokens per job, 1,410 tok/s); Qwen 500 in 20.4 min (2,222, 906) |
| 4 sync point | judge pilot 0 of 412 without a verdict on each model; 100% of generations reached the final answer on each; projection 8.32 A100-h on the pod at k = 2 for both: **proceed** |
| 5 adapter arms, `none` k1 | both vLLM parity gates pass (Muse 0.0138 within its band 0.0151; Qwen 0.0130 within 0.0130); all 4,000 generations reached the final answer, none at the cap |
| 6 the full judge | 91,600 calls (824 from the pilot), 0 without a verdict, 5,292 tok/s; no guard trip |

Wall 3.83 h; spend about 7.8 A100-h on this pod and about 8.5 with the first attempt, of the 9.0
envelope (the author let the session run to its own cap). The overlap check found no MoReBench
dilemma sharing a word 8-gram with the adapters' 40 training scenarios, so the headline runs on all 500.

## 2. The headline (response field, 500 scenarios, k = 2, adapter − none, code score of record)

| model | none | adapter | Δ Regular [95% CI] | Δ length (chars) | Δ Hard | registered branch |
|---|---|---|---|---|---|---|
| Muse Glimmer 30B (T1.6 care-DPO, known-good) | 76.06 | 74.59 | **−1.47 [−2.01, −0.97]** | −215 [−249, −183] | +0.75 [+0.55, +0.95] | process-quality cost |
| Qwen3.8-27B (care-DPO_Q, known-non-specific) | 78.27 | 74.42 | **−3.85 [−4.48, −3.22]** | −1,476 [−1,579, −1,373] | +2.18 [+1.99, +2.39] | process-quality cost |

Each adapter is a single training draw: the verdicts are about these adapters, not the recipe.
Observed MDE80: 0.76 (Muse) and 0.87 (Qwen); both costs are well past it. The 400-scenario subset reads
the same (−1.75 [−2.38, −1.15]; −3.76 [−4.44, −3.09]). Both models' length CIs exclude 0, so the
length rule puts Hard beside Regular: **Hard rises on both while Regular falls** (MRB-A13). The
registered length rule names only the gain case; the cost here runs opposite to Hard, which is
consistent with shorter answers covering fewer criteria (R_length) as well as with worse reasoning
(R_quality). The separating read is zero GPU on these rows and needs an amendment first.

**Dimensions (share of criteria met, Δ [CI], Holm across five):**

| dimension | Muse | Qwen |
|---|---|---|
| identifying | −1.53 [−2.41, −0.68], p 0.005 | −3.49 [−4.50, −2.50], p < 0.001 |
| logical process | −1.54 [−2.67, −0.41], p 0.04 | −4.81 [−6.10, −3.40], p < 0.001 |
| clear process | −0.75 [−2.35, +0.87] | −4.92 [−6.75, −3.10], p < 0.001 |
| helpful outcome | −1.14 [−2.33, +0.06] | −2.28 [−3.57, −0.97], p < 0.001 |
| harmless outcome | −0.32 [−1.83, +1.10] | −1.13 [−2.95, +0.58] |

Neither adapter moves Harmless Outcome detectably (per-dimension MDE about 2.3).

## 3. Registered secondaries

- **Role (v0.9.1; Agent predicted).** Muse: Agent −2.01 [−2.83, −1.19], Advisor −1.10 [−1.76, −0.40],
  interaction −0.91 [−2.05, +0.19]: no role difference detectable at |Δ_Agent − Δ_Advisor| ≳ 1.55.
  Qwen: Agent −4.71 [−5.65, −3.76], Advisor −3.25 [−4.03, −2.43], interaction −1.46 [−2.66, −0.24]:
  the literal registered branch reads "the opposite of the prediction", while the Agent role carries
  the larger effect in magnitude on both models. The branches were written for a positive transfer;
  which reading the registration meant is the author's call (MRB-A14).
- **Separation, Muse − Qwen (known-good against known-non-specific).** Regular ΔΔ +2.38 [+1.58,
  +3.20] and Logical Process +3.27 [+1.61, +4.98]: MoReBench's response field separates the two
  adapters, Muse's costing less. Harmless Outcome +0.82 [−1.30, +2.85]: no separation detectable at
  |ΔΔ| ≳ 2.92, which is not a verdict against MoReBench (ST's non-specificity was measured on
  rule-right items). Qwen's much larger length shift (−1,476 against −215) is a rival for the
  separation (MRB-A13).

## 4. The merge and parity gates; the MoralBench rider (MB_PLAN Amendment 5)

| model | adapter vs base TV | merged vs attached | merged vs base | kept share | weight change kept in bf16 | vLLM parity (band) |
|---|---|---|---|---|---|---|
| Muse | 0.222 | 0.015 | 0.220 | 0.93 | 99.5% | 0.0138 (0.0151) |
| Qwen | 0.029 | 0.012 | 0.029 | 0.60 | 87.5% | 0.0130 (0.0130) |

Both gates pass as registered. Qwen's merged copy keeps about 60% of its adapter's dose-0 letter
shift, so Qwen's Δ may understate the attached adapter's effect (MRB-A15).

**MoralBench R2 (dose 0, both orders; the `none` rows are the Phase 2 pre-steering baseline, the
adapter rows a drift-monitor read).** Muse's adapter moves the letter bias toward A on every set (ΔL
+0.16 to +0.30, every CI excluding 0) and raises the MFQ binary expected score +2.24 [+0.43, +4.09];
the content effect K moves by at most 0.07. Qwen's adapter barely moves either (|ΔL| ≤ 0.03; MFQ
binary +0.51 [+0.18, +0.91]). Under MB's drift-monitor rule (Amendment 1), Muse's shift is mostly a
letter preference (ΔL up to +0.30) with small content-effect shifts (|ΔK| ≤ 0.07, two of four CIs
excluding 0).

## 5. Referee pass

1. *"The 'cost' is a length effect: the adapters answer more briefly, and Regular counts criteria."*
   Answered in part by the registered fork (§8): length accounts for 0.41 of Muse's cost and 0.27 of
   Qwen's, and a cost remains at matched length on both (−0.87, −2.83). Conceded: the adjustment uses
   the model's own length coupling, so a coverage loss specific to the adapter's shorter answers would
   survive it (SYNTHESIS, MRB-05's counter-reading).
2. *"Qwen's adapter on vLLM is a weakened adapter."* Conceded in part: the merge keeps 60% of its dose-0
   letter shift (MRB-A15); the gates passed as registered. Qwen's cost is then a lower bound. Muse's merge
   keeps 93%.
3. *"Two single draws cannot speak for the recipe."* Conceded, as registered: each verdict is about one
   adapter. The separation between them is about these two adapters, and it carries Qwen's larger
   length shift as a rival.

## 6. Anticipated review

1. Hard against Regular in opposite directions → MRB-A13 → a registered length-matched read (zero GPU).
2. The role branches' sign convention → MRB-A14 → the author's ruling on what v0.9.1 meant.
3. Qwen's 60% kept share → MRB-A15 → HF-attached generation on ~50 scenarios (~0.3 A100-h; the envelope
   has ~0.5 left).
4. Not planned before: Muse's MoralBench letter-bias shift (ΔL up to +0.30) means a MoralBench score of
   the steered Muse moves with its letter preference; any MB drift claim carries L beside the score.

Question behind the question: whether the steering line should read MoReBench's response field at
matched length. If MRB-A13 resolves as R_length, Regular measures how much an answer covers, and a
length-controlled score is the steering readout; if R_quality, Regular stands as it is.

## 7. Decisions for the author

1. **MRB-A13:** register the length-matched read (Δscore against Δlength per scenario) as a fork.
   Recommended: yes, before any further framing.
2. **MRB-A14:** rule on the role branches: literal (Qwen "opposite") or by magnitude (Agent, as predicted).
3. **MRB-A15:** run the HF-attached check on Qwen (~0.3 A100-h), or carry the lower-bound note.
4. **The ds-data backup** of `outputs/p2/_raw/` when the drive is mounted.

## 8. After the gate (author's decisions; MRB_PLAN v0.9.3; registered before computing)

**The length-matched read (MRB-A13; `outputs/p2/p2_length_read.json`).** A fork beside the
registered "process-quality cost".

| model | length coupling β (`none`, within scenario) | Δ | Δ length | Δ_adj (cost at matched length) | share explained by length | cross-scenario intercept |
|---|---|---|---|---|---|---|
| Muse | +2.83 per 1,000 chars [1.45, 4.21] | −1.47 | −0.215 k | **−0.87 [−1.46, −0.23]** | 0.41 [0.21, 0.76] | −0.82 [−1.43, −0.20] |
| Qwen3.8 | +0.70 [0.29, 1.20] | −3.85 | −1.476 k | **−2.83 [−3.70, −1.90]** | 0.27 [0.11, 0.48] | −2.72 [−3.54, −1.89] |

Reading by v0.9.3's rule, both models: **part length, part reasoning**; a cost remains at matched
length, and the two estimators agree. The author's prior (Muse's small length shift unlikely to
explain all of its cost) holds. Muse's share is larger than its 5.5% length shift suggests because its
score rises steeply with length.

**The role breakdown (MRB-A14).** The registered prediction ("Agent is the locus of any transfer") did
not specify a sign; both readings are reported. Literal: Qwen "the opposite of the prediction", Muse
no detectable difference. Magnitude: the Agent role carries the larger effect on both, resolved on Qwen.

**MRB-A15.** Qwen's Δ is carried as a lower bound on its attached adapter's effect; no HF-attached check
unless Qwen's number enters a paper.

**The DPO pairs' letter positions (MRB-A16; `outputs/p2_pair_letters.json`).** Pairs rebuilt with ST's own
rules and checked against each adapter's recorded counts (Muse 40 rule-harm; Qwen 32 rule-harm and
7 rule-right; three options in every pair). Registered read, chosen against rejected at A: Muse 11/40
against 12/40, difference −0.03 [−0.25, +0.18]; Qwen 4/39 against 10/39, −0.15 [−0.31, 0.00]. No
chosen-at-A excess is detectable, so positional learning toward A does not explain Muse's MoralBench
shift (an excess up to about +0.18 is not excluded at 40 pairs). Descriptive, found after the table:
Muse's pairs push away from B (−0.25 [−0.45, −0.05]) and toward C (+0.28), and on two-option items
moving away from B is moving toward A, a reading (R_B).

**MRB-A16's check (MRB_PLAN v0.9.4; `outputs/p2_b_avoidance.json`).** On ST's dose-0 reads for Muse (1,056
matched rows, 132 scenarios), care-DPO against `none`, position effects at fixed content: A +0.232
[+0.209, +0.253], B −0.116 [−0.131, −0.101], C −0.116 [−0.130, −0.102]; content effects (toward the
protect option +0.100, away from the rule option −0.201) beside them. The registered branch reads R_B,
mixed, with R_A: B falls no more than C, so the positional shift is an A preference at fixed content,
not a B-specific avoidance. It accounts for Muse's MoralBench shift toward A; the pairs' letters do not
(no chosen-at-A excess), and base-DPO shows no A preference (A −0.042). What in care-DPO's steered
traces teaches it is open.

**Backup.** `outputs/p2/_raw/` (243 MB) copied to ds-data, sha256 verified and recorded in its manifest.

Framing written after the length-matched read (author, 2026-11-02): the headline above, and one
standing claim in SYNTHESIS (MRB-05).

