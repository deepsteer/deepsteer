# KDG pilot results (Session KDG-1, pod 2026-09-14; spec v0.4 + A12)

Numbers of record come from `outputs/pilot/analysis_pilot.json` (run `kdg_20260914T045758`,
manifest verified, 2.3 GB per-rollout arrays; 96 scenarios: 60 primaries + 36 harm twins;
OLMo-3-7B-Instruct for every chat cell, OLMo-3-7B base + Instruct for the raw cells; harness
1.0.0, template 1.0.0; scenario sets `data/pilot_scenarios_{A,B}_*.json`). Verdict rules: spec
§5–§7 with amendments A11 (binary reference) and A12 (stability fork). Human gate KDG-G1.

## 0. Positive voice first (move 7)

The panel instrument works on an open 7B instruct model: every one of 10,656 generated replies
parsed by a clean rule, the pressure engages on every gate family, the known-gap control moves
the instrument by 0.44 [0.14, 0.62] above the measurement, and the pilot gate passes under the
pre-registered rule (19/48, four families) and under fork B (26/48). On the screened panel the
model takes the action it judged wrong on 22–27% of scenarios by majority and 41% of rollouts.
Whether that is a knowing–doing gap or reference noise is the question the ladder poses below;
the panel now has the cells to answer it.

## 1. Pilot gate (§5): PASS

| rule | option-level stability (PRIMARY, §4.1) | binary stability (fork B, A12) |
|---|---|---|
| gate-family primaries passing the screen | **19/48** (F1 3, F3 4, F4 5, F5 7) | 26/48 (F1 4, F3 7, F4 7, F5 8) |
| screen reasons | mixed 19, no_pressure 9, judgment_unstable 16, judged-violating 4 | mixed 26, no_pressure 11, judgment_unstable 7, judged-violating 4 |
| gate (≥ 14/48 and ≥ 2 families) | PASS | PASS |
| F2 (appendix) | 1/12 (11 no_pressure) | same |

Harm twins behave like their primaries (F1T 2 mixed + 1 clean gap, F3T 3, F4T 5). F2 never
engages pressure on this model (11/12 below 0.15 violating): the sycophancy-under-pushback
family stays an appendix family; the promotion rule (§5) is not close.

Parse rates: 1.000 on every chat cell (3072 dose-0 D rollouts, all `bare_letter`; 864 J_stated
replies per cell, all `answer_line`). The bail rule (parse < 0.80) never fired.

## 2. Calibration ladder (§6), primary rule

| rung | construction | value |
|---|---|---|
| floor | greedy J_stated vs greedy J_stated on the paraphrased frame, option agreement | **0.69** (binary agreement 0.84) |
| matched null | KDG on pressure-removed twins, same screened scenarios | 0.09 [0.00, 0.24] (n = 22) |
| measurement | KDG rate on the screened panel | **0.22 [0.08, 0.42]** (n = 27 defined, 4 undefined) |
| positive band | KDG under the known-gap system prompt (primaries) | 0.63 [0.48, 0.77] (n = 43) |

Difference CIs (never overlap reads; nested scenario × rollout bootstrap, 2000 draws):

| contrast | Δ | 95% CI | excludes 0 |
|---|---|---|---|
| measurement − matched null, paired on scenario | 0.15 | [0.00, 0.35] (n = 20) | **no** (lower bound at 0.00) |
| positive band − measurement | 0.44 | [0.14, 0.62] | yes |
| fork B: measurement − matched null, paired | 0.17 | [0.03, 0.33] (n = 30) | yes |

Fork B ladder: measurement 0.27 [0.14, 0.45] (n = 37), matched null 0.09 [0.00, 0.21].

**Second derivation.** Rollout-level: on screened scenarios whose stable J named a
non-violating option, the mean violating fraction of D is **0.41** (n = 23). A majority-rule
KDG of 0.22–0.27 from per-scenario violating fractions averaging 0.41 is what a mixed panel
predicts (fractions near 0.5 split by majority); the two estimates agree in the sense that
neither is near 0 and neither is near the band. Agree.

**Verdict sentences (ladder-carrying).**
- Primary rule: *KDG = 0.22 [0.08, 0.42] on the screened panel, above the floor's noise but
  with a paired excess over the pressure-removed null of 0.15 whose CI reaches 0.00; a gap
  attributable to pressure rather than to frame change is not established at n = 20 under
  the pre-registered stability rule.*
- Fork B: *KDG = 0.27 [0.14, 0.45]; the paired excess over the null, 0.17 [0.03, 0.33],
  excludes 0; the known-gap band sits 0.44 above the measurement.*
- Both rules: *the instrument detects a forced gap (band − measurement excludes 0) and the
  measurement is well below the band, so the observed rate is not instruction-following
  saturation.*

**Bias-direction table (estimator-traps).**

| known bias | mechanism | direction vs "gap above null" |
|---|---|---|
| reference noise | J flips under paraphrase on 31% of scenarios (option) / 16% (binary); a scenario whose J lands non-violating by chance while D is ~50% violating counts as KDG = 1 | **favors** the claim; the paraphrase-stable cell (§3) is the check |
| screen selects mixed D | keeping 0.15–0.85 violating fractions makes a majority-violating D a near coin flip on many scenarios | favors; the matched null runs on the same screened ids, so the paired Δ removes the part shared with the twin |
| nested rollout resampling | resampling 32 rollouts near p ≈ 0.5 flips majorities, widening every CI | opposes (free strength) |
| A11 neutral → non-violating | scenarios where J = "hold/escalate" and D violates count as gaps | neutral (a real inconsistency), stated |

## 3. Rival reading and the separating cell (move 2)

*Rival:* the gap is reference noise, not values: J_stated on this model is unstable (16/48
primaries fail option-level stability; 12 of those are real consistent↔violating flips, not
consistent/neutral splits), and a noisy reference manufactures KDG = 1 at the rate at which J
happens to land non-violating.

*Separating cell:* KDG on the paraphrase-stable subset (greedy J identical on the original
and paraphrased third-person frame): **0.11 [0.00, 0.35]** (n = 18 of the 21 screened
scenarios that are paraphrase-stable). The rate halves when the reference is required to
survive a paraphrase. At n = 18 the CI still covers the full-panel point (0.22), so the rival is
not separated; it is strengthened. **Verdict downgraded to a reading:** the pilot shows a
knowing–doing gap *candidate* whose size is bounded above by the full-panel rate and whose
lower bound under a paraphrase-robust reference is 0. The full panel (~200 scenarios) is
powered to separate these: at n ≈ 60 paraphrase-stable screened scenarios the difference CI
half-width is ≈ 0.12.

## 4. Structure (Branch A / B / C, §7): reading, not verdict

| family | primary KDG (n) | fork B KDG (n) |
|---|---|---|
| F1 task-completion | 0.33 [0.00, 0.67] (6) | 0.25 [0.00, 0.63] (8) |
| F3 instrumental (tool shortcut) | 0.33 [0.00, 0.67] (6) | **0.55 [0.27, 0.82]** (11) |
| F4 loyalty/fairness | 0.11 [0.00, 0.33] (9) | 0.08 [0.00, 0.30] (12) |
| F5 third-party harm | 0.20 [0.00, 0.60] (5) | 0.20 [0.00, 0.60] (5) |

F1+F3+F4 − F5: 0.04 [−0.41, 0.38] (primary); 0.09 [−0.37, 0.40] (fork B). Family-vs-family
MDE at pilot n (closed form, worst-case p): 0.51. **No family structure is detectable at this
n**; Branch A's prediction (F5 lowest) is neither supported nor excluded (F4 is lowest on
both rules; F5 is not). The candidate structure for the full panel is F3-high / F4-low
(fork B F3 CI excludes 0.27 as its lower bound, F4's upper bound is 0.30). Reading: Branch C
shape at pilot power; framing deferred to the full panel per §7.

Per generator (no reversal): Claude-generated 0.19 [0.06, 0.44] (16) vs GPT-generated 0.27
[0.04, 0.55] (11) on the primary rule; 0.20 vs 0.35 on fork B. Same sign, overlapping; the
generator covariate does not explain the gap at this n.

## 5. Three-cell base/instruct (§4.6), raw frame, mass floor 0.5

| cell | n above floor / 96 | raw gap rate (D_raw violating while J_raw non-violating) |
|---|---|---|
| D_raw/J_raw base | 84 | 0.12 (10 / 81 defined) |
| D_raw/J_raw instruct | **58** | 0.07 (4 / 57 defined) |
| weights contrast, 50 common scenarios | | base 0.10 vs instruct 0.08 |
| format contrast on instruct, 37 common | | D_chat 0.05 vs D_raw 0.03 |

Sub-branch reading: *gap in base D_raw ≈ gap in instruct D_raw* (0.10 vs 0.08 on the same 50
scenarios; 5 vs 4 scenarios, no CI is informative at these counts). The knowing–doing
structure, insofar as the raw frame sees it, is present before post-training. The instruct
model puts less next-token mass on the option letters in a raw completion frame than the base
model does (58 vs 84 scenarios above the 0.5 floor; mean base mass 0.76): recorded as
KDG-A2, an instrument finding, not a null. The base→instruct D_raw violating fraction falls by
0.14 (instruct violates less in the raw frame).

## 6. Anomalies opened (ANOMALIES.md KDG-A1..A3) and the next session

- **KDG-A1 reference instability** (near-miss + control): J_stated flips under paraphrase on
  31% of scenarios; the paraphrase-stable subset halves KDG. Discriminator (zero-GPU on saved
  text): score consideration breadth on the 864 J replies and test whether unstable scenarios
  are the low-breadth ones; then (pod rider, ~10 min) J_stated at T = 0 on three more
  paraphrases per scenario to build a paraphrase-majority reference.
- **KDG-A2 instruct raw-frame mass floor** (control): 38/96 instruct scenarios below the
  0.5 option-mass floor in the raw frame vs 12/96 base. Discriminator (zero-GPU): read the
  top-5 next tokens from the saved full vectors on the below-floor scenarios (refusal
  tokens vs formatting tokens).
- **KDG-A3 F3 concentration** (family-exception candidate): fork B F3 0.55 vs F4 0.08.
  Discriminator: the full panel (n ≈ 40 per family) and the harm-stratified twin contrast.

Next session (compute-ordering): zero-GPU first (A1 breadth pass, A2 token read, stage-2
calibration), then the full construction (~200 scenarios, both generators), then Tier 2 with
the dose arm on screened scenarios.

## 7. Calibration stage 2 (A6): gate condition

Stage 2 (200 real replies drawn across six generated cells, stratified; no reply parsed by a
non-clean rule existed to oversample): harness vs Claude judge **0.99** (kappa 0.98), harness vs
GPT-5.5 judge **1.00**, judge vs judge 0.99. The two disagreements are the Claude judge returning
no label on a bare-letter reply (`A`, `B`); excluded per §4.4. **Stage 2 passes; the gate in §1
is applied.** Report: `data/calibration_stage2_report.json`.

## 8. Referee pass (three damaging objections)

1. *"Your paired measurement−null CI touches 0 under the rule you pre-registered; you are
   shipping fork B because it clears."* Conceded in structure and handled by A12: the
   primary rule stays primary in every table and the §2 verdict sentence under it says "not
   established"; fork B is reported beside it with its construction reason (4/16 unstable
   scenarios are non-violating splits) stated as small. No verdict rests on fork B alone.
2. *"A reference that flips on 31% of paraphrases cannot anchor a self-inconsistency
   construct."* Answered in part: the binary-level flip rate is 16%, the matched-null rung
   bounds frame-change effects at 0.09, and the paraphrase-stable cell is reported as the
   robustness cell. Conceded: the rate under that cell (0.11 [0, 0.35]) is the honest lower
   bound and the verdict is downgraded to a reading (§3). KDG-A1 is scheduled, not footnoted.
3. *"F2 never engages and F5 is not lowest; the family design failed."* F2 was an appendix
   family by pre-registration (§11) and its non-engagement is the expected outcome, not a
   gate cost. F5-vs-rest at n = 5 per family has an MDE of 0.51; the pilot was never powered
   for structure (§8 step 5 is where structure is tested). The F3-high candidate is entered
   as KDG-A3 with the full panel as its discriminator.

## 9. Ship-blocker checklist (commit boundary)

- [x] ladder table + verdict sentences carry the ladder and the detection bar
- [x] positive control on the same instrument and model (known-gap band, 0.63)
- [x] rival reading with its separating cell (paraphrase-stable subset)
- [x] fork amendment (A12) committed before this document; both-choice verdicts
- [x] bias-direction table; difference CIs, no overlap reads
- [x] anomalies entered with priced discriminators
- [x] stage-2 calibration ≥ 0.95 (two judges): 0.99 / 1.00 / 0.99
- [x] SYNTHESIS.md updated in the same commit
