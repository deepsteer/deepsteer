# CST T0.4: power table from saved arrays (draft for the G0 report)

Zero GPU, zero API. Script `scripts/t04_power.py`; numbers `outputs/t04_power.json` (every source
file listed there with its sha256; seed 20261104, 10,000 bootstrap draws). Subject of every row is
Qwen3.8-27B under `none` unless stated. The ST rollout files it reads are untracked pod pulls; their
hashes are the provenance.

## 1. S on the pressure triple: what the saved rollouts can say

S does not exist yet. The proxy is the per-scenario **departure share**: among parsed thinking-on
rollouts (effort medium, cap 1,536, T 0.7), the share whose action is the norm-violating option.
S is conditional on an applied tenet, so only a fraction f of the n·k rollouts enter it, and a
scenario with no applied rollout in either arm drops out. The closed form carries both:

MDE = 2.80 · sqrt((2·w·E[1/m | m > 0] + 2·τ²) / (n · (1 − (1 − f)^k)²)), m ~ Binomial(k, f),

with w = E_s[p_s(1 − p_s)] (within-scenario Bernoulli variance) and τ² the run-to-run excess
measured on the twin.

**Variance components (thinking on, `none`):**

| set | n | mean departure | w | scenarios at 0 / at 1 |
|---|---|---|---|---|
| H1_held | 132 | 0.615 | 0.098 | 0.17 / 0.36 |
| H1_held engaged (ST's Qwen subset) | 17 | 0.250 | 0.175 | 0.29 / 0 |
| panel rule-right (ST specificity set) | 60 | 0.008 | 0.005 | 0.98 / 0 |
| cell A′ forced items | 102 | 0.271 | 0.103 | 0.44 / 0.08 |

**Twin-noise band (matched null).** The only thinking-on none-vs-none twin on this model is H1_held:
ST's selection run (st2q, vLLM) against its independent re-run (st2qp, vLLM bf16), 8 rollouts each.
Mean difference −0.008, 95% band [−0.035, +0.017]. The per-scenario difference's SD (0.154) sits
below the binomial-only prediction (0.161), so τ² = 0: no run or engine excess is detectable on H1.
A within-run split by rollout index (0-3 against 4-7) reads −0.031 [−0.063, 0.000]; rollout index
carries the letter order, so that split is not a clean null and is not used.

**MDE on a paired ΔS (arm − `none`), by k and applied fraction f:**

| set (variance basis) | k | f 0.3 | f 0.5 | f 0.8 | f 1.0 |
|---|---|---|---|---|---|
| panel, n 586 (H1_held w) | 2 / 4 / 8 | 0.096 / 0.059 / 0.039 | 0.063 / 0.041 / 0.028 | 0.044 / 0.030 / 0.021 | 0.036 / 0.026 / 0.018 |
| panel, n 586 (worst case w 0.25) | 2 / 4 / 8 | 0.153 / 0.093 / 0.062 | 0.100 / 0.066 / 0.045 | 0.070 / 0.048 / 0.033 | 0.058 / 0.041 / 0.029 |
| H1_held, n 132 | 2 / 4 / 8 | 0.202 / 0.123 / 0.082 | 0.132 / 0.087 / 0.059 | 0.092 / 0.063 / 0.044 | 0.076 / 0.054 / 0.038 |
| cell A′ forced, n 102 | 2 / 4 / 8 | 0.235 / 0.143 / 0.095 | 0.153 / 0.101 / 0.069 | 0.107 / 0.074 / 0.051 | 0.089 / 0.063 / 0.045 |
| H1 engaged, n 17 | 2 / 4 / 8 | 0.751 / 0.458 / 0.303 | 0.488 / 0.324 / 0.219 | 0.341 / 0.235 / 0.162 | 0.284 / 0.201 / 0.142 |

**Second derivation (agree).** The empirical scenario bootstrap of the observed twin differences on
H1_held, with rollouts split into disjoint blocks of k, against the closed form at f = 1:
k 8: 0.037 against 0.038; k 4: 0.055 against 0.054; k 2: 0.075 against 0.076. The two agree within
3% at every k.

## 2. The operator condition's scale on Qwen (dose 0 only)

On the full panel (586, t10b screen, dose 0, letter readout), the known-gap cell (the frontier
exhibit's operator instruction) puts 0.769 of the option mass on the violating action, against 0.115
in the user-pressure cell and 0.084 in the pressure-removed cell.

| contrast | mean [95% CI] | per-scenario SD |
|---|---|---|
| g_band = known-gap − J (primary) | 0.653 [0.633, 0.672] | 0.237 |
| operator − user (known-gap − D primary) | 0.655 [0.636, 0.674] | 0.235 |
| operator − plain (known-gap − D pressure-removed) | 0.685 [0.666, 0.704] | 0.235 |
| user − plain (D primary − D pressure-removed) | 0.031 [0.022, 0.039] | 0.105 |
| twin, known-gap on two pods (293 scenarios) | −0.0001 [−0.0015, +0.0012] | |

The frontier reference (GPT-6.1 Sol, effort medium): g_band 0.900 [0.877, 0.921]. The screen's
64-scenario C0 value on Qwen was 0.689 [0.629, 0.746]; the full panel gives 0.653. On this read the
operator instruction moves the action about 20 times as far as the user-turn incentive does, so the
user and operator conditions are not two doses of one pressure, and ΔS_op and ΔS_user will sit on
different scales.

## 3. ΔP on H1_held (ST's secondary)

| model, set | n | P_none | closed-form null MDE (k 8) | MDE recorded with an arm | regression to the mean (P_re − P_sel) |
|---|---|---|---|---|---|
| Qwen3.8-27B, engaged | 17 | 0.250 | 0.142 | 0.163 (care-DPO_Q), 0.100 (care-DPO_Q2) | −0.088 [−0.175, −0.005] |
| Muse Glimmer 30B, engaged | 54 | 0.123 | 0.056 | 0.104 (care-DPO) | not computed |

At n 17 the engaged subset resolves only effects of about 0.15 and above. The selection itself
moves P_none by −0.09 on re-run, which is why ST read every ΔP against an independent `none`.

## 4. MoReBench response field at matched length (MRB v0.9.3 estimator, k 2)

From the P2 rows (`none` and the attached ST adapter), random 200-scenario subsets (20 subsets, 400
bootstrap draws each):

| model (basis) | MDE on Δ_adj, 500 | MDE on Δ_adj, 200 | bar −0.87: failure detected (80%) if the true cost is below | "beats −0.87" (80%) needs a true cost above |
|---|---|---|---|---|
| Qwen3.8-27B (care-DPO_Q rows) | 1.37 | 2.05 | −2.92 | +1.18 |
| Muse Glimmer 30B (care-DPO rows) | 0.83 | 1.30 | −2.17 | +0.43 |

The `none`-only null (k0 against k1, raw score, scaled to k 2 at n 200) gives 1.13 on both models.
Read: at 200 scenarios the bar's interval half-width exceeds the bar's own distance from 0, so the
200-subset read cannot place an arm on either side of −0.87 unless its cost is about −2.9 or worse
(Qwen basis). Qwen care-DPO_Q's matched-length cost of record (−2.83) sits at that edge. The plan's
claim "beat −0.87" is not testable at n 200; at 500 it needs a true cost above about +0.5 (Qwen
basis).

## 5. MACHIAVELLI (MCH rows, k 2, 30 test games, 1,024 tokens)

| model | violations MDE (formula; measured paired SE) | reward MDE (formula; measured) | ethics prompt's effect |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 3.7; 3.6 | 2.8; 2.4 | violations −10.7, reward −0.6 |
| GPT-OSS-20B | 5.9; 4.3 | 2.9; 2.9 | violations −17.4, reward −0.2 |

The CST subject is Qwen3.8-27B with thinking on; neither row is that model (bias table).

## 6. Capability (t15_capability_v1, n 200, Qwen st2qp rows)

Accuracy under `none` 0.865; paired discordance 0.08-0.09 against the three adapters; paired SE 0.020
to 0.021; MDE 0.056-0.059. The bar of record is a point rule (Δ ≥ −0.02), not a CI rule: with SE
0.021, an arm with no true loss fails it 17% of the time, an arm with a true loss of 0.02 passes it
half the time, and one with a true loss of 0.06 passes it 3% of the time.

## 7. Recommendations

- **Panel (586) under plain / user / operator: k = 4.** MDE on ΔS 0.041 at f 0.5 (0.066 on the
  worst-case variance), against operator-condition effects on the dose-0 read of 0.65.
- **H1_held (132): k = 8.** MDE 0.059 at f 0.5; k 4 gives 0.087, coarser than ST's ΔP resolution.
- **Cell A′ forced (102): k = 8.** MDE 0.069 at f 0.5.
- **H1 engaged (17): keep ST's k = 8 for ΔP and do not read S on it alone** (MDE 0.22 at f 0.5).
- **f is the largest unknown.** Measure it on A0's first 20 panel scenarios in S0 and re-run this
  table before any arm trains; below f 0.3 the panel MDE at k 4 passes 0.06.
- **MoReBench: read the full 500 on every arm, or treat the 200 subset as a gross-failure screen
  only** (MDE 2.05 at 200 on the Qwen basis cannot resolve a −0.87 bar).
- **MACHIAVELLI: k = 2 as MCH.** MDE 3.7 on violations (other model).
- **Capability: n 200 as ST.** State the bar's pass probabilities beside it, or move to a CI rule by
  amendment.

## 8. Bias-direction table

| known bias | mechanism | direction relative to the stated MDEs |
|---|---|---|
| applied-tenet fraction varies by scenario | Jensen: E[1 − (1 − f_s)^k] ≤ 1 − (1 − f̄)^k, so more scenarios drop than the iid-f form predicts | MDEs understated (retraction risk); measure f_s in S0 |
| null variance only | an arm's effect varies across scenarios and inflates Var(d_s); the recorded arm MDEs run 15% to 85% above the null form (Qwen 0.163 vs 0.142; Muse 0.104 vs 0.056) | MDEs understated |
| applied-tenet rollouts may be more decisive | rollouts that cite a tenet may sit nearer 0 or 1, lowering w | MDEs overstated (free strength) |
| panel variance borrowed from H1_held | no thinking-on read of the panel's pressure items exists on Qwen; H1's w 0.098, worst case 0.25 shown | unknown, bounded by the worst-case row |
| judge noise in the applied-tenet label | misattributed tenets add variance and attenuate differences | MDEs understated, effects attenuated |
| twin across engines with τ² = 0 | the only thinking-on twin finds no run excess; a session on another engine could add one | MDEs understated if engines change |
| MRB basis includes adapter heterogeneity | SE from arm rows exceeds a null SE (the `none`-only null gives 1.13 against 2.05) | MoReBench MDE overstated |
| MCH rows are another model | Qwen3-30B-A3B-Instruct-2507, non-thinking, MoE | unknown |

## 9. Not grounded in a file

- Thinking-on departure on the panel's primary (pressure) items for Qwen3.8-27B: only the 60
  rule-right items and dose-0 letter reads exist.
- Thinking-on behavior under the operator (known-gap) condition on Qwen: dose 0 only.
- f (the applied-tenet fraction) and the panel's κ on tenet application: no data.
- A0s's effect size (the constitution as a system prompt), which R2 (d) gates on: no data.
- MACHIAVELLI and MoReBench on Qwen3.8-27B with a constitution-trained arm: no data; the bases above
  are other models or other adapters.
- A thinking-on none-vs-none twin on the panel: none exists; the dose-0 panel twin is numeric only
  (±0.0013).
- The rule that picks MoReBench's 200 scenarios: not registered; random subsets were used here.

## 10. Re-run for the ordered-option share (Amendment 14 item 3, 2026-11-08)

Script `scripts/t04d_ordered_power.py`, record `outputs/t04d_ordered_power.json`; S0a's operator
cell, 458 clear items, k 4 per arm, every rollout with an action (no f thinning). The script asserts
its shares equal the report's (`s0a_reads.json`): A0 0.311, A0s 0.031, gain −0.280. Within-scenario
variance w: A0 0.138, A0s 0.022. Effect heterogeneity of A0s − A0 across scenarios: 0.057.

| k | arm − A0, null MDE | arm − A0s, null MDE | arm − A0s, MDE with A0s's heterogeneity | A0s band half-width (95%) | arm gain / A0s gain, MDE |
|---|---:|---:|---:|---:|---:|
| 2 | 0.049 | 0.020 | 0.037 | 0.034 | 0.13 |
| **4** | **0.034** | **0.014** | **0.034** | **0.029** | **0.12** |
| 6 | 0.028 | 0.011 | 0.033 | 0.027 | 0.12 |
| 8 | 0.024 | 0.010 | 0.033 | 0.026 | 0.12 |

**Second derivations (agree).** A0's k 4 split into disjoint k 2 blocks: bootstrap SE of the
null-twin difference 0.0171 against the closed form's 0.0173. A0s − A0 at k 4: bootstrap SE 0.0147
against 0.0146 with the measured heterogeneity (0.0093 without it, so the heterogeneity term is
most of the variance).

**k stays 4.** Past k 4 the comparison of an arm with A0s is floored by scenario heterogeneity
(0.034 → 0.033 at k 8), and A0s's band narrows by 0.003; raising k would also re-generate A0s, the
costliest arm to read (CST-A11). An arm's gain is resolved against A0s to about 0.034, or 12% of
A0s's gain.

**The specificity bar (Amendment 14 item 4) is below cell A′'s resolution.** On 51 pairs at k 8
the shift on the rule-right originals has SE 0.029 (MDE 0.082). Against the 0.03 bar: an arm with
a true shift of 0 passes a point rule with probability 0.85 and a CI rule (upper bound below 0.03)
with probability 0.17; a true shift of 0.03 passes the point rule half the time; A0s's measured
+0.088 passes neither (0.02, <0.001). Raising k to 32 at best halves the SE (0.015), and the CI
rule's pass rate at a true 0 is then about 0.5. Returned to the author: the rule's form (point or
CI) and whether the bar's n grows (more rule-right items, e.g. ST's 60 specificity items).

| known bias | mechanism | direction relative to the stated MDEs |
|---|---|---|
| heterogeneity borrowed from A0s − A0 | an arm close to A0s may vary across scenarios less than A0s does against A0 | arm − A0s MDE overstated, bounded below by the null column |
| one session, one engine | no run-to-run excess measured across sessions | MDEs understated if an arm's read runs on another engine or batch |
| capped rollouts excluded | 7 A0s rollouts capped of 12,304; an arm with longer traces loses more | small; grows with an arm's trace length |
| specificity SE from one A0s − A0 contrast | an arm's per-item variance on A′ differs from A0s's | unknown |
