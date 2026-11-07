# CST line re-forecast at measured rates (Amendment 8 item 5; 2026-11-06)

For the author before S0a launches. Source: `scripts/t09_reforecast.py` writes
`outputs/t09_reforecast.json`; the second VALIDATE's records are in `outputs/validate/`. Dollars at
$1.6 to $2.0 per A100-80GB hour; envelope $500 to $1,000; envelope of record 150 A100-h; the dollar
cap governs. Spent so far: the two VALIDATE pods, 2.8 pod-hours (about $5).

## 1. Unit prices

| unit | value | basis |
|---|---|---|
| A0 generation, thinking on | 819 tokens/s; 6,888 rollouts/h at 428 tokens | measured, 256 rows |
| Generation with `C_anth` in context (A0s) | 150 tokens/s; 452 rollouts/h at 1,197 tokens | measured, 128 rows; CST-A11 |
| SDF / SFT / DPO training throughput | 411 tokens/s; 10M tokens in 6.8 h | measured (V5) |
| Judge scoring, one arm (12,304 rollouts) | gpt-oss-120b 0.9 h; Olmo-3.1-32B-Instruct 2.2 h; third judge 2.2 h | measured; the third (Mistral) estimate set equal to Olmo's |
| Full read of one trained arm | 10.0 h (generation 2.5 at 600 tokens, judges 5.2, capability 1.2, N 1.0, dose-0 0.1) | judges and generation measured; capability carried; N estimated |
| Finalist bars | 3.6 h (MRB 500 k 2 1.5, MACHIAVELLI k 2 1.75, refusal 0.3) | carried from MRB and MCH |
| DPO, one draw of 3,000 pairs, 3 epochs | 16.1 h (1 epoch: 5.4) | 411 tokens/s, 1,100-token pairs, reference forward at 0.2 |

The plan's §8 unit prices were 3.5 to 4 h per adapter read and about 2 h per DPO draw. The measured
read costs about 2.5 times as much: three judges on 12,304 rollouts add 5.2 h even at the kept
judges' short outputs (medians 76 and 223 tokens). A DPO draw at §4.3's 2,000 to 4,000 pairs and
the 3-epoch cap costs 8 times ST's 40-pair draw.

## 2. Forecast by session (A100-h)

| session | current engine | engine fixed (CST-A11's R_engine) |
|---|---:|---:|
| Probe pod: engine tuning, Mistral judge check, the letter fix on live judges | 1.5 | 1.5 |
| S0a (measured; panel gpt-oss + Olmo-Instruct + Mistral, cap 4,096, 10% margin) | 47.5 | 23.1 |
| S0b (A0s riders, N on A0 and A0s, A0s's length-matched fork) | 11.1 | 6.9 |
| S1 (§4.1 with the document in context and its self-critique pass, §4.2, A1, A1c, recovery) | 54.9 | 24.7 |
| S2 (A1 and A1c reads, A1-without-recovery format read, the winner's bars) | 24.3 | 24.3 |
| S3 (A2 two draws, A2 without the rewrite, A3; reads and bars) | 128.4 | 115.5 |
| **Partner recipe, S0a-S3** | **267.8 ($428-536)** | **196.0 ($314-392)** |
| S4 (B1 on the sibling; estimate) | 29.0 | 29.0 |
| S5 (A4, A4f; gated by R3; estimate) | 57.2 | 57.2 |
| **Whole line** | **354.0 ($566-708)** | **282.2 ($452-564)** |

The engine case is the largest single uncertainty: about 72 A100-h across S0a, S0b, S1 and S3,
because §4.1's documents and §4.3's rewrites are generated with the document in context, the path
that runs at 150 tokens/s now. The probe that decides it costs about 0.3 A100-h.

## 3. Ranked cut list (applied cumulatively to the partner-recipe path)

| rank | cut | saves (fixed / current) | recipe after (fixed / current) | loses | amendment? |
|---|---|---:|---:|---|---|
| 1 | DPO at 1 epoch, not 3 | 32.1 / 32.1 | 163.9 / 235.6 | dose; ST-A38 moved protection with pair count, not epochs | no (inside "≤ 3") |
| 2 | Defer A2 without the rewrite | 18.7 / 18.7 | 145.2 / 216.9 | the lab's rewrite-ablation replication | yes |
| 3 | A0s on the operator condition only (panel k 4, H1 operator k 8) | 8.1 / 25.8 | 137.0 / 191.1 | A0s's plain, user and operator_pr cells; A0s is a comparator and its band is provisional until S0b | yes |
| 4 | One A2 draw, not two | 20.1 / 26.6 | 116.9 / 164.5 | the draw-to-draw read ST-A38 asks for | yes |
| 5 | SDF corpus 5M tokens, not 10M | 11.9 / 26.9 | 105.0 / 137.6 | dose (the 1M-token pilot prices it anyway) | yes |
| — | S5 (already gated by R3) | 57.2 | — | GRPO | registered gate |
| — | S4, B1 on the sibling | 29.0 | — | the stage question on the sibling | yes (author) |

With the engine fixed, cuts 1 and 2 bring the partner recipe to 145 A100-h ($232-290), inside the
envelope of record. With the engine as it is, cuts 1-5 reach 138 A100-h.

## 4. Found while pricing

- §5 says the A2-without-rewrite ablation rides along "at zero extra training". It needs its own DPO
  run (chosen = a second sample), so it is priced here as one draw at the same dose (cut 2).
- §4.1's corpus and its self-critique pass are generated with `C_anth` in context, so their price
  moves with CST-A11; the 1M-token pilot in S1 measures it at the engine of record.

## 5. Bias-direction table for the forecast

| known bias | mechanism | direction |
|---|---|---|
| A0s's rate applied to every condition | measured on the operator condition; plain and user prompts are shorter | overprices S0a |
| Relabel and long-prompt calls priced at the probe's seconds per call | 8 calls ran concurrently; 72 calls batch better | overprices (small) |
| Trained arms' generation at 600 tokens per rollout | the toy SDF adapter's median; a stronger dose may lengthen traces toward A0s's 1,197 | either way; at 1,197 tokens each arm read costs 2.5 h more |
| DPO throughput from packed 2,048-token SDF sequences | 1,100-token pairs pad and pack less well | underprices DPO |
| Mistral's scoring rate set equal to Olmo-Instruct's | unmeasured; dense 24B, short outputs | either way; range 0.9 to 4.4 h per arm |
| Capability, MRB, MACHIAVELLI carried from other lines | other models, other engines | unknown |
| S4 and S5 | estimates, not built from measured units | unknown; S5 is gated |

## 6. Recommendation

1. The probe pod first (1.5 A100-h, about $3): three engine configurations on A0 and A0s (prefix
   caching off for A0, MTP off, `max_num_seqs`), Mistral-Small-3.2-24B's judge check with the
   validity read (concluded option against the subject's answer), and the letter-mapped scorer on
   the kept judges. Its result picks the column of §2.
2. Cut 1 (DPO at 1 epoch) needs no amendment and saves the most; cut 2 needs one sentence. The rest
   are the author's choice against the dollar cap once the probe reports.
