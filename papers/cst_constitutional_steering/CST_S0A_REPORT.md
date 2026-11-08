# CST S0a report (2026-11-08): A0 and A0s on Qwen3.8-27B

S0a (`S0A_BUILD.md`; Amendments 7-13) read A0 (`none`) and A0s (`C_anth` in the system message)
with the instruments every later arm is read with. Pods: four sessions, 3.7 + 8.3 + 4.5 + about
1.3 h, plus two stopped launches (CST-A13, the launcher fix). Records: `outputs/s0a_core.json`,
`outputs/s0a_reads.json`, `outputs/claude_stenet_a0.json`, `outputs/claude_stenet_a0s.json`; raw
rollouts and replies under `outputs/s0a/` (gitignored). Scripts: `scripts/s0a_analysis.py`,
`scripts/s0a_reads.py`.

## 1. What ran

| | A0 | A0s |
|---|---:|---:|
| rollouts (panel 9,376, H1_held 2,112, cell A′ 816) | 12,304 | 12,304 |
| truncated at cap 4,096 | 0 | 7 |
| operator-cell trace median (clear items) | 422 | 1,093 |
| gpt-oss-120b blind scores (panel and H1) | 11,488 | 11,488 |
| Olmo's 10% | 1,148 | 1,148 |
| Claude S_tenet read, seeded 200 × 4 (scored / unscored / pending) | 759 / 41 / 0 | 241 / 15 / **544** |

## 2. The instruments (gates and calibration)

- **S_concl's scorer (gpt-oss-120b, blind, temperature 1, effort low).** Retest on 100 A0 rollouts:
  concluded option kappa 1.00 same-order and 1.00 permuted (the gate passes on this task); applied
  tenets 0.74 [0.68, 0.79] and 0.72 [0.66, 0.78] (gpt-oss holds no tenet task after Amendment 11).
  Its blind conclusion equals the subject's answer on 11,475 of 11,475 A0 rollouts. Olmo's 10%
  disagrees on the concluded option on 0.6% (A0) and 1.3% (A0s).
- **S_tenet's reader (Claude).** Probe retest 0.84 on applied tenets (deference 0.86). On A0's
  read: 5.1% unscored after three attempts; stops independent of scenario family (chi-square p
  0.52). The validity bar of record (agreement with the frontier second opinion on 200 rollouts)
  is not yet read.
- **Layer A (reported, Amendment 13 item 3):** the frontier tier 51 of 57 settled items; gpt-oss
  20 of 27 settled, 20 of 51 overall.
- **Third-family votes (Amendment 13 item 4):** Olmo's self-agreement kappa 0.31 (`T_core`) and
  0.42 (`T_conf`), below 0.8; `T_core` stays 116 two-family matches with 214 disputed; the 13
  `T_conf` pairs stay out of training.
- **Capability parser (Amendment 13 item 1, CST-A14):** leading answer first; ST's beside.

## 3. Results

**S_concl** (the registered primary) is **0.000 on every cell for both arms** (headline operator
cell, clear items: A0 0 of 1,826, A0s 0 of 1,829; largest anywhere 0.0015). No-conclusion rates
0.0-0.4%. This is CST-A15: at baseline the actions follow the traces' conclusions.

**Actions on the 458 clear directed-away items** (shares among rollouts with an action; Δ A0s − A0,
scenario-paired bootstrap 95%):

| cell | ordered option, A0 → A0s | Δ | directed option, A0 → A0s | Δ | o3, A0 → A0s |
|---|---|---|---|---|---|
| plain | 0.005 → 0.003 | −0.002 [−0.007, 0.001] | 0.858 → 0.859 | +0.001 | 0.083 → 0.085 |
| user | 0.004 → 0.003 | −0.001 | 0.822 → 0.823 | +0.001 | 0.141 → 0.144 |
| **operator** | **0.311 → 0.031** | **−0.280 [−0.310, −0.253]** | 0.572 → 0.783 | +0.211 [0.182, 0.239] | 0.099 → 0.155 |
| operator_pr | 0.382 → 0.039 | −0.343 [−0.372, −0.315] | 0.540 → 0.799 | +0.259 | 0.052 → 0.113 |

**S_tenet (A0, operator, seeded 200 × 4):** 0.404 [0.349, 0.459]; f 0.841; silent 2.1%,
unsettled-only 8.6%, mixed 0, unscored 5.1%, no action 0.5%. **A0s's S_tenet is not read:** 544 of
its 800 rollouts are pending (the reads stopped at the account's usage limit); the 241 scored are
the families read first and are not a sample of the subset.

**ST's outcomes (H1_held's 82 clear items, the protective option):** plain 0.655 → 0.802, ΔP
+0.146 [0.105, 0.191]; operator (ordering the rule) 0.024 → 0.251, +0.227 [0.166, 0.294]. E at
dose 0 and the forced-choice read on cell A′ are saved and not yet computed.

**Capability (n 800, paired, parser of record):** A0 0.916, A0s 0.924, Δ +0.008 [−0.008, +0.023]:
the bar (CI not entirely below −0.02) passes. ST's parser: Δ +0.035 [+0.011, +0.060] (CST-A14's
style effect).

**Recall on A0 (per token; margin = true − paraphrase 1, floor = paraphrase 1 − paraphrase 2):**
public domain (the positive control) margin 3.55 nats against floor 0.21, 90% of windows above the
floor's 95th percentile; `C_anth` 0.75 against 0.21, 15%; `C_spec` 0.67 against 0.30, 19%.

## 4. Readings and their rivals

- **A0s removes most operator-induced compliance** (−0.28 on the headline cell) and leaves the
  unpressured cells unchanged. Rival: the A0s traces are 2.6 times longer (CST-A8); the effect may
  ride the length, not the content. Separating cell: S0b's length-matched fork and N on A0s
  (Amendment 8 item 4); until then the A0s band is provisional. The o3 rule (Amendment 6): o3 rises
  +0.056 while the directed option rises +0.211, so most of the shift goes to the directed option,
  not to "stopped acting".
- **S_concl cannot register the effect at baseline** (CST-A15). Rival: the blind judge reads the
  conclusion off the trace's closing lines, so a gap would be invisible to it. Separating cell
  (zero GPU): rescore with the closing lines cut.
- **Recall:** the document's text is somewhat more predictable to Qwen than a paraphrase (15% of
  `C_anth` windows above the floor's 95th percentile, against 90% for public-domain text). Rival:
  the paraphrases are less fluent than the source, which favors the source without memorization;
  the floor (two paraphrases) bounds that only in part.

## 5. Referee pass

1. *"Your primary statistic is zero on both arms, so S0a measured nothing."* Conceded for S_concl
   (CST-A15, escalated). The action shares and ΔP are pre-registered secondaries (ST's outcomes) and
   move with CI-clean margins; S_tenet registers the operator effect on A0.
2. *"A0s's effect is length, not content."* Not answered at S0a; it is the registered S0b read,
   and the band is provisional by rule.
3. *"The S_tenet read is incomplete and its reader was rate-limited."* Conceded for A0s (not
   reported as a number); A0's read is complete with 5.1% unscored and stops independent of family.

## 6. Open (for G1)

The author's ruling on CST-A15 (which statistic heads G1); A0s's S_tenet read (544 pending); the
frontier second opinion (validity bar of record; Claude 200, then Codex after its quota); E at dose
0 and the forced-choice read (ST's definitions, `analyze_phase1_session_a.four_cells`,
`analyze_st2qp.forced_read`); the S_concl cut-lines discriminator; S0b.
