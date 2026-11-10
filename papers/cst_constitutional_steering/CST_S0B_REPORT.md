# CST S0b-1 report (2026-11-09): A0n, the CST-A15 cut read, the CST-A16 dose-0 read

S0b-1 (`S0B_BUILD.md` §4; Amendments 16-19, 23) read A0n (Faraday's lectures, 34,173 tokens, in
A0s's system-message slot under a neutral header), ran the CST-A15 cut stage on A0, A0s and A0n, and
read A0n at dose 0. One A100, 5.11 h against a price of 4-6.5 h; the pilot gate passed (parse 1.00,
capped 0, 1,854 rollouts/h). Records: `outputs/s0b_a0n_reads.json`, `outputs/s0b_cut_reads.json`,
`outputs/s0b_dose0_reads.json`; scripts `scripts/s0b_{a0n,cut,dose0}_reads.py`; raw rollouts and
replies under `outputs/s0b/` (gitignored).

## 1. A0n: the prefix carries about half of the in-context effect (CST-A20)

Operator cell, 458 clear items, ordered-option share: A0 0.311, **A0n 0.153**, A0s 0.031.

| read | value |
|---|---|
| A0n − A0 | −0.158 [−0.184, −0.133] |
| ratio (A0n − A0)/(A0s − A0), probability scale (rider a, of record) | 0.563 [0.497, 0.629]; lower bound 0.493-0.496 across bootstrap seeds: **partial**, a near-miss of the 0.5 bar |
| ratio, log-odds scale (the fork) | 0.343 [0.281, 0.414] |
| relative reductions | the prefix 0.51 (A0n against A0); the content given the prefix 0.80 (A0s against A0n) |
| references for every arm (Amendment 23 item 2) | full band −0.280; content band −0.123 additive, −0.238 carried to A0's odds |
| operator-cell trace medians | A0 432, A0n 519, A0s 1,128 tokens |
| dose 0, operator cell (the second derivation) | A0n − A0 −0.189, A0s − A0 −0.335: ratio 0.56 with no trace generated |

**R_injection (descriptive; patterns registered in Amendment 23 item 3):** injection or test talk in
0.41 of A0's traces, 0.62 of A0n's, 0.05 of A0s's; the order is attributed to the operator in every
arm. Ordered share with and without such talk: A0 0.08 and 0.47, A0n 0.07 and 0.30, A0s 0.06 and
0.03. The rise in injection talk accounts for **0.31 to 0.53** of A0n's drop by the two decomposition
orders, so it carries a third to a half of the prefix effect.

## 2. CST-A15: R_conclusion under both cuts (the guard framing stands)

Residual-free rows (no choice stated in the trace's last two sentences after the cut), the verdict's
scope (Amendment 19 item 6):

| | two-sentence cut (registered) | decision-block cut (fork) |
|---|---|---|
| retest kappa on the concluded option (rider b) | 0.98 | 0.90 |
| splice: concluded option follows the closing | 0.357 [0.287, 0.433] | 0.389 [0.322, 0.455] |
| A0 cut-lines S_concl | 0.004 [0.000, 0.009] | 0.031 [0.022, 0.041] |
| A0s / A0n cut-lines S_concl | 0.002 / 0.006 | 0.054 / 0.031 (within-arm only) |

The splice CI lies below 0.5 under both cuts: the blind judge reads the reasoning's body, not the
closing lines. With the closing lines cut, S_concl stays below 0.05 on A0. **CST-A15 resolves as
R_conclusion:** at baseline Qwen's actions follow its traces' conclusions; S_concl is a guard at its
floor, and an arm that raises it opens a reasoning-action gap (Amendment 14 item 1). The block cut
removes about 13 sentences of A0s's reasoning, which is why its A0s value is not read across arms.

## 3. CST-A16: the dose-0 shift is specific to the constitution's content

P(violating) at dose 0 on the clear items, A0n − A0 against A0s − A0: plain −0.024 [−0.032, −0.016]
against +0.043; user −0.023 [−0.032, −0.014] against +0.061; KDG's judgment cell +0.023
[+0.015, +0.032] against +0.113. The neutral prefix does not reproduce A0s's shift toward the
violating option on the unpressured cells (it moves the other way on plain and user), which rules out
R_prefix_readout; it also falls just outside R_disposition's registered ±0.015 band, so neither
registered branch reads exactly, and CST-A16 records the result as a reading: the dose-0 shift
belongs to `C_anth`'s content, and dose-0 reads stay beside the reasoned reads (Amendment 14 item 5).

## 4. Referee pass

1. *"A0n is not neutral: a scholarly frame changes the deployment the model infers."* Partly
   conceded: frame talk appears in 0.06 of A0n's traces, and R_document is one of CST-A20's three
   readings. The next document-engine pod's variants (the order before the document; a short neutral
   prefix) separate a frame from a length effect (Amendment 23 item 6).
2. *"Your ratio's lower bound is 0.497; the branch is a coin flip."* Answered: the bound is 0.493 to
   0.496 across ten bootstrap seeds, stably below the bar, so the registered rule reads partial; the
   near-miss and the log-odds fork (0.343) are stated beside it, and no gain tier was moved because of
   it (Amendment 23 item 2).
3. *"The cut read cannot detect a gap the judge never sees."* Answered by the splice control: when a
   body and a closing disagree, the judge follows the body on most traces (the closing on 0.36-0.39),
   so it reads reasoning, and the cut S_concl stays near 0 on A0 with a retest kappa of 0.98.
