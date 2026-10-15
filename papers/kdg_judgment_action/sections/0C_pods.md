# Appendix C. Pilot, full panel, and union: the numbers by pod {#app:pods}

**Sample flow (v2).** \Cref{tab:flow} gives every scenario set the paper reads and how it is formed. Each row
carries its own pressure-removed twin as a second pair of frames, so pressure-removed twins are not counted
as rows; "harm twins" are separate rows that vary the level of harm.

| set | scenarios | formed as | read in |
|------------|------|-------------------|-------------|
| pilot pod | 96 | 60 primaries + 36 harm twins | \Cref{tab:pods}, first column |
| full-panel pod | 320 | the pilot's 96 + 140 primaries + 84 harm twins | \Cref{tab:pods}, second column |
| round 2 | 120 | 48 primaries + 32 harm twins + 40 swapped F4 scenarios | \Cref{tab:pods} union; \Cref{tab:swap} |
| union of the panel pods | 440 | 248 primaries + 152 harm twins + 40 swaps | \Cref{gap}, \Cref{reference}, \Cref{structure} |
| analysis set | 397 | 440 less the 40 swaps and 3 flagged F4 harm twins | \Cref{base}, \Cref{app:three-cell} |
| gate-family primaries | 208 | 248 less the 40 F2 primaries | the screen (\Cref{instruments}): 74 pass |
| round 3 | 192 | 112 primaries + 80 harm twins (3 flagged F4 harm twins) | the 586 only |
| letter-only union | 586 | 632 rows less 40 swaps and 6 flagged: 360 primaries + 226 harm twins | \Cref{base}, \Cref{recipe}, GPT-OSS-20B |
| rule-harms set (separate) | 264 | 270 generated (135 per generator), kept by both raters | \Cref{rule-harm} |

Table: Sample flow from the generated rows to each analysis set. Screened subsets (136, 130, 110 on OLMo-3;
each model's own screen in \Cref{tab:recipe}) are drawn from these sets and named where used. {#tab:flow}

| quantity | pilot (96) | full panel (320) | union (440) |
|---|---|---|---|
| gate-family primaries screened | 19/48 | 55/160 | 74/208 |
| binary gap, screened | 0.22 [0.08, 0.42] (27) | 0.20 [0.14, 0.31] (95) | 0.19 [0.13, 0.28] (126) |
| matched null | 0.09 [0.00, 0.24] (22) | 0.11 [0.05, 0.19] (85) | 0.10 [0.06, 0.17] (108) |
| paired excess over null | 0.15 [0.00, 0.35] (20) | 0.10 [0.01, 0.20] (78) | 0.10 [0.02, 0.18] (100) |
| positive band | 0.63 [0.48, 0.77] (43) | 0.57 [0.48, 0.65] (137) | 0.58 [0.51, 0.65] (196) |
| rollout-level violating fraction | 0.41 (23) | 0.39 (79) | 0.38 (103) |
| floor: paraphrase agreement, option / binary | 0.69 / 0.84 | 0.70 / 0.81 | 0.70 / 0.80 |
| L1 excess (binary) | not run | 0.11 [0.01, 0.21] (73) | 0.11 [0.02, 0.19] (92) |
| L2 excess (binary) | not run | 0.06 [0.00, 0.24] (34) | 0.05 [−0.02, 0.19] (43) |
| L2 excess (continuous) | not run | not run | 0.079 [0.029, 0.128] (67) |
| raw-frame argmax gap, base vs instruct (shared) | 0.10 vs 0.08 (50) | 0.136 vs 0.089 (169) | 0.145 vs 0.094 (235) |

Table: Numbers of record by pod. Bracketed values are 95% bootstrap CIs over scenarios, as printed in v1 (two-way
intervals for the union column are in \Cref{tab:twoway}); parenthesized values
the number of scenarios with a defined statistic. The pilot's paraphrase-stable robustness cell
(0.11 at 18 scenarios) did not replicate (0.17 at 90). The raw-frame row is the argmax reading
without a null; the calibrated version is in Appendix D. {#tab:pods}

| family | primaries | screened | mixed | no pressure | judged violating | judgment unstable |
|---|---|---|---|---|---|---|
| F1 task completion | 56 | 25 | 25 | 16 | 4 | 11 |
| F3 instrumental | 56 | 17 | 17 | 8 | 6 | 25 |
| F4 loyalty/fairness | 40 | 11 | 11 | 7 | 1 | 21 |
| F5 third-party harm | 56 | 21 | 21 | 7 | 11 | 17 |
| F2 social cost (appendix) | 40 | 6 | 6 | 34 | 0 | 0 |

Table: Screen outcomes on the panel of record, gate-family primaries by family. "Judged
violating" scenarios are violated by the model and judged violating by it, so they carry no
contradiction; "judgment unstable" scenarios have no stable reference. {#tab:screen}

| origin | paraphraser | original gap (screened) | swapped gap (screened) | swapped − original, paired |
|---|---|---|---|---|
| Claude-written (20) | GPT | 0.00 [0.00, 0.50] (4) | 0.10 [0.00, 0.33] (10) | 0.00 [−0.25, 0.00] (5) |
| GPT-written (20) | Claude | 0.33 [0.00, 0.80] (6) | 0.11 [0.00, 0.33] (9) | −0.09 [−0.27, 0.00] (11) |

Table: The F4 paraphrase-swap cell on the binary readout. With four to eleven defined scenarios
per cell the swap is inconclusive; on the continuous readout the acting mass moves toward the
paraphraser by −0.03 [−0.10, 0.03] and −0.07 [−0.17, 0.02]. {#tab:swap}

Pod wall time on one A100-80GB: 58 minutes for the pilot's instruct cells (judgment cells run
one scenario at a time), 3 hours 20 minutes for the full panel with judgment prompts batched
across scenarios (about 16 minutes per judgment cell at 320 scenarios), 1 hour 10 minutes for
round 2. Every cell saved per-rollout text, parsed option, harness label, option order, and the
full next-token log-probability vector at the decision position; the three pods total 15 GB of
arrays with a verified manifest each. The raw-frame cells on both models take about 30 seconds
per 96 scenarios.
