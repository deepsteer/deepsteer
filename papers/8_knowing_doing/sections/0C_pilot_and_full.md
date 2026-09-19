# Appendix C. Pilot, full panel, and union: the numbers by pod {#app:pods}

| quantity | pilot (96) | full panel (320) | union (440) |
|---|---|---|---|
| gate-family primaries screened | 19/48 | 55/160 | 74/208 |
| binary KDG, screened | 0.22 [0.08, 0.42] (27) | 0.20 [0.14, 0.31] (95) | 0.19 [0.13, 0.28] (126) |
| matched null | 0.09 [0.00, 0.24] (22) | 0.11 [0.05, 0.19] (85) | 0.10 [0.06, 0.17] (108) |
| paired excess over null | 0.15 [0.00, 0.35] (20) | 0.10 [0.01, 0.20] (78) | 0.10 [0.02, 0.18] (100) |
| positive band | 0.63 [0.48, 0.77] (43) | 0.57 [0.48, 0.65] (137) | 0.58 [0.51, 0.65] (196) |
| rollout-level violating fraction | 0.41 (23) | 0.39 (79) | 0.38 (103) |
| floor: paraphrase agreement, option / binary | 0.69 / 0.84 | 0.70 / 0.81 | 0.70 / 0.81 |
| A13 L1 excess | not run | 0.11 [0.01, 0.21] (73) | 0.11 [0.02, 0.19] (92) |
| A13 L2 excess (binary) | not run | 0.06 [0.00, 0.24] (34) | 0.05 [−0.02, 0.19] (43) |
| A13 L2 excess (continuous) | not run | not run | 0.079 [0.029, 0.128] (67) |
| raw-frame gap, base vs instruct (shared) | 0.10 vs 0.08 (50) | 0.136 vs 0.089 (169) | 0.145 vs 0.094 (235) |

Table: Numbers of record by pod. Bracketed values are 95% bootstrap CIs; parenthesized values
the number of scenarios with a defined statistic. The pilot's paraphrase-stable robustness cell
(0.11 at 18 scenarios) did not replicate (0.17 at 90). {#tab:pods}

Pod wall time on one A100-80GB: 58 minutes for the pilot's instruct cells (judgment cells run
one scenario at a time), 3 hours 20 minutes for the full panel with judgment prompts batched
across scenarios (about 16 minutes per judgment cell at 320 scenarios), 1 hour 10 minutes for
round 2. Every cell saved per-rollout text, parsed option, harness label, option order, and the
full next-token log-probability vector at the decision position; the three pods total 15 GB of
arrays with a verified manifest each.
