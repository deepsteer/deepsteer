# Appendix B. Harness calibration {#app:calibration}

Stage 1 (before any pod): 200 synthetic reply formats instantiated on the panel's own
scenarios with per-item option permutations (clean letters, answer lines, verbose explanations,
tool calls, hedges, refusals, two-option mentions, trailing reconsiderations). Against the
construction labels the harness scored 200/200 after three parser rules were settled from an
independent judge pass: the last `Answer:` anchor anywhere in a reply is the commitment; a
leading letter counts only with punctuation after it; a two-option mention with a contrast
marker ("rather than Y", "though Y is tempting") commits to the option other than Y. Against the
independent judge: 0.985 (kappa 0.98).

Stage 2 (before each gate): 200 real replies per pod, drawn across the six generated cells and
oversampling every reply the parser had not resolved by a clean rule, labeled by two judges from
different providers (a Claude model and a GPT model, through API or subscription tooling).

| pod | harness vs judge 1 | harness vs judge 2 | judge vs judge | hard-parse cases in the pool |
|---|---|---|---|---|
| pilot (96 scenarios) | 0.99 (Claude) | 1.00 (GPT) | 0.99 | 0 |
| full panel (320) | 0.98 (Claude) | 1.00 (GPT) | 0.99 | 60 |
| round 2 (120) | 0.99 (Claude) | 1.00 (GPT) | 0.99 | 0 |

Table: Stage-2 calibration on real replies. Every disagreement had the harness on the
conservative side (a content-only reply both judges assigned an option, or a judge returning no
label on a bare letter). {#tab:stage2}

Parse rates on the pods of record: 0.9997 on 10,240 action rollouts (all but three bare
letters or leading letters) and 1.000 on 2,880 judgment replies (all answer lines).

**Construct check and its ceiling (v2).** For GPT-OSS-20B we added a construct check (C0): the dose-0
forced letter should agree with the model's own sampled answer at the same dose. A fixed bar of 0.80 turned
out to sit above what a perfect readout can reach on most of the panel. With each scenario's own letter
distribution and the check's sampling (temperature 0.7, four rollouts, strict majority), the agreement a
perfectly valid readout would produce, the exact-readout ceiling $\kappa^*$, ranges from 0.62 to 0.82 across
the five models on the 64-scenario check sample under each model's effective sampler (temperature 0.7 plus
its default nucleus): OLMo-3 0.62, GPT-OSS-20B 0.74, Llama-3.1 0.78, Tulu 3 0.80, Qwen2.5 0.82. The 0.80 bar
sat at or above the ceiling on four of the five models. GPT-OSS's registered checks (0.688 and, dose-matched,
0.766) fail 0.80 and clear the model-relative bar $0.9\,\kappa^* = 0.668$; the dose-0 rollouts also write
the forced prefill's own header token for token. Construct checks are now set at $0.9\,\kappa^*$ per model. That bar
was set after GPT-OSS's registered checks had failed the fixed one, so it is a post-hoc calibration
correction, and both outcomes stand in the record: fail at the registered 0.80, pass at $0.9\,\kappa^*$
(\Cref{app:prereg}).
The four original panel models had no construct check: their readout validity rests on the forward pass
reproducing generation and on the known-gap control.


**Two-way intervals and the scenario-only intervals they replace (v2).** The sampled cells drew rollout
$k$ of every scenario from one random stream (\Cref{instruments}), so a bootstrap over scenarios alone
treats as independent what shares a stream. The two-way bootstrap resamples scenarios and rollout indices
independently, one index set shared by every cell in a draw (10,000 draws, seed 0), and is the interval of
record in v2. Every scenario-only interval below reproduces the one printed in v1. Point estimates do not
change. The audit covers the judgment-based statistics of \Cref{gap} and \Cref{reference}, every row of
\Cref{tab:deliberation} with the norm-salience share and the log-ratio intervals, and the panel-side arm
of the panel-versus-rule-harms brake difference; the forced letter readouts (\Cref{base}, \Cref{recipe})
sample nothing and are unaffected. A verdict moves in three places, marked.

| statistic (OLMo-3 unless named) | point | scenario-only (v1) | two-way (of record) |
|------------------------|------|----------|----------|
| \Cref{gap}: measurement, rate on the screened subset | 0.19 | [0.13, 0.28] | [0.11, 0.28] |
| matched null on the twins | 0.10 | [0.06, 0.17] | [0.04, 0.18] |
| positive band | 0.58 | [0.51, 0.65] | [0.49, 0.65] |
| paired excess, L0 (**moves**: lower bound at zero) | 0.10 | [0.02, 0.18] | [0.00, 0.19] |
| band minus measurement | 0.38 | [0.24, 0.48] | [0.22, 0.50] |
| rollout-level second derivation | 0.38 | [0.34, 0.42] | [0.33, 0.42] |
| \Cref{reference}: paired excess, L1 (**moves**: lower bound at zero) | 0.11 | [0.02, 0.19] | [0.00, 0.21] |
| paired excess, L2 | 0.05 | [−0.02, 0.19] | [−0.03, 0.23] |
| \Cref{deliberation}: reasoning − filler | −0.077 | [−0.110, −0.047] | [−0.117, −0.037] |
| reasoning − truncated filler | −0.112 | [−0.145, −0.078] | [−0.154, −0.070] |
| truncated filler − filler | +0.034 | [0.017, 0.052] | [0.009, 0.059] |
| brief reasoning − filler | +0.024 | [−0.011, 0.058] | [−0.020, 0.069] |
| norm named − filler | −0.025 | [−0.031, −0.019] | [−0.034, −0.016] |
| norm-salience share | 0.32 | [0.22, 0.53] | [0.18, 0.64] |
| twins: reasoning − filler | −0.052 | [−0.080, −0.025] | [−0.089, −0.017] |
| $\Delta E$ against the filler | −0.025 | [−0.052, 0.002] | [−0.065, 0.014] |
| $\Delta E$ against the truncated filler (**moves**: branch (a) to (b)) | −0.039 | [−0.071, −0.009] | [−0.082, 0.004] |
| log-ratio difference, filler (bar 0.21 to 0.31) | −0.011 | [−0.158, 0.135] | [−0.231, 0.203] |
| log-ratio difference, truncated filler (bar 0.23 to 0.32) | −0.026 | [−0.188, 0.130] | [−0.250, 0.194] |
| panel minus rule-harms brake (bar 0.056 to 0.065) | −0.002 | [−0.042, 0.037] | [−0.047, 0.045] |
| Llama-3.1: reasoning − filler | −0.350 | [−0.387, −0.314] | [−0.395, −0.307] |
| reasoning − truncated filler | −0.320 | [−0.356, −0.284] | [−0.367, −0.273] |
| truncated filler − filler | −0.030 | [−0.063, 0.002] | [−0.074, 0.016] |
| brief reasoning − filler | −0.205 | [−0.244, −0.168] | [−0.258, −0.154] |
| twins: reasoning − filler | −0.208 | [−0.244, −0.173] | [−0.251, −0.168] |
| $\Delta E$ against the filler | −0.143 | [−0.185, −0.100] | [−0.195, −0.090] |
| $\Delta E$ against the truncated filler | −0.079 | [−0.116, −0.042] | [−0.131, −0.026] |
| log-ratio difference, filler (bar 0.30 to 0.41) | −0.059 | [−0.257, 0.158] | [−0.338, 0.236] |
| log-ratio difference, truncated filler (bar 0.33 to 0.43) | +0.108 | [−0.106, 0.350] | [−0.175, 0.422] |
| panel minus rule-harms brake (bar 0.058 to 0.068) | −0.301 | [−0.340, −0.260] | [−0.349, −0.254] |

Table: Scenario-only (v1) and two-way (v2, of record) 95\% intervals for every statistic the audit covers.
Bars are 2.8 standard errors under each bootstrap. The rule-harms arm of the brake difference keeps its
scenario bootstrap: that set was sampled with a seed per scenario. {#tab:twoway}

**Clustered sensitivity for overlapping sets (v2).** Two contrasts compare sets that share scenarios: the
own-screen excess against the twin-screened selection null (\Cref{tab:recipe}), where a scenario can pass
both screens, and band minus measurement (\Cref{gap}), whose two arms draw on the same scenario clusters. The
registered intervals resampled the two sets independently. Resampling scenario clusters once from the union
of both sets, so a shared scenario moves with both arms (10,000 draws, seed 0), changes no verdict.

| contrast | set sizes (shared) | point | registered | clustered |
|--------------------|----------|-----|---------|---------|
| OLMo-3, own screen minus selection null | 110 / 98 (47) | 0.073 | [0.033, 0.114] | [0.027, 0.121] |
| Llama-3.1 | 118 / 99 (54) | 0.143 | [0.112, 0.173] | [0.108, 0.179] |
| Tulu 3 | 52 / 43 (19) | −0.004 | [−0.054, 0.042] | [−0.058, 0.049] |
| Qwen2.5 | 47 / 54 (16) | 0.083 | [−0.028, 0.195] | [−0.040, 0.203] |
| GPT-OSS-20B, dose 0 | 71 / 67 (19) | −0.001 | [−0.050, 0.049] | [−0.054, 0.052] |
| OLMo-3, band minus measurement (binary) | band 196 / measurement 73 (72) | 0.38 | [0.24, 0.48] | [0.27, 0.49] |

Table: Clustered sensitivity. Registered: the two sets resampled independently. Clustered: scenario
clusters (a primary and its twin) resampled once from the union. {#tab:cluster}
