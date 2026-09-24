# Appendix D. The three-cell contrast in detail {#app:three-cell}

**Pre-registered rules (amendment A17, committed before computation).** Readouts: the argmax
gap indicator (majority over eight option orders) and the continuous gap $g_{\mathrm{raw}} =
p_D - p_J$, with $p$ the violating option's mass normalized over the displayed letters, from the
saved per-permutation option log-probabilities; mass floor 0.5 on both frames. Matched null:
each model's pressure-removed raw twins. Statistics: scenario-level bootstrap, 2000 draws, seed
0, paired on the shared subset; difference CIs, never overlap reads. Quantities: $E$ per model
(paired primary minus twin), $\Delta_g$ (paired base minus instruct on the primaries), $\Delta_E$
(paired base minus instruct on $E$). Sub-branch rules on the continuous readout: *present* if a
model's $E$ excludes zero; *inherited, not installed* if present in base and $\Delta_E$ includes
zero; *inherited and narrowed* if present in base and $\Delta_E$ excludes zero with base larger;
*installed* if absent in base and present in instruct; *widened* if present in both and
$\Delta_E$ excludes zero with instruct larger; *template carries it* if neither raw $E$ excludes
zero while the chat excess does; otherwise under-powered, with the MDE stated. Selection check:
base's $E$ on all its above-floor scenarios against its $E$ on the shared subset, flagged if they
differ by more than the shared CI half-width.

| quantity | base | instruct |
|---|---|---|
| scenarios above the floor on the primary (of 397) | 359 | 242 |
| above the floor on primary and twin | 354 | 208 |
| $p_D$, $p_J$ under pressure (all above floor) | 0.354, 0.313 | 0.263, 0.261 |
| $g$ under pressure | 0.041 [0.034, 0.049] | 0.002 [−0.022, 0.024] |
| $g$ on the twin | 0.024 [0.017, 0.030] | −0.038 [−0.059, −0.015] |
| $E$, paired | 0.017 [0.012, 0.022] (354) | 0.039 [0.016, 0.061] (208) |
| binary gap rate | 0.154 [0.117, 0.194] (350) | 0.098 [0.064, 0.137] (234) |
| binary null rate | 0.127 [0.095, 0.166] (338) | 0.056 [0.026, 0.092] (195) |
| binary $E$, paired | 0.024 [−0.021, 0.065] (338) | 0.051 [0.010, 0.097] (195) |

Table: Each model on all of its above-floor scenarios. {#tab:three-cell-all}

| shared subset (192 with twins; 171 binary) | value |
|---|---|
| $p_D$ twin $\to$ primary, base | 0.306 $\to$ 0.354 |
| $p_J$ twin $\to$ primary, base | 0.280 $\to$ 0.311 |
| $p_D$ twin $\to$ primary, instruct | 0.192 $\to$ 0.277 |
| $p_J$ twin $\to$ primary, instruct | 0.224 $\to$ 0.263 |
| $\Delta_E$, continuous | −0.028 [−0.049, −0.007]; MDE 0.030 |
| acting side, base minus instruct | −0.037 [−0.062, −0.012] |
| judging side, base minus instruct | −0.009 [−0.023, 0.004] |
| $\Delta_E$, binary | −0.012 [−0.076, 0.053] |
| selection check: base $E$ all vs shared | 0.017 vs 0.018; half-width 0.007; not flagged |

Table: The paired contrasts on the shared subset. {#tab:three-cell-shared}

| slice | $n$ | $E_{\mathrm{base}}$ | $E_{\mathrm{instruct}}$ | $\Delta_E$ |
|---|---|---|---|---|
| pilot-written scenarios (prompt 1.0.0) | 45 | 0.013 [0.002, 0.025] | 0.016 [−0.015, 0.049] | −0.003 [−0.032, 0.027] |
| later-written scenarios (prompt 1.1.0) | 147 | 0.020 [0.011, 0.028] | 0.055 [0.029, 0.083] | −0.036 [−0.063, −0.011] |
| primaries | 105 | | | −0.030 [−0.060, 0.000] |
| harm twins | 87 | | | −0.026 [−0.056, 0.002] |
| F1 | 62 | 0.012 [0.002, 0.022] | 0.035 [0.002, 0.070] | −0.023 [−0.057, 0.009] |
| F3 | 60 | 0.037 [0.022, 0.052] | 0.066 [0.029, 0.106] | −0.029 [−0.068, 0.006] |
| F4 | 33 | 0.015 [0.002, 0.030] | 0.073 [0.030, 0.122] | −0.058 [−0.103, −0.015] |
| F5 | 28 | 0.009 [−0.005, 0.024] | 0.014 [−0.069, 0.089] | −0.005 [−0.080, 0.079] |

Table: Slices of the shared subset. The pilot's 96 scenarios were re-run in the full-panel pod
with bit-identical raw-frame values (maximum absolute difference 0.0 over 192 model-scenario
pairs), so the pilot block is a scenario-subset check, not an independent replication. The
pilot-written and later-written subsets are not separated from each other at these counts; the
pooled number is the number of record and the prompt version is recorded as a covariate.
{#tab:three-cell-slices}

**Exploratory items under the same amendment.** Six pairwise family contrasts on the chat
continuous gap $g$ over screened scenarios: F1 − F3 0.09 [0.02, 0.16]; F1 − F4 0.03
[−0.06, 0.13]; F1 − F5 0.05 [−0.07, 0.18]; F3 − F4 −0.06 [−0.16, 0.05]; F3 − F5
−0.04 [−0.17, 0.08]; F4 − F5 0.02 [−0.14, 0.16]. Per-family paired excess over the
pressure-removed null: F1 0.076 [0.020, 0.129] (43); F3 0.020 [−0.038, 0.079] (40); F4 0.121
[0.054, 0.195] (26); F5 −0.008 [−0.090, 0.073] (21). The binary excess predicted from the
per-scenario acting mass (net crossings of 0.5 from twin to primary on paired screened scenarios
with a non-violating reference) is 0.136 [0.068, 0.216] on 88 scenarios against the observed
0.100 [0.030, 0.170] on the same 100 scenarios; 20 scenarios cross upward and one downward.
