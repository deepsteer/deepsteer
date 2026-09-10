# Appendix F. Removability battery {#app:removability}

«SKELETON (W4-1, 2026-09-10): section heads + CLAIMS-anchored sentences only; prose in W4-3. Every
number below carries its CLAIMS id; «CHECK:» marks a number not yet traced.»

Paper-B disposition of record (WRITEUP_PHASE_PLAN Phase W4 item 4): the behavioral / interventional
companion folds into this and the next two appendices. This appendix carries the single-direction
refusal-ablation battery across the three instruct families, the behavioral side of the
representational dissociation in \Cref{fresh-gate}, and the mechanism-level reading of
\Cref{app:llama-anatomy}.

## F.1 The battery and its readouts {#app:removability-battery}

Single-direction refusal ablation (Arditi-style orthogonalization of the attention out-projection
and the multilayer-perceptron down-projection at the swept layer), read against four outcomes:
refusal rate on harmful requests, fresh per-foundation probe accuracy, moral-subspace effective
dimensionality, and behavioral moral judgment on the 48-scenario battery, plus persona-shift
compliance (\Cref{app:persona}). Ablation layers are chosen by a depth-fraction sweep per model.

- [PB-01] Single-direction refusal ablation at each model's swept best layer (OLMo layer 19, depth
  0.59; Qwen layer 14, depth 0.50; Llama layer 13, depth 0.41) leaves probe accuracy 1.0 → 1.0 and
  effective dimensionality 5 → 5 on all three models.
- [P6-02] Refusal removability is family-dependent: OLMo 0.575 → 0.000, Qwen 1.000 → 0.000, Llama only
  0.900 → 0.475.
- [P5-04] On OLMo-3 the refusal direction projects only 0.10 of its norm into the moral subspace (mean
  $|\cos|$ 0.06); ablating it drops refusal 0.25 → 0.00 while comprehension (base-to-fresh cosine
  0.749, probe accuracy 1.0, effective dimension 5) and moral judgment (0.73 vs 0.75) are untouched.
  «CHECK: the OLMo refusal rate here is 0.25 (Paper 5, borderline + harmful set) against 0.575 in
  PB-01/P6-02 (Paper 6, harmful set); state both sets by construction.»

## F.2 Behavioral judgment under ablation {#app:removability-judgment}

- [PB-03] Behavioral moral judgment under ablation: OLMo 0.75 → 0.79, Qwen 0.875 → 0.812, Llama
  0.75 → 0.604.
- [P6-03] Llama's drop is a $-21\sigma$ outlier against the magnitude-matched random null
  (0.747 $\pm$ 0.007) and dose-dependent (Spearman 1.0). «CHECK: FL main text dropped the σ count
  per the self-review ("far outside the random-ablation band"); use the same wording here.»
- [PB-04] Dose-response: the judgment drop is 0.083 (95% bootstrap CI [0.02, 0.17], $n=48$) at
  $\alpha=0.5$ while the refusal rate is still 0.90, and 0.146 (CI [0.06, 0.25]) at $\alpha=1$; the
  matched-random null stays at 0.74 even at twice the full-ablation magnitude; ablating the persona
  direction leaves judgment at 0.75.
- [P6-04] The anomaly is resolved upstream by the depth-matched Llama battery (\Cref{app:llama-depth}):
  a broad-moral reader that commits early.

## F.3 What removability does and does not show {#app:removability-scope}

- [D3-09, D3-08] On OLMo the ablated direction reads a rank-1 harm slice; 73% of refusal's causal
  input lies outside the rank-16 moral basis, which is why a rank-one edit removes it without touching
  comprehension.
- Scope: the "low-rank read → removable" link is $n=1$ on the causal side (OLMo) and correlational on
  the panel (SELF_REVIEW should-fix); this appendix reports the battery, it does not claim the link is
  general. «CHECK: cross-reference the Limitations paragraph that carries this scope.»
