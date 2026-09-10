# Appendix G. Distributed refusal {#app:distributed}

«SKELETON (W4-1, 2026-09-10): section heads + CLAIMS-anchored sentences only; prose in W4-3.»

Two independent observations that refusal is a distributed write, not a single-direction switch:
the GPT-OSS held-out ablation battery (Paper 7 §4.3) and the OLMo-3 per-head write attribution into
the decision channel (D3 Stage 1, the anatomy behind \Cref{reads-harm} and \Cref{app:write}).

## G.1 GPT-OSS: no single direction ablates refusal {#app:distributed-gptoss}

- [P7-03, PB-06] On GPT-OSS-20B no single direction ablates refusal on a held-out, category-diverse
  set: the end-of-prompt refusal direction (estimated on a category-spanning training draw)
  coherently flips 4% of held-out refusals, the CoT-last direction flips none, and the CoT-mean
  direction removes refusal in 88% of cases only by driving generation into incoherence, which the
  coherence filter excludes.
- Consequence for the causal program: a representation that is not a bottleneck for refusal cannot be
  the moral subspace either; the single-subspace load-bearing test is not cleanly runnable on this
  model, which is why the GPT-OSS read axis is correlational (\Cref{gpt-oss}, D1-18/19) and the
  causal C1-MoE stays held.

## G.2 OLMo-3: a distributed write into a narrow channel {#app:distributed-olmo}

- [D3-01] Refusal is written into the ~13-dimensional decision-site channel by a distributed set of
  heads led by L16 H23; cumulative channel-matched specificity is 44% at the top 10 heads and ~62
  heads are needed for 80% ($k$ hit its cap of 10). «CHECK / NI-9: FL §7 was reconciled to the saved
  curve as 11.7% (top head) / 45% (top ten) / ~67 heads for 80% (OPEN_THREADS H4 follow-up (a));
  CLAIMS D3-01/02 still carry 11.6% / 44% / ~62. Pin one set of record before prose.»
- [D3-02, PB-07] Top writers by channel-matched specificity: L16 H23 write +0.742 (specificity
  +0.756, 11.6% of total «CHECK NI-9»); L15 H2 +0.302 / +0.368; L14 H19 +0.334 / +0.347; L15 H0
  +0.265 / +0.285; L11 H20 +0.246 / +0.274; L16 H21 +0.172 / +0.197; L14 H22 +0.178 / +0.193;
  L15 H6 +0.175 / +0.189; L13 H29 +0.139 / +0.144; L15 H15 −0.130 / −0.142 (the sole anti-refusal
  writer). Writers span layers 11–16.
- [D3-03] Multilayer perceptrons contribute 38% of the decision-site write (fraction 0.384, below the
  0.50 Jacobian threshold, above the 0.23 the un-folded run reported).
- [D3-04] Folding the per-layer RMSNorm gain brings Stage-1 reconstruction from 3.05 to 0.9999
  (two-sided band [0.90, 1.10]); the fold is exact to $10^{-9}$ (methods note, A3).
- [D3-05] All ten top writers are labeled neither-moral-nor-harm: none clears the moral-family band,
  none is a clean copy-head-for-harm; V_moral fraction 0.15–0.28 with comparable harm loading.
- [D3-14] Llama-3.1's anatomy is OLMo-like (pre-norm reconstruction 1.0008, distributed write,
  multilayer-perceptron share 0.30, all writers neither) (\Cref{app:llama-anatomy}).

## G.3 Reading the two together {#app:distributed-reading}

- Both models: many small writes into a low-rank control channel (a refusal cone, not a single
  direction), consistent with the ~9–15-dimensional bottleneck of \Cref{bottleneck} (D2-02, D3-20).
- The distributed write is what the Direction-2 target would have to move (\Cref{discussion}); the
  per-head arrays are in the supplement (`head_attribution.csv`) «CHECK: confirm the CSV carries the
  reconciled curve (NI-9)».
