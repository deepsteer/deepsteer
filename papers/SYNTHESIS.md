# Synthesis — how the refusal decision relates to the moral subspace

Program-level thesis across Directions 1–3 (OLMo-3-7B primary). Updated 2026-07-02 after the D3
powered hardening run. Numbers of record live in each direction's RESULTS; this file states the
throughline and is re-dated on each substantive change.

## Thesis (routing form)

The refusal decision **routes around** the moral subspace. It reads a **specific but minority** moral
substrate, dominated by the harm percept, and it writes into a **narrow control-token channel** at the
decision site that moral content largely does not reach. The geometric orthogonality measured upstream
(D1, D2) is therefore not a weak-instrument artifact at the readout; it is a property of *where the
refusal computation writes* (a low-rank channel) and *how little of the moral subspace it reads*.
One question stays open: whether the moral content refusal does not read is genuinely non-`V_moral`,
or whether rank-3 `V_moral` is too small a window on the same content. The rank sweep (Amendment 4)
resolves it.

## The three legs

- **D1 (geometry of the direction).** Refusal projects **below the moral-family band at every rung**
  (held-one-out `p(d_src | others)` bands; refusal in-trace peak included). Even the program's highest
  refusal projection is less moral-adjacent than a held-out moral direction. Refusal does **not
  crystallize** from a pretraining precursor (cos 0.155 base→instruct), unlike the moral subspace
  (cos 0.999); it is a fresh post-training construction in a low-variance channel.

- **D2 (geometry of the decision).** The decision site (`final_pre_assistant`) is a **~10–15-dim
  control-token bottleneck** (participation ratio 14.7 / 8.6 / 10.2 on OLMo / Qwen / Llama; band below
  the covariance null → position-invalid for content). Refusal-decision ⊥ judgment-decision at |cos|
  below even the low-dim random level. Content and decision do not coexist at one valid position, so
  content-vs-decision orthogonality is **architecturally guaranteed**, and any coupling must ride the
  **heads that write the bottleneck**. That is the D3 target.

- **D3 (anatomy of the decision, causal).** Refusal is a **distributed write** into that ~13-dim
  channel: led by L16 H23 but needing ~62 heads for 80% of the specificity, plus a 38% MLP share;
  every top writer reads content only weakly `V_moral`-aligned. Interchange patching then shows
  `V_moral` **is a specific but minority refusal substrate**: the V_moral-restricted patch moves
  refusal significantly more than a random rank-3 restriction (Δ = |restricted| − |random| = 0.031,
  95% CI **[0.020, 0.043]**, paired, excludes 0), carrying ~34% of the full-content effect, with the
  **complement carrying ~76%** (near-additive). The **harm rank-1 direction** accounts for most of
  what V_moral rank-3 captures (harm-restricted −0.026 ≈ V_moral-restricted −0.028), and the harm
  direction sits partly inside V_moral (`frac(V_moral, d_harm) = 0.46`). Whether refusal reads V_moral
  **less** than judgment does — the claim that would explain the D1/D2 nulls outright — is
  **`under_transfer`** (ratio-of-ratios `R_judgment − R_refusal = 0.18`, CI includes 0). Open pending
  the rank sweep + the harm-partialed identification cell.

## What is settled vs open

- **Settled.** The write anatomy (distributed, channel-shaped, harm-adjacent read). `V_moral` is a
  *specific* refusal substrate, not a random-subspace artifact (Δ CI excludes 0). The refusal-relevant
  moral content is largely the **harm percept**, not the broad moral subspace.
- **Open (rank sweep, Amendment 4).** Is the residual routing genuinely non-`V_moral`
  (harm-saturating), a rank-3 truncation (broad-moral), or a linear-transport ceiling
  (instrument-ceiling)? The three shape verdicts are frozen and all publishable. No cross-model panel
  runs until this resolves.

## Method spine (portable, promoted to ANOMALIES)

A1 (covariance nulls degenerate in massive-activation families → standardize), A2 (band-below-null ⇒
position-invalid instrument), A3 (reordered-norm architectures overshoot per-head OV attribution ~3×
→ fold the norm). Plus the estimator discipline this program keeps re-learning: **an absolute
"one-clears-MDE-one-doesn't" comparison is the overlap fallacy** — normalize to a within-outcome ratio
and gate on a bootstrap CI, which is what reclassified the D3 headline from a clean claim to the honest
`under_transfer`.
