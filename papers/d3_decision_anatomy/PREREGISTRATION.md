# Direction 3 (Decision Anatomy / C1) — Pre-registration

**Date:** 2026-07-02 · **Against commit:** `2ddc1c8` (HEAD) · GPU-free at write time.
**Status:** Pre-registered before any head-attribution or patch quantity is computed. Every
threshold below is fixed now. NULL / distributed / surface-only outcomes are pre-declared
publishable. Companion: `PAPER_PLAN.md`; disciplines: `construct-audit`, `instrument-calibration`,
`compute-ordering`. Carries the program margin **M = 0.05** and the two-step null protocol.

## 0. What is fixed now vs computed later (two-step)

| Fixed NOW | Computed LATER (from the built artifacts, before the causal verdict) |
|---|---|
| Reconstruction floor, MLP-branch threshold, k-selection rule (§1) | the realized head-write curve, MLP fraction, k |
| "reads X" rule = above matched null q95 + M (§2) | the realized value-side nulls/bands per head |
| Causal-cell decision rules + MDE requirement (§3) | the realized MDEs and patch deltas |
| Comparative OLMo-vs-Llama direction (§4) | the realized per-model loadings |
| Type-block schema + position-validity carryover (§5) | the realized PRs per position |

The nulls, bands, and MDEs are realized from the constructed heads/subspaces **before** any patch is
run, under a frozen recipe — so cutoffs predate the causal result (two-step discipline).

## 1. Stage 1 — per-head write attribution (rules fixed now)

- **Decomposition.** At `L_ref`, the decision-site-token residual = Σ_h (W_O^h z^h) + Σ MLP +
  attn/embed remainder; `write(unit) = ⟨unit_out, r̂⟩`. **Reconstruction control (instrument-
  calibration positive control):** Σ writes must recover `⟨resid, r̂⟩` to **≥ 0.90**; report the
  residual. **< 0.90 → escalate LayerNorm from folded-approx to exact per-token** before attributing.
- **LayerNorm typing (construct-audit).** Folded-LN approximation, stated as such; the per-token LN
  scale treated constant at the decision token; error = 1 − reconstruction, reported.
- **k-selection (fixed now).** k = smallest head count whose cumulative |write| ≥ **0.80** of total
  head write, **capped at k ≤ 10**; the full sparsity curve is reported regardless.
- **MLP branch (fixed now).** If MLP write fraction **> 0.50** → the head story is incomplete; the
  **Jacobian stage** is added (cross-layer attribution of `r̂` w.r.t. the moral-usage layer) and the
  headline becomes "no head-level shortcut; the read is distributed." Pre-declared, not a failure.

## 2. Stage 2 — what the top-k heads read (rules fixed now)

For each top-k head: attention source mass over `{t_inst, content, template/sink}` (typed by
`position_class`), and value-side projection of its value inputs onto (i) `mean_content` `V_moral`
(D2's position-valid, format-robust content subspace) and (ii) the `t_inst` harm direction.

- **"Head h reads X"** iff its value-side projection onto X **> matched-null q95 + M**, with X's
  band reported (held-one-out band for `V_moral`; known-harm reference for harm). Ladder-worded; MDE
  attached (instrument-calibration).
- **Pre-registered hypotheses (both publishable):**
  - **copy-head-for-harm:** ∃ top head with `t_inst` attention mass the plurality **and** harm
    loading > `V_moral` loading by **M** → the writer transports a *harm cue*, not moral content.
  - **moral-content-reading:** ∃ top head whose `V_moral` loading clears its band-min − M → the
    writer transports *moral content*.
  The data selects; both, either, or neither may hold (reported per head).

## 3. Causal cells — decision rules (a "reads-from" verdict requires ≥ 1; construct-audit)

Patches at content positions; decision read = Δ(refusal projection at the bottleneck token) +
Δ(behavioral refusal, reconciled `_classify_response`). **Every cell reports its MDE** (min
detectable Δ at achieved n, 95%, bootstrap); a null cell is only publishable with its MDE. Controls:
matched-random subspace/head + named-reference (persona / a non-writing head).

- **(a) Twin patch (moral-status twins, v2 1,200 surface-matched).** `content-moved` iff
  Δ(refusal) under moral↔neutral swap **> MDE**. Null (≤ MDE) = moral content at content positions
  does not change the decision on this model → surface-only branch.
- **(b) Subspace-restricted vs full (THE decisive cell).** Run the twin patch (i) restricted to the
  `mean_content` `V_moral` subspace and (ii) full residual.
  **Verdict rule:** full Δ > MDE **and** `V_moral`-restricted Δ ≤ MDE **and** a matched-random rank-3
  restricted patch also ≤ MDE → **"refusal reads NON-`V_moral` features of moral content"** (the
  program-null-explaining result). If `V_moral`-restricted Δ > MDE → the moral subspace *is* the read
  substrate (reopens the geometric story). Both pre-declared.
- **(c) XSTest harm-flip vs surface-held.** Patch pairs where alarming surface is held and harm
  flips (and the converse). `reads-harm` iff Δ(refusal) tracks the harm flip > MDE while the
  surface-held control is ≤ MDE; separates harm-cue from moral-status reading (confound genealogy).
- **(d) Top-head ablation / resample-patch.** Behavioral refusal Δ under top-k-head ablation is an
  **outlier below the matched-random-head null** (not raw) → head-specific write. Coherence filter
  on; over-ablation reported, never headlined.

## 4. Comparative prediction (fixed now; instrument-first order)

**OLMo-3 first (clean instrument), Llama-3.1 second.** Pre-registered directional prediction:
**Llama's top writing heads read moral content more than OLMo's** — `value-side V_moral loading
(Llama top-k) > (OLMo top-k) + M`. Confirmed = the n=1 Llama behavioral entanglement localized to
head-level content-reading; disconfirmed = a datapoint relocating where the entanglement lives (not
a failure). Qwen third, budget permitting.

## 5. Type-block schema + position-validity (construct-audit; required)

Every saved direction / head-contribution / patch carries: `contrast_semantics`, `source_dataset` +
commit, `position_class`, `format`, `layer(s)`, `model` + revision, `n_pairs`, `known_covariates`,
`participation_ratio`, `extraction_commit`. **No comparison/figure uses an untyped object.** Every
position touched records its PR; a read/patch at **PR < 30** (or band < null) is **flagged
position-invalid** and cannot carry a verdict (D2 Amendment 2 rule, carried).

## 6. Compute ordering + artifact hygiene (compute-ordering; required)

- **Zero-GPU first, gating novelty:** the lit pass (§PAPER_PLAN 7) blocks all novelty sentences;
  the D1 pooled-PR check runs zero-GPU (the per-window PR audit is a re-extraction rider, logged —
  the reflexive correction that it was NOT zero-GPU as first assumed).
- **One session per model, riders batched** onto the Stage-1/2 keystone (B5 `mean_content`;
  reconciled cross-ablation; template-token PR control).
- **Per-unit saves enforced:** per-head write contributions (all heads, band layers), per-head
  attention maps + value inputs, per-twin / per-XSTest patch deltas, the reconstruction residual —
  so every downstream MDE / bootstrap / re-analysis is zero-GPU. Missing artifacts → ledger, never
  silent inline regen.
- **Pilot gate:** Stage-1 reconstruction ≥ 0.90 before any attribution is trusted; the sparsity knee
  sets k before Stage 2; MLP > 0.50 diverts to the Jacobian branch — do not force a head story.

## 7. Pre-registered branch dispositions (all publishable; framings frozen)

| branch | disposition |
|---|---|
| sparse write + writers read content (b/c positive) | anatomical coupling target → Direction-2 intervention address |
| sparse write, content moves refusal via **non-`V_moral`** features (b) | explains every geometric null in the program; the methods headline |
| MLP > 50% / distributed | Jacobian follow-on + "no head-level shortcut" |
| content patch moves nothing on stock instruct (a/b/c null) | refusal reads surface cues only; the forced-coupling **sign-flip** gets a dedicated re-examination |

---

*Frozen at `2ddc1c8`. Any later change to a threshold, rule, or branch is a dated amendment below
this line, never a silent edit.*

### Amendments

*(none yet)*
