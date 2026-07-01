# Direction 1 — Gate-critical hook verification

**Date:** 2026-06-26 · **Against commit:** `107f9f3` (HEAD) · **GPU-free.**

Scope: verify that the three repository hooks the pre-registered gates depend on
exist and behave the way the Direction 1 plan assumes, *before* pre-registering
against them. This is the "slice of Phase-4-scope-check folded into the
pre-registration" — not the full Section-12 inventory, only the gate-critical
pieces. If any hook were shaped differently than assumed, it would change how the
gate is constructed; this log records that they do (with two corrections).

---

## Hook 1 — `heretic_ablation.last_token_means` (G3 Point A) — ✅ MATCHES

- **File:** `papers/5_moral_alignment/scripts/heretic_ablation.py:74-95`
- **Signature:** `last_token_means(model, prompts, input_format, layers) -> dict[int, np.ndarray]`
- **Returns:** `{layer: mean_vec}`, each `mean_vec` a float32 `(hidden,)` array — the
  mean over prompts of the last-input-token residual activation at that layer.
- **Refusal-direction construction** (`main`, lines 170-176): per layer,
  `r = h_means[L] - s_means[L]`, then `refusal = r / ||r||`. This is a **single
  unit vector per layer**, which is exactly the object G3 projects onto `V_moral`.
- **Bonus (also verified):** `heretic_ablation.subspace_projection_fraction(r, basis)`
  (lines 98-103) already computes `||proj_span(basis)(r)|| / ||r||` via least squares.
  This is the *same function that produced the committed 0.1044 MFT baseline*, so
  reusing it with `V_moral`'s basis makes Point A apples-to-apples with Paper 5.

**Construction notes carried into the pre-registration (none change G3's shape):**
1. The committed baseline (`outputs/heretic/refusal_morality_geometry.json`) is at
   **layer 16**, not the script default `int(0.6 * n_layers) = 19` for 32 layers — it
   was run with an explicit `--refusal-layer 16`. G3 Point A is therefore pinned to
   **layer 16** for direct comparability, with the stable band (layers 15–31) reported
   for robustness.
2. `subspace_projection_fraction` uses `lstsq` on the raw `basis`, so it tolerates a
   non-orthonormal basis, but `V_moral`'s basis will be SVD left singular vectors
   (orthonormal by construction), making the fraction exactly the in-subspace norm.
3. `input_format`: base checkpoints have no chat template → `raw`; the instruct-time
   refusal gate (Point B) → `chat`. Matches the `direction_utils` pooling discipline.

**Verdict:** behaves as the plan assumes. No change to G3 construction required.

---

## Hook 2 — `direction_utils` (every projection) — ✅ MATCHES, with one convention fork

- **File:** `deepsteer/directions/extraction.py` (moved to the core library 2026-07-01; imported as `from deepsteer.directions import extraction as du`)
- **Verified present and numpy/torch-only (deterministic, `PROBE_SEED = 42`):**
  - `project_scores(X, direction)` (258) — project rows onto a direction.
  - `transfer_metrics(X, y, direction)` (287) → `acc_midpoint`, `auc`, `auc_abs`,
    `threshold`. **`acc_midpoint` is the G2 accuracy metric** (fixed-direction transfer
    accuracy, one mild centering param).
  - `mean_diff_direction(X, y)` (164), `probe_weight_direction(...)` (172).
  - `cosine` (338), `cosine_matrix` (321), `effective_dimensionality` (327),
    `save_directions` / `load_directions` (350/362).

**Convention fork that changes a gate denominator — captured in the pre-registration:**
`effective_dimensionality` **centers** the matrix before SVD
(`mat - mat.mean(axis=0)`, line 332). Paper 6 used an **uncentered** eff-rank
(per project memory). For `V_moral`'s rank — which is the *denominator the G3 null
depends on* — the uncentered convention is correct: the shared moral-valence axis is
signal, not nuisance, and centering would remove it and **understate** the rank. The
pre-registration (§5) fixes uncentered eff-rank at variance-threshold 0.9 and notes
that `effective_dimensionality` must **not** be called as-is for `V_moral`'s rank.

**Verdict:** projection primitives behave as assumed; the one centered-vs-uncentered
fork is overridden explicitly in the pre-registration.

---

## Hook 3 — Dataset audit harness (G2's upstream) — ⚠️ PARTIAL (real correction to plan §12)

The plan (§4.2, §12) lists a "`DATASET_GUIDELINES.md` audit harness (Phase 1)" to be
re-run at v2 thresholds. What actually exists:

- **Mechanical gates — runnable, reusable:** `deepsteer/datasets/validation.py`
  `validate_pairs(...)` (line 81) applies four gates in order — length ratio, content-word
  overlap, moral-keyword scan on the neutral, dedup. These exist as code and can be
  re-run on new pairs directly.
- **LLM-scored construction gates (§1.1 inanimate subjects, §1.2 structural
  parallelism, §1.5 accidentally-moral neutrals) — DEFINED but runner NOT committed.**
  `deepsteer/datasets/DATASET_AUDIT.md` reports these were scored with Claude Sonnet 4.6
  at a score-3 failure threshold, but states "Full per-pair results were in `/tmp/`"
  (line 7). The grep for an audit runner (`def *audit*`, `*audit*.py`,
  inanimate/structural/accidentally scorers) returns **no committed script** — only the
  gate definitions in `DATASET_GUIDELINES.md`/`DATASET_AUDIT.md` and the one-time result.

**Why this matters for G2 (and why it is surfaced now, not at Phase 1):**
`DATASET_AUDIT.md` records the v2 moral set at **44.0% accidentally-moral neutrals**,
**73.6% any-fail**, **317 clean of 1,200**. Accidentally-moral neutrals are precisely the
construction defect that makes the G2 paraphrase gate harder to pass (the probe can read
residual moral weight in the "neutral" rather than memorized vs. structural moral
content). Phase 1 must therefore **budget to rebuild the LLM-scored §1.1/§1.2/§1.5 gate
runner**, not assume it is present. The mechanical `validate_pairs` half is reusable as-is.

**Verdict:** half-present. Mechanical gates reusable; LLM-scored construction gates are a
Phase-1 build item. This is a correction to the plan's Section-12 reuse inventory; it does
**not** alter G2's or G3's pre-registered definition.

---

## Summary

| Hook | Gate it serves | Status | Action taken |
|---|---|---|---|
| `last_token_means` + `subspace_projection_fraction` | G3 Point A | ✅ matches | Pin Point A to layer 16; reuse the baseline's own projection fn |
| `direction_utils` projection primitives | G2 + all projections | ✅ matches | Use `transfer_metrics.acc_midpoint` for G2 accuracy |
| `effective_dimensionality` | G3 null denominator | ⚠️ centers | Override: uncentered eff-rank (PREREGISTRATION §5) |
| `validate_pairs` (mechanical) | G2 upstream | ✅ reusable | Re-run on new pairs in Phase 1 |
| LLM-scored §1.1/§1.2/§1.5 gates | G2 upstream | ⚠️ runner not committed | Phase-1 build item (not a present hook) |

Two corrections fell out of this check — the centered-`eff_dim` fork and the
half-present audit harness — and both are recorded where they bite. Neither changes
the construction of G2 or G3, so the pre-registration below is safe to commit.
