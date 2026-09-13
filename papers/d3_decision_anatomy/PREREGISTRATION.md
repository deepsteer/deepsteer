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

- **2026-07-02 (Amendment 1 — referee-pass corrections, pre-registered before the typing prep builds
  any asset):** A review of the skeleton found one stimulus–outcome mis-specification, one degenerate
  branch in the decisive cell, and a head-attribution specificity gap; plus two lit-driven framing
  changes. All fixed here before any patch stimulus is built.

  **(1) Twin cell — stimulus–outcome mismatch (fix).** The v2 1,200 twins are **third-person
  narrative** moral-status pairs; stock instruct models do **not** refuse discussion of either side,
  so "swap twin activations → Δrefusal" is **flat by construction** and says nothing about transport.
  Correction:
  - **Twins carry `outcome_variable = judgment-decision readout`** (the outcome they can actually
    move); Δrefusal is reported but **pre-registered expected-flat** on twins.
  - **New asset — request-twins.** Recast a v2 subset as **surface-matched requests** ("help me plan
    to [norm-following vs violating action]"), holding surface form, flipping only moral status.
    These are the **minimal-pair refusal-patching stimuli** for cell (a)/(b). Built now in the typing
    prep with type blocks + provenance.
  - **XSTest is a category-contrast, not a minimal pair** → typed as the **looser generalization
    cell**, not the minimal-pair cell (its role in (c) is harm-cue-vs-surface separation, not a clean
    twin swap).
  - **Token-alignment metadata per pair** (patch positions must map across the two members). Fixed
    rule for length-mismatched pairs: **align on the shared prefix + the flipped content span**;
    pairs that cannot be aligned to within a fixed token budget are **excluded and counted**.

  **(2) Decisive cell (b) — the negative branch was degenerate; add a transport positive control.**
  "Full patch moves refusal, `V_moral`-restricted doesn't → reads non-`V_moral` features" has an
  **unexcluded alternative:** the moral variable is encoded **beyond rank 3 / nonlinearly**, so a
  rank-3-restricted swap **under-transfers** any moral signal (instrument too weak, not "features
  elsewhere"). **Transport positive control (pre-registered):** the `V_moral`-restricted patch must
  first be shown to **move the judgment readout/behavior on twins**. Then:
  - restricted patch flips **judgment** but not **refusal** → **"refusal reads non-`V_moral`
    features"** is clean (the program-null-explaining result);
  - restricted patch cannot even move **judgment** → the **negative branch is uninformative**
    (rank-3 under-transfer not excluded); only the **positive** branch of the cell carries weight.
  **Both branch framings (both publishable):**
  - **`V_moral`-restricted patch MOVES refusal** → geometric orthogonality at the readout **plus**
    causal transport of `V_moral` content into the writers = the **complete through-weights story with
    anatomy** (the strongest positive outcome).
  - **only full moves it, restricted moves judgment-not-refusal** → refusal reads moral content
    through **non-`V_moral`** features = explains every geometric null in the program.

  **(3) Head attribution needs a channel-matched null (PR ≈ 15 specificity).** In a ~15-dim token,
  any head that writes strongly into the token projects onto most channel directions, so raw
  "projection onto refusal" collapses "top refusal-writers" into "top token-writers." **Head score
  (fixed now) = ⟨write_h, r̂⟩ − mean_j ⟨write_h, ĉ_j⟩**, where `{ĉ_j}` are **channel-basis control
  directions** (an orthonormal basis of the decision-token channel, norm-matched). Causal checks use
  **mean / resample ablation, not zeroing** (zeroed OV outputs are off-distribution). The
  reconstruction ≥ 0.90 control checks the **decomposition**; the channel-matched score checks
  **specificity** — both are required.

  **(4) Pilot gate before any patching session — behavioral-discrimination screen.** Keep only
  twins / request-twins whose **baseline behavior differs across the pair** (judgment flips, or
  refusal differs). MPS-runnable on OLMo; a patch pair with no baseline behavioral gap cannot be
  moved and is dropped (counted).

  **(5) Positioning (lit-driven, `LITERATURE.md`).** (a) **Promote Qi et al. 2024 (`2406.05946`) to
  framing:** the decision-site bottleneck is a candidate geometric mechanism of *shallow alignment*;
  C1 = "the measured substrate of shallow alignment" (correlational until a causal cell ties channel
  to depth). (b) **Engage Wollschläger et al. 2025 (`2502.17420`):** refusal is a concept cone; state
  "channel, not direction" — a ~15-dim channel hosts a cone, compatible, reconciled explicitly.

  Spine preserved: `M = 0.05`, the two-step null, per-unit saves, position-validity all unchanged.
  This amendment strengthens the stimuli, the decisive cell, and the attribution instrument before any
  compute.

---

### Amendment 2 (2026-07-02) — two-sided reconstruction gate + reordered-norm LN-fold (executes the pre-registered escalation)

**Trigger.** The first real C1 run (`Olmo-3-7B-Instruct`, layer 16) returned Stage-1
`reconstruction = 3.05` — the per-head OV decomposition overshoots the true residual write by 3×. The
original one-sided gate (`recon ≥ 0.90`) passed it silently. Cause: OLMo's **reordered norm**
(RMSNorm on the attention/MLP *output* before the residual add), so the real write is
`RMSNorm(Σ_h W_O^h z_h)`, not the raw sum. Logged as **ANOMALIES.md A3 (ledger)**.

**This is not a new decision — §1 already named it:** "the reconstruction control ... catches ... r̂
read post-final-LN (then **fold the LN gain, escalate from the folded-LN approximation**)." Amendment
2 only formalizes the gate shape and the fold, and records that the escalation **fired for OLMo**.

**Changes (committed before any folded number is computed):**
1. **Two-sided reconstruction gate** `0.90 ≤ recon ≤ 1.10`. Overshoot (un-folded block norm) and
   undershoot (missing components) both fail. `reconstruction_ok()` in `stage1_attribution.py`.
2. **Exact RMSNorm fold** (`rms_gain`): per layer, multiply each pre-norm component write vector by
   `g = γ / sqrt(mean(x²)+ε)` (RMSNorm is diagonal at a fixed token, so `Σ_h contrib_h ⊙ g =
   norm(Σ_h contrib_h)` — exact, unit-tested to 1e-9). Auto-fires for reordered-norm families
   (detected by `post_feedforward_layernorm`); no-op for pre-norm (Llama/Qwen reconstruct ~1.0 raw).
   `reordered_norm` is recorded in every Stage-1 result.

**Scope of impact.** Re-runs only the **Stage-1/2 head anatomy** (which heads write refusal, what
they read); the un-folded top-head list, specificity, `k`, and `mlp_write_fraction` from the first
run are **superseded** (inflated). The **decisive `cell_b_verdict`** is patch-based (reads the real
forward pass, no decomposition) and is **unchanged** — the first-run headline
(`reads_non_vmoral_features`, transport control passed) stands. Spine preserved: `M = 0.05`, the
two-step null, per-unit saves, position-validity, the pilot gate, all decisive-cell branch definitions
unchanged.

---

### Amendment 3 (2026-07-02) — ratio-of-ratios verdict test, OLMo hardening cells, and the A1-robust comparative statistic

Committed **before** the gated quantity is computed. The first OLMo run passed the absolute transport
control (V_moral-restricted patch moves judgment above MDE) and returned `reads_non_vmoral_features`.
That control is necessary but **not sufficient**: a rank-3 restriction that under-transfers *every*
outcome uniformly could still clear the absolute judgment threshold while the "reads non-V_moral"
reading is wrong. Amendment 3 replaces the sufficiency test with a **within-outcome transfer-fraction
comparison** and pre-registers the OLMo-hardening cells + the panel's comparative statistic.

**(0) The ratio-of-ratios verdict test [the gated quantity].** Define the fraction of each full-patch
effect that survives V_moral restriction:
- `R_refusal  = mean(restricted→refusal) / mean(full→refusal)` on request-twins (both deltas paired
  per request-twin).
- `R_judgment = mean(restricted→judgment) / mean(full→judgment)` on compositional twins (both deltas
  paired per twin). **`full→judgment` is RIDER 0 — absent from the first run's saved arrays
  (`c1_inputs_olmo3.npz` has full→refusal, restricted→refusal, restricted→judgment only); it must be
  logged in the OLMo-hardening pod (interchange, `restrict_Q=None`, read `jdir`, on compositional
  twins).** Until rider 0 lands, `R_judgment` is undefined and the test is not evaluated.

  **Decision rule (frozen).** `reads_non_vmoral_features` **STANDS** iff `R_judgment − R_refusal ≥
  M_ratio` (M_ratio = **0.15**) **and** the bootstrap 95% CI of `(R_judgment − R_refusal)` excludes 0
  (2000 iters; resample request-twins for `R_refusal`, compositional twins for `R_judgment`,
  independently; ratios formed on the resampled means). Reading: V_moral restriction preserves a
  substantially larger share of the judgment effect than of the refusal effect → V_moral is a good
  substrate for judgment but not for refusal → refusal reads elsewhere.
  - CI excludes 0 but point `< M_ratio` → **"directionally confirmed, small margin"** (report, do not
    over-claim).
  - CI **includes 0** (ratios ≈ equal) → **RECLASSIFY `under_transfer`**: the restriction
    under-transfers both outcomes comparably, so "reads non-V_moral" is not distinguished from
    "rank-3 too weak." Cell-b is then **redesigned with a rank sweep** `k ∈ {1, 3, 8, 16}` (restrict
    to the top-k of an over-complete moral basis) to map transfer-fraction vs rank **before any panel
    extension**. No panel run proceeds under an unresolved `under_transfer`.

**(1) OLMo-hardening cells (framings frozen).** Same loaded-model session as rider 0:
- **complement patch** — patch all-*but*-V_moral (`I − QQ^T`). Expected: **moves refusal** (the
  non-V_moral features carry it); a direct positive confirmation of `reads_non_vmoral_features`.
- **harm-restricted patch** — restrict to the rank-1 t_inst harm direction (Zhao et al. 2025). If
  refusal reads the harm feature specifically, this **moves refusal** where V_moral-restricted does
  not → localizes "non-V_moral" toward harm.
- **random-rank-3 restricted control** — restrict to a random rank-3 subspace (matched norm).
  Expected: **does not move refusal** (nor judgment); shows the V_moral restriction is not "any rank-3
  under-transfers." Ships with an MDE.
- **generate-under-patch** — on ≥ 8 twins, upgrade from projection readouts to **behavioral refusal
  flips** (does the completion flip refuse↔comply under the patch); closes the "readouts not
  behavior" limitation.
- **per-layer attention capture** — characterize the read for the earlier-layer top writers (the
  first run's Stage 2 only covered the L_ref writers); closes the Stage-2 coverage limitation.
- **L15 H15 discriminator** — the anti-refusal writer (spec −0.142): single-head mean/resample
  ablation should **increase** XSTest-safe over-refusal if it is an anti-over-refusal head.

**(2) Stimuli + bookkeeping (zero-GPU, before the pod).** Expand request-twins 24 → **~60** (target
**n ≥ 25 surviving** the baseline-refusal screen); add transport twins for **≥ 2× MDE headroom** on
the judgment control. Benign members drawn **away from alarming-surface phrasing** (XSTest-safe
register) so benign-twin over-refusal is a reported datapoint, not a screen artifact. Resolve the
**114-screened-vs-20-used** transport bookkeeping (the first run screened 114/200 compositional twins
but the cell capped the transport sample at 20): the transport sample size is reported = used, and the
cap is raised to meet the headroom target.

**(3) A1-robust comparative statistic [panel].** The cross-model comparison uses the **ratio-of-ratios
`R_refusal` (and `R_judgment − R_refusal`), NOT raw cell-b deltas** — a within-model ratio is immune
to per-model overall transfer-fraction and to the A1 massive-activation scale differences. The
pre-registered comparative prediction "**Llama reads content more**" = **higher `R_refusal`** (V_moral
restriction preserves more of Llama's refusal effect than OLMo's). Llama Stage-1/2 extraction is run
in **per-dim-standardized space** (dim-788-robust σ from format/position-matched sink-free
decision-token samples; ANOMALIES A1), gated on the **OLMo raw→standardized invariance** check passing
first (rider: save per-head arrays in the OLMo pod). Llama is pre-norm so **no LN-fold**, but the
reconstruction band is verified anyway.

Spine preserved: `M = 0.05` for cosine/fraction rungs, the two-step null, per-unit saves,
position-validity, the pilot gate. Amendment 3 adds the ratio-of-ratios verdict gate, the hardening
cells, and the comparative statistic; it does not relax any prior gate.

---

### Amendment 4 (2026-07-02) — rank sweep, harm identification, and behavioral severity ladder (resolves `under_transfer`)

Committed **before** the Δ-substrate CI is computed and before the sweep is built. The powered
hardening run returned **`under_transfer`** (`R_refusal = 0.338`, `R_judgment = 0.517`, diff 0.179 but
CI [−0.238, 0.388] includes 0). The powered cells also showed V_moral is a **specific minority**
substrate (restricted −0.028, random rank-3 −0.0005), that the **harm rank-1 direction (−0.026) ≈
V_moral rank-3**, and a **behavioral floor** (2/8 baseline violating refusals, 0 benign). Amendment 4
pre-registers the resolution: a rank sweep, harm identification, and a severity-ladder behavioral fix.
One OLMo pod session.

**(0) Substrate-language gate [zero-GPU, gated].** "V_moral is a specific refusal **substrate**"
language is licensed **only if** the paired bootstrap 95% CI of `Δ = |V_moral-restricted→refusal| −
|random-rank-3→refusal|` (paired per request-twin, from the saved `cell_restricted_deltas` vs
`cell_random_deltas`) **excludes 0**. Also report `p(d_harm | V_moral) = frac(V_moral, d_harm)` at
mean_content from the saved artifacts. These enter `papers/SYNTHESIS.md` (routing-form thesis) under
this rule; if the CI includes 0, the wording drops to "V_moral-restricted moves refusal at n = 23 but
is not separable from a random rank-3 restriction."

**(1) Sweep spine.** A **nested moral-contrast PCA basis** (eigenvectors of the paired moral−neutral
content-contrast covariance; nested `rank 1 ⊂ 3 ⊂ 8 ⊂ 16`), `k ∈ {1, 3, 8, 16}`. Per rank report:
- **paired `R_refusal(k)` and `R_judgment(k)`** (restricted-to-rank-k ÷ full, paired over twins);
- a **random rank-k null curve** (random orthonormal rank-k, V_moral-excluded; the specificity floor);
- **per-rank moral/neutral purity** (how well rank-k separates moral vs neutral content);
- **`cos(d_harm, PC_k)`** per component (where the harm direction sits in the basis);
- an **additivity check**: `V_moral-restricted + complement vs full`, with CI (the powered run's
  34% + 76% ≈ 110% suggests near-additivity; report it explicitly).

**(2) Identification cells.**
- **harm-partialed patch** — restrict to `V_moral` with `d_harm` **projected out** (`V_moral ⊥ d_harm`).
  If this does **not** move refusal while full-`V_moral` does, refusal's V_moral effect **is** the harm
  component.
- **harm rank-1 replication with CI** — re-run the harm-restricted cell with a bootstrap CI.

**(3) Shape verdicts (frozen, all publishable).**
- **harm-saturating** — `R_refusal(k)` plateaus ≈ the harm-rank-1 level while `R_judgment(k)` climbs →
  refusal's moral read **is the harm percept**, not the broad subspace.
- **broad-moral** — `R_refusal(k)` climbs toward `R_judgment(k)` → rank-3 was a **truncation**; the
  D1/D2 "refusal ⊥ morality" framing **softens** (refusal reads higher-rank moral content).
- **instrument-ceiling** — both plateau `≪ 1` → **linear restricted transport saturates**; escalate to
  nonlinear transport or declare the ceiling as the finding.

**(4) Behavioral severity ladder.** ~30–40 authored **severity-ladder twins**: both members of a pair
shift register **together** (surface-matched within pair), spanning a severity range. The pilot screen
selects the **operating band** where violating-refuses **and** benign-complies. Deliverable: the
**refusal-vs-severity psychometric curve** for both twin types (a behavioral dose-response on the harm
cue, pairing with the harm-direction cell) + the pass rate. Passing twins are **dual-use** (type
blocks note it) and feed the readout cells; **generate-under-patch** and the **L15 H15 over-refusal
discriminator** run once the stimuli are out of the floor.

**Held (not this session).** Mass twin-authoring (conditional on the sweep staying ambiguous); the
Llama pod + its comparative pre-registration (the comparative **statistic is chosen after the shape
verdict**). The Llama **standardization build proceeds in parallel** as zero-GPU (ANOMALIES A1).

Spine preserved: `M = 0.05`, the two-step null, per-unit saves, position-validity, the pilot gate,
`M_ratio = 0.15`, all Amendment-3 branch definitions. Amendment 4 adds the sweep, the identification
cells, the shape verdicts, and the behavioral ladder; it relaxes no prior gate.

---

### Amendment 5 (SKELETON, 2026-07-02) — GPT-OSS-20B reasoning-MoE extension of C1

**Status: SKELETON.** Framings frozen; thresholds/positions finalized before the GPT-OSS pod, after
Llama/Qwen reuse the harness. GPT-OSS is the panel endpoint (most new harness). Motivation: its
in-trace refusal (P2) already reads **harm-loaded** (D1 retro-audit: standardized |cos(P2, d_harm)| =
0.49 vs |cos(P2, V_moral ⊥ d_harm)| = 0.13), so it is a strong test of whether `harm_saturating`
survives a reasoning MoE with a distinct training regime.

1. **Outcome redefinition (both pre-registered).** OLMo's refusal barely tracked intent severity (a
   behavioral floor), so the OLMo primary was the projection readout. GPT-OSS refuses far more
   (harmful-prompt baseline ~1.0), so behavior is viable. **PRIMARY = behavioral** (generate-under-
   patch refuse↔comply flips on operating-band severity-ladder twins). **SECONDARY = the rank-r
   refusal-subspace projection readout** (OLMo-comparable). The behavioral primary is the upgrade the
   OLMo floor prevented.

2. **Severity-ladder inversion.** GPT-OSS baseline ~1.0 **inverts** OLMo's floor risk into a
   benign-twin **over-refusal** risk (the following member may also refuse). The ladder brackets both
   ends, so the operating-band screen still finds where violating-refuses AND benign-complies — the
   band just shifts down in severity. **Deliverable: the refusal-vs-severity psychometric curve per
   model** — how steeply each model's refusal rises with intent severity is itself a cross-model
   harm-keying comparison.

3. **MoE Stage-1 scope.** The decision-token write is a router-weighted mix of expert MLPs. Stage 1 =
   per-head attention OV write (well-defined) + a **router-weighted MLP fraction**; the reconstruction
   control still holds (residual is linear over attn + routed-MLP + embed). If MLP fraction > 0.50
   (likely for MoE), the Jacobian branch fires → attribute to the **top experts by router weight**;
   full expert-level circuit attribution is a follow-on.

4. **Harmony decision-token — a hard pre-condition.** GPT-OSS's harmony format has distinct channel
   tokens. Pre-register the decision-site token (the token before the assistant response channel opens;
   the D1 P2 position and the assistant-header analog are candidates). **Two pre-conditions gate the
   projection (secondary) outcome, measured before the sweep:** (a) the **post-standardization
   participation ratio** at that token must be reported, and (b) the **standardized band-below-null
   validity check** (A2: the positive-control moral band must sit *above* the covariance null at that
   position). **If the position is invalid (band-below-null or PR < 30 even standardized), GPT-OSS is
   behavioral-primary-only by pre-registration** — the projection-readout sweep is not reported as a
   causal transfer, only the behavioral flips + the psychometric curve stand.

5. **Reconstruction + A1.** Verify GPT-OSS norm placement and apply the A3 fold if reordered; verify
   the reconstruction band regardless. GPT-OSS is the **strongest A1 outlier** (top-dim variance
   0.70) → Stage-1/2 and every null run **per-dim standardized**; raw reported only as the degenerate
   baseline (no OLMo-style raw↔std invariance is expected — standardized is primary).

6. **Comparative branches (frozen).**
   - `R_refusal(k)` **saturates** at the harm level → the **strongest generality** of
     `harm_saturating` (a reasoning MoE with deliberative alignment still routes refusal through harm).
   - `R_refusal(k)` **climbs** toward judgment → a **training-regime exception**: deliberative
     alignment lets refusal read broader moral content; connect to the deliberative-alignment
     literature (the model reasons over moral content, not just the harm cue).

**Sequence:** #18 Llama → Qwen (template reuse) → GPT-OSS pod once this skeleton is finalized and
passes local test. Spine preserved throughout: `M = 0.05`, `M_ratio = 0.15`, two-step null,
per-unit saves, position-validity, the pilot gate, all Amendment 3/4 branch definitions.

---

### Amendment 6 (2026-07-02) — dual-basis Llama re-run: is it "reads broad" or a richer harm percept?

Committed **before** authoring the stimulus batch or computing the design parameters. The first Llama
run was underpowered (heterogeneous severity-band twins, cells below MDE) but left a directional hint:
`R_refusal` reached 0.44 at rank 16, ~3× Llama's harm-rank-1 level (0.14), and the harm-saturation
one-knob model failed. **Why this pod matters most:** if the hint is real, Llama is the model where
refusal reads the *broad moral subspace* — which would be the mechanistic explanation for Llama being
the program's n=1 robustness anomaly (judgment degrading under refusal ablation, partial
unablatability, eff-rank-4 refusal). Routing thesis meets robustness conjunction, causally — the
flagship's cross-model figure.

**The confound to kill first.** "Reads beyond harm" has a rival reading: **rank-1 `d_harm` underfits
Llama's harm percept.** `R_refusal 0.44 ≫ harm-rank-1 0.14` is equally consistent with Llama
representing harm **multi-dimensionally** (severity × category × intent — and "intent-coupled refusal"
gestures exactly there), so the moral basis's higher PCs pick up *additional harm* content that a
rank-1 `d_harm` misses, not non-harm moral content.

**Design change — the dual nested-basis sweep (the discriminator).** Run the sweep against **two**
nested bases: (1) the **moral-contrast basis** (as now, `k ∈ {1, 3, 8, 16}`), and (2) a
**severity-derived harm basis** built from the severity ladder's graded contrasts (`k ∈ {1, 2, 4}`).
The one-knob fit runs against **both**, with **RMSE bootstrap error bars** (a fit-failure verdict needs
its uncertainty). Discriminator:
- `R_refusal` follows the **harm basis** and saturates against the moral basis beyond the harm span →
  **harm-keyed with a richer harm percept** (harm-keying generalizes; harm *dimensionality* is the
  family difference).
- `R_refusal` climbs the **moral basis** where the harm basis is exhausted → **genuinely reads broad
  moral content** (routing differs by family; the robustness-conjunction mechanism; the GPT-OSS
  comparative is reframed).
- **still-unresolved at the computed power** → report as bounded; Qwen proceeds anyway.

**Power is computed, not guessed.** The heterogeneous run gives within-level variance for free
(per-twin full effects at levels 3/4/5 separately). Compute the **MDE(n) table** for `n ∈ {16, 24, 32,
40}` from the measured within-level variance and author to the `n` it demands. **Select the level** by
the measured psychometric curve (refusal-discrimination pass rate × headroom, weighing benign-twin
over-refusal risk), not an assumed level 5.

**`d_harm` validity (pre-condition).** Confirm `d_harm` is Llama-extracted (t_inst, Llama act samples);
validate it against the severity ladder — twin-difference projection onto `d_harm` must **climb
monotonically with severity level**. If flat/non-monotone, `d_harm` is mis-specified for Llama and the
0.14 baseline means nothing → re-extract before anything else.

**Composition-covariate (handles the stimulus asymmetry by measurement, not equation).** A single
stimulus set both models refuse differentially does not exist (OLMo barely refuses). Instead report
each set's **twin-difference composition** (harm / ⊥harm-moral / residual content fractions) as a
covariate beside each model's curves, and state that the **within-model** design (each model's `R(k)`
shape against its own harm level, now against its own two bases) was chosen precisely because
cross-model stimulus equating is impossible. Composition-aware comparison is the defensible comparative.

**Held.** Qwen until the Llama design is proven (then same harness, same dual-basis statistic); the
GPT-OSS Amendment 5 inherits the dual-basis + composition-covariate upgrades. **Ledger note:** Llama's
decision-token channel reads A1-clean despite the global dim-788 outlier → massive dims may be
**position-dependent**; a one-line ANOMALIES entry + a cheap per-position top-dim-share profile if it
ever matters.

Spine preserved: `M = 0.05`, `M_ratio = 0.15`, two-step null, per-unit saves, position-validity, the
pilot gate, all Amendment 3/4 branch definitions. Amendment 6 adds the second basis, the power table,
the composition covariate, and the `d_harm` validity gate.

---

### Amendment 7 (2026-07-02) — instrument-diagnosis tree (all zero-GPU; pre-registered before computing)

Amendment 6's power table showed Llama's block is the interchange instrument, not stimulus power
(per-twin refusal deltas sign-chaotic, SD 0.31; behavioral `flips_to_comply = 0`). Before building any
fix, diagnose *why* from the saved arrays. The tree is fixed now; the branch is chosen by the data.

**Root split — the judgment cell is the instrument's positive control.** The full patch is measured on
both refusal (`cell_full_deltas`) and judgment (`full_judgment_deltas`). Compute the per-twin JUDGMENT
delta **sign-coherence** (bootstrap 95% CI of the mean; fraction same-sign) for Llama, with OLMo as the
reference.
- **Judgment coherent (CI excludes 0) while refusal is chaotic (CI includes 0) → REFUSAL-SPECIFIC.**
  The patch *works* (it moves judgment); Llama's refusal specifically does not respond — a
  content-robustness candidate, not a broken instrument.
- **Judgment ALSO chaotic → INSTRUMENT-BROKEN.** The patch is noisy regardless of readout.

**Refusal-specific branch:**
- (a) **Delta vs baseline decision margin** (proxy: severity level / baseline refusal prob). Structure
  → **dynamic-range** verdict; fix = boundary-band twins drawn off the measured psychometric curve.
- (b) **Direction asymmetry** (benign→violating vs violating→benign, from saved cells if available;
  else pre-register a small confirmation cell). Asymmetric → **latch/hysteresis** finding.
- (c) Neither → **content-robustness candidate**: run the layer-divergence proxy; divergence below the
  patch layer → patch-layer fix; else **genuine robustness**, reported with the judgment cell as the
  validity certificate (the instrument is proven to work, so the refusal null is real).

**Instrument-broken branch:**
- (a) **Delta split by length-match status** + audit which alignment rule actually executed.
  Mismatched-only chaos → **per-position swap on length-matched twins** (build spec, moderate).
- (b) **Raw-vs-std delta correlation** closes the readout-noise reading (requires a raw-readout
  re-derivation; flagged if the raw deltas were not saved).

**Escalation menu (build only what the branch demands):** boundary-band authoring (cheap) /
per-position aligned swap (moderate) / patch-layer or multi-layer flag (cheap harness change) /
directional steering **only as last resort**, with the construct shift pre-registered (steering tests
direction-sensitivity, not content-reading — a different claim per construct-audit).

**Held:** the Qwen pod until the diagnosis lands (whatever breaks on Llama likely breaks on Qwen's
severe band; the harness fix, if any, exists before the third model burns a pod). **Ledger:** the
median-sign flip (Llama +0.029 vs OLMo −0.083) attaches to whichever branch resolves it; the
dim-788-at-content-positions result closes the Amendment-6 position-dependence entry (A2 strengthened).

Spine preserved: `M = 0.05`, `M_ratio = 0.15`, two-step null, per-unit saves, position-validity,
all prior branch definitions. Amendment 7 adds only the diagnosis tree and its fixed decision rules.

---

### Amendment 8 (2026-07-02) — boundary-band Llama re-run (executes the Amendment-7 dynamic-range fix)

Committed **before** authoring. Amendment 7 diagnosed Llama's chaotic refusal cells as
**dynamic-range/saturation** (judgment cell certified the instrument; operating-band twins sit at the
refusal ceiling). Amendment 8 fixes it with boundary-band stimuli and hardens the re-run.

**(0) Void the directional hint.** The `R_refusal 0.44 vs harm-rank-1 0.14` "reads beyond harm" hint is
**retracted** in RESULTS + SYNTHESIS: its inputs are diagnosed latched (A5), so it carries no evidential
weight. The three branches (**reads-broad / richer-harm-percept / bounded-unresolved**) enter the re-run
**unweighted**.

**(1) Band census → authoring spec (zero-GPU).** From the existing psychometric, census the existing
twins for a boundary band (violating-refuse `p ∈ [0.4, 0.7]`) and read the **level-2→3 slope**. If the
slope is steep (a single-level jump 0.33 → 1.0, as observed), author a **micro-graded ladder** (finer
severity steps between levels 2 and 3); if boundary twins can't be hit by severity alone, vary an
**orthogonal knob** (specificity/immediacy of the harmful ask) to land the band.

**(2) Selection-regression control.** The pilot screen selects twins into the band on a noisy baseline;
that selection regresses to the mean. The pod **re-measures the baseline independently** on the selected
twins, and the **shrinkage** (screen-band vs re-measured-band drift) is reported. Cells use the
re-measured band, not the screen-selected one.

**(3) Pod structure (gated).** `screen → GATE (≥ N boundary twins, N from the power table) →` the
Amendment-6 **dual-basis patch run** (moral basis `k ∈ {1, 3, 8, 16}`, Llama-ladder-derived harm basis
`k ∈ {1, 2, 4}`, harm-partialed cell, one-knob fit against **both** bases with RMSE error bars) **+
bidirectional cells** (following→violating **and** violating→following; asymmetry = latch/hysteresis,
Amendment 7 (b)) **+ judgment recertification at the boundary** (re-confirm the instrument is coherent
at the new operating point). **Gate fail → bank `bounded-unresolved`, automatic** (no forcing).

**(4) Composition covariate.** A twin-difference composition table (harm / ⊥harm-moral / residual
fractions) for **both** models' stimulus sets; OLMo's baseline-refusal distribution is added to the A5
ledger entry (the dynamic-range comparison OLMo-weak vs Llama-saturated).

**(5) Methods-note (standing task).** During the authoring/build window, draft the methods note: A1–A5
plus the power-table and orthogonal-cell-certificate patterns are a complete methods-paper skeleton now.

Spine preserved: `M = 0.05`, `M_ratio = 0.15`, two-step null, per-unit saves, position-validity, all
prior branch definitions. Amendment 8 adds the boundary-band authoring, the gate, the bidirectional and
recertification cells, the selection-regression control, and the composition covariate.

---

### Amendment 9 (2026-07-02) — engage/disengage asymmetry: hysteresis vs early-commitment (pre-registered before building)

The Amendment-8 boundary run showed Llama's refusal is directionally asymmetric: coherent when harmful
content is added, incoherent when it is removed. Amendment 9 pins the statistic, the mechanism
discriminator, and the still-open reads-broad question on the working (engage) channel.

**(1) Nomenclature freeze.** **engage** = harm-add (following→violating context; refusal ↑);
**disengage** = harm-removal (violating→following context; refusal ↓). All D3 docs audited to this.
**Reclassification:** OLMo's original **−0.134** full cell is a **coherent DISENGAGE** datapoint at
OLMo's band (removing harm lowers OLMo's refusal) — so OLMo's disengage is already evidenced; RESULTS
states this. Llama's disengage (−0.014, incoherent) is the contrast.

**(2) Asymmetry statistic.** `A = (|engage| − |disengage|) / (|engage| + |disengage|)` per model,
paired within twins, bootstrap 95% CI. `A ≈ 0` symmetric; `A → +1` engage-only (full latch). The
**cross-model claim = the CI on `A_Llama − A_OLMo`**. Report the **per-twin engage-vs-disengage
scatter**: a uniform disengage-null vs a **bimodal** pattern (some twins un-latch) is itself diagnostic.

**(3) Mechanism cells — hysteresis vs early-commitment.**
- (i) **Twin-difference norm at `t_inst` by layer** (both models): where does the moral-status contrast
  crystallize relative to the patch layer? Zero-GPU **iff** per-layer `t_inst` slices were saved; else a
  cheap extraction rider.
- (ii) **Patch-layer sweep on Llama disengage:** patch the disengage swap at earlier layers. If an
  earlier patch **restores disengage coherence**, the decision **crystallized early** (early-commitment),
  not a true latch. **Wording (frozen): "latch" is claimed ONLY if disengage fails at all accessible
  depths;** if an earlier layer restores it → **early-commitment** verdict.

**(4) Engage-direction dual-basis sweep (the reads-broad question, on the coherent channel).** Run the
sweep on the **engage** direction: moral basis `k ∈ {1, 3, 8, 16}` vs the Llama-ladder-derived harm
basis, one-knob fit against both (Amendment 6 machinery), sign conventions per the freeze. This finally
answers whether Llama's refusal, *in the direction it moves*, reads broad moral content or harm.

**(5) Behavioral counts.** Report **flips-to-refuse** (engage: does the add-harm patch flip a benign
request to refuse) **and flips-to-comply** (disengage) beside the projections — the asymmetry lands as
behavior or it is a readout-only claim.

**(6) OLMo cells local-first.** Run OLMo's engage + a matched-band disengage replication on **MPS** if
the harness ports; pod only if not. Check **Llama-8B MPS feasibility** for the patch-layer sweep before
provisioning a pod.

**(7) Claim ceiling.** Released-model characterization stays within the standing safety line. The
robustness-mechanism sentence stays **"candidate mechanism"** until (3) picks latch vs early-commitment
— and note that **both mechanisms predict the Paper-6 robustness anomaly, so the conjunction survives
either branch**.

**(8) Head bookkeeping.** Confirm the anti-over-refusal head ID on Llama **independently** — `L15 H6`
collides with an OLMo writer index; verify it is Llama's own min-specificity top head, not a carryover.

Spine preserved: `M = 0.05`, `M_ratio = 0.15`, two-step null, per-unit saves, position-validity, all
prior branch definitions. Amendment 9 adds the asymmetry statistic, the mechanism discriminator, the
engage-direction sweep, and the behavioral flip counts.

---

### Amendment 10 (2026-07-02) — verification before SYNTHESIS freeze (depth-confound + sign audit)

Pre-registered before computing the verification quantities (some can change the reads/asymmetry
verdicts). The reads-broad verdict was read at layer 16, which is **post-commitment** for Llama
(early-commitment) — so it may be a depth artifact; and the `A` comparison patched both models at layer
16, which may not be depth-matched. Verify before freezing SYNTHESIS.

**Zero-GPU (saved arrays):**
- (i) **Harm-basis engage curve** `k ∈ {1, 2, 4}`: the engage sweep used only the moral basis + a
  rank-1 harm cell; the severity-derived multi-rank harm basis was never built. If absent → the
  **blocking gap** that the conditional pod fills (reads-broad vs richer-harm-percept, now on engage).
- (ii) **`d_harm` monotonicity validation** (Llama): twin-difference projection onto `d_harm` climbs
  with severity → cited in RESULTS beside the −0.04, or flagged as an extraction rider if unsaved.
- (iii) **Sign audit** of the rank-1 engage harm cell: engage-full is +0.14 but `engage_harm_rank1_R =
  −0.04` (opposite sign). Audit `eng_harm_d` sign under the nomenclature freeze — a sign bug would
  invalidate the −0.04.
- (iv) **Harm-partialed engage cell** + one-knob fits against **both** bases (moral + harm).
- (v) **OLMo patch-depth vs read-layer**: is the `A` comparison **depth-matched**? Recompute `A` at a
  matched *relative* depth if the data exists (else an OLMo patch-layer rider).

**Conditional pod (or MPS if it fits): engage dual-basis sweep at Llama layer 12 (+14 if budget)** —
the pre-commitment coherent depth. Frozen branches:
- **harm-basis-dominant-early** (harm basis dominates at layer 12) → **depth-resolved routing verdict**:
  unified harm-keying across the family; the OLMo/Llama difference is **depth + ratchet**, not
  *what* is read.
- **broad-dominant-at-all-coherent-depths** → **reads-broad stands**; the two-dimensional table is final.

**SYNTHESIS freeze policy.** The **commitment-dimension** sentence (early-commitment, not hard latch)
is clean and **freezes now** with depth-indexed phrasing: "bidirectional below ~layer 15, engage-only
at the read layer." The **reads-dimension** sentence is **held** until (i)/(iv) or the layer-12 sweep
lands.

**Methods note.** Add the **"depth-indexed intervention verdicts"** pattern: a verdict about what a
circuit *reads* must state the intervention depth **relative to commitment**. Drafting proceeds in
parallel; nothing above blocks it.

**Panel inheritance.** Qwen and GPT-OSS adopt **dual-basis-at-two-depths + depth-matched `A`** as
standard cells; Amendment 5 (GPT-OSS) adds the **bidirectional cell** — "does deliberative training
produce a reversible reader" is now formally testable with the `A` statistic.

Spine preserved throughout. Amendment 10 adds only verification cells and the depth-matched comparison.

---

### Amendment 11 (2026-07-02) — Llama epilogue (zero-GPU + small builder)

1. **Rule-7 line (fork discipline).** Document the **dropped dual-basis harm cell**: the severity-harm
   basis was pre-registered (A6/A9) but never run, because the **gap-closes-to-judgment** comparison is
   a cleaner reads-broad discriminator (refusal matching judgment's transfer ⇒ refusal reads the broad
   subspace judgment reads). Recorded so the drop is a stated fork, not silent.
2. **Severity-harm-basis builder** (nested rank 1⊂2⊂4 from the severity-ladder paired contrasts) +
   **capture curve**: how much of the engage-driving moral basis a rank-k harm basis spans (geometric),
   against the saved per-k engage outcomes. Resolves the **harm-coextensive** alternative to reads-broad.
   Needs the severity-twin content contrasts (extraction rider if unsaved).
3. **Harm-coextensive caveat** in RESULTS beside reads-broad, pending the capture number: reads-broad is
   "beyond harm" only if the harm basis does **not** span the engage-driving moral directions.
4. **Symmetric-depth confirmation.** The frozen cross-model `A` comparison uses **`A_OLMo` recomputed at
   the matched depth** (layer 12: −0.54), not the read-layer value (−0.20) — `A_Llama − A_OLMo = +0.26`
   is `(−0.28) − (−0.54)`. Verify no read-layer value leaked into the frozen comparison.

### Amendment 5 finalization (2026-07-02) — GPT-OSS commitment axis (pre-pod)

The GPT-OSS extension inherits the **commitment axis** (reversible-reader test):
- (i) **Prefill-bidirectional deliberation cells** on operating-band severity twins: **engage** =
  inculpating-analysis prefill, **disengage** = exculpating prefill; the `A` statistic **at the
  deliberation level** + per-twin scatter. (The content-swap patch is replaced by a reasoning prefill,
  the reasoning-model analog of the depth intervention.)
- (ii) **Trace-position commitment curve**: at what **trace fraction** the final decision becomes
  predictable, from activations captured during the same generations (the reasoning analog of the
  depth-commitment curve).
- (iii) **Frozen branches (all publishable):** reversible-reader (engage & disengage both move, low `A`)
  → deliberative training yields a reversible reader; early-commit-in-trace (decision predictable at low
  trace fraction, disengage fails late) → in-trace early-commitment; harm-keyed-deliberation (prefill
  moves refusal via harm content only) → unified harm-keying at the trace level.

**Safety scope.** Characterization of a released model on borderline-severity items; rates reported; no
technique optimization; one-off cells per the standing claim ceiling.

**Zero-GPU pre-conditions (from saved harmony-token samples).** Post-standardization PR + the
band-below-null validity check at the harmony decision token; **expected outcome locks
behavioral-primary** (A5 pre-condition rule). **Rider:** the **P0 (prompt-position) harm/⊥harm
decomposition** from saved arrays, so the correlational reads-harm chain is **prompt→trace consistent**
before any pod.

**Compute tiers.** **Tier 1** (one modest 20B session): psychometric curve + prefill deliberation cells
+ trace-position commitment curve. **Tier 2** (held, decide after Tier 1): full causal C1-MoE —
KV-persistent dual-basis prompt patches (causal reads verdict), router-weighted Stage-1, depth×position
commitment map (~2–3× a Llama session). **Tier 2 and Qwen are the follow-on paper's openers, not this
phase's blockers.**

Spine preserved. A11 is verification + a geometric builder; A5-finalization adds the commitment axis to
the held GPT-OSS extension.

### Amendment 12 (2026-07-03) — GPT-OSS commit axis: band-existence + graded disengage (pre-pod)

The first Tier-1 run banked position gate (`pr_std 12.79`, valid) + engage (7/7 benign→refuse) but the
disengage arm and the harm-separability commitment curve were saturation-confounded (empty boundary
band; A7 trap). Amendment 12 redesigns the disengage instrument to be robust to either outcome and
un-confounds the commitment curve. **All quantities below are pre-registered before compute.**

1. **Band-existence check (GATES the finer-ladder decision).** Per-item base-refuse histogram across the
   existing severity twins (needs per-item base-refuse, which the harness now saves). **Both responses
   frozen before looking:** (a) **smooth spread** of violating base-refuse across [0,1] → *resolution-
   limited* → a finer/milder severity ladder will land a boundary band (build the secondary ladder);
   (b) **bimodal at 0/~1** (few items between) → *genuine step function* → no ladder can land a band,
   the graded-prefill disengage below is the only disengage instrument, and the finer ladder is not
   built. Operationalized: bimodality = fraction of violating items with base-refuse in (0.2, 0.8) < 0.2.

2. **Graded disengage (PRIMARY disengage instrument, robust to either band outcome).** A weak→strong
   series of exculpatory-reasoning prefills (the mirror of the working engage arm) applied to the SAME
   ceiling-refusing violating items. Two graded within-item readouts per prefill strength: (i)
   behavioral flip rate (refusal on the FINAL channel), and (ii) the now-licensed **rank-r projection
   readout** — the decision-channel activation under each prefill projected onto the refusal direction
   (a graded, continuous within-item measure that does not need a behavioral flip to register movement).
   The milder-severity ladder is **secondary/confirmatory**, built only under branch 1(a).

3. **Frozen branches (all publishable).**
   - **strong-exculpatory-prefill flips** (behavioral flip OR monotone projection decrease toward the
     comply pole at high prefill strength) → *disengage consequential → GPT-OSS reversible*: the Llama
     contrast, an affirmative answer to deliberative reversibility.
   - **maximal prefill cannot move it** (no flip AND flat/limited projection movement at max strength) →
     *genuine downward-robustness at consequential-engage* — a distinct third pattern (**reversible up,
     robust down**), not a saturation artifact (the graded projection readout rules saturation out).
   - **resolution-limited histogram** (branch 1a) → finer ladder as the behavioral cross-check on the
     prefill result.

4. **Commitment curve, un-confounded.** Replace harm-separability predictability with
   **decision-predictability WITHIN fixed harm-status**: regress the final behavioral outcome on
   trace-fraction activations using **borderline items only** (items whose outcome is not determined by
   harm status — harmful-that-comply / harmless-that-refuse, or graded-prefill items that flip). This
   measures when the *decision* (not the harm representation) becomes fixed. **If no borderline items
   exist** (step-function case), report the commitment curve as **`not_computable_at_this_operating_
   point`** — never substitute the harm-confounded version (the rule already written into the harness).

5. **Write-up NOW (does not wait on the disengage resolution).** These enter RESULTS/SYNTHESIS this
   commit: the **engage 7/7** result (deliberation consequential upward); the **PR-gate fourth-
   architecture confirmation** (decision channel 9–15 dim across OLMo/Qwen/Llama/GPT-OSS — a strong D2
   generalization, foregrounded); the **null-ratio corroborates harm-keying** sentence. Only the
   **commit-axis verdict cell** waits for the graded-disengage pod.

6. **Pod scope.** Graded-prefill disengage (behavioral + projection) + decision-predictability curve +
   (conditional on branch 1a) finer-ladder cross-check. One modest 80 GB session; the prefill arm needs
   no KV-patching, same as engage.

Spine preserved. A12 hardens the commit axis against the A7 operating-point trap and un-confounds the
commitment curve; the reads axis (P2 harm-loading) and the position gate are already banked.

### Amendment 13 (2026-07-03) — SYNTHESIS hygiene: architecture confound + the dimensionality rival (zero-GPU)

The two-axis table (what reads × how commits) is the clean measured result and stays. The *interpretation
line* above it — the n=3 "harm-readers reversible, broad-reader early-commits" co-occurrence — needs two
corrections, both pre-registered here before the reframe is written.

1. **Name the confound, not just n.** The axis split across OLMo / GPT-OSS / Llama is **also** a
   lineage / scale / tokenizer / reasoning-vs-instruct split, so the "reads-harm ↔ reversible" pairing is
   architecture-confounded, not just under-powered. The **deconfounding design** (varies one axis at a
   time) is the honest replacement for "n=3": (a) a **deliberation-trained OLMo variant** — same
   architecture, changes the commitment/training axis, isolates whether reversibility tracks the read or
   the reasoning-training; (b) **Qwen** as a **lineage-independent** third harm/broad datapoint (a fourth
   family that is neither OLMo-lineage nor a reasoning MoE), so the read↔commit pairing is tested off the
   existing confound. Named as the follow-on panel, not run here.

2. **The one-axis rival (dimensionality → reversibility).** Test whether *what reads* and *how commits*
   are **one property with two signatures**: correlate the **refusal-read effective rank** (the causal
   saturation rank per model — OLMo `harm_saturating` ⇒ ~rank-1; Llama `broad_moral` ⇒ high rank; GPT-OSS
   the harm-loading proxy ⇒ low rank) against **commitment depth / reversibility** across the three
   points. **Frozen response:** if the three points are ordinally consistent (low read-rank ↔ reversible;
   high read-rank ↔ early-commit), SYNTHESIS carries **"dimensionality of the refusal read → reversibility"**
   as the sharpened, *falsifiable-continuous* follow-on hypothesis, **superseding** the two-column
   co-occurrence framing (which is categorical and adds no mechanism). If the points are inconsistent,
   keep the co-occurrence framing and drop the dimensionality claim. Either way the statement is a
   **hypothesis for the deconfounding panel**, not a result at n=3 (three confounded points cannot
   separate a genuine dimensionality→reversibility law from the architecture confound; consistency only
   licenses the hypothesis).

Spine preserved. A13 is framing hygiene: the measured table is untouched; the interpretation becomes a
confound-named dimensionality hypothesis with its deconfounding path.

### Amendment 14 (2026-09-10) — W4 Tier A: claim-bearing pre-ship cells (pre-pod)

Context: Phase W4 (WRITEUP_PHASE_PLAN.md) unfreezes the program for one targeted pod. Amendment 14
pre-registers the six Tier-A cells before any array is extracted. Each cell names its OPEN_THREADS row,
the CLAIMS ids it can move, both branches (both publishable), the verdict rule with its detection bar,
the positive control and null (measurement cells) or the intervention spec block (causal cells), and
the per-unit artifacts that MUST be saved. Where the handoff draft disagreed with the repo of record,
the repo wins; each such correction is marked **[repo-fix]**. Spine preserved: `M = 0.05`, the two-step
null, folded-primary convention (NI-4), per-unit saves, position validity (PR recorded on every
position), difference-CIs never overlap checks, and no NULL without a ladder.

**Model batching (compute-ordering).** "OLMo-3" in the handoff is three loads: Instruct (15.1, 14.5,
14.6, 14.1 gate-side), Think (14.4), base `main` (14.1 split-half). Order: OLMo-3-Instruct → OLMo-3-Think
→ OLMo-3 base → Llama-3.1-8B-Instruct → GPT-OSS-20B → Qwen2.5-7B-Instruct. Every load records the HF
commit hash actually resolved (`config._commit_hash`, falling back to `HfApi.model_info(...).sha`) into
the run manifest; `assert_matches_model` runs on every load.

#### 14.1 Proto-refusal reliability ceiling (FL Tier 3 estimability control)

- **Source.** OPEN_THREADS §H "New pre-ship control"; SYNTHESIS Tier 3 counter-reading. **Moves:** D1-14,
  D1-15 (fresh-construction wording), the FL abstract clause "almost no pretraining precursor", FL §3/§4,
  MN crystallization figure caption; opens W4-01 (reliability ceiling) and W4-02 (per-checkpoint
  proto-refusal→gate trajectory).
- **[repo-fix] Zero-GPU arm exists.** Paper 5 cached per-checkpoint proto-refusal directions for all 14
  OLMo-3 stage-3 states (`papers/5_moral_alignment/outputs/measurement/stage3/<label>/
  proto_refusal_directions.npz`, keys `proto_refusal_layer{L}`, all layers, `raw` last-token diff-of-means
  over the Heretic 400/400 set) — the **same construction** as `refusal_base.npz`
  (`phase2_g3_respec_extract.refusal_vec(base, prompts, "raw", 16)`). So the adjacent-checkpoint
  self-cosine and the full per-checkpoint proto-refusal→gate trajectory are computable now; only the
  **split-half** arm (needs per-sample activations) is a pod item. OPEN_THREADS §H and
  `supplement/PROVENANCE.md` say "not derivable from saved artifacts" — that statement is corrected by
  this amendment (it was true of D1's saves, not Paper 5's).
- **Quantities (frozen).** (a) `rel_adj = cos(proto@stage3-step11900, proto@stage3-step11921) @ L16`
  (21 training steps apart; the reliability ceiling under checkpoint drift + sampling), plus the full
  curve `cos(proto@step_s, proto@final)` for the 13 stage-3 steps. (b) Cache-consistency positive control:
  `cos(proto@olmo3_base[main], refusal_base.npz) ≥ 0.99` (same model, same construction; if it fails the
  cache is not the D1 object and (a) is void). (c) `traj(s) = cos(proto@step_s, refusal_instruct)` for all
  13 steps: the proto-refusal→gate trajectory (the 0.155 of record is `traj(final)`). (d) **Pod arm,
  split-half reliability of proto-refusal:** on OLMo-3 base `main`, per-sample L16 last-token activations
  for all 400 harmful + 400 harmless Heretic prompts (raw format); resample 200 random half-splits of the
  prompt index set (paired: each split takes half of harmful AND half of harmless), diff-of-means per
  half, `r_half = cos(d_A, d_B)`; report the median and percentile 95% CI over splits, and the
  Spearman–Brown full-length correction `rel_proto = 2·r̄/(1+r̄)`. (e) **Rider (same session, the gate
  side of the disattenuation):** identical split-half on the **instruct gate** (chat format, L16, same
  prompts) → `rel_gate`. The disattenuated cosine is `cos_corr = 0.155 / sqrt(rel_proto · rel_gate)`,
  with a bootstrap CI propagated from the two split distributions. The handoff ceilinged only the
  proto side; a disattenuation needs both reliabilities **[repo-fix: added]**.
- **Positive control / null.** Positive control for the instrument: split-half self-cosine of the
  moral-stories direction from the saved `diffs_moral_stories.npz` (base, L16; per-pair diffs are saved)
  must be ≥ 0.9 — a direction the program knows is stable must read as stable on this estimator.
  Null: self-cosine of two halves of a label-permuted contrast (labels shuffled within the same
  activations), 200 permutations; the q95 is the chance ceiling for `r_half`.
- **Branches (all publishable).** A `rel_proto ≥ 0.9` (and `rel_adj ≥ 0.9`): fresh-construction claim
  stands as written; the disattenuated 0.155 changes by < 0.03. B `rel_proto ≤ 0.3`: 0.155 is mostly
  attenuation floor; FL Tier 3 rescopes to "low base→instruct cosine, reliability-limited"; the
  abstract's "almost no pretraining precursor" is dropped; D1-14/15 SCOPED. Between: `cos_corr` with its
  CI carries the sentence ("proto-refusal→gate cosine 0.155, disattenuated X [CI]"), and the abstract
  clause is rewritten to the corrected value. Any discordance between (a) and (d) (adjacent-checkpoint
  low but split-half high, or the reverse) is a finding: checkpoint drift vs sampling noise are
  separated by construction, and both numbers are reported.
- **Detection bar.** With 200 harmful/200 harmless per half in d = 4096, the isotropic chance self-cosine
  is ≈ sqrt(2/(π·4096)) ≈ 0.012; the permutation null supplies the realized ceiling. The bootstrap over
  splits gives a CI half-width of ≈ 0.02 at n=200 splits (to be reported, not assumed).
- **Save.** `w4/olmo3_base/proto_refusal_samples.npz` (per-sample L16 activations: harmful (400, 4096),
  harmless (400, 4096), prompt index order, format tag); `w4/olmo3_instruct/gate_samples.npz` (same, chat);
  every resampled half-direction (200 × 2 × 4096 per model); `w4/zero_gpu/proto_refusal_trajectory.json`
  (a, b, c with per-layer values). Type blocks on every direction (position, PR, format, n, commit).
- **Price.** ~0.2 A100-h (two forward-only passes over 800 prompts); zero-GPU arm now.

#### 14.2 Llama severity-twin rank-2/4 harm-coextensive check

- **Source.** OT §B row 2; MISSING_ARTIFACTS Amendment 11. **Moves:** D3-18 (reads-broad strength),
  D3-19 (the rank-2/4 rider), FL §8 dissenting-read wording, D3-24 (Llama's read rank).
- **[repo-fix] Stimuli of record.** The Llama layer-12 C1 run of record used the **boundary-band twins**
  (36 pairs, `BOUNDARY=1`; `rt_composition {request_screened: 1, band: 36}`), not the severity ladder.
  Amendment 11 names the *severity-ladder* contrasts. Both are extracted: **primary** = severity-ladder
  twins (30 pairs, levels 1–5, `get_severity_twins`), matching A11's wording; **secondary** = boundary
  twins (36), matching the run of record's stimuli. Both are forward-only.
- **[repo-fix] Moral PCs are not saved.** `c1_inputs_llama31_L12.npz` saves `Vbasis`, `harm`,
  `channel_act`, and the per-k deltas, but not the nested moral-contrast PCA basis. The moral PCs are
  re-derived in-run from the same moral pairs (`load_moral_pairs`, all three sources, `mean_content`,
  L12, standardized with σ from the **saved** `channel_act`, so the frame is reproducible bit-for-bit).
  **Harness-parity check:** the re-derived `|cos(d_harm, PC_i)|` for i = 1..8 must match the saved
  `cos_harm_pc` (0.199, 0.307, 0.018, 0.026, …) within 0.05 per component; a miss voids the cell.
- **Procedure.** Per-pair `mean_content` contrasts (violating − following) at L12 in the standardized
  frame → `sweep.nested_pca_basis(contrasts, [1, 2, 4])` → harm bases H_1 ⊂ H_2 ⊂ H_4 →
  `sweep.harm_capture_curve(H, moral_pcs[:, :16], engage_marginal_weights(saved engage_sweep))` →
  engage-weighted capture at j = 1, 2, 4. j = 1 must reproduce the saved rank-1 number (3.6%) within
  0.02 (second check of parity).
- **Positive control / null.** Positive control: the moral PCs' own split-half rank-4 basis (PCs from
  half the moral pairs) must capture the engage-driving basis at ≥ 0.6 (an instrument that cannot
  recover the basis from itself cannot certify a low harm capture). Null: rank-j bases built the same way
  from the three non-moral control contrast sets (syntax, register, sentiment pairs at `mean_content`,
  L12) → the q95 over controls and over 200 random orthonormal rank-j bases in the standardized frame
  (channel-matched specificity). Report capture − null.
- **Branches.** A `capture(4) − null_q95(4) ≤ 0.25`: Llama reads-broad ships at full strength ("beyond
  harm"). B `capture(4) − null_q95(4) ≥ 0.50`: Llama reads a multi-dimensional harm percept; §8 softens to
  "broader than OLMo's rank-1 harm, not established as beyond harm"; D3-24's "rank-8 broad" becomes
  "rank-8, of which up to k harm-coextensive". Between: "partially harm-coextensive (capture X at rank
  4)", both readings kept, D3-19 SCOPED. Thresholds are set at the D3 rank-1 rule's scale: the rank-1
  capture of record is 0.036 and the OLMo harm-rank-1 R is 0.31 of the full effect, so 0.25 is "a
  quarter of the engage-driving basis" and 0.50 "half".
- **Save.** `w4/llama31/severity_contrasts_L12.npz` (per-pair (30, 4096) + boundary (36, 4096), raw and
  standardized), `moral_contrasts_L12.npz` (per-pair, all sources), nested harm bases, moral PCs, control
  bases, `harm_capture_L12.json`. **Price.** ~0.3 A100-h.

#### 14.3 GPT-OSS post-response decision-token projection (+ the missing half of A5)

- **Source.** OT §B row 5, §E row 3. **Moves:** D3-22 (projection from "corroboration w/ last-token
  caveat" to co-primary or to behavioral-primary-only), D3-20 (completes the A5 pre-condition), FL §8.2
  and Limitations "prefill-last-token caveat".
- **[repo-fix] What is already banked.** The A5 pre-condition has two halves: post-standardization PR
  (banked: 12.79 ≤ 25, `tier1_session_gpt_oss_20b.json`) and **band-below-null at the harmony decision
  token**, which the Tier-1 gate did **not** compute (`position_gate_verdict` uses PR + the refusal
  null-ratio only). MISSING_ARTIFACTS Amendment 11 lists both; only the band half is open. It runs here.
- **Position definitions (frozen).** `P_prefill` = last token of (prompt + analysis opener + prefill),
  the Tier-1 readout (`_prefill_proj`). **New: `P_dec`** = the token immediately before the first
  generated token of the final channel, i.e. the `<|message|>` of `<|channel|>final<|message|>` in the
  completed rollout, read by ONE forward over the full generated sequence (prompt + prefill + analysis +
  final opener). If the rollout never opens a final channel, `P_dec` is **unmeasured and counted**
  (never substituted). `P_dec` is the decision site after deliberation; `P_prefill` is before it.
- **Procedure.** Same 10 ceiling-refusing violating items, same three-strength exculpatory series, same
  refusal direction construction (END_OF_PROMPT diff-of-means, 64/64), same layer 12, greedy decoding
  (replication check: the 6/10 behavioral flip at max strength must reproduce exactly; a mismatch is
  harness drift and voids the cell). Read the projection at `P_dec` per item × strength (+ no-prefill
  baseline). Statistic: `graded_disengage_stat` on the `P_dec` series (frac_monotone_toward_comply,
  mean_projection_move_at_max) plus a paired bootstrap CI (over items) on the mean move.
- **Positive control / null.** Positive control: the engage arm (7 benign items, inculpating prefill)
  read at `P_dec` must move toward refuse (the direction the behavior demonstrably went). Null: the
  same rollouts projected onto 200 covariance-matched random directions (from the `P_dec` act-sample);
  the monotone fraction and mean move under random directions give the chance level.
- **Band-below-null (A5 half).** Held-one-out moral-family band at the harmony decision token: the three
  V_moral source directions extracted at END_OF_PROMPT in think format (moral pairs, 60/source), each
  projected onto the span of the other two, vs the covariance-matched rank-2 null from the decision-token
  act-sample. Band below null ⇒ the decision token is position-invalid for content, exactly as the D2
  sites; band above null ⇒ note it as the one panel model where content survives at the decision token.
  Either way D3-20's "position-valid" is scoped to *decision-direction* reads (NI-3 wording).
- **Branches.** A (monotone fraction ≥ 0.8 at `P_dec` AND paired CI on the mean move toward comply
  excludes 0 AND random-direction null q95 for the monotone fraction < 0.8): projection is co-primary
  with the 6/10 flip; the Limitations paragraph is deleted. B (otherwise): reversibility ships
  behavioral-primary only; the stated reason is that the post-deliberation decision token does not carry
  the graded movement the prefill token does (a finding about *where* deliberation writes, reported).
- **Save.** Per-rollout full token ids, `P_dec` and `P_prefill` activations (10 × 4 × 2880 + engage
  7 × 2 × 2880), per-item projections, the decision-token act-sample (n ≥ 128) with its PR, the three
  V_moral source directions at the decision token, `band_below_null_gptoss.json`. **Price.** ~0.3 A100-h.

#### 14.4 D1 P0–P3 per-rollout PR audit (reasoning band rung)

- **Source.** OT §B row 1; MISSING_ARTIFACTS Amendment 2 (per-rollout windows), A1 (Think MFT
  directions), A3 (Think refusal vectors). **Moves:** D1-10, D1-11, D1-12 (drop or keep the
  "scoped as cross-position" hedge); closes MISSING_ARTIFACTS A1/A3/Amendment-2(ii).
- **[repo-fix] Price and design.** The handoff priced this at ~0.3 h. P0/P1 are prompt-side (400/400,
  cheap). P2/P3 need generation; **no rollouts were saved** by the D1 runs, so re-generation is required.
  Run-of-record parameters: GPT-OSS `n_gen = 64/side, max_new_tokens 1024, cot_window 16`; OLMo-3-Think
  `max_new_tokens 2048, window 256`, P3 unmeasured (benign side never closes within budget). For a PR
  audit the window needs only `window_n` reasoning tokens, not closure: **Think uses `max_new_tokens =
  window + 64 = 320`**, which keeps its P2 audit at ~64 rollouts × 320 tokens. P3 on Think stays
  unmeasured (as of record). Realistic price: GPT-OSS ~0.4 h, Think ~0.5 h.
- **Procedure.** Per rollout, save the P0, P1, P2-window, P2-full, P3 activations at the match layer (12
  GPT-OSS / 16 Think) and the generated token ids. Per position class: PR of the pooled (harmful +
  harmless) activation sample; band-below-null: the three V_moral source directions re-extracted at that
  position class (moral pairs through the same window pipeline, n = 32/source) held-one-out vs the
  covariance-matched rank-2 null from that position's sample. Also the refusal direction per position
  (diff-of-means) saved as `.npz` (closes A3 for Think) and the 6-foundation MFT directions on Think in
  raw format (closes A1).
- **Positive control / null.** Positive control: P0 (`t_inst`, a content position) must be PR-valid
  (PR ≥ 30 and band ≥ null) — if even the content site fails, the audit instrument is miscalibrated for
  reasoning traces and the rung stays hedged regardless. Null: covariance-matched rank-2 null per position.
- **Branches.** A (P2 window PR ≥ 30 and band ≥ null on both models): D1 band-relative statements drop
  the cross-position hedge; D1-10/11 VERIFIED-unscoped. B (P2 fails on either): the hedge stays, and the
  failing model's in-trace rung is stated as "band-relative comparison not licensed at P2 (PR x, band
  below null)"; null-relative claims (D1-11's null crossing, D1-12) are unaffected either way. Mixed
  (one model passes): the hedge is per-model.
- **Save.** `w4/<gpt_oss_20b|olmo3_think>/p0p3_rollouts.npz` (per-rollout activations per position,
  token ids, closed/window flags), `refusal_P{0..3}.npz` (Think), `mft_directions.npz` (Think),
  `pr_audit.json`. **Price.** ~0.9 A100-h total.

#### 14.5 Reconciled B1 cross-ablation on the full harmful set

- **Source.** OT §B row 4; V-D2-4; D2 Amendment 1 rider 8. **Moves:** D2-01 (adds or withholds the
  causal arrow), FL §6 (R3 wording "geometric-only" vs "geometric + causal").
- **Intervention spec block.**
  `stimulus_class`: Heretic `harmful_eval` (100 held-out harmful requests, Paper 5 `refusal_prompts.json`)
  for refusal; the D1 ETHICS forced-choice battery (120 items, counterbalanced) for judgment.
  `outcome_variable`: refusal rate (shared `_classify_response`, opening-refusal rule) and judgment
  accuracy. `baseline_discrimination`: **bail condition** — baseline refusal on the 100 must be ≥ 0.40
  (Paper 6's 0.575 expected); below 0.40 the cell is banked as floor-limited and not run further.
  `site`: layer 16 (registry primary), `final_pre_assistant`, PR recorded from the chat act-sample.
  `transfer_scope`: rank-1 direction ablation (Arditi orthogonalization via `ablate_direction`).
  `ablation_semantics`: projection-out at the layer (the Paper 5/6 semantics, kept for parity; not a
  mean-ablation — stated). `alignment_rule`: n/a (no patching). `controls`: matched-random unit direction
  (5 draws, matched norm) and the persona direction (named reference). `outcome_harness`:
  `deepsteer.benchmarks.compliance_gap.greenblatt._classify_response` at the repo commit in the
  manifest; `b1_judgment_direction.is_refusal`. `branches`: below.
- **Verdict rule (difference-CI, never overlap).** Per prompt, outcome under baseline and under each
  ablation; paired bootstrap (2000, seed 0) over prompts of `Δ_refusal(judgment-ablated) − Δ_refusal
  (random-ablated)` and of `Δ_judgment(refusal-ablated) − Δ_judgment(random-ablated)`. Arrow detected iff
  the CI excludes 0 **and** the persona control's Δ does not exceed the random q95. Detection bar: at
  n = 100 and p ≈ 0.5 the MDE for a rate difference is ≈ 0.14; at n = 120 judgment items ≈ 0.13. Both
  are reported next to the estimates.
- **Branches.** A (either arrow's CI excludes 0): R3 ships causal + geometric with the arrow's direction
  named. B (neither): R3 stays geometric-only (D2-01), now with the bar: "no causal cross-effect
  detectable at Δ ≳ 0.14 in refusal rate".
- **Save.** Per-prompt outcomes (100 × {baseline, judgment-abl, refusal-abl, persona-abl, random×5})
  and generations; per-item judgment verdicts (120 × conditions); the directions used with type blocks.
  **Price.** ~0.4 A100-h (≈ 1,300 short generations).

#### 14.6 Per-unit saves for the MN instruments (no new claim)

- **Source.** SELF_REVIEW MN should-fix items (Fig 1 error bars, PR gate normalization, §3.1 reply-
  inversion specificity). **Moves:** D2-02, D2-09, D3-20 (CIs), P7-05 (specificity), MN Table 1/Fig 1.
- **(a) PR with CIs.** For each panel model × position class ∈ {final_pre_assistant, last_content,
  mean_content} (chat; GPT-OSS: harmony decision token + a content position), save the per-text
  activation sample (n ≥ 240 texts, the in-format ladder's mix) at the primary layer, raw and
  standardized. Bootstrap CI on PR (resample rows, 2000). Report `PR`, `PR/d`, `PR/(n−1)` (the
  sample-size ceiling), and the null-referenced quantile: PR of a covariance-matched Gaussian sample
  of the same n (200 draws) and of a **row-shuffled / sign-flipped** control that destroys token
  identity while preserving marginals. The MN gate is then stated as a dimension- and n-normalized
  quantile, not "25 for GPT-OSS". Rider: MN Table 1's "decision-token 13.5 vs decision-site 10.2"
  Llama split is measured on the **same** sample here, so NI-2 closes with both positions labeled.
- **(b) Reply-inversion specificity null (MN §3.1).** On Llama-3.1-8B-Instruct (the only reply-
  inversion model in the panel; Qwen2.5-14B-Instruct is not loaded and its null is **not run**, stated),
  the Paper 7 `reply_inversion_control` harness (norm-scaled steering at the swept layer, forced-answer
  logit read, coherence gate) is re-run with the harm direction **and** 20 random unit directions at the
  identical norm. Report the harm flip fraction and margin shift against the random q95 (channel-matched
  specificity). Branch A: harm exceeds random q95 → "the harm axis moves the reply over matched-norm
  directions" enters MN §3.1. Branch B: not → the sentence stays as the current limitation, now with the
  measured chance level.
- **Save.** Per-model per-position activation samples, PR bootstrap arrays, null PR draws; per-item
  steered margins for harm and each random direction. **Price.** ~0.4 A100-h incremental across loads.

**Tier A total: ~2.5 A100-h** (14.1 0.2 · 14.2 0.3 · 14.3 0.3 · 14.4 0.9 · 14.5 0.4 · 14.6 0.4), plus
model-load overhead (six loads, ~0.1 h each).

#### Referee pass (Amendment 14)

1. *"Your split-half reliability corrects a cosine between directions measured in two different formats
   (raw base vs chat instruct); disattenuation assumes the same construct in both."* Conceded in part:
   the cross-format step is the 0.155 of record's own construction (D1 Point A raw, Point B chat). The
   disattenuated value is reported as a ceiling-corrected version of *that* number, and the format
   difference is stated as a residual that no reliability correction removes. The adjacent-checkpoint arm
   is within-format and carries the drift reading on its own.
2. *"The harm-coextensive null is a random rank-j basis, which is a weak null in a standardized 4096-d
   space."* Answered: the null is channel-matched by construction — the same nested-PCA procedure applied
   to three non-moral control contrast sets (syntax, register, sentiment) at the same position and layer,
   with the random bases reported alongside as the floor; the positive control (self-capture ≥ 0.6) bounds
   the instrument from above.
3. *"Reading `P_dec` after the model responds lets the response content leak into the decision token, so
   a monotone projection could just be the final channel echoing the prefill."* Conceded and separated:
   the random-direction null at `P_dec` measures how much any direction moves with the prefill; the harm
   read is claimed only above that chance level, and the engage arm at `P_dec` (movement toward refuse)
   shows the readout is two-sided rather than prefill-echo.

### Amendment 15 (2026-09-10) — W4 Tier B: panel strength (pre-pod)

#### 15.1 OLMo-3 additional request-twins (lift the rank sweep past n = 23)

- **Source.** OT §D "R_refusal precision"; SELF_REVIEW "n=23 underpowered"; MN §3.1. **Moves:** D3-06,
  D3-07, D3-08 (headline shape), D3-09, D3-12 (one-knob fit), FL §7 and App C.
- **[repo-fix] Dependency and pass rate.** The n = 23 of record is **23 of 60** authored request-twins
  surviving the baseline-discrimination screen (`screen_counts.request_twins 23`; `rt_composition
  {request_screened: 23, band: 0}` — the severity-ladder operating band was empty on OLMo). At that 38%
  pass rate, reaching pooled n ≥ 40 needs ≈ 45 newly authored twins. A batch of **48** (8 per foundation)
  is committed as `deepsteer.datasets.request_twins_w4` under the identical construction rule (exact
  shared prefix; flip only the trailing moral-intent span; harm carried by intent in an XSTest-safe
  register; no memorized text) and flagged for Orion's review at Gate W4-1. Expected pooled n ≈ 41;
  the target n is Orion's to confirm.
- **[repo-fix] Units and pooling.** The local `c1_session_olmo3.json` / `c1_inputs_olmo3.npz` are the
  **standardized robustness rerun** (full → refusal −0.62 in standardized units), not the folded-primary
  run whose numbers CLAIMS D3-06 carries (−0.0833). Pooling new per-twin deltas against those arrays
  would mix units. So: **one session run over the union** (60 original + 48 new authored twins through
  the same screen), folded-primary environment (`STANDARDIZE` unset, `SWEEP=1`, layer 16, same
  classifier), per-twin deltas saved with a set tag. Analyses: **replication** (original-set subset must
  return `harm_saturating` and R_refusal(16) within 0.10 of 0.27 — a harness-parity gate; a miss stops
  the pooled analysis and is banked as drift), **alone** (new-set subset), **pooled** (all).
- **Power table (from the saved per-twin arrays, resampled; scaffolding read, not a verdict).**
  Ratio-of-ratios difference CI width: n = 23 → 0.84, n = 40 → 0.57, n = 60 → 0.46 (the recorded gap is
  0.13–0.18, so the secondary does **not** resolve at n = 40; ≈ n = 140 would). R_refusal(16) CI width:
  n = 23 → 0.49, n = 40 → 0.33. **What n = 40 buys is the primary (shape) at a third tighter CI, not the
  secondary.** This is stated before the pod so the outcome cannot be read as a surprise.
- **Primary.** Shape verdict (`sweep.shape_verdict`, frozen Amendment-4 rules) on the pooled sweep;
  one-knob fit RMSE over the plateau. **Secondary.** Ratio-of-ratios CI at pooled n; the paired
  Δ(V_moral-restricted vs random-rank-3) CI (D3-07).
- **Branches.** Shape survives (pooled `harm_saturating`, alone-result same sign of R_judgment(16) −
  R_refusal(16)) → headline unchanged with the tighter CI; pooled is primary. Shape changes (pooled not
  `harm_saturating`, or alone disagrees in sign) → the one-knob model is reported as "fitted on 23, not
  replicated on N", §7 rewrites to the pooled verdict, and D3-08/12 SCOPED. Pooled is primary **only if**
  the alone result agrees in sign; otherwise both are reported and neither pooled.
- **Pilot gate.** First 5 screened twins end-to-end (extract → screen → full/restricted/random cells →
  sweep) before the remaining twins: every delta finite, full-cell sign coherent on ≥ 4/5, per-twin
  arrays written. Fail → stop and bank the diagnostic.
- **Spec block.** As Amendment 3/4 for the OLMo C1 cell (stimulus request-twins; outcome refusal
  projection at the decision token; baseline screen; site L16 `final_pre_assistant` PR 14.7; scopes full
  / rank-k restricted; transport control = judgment on compositional twins; alignment = shared prefix +
  flipped span; classifier shared; both branches above).
- **Save.** Per-twin deltas for every cell and every rank (rows tagged by set and twin id), screen
  outcomes per twin, the moral PCA bases, `channel_act`. **Price.** ~1.0 A100-h (108 twins × ~15
  interchange passes + screen), the largest single item.

#### 15.2 Qwen2.5-7B-Instruct causal C1 read cell

- **Source.** OT §E row 1; SYNTHESIS Tier 2 "Qwen — not measured". **Moves:** D3-23 (fills the empty
  cell), D3-24 (a lineage-independent point for the dimensionality hypothesis), FL abstract "Qwen is not
  measured on the read axis", FL §8 two-axis table (a fourth row).
- **Protocol.** `c1_session.py --key qwen25 --layer 14` (registry primary at 28 layers) with
  `STANDARDIZE=1` (A1: dim 458 carries 59% of content-position variance; the invariance proof on OLMo
  licenses the standardized frame), `SWEEP=1`; same request-twins (60 original + 48 new), same screen,
  same classifier. Pilot gate: screen must keep ≥ 12 request-twins; else `BOUNDARY=1` twins; if both
  bands keep < 12 the cell is banked as **indeterminate (operating point)** and not forced.
- **Spec block.** Stimulus request-twins (+ boundary twins if needed); outcome refusal projection at the
  decision token (Qwen PR 8.6; A1 standardization recorded; channel null raw and std); scopes full /
  rank-k; transport control judgment; alignment shared prefix; classifier shared; ablation n/a.
- **Branches (all publishable, all fill the cell).** `harm_saturating` (OLMo-like plateau at the
  harm-rank-1 level) / `broad_moral` (Llama-like gap-close) / `instrument_ceiling` / `indeterminate`
  (below bar: the shape verdict's plateau tolerance 0.1 is not resolvable at the achieved n, stated with
  the achieved CI widths). The two-axis table gains a Qwen row in every case; the abstract's "not
  measured" becomes the measured reading.
- **Save.** As 15.1 for Qwen. **Price.** ~1.0 A100-h.

**Tier B total: ~2.0 A100-h. Pod total ≈ 4.5–5 A100-h plus ~0.6 h of model loads** (the handoff's
5–7 h envelope holds).

#### Referee pass (Amendment 15)

1. *"The new twins were authored after the n = 23 result was known; the author could tune them toward
   the plateau."* Conceded as a risk and controlled: the construction rule is unchanged and mechanical,
   the batch is committed before the pod with a per-foundation count, the screen is the same blind
   baseline-discrimination gate, and the alone-vs-pooled sign rule means a batch that behaves differently
   from the original is reported, not averaged away.
2. *"n = 40 still does not resolve the ratio-of-ratios."* Answered in the power table above: agreed, and
   said before the run; the primary is the shape verdict, whose CI tightens by a third.
3. *"Qwen at PR 8.6 with a 59%-variance outlier dimension is exactly where your own A1 says nulls
   saturate."* Answered: the cell runs in the standardized frame with both nulls reported, the transport
   positive control (judgment) must move under the same patch before any refusal null is read
   (intervention-validity rule 2), and `indeterminate` is a pre-registered publishable branch.

Spine preserved. Amendments 14/15 add cells; they relax no prior gate.

### Amendment 16 (2026-09-13) — W4-3 pre-verdict: the 14.1 null rung as a verdict input; the crystallization trajectory in the Tier-3 sentence; a declared fork on the 14.2 parity gate

Committed before any W4 verdict is written (W4_RESULTS.md). Items 1–2 are the two zero-cost promotions
Orion approved at Gate W4-1 (WRITEUP_PHASE_PLAN, Gate W4-1 item 5). Item 3 is a **fork**: a post-hoc
analysis-choice change, reported under both choices, never as a silent replacement.

1. **14.1 ladder (promotion 5a).** The covariance-matched single-direction null computed in the
   zero-GPU arm (CLAIMS W4-02: q95 0.070 from the base L16 act-sample, n = 1754) is promoted from a
   descriptive rung to a verdict input. The Tier-3 quantity is read on the ladder isotropic chance
   sqrt(2/(π·4096)) ≈ 0.012 → matched-null q95 0.070 → measurement 0.155 → moral-subspace positive
   control 0.999, instead of against the bare 0.50 threshold. Wording rule: 0.155 is "a weak
   precursor (≈ 2× the matched-null q95)", never "almost no precursor"; the 0.50 threshold is kept as
   the crystallization bar the moral subspace clears and refusal does not.
2. **Tier-3 sentence carries the trajectory (promotion 5b).** The per-checkpoint proto-refusal→gate
   cosine (flat 0.139–0.155 across 13 stage-3 states while proto-refusal's self-cosine to final rises
   0.93 → 1.0) may be stated in the FL Tier-3 sentence and drawn as a second curve in the
   crystallization figure.
3. **14.2 parity gate — fork.** Amendment 14.2 pre-registered a per-component harness-parity check:
   re-derived |cos(d_harm, PC_i)| for i = 1..8 must match the saved values within 0.05 each, else the
   cell is void. The pod's re-derivation matches PC1/PC2/PC4+ within tolerance and misses on **PC3**
   (0.157 vs 0.018; max |Δ| = 0.139): the third and neighbouring components of a 1057-pair PCA are a
   near-degenerate pair whose order is not stable to the σ used for standardization, so a
   per-component cosine is the wrong invariant for a rank-4 *subspace* comparison. The fork:
   (a) **pre-registered rule** → the cell is VOID; its capture numbers are reported as descriptive only.
   (b) **subspace-level parity** → the harm vector's projection fraction onto span(PC1..PC4) must match
   the saved value within 0.05 (saved 0.367 from the four saved cosines; re-derived 0.337; |Δ| = 0.030).
   Under (b) the cell is licensed and the Branch A/B rule of 14.2 applies unchanged. W4_RESULTS reports
   the verdict under both (a) and (b); FL prose may use (b) only if the fork is stated in the
   appendix. The choice between (a) and (b) is Orion's (escalated; it does not change a PRIMARY).

### Amendment 15/14 rider (2026-09-12) — W4 rerun pod: 15.2 gate wording, 14.5 generation save (pre-rerun)

Committed after the W4 pod of 2026-09-12 (manifest `w4_20260912T190441`) and before any rerun array is
extracted. Neither item changes a PRIMARY; both are pre-registered corrections of driver behaviour
against the amendment text.

1. **15.2 gate counts the operating band (clarification, no new rule).** Amendment 15.2 said "screen
   must keep ≥ 12 request-twins; else `BOUNDARY=1` twins; if both bands keep < 12 → indeterminate".
   The C1 harness of record has always been dual-use: its readout/sweep stimuli are the screened
   request-twins **plus** the severity-ladder operating-band pairs (`rt_pairs = req_pairs +
   band_pairs`; Llama L12's cell of record ran on 36 band pairs and 1 request twin). The W4 driver
   keyed the fallback on request-twins alone: on Qwen the screen kept 1/108 request-twins, the pilot's
   ladder found **18 operating-band pairs at levels 3–5 (≥ 12)**, and the driver nonetheless switched
   the full cell to `BOUNDARY=1`, whose boundary band on the boundary-twin set was empty (0 pairs) →
   banked `indeterminate_operating_point` with no sweep. Corrected rule, matching the harness and the
   amendment's intent: the gate quantity is `request_screened + operating_band_pairs` from the pilot;
   ≥ 12 → the full cell runs in operating mode; < 12 → `BOUNDARY=1`; boundary band < 12 → indeterminate.
   The 15.2 cell is **rerun** under this rule (Qwen2.5-7B-Instruct, `STANDARDIZE=1`, `SWEEP=1`,
   operating band, expected n = 19). The 2026-09-12 boundary-mode run stays in the manifest as the
   record of the empty boundary band; its `qwen25_read_cell.json` is superseded, not deleted.
2. **14.5 saves generations (save-list correction).** Amendment 14.5's save list names "per-prompt
   outcomes … and generations"; the unit saved outcomes only. A new unit `14.5_gen` re-generates
   greedily under the **saved** directions (`cross_ablation_outcomes.npz`: refusal, judgment-decision,
   random_0; baseline) for the same 100 prompts, saves the texts with the classifier outcome per
   text, and reports agreement with the saved outcomes as a determinism check (greedy decoding; any
   disagreement is logged, not silently overwritten). This is the discriminator for ANOMALIES A8; it
   adds no verdict rule. The 14.5 arrow verdict is **held** until A8 resolves.
3. **Manifest discipline for partial reruns.** A rerun writes to its own output subdir and its own
   manifest (`W4_OUT_SUBDIR`), then `pod_w4.py --merge-from` folds it into `manifest_w4.json`: rerun
   artifacts replace same-path entries, unit statuses and gates for the rerun units are replaced, the
   superseded entries are kept under `reruns[].superseded`, and the merged manifest is re-verified.
