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
