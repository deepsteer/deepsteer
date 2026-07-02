# Direction 3 — Anatomy of the Refusal Decision (the "C1" experiment)

**Provisional.** This is the C1 causal follow-on promoted from `papers/d2_decision_coupling/`
(Phase C). Numbering is the program owner's call.

**One line.** D2 showed the refusal decision and the moral-judgment decision are orthogonal *and*
that the decision site is a **~10–15-dimensional control-token bottleneck** (participation ratio
14.7 / 8.6 / 10.2 on OLMo-3 / Qwen2.5 / Llama-3.1, all position-invalid) that moral content does not
reach; so content-vs-decision orthogonality is **architecturally guaranteed** and any
comprehension→decision coupling must be carried by the **attention heads that write into the
bottleneck**. This paper finds those heads and asks whether they **transport moral content** into
the decision channel.

This plan is written under three program disciplines, each with a ship-blocker checklist that gates
the write-up: `construct-audit` (type every direction; measure the diagonal cells; a causal cell is
required before any "reads-from" claim), `instrument-calibration` (no null/mediation verdict without
a positive control + ladder + MDE on *this* instrument), `compute-ordering` (zero-GPU first; rank
GPU items by ΔDecision/GPU-hour; per-unit saves).

---

## 1. Contribution positioning (write AFTER the zero-GPU lit pass, §7)

**Not** "attention heads matter for safety" (known: Arditi et al. single-direction refusal
mediation; a growing literature on safety/refusal-relevant heads — verify exact refs before citing,
per the program's citation-verification rule). **The contribution is three specific things:**

1. **The decision site is a *measured* low-rank control channel** (PR ~10–15, cross-model, caught by
   a calibrated positive-control band — D2/`ANOMALIES.md` A2). This reframes "refusal is a
   direction" as "refusal is written into a narrow bus," and makes head attribution a question about
   **what gets transported into a channel of known dimensionality**, not a search.
2. **Content-transport as the coupling test.** We do not ask "do heads write refusal" (they must). We
   ask whether the writing heads **read moral content** (`mean_content` `V_moral`) vs read only a
   harm *cue* (`t_inst` harmfulness, Zhao et al.) vs read surface alarm — the content×decision
   *through-weights* cell that seven prior papers left unmeasured (`construct-audit` design matrix).
3. **A negative-space result if it lands there:** a content patch that moves refusal via
   *non-`V_moral`* features of moral content (causal cell b) would explain **every geometric null in
   the program** at once — orthogonality with causal coupling, the sign-flip included.

## 2. The question, as a design-matrix cell

`construct-audit` forbids a "disconnected" claim without the diagonal + a causal cell. D2 filled
decision-vs-decision (R3 dissociation) and content-vs-content (MFT↔V_moral). This paper fills the
**last missing cell: the causal content→decision arrow through weights** — the one the same-layer /
same-position geometry provably cannot see (Paper 1 storage-vs-usage divergence, reflexively
applied). Verdict language is therefore "reads-from / does-not-read-from," established by
intervention, never by geometry.

## 3. Stage 1 — who writes the decision (per-head direct attribution)

At the refusal direction's layer `L_ref`, decompose the residual at the **decision-site token** into
per-head OV contributions and per-MLP contributions, and project each onto the unit refusal
direction `r̂`:
`write(head h) = ⟨ (W_O^h · z^h) , r̂ ⟩`, `write(MLP) = ⟨ mlp_out , r̂ ⟩`, summed to the residual's
projection onto `r̂` (reconstruction check, §6 calibration).

- **LayerNorm handling (typed, construct-audit).** Use the **folded-LN approximation** (fold the
  final LN gain into the readout; treat the per-token LN scale as a constant at the decision token).
  Stated as an approximation, its reconstruction error reported; if reconstruction < 0.9 of the true
  projection, escalate to exact per-token LN.
- **Deliverables:** the **head-sparsity curve** (cumulative |write| vs head rank) and the
  **unexplained MLP fraction**.
- **Pre-registered branch (fixed now):** if MLPs carry **> ~50%** of the write, the head story is
  incomplete → the **Jacobian stage** (cross-layer attribution of `r̂` w.r.t. earlier residual) is
  added; this is a branch, not a failure (see §6 branches).

## 4. Stage 2 — what the writers read (source + value-side content)

For the top-`k` writing heads (k set by the sparsity curve's knee, pre-registered rule §PREREG):

- **Attention source distribution:** the head's attention mass over position classes at the decision
  token — `t_inst` (instruction/harm) vs content vs template/sink positions. Typed by
  `position_class` (construct-audit).
- **Value-side content:** project the head's **value inputs** (the things it copies) onto (i)
  `mean_content` `V_moral` (the *format-robust, position-valid* content subspace from D2 — NOT the
  raw or the bottleneck one), and (ii) a **`t_inst` harm direction** (Zhao et al. harmfulness at the
  last instruction token). Both references are typed + calibrated (band + null, §6).
- **Pre-registered copy-head hypothesis:** at least one top head is a **copy head moving `t_inst`
  harmfulness into the decision token** (high `t_inst` attention + value-side loading on the harm
  direction ≫ on `V_moral`). The competing hypothesis is a **moral-content-reading head** (value-side
  loading on `V_moral` above its band). Both pre-declared; the data picks.

## 4b. Causal cells (the decisive experiments — a "reads-from" claim needs ≥ 1)

All patches at content positions, decision measured at the bottleneck token (refusal projection) +
behaviorally (refusal rate, reconciled classifier). Controls per `construct-audit`: matched-random
+ named-reference heads, and the twin-surface match removes the register/valence/topic covariates.

- **(a) Twin patching.** Swap content-position activations between **moral-status twins** (the v2
  1,200 minimal pairs: surface-matched, only moral status flips). Measure Δ(refusal projection) at
  the bottleneck + Δ(behavior). Establishes whether moral *content* at content positions changes the
  refusal decision at all.
- **(b) Subspace-restricted patch (the decisive cell).** The **same** patch restricted to the
  `mean_content` `V_moral` subspace vs the **full residual**. **Full moves refusal but
  `V_moral`-restricted does not ⇒ the read uses *non-`V_moral`* features of moral content** — which
  would explain every geometric null in the program (orthogonality + causal coupling). The
  complementary outcome (`V_moral`-restricted moves it) would mean the moral subspace *is* the read
  substrate, reopening the geometric story.
- **(c) XSTest safe↔unsafe patching.** Patch between XSTest pairs where **alarming surface is held
  and harm flips** (and its converse: safe with alarming words). Separates **harm-cue reading** from
  **moral-status reading** — the confound-genealogy control for "surface alarm."
- **(d) Top-head ablation / resample-patch → behavior**, with matched-random and named-reference
  head controls; report the behavioral Δ against the matched-random null (outlier test), never raw.

## 5. Branches (all publishable; framings frozen before data)

| observed | reading | program consequence |
|---|---|---|
| **sparse write + writers read `V_moral`/harm content (causal b/c positive)** | anatomical content→decision coupling exists, localized | **the Direction-2 intervention target** (train the write, or the read) — a concrete address |
| **sparse write, but content patch moves refusal only via non-`V_moral` features (b)** | refusal reads moral content through features the moral subspace doesn't span | **explains every geometric null in the program** at once; the headline methods result |
| **distributed / MLP-mediated (> 50% MLP)** | no head-level shortcut | Jacobian follow-on + a "no clean head circuit; the read is diffuse" methods claim |
| **content patch moves nothing on stock instruct (a null, b/c null)** | refusal reads surface cues only on the aligned model | and the **coupled-model sign-flip** (forced-coupling arm) gets a dedicated re-examination — the coupling was training-induced, not native |

## 6. Instrument calibration (before any mediation/null verdict — instrument-calibration)

- **Reconstruction positive control (Stage 1).** The per-head+MLP decomposition must **reconstruct**
  the residual's projection onto `r̂` to ≥ 0.9; report the residual-of-decomposition. A decomposition
  that doesn't sum to the whole cannot attribute it.
- **Calibration ladder for every value-side projection** (`V_moral`, harm): floor `√(k/d)` → matched
  null q50/q95 (covariance-matched at the value-input distribution) → measurement ± bootstrap CI →
  positive band (held-one-out for `V_moral`; a known-harm reference for the harm axis) → named
  references. Verdict sentences carry the ladder ("head H reads V_moral at 0.X — above null q95 0.Y,
  inside the moral band [..]"), never "head H reads morality."
- **MDE on every causal cell.** State the minimum Δ(refusal projection) / Δ(behavior) detectable at
  the achieved n (twin count, rollout count) at 95% — so a null patch has teeth
  ("no `V_moral`-restricted effect, MDE ≈ 0.0X").
- **Control-validity re-verify.** The `t_inst` harm direction's defining property (encodes
  harmfulness) is re-measured **on this model at this position** before it is used as a bar; the
  `mean_content` `V_moral` position-validity (PR ≥ 30, band ≥ null) is re-checked per model (it was
  established in D2, re-verify in the C1 forward-pass context).
- **Position-validity carries over.** Every position touched (decision token, `t_inst`, content)
  records its `participation_ratio`; a patch/read at a PR < 30 bottleneck is flagged, not trusted.

## 7. Zero-GPU pre-build (compute-ordering: before any pod, before novelty claims)

1. **Lit pass** on safety/refusal-relevant attention heads + refusal mediation, to fix the
   contribution wording (§1) as "moral-content transport into a measured low-rank decision channel."
   Verify every cited claim against the primary source (program rule). **Novelty sentences are
   blocked until this lands.**
2. **D1 P0–P3 PR audit.** *Reflexivity catch:* the D2 amendment assumed this was zero-GPU, but the
   D1 reasoning runs saved only the P-window *direction vectors*, not the per-rollout window
   *activations* — so per-window PR is **not** computable zero-GPU. What IS zero-GPU: the pooled
   `act_sample.npz` PR for Think/GPT-OSS (a coarse check). The per-window PR audit is a small
   re-extraction rider (logged in `MISSING_ARTIFACTS.md`), not a pod trip of its own.
3. **Twin + XSTest cell prep.** Verify the v2 1,200 minimal pairs and the committed
   `xstest_borderline.json` are usable as patch sources as-is; type them (construct-audit).

## 8. Models & order (compute-ordering: ΔDecision-ranked; instrument-first)

1. **OLMo-3 first** — the clean instrument (well-conditioned activations; the D2 anchor).
2. **Llama-3.1 second** — with a **pre-registered comparative prediction:** *Llama's top writing
   heads read moral content (value-side `V_moral` loading) more than OLMo's* — the n=1 behavioral
   entanglement (Llama's judgment drop under refusal ablation; the forced-coupling family) explained
   at head level. A confirmed prediction is a strong result; a disconfirmed one is a datapoint about
   where the entanglement actually lives.
3. **Qwen2.5 third, budget permitting.**

## 9. Riders (ride the same loaded-model sessions — compute-ordering batching)

- **B5 (R8 moral-fragility baseline)** on the `mean_content` subspace (type block records the
  position switch from the bottleneck to the content position; the standing metric for a future
  Direction-2 intervention).
- **Reconciled cross-ablation** (R3's causal arrow) on the **full** harmful set with the shared
  `_classify_response` classifier (the B1-floor fix already landed).
- **Template-token PR-profile control:** confirm the ~10–15-dim bottleneck is the assistant-header
  token specifically (vs other template tokens), so "decision-site bottleneck" is precise.

## 10. Compute plan (session template per model)

```
SESSION (est. 2–4 h, one model: OLMo-3 first)
  keystone:    Stage 1 (per-head write attribution) + Stage 2 (source + value-side content)
               -> the ΔDecision item: which heads write, and do they read moral content
  riders:      causal cells (a) twin patch, (b) V_moral-restricted vs full, (c) XSTest patch,
               (d) top-head ablation; B5 mean_content; reconciled cross-ablation; template PR control
  pilot gate:  reconstruction >= 0.9 (Stage 1) before trusting attribution; head-sparsity knee
               sets k before Stage 2; if MLP > 50% -> Jacobian branch (do not force a head story)
  depends on:  D2 refusal.npz + b1_judgment_dir + chat_vmoral(mean_content) per model (uploaded);
               v2 1,200 twins; xstest_borderline.json; t_inst harm direction (extract in-session)
  saves:       per-head write contributions (all heads, band layers), per-head attention maps +
               value inputs, per-twin/per-XSTest patch deltas, reconstruction residual — PER UNIT
               (compute-ordering: keeps every downstream bootstrap/MDE zero-GPU)
  gate after:  the §5 branch table -> Direction-2 intervention target OR methods-null OR Jacobian
```
Stage 1–2 are hook-cached forward passes (~1–2 h/model); the patching cells ~1–2 h/model. One
session per model with riders. **Zero-GPU (§7) runs first and gates the novelty framing.**

## 11. Out of scope (explicit)

- No training/intervention here (that is Direction 2; this hands it a target).
- No claim of a "safety circuit" beyond the measured write→read cells; distributed outcomes are
  reported as such.
- Geometry never establishes a "reads-from" verdict — only the causal cells do.

## Ship-blockers (must pass before the write-up commits)

**construct-audit:** [ ] every direction/head-contribution typed · [ ] the content→decision causal
cell measured (not just geometry) · [ ] confound genealogy handled by the twin-surface match + XSTest
harm-flip · [ ] construct constant across the position sweep (decision vs content vs `t_inst` typed
separately) · [ ] normalization audited cross-model (per-model nulls).
**instrument-calibration:** [ ] Stage-1 reconstruction ≥ 0.9 · [ ] ladder on every value-side
projection · [ ] MDE on every causal cell · [ ] harm + `V_moral` control properties re-verified in
context · [ ] controls named by construction.
**compute-ordering:** [ ] §7 zero-GPU done + lit pass gating novelty · [ ] each GPU item's
ΔDecision line present · [ ] riders batched onto the keystone session · [ ] per-unit saves specified.
