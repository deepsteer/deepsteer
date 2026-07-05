# OPEN_THREADS.md — audited thread register (W0, 2026-07-03)

Every open thread classified. **Nothing is silently omitted:** a blocking thread means the
affected claim ships in its scoped wording (stated here and in CLAIMS.md), not that the claim
is dropped.

Phase is FROZEN (no pods). Disposition codes:
- **CLOSED** — resolved (this session or earlier); no action.
- **BLOCKS→SCOPED** — blocks a headline claim; the claim ships scoped (wording given). Closing
  it needs a pod → parked for the next phase with a price.
- **ZERO-GPU** — closeable now without a pod (a read, an edit, a prose fix); scheduled.
- **OPTIONAL** — follow-on; parked, does not affect any shipping claim.
- **ESCALATE** — needs Orion (thesis/scope/publication/safety, or unresolvable from repo).

Prices are rough A100-hour estimates for the parked pods.

---

## A. Closed (verified this session)

| thread | disposition | evidence |
|---|---|---|
| A6 incorporation into MN skeleton | **CLOSED** | `METHODS_NOTE.md` lists A1–**A6** (A6 landed 2026-07-03) |
| Amendment 13 wording in SYNTHESIS (dimensionality rival + architecture confound) | **CLOSED** | SYNTHESIS "Held (panel breadth)" carries the A13 one-axis rival + confound sentences |
| null-ratio-corroborates-harm-keying sentence in GPT-OSS RESULTS | **CLOSED** | D3 RESULTS GPT-OSS block: "The null-ratio corroborates harm-keying at the reflexive site" (null-ratio 372) |
| OLMo depth-matched A recomputation (symmetric −0.54 behind the −0.28 comparison) | **CLOSED** | D3 A10: A_OLMo@12 = −0.54 (CI [−0.81, −0.32]); the read-layer −0.20 no longer used in the frozen difference (+0.26) |
| Heading-string alignment (boot-sequence heading verbatim) | **CLOSED** | only `WRITEUP_PHASE_PLAN.md` + `CLAUDE.md` reference it; both use "Research boot sequence" |
| Paper 6 back-audit (Qwen/Llama geometric cells vs the vulnerable null) | **CLOSED (CLEAN)** | ANOMALIES A1: Paper 6 uses a permutation test + raw projection fraction, never the covariance null; no number changes |
| MISSING_ARTIFACTS.md location vs boot sequence | **CLOSED** | moved to `papers/MISSING_ARTIFACTS.md`; CLAUDE.md line 175 matches |
| W3-reproducibility risk: gitignored `outputs/` arrays/figures lost? | **CLOSED (present locally)** | flagship `c1_inputs_olmo3.npz` (19MB) + `c1_session_olmo3.json` + `one_knob_olmo3.png`, D1 calibration JSONs + `a1_ladder.{png,pdf}`, P5/P6 figure PDFs all on disk (d1 114 / d2 101 / d3 18 files). W3 distillation path viable; not lost, just uncommitted |
| CLAIMS numbers = agent extractions? | **CLOSED (spot-checked)** | D2 PR 14.7/8.6/10.2, P5 0.999 / 0.10 / 0.25→0.00, P6 0.104/0.127/0.071 all match primaries exactly; non-spine numbers primary-verified at sentence drafting |
| **Bib verification (W0 addendum 2, priority)** | **CLOSED (CLEAN)** | Paper-1 bib verified against arXiv: `olmo2_2025` (2501.00656, title/id ✓, cosmetic author-form `OLMo Team`→`Team OLMo`), `ren2026apex` (2602.03586 ✓), `borras2022walking` (2212.10430 ✓), `qian2024trustworthiness` (2402.19465 ✓). Named/memory concerns clean: `hubinger2024sleeper` (2401.05566 ✓, Paper 2), `muennighoff2024olmoe` (2409.02060, first-6 authors match arXiv — the flagged confabulation is already fixed). **No Paper-1 errata.** |
| **Retro-checks (W0 addendum 3)** | **CLOSED (CLEAN)** | (a) σ\* framing already in P1 §4.4 + lines 162–182 (raw = within-layer; RMS cross-layer) → propagate one-line note to MN/DUO. (b) onset-ordering caveat present (P1 lines 41–46: onset tracks unigram-TF-IDF lexical floor; separates encoding from difficulty). (c) P1 probing = mean-pooled content positions (full-rank by construction, not the decision-token bottleneck) → PR-valid, no audit. |
| **NI-1 base rank-3 V_moral null** | **RESOLVED (partial)** | committed `a1_ladder_base.json` `null_q95 = 0.2913` → 0.291 of record; RESULTS G3-table 0.31 may be a distinct G3 null (confirm at drafting); verdict robust |
| **NI-5 Paper-1 §4.3 vs stale c3/RESULTS.md** | **RESOLVED (not errata)** | v2 c3 JSONs carry the paper's numbers (peak_acc 0.740/0.750/0.750, σ* floats 7.3852/5.6351/6.9413); stale markdown used cap-at-max — refresh it (doc hygiene) |

---

## B. Blocks a claim → ships scoped (pod-gated; parked with price)

| thread | scoped wording the claim ships in | price |
|---|---|---|
| **D1 P0–P3 PR audit** (band rung is cross-position; queued in MISSING_ARTIFACTS) | D1 reasoning-gradient band-relative statements ship "scoped as cross-position; P2's null-relative crossing stands." Null-relative claims (D1-11) unaffected. | ~0.3h re-extraction (per-rollout P0–P3 window acts) |
| **Llama rank-2/4 severity-twin harm-coextensive check** (unsaved severity-twin contrasts) | Llama reads-broad ships carried by the rank-1 rejection (3.6%, D3-19) + gap-closes (D3-18); the *strongest* "beyond harm" strength claim ships "rank-1 rejected; rank-2/4 rides an extraction rider, prior against coextensivity." | ~0.5h re-extract severity-ladder contrasts at Llama L12 |
| **GPT-OSS reads-axis is correlational** (no causal sweep; Tier-2 held) | GPT-OSS "reads harm" ships as **correlational** (P2 harm-loading 0.49 vs 0.13, prompt→trace 0.977, D1-18/19); causal parity is Tier-2 (OT §E). | ~2–3× a Llama session (Tier-2 C1-MoE) |
| **Reconciled cross-ablation on the full harmful set** (base 0.167 floor = harness drift, V-D2-4) | The R3 causal *arrow* ships "queued; the R3 geometric dissociation (D2-01) is banked and does not depend on it." | ~0.5h (B1 re-run, reconciled classifier) |
| **GPT-OSS in-trace decision-token projection refinement** (last-token-of-prefill caveat) | GPT-OSS reversibility ships with the behavioral flip (6/10, D3-22) as **primary** and the projection as **corroboration with the last-token caveat stated**. | ~0.3h (re-read the in-trace decision token after the model responds) |

---

## C. Zero-GPU (closeable now; scheduled into W1/W2 drafting)

| thread | action | when |
|---|---|---|
| **NI-2** Llama decision-channel PR 13.5 (D3 C1) vs 10.2 (D2 ladder) | print both with position labels ("decision-token C1 13.5; in-format-ladder 10.2"), or pick one as "the Llama bottleneck PR" — propose printing both | W1 (MN figure) / W2 (FL beat 3) |
| **NI-3** "position-valid" vs "position-invalid" reconciliation | write the one sentence: the bottleneck is position-**invalid for content projection-fraction** tests (band-below-null) but **valid for decision-direction** reads (R3, GPT-OSS refusal projection); the "25 ceiling" is a separate MoE PR sanity gate | W1/W2 (recommend drafting the sentence at W1, MN A2) |
| **NI-4** OLMo R_refusal/harm-ceiling drift (folded-primary vs standardized) | state the convention: folded-primary (0.31 peak / 0.27 plateau, harm_rank1_R 0.31) is the headline; standardized-sweep (0.33 ≈ 0.35) is the robustness check; stop conflating rank-3 peak with rank-16 plateau | W2 (FL beat 5) |
| **NI-6** OLMo-Think P2 "~0.009 below null" | fix prose to the table arithmetic (0.35 vs margin 0.354); verdict (near-miss NULL) unaffected | W2 |
| **NI-7 / NI-8** ETHICS n 197/199; persona c 0.506/0.510; Figure-4 name collision; onset 4K/5K | pick canonical values; renumber the §4.3 "Figure 4"; always attach "4-seed-mean" | W1 (P1) / W2 |
| **σ\* framing carry-over** (retro-check (a) closed clean) | P1 §4.4 already states raw = within-layer, RMS cross-layer; propagate the one-line note to MN + DUO cross-refs | W1 (MN + DUO) |
| **R5 std-vs-projout Llama numbers** (only Qwen quoted) | note the asymmetry; the in-format ladder is the discriminator (content subspace is format-robust); optional cleanup | W2 |

---

## D. Optional follow-on (parked; affect no shipping claim)

- **B5 / R8 moral-fragility baseline** — HELD, "not needed for the headline"; the standing metric
  for a future Direction-2 intervention (σ\* ratio ratified 0.5×). Parked.
- **C1 attention-head attribution at the D2 bottleneck** — "the real next experiment," gated at
  D2 Gate B; D3 already delivered the OLMo anatomy, so this is Direction-2 work. Parked.
- **R6 / R7 / B4** — F2 rotation-specificity (≥15°), Fisher-combined P2 (exploratory), Llama B4
  rank-k sweep. Parked.
- **Cells (c)/(d) on OLMo** — XSTest generalization + full mean/resample ablation battery. Parked.
- **Cross-model transfer-robustness check** (instruct refusal × Base-V_moral). Parked.
- **Track-4 cross-register** (28 pairs, directional/exploratory). Parked (read as designed).
- **Cross-layer Stage-2 approximation** (earlier-layer writers vs read-layer V_moral). Ships as a
  stated limitation; no action.
- **R_refusal precision** (ratio-of-ratios CI wide at n=23) — the sweep resolves the *shape*; ships
  as a stated limitation. More twins would tighten it. Parked.
- **xstest_borderline.json provenance** — dependency for B5 session 1, not the headline. Parked.
- **OT-9 — whitened / PR-matched fragility control (P1 residual gradient).** §4.4 reports a residual
  ~2× (1.8–3.1×) layer-depth ratio surviving RMS normalization, explicitly **not claimed** as a
  genuine gradient: RMS controls per-layer scale, not covariance shape. The test that would establish
  (or kill) a genuine residual gradient is injecting noise in each layer's **whitened** (or
  participation-ratio-matched) basis, re-running the 37-checkpoint fragility trajectory. **Optional**
  (the v2 claims do not depend on it). Price: ~0.5–1h (re-extract + whiten + re-probe across
  checkpoints; comparable to the original refragility run). Feeds a possible P1 v3 or the methods note.

---

## E. Held register (priced for the next-phase decision)

| held item | what it buys | price |
|---|---|---|
| **Qwen2.5-7B causal C1** | the lineage-independent 4th harm/broad datapoint for the A13 deconfounding panel; stronger A1 caveat (dim 458 = 59%) | ~1 Llama-session |
| **Tier-2 causal C1-MoE (GPT-OSS)** | makes the GPT-OSS reads-harm verdict *causal* (KV-persistent dual-basis patches, router-weighted Stage-1, depth×position commitment map) | ~2–3× a Llama session |
| **In-trace decision-token projection refinement (GPT-OSS)** | a cleaner reversibility readout (decision token *after* the model responds, not the prefill last token) | ~0.3h |
| **P2 SNR-normalized fragility fix** | unblocks HELD-P2 (raw σ\* "scale meter" → SNR-normalized); links to the NI-5/OT-1 σ\* family | ~0.5h analysis + possible re-extract |
| **Direction-2 intervention (widen what the writing heads read)** | the program's forward target; needs its own prereg + human go (safety-adjacent) | new program |
| **A13 deconfounding panel** (OLMo-3-7B-Think + Qwen) | tests dimensionality→reversibility by varying one axis at a time | Think + Qwen sessions |

---

## F. Escalations (Orion decides)

**F2/F3/F4 RESOLVED (see §A). F1 is now RESOLVED too (arXiv v1 read directly): the DUO needs a
v2. The remaining decision is the erratum WORDING (Orion — it scopes a published abstract-level
finding), and that build is W3, beyond this gate.**

1. **OT-1 (F1) — Paper 1 v2 → RESOLVED: v2 REQUIRED (errata-class).** Read arXiv:2606.11375v1
   (9 Jun 2026) directly: it **lacks §4.4** (the RMS-normalization control, added post-submission
   `ddff03e`/`1300d50`, 12–13 Jun). It states the raw layer-depth fragility gradient (P1-04) as
   **abstract Finding 2** + Table 2 (late 10.0 / early 1.8) + Figures 3–4, and the post-saturation
   decline (P1-05) raw (Fig 3). v1 carries **one passing hedge** (§5.2: "a low critical noise can
   also reflect activation-scale changes … rather than margin or redundancy alone") but no
   quantified control and no scoping of Finding 2. So §4.4's finding (the gradient is *largely*
   activation scale: ~2× under RMS, ordering fails at 8/37 checkpoints) materially changes a
   published headline → **errata-class → v2**. Mitigation: v1 pre-flagged the possibility, so the
   erratum is a quantified confirmation of a stated caveat, not a reversal. §4.3 numbers in v1 are
   already the correct v2 ones (NI-5 clean in the published paper too).
   **→ UPDATE 2026-07-04: Template A approved (Finding 2 = confound-isolation; both sub-claims
   VOID per V-P1-1/2). v2 PDF built (`papers/1_accuracy_vs_fragility.pdf`), abstract reworded, §4.4
   residual non-claim added, citation fix applied, v2 comment drafted. Held for two pre-submission
   bib flags (ref-list omits OLMo-2/3 entries; 2025a/b disambiguation), not for wording. Cross-paper
   dependency: Paper 3 cites this paper's §4.4 (v2-only), so v2 is the version its citation resolves
   against — another reason v2 ships.**
2. **~~Bib verification~~ → CLOSED CLEAN (§A).** No Paper-1 errata; one cosmetic author-form nit
   (`OLMo Team`→`Team OLMo` in `olmo2_2025`) to fix at drafting. The OLMoE confabulation MEMORY
   flagged is already fixed in the committed bib.
3. **~~NI-5 §4.3~~ → RESOLVED (§A).** Paper §4.3 is artifact-backed; refresh the stale
   `c3/RESULTS.md` markdown (doc hygiene, not errata).
4. **~~NI-1 base null~~ → RESOLVED (§A).** Committed `a1_ladder_base.json` = 0.291; reconcile the
   RESULTS 0.31 (possible distinct G3 null) at drafting; verdict robust.
5. **Paper-B disposition** — companion note now / FL appendices / defer to Direction-2 (Gate W2).
6. **C2 counterfactual moral-consistency DPO** — HELD, training out of scope, safety-adjacent
   (needs a separate Direction-2 prereg + explicit go).

---

## G. Gate-W0 decision summary (what Orion is asked to rule on)

1. Confirm the §0 unit structure (MN / FL / DUO / HELD-P2 / PAPER-B).
2. Thread dispositions: approve the B (scoped) wordings and the C (zero-GPU) schedule; approve
   parking D/E for the next phase at the stated prices.
3. **F1 RESOLVED: v2 is required** (v1 lacks §4.4; P1-04/05 errata-class). The open decision is
   the **erratum wording** — it scopes abstract-level Finding 2, so it's yours; the v2 build is W3
   (held this phase).
4. Paper-B disposition (F5) — or defer to Gate W2 per the plan.

---

## H. Self-review escalations (2026-07-04 hostile-reviewer pass; detail in SELF_REVIEW.md)

Both drafts were reviewed from the delivered PDF in the external-reviewer frame. The shared
root objection is overclaiming relative to the evidence; the AUTO fixes (honest scoping, anchor
corrections, notation/limitations, per-mode coverage, related-work positioning) are applied and
built. The following are thesis/scope/publication calls left to Orion. Each is self-contained
here so it survives independent of the review scratch doc.

| # | escalation | why it is Orion's | options |
|---|---|---|---|
| **H1** | **FL title** over-generalizes a single-model causal result. Table 1 shows Llama refusal reads *broad* moral content by interchange (transfer 0.85 ≈ judgment 0.79); the titular "reads harm, not the moral subspace" is causally supported on OLMo (n=23) and contradicted on Llama. | the title scopes the whole contribution | keep as-is (panel-general claim); **or** "Refusal Reads a Low-Rank Slice of Moral Content"; **or** scope to "On OLMo-3, refusal reads the harm percept" + make §8 explicitly a cross-model variation study |
| **H2** | **FL contribution scope** — the causal rank sweep is one model (OLMo, n=23, near the MDE); GPT-OSS is correlational, Qwen never appears on the read axis. | rescoping the headline is a framing decision | AUTO already softened abstract/§1 to "causal on OLMo; panel is a cross-architecture consistency check with one dissenting read (Llama)"; the whole-contribution rescope to single-model-causal is yours |
| **H3** | **Publish P3** — FL §3 (compositional moral encoding) rests on `reblitzrichardson2026geometry` ("How LMs Organize Competing Moral Frameworks"), cited once, currently an in-series `note={Paper 3, this series}`. | publication decision | publish P3 (gives a real cite) / fold the two numbers into an FL appendix / leave as in-series pointer |
| **H4** | **Release underlying arrays** — 5/6 MN modes and several FL cells cite numbers whose backing lives in uncommitted local `outputs/` files, so a reviewer cannot check them from the page. | data-release decision (repo policy: derived arrays uncommitted) | release per-unit arrays as a supplement / cite the source papers where each number is primary / ship scoped with the honest verifiability caveat (AUTO-stated) |
| **H5** | **MN scope** — "protocols others can run" vs. externally-validated. All evidence is one program on one panel (OLMo-3/Qwen2.5/Llama-3.1/GPT-OSS). | scope of the note's claim | honest rescope to "lessons from one program, offered as portable" (AUTO-hedged now, Limitations added) vs. commit to a broader external-validation claim (needs cross-program evidence) |

**[POD] items surfaced (frozen phase cannot run; AUTO edits scope the claims instead):** the full
read×commit grid on every panel model; more request-twins to lift n=23 past the MDE; a
decorrelating control for the refusal/harm genealogy (harmful-but-complied / benign-but-refused);
a non-moral positive-projection control for the calibration ladder (persona is moral-adjacent).

### Resolved 2026-07-04/05 (Orion)

- **H1 — RESOLVED.** New FL title *"Refusal Reads Only a Slice of What the Model Knows"* +
  subtitle *"Harm-Keyed Routing and Its Exceptions Across Model Families."* The earlier
  *"...Slices of Moral Content"* proposal was declined on accuracy: it framed harm as moral
  content, contradicting the 76%-off-subspace finding; the chosen title keeps harm and the
  subspace distinct and flags the exceptions (Llama-broad, GPT-OSS-reversible) in the subtitle.
- **H2 — RESOLVED (keep AUTO scope + one sentence).** Abstract now carries per-cell labels:
  "causal on OLMo-3, corroborative on GPT-OSS (correlational), contrastive on Llama (causal,
  broad); Qwen absent on the read axis."
- **H3 — RESOLVED.** P2/P3 arXiv-forthcoming; FL's P3 self-cite note de-labeled and the eprint
  id drops in at publication.
- **H4 — IN PROGRESS.** Building `deepsteer/supplement/` (manifest-indexed, distilled artifacts
  only — per-head contribs, nulls, PR profiles, ladders, sweep outcomes; not raw activations).
  MN cites comprehensively; FL cites per quantitative figure + ships regen scripts with pinned
  revisions/seeds; shared arrays live once. Raw activations on reviewer request.
- **H5 — RESOLVED.** MN abstract adds "four architectures across three families within a single
  program; external replication across programs is future work."

**Program-thesis pass (title + both abstracts).** Anchored adjectives (0.869→0.999; 40°/0.757;
0.155; 9–15 dim; |cos|<0.10 null 0.41; 0.66 vs 0.31; 76% off); construction-named cosines
(checkpoint-to-final crystallization vs proto-refusal-to-gate — corrected a "base-to-aligned
0.999" mislabel in the abstract, §1, §3, §11); detection bar on the orthogonality verdict;
per-cell causal-vs-correlational labels.

**Open (Orion): SYNTHESIS thesis scope.** Proposal to restructure the single OLMo-3-scoped
thesis into three evidence tiers (panel-level structural / OLMo-causal family-varying read /
OLMo-only checkpoint crystallization), mirroring the FL paper and the new title. Plus a
rule-9 fix to SYNTHESIS line 40 ("architecturally guaranteed" → "structurally favored").
Awaiting sign-off before the SYNTHESIS edit lands.
