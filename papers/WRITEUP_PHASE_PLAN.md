# Write-up & Packaging Phase — Plan WP-1 (2026-07-03)

**For:** Claude Code (Fable 5.1 for the judgment sessions W4-1 and W4-3; Sonnet 5 for the
pod-driver session W4-2; see Phase W4), cold start after `/clear`. Context = this document + the
committed repo. First action every session: Run CLAUDE.md "Research boot sequence" then this plan.
**Phase mode: experiments are FROZEN, with one exception.** Phase W4 (2026-09-10) unfreezes the
program for exactly one targeted pod (Tier A + Tier B below); everything else stays
zero-GPU claim-verification, and any other tempting experiment goes to OPEN_THREADS.md with a
cost estimate and waits for a gate. The research program (D1→D2→D3, three-model panel, two-axis resolution) is complete
per `papers/SYNTHESIS.md`; this phase converts it into papers.

---

## Phase W0 — Ledger reconciliation & claim inventory (first session, zero-GPU)

The `/clear` means no working memory of which riders closed. Reconstruct from the record:
walk every amendment trail (`papers/d1_moral_subspace/PREREGISTRATION.md` + CALIBRATION
docs, `papers/d2_decision_coupling/PREREGISTRATION.md`, `papers/d3_decision_anatomy/
PREREGISTRATION.md` Amendments 1–13), `papers/ANOMALIES.md`, `MISSING_ARTIFACTS.md`,
`papers/SYNTHESIS.md`, and all RESULTS.md files. Produce three documents:

1. **`papers/PACKAGING.md`** — the old→new mapping in §1 below, expanded to section-level
   provenance (each section of each new paper lists the source docs/results it draws from).
2. **`papers/CLAIMS.md`** — every claim that will appear in any paper, one row each:
   `id | claim (exact anchored sentence) | status (verified / CI'd / scoped / held / void) |
   artifact path | figure?`. Every number in any draft must trace to a CLAIMS row. Voided
   claims (e.g. the layer-16 asymmetry A=+0.82; the pre-boundary Llama directional hint)
   are listed as VOID with their replacement, so they cannot re-enter prose.
3. **`papers/OPEN_THREADS.md`** — audit at minimum these candidates (status uncertain at
   plan-writing; verify each): B5 fragility baseline (R8) — run or still held?; reconciled
   cross-ablation on the full harmful set; D1 P0–P3 PR audit (queued in MISSING_ARTIFACTS);
   OLMo depth-matched A recomputation (symmetric treatment behind the −0.28 vs −0.20
   comparison); A6 incorporation into the methods-note skeleton (skeleton lists A1–A5);
   Amendment 13 wording landed in SYNTHESIS (dimensionality one-axis rival + architecture
   confound); the null-ratio-corroborates-harm-keying sentence in GPT-OSS RESULTS; P2's SNR
   normalization fix; held items (Qwen, Tier-2 causal C1-MoE, in-trace decision-token
   projection refinement). Classify each: **blocks a claim** (must close or the claim ships
   scoped) vs **optional** (follow-on). Nothing is silently omitted: an open thread means
   the affected claim ships in its scoped wording, stated in CLAIMS.md.

Addendums to the above:

1. Extend CLAIMS.md to Paper 1's published headline claims; disposition each: confirmed / scoped / errata, with artifact pointers.
2. Bib verification (priority): reconcile PAPER_PLAN's open citation-verification item against the published arXiv v1 — fetch each flagged reference's abstract page, diff author lists/venues/IDs against the published bib (olmo2_2025, hubinger2024sleeper named concerns first). Pass ran pre-submission → close the stale item; discrepancies in the published bib → errata-class, triggers v2.
3. Retro-checks: σ* absolute-noise definitional footnote needed? (within-model comparisons expected fine — state it); onset-ordering difficulty caveat vs current framing; probing-position PR validity on the P1 configs (expected clean).
Heading-string alignment pass: all docs referencing the boot sequence use the merged CLAUDE.md heading verbatim.
Conditional W3 item: Paper 1 arXiv v2, gated on (2)/(3) — errata-class → v2 required; scoping-only → minimal v2 bundled with the MN arXiv date; clean → no v2, forward papers cite v1 and the audit rows in CLAIMS.md record the confirmations.




**Gate W0 (Orion):** review CLAIMS + OPEN_THREADS; decide thread dispositions
(close-now-zero-GPU / scope the claim / defer) and confirm the §1 structure.

---

## §1 — The paper structure (revamped; supersedes the earlier informal restructure)

Package by claim, not chronology (program-thesis packaging principle). New instruments
absorb superseded papers; prereg/amendment trails stay as repo documents both papers cite
(a public pre-registration trail is a credibility asset — reference it explicitly).

| unit | thesis | absorbs / draws on | status |
|---|---|---|---|
| **MN — methods note** (*Instruments before verdicts*, `papers/METHODS_NOTE.md`) | interpretability instruments fail in specific, diagnosable ways; calibrate → certify with an orthogonal cell → power before pod → depth-indexed verdicts | ANOMALIES A1–A6; ratio-of-ratios, power-table, orthogonal-cell-certificate, operating-point/dynamic-range, depth-indexed-verdict, MDE-crossing (trap 12) patterns; the 4-architecture bottleneck as *motivating discovery* | draft skeleton exists → W1 |
| **FL — flagship** (routing + commitment) | refusal reads the harm percept through a narrow control-token bottleneck; families differ in what refusal reads × how it commits | D1 (calibration, crystallization, P0–P3), D2 (decision-vs-decision, bottleneck, format-robustness), D3 (anatomy, sweeps, two-axis panel), P4 (preliminary causal validation — fully absorbed), P5 dissociation core, P6 representational cells + robustness anomaly, P7 decision-point/trace findings + distributed refusal | outline → W2 |
| **P1 + P3** | pretraining duo (emergence; competing frameworks) | untouched; FL cites P1 (published) | done / standalone |
| **P2** | MoE dilution | standalone, held pending SNR-normalized fragility fix (OPEN_THREADS) | held |
| **Paper B** | behavioral/interventional companion | *gated decision at W2* — candidates: ART forced-coupling arc + sign flip, F2 rotation + specificity control, persona/Assistant-Axis + persona-shift compliance, removability battery detail, B5 (if run), distributed-refusal detail. Dispositions: (a) companion note now, (b) FL appendices, (c) defer to the Direction-2 intervention paper | decide at Gate W2 |

**Double-claiming rule:** the bottleneck *finding* (PR 9–15 across OLMo/Qwen/Llama/GPT-OSS)
lives in FL; the *validity protocol* it motivated (band-below-null tell, PR gate,
standardization + invariance proof) lives in MN. One cross-reference sentence each way;
neither paper claims the other's contribution as novel.

---

## Phase W1 — Methods note to arXiv-ready (standing task; nothing blocks it)

Flesh the skeleton: per-anomaly section = failure as it first appeared → the tell → the
protocol → the check that certifies the fix (each with real numbers; the case studies are
the paper). Add A6; add the estimator/intervention patterns as a section each; appendix =
the ship-blocker checklists (portable form of the skills). Deliverables: figures
(bottleneck PR bar ×4 architectures; band-below-null ladder example; the +0.82→−0.28
depth-collapse as the depth-indexed exemplar), reproducibility statement, ~8–12 pages.
Optional (flag at gate, Orion decides): extract a small `deepsteer.validity` module
(ladder, PR gate, covariance-matched nulls, power table) as the open-core companion.
**Gate W1:** full draft → external review pass (Orion routes through the review channel).

## Phase W2 — Flagship outline, then prose

Outline first; **no prose before Gate W2 approves the outline.** The claim arc, seven beats:
1. Moral comprehension is pretraining-native and survives alignment (P1/P3 cite;
   crystallization cos → 0.999).
2. The refusal gate is a fresh post-training construction (proto-refusal→gate cos 0.155) in
   a low-variance channel.
3. The decision site is a ~9–15-dim control-token bottleneck on four architectures; content
   and decision never co-locate — content-vs-decision orthogonality is structurally favored
   (D2 + the position-validity protocol, MN-cited).
4. Decision-vs-decision: refusal ⊥ moral-judgment decisions panel-wide (R3), stated with
   detection bars and the calibrated bands.
5. Causal anatomy (OLMo): distributed write (~62 heads, 38% MLP) into the channel;
   interchange + nested rank sweep → `harm_saturating`, one-knob fit RMSE 0.036; judgment
   reads broadly on the *same patches* — the within-model contrast proving readability.
6. Cross-model two-axis panel: Llama reads broad + early-commits (depth-verified,
   A11-hardened); GPT-OSS reads harm (P2 harm-loaded 0.49 vs 0.13) + reversible
   (engage 7/7, graded-disengage 6/10, monotone projection); two-axis table; the
   dimensionality one-axis hypothesis with the architecture-confound caveat (A13 wording)
   as the follow-on frame.
7. Implications: a mechanism for shallow alignment (the wrapper reads the harm percept over
   a narrow bus); the Direction-2 target (widen what the writing heads read); GPT-OSS as
   existence proof that deliberation can be load-bearing and reversible; standing safety
   scope (characterization of released models; no removability optimization).

Money figures: 0.999-vs-0.155 pair; D1 calibrated ladder; bottleneck PR bar ×4; OLMo
R_refusal(k) vs R_judgment(k) + one-knob fit; Llama gap-close + depth-gated disengage;
GPT-OSS psychometric step + graded-prefill monotone panel; the two-axis table.
Limitations section must include: n=3 architecture confound + the one-axis rival; GPT-OSS
reads-axis is correlational (Tier-2 held); readout-vs-behavior scope per cell; the
prefill-last-token caveat; stimulus-composition covariates across model bands.
Title candidates (Orion picks): *"What refusal reads: harm routing and commitment dynamics
in open-weight language models"*; *"Refusal reads the harm percept: routing, bottlenecks,
and reversibility across model families."*
Claim-language pass (program-thesis rules 1–11) + referee pass are outline deliverables,
not afterthoughts. **Gate W2:** outline + figure list + Paper-B disposition.

## Phase W3 — Full drafts + repo alignment

FL prose section-by-section against the approved outline (each section PR-able alone, every
number CLAIMS-traced); `papers/README.md` rewritten to the new structure with pointers to
the absorbed papers' trails; **figure reproducibility**: outputs/ is gitignored, so every
figure needs a regeneration script reading committed inputs or documented local arrays —
commit small distilled `figure_data/*.csv` where the source arrays are local-only; anything
irreproducible is flagged at gate, never silently shipped. SYNTHESIS.md remains the living
abstract source. **Gate W3:** full FL draft → external review → arXiv decision (Orion).

---

## Standing rules for this phase

1. No pods. Zero-GPU verification only; anything else → OPEN_THREADS with cost.
2. Every number in prose has a CLAIMS.md id; every claim has its anchored sentence fixed
   there first.
3. Commit-boundary blockers hold: referee pass + SYNTHESIS update in the same commit as any
   RESULTS/draft milestone.
4. Voided results (CLAIMS status VOID) may be *discussed as methods lessons in MN* but never
   as findings in FL.
5. Escalate to Orion: venue/authorship/timing, Paper-B disposition, anything touching the
   safety scope, any claim whose verification would need a pod.
6. Methods note is the default filler task whenever a gate is pending.

## Phase success criteria

MN submitted (or Orion-approved final) · FL full draft through referee pass · PACKAGING /
CLAIMS / OPEN_THREADS complete and clean · held-thread register (Qwen, Tier-2, projection
refinement, P2 SNR, Direction 2) accurate and priced for the next phase decision.

---

## Phase W4 — venue-quality pod (2026-09-10)

**Decisions of record (Orion, 2026-09-10; do not reopen in-session).**

1. **Bar.** Venue quality before arXiv for MN and FL. The July self-review left both at
   borderline-reject for overclaiming; every AUTO fix and H1–H5 escalation is resolved (OPEN_THREADS
   §H). What remains are the [POD] items the frozen phase could only scope. W4 runs them.
2. **Unfreeze scope.** Exactly one targeted A100 pod, pre-registered as D3 Amendments 14 (Tier A)
   and 15 (Tier B) in `papers/d3_decision_anatomy/PREREGISTRATION.md`. Tier A = claim-bearing and
   cheap: 14.1 proto-refusal reliability ceiling; 14.2 Llama rank-2/4 harm-coextensive; 14.3 GPT-OSS
   post-response decision-token projection (+ the band-below-null half of the A5 pre-condition);
   14.4 D1 P0–P3 per-rollout PR audit (+ Think MFT/refusal saves); 14.5 reconciled B1
   cross-ablation; 14.6 per-unit saves for the MN instruments. Tier B = panel strength: 15.1 OLMo-3
   additional request-twins; 15.2 Qwen2.5-7B-Instruct causal C1 read cell.
3. **Everything else stays HELD at its OPEN_THREADS price.** Tier-2 GPT-OSS causal C1-MoE (§E,
   ~2–3 Llama-sessions), Direction 2 (new program, safety-adjacent), the A13 deconfounding panel
   (Think + Qwen sessions), C2 counterfactual-consistency DPO (§F.6, training, safety-adjacent), B5
   fragility baseline, OT-9 whitened-fragility control. Not in Tier A/B means not in this pod.
4. **Paper-B disposition: FL appendices.** The behavioral/interventional companion folds into FL as
   appendices F (removability battery, from P6 §4.2 + P5 §4.4), G (distributed refusal, from P7 §4.3 +
   D3 Stage 1), H (persona / Assistant-Axis + persona-shift compliance, from P5 §4.3 + P6 §4.2–4.3).
   No companion note. Skeletons land in W4-1; prose in W4-3.
5. **Raw-array release channel: Zenodo** (DOI of record). Public record = every per-unit array not
   derived from a non-commercial source; MORABLES-derived caches are excluded or restricted with a
   regeneration recipe. Plan in `deepsteer/supplement/RELEASE_PLAN.md`; DOI minted in W4-3.
6. **MN and FL submit as a pair.**
7. **Model assignments.** W4-1 (this session: rebuild, amendments, driver, skeletons, release plan,
   gate summary) = Fable 5.1. W4-2 (pod driver on RunPod, manifests, sync; no verdicts) = Sonnet 5.
   W4-3 (verdict rules, CLAIMS, prose, Zenodo DOI, hostile-reviewer pass, `make arxiv`) = Fable 5.1.

**Session plan (compute-ordering template).**

```
SESSION W4-2 (est. 5–8 A100-80GB h; batched by loaded model, in this order)
  model batch 1  OLMo-3-7B-Instruct         15.1 twins (pilot gate n=5 first) · 14.5 · 14.6 · 14.1 gate split-half
  model batch 2  OLMo-3-7B-Think            14.4 P0–P3 per-rollout (+ MFT dirs, refusal .npz)
  model batch 3  OLMo-3-7B base (main)      14.1 proto-refusal split-half (per-sample saves)
  model batch 4  Llama-3.1-8B-Instruct      14.2 severity/boundary contrasts + moral PCs · 14.6 · reply-inversion null
  model batch 5  GPT-OSS-20B                14.3 post-response decision token + band-below-null · 14.4 · 14.6
  model batch 6  Qwen2.5-7B-Instruct        15.2 C1 read cell · 14.6
  keystone:      15.1 (the flagship n) and 14.1 (the Tier-3 counter-reading) — both branches change FL wording
  pilot gates:   15.1 first 5 twins end-to-end (deltas finite, sign-coherent full cell); 15.2 screen >= 12
                 twins in band (else BOUNDARY=1; else bank indeterminate); 14.5 base refusal >= 0.40
  depends on:    W4 twin batch committed (deepsteer/datasets/request_twins_w4.py); Amendments 14/15
                 committed; Paper 5 stage-3 proto-refusal caches present locally for the zero-GPU arm
  saves:         per-unit arrays per Amendment 14/15 artifact lists; manifest JSON with sha256 + HF
                 revision hash actually loaded (fixes FL App E.4 "default branch")
  gate after:    W4-3 (verdicts) -> Gate W4-3 (Orion reads review + PDFs -> submit paired)
```

The zero-GPU-first layer is in Amendment 14.1: the adjacent-checkpoint arm is computable now from
Paper 5's cached per-checkpoint proto-refusal directions (`outputs/measurement/stage3/`, 14 states,
same construction as `refusal_base.npz`), so only the split-half arm needs the pod.

### Gate W4-1 — what Orion confirms before the pod runs (2026-09-10)

**Banked this session (commits 63cffcf … ):** FL rebuilt against the P3 arXiv id (MN verified);
this W4 section; D3 Amendments 14/15 (pre-registered before any extraction) with the SYNTHESIS branch
table; `scripts/pod_w4.py` + `scripts/w4/` (dry-run known to run end-to-end; 413 tests green) and
`scripts/remote_w4.sh`; the 48-twin W4 batch; `deepsteer.geometry.{reliability,participation}`;
FL appendix skeletons F/G/H with CLAIMS rows PB-01..07 and W4-01..10; `supplement/RELEASE_PLAN.md`;
the MISSING_ARTIFACTS closure map; the zero-GPU arm of 14.1.

**Zero-GPU result already in hand (14.1 drift arm; CLAIMS W4-01/02).** Cache-consistency control
0.99999998 (Paper 5 `olmo3_base` proto-refusal = D1 `refusal_base.npz`); adjacent-checkpoint
self-cosine 0.9999999 (stage3-step11900 vs 11921); proto-refusal crystallizes 0.93 (step 1000) → 1.0
(final) across the anneal; proto→gate cosine flat at 0.139–0.155 on all 13 states; covariance-matched
single-direction null q95 0.070 (descriptive rung). Reading: Branch A on the drift arm; the split-half
arm still bounds prompt-sampling reliability (identical prompts sit on both sides of the drift arm, so
it cannot). Formal Tier-3 verdict at W4-3.

**Confirm (decisions only Orion can make).**

1. **Amendment wording.** 14.1 disattenuation rider (instruct-gate split-half added; both
   reliabilities needed); 14.2 thresholds 0.25 / 0.50 over the control null; 14.3 `P_dec` definition
   (token before the first final-channel token); 14.5 bail at baseline refusal < 0.40; 15.1 the
   replication → alone → pooled rule with the sign agreement condition; 15.2 `indeterminate` as a
   publishable branch.
2. **Twin target n and the batch.** The 48 authored twins (`deepsteer/datasets/request_twins_w4.py`)
   need your read for construction fidelity (exact prefix, intent-carried harm, no alarming lexicon).
   At the recorded 38% screen pass rate the pooled n is ≈ 41. The power table says n = 40 tightens the
   shape CI by a third but does **not** resolve the ratio-of-ratios (≈ n = 140 would). Options: accept
   n ≈ 41 as the shape-only lift (default); author a second batch toward n ≈ 60 (+0.3 h, ratio still
   unresolved); or drop 15.1 and spend the hour on 14.4 breadth.
3. **Pod hour budget.** Per cell (A100-80GB): 14.1 0.2 · 14.2 0.3 · 14.3 0.3 · 14.4 0.9 (GPT-OSS 0.4 +
   Think 0.5; the handoff's 0.3 assumed saved rollouts that do not exist) · 14.5 0.4 · 14.6 0.4 · 15.1
   1.0 · 15.2 1.0 · six model loads ≈ 0.6 → **≈ 5.1 A100-h**, inside the 5–7 h envelope. 80 GB card is
   mandatory (GPT-OSS bf16 ~40 GB); DISK_GB ≥ 250 for six models.
4. **Model order.** OLMo-3-Instruct → OLMo-3-Think → OLMo-3 base (`main`) → Llama-3.1-8B-Instruct
   (gated; HF_TOKEN on the pod) → GPT-OSS-20B → Qwen2.5-7B-Instruct. "OLMo-3" is three loads; the
   handoff's four-model order is preserved at the family level.
5. **Two zero-cost promotions for W4-3 (need your yes).** (a) Promote the descriptive null rung to a
   verdict input by dated amendment, so the 0.155 rides a ladder (isotropic 0.012 / matched q95 0.070
   / measurement 0.155 / moral crystallization 0.999) instead of the bare 0.50 threshold. (b) Let the
   FL Tier-3 sentence carry the per-checkpoint trajectory (W4-02) as a second crystallization curve.

**Anticipated review (pre-review protocol).**

1. *Drift arm at 0.9999999 with the same 800 prompts on both sides* → estimator-traps (common-mode
   sampling) → the split-half is the load-bearing reliability, the drift arm bounds drift only.
   Implemented: both arms pre-registered as separate quantities; the write-up must not average them.
2. *0.155 is ~2× the matched null q95 (0.070), so "almost no pretraining precursor" is the wrong
   adjective regardless of reliability* → program-thesis anchored adjectives → the abstract clause
   becomes "a weak precursor (0.155; matched-null q95 0.07) against 0.999 for the moral subspace".
   Implemented: rung computed. Open: the amendment in (5a) (zero cost, W4-3).
3. *The W4 twins were authored knowing the plateau* → construct-audit register drift → set tags,
   alone/pooled sign rule, replication gate on the original 60. Open: none priced; a blind screen by
   a second author would add nothing computational.
4. *Think P3 stays unmeasured at cap 320* → the in-trace rung on Think is window-only whatever 14.4
   returns; the P3 hedge survives the pod by design. Implemented: stated in 14.4.
5. *14.5 harness parity: if the reconciled baseline lands near 0.40 the arrow is unpowered* → bail is
   pre-registered; Open: extend to the 400 harmful train prompts (+0.3 h) if the baseline is 0.40–0.50.
   Decide at the pilot read, not after the cell.

**Implemented now:** riders 1, 3, 4 (in the amendment text and the driver), the rung behind 2.
**Open (with costs):** 5a/5b amendments (0 h), second twin batch (+0.3 h), 14.5 extension (+0.3 h).
**Question behind the question:** whether ~5 A100-h buys venue quality depends on which objection a
referee leads with. The self-review's lead objections were n = 23 near the MDE, the Tier-3
counter-reading, and Qwen's empty read cell; 15.1, 14.1, and 15.2 answer exactly those three in both
branches, which is why they are the keystones. 14.2–14.4 remove hedges rather than change verdicts;
if the budget must shrink, they go first, in the order 14.4 (0.9 h) → 14.2 → 14.3.
