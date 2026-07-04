# Write-up & Packaging Phase — Plan WP-1 (2026-07-03)

**For:** Claude Code (Opus 4.8), cold start after `/clear`. Context = this document + the
committed repo. First action every session: Run CLAUDE.md "Research boot sequence" then this plan.
**Phase mode: experiments are FROZEN.** Zero-GPU claim-verification only. Any tempting
experiment goes to OPEN_THREADS.md with a cost estimate and waits for a gate — no pods this
phase. The research program (D1→D2→D3, three-model panel, two-axis resolution) is complete
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
