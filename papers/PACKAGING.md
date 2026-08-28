# PACKAGING.md — old→new provenance map (W0, 2026-07-03)

How the seven chronological papers (1–7) and the D1→D3 program repackage into the
publication units. Package by claim, not chronology (program-thesis principle). The
prereg/amendment trails stay as repo documents both new papers cite — a public
pre-registration trail is a credibility asset, referenced explicitly.

Claim ids below refer to `papers/CLAIMS.md`.

## §0 — Unit map (supersedes the plan's §1 table; verified against results text)

| unit | thesis | absorbs / draws on | status |
|---|---|---|---|
| **MN** — methods note (*Instruments before verdicts*, `METHODS_NOTE.md`) | interpretability instruments fail in specific, diagnosable ways: calibrate → certify with an orthogonal cell → power before pod → depth-indexed verdicts | ANOMALIES A1–A6; the estimator/intervention patterns; the 4-architecture bottleneck as motivating discovery; P1 §4.4 + P2 "scale meter" cited as prior art on scale confounds (thematic, **not** formal absorption) | skeleton exists (A1–A6 landed) → W1 |
| **FL** — flagship (routing + commitment) | refusal reads the harm percept through a narrow control-token bottleneck; families differ in what refusal reads × how it commits | D1 (calibration, crystallization, P0–P3), D2 (decision-vs-decision, bottleneck, format-robustness), D3 (anatomy, sweeps, two-axis panel); P4 (preliminary causal, concept-absorbed), P5 (dissociation core), P6 (representational cells + robustness anomaly), P7 (decision-point/trace + distributed refusal) | outline → W2 |
| **DUO** — P1 + P3 | pretraining duo (emergence; competing frameworks) | untouched; FL cites P1 (published) for beat 1 | done / standalone (P1 v2 gated, see OPEN_THREADS OT-1) |
| **HELD-P2** | MoE dilution | standalone, held pending SNR-normalized fragility fix | held (OT-2) |
| **PAPER-B** | behavioral/interventional companion | disposition at Gate W2 | candidates below |

**Double-claiming rule.** The bottleneck *finding* (PR 9–15 across OLMo/Qwen/Llama/GPT-OSS)
lives in FL (D2-02, D3-20); the *validity protocol* it motivated (band-below-null tell, PR
gate, standardization + invariance proof) lives in MN (A2). One cross-reference each way;
neither paper claims the other's contribution as novel.

---

## §1 — MN section-level provenance

Each per-anomaly section = failure as it first appeared → the tell → the protocol → the
check that certifies the fix (real numbers; the case studies are the paper).

| MN section | draws on | key claim ids / numbers |
|---|---|---|
| 1. Intro — the instrument problem | SYNTHESIS "Method spine"; the 4-architecture bottleneck as motivating discovery | D2-02, D3-20 (PR 14.7/8.6/10.2/12.79) |
| 2. Decision-site instrument + calibration ladder | ANOMALIES A2, A1, A3; D2 RESULTS; D1 CALIBRATION_RESULTS (the calibrated ladder exemplar) | D2-03 (band-below-null 0.557), D2-04 (three-estimate convergence), D3-04 (LN-fold 3.05→0.9999), D1-08 (ladder) |
| 3. Verdict discipline — ratio-of-ratios, power tables, orthogonal-cell certificate | ANOMALIES A5(2); D3 Amendments 3/6/7; the ratio-of-ratios reclassification | D3-06/07 (MDE cells), V-D3-2 (under_transfer), A5 power-table (Llama bounded-unresolved) |
| 4. Stimulus discipline — operating point, severity ladders, boundary bands | ANOMALIES A5(2), A6; D3 Amendments 7/8/12; P7 §4.6 readout failure | D3-13 (empty band), D3-15 (boundary twins), D3-22 (graded disengage), P7-06 |
| 5. Depth discipline — commitment-relative verdicts | ANOMALIES (A5 depth note); D3 Amendments 9/10 | D3-16/17 (early-commitment; +0.82→−0.28), V-D3-8 |
| 6. Case study — the D3 program, each amendment as a caught failure | D3 PREREGISTRATION Amendments 1–13; D3 RESULTS | whole D3 block; VOID register V-D3-1..8 |
| 7. Checklist — prereg + verification gates as a reusable protocol | CLAUDE.md hard gates; the five skills | (appendix; ship-blocker checklists) |

MN prior-art cross-refs (not absorption): P1 §4.4 (RMS activation-scale confound, D1-06
context / P1-04/05) and P2 "scale meter" (P2-02) are the same "instruments before verdicts"
lesson, kept inside P1/P2. One MN sentence cites them as prior art.

Money figures (MN originals): bottleneck PR bar ×4 architectures; band-below-null ladder
example (D1 a1_ladder); the +0.82→−0.28 depth-collapse as the depth-indexed exemplar.

---

## §2 — FL section-level provenance (the seven-beat arc)

| FL beat | thesis sentence | draws on (source docs) | key claim ids | figure |
|---|---|---|---|---|
| 1. Moral comprehension is pretraining-native and survives alignment | crystallization cos → 0.999 | P5 §4.1–4.2 (three_curve), P3/P1 cite, D1 (subspace construction) | P5-01/02, P3-01, D1-02/05 | three_curve (0.999) |
| 2. The refusal gate is a fresh post-training construction in a low-variance channel | proto-refusal→gate cos 0.155 | D1 CALIBRATION_RESULTS A3/A5; P5 §4.4 | D1-13/14/15, P5-04 | 0.999-vs-0.155 pair |
| 3. The decision site is a ~9–15-dim control-token bottleneck on four architectures | content and decision never co-locate → orthogonality structurally favored | D2 RESULTS + Amendment 2 (position-validity protocol, MN-cited); D3-20 (GPT-OSS) | D2-02/03/09, D3-20 [NI-2, NI-3] | bottleneck PR bar ×4 |
| 4. Decision-vs-decision: refusal ⊥ moral-judgment decisions panel-wide | stated with detection bars + calibrated bands | D2 RESULTS (R3); D1-08 bands | D2-01, D1-08 | — |
| 5. Causal anatomy (OLMo): distributed write into the channel; interchange + rank sweep → harm_saturating | judgment reads broadly on the *same* patches (within-model readability proof) | D3 RESULTS (Stage 1/2, decisive cell, sweep, one-knob) | D3-01..12 [NI-4] | one_knob_olmo3 |
| 6. Cross-model two-axis panel | Llama reads broad + early-commits; GPT-OSS reads harm + reversible | D3 RESULTS (Llama arc A6–A11, GPT-OSS A5/A12); P6 §4.4 (Llama anomaly, resolved); P7 (GPT-OSS inputs) | D3-14..24, P6-03/04, P7-01..05 [NI-2] | llama_dose_response, GPT-OSS graded panel, two-axis table |
| 7. Implications | shallow-alignment mechanism; Direction-2 target; deliberation load-bearing & reversible; safety scope | SYNTHESIS thesis + "what is settled vs open"; D1-18/19 (GPT-OSS correlational) | D3-23/24, D1-18/19 | — |

**FL absorbs — verified per-paper (see CLAIMS for the sentences):**
- **P4 → concept only.** Foundation-specific causal directions + dose-response (P4-01) carry
  forward as a preliminary; the OLMo-2 1B log-prob/SAE numbers are superseded by D3
  (V-P4-1). "Fully absorbed" confirmed: P4 has no standalone future.
- **P5 → dissociation core, scoped.** P5-01..04 are the single-model seed of the D-series;
  P5's "not routed through morality" ships scoped by D3's harm_saturating (P5-05).
- **P6 → representational cells + a now-answered anomaly.** The three-family decomposition
  (P6-01/02) is FL beat 4's cross-model backbone; the Llama entanglement anomaly (P6-03) is
  resolved upstream by D3 (P6-04) and is also the MN depth-indexed exemplar.
- **P7 → decision-point/trace + distributed refusal.** P7-01..05 carry forward; P7's open
  GPT-OSS commit question is resolved by D3 Amendment 12 (D3-22); P7-06 feeds MN A6.

Limitations FL must state (plan §W2): n=3 architecture confound + the one-axis rival
(D3-24); GPT-OSS reads-axis is correlational, Tier-2 held (OT-8); readout-vs-behavior scope
per cell; the prefill-last-token caveat (D3-22); stimulus-composition covariates across
bands.

---

## §3 — DUO (P1 + P3) provenance

Standalone pretraining duo; FL cites P1 for beat 1 but does not absorb it. P1 is published
(arXiv:2606.11375; v2 with the §4.4 scale control submitted 25 Aug 2026); P2 is published
(arXiv:2608.25231, 25 Aug 2026); P3 is published (arXiv:2608.27402, 27 Aug 2026); MN is
upload-ready, held to submit as a pair with FL. **W0 resolved the v1→v2 question: a v2
is required** — v1 lacks §4.4, so P1-04/P1-05 are errata-class (OT-1/F1); the erratum wording
(it scopes abstract-level Finding 2) is escalated to Orion and the build is W3. Bib and §4.3 are
both clean in v1. Claim ids P1-01..07, P3-01..03. The σ*-normalization control (P1 §4.4) links
P1, P3, and HELD-P2 into one claim family.

## §4 — PAPER-B candidates (disposition at Gate W2)

Behavioral/interventional companion. Candidates, each with a disposition option
(a companion note now / b FL appendix / c defer to the Direction-2 intervention paper):

| candidate | source | note |
|---|---|---|
| ART forced-coupling arc + sign flip | prior repo (Phase 3) + D2 motivating evidence D2-motiv | interventional; degenerate result is itself a finding |
| F2 rotation + specificity control | P5 §4.2 + D2 R6 (held) | needs R6 (≥15° rule), B3 |
| persona / Assistant-Axis + persona-shift compliance | P5 §4.3, P6 §4.3 | behavioral dissociation |
| removability battery detail | P6 §4.2 | family-dependent removability |
| distributed-refusal detail | P7 §4.3, D3 Stage 1 | anatomy |
| B5 fragility baseline (if run) | D2 R8 (held) | standing Direction-2 metric |

Recommendation deferred to Gate W2 per plan; default = FL appendices for the ones already
cited (removability, distributed-refusal), companion note only if the ART arc is wanted as a
standalone interventional result.

---

## §5 — Repo documents both papers cite (the pre-registration trail)

- `d1_moral_subspace/{PREREGISTRATION, CALIBRATION_PREREG, CALIBRATION_RESULTS, RESULTS}.md`
- `d2_decision_coupling/{PREREGISTRATION, RESULTS}.md`
- `d3_decision_anatomy/{PREREGISTRATION (Amendments 1–13), RESULTS, PAPER_PLAN}.md`
- `ANOMALIES.md` (A1–A6, the MN backbone), `SYNTHESIS.md` (living abstract source),
  `MISSING_ARTIFACTS.md`, `CLAIMS.md`, `OPEN_THREADS.md`.
