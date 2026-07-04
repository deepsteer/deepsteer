# CLAIMS.md — master claim ledger (W0, 2026-07-03)

Every number that appears in any paper draft must trace to a row here. One row per claim:
the **anchored sentence** is the exact wording (with numbers) that may enter prose; if a
draft states a number this ledger does not carry, the draft is wrong until a row is added.

Built at Phase W0 from: `d1_moral_subspace/{PREREGISTRATION,CALIBRATION_PREREG,
CALIBRATION_RESULTS,RESULTS}.md`, `d2_decision_coupling/{PREREGISTRATION,RESULTS}.md`,
`d3_decision_anatomy/{PREREGISTRATION,RESULTS,PAPER_PLAN}.md`, `ANOMALIES.md`,
`SYNTHESIS.md`, `MISSING_ARTIFACTS.md`, and `papers/{1..7}_*/sections/04_results.md`.

## Legend

**status** — `VERIFIED` (measured, gate/control passed) · `CI` (has a bootstrap/analytic CI,
stated) · `SCOPED` (true only under a stated restriction; the restriction ships in the
sentence) · `HELD` (not yet run / rides an unsaved artifact) · `VOID` (superseded or
retracted; listed in the VOID register with its replacement, never shipped as a finding).

**unit** — where the claim lands: `FL` flagship · `MN` methods note · `DUO` pretraining duo
(P1+P3, standalone; FL cites) · `HELD-P2` Paper 2 (held) · `PAPER-B` companion candidate
(disposition at Gate W2) · `REF` reference/motivation only (not a headline).

**artifact** — path the number traces to. `gitignored` = under `outputs/` (regeneration
script or committed `figure_data/*.csv` required at W3, plan §W3). `none stated` = the source
doc attaches no path (a W3 reproducibility rider, logged in OPEN_THREADS).

---

## ⚠ NUMBER-INTEGRITY FLAGS (pin before any prose uses these)

These are discrepancies surfaced during reconciliation. Each blocks the specific sentence
that cites it until the number of record is fixed. None blocks a *verdict* (the shapes and
signs are robust); all block *a printed scalar*.

| # | flag | competing values | resolution needed | blocks |
|---|---|---|---|---|
| NI-1 | **Base rank-3 V_moral null q95** | 0.291 (locked-framings amendment) / 0.308 (CALIBRATION_PREREG §1 table) / 0.31 (RESULTS headline table) | **RESOLVED (partial):** committed `a1_ladder_base.json` `null_q95 = 0.2913` → the calibration-ladder value of record is **0.291**; the RESULTS G3-table 0.31 is either drift or a *distinct* G3 rank-matched null (no separate `g3_base.json` located) — confirm which artifact the G3 headline cites at drafting. Verdict robust either way (0.33 < 0.291+0.05). | D1 G3 headline sentence |
| NI-2 | **Llama decision-channel PR** | 13.5 (D3 C1 decision token) / 10.2 (D2 in-format ladder, four-family list) | **RESOLVED:** 10.2 (D2 in-format ladder) is of record for the four-family bar (comparable to OLMo 14.7 / Qwen 8.6, all D2 ladder); 13.5 is the D3-C1 decision-token measurement (different harness), a second position. Both position-labeled; both < 30. | FL beat 3 four-architecture PR bar; MN bottleneck figure |
| NI-3 | **decision-channel "position-valid" vs "position-invalid"** | GPT-OSS PR 12.79 called "position-valid (25 ceiling)" in D3 RESULTS; PAPER_PLAN §1 calls OLMo/Qwen/Llama bottlenecks "all position-invalid"; prereg §5 rule = "PR<30 → position-invalid" | **RESOLVED (MN §2.1):** the bottleneck is position-**invalid for content projection-fraction** tests (band-below-null) but position-**valid for decision-direction** reads (R3 cosine, GPT-OSS refusal projection). The "25 ceiling" is a separate MoE PR sanity gate, not the content rule. | FL beat 3–4; MN §2.1 |
| NI-4 | **OLMo R_refusal / harm ceiling number drift** | folded-primary: R_refusal 0.31 (rank-3 peak) / 0.27 (rank-16 plateau), harm_rank1_R 0.31 · standardized-sweep: 0.33 ≈ harm 0.35 · A13 prose: 0.33 ≈ 0.355 | state a which-is-which convention (folded-primary is the headline; standardized is the robustness check); stop conflating rank-3 peak (0.31) with rank-16 plateau (0.27) as "the harm level" | FL beat 5; SYNTHESIS thesis paragraph |
| NI-5 | **Paper 1 §4.3 vs cited artifact `c3/RESULTS.md`** | paper: acc 0.740/0.750/0.750, σ* 7.38/5.63/6.94 · stale markdown: acc 0.812/0.802/0.802, σ* 10.0/9.42/10.0 | **RESOLVED:** the committed v2 c3 JSONs (`{narrative,declarative,general_control}_moral.json`) contain the paper's numbers — `final_probing.peak_accuracy` 0.740/0.750/0.750 and the σ* summary floats 7.3852/5.6351/6.9413 are present. The paper §4.3 is **artifact-backed and correct**; the stale `c3/RESULTS.md` markdown used a cap-at-max aggregation and just needs refreshing (doc hygiene). **NOT errata.** | DUO / P1 §4.3 |
| NI-6 | **OLMo-Think P2 near-miss magnitude** | "~0.009 below its null" (prose) vs table (P2 0.35, q95+M 0.354 → 0.004 below the *margin*, +0.046 above q95) | fix the prose to the table arithmetic; the verdict (near-miss NULL) is unaffected | D1 reasoning-gradient sentence |
| NI-7 | **ETHICS held-out n** | 197 (table) vs 199 (prose) | minor; pick one | D1 G2 coverage sentence |
| NI-8 | **minor rounding** | instruct persona c 0.506 vs 0.510; Paper-1 "Figure 4" name collision (§4.3 profiles vs `figure_4_rms_control.pdf`); compositional onset 4K (seed-42) vs 5K (4-seed-mean) | pick canonical values / renumber figure / always attach "4-seed-mean" qualifier | assorted |

---

## D1 — moral subspace / refusal geometry (→ FL beats 1–2, 4; MN)

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| D1-01 | G3 = NULL: base proto-refusal projects 0.33 onto base rank-3 V_moral (null q95 0.291 committed [NI-1], persona c 0.51) and the instruct gate projects 0.14 onto instruct V_moral (null q95 0.26, persona c 0.51); across the rank sweep (1→3 sources) refusal never clears q95+0.05. | VERIFIED | FL | `a1_ladder_base.json`; `outputs/phase2/*_g3*_result.json` (gitignored) | none |
| D1-02 | V_moral is the orthonormalized span of three distinguishable moral mean-diff directions (Moral Stories, Understanding Fables, ETHICS), cos(d_fables,d_moral)=0.53, cos(d_ethics,d_moral)=0.36, effective rank 3. | SCOPED (one rank-3 construction) | FL | `<tag>/moral_directions.npz`, `axis_directions.npz` (gitignored) | none |
| D1-03 | The instruct refusal gate projects 0.144 onto rich rank-3 V_moral (null q95 0.266) and 0.155 onto the 6-foundation MFT span (null q95 0.252) — both NULL; Paper 5's raw 0.1044 was never null-judged. | VERIFIED | FL | gitignored | none |
| D1-04 | The 6-foundation MFT span and rank-3 V_moral project onto each other at 0.56 / 0.62 (base) and 0.56 / 0.62 (instruct), inside the moral-family band and above persona — related-but-distinct, neither nests. | VERIFIED | FL | `a1_ladder_<tag>.json` (gitignored) | none |
| D1-05 | V_moral is richer but *lower*-dimensional than MFT (3 source directions vs 6; eff-dim 3 vs 4); its contribution is construct-diversity + distinguishability + contamination-resistance, not more dimensions. | VERIFIED | FL | gitignored | none |
| D1-06 | RMS-normalized Track-1 σ* on rank-3 V_moral is narrative 4.86 vs 0.0 (6-foundation MFT) and declarative 9.56 / 9.56 (tied): the MFT probe collapses at the smallest noise tested while V_moral survives to σ*=4.86. | VERIFIED | FL / MN | `base/track1_result.json` (gitignored) | none |
| D1-07 | G2 = PASS: Moral Stories narrative acc_surf 0.667 / acc_para 0.677 (gap −0.011); fables held-out 0.967/0.967 (+0.000); ETHICS held-out 0.701/0.761 (−0.061, n=197 [NI-7]); ETHICS extraction 0.787/0.813 (−0.026) — all clear, the direction reads structure not memorized text. | VERIFIED | FL | none stated | none |
| D1-08 | Held-one-out moral-family bands: base [0.537, 0.664], instruct [0.523, 0.637], think [0.537, 0.667], gpt_oss [0.649, 0.764]; every refusal point on every model lands below its tag's band (base band-min 95% CI [0.47, 0.53]). | VERIFIED | FL | `a1_ladder_<tag>.json`, `a4_bootstrap_<tag>.json` (gitignored) | a1_ladder.{png,pdf} |
| D1-09 | Refusal is sub-band with the paired Δ-CI excluding 0 at every point except the single GPT-OSS in-trace P2 window (Δ̂ 0.127, percentile CI [−0.029, 0.162] includes 0; BCa [0.089, 0.212] excludes 0) → Option 2 wording, locks at B3. | CI | FL | `a4_bootstrap_gpt_oss.json` (B=2000, seed 0, gitignored) | none |
| D1-10 | In both reasoning models the refusal projection is smallest at the gate, largest in-trace: OLMo-3-Think P1 0.10 / P0 0.29 / P2 0.35; GPT-OSS P1 0.19 / P0 0.47 / P2 0.52 / P3 0.25. | SCOPED (band-relative = cross-position; null-relative stands) | FL | `a1_ladder_<tag>.json` (gitignored) | a1_ladder.{png,pdf} |
| D1-11 | OLMo-Think in-trace P2 (0.35) is a near-miss below its rank-matched null margin [NI-6]; GPT-OSS P2 (0.52) crosses its null (0.32–0.34) but stays below persona (0.60) and the moral-family band [0.65, 0.76] — voice-adjacent, not moral-content-adjacent. | VERIFIED | FL | gitignored | a1_ladder.{png,pdf} |
| D1-12 | Combined in-trace P2 (Fisher, EXPLORATORY): Think p=0.015, GPT-OSS p=0.0005 → χ²(4)=23.5, p≈1.0e-4 — the in-trace peak is a real above-random effect, still sub-persona and sub-band. | SCOPED (exploratory, not a gate) | FL | `a4_combined_p2.json` (gitignored) | none |
| D1-13 | The wired refusal gate lives in a low-variance channel: instruct P_B at percentile 0.0 (≤q10) and all four GPT-OSS positions ≤0.2 (≤q10), while V_moral axes and persona sit at ordinary-to-high variance; the base proto-refusal is *not* narrow (percentile 37.4). | VERIFIED | FL | `a3_variance_percentile_<tag>.json` (gitignored) | none |
| D1-14 | Refusal does not crystallize from a pretraining precursor: cos(proto-refusal_base, refusal_instruct) @ L16 = 0.155, below the 0.50 trigger — contrast the moral subspace's cos(base, fresh) → 0.999 (Paper 5). The instruct gate is substantially a post-training construction. | VERIFIED | FL | `a5_proto_refusal_continuity.json` (gitignored) | none |
| D1-15 | A3 + A5 converge: refusal is a freshly-built, low-variance post-training gate (narrow in the wired gate, absent in the percentile-37 precursor), not a moral-content-derived direction — explaining both easy Heretic ablation and its below-null projection. | VERIFIED (synthesis) | FL | A3+A5 JSONs (gitignored) | none |
| D1-16 | Single-source moral salience is rank-1: Moral Stories yields one dominant moral direction carrying 7.5% of per-pair-difference variance atop a flat content tail — no low-rank moral subspace inside one source. | VERIFIED | REF/MN | none stated | none |
| D1-17 | Eff-dim thresholding on diff vectors measures content rank, not moral rank: the pooled-diff spectrum is elbow-less (singvals 31,18,15,14,…) so uncentered eff-dim@0.90 = 385, where refusal, persona, and random all project ~0.7–0.8 (degenerate for every direction). | VERIFIED (cautionary) | MN | none stated | none |
| D1-18 | Retro-audit (D3 routing lens): GPT-OSS in-trace P2 is harm-loaded — standardized \|cos(P2, d_harm)\| = 0.49 vs \|cos(P2, V_moral ⊥ d_harm)\| = 0.13 (raw 0.57 vs 0.22); standardization sharpens the gap 3.8×. | SCOPED (correlational, cross-position) | FL | `outputs/phase2/gpt_oss/harm_audit.json` (gitignored) | none |
| D1-19 | Prompt→trace harm consistency: at prompt P0 (t_inst) std \|cos(P0, d_harm)\| = 0.977 vs 0.001 (near-purely harm); in-trace P2 stays harm-dominant but attenuates (0.49 vs 0.13). | SCOPED (correlational) | FL | `harm_audit.json` (gitignored) | none |
| D1-20 | GPT-OSS V_moral is genuinely moral: separates moral/neutral at acc 0.67 with distinct source axes (cos 0.46–0.66); its elevated persona reflects general entanglement (moral↔persona cos 0.30 vs OLMo 0.24). | VERIFIED | FL | none stated | none |

## D2 — decision-vs-decision coupling / the bottleneck (→ FL beats 3–4; MN A2)

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| D2-01 | Refusal-decision ⊥ judgment-decision at the decision site on every model: \|cos\| 0.10 (OLMo, null q95 0.41), 0.32 (Qwen, 0.42), 0.08 (Llama, 0.51); margins 0.35 / 0.15 / 0.48 all clear the MDE (null q95 + M, M=0.05). | VERIFIED | FL | none stated (from `acts_headline`) | none |
| D2-02 | The chat decision-site token (final_pre_assistant) is a ~9–15-dim control-token bottleneck: participation ratio 14.7 (OLMo) / 8.6 (Qwen) / 10.2 (Llama) [NI-2], vs full-rank-healthy content positions (PR 40+/33+/35+). | VERIFIED | FL / MN | none stated | none |
| D2-03 | The bottleneck is position-*invalid for content projection tests*: the positive-control moral band [0.40, 0.47] sits below the covariance null 0.557, so any projection-fraction result there is uninterpretable — the band-below-null tell [NI-3]. | VERIFIED | FL / MN | none stated | none |
| D2-04 | Three independent estimates converge on ~15 dims: √(3/14.7)=0.45 ↔ null q95 0.557 ↔ the R3 pairwise-null 0.41–0.51. | VERIFIED (second-derivation) | MN | none stated | none |
| D2-05 | V_moral is FORMAT-ROBUST: at the valid mean_content position the in-format moral-family band matches the raw band on all three models (OLMo [0.54,0.64]≈[0.52,0.64]; Qwen [0.47,0.57]≈[0.46,0.56]; Llama [0.50,0.56]≈[0.44,0.54]). | VERIFIED (register-scoped reading VOID, see V-D2-1) | FL | none stated | none |
| D2-06 | R2/G3 (content projection at the decision site) is not well-posed cross-position: decision directions live only at the control-token bottleneck, content only at content positions, and they do not coexist at a valid position — the non-coexistence is the mechanism, not a limitation. | VERIFIED (re-typed; numbers non-verdict) | FL | none stated | none |
| D2-07 | Massive-activation outlier dims saturate the covariance null (Qwen dim 458 = 59% of variance, Llama dim 788 = 32%, OLMo top dim 1.4%), so raw Qwen/Llama R2/R3/R5 geometry is uninterpretable as run → standardized + in-format recompute (ANOMALIES A1). | VERIFIED (methods finding) | MN | `../ANOMALIES.md` A1 | none |
| D2-08 | Post-standardization the switch hides nothing positive: refusal at the (invalid) PRIMARY projects 0.497 < its 0.557 null; the PRIMARY was switched to mean_content by the pre-existing validity rule, not by outcome. | VERIFIED (audit) | MN | none stated | none |
| D2-09 | participation_ratio is a required type-block field; positions with PR < 30 are flagged position-invalid at extraction (all three decision sites, 14.7/8.6/10.2, fall below 30). | VERIFIED (protocol) | MN | none stated | none |

## D3 — decision anatomy, causal (→ FL beats 5–6; the flagship core)

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| D3-01 | On OLMo-3-7B-Instruct refusal is written into the ~13-dim decision-site channel by a distributed set of heads (led by L16 H23), cumulative channel-matched specificity 44% at the top 10 heads, ~62 heads for 80% (k hit its cap of 10). | VERIFIED | FL | `outputs/c1_session_olmo3.json`, `c1_inputs_olmo3.npz` (gitignored) | none |
| D3-02 | L16 H23 writes +0.742 onto refusal (channel-matched specificity +0.756), alone carrying 11.6% of total specificity; writers span layers 11–16; L15 H15 is the sole anti-refusal writer (−0.130). | VERIFIED | FL | `c1_inputs_olmo3.npz` (gitignored) | none |
| D3-03 | MLPs contribute 38% of the decision-site write (mlp_write_fraction 0.384, below the 0.50 Jacobian threshold, above the 0.23 the un-folded run reported). | VERIFIED | FL | `c1_session_olmo3.json` (gitignored) | none |
| D3-04 | Folding the per-layer RMSNorm gain brings Stage-1 reconstruction from 3.05 to 0.9999 (reordered_norm, two-sided band [0.90,1.10]); the LN-fold is exact (unit-tested to 1e-9). | VERIFIED (positive control) | FL / MN | `stage1_attribution.py`; `../ANOMALIES.md` A3 | none |
| D3-05 | All ten top writers are labeled `neither` — none clears the moral-family band, none is a clean copy-head-for-harm; V_moral fraction 0.15–0.28 with comparable harm loading, split into instruction-attenders and content-attenders. | VERIFIED (NULL on both hypotheses) | FL | `c1_inputs_olmo3.npz` (gitignored) | none |
| D3-06 | Powered decisive cells (n=23 request-twins): full→refusal −0.0833, V_moral-restricted→refusal −0.0282, complement→refusal −0.0636, harm-rank-1→refusal −0.0261, random-rank-3→refusal −0.0005, full→judgment +0.0459, restricted→judgment +0.0237 (refusal MDE 0.0238, judgment MDE 0.0086). | CI | FL | `c1_inputs_olmo3.npz` (gitignored) | none |
| D3-07 | V_moral is a *specific* refusal substrate: V_moral-restricted moves refusal more than a random rank-3 (Δ=0.031, paired 95% CI [0.020, 0.043], excludes 0). | CI | FL | `c1_inputs_olmo3.npz` (gitignored) | none |
| D3-08 | **HEADLINE — `harm_saturating`:** as k ∈ {1,3,8,16}, R_judgment climbs 0.05→0.46→0.59→0.66 while R_refusal saturates 0.01→0.31→0.26→0.27 at the harm-rank-1 level (harm_rank1_R 0.31); random-null ~0 at every rank; per-rank purity 0.97–0.99 [NI-4]. | VERIFIED (shape verdict) | FL | `c1_inputs_olmo3.npz` (gitignored) | one_knob_olmo3.png |
| D3-09 | ~73% of refusal's causal twin-difference input lies outside the rank-16 moral basis (1 − R_refusal(16); 69% at the rank-3 peak). | VERIFIED | FL | gitignored | none |
| D3-10 | Identification: harm-restricted (−0.0261) ≈ full V_moral (−0.0282), but the harm-partialed patch (V_moral ⊥ d_harm) still moves refusal −0.0133 (95% CI [−0.023, −0.005], excludes 0) — harm-dominant with a resolvable residual non-harm moral read; harm(V_moral)=0.46. | CI | FL | gitignored | none |
| D3-11 | PC1 — the highest-variance (purity 0.974), most harm-aligned (cos 0.35) contrast component — is causally inert: rank-1 restriction moves neither readout (R_refusal(1) 0.01, R_judgment(1) 0.05). Variance is not causal relevance (ANOMALIES A4). | VERIFIED | FL / MN | `../ANOMALIES.md` A4 | none |
| D3-12 | The sweep collapses to one free parameter: R_refusal(k) ≈ min(harm_ceiling, R_judgment(k)), ceiling ≈ 0.31, fitting the plateau (k≥3) at RMSE 0.036 (residual −0.002 at k=3) while harm-amplitude alternatives miss by 0.10–0.24; rank-1 over-predicts (measured 0.013 vs predicted 0.052, the A4 nonlinearity). | VERIFIED (flagship fit) | FL | gitignored | one_knob_olmo3.png |
| D3-13 | OLMo-3's refusal on intent-harmful requests reaches only ~17% at top severity (violating 0/0.17/0/0.17/0.17; benign 0), so the operating band is empty — a model property (weak intent-coupling) coherent with harm_saturating, not a stimulus artifact. | SCOPED (limitation-as-finding) | FL | `c1_session_olmo3.json` (gitignored) | none |
| D3-14 | Llama-3.1-8B anatomy is OLMo-like: pre-norm reconstruction 1.0008 (no fold), A1-clean decision channel (PR 13.5 [NI-2], null 0.148→0.114), distributed write, MLP 0.30, all writers `neither`; Llama's refusal tracks intent severity (baseline refusal 9/10). | VERIFIED | FL | none stated | none |
| D3-15 | Llama's refusal is directionally asymmetric at the boundary band (36 micro-graded twins): engage (add harm) +0.142, CI [+0.086, +0.212], coherent (sign-frac 0.81); disengage (remove harm) −0.014, CI [−0.084, +0.052], incoherent (sign-frac 0.51). | CI | FL | none stated | none |
| D3-16 | Llama's disengage is coherent at layers 8/12/14 (patch-layer sweep, Amendment 9: −0.12/−0.11/−0.20, CIs exclude 0; the depth-matched full cell at layer 12 reads −0.57, Amendment 10 — both coherent) but incoherent at layer 16 (−0.014) → EARLY-COMMITMENT (crystallizes before the read layer); OLMo's disengage is coherent at 16 (−0.62), so OLMo commits at/after the read layer. | CI / VERIFIED | FL | none stated | none |
| D3-17 | Cross-model asymmetry is depth-verified at matched layer 12: A_Llama = −0.28 (CI [−0.47, +0.03]), A_OLMo@12 = −0.54 (CI [−0.81, −0.32]), difference +0.26 — the read-layer +0.82 was a post-commitment artifact, so the asymmetry is a *consequence* of early-commitment, not a third property [see V-D3-8]. | CI (re-attribution) | FL / MN | none stated | none |
| D3-18 | At matched depth (layer 12) Llama reads BROAD (`broad_moral`: R_refusal 0.85 ≈ R_judgment 0.79, gap closes; harm-rank-1 only 0.59) while OLMo stays harm-keyed (R_refusal 0.43 < R_judgment 0.53, gap open). | VERIFIED | FL | none stated | none |
| D3-19 | The harm-coextensive alternative is rejected at rank 1: the request-twin d_harm spans only 3.6% of the engage-driving moral basis (engage weight on PC2–PC3 where d_harm captures 9.4% / 0.03%). The rank-2/4 severity-ladder version is a stated extraction rider (unsaved severity-twin contrasts). | SCOPED (rank-1 verified; rank-2/4 HELD) | FL | `harm_coextensive.py`; `sweep.harm_capture_curve` | none |
| D3-20 | GPT-OSS-20B position gate PASSES: the harmony END_OF_PROMPT decision channel has post-standardization PR 12.79, position-valid [NI-3]; across four families the decision site is a 9–15-dim control-token bottleneck (OLMo 14.7 / Qwen 8.6 / Llama 10.2 / GPT-OSS 12.79). | VERIFIED | FL / MN | none stated | none |
| D3-21 | GPT-OSS deliberation is consequential: an inculpating-analysis prefill flips unsaturated benign requests to refuse 7/7 (Wilson 95% [0.65, 1.0]). | CI | FL | none stated | none |
| D3-22 | **GPT-OSS is a REVERSIBLE READER:** graded exculpatory prefill flips ceiling-refusing violating items to comply 6/10, and the decision-channel refusal projection moves monotonically toward comply in all 10 items (frac_projection_moved 1.0, frac_monotone 1.0); the first-run disengage 0/7 was the A7 saturation trap (band_existence step_function, 5.6% mid-band). | VERIFIED (behavioral flip primary; projection corroborates w/ last-token caveat) | FL | none stated | none |
| D3-23 | The two-axis table (measured, stands): *what* refusal reads — OLMo & GPT-OSS harm (transfer < judgment, saturates), Llama broad moral (transfer ≈ judgment, gap closes); *how* it commits — OLMo at/after read layer, GPT-OSS reversible reader, Llama early. | VERIFIED | FL | composite | (two-axis table) |
| D3-24 | The dimensionality→reversibility hypothesis (A13): the three points are ordinally consistent (OLMo/GPT-OSS ~rank-1 harm → reversible; Llama ~rank-8 broad → early-commit), *licensing* "dimensionality of the refusal read → reversibility" as a falsifiable follow-on, but not confirming it — the split is architecture-confounded at n=3. | HELD (hypothesis, not result) | FL | composite | none |

## P4 — causal_validation (→ FL, "preliminary, fully absorbed"; D3 supersedes the numbers)

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| P4-01 | Concept (carries forward): moral directions are causal and foundation-specific — ablation specificity strengthens with depth (mean −0.16/−0.39/−0.63 at L4/8/12), steering is dose-response specific (low-α specific, high-α amplifying). | VERIFIED (concept; OLMo-2 1B) | FL (concept only) | Paper 4 §4.1–4.2 | placeholder (no PDF) |
| P4-02 | The specific P4 log-prob numbers and SAE-overlap cell are superseded by D3's OLMo-3 interchange + rank sweep; P4 has no standalone future (fully absorbed). | VOID-as-headline (concept retained) | FL | — | — |

## P5 — moral_alignment (→ FL, "dissociation core")

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| P5-01 | Comprehension is pretraining-native: all 25 OLMo-3 training states reach 100% probe accuracy, eff-dim 5, transfer AUC ≈ 1.0; direction-preservation cosine rises 0.869 (step 1000) → 0.999 (step 11921). | VERIFIED | FL | Paper 5 §4.1, three_curve | three_curve |
| P5-02 | Post-training reorients, does not re-teach: base→SFT direction cosine 0.999 → 0.757 (~40° rotation), then DPO 0.757 and all RLVR substeps 0.757–0.759 hold; eff-dim 5 throughout. | VERIFIED | FL | Paper 5 §4.2 | dendrogram_compare |
| P5-03 | Comprehension and compliance are only weakly coupled: φ −0.19→+0.02→+0.05 (SFT→DPO→Instruct); P(comply\|comprehend) 0.77 vs P(comply\|¬comprehend) 0.73; persona decodable ~0.94 but ⊥ morality (mean \|cos\| 0.076→0.085). | VERIFIED | FL | Paper 5 §4.3 | — |
| P5-04 | Refusal projects only 0.10 of its norm into the moral subspace (mean \|cos\| 0.06); ablating it drops refusal 0.25→0.00 while comprehension (base-to-fresh cosine 0.749, probe acc 1.0, eff-dim 5) and moral judgment (0.73 vs 0.75) are untouched. | VERIFIED | FL | Paper 5 §4.4–4.5, dissociation | dissociation, dissociation_2x2 |
| P5-05 | P5's "compliance is not routed through moral representations" ships **scoped** by D3 to "refusal reads only the harm sliver of V_moral (harm_saturating), a low-rank slice nearly orthogonal to the bulk" — a mechanistic refinement, not a contradiction. | SCOPED (by D3) | FL | Paper 5 + D3-08 | — |

## P6 — cross_model (→ FL, "representational cells + robustness anomaly")

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| P6-01 | Refusal is 98–99% residual and ⊥ morality in every family: moral projection fraction 0.104 (OLMo) / 0.127 (Qwen) / 0.071 (Llama), mean \|cos\| 0.04–0.075, single-direction AUC 1.00, single-vs-full-rank gap 0.000; reproduces P5's anchor (0.104 vs published 0.1044). | VERIFIED | FL | Paper 6 §4.1, phase1_decomposition | phase1_decomposition |
| P6-02 | Ablation preserves linear moral representation everywhere (probe acc 1.0→1.0, eff-dim 5→5); refusal removability is family-dependent: OLMo 0.575→0.000, Qwen 1.000→0.000, Llama only 0.900→0.475. | VERIFIED | FL | Paper 6 §4.2 | — |
| P6-03 | Llama-3.1 refusal is entangled with moral judgment (the robustness anomaly): judgment 0.75→0.604 at the best ablation layer, a −21σ outlier vs matched-random (0.747±0.007), dose-dependent (Spearman 1.0). | VERIFIED | FL | Paper 6 §4.4, llama_dose_response | llama_dose_response |
| P6-04 | P6's Llama anomaly ("a question for the next study") is RESOLVED upstream by D3: Llama is a broad-moral reader that early-commits (A_Llama −0.28 at matched depth, A_Llama−A_OLMo CI excludes 0, depth-verified). | VERIFIED (resolved) | FL | P6-03 + D3-16/17/18 | — |

## P7 — reasoning (→ FL, "decision-point/trace findings + distributed refusal")

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| P7-01 | Harmfulness and refusal are separately encoded at t_inst (d' 5.01 GPT-OSS / 4.39 Llama-distill / 5.01 Qwen-distill; cosine(harm t_inst, refusal t_post) 0.16/0.11/0.16), extending Zhao et al. to reasoning models incl. RL-deliberative GPT-OSS. | VERIFIED | FL | Paper 7 §4.1 | fig1_harmfulness_vs_refusal |
| P7-02 | Comprehension is distributed and displaced from the decision: end-of-prompt direction 99.1% residual, trace moral content peaks in the first third and is lowest at the decision (position 1.0). | VERIFIED | FL | Paper 7 §4.2 | fig4_trace_profile |
| P7-03 | Refusal is distributed on GPT-OSS: no single direction ablates it (end-of-prompt 4%, CoT-last 0%, CoT-mean 88% but only via incoherence). | VERIFIED | FL | Paper 7 §4.3 | fig5_distributed_refusal |
| P7-04 | Harmfulness is largely distinct from moral foundations: in-subspace fraction 0.18 (GPT-OSS) / 0.11 (distills), 3.0–3.9× the √(k/d)≈0.04 chance floor, ~85% outside the moral subspace — the same "reads a harm sliver" object D3 formalizes as harm_saturating. | VERIFIED | FL | Paper 7 §4.4 | fig3_harmfulness_vs_moral |
| P7-05 | Harmfulness is causally validated by reply-inversion: Qwen2.5-14B-Instruct shift +17.4 flips 33%, Llama-3.1-8B-Instruct +3.0 flips 23%; the earlier raw diff-of-means null (0.44–0.49 of residual norm) was a magnitude artifact. | VERIFIED | FL / MN | Paper 7 §4.5 | fig2_causal_validation |
| P7-06 | Reasoning models defeat clean judgment readouts (regex / final-answer / forced-logit), while Qwen2.5-14B-Instruct is clean on instruct (24/24 harmless-safe, 24/24 harmful-harmful) — the operating-point/readout lesson (MN A6). | VERIFIED | FL / MN | Paper 7 §4.6 | — |

## DUO — P1 (published) + P3 (pretraining duo, standalone; FL cites, does not absorb)

Paper 1 published-headline disposition (plan W0 addendum 1). Full P3 headline claims are
carried in PACKAGING §DUO; the load-bearing dispositions:

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| P1-01 | Staged emergence: moralized words onset step 1K < compositional moral step 5K (4-seed-mean, 0.709±0.025) < syntax step 6K; onset order tracks unigram TF-IDF floor. | CONFIRMED | DUO | `phase_c2/`, `phase_c4_compositional/` (gitignored) | Table 1 / Figure 1 |
| P1-02 | Compositional encoding is real, not bag-of-words: leave-construction-out transfer 0.848 (≈ in-dist 0.858) vs BoW 0.598; role_reversal (floor 0.57) decodes 0.85 (lift +0.28). | CONFIRMED | DUO | Figure 5; gitignored | Figure 5 |
| P1-03 | Accuracy saturates ~step 4K (0.586→0.942→0.953) while raw critical noise keeps evolving (0.52→18.31 peak→4.69). | CONFIRMED (as raw practical-sensitivity metric) | DUO | Table 2; gitignored | Figure 2/3 |
| P1-04 | The raw late-vs-early layer-depth fragility gradient (7–15×) is **largely activation scale**: under RMS normalization it attenuates to ~2× (1.8–3.1×) and the ordering fails at 8/37 checkpoints; the residual ~2× is **not claimed** as a genuine gradient (RMS controls scale, not covariance shape; whitened/PR-matched control not run — OT-9). | **ERRATA→v2, Template A approved 2026-07-04.** Original v1 raw-gradient-as-robustness reading is VOID (V-P1-1). | DUO | §4.4; `figure_4_rms_control.pdf` | figure_4_rms_control |
| P1-05 | The raw post-saturation fragility decline (σ* 18.3→4.7) is withdrawn: flat under RMS normalization (13.8→15.0), a scale artifact, not encoding change. | **ERRATA→v2, Template A approved 2026-07-04.** Original v1 "fragility keeps resolving structure" reading is VOID (V-P1-2). | DUO | §4.4 | — |
| P1-06 | Data curation reshapes robustness not accuracy: LoRA corpora identical in accuracy (0.740/0.750/0.750), differ in fragility (σ* 7.38/5.63/6.94), declarative fragile at 10/16 layers — matched-layer, scale-safe per §4.4. | CONFIRMED (artifact-backed by v2 c3 JSONs; stale c3/RESULTS.md to refresh, [NI-5]) | DUO | §4.3; v2 c3 JSONs (`{narrative,declarative,general_control}_moral.json`) | Figure 4 (§4.3) |
| P1-07 | Compositional-probe fragility decline is real, not single-seed: 4-seed gap 4.65−2.46 = 2.19 ≥ 1.0, max endpoint std 0.84 < 2.19 (pre-registered rule passes). | CONFIRMED | DUO | Table 3; gitignored | Table 3 |
| P3-01 | Foundations integrate rather than collapse: mean off-diagonal cosine 0.232–0.274 (uniformly positive), eff-dim 5 at every layer, PC1 0.379 of variance vs 0.179 random; the dendrogram does not recover the MFT individualizing/binding split (permutation min p=0.32). | CONFIRMED | DUO | Paper 3 §4.1–4.3 | fig1_cosine_heatmap, fig3_dendrogram |
| P3-02 | Fragility is cross-architecture not cross-foundation: dense σ* mean 5.0 vs MoE 2.2 (~2.3×), shrinking to ~1.2× under RMS normalization; binding−individualizing not significant. | CONFIRMED (RMS-dependent, linked to NI-5 family) | DUO | Paper 3 §4.7 | exp7_mean_critical_bars |
| P3-03 | 7B replication: cosine 0.193–0.244, eff-dim 5 at all 32 layers, dilemma matched 0.090 vs mismatched 0.032 (~2.8×); unsupervised clustering does not recover MFT (AMI peak 0.032). | CONFIRMED | DUO | Paper 3 scale_comparison_* | scale_comparison_* |

## HELD-P2 — moe_output_dilution (held pending SNR-normalized fragility fix)

| id | claim (anchored sentence) | status | unit | artifact | figure |
|---|---|---|---|---|---|
| P2-01 | OLMoE and OLMo-2 reach the same moral-probe accuracy (~99%) but OLMoE is 4.2× more fragile (mean σ* 0.92 vs 3.81); no expert moral specialization (1020/1024 probes >75%, Gini 0.016–0.023); router content-agnostic (max routing preference 1.8%). | HELD (raw-σ* framing pends SNR fix) | HELD-P2 | Paper 2 §4.1–4.3 | Figure 1/2 |
| P2-02 | Output dilution is the fragility mechanism: component σ* router 9.56 / expert 1.8 / aggregated 0.6; dense MLP output 74× larger than the MoE block; raw σ* framed as a "scale meter" (cites Paper 1's RMS control). | HELD (linked to NI-5 σ* family) | HELD-P2 | Paper 2 §4.4, Figure 3 | Figure 3 |

---

## VOID REGISTER (may be discussed as methods lessons in MN; never a finding in FL)

| id | voided claim | replacement | source |
|---|---|---|---|
| V-D1-1 | §5 eff-dim convention (uncentered eff-dim@0.90 of pooled diffs) | rank-3 span of {d_moral, d_fables, d_ethics} (eff-dim@0.90=385 is degenerate: null 0.80, refusal ~0.7–0.8) — retained as a cautionary finding (D1-17) | D1 prereg 2026-06-28 |
| V-D1-2 | G-AXIS MORABLES cross-source pooling (floor 0.67) | V_moral single-source then rank-3 re-spec; MORABLES dropped (CC-BY-NC + ~79% non-re-derivable) | D1 prereg 2026-06-27 |
| V-D1-3 | `</think>` detector using fixed 3-token subsequence [524, 27963, 29] | anchor on suffix [27963, 29] + validate `</` per occurrence (BPE merges the leading `</`) | D1 prereg 2026-06-29 |
| V-D1-4 | Naive full-reasoning-span P2 (in-trace mean over whole trace) | symmetric first-N-token window (N=256), same N both sides (closed-rate contrast otherwise) | D1 prereg 2026-06-30 |
| V-D1-5 | Persona as "non-moral semantic control" | "moral-adjacent voice reference" (persona 0.51 sits just below the band); strong-form "below a known non-moral axis" HELD for B3 (R5) | D1 calibration |
| V-D2-1 | V_moral is "register-scoped" (format_robust=false) | V_moral is FORMAT-ROBUST (invalid-position artifact at final_pre_assistant); band matches at valid mean_content (D2-05) | D2 Amendment 2 |
| V-D2-2 | Raw Qwen/Llama R2/R3/R5 geometry | standardized + top-k projection-out + in-format ladder (massive-activation null saturation; ANOMALIES A1); post-fix R3 0.32/0.08 survive | D2 Amendment 1 |
| V-D2-3 | R2/G3 as a coupling verdict | R3 decision-vs-decision cosine (D2-06); R2/G3 numbers exist but are non-verdict cross-position — do not leak as coupling numbers | D2 Amendment 2 |
| V-D2-4 | Cross-ablation R3(iii) at base refusal 0.167 | reconcile eval configs (harness drift vs Paper 6's 0.575), rerun on full harmful set with shared `_classify_response` — QUEUED (OPEN_THREADS) | D2 Amendment 1 |
| V-D3-1 | Un-folded Stage-1/2 anatomy (reconstruction 3.05, MLP 0.23) | RMSNorm-folded anatomy (0.9999, MLP 0.384); decisive patch cell unchanged | D3 Amendment 2 |
| V-D3-2 | n=11 `reads_non_vmoral_features` headline | `under_transfer` (n=23; absolute transport control was necessary-not-sufficient) | D3 Amendment 3 |
| V-D3-3 | `under_transfer` verdict | `harm_saturating` (rank sweep; D3-08) | D3 Amendment 4 |
| V-D3-4 | Llama R_refusal 0.44 (rank 16) vs harm-rank-1 0.14 "reads beyond harm" hint | VOIDED — denominator latched/saturated (ANOMALIES A5); three branches re-enter unweighted, resolved by depth-matched broad_moral (D3-18) | D3 Amendment 8 |
| V-D3-5 | Llama underpowered R_refusal_k / R_judgment_k>1 / ratio-of-ratios CI [−2.3, 4.9] | VOIDED as denominator-latched; the reverse (disengage) direction is the clean channel | D3 Amendment 8 |
| V-D3-6 | GPT-OSS first-run disengage 0/7 "irreversible" read | A7 saturation trap (empty boundary band); graded disengage 6/10 → reversible reader (D3-22) | D3 Amendment 12 |
| V-D3-7 | n=3 categorical "harm-readers reversible / broad-reader early-commits" co-occurrence framing | confound-named dimensionality→reversibility hypothesis (D3-24); the measured two-axis table (D3-23) is retained | D3 Amendment 13 |
| **V-D3-8** | **CANONICAL: A_Llama = +0.82 (read layer 16), engage-dominant/latch-like; A_Llama−A_OLMo = +1.03** | **depth-matched layer 12: A_Llama = −0.28, A_OLMo@12 = −0.54, difference +0.26; asymmetry is a consequence of early-commitment, not a third property (D3-17)** | D3 Amendments 10/11 |
| V-P4-1 | P4 specific log-prob ablation/SAE numbers (OLMo-2 1B) as headline causal evidence | D3 OLMo-3 interchange + rank sweep (harm_saturating); P4 concept (foundation-specific causality) retained, numbers superseded | packaging |
| **V-P1-1** | **v1 Finding 2 (arXiv:2606.11375v1): a layer-depth robustness gradient develops monotonically; late layers tolerate far more noise than early ones, read as encoding robustness** | the raw gradient (7–15×) is **largely activation scale**; under RMS it attenuates to ~2× (1.8–3.1×; ordering fails 8/37) and **no genuine residual gradient is claimed** (whitened/PR-matched control not run) → P1-04 | arXiv v2 §4.4 / W0-F1 |
| **V-P1-2** | **v1: the post-saturation evolution of critical noise (raw σ* 18.3→4.7) as continued representational change ("fragility keeps resolving structure")** | a **scale artifact**; flat under RMS (13.8→15.0); **withdrawn** → P1-05 | arXiv v2 §4.4 / W0-F1 |
