# FL — accumulated gates & feedback for Orion (review after the rough draft)

Per the write-up plan, Gate W2 (outline + figure list + Paper-B disposition) is being
**deferred** this pass: Orion asked to rough out ALL of FL to a reviewable PDF and collect the
gate/feedback decisions here rather than block. Decisions below are working choices; each is a
Gate-W2 item to confirm or change.

## Working title (Orion picks)
- **Working:** *Refusal Reads the Harm Percept, Not the Moral Subspace* (title case, no colon, claim-forward — the MN style you approved).
- Plan candidates (both have colons; the MN pass dropped colons): "What refusal reads: harm routing and commitment dynamics in open-weight language models"; "Refusal reads the harm percept: routing, bottlenecks, and reversibility across model families."
- Other no-colon options: *What Refusal Reads and How It Commits in Open-Weight Language Models*; *How Refusal Routes Around the Moral Subspace*.

## Structure (working) — confirm the beat→section mapping
1 Introduction · 2 Setup (models + instruments) · 3 Comprehension is pretraining-native (beat 1) ·
4 The refusal gate is a fresh add-on (beat 2) · 5 The decision site is a control-token bottleneck
(beat 3) · 6 Refusal ⊥ moral judgment at the decision (beat 4) · 7 What refusal reads: the harm
percept (beat 5, OLMo causal) · 8 Across families: what it reads × how it commits (beat 6) ·
9 Discussion · 10 Limitations · 11 Conclusion.

## Figures (working) — confirm the money-figure list
- Fig 1 bottleneck PR bar ×4 (reused from MN). Fig 2 crystallization 0.999-vs-0.155 (new).
  Fig 3 calibrated ladder (reused). Fig 4 OLMo one-knob sweep R_refusal(k) vs R_judgment(k) (new).
  Fig 5 depth collapse +0.82→−0.28 (reused). Fig 6 GPT-OSS graded-prefill monotone (new).
  Two-axis table (LaTeX, §8).

## Open decisions carried in
- **Paper-B disposition** (companion note / FL appendices / defer to the Direction-2 paper) — plan defers to Gate W2.
- **Bib**: assembled from the verified bibs of Papers 1/5/6/7; any citation the drafting flags «CITE:» (e.g. Constitutional-AI / RLHF / alignment-faking for the shallow-alignment framing) needs a verification pass before submission. FL **absorbs** P5/P6/P7 (does not cite them as separate papers); cites Paper 1 (published) + P3 (geometry) for the pretraining duo, and the methods note as "in preparation".
- **Safety scope** wording (characterization of released models; no removability optimization) — Orion sign-off.
- **NI flags** carried into prose: NI-2 (Llama PR 10.2 of record, 13.5 labeled), NI-3 (position valid/invalid reconciling sentence), NI-4 (R_refusal folded-primary vs standardized convention).

## Drafting flags (populated by the draft)
(the drafting agent appends «CITE:» / «CHECK:» items here)

- «CITE: Constitutional AI (Bai et al. 2022, "Constitutional AI: Harmlessness from AI
  Feedback", arXiv:2212.08073)» — used in §1 for the post-hoc-alignment framing alongside RLHF
  (`ouyang2022instructgpt`) and DPO (`rafailov2023dpo`). No bib key exists; a placeholder
  «CITE:» marker is left inline in `sections/01_introduction.md`. Add the verified key + swap
  the marker for `[@key]` before build.
- «CHECK: duplicate bib keys in build/references.bib» — `olmo3_2025` appears twice (an
  `@article` at ~L168 and an `@misc` at ~L317), `pew2025heretic` twice (both `@misc`, ~L312 and
  ~L400), and `wang2025persona` twice (~L303 `@misc`, ~L423 `@article`). bibtex emits "Repeated
  entry" and uses one arbitrarily; de-duplicate before the build. Cited-from-FL keys among these:
  `olmo3_2025` (§2) and `pew2025heretic` (§1/§4/§9); `wang2025persona` is not cited by FL but is a
  general bib hazard.
- «CHECK: three figures referenced but not yet in figures/» — `fl_crystallization.pdf` (§3,
  \Cref{fig:crystal}), `fl_one_knob.pdf` (§7, \Cref{fig:oneknob}), `fl_gpt_oss_reversibility.pdf`
  (§8, \Cref{fig:reversibility}) are being produced in parallel. `fl_bottleneck_pr.pdf`,
  `fl_calibration_ladder.pdf`, `fl_depth_collapse.pdf` are present. Build will have missing-graphic
  warnings until the three land.
- «CHECK: P4 preliminary numbers» — §7 keeps the P4 concept (foundation-specific causal moral
  directions, depth-strengthening ablation) as a one-sentence preliminary WITHOUT the specific
  OLMo-2-1B scalars (−0.16/−0.39/−0.63) and WITHOUT a self-citation, per the "no unpublished-paper
  refs except Paper 1 + geometry duo" rule. Confirm this framing is what Orion wants (the alternative
  is to drop the preliminary sentence entirely).
- «CHECK: absorbed-paper findings presented as this work's own» — the crystallization trajectory
  (§3), the dissociation/ablation numbers (§4), the Llama robustness anomaly −21σ (§8), and the
  GPT-OSS harm-audit + reasoning harmfulness cells (§8) come from the absorbed Papers 5/6/7 and are
  written as FL's own results (no self-cite), per the hard rule. External priors are cited where
  extended (`zhao2025harmfulness`, `arditi2024refusal`).

## Bib verification — DONE (2026-07-04)
All 21 cited entries checked against primary sources (arXiv abstract pages / the tool repo):
- **Verified exact** (title / ID / first author): arditi2024refusal (2406.11717), bai2022constitutional
  (2212.08073), ouyang2022instructgpt (NeurIPS 2022, 2203.02155), rafailov2023dpo (NeurIPS 2023,
  2305.18290), greenblatt2024faking (2412.14093), zhao2025harmfulness (2507.11878, "LLMs Encode
  Harmfulness and Refusal Separately"), grattafiori2024llama3 (2407.21783), openai2025gptoss
  (2508.10925), qwen2025qwen25 (2412.15115), olmo3_2025 (2512.13961), belrose2023leace (2306.03819),
  hendrycks2021ethics (2008.02275), park2024linear (2311.03658), zou2023repe (2310.01405),
  meng2022rome (2202.05262), hubinger2024sleeper (2401.05566), pew2025heretic (github.com/p-e-w/heretic).
- **Fixed:** grattafiori2024llama3 author was "Llama Team, AI @ Meta" → corrected to "Grattafiori,
  Aaron and others" (arXiv canonical first author; matches the key).
- **Not re-verified (low-risk, standard):** graham2013mft, haidt2012righteous (well-known
  moral-psychology book/journal, inherited from Paper 1's published bib).
- **Publication-status gate (not a verification issue):** reblitzrichardson2026geometry (P3, the
  pretraining-duo cite) is unpublished with no arXiv ID — either P3 goes to arXiv or the cite becomes
  "in preparation" before FL submission. reblitzrichardson2026fragility (Paper 1) is published (v2 in progress).

## Appendix drafting flags (populated 2026-07-04, appendices A–E)

- «CHECK: OLMo-3 instruct repo id.» The appendix write-brief proposed
  `allenai/Olmo-3-1025-7B-Instruct` (date-stamped) for the instruct model, but the repo id of
  record across the D-series docs and the D2 model table is `allenai/Olmo-3-7B-Instruct` (NO date
  stamp; the base is date-stamped `allenai/Olmo-3-1025-7B`, the instruct is not — a known Ai2
  naming inconsistency). App E (`0E_reproducibility.md` §E.4) uses `allenai/Olmo-3-7B-Instruct`,
  the id of record. Confirm against HF before submission (do not "fix" it to the date-stamped form).
- «CHECK: model revisions/pins.» No explicit checkpoint revision (commit/branch) is pinned in the
  D-series docs — drivers read the live model and `assert_matches_model` on layer/hidden/model_type
  rather than pinning a revision. App E §E.4 states "default branch at the pinned transformers
  version." If a specific revision was used for the A100 runs, add it to the E.4 table before
  submission.
- «CHECK: Qwen/Llama base-vs-instruct in §2.1 prose.» The setup section names the panel
  "Qwen2.5-7B" and "Llama-3.1-8B" (no `-Instruct` suffix) but describes four *chat* models; the
  D2/D3 runs of record use `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`. App E
  §E.4 lists the `-Instruct` repos (the actual runs). Confirm the §2.1 prose intends the instruct
  variants (this is a body-section wording call for the polishing agent, flagged here for
  consistency with the appendix repo table).
- «CHECK: OLMo hidden-dim / Qwen hidden-dim not stated in appendices.» OLMo-3 and Llama hidden 4096
  are confirmed in the D-series; Qwen2.5-7B and GPT-OSS (2880) hidden dims were NOT restated in the
  appendices to avoid inferring un-sourced values. If a hidden-size column is wanted in E.4, source
  Qwen's from the registry first.
- Numbers with a stated convention caveat carried into the appendix tables (all traceable, none
  invented): the base rank-3 null q95 is the calibration-ladder value of record (0.291, matching the
  body §4), not the 0.31 that appears in an older results table (NI-1). The Llama decision-channel
  participation ratio of record is 10.2 (in-format ladder), with 13.5 labeled as the second-position
  decision-token harness (NI-2) — both stated in App D §D.1. The OLMo refusal-transfer plateau uses
  the folded-primary values (R_refusal 0.31 at rank-3 peak, 0.27 at rank-16; ceiling 0.31), matching
  the body §7 and figure CSV (NI-4).

## Appendices + quality pass + figure restyle (2026-07-04)
- Five appendices added (A directions/stimuli, B calibration/nulls/controls [cites the methods note], C causal-anatomy tables, D per-model panel detail, E reproducibility) + the `\appendix` block in main.tex. FL is now 27 pp.
- Main text polished for reviewer clarity; heavy detail trimmed into `\Cref{app:*}` while headline numbers stay in-text.
- Figures (FL + MN) restyled to Paper 1's look (Material palette, descriptive suptitles + lettered panel titles, bold value labels, black bar edges, o-/s- markers). No data dropped.
- **Calibration-ladder figure corrected:** the reused two-position validity figure was replaced with the intended per-model refusal-below-band ladder (Base 0.33 / Instruct 0.14 / GPT-OSS 0.52, all below their moral-family bands), matching its caption. The MN keeps the two-position validity ladder (correct there).
- §2 now states the panel is the instruction-tuned checkpoints (the runs of record); short model forms used elsewhere.
