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
