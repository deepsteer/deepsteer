# Paper 8 — decisions for Orion after the 2026-09-19 upgrade pass

Working choices made in this pass; each is a gate item to confirm or change. The paper builds from
`sections/*.md` via `build/Makefile`; figures regenerate from committed data via
`figure_data/regen_kdg_figures.py`. Nothing here is pushed.

## 1. Title (author picks)

**Working title:** *A Language Model Acts Against Its Own Moral Judgment* — subtitle *A
Pre-Registered, Calibrated Panel of the Judgment–Action Gap on OLMo-3, Base and Instruct*.

Why the old title had to go: "Knowing Without Doing" is a near-duplicate of Huang et al. 2026's
title *Knowing But Not Doing* (arXiv:2601.07972), the closest prior panel; a reader would file this
paper as a replication of theirs. The *term* "knowing–doing gap" does match existing usage (Pfeffer
& Sutton 2000; Schmied et al. 2025; Cheng et al. 2026; Huang et al. 2026) and stays in the body at
first mention and as the internal identifier (KDG) in code and data. The paper's primary term is now
**judgment–action gap**, the moral-psychology name for exactly this construct (Blasi 1980,
*Psychological Bulletin* 88(1); verified), which also avoids the epistemic claim in "knowing" (the
panel measures a stated judgment, not knowledge).

Alternatives, all claim-forward, none colliding with prior titles:
- *The Judgment–Action Gap Is Present Before Alignment* (claim; rests on §7, which is one lineage)
- *What the Model Judges Right and What It Does* (descriptive; echoes the flagship's title form)
- *Judging Right, Acting Otherwise* (short; slightly informal for this program's register)
- *Pressure Moves Actions Away From Judgments, Before and After Alignment* (the §7 finding; long)

## 2. What changed in the science (needs your sign-off because it re-scopes a committed reading)

Amendment A17 (committed 250c8b5 before computation) gave the base-vs-instruct contrast the CIs it
never had. Result (`KDG_RESULTS.md` §13): the pressure-attributable raw-frame gap is present in base
(E 0.017 [0.012, 0.022]) **and larger after post-training** (0.046 vs 0.018 on the 192 shared
scenarios; paired Δ 0.028 [0.007, 0.049]; on the acting side), while post-training reverses the
no-pressure frame gap (+0.024 → −0.038). The 2026-09-15 reading "inherited, not larger after
post-training" rested on argmax rates without a null and is superseded; CLAIMS KDG-16/22 keep their
numbers as argmax readings and lose the interpretation; KDG-29..34 replace it; SYNTHESIS carries
the revised thesis sentence. Verdict wording ("widened") was fixed in A17 before the numbers.
**This is a change to a committed reading and is escalated here rather than decided.**

## 3. Anticipated review (the riders a hostile reviewer attaches), with what was done

1. *Seventeen amendments in six days is a lab notebook, not a pre-registration* → program-thesis /
   estimator-traps → the paper now states the split (eleven before any model data; four committed
   before the computation they license; two post-hoc forks with both verdicts) in §1 and Table A;
   the amendment table carries a "committed" column. (done)
2. *Your central novel claim had no confidence interval and the point estimate showed post-training
   reducing the gap* → A17 → paired CIs on both readouts, raw-frame null, side decomposition,
   selection check, slices; the finding changed and the paper says so. (done)
3. *The continuous readout was introduced after the binary one failed* → §3.5 now states the timing
   (committed after the strictest-level binary result was known, before any vector was read) and the
   coherence gate that could have voided it. (done)
4. *Family MDE 0.40 on a 0.19 effect tests nothing* → §6 states the MDE and the observed CI
   half-width; the continuous-instrument contrasts are reported as exploratory with the chance level
   (0.26 for six contrasts); KDG-A5 opened with a priced discriminator. (done; the pod is open)
5. *The judges are LLMs and the blind reader is the author* → §3.1 and §9 say so plainly. (done)
6. *The known-gap band changes the system-prompt slot, not only the instruction* → §3.3 states it;
   the same-slot control (operator prompt that instructs the consistent action) remains unrun (~5 min).
7. *The chat null (0.12) is huge; what is it?* → §3.3/§4 name the deliberation asymmetry (deliberated
   judgment vs immediate action) as a component of the null; the paired excess is immune, the
   absolute rate is not. (done)
8. *The instruct raw frame is format-invalid on 39% of scenario-frames* → §9 + selection check;
   KDG-A6 discriminator priced (~5 min). (open)

Implemented now: 1–5, 7. Open (with costs): 6 (~5 min pod), 8 (~5 min pod + zero-GPU leg), and the
items in §4 below. Question behind the question: whether the paper's third headline should be
"present before alignment" (robust, one-lineage) or "widened by alignment on the acting side" (the
pre-registered verdict, mechanistically richer, rests on the raw-frame instruct cell). The draft
leads with the first and carries the second with its rivals; the title does not commit to either.

## 4. Immediate scientific gaps and their prices (compute-ordering: zero-GPU first)

| gap | discriminator | cost | what it changes |
|---|---|---|---|
| Is the instruct model's negative no-pressure raw gap a persona default or a raw-frame artifact? (KDG-A6) | zero-GPU: p_D(twin) vs option mass on the shared 192; pod: letter-only chat-template J on the twins | 0 + ~5 min | keeps or drops the "post-training lowers the baseline" half of §7 |
| Is the widened acting-side sensitivity goal-following? | deliberation-dose arm with the filler control (three arms × 136 screened × 16 rollouts) | ~90 min A100 | the mechanism sentence in §8; whether the S1 cell looks at the decision token or the trace |
| Exploratory family structure on the continuous instrument (KDG-A5) | 16 more F3 and F5 primaries per generator (CLI subagents, no API) + one kdg2-profile pod; zero-GPU first: F3 tool-menu structure covariate | ~1 h GPU | Branch A in excess units (F5 not pressure-attributable) vs Branch C |
| Is "widened" a prompt-version effect? | version-stratified generation round (the same pod as above) | shared | scopes §7 to the later-written construction or not |
| One lineage for base-vs-instruct | Qwen2.5-7B base + instruct, raw cells only (forward passes) | ~2 h A100 | whether "lower baseline, higher pressure sensitivity" is alignment's signature or OLMo-3's |
| One model for the chat panel (Huang cross-model rival) | Tier 2 (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct), full ladder | ~9 h A100 | whether the model axis collapses |
| Known-gap band differs in the system-prompt slot | same-slot consistent-instruction control | ~5 min | bounds the band's construct validity |

Budget note: SYNTHESIS records compute and API budgets exhausted on 2026-09-15; every row above is
priced for the day they are funded; the two zero-GPU legs can run now.

## 5. Writing changes made (for the record)

- Structure: sections are claims, each opening with its sentence; intro leads with the external stake
  (agentic misalignment, alignment faking, the flagship's "knows more than refusal uses") and gives
  the argument in four steps; a prior-art table states the delta; discussion composes the three
  findings into one account with the goal-following rival named and priced.
- Terminology: "generator" everywhere (never "writer"/"provider"); pod names and amendment codes
  only in the appendices; "judgment–action gap" as the term, KDG as the identifier.
- Citations: the flagship result now cites the flagship (arXiv:2609.14759; the draft cited Paper 1,
  arXiv:2606.11375, which is the fragility paper); bib rebuilt with verified entries only (Blasi
  1980, Pfeffer & Sutton 2000, Schmied et al. 2025 added; duplicates removed).
- Figures: six, restyled to the Paper-1/flagship convention (Material palette, lettered panels,
  bold value labels, larger type); new data figure (per-scenario scatter) and new three-cell figure
  with CIs; the families figure's colliding axis labels fixed; every figure has a matched CSV.
- Tables: prior-art delta, readout type blocks, bias-direction audit, strictness decomposition,
  three-cell contrast, screen outcomes by family, F4 swap, three-cell detail and slices.

## 6. Still to confirm before arXiv

- Author sign-off on §2 above (re-scoping of the 2026-09-15 reading) and on the title.
- The per-rollout arrays (15 GB) are "available on request" in App F; a Zenodo deposit like the
  flagship's would let the sentence say "deposited".
- KDG-A5/A6 zero-GPU legs (an afternoon) before submission; both change wording, not verdicts.

## 7. Build notes (2026-09-19)

- **pdfTeX segfault, worked around.** This machine's TeX Live (2026basic) pdfTeX crashes with
  SIGSEGV (crash reports in `~/Library/Logs/DiagnosticReports/pdftex-*.ips`) whenever a natbib
  citation link straddles a page break; the console shows "\pdfendlink ended up in different
  nesting level than \pdfstartlink" just before the crash, and latexmk then loops because each
  crashed pass truncates `main.aux`. Bisected on §1 (no `\citep` → clean; no `\Cref` → still
  crashes). Workaround in `main.tex`: citation links are coloured but carry no link annotation
  (`\hyper@natlinkstart/end` redefined); cross-reference and URL links are unchanged. The
  flagship's preamble is otherwise identical, so the flagship would hit this on any edit that
  moves a citation onto a page boundary. Remove the workaround if TeX Live is updated.
- **Longtable + pending float clipped content.** The pandoc longtables for the small captioned
  tables (families, strictness decomposition, three-cell) interacted with a pending `[tbp]`
  figure to produce an overfull vbox that silently dropped half a page (the §5 verdict paragraph
  and Table 5 vanished from the render). Those three tables and the prior-art table are now raw
  `table` floats with `tabular` in the markdown, which also keeps them on one page. The readouts,
  bias, and appendix tables remain pandoc longtables and render correctly.
- Build from a clean state: `make -B -C build sections`, then `pdflatex`, `bibtex`, and pdflatex
  until stable (three passes). `latexmk` converges too once the workaround is in place.
- Typewriter text (`\texttt`, `\path`) is set from bitmap PK fonts (`ectt*`) on this install
  because `cm-super` is absent; the flagship has the same property. Installing `cm-super` gives
  Type 1 outlines for arXiv.


## 2026-09-28 — Session C gate: paper restructured to four claims; hold lifted

**Hold on the KDG paper: lifted** (author, 2026-09-28), with the restructured draft below.

Arc (author): (1) the gap exists, template-valid, on OLMo-3 (§4–§6); (2) the instrument withdrew our own
claim, and post-training stages do not resize the gap within OLMo-3 (§7); (3) the gap follows the
post-training recipe: four models, positive controls, the same-base Meta vs Tulu contrast (§8, new);
(4) moral deliberation reduces the gap on both recipes that carry it, with the salience share (§9, new).
Discussion, limitations and conclusion renumbered to §10–§12 and rewritten to the arc; abstract and §1
follow. Prose rules applied: "carries", never "installs"; no size comparison between the OLMo-3
(truncated) and Llama-3.1 (mostly completed) dose effects; Llama base's zero descriptive; every null
carries its positive-control number and detection bar. Builds clean, 27 pp.

**Title: two options (author picks); the working title no longer fits a four-model paper.**
- *Whether a Language Model Acts Against Its Own Moral Judgment Depends on How It Was Post-Trained*
  (claim-forward; leads with finding 3; long).
- *Acting Against One's Own Moral Judgment: A Calibrated Panel Across Four Post-Training Recipes*
  (descriptive; keeps the construct in front and states the scope).

Open, deferred (not gates): like-for-like dose run on a shared scenario set with a completion budget
(~3 GPU-h), required before any slide shows both dose numbers together; salience-control extension
(~2 GPU-h) behind it. The DPO-stage at-rest lean (KDG-A8, replicated on Tulu 3) entered the paper at the
draft gate (below).

## Draft gate (2026-09-28, author decisions, executed)

- **Title chosen:** *Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own
  Moral Judgment* (no subtitle).
- **Review item 1:** Tulu 3 (arXiv:2411.15124), Llama 3 (arXiv:2407.21783) and Qwen2.5
  (arXiv:2412.15115) entered the bibliography only after each was fetched at its primary source
  (arXiv API: title, authors as printed, id); cited in §8.
- **Review item 2:** Appendix A carries the Phase 1 amendments table (P1-A1..A9, commit timing, forks
  P1-A1 and P1-A5); the abstract cites seventeen panel and nine Phase 1 amendments.
- **Review item 3:** Figure 7 (§8) puts the known-gap positive control and the four gaps on one axis
  with the 0.10 validation bar, plus a zoomed panel with the bases (descriptive); Figure 8 (§9) shows the
  dose contrasts on separate scales (sizes not compared). CVD-validated palette (indigo/red, gray for
  controls and bases); data in `figure_data/kdg_recipe.csv`, `kdg_deliberation.csv`.
- **Review item 4:** the clarifying sentence (0.018 whole-panel vs 0.030 screened vs 0.19 majority-vote
  readout) is in §8 and the abstract.
- **KDG-A8 in:** §7 "What post-training does change" (DPO at-rest lean, model-free n = 586; RL step is
  sharpening) and §8 "The DPO-stage lean replicates" (Tulu 3). Scoped: at-rest only; the
  pressure-attributable part unchanged; mechanism deferred to the recipe paper's ablation.
- Build: 30 pp., zero undefined references, zero overfull boxes, no pdfTeX crash reports.
  **Next gate: author reads the PDF.**

## PDF review (2026-09-28, author via Fable, executed)

- §1: the "base-versus-instruct cell none of the prior panels contain" sentence is replaced by the three
  additions beyond Table 1 (per-model positive control, same-base two-recipe contrast, deliberation arm
  with truncation and norm-salience controls); Table 1's "this panel" row matches; the pods paragraph
  names the OLMo-3 pods plus the Phase 1 sessions (OLMo-3 stages, Llama-3.1 Meta, Tulu 3 at three stages,
  Qwen2.5).
- §8: Tulu 3 DPO E step reported as touching zero (0.006, 0.000 to 0.012) below its bar of ~0.008 at
  n = 586; RL step bar ~0.005.
- PDF metadata title and author set; Strakhov and Claude cited as "Strakhov and Claude (2025)" (bib second
  author "Claude", with the printed byline in the note); Appendix A notes that verdict rules are quoted as
  registered and the body says "carries". Build clean, 30 pp. **Next gate: author.**

## 2026-10-01: stage-claim wording (KDG-A12, P1-A10; author)

- **Number of record for stage claims:** the 586 set screened by no model (same diluted set at every
  stage; fair, conservative stage contrast). The 136 screened set is the secondary with its own per-step
  bars (DPO 0.019, RL 0.012; the earlier single "about 0.013" was the RL-step bar applied to both).
- **"Not resized" retired for "does not enlarge, on either readout"** (§1, §7, §10). Per unit of output
  scale DPO's step is a decrease (−0.039, −0.069 to −0.008), which §7 explains as arithmetic (sharpening
  without added pull); "on either readout" keeps §1/§10 from reading as a mechanism claim about DPO.
  The KDG-A8 sentence carries the same qualifier (required, else §7 contradicts it); §8's Tulu sentence
  is scoped to the probability scale. Rebuilt clean, 30 pp; new sentences checked in the PDF text.
- **RL step:** +0.004 at its 0.005 bar on probability (lower bound +0.00041, 10,000 resamples), +0.005
  (−0.016 to 0.026) per unit of scale; "adds" not written. P1-A10 reading: **unresolved** ("weaker" =
  weaker conclusion; author, 2026-10-01). The within-RL sweep (eight registered RL checkpoints, ~2.5
  A100-h, 586 set only) is priced into the F6–F8 pod plan as an optional OLMo-3 extra with its own
  pre-registration; decided at the pod gate.
- KDG-A13 (DPO per-scale sign opposite on OLMo-3 and Tulu 3; base and recipe confounded): ledger only, no
  prose; recipe-paper candidate beside RL-Zero.

## 2026-10-04: arXiv preparation (author sequencing: paper on arXiv before any Phase 2 generation)

- **Fonts:** `cm-super` installed in tlmgr user mode (`~/Library/texmf`; `updmap-user` map enabled), so
  typewriter text now embeds as Type 1 outlines. `pdffonts`: 17 Type 1 + 27 CID TrueType (matplotlib
  figures), **no Type 3**. The pdfTeX citation-link workaround stays. Clean build: 0 errors, 0 undefined
  references, 30 pp., no crash reports. Caveat for this machine: the user-level font map shadows the
  system one; after a system TeX Live update, run `updmap-user` again.
- **KDG-A6 zero-GPU leg:** already run as Z2 (KDG_RESULTS §14: no concentration at low option mass) and
  the claim was resolved R_b by the Session A pod; the paper withdraws the "lowers the baseline" reading
  (§7, §10). Nothing further to run.
- **KDG-A5 zero-GPU leg** (pre-registered 0b69263, run 2026-10-04): unresolved (Δ −0.05, −0.15 to 0.09,
  MDE 0.17; the prohibited tool is the only fast route in 36 of 40 screened F3 scenarios). §6 gains one
  sentence saying so; no verdict changes.
- **App F:** keeps "available on request"; the Zenodo deposit is a v2 item, not a v1 gate (author).
- **Scope fence:** nothing from Phase 2 (the 2026-10-02 commits: p2a, G2′, A4a/A4b) is in the paper
  (checked by grep over the sections).

## 2026-10-05: pre-submit decisions (author), executed

- **Withdrawn-claim framing removed everywhere** (abstract, §1 claims 2 and 5 and roadmap, §7 title and
  paragraph, §8, §10 heading and the persona sentence, §12): the finding stays as a methods result with its
  pre-registered test (reading a chat model outside its template reverses its at-rest sign, −0.038 raw
  against +0.055 under the template on OLMo-3; present on both Ai2 recipes, not on Meta's). §7 is titled
  "Before and after post-training, and a format check that changed a reading"; the §10 paragraph heading is
  "A format check that changed a reading". Grep of the PDF text and the source for "earlier claim",
  "withdr", "corrected", "demot": 0 (the parser-appendix "non-demoted option" reworded as "the option
  other than Y").
- **"In short." paragraphs** at the top of §7 (stage null with its bars in the same sentence and the
  positive-control clause; values from the 586 set of record: 0.015 / 0.014 / 0.018, step bars 0.008 and
  0.005) and §8 (as drafted).
- **Figure 6 replaced** by the template-valid stage profile (`kdg_stages`: 586 and 136 sets with CIs, the
  base as a gray raw-frame description, the final checkpoint's positive control beside on one axis); the
  raw-frame three-cell figure moved to Appendix D (now Figure 9) with a caption that opens with what it does
  not show. Figure data `figure_data/kdg_stages.csv`; zero GPU.
- **§10:** one sentence naming the direct test of the pretraining thesis (one recipe applied to bases that
  differ only in pretraining; a training experiment) as the recipe paper's experiment.
- **Correction found while drawing Figure 6:** the known-gap CI is 0.47 to 0.52 (0.4679, 0.5248), not
  "0.47 to 0.53" as first written on 2026-10-04 (0.525 rounded twice); fixed in §7.
- **Build and bundle:** clean (0 errors, 0 undefined, 0 overfull), 30 pp., no Type 3; tarball rebuilt (9
  figures, 33 entries, 291 KB) and compiled from a fresh extraction (0 errors, 0 missing). arXiv abstract
  1,896 characters after the swap (limit 1,920); `ARXIV_SUBMISSION.md` regenerated.

## 2026-10-05: adversarial review of 538d535 (Fable; R1–R8, P1–P5, GPU-1), executed

- **R1 (deliberation arm undifferenced):** §9 retitled "Moral deliberation before acting moves the action
  toward the model's judgment"; "What this says" states that the pressure-removed twins were not run under
  deliberation; §10's goal-following paragraph is conditional ("whether deliberation loosens it is open");
  §11 prices the twin cell (P1-A12, about three GPU-hours, not scheduled). Wording note: the suggested
  "a same-length non-moral task does not" was not adopted, because only reasoning minus filler was measured;
  §9 and Figure 8 say "further than" / "more than a same-length non-moral task does".
- **R2 (own-screen excess):** P1-A11 (c263eb2) and P1-A13 (1d2616d) pushed before each computation. Table 3
  gains a column, own screen: excess (n; bar); minus selection null. OLMo-3 0.084 (110; 0.04), 0.073;
  Llama 0.100 (118; 0.03), 0.143; Tulu 3 0.051 (52; 0.04), −0.004; Qwen2.5 0.187 (47; 0.10), 0.083
  (−0.028 to 0.195). Reading (ii) triggered: "show none on the whole panel" everywhere the split is stated.
- **R3:** §8 sentence that a recipe carrying the gap has one measured property, not a rank among recipes;
  abstract's chat-template sentence names no lab.
- **R4/R5/R8:** "about a third (0.22 to 0.53)" in §10 and §12; §11 7–8B scale sentence; "mechanism" →
  "dissociation" in §1.
- **R6 (number hygiene):** known-gap 0.497 (0.468 to 0.525) in every place (three decimals, avoiding the
  double-rounding risk); "354 above-floor primary–twin pairs". One rounding error found and fixed during
  the final check: Qwen2.5's own-screen mean is 0.18745, printed as 0.188 in §8, Table 3, KDG_RESULTS §23,
  CLAIMS KDG-58 and ANOMALIES; now 0.187. No other mismatches; two naming risks listed for the author
  (OLMo-3's positive control as 0.58 binary / 0.60 continuous / 0.497 letter-only; screened sets of 136 /
  110 / 130).
- **R7:** Table 1 caption says "to our reading"; the "no" entries were not re-verified paper by paper.
- **P1–P5:** abstract order (chat-template sentence after the same-base sentence; "with nothing at stake");
  §1 five steps as an enumerated list, Strakhov details in the Table 1 caption; "In short" for §4 and §9;
  in-figure headlines for Figures 1 and 8; "pressure-attributable excess" glossed at first use.
- **GPU-1:** P1-A12 pre-registered and priced (about 3.1 A100-h); waits for the author's go. **GPU-2:** none.
- **Build and bundle:** see ARXIV_SUBMISSION.md (31 pp., abstract counts there).

## 2026-10-05: decisions on 6a4a957 (author), executed

- **§9 wording:** "further than" / "more than a same-length non-moral task does" accepted.
- **Table 1, paper by paper** (full text of all nine read; caption keeps "to our reading" and now says
  chance-level floors are not counted): Huang "no; value-selection check"; Gu "one-factor prompt pairs; no";
  Backmann "neutral base game; no"; Cheng/Basu "random-steering controls (Basu); no"; Strakhov, Shen,
  Rakshit, Hosseini stay "no; no". Base-model column unchanged (Basu's Steerling-8B is a base model, but the
  gap is measured on Qwen2.5-7B-Instruct only).
- **GPU-1 (P1-A12) run** on pod p1d (A100-SXM4-80GB, 21:19–00:14 UTC, about 3.25 pod-hours; one aborted
  mis-launch pod before it, terminated during sync). Registered verdict: Llama branch (a), OLMo-3 branch (b)
  (vs the truncated filler: (a)). Post-hoc ratio fork P1-A14 (pushed 3500a2e before its CI; disclosure of the
  descriptive numbers seen first): **proportional on both models and both references**; one common ratio
  predicts the probability-scale ΔE on both. §9, §10, §11, §12, §1 step four and the abstract's last
  paragraph (PDF) and deliberation sentence (arXiv) updated: deliberation brakes the violating action by the
  same fraction with and without the pressure; not the incentive's pull specifically, at bars 0.21 / 0.30 on
  the log ratio. KDG_RESULTS §24 with referee pass; CLAIMS KDG-59 and scope notes on KDG-42 / KDG-53.
- **R2 propagation:** "Tulu 3 shows none on either read; Qwen2.5 shows none on the whole panel and is
  unresolved on its own screened scenarios (0.083, −0.028 to 0.195; n = 47, bar about 0.16)" in the abstract
  (PDF and arXiv), §1, §8, §10, §11, §12 and the Figure 7 headline; §8 says what the selection-matched null is
  and why; KDG-A19 open with its price (about 1,700 scenarios and one A100-hour), price in §11. Author
  confirmed: bar 0.16 (the 0.083's own), 0.497 on 360 union primaries, 130 = 136 less six F2.
- **Two sentences, no number changes:** §8 (the control on three readouts: 0.58 / 0.60 / 0.497) and §3.2
  (136 / 110 / 130).
- **Amendment record:** Appendix A's Phase 1 table gains P1-A10 to P1-A13 with push hashes (A13 labelled
  post-hoc); P1-A14 added after the GPU-1 result; counts recounted in §1, §11 and Appendix A (fourteen Phase 1: six
  before data, four before computation, three post-hoc forks, one post-hoc addition; four post-review).
- **p2b:** nothing in the paper; SYNTHESIS and the pitch-edits draft carry the token-ladder state of record and
  no "does not fade" line; KDG-A16 logged unresolved.

## 2026-10-05: pre-submit deliberation wording fix (author, on c1516d3), executed

- §10 heading: "Goal-following is the parsimonious mechanism; deliberation brakes the action, with or
  without the incentive." Body, §9 In short and "What this says", §1 step four and §12: the pre-registered
  branch per model first (Llama (a), OLMo-3 (b)), then the post-hoc ratio reading (P1-A14) as consistent with
  one proportional cut on both, not separating a pressure-specific part smaller than about a fifth (OLMo-3) or
  a quarter (Llama) of the brake. "Not the goal" / "not the incentive specifically" / "by the same fraction"
  removed from the paper (supersedes the c1516d3 wording recorded above); RESULTS §24, SYNTHESIS and CLAIMS
  follow the same rule.
- Abstract, both versions: "...with or without the pressure"; PDF last paragraph "...brakes the violating
  action with or without the incentive." "The gap is a measurable target..." stays as the closing sentence;
  the release is stated in the paper.

## 2026-10-06: abstract release sentence removed; references verified (author request)

- "We release the panel, the harness, the pre-registration with its amendments, and the per-scenario
  arrays." removed from the PDF abstract (the arXiv version already lacked it). The paper had no repository
  URL anywhere; Appendix F now gives <https://github.com/deepsteer/deepsteer/> (public, HTTP 200), as FL and
  the methods note do. Appendix F also brought up to date: round-3 scenario files, the multi-model driver,
  runner, spec (P1-A1 to P1-A14) and analysis scripts, every model read with `models.yaml` pinning, the
  calibration v1 filename, KDG_RESULTS sections 1 to 24, and bootstrap draws (2,000 panel; 10,000 Phase 1).
  Every file named there is tracked.
- **References, all 24 checked against primary sources** (arXiv API, ACL Anthology, PMLR, Crossref, NeurIPS
  proceedings, Hugging Face model card, Open Library, the values.md page): no fabricated or misordered author
  lists. Fixed: `pan2023machiavelli` cited ICML with arXiv's 10-author list; now the PMLR record (9 authors,
  no Jonathan Ng; vol. 202, pp. 26837–26867; PMLR URL; arXiv ID in the note), both lists confirmed directly;
  `olmo3_2025` corporate author "Team OLMo" → "Team Olmo" with Allyson Ettinger, per the official citation on
  the Olmo-3-7B-Instruct model card. Added: DOIs for Shen et al. (ACL), Shao et al. (NeurIPS, with vol. 37,
  pp. 89373–89407) and Blasi (APA). Every URL and DOI loads (HTTP 200; the APA DOI returns 403 to scripts
  and 200 to a browser). Left as they are (correct as cited): arXiv citations for Sharma et al. (also ICLR 2024)
  and Tulu 3 (also COLM 2025); "and others" truncations for Greenblatt, Hubinger, Llama 3 and Qwen2.5.
- Not changed (other papers): FL and Paper 1 also cite Olmo 3 as "Team OLMo".
- Build 33 pp., no Type 3; bundle rebuilt and fresh-compiled (33 entries, 296 KB); PDF abstract 2,610
  characters, arXiv abstract 1,887 (unchanged).
