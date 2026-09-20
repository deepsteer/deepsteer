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
