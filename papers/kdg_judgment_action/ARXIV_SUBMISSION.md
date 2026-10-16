# arXiv submission: KDG paper (v1 posted; v2 replacement prepared 2026-10-13, revised 2026-10-15)

**Posted: arXiv:2610.08670 (v1, 2026-10-06; title and author verified against the arXiv API).** `apply_arxiv_id.py` applied the same day.

Prepared 2026-10-04, updated through 2026-10-06 (pre-submit decisions, adversarial review, GPU-1, references) for the author's submission (author submits; this file is the metadata of record).
Source: commit of record at submission time; tarball `build/arxiv.tar.gz` (gitignored, rebuilt below).

## Fields

**Title.** Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment

**Authors.** Orion Reblitz-Richardson

**Abstract** (v2, revised 2026-10-15 after the external review, P1-A17: 1,916 characters, plain text, no
LaTeX macros; the same words as the PDF abstract, here as one paragraph. Changes from the 2026-10-13 text:
"posed twice to the same model, as the agent choosing what to do" became "posed to one model in two separate
calls, as the agent choosing among fixed options" (item 4: closed-menu choices in separate calls); "so no
outside ethics enters the measurement" became "and the scenario's construction fixes which option counts as
the violation" (item 4); "on about one scenario in five" became "on one in five of the scenarios screened on
its own actions" (item 2); to fit, the clause "the same weights can yield a model that carries the gap or one
that does not," was cut from the closing sentence, which the Meta/Tulu sentence already states (author to
confirm). Referee notes: "more often than with the pressure removed" now rests on the log-probability
readout, since the binary paired excess's two-way lower bound is at zero; "not detectably diminished" is
OLMo-3's panel-minus-rule-harms difference, -0.002 [-0.047, +0.045] with the panel arm two-way. The v1
abstract of record is in git history at 9659446.):

An agent that can say an action is wrong and then take it anyway is a different failure from one that does not know better, and evaluations of stated values cannot see it. We ask how often open models act against their own moral judgment under pressure, and what decides and reduces it. We build a pre-registered panel of 248 scenarios across five kinds of pressure. Each scenario is posed to one model in two separate calls, as the agent choosing among fixed options and in the third person asking which is right; the model's own judgment is the reference, and the scenario's construction fixes which option counts as the violation. Every scenario has a twin with the pressure removed, and every model gets a positive control that bounds what the instrument can see. On OLMo-3-7B-Instruct, the model takes the action it judged wrong on one in five of the scenarios screened on its own actions, more often than with the pressure removed. Whether the gap appears depends on the post-training recipe: OLMo-3 and Meta's Llama-3.1-8B-Instruct carry it; Tulu 3, Qwen2.5-7B-Instruct and OpenAI's GPT-OSS-20B, read without reasoning, show none within stated detection bars. Meta's recipe and Ai2's Tulu 3 share the same Llama-3.1 weights; only Meta's carries the gap. On both models that carry it, reasoning about the stakes before acting moves the choice back toward the model's own judgment. What that deliberation tracks is the stated rule: where following a written rule harms an absent third party, it still moves OLMo-3 and GPT-OSS-20B toward the rule, not detectably diminished, while Llama-3.1's brake shrinks to about a seventh. Whether a model acts on its own moral judgment is therefore not settled by pretraining, and deliberation moves the action toward the stated rule, which protects people only when the rule does. The gap is a measurable target for post-training recipes, not a fixed property of the model.

**Comments.** 40 pages (v2, 2026-10-15: v1 was 33; v2 adds the harness corrections, the GPT-OSS-20B row, the C0 note, §9.1, the estimands table, and the Appendix A/B/C/F additions of P1-A17)
(The flagship's comments also listed figures, tables, the Zenodo DOI and the code URL; per the author,
this one is the page count only. The code URL is in the paper.)

**Primary category.** cs.LG. **Cross-lists.** cs.AI, cs.CL. Matches the flagship (arXiv:2609.14759,
read from the arXiv API 2026-10-04); no reason to differ: same program, same audience.

**License.** CC BY 4.0, matching the flagship (its abstract page links creativecommons.org/licenses/by/4.0).

## Source bundle

- `\pdfoutput=1` on line 1 of `main.tex`; `main.bbl` included (no BibTeX run needed on arXiv); the nine
  figures as PDF under `figures/`, `\graphicspath{{figures/}}`; `neurips_2025.sty`; `sections/*.tex`.
- Built once from scratch from the extracted tarball (three pdflatex passes, no BibTeX): 0 errors,
  0 missing files, 0 undefined references, 33 pages, no Type 3 fonts. 33 entries, 296 KB (rebuilt 2026-10-06 at 9659446).
- **v2 build of record (2026-10-15, P1-A17 item 9).** The Makefile's latexmk call no longer ignores failures
  (the leading `-` is removed, so a failed build fails `make`). `make arxiv` rebuilt `build/arxiv.tar.gz`
  (40 files, 1,581,008 bytes, SHA-256 prefix 257f64e2040897b6; rebuilt 2026-10-16 with the Zenodo DOI 10.5281/zenodo.23425333 in Appendix F and three introduction semicolons removed); extracted into an empty temporary directory and
  compiled there with three `pdflatex -halt-on-error` passes and no BibTeX: every pass exit 0, 0 errors,
  0 undefined references, 0 missing files, 0 overfull boxes, 40 pages, no Type 3 fonts.
- Rebuild: stage as in that check (the Makefile's `arxiv` target copies every file in `figures/` and
  runs latexmk, which is crash-prone on this machine; see KDG_GATES build notes).

## After the ID arrives

`python3 papers/kdg_judgment_action/apply_arxiv_id.py <ID>` (`--dry-run` prints the edits first): CITATION
entry, a Papers section in the root README, a Published section in papers/README (lists FL, MN and this
paper, since papers/README currently stops at Paper 4), a SYNTHESIS line, and the orionr.com
`publications.html` entry (the `orionr.github.io` repo; committed there separately).
