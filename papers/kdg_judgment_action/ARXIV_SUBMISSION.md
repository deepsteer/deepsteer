# arXiv submission: KDG paper (v1 posted; v2 replacement prepared 2026-10-13)

**Posted: arXiv:2610.08670 (v1, 2026-10-06; title and author verified against the arXiv API).** `apply_arxiv_id.py` applied the same day.

Prepared 2026-10-04, updated through 2026-10-06 (pre-submit decisions, adversarial review, GPU-1, references) for the author's submission (author submits; this file is the metadata of record).
Source: commit of record at submission time; tarball `build/arxiv.tar.gz` (gitignored, rebuilt below).

## Fields

**Title.** Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment

**Authors.** Orion Reblitz-Richardson

**Abstract** (v2, 2026-10-13: 1,920 characters, plain text, no LaTeX macros; the same words as the PDF abstract,
which the author rewrote for v2, here as one paragraph. Referee pass: "undiminished" became "not detectably diminished"
(OLMo-3's panel-vs-H1 difference, -0.002 [-0.042, +0.037], cannot exclude a reduction of about 0.04), which brings the
count to exactly the 1,920 limit; "of its panel size" after "about a seventh" was left out to fit (the body states it).
The v1 abstract of record is in git history at 9659446.):

An agent that can say an action is wrong and then take it anyway is a different failure from one that does not know better, and evaluations of stated values cannot see it. We ask how often open models act against their own moral judgment under pressure, and what decides and reduces it. We build a pre-registered panel of 248 scenarios across five kinds of pressure. Each scenario is posed twice to the same model, as the agent choosing what to do and in the third person asking which option is right; the model's own judgment is the reference, so no outside ethics enters the measurement. Every scenario has a twin with the pressure removed, and every model gets a positive control that bounds what the instrument can see. On OLMo-3-7B-Instruct, the model takes the action it judged wrong on about one scenario in five, more often than with the pressure removed. Whether the gap appears depends on the post-training recipe: OLMo-3 and Meta's Llama-3.1-8B-Instruct carry it; Tulu 3, Qwen2.5-7B-Instruct and OpenAI's GPT-OSS-20B, read without reasoning, show none within stated detection bars. Meta's recipe and Ai2's Tulu 3 share the same Llama-3.1 weights; only Meta's carries the gap. On both models that carry it, reasoning about the stakes before acting moves the choice back toward the model's own judgment. What that deliberation tracks is the stated rule: where following a written rule harms an absent third party, it still moves OLMo-3 and GPT-OSS-20B toward the rule, not detectably diminished, while Llama-3.1's brake shrinks to about a seventh. Whether a model acts on its own moral judgment is therefore not settled by pretraining: the same weights can yield a model that carries the gap or one that does not, and deliberation moves the action toward the stated rule, which protects people only when the rule does. The gap is a measurable target for post-training recipes, not a fixed property of the model.

**Comments.** 34 pages (v2, 2026-10-13: v1 was 33; v2 adds the harness corrections, the GPT-OSS-20B row, the C0 note and §9.1; check against the final PDF at submission)
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
- Rebuild: stage as in that check (the Makefile's `arxiv` target copies every file in `figures/` and
  runs latexmk, which is crash-prone on this machine; see KDG_GATES build notes).

## After the ID arrives

`python3 papers/kdg_judgment_action/apply_arxiv_id.py <ID>` (`--dry-run` prints the edits first): CITATION
entry, a Papers section in the root README, a Published section in papers/README (lists FL, MN and this
paper, since papers/README currently stops at Paper 4), a SYNTHESIS line, and the orionr.com
`publications.html` entry (the `orionr.github.io` repo; committed there separately).
