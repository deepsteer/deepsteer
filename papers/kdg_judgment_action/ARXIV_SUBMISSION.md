# arXiv submission: KDG paper (v1)

Prepared 2026-10-04 for the author's submission (author submits; this file is the metadata of record).
Source: commit of record at submission time; tarball `build/arxiv.tar.gz` (gitignored, rebuilt below).

## Fields

**Title.** Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment

**Authors.** Orion Reblitz-Richardson

**Abstract** (1,833 characters, plain text, no LaTeX macros; condensed from the PDF abstract, which is
2,456 characters, over arXiv's 1,920 limit; every sentence is the paper's own):

Language models increasingly act as agents. An agent that says an action is wrong and then takes it anyway is a different failure from one that does not know better, and evaluations of stated values cannot see it. We build a pre-registered panel of 248 scenarios across five kinds of pressure. Each scenario is posed twice to the same model, once as the agent choosing what to do and once in the third person asking which option is right, so the model's own judgment is the reference. Every scenario has a twin with the pressure removed, and every model gets a positive control in which its operator orders the violating action, so that a missing gap can be told apart from an instrument that cannot see one. On OLMo-3-7B-Instruct, the model takes the action it judged wrong on about one in five of the scenarios that pressure it, more often than on the same scenarios with the pressure removed. Reading the model in its own chat template also withdrew one of our earlier claims, that post-training makes it more cautious at rest. Across four instruct models the gap depends on the post-training recipe: OLMo-3 and Meta's Llama-3.1-8B-Instruct carry it, while Tulu 3 and Qwen2.5-7B-Instruct show none that the validated instrument detects (above about 0.01 to 0.02 in probability). Meta's recipe and Ai2's Tulu 3 start from the same Llama-3.1 weights, and only Meta's carries the gap. On both models that carry it, reasoning about the stakes before acting moves the choice back toward the model's own judgment, compared with a non-moral task of the same length; on OLMo-3, naming the norm at stake does about a third of that. The gap is a measurable target for post-training recipes, not a fixed property of pretrained weights. We release the panel, the harness, the pre-registration with its amendments, and the per-scenario arrays.

**Comments.** 30 pages
(The flagship's comments also listed figures, tables, the Zenodo DOI and the code URL; per the author,
this one is "30 pages" only. The code URL is in the paper.)

**Primary category.** cs.LG. **Cross-lists.** cs.AI, cs.CL. Matches the flagship (arXiv:2609.14759,
read from the arXiv API 2026-10-04); no reason to differ: same program, same audience.

**License.** CC BY 4.0, matching the flagship (its abstract page links creativecommons.org/licenses/by/4.0).

## Source bundle

- `\pdfoutput=1` on line 1 of `main.tex`; `main.bbl` included (no BibTeX run needed on arXiv); the eight
  figures as PDF under `figures/`, `\graphicspath{{figures/}}`; `neurips_2025.sty`; `sections/*.tex`.
- Built once from scratch from the extracted tarball (three pdflatex passes, no BibTeX): 0 errors,
  0 missing files, 0 undefined references, 30 pages, no Type 3 fonts. 32 entries, 265 KB.
- Rebuild: stage as in that check (the Makefile's `arxiv` target copies every file in `figures/` and
  runs latexmk, which is crash-prone on this machine; see KDG_GATES build notes).

## After the ID arrives

`python3 papers/kdg_judgment_action/apply_arxiv_id.py <ID>` (`--dry-run` prints the edits first): CITATION
entry, a Papers section in the root README, a Published section in papers/README (lists FL, MN and this
paper, since papers/README currently stops at Paper 4), a SYNTHESIS line, and the orionr.com
`publications.html` entry (the `orionr.github.io` repo; committed there separately).
