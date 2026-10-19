---
name: paper-prose
description: >-
  Strip AI-writing tells from DeepSteer release prose. Use this skill before committing changes to
  papers/*/sections/, READMEs, abstracts, or arXiv/OpenReview submission text, and whenever a
  draft is about to leave the repo. In this repo it replaces the general deslop skill. Runs after
  program-thesis: that skill owns verdict wording, hedges and detection-bar language; this one
  changes surface wording only and never strengthens or weakens a claim. Fires on: "prose pass",
  "polish", "em dash", "AI tells", "deslop", "camera-ready", "arXiv text", "abstract". Pairs with
  scripts/prose_lint.py (counts em dashes, "not X but Y" frames, trailing participle clauses, and
  tell vocabulary per file and per 1k words; flags any section over 2 em dashes).
---

# Paper Prose

**Core principle: fix the wording, not the science.** A prose pass that moves a claim, even
slightly, is a program-thesis change made without the referee pass. Surface edits only.

## Order of operations

1. `program-thesis` has run on the text (anchored adjectives, verdict wording, detection bars,
   SYNTHESIS current). Do not start a prose pass on text whose claims are still moving.
2. `python scripts/prose_lint.py papers/<paper>/sections/*.md` for the before numbers.
3. Edit per the rules below, with `references/ai-writing-tells.md` as the catalog.
4. Re-run the lint for the after numbers. Every section at 2 or fewer em dashes.
5. `scripts/package_arxiv.py` prints the same lint as a warning at packaging time.

## Scope

In scope: `papers/*/sections/*.md`, `README.md` files, abstracts, arXiv and OpenReview form text
(`build/arxiv_abstract.txt` and the like), cover letters to venues.

Out of scope (internal documents, written for the author and agents, where precision beats
polish): `SYNTHESIS.md`, `ANOMALIES.md`, `*_SPEC.md`, `*_GATES.md`, `*RESULTS*.md`,
`PREREGISTRATION.md`, plans, ledgers, `CLAIMS.md`, and anything under `papers/` that is not a
paper section. Do not run a prose pass on these, and do not "fix" them while passing through.

## Never touch

- Math (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`), code spans and blocks.
- `\cite`, `\ref`, `\label`, `[@key]`, `{#sec:...}` and every other cross-reference or label.
- Numbers, CIs, n, units, model ids, table contents.
- The standard acknowledgments block (its text is fixed in CLAUDE.md, identical across papers).
- Claim strength. "consistent with", "we cannot exclude", "no coupling detectable at |cos| ≳ 0.5"
  and every other hedge or detection bar is program-thesis wording. Keep it verbatim. Do not add a
  hedge either.

## Rules (moved from CLAUDE.md "Prose Style", 2026-10-19)

- Minimize em dashes. Use commas, semicolons, colons, parentheses, or separate sentences instead.
  Bar: 2 per section (one `sections/*.md` file); the lint flags anything above it.
- Avoid entirely: "underscores", "landscape", "paradigm", "nuanced", "multifaceted", "tapestry",
  "holistic", "cutting-edge", "at the forefront", "a testament to".
- Prefer "shows" or "finds" over "reveals"; "use" over "leverage" or "utilize"; "important" over
  "crucial" or "critical" (unless literally about criticality).
- Replace vague intensifiers ("meaningful", "key insight", "key takeaway") with the specific claim.
  If the specific claim would change the claim's strength, leave the sentence and flag it for
  program-thesis instead.
- Don't hedge with "This suggests" or "It is worth noting that"; state the point directly.
  Exception: when "suggests" is the calibrated verdict word, it stays (see Never touch).
- Don't open sentences with "Importantly,", "Notably,", "Crucially,", "Interestingly,", or
  "Specifically,".
- Write like a careful human researcher, not a summarizer. Prefer short direct sentences over long
  ones joined by dashes.

## From the guide (references/ai-writing-tells.md)

The guide is descriptive; density is the tell, not any single item. The patterns that recur in
this repo's drafts:

- "Not X but Y" / "not just" / "not only X but also Y": state Y. Keep the frame only when X is a
  reading someone actually holds (a named rival reading is content, not a tell).
- Trailing participle clauses (", highlighting...", ", underscoring...", ", reflecting..."): drop
  the clause or make it a separate sentence with its support.
- Section-closing summaries ("Overall,", "In summary,"): end on the last substantive point.
- Transition stacks (Moreover, Furthermore, Additionally): delete most.
- Elegant variation: one name per object. Calling the same direction "the refusal direction",
  "the gate" and "the decision axis" in one section reads as three objects to a referee.
- Rule-of-three lists: keep three only when there are three things.

## Domain-term allowlist

These are technical terms here, not tells. Do not replace them:
harness, robust / robustness, align / alignment, steering, loss landscape, and "reveal" when it
means unmask (a hidden state or a masked token is revealed). "landscape" in any other sense and
"reveals" meaning "shows" are still tells. The lint cannot tell senses apart, so its "reveal" count
needs a human read.

## Vague attribution

"Prior work has shown", "it is well known", "recent studies find" with no citation: do not delete
and do not rewrite. Flag it for a `\cite` (or `[@key]`) in the report back to the author; the
citation has to be found and verified against the primary source, which is not a prose edit.

## Report back

Before/after lint numbers per paper; the list of files changed; a few sample rewrites (old → new);
every vague attribution flagged; any sentence left alone because the fix would move a claim.

## Maintenance

`references/ai-writing-tells.md` is refreshed monthly from Wikipedia's "Signs of AI writing". The
refresh replaces that file in place (never a dated copy at the repo root; `.gitignore` blocks
`/ai-writing-tells-*.md`). When the guide's vocabulary list changes, update `VOCAB` in
`scripts/prose_lint.py` to match, minus the allowlist above.
