# Dataset licenses & provenance

Source licenses for the committed derived datasets in `deepsteer/datasets/`. The DeepSteer
repository is **Apache-2.0**; everything committed here is Apache-2.0-compatible. Generated
content (LLM-produced neutrals, retellings, register re-renderings, paraphrases) is original
to this project and **Apache-2.0**. Source halves carry their upstream license, named below.

## `d1_vmoral_v1.json` — the single-source V_moral dataset

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| Moral Stories situations + moral actions | `demelin/moral_stories` (Emelin et al., EMNLP 2021) | **MIT** | upstream GitHub `LICENSE` ("MIT License") + HF card, 2026-06-27 | `train`, `eval_g2_indist` (the V_moral training + in-distribution eval) |
| ETHICS commonsense scenarios | `hendrycks/ethics` (Hendrycks et al., ICLR 2021) | **MIT** | HF card metadata, 2026-06-27 | `eval_generalization_probe` (held-out, zero in training) |
| Generated neutrals / declarative re-renderings / paraphrases | this project (Claude) | **Apache-2.0** | — | all splits |

MIT and Apache-2.0 both permit research **and commercial** use with attribution / notice, so
the dataset is clean for DeepSteer's Apache-2.0 / commercial posture. Cite the two source
papers when using the dataset.

### Why MORABLES is NOT here

MORABLES (`cardiffnlp/Morables`, Marcuzzo et al. 2025) is **CC-BY-NC-4.0** (NonCommercial),
verified from its HF card. Its NonCommercial restriction cannot be sublicensed under
Apache-2.0, and the retellings derive from its (largely modern, e.g. Gibbs/Perry)
expression, so committing them would inject an NC carve-out into the dataset that defines
`V_moral` — making DeepSteer's own commercial use of `V_moral` inherit NC. It was therefore
**dropped**: `V_moral` is single-source (Moral Stories). A public-domain re-derivation was
evaluated and rejected — only ~21% of our MORABLES selection is canonical enough to retell
from public-domain content (the rest is obscure Perry-index fables in neither public-domain
editions nor model knowledge). See `papers/d1_moral_subspace/PREREGISTRATION.md`
(single-source amendment). A fable-based extension may be revisited if a clean public-domain
source/method becomes viable. Only NC-safe **index identifiers** (fable aliases) ever appear
in the repo (`papers/d1_moral_subspace/partition_ids.json`); no MORABLES morals or
story text are committed.

## `papers/d2_decision_coupling/data/xstest_borderline.json` — XSTest borderline subset

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| 40 borderline prompts (20 safe over-refusal-prone + 20 mild unsafe-contrast) | `paul-rottger/xstest` (Röttger et al., NAACL 2024; arXiv:2308.01263), commit `d7bb5bd` | **CC-BY-4.0** | source-repo README + LICENSE, 2026-07-01 | D2 B2 outcome-conditioned P2 pilot; D2 B5 over-refusal arm |

CC-BY-4.0 permits research **and commercial** use with attribution + indication of changes;
both are satisfied by the NOTICE entry and the provenance block inside the JSON. **Prompts
only, no model completions** — pulled from the source repo, NOT the HF `xstest-v2-copy` mirror
(whose bundled completions carry Meta / Mistral / OpenAI terms). Original prompt text is
unmodified; the only modification is subset selection (documented in the file). Cite the XSTest
paper when using the subset. Rebuild: `papers/d2_decision_coupling/scripts/build_xstest_borderline.py`.

## Emergent-misalignment fixtures and corpora (vendored, unmodified)

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| `benchmarks/compliance_gap/em_fixtures/first_plot_questions.yaml` (8 first-plot questions + judge prompts) | `emergent-misalignment/emergent-misalignment` (Betley et al., 2025) | **MIT**, Copyright (c) 2025 emergent-misalignment | GitHub API license field + upstream `LICENSE`, 2026-10-06 | `EMBehavioralEval`; shipped in the wheel |
| `datasets/corpora/emergent_misalignment/{insecure,secure}.jsonl` (6,000 records each) | same repo, `data/` | **MIT** (same) | same | Paper 2 EM replication LoRA runs; repo only, not shipped in the wheel |

MIT permits research and commercial use with the copyright and permission notice retained;
the NOTICE entry carries it. Cite Betley et al. when using either.

## `papers/5_moral_alignment/refusal_prompts.json`: the Heretic refusal prompt set

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| 400 + 100 harmful prompts | `mlabonne/harmful_behaviors` (from AdvBench, `llm-attacks/llm-attacks`) | **MIT** (AdvBench repo); the HF card states no license | GitHub API license field, 2026-10-06 | refusal, proto-refusal, position and two-site directions (Papers 5–7, D1–D3, W4) |
| 400 + 100 harmless prompts | `mlabonne/harmless_alpaca` (Stanford Alpaca instructions) | **CC BY-NC 4.0** (`tatsu-lab/alpaca`); the HF card states no license | rows checked verbatim against `tatsu-lab/alpaca`, 2026-10-06 | same |

The selection follows Heretic's `config.default.toml` (Heretic's code is AGPL-3.0; only the
dataset names are taken from it, not code). The harmless half carries the NonCommercial term, so
arrays computed from this set are treated as NC-derived under the supplement's exclusion rule:
the 36 that left the `papers/` tree are in the restricted record 10.5281/zenodo.23203126 (CC
BY-NC 4.0, access on request), not the public one. The earlier FL/MN deposit
(10.5281/zenodo.22731361) describes this set as "MIT-licensed upstream"; that is incorrect for
the harmless half, and how to treat the Alpaca-derived arrays in that deposit is an open author
decision (`papers/ANOMALIES.md` A12).
