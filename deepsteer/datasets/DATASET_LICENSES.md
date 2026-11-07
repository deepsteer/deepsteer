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
dataset names are taken from it, not code). The harmless half carries the NonCommercial term.
The FL/MN deposit (10.5281/zenodo.22731361) describes the set as "MIT-licensed upstream", which
is incorrect for the harmless half.

**Author judgment (2026-10-07).** In this case the derived statistics (directions, class-mean
contrasts, activation samples and projections computed from the prompts) are not bound by the
NonCommercial term: they do not reproduce or redistribute the Alpaca text. The arrays already
published in 10.5281/zenodo.22731361 stay public under CC BY 4.0. **Policy going forward:**
arrays computed from NC-licensed stimuli are still deposited restricted (CC BY-NC 4.0, access on
request), as the 36 that left the `papers/` tree are in 10.5281/zenodo.23203126. The prompt
text itself is never redistributed. See `papers/ANOMALIES.md` A12.

## MoralBench: not vendored (runtime fetch only)

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| `questions/` (88 items), `answers/` (4 key files), `template/moral_system.txt` | `agiresearch/MoralBench` (Ji et al., arXiv:2406.04428), commit `84100440a28d040eca27df32b93f44685ad69ce5` | **none** (no LICENSE file, no license in README; the paper's Drive mirror states no terms) | upstream tree at the pinned commit, 2026-10-29 and 2026-10-30 | `deepsteer.datasets.moralbench` (fetched into the user cache; never committed, never in the wheel); MB replication (`papers/MB_PLAN.md` §9) |
| `moralbench_manifest.json` | this project | **Apache-2.0** (file paths and sha256 hashes only) | — | loader verification; shipped in the wheel |

Nothing of MoralBench's text, prompt or answer keys is in the repository or the wheel. Tracked
MB outputs hold item ids, foundation labels, parsed letters, readouts, scores and prompt hashes
(`MB_PLAN.md` §9.2). Seven `6_concepts` items are verbatim Clifford et al. (2015) MFV vignettes,
third-party text that MoralBench could not license in any case.

## MoReBench: fetched at a pinned revision (runtime fetch only)

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| `morebench_public.csv` (500 scenarios, 11,450 criteria), `morebench_theory.csv` (150), the card | `morebench/morebench` (Chiu, Lee, et al., arXiv:2510.16380), HF revision `d900e89f942f784f8e0113935034b4f0541a0b55` | **CC BY 4.0** (card YAML; not gated); evaluation use of the whole dataset, the expert-written scenarios and their rubric criteria included, confirmed in writing by MoReBench's author (2026-11-07; below) | HF API and card, 2026-10-30; author's written confirmation, 2026-11-07 | `deepsteer.datasets.morebench` (fetched into the user cache; not committed, not in the wheel); MRB replication (`papers/MRB_PLAN.md`) |
| the 100 `expert_written_*` scenarios | APPE Ethics Bowl cases (CC BY-NC-ND 4.0 in some years, all rights reserved in others), Ethics Unwrapped (UT Austin; copyrighted, free use with credit, no license), Cheung, Maier, Lieder, PNAS 2025 (article CC BY-NC-ND 4.0; OSF materials unlicensed) | **evaluation: permitted**, the rubric criteria included (MoReBench's author, in writing, 2026-11-07; ANOMALIES MRB-A7 resolved for evaluation). **Training or adaptation: not permitted until a separate decision** records its basis (the sources' own terms are CC BY-NC-ND 4.0 or unlicensed). Fetch-only | source pages, 2026-10-30; author's written confirmation, 2026-11-07 | same; no committed text, no adaptations |
| `deepsteer/benchmarks/moral_reasoning/morebench/prompts.py` (generation and judge instruction strings) | `morebench/morebench` code, commit `e6689bbb9d742c48ea78ea07acd7931eba05b213` | **MIT**, Copyright (c) 2025 MoReBench (`LICENSE_MOREBENCH`, copied unmodified) | upstream `LICENSE`, 2026-10-30 | the MoReBench harness; shipped in the wheel with its license |
| `morebench_manifest.json` | this project | **Apache-2.0** (file names and sha256 hashes only) | — | loader verification; shipped in the wheel |

DailyDilemmas (`kellycyy/daily_dilemmas`) and AIRiskDilemmas (`kellycyy/AIRiskDilemmas`), the
sources of the other 400 public scenarios, are CC BY 4.0 on their own cards.

## MACHIAVELLI: fetched at pinned hashes (runtime fetch only; MCH-G0; only the manifest is in the wheel)

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| game bundle `game_data/` (scenes, trees, game source, annotations, normalization coefficients) | Google Drive file `19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz` linked from `aypan17/machiavelli` (Pan et al., arXiv:2304.03279) at `ebb44e1`; zip sha256 `e13d07b4…` | **none** (no terms in the bundle or README; the games are Choice of Games' commercial titles) | README, `LICENSE` scope and bundle contents, 2026-11-02 | `deepsteer.datasets.machiavelli` (fetched into the user cache, never committed, never in the wheel); MCH replication (`papers/MCH_PLAN.md`) |
| conscience weights `multi_deberta-v3-large_1e-05_32_3.pkl` | Google Drive file `1MD0N-0IfJx4mCUnAqH9yhjqBz2u00c9a` (same README) | **none** | 2026-11-02 | DRRN +shaping only (not planned); hashed, not loaded |
| engine and agent code; `main_results*.csv` | `aypan17/machiavelli` at `ebb44e1` (CSVs and prompts unchanged since `9f82117`) | **MIT**, Copyright (c) 2022 Alex Pan | upstream `LICENSE`, 2026-11-02 | engine fetched at the pin; `deepsteer/benchmarks/agentic/machiavelli/prompts.py` copies the prompt strings, with `LICENSE_MACHIAVELLI` (copied unmodified) beside it; both shipped in the wheel (moved from `papers/` after G1, Amendment 7) |
| `deepsteer/datasets/machiavelli_manifest.json` | this project | **Apache-2.0** (file paths and sha256 hashes only: the engine's 97 files at `ebb44e1`, the bundle zip and its 3,459 files) | — | `deepsteer.datasets.machiavelli` verification; shipped in the wheel |
| `papers/mch_machiavelli/outputs/g0_random*.jsonl.gz` | this project (runs of the MIT engine on the bundle) | **Apache-2.0**; seeds, actions, rewards and observation hashes only. The per-episode harm counts are not tracked: with seed-exactness they would rebuild the bundle's coefficient file (author decision at MCH-G0) | — | MCH-G0 statistics (counts regenerate from the seeds) |

## CST Layer A evaluation sets and CST sources (CST-G0, 2026-11-04; nothing fetched or committed yet)

Layer A sets carry human labels and are evaluation only: no CST adapter, corpus or prompt trains on
them (`papers/CST_PLAN.md` §2.3, §6.5). Text is never committed; items are referenced by upstream id.

| Component | Source | Upstream license | Verified | Feeds |
|---|---|---|---|---|
| MoralExceptQA (`data/complete_file.json`) | HF `feradauto/MoralExceptQA` at `def71b74159a8460ce977fc2ace42e32947fb3fa` (Jin et al., NeurIPS 2022, arXiv:2210.01478); also GitHub `feradauto/MoralCoT` | **CC BY-NC-SA 4.0** (HF card, "Licensing Information"; the card's BibTeX copyright line is the paper's, not the data's) | HF card, 2026-11-04 | **not used** (CST Amendment 7: NC-SA on an input to a partner deliverable) |
| Scruples Anecdotes and Dilemmas v1.0 data | `ai2-mosaic-public/projects/scruples/v1.0/data/{anecdotes,dilemmas}.tar.gz` (Lourie, Le Bras, Choi, AAAI 2021, arXiv:2008.09094) | **none stated** for the data (both tarballs read; no license line or file); Anecdotes are Reddit posts. The code (`allenai/scruples` at `c79697e9`) is **Apache-2.0**. The HF mirror `tasksource/scruples`'s `apache-2.0` tag is a third party's | tarball contents and repo `LICENSE`, 2026-11-04 | **CST Layer A** (Amendment 7): the Dilemmas part, 50 pairs with at least 4 of 5 annotators agreeing, evaluation only by id; never trains, text never committed or redistributed |
| ETHICS | `hendrycks/ethics` | **MIT** (row above) | 2026-06-27 | CST Layer A fallback (commonsense part, MIT) if Scruples cannot serve; the V_moral held-out rule applies (`MORAL_EVAL_REPLICATIONS.md`) |
| DailyDilemmas | `kellycyy/daily_dilemmas` at `372352c4` | **CC BY 4.0** (card YAML) | 2026-11-06 | **not Layer A** (CST Amendment 7, 2026-11-06): its per-action value attributions are GPT-4-generated and it carries no human verdict on the action; MoReBench's scenario source only |
| Constitution-audit tenet lists (`tenets/*.md`) | `ajobi-uhc/constitution-audits` at `f5e80c1` (Jakkli, Rajamanoharan, Nanda, arXiv:2605.24229) | **none** at the repo root (MIT covers only its bundled `petri/`) | repo tree, 2026-11-04 | cited only; CST decomposes the documents itself (`T0.1`) and copies nothing from these lists |
| Grad-Nec reference (`eval/faithfulness_metrics_clean.py`, `grad_cot_necessity`) | `safety-research/faithful-cot` at `f9e1f29564fcd8d3a50507766e028547cdbbbd23` (Jia, Benton, Easley, arXiv:2605.24286) | **none** at the repo root (Apache-2.0 covers only the vendored VERL fork under `train/`) | GitHub API and tree, 2026-11-04 | reference for a re-implementation if the author so decides at G0; not vendored |
| Open Character Training data | HF datasets of Maiya et al. (arXiv:2511.01689) | **other** (inherits LIMA's terms, non-commercial); code MIT | card, 2026-11-04 | not used; CST generates its own introspection transcripts |
