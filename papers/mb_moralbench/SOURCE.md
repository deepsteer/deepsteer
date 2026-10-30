# MoralBench source of record (MB-G0, 2026-10-30)

| what | value |
|---|---|
| paper | Ji, Chen, Jin, Xu, Hua, Zhang, "MoralBench: Moral Evaluation of LLMs", arXiv:2406.04428 (v1 2024-06-06, v2 2025-07-04); SIGKDD Explorations 27(1):62-71, 2025 |
| code + data | `https://github.com/agiresearch/MoralBench`, commit `84100440a28d040eca27df32b93f44685ad69ce5` (= `main` HEAD on 2026-10-30; 5 commits, last 2025-08-30) |
| other refs | `update-readme` / `refs/pull/2/head` at `63ee578`, an ancestor of the pinned commit |
| license | none: no LICENSE file, no license text in README or in the Drive folder (checked 2026-10-30) |
| fetch | `deepsteer.datasets.moralbench.fetch_moralbench()` clones into `~/.cache/deepsteer/moralbench/<sha>/` (`$DEEPSTEER_CACHE` overrides) |
| manifest | `deepsteer/datasets/moralbench_manifest.json`: 93 files (88 questions, 4 answer files, 1 system prompt), path + sha256 only |
| Drive mirror | folder `1k93YZJserYc2CkqP8d4B3M3sgd3kA8W7` ("LLM Moral Benchmark") holds one file, `questions/moral benchmark.zip`, sha256 `0dc19432…a990963a`; questions only (no answers, no template); all 88 question files byte-identical to the pinned commit, as are the 88 `.ipynb_checkpoints` copies |
| sampling in the paper | "For all models, we use a temperature of 0.7." arXiv:2406.04428v2, p. 5, §4.1 LLM Backbones (not in v1); v2 also names `gpt-3.5-turbo-1106` and `gpt-4-0613` there |
| closed models | `gpt-3.5-turbo-1106` shut down 2026-09-28 (OpenAI notice of 2025-09-26); `gpt-4-0613` and alias `gpt-4` shut down 2026-10-23 (notice of 2026-04-22); `https://developers.openai.com/api/docs/deprecations` (formerly `platform.openai.com/docs/deprecations`), checked 2026-10-30 |
| Clifford norms (MB-A4) | `deepsteer/datasets/mfv_clifford2015.json` has no wrongness ratings (checked first); `http://scottaclifford.com/wp-content/uploads/2012/02/Full-Published-Norms.xlsx` (linked from Clifford's research page) fetched to `~/.cache/deepsteer/mfv_clifford2015/` only, sha256 `a198b6f48de762bdac1cc6e97ea37341404acecd206161b94601df7f28708cfe`, 2026-10-30 |
| `main.py` | not opened (clean room, MB_PLAN §9.2); its behaviour is taken from MB_PLAN §1 |

## What the pinned commit holds

| set | items | per foundation | key letters | notes |
|---|---|---|---|---|
| `MFQ_30` | 20 | 4 × harm, fairness, ingroup, authority, purity | A/B | 10 items "Agree/Disagree", 10 items "Correct/Not Correct"; A + B = 5 on all 20 |
| `6_concepts` (MFV) | 24 | 4 × the five + liberty | A/B | all "Agree/Disagree"; A + B = 4 on all 24 |
| `MFQ_30_compare` | 20 | 4 × five | A/B | `ingroup_2` scores A and B equally (any A/B answer earns the point) |
| `6_concepts_compare` | 24 | 4 × six | A/B | `ingroup_2` and `ingroup_3` are the same item with opposite keys; `fairness_2` and `fairness_3` are the same item with the same key |

Every answer file also carries keys with no question file: `life_boat` and `trolley_tracks`
(three-letter keys) in all four, and `liberty_1`–`liberty_4` in the two MFQ files (values differ
from the MFV liberty keys). The loader ignores them; items come from question files.

The paper describes MFQ-30 as 30 items and the MFV as 132 vignettes; the benchmark uses 20 and
24. Seven `6_concepts` vignettes are verbatim Clifford et al. (2015) MFV text that the repo
already holds in `deepsteer/datasets/mfv_clifford2015.json` (Paper 3), so part of MoralBench's
text is third-party, as with LEB's instruments.
