# Dataset licenses & provenance

Source licenses for the committed derived datasets in `deepsteer/datasets/`. The DeepSteer
repository is **Apache-2.0**; everything committed here is Apache-2.0-compatible. Generated
content (LLM-produced neutrals, retellings, register re-renderings, paraphrases) is original
to this project and **Apache-2.0**. Source halves carry their upstream license, named below.

## `direction1_vmoral_v1.json` — the single-source V_moral dataset

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
editions nor model knowledge). See `papers/direction1_moral_subspace/PREREGISTRATION.md`
(single-source amendment). A fable-based extension may be revisited if a clean public-domain
source/method becomes viable. Only NC-safe **index identifiers** (fable aliases) ever appear
in the repo (`papers/direction1_moral_subspace/partition_ids.json`); no MORABLES morals or
story text are committed.
