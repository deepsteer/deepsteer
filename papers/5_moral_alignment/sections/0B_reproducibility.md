# Appendix B. Reproducibility

**Models.** OLMo-3 7B \citep{olmo3_2025}: base `allenai/Olmo-3-1025-7B`
(+ stage-3 anneal revisions `stage3-step1000`--`stage3-step11921`),
`allenai/Olmo-3-7B-Instruct-SFT`, `allenai/Olmo-3-7B-Instruct-DPO`, and
`allenai/Olmo-3-7B-Instruct` (+ RLVR revisions `step_050`--`step_400`). All at
fp16. The full checkpoint inventory and the 25-state grid are in
`checkpoint_inventory.json` and `checkpoint_grid.json`.

**Datasets.** Foundation probing uses the v2 minimal-pair set (40 pairs per
foundation); persona probing uses the 240-pair persona set; behavioral judgment
uses 48 Moral-Foundations scenarios across four difficulty levels; the refusal
direction uses Heretic's exact set, `mlabonne/harmful_behaviors` and
`mlabonne/harmless_alpaca` (column `text`, first 400 train prompts each, in
dataset order).

**Probing.** Mean-pooled residual activations; one seeded `Linear` probe per
foundation per layer (50 epochs, lr $10^{-2}$, BCE); unit-norm weight vector =
direction. Bootstrap (200 resamples) on the base gives a stable band of
layers 15--31 across all foundations; geometry is reported over that band. The
8 full-attention layers (3, 7, \dots, 31) show no periodicity in any layer-wise
metric.

**Refusal ablation.** Difference-of-means of the final-token residual (chat
template, generation prompt) between harmful and harmless prompts at layer 16;
unit direction orthogonalized out of every layer's attention out-projection and
MLP down-projection ($W \leftarrow W - \hat d\hat d^\top W$). The refusal
classifier scores an opening refusal (e.g.\ "I'm sorry, but I can't \dots") as a
refusal regardless of length, correcting a failure mode that scored long
declines as compliance.

**Runs.** Probing and geometry on Apple MPS (base) and a single A100 (pipeline);
the disk-bounded pipeline sweep purges each checkpoint's weights after
processing. Commands are in `papers/5_moral_alignment/runpod/`
(`ONLY="pipeline coupling"` for the pipeline, `ONLY="heretic"` for the
ablation). Per-state results JSON are committed under `outputs/`; per-checkpoint
direction caches are regenerable and not committed.

**Note on references.** External citations were verified against primary
sources (arXiv abstract pages and publisher records): the refusal-direction
\citep{arditi2024refusal}, alignment-faking \citep{greenblatt2024faking},
persona-features \citep{wang2025persona}, OLMo-3 \citep{olmo3_2025}, and
Heretic \citep{pew2025heretic} entries were confirmed for title, first author,
and identifier. The companion-paper entries (Papers 1--4 of this series) cite
working titles.
