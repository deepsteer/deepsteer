# Reproducibility {#reproducibility .unnumbered}

Each figure in this note ships with a regeneration script that reads a committed CSV under
the convention `papers/figure_data/mn_*.csv` (`mn_bottleneck_pr.csv`, `mn_ladder.csv`,
`mn_depth_collapse.csv`); the analysis outputs are gitignored, so the committed CSV plus its
script is the reproducibility contract for every figure.

Every number in this note, the participation-ratio profiles, the calibrated covariance nulls,
the positive-control ladders, the per-head write attribution, and the rank-sweep outcomes, is
indexed in the shared supplement `deepsteer/supplement/MANIFEST.json` (public repository:
<https://github.com/deepsteer/deepsteer/>), each with a content hash,
its provenance, and the figure or table it backs. The two instruments this note shares with the
companion flagship (the decision-site participation-ratio profile and the depth-asymmetry panel)
live in the supplement once and are cited by both papers; `deepsteer/supplement/scripts/verify.py`
asserts the note's plotting copies match the canonical values, so a shared number can change in
only one place. Model ids, decision layers, standardization settings, and seeds are pinned in
`deepsteer/supplement/PROVENANCE.md`; raw activation caches are available on reviewer request.
