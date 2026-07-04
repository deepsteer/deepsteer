# Reproducibility {#reproducibility .unnumbered}

Each figure in this note ships with a regeneration script that reads a committed CSV under
the convention `papers/figure_data/mn_*.csv` (`mn_bottleneck_pr.csv`, `mn_ladder.csv`,
`mn_depth_collapse.csv`); the analysis outputs are gitignored, so the committed CSV plus its
script is the reproducibility contract for every figure.
