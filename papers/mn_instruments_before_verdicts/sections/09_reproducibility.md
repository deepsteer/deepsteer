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
`deepsteer/supplement/PROVENANCE.md`. Every array this note cites (the participation-ratio
samples with their bootstrap and null draws, the two-position validity ladder arrays, the
depth-matched per-twin deltas at layers 12 and 16, the RMSNorm-fold reconstruction arrays, and
the reply-inversion margins for the harm direction and twenty matched-norm random directions)
lives in the flagship's Zenodo deposit (`deepsteer_fl_arrays_v1.tar.zst`, CC BY 4.0, DOI
10.5281/zenodo.[reserved before submission]); the deposit manifest lists the note's files by path
and SHA-256 so no shared array is duplicated.
