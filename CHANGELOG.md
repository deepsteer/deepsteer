# Changelog

The API is alpha and will change between minor versions.

## [0.2.0]

Namespace hygiene and CI (LIBRARY_RELEASE_PLAN §B, §C).

### Moved
- `deepsteer.supplement` → `papers/supplement/` in the repository (paper data, provenance and
  build scripts, never library API). `import deepsteer.supplement` now raises `ImportError`
  naming the new location. A stub `deepsteer/supplement/README.md` maps the paths that the
  FL/MN papers cite to their new locations.

### Added
- `__all__` on `deepsteer` and `deepsteer.viz` (the other subpackages already had one).
  Names not in `__all__` are private by convention.
- Import contracts (import-linter, checked in CI): `deepsteer` imports nothing from
  `papers`, `scripts` or `tests`; `deepsteer.geometry` and the direction algorithms
  (`mean_diff`, `leace`, `compare`, `probe_weight`) import neither torch nor transformers.
- Experimental markers: `deepsteer.kdg`, `EMBehavioralEval`, `PersonaFeatureProbe`,
  `PersonaActivationScorer` and `CompositionalMoralProbe` serve one paper each and are not
  stable API (ARCHITECTURE.md, Experimental).
- CI (`.github/workflows/ci.yml`): ruff, import contracts, fast tests on Python 3.10 and
  3.12, and the installed-wheel smoke test on every push and PR to `main`.

### Changed
- `extract_mean_diff_directions` and `extract_leace_directions` accept any array-like
  (numpy arrays or CPU torch tensors). They previously required torch tensors (they called
  `.numpy()`), although the package documented numpy inputs. Results are bit-identical
  for torch input.
- Lint scope is `ruff check .` with `papers/` and the frozen W4 harness excluded;
  prose-data modules exempt from E501. Ruff pinned to 0.16.7 and import-linter to 2.15 in
  the `dev` extra. Unused imports removed from library modules (none was imported elsewhere
  in the repository).

### Fixed
- `build_probing_dataset(model=...)` and `legacy_pool=True` were silently ignored under the
  v2 default (the bundled dataset was returned first). They now raise `ValueError` pointing
  to `dataset_version="v1"`, the only path that uses them.
- `geometry.permutation_test` (and `permutation_test_mft`) counted ties with the observed
  statistic by float rounding, so p varied by platform. Mirror partitions (the two groups
  swapped) and index reorderings of one split are mathematically tied; they are now counted
  within 1e-12. For 6 items split 3/3 the exact floor is p = 0.10, so the test cannot reject
  at 0.05. See `papers/ANOMALIES.md` A11.
- `full_geometric_analysis(directions, labels=...)` with labels other than the six MFT
  foundations and no `groups` raised `IndexError` (it applied the MFT index split to any
  label set). The MFT default now applies only to `FOUNDATION_ORDER`; other label sets get
  no permutation test.
- Software title, citation and PyPI summary: "DeepSteer: Monitor, measure, and steer alignment
  from LLM pretraining through post-training" (was "Evaluating and Steering Alignment Depth in
  LLM Pre-Training").
- README rewritten and cut from about 700 to about 130 lines: findings and papers first,
  one working quick-start example, a module table; benchmark walkthroughs live in docstrings
  and testing commands in CONTRIBUTING. Its old API example passed the wrong arguments.

## [0.1.1]

First working release on PyPI (LIBRARY_RELEASE_PLAN §A). Supersedes 0.1.0, which is yanked.

### Fixed
- The wheel now ships the data files the library reads at runtime:
  `datasets/moral_probing_v2.json`, `datasets/dilemma_pairs_{final,validated}.json`, and the
  emergent-misalignment first-plot fixture (with its upstream MIT `LICENSE`). In 0.1.0 they
  were missing, and `build_probing_dataset()` silently fell back to the v1 minimal-pair
  pipeline instead of loading the 1,200-pair v2 benchmark.
- `build_probing_dataset()` now raises `FileNotFoundError` when v2 is selected (the default)
  and its file is missing, instead of silently building the v1 dataset. Request v1 with
  `dataset_version="v1"`.

### Added
- `scripts/wheel_smoke.py`: installed-wheel check that asserts the v2 dataset loads and
  `__version__` matches the package metadata.
- Release workflow publishing to PyPI via trusted publishing on `v*` tags.
- NOTICE and `DATASET_LICENSES.md` entries for the vendored emergent-misalignment files (MIT).

## [0.1.0] - 2026-04-16

Uploaded to PyPI without package data (see 0.1.1, Fixed). Not tagged in git.
