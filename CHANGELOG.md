# Changelog

The API is alpha and will change between minor versions.

## [Unreleased]

### Added
- CI (`.github/workflows/ci.yml`): ruff, fast tests on Python 3.10 and 3.12, and the
  installed-wheel smoke test on every push and PR to `main` (LIBRARY_RELEASE_PLAN §B).

### Fixed
- `geometry.permutation_test` (and `permutation_test_mft`) counted ties with the observed
  statistic by float rounding, so p varied by platform. Mirror partitions (the two groups
  swapped) and index reorderings of one split are mathematically tied; they are now counted
  within 1e-12. For 6 items split 3/3 the exact floor is p = 0.10, so the test cannot reject
  at 0.05. See `papers/ANOMALIES.md` A11.

### Changed
- Lint scope is `ruff check .` with `papers/`, the frozen W4 harness, and
  `deepsteer/supplement` excluded; prose-data modules exempt from E501. Ruff pinned to
  0.16.7 in the `dev` extra. Unused imports removed from library modules (none was imported
  elsewhere in the repository).

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
