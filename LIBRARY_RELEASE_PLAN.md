# LIBRARY_RELEASE_PLAN.md — make `deepsteer` an installable library without moving the papers

Brief for Claude Code. Drop this at the repo root next to `RESEARCH_PLAN.md`. Work it top to
bottom; each phase ends in a PR and a human gate. Do not start a phase until the previous
one is merged. Do not interleave this work with pod sessions — if a KDG/FL pod is in flight,
stop at the nearest gate and wait.

## 0. Decision and scope

**Decision (2026-10-06):** keep the monorepo. `papers/` stays where every arXiv citation and
permalink points. The library becomes installable and its namespace gets tightened. A repo
split is explicitly deferred; see §7 for the only conditions under which it is revisited.

**Why this is enough.** Inspection of `main` on 2026-10-06:

- Dependency direction is already one-way: `papers/` imports `deepsteer.*`; nothing in
  `deepsteer/`, `scripts/`, or `tests/` imports from `papers/`. Nothing structural blocks a
  clean package.
- Weight is artifacts, not code: `deepsteer/` is 16 MB (12 MB is `datasets/corpora/`),
  `papers/` is 310 MB on disk, 127 MB git-tracked, 122 tracked `.npz/.pt/.safetensors/.bin`
  files, and a shallow clone pulls ~226 MB of `.git`.
- There is no install path except cloning: no tags, no releases, version `0.1.0`, nothing
  on PyPI.
- `pyproject.toml` has no `package-data`, but `datasets/pipeline.py` and
  `datasets/loaders.py` locate `moral_probing_v2.json` and other JSON files via
  `Path(__file__)`. **A wheel built today would install without those files and
  `build_probing_dataset()` would fail.** This is the first thing to fix.
- Paper-specific code lives in the library namespace: `deepsteer.kdg` (active KDG
  workstream, has `tests/kdg/` and `scripts/pod_kdg_*` callers) and `deepsteer.supplement`
  (supplement-generation tooling with its own MANIFEST/PROVENANCE/RELEASE_PLAN).
- 129 scripts under `papers/` use `sys.path` hacks. Leave them alone; they are not
  user-facing and this plan does not touch `papers/` code except to remove binary
  artifacts.

## 1. Hard constraints (never violate)

1. **No history rewrite.** No `filter-repo`, no `filter-branch`, no force-push to `main`,
   no BFG. Artifacts are removed from the tree tip only; they stay reachable at their
   historical SHAs.
2. **No URL breaks.** Every existing path under `papers/` that a paper could cite keeps
   resolving. If a file must go, leave a stub README at that path pointing to the
   replacement and the DOI.
3. **Don't break a pod mid-flight.** Before moving anything that `scripts/pod_*` or
   `tests/scripts/test_pod_*` import, confirm with the human that no pod session depends on
   the current import path this week.
4. **Compatibility shims for one release.** Any moved module keeps a shim at the old import
   path that re-exports and emits `DeprecationWarning`, removed in the release after next.
5. **Methodology skills stay.** `.claude/skills/*`, `CLAUDE.md` hard gates, and the
   prereg/verification discipline are untouched by this plan.
6. **No new runtime dependencies in core.** `deepsteer` core deps stay as listed in
   `pyproject.toml`; anything analysis-only stays behind an extra.

## 2. Phase A — installable as-is (target: one session)

Goal: `pip install deepsteer` works from PyPI, in a clean venv, with only core deps, and
`build_probing_dataset(target_per_foundation=10)` runs.

Tasks:

- [ ] **Package data.** Add to `pyproject.toml`:
  `[tool.setuptools.package-data] deepsteer = ["**/*.json", "**/*.md", "**/*.txt", "**/*.jsonl"]`
  (adjust the glob to what `datasets/` and `benchmarks/*/em_fixtures` actually read —
  grep for `Path(__file__)`, `open(`, `importlib.resources` across `deepsteer/` and list
  every non-`.py` file the library loads at runtime). Exclude `papers/` and `outputs/` via
  `[tool.setuptools.packages.find] exclude = ["papers*", "outputs*", "tests*", "scripts*"]`.
- [ ] **Build and verify in a clean venv.** `python -m build`, then in a fresh venv:
  `pip install dist/*.whl`, `python -c "import deepsteer; from deepsteer.datasets import build_probing_dataset; d = build_probing_dataset(target_per_foundation=5); print(len(d.train))"`.
  Also `python -c "from deepsteer.directions import extract_mean_diff_directions; from deepsteer.geometry import full_geometric_analysis; from deepsteer.causal import ablation_sweep"`.
  Record the wheel size in the PR description. If it exceeds ~25 MB, report which files
  dominate and stop for a decision (likely `datasets/corpora/`); do not lazy-load yet.
- [ ] **Version and tag.** Bump to `0.1.0` if not already there (it is), add a short
  `CHANGELOG.md` with a `0.1.0` entry ("first installable release; API is alpha and will
  change"), tag `v0.1.0` on the merge commit.
- [ ] **PyPI.** Check that the name `deepsteer` is free; if it isn't, stop and ask. Set up
  `.github/workflows/release.yml` using PyPI trusted publishing (OIDC) triggered on
  `v*` tags. The human creates the PyPI project and the trusted-publisher entry; the
  workflow does the upload. Do not store API tokens in the repo.
- [ ] **README install section.** Lead with `pip install deepsteer` (and
  `pip install "deepsteer[api,lora]"`); move the `-e` instructions under a
  "Developing / reproducing papers" heading that also says:
  `git clone --filter=blob:none https://github.com/deepsteer/deepsteer.git`.
  Add the same clone line to `CONTRIBUTING.md`.
- [ ] **Fix stale docs.** `CLAUDE.md`'s architecture tree shows `papers/` and `outputs/`
  under `deepsteer/`; they are repo-root siblings. Correct it there and in
  `ARCHITECTURE.md` if the same error appears.

Gate A (human): PyPI name confirmed, trusted publisher created, wheel size accepted.

## 3. Phase B — CI (target: half a session)

- [ ] `.github/workflows/ci.yml` on push/PR to `main`:
  1. `ruff check .`
  2. `pytest tests/ -m "not slow and not regression"` on Python 3.10 and 3.12.
  3. Build the wheel; install it in a clean venv; run the Phase A import/dataset smoke
     script as a separate job (this is what catches package-data regressions).
  4. `lint-imports` (see Phase C) once the contract file exists.
- [ ] Add a README badge for CI. No coverage service.

Gate B: CI green on `main`.

## 4. Phase C — namespace hygiene (target: one to two sessions)

Goal: what `import deepsteer` exposes is the library; research-program code is visibly
separate. Two different treatments, because the two modules are in different states.

### C1. `deepsteer.supplement` → `papers/supplement/`

- [ ] Grep every importer of `deepsteer.supplement` (expected: only `papers/` scripts and
  possibly `scripts/package_arxiv.py`). List them in the PR.
- [ ] `git mv deepsteer/supplement papers/supplement`. Keep MANIFEST.json, PROVENANCE.md,
  RELEASE_PLAN.md, `cells/`, `figure_data/`, `scripts/` intact — these are the supplement's
  provenance record and must not be rewritten.
- [ ] Leave `deepsteer/supplement/__init__.py` as a shim that emits `DeprecationWarning`
  and, if anything imports it, raises a clear `ImportError` naming the new location
  (it was never library API, so a hard pointer is acceptable). Fix the importers.

### C2. `deepsteer.kdg` — mark experimental now, relocate after the KDG paper ships

Rationale: `kdg/` is the active workstream (`tests/kdg/`, `scripts/pod_kdg_*`,
`papers/kdg_judgment_action`). Moving it while the pod runs violates constraint 3 for no
user benefit. Instead:

- [ ] Add a module docstring to `deepsteer/kdg/__init__.py`:
  "Experimental. Research-program code for the knowing–doing gap panel. Not part of the
  stable library API; expect it to move to `papers/kdg_panel/` or a `deepsteer.research`
  namespace once the KDG paper is published."
- [ ] Add an `experimental` section to `ARCHITECTURE.md` listing `deepsteer.kdg` and any
  benchmark classes that exist only to serve one paper (`EMBehavioralEval`,
  `PersonaFeatureProbe`, `PersonaActivationScorer` are candidates — confirm each by
  checking whether any non-`papers/` caller exists). Do not move the benchmarks.
- [ ] Open a GitHub issue "Relocate deepsteer.kdg after KDG publication" referencing this
  plan so it is not forgotten.

### C3. Public surface

- [ ] Define `__all__` in `deepsteer/__init__.py` and in each of `directions`, `geometry`,
  `causal`, `core`, `datasets`, `steering`, `viz`. The README's "Library API" section is
  the spec for what belongs there. Anything not in `__all__` is private by convention.
- [ ] Add `importlinter` to the `dev` extra and a contract in `pyproject.toml`:
  - `deepsteer` must not import `papers`, `scripts`, or `tests`.
  - `deepsteer.directions` and `deepsteer.geometry` must not import `torch` or
    `transformers` (README promises they are pure numpy; make the promise checkable).
  Wire `lint-imports` into CI.
- [ ] Tag `v0.2.0` with a CHANGELOG entry listing the move and the experimental markers.

Gate C (human): confirm the experimental list; confirm no pod is importing
`deepsteer.supplement`.

## 5. Phase D — artifacts out of the tree (target: one session, plus Zenodo upload by human)

Goal: `papers/` no longer carries binaries at the tip. This matches the existing decision
that Zenodo is the DOI of record for raw arrays; this phase is its implementation.

- [ ] **Inventory.** `git ls-files papers | grep -E '\.(npz|npy|pt|bin|safetensors|ckpt)$'`
  plus `tokenizer.json`/adapter directories under `papers/*/outputs/`. Produce
  `papers/ARTIFACT_MANIFEST.json`: path, size, sha256, which paper/cell produced it, which
  script reads it. Flag anything NC-derived (per the existing exclusion rule) so it is
  restricted rather than published.
- [ ] **Readers.** For every path in the manifest, find the reader (grep the path and its
  basename across `papers/` and `tests/regression/`). Change readers to go through one
  helper, `papers/build_common/artifacts.py::get(path)`, which returns the local file if
  present, otherwise downloads from the Zenodo record by DOI and verifies sha256. Cache in
  `papers/_artifacts/` (gitignored).
- [ ] **Regression tests.** `tests/regression/` must `pytest.skip` with the fetch command
  when an artifact is absent, not fail.
- [ ] **Human uploads to Zenodo** and supplies the DOI; write it into the manifest.
- [ ] **Remove at tip.** `git rm` the binaries. For each removed `outputs/` directory leave
  a `README.md`: what was here, the Zenodo DOI, the SHA at which it was last in-tree, and
  the fetch command. Tag the commit before removal `artifacts-last-in-tree`.
- [ ] Add the extensions to `.gitignore` under `papers/` so they cannot come back by
  accident; new pod outputs land in `papers/_artifacts/` or are uploaded.

Gate D (human): manifest reviewed; NC-derived items confirmed restricted; Zenodo record
published.

Note: this does not shrink `.git` for people who clone full history — constraint 1 rules
that out. It does shrink `--filter=blob:none` clones and GitHub's tarball, and it stops the
growth. That is the intended outcome.

## 6. Phase E — make the library usable by agents (optional, after D)

- [ ] A `.claude/skills/deepsteer-library/SKILL.md` for *consumers* (distinct from the
  methodology skills): how to load a model, collect activations, extract a direction, run
  the geometry panel, run an ablation and a steering sweep — with the exact function
  signatures and the construct-audit/intervention-validity gates referenced. Keep it under
  200 lines. This is the thing a lab's agent reads instead of reimplementing `directions/`.
- [ ] `llms.txt` at the repo root: one paragraph on what the library is, the install line,
  and links to README sections. Point it at the skill.
- [ ] Docstrings on every `__all__` symbol in `directions`, `geometry`, `causal`, `core`:
  one-line summary, args, returns, shape conventions. No docs site; docstrings plus README
  are the docs for now.

## 7. When to revisit a split

Reopen only if one of these becomes true: a second maintainer takes ownership of the
library; a lab or product needs an API-stability promise (1.0); or release cadence for the
library diverges from the research program's. If reopened, the mechanism is **extract the
library** (new repo from `--subdirectory-filter deepsteer` on a clone, under a new name
such as `deepsteer/deepsteer-core`) — not extract the papers — because that is the only
split that produces a light clone without rewriting the cited repo's history. Do not
recreate a repo under the old name; GitHub drops the rename redirect and SHA permalinks
would break.

## 8. Working agreements for this plan

- One PR per phase (C may be two: C1 and C2+C3). PR description lists every moved or
  deleted path and every shim added.
- Run the clean-venv wheel smoke test locally before opening any PR that touches
  `pyproject.toml`, `deepsteer/datasets/`, or package layout.
- If a task reveals something this plan got wrong about the repo, stop, note it in the PR,
  and ask — don't improvise a different structure.
- Commit messages: imperative, one line, body explains why; reference this file as
  `LIBRARY_RELEASE_PLAN §<phase>`.
