# DeepSteer paper supplement

Distilled, manifest-indexed artifacts behind the quantitative claims in the two
papers of this series:

- **Flagship** — *Refusal Reads Only a Slice of What the Model Knows*
- **Methods note** — *Calibrating Interpretability Instruments Before Trusting Their Verdicts*

Every figure and headline number in both papers resolves to a file here. Shared
artifacts (used by both papers) live here once; the papers cite this supplement
rather than duplicating them.

## What is and isn't here

**Included (distilled artifacts only):** per-head write contributions, calibrated
nulls, participation-ratio profiles, positive-control ladders, and interchange
rank-sweep outcomes — the numbers plotted and tabulated in the papers.

**Not included here: the per-unit arrays.** They are released on Zenodo as the DOI of
record, 10.5281/zenodo.22731361 (v1; https://zenodo.org/records/22731361;
`deepsteer_fl_arrays_v1.tar.zst`, 336 files, 2.8 GB uncompressed, CC BY 4.0;
built by `scripts/build_release.py`, plan in `RELEASE_PLAN.md`). Caches whose stimuli are
MORABLES retellings inherit that corpus's non-commercial license and are excluded from the
public record with a regeneration recipe (`REGENERATE.md` in the deposit); the headline
arrays are all deposited. Every distilled artifact here is reproducible from the deposited
arrays via the run scripts named in `PROVENANCE.md`.

## Layout

```
supplement/
├── MANIFEST.json     index of every artifact: sha256, provenance, which paper
│                     figures/tables cite it, and CSV column schemas
├── PROVENANCE.md     pinned model ids/revisions, seeds, harness + classifier versions
├── figure_data/      plot-ready distilled CSVs (one canonical copy each)
│   ├── bottleneck_pr.csv                PR by model x position x normalization  [shared]
│   ├── depth_asymmetry.csv              read-vs-commit A by model x layer, CIs   [shared]
│   ├── calibration_ladder_permodel.csv  per-model band / refusal / persona / null
│   ├── calibration_ladder_position.csv  two-position validity ladder (methods note)
│   ├── rank_sweep.csv                   R_refusal / R_judgment over k (OLMo-3)
│   ├── crystallization.csv              moral 0.869->0.999 vs proto-refusal 0.155
│   ├── reversibility.csv                GPT-OSS graded flips with Wilson CIs
│   └── head_attribution.csv             OLMo-3 Stage-1 per-head write attribution
├── cells/            distilled session summaries (JSON) — decisive cells, nulls,
│                     MDEs, verdicts; no raw activations
└── scripts/
    ├── build.py      regenerate MANIFEST.json (deterministic; sorted, no timestamps)
    └── verify.py     integrity (sha256 vs manifest) + shared-array mirror-drift check
```

## Shared arrays live once

`bottleneck_pr.csv` and `depth_asymmetry.csv` are used by both papers. The
canonical copy lives here; each paper keeps a small plotting mirror under its own
`figure_data/` so its arXiv tarball builds standalone. `scripts/verify.py` asserts
the mirrors carry the same values as the canonical file, so a number can only be
changed in one place — drift fails the check rather than passing silently.

## Reproduce

```bash
python3 deepsteer/supplement/scripts/build.py    # rebuild MANIFEST.json
python3 deepsteer/supplement/scripts/verify.py   # integrity + mirror check
```

Figures are regenerated from these CSVs by each paper's `figure_data/regen_*.py`
(no GPU, no network). The distilled `cells/*.json` are copied once from the run
outputs named in `PROVENANCE.md`; re-deriving them from raw activations needs the
raw caches (reviewer request) and the run scripts.

## Licensing

Code and distilled numeric artifacts here are released under the repository
license. One upstream stimulus source (MORABLES) is CC-BY-NC; the moral-content
directions and any activation caches derived from it inherit that non-commercial
restriction. The distilled numbers in this supplement are summary statistics, not
redistributions of the licensed text; the raw activation caches, which are closer
to the source, are the reason raw data is request-gated rather than posted.
