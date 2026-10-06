# Raw-array release plan (Zenodo, DOI of record)

**Status 2026-09-13 (W4-3): PUBLISHED.** Version DOI of record **10.5281/zenodo.22731361** (v1; concept DOI 10.5281/zenodo.22731360 resolves to the latest version); record https://zenodo.org/records/22731361; all five files' MD5 verified against the local build (tarball e2904a9143fc384df1fabdb1c5a641cc). Papers cite the version DOI. Build record: `scripts/build_release.py` wrote
`outputs/zenodo_v1/`: `deepsteer_fl_arrays_v1.tar.zst` (336 files, 2.77 GB uncompressed, 2.35 GB
compressed, sha256 `c229dd72594186d1c52083bd6ee1d2ac05e1c1e6f17ba1a68e0e964e15945ba5`), `MANIFEST.json`
(per file: path, bytes, sha256, group, unit/model/source run, HF repo + commit + dtype for every W4
file, license, cited_by; 241 MN-referenced paths), `PROVENANCE.md`, `REGENERATE.md`, `LICENSE`. No
separate MN tarball: every array the methods note cites lives in the FL tarball and is referenced by
path + sha256 (§1's shared-arrays-live-once rule). Exclusions applied per §2 (MORABLES-derived files,
datasets under `outputs/full/`, smoke/dry/pilot trees, weights). W4 files were re-hashed against
`manifest_w4.json` before staging. **Orion:** reserve the DOI on Zenodo, replace
`10.5281/zenodo.[reserved before submission]` in FL App E, MN §9 and `supplement/README.md`, upload
the five files, publish; then rebuild both papers.

Decision of record (Orion, 2026-09-10; WRITEUP_PHASE_PLAN Phase W4 item 5): the per-unit arrays
behind the flagship (FL) and the methods note (MN) are released on **Zenodo** as the DOI of record.
The public record holds every array not derived from a non-commercial source; MORABLES-derived
caches are excluded or restricted, each with a regeneration recipe. Nothing is uploaded in Session
W4-1; the DOI is minted in Session W4-3 after the pod arrays land and the manifests verify.

## 1. What goes in the public record

Per paper, one tarball of per-unit arrays plus the `MANIFEST.json` that indexes them. "Per-unit"
means the unit of independent variation the statistics resample over (prompt, pair, twin, rollout,
head), so every CI in either paper is re-derivable from the record without a model.

**FL tarball (`deepsteer_fl_arrays_v1.tar.zst`).**

| group | arrays | source run | license status |
|---|---|---|---|
| Decision anatomy (OLMo-3, Llama-3.1) | per-head write contributions + channel-matched specificity (544 heads × d), per-request-twin interchange deltas for every cell and every rank (`c1_inputs_*.npz`), `channel_act`, `Vbasis`, `harm`, `refusal` | D3 C1 sessions (incl. depth-L12) + W4 15.1 union run | public (request-twins hand-authored; V_moral from Moral Stories / fables / ETHICS, see §2) |
| Rank sweep + one-knob | `sweep_refusal`, `sweep_judgment`, `sweep_random` per k, per twin | D3 Amendment 4 + W4 15.1 | public |
| GPT-OSS Tier 1 + W4 14.3 | decision-token act-sample, refusal direction, per-item graded projections at `P_prefill` and `P_dec`, per-rollout token ids, V_moral sources at the decision token | D3 Amendment 12 + W4 14.3 | public (Heretic prompt set is MIT-licensed upstream; boundary twins hand-authored) |
| D1 P0–P3 | per-rollout window activations, refusal directions per position, MFT directions (Think, GPT-OSS) | W4 14.4 | public (Heretic prompts; MFT v2 pairs are ours) |
| Proto-refusal reliability | per-sample base and instruct last-token activations (800 × 4096 each), split-half directions, per-checkpoint proto-refusal directions (13 stage-3 states), trajectory JSON | W4 14.1 + Paper 5 caches | public |
| Cross-ablation | per-prompt outcomes under every condition, directions used | W4 14.5 | public |
| Calibration ladders | per-pair diff arrays behind the held-one-out bands, null resample arrays, PR profiles per position class (raw + standardized) with bootstrap arrays | D1 phase 2 + D2 in-format + W4 14.6 | **partly restricted**: Moral Stories / ETHICS per-pair diffs are public; any array whose *pairs* are MORABLES retellings is excluded (see §2) |
| Distilled cells | the 14 `supplement/{cells,figure_data}` artifacts (already public in-repo) | — | public |

**MN tarball (`deepsteer_mn_arrays_v1.tar.zst`).** The MN-cited subset: PR profiles with bootstrap
arrays for every (model × position × normalization) cell of MN Table 1 / Fig 1; the two-position
validity ladder arrays; the depth-asymmetry per-twin deltas at layers 12 and 16; the RMSNorm-fold
reconstruction arrays; the reply-inversion margins (harm vs 20 random directions, W4 14.6b). Shared
arrays with FL are **not duplicated**: the MN manifest references the FL tarball path + sha256 for
them (the supplement's shared-arrays-live-once rule, extended across deposits).

## 2. What is excluded or restricted, with regeneration recipes

- **MORABLES-derived caches (CC-BY-NC).** Any activation cache, per-pair diff array, or direction
  whose stimulus pairs are MORABLES retellings inherits the NC restriction and does **not** enter the
  public record. Of record, V_moral's three sources are Moral Stories, the *public-domain-derived*
  Understanding-Fables retellings, and ETHICS (D1 V-D1-2 dropped MORABLES from V_moral), so the
  headline arrays are clean. The exclusion bites on the early D1 G-axis / MORABLES pooling
  experiments only. Disposition: **excluded** from the public record; a **restricted Zenodo record**
  (access on request, NC terms stated) holds them if a reviewer needs the pre-V-D1-2 trail.
  Regeneration recipe: `papers/d1_moral_subspace/scripts/generate_morables.py` (needs the MORABLES
  release under its own license) → `phase2_extract.py --model allenai/Olmo-3-1025-7B` at L16.
- **Model weights, HF caches, LoRA adapters**: never deposited; the manifest pins repo + commit hash.
- **Generated rollout text**: token ids are deposited (needed to re-read positions); decoded text is
  regenerable from ids + tokenizer and is not deposited separately.
- **Anything gated (Llama-3.1)**: activations are derived data and are deposited; the manifest notes
  the gated base and the accepted-license requirement to regenerate.

## 3. Deposit layout

```
zenodo:<DOI>/
├── deepsteer_fl_arrays_v1.tar.zst      one tarball per paper
├── deepsteer_mn_arrays_v1.tar.zst
├── MANIFEST.json                        every file in both tarballs: path, bytes, sha256, unit,
│                                        model, source run, HF commit hash of the model that
│                                        produced it, license tag (public | restricted-NC), and
│                                        the paper figure / table / CLAIMS id it backs
├── PROVENANCE.md                        copy of deepsteer/supplement/PROVENANCE.md at the deposit commit
├── REGENERATE.md                        per-group regeneration recipes (script + args + commit)
└── LICENSE                              CC BY 4.0 for the arrays; code license stated by pointer
```

The manifest is built by extending `deepsteer/supplement/scripts/build.py` to walk the W4 output
tree (`papers/d3_decision_anatomy/outputs/w4/manifest_w4.json` already carries per-file sha256 and
per-load commit hashes; the deposit manifest merges the run manifests, never re-types a hash) and is
verified by `scripts/verify.py` before upload. The DOI is reserved first so it can be printed in the
papers, then the files are published.

## 4. Metadata

- **Title:** DeepSteer per-unit arrays: what refusal reads (flagship) and instruments before verdicts (methods note)
- **Authors:** Reblitz-Richardson, Orion (Distiller Labs)
- **License:** CC BY 4.0 for the deposited arrays and manifests (code stays under the repository license, by pointer)
- **Related identifiers** (`isSupplementTo` / `cites`): arXiv:2606.11375 (Paper 1 v2), arXiv:2608.25231 (Paper 2), arXiv:2608.27402 (Paper 3), arXiv:2609.14759 (FL), arXiv:2609.14754 (MN); the FL and MN ids were added once they existed
- **Keywords:** language models, refusal, moral representations, interpretability, activation patching, participation ratio
- **Version:** v1 = the W4 run of record; any later regeneration is a new Zenodo version with its own DOI and a changelog line
- **Description:** two sentences stating what the arrays are, that every CI in both papers is re-derivable from them without a model, and the NC exclusion with its restricted-record pointer

## 5. Replacement sentence for the three "on request" sites

Replace, at FL App E.7 (`0E_reproducibility.md`), MN §9 (`09_reproducibility.md`), and
`deepsteer/supplement/README.md` ("available to reviewers on request"):

> The per-unit arrays behind every figure and confidence interval (per-sample activations at each
> position class, per-pair contrasts, per-rollout window activations, per-twin interchange outcomes,
> per-head attributions) are deposited on Zenodo under CC BY 4.0 at doi:10.5281/zenodo.«DOI», indexed
> by a manifest with a sha256 per file and the HuggingFace commit hash of the model that produced it.
> Arrays derived from one non-commercial stimulus source (MORABLES) are excluded from the public
> record and available under their upstream terms in a restricted companion record, with the
> regeneration recipe in the deposit.

«DOI» is filled in Session W4-3 after minting; the sentence is otherwise final.

## 6. Session W4-3 checklist (not this session)

1. `python3 scripts/pod_w4.py --verify-manifest` clean; W4 arrays synced.
2. Build the deposit manifest (extend `build.py`); `verify.py` clean; license tag on every file; no
   MORABLES-derived path in the public tarballs (grep the stimulus provenance field).
3. Reserve the DOI; print it in FL App E.7, MN §9, supplement README; rebuild both PDFs.
4. Upload; publish; record the DOI + deposit commit in `papers/README.md` and PROVENANCE.md.
