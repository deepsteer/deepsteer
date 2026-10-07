# Binary artifacts moved to Zenodo

The arrays listed below were removed from this directory at the tip of `main` (LIBRARY_RELEASE_PLAN
§D). They stay in git history: the last commit that contains them is
[`341487e`](https://github.com/deepsteer/deepsteer/tree/341487e4a7b67046fb0e4ecf7dede893c52a3012/papers/7_reasoning/outputs/ds_r1_llama8b) (tag `artifacts-last-in-tree`).

Fetch and verify from a clone:

```bash
python3 papers/build_common/artifacts.py fetch papers/7_reasoning/outputs/ds_r1_llama8b/<file>
```

Scripts that read these arrays without loading a model resolve them automatically
(`papers/build_common/artifacts.py`). Records: public 10.5281/zenodo.23202782 (CC BY 4.0); restricted
10.5281/zenodo.23203126 (CC BY-NC 4.0, access on request).

| File | Bytes | sha256 | Where |
|---|---|---|---|
| `exp1_probe_directions.npz` | 3199646 | `d0fa68cb91c55c03…` | [public](https://doi.org/10.5281/zenodo.23202782) |
| `persona_directions.npz` | 1065902 | `e2b3107f9404995f…` | [public](https://doi.org/10.5281/zenodo.23202782) |
| `position_directions.npz` | 566768 | `97115a0efc0fcc0b…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
| `position_headline_vectors.npz` | 4195574 | `eb296f41202bb95f…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
| `two_site_headline_vectors.npz` | 6293226 | `8662b041faac2e32…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
| `two_site_refusal_directions.npz` | 848900 | `8f8a077e12e55a85…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
