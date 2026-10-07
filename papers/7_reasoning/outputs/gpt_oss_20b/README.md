# Binary artifacts moved to Zenodo

The arrays listed below were removed from this directory at the tip of `main` (LIBRARY_RELEASE_PLAN
§D). They stay in git history: the last commit that contains them is
[`341487e`](https://github.com/deepsteer/deepsteer/tree/341487e4a7b67046fb0e4ecf7dede893c52a3012/papers/7_reasoning/outputs/gpt_oss_20b) (tag `artifacts-last-in-tree`).

Fetch and verify from a clone:

```bash
python3 papers/build_common/artifacts.py fetch papers/7_reasoning/outputs/gpt_oss_20b/<file>
```

Scripts that read these arrays without loading a model resolve them automatically
(`papers/build_common/artifacts.py`). Records: public 10.5281/zenodo.23202782 (CC BY 4.0); restricted
10.5281/zenodo.23203126 (CC BY-NC 4.0, access on request).

| File | Bytes | sha256 | Where |
|---|---|---|---|
| `exp1_probe_directions.npz` | 1699294 | `ae77b5778ebfd452…` | [public](https://doi.org/10.5281/zenodo.23202782) |
| `persona_directions.npz` | 565950 | `7d6c801c47f4f7fc…` | [public](https://doi.org/10.5281/zenodo.23202782) |
| `position_directions.npz` | 306952 | `bed47dad8d60c82a…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
| `position_headline_vectors.npz` | 2950390 | `1e288165be946e68…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
| `two_site_headline_vectors.npz` | 4264170 | `8daa9785ee68acad…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
| `two_site_refusal_directions.npz` | 459468 | `24b7c159b060462d…` | [restricted](https://doi.org/10.5281/zenodo.23203126) |
