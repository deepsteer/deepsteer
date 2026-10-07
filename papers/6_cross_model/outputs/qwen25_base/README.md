# Binary artifacts moved to Zenodo

The arrays listed below were removed from this directory at the tip of `main` (LIBRARY_RELEASE_PLAN
§D). They stay in git history: the last commit that contains them is
[`341487e`](https://github.com/deepsteer/deepsteer/tree/341487e4a7b67046fb0e4ecf7dede893c52a3012/papers/6_cross_model/outputs/qwen25_base) (tag `artifacts-last-in-tree`).

Fetch and verify from a clone:

```bash
python3 papers/build_common/artifacts.py fetch papers/6_cross_model/outputs/qwen25_base/<file>
```

Scripts that read these arrays without loading a model resolve them automatically
(`papers/build_common/artifacts.py`). Records: public 10.5281/zenodo.23202782 (CC BY 4.0); restricted
10.5281/zenodo.23203126 (CC BY-NC 4.0, access on request).

| File | Bytes | sha256 | Where |
|---|---|---|---|
| `exp1_probe_directions.npz` | 2455614 | `5d5b736c68c8b582…` | [public](https://doi.org/10.5281/zenodo.23202782) |
| `persona_directions.npz` | 817974 | `7c936a09ee4753df…` | [public](https://doi.org/10.5281/zenodo.23202782) |
