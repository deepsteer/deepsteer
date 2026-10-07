# Binary artifacts moved to Zenodo

The arrays listed below were removed from this directory at the tip of `main` (LIBRARY_RELEASE_PLAN
§D). They stay in git history: the last commit that contains them is
[`341487e`](https://github.com/deepsteer/deepsteer/tree/341487e4a7b67046fb0e4ecf7dede893c52a3012/papers/1_accuracy_vs_fragility/outputs/phase_c_tier2/c3/confound_analytic) (tag `artifacts-last-in-tree`).

Fetch and verify from a clone:

```bash
python3 papers/build_common/artifacts.py fetch papers/1_accuracy_vs_fragility/outputs/phase_c_tier2/c3/confound_analytic/<file>
```

Scripts that read these arrays without loading a model resolve them automatically
(`papers/build_common/artifacts.py`). Records: public 10.5281/zenodo.23202782 (CC BY 4.0); restricted
10.5281/zenodo.23203126 (CC BY-NC 4.0, access on request).

| File | Bytes | sha256 | Where |
|---|---|---|---|
| `margins_analytic1000.npz` | 65446 | `a303d672ee7b3e0e…` | [public](https://doi.org/10.5281/zenodo.23202782) |
| `margins_smoke_analytic.npz` | 16258 | `86303883e4656bb4…` | [public](https://doi.org/10.5281/zenodo.23202782) |
