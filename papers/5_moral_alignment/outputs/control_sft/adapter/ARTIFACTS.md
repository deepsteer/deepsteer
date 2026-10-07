# Binary artifacts moved to Zenodo

The arrays listed below were removed from this directory at the tip of `main` (LIBRARY_RELEASE_PLAN
§D). They stay in git history: the last commit that contains them is
[`341487e`](https://github.com/deepsteer/deepsteer/tree/341487e4a7b67046fb0e4ecf7dede893c52a3012/papers/5_moral_alignment/outputs/control_sft/adapter) (tag `artifacts-last-in-tree`).

Fetch and verify from a clone:

```bash
python3 papers/build_common/artifacts.py fetch papers/5_moral_alignment/outputs/control_sft/adapter/<file>
```

Scripts that read these arrays without loading a model resolve them automatically
(`papers/build_common/artifacts.py`). Records: public 10.5281/zenodo.23202782 (CC BY 4.0); restricted
10.5281/zenodo.23203126 (CC BY-NC 4.0, access on request).

| File | Bytes | sha256 | Where |
|---|---|---|---|
| `tokenizer.json` | 7137558 | `18e309ad7f9c6003…` | not deposited; regenerate: Stock allenai/Olmo-3-1025-7B tokenizer (chat_template.jinja from allenai/Olmo-3-7B-Instruct). Full: same art_sft.py comm |
