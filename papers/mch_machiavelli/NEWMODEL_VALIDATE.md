# New-model rows: VALIDATE pod report

Pod of 2026-11-03 (one A100-SXM4-80GB, 0.43 h wall), `runpod/remote_mch_newmodel.sh` at `c353bff`,
VALIDATE=1. Rules: `papers/MCH_PLAN.md` Amendments 4, 6, 7 and 8. Outputs:
`outputs/newmodel/validate_*`, `validate_decision.json`, `session_validate.log`.

| check | result |
|---|---|
| stack | vLLM 0.31.0+cu129, torch 2.13.0+cu129, transformers 5.17.0, gym 0.25.1, asteval 1.0.10: OK |
| vLLM smoke | OK |
| models | GPT-OSS-20B at `6cee5e81`, Qwen3-30B-A3B-Thinking-2507 at `144afc2f` |
| engine and bundle | fetched from GitHub and Google Drive, every file hash-verified; the local-cache fallback was not needed |
| gate tests | 18 passed (12 runner, 6 fetch), none skipped |
| render parity | both models, both agents: the pod's first render equals the local test's |
| setup | 0.26 h |

| model × agent (30 decisions each) | first-attempt parse | fallback | mean attempts | replies at the 256 cap | loop overrides |
|---|---|---|---|---|---|
| GPT-OSS-20B CoTAgent | 0.70 | 0.03 | 1.67 | 0.42 | 4 |
| GPT-OSS-20B GoodCoTAgent | 0.90 | 0.00 | 1.17 | 0.14 | 4 |
| Thinking-2507 CoTAgent | 0.10 | **0.70** | 4.13 | 0.91 | 4 |
| Thinking-2507 GoodCoTAgent | 0.27 | **0.53** | 3.53 | 0.87 | 4 |

- **Amendment 8's swap fired:** Thinking-2507's fallback exceeds 10% for both agents; the main run
  uses Qwen3-30B-A3B-Instruct-2507 at `0d7cf239` (`validate_decision.json`). Cause located (MCH-A8):
  the model reasons in prose through the closed think block.
- **GPT-OSS-20B** sits inside the bar at VALIDATE (0.03 and 0.00); its fallback of record is read at
  the pilot. Its CoTAgent retries are mostly the paper's 256-token cap cutting the JSON's "reasoning"
  field (42% of attempts at the cap), as the paper's own path would.
- **None** of these prompts would have been cut by the paper's chat limit.
- **Instruct-2507 has not run on a pod.** The main run checks its render against the local test's
  before stage 3 and applies the 10% fallback bar to its row.
- **The VALIDATE summary step** crashed after the decision was written (its glob matched
  `validate_decision.json`); fixed to directories only and run locally on these outputs.
- **Rates at VALIDATE's concurrency (4 episodes):** GPT-OSS 2.2e-4 A100-h per decision,
  Thinking-2507 1.1e-3. The pilot runs 40 episodes at once and measures the rate the stop rule uses.
