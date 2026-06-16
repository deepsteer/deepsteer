# Paper 5 (Phase 2) RunPod runbook

Mirrors the Paper 3 pattern: a local launcher (`run_session.sh`) that spins up
the cheapest available GPU, syncs the repo, runs `remote_experiments.sh`
detached on the pod, streams the log, downloads results, and **always
terminates the pod on exit** (so GPU time is never leaked).

## Cost model

The dominant cost is downloading 25 distinct 7B checkpoints (~14 GB each) for
the pipeline sweep. Mitigations baked in:

- `pipeline_study.py --purge-hf-cache` deletes each repo's weights after
  processing, so **peak disk ≈ one model**, not 350 GB.
- Coupling runs only on the 3 instruct-capable post-training states (pretraining
  checkpoints can't comply, so coupling there is uninformative).
- Step toggles (`ONLY=`, `RUN_*`) and a cheap `VALIDATE=1` smoke avoid paying
  for the full sweep before the code path is confirmed on the pod.
- The repo sync ships only the `deepsteer` package + **this paper's** outputs
  (~37 MB). Shared excludes in `papers/runpod_common/rsync_exclude.txt` drop
  model blobs, activation caches, and every other paper's outputs; each launcher
  re-includes just its own via `--include '/papers/<self>/outputs/***'`.

Rough estimate: validation ~5–15 min; Sprint 1 ~30–45 min; Sprint 2 pipeline
~2 h (download-bound) + coupling ~20 min. Whole session well under the
$12–18 budget.

## Local pre-flight (already done)

Run on MPS before touching a GPU — these produce the base directions the pod
consumes and confirm the code paths:

- `exp1_2_3_framework_geometry.py` on `Olmo-3-1025-7B` → `outputs/olmo3_base/`
  (probes 100% peak, geometry eff-dim 5, bootstrap stable band layers 15–31)
- `persona_probe_base.py` + `persona_morality_angles.py` → base persona baseline
- `probe_transfer.py` self-transfer on base → AUC 1.0, cos(base,fresh)=1.0
  (validates the transfer code path end-to-end on real weights)

The base outputs (`.npz`/`.json`, a few MB) are rsynced **up** to the pod; model
blobs are excluded from rsync.

## On the GPU

```bash
export RUNPOD_API_KEY=...                 # and an SSH keypair at ~/.ssh/id_ed25519

# 1. cheap validation run (one instruct load + a 2-state pipeline smoke), then stop
VALIDATE=1 ./run_session.sh

# 2. inspect outputs/_validate_*; decide raw-vs-chat input format (Sprint 1.1/1.2)

# 3. Sprint 1 (instruct de-risk): probe transfer raw+chat, behavioral, persona
ONLY="transfer behavioral persona" ./run_session.sh

# 4. Sprint 2 (full grid). Pipeline probes raw (Sprint 1 decision); coupling +
#    behavioral use chat internally.
ONLY="pipeline coupling" ./run_session.sh
```

Or one shot: `./run_session.sh`.

### Useful env overrides

| Var | Default | Purpose |
|---|---|---|
| `VALIDATE` | 0 | 1 = cheap smoke then stop |
| `ONLY` | (all) | `transfer behavioral persona pipeline coupling` |
| `INPUT_FORMAT` | raw | pipeline probing format (Sprint 1 decided raw; coupling/behavioral use chat) |
| `STABLE_LAYER` | 16 | coupling read layer (in the 15–31 stable band) |
| `KEEP_POD` | 0 | 1 = leave pod running to debug |
| `REUSE_POD_ID` | — | attach to an existing pod (no create/terminate) |
| `GPU_TYPES` | A100→…→A5000 | ordered fallback list |

## After the session (local)

```bash
python papers/5_moral_alignment/scripts/pipeline_figures.py \
    --pipeline-dir papers/5_moral_alignment/outputs/pipeline --layer 16
```

Sprint 3 (Heretic ablation) is a separate, shorter session — run
`heretic_ablation.py` with a real Arditi/Heretic harmful/harmless prompt set
(replace the placeholder) and re-run the measurement battery on the ablated
model.
