# RunPod Session — Paper 3 7B GPU Work

Programmatic spin-up / execute / download / tear-down for the GPU-bound parts of
the Paper 3 extensions sprint (Extension A, plus C.2 causal and B/E when their
scripts exist). Designed to minimize billed GPU time: the pod is created right
before the run and **always** terminated afterward via a shell `trap`, including
on error or Ctrl-C.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `run_session.sh` | Local | Orchestrator: create pod, rsync repo up, exec, rsync results down, terminate |
| `remote_experiments.sh` | Pod | The actual experiment plan (toggle-driven, fail-soft) |

## Prerequisites

- Local tools: `curl`, `jq`, `ssh`, `rsync` (the orchestrator checks for these).
- `RUNPOD_API_KEY` exported (RunPod console → Settings → API Keys).
- An SSH keypair at `$SSH_KEY` (default `~/.ssh/id_ed25519`). The **public** key
  is injected into the pod via the `PUBLIC_KEY` env var, so the official RunPod
  PyTorch image authorizes you and starts `sshd`. You do *not* need to pre-register
  the key in the RunPod web UI for this path.

## Usage

```bash
export RUNPOD_API_KEY=...            # required
cd papers/3_moral_geometry/runpod

./run_session.sh                     # default session (see toggles below)
RUN_DILEMMA=1 ./run_session.sh       # also run the A.4 stretch step
KEEP_POD=1 ./run_session.sh          # leave the pod running for debugging
REUSE_POD_ID=<id> ./run_session.sh   # attach to an existing pod (no create/terminate)
```

## What runs (and minimizing compute)

Steps are toggled by env vars (`1` = run). Defaults:

| Toggle | Default | Step | Notes |
|---|---|---|---|
| `RUN_BOOTSTRAP` | 1 | A.2 Exp 3 bootstrap stability | Uses `--bootstrap-only`: reuses the synced 7B Exp 1 directions, **skips re-running Exp 1/2**. Longest step (~2 h at `N_BOOTSTRAP=200`). |
| `RUN_FRAGILITY` | 1 | A.3 Exp 7 framework fragility | `--olmo-only`. ~1 h. |
| `RUN_CAUSAL` | 1 | C.2 direction ablation + steering injection | Outputs moved to `outputs/probe_engineering_7B/` so they don't clobber 1B. |
| `RUN_DILEMMA` | 0 | A.4 dilemma probing + geometry | Stretch; off by default. |
| `RUN_TAXONOMY` | 1 | B data-driven taxonomy | **Auto-skips** until `scripts/data_driven_taxonomy.py` exists. |
| `RUN_EXTERNAL` | 1 | E MFV external robustness | **Auto-skips** until `scripts/external_dataset_robustness.py` exists. |

Cost levers:
- The `--bootstrap-only` flag is the main saver — the existing committed 7B
  directions (`outputs/exp1_2_3_7B/exp1_probe_directions.npz`) ride along in the
  rsync, so Exp 3 doesn't recompute Exp 1/2.
- Trap-based teardown means a crashed run still terminates the pod.
- `remote_experiments.sh` is fail-soft (no `set -e`): a single failed experiment
  doesn't abort the session, and partial results are still downloaded.
- Lower `N_BOOTSTRAP` (e.g. `N_BOOTSTRAP=50`) for a cheap smoke test first.

Rough estimate at the defaults: ~3.5–4 GPU-hours, ~$4–5 on an A100 80 GB.

## After the run (local)

New 7B outputs land under `papers/3_moral_geometry/outputs/` (e.g.
`exp1_2_3_7B/exp3_bootstrap_stability.json`, `exp7_fragility_7B/`,
`probe_engineering_7B/`). Then regenerate the scale-comparison figures with real
7B fragility data:

```bash
python papers/3_moral_geometry/scripts/scale_comparison_figures.py
```

## Notes / caveats

- **GPU selection / capacity.** OLMo-2 7B is ~14 GB in fp16, so a 24-48 GB card
  is plenty. The orchestrator tries an ordered list of GPU types across both
  clouds until one deploys (handles `SUPPLY_CONSTRAINT` automatically):
  - `GPU_TYPES="a,b,c"` — custom ordered candidate list (default spans 80 GB ->
    48 GB -> 24 GB cards).
  - `GPU_TYPE="NVIDIA L40S"` — force a single type.
  - `CLOUD_TYPES="SECURE,COMMUNITY"` — default order; set `CLOUD_TYPES=SECURE`
    to stay on RunPod's own DCs only (community hosts are cheaper but less
    consistent about exposing a public SSH port).
- `IMAGE`, `DISK_GB` are also overridable via env.
- Uses RunPod's GraphQL API directly (`curl` + `jq`); no `runpodctl` needed.
- If you Ctrl-C during the run, the trap still fires and terminates the pod.
  Always confirm in the RunPod console that the pod is gone if you saw a
  terminate warning.
