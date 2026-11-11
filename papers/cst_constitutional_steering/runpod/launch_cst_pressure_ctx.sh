#!/usr/bin/env bash
# Launch card: the CST pressure line's session 2-ctx (P-ctx, P-ctx-n) (Amendments 27-28; PRESSURE_BUILD.md).
# The author runs, after reading 2-roll's contrast report (the line gate passed):
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_pressure_ctx.sh
#
# One A100-80GB, priced at about 5.9 A100-h (P-ctx 1.88 h and P-ctx-n 1.57 h of training, measured from 2-roll) (watchdog 9.0 h). Synced by name: the pinned C_anth
# and A0n texts, and 2-roll's scenarios, labels, contrast report and every rollout chunk.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pressure
python3 -c "import json,sys; sys.exit(0 if json.load(open('$P/$O/contrast.json'))['line_gate']['passes'] else 1)" \
  || { echo "BAIL: 2-roll's line gate did not pass (or contrast.json is missing): the line is the author's"; exit 1; }
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/a0n_faraday_v1.txt"
for f in scenarios.json act.jsonl act.json contrast.json; do SYNC="$SYNC,$O/$f"; done
for f in "$P/$O"/roll/*/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
export SELF_PAPER=$P RESULTS_SUBPATH=$O CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_pressure_ctx.sh MAX_SYNC_GB=2 DISK_GB=300
export POD_NAME=deepsteer-cst-pressure-ctx GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
