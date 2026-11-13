#!/usr/bin/env bash
# Launch card: the loop's second iteration (CST Amendment 44 item 5). The author runs:
#
#   export RUNPOD_API_KEY=... HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/launch_cst_loop2.sh
#
# One A100-80GB, priced at about 7.0 A100-h (watchdog 7.5 h); results in outputs/loop2_v1. Synced
# by name: the loop pod's set (the cache, 2-ctx's P-ctx record and adapter, S0a's cell A' rollouts,
# C_anth and the eval index, the smoke pass's fixed adapter and its letters), and from the loop pod
# (outputs/loop_v1): the rule-right kind (scenarios, the teacher's rollouts and labels), the selected
# content cell and the cell rows of A0 (the registered reference) and of config 2's first draw.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=papers/cst_constitutional_steering
O=outputs/pressure
[ -f "$P/data/loop_cache_v1.json" ] || { echo "BAIL: the cache record is missing"; exit 1; }
SYNC="outputs/_sync/C_anth.txt,outputs/_sync/eval_8gram_hashes_v1.npy"  # the eval index: write_kind's check
for f in scenarios.json act.jsonl act.json contrast.json train_p_ctx.json; do SYNC="$SYNC,$O/$f"; done
for f in "$P/$O"/roll/a0s/chunk_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
d=$(python3 -c "import json; p=list(json.load(open('$P/$O/train_p_ctx.json'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
[ -f "$P/$d/adapter_model.safetensors" ] || { echo "BAIL: $d's adapter is not here"; exit 1; }
SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
for f in "$P"/outputs/s0a/gen/a0/forced_plain_*.jsonl "$P"/outputs/s0a/gen/a0s/forced_plain_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
F=outputs/pilot_sdf  # Amendment 31 item 2: the smoke pass's fixed adapter and its saved HF letters
SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json"
K=outputs/loop_v1
[ -f "$P/$K/cell_gate.json" ] || { echo "BAIL: the loop pod's outputs are not here"; exit 1; }
for f in kinds/rule_right/scenarios.json kinds/rule_right/act.jsonl kinds/rule_right/act.json cell_gate.json; do SYNC="$SYNC,$K/$f"; done
for f in "$P/$K"/kinds/rule_right/roll/a0s/chunk_*.jsonl "$P/$K"/gen/a0/cell_plain_*.jsonl "$P/$K"/gen/c2_rule_right/cell_plain_*.jsonl; do SYNC="$SYNC,${f#$P/}"; done
export SELF_PAPER=$P RESULTS_SUBPATH=outputs/loop2_v1 CST_SESSION=1 SYNC_OUTPUTS="$SYNC"
export REMOTE_SCRIPT=$P/runpod/remote_cst_loop2.sh MAX_SYNC_GB=3 DISK_GB=300
export POD_NAME=deepsteer-cst-loop2 GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe"
exec bash papers/runpod_common/run_session.sh
