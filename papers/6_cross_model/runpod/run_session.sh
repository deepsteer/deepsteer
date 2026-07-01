#!/usr/bin/env bash
# Programmatic RunPod session for the Paper 6 (cross-model) Phase 1 GPU work.
#
#   spin up GPU -> rsync repo -> run per-model decomposition chain -> rsync back
#   -> terminate
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C. Set
# KEEP_POD=1 to leave it running for debugging.
#
# Each family pulls a base + an instruct (~14-16 GB fp16 each); run_phase1.py
# processes them sequentially and releases between models, so disk stays bounded.
# Llama-3.1 is gated: the pod's HF_TOKEN must have accepted-license access or its
# steps 401 (OLMo + Qwen still complete; the run summary marks llama31 FAILED).
#
# Recommended flow (cost-minimizing):
#   1. Local gate already done: python papers/6_cross_model/scripts/local_test.py
#   2. VALIDATE=1 MODELS=qwen25 ./run_session.sh   # cheap non-OLMo smoke (Phase 0c)
#   3. inspect outputs/qwen25_*; confirm hooks/decomposition produce sane JSON
#   4. ./run_session.sh                            # full 3-model Phase 1
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export HF_TOKEN=...                            # forwarded to the pod (Llama gate)
#   VALIDATE=1 MODELS=qwen25 ./run_session.sh      # cheap validation, then stop
#   ./run_session.sh                               # full session (all 3 families)
#   MODELS="olmo3,qwen25" ./run_session.sh         # subset
#   KEEP_POD=1 ./run_session.sh                    # don't terminate at the end
#   REUSE_POD_ID=<id> ./run_session.sh             # attach to an existing pod
set -euo pipefail

# ---------------------------- config (override via env) ----------------------
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
GPU_TYPE="${GPU_TYPE:-}"
GPU_TYPES="${GPU_TYPES:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation,NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA RTX A5000}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE,COMMUNITY}"
[ -n "$GPU_TYPE" ] && GPU_TYPES="$GPU_TYPE"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-120}"      # base + instruct per family, released between models
VOLUME_GB="${VOLUME_GB:-0}"
POD_NAME="${POD_NAME:-deepsteer-p6-xmodel}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"

# Step config (passed through to the remote runner). REMOTE_SCRIPT selects the
# phase: remote_phase1.sh (decomposition) or remote_phase2.sh (ablation battery).
VALIDATE="${VALIDATE:-0}"
MODELS="${MODELS:-all}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-papers/6_cross_model/runpod/remote_phase1.sh}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Standardized sync: shared universal excludes + keep only THIS paper's outputs.
SELF_PAPER="papers/6_cross_model"
RSYNC_EXCLUDE="$REPO_ROOT/papers/runpod_common/rsync_exclude.txt"

source "$REPO_ROOT/papers/runpod_common/session_lib.sh"
rp_require_bins
trap cleanup EXIT            # arm teardown before any pod exists
rp_provision_pod
rp_wait_for_ssh
rp_sync_up

# ---------------------------------- execute ----------------------------------
REMOTE_LOG="$REMOTE_DIR/session.log"
REMOTE_DONE="$REMOTE_DIR/.session_done"
echo ">> Monitor from another terminal:"
echo "     ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$SSH_HOST 'tail -f $REMOTE_LOG'"
echo ">> Launching Phase 1 detached on pod (log: $REMOTE_LOG)"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   ( PYTHONUNBUFFERED=1 REPO_DIR=$REMOTE_DIR \
     VALIDATE=$VALIDATE MODELS='$MODELS' \
     HF_TOKEN='$HF_TOKEN' HUGGING_FACE_HUB_TOKEN='$HF_TOKEN' \
     setsid bash $REMOTE_SCRIPT > '$REMOTE_LOG' 2>&1 < /dev/null & ) >/dev/null 2>&1"

echo ">> Launched. Streaming log below (run survives SSH drops; Ctrl-C terminates pod)."
echo "----------------------------------------------------------------------"
SEEN=0
while true; do
  CHUNK="$(ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "tail -n +$((SEEN + 1)) '$REMOTE_LOG' 2>/dev/null" 2>/dev/null || true)"
  if [ -n "$CHUNK" ]; then
    printf '%s\n' "$CHUNK"
    SEEN=$((SEEN + $(printf '%s\n' "$CHUNK" | wc -l)))
  fi
  if ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "test -f '$REMOTE_DONE'" 2>/dev/null; then
    FINAL="$(ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" "tail -n +$((SEEN + 1)) '$REMOTE_LOG' 2>/dev/null" 2>/dev/null || true)"
    [ -n "$FINAL" ] && printf '%s\n' "$FINAL"
    break
  fi
  sleep 3
done
echo "----------------------------------------------------------------------"
echo ">> Phase 1 finished (sentinel detected)."

# --------------------------------- download ----------------------------------
echo ">> Downloading results (model blobs excluded)"
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  --exclude 'ablated_model/' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/papers/6_cross_model/outputs/" \
  "$REPO_ROOT/papers/6_cross_model/outputs/"

echo ">> Done. Phase 1 outputs under papers/6_cross_model/outputs/{key}/refusal_decomposition.json."
echo ">> Next (local): cross-model table -> python papers/6_cross_model/scripts/phase1_table.py"
# pod terminated by the EXIT trap
