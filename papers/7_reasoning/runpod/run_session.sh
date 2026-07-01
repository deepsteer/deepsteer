#!/usr/bin/env bash
# Programmatic RunPod session for Paper 7 (reasoning refusal) Phase 0 GPU work.
#
#   spin up GPU -> rsync repo -> run Phase 0 chain (precision gate + smoke +
#   validity anchor) -> rsync back -> terminate
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C. Set
# KEEP_POD=1 to leave it running for debugging.
#
# GPT-OSS-20B is loaded DEQUANTIZED to bf16 (~40 GB) for a clean intervention, so
# the default GPU pool is 80 GB-class (A100-80GB / H100). The two distills (8B/14B)
# fit smaller cards; a smoke- or anchor-only run can use a 48 GB card via GPU_TYPES.
#
# Recommended flow (cost-minimizing):
#   1. Local gate (no GPU):  python papers/7_reasoning/scripts/local_test.py
#   2. Dry-run the chain:     python papers/7_reasoning/scripts/run_phase0.py --dry-run
#   3. Cheap GPU smoke:       VALIDATE=1 STAGE=smoke ./run_session.sh
#   4. Validity anchor:       STAGE=anchor ./run_session.sh   (confirm ~99% residual)
#   5. Precision gate:        STAGE=precision ./run_session.sh (GPT-OSS 80GB card)
#   6. Or everything at once: ./run_session.sh
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export HF_TOKEN=...                            # only if a repo is gated (panel is public)
#   VALIDATE=1 STAGE=smoke ./run_session.sh        # cheap validation, then stop
#   STAGE=precision ./run_session.sh               # just the GPT-OSS precision gate
#   ./run_session.sh                               # full Phase 0 (all stages)
#   KEEP_POD=1 ./run_session.sh                    # don't terminate at the end
#   REUSE_POD_ID=<id> ./run_session.sh             # attach to an existing pod
set -euo pipefail

# ---------------------------- config (override via env) ----------------------
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
GPU_TYPE="${GPU_TYPE:-}"
# 80 GB-class first (GPT-OSS bf16 ~40 GB + activations); smaller cards can run a
# smoke/anchor-only pass on the distills via GPU_TYPES override.
GPU_TYPES="${GPU_TYPES:-NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe,NVIDIA H100 NVL}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE,COMMUNITY}"
[ -n "$GPU_TYPE" ] && GPU_TYPES="$GPU_TYPE"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-200}"      # gpt-oss(~13GB dl)+distills(~16/28GB) + HF cache headroom
VOLUME_GB="${VOLUME_GB:-0}"
POD_NAME="${POD_NAME:-deepsteer-p7-reason}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"

# Step config (passed through to the remote runner).
VALIDATE="${VALIDATE:-0}"
STAGE="${STAGE:-all}"          # Phase 0: all | precision | smoke | anchor
MODELS="${MODELS:-all}"        # Phase 1: all | comma list of registry keys
N="${N:-}"                     # Phase 1: prompts/class override
MAXTOK="${MAXTOK:-}"           # Phase 1: max-new-tokens override
PIP_EXTRA="${PIP_EXTRA:-}"     # e.g. "transformers>=4.55" if gpt_oss unsupported
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
# Phase 0: remote_phase0.sh (default). Phase 1: REMOTE_SCRIPT=.../remote_phase1.sh
REMOTE_SCRIPT="${REMOTE_SCRIPT:-papers/7_reasoning/runpod/remote_phase0.sh}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# Standardized sync: shared universal excludes + keep only THIS paper's outputs.
SELF_PAPER="papers/7_reasoning"
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
echo ">> Launching Phase 0 [stage=$STAGE] detached on pod (log: $REMOTE_LOG)"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   ( PYTHONUNBUFFERED=1 REPO_DIR=$REMOTE_DIR \
     VALIDATE=$VALIDATE STAGE='$STAGE' MODELS='$MODELS' N='$N' MAXTOK='$MAXTOK' \
     PIP_EXTRA='$PIP_EXTRA' \
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
echo ">> Phase 0 finished (sentinel detected)."

# --------------------------------- download ----------------------------------
echo ">> Downloading results (model blobs excluded)"
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  --exclude 'ablated_model/' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/papers/7_reasoning/outputs/" \
  "$REPO_ROOT/papers/7_reasoning/outputs/"

echo ">> Done. Phase 0 outputs under papers/7_reasoning/outputs/{key}/."
echo ">> Inspect: precision_gate.json (0d), {ds_r1_llama8b}/smoke.json (0c),"
echo ">>          {ds_r1_llama8b}/refusal_decomposition_eop.json (validity anchor, expect ~99% residual)."
# pod terminated by the EXIT trap
