#!/usr/bin/env bash
# Programmatic RunPod session for Direction 1 Phase 2 (single-source V_moral GPU run).
#
#   spin up GPU -> rsync repo -> run the Phase 2 sequence (base + instruct chains, G2,
#   Track-1, two same-model-point G3) -> rsync results back -> terminate.
#
# The pod is ALWAYS terminated on exit (trap), including on error or Ctrl-C. Set KEEP_POD=1
# to leave it running for debugging.
#
# Memory peak is ONE 7B model at a time (extract loads base, then instruct; G3 loads base
# then instruct; each model.release()d before the next), so a single 48 GB card is plenty;
# 80 GB is fine too. Disk holds OLMo-3-7B (~14 GB) + OLMo-3-7B-Instruct (~14 GB) [+ the tiny
# OLMo-2-1B for the smoke] + HF cache headroom.
#
# Recommended flow (cost-minimizing; the no-GPU gates already pass locally):
#   1. Local gate (done):  python papers/d1_moral_subspace/scripts/phase2_local_test.py
#   2. Local dry run (done): VALIDATE=1 bash papers/d1_moral_subspace/runpod/phase2_session.sh
#   3. Cheap GPU smoke:     VALIDATE=1 ./run_session.sh          (tiny model on the pod)
#   4. Real run:            ./run_session.sh                     (OLMo-3-7B Base + Instruct)
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export HF_TOKEN=...                          # OLMo-3 is public; pass for rate limits / gating
#   VALIDATE=1 ./run_session.sh                  # cheap GPU plumbing smoke, then stop
#   ./run_session.sh                             # full Phase 2 (real models)
#   KEEP_POD=1 ./run_session.sh                  # don't terminate at the end
#   REUSE_POD_ID=<id> ./run_session.sh           # attach to an existing pod
set -euo pipefail

# ---------------------------- config (override via env) ----------------------
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first}"
GPU_TYPE="${GPU_TYPE:-}"
# 48 GB-class first (one 7B at a time fits in ~20 GB); 80 GB as fallback.
GPU_TYPES="${GPU_TYPES:-NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S,NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3}"
CLOUD_TYPES="${CLOUD_TYPES:-SECURE,COMMUNITY}"
[ -n "$GPU_TYPE" ] && GPU_TYPES="$GPU_TYPE"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${DISK_GB:-150}"      # two 7B (~28 GB) + tiny smoke model + HF cache headroom
VOLUME_GB="${VOLUME_GB:-0}"
POD_NAME="${POD_NAME:-deepsteer-d1-phase2}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/deepsteer}"

# Step config (passed through to the remote runner).
VALIDATE="${VALIDATE:-0}"
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3-1025-7B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-allenai/Olmo-3-7B-Instruct}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"  # pinned (smoke-validated); override if needed
PIP_EXTRA="${PIP_EXTRA:-}"     # e.g. "transformers==X" extra override
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
PILOT_N="${PILOT_N:-}"         # >0 -> cheap subset on the real model (think refusal pilot)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"  # generation cap override (think refusal)
P23_N="${P23_N:-}"             # >0 -> cap the P2/P3 GENERATION subset/side (P0/P1 stay full)
COT_WINDOW_N="${COT_WINDOW_N:-}"  # P2 in-trace symmetric window (first N reasoning tokens; def 256)
REMOTE_SCRIPT="${REMOTE_SCRIPT:-papers/d1_moral_subspace/runpod/remote_phase2.sh}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

# SELF_PAPER + RESULTS_SUBPATH are env-overridable so this launcher also drives sibling papers
# (e.g. Direction-2 session 1: SELF_PAPER=papers/d2_decision_coupling RESULTS_SUBPATH=outputs).
# Defaults reproduce the original Direction-1 Phase-2 behaviour exactly.
SELF_PAPER="${SELF_PAPER:-papers/d1_moral_subspace}"
RESULTS_SUBPATH="${RESULTS_SUBPATH:-outputs/phase2}"
RSYNC_EXCLUDE="$REPO_ROOT/papers/runpod_common/rsync_exclude.txt"

source "$REPO_ROOT/papers/runpod_common/session_lib.sh"
rp_require_bins
trap cleanup EXIT            # arm teardown before any pod exists
rp_provision_pod
rp_wait_for_ssh
rp_sync_up

# Optional extra paths pushed to the pod verbatim AFTER the filtered sync (e.g. gitignored
# source data a sibling paper's extraction needs). Space-separated, repo-relative; rsync -R
# recreates each path under $REMOTE_DIR. Transient pod use only (never committed to the repo).
if [ -n "${SYNC_EXTRA:-}" ]; then
  for p in $SYNC_EXTRA; do
    echo ">> extra-sync: $p"
    # cd into the repo root and use a RELATIVE source so rsync -R recreates exactly "$p" under
    # $REMOTE_DIR. macOS rsync ignores GNU's "/./" relative-root marker (it copied the full
    # absolute path instead), so anchor via cwd rather than the dot marker.
    ( cd "$REPO_ROOT" && rsync -az -R --exclude-from="$RSYNC_EXCLUDE" -e "ssh ${SSH_OPTS[*]}" \
        "$p" "root@$SSH_HOST:$REMOTE_DIR/" )
  done
fi

# ---------------------------------- execute ----------------------------------
REMOTE_LOG="$REMOTE_DIR/session.log"
REMOTE_DONE="$REMOTE_DIR/.session_done"
echo ">> Monitor from another terminal:"
echo "     ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no root@$SSH_HOST 'tail -f $REMOTE_LOG'"
echo ">> Launching Phase 2 [VALIDATE=$VALIDATE] detached on pod (log: $REMOTE_LOG)"
ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
  "cd $REMOTE_DIR && rm -f '$REMOTE_DONE' '$REMOTE_LOG' && \
   ( PYTHONUNBUFFERED=1 REPO_DIR=$REMOTE_DIR \
     VALIDATE=$VALIDATE BASE_MODEL='$BASE_MODEL' INSTRUCT_MODEL='$INSTRUCT_MODEL' \
     TRANSFORMERS_VERSION='$TRANSFORMERS_VERSION' PIP_EXTRA='$PIP_EXTRA' \
     PILOT_N='$PILOT_N' MAX_NEW_TOKENS='$MAX_NEW_TOKENS' P23_N='$P23_N' COT_WINDOW_N='$COT_WINDOW_N' \
     STEPS='${STEPS:-}' MODELS='${MODELS:-}' B5_N_RANDOM='${B5_N_RANDOM:-}' B5_N_DIR='${B5_N_DIR:-}' \
     B5_SIGMA_GRID='${B5_SIGMA_GRID:-}' SFT_MODEL='${SFT_MODEL:-}' MORAL_ROTATION_DEG='${MORAL_ROTATION_DEG:-}' \
     SWEEP='${SWEEP:-}' STANDARDIZE='${STANDARDIZE:-}' ROBUSTIFY='${ROBUSTIFY:-}' \
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
echo ">> Phase 2 finished (sentinel detected)."

# --------------------------------- download ----------------------------------
echo ">> Downloading results ($SELF_PAPER/$RESULTS_SUBPATH; npz/json only, no model weights)"
mkdir -p "$REPO_ROOT/$SELF_PAPER/$RESULTS_SUBPATH"
rsync -az \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.safetensors' \
  -e "ssh ${SSH_OPTS[*]}" \
  "root@$SSH_HOST:$REMOTE_DIR/$SELF_PAPER/$RESULTS_SUBPATH/" \
  "$REPO_ROOT/$SELF_PAPER/$RESULTS_SUBPATH/"

echo ">> Done. Results under $SELF_PAPER/$RESULTS_SUBPATH/ (see the remote runner's log for the file list)."
# pod terminated by the EXIT trap
