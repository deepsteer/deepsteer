#!/usr/bin/env bash
# Phase 2d-A: reply-inversion positive control on direct-answering INSTRUCT models
# (Zhao's setup). Qwen2.5-14B-Instruct (ungated) + Llama-3.1-8B-Instruct (gated;
# needs HF_TOKEN). No GPT-OSS -> no torch-trio upgrade needed.
set -uo pipefail
REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
REPOS="${REPOS:-Qwen/Qwen2.5-14B-Instruct,meta-llama/Llama-3.1-8B-Instruct}"
NTRAIN="${NTRAIN:-96}"
NTEST="${NTEST:-24}"
ALPHAS="${ALPHAS:-0.5,1,2,4}"   # steering as multiples of the residual norm
cd "$REPO_DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
trap 'touch "$REPO_DIR/.session_done"' EXIT

pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -1
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1 || true
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] && echo ">> HF_TOKEN: set (gated Llama ok)" \
  || echo ">> HF_TOKEN: UNSET -> Llama-3.1-Instruct will 401 (Qwen still runs)"

OUT="$REPO_DIR/papers/7_reasoning/outputs/control"
IFS=',' read -ra _REPOS <<< "$REPOS"
RC=0
for repo in "${_REPOS[@]}"; do
  tag="$(echo "$repo" | sed 's|.*/||; s|[^A-Za-z0-9]|_|g')"
  echo ">> reply-inversion positive control: $repo"
  python papers/7_reasoning/scripts/reply_inversion_control.py \
    --repo "$repo" --prompts papers/5_moral_alignment/refusal_prompts.json \
    --n-train "$NTRAIN" --n-test "$NTEST" --alphas "$ALPHAS" \
    --output "$OUT/${tag}_inversion.json" || { echo ">> $repo FAILED (continuing)"; RC=1; }
done
echo ">> control exit code: $RC"
exit 0   # don't fail the session if one (e.g. gated) model errors; per-model JSON tells the story
