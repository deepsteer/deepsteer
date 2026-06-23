#!/usr/bin/env bash
# Remote Phase 2c runner — Llama random-direction ablation control.
# Resolves whether Llama's moral-judgment drop under refusal ablation is
# refusal-specific or collateral (magnitude-matched random null + persona control).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
trap 'touch "$REPO_DIR/.session_done"' EXIT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

echo ">> cpu: $(nproc 2>/dev/null || echo '?') vCPUs; threads capped at $OMP_NUM_THREADS"
echo ">> python: $(python --version 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -2 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -2

if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo ">> HF_TOKEN: set"
else
  echo ">> HF_TOKEN: UNSET -- gated Llama will 401"
fi

# Ablation layer = the one the sweep chose for Llama (committed); fallback L13.
LAYER=$(python -c "import json;print(json.load(open('papers/6_cross_model/outputs/llama31/ablation_sweep.json'))['chosen_layer'])" 2>/dev/null || echo 13)
echo ">> Llama random-ablation control @L$LAYER (N=${N_RANDOM:-8} random + persona control)"

python papers/6_cross_model/scripts/random_ablation_control.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --refusal-npz papers/6_cross_model/outputs/llama31_instruct/refusal_directions.npz \
  --persona-npz papers/6_cross_model/outputs/llama31_instruct/persona_directions.npz \
  --layer "$LAYER" --n-random "${N_RANDOM:-8}" \
  --output papers/6_cross_model/outputs/llama31/random_ablation_control.json
RC=$?
echo ">> control exit code: $RC"
exit $RC
