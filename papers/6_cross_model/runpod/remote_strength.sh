#!/usr/bin/env bash
# Remote Phase 2d runner — Llama refusal-ablation strength sweep (dose-response).
# Tests whether the moral-judgment drop tracks refusal removal monotonically
# (genuine coupling) or cliffs at the incomplete-ablation point (destabilization).
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

LAYER=$(python -c "import json;print(json.load(open('papers/6_cross_model/outputs/llama31/ablation_sweep.json'))['chosen_layer'])" 2>/dev/null || echo 13)
echo ">> Llama ablation-strength sweep @L$LAYER (alphas=${ALPHAS:-0,0.5,1.0,1.5,2.0})"

python papers/6_cross_model/scripts/ablation_strength_sweep.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --refusal-npz papers/6_cross_model/outputs/llama31_instruct/refusal_directions.npz \
  --prompts papers/5_moral_alignment/refusal_prompts.json \
  --layer "$LAYER" --alphas "${ALPHAS:-0,0.5,1.0,1.5,2.0}" \
  --null-alphas "${NULL_ALPHAS:-1.0,2.0}" --n-null "${N_NULL:-3}" --n-boot "${N_BOOT:-5000}" \
  --output papers/6_cross_model/outputs/llama31/ablation_strength_sweep.json
RC=$?
echo ">> strength sweep exit code: $RC"
exit $RC
