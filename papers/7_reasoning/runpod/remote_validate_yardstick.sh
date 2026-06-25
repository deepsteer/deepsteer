#!/usr/bin/env bash
# Path A gate: held-out, category-diverse refusal-yardstick validation on GPT-OSS.
set -uo pipefail
REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
trap 'touch "$REPO_DIR/.session_done"' EXIT

pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -1
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1 || true
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
if ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> torch trio -> 2.6.0+cu124 for GPT-OSS..."
  pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2 || echo "WARN torch upgrade failed"
fi
echo ">> held-out yardstick validation"
python papers/7_reasoning/scripts/validate_yardstick.py --key gpt_oss_20b \
  --prompts papers/5_moral_alignment/refusal_prompts.json \
  --n-train-eop "${NTRAIN_EOP:-128}" --n-train-cot "${NTRAIN_COT:-24}" \
  --n-test "${NTEST:-24}" --max-new-tokens "${MAXTOK:-512}" \
  --output papers/7_reasoning/outputs/gpt_oss_20b/yardstick_validation.json
echo ">> validate exit $?"
