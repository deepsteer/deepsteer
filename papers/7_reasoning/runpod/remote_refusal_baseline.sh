#!/usr/bin/env bash
# Remote clean-refusal baseline diagnostic (Phase 2b precondition). Cheap: no
# ablation loop, one budget, panel loop. Executed ON the pod by run_session.sh
# (REMOTE_SCRIPT=.../remote_refusal_baseline.sh).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
MODELS="${MODELS:-all}"
N="${N:-24}"
MAXTOK="${MAXTOK:-1024}"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
trap 'touch "$REPO_DIR/.session_done"' EXIT

echo ">> python: $(python --version 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -2 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -2
if pip install -q --break-system-packages hf_xet >/dev/null 2>&1; then
  export HF_XET_HIGH_PERFORMANCE=1; echo ">> hf_xet: enabled"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

case ",$MODELS," in
  *",gpt_oss_20b,"*|*",all,"*) NEED_TORCH=1 ;; *) NEED_TORCH=0 ;;
esac
if [ "$NEED_TORCH" = 1 ] && ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> upgrading torch trio to 2.6.0+cu124 for GPT-OSS..."
  pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 || echo ">> WARN: torch upgrade failed."
  echo ">> torch now: $(python -c 'import torch;print(torch.__version__, hasattr(torch,"accelerator"))' 2>&1)"
fi

echo ">> refusal baseline: models=$MODELS n=$N max_new_tokens=$MAXTOK"
python papers/7_reasoning/scripts/refusal_baseline.py --models "$MODELS" \
  --prompts papers/5_moral_alignment/refusal_prompts.json \
  --n "$N" --max-new-tokens "$MAXTOK" --output-dir papers/7_reasoning/outputs
RC=$?
echo ">> refusal_baseline exit code: $RC"
exit $RC
