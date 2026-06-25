#!/usr/bin/env bash
# Remote Phase 2c — t_inst/t_post-inst diff-of-means + clustering (Zhao et al.).
set -uo pipefail
REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
MODELS="${MODELS:-all}"
VALIDATE="${VALIDATE:-0}"
N="${N:-}"
cd "$REPO_DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
trap 'touch "$REPO_DIR/.session_done"' EXIT

pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -1
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1 || true
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

case ",$MODELS," in
  *",gpt_oss_20b,"*|*",all,"*) NEED_TORCH=1 ;; *) NEED_TORCH=0 ;;
esac
if [ "$NEED_TORCH" = 1 ] && ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> torch trio -> 2.6.0+cu124 for GPT-OSS..."
  pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2 || echo "WARN torch upgrade failed"
fi

ARGS="--models $MODELS"
[ -n "$N" ] && ARGS="$ARGS --n $N"
echo ">> Phase 2c [VALIDATE=$VALIDATE] $ARGS"
VALIDATE="$VALIDATE" python papers/7_reasoning/scripts/run_phase2c.py $ARGS
RC=$?
echo ">> run_phase2c exit code: $RC"
exit $RC
