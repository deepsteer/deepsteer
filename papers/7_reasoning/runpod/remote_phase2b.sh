#!/usr/bin/env bash
# Remote Phase 2b runner — causal load-bearing ablation (moral / persona / refusal
# yardstick / random floor), distills at two budgets. Executed ON the pod by
# run_session.sh (REMOTE_SCRIPT=.../remote_phase2b.sh).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
MODELS="${MODELS:-all}"
VALIDATE="${VALIDATE:-0}"
N="${N:-}"
NRANDOM="${NRANDOM:-}"
MAXTOK="${MAXTOK:-}"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

trap 'touch "$REPO_DIR/.session_done"' EXIT

echo ">> cpu: $(nproc 2>/dev/null || echo '?') vCPUs; python: $(python --version 2>&1)"
python -c 'import torch; print(">> cuda:", torch.cuda.is_available(),
  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))' 2>&1 || true

echo ">> installing deepsteer (editable)..."
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -2 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -2
[ -n "${PIP_EXTRA:-}" ] && pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -2 || true

if pip install -q --break-system-packages hf_xet >/dev/null 2>&1; then
  export HF_XET_HIGH_PERFORMANCE=1; echo ">> hf_xet high-performance: enabled"
else
  echo ">> hf_xet: install failed; default downloader"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

case ",$MODELS," in
  *",gpt_oss_20b,"*|*",all,"*) NEED_TORCH=1 ;;
  *) NEED_TORCH=0 ;;
esac
if [ "$NEED_TORCH" = 1 ] && ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> upgrading torch trio to 2.6.0+cu124 (matched torchvision/torchaudio) for GPT-OSS..."
  pip install -q --break-system-packages \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 \
    || echo ">> WARN: torch trio upgrade failed; GPT-OSS load will fail-fast."
  echo ">> torch now: $(python -c 'import torch, torchvision; print(torch.__version__, "tv", torchvision.__version__, "accel", hasattr(torch,"accelerator"))' 2>&1)"
fi

[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] && echo ">> HF_TOKEN: set" || echo ">> HF_TOKEN: unset (panel repos public)"

ARGS="--models $MODELS"
[ -n "$N" ] && ARGS="$ARGS --n $N"
[ -n "$NRANDOM" ] && ARGS="$ARGS --n-random $NRANDOM"
[ -n "$MAXTOK" ] && ARGS="$ARGS --max-new-tokens $MAXTOK"
echo ">> Phase 2b [VALIDATE=$VALIDATE] $ARGS"

VALIDATE="$VALIDATE" python papers/7_reasoning/scripts/run_phase2b.py $ARGS
RC=$?
echo ">> run_phase2b exit code: $RC"
exit $RC
