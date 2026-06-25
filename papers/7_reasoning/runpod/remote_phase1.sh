#!/usr/bin/env bash
# Remote Phase 1 runner — per-model subspace + two-site (EOP/CoT) decomposition.
# Executed ON the pod by run_session.sh (REMOTE_SCRIPT=.../remote_phase1.sh).
# Always drops the .session_done sentinel on exit so the launcher never leaks the pod.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
MODELS="${MODELS:-all}"          # all | comma list of registry keys
VALIDATE="${VALIDATE:-0}"
N="${N:-}"                       # prompts/class override (run_phase1 default 64)
MAXTOK="${MAXTOK:-}"             # max-new-tokens override (default 512)
cd "$REPO_DIR"

# Per-layer probes (exp1/persona/LDA gap) train on CPU; cap threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

trap 'touch "$REPO_DIR/.session_done"' EXIT

echo ">> cpu: $(nproc 2>/dev/null || echo '?') vCPUs; OMP/MKL capped at $OMP_NUM_THREADS"
echo ">> python: $(python --version 2>&1)"
python -c 'import torch; print(">> cuda:", torch.cuda.is_available(),
  (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))' 2>&1 || true

echo ">> installing deepsteer (editable)..."
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -2 \
  || pip install -q --break-system-packages -e . 2>&1 | tail -2
if [ -n "${PIP_EXTRA:-}" ]; then
  echo ">> PIP_EXTRA: installing $PIP_EXTRA"
  pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -2 || true
fi

# Resilient large-shard downloads (Xet high-performance client; hf_transfer is a
# no-op in recent huggingface_hub). Set before any python imports huggingface_hub.
if pip install -q --break-system-packages hf_xet >/dev/null 2>&1; then
  export HF_XET_HIGH_PERFORMANCE=1
  echo ">> hf_xet high-performance transfer: enabled"
else
  echo ">> hf_xet: install failed; using the default downloader"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

# GPT-OSS needs torch>=2.6 (mxfp4 quantizer calls torch.accelerator) and its
# matched torchvision/torchaudio; only upgrade when GPT-OSS is in the run.
case ",$MODELS," in
  *",gpt_oss_20b,"*|*",all,"*|",all,") NEED_TORCH=1 ;;
  *) [ "$MODELS" = "all" ] && NEED_TORCH=1 || NEED_TORCH=0 ;;
esac
if [ "$NEED_TORCH" = 1 ] && ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> torch $(python -c 'import torch;print(torch.__version__)') lacks torch.accelerator; "\
"upgrading torch trio to 2.6.0+cu124 (matched torchvision/torchaudio) for GPT-OSS..."
  pip install -q --break-system-packages \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 \
    || echo ">> WARN: torch trio upgrade failed; GPT-OSS load will fail-fast."
  echo ">> torch now: $(python -c 'import torch, torchvision; print(torch.__version__, "tv", torchvision.__version__, "accel", hasattr(torch,"accelerator"))' 2>&1)"
fi

if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo ">> HF_TOKEN: set"
else
  echo ">> HF_TOKEN: UNSET -- panel repos are public; only the base longitudinal probe is gated."
fi

ARGS="--models $MODELS"
[ -n "$N" ] && ARGS="$ARGS --n $N"
[ -n "$MAXTOK" ] && ARGS="$ARGS --max-new-tokens $MAXTOK"
echo ">> Phase 1 [VALIDATE=$VALIDATE] $ARGS"

VALIDATE="$VALIDATE" python papers/7_reasoning/scripts/run_phase1.py $ARGS
RC=$?
echo ">> run_phase1 exit code: $RC"
exit $RC
