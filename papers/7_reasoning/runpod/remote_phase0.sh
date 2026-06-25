#!/usr/bin/env bash
# Remote Phase 0 runner — executed ON the pod by run_session.sh.
#
# Runs the Paper 7 Phase 0 GPU chain (run_phase0.py): GPT-OSS-20B precision gate +
# positive control, reasoning-hook smoke, and the END-OF-PROMPT validity anchor.
# Always drops the .session_done sentinel on exit (even on failure) so the
# launcher never leaks the billed pod.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
STAGE="${STAGE:-all}"          # all | precision | smoke | anchor
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"

# Probes (exp1/persona/LDA gap) train on CPU; cap threads so tiny matmuls don't
# thrash every core while the GPU idles (same numerics, faster). Set before import.
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
# Optional extra installs (e.g. a newer transformers for gpt_oss): PIP_EXTRA="transformers>=4.55"
if [ -n "${PIP_EXTRA:-}" ]; then
  echo ">> PIP_EXTRA: installing $PIP_EXTRA"
  pip install -q --break-system-packages -U $PIP_EXTRA 2>&1 | tail -2 || true
fi

# Faster, more resilient large-shard downloads (GPT-OSS ~13 GB, distills 16-28 GB).
# Recent huggingface_hub replaced hf_transfer with Xet as the default backend
# (HF_HUB_ENABLE_HF_TRANSFER is now a no-op), so enable Xet's high-performance rust
# client instead. Set in the shell BEFORE any python imports huggingface_hub.
# Falls back cleanly if hf_xet can't install.
if pip install -q --break-system-packages hf_xet >/dev/null 2>&1; then
  export HF_XET_HIGH_PERFORMANCE=1
  echo ">> hf_xet high-performance transfer: enabled (HF_XET_HIGH_PERFORMANCE=1)"
else
  echo ">> hf_xet: install failed; using the default downloader"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

echo ">> transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"
if [ "$STAGE" = "all" ] || [ "$STAGE" = "precision" ]; then
  # GPT-OSS's mxfp4 quantizer calls torch.accelerator (added in torch 2.6). The
  # base image ships torch 2.4, so upgrade IN PLACE to 2.6.0+cu124 — matches the
  # image's CUDA 12.4 (same host-driver requirement), lower risk than jumping to a
  # cu12.8 image. torch's deps are already satisfied from the 2.4 install.
  if ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
    TV="$(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo ">> torch $TV lacks torch.accelerator (mxfp4 quantizer needs torch>=2.6); upgrading the torch trio to 2.6.0+cu124..."
    # Upgrade torch + torchvision + torchaudio TOGETHER to their matched set. The
    # image's torchvision/torchaudio were built against torch 2.4; bumping torch
    # alone leaves torchvision's compiled ops (torchvision::nms) unloadable, and
    # transformers imports torchvision in image_utils (a core path via
    # modeling_layers), which cascades into a GptOssForCausalLM import failure.
    pip install -q --break-system-packages \
      "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
      --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 \
      || echo ">> WARN: torch trio upgrade failed; the precision FIT check will fail-fast with a clear error."
    echo ">> torch now: $(python -c 'import torch, torchvision; print(torch.__version__, "tv", torchvision.__version__, "accelerator:", hasattr(torch, "accelerator"))' 2>&1)"
  else
    echo ">> torch $(python -c 'import torch; print(torch.__version__)') already has torch.accelerator (ok)"
  fi
  # GPT-OSS arch support in transformers (bf16-dequant needs no mxfp4 kernel).
  python -c 'from transformers import AutoConfig; AutoConfig.for_model("gpt_oss"); print(">> gpt_oss arch: supported")' 2>&1 \
    || echo ">> WARN: this transformers may not support gpt_oss; set PIP_EXTRA='\''transformers>=4.55'\'' if the FIT check fails."
fi

if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo ">> HF_TOKEN: set"
else
  echo ">> HF_TOKEN: UNSET -- needed only if a repo is gated (panel repos are public)."
fi

ONLY=""
[ "$STAGE" != "all" ] && ONLY="--only $STAGE"
echo ">> Phase 0 [VALIDATE=$VALIDATE] stage=$STAGE"

VALIDATE="$VALIDATE" python papers/7_reasoning/scripts/run_phase0.py $ONLY
RC=$?
echo ">> run_phase0 exit code: $RC"
exit $RC
