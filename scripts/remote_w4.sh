#!/usr/bin/env bash
# Remote runner for the W4 venue-quality pod (D3 PREREGISTRATION Amendments 14 + 15).
# Six model loads in the pre-registered order, Tier A then Tier B, per-unit saves + manifest.
# Launch via the shared launcher (80 GB card REQUIRED: GPT-OSS-20B bf16 ~40 GB):
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe" \
#   DISK_GB=250 REMOTE_SCRIPT=scripts/remote_w4.sh \
#     SELF_PAPER=papers/d3_decision_anatomy RESULTS_SUBPATH=outputs/w4 \
#     SYNC_EXTRA="papers/d1_moral_subspace/outputs/full papers/d2_decision_coupling/outputs/olmo3 \
#                 papers/d3_decision_anatomy/outputs/c1_inputs_llama31_L12.npz \
#                 papers/d3_decision_anatomy/outputs/c1_session_llama31_L12.json \
#                 papers/d1_moral_subspace/outputs/phase2/base/diffs_moral_stories.npz \
#                 papers/d1_moral_subspace/outputs/phase2/instruct/diffs_moral_stories.npz \
#                 papers/d1_moral_subspace/outputs/phase2/refusal_base.npz \
#                 papers/d1_moral_subspace/outputs/phase2/refusal_instruct.npz \
#                 papers/5_moral_alignment/outputs/measurement/stage3" \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   Partial rerun (e.g. 15.2 on the operating band + 14.5_gen), into its own subdir, then merged:
#     W4_MODELS=qwen25,olmo3_instruct W4_UNITS=15.2,14.5_gen W4_OUT_SUBDIR=outputs/w4_rerun1 \
#       RESULTS_SUBPATH=outputs/w4_rerun1 ... ./papers/d1_moral_subspace/runpod/run_session.sh
#     python3 scripts/pod_w4.py --merge-from papers/d3_decision_anatomy/outputs/w4_rerun1 --reason "..."
#   (prepend VALIDATE=1 for the no-model dry run on the pod; W4_MODELS / W4_UNITS subset the run.
#    The last three SYNC_EXTRA paths feed the 14.1 zero-GPU arm + end-of-run disattenuation; without
#    them the driver records 14.1_zero_gpu = missing_inputs, as the 2026-09-12 run did.)
#
# Flow (compute-ordering + test-gates-before-GPU):
#   1. local gates (no model): pytest tests/scripts/test_pod_w4.py + pod_w4.py --dry-run
#   2. VALIDATE=1: the same dry run on the pod (plumbing), then exit
#   3. real run: zero-GPU arm -> OLMo-3-Instruct (15.1 pilot gate first) -> Think -> base -> Llama ->
#      GPT-OSS -> Qwen; manifest checkpointed after every unit; verify-manifest at the end
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

# W4_OUT_SUBDIR (default outputs/w4): a partial rerun MUST write to its own subdir (e.g.
# outputs/w4_rerun1, launched with RESULTS_SUBPATH=outputs/w4_rerun1) and be folded into the main
# tree afterwards with `pod_w4.py --merge-from`; a fresh Manifest in outputs/w4 would clobber the
# full manifest of record on rsync-back.
OUT="$REPO_DIR/papers/d3_decision_anatomy/${W4_OUT_SUBDIR:-outputs/w4}"; mkdir -p "$OUT"

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 || pip install -q --break-system-packages -e . 2>&1 | tail -1
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate pytest 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
if ! python -c 'import torch; torch.accelerator' 2>/dev/null; then
  echo ">> upgrading torch trio to 2.6.0+cu124 for GPT-OSS mxfp4..."
  pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3 || echo ">> WARN torch upgrade failed"
fi
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"
[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ] && echo ">> HF_TOKEN: set" || echo ">> HF_TOKEN: unset (Llama-3.1 is gated!)"

# ---- no-model gates always run first ----
echo ">> W4 local gates:"
python -m pytest -q tests/scripts/test_pod_w4.py tests/geometry/test_reliability.py \
  tests/geometry/test_participation.py tests/datasets/test_request_twins_w4.py || { echo "LOCAL GATE FAILED"; exit 1; }
python scripts/pod_w4.py --dry-run --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE: dry run OK on the pod. Launch without VALIDATE for the real run."; exit 0
fi

# ---- VRAM guard (GPT-OSS needs 80 GB) ----
VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 70 ] && { echo "FATAL: W4 includes GPT-OSS-20B (bf16 ~40 GB); need an 80 GB card."; exit 1; }

# ---- real run ----
ARGS=""
[ -n "${W4_MODELS:-}" ] && ARGS="$ARGS --models $W4_MODELS"
[ -n "${W4_UNITS:-}" ] && ARGS="$ARGS --units $W4_UNITS"
[ -n "${W4_TIER:-}" ] && ARGS="$ARGS --tier $W4_TIER"
echo "==================== W4 pod: Tier A then Tier B ($ARGS) ===================="
python scripts/pod_w4.py --out "$OUT" $ARGS
RC=$?
python scripts/pod_w4.py --verify-manifest --out "$OUT" || echo "WARN: manifest verify reported mismatches"
echo ">> W4 done (exit $RC). rsync-back -> papers/d3_decision_anatomy/outputs/w4/ (manifest_w4.json + per-unit arrays)."
exit $RC
