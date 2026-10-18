#!/usr/bin/env bash
# Remote runner for the ST T1.0 screen (papers/ST_PLAN.md v0.2.7 §2, §7; Amendments
# 2-5; T0.11). Orion launches (keys stay in Orion's terminal). One model loaded at a time, in
# MODELS order; each runs gates -> padded check -> trace-length pilot (step-down, re-price) ->
# VALIDATE -> panel -> H1 screen back to back before unloading; the last output is the re-priced
# Tier 1 table (outputs/t10/tier1_reprice.md) for G1.
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe" \
#   DISK_GB=250 REMOTE_SCRIPT=papers/st_live_reference/runpod/remote_st_screen.sh \
#     SELF_PAPER=papers/st_live_reference RESULTS_SUBPATH=outputs/t10 \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   VALIDATE=1 <same> ......  stack upgrade, local gates (incl. tiny real checkpoints of all three
#                             architectures) and the three dry runs on the pod, then exit. FIRST.
#   MODELS="olmo3_think" <same>  a subset, in the given order.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
MODELS="${MODELS:-olmo3_think qwen38_27b muse_glimmer_30b}"
MODELS="${MODELS//,/ }"
CLEAN_HF_CACHE="${CLEAN_HF_CACHE:-1}"   # drop each model's weights after its unit (disk)
PIP_INSTALL="${PIP_INSTALL:-1}"         # 0 = local VALIDATE on a machine with the stack already
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
PY="${PY:-python}"
trap 'touch "$REPO_DIR/.session_done"' EXIT

OUT="$REPO_DIR/papers/st_live_reference/outputs/t10"; mkdir -p "$OUT"
S=papers/st_live_reference/scripts/pod_st_screen.py
DATE_PIN="$(date -u +%F)"   # one date for the whole session (Muse renders it; Amendment 2 item 6)
echo ">> date pin: $DATE_PIN   models: $MODELS"

if [ "$PIP_INSTALL" = "1" ]; then
  # transformers 5.17.0 carries muse_glimmer (absent from the pods' default 5.12.1) and needs
  # torch >= 2.5; the image ships 2.4, so upgrade the trio exactly as W4/GPT-OSS did
  if ! $PY -c 'import torch,sys; v=tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(0 if v>=(2,6) else 1)' 2>/dev/null; then
    echo ">> torch $($PY -c 'import torch;print(torch.__version__)') < 2.6; upgrading the trio to 2.6.0+cu124 (W4 stack)..."
    pip install -q --break-system-packages "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
      --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
  fi
  pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 || pip install -q --break-system-packages -e . 2>&1 | tail -1
  pip install -q --break-system-packages "transformers==5.17.0" -U accelerate pytest pyyaml 2>&1 | tail -1 || true
  pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
fi
echo ">> torch: $($PY -c 'import torch;print(torch.__version__)' 2>&1)  transformers: $($PY -c 'import transformers;print(transformers.__version__)' 2>&1)  cuda: $($PY -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
$PY -c 'import transformers,sys; sys.exit(0 if transformers.__version__=="5.17.0" else 1)' \
  || { echo "FATAL: transformers is not 5.17.0 (Amendment 4 item 7)"; exit 1; }

# ---- inputs the screen needs (committed; never regenerated here) ----
for f in papers/st_live_reference/split_v1.json papers/kdg_panel/data/harm_divergence/hdiv_set.json \
         papers/kdg_panel/data/harm_divergence/h1_kept_ids.json papers/kdg_panel/models.yaml; do
  [ -f "$f" ] || { echo "FATAL: $f missing"; exit 1; }
done

# ---- no-model gates always run first (tiny real checkpoints exercise generate, hooks, padding
# and the VLM loaders on this torch/transformers before any large download) ----
echo ">> ST local gates:"
$PY -m pytest -q tests/scripts/test_st_split.py tests/scripts/test_st_formats.py \
  tests/scripts/test_st_screen.py tests/core/test_model_interface_vlm.py tests/kdg \
  -m "slow or not slow" || { echo "LOCAL GATE FAILED"; exit 1; }
for m in $MODELS; do
  $PY $S --model "$m" --dry-run --date-pin "$DATE_PIN" --out "$OUT/_dry" \
    || { echo "DRY RUN FAILED ($m)"; exit 1; }
done
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE: gates + dry runs OK. Launch without VALIDATE for the real run."; exit 0
fi

VRAM_GB="$($PY -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 75 ] && { echo "FATAL: need an 80 GB GPU (Muse Glimmer 30B is ~60 GB in bf16)."; exit 1; }

hf_cache_dir() {  # the HF hub cache directory of a registry key's repo
  $PY - "$1" <<'PY'
import sys, yaml
from huggingface_hub.constants import HF_HUB_CACHE
repo = yaml.safe_load(open("papers/kdg_panel/models.yaml"))["st"][sys.argv[1]]["repo"]
print(f"{HF_HUB_CACHE}/models--{repo.replace('/', '--')}")
PY
}

RC_ALL=0
for m in $MODELS; do
  echo "==================== t10/$m ===================="
  $PY $S --model "$m" --date-pin "$DATE_PIN" --out "$OUT"; rc=$?
  echo ">> $m exit $rc (0 ok, 2 gate failed, 3 envelope stop; banked either way)"
  $PY -c "import sys; sys.path.insert(0,'papers/kdg_panel/scripts'); from kdg_pod_lib import verify_manifest as v; bad=v('$OUT/$m/manifest_kdg.json'); print('manifest verify:', 'OK' if not bad else bad[:5])" \
    || echo "WARN: $m manifest verify did not run"
  [ "$rc" -ne 0 ] && RC_ALL=$rc
  if [ "$CLEAN_HF_CACHE" = "1" ]; then
    d="$(hf_cache_dir "$m")"; [ -d "$d" ] && rm -rf "$d" && echo ">> removed $d"
  fi
done

echo "==================== t10/reprice ===================="
$PY papers/st_live_reference/scripts/reprice_tier1.py --screen-dir "$OUT" || echo "WARN: re-price failed"
echo ">> ST T1.0 done (rc=$RC_ALL). rsync-back -> papers/st_live_reference/outputs/t10/"
exit $RC_ALL
