#!/usr/bin/env bash
# Remote runner for the KDG pilot pod (KDG_PANEL_SPEC §8 step 4 / §11). Two model loads:
# OLMo-3-7B-Instruct (chat + raw cells), then OLMo-3 base (raw cells). Launch via the shared
# launcher from the repo root (Orion runs this; keys stay in Orion's terminal):
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe" \
#   DISK_GB=120 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_pilot.sh \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/pilot \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#
#   VALIDATE=1 <same> ......  no-model dry run on the pod (plumbing), then exit — run this FIRST.
#   KDG_MODELS=olmo3_instruct KDG_UNITS=d_chat_dose0,j_stated  subset the run.
#
# Flow (compute-ordering + test-gates-before-GPU): local gates (pytest + dry run) -> VALIDATE
# exit -> real run -> verify-manifest. Outputs rsync back to papers/kdg_panel/outputs/pilot/.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

OUT="$REPO_DIR/papers/kdg_panel/${KDG_OUT_SUBDIR:-outputs/pilot}"; mkdir -p "$OUT"

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 || pip install -q --break-system-packages -e . 2>&1 | tail -1
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate pytest pyyaml 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"

# ---- no-model gates always run first ----
echo ">> KDG local gates:"
python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_pilot.py || { echo "LOCAL GATE FAILED"; exit 1; }
python papers/kdg_panel/scripts/pod_kdg_pilot.py --dry-run --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
N_SCEN="$(python - <<'PY'
import glob, json
n = sum(len(json.load(open(p))["scenarios"]) for p in glob.glob("papers/kdg_panel/data/pilot_scenarios_*.json"))
print(n)
PY
)"
echo ">> scenario count on pod: $N_SCEN"
[ "${N_SCEN:-0}" -ge 30 ] || { echo "FATAL: fewer than 30 scenarios synced; the scenario set is not the committed pilot set"; exit 1; }
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE: dry run OK on the pod. Launch without VALIDATE for the real run."; exit 0
fi

VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 38 ] && { echo "FATAL: need >= 40 GB for 7B bf16 + 32-way batched generation."; exit 1; }

ARGS=""
[ -n "${KDG_MODELS:-}" ] && ARGS="$ARGS --models $KDG_MODELS"
[ -n "${KDG_UNITS:-}" ] && ARGS="$ARGS --units $KDG_UNITS"
echo "==================== KDG pilot pod ($ARGS) ===================="
python papers/kdg_panel/scripts/pod_kdg_pilot.py --out "$OUT" $ARGS
RC=$?
python papers/kdg_panel/scripts/pod_kdg_pilot.py --verify-manifest --out "$OUT" || echo "WARN: manifest verify reported mismatches"
echo ">> KDG pilot done (exit $RC). rsync-back -> papers/kdg_panel/outputs/pilot/ (manifest_kdg.json + per-cell jsonl/npz)."
exit $RC
