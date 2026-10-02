#!/usr/bin/env bash
# Remote runner for KDG Phase 2 (papers/KDG_F6_F8_SPEC.md §10). Orion launches (keys stay in
# Orion's terminal); KDG_PROFILE selects the session:
#
#   KDG_PROFILE=p2a  pilot: OLMo-3-7B-Instruct and Llama-3.1-8B-Instruct (Meta), one load each:
#                    F6–F8 pilot cells (P2PILOT), turns-since-norm rider on the model's own Phase 1
#                    screen (TSN), forward-vs-generate check (VALIDATE)
#   KDG_PROFILE=p2b  turns-since-norm follow-ups (P2-A4): Llama-3.1 Meta counterbalanced filler order
#                    (TSN_ROT, KDG-A16) + token-distance ladder (TSN_LEN, 600 / 2,000 tokens) on both
#                    models, each on its own Phase 1 screen, with VALIDATE (incl. a 2,000-token prompt)
#   KDG_PROFILE=p2a_e1 / p2a_e2  extras E1 (OLMo-3 SFT/DPO) and E2 (within-RL sweep): refused here
#                    until the author schedules them at the pod gate (E2 also needs P2-A1 pushed)
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe" \
#   DISK_GB=150 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase2.sh KDG_PROFILE=p2a \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p2a \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   VALIDATE=1 <same> ......  gates + dry run on the pod, then exit. Run this FIRST.
#   (Needs HF_TOKEN exported with Llama-3.1 access.)
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
PROFILE="${KDG_PROFILE:-}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

case "$PROFILE" in
  p2a|p2b) ;;
  p2a_e1|p2a_e2) echo "FATAL: $PROFILE is an optional extra (spec §10) and is not scheduled; the author decides at the pod gate"; exit 1;;
  *) echo "FATAL: KDG_PROFILE must be p2a or p2b (got '$PROFILE')"; exit 1;;
esac
OUT="$REPO_DIR/papers/kdg_panel/outputs/$PROFILE"; mkdir -p "$OUT"
S=papers/kdg_panel/scripts/pod_kdg_phase1.py
D=papers/kdg_panel/data

echo ">> cuda: $(python -c 'import torch;print(torch.cuda.is_available())' 2>&1)"
pip install -q --break-system-packages -e ".[all]" 2>&1 | tail -1 || pip install -q --break-system-packages -e . 2>&1 | tail -1
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
pip install -q --break-system-packages "transformers==$TRANSFORMERS_VERSION" -U accelerate pytest pyyaml 2>&1 | tail -1 || true
pip install -q --break-system-packages hf_xet >/dev/null 2>&1 && export HF_XET_HIGH_PERFORMANCE=1
echo ">> transformers: $(python -c 'import transformers;print(transformers.__version__)' 2>&1)"

# ---- no-model gates always run first ----
echo ">> KDG local gates:"
python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py tests/scripts/test_pod_kdg_phase2.py \
  || { echo "LOCAL GATE FAILED"; exit 1; }
# author rule (2026-10-01): the Phase 3 action-position pre-registration exists before this pod
[ -f papers/KDG_PHASE3_SPEC.md ] || { echo "FATAL: papers/KDG_PHASE3_SPEC.md missing (Phase 3 prereg is due before the Phase 2 pod)"; exit 1; }
ITEMS="$(ls $D/p2pilot_items_A_*.json $D/p2pilot_items_B_*.json 2>/dev/null | tr '\n' ' ')"
if [ "$PROFILE" = "p2a" ]; then
[ "$(echo $ITEMS | wc -w)" -eq 2 ] || { echo "FATAL: need exactly one merged item file per half (got: $ITEMS)"; exit 1; }
python - $ITEMS <<'PY' || { echo "FATAL: item files not pilot-complete"; exit 1; }
import sys
from deepsteer.kdg.phase2 import load_items
n = {}
for p in sys.argv[1:]:
    items, meta = load_items(p)
    assert meta.get("external_label_rater"), f"{p}: external labels missing (spec §2.1)"
    for it in items:
        n[it.family] = n.get(it.family, 0) + 1
assert n == {"F6": 24, "F7": 12, "F8": 12}, n
print(">> pilot items:", n)
PY
fi
for f in screened_ids_a17_union.json screened_ids_llama31_meta.json tsn_filler_turns.json \
         tsn_filler_turns_600.json tsn_filler_turns_2000.json; do
  [ -f "$D/$f" ] || { echo "FATAL: $D/$f missing"; exit 1; }
done

VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 38 ] && { echo "FATAL: need >= 40 GB for 7-8B bf16."; exit 1; }

if [ "$PROFILE" = "p2a" ]; then
  python $S --dry-run --models olmo3_instruct,olmo3_sft --units P2PILOT,TSN,VALIDATE --items $ITEMS \
    --scenario-ids-file $D/screened_ids_a17_union.json --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
else
  python $S --dry-run --models llama31_instruct_meta --units VALIDATE,TSN_ROT,TSN_LEN \
    --scenario-ids-file $D/screened_ids_llama31_meta.json --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
fi
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE: gates + dry run OK on the pod. Launch without VALIDATE for the real run."; exit 0
fi

step() {  # step <name> <driver args...>
  local name="$1"; shift
  echo "==================== $PROFILE/$name ===================="
  python $S --out "$OUT/$name" "$@"; local rc=$?
  python $S --verify-manifest --out "$OUT/$name" || echo "WARN: $name manifest verify reported mismatches"
  return $rc
}

# Bail (spec §12): VALIDATE runs first in each step's unit list; a forward-vs-generate mismatch fails
# that unit and is recorded; the pilot cells still run and the gate analysis reports G6 per model.
if [ "$PROFILE" = "p2a" ]; then
  step olmo3_instruct --models olmo3_instruct --units VALIDATE,P2PILOT,TSN --items $ITEMS \
    --scenario-ids-file $D/screened_ids_a17_union.json
  step llama31_instruct_meta --models llama31_instruct_meta --units VALIDATE,P2PILOT,TSN --items $ITEMS \
    --scenario-ids-file $D/screened_ids_llama31_meta.json
else  # p2b: the KDG-A16 discriminator first (author), then the ladder
  step llama31_instruct_meta --models llama31_instruct_meta --units VALIDATE,TSN_ROT,TSN_LEN \
    --scenario-ids-file $D/screened_ids_llama31_meta.json
  step olmo3_instruct --models olmo3_instruct --units VALIDATE,TSN_LEN \
    --scenario-ids-file $D/screened_ids_a17_union.json
fi
echo ">> KDG Phase 2 $PROFILE done. rsync-back -> papers/kdg_panel/outputs/$PROFILE/ (one manifest per step)."
