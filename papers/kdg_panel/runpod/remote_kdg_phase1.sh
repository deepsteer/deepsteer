#!/usr/bin/env bash
# Remote runner for KDG Phase 1 (papers/KDG_PHASE1_SPEC.md §5). Orion launches (keys stay in
# Orion's terminal); KDG_PROFILE selects the session:
#
#   KDG_PROFILE=p1a  Session A (OLMo-3: stage raw cells, new rows, C1, stage chat, dose arm)
#   KDG_PROFILE=p1a_fix  re-run of the p1a cells lost in the download (stages_raw, SFT stage chat)
#   KDG_PROFILE=p1b  Session B (Llama-3.1 base + Meta instruct, Tulu-3 stages, Qwen2.5: raw cells)
#
#   GPU_TYPES="NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe" \
#   DISK_GB=200 REMOTE_SCRIPT=papers/kdg_panel/runpod/remote_kdg_phase1.sh KDG_PROFILE=p1a \
#     SELF_PAPER=papers/kdg_panel RESULTS_SUBPATH=outputs/p1a \
#     ./papers/d1_moral_subspace/runpod/run_session.sh
#   VALIDATE=1 <same> ......  gates + dry run on the pod, then exit. Run this FIRST.
#   (Session B also needs HF_TOKEN exported with Llama-3.1 access.)
#
# Every step writes its own manifest under outputs/<profile>/<step>/ and is verified after it
# runs. A failed step is recorded and the session continues with the next step (spec §7 bails
# are the exceptions, marked below).
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
PROFILE="${KDG_PROFILE:-}"
cd "$REPO_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'touch "$REPO_DIR/.session_done"' EXIT

case "$PROFILE" in p1a|p1b|p1a_fix) ;; *) echo "FATAL: KDG_PROFILE must be p1a, p1b or p1a_fix (got '$PROFILE')"; exit 1;; esac
# p1a_fix writes into outputs/p1a (it re-runs Session A cells lost in the 2026-09-27 download)
OUT="$REPO_DIR/papers/kdg_panel/outputs/${PROFILE%_fix}"; mkdir -p "$OUT"
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
python -m pytest -q tests/kdg tests/scripts/test_pod_kdg_phase1.py || { echo "LOCAL GATE FAILED"; exit 1; }
N_ROWS="$(python - <<'PY'
import glob, json
n = sum(len(json.load(open(p))["scenarios"]) for p in glob.glob("papers/kdg_panel/data/*_scenarios_*.json"))
print(n)
PY
)"
echo ">> scenario rows on pod: $N_ROWS"
[ "${N_ROWS:-0}" -ge 580 ] || { echo "FATAL: fewer than 580 scenario rows: the Z4 top-up (round 3) is not synced"; exit 1; }
ls $D/round3_scenarios_*.json >/dev/null 2>&1 || { echo "FATAL: round3 scenario files missing"; exit 1; }
[ -f $D/screened_ids_a17_union.json ] || { echo "FATAL: screened id list missing"; exit 1; }
if [ "$PROFILE" != "p1b" ]; then
  python $S --dry-run --models olmo3_instruct,olmo3_sft,olmo3_base --units RAW,C1,C3CHAT,VALIDATE,DOSE,DOSE_BF,DOSE_LONG --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
else
  [ -n "${HF_TOKEN:-}" ] || { echo "FATAL: Session B needs HF_TOKEN (gated Llama-3.1)"; exit 1; }
  python $S --dry-run --models llama31_base,tulu3_sft,qwen25_base --units RAW --out "$OUT/_dry" || { echo "DRY RUN FAILED"; exit 1; }
fi
if [ "$VALIDATE" = "1" ]; then
  echo ">> VALIDATE: gates + dry run OK on the pod. Launch without VALIDATE for the real run."; exit 0
fi

VRAM_GB="$(python -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1e9)) if torch.cuda.is_available() else 0' 2>/dev/null || echo 0)"
echo ">> GPU VRAM: ${VRAM_GB} GB"
[ "${VRAM_GB:-0}" -lt 38 ] && { echo "FATAL: need >= 40 GB for 7-8B bf16 + batched generation."; exit 1; }

step() {  # step <name> <driver args...>
  local name="$1"; shift
  echo "==================== $PROFILE/$name ===================="
  python $S --out "$OUT/$name" "$@"; local rc=$?
  python $S --verify-manifest --out "$OUT/$name" || echo "WARN: $name manifest verify reported mismatches"
  return $rc
}
R3="$(ls $D/round3_scenarios_*.json | tr '\n' ' ')"

if [ "$PROFILE" = "p1a_fix" ]; then
  # re-run of the cells lost in the p1a download (KDG_RESULTS §15.4): stage raw cells + SFT chat
  step stages_raw --models olmo3_sft,olmo3_dpo --units RAW
  step stages_chat_sft --models olmo3_sft --units C3CHAT
  # P1-A5 budget-forced dose arm (no probe gate: the forced readout always yields a decision)
  step final_dose_bf --models olmo3_instruct --units DOSE_BF --scenario-ids-file $D/screened_ids_a17_union.json
  python - <<PY
import json
allids = json.load(open("$D/screened_ids_a17_union.json"))["ids"]
ids = [i for f in ("F1", "F3", "F4", "F5") for i in [x for x in allids if x.startswith(f + "-")][:4]]
json.dump(ids, open("$OUT/dose_long_ids.json", "w"))
PY
  step dose_long --models olmo3_instruct --units DOSE_LONG --scenario-ids-file "$OUT/dose_long_ids.json"
elif [ "$PROFILE" = "p1a" ]; then
  # keystone: C3 stage raw cells on the enlarged union (pilot gate n_shared is computed at analysis)
  step stages_raw --models olmo3_sft,olmo3_dpo --units RAW
  # new rows on the existing models: base raw; final raw + KDG-2 chat ladder
  step base_new --models olmo3_base --units RAW --scenarios $R3
  step final_new --models olmo3_instruct --units RAW,KDG2 --scenarios $R3
  # bail gate for the letter-chat readout: forward pass must reproduce generation's first step
  step final_validate --models olmo3_instruct --units VALIDATE
  if python -c "import json,sys;d=json.load(open('$OUT/final_validate/olmo3_instruct/forward_matches_generate.json'));sys.exit(0 if d['max_abs_nats']<=0.05 else 1)"; then
    step final_c1 --models olmo3_instruct --units C1
    step stages_chat --models olmo3_sft,olmo3_dpo --units C3CHAT
  else
    echo "BAIL: forward-pass readout != generation first step; C1 and stage chat cells skipped (spec §7)"
  fi
  # dose arm last, behind the anchor probe (spec C2 bail: anchor found in >= 80% of rollouts)
  python - <<PY
import json
# 4 per family (first by id), so a family-specific anchor failure (e.g. F3's tool menu) shows
allids = json.load(open("$D/screened_ids_a17_union.json"))["ids"]
ids = [i for f in ("F1", "F3", "F4", "F5") for i in [x for x in allids if x.startswith(f + "-")][:4]]
json.dump(ids, open("$OUT/dose_probe_ids.json", "w"))
PY
  step dose_probe --models olmo3_instruct --units d_chat_dose2,d_chat_dose2_filler --scenario-ids-file "$OUT/dose_probe_ids.json"
  if python - <<PY
import json, sys
ok = []
for arm in ("d_chat_dose2", "d_chat_dose2_filler"):
    rows = [json.loads(l) for l in open("$OUT/dose_probe/olmo3_instruct/" + arm + ".jsonl")]
    ok.append(sum(r["decision_step"] is not None and r["decision_step"] >= 0 for r in rows) / len(rows))
print("anchor-found rates", ok)
sys.exit(0 if min(ok) >= 0.8 else 1)
PY
  then
    step final_dose --models olmo3_instruct --units DOSE --scenario-ids-file $D/screened_ids_a17_union.json
  else
    echo "BAIL: dose anchor-found rate < 0.8; dose arm stopped (a cap change is a fork amendment)"
  fi
else
  step raw_lineages --models llama31_base,llama31_instruct_meta,tulu3_sft,tulu3_dpo,tulu3_final,qwen25_base,qwen25_instruct_p1 --units RAW
  # P1-A4: every instruct model also gets the neutral letter-only chat cells (no raw-only instruct findings)
  step chat_lineages --models llama31_instruct_meta,tulu3_sft,tulu3_dpo,tulu3_final,qwen25_instruct_p1 --units C3CHAT
fi
echo ">> KDG Phase 1 $PROFILE done. rsync-back -> papers/kdg_panel/outputs/$PROFILE/ (one manifest per step)."
