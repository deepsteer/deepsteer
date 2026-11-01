#!/usr/bin/env bash
# Remote runner for MRB Phase 2 (papers/MRB_PLAN.md v0.9.2; MB_PLAN Amendment 5): Muse Glimmer 30B
# and Qwen3.8-27B, base and with ST's adapter, on the MoReBench response field, with the
# MoralBench rider. Orion launches; the launcher passes HF_TOKEN. G1's vLLM venv and stack checks,
# plus ST's peft pin for the HF phase. No separate VALIDATE (v0.9.2): its checks are stages 1-4.
#
#   1 setup and gates   2 HF phase, per model (rider, merge, merge gate)   3 none k0 on 500, per
#   model (the rates)   4 sync point: judge pilot, pilot criteria, projection (proceed or STOP, k per
#   model)   5 adapter at k 0..k-1 behind the parity gate, then none k1 where k = 2   6 the full
#   judge, k0 first. Every step resumes by job id, so a relaunch on the pulled outputs continues.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
OUT_ROOT="$REPO_DIR/papers/mrb_morebench/outputs"
[ "$VALIDATE" = 1 ] && { echo "BAIL: Phase 2 has no separate VALIDATE (MRB_PLAN v0.9.2)"; exit 1; }
SUB=p2
OUT="$OUT_ROOT/$SUB"
mkdir -p "$OUT"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
POD=papers/mrb_morebench/scripts/mrb_p2_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> MRB-P2 start $(date -u +%FT%TZ) VALIDATE=$VALIDATE out=$OUT"

ENVELOPE_H=9.0
MAX_H=$(python3 -c "print($ENVELOPE_H / 2)")  # the envelope's wall on two GPUs, from the start
DEADLINE=$(python3 -c "print($START + $MAX_H * 3600 - 600)")
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f mrb_p2_pod.py ) &
WATCHDOG=$!

# ---- the vLLM venv (G1's block, unchanged pins) plus ST's peft for the HF phase ----
VENV=/workspace/venv_vllm
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f
CU129_INDEX=https://download.pytorch.org/whl/cu129
CONSTRAINTS=papers/st_live_reference/runpod/vllm_constraints.txt
if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
  "$VENV/bin/pip" install -q "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
    "torchaudio==2.11.0+cu129" --index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
    "transformers==5.17.0" pyyaml ninja pytest -c "$CONSTRAINTS" --extra-index-url "$CU129_INDEX" \
    2>&1 | tail -2
  "$VENV/bin/pip" install -q -e ".[lora]" "peft==0.21.2" "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, shutil, torch, vllm  # noqa: E401
import peft  # noqa: F401
v = {p: md.version(p) for p in ("vllm", "torch", "transformers", "peft")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and v["peft"] == "0.21.2"
      and torch.cuda.is_available() and (torch.version.cuda or "").startswith("12.9")
      and shutil.which("ninja") is not None)
print(">> stack:", v, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
EOF
$PY - <<'EOF' || { echo "BAIL: vLLM smoke (engine start-up or sampler)"; exit 1; }
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, gpu_memory_utilization=0.3)
out = llm.generate(["Hello"], SamplingParams(temperature=0.6, top_p=0.95, top_k=0, seed=0,
                                             max_tokens=8))
print(">> vLLM smoke OK:", len(out[0].outputs[0].token_ids), "tokens at top-p 0.95")
EOF
NGPU=$($PY -c "import torch; print(torch.cuda.device_count())")
[ "$NGPU" -lt 2 ] && { echo "BAIL: MRB-P2 needs 2 GPUs (one model per GPU; the judge is tensor-parallel), found $NGPU"; exit 1; }

# ---- models, adapters, data ----
$PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"muse_glimmer_30b": ("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
        "qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
        "gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")}
out = {k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
       for k, (r, v) in pins.items()}
print(json.dumps(out))
EOF
echo ">> models: $(cat "$OUT/models.json")"
M120=$($PY -c "import json; print(json.load(open('$OUT/models.json'))['gpt_oss_120b'])")
$PY - <<'EOF' || { echo "BAIL: an adapter is missing or not the registered one"; exit 1; }
import sys
sys.path.insert(0, "papers/mrb_morebench/scripts")
import mrb_p2_pod as Z
for m, d in Z.ADAPTERS.items():
    got = Z.sha_file(d / "adapter_model.safetensors")
    print(">> adapter", m, got[:16], "OK" if got == Z.ADAPTER_SHA[m] else "MISMATCH")
    if got != Z.ADAPTER_SHA[m]:
        raise SystemExit(1)
EOF
$PY -c 'from deepsteer.datasets.morebench import fetch_morebench; from deepsteer.datasets.moralbench import fetch_moralbench; print(">> data:", fetch_morebench(), fetch_moralbench())' \
  || { echo "BAIL: data fetch or manifest check"; exit 1; }
SETUP_H=$(hours)
echo ">> setup done at ${SETUP_H} h"
# the full output goes to gate_tests.log (downloaded with the results): the first session's bail
# here left only pytest's summary line, not the failing test's traceback
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_mrb_p2_pod.py tests/scripts/test_mrb_p2_analyze.py \
  tests/benchmarks/test_morebench_scoring.py tests/datasets/test_morebench.py > "$OUT/gate_tests.log" 2>&1
GATE_RC=$?
tail -2 "$OUT/gate_tests.log"
[ "$GATE_RC" = 0 ] || { echo "BAIL: gate tests (traceback in gate_tests.log)"; exit 1; }
echo ">> STAGE 1 done (setup and gates) at $(hours) h"

pod() {  # pod <gpu> <log> <args...>
  local gpu="$1" log="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$POD" "$@" --out "$OUT" --deadline "$DEADLINE" >> "$OUT/$log" 2>&1
  local rc=$?; tail -1 "$OUT/$log" | cut -c1-400; echo ">> $log rc=$rc $(date -u +%T)"; return $rc
}
first_half() {  # stages 2-3 for one model: the HF phase, then none at k0 on all 500
  local gpu="$1" m="$2"
  pod "$gpu" "hf_$m.log" hf-phase --model "$m" || { echo ">> $m HF phase failed"; return 1; }
  pod "$gpu" "gen_${m}_none.log" generate --model "$m" --arm none --k-from 0 --k 1 || return 1
}
first_half 0 muse_glimmer_30b & C0=$!
first_half 1 qwen38_27b & C1=$!
wait $C0; R0=$?; wait $C1; R1=$?
[ $((R0 + R1)) -gt 0 ] && { echo "BAIL: stage 2 or 3 failed (muse rc=$R0, qwen rc=$R1); stop and report"; exit 1; }
echo ">> STAGE 3 done (HF phases, none k0) at $(hours) h"

# ---- stage 4: the sync point ----
pod 0,1 judge_pilot.log judge --pilot --local-dir "$M120" || { echo "BAIL: the judge pilot failed"; exit 1; }
SPENT=$(python3 -c "print(round(2 * ($(hours) + 0.05), 4))")
$PY "$POD" project-live --out "$OUT" --spent-a100-h "$SPENT" --envelope-h "$ENVELOPE_H" > "$OUT/project.log" 2>&1
PROJ="$OUT/projection.json"
[ -f "$PROJ" ] || { echo "BAIL: the sync point wrote no projection"; exit 1; }
echo ">> projection: $(python3 -c "import json; d=json.load(open('$PROJ')); print(d['decision'], d.get('stop_reason', ''), 'k', d['k'], 'total A100-h', d['session_a100_h'], 'rest cap', d['rest_max_wall_h'], {m: (c['no_verdict_share'], c['final_reached_share']) for m, c in d['pilot_criteria'].items()})")"
DECISION=$(python3 -c "import json; print(json.load(open('$PROJ'))['decision'])")
[ "$DECISION" = "STOP" ] && { echo "BAIL: STOP at the sync point; stop and report (MRB_PLAN v0.9.2)"; exit 1; }
KMUSE=$(python3 -c "import json; print(json.load(open('$PROJ'))['k']['muse_glimmer_30b'])")
KQWEN=$(python3 -c "import json; print(json.load(open('$PROJ'))['k']['qwen38_27b'])")
DEADLINE=$(python3 -c "import json; print(min($DEADLINE, $(date +%s) + json.load(open('$PROJ'))['rest_max_wall_h'] * 3600 - 600))")
echo ">> STAGE 4 done at $(hours) h; k: muse $KMUSE, qwen $KQWEN"

# ---- stages 5-6 ----
second_half() {  # stage 5 for one model: the adapter arm at k 0..k-1, then none's k1 where k = 2
  local gpu="$1" m="$2" k="$3"
  pod "$gpu" "gen_${m}_adapter.log" generate --model "$m" --arm adapter --k-from 0 --k "$k" || return 1
  [ "$k" = 2 ] && { pod "$gpu" "gen_${m}_none.log" generate --model "$m" --arm none --k-from 1 --k 2 || return 1; }
  return 0
}
second_half 0 muse_glimmer_30b "$KMUSE" & C0=$!
second_half 1 qwen38_27b "$KQWEN" & C1=$!
wait $C0; wait $C1
echo ">> STAGE 5 done at $(hours) h"
pod 0,1 judge.log judge --local-dir "$M120"
echo ">> STAGE 6 done at $(hours) h"

kill "$WATCHDOG" 2>/dev/null
rm -rf "$OUT_ROOT/${SUB}_merged"
$PY - "$OUT" <<'EOF'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
(out / "manifest.json").write_text(json.dumps({"files": files}, indent=1) + "\n")
print(f">> manifest: {len(files)} files")
EOF
echo ">> MRB-P2 end $(date -u +%FT%TZ) after $(hours) h"
