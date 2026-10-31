#!/usr/bin/env bash
# Remote runner for MRB-G1 (papers/MRB_PLAN.md v0.3 §5). Orion launches; the launcher passes
# HF_TOKEN and VALIDATE. One vLLM venv (ST's pinned stack: vLLM 0.31.0 cu129, torch 2.13,
# transformers 5.17.0) runs generation, judging and the MB rider.
#
#   VALIDATE=1 (1 x A100-80GB): stack, access, data, gate tests; a generation probe per model
#     (16 scenarios x 4); gpt-oss-120b judging the first generation's criteria (~2,000 calls: the
#     envelope condition, "gpt-oss-120b runs on the pod's GPU type before any judging"); the MB
#     rider smoke; then projection.json: the session's A100-hours under each cut level and the
#     registered choice (none, a, a+b, a+b+c, or STOP).
#   session (2 x A100-80GB; needs outputs/g1_validate/projection.json via SYNC_OUTPUTS):
#     generation of both models in parallel, the MB rider, then two judge shards in parallel on
#     the tiers the projection chose, under a deadline and a watchdog.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
OUT_ROOT="$REPO_DIR/papers/mrb_morebench/outputs"
SUB=g1; [ "$VALIDATE" = 1 ] && SUB=g1_validate
OUT="$OUT_ROOT/$SUB"
mkdir -p "$OUT"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
POD=papers/mrb_morebench/scripts/mrb_g1_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> MRB-G1 start $(date -u +%FT%TZ) VALIDATE=$VALIDATE out=$OUT"

if [ "$VALIDATE" = 1 ]; then
  MAX_H="${SESSION_MAX_HOURS_VALIDATE:-0.75}"
else
  PROJ="$OUT_ROOT/g1_validate/projection.json"
  [ -f "$PROJ" ] || { echo "BAIL: $PROJ missing (sync it with SYNC_OUTPUTS)"; exit 1; }
  DECISION=$(python3 -c "import json; print(json.load(open('$PROJ'))['decision'])")
  echo ">> projection decision: $DECISION"
  [ "$DECISION" = "STOP" ] && { echo "BAIL: the projection overruns after cut (c); stop and report (MRB_PLAN v0.3)"; exit 1; }
  TIERS=$(python3 -c "import json; print(','.join('T%d' % t for t in json.load(open('$PROJ'))['chosen']['tiers']))")
  MAX_H=$(python3 -c "import json; print(json.load(open('$PROJ'))['session_max_wall_h'])")
  echo ">> session tiers $TIERS, wall cap ${MAX_H} h"
fi
DEADLINE=$(python3 -c "print($START + $MAX_H * 3600 - 600)")
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f mrb_g1_pod.py ) &
WATCHDOG=$!

# ---- the vLLM venv (ST's block, unchanged pins) ----
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
  "$VENV/bin/pip" install -q -e . "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
PATH="$VENV/bin:$PATH" $PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, torch, vllm  # noqa: E401
from vllm.model_executor.models import gpt_oss, qwen3_moe  # noqa: F401
v = {p: md.version(p) for p in ("vllm", "torch", "transformers")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and torch.cuda.is_available()
      and (torch.version.cuda or "").startswith("12.9"))
print(">> stack:", v, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
EOF
NGPU=$($PY -c "import torch; print(torch.cuda.device_count())")
[ "$VALIDATE" != 1 ] && [ "$NGPU" -lt 2 ] && { echo "BAIL: the session needs 2 GPUs, found $NGPU"; exit 1; }

# ---- models (the GPT-OSS repos' original/ and metal/ copies are skipped) and data ----
$PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"gpt_oss_20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
        "qwen3_30b": ("Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"),
        "gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")}
out = {k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
       for k, (r, v) in pins.items()}
print(json.dumps(out))
EOF
echo ">> models: $(cat "$OUT/models.json")"
M20=$($PY -c "import json; print(json.load(open('$OUT/models.json'))['gpt_oss_20b'])")
MQW=$($PY -c "import json; print(json.load(open('$OUT/models.json'))['qwen3_30b'])")
M120=$($PY -c "import json; print(json.load(open('$OUT/models.json'))['gpt_oss_120b'])")
$PY -c 'from deepsteer.datasets.morebench import fetch_morebench; from deepsteer.datasets.moralbench import fetch_moralbench; print(">> data:", fetch_morebench(), fetch_moralbench())' \
  || { echo "BAIL: data fetch or manifest check"; exit 1; }
SETUP_H=$(hours)
echo ">> setup done at ${SETUP_H} h"
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_mrb_g1_pod.py \
  tests/benchmarks/test_morebench_scoring.py tests/datasets/test_morebench.py 2>&1 | tail -2
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

V=""; [ "$VALIDATE" = 1 ] && V="--validate"
pod() {  # pod <gpu> <log> <args...>
  local gpu="$1" log="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$POD" "$@" --out "$OUT" $V --deadline "$DEADLINE" > "$OUT/$log" 2>&1
  local rc=$?; tail -1 "$OUT/$log" | cut -c1-400; echo ">> $log rc=$rc $(date -u +%T)"; return $rc
}

if [ "$VALIDATE" = 1 ]; then
  pod 0 gen_gpt_oss_20b.log generate --model gpt_oss_20b --local-dir "$M20"
  pod 0 gen_qwen3_30b.log generate --model qwen3_30b --local-dir "$MQW"
  pod 0 judge.log judge --shard 0/1 --tiers T1 --local-dir "$M120" \
    || { echo "BAIL: gpt-oss-120b did not judge on this GPU type"; exit 1; }
  pod 0 mb_rider.log mb-rider --local-dir "$M20"
  VH=$(python3 -c "print(round($(hours) + 0.05, 4))")   # + the manifest and download
  $PY "$POD" project --out "$OUT" --validate-hours "$VH" --setup-hours "$SETUP_H" > "$OUT/project.log" 2>&1
  echo ">> projection: $(python3 -c "import json; d=json.load(open('$OUT/projection.json')); print(d['decision'], [(l['cut'], l['session_a100_h']) for l in d['levels']], d['measured'])")"
else
  pod 0 gen_gpt_oss_20b.log generate --model gpt_oss_20b --local-dir "$M20" & G0=$!
  pod 1 gen_qwen3_30b.log generate --model qwen3_30b --local-dir "$MQW" & G1=$!
  wait $G0; wait $G1
  pod 1 judge1.log judge --shard 1/2 --tiers "$TIERS" --local-dir "$M120" & J1=$!
  pod 0 mb_rider.log mb-rider --local-dir "$M20"
  pod 0 judge0.log judge --shard 0/2 --tiers "$TIERS" --local-dir "$M120" & J0=$!
  wait $J0; wait $J1
fi

kill "$WATCHDOG" 2>/dev/null
$PY - "$OUT" <<'EOF'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1])
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
(out / "manifest.json").write_text(json.dumps({"files": files}, indent=1) + "\n")
print(f">> manifest: {len(files)} files")
EOF
echo ">> MRB-G1 end $(date -u +%FT%TZ) after $(hours) h"
