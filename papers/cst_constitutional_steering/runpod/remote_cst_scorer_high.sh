#!/usr/bin/env bash
# Remote runner for the CST scorer mini-pod (Amendment 10 item 1), one A100-80GB. Orion launches;
# the launcher passes HF_TOKEN. VALIDATE's pinned vLLM 0.31.0 stack.
#
#   setup (venv, stack check, vLLM smoke, gpt-oss-120b, gate tests) -> the probe's 128 saved A0
#   rollouts copied in (outputs/probe/gen_a0.jsonl, synced) -> gpt-oss-120b at effort high, nine
#   blind passes at temperature 1 (six same-order, three permuted) -> summary_high (single-sample
#   and majority-of-three kappas, per tenet kind; the price of the operator-only majority). No
#   subject generation. Wall cap 1.5 h.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/scorer_high"
RAWD="$REPO_DIR/papers/cst_constitutional_steering/outputs_raw/scorer_high"
mkdir -p "$OUT" "$RAWD"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session_scorer_high.log" 2>/dev/null; cp "$RAWD"/*.json "$RAWD"/*.jsonl "$OUT"/ 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
POD=papers/cst_constitutional_steering/scripts/cst_probe_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST SCORER-HIGH start $(date -u +%FT%TZ) out=$OUT"

MAX_H=1.5
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f cst_probe_pod.py ) &
WATCHDOG=$!

# ---- the vLLM venv (VALIDATE's block, unchanged pins) ----
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
    "transformers==5.17.0" "peft==0.21.2" pyyaml ninja pytest tokenizers -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q -e "." "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, shutil, torch, vllm, peft  # noqa: E401,F401
v = {p: md.version(p) for p in ("vllm", "torch", "transformers", "peft")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and v["peft"] == "0.21.2"
      and torch.cuda.is_available() and (torch.version.cuda or "").startswith("12.9")
      and shutil.which("ninja") is not None)
print(">> stack:", v, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
EOF
$PY - <<'EOF' || { echo "BAIL: vLLM smoke"; exit 1; }
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, gpu_memory_utilization=0.3)
out = llm.generate(["Hello"], SamplingParams(temperature=0.2, top_p=1.0, seed=0, max_tokens=8))
print(">> vLLM smoke OK:", len(out[0].outputs[0].token_ids), "tokens")
EOF
PROBE="$REPO_DIR/papers/cst_constitutional_steering/outputs/probe"
[ -f "$PROBE/gen_a0.jsonl" ] || { echo "BAIL: the probe's saved rollouts did not sync"; exit 1; }
cp "$PROBE/gen_a0.jsonl" "$PROBE/judge_gpt_oss_120b.json" "$RAWD"/

# ---- the scorer of record at its pin ----
$PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")}
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/models.json")"
echo ">> setup done at $(hours) h"
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_cst_probe.py tests/scripts/test_cst_validate.py \
  2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $POD --out "$RAWD" "$@"; }
run --stage judge --judge gpt_oss_120b_high || { echo "BAIL: judge gpt_oss_120b_high failed"; exit 1; }
echo ">> judge gpt_oss_120b_high done at $(hours) h"
run --stage summary_high || echo ">> summary_high failed"
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
