#!/usr/bin/env bash
# Remote runner for the CST probe pod (Amendment 9 item 4), one A100-80GB. Orion launches; the
# launcher passes HF_TOKEN. ST's, MRB's and MCH's pinned vLLM 0.31.0 stack (VALIDATE's block).
#
#   setup (venv, stack check, vLLM smoke, models, gate tests) -> engine, one process per
#   configuration (st first: its A0 rollouts feed the judges) -> judge gpt-oss-120b (four passes)
#   -> judge Olmo-3.1-32B-Instruct -> summary. Each stage writes outputs/probe/<stage>.json; a failed
#   engine configuration does not stop the others; the judges need the st configuration. Wall cap
#   3.0 h.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/probe"
RAWD="$REPO_DIR/papers/cst_constitutional_steering/outputs_raw/probe"
mkdir -p "$OUT" "$RAWD"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session_probe.log" 2>/dev/null; cp "$RAWD"/*.json "$RAWD"/*.jsonl "$OUT"/ 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
POD=papers/cst_constitutional_steering/scripts/cst_probe_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST PROBE start $(date -u +%FT%TZ) out=$OUT"

MAX_H=3.0
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
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: the pinned constitution text did not sync"; exit 1; }

# ---- models: the subject, the scorer of record and the 10% reader at their pins ----
$PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
        "gpt_oss_120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
        "olmo31_32b_instruct": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae")}
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/models.json")"
echo ">> setup done at $(hours) h"
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_cst_probe.py tests/scripts/test_cst_validate.py \
  2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $POD --out "$RAWD" "$@"; }
run --stage engine --config st || { echo "BAIL: engine st failed; the judges need its rollouts"; exit 1; }
echo ">> engine st done at $(hours) h"
for C in no_mtp no_mtp_seqs128 no_prefix_no_mtp; do
  run --stage engine --config "$C" || echo ">> engine $C failed"
  echo ">> engine $C done at $(hours) h"
done
for J in gpt_oss_120b olmo31_32b_instruct; do
  run --stage judge --judge "$J" || echo ">> judge $J failed"
  echo ">> judge $J done at $(hours) h"
done
run --stage summary || echo ">> summary failed"
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
