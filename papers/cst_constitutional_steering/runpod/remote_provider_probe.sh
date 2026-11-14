#!/usr/bin/env bash
# Remote runner for a second provider's compatibility probe (CST Amendment 51 item A), started by
# probe_provider.sh. Provider-neutral: a Linux host with one A100 80GB, /workspace, Python 3.11.
# Stops at the first failed go/no-go item. Writes outputs/provider_probe_v1.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
OUT="$C/provider_probe_v1"
mkdir -p "$OUT/logs"
START=$(date +%s)
trap 'cp "$REPO_DIR/probe.log" "$OUT/logs/" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
export HF_TOKEN="$(cat "$HOME/.hf_token" 2>/dev/null)"
PROBE=papers/cst_constitutional_steering/scripts/provider_probe.py
TPOD=papers/cst_constitutional_steering/scripts/cst_teacher_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> provider probe start $(date -u +%FT%TZ) out=$OUT"

# ---- A1, before anything is installed: the host's facts ----
{ echo "python: $(python3 --version 2>&1)"; echo "user: $(id)"; \
  echo "sudo: $(sudo -n true 2>/dev/null && echo yes || echo no)"; echo "kernel: $(uname -a)"; \
  echo "cpus: $(nproc)"; free -g | head -2; df -h "$REPO_DIR" | tail -1; \
  echo "workspace: $(readlink -f /workspace)"; nvidia-smi; } > "$OUT/logs/facts.log" 2>&1
cat "$OUT/logs/facts.log"
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$PYV" = 3.11 ] || [ "${PROBE_ALLOW_PY:-0}" = 1 ] || { echo "STOP (A1): Python $PYV, not 3.11; the author decides (PROBE_ALLOW_PY=1 to go on)"; exit 1; }

# ---- A1-A2: the runner's venv exactly as pinned, touching nothing in the system CUDA (python3-venv
# from the distribution if the image lacks it; no CUDA package is touched) ----
python3 -m venv --help >/dev/null 2>&1 && python3 -c 'import ensurepip' 2>/dev/null \
  || { echo ">> installing python3-venv"; sudo apt-get install -y -q "python3.11-venv" >> "$OUT/logs/pip.log" 2>&1 || sudo apt-get install -y -q python3-venv >> "$OUT/logs/pip.log" 2>&1; }
VENV=/workspace/venv_vllm
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f
CU129_INDEX=https://download.pytorch.org/whl/cu129
CONSTRAINTS=papers/st_live_reference/runpod/vllm_constraints.txt
if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip >> "$OUT/logs/pip.log" 2>&1
  "$VENV/bin/pip" install -q "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
    "torchaudio==2.11.0+cu129" --index-url "$CU129_INDEX" >> "$OUT/logs/pip.log" 2>&1
  "$VENV/bin/pip" install -q "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
    "transformers==5.17.0" "peft==0.21.2" pyyaml ninja pytest tokenizers -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" >> "$OUT/logs/pip.log" 2>&1
  "$VENV/bin/pip" install -q -e ".[machiavelli]" "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" >> "$OUT/logs/pip.log" 2>&1
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' > "$OUT/logs/stack.log" 2>&1; ST=$?

import importlib.metadata as md, shutil, torch, vllm, peft  # noqa: E401,F401
import gym, asteval  # noqa: F401,E401  (the MACHIAVELLI extra)
v = {p: md.version(p) for p in ("vllm", "torch", "transformers", "peft", "gym")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and v["peft"] == "0.21.2"
      and v["gym"] == "0.25.1" and torch.cuda.is_available()
      and (torch.version.cuda or "").startswith("12.9") and shutil.which("ninja") is not None)
print(">> stack:", v, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
EOF
cat "$OUT/logs/stack.log"
python3 -c "import json; json.dump({'ok': $ST == 0}, open('$OUT/stack.json', 'w'))"
[ "$ST" = 0 ] || { echo "STOP (A2): the pinned stack check failed; delete the instance and report $OUT/logs/stack.log and pip.log in full"; exit 1; }
$PY - <<'EOF' || { echo "BAIL: vLLM smoke"; exit 1; }
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, gpu_memory_utilization=0.3)
out = llm.generate(["Hello"], SamplingParams(temperature=0.2, top_p=1.0, seed=0, max_tokens=8))
print(">> vLLM smoke OK:", len(out[0].outputs[0].token_ids), "tokens")
EOF


# ---- synced inputs: P-ctx's record and adapter, C_anth, the finalist run's decisions, the smoke
# pass's fixed adapter and its letters, P-ctx's training rollouts, the game bundle ----
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: C_anth did not sync"; exit 1; }
cp "$C/teacher_in/train_p_ctx.json" "$OUT/" || { echo "BAIL: P-ctx's record did not sync"; exit 1; }
for f in decisions.jsonl episodes.jsonl; do
  [ -f "$C/finalist_c2_rule_right_mach/mach/$f" ] || { echo "BAIL: the finalist run's $f did not sync"; exit 1; }
done
[ -f "$C/pressure/scenarios.json" ] || { echo "BAIL: P-ctx's training scenarios did not sync"; exit 1; }
PART=mach $PY - <<'EOF' > "$OUT/logs/models_probe${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json, os
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")}
if os.environ["PART"] == "bars":  # MoReBench's judge
    pins["gpt_oss_120b"] = ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
got = {k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
       for k, (r, v) in pins.items()}
if os.environ["PART"] == "mach":  # the gate tests' dry judge reads gpt-oss's tokenizer only
    got["gpt_oss_120b_tokenizer"] = snapshot_download(
        "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json",
                        "generation_config.json", "chat_template*"])
print(json.dumps(got))
EOF
echo ">> models: $(cat "$OUT/logs/models_probe${SESSION}.json")"
echo ">> models: $(cat "$OUT/logs/models_probe.json")"
$PY -c "from pathlib import Path; from deepsteer.datasets.machiavelli import fetch_machiavelli; print('>> engine:', fetch_machiavelli(zip_path=Path('/workspace/mch_bundle/game_data.zip')))" \
  || { echo "BAIL: the synced bundle failed its hash check"; exit 1; }
$PY -c 'from deepsteer.datasets.morebench import fetch_morebench; print(">> data:", fetch_morebench())' || { echo "BAIL: MoReBench download"; exit 1; }
echo ">> setup done at $(hours) h"

# ---- A3: the gate tests ----
TESTS="tests/scripts/test_cst_teacher.py tests/scripts/test_cst_finalist.py tests/scripts/test_cst_parity.py tests/scripts/test_provider_probe.py"
$PY -m pytest -q -p no:cacheprovider $TESTS > "$OUT/logs/gate_tests.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests.log" | tail -1
python3 -c "import json; json.dump({'ok': $GT == 0}, open('$OUT/gate_tests.json', 'w'))"
[ "$GT" = 0 ] || { echo "STOP (A3): gate tests (see logs/gate_tests.log)"; exit 1; }

run() { "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
# ---- A4: the engine of record at 128 sequences, the fixed smoke check (Amendment 31) ----
run $PY $TPOD --out "$OUT" --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "STOP (A4): the fixed smoke check did not pass"; exit 1; }
# ---- A5: the HF path ----
run $PY $PROBE --out "$OUT" --stage hf_fixed
# ---- A6: throughput at saturation; A7: training ----
run $PY $PROBE --out "$OUT" --stage tput --mode plain --minutes 30
run $PY $PROBE --out "$OUT" --stage tput --mode doc --minutes 15
run $PY $PROBE --out "$OUT" --stage train50
echo ">> done at $(hours) h"
