#!/usr/bin/env bash
# Remote runner for the MCH thinking-on cell under Amendment 11 (papers/MCH_PLAN.md §5; not a
# reproduction): ThinkCoTAgent and ThinkGoodCoTAgent on GPT-OSS-20B (effort medium) only, k = 2, the
# 80 episodes the pilot did not run. Orion launches; the launcher passes HF_TOKEN. ST's and MRB's
# pinned vLLM stack. An A40 first, an A100 behind it; one A40 hour counts 0.32 A100-h.
#
#   setup (venv, stack, smoke, model, engine + bundle hash-verified, gate tests) -> stage 0 twice
#   (render parity and 15 decisions per agent, without and with n-gram speculative decoding) ->
#   a11_forecast (the faster mode; the longest episode at its measured rate, against the 4.8 A100-h
#   left; a forecast over it stops the pod) -> the cell. Every stage resumes. Every trace goes to the
#   gitignored _raw/ of its stage. The pilot's episodes.jsonl comes with SYNC_OUTPUTS so the cell
#   skips the pilot's episodes.
# The game bundle comes from Google Drive; if that fails or its hash mismatches, the pod prints
# NEED_SYNC mch_bundle and the launcher (ON_DEMAND_SYNC) copies the locally verified zip here.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/mch_machiavelli/outputs/thinking"
mkdir -p "$OUT"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session_thinking_a11.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
POD=papers/mch_machiavelli/scripts/mch_thinking_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> MCH thinking-on cell (Amendment 11) start $(date -u +%FT%TZ) out=$OUT"
[ -f "$OUT/pilot_gpt_oss_20b/episodes.jsonl" ] \
  || { echo "BAIL: the pilot's episodes.jsonl is not on the pod (SYNC_OUTPUTS)"; exit 1; }

# the GPU sets the wall cap: 4.7 A100-h (4.8 left, 0.1 for start-up) in that GPU's hours
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in
  *A40*) A100H_PER_H=0.32 ;;
  *A100*) A100H_PER_H=1.0 ;;
  *) echo "BAIL: no A100-h conversion registered for '$GPU_NAME'"; exit 1 ;;
esac
MAX_H=$(python3 -c "print(round(4.7 / $A100H_PER_H, 2))")
echo ">> GPU $GPU_NAME: $A100H_PER_H A100-h per hour, wall cap $MAX_H h"
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f mch_thinking_pod.py ) &
WATCHDOG=$!
# 80 engine workers at about 0.3 GB each (measured); 20 GB kept for vLLM's own process
CONC=$(python3 - <<'EOF'
from pathlib import Path
lim = Path("/sys/fs/cgroup/memory.max")
v = lim.read_text().strip() if lim.exists() else "max"
if not v.isdigit():
    old = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    v = old.read_text().strip() if old.exists() else "max"
mem = int(v) if v.isdigit() and int(v) < 1 << 50 else None
if mem is None:
    mem = next(int(x.split()[1]) * 1024 for x in open("/proc/meminfo") if x.startswith("MemTotal"))
print(max(8, min(96, int((mem / 1e9 - 20) / 0.35))))
EOF
)
echo ">> host memory allows $CONC engine workers"

# ---- the vLLM venv (MRB-G1's block, unchanged pins) plus the MACHIAVELLI extra ----
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
  "$VENV/bin/pip" install -q -e ".[machiavelli]" "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, shutil, torch, vllm  # noqa: E401
from vllm.model_executor.models import gpt_oss, qwen3_moe  # noqa: F401
import gym, asteval  # noqa: F401,E401  (the MACHIAVELLI extra)
v = {p: md.version(p) for p in ("vllm", "torch", "transformers", "gym", "asteval")}
ok = (v["vllm"].startswith("0.31.0") and v["transformers"] == "5.17.0" and v["gym"] == "0.25.1"
      and torch.cuda.is_available() and (torch.version.cuda or "").startswith("12.9")
      and shutil.which("ninja") is not None)
print(">> stack:", v, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
EOF
$PY - <<'EOF' || { echo "BAIL: vLLM smoke (engine start-up or sampler)"; exit 1; }
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, gpu_memory_utilization=0.3)
out = llm.generate(["Hello"], SamplingParams(temperature=0.2, top_p=1.0, seed=0, max_tokens=8))
print(">> vLLM smoke OK:", len(out[0].outputs[0].token_ids), "tokens at T 0.2")
EOF

# ---- the model (GPT-OSS's original/ and metal/ copies skipped), engine and bundle ----
$PY - <<'EOF' > "$OUT/models_a11.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"gpt_oss_20b": snapshot_download(
    "openai/gpt-oss-20b", revision="6cee5e81ee83917806bbde320786a8fb61efebee",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/models_a11.json")"
# the bundle: Google Drive first; on a failure or a hash mismatch, the locally verified zip copied
# here by the launcher (ON_DEMAND_SYNC; Amendment 8). Never re-hosted.
BUNDLE_PUSH=/workspace/mch_bundle/game_data.zip
if ! $PY -c 'from deepsteer.datasets.machiavelli import fetch_machiavelli; print(">> engine:", fetch_machiavelli())'; then
  echo ">> the Drive download failed or its hash mismatched; asking for the locally verified bundle"
  echo "NEED_SYNC mch_bundle"
  for _ in $(seq 1 120); do [ -f "$BUNDLE_PUSH" ] && break; sleep 10; done
  [ -f "$BUNDLE_PUSH" ] || { echo "BAIL: no bundle after 20 min (launch with ON_DEMAND_SYNC)"; exit 1; }
  $PY -c "from pathlib import Path; from deepsteer.datasets.machiavelli import fetch_machiavelli; print('>> engine:', fetch_machiavelli(zip_path=Path('$BUNDLE_PUSH')))" \
    || { echo "BAIL: the synced bundle failed its hash check"; exit 1; }
fi
echo ">> setup done at $(hours) h"
# the gate tests; keep pytest's own summary line (worker processes print gym's warning after it)
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_mch_thinking_pod.py \
  tests/datasets/test_machiavelli.py 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }


# ---- stage 0 twice (Amendment 11): without, then with n-gram speculative decoding ----
$PY $POD --stage stage0 --model gpt_oss_20b --spec none --tag _a11_none --out "$OUT" \
  || { echo "BAIL: stage 0 (no speculative decoding)"; exit 1; }
$PY $POD --stage stage0 --model gpt_oss_20b --spec ngram --tag _a11_ngram --out "$OUT" \
  || echo ">> stage 0 with n-gram decoding failed; the cell runs without it"

# ---- the forecast: the faster mode, the longest episode at its rate, against 4.8 A100-h ----
$PY $POD --stage a11_forecast --model gpt_oss_20b --gpu "$GPU_NAME" --elapsed-h "$(hours)" \
  --out "$OUT" > /dev/null
RC=$?
[ -f "$OUT/a11_forecast.json" ] || { echo "BAIL: a11_forecast wrote nothing"; exit 1; }
echo ">> forecast: $(tr -d '\n' < "$OUT/a11_forecast.json")"
[ "$RC" = 0 ] || { echo ">> STOP: the forecast is over the 4.8 A100-h left (Amendment 11)"; kill $WATCHDOG 2>/dev/null; exit 0; }
SPEC=$($PY -c "import json; print(json.load(open('$OUT/a11_forecast.json'))['spec'])")

# ---- the cell: the 80 non-pilot episodes, longest games first ----
echo ">> cell: spec=$SPEC concurrency=$CONC at $(hours) h"
$PY $POD --stage cell --model gpt_oss_20b --k 2 --spec "$SPEC" --concurrency "$CONC" --out "$OUT" \
  || { echo "BAIL: cell gpt_oss_20b"; exit 1; }
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
