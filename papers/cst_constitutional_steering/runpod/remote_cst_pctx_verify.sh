#!/usr/bin/env bash
# Remote runner for the P-ctx verification pod (CST Amendment 35). One A100-80GB, priced at or under
# 10.5 A100-h (watchdog 11.0 h). Orion launches with CST_SESSION=1 (launch_cst_pctx_verify.sh) once
# 2-ctx's download has landed. On one session base: the fixed smoke check; the base's screen read
# and A0s's operator cell on one engine, P-ctx's operator cell; capability at n 800, the
# injected-order cell and H1_held on the saved P-ctx; a second P-ctx draw and P-ctx-f (Amendment
# 36: the teacher's contradicting rollouts filtered out) with their screen reads; the held-out cell
# at k 4 on the base, the draw and P-ctx-f (about 9.6 h). Writes outputs/pressure_verify.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
OUT="$C/pressure_verify"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_pctx_verify${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=11.0;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
VERIFY=papers/cst_constitutional_steering/scripts/cst_pctx_verify_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST P-ctx verification pod session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(pctx_verify|pressure_train|sdf_norec|s0a)_pod.py' ) &
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

# ---- synced inputs: 2-ctx's records and adapters (P-ctx, P-ctx-n), 2-roll's scenarios, labels,
# contrast and rollouts (the second draw's training set), C_anth (A0s), the pressure scenarios ----
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: C_anth did not sync"; exit 1; }
for f in scenarios.json act.jsonl act.json contrast.json; do
  cp "$C/pressure/$f" "$OUT/" || { echo "BAIL: 2-roll's $f did not sync"; exit 1; }
done
for a in a0 a0s a0n; do
  mkdir -p "$OUT/roll/$a"
  cp "$C"/pressure/roll/$a/chunk_*.jsonl "$OUT/roll/$a/" 2>/dev/null || { echo "BAIL: 2-roll's $a rollouts did not sync"; exit 1; }
done
for r in p_ctx p_ctx_n; do
  cp "$C/pressure/train_$r.json" "$OUT/" || { echo "BAIL: 2-ctx's train_$r.json did not sync"; exit 1; }
  d=$(python3 -c "import json; print(list(json.load(open('$OUT/train_$r.json'))['adapters'].values())[-1])")
  [ -f "$d/adapter_model.safetensors" ] || { echo "BAIL: 2-ctx's $r adapter did not sync ($d)"; exit 1; }
done

$PY - <<'EOF' > "$OUT/logs/models_pctx_verify${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"]),
    "gpt_oss_120b": snapshot_download(  # the guard and held-out cells' action reads
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_pctx_verify${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_pctx_verify.py tests/scripts/test_cst_pressure_train.py tests/scripts/test_cst_pressure_pod.py tests/scripts/test_cst_parity.py tests/scripts/test_cst_pilot_train.py"
$PY -m pytest -q -p no:cacheprovider $TESTS 2>&1 | grep -E '[0-9]+ (passed|failed)' | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

run() { $PY $VERIFY --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }

# the smoke pass (Amendments 28 item 3, 31): a failure stops the pod in its first minutes
run --stage smoke_hf
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
echo ">> smoke pass done at $(hours) h"

# item 1: the session base's screen read and A0s on one engine; P-ctx's operator cell
run --stage parity_hf --tags p_ctx,p_ctx_n
run --stage session
run --stage read_op --tags p_ctx
run --stage parity_only --tags p_ctx_n
# item 2: the saved P-ctx
run --stage capability --tags base,p_ctx
run --stage guard --tags base,p_ctx,p_ctx_n
run --stage guard_act
run --stage h1 --tags base,p_ctx
# item 3: the second draw and its screen read; Amendment 36 item 3a: P-ctx-f and its screen read
run --stage train_draw2
run --stage parity_hf --tags p_ctx_draw2
run --stage read --tags p_ctx_draw2
run --stage train_filtered
run --stage parity_hf --tags p_ctx_f
run --stage read --tags p_ctx_f
# Amendment 36 item 3b: the held-out free-form cell at k 4 on the base, the second draw and P-ctx-f
run --stage heldout --tags base,p_ctx_draw2,p_ctx_f
run --stage heldout_act
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
