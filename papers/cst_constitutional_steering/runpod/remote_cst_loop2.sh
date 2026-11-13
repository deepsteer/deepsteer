#!/usr/bin/env bash
# Remote runner for the loop's second iteration (CST Amendment 44 item 5): config 2's second draw
# on a new seed. One A100-80GB, priced at about 7.0 A100-h (watchdog 7.5 h). Orion launches with
# CST_SESSION=1 (launch_cst_loop2.sh). The cache and the loop pod's rule-right kind in (both checked
# against committed records), the draw trained on the same data, then the frozen read on one session
# base with the content cell and cell A' first, the base's rule-keeping on the rule-right scenarios,
# the rest of the frozen read and the report. Writes outputs/loop2_v1.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
OUT="$C/loop2_v1"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SESSION="${CST_SESSION:-}"
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_loop2${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=7.5;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
LOOP=papers/cst_constitutional_steering/scripts/cst_loop2_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST loop2 session $SESSION start $(date -u +%FT%TZ) out=$OUT (watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_(loop2|loop|pressure_train|sdf_norec|s0a)_pod.py' ) &
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

# ---- synced inputs: the cache (2-roll's scenarios, labels, contrast and A0s rollouts), 2-ctx's
# P-ctx record and adapter (the candidate of record), S0a's cell A' rollouts (the teacher's A'
# shift), C_anth, the pressure scenarios (the guard cell) ----
for f in C_anth.txt eval_8gram_hashes_v1.npy; do  # the eval index: write_kind's 8-gram check
  [ -f "$CST_CONST_DIR/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done
for f in scenarios.json act.jsonl act.json contrast.json train_p_ctx.json; do
  cp "$C/pressure/$f" "$OUT/" || { echo "BAIL: $f did not sync"; exit 1; }
done
mkdir -p "$OUT/roll/a0s"
cp "$C"/pressure/roll/a0s/chunk_*.jsonl "$OUT/roll/a0s/" 2>/dev/null || { echo "BAIL: 2-roll's A0s rollouts did not sync"; exit 1; }
d=$(python3 -c "import json; print(list(json.load(open('$OUT/train_p_ctx.json'))['adapters'].values())[-1])")
[ -f "$d/adapter_model.safetensors" ] || { echo "BAIL: P-ctx's adapter did not sync ($d)"; exit 1; }
ls "$C"/s0a/gen/a0/forced_plain_*.jsonl "$C"/s0a/gen/a0s/forced_plain_*.jsonl >/dev/null 2>&1 || { echo "BAIL: S0a's cell A' rollouts did not sync"; exit 1; }
[ -f "papers/cst_constitutional_steering/data/content_cell_v1.json" ] || { echo "BAIL: the content cell is missing"; exit 1; }
# the loop pod's rule-right kind (scenarios, the teacher's rollouts and labels), its selected content
# cell and A0's rollouts on it (the registered reference), config 2's first-draw cell rows (beside)
K1="$C/loop_v1/kinds/rule_right"
for f in scenarios.json act.jsonl act.json; do
  [ -f "$K1/$f" ] || { echo "BAIL: the rule-right kind's $f did not sync"; exit 1; }
done
mkdir -p "$OUT/kinds/rule_right/roll/a0s"
cp "$K1"/scenarios.json "$K1"/act.jsonl "$K1"/act.json "$OUT/kinds/rule_right/"
cp "$K1"/roll/a0s/chunk_*.jsonl "$OUT/kinds/rule_right/roll/a0s/" 2>/dev/null || { echo "BAIL: the teacher's rule-right rollouts did not sync"; exit 1; }
cp "$C/loop_v1/cell_gate.json" "$OUT/" || { echo "BAIL: the loop pod's cell gate did not sync"; exit 1; }
ls "$C"/loop_v1/gen/a0/cell_plain_*.jsonl "$C"/loop_v1/gen/c2_rule_right/cell_plain_*.jsonl >/dev/null 2>&1 || { echo "BAIL: the loop pod's cell rows did not sync"; exit 1; }

$PY - <<'EOF' > "$OUT/logs/models_loop2${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
print(json.dumps({"qwen38_27b": snapshot_download(
    "Qwen/Qwen3.8-27B", revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ignore_patterns=["original/*", "metal/*"]),
    "gpt_oss_120b": snapshot_download(  # the action reads
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    ignore_patterns=["original/*", "metal/*"])}))
EOF
echo ">> models: $(cat "$OUT/logs/models_loop2${SESSION}.json")"
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_loop2.py tests/scripts/test_cst_loop.py tests/scripts/test_cst_pctx_verify.py tests/scripts/test_cst_pressure_train.py tests/scripts/test_cst_pressure_pod.py tests/scripts/test_cst_parity.py tests/scripts/test_cst_pilot_train.py tests/scripts/test_cst_sdf_norec.py"
# the full pytest output stays in the downloaded logs (the first launch kept only its summary line)
$PY -m pytest -q -p no:cacheprovider $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

run() { $PY $LOOP --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
D=c2_draw2

run --stage cache_check
[ -f "$OUT/cache_check.json" ] || { echo "BAIL: the cache check failed (see the log)"; exit 1; }
run --stage kind_check
[ -f "$OUT/kind_check.json" ] || { echo "BAIL: the kind check failed (see the log)"; exit 1; }
# the smoke pass (Amendments 28 item 3, 31): a failure stops the pod in its first minutes
run --stage smoke_hf
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
echo ">> smoke pass done at $(hours) h"
run --stage train
[ -f "$OUT/train_$D.json" ] || { echo "BAIL: the draw did not train (see the log)"; exit 1; }
run --stage parity_hf --tags $D
# the content cell and cell A' first (Amendment 44 item 5)
run --stage read_a2 --tags $D
run --stage cell_read --tags $D
run --stage cell_base
# the base's rule-keeping on the rule-right scenarios (Amendment 40 item 3)
run --stage kind_base
run --stage kind_base_act
# the rest of the frozen read
run --stage read_rest --tags $D
run --stage heldout --tags base,$D
run --stage heldout_act
run --stage guard --tags base,$D
run --stage guard_act
run --stage h1 --tags base,$D
run --stage capability --tags base,$D
run --stage report
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
