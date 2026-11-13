#!/usr/bin/env bash
# Remote runner for the MoReBench pod, stage 1 (CST Amendment 46 item B). One A100-80GB, priced at
# about 9 A100-h against the 10.5 bound (stages stop at 10.25 h, watchdog 11.0 h). Orion launches with
# CST_SESSION=1 (launch_cst_mrb2x2.sh). P-ctx's and P-ctx-n's records and adapters in by name
# (outputs/mrb2x2_in), C_anth for A0s. The smoke pass and both adapters' parity, then A0s (the gate
# at load on its answer length), the judge, P-ctx with the session check, the judge, P-ctx-n unless
# the gate moved it to a later pod, the judge. Writes outputs/mrb2x2_v1.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
SESSION="${CST_SESSION:-}"
PRICE_H=10.5; WATCH_H=11.0
OUT="$C/mrb2x2_v1"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_mrb2x2${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;  # mach: Amendment 43 item 2
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
# the stages stop 0.25 h before the price (MoReBench's judge and MACHIAVELLI read it); the rest is
# the download
export CST_PRICE_H=$PRICE_H
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
POD=papers/cst_constitutional_steering/scripts/cst_mrb2x2_pod.py
export CST_START_EPOCH=$START
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST MoReBench stage 1 session $SESSION start $(date -u +%FT%TZ) out=$OUT (price ${PRICE_H} h, watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_mrb2x2_pod.py' ) &
WATCHDOG=$!

# ---- the vLLM venv (VALIDATE's block, unchanged pins) plus the MACHIAVELLI extra ----
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
  "$VENV/bin/pip" install -q -e ".[machiavelli]" "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
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
$PY - <<'EOF' || { echo "BAIL: vLLM smoke"; exit 1; }
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=512, gpu_memory_utilization=0.3)
out = llm.generate(["Hello"], SamplingParams(temperature=0.2, top_p=1.0, seed=0, max_tokens=8))
print(">> vLLM smoke OK:", len(out[0].outputs[0].token_ids), "tokens")
EOF

# ---- synced inputs: the candidate's record and adapter, C_anth, the smoke pass's fixed adapter
# and its letters (Amendment 31) ----
[ -f "$CST_CONST_DIR/C_anth.txt" ] || { echo "BAIL: C_anth did not sync"; exit 1; }
for t in p_ctx p_ctx_n; do
  cp "$C/mrb2x2_in/train_$t.json" "$OUT/" || { echo "BAIL: $t's record did not sync"; exit 1; }
  d=$(python3 -c "import json; print(list(json.load(open('$OUT/train_$t.json'))['adapters'].values())[-1])")
  [ -f "$d/adapter_model.safetensors" ] || { echo "BAIL: $t's adapter did not sync ($d)"; exit 1; }
done
for f in outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors outputs/pilot_sdf/parity/hf_d_doc_1M.json outputs/pilot_sdf/parity/hf_base.json; do
  [ -f "papers/cst_constitutional_steering/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

PART=bars $PY - <<'EOF' > "$OUT/logs/models_mrb2x2${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
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
echo ">> models: $(cat "$OUT/logs/models_mrb2x2${SESSION}.json")"
$PY -c 'from deepsteer.datasets.morebench import fetch_morebench; from deepsteer.datasets.moralbench import fetch_moralbench; print(">> data:", fetch_morebench(), fetch_moralbench())' \
  || { echo "BAIL: MoReBench or MoralBench download"; exit 1; }
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_mrb2x2.py tests/scripts/test_cst_finalist.py tests/scripts/test_cst_parity.py tests/scripts/test_mrb_p2_pod.py"
# the full pytest output stays in the downloaded logs
$PY -m pytest -q -p no:cacheprovider $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

run() { $PY $POD --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }

# the smoke pass (Amendment 31) and the candidate's parity: a failure stops the pod
run --stage smoke_hf
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
run --stage parity_hf  # HF letters in their own process; the served read follows in a new one
run --stage parity
for t in p_ctx p_ctx_n; do
  [ -f "$OUT/parity/$t.json" ] || { echo "BAIL: the parity stage did not finish for $t (see the log); nothing was measured"; exit 1; }
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/parity/$t.json')).get('read') else 1)" 2>/dev/null \
    || { echo "BAIL: $t fails served parity; it is not read"; exit 1; }
done
echo ">> smoke and parity done at $(hours) h"
# A0s first (the gate at load on its answer length), then P-ctx, then P-ctx-n (Amendment 46 item B1, B3)
run --stage a0s
run --stage judge
run --stage adapter --tag p_ctx  # with the session check (Amendment 42 item 4)
run --stage judge
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/gate.json'))['p_ctx_n'] == 'this pod' else 1)" 2>/dev/null; then
  run --stage adapter --tag p_ctx_n
  run --stage judge
else
  echo ">> the gate at load moved P-ctx-n to a later pod (gate.json)"
fi
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
