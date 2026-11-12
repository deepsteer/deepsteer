#!/usr/bin/env bash
# Remote runner for the finalist-bars pod (CST Amendments 41, 42), split in two pods that run in
# parallel (Amendment 42 item 3); PART comes from the wrapper (remote_cst_finalist_bars.sh or
# remote_cst_finalist_mach.sh). One A100-80GB each. Orion launches with CST_SESSION=1
# (launch_cst_finalist.sh). The candidate's training record and adapter come in by name
# (outputs/finalist_in_<part>/train_<tag>.json, exactly one); the base and the candidate on one vLLM
# engine with LoRA; the smoke pass and the candidate's parity open both pods.
#   bars (10.5 A100-h): refusal retention, MoralBench, MoReBench generation and the session check,
#     capability at n 800 unless the candidate is P-ctx (read at n 800 on the verification pod),
#     then MoReBench's judge on one GPU to the deadline (resumable).
#   mach (13.0 A100-h, Amendment 43's exception; watchdog 14.0 h): MACHIAVELLI on both arms at the
#     4,096-token budget, the gate at full load and the deadline. Every decision is saved as it is
#     taken; a stage that dies is started again (up to 3 times) and resumes from the saved
#     decisions, and a relaunch resumes from the files the card syncs back (Amendment 43 item 1).
# Writes outputs/finalist_<tag>_<part>.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
SESSION="${CST_SESSION:-}"
case "${PART:-}" in bars) PRICE_H=10.5; WATCH_H=11.0;; mach) PRICE_H=13.0; WATCH_H=14.0;;
  *) echo "BAIL: PART must be bars or mach (run a wrapper)"; touch "$REPO_DIR/.session_done"; exit 1;; esac
REC=$(ls "$C"/finalist_in_"$PART"/train_*.json 2>/dev/null)
[ "$(printf '%s\n' "$REC" | grep -c .)" = 1 ] || { echo "BAIL: exactly one candidate record in finalist_in_$PART (got: $REC)"; touch "$REPO_DIR/.session_done"; exit 1; }
TAG=$(basename "$REC" .json); TAG=${TAG#train_}
OUT="$C/finalist_${TAG}_$PART"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_finalist${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;  # mach: Amendment 43 item 2
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
# the stages stop 0.25 h before the price (MoReBench's judge and MACHIAVELLI read it); the rest is
# the download
export CST_PRICE_H=$PRICE_H
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
POD=papers/cst_constitutional_steering/scripts/cst_finalist_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> CST finalist ($PART) session $SESSION start $(date -u +%FT%TZ) candidate=$TAG out=$OUT (price ${PRICE_H} h, watchdog ${MAX_H} h)"

( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f 'cst_finalist_pod.py' ) &
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
cp "$REC" "$OUT/" || { echo "BAIL: the candidate's record did not copy"; exit 1; }
[ -f "$OUT/mach/decisions.jsonl" ] && echo ">> mach: resuming from $(wc -l < "$OUT/mach/decisions.jsonl") saved decisions"
d=$(python3 -c "import json; print(list(json.load(open('$REC'))['adapters'].values())[-1])")
[ -f "$d/adapter_model.safetensors" ] || { echo "BAIL: the candidate's adapter did not sync ($d)"; exit 1; }
for f in outputs/pilot_sdf/adapters/d_doc/1e+06/adapter_model.safetensors outputs/pilot_sdf/parity/hf_d_doc_1M.json outputs/pilot_sdf/parity/hf_base.json; do
  [ -f "papers/cst_constitutional_steering/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done

PART=$PART $PY - <<'EOF' > "$OUT/logs/models_finalist${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json, os
from huggingface_hub import snapshot_download
pins = {"qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")}
if os.environ["PART"] == "bars":  # MoReBench's judge
    pins["gpt_oss_120b"] = ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/logs/models_finalist${SESSION}.json")"
$PY -c 'from deepsteer.datasets.morebench import fetch_morebench; from deepsteer.datasets.moralbench import fetch_moralbench; print(">> data:", fetch_morebench(), fetch_moralbench())' \
  || { echo "BAIL: MoReBench or MoralBench download"; exit 1; }
# the game bundle: Google Drive first; on a failure or a hash mismatch, the locally verified zip
# copied here by the launcher (ON_DEMAND_SYNC; MCH Amendment 8). Never re-hosted.
BUNDLE_PUSH=/workspace/mch_bundle/game_data.zip
if [ "$PART" = mach ] && ! $PY -c 'from deepsteer.datasets.machiavelli import fetch_machiavelli; print(">> engine:", fetch_machiavelli())'; then
  echo ">> the Drive download failed or its hash mismatched; asking for the locally verified bundle"
  echo "NEED_SYNC mch_bundle"
  for _ in $(seq 1 120); do [ -f "$BUNDLE_PUSH" ] && break; sleep 10; done
  [ -f "$BUNDLE_PUSH" ] || { echo "BAIL: no bundle after 20 min (launch with ON_DEMAND_SYNC)"; exit 1; }
  $PY -c "from pathlib import Path; from deepsteer.datasets.machiavelli import fetch_machiavelli; print('>> engine:', fetch_machiavelli(zip_path=Path('$BUNDLE_PUSH')))" \
    || { echo "BAIL: the synced bundle failed its hash check"; exit 1; }
fi
echo ">> setup done at $(hours) h"

TESTS="tests/scripts/test_cst_finalist.py tests/scripts/test_cst_parity.py"
if [ "$PART" = mach ]; then TESTS="$TESTS tests/scripts/test_mch_thinking_pod.py"; else TESTS="$TESTS tests/scripts/test_mrb_p2_pod.py"; fi
# the full pytest output stays in the downloaded logs
$PY -m pytest -q -p no:cacheprovider $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

run() { $PY $POD --candidate "$TAG" --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }

# the smoke pass (Amendment 31) and the candidate's parity: a failure stops the pod
run --stage smoke_hf
[ -f "$OUT/smoke_hf.json" ] || { echo "BAIL: smoke_hf failed (see the log)"; exit 1; }
run --stage smoke_serve
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/smoke_serve.json'))['parity_passes'] else 1)" 2>/dev/null \
  || { echo "BAIL: smoke_serve failed (see smoke_serve.json and the log)"; exit 1; }
run --stage parity
python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/parity/$TAG.json')).get('read') else 1)" 2>/dev/null \
  || { echo "BAIL: the candidate fails served parity; it is not read"; exit 1; }
echo ">> smoke and parity done at $(hours) h"
if [ "$PART" = bars ]; then
  run --stage rr
  run --stage mb
  run --stage mrb_gen  # the candidate's 1,000 rollouts, then the session check (Amendment 42 item 4)
  [ "$TAG" = p_ctx ] || run --stage capability  # Amendment 42 item 5
  run --stage mrb_judge  # to the deadline (resumable)
else
  # the gate at load, then the deadline; a run that ends writes a run_meta line, a run that dies
  # does not, and is started again from the saved decisions (Amendment 43 item 1)
  for try in 1 2 3; do
    before=$(cat "$OUT/mach/run_meta.jsonl" 2>/dev/null | wc -l)
    run --stage mach --elapsed-h "$(hours)" --reserve-h 0.3
    after=$(cat "$OUT/mach/run_meta.jsonl" 2>/dev/null | wc -l)
    [ "$after" -gt "$before" ] && break
    [ "$(date +%s)" -lt "$CST_DEADLINE_EPOCH" ] || { echo ">> mach: the run died after the deadline; not resumed"; break; }
    echo ">> mach: the run died (try $try); resuming from the saved decisions"
  done
fi
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
