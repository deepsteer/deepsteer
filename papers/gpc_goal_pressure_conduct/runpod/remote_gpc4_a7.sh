#!/usr/bin/env bash
# Remote runner for GPC-4's Amendment 7 (GPC_PLAN.md §15, §15.1, §15.2), on Thunder, with no host: the subject's
# vLLM server under 16,384; the fidelity check under the server; the standing control (24 held-out grid slots) and
# its bail; then the cold renders, the named probe at each run's earliest state, the named probe at the 48 states
# and probe B at the 32 act states, all by letter. No tool loop, no judge, no control document, no restricted keys.
# Writes papers/gpc_goal_pressure_conduct/outputs/gpc4_a7 (stages stop at 10.25 h unless moved; watchdog ceiling
# 14.0 h). The states are an input (outputs/gpc4_in/a7, gitignored), never printed.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
G="$REPO_DIR/papers/gpc_goal_pressure_conduct"
C="$REPO_DIR/papers/cst_constitutional_steering/outputs"
SESSION="${CST_SESSION:-}"
PRICE_H=10.5; WATCH_H=11.0
OUT="$G/outputs/gpc4_a7"
mkdir -p "$OUT/logs"
START=$(date +%s)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
source "$REPO_DIR/papers/runpod_common/session_limits.sh"
# the exit trap must reach the sentinel on every path: on GPC-4's first launch it opened with `kill $RW`,
# which an early bail (before RW is set) made an unbound-variable error under set -u, so the trap stopped
# there, no sentinel was written, and the launcher's stream loop waited on an idle pod
trap 'set +u +e; { kill ${RW:-} ${WATCHDOG:-} ${JDL:-}; } 2>/dev/null; limits_save "$OUT/logs" || true; cp "$REPO_DIR/session.log" "$OUT/logs/session_gpc4${SESSION:-}_${STAMP:-}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
case "$SESSION" in 1) MAX_H=$WATCH_H;;
  *) echo "BAIL: CST_SESSION must be 1"; exit 1;; esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$C/_sync"
# the token: RunPod's launcher passes it from /root/.hf_token; elsewhere it is ~/.hf_token (mode 600)
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.hf_token" 2>/dev/null)}"
PYBIN="${CST_PYTHON:-python3}"
export CST_PRICE_H=$PRICE_H CST_START_EPOCH=$START
export CST_DEADLINE_EPOCH=$(python3 -c "print(int($START + ($PRICE_H - 0.25) * 3600))")
# editable limits (papers/runpod_common/session_limits.py): stages stop at 10.25 h, the watchdog at
# 11.0 h, both movable during the run with set_limits.sh (watchdog at most 14.0 h); each change logged
limits_init "$REPO_DIR/.session_limits.json" "$START" "$(python3 -c "print($PRICE_H - 0.25)")" "$WATCH_H" \
  || { echo "BAIL: the limits file"; exit 1; }
GP=papers/gpc_goal_pressure_conduct/scripts/gpc4_pod.py
TPOD=papers/cst_constitutional_steering/scripts/cst_teacher_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> GPC-4 session $SESSION start $(date -u +%FT%TZ) out=$OUT (stages stop at $(python3 -c "print($PRICE_H - 0.25)") h, watchdog ${MAX_H} h)"
[ "$($PYBIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.11 ] || { echo "BAIL: $PYBIN is not Python 3.11"; exit 1; }
$PYBIN -c "import os, sysconfig; raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" \
  || { echo "BAIL: no Python.h for $PYBIN (Triton compiles against it)"; exit 1; }

limits_watchdog 'gpc4_pod.py|gpc4_vm.py|cst_teacher_pod.py|vllm'  # sets WATCHDOG; re-reads the limits file

# ---- the vLLM venv (CST's pins, unchanged) plus the MACHIAVELLI extra ----
VENV=/workspace/venv_vllm
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256=5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f
CU129_INDEX=https://download.pytorch.org/whl/cu129
CONSTRAINTS=papers/st_live_reference/runpod/vllm_constraints.txt
if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating $VENV"
  $PYBIN -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
  "$VENV/bin/pip" install -q "torch==2.13.0+cu129" "torchvision==0.28.0+cu129" \
    "torchaudio==2.11.0+cu129" --index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q "vllm @ ${VLLM_WHEEL_URL}#sha256=${VLLM_WHEEL_SHA256}" \
    "transformers==5.17.0" "peft==0.21.2" pyyaml ninja pytest tokenizers -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
  "$VENV/bin/pip" install -q -e ".[machiavelli]" "transformers==5.17.0" -c "$CONSTRAINTS" \
    --extra-index-url "$CU129_INDEX" 2>&1 | tail -2
fi
# G5's judge runs upstream's classifiers (BeautifulSoup with lxml)
"$VENV/bin/pip" install -q beautifulsoup4 lxml -c "$CONSTRAINTS" 2>&1 | tail -1
# the gates read each ImpossibleBench sample's tool calls from its zstd .eval log (GPC_PLAN.md §11.15),
# with Inspect's own zstandard version
"$VENV/bin/pip" install -q "zstandard==0.25.0" 2>&1 | tail -1
"$VENV/bin/python" -c "import zstandard" || { echo "BAIL: zstandard did not install"; exit 1; }
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
$PY - <<'EOF' || { echo "BAIL: vLLM venv not usable"; exit 1; }
import importlib.metadata as md, shutil, torch, vllm, peft  # noqa: E401,F401
import gym, asteval  # noqa: F401,E401  (the MACHIAVELLI extra)
import bs4, lxml  # noqa: F401,E401  (G5's classifiers)
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

# ---- CST's inputs, synced by name under GPC's outputs, put back at their paths; GPC-1's and GPC-4's ----
I="$G/outputs/gpc4_in"
[ -d "$I/cst/outputs" ] || { echo "BAIL: the CST inputs did not sync ($I)"; exit 1; }
cp -r "$I/cst/outputs/." "$C/"
for f in adapters/c1/612732/adapter_model.safetensors conduct/slots.json conduct/holdout_base.jsonl; do
  [ -f "$G/outputs/gpc1_v1/$f" ] || { echo "BAIL: GPC-1's $f did not sync"; exit 1; }
done
for f in outputs/_sync/eval_8gram_hashes_v2.npy data/c_bounds_v1.json data/bounds_grid_v1.json \
         data/c_conduct_v1.json data/a6_renders_v1.json data/odcv_pressure_v1.json \
         outputs/gpc4_in/a7/states.jsonl; do
  [ -f "$G/$f" ] || { echo "BAIL: $f did not sync"; exit 1; }
done
mkdir -p "$OUT/a7"
# P-ctx, put back above by the copy into $C (which already ends in .../outputs): GPC-4's launch 3 stopped
# at prep on a doubled outputs/ in this path, so its two files are checked here, minutes into the session
export GPC4_PCTX="$C/pressure/adapters/p_ctx/1.94053e+06"
for f in adapter_config.json adapter_model.safetensors; do
  [ -f "$GPC4_PCTX/$f" ] || { echo "BAIL: P-ctx's $f is not at $GPC4_PCTX"; exit 1; }
done
# upstream code and data at their pins (fetched here, never committed)
git clone -q https://github.com/McGill-DMaS/ODCV-Bench /workspace/ODCV-Bench && \
  git -C /workspace/ODCV-Bench checkout -q 6924dafb3a3f84fc320fbe9f0a08203d67d1d488 || { echo "BAIL: ODCV-Bench"; exit 1; }
git clone -q https://github.com/anthropic-experimental/agentic-misalignment /workspace/agentic-misalignment && \
  git -C /workspace/agentic-misalignment checkout -q ea0630e1a3eaae7f9f9740fd2703229d3854ccda || { echo "BAIL: agentic-misalignment"; exit 1; }
export GPC4_ODCV_CLONE=/workspace/ODCV-Bench GPC_AM_DIR=/workspace/agentic-misalignment GPC4_IB_DATA=/workspace/ib_data/data
PART=mach $PY - <<'EOF' > "$OUT/logs/models_gpc4${SESSION}.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
got = {"qwen38_27b": snapshot_download("Qwen/Qwen3.8-27B",
                                       revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                                       ignore_patterns=["original/*", "metal/*"])}
got["gpt_oss_120b_tokenizer"] = snapshot_download(  # the CST gate tests' dry judge
    "openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    allow_patterns=["tokenizer*", "special_tokens_map.json", "config.json",
                    "generation_config.json", "chat_template*"])
print(json.dumps(got))
EOF
echo ">> models: $(cat "$OUT/logs/models_gpc4${SESSION}.json")"
# §15: no judge in this session, so its weights are not downloaded
# the game bundle: Google Drive first; on a failure or a hash mismatch, the locally verified zip
# copied here by the launcher (ON_DEMAND_SYNC; MCH Amendment 8). Never re-hosted.
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

TESTS="tests/scripts/test_gpc4.py tests/scripts/test_gpc3.py tests/scripts/test_gpc2.py tests/scripts/test_gpc1.py tests/scripts/test_session_limits.py tests/scripts/test_cst_teacher.py tests/scripts/test_cst_parity.py tests/scripts/test_mch_thinking_pod.py"
# the slow dry chains are Z9's local check (run in the sync emulation). The pod runs only tests of code
# it executes: the host-side tests (the launcher, the IAM policy, cloud-init, the rehearsal plan, verify
# and the host's parts of gpc4_vm.py) are deselected here and run in CI, since both of GPC-4's first two
# launches bailed on a host-side test failing only in the pod's environment (GPC_PLAN.md §11.12)
GATE_MARK="not slow and not host_side"
GATE_FLOOR=239   # the gate's collected count when set (Amendment 7, §15.2); fewer means a file failed to collect
NCOL=$($PY -m pytest --collect-only -q -p no:cacheprovider -m "$GATE_MARK" $TESTS 2>/dev/null | grep -c '::' || true)
echo ">> gate tests collected: $NCOL (floor $GATE_FLOOR)"
[ "$NCOL" -ge "$GATE_FLOOR" ] || { echo "BAIL: the gate collected $NCOL tests, under its floor of $GATE_FLOOR"; exit 1; }
$PY -m pytest -q -p no:cacheprovider -m "$GATE_MARK" $TESTS > "$OUT/logs/gate_tests${SESSION}.log" 2>&1; GT=$?
grep -E '[0-9]+ (passed|failed)' "$OUT/logs/gate_tests${SESSION}.log" | tail -1
grep -E '^FAILED' "$OUT/logs/gate_tests${SESSION}.log" | head -5
[ "$GT" = 0 ] || { echo "BAIL: gate tests (see logs/gate_tests${SESSION}.log)"; exit 1; }

echo setup > "$OUT/.stage"
python3 papers/cst_constitutional_steering/scripts/rate_watch.py --log "$REPO_DIR/session.log" \
  --stage-file "$OUT/.stage" --out "$OUT/logs/rate_windows.jsonl" \
  --forecast '{"bounds": 1800, "stage4": 1800}' &
RW=$!

# GPC4_DRY=1 (the standing pre-launch step's local run of this block) adds --dry-run to every stage; unset
# on the pod, it adds nothing
DRY=${GPC4_DRY:+--dry-run}
run() { echo "$2" > "$OUT/.stage"; $PY "$1" --out "$OUT" "${@:3}" $DRY || echo ">> stage failed: ${*:3}"; echo ">> stage done at $(hours) h: ${*:3}"; }
fits() { $PY $GP --out "$OUT" --stage fits --next "$1" $DRY; }
ok() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/$1')).get('$2') else 1)" 2>/dev/null; }
stop() { echo ">> $1"; echo ">> done at $(hours) h"; kill ${WATCHDOG:-} ${JDL:-} 2>/dev/null; exit 0; }
A7=papers/gpc_goal_pressure_conduct/scripts/gpc4_a7_pod.py
export GPC4_VLLM="$VENV/bin/vllm"
end_session() { echo ">> $1"; run $GP serve --stage serve --part down; stop "$1"; }

# a. C1 registered, the summed adapter built (exact); each adapter's letters on HF, then its served parity
run $GP prep --stage prep
run $GP parity_hf --stage parity_hf
run $GP parity --stage parity
# b. GPC_PLAN.md §15 (Amendment 7): the server under 16,384 (the depth cell's settings), its check
run $GP serve --stage serve --part up
run $GP serve_check --stage serve_check
$PY $GP --out "$OUT" --stage server_ok $DRY || end_session "the subject's server is dead: the session ends (GPC_PLAN.md §11.17 item 1)"
# c. the fidelity check under the server's own rendering (§15.2), then the standing control and its bail (§15):
#    with the control failed the instrument failed this session, nothing is read, and the session ends
run $A7 fidelity --stage fidelity
run $A7 control --stage control
run $A7 decide --stage decide
ok a7/decision.json control_ok || end_session "the control failed (see a7/decision.json): nothing is read (GPC_PLAN.md §15)"
# d. the reference, the probe-artifact check, the named probe at the brink, probe B (§15, §15.1)
run $A7 cold --stage cold
run $A7 early --stage early
run $A7 named --stage named
run $A7 own --stage own
run $GP serve --stage serve --part down
stop "all stages run"
