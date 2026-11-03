#!/usr/bin/env bash
# Remote runner for the MCH thinking-on rows (papers/MCH_PLAN.md §5, Amendment 10; not a
# reproduction): ThinkCoTAgent and ThinkGoodCoTAgent on GPT-OSS-20B (effort medium) and
# Qwen3-30B-A3B-Thinking-2507 (native thinking), one A100-80GB. Orion launches; the launcher passes
# HF_TOKEN. ST's and MRB's pinned vLLM stack.
#
#   setup (venv, stack, smoke, models, engine + bundle hash-verified, gate tests) -> stage 0 (render
#   parity and 15 decisions per model x agent; a failure stops the pod) -> the GPT-OSS pilot and its
#   fallback bar -> the Qwen pilot -> decide (the bar, k, the price against 8.0 A100-h) -> both cells
#   at k. Every stage resumes. Every trace goes to the gitignored _raw/ of its stage.
# The game bundle comes from Google Drive; if that fails or its hash mismatches, the pod prints
# NEED_SYNC mch_bundle and the launcher (ON_DEMAND_SYNC) copies the locally verified zip here.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/mch_machiavelli/outputs/thinking"
mkdir -p "$OUT"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session_thinking.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
POD=papers/mch_machiavelli/scripts/mch_thinking_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> MCH thinking-on rows start $(date -u +%FT%TZ) out=$OUT"

# wall cap 8.5 h (Amendment 10: the cell's ceiling is 8.0 A100-h on one GPU)
MAX_H=8.5
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f mch_thinking_pod.py ) &
WATCHDOG=$!

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

# ---- models (GPT-OSS's original/ and metal/ copies skipped), engine and bundle ----
$PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json
from huggingface_hub import snapshot_download
pins = {"gpt_oss_20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
        "qwen3_30b": ("Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d")}
print(json.dumps({k: snapshot_download(r, revision=v, ignore_patterns=["original/*", "metal/*"])
                  for k, (r, v) in pins.items()}))
EOF
echo ">> models: $(cat "$OUT/models.json")"
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

# ---- stage 0 (Amendment 10): render parity and a smoke per model; a failure stops the pod ----
for M in gpt_oss_20b qwen3_30b; do
  $PY $POD --stage stage0 --model $M --out "$OUT" || { echo "BAIL: stage 0 $M"; exit 1; }
done
$PY $POD --stage stage0_check --out "$OUT" || { echo ">> STOP: stage 0 failed (no parsed answer on a model)"; kill $WATCHDOG 2>/dev/null; exit 0; }
echo ">> stage 0 passed: $(tr -d '\n' < "$OUT/stage0_check.json")"

# ---- stage 1: the pilots, the fallback bar after each, then the decision ----
$PY $POD --stage pilot --model gpt_oss_20b --out "$OUT" || { echo "BAIL: pilot gpt_oss_20b"; exit 1; }
$PY $POD --stage pilot_check --model gpt_oss_20b --out "$OUT" || { echo ">> STOP: GPT-OSS pilot fallback above the bar ($(tr -d '\n' < "$OUT/pilot_check_gpt_oss_20b.json"))"; kill $WATCHDOG 2>/dev/null; exit 0; }
$PY $POD --stage pilot --model qwen3_30b --out "$OUT" || { echo "BAIL: pilot qwen3_30b"; exit 1; }
$PY $POD --stage decide --out "$OUT" || { echo "BAIL: decide"; exit 1; }
CONT=$($PY -c "import json; print(int(json.load(open('$OUT/decision.json'))['continue_']))")
K=$($PY -c "import json; print(json.load(open('$OUT/decision.json'))['k'])")
$PY - "$OUT/decision.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for m, v in d["models"].items():
    r = v["rates"]
    print(f">> pilot {m}: " + "; ".join(
        f"{a} fallback {r[a]['fallback_rate']:.3f} cap-hit {r[a]['cap_hit_rate']:.3f} "
        f"capped decisions {r[a]['decisions_with_cap_hit']} cut by cap rule "
        f"{r[a]['decisions_cut_by_cap_rule']} median trace {r[a]['median_trace_chars']:.0f} chars"
        for a in ("ThinkCoTAgent", "ThinkGoodCoTAgent")) + f"; {r['a100h_per_decision']:.2e} A100-h/decision; k {v['k']}")
print(f">> price A100-h at k {d['k']}: {d['price_a100h']:.2f} (at k 2: {d['price_at_k2']:.2f}); envelope {d['envelope_a100h']}")
EOF
echo ">> stage 1 decision: continue=$CONT k=$K reason=$($PY -c "import json; print(json.load(open('$OUT/decision.json'))['reason'])")"
[ "$CONT" = 1 ] || { echo ">> STOP after stage 1 (Amendment 10)"; kill $WATCHDOG 2>/dev/null; exit 0; }

# ---- the cells at k ----
$PY $POD --stage cell --model gpt_oss_20b --k "$K" --out "$OUT" || { echo "BAIL: cell gpt_oss_20b"; exit 1; }
$PY $POD --stage cell --model qwen3_30b --k "$K" --out "$OUT" || { echo "BAIL: cell qwen3_30b"; exit 1; }
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
