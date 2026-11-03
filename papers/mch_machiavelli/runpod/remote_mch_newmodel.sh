#!/usr/bin/env bash
# Remote runner for the MCH new-model rows (papers/MCH_PLAN.md §5, Amendments 4, 6, 7): CoTAgent
# and GoodCoTAgent on GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507, one A100-80GB. Orion launches;
# the launcher passes HF_TOKEN and VALIDATE. ST's and MRB's pinned vLLM stack.
#
#   VALIDATE=1  stack, models, engine + bundle (hash-verified), gate tests, then both models on
#               2 games x episode 0 x both agents, 15 decisions each, behind the render-parity check
#   VALIDATE=0  the same setup, then stage 1 (the GPT-OSS pilot) -> decide (stop or continue, k)
#               -> stage 2 (GPT-OSS cell at k) -> stage 3 (Qwen cell at k). Every stage resumes.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/mch_machiavelli/outputs/newmodel"
mkdir -p "$OUT"
START=$(date +%s)
trap 'cp "$REPO_DIR/session.log" "$OUT/session_$([ "$VALIDATE" = 1 ] && echo validate || echo run).log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
POD=papers/mch_machiavelli/scripts/mch_newmodel_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> MCH new-model rows start $(date -u +%FT%TZ) VALIDATE=$VALIDATE out=$OUT"

# wall caps: VALIDATE 1 h; the run 7 h (the cell's envelope is about 6 A100-h on one GPU)
MAX_H=$([ "$VALIDATE" = 1 ] && echo 1.0 || echo 7.0)
( trap 'kill $S 2>/dev/null; exit 0' TERM
  sleep "$(python3 -c "print(int($MAX_H * 3600))")" & S=$!; wait $S
  echo ">> WATCHDOG: ${MAX_H} h reached"; pkill -f mch_newmodel_pod.py ) &
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
$PY -c 'from deepsteer.datasets.machiavelli import fetch_machiavelli; print(">> engine:", fetch_machiavelli())' \
  || { echo "BAIL: engine or bundle fetch, or a hash mismatch"; exit 1; }
echo ">> setup done at $(hours) h"
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_mch_newmodel_pod.py \
  tests/datasets/test_machiavelli.py 2>&1 | tail -2
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

if [ "$VALIDATE" = 1 ]; then
  for M in gpt_oss_20b qwen3_30b; do
    $PY $POD --stage validate --model $M --out "$OUT" || { echo "BAIL: validate $M"; exit 1; }
  done
  $PY - <<'EOF'
import gzip, json, pathlib
for d in sorted(pathlib.Path("papers/mch_machiavelli/outputs/newmodel").glob("validate_*")):
    rows = [json.loads(x) for x in gzip.open(d / "decisions.jsonl.gz", "rt")]
    meta = [json.loads(x) for x in (d / "run_meta.jsonl").read_text().splitlines()][-1]
    for a in ("CoTAgent", "GoodCoTAgent"):
        r = [x for x in rows if x["agent"] == a]
        print(f">> {d.name} {a}: {len(r)} decisions, first-attempt parse "
              f"{sum(x['attempts'] == 1 and not x['fallback'] for x in r) / len(r):.2f}, fallback "
              f"{sum(x['fallback'] for x in r) / len(r):.2f}, mean attempts "
              f"{sum(x['attempts'] for x in r) / len(r):.2f}")
    print(f">> {d.name}: {meta['decisions']} decisions in {meta['wall_s']} s "
          f"({meta['wall_s'] / 3600 / max(1, meta['decisions']):.2e} A100-h per decision)")
EOF
  echo ">> VALIDATE done at $(hours) h"
  kill $WATCHDOG 2>/dev/null
  exit 0
fi

# ---- stage 1: the GPT-OSS pilot, then the decision (Amendment 6) ----
$PY $POD --stage pilot --model gpt_oss_20b --out "$OUT" || { echo "BAIL: pilot"; exit 1; }
$PY $POD --stage decide --out "$OUT" || { echo "BAIL: decide"; exit 1; }
CONT=$($PY -c "import json; print(int(json.load(open('$OUT/pilot/decision.json'))['continue']))")
K=$($PY -c "import json; print(json.load(open('$OUT/pilot/decision.json'))['k'])")
echo ">> stage 1 decision: continue=$CONT k=$K ($(cat "$OUT/pilot/decision.json" | tr -d '\n' | head -c 600))"
[ "$CONT" = 1 ] || { echo ">> STOP after stage 1 (Amendment 6)"; kill $WATCHDOG 2>/dev/null; exit 0; }

# ---- stages 2 and 3 at the pilot's k ----
$PY $POD --stage cell --model gpt_oss_20b --k "$K" --out "$OUT" || { echo "BAIL: cell gpt_oss_20b"; exit 1; }
$PY $POD --stage cell --model qwen3_30b --k "$K" --out "$OUT" || { echo "BAIL: cell qwen3_30b"; exit 1; }
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
