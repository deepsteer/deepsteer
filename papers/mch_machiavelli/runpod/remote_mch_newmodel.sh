#!/usr/bin/env bash
# Remote runner for the MCH new-model rows (papers/MCH_PLAN.md §5, Amendments 4, 6, 7): CoTAgent
# and GoodCoTAgent on GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507, one A100-80GB. Orion launches;
# the launcher passes HF_TOKEN and VALIDATE. ST's and MRB's pinned vLLM stack.
#
#   VALIDATE=1  stack, models, engine + bundle (hash-verified), gate tests, then both models on
#               2 games x episode 0 x both agents, 15 decisions each, behind the render-parity check
#               then writes validate_decision.json (Amendment 8: the main run's Qwen checkpoint)
#   VALIDATE=0  needs validate_decision.json (SYNC_OUTPUTS); the same setup, then stage 1 (the
#               GPT-OSS pilot at 1,024 tokens into pilot_1024/, then the Qwen rate leg;
#               Amendment 9) -> decide (stop or continue, k) -> stage 2 (GPT-OSS cell at k) ->
#               stage 3 (the Qwen cell at k, on the checkpoint VALIDATE chose). Every stage resumes.
#               The 256-token pilot (pilot/) is the protocol-of-record verdict and is not rerun.
# The game bundle comes from Google Drive; if that fails or its hash mismatches, the pod prints
# NEED_SYNC mch_bundle and the launcher (ON_DEMAND_SYNC) copies the locally verified zip here.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
VALIDATE="${VALIDATE:-0}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/mch_machiavelli/outputs/newmodel"
mkdir -p "$OUT"
START=$(date +%s)
# the run's log is session_run_1024.log (Amendment 9), so it never overwrites the 256-token run's
trap 'cp "$REPO_DIR/session.log" "$OUT/session_$([ "$VALIDATE" = 1 ] && echo validate || echo run_1024).log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
POD=papers/mch_machiavelli/scripts/mch_newmodel_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
echo ">> MCH new-model rows start $(date -u +%FT%TZ) VALIDATE=$VALIDATE out=$OUT"

# wall caps: VALIDATE 1 h; the run 5.5 h (Amendment 9: 7 A100-h total, 1.5 already spent)
MAX_H=$([ "$VALIDATE" = 1 ] && echo 1.0 || echo 5.5)
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

# ---- which Qwen (Amendment 8): VALIDATE runs Thinking-2507; the main run reads VALIDATE's choice ----
if [ "$VALIDATE" = 1 ]; then
  QW=qwen3_30b
else
  [ -f "$OUT/validate_decision.json" ] || { echo "BAIL: no validate_decision.json (launch with SYNC_OUTPUTS=outputs/newmodel/validate_decision.json)"; exit 1; }
  QW=$($PY -c "import json; print(json.load(open('$OUT/validate_decision.json'))['qwen_model'])")
  echo ">> Qwen checkpoint for the main run: $QW ($(tr -d '\n' < "$OUT/validate_decision.json" | head -c 300))"
fi

# ---- models (GPT-OSS's original/ and metal/ copies skipped), engine and bundle ----
QW="$QW" $PY - <<'EOF' > "$OUT/models.json" || { echo "BAIL: model access or download"; exit 1; }
import json, os
from huggingface_hub import snapshot_download
pins = {"gpt_oss_20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
        "qwen3_30b": ("Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"),
        "qwen3_30b_instruct": ("Qwen/Qwen3-30B-A3B-Instruct-2507",
                               "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe")}
want = ("gpt_oss_20b", os.environ["QW"])
print(json.dumps({k: snapshot_download(pins[k][0], revision=pins[k][1],
                                       ignore_patterns=["original/*", "metal/*"]) for k in want}))
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
$PY -m pytest -q -p no:cacheprovider tests/scripts/test_mch_newmodel_pod.py \
  tests/datasets/test_machiavelli.py 2>&1 | tail -2
[ "${PIPESTATUS[0]}" = 0 ] || { echo "BAIL: gate tests"; exit 1; }

if [ "$VALIDATE" = 1 ]; then
  for M in gpt_oss_20b qwen3_30b; do
    $PY $POD --stage validate --model $M --out "$OUT" || { echo "BAIL: validate $M"; exit 1; }
  done
  $PY $POD --stage validate_decide --out "$OUT" || { echo "BAIL: validate_decide"; exit 1; }
  echo ">> Amendment 8 decision: $(tr -d '\n' < "$OUT/validate_decision.json")"
  $PY - <<'EOF'
import gzip, json, pathlib
# directories only: validate_decision.json matches the pattern too (the first VALIDATE's traceback)
for d in sorted(p for p in pathlib.Path("papers/mch_machiavelli/outputs/newmodel").glob("validate_*")
                if p.is_dir()):
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

# ---- stage 1 (Amendment 9): the GPT-OSS pilot at 1,024 tokens, the Qwen rate leg, the decision ----
TOK=1024
PD=pilot_1024
$PY $POD --stage pilot --model gpt_oss_20b --max-tokens $TOK --pilot-dir $PD --out "$OUT" || { echo "BAIL: pilot"; exit 1; }
$PY $POD --stage qwen_leg --model "$QW" --max-tokens $TOK --out "$OUT" || { echo "BAIL: qwen_leg"; exit 1; }
$PY $POD --stage decide --pilot-dir $PD --out "$OUT" || { echo "BAIL: decide"; exit 1; }
CONT=$($PY -c "import json; print(int(json.load(open('$OUT/$PD/decision.json'))['continue']))")
K=$($PY -c "import json; print(json.load(open('$OUT/$PD/decision.json'))['k'])")
$PY - "$OUT/$PD/decision.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for name, r in (("pilot 1024", d["pilot"]), ("qwen leg 1024", d["qwen_leg"]),
                ("record 256", d["protocol_of_record_256"])):
    if r:
        print(">> " + name + ": " + "; ".join(
            f"{a} fallback {r[a]['fallback_rate']:.3f} cap-hit {r[a]['cap_hit_rate']:.3f}"
            for a in ("CoTAgent", "GoodCoTAgent")) + f"; {r['a100h_per_decision']:.2e} A100-h/decision")
print(">> price A100-h:", {k: round(v, 2) for k, v in d["price_a100h"].items()}, "envelope", d["envelope_a100h"])
EOF
echo ">> stage 1 decision: continue=$CONT k=$K reason=$($PY -c "import json; print(json.load(open('$OUT/$PD/decision.json'))['reason'])")"
[ "$CONT" = 1 ] || { echo ">> STOP after stage 1 (Amendments 6, 9)"; kill $WATCHDOG 2>/dev/null; exit 0; }

# ---- stages 2 and 3 at the pilot's k, 1,024 tokens ----
$PY $POD --stage cell --model gpt_oss_20b --k "$K" --max-tokens $TOK --pilot-dir $PD --out "$OUT" || { echo "BAIL: cell gpt_oss_20b"; exit 1; }
$PY $POD --stage cell --model "$QW" --k "$K" --max-tokens $TOK --pilot-dir $PD --out "$OUT" || { echo "BAIL: cell $QW"; exit 1; }
echo ">> done at $(hours) h"
kill $WATCHDOG 2>/dev/null
