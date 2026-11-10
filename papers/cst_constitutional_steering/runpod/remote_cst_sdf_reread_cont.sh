#!/usr/bin/env bash
# Continuation of the re-read pod's session 2 (launch_cst_sdf_reread.sh). Its runner died with the
# pod's sshd at about 2.2 h (11:36 PST, 2026-11-10); the D-neutral corpus stage survived and finished,
# and nothing started the next stage. Started by the lead with the author's approval, detached
# (setsid nohup), appending to session.log. It runs the original runner's remaining lines unchanged
# and keeps the original session's clock, so the 9.0 h price check holds and the original watchdog
# (still alive) keeps the 10.5 h cap; the same EXIT trap writes the log copy and .session_done.
set -uo pipefail

REPO_DIR="${REPO_DIR:-/workspace/deepsteer}"
cd "$REPO_DIR"
OUT="$REPO_DIR/papers/cst_constitutional_steering/outputs/pilot_sdf"
SESSION=2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
START="${START_EPOCH:-1794331341}"  # 2026-11-10T17:22:21Z, the original session's start
PRICE_H=9.0
trap 'cp "$REPO_DIR/session.log" "$OUT/logs/session_sdf_reread${SESSION}_${STAMP}.log" 2>/dev/null; touch "$REPO_DIR/.session_done"' EXIT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_ENABLE_HF_TRANSFER=0
export CST_CONST_DIR="$REPO_DIR/papers/cst_constitutional_steering/outputs/_sync"
VENV=/workspace/venv_vllm
PY="${PY_OVERRIDE:-$VENV/bin/python}"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
PILOT=papers/cst_constitutional_steering/scripts/cst_pilot_sdf_pod.py
hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }
run() { $PY $PILOT --out "$OUT" "$@" || echo ">> stage failed: $*"; echo ">> stage done at $(hours) h: $*"; }
under_price() { python3 -c "import sys,time; sys.exit(0 if (time.time()-$START)/3600 < $PRICE_H else 1)"; }
gate_ok() { python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/corpus_gate_$1.json'))['proceed'] else 1)" 2>/dev/null; }
echo ">> continuation of session $SESSION at $(hours) h ($(date -u +%FT%TZ)): the runner died with sshd"

if gate_ok d_neutral; then
  run --stage train --rung d_neutral
  run --stage parity_hf --tags d_neutral_1M_rec
  run --stage read --tags d_neutral_1M_rec
else echo ">> D-neutral corpus gate failed: D-neutral not trained"; fi
run --stage read --tags d_chat_1M_rec
if under_price; then run --stage read --tags d_doc_03M_rec; else echo ">> over price: D-doc 0.3M reads skipped"; fi
if under_price; then run --stage read --tags d_doc_03M; else echo ">> over price: D-doc 0.3M without recovery skipped"; fi
if under_price; then run --stage read --tags d_raw_rec; else echo ">> over price: D-raw skipped"; fi

$PY - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
print(">> parity:", json.dumps({p.stem: json.loads(p.read_text())["passes"]
                                for p in sorted(o.glob("parity/*.json")) if not p.stem.startswith("hf_")}))
print(">> reads:", sorted(p.name for p in o.glob("gen/*") if p.is_dir()))
PYEOF
echo ">> done at $(hours) h"
