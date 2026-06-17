#!/usr/bin/env bash
# Local pre-flight for the Paper 6 Sprint 5 dependency code, run BEFORE any GPU spend.
#
# Two gates:
#   1. Unit-level checks (no large download): basis math, ablation hooks, CE DiD,
#      empty-basis control — on a small cached model with synthetic directions.
#   2. (REAL=1) Capped real run on the cached OLMo-3 7B base with the real Phase 2
#      directions — the exact production path at small scale.
#
# Usage:
#   bash papers/6_ablation_resistance/runpod/local_test.sh        # unit checks only (fast)
#   REAL=1 bash papers/6_ablation_resistance/runpod/local_test.sh # + real 7B (loads ~14 GB from cache)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SMALL_MODEL="${SMALL_MODEL:-allenai/OLMo-2-0425-1B}"
SMALL_DEVICE="${SMALL_DEVICE:-cpu}"

echo ">> Local pre-flight (model=$SMALL_MODEL device=$SMALL_DEVICE real=${REAL:-0})"
# Avoid empty-array expansion (macOS bash 3.2 errors under `set -u`); branch instead.
if [ "${REAL:-0}" = 1 ]; then
  python papers/6_ablation_resistance/scripts/local_test.py \
    --model "$SMALL_MODEL" --device "$SMALL_DEVICE" --real
else
  python papers/6_ablation_resistance/scripts/local_test.py \
    --model "$SMALL_MODEL" --device "$SMALL_DEVICE"
fi
rc=$?
if [ "$rc" -eq 0 ]; then
  echo ">> Local pre-flight PASSED. Next: RunPod dry-run ->"
  echo "     export RUNPOD_API_KEY=...; VALIDATE=1 papers/6_ablation_resistance/runpod/run_session.sh"
else
  echo ">> Local pre-flight FAILED (rc=$rc). Fix before spending GPU time."
fi
exit "$rc"
