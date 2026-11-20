#!/usr/bin/env bash
# The standing step before any GPC-4 launch (the lead, after launches 1 and 2): run the pod's gate tests
# locally under the environment the pod runner has exported when it runs them, once quietly and then
# several times under heavy CPU load, all passing. Launch 1 bailed on a test that read an environment
# name the runner sets; launch 2 on a test whose order depended on load. Both were host-side tests,
# which the pod's gate now deselects (marker host_side; CI still runs them).
#
#   bash papers/gpc_goal_pressure_conduct/runpod/gpc4_pod_gates_local.sh        # LOAD_RUNS=3 by default
#
# In order: every path the runner exports into the repo exists; prep's CPU part (the summed adapter
# built from the runner's own paths and GPC-1's C1, exact); the runner's own stage block run dry, start
# to finish, under its real exports and the real local inputs (the pod-only /workspace paths mapped to
# local copies at the runner's pins, cached in GPC4_LOCAL_CACHE); the served tool parser read on the
# subject's own tool call with the runner's own vLLM (§11.18); then the gate, quietly and under load.
# Reads the runner's own marker expression, floor, TESTS list and `export` lines (those before its gate
# tests), so they never drift. Leaves out PATH (the local interpreter stays) and HF_TOKEN (never
# copied). Prints the variables' names only, never their values, the collected count against the floor,
# and each run's summary; exits nonzero on any failure.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=papers/gpc_goal_pressure_conduct/runpod/remote_gpc4.sh
LOAD_RUNS=${LOAD_RUNS:-3}
gate_at=$(grep -n 'pytest -q -p no:cacheprovider -m "\$GATE_MARK" \$TESTS' "$R" | head -1 | cut -d: -f1)
[ -n "$gate_at" ] || { echo "BAIL: the runner's gate-test line not found"; exit 1; }
tests=$(head -n "$gate_at" "$R" | grep -E '^TESTS=' | tail -1)
mark=$(head -n "$gate_at" "$R" | grep -E '^GATE_MARK=' | tail -1)
floor=$(head -n "$gate_at" "$R" | grep -E '^GATE_FLOOR=' | tail -1 | sed 's/ *#.*//')
[ -n "$tests" ] && [ -n "$mark" ] && [ -n "$floor" ] || { echo "BAIL: the runner's TESTS, GATE_MARK or GATE_FLOOR not found"; exit 1; }
exports=$(head -n "$gate_at" "$R" | grep -E '^export ' | grep -vE '^export (PATH|HF_TOKEN)=')
PY=${PY:-/opt/homebrew/bin/python3.13}
run_gate() {  # <label>
  bash -c "
set -euo pipefail
REPO_DIR=/workspace/deepsteer G=\$REPO_DIR/papers/gpc_goal_pressure_conduct
C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm
START=\$(date +%s) PRICE_H=10.5 SESSION=1
$exports
$tests
$mark
$floor
cd '$PWD'
if [ '$1' = quiet ]; then
  echo \">> the pod runner's exports at its gate tests: \$(printf '%s\n' \"$exports\" | grep -oE '[A-Z_0-9]+=' | tr -d = | sort -u | tr '\n' ' ')\"
  n=\$('$PY' -m pytest --collect-only -q -p no:cacheprovider -m \"\$GATE_MARK\" \$TESTS 2>/dev/null | grep -c '::' || true)
  echo \">> the pod's gate (-m \\\"\$GATE_MARK\\\"): \$n tests collected, floor \$GATE_FLOOR\"
  [ \"\$n\" -ge \"\$GATE_FLOOR\" ] || { echo 'BAIL: under the floor'; exit 1; }
fi
'$PY' -m pytest -q -p no:cacheprovider -m \"\$GATE_MARK\" \$TESTS 2>&1 | tail -1
exit \${PIPESTATUS[0]}
"
}
# every path the runner exports into the repo exists (launch 3 stopped at prep on a doubled outputs/ in
# GPC4_PCTX): the runner's paths are evaluated with the repo here, where the synced inputs come from
all_exports=$(grep -E '^export ' "$R" | grep -vE '^export (PATH|HF_TOKEN)=')
missing=$(bash -c "
REPO_DIR='$PWD' G=\$REPO_DIR/papers/gpc_goal_pressure_conduct
C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm
START=\$(date +%s) PRICE_H=10.5 SESSION=1
$all_exports
for v in \$(printf '%s\n' \"$all_exports\" | grep -oE '[A-Z_0-9]+=' | tr -d =); do
  val=\${!v}; case \"\$val\" in '$PWD'/*) [ -e \"\$val\" ] || echo \"\$v=\${val#'$PWD'/}\";; esac
done" 2>/dev/null)
n_paths=$(bash -c "REPO_DIR='$PWD' G=\$REPO_DIR/papers/gpc_goal_pressure_conduct C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in VENV=/workspace/venv_vllm START=0 PRICE_H=10.5 SESSION=1
$all_exports
for v in \$(printf '%s\n' \"$all_exports\" | grep -oE '[A-Z_0-9]+=' | tr -d =); do case \"\${!v}\" in '$PWD'/*) echo x;; esac; done" 2>/dev/null | wc -l | tr -d ' ')
[ -z "$missing" ] || { echo "BAIL: the runner exports paths into the repo that do not exist: $missing"; exit 1; }
echo ">> the runner's $n_paths exported repo paths all exist"
# prep's CPU part on the real files: the summed adapter from the runner's own GPC4_PCTX (evaluated
# with the repo here) and C1 from GPC-1's record, as prep builds it on the pod (§11.14 item 3)
CACHE=${GPC4_LOCAL_CACHE:-$HOME/.cache/deepsteer/gpc4_local}
mkdir -p "$CACHE"
pctx=$(bash -c "REPO_DIR='$PWD' G=\$REPO_DIR/papers/gpc_goal_pressure_conduct C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in START=0 PRICE_H=10.5 SESSION=1
$all_exports
echo \$GPC4_PCTX")
# as on the pod, the paths go through a symlink to the repo (/workspace is one there; launch 4's build
# failed comparing an unresolved path with the resolved repo root), and the pod's own build_sum runs
ln -sfn "$PWD" "$CACHE/wsroot"
pctx_ws="$CACHE/wsroot/${pctx#$PWD/}"
"$PY" - "$pctx_ws" "$CACHE/wsroot" "$CACHE/prep_cpu/sum/1" <<'PYEOF' || { echo "BAIL: prep's CPU part failed"; exit 1; }
import sys
from pathlib import Path
sys.path.insert(0, "papers/gpc_goal_pressure_conduct/scripts")
import gpc2_pod as P2
import gpc4_pod as P4
ws = Path(sys.argv[2])
c1 = P2.c1_dir(Path("papers/gpc_goal_pressure_conduct/outputs/gpc4_v1").resolve())
c1_ws = ws / Path(c1).relative_to(Path.cwd().resolve())
rec = P4.build_sum(Path(sys.argv[3]), str(c1_ws), sys.argv[1])
print(f">> prep's CPU part: the pod's build_sum makes the summed adapter through a symlinked repo path, "
      f"as on the pod (rank {rec['rank']}, alpha {rec['alpha']}, {rec['modules']} modules, exact)")
PYEOF
# the runner's own stage block, dry, start to finish, under its real exports and against the real
# local inputs (GPC-1's outputs linked in). The pod-only /workspace paths (the two clones, the
# ImpossibleBench data) map to local copies at the runner's own pins, fetched here if missing.
odcv_pin=$(grep -oE 'ODCV-Bench checkout -q [0-9a-f]{40}' "$R" | awk '{print $NF}')
am_pin=$(grep -oE 'agentic-misalignment checkout -q [0-9a-f]{40}' "$R" | awk '{print $NF}')
ib_rev=$(grep -oE 'impossible_livecodebench", repo_type="dataset", revision="[0-9a-f]{40}' "$R" | grep -oE '[0-9a-f]{40}')
fetch() {  # <dir> <url> <pin>
  [ -d "$1/.git" ] || git clone -q "$2" "$1"
  git -C "$1" checkout -q "$3" && [ "$(git -C "$1" rev-parse HEAD)" = "$3" ] || { echo "BAIL: $1 is not at $3"; exit 1; }
}
fetch "$CACHE/ODCV-Bench" https://github.com/McGill-DMaS/ODCV-Bench "$odcv_pin"
fetch "$CACHE/agentic-misalignment" https://github.com/anthropic-experimental/agentic-misalignment "$am_pin"
[ -d "$CACHE/ib_data/data" ] || "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('fjzzq2002/impossible_livecodebench', repo_type='dataset', revision='$ib_rev', local_dir='$CACHE/ib_data')" >/dev/null
B=$CACHE/blockdry; rm -rf "$B"; mkdir -p "$B/outputs"
ln -sfn "$PWD/papers/gpc_goal_pressure_conduct/outputs/gpc1_v1" "$B/outputs/gpc1_v1"
for port in 8000 8001; do
  ! lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1 || { echo "BAIL: port $port is in use (the dry block serves its stub and control there)"; exit 1; }
done
{ echo 'set -uo pipefail'; echo "cd '$PWD'"
  echo "REPO_DIR='$PWD' G=\$REPO_DIR/papers/gpc_goal_pressure_conduct C=\$REPO_DIR/papers/cst_constitutional_steering/outputs I=\$G/outputs/gpc4_in"
  echo "VENV=/workspace/venv_vllm SESSION=1 START=\$(date +%s) PRICE_H=10.5"
  echo "PY='$PY' GP=papers/gpc_goal_pressure_conduct/scripts/gpc4_pod.py OUT='$B/outputs/gpc4_v1'"
  echo 'mkdir -p "$OUT/logs" "$OUT/vm_results"; echo ok > "$OUT/logs/judge_download.log"'
  echo 'hours() { python3 -c "import time; print(round((time.time() - $START) / 3600, 4))"; }'
  echo "$all_exports"
  echo "export GPC4_ODCV_CLONE='$CACHE/ODCV-Bench' GPC_AM_DIR='$CACHE/agentic-misalignment' GPC4_IB_DATA='$CACHE/ib_data/data' GPC4_DRY=1"
  echo '( sleep 1 ) & WATCHDOG=$!; ( true ) & JDL=$!'
  sed -n '/^echo setup > "\$OUT\/.stage"/,$p' "$R"
} > "$B/run_block.sh"
bash "$B/run_block.sh" > "$B/run.log" 2>&1; brc=$?
[ -f "$B/outputs/gpc4_v1/serve.pid" ] && kill "$(cat "$B/outputs/gpc4_v1/serve.pid")" 2>/dev/null
pkill -f "$B/outputs/gpc4_v1" 2>/dev/null || true
if [ "$brc" != 0 ] || ! grep -q '^>> all stages run' "$B/run.log" || grep -qE '^>> stage failed|Traceback' "$B/run.log"; then
  grep -E '^>> stage failed|Traceback|Error' "$B/run.log" | head -5; echo "BAIL: the stage block did not run dry to its end (see $B/run.log)"; exit 1
fi
"$PY" - "$B/outputs/gpc4_v1" <<'PYEOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
r = lambda n: json.loads((o / f"{n}.json").read_text()) if (o / f"{n}.json").exists() else {}
stages = sum(1 for _ in open(o.parent.parent / "run.log") if _.startswith(">> stage done"))
print(f">> the runner's stage block ran dry to its end ({stages} stage calls): adapters usable "
      f"{r('prep').get('usable')}; arm smoke " + json.dumps({a: [k for k in ('odcv', 'ib') if v[k]]
      for a, v in r('arm_smoke').get('arms', {}).items()}) + f"; arms skipped "
      f"{sorted(r('arms').get('skipped', {}))}; stage 3 {r('gate3').get('skipped') or 'ran'}; "
      f"stage 4 {r('stage4').get('reason') or 'ran'}")
PYEOF
echo ">>   (dry, the stand-in model cannot load the real 27B adapters, so C1's and the sum's parity fail and"
echo ">>   their consumers take the skip path; the slow dry chain runs those paths with dry adapters)"
# the served tool parser on the subject's own call (§11.18; launch 5's hermes parsed none of Qwen3.8's XML
# calls): the runner's own vLLM wheel (its URL and sha256), its Python sources only (no compiled part
# runs), in a venv over this interpreter with the few modules the parser's imports need; then the gate's
# parser test runs here instead of skipping
wurl=$(grep -E '^VLLM_WHEEL_URL=' "$R" | cut -d'"' -f2); wsha=$(grep -E '^VLLM_WHEEL_SHA256=' "$R" | cut -d= -f2)
[ -n "$wurl" ] && [ -n "$wsha" ] || { echo "BAIL: the runner's VLLM_WHEEL_URL or VLLM_WHEEL_SHA256 not found"; exit 1; }
VW=$CACHE/vllm_$wsha
if [ ! -f "$VW/src/vllm/tool_parsers/__init__.py" ]; then
  mkdir -p "$VW/src"
  [ -f "$VW/vllm.whl" ] || curl -sSL -o "$VW/vllm.whl" "$wurl"
  [ "$(shasum -a 256 "$VW/vllm.whl" | cut -d' ' -f1)" = "$wsha" ] || { rm -f "$VW/vllm.whl"; echo "BAIL: the vLLM wheel's sha256 does not match the runner's"; exit 1; }
  # unzip's * crosses directories, so one pattern takes every .py (a second, unmatched one exits 11)
  ( cd "$VW/src" && unzip -q -o ../vllm.whl 'vllm/*.py' ) || { echo "BAIL: the vLLM wheel did not unpack"; exit 1; }
  rm -f "$VW/vllm.whl"
fi
if [ ! -x "$VW/venv/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$VW/venv"
  "$VW/venv/bin/pip" install -q cachetools==8.0.2 cbor2==6.3.0 fastapi==0.165.0 ijson==3.6.0 \
    mistral_common==1.13.1 msgspec==0.22.0 openai-harmony==0.0.8 partial-json-parser==0.2.1.1.post7 \
    prometheus_client==0.26.0 py-cpuinfo==9.0.0 pybase64==1.5.2 pyzmq==27.3.0 uvloop==0.23.0 \
    xgrammar==0.2.9 2>&1 | grep -v notice || true
fi
export GPC4_VLLM_PY=$VW/venv/bin/python GPC4_VLLM_SRC=$VW/src
tp=$("$PY" -c "import sys; sys.path.insert(0, 'papers/gpc_goal_pressure_conduct/scripts'); import gpc4_pod as P4; print(P4.TOOL_PARSER)")
probe() { "$GPC4_VLLM_PY" -I papers/gpc_goal_pressure_conduct/scripts/gpc4_parser_probe.py --parser "$1" --vllm-src "$GPC4_VLLM_SRC" 2>/dev/null | tail -1; }
p_ok=$(probe "$tp") || { echo "$p_ok"; echo "BAIL: vLLM's $tp parser does not read the subject's own tool call"; exit 1; }
p_ctl=$(probe hermes) && { echo "$p_ctl"; echo "BAIL: the probe's control (hermes) parsed the XML call: the probe cannot tell"; exit 1; }
echo ">> the served tool parser ($tp, vLLM at the runner's wheel) reads the subject's own call: $(echo "$p_ok" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["calls"])'); its control (hermes) parses none"
out=$(run_gate quiet) || { echo "$out"; echo "BAIL: the quiet run failed"; exit 1; }
echo "$out" | sed '$d'; echo ">> run 1, quiet: $(echo "$out" | tail -1)"
burners=()
stop_load() { [ ${#burners[@]} -gt 0 ] && kill "${burners[@]}" 2>/dev/null; burners=(); }
trap stop_load EXIT
for _ in $(seq 1 $(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) * 2 ))); do ( while :; do :; done ) & burners+=($!); done
fails=0
for k in $(seq 1 "$LOAD_RUNS"); do
  out=$(run_gate loaded) || fails=$((fails + 1))
  echo ">> run $((k + 1)), under load (two busy loops a core): $out"
done
stop_load
[ "$fails" = 0 ] || { echo "BAIL: $fails of $LOAD_RUNS loaded runs failed"; exit 1; }
echo ">> the standing step passed: 1 quiet run and $LOAD_RUNS under load"
