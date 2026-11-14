#!/usr/bin/env bash
# Laptop-side driver for a second provider's compatibility probe (CST Amendment 51 item A). The
# author creates and deletes the instance with the provider's own CLI (the API key stays on the
# laptop and never reaches the instance or this script) and passes the SSH target here: an ssh
# config alias (Thunder: `tnr connect <id>` writes `tnr-<id>`, user ubuntu, its own key) or
# user@host (with SSH_PORT and SSH_KEY set):
#
#   export HF_TOKEN=...
#   bash papers/cst_constitutional_steering/runpod/probe_provider.sh facts   TARGET
#   bash papers/cst_constitutional_steering/runpod/probe_provider.sh py311   TARGET   (then run with PROBE_PYTHON=python3.11)
#   bash papers/cst_constitutional_steering/runpod/probe_provider.sh run     TARGET
#   bash papers/cst_constitutional_steering/runpod/probe_provider.sh pull    TARGET
#   bash papers/cst_constitutional_steering/runpod/probe_provider.sh verdict PRICE_PER_HOUR
#
# facts: A1's go/no-go read only (nothing installed or downloaded). run: the /workspace symlink into
# the home directory, the repo with the launcher's own sync filters, the synced inputs and the game
# bundle, HF_TOKEN to a mode-600 file over the SSH channel, then remote_provider_probe.sh under nohup
# (about 2 h). pull: the results to outputs/provider_probe_v1 (run it again until the log says done).
# verdict: A9's line from the pulled records and the price the provider reports.
set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO_ROOT=$(pwd)
P=papers/cst_constitutional_steering
MODE=${1:?facts|run|pull|verdict}
if [ "$MODE" = verdict ]; then
  exec /opt/homebrew/bin/python3.13 "$P/scripts/provider_probe_verdict.py" "$P/outputs/provider_probe_v1" "${2:?price per hour}"
fi
R=${2:?ssh target: an ssh config alias or user@host}
SSH_OPTS=(-o ConnectTimeout=20 -o ServerAliveInterval=30)
[ -n "${SSH_PORT:-}" ] && SSH_OPTS+=(-p "$SSH_PORT")
[ -n "${SSH_KEY:-}" ] && SSH_OPTS+=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
RD=/workspace/deepsteer

case "$MODE" in
py311)  # Python 3.11 beside the host's python3, from the deadsnakes PPA (no CUDA package touched).
  # add-apt-repository is not used: it runs on the host's python3, which on Thunder is 3.12 without
  # Ubuntu's apt_pkg. The PPA's key comes from Ubuntu's keyserver and is checked by fingerprint.
  ssh "${SSH_OPTS[@]}" "$R" 'set -e; FP=F23C5A6CF475977595C89F51BA6932366A755776
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$FP" | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/deadsnakes.gpg
    gpg --show-keys --with-colons /etc/apt/keyrings/deadsnakes.gpg | grep -q "fpr:::::::::$FP:" || { echo "STOP: the deadsnakes key fingerprint does not match"; exit 1; }
    echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/deadsnakes.list >/dev/null
    sudo apt-get update -q >/dev/null 2>&1 || echo ">> apt-get update returned nonzero (post-update hooks on python3); installing anyway"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.11 python3.11-venv >/dev/null
    python3.11 --version; python3 --version'
  ;;
facts)
  ssh "${SSH_OPTS[@]}" "$R" 'echo "python: $(python3 --version 2>&1)"; echo "user: $(id)"; echo "sudo: $(sudo -n true 2>/dev/null && echo yes || echo no)"; echo "home: $HOME"; df -h "$HOME" | tail -1; echo "workspace: $(readlink -f /workspace 2>/dev/null || echo none)"; nvidia-smi'
  ;;
run)
  [ -n "${HF_TOKEN:-}" ] || { echo "BAIL: HF_TOKEN is not set"; exit 1; }
  # /workspace: a symlink into the persistent home directory (Amendment 51 item A3)
  ssh "${SSH_OPTS[@]}" "$R" '[ -e /workspace ] || { mkdir -p "$HOME/workspace" && sudo ln -s "$HOME/workspace" /workspace; } ; [ -w /workspace ] || sudo chown "$(id -u):$(id -g)" /workspace; mkdir -p /workspace/deepsteer /workspace/mch_bundle; command -v rsync >/dev/null || sudo apt-get install -y -q rsync'
  # the teacher pod's inputs (copied as its card does), P-ctx's training rollouts, the bundle
  rm -rf "$P/outputs/teacher_in" && mkdir -p "$P/outputs/teacher_in" && cp "$P/outputs/pressure/train_p_ctx.json" "$P/outputs/teacher_in/"
  M=outputs/finalist_c2_rule_right_mach/mach
  d=$(python3 -c "import json; p=list(json.load(open('$P/outputs/pressure/train_p_ctx.json'))['adapters'].values())[-1]; print(p.split('/papers/cst_constitutional_steering/')[1])")
  F=outputs/pilot_sdf
  SYNC="outputs/_sync/C_anth.txt,outputs/teacher_in/train_p_ctx.json,$M/decisions.jsonl,$M/episodes.jsonl"
  SYNC="$SYNC,$d/adapter_model.safetensors,$d/adapter_config.json"
  SYNC="$SYNC,$F/adapters/d_doc/1e+06/adapter_model.safetensors,$F/adapters/d_doc/1e+06/adapter_config.json"
  SYNC="$SYNC,$F/parity/hf_d_doc_1M.json,$F/parity/hf_base.json,outputs/pressure/scenarios.json"
  for f in "$P"/outputs/pressure/roll/a0s/chunk_*.jsonl; do SYNC="$SYNC,${f#"$P"/}"; done
  export SELF_PAPER=$P RESULTS_SUBPATH=outputs/provider_probe_v1 SYNC_OUTPUTS="$SYNC"
  RSYNC_EXCLUDE=$REPO_ROOT/papers/runpod_common/rsync_exclude.txt
  # shellcheck source=/dev/null
  source papers/runpod_common/session_lib.sh
  rp_output_filters
  rsync -az --delete ${RP_PRE_FILTERS[@]+"${RP_PRE_FILTERS[@]}"} --exclude-from "$RSYNC_EXCLUDE" "${RP_OUT_FILTERS[@]}" \
    -e "ssh ${SSH_OPTS[*]}" "$REPO_ROOT/" "$R:$RD/"
  IFS=',' read -ra paths <<< "$SYNC"
  for p in "${paths[@]}"; do
    ( cd "$REPO_ROOT" && rsync -az -R -e "ssh ${SSH_OPTS[*]}" "$P/$p" "$R:$RD/" )
  done
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOME/.cache/deepsteer/mch/dl/game_data.zip" "$R:/workspace/mch_bundle/game_data.zip"
  printf '%s' "$HF_TOKEN" | ssh "${SSH_OPTS[@]}" "$R" 'umask 077 && cat > "$HOME/.hf_token" && chmod 600 "$HOME/.hf_token"'
  ssh "${SSH_OPTS[@]}" "$R" "cd $RD && rm -f .session_done && PROBE_PYTHON=${PROBE_PYTHON:-python3} PROBE_ALLOW_PY=${PROBE_ALLOW_PY:-0} nohup bash $P/runpod/remote_provider_probe.sh > probe.log 2>&1 < /dev/null &"
  echo ">> started; follow with: ssh ${SSH_OPTS[*]} $R tail -f $RD/probe.log"
  ;;
pull)
  mkdir -p "$P/outputs/provider_probe_v1"
  rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude '*.safetensors' "$R:$RD/$P/outputs/provider_probe_v1/" "$P/outputs/provider_probe_v1/" \
    || echo ">> nothing to pull yet (the runner has not started)"
  ssh "${SSH_OPTS[@]}" "$R" "grep -E '^>>|STOP|BAIL' $RD/probe.log | tail -5; [ -f $RD/.session_done ] && echo SESSION_DONE" || true
  ;;
*) echo "usage: facts|py311|run|pull TARGET, or verdict PRICE"; exit 1;;
esac
