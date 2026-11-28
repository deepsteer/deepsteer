#!/usr/bin/env bash
# The real-bash sandbox's setup (GPC_PLAN.md §19.18): bubblewrap, and the 16 slot users the sandboxed commands run as
# (c2sbx00 to c2sbx15, no home, no login shell, no sudo). Run as root (the local container) or by a user with
# passwordless sudo (the pod). Idempotent. Nothing here loosens the host's namespace settings: if the kernel does not
# allow unprivileged namespaces, the self-test names it and the session stops (§19.18 item 4).
set -euo pipefail
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
if ! command -v bwrap >/dev/null || ! command -v prlimit >/dev/null || ! command -v pgrep >/dev/null; then
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q bubblewrap util-linux procps >/dev/null 2>&1 || {
    $SUDO apt-get update -q >/dev/null 2>&1 || true
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -q bubblewrap util-linux procps >/dev/null; }
fi
for i in $(seq -w 0 15); do
  id "c2sbx$i" >/dev/null 2>&1 || $SUDO useradd --system --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin "c2sbx$i"
done
# sudo warns on stderr when the host name does not resolve, and that warning would reach every command's output
grep -qw "$(hostname)" /etc/hosts || echo "127.0.1.1 $(hostname)" | $SUDO tee -a /etc/hosts >/dev/null
echo ">> sandbox setup: $(bwrap --version), slot users $(getent passwd | grep -c '^c2sbx'), kernel $(uname -r)"
