#!/usr/bin/env python3
"""C2's real-bash sandbox (GPC_PLAN.md §19.18; the lead's 2026-11-27 4:15 PM items 1 to 4): the environment for the
headroom pilot and cycle 1, in place of the emulator, which stays only as the Mac dry run's stand-in.

**A loop** gets its own directory, with the family's files, its real scorer (`c2_scorers.py`) and the directories
its task writes to. Each command is a fresh `bash -c` under bubblewrap with the directory mounted at `/app`, the
working directory `/app`, 30 s a command and an output cap of 8,192 bytes. After each command the tree is read back
from the directory (regular files only; a symlink is recorded by its target and never followed), and the outcome is
read as the emulator's is: the shortcut from the tree's diff against the original and the family's own check.

**Isolation** (each command):
- *no network:* every namespace unshared (`--unshare-all`), so only a loopback device exists;
- *no view of the host:* only `/usr` (and its `/bin`, `/lib` links) and a short list of `/etc` files are mounted,
  read-only; a generated `/etc/passwd`; no home directory, no token file, no repository, no `/workspace`;
- *writes only under `/app`:* everything else is read-only or a per-command `tmpfs` (`/tmp`);
- *a clean environment:* `--clearenv`, then the emulator's variables (no `HF_TOKEN`, nothing from the host);
- *limits:* the command runs as a slot user of its own (`c2sbx00` to `c2sbx15`, one a concurrent loop, so a process
  limit in one loop never starves another or the server), under `prlimit` (256 processes, 2 GiB of address space a
  process, 64 MiB a file, 30 CPU-seconds, 256 open files, no core files) and `timeout -s KILL 30`; the pid
  namespace dies with it, so a killed command leaves nothing running. A loop's directory over 64 MiB ends the loop
  with an error.

**The self-test** (`selftest`), the pod's first step before the model download (item 4), names each failure: the
network is reachable; the token file is readable; a write outside /app succeeds; a runaway command is not killed;
the output cap does not hold; and beside them, the host's environment, home or repository visible, and the process
and memory limits not holding. Each check has its positive control outside the sandbox (the host reaches the
network, reads the token file). Any failure bails. A host that does not allow the namespaces stops with that named,
and there is no fallback to an unisolated subprocess (§12).

    python c2_sandbox.py selftest [--out logs/sandbox_selftest.json] [--token-file ~/.hf_token]
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_env as E  # noqa: E402
from c2_scorers import scorer_source  # noqa: E402
from c2_shell import ENV  # noqa: E402

TIMEOUT_S = 30
OUT_CAP = 8192
DIR_CAP = 64 * 2**20
FILE_READ_CAP = 2**20
SLOTS = 16
SLOT_USER = "c2sbx{:02d}"
LIMITS = {"nproc": 256, "as": 2 * 2**30, "fsize": 64 * 2**20, "cpu": TIMEOUT_S, "nofile": 256, "core": 0}
BASE = Path(os.environ.get("C2_SANDBOX_BASE", "/tmp/c2_sandbox"))
ETC = ("alternatives", "ld.so.cache", "ld.so.conf", "ld.so.conf.d", "localtime", "timezone", "nsswitch.conf",
       "hosts", "magic", "magic.mime", "mime.types", "debian_version", "os-release", "lsb-release", "inputrc")
# the directories each family's task writes to (from its routes): a real shell needs them to exist
OUT_DIRS = {"D1-uptime-report": ["reports"], "D2-grant-summary": ["summary"], "I1-assay-qc": ["reports"],
            "I2-fridge-log": ["reports"], "R1-deploy-role": ["reports"], "R2-compute-quota": ["reports"],
            "M1-renewal-email": ["outbox"], "M2-volunteer-ask": ["outbox"], "X1-owed-refund": [],
            "X2-invented-quote": ["drafts"]}


class SandboxUnavailable(RuntimeError):
    """The host cannot run the sandbox (no bubblewrap, no slot users, or no namespaces). There is no fallback."""


def available() -> bool:
    return sys.platform.startswith("linux") and shutil.which("bwrap") is not None and _slot_uid(0) is not None


def _slot_uid(i: int) -> int | None:
    import pwd

    try:
        return pwd.getpwnam(SLOT_USER.format(i)).pw_uid
    except KeyError:
        return None


_local = threading.local()
_next = iter(range(10**9))
_lock = threading.Lock()


def _slot() -> int:
    """One slot user a worker thread, so concurrent loops never share a process limit."""
    if not hasattr(_local, "slot"):
        with _lock:
            _local.slot = next(_next) % SLOTS
    return _local.slot


def _as_slot(slot: int) -> list[str]:
    user = SLOT_USER.format(slot)
    if os.geteuid() == 0:
        uid = _slot_uid(slot)
        return ["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--"]
    return ["sudo", "-n", "-u", user, "--"]


def _support() -> Path:
    """The generated /etc/passwd and /etc/group the sandbox sees (the emulator's user, `agent`)."""
    d = BASE / "etc"
    if not (d / "passwd").exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / "passwd").write_text("root:x:0:0::/root:/usr/sbin/nologin\nagent:x:1000:1000::/home/agent:/bin/bash\n"
                                  "nobody:x:65534:65534::/nonexistent:/usr/sbin/nologin\n")
        (d / "group").write_text("root:x:0:\nagent:x:1000:\nnogroup:x:65534:\n")
        for p in (BASE, d, d / "passwd", d / "group"):
            p.chmod(0o755 if p.is_dir() else 0o644)
    return d


def bwrap_argv(app_dir: Path) -> list[str]:
    a = [shutil.which("bwrap") or "bwrap", "--unshare-all", "--unshare-user", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
         "--hostname", "sandbox", "--uid", "1000", "--gid", "1000", "--ro-bind", "/usr", "/usr"]
    for d in ("bin", "sbin", "lib", "lib32", "lib64", "libx32"):
        p = Path("/") / d
        if p.is_symlink():
            a += ["--symlink", os.readlink(p), str(p)]
        elif p.is_dir():
            a += ["--ro-bind", str(p), str(p)]
    a += ["--dir", "/etc"]
    for f in ETC:
        if (Path("/etc") / f).exists():
            a += ["--ro-bind", f"/etc/{f}", f"/etc/{f}"]
    s = _support()
    a += ["--ro-bind", str(s / "passwd"), "/etc/passwd", "--ro-bind", str(s / "group"), "/etc/group",
          "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", str(app_dir), "/app", "--chdir", "/app",
          "--remount-ro", "/", "--clearenv"]
    for k, v in ENV.items():
        a += ["--setenv", k, v]
    return a


def run_in_box(command: str, app_dir: Path, slot: int, timeout: int = TIMEOUT_S, cap: int = OUT_CAP
               ) -> tuple[str, int, dict]:
    """One command, a fresh `bash -c` in the sandbox: (output, exit code, facts: elapsed, bytes, truncated, killed).
    Output is read while the command runs, and past the cap it is drained and dropped."""
    lim = [f"--{k}={v}" for k, v in LIMITS.items()]
    argv = (_as_slot(slot) + ["prlimit", *lim, "--", "timeout", "-s", "KILL", str(timeout)]
            + bwrap_argv(app_dir) + ["--", "/bin/bash", "-c", command])
    t0 = time.time()
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True)
    kept, total, deadline = bytearray(), 0, t0 + timeout + 15
    fd = p.stdout.fileno()
    while True:
        r, _, _ = select.select([fd], [], [], max(0.0, deadline - time.time()))
        if not r:  # the backstop: timeout(1) should have killed it 15 s ago
            p.kill()
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        total += len(chunk)
        if len(kept) < cap:
            kept += chunk[: cap - len(kept)]
    code = p.wait()
    elapsed = time.time() - t0
    out = kept.decode("utf-8", errors="replace")
    if total > cap:
        out += f"\n[output truncated: {total} bytes in all, the first {cap} shown]"
    killed = code in (137, -9) and elapsed >= timeout - 1
    if killed:
        out += f"\nbash: the command was stopped after {timeout} s"
    if code != 0 and elapsed < 5 and out.startswith(("bwrap: ", "sudo: ", "setpriv: ", "prlimit: ")):
        # the sandbox did not start (its own error, before any bash ran): never answered as the command's output
        raise SandboxUnavailable(f"the sandbox did not start: {out[:160]!r}")
    return out, code, {"elapsed_s": round(elapsed, 3), "bytes": total, "truncated": total > cap, "killed": killed}


def _as_slot_run(slot: int, argv: list[str]) -> None:
    subprocess.run(_as_slot(slot) + argv, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_tree(app_dir: Path) -> dict[str, str]:
    """The tree as the outcome reads it: regular files as text (up to 1 MiB), a symlink as its target, never
    followed; a directory over the cap raises."""
    tree, size = {}, 0
    for d, dirs, names in os.walk(app_dir, followlinks=False):
        for n in names + [x for x in dirs if (Path(d) / x).is_symlink()]:
            full = Path(d) / n
            rel = str(full.relative_to(app_dir))
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                tree[rel] = f"(symlink to {os.readlink(full)})"
            elif stat.S_ISREG(st.st_mode):
                size += st.st_size
                if st.st_size > FILE_READ_CAP:
                    tree[rel] = f"(a file of {st.st_size} bytes)"
                else:
                    with open(full, "rb") as f:
                        tree[rel] = f.read().decode("utf-8", errors="replace")
    if size > DIR_CAP:
        raise RuntimeError(f"sandbox: the loop's directory is over {DIR_CAP} bytes")
    return tree


def sandbox_family(spec, twin: bool) -> E.Family:
    """The family with its real scorer in place of the emulator's stub."""
    fam = spec.family(twin)
    fam.files = dict(fam.files) | {spec.scorer_path: scorer_source(spec, twin)}
    return fam


@dataclass
class SandboxEnv(E.Env):
    """`c2_env.Env`'s interface (run, close, outcome, the log, the tree) over a real bash."""

    dirs: list = field(default_factory=list)
    app_dir: Path | None = None
    slot: int = 0
    facts: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not available():
            raise SandboxUnavailable("no bubblewrap or no slot users on this host (run c2_sandbox_setup.sh)")
        self._orig = dict(self.family.files)
        self.slot = _slot()
        _support()
        self.app_dir = Path(tempfile.mkdtemp(prefix="loop-", dir=BASE))
        self.app_dir.chmod(0o777)
        for d in self.dirs:
            (self.app_dir / d).mkdir(parents=True, exist_ok=True)
        for rel, text in self._orig.items():
            p = self.app_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            p.chmod(0o777 if text.startswith("#!") else 0o666)
        for d, sub, _ in os.walk(self.app_dir):
            for x in sub:
                (Path(d) / x).chmod(0o777)
        self.tree = read_tree(self.app_dir)

    def run(self, command: str) -> tuple[str, int]:
        self.turn += 1
        out, code, facts = run_in_box(command, self.app_dir, self.slot)
        _as_slot_run(self.slot, ["chmod", "-R", "a+rwX", str(self.app_dir)])  # what the agent made, readable here
        self.tree = read_tree(self.app_dir)
        self.facts.append(facts)
        self._record(command, out, code)
        return out, code

    def outcome(self) -> dict:
        return super().outcome() | {"shell": "sandbox", "timeouts": sum(f["killed"] for f in self.facts),
                                    "truncated": sum(f["truncated"] for f in self.facts)}

    def cleanup(self) -> None:
        if self.app_dir and self.app_dir.exists():
            _as_slot_run(self.slot, ["chmod", "-R", "a+rwX", str(self.app_dir)])
            shutil.rmtree(self.app_dir, ignore_errors=True)


def make_env(spec, twin: bool) -> SandboxEnv:
    return SandboxEnv(sandbox_family(spec, twin), dirs=list(OUT_DIRS.get(spec.fid, [])))


# ---- the isolation self-test (item 4) ----------------------------------------------------------------------------
PROBE_NET = ("python3 -c \"import socket; socket.create_connection(('{host}', {port}), timeout=4); print('REACHED')\""
             " 2>&1; getent hosts huggingface.co && echo RESOLVED")


def selftest(token_files: list[Path], repo: Path | None) -> dict:
    """Each check by name, with its positive control outside the sandbox where one exists; `ok` only when every
    check passes and every control holds."""
    if not available():
        return {"ok": False, "failures": ["the sandbox is not installed (bubblewrap or the slot users missing)"]}
    BASE.mkdir(parents=True, exist_ok=True)
    app = Path(tempfile.mkdtemp(prefix="selftest-", dir=BASE))
    app.chmod(0o777)
    checks, fail = {}, []

    def box(cmd: str, **kw):
        return run_in_box(cmd, app, 0, **kw)

    out, code, _ = box("echo sandbox-up")
    if out.strip() != "sandbox-up":
        shutil.rmtree(app, ignore_errors=True)
        return {"ok": False, "failures": [f"the namespaces the sandbox needs are not allowed here: {out[:200]!r}"],
                "checks": {"namespaces": {"pass": False}}}
    checks["namespaces"] = {"pass": True}
    out, _, _ = box("python3 -c 'print(6 * 7)'; awk 'BEGIN {print 6 * 7}'; sed --version | head -1")
    checks["tools"] = {"pass": out.splitlines()[:2] == ["42", "42"]}  # the scorers' interpreter, and awk
    if not checks["tools"]["pass"]:
        fail.append("python3 or awk does not run in the sandbox (the scorers need python3)")

    # 1. the network is reachable (the control: the host reaches it)
    host_reach = subprocess.run([sys.executable, "-c", "import socket; socket.create_connection(('1.1.1.1', 443), "
                                 "timeout=4)"], capture_output=True).returncode == 0
    out, _, _ = box(PROBE_NET.format(host="1.1.1.1", port=443))
    checks["network"] = {"pass": "REACHED" not in out and "RESOLVED" not in out, "control_host_reaches": host_reach}
    if not checks["network"]["pass"]:
        fail.append("the network is reachable")
    if not host_reach:
        fail.append("the network check has no positive control (the host itself reached nothing)")

    # 2. the token file is readable (the control: the runner reads it)
    tok = {}
    for t in token_files:
        t = t.expanduser()
        host_ok = t.is_file() and os.access(t, os.R_OK)
        out, _, _ = box(f"test -r {t} && echo READABLE; test -e {t} && echo EXISTS; true")
        tok[str(t)] = {"host_reads": host_ok, "box_reads": "READABLE" in out, "box_sees": "EXISTS" in out}
    checks["token_file"] = {"pass": not any(v["box_reads"] or v["box_sees"] for v in tok.values()),
                            "control_host_reads": any(v["host_reads"] for v in tok.values()), "files": tok}
    if not checks["token_file"]["pass"]:
        fail.append("the token file is readable")
    present = [t for t in token_files if t.expanduser().exists()]
    checks["token_file"]["present_on_host"] = len(present)
    if present and not checks["token_file"]["control_host_reads"]:
        fail.append("the token-file check has no positive control (the token file is not readable on the host)")

    # 3. a write outside /app succeeds (and /tmp inside never reaches the host's)
    marker = f"c2sbx-selftest-{os.getpid()}"
    targets = ["/", "/usr", "/etc", "/home", "/var/tmp", "/proc"] + ([str(repo)] if repo else []) + [str(BASE)]
    cmd = "; ".join(f"echo x > {d.rstrip('/')}/{marker} 2>/dev/null && echo WROTE:{d}" for d in targets)
    cmd += f"; echo x > /tmp/{marker} && echo TMP_INSIDE"
    out, _, _ = box(cmd)
    leaked = [d for d in targets if f"WROTE:{d}" in out] + [d for d in targets if Path(d, marker).exists()]
    leaked += ["/tmp (reached the host)"] if Path("/tmp", marker).exists() else []
    checks["write_outside_app"] = {"pass": not leaked, "leaked": leaked, "tmp_inside_ok": "TMP_INSIDE" in out}
    if leaked:
        fail.append(f"a write outside /app succeeds: {leaked}")
    out, _, _ = box(f"echo hello > /app/{marker} && cat /app/{marker}")
    if not (app / marker).exists() or "hello" not in out:
        fail.append("a write under /app does not reach the loop's directory")

    # 4. a runaway command is not killed (and nothing of it survives)
    t0 = time.time()
    out, code, facts = box("sleep 300 & sleep 300 & exec sleep 300")
    took = time.time() - t0
    time.sleep(1)
    left = subprocess.run(["pgrep", "-u", SLOT_USER.format(0)], capture_output=True, text=True).stdout.split()
    checks["runaway"] = {"pass": facts["killed"] and took < TIMEOUT_S + 10 and not left, "took_s": round(took, 1),
                         "exit": code, "left_running": len(left)}
    if not checks["runaway"]["pass"]:
        fail.append("a runaway command is not killed")

    # 5. the output cap does not hold
    out, _, facts = box("yes | head -c 5000000")
    checks["output_cap"] = {"pass": len(out.encode()) <= OUT_CAP + 200 and facts["bytes"] >= 5_000_000,
                            "kept": len(out.encode()), "bytes": facts["bytes"]}
    if not checks["output_cap"]["pass"]:
        fail.append("the output cap does not hold")

    # beside them: the environment, the host's home and repository, the process and memory limits
    os.environ.setdefault("C2_SELFTEST_CANARY", "canary-not-for-the-box")
    out, _, _ = box("env")
    names = sorted(ln.split("=", 1)[0] for ln in out.splitlines() if "=" in ln)
    checks["environment"] = {"pass": set(names) <= set(ENV) | {"SHLVL", "_", "OLDPWD"}, "names": names}
    if not checks["environment"]["pass"] or "HF_TOKEN" in names:
        fail.append(f"the host's environment reaches the sandbox: {sorted(set(names) - set(ENV))}")
    seen = [p for p in ["/root", "/workspace", str(Path.home())] + ([str(repo)] if repo else [])
            if "EXISTS" in box(f"test -e {p} && echo EXISTS; true")[0]]
    hl = box("ls -A /home 2>/dev/null | wc -l")[0].strip()
    checks["host_paths"] = {"pass": not seen and hl in ("0", ""), "visible": seen, "home_entries": hl}
    if not checks["host_paths"]["pass"]:
        fail.append(f"the host's home or repository is visible: {seen or '/home has entries'}")
    out, _, _ = box("python3 -c \"import subprocess\nps = []\nfor i in range(400):\n    try:\n        "
                    "ps.append(subprocess.Popen(['sleep', '3']))\n    except OSError:\n        break\n"
                    "print('STARTED', len(ps))\"")
    try:
        started = int(out.split("STARTED")[-1].split()[0])
    except (ValueError, IndexError):
        started = -1
    checks["process_limit"] = {"pass": 0 < started < LIMITS["nproc"], "started_of_400": started}
    if not checks["process_limit"]["pass"]:
        fail.append("the process limit does not hold")
    out, code, _ = box("python3 -c \"b = bytearray(3 * 2**30); print('ALLOCATED')\" 2>&1 | tail -1")
    checks["memory_limit"] = {"pass": "ALLOCATED" not in out and "MemoryError" in out}
    if not checks["memory_limit"]["pass"]:
        fail.append("the memory limit does not hold")

    _as_slot_run(0, ["chmod", "-R", "a+rwX", str(app)])
    shutil.rmtree(app, ignore_errors=True)
    for p in (Path("/tmp", marker),):
        p.unlink(missing_ok=True)
    return {"ok": not fail, "failures": fail, "checks": checks, "limits": LIMITS, "timeout_s": TIMEOUT_S,
            "out_cap": OUT_CAP, "bwrap": subprocess.run(["bwrap", "--version"], capture_output=True,
                                                         text=True).stdout.strip(), "kernel": os.uname().release}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--token-file", type=Path, action="append", default=None)
    ap.add_argument("--repo", type=Path, default=HERE.parents[2])
    a = ap.parse_args()
    rec = selftest(a.token_file or [Path("~/.hf_token"), Path("~/.cache/huggingface/token")], a.repo)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rec, indent=1))
    print(">> sandbox selftest:", json.dumps({"ok": rec["ok"], "failures": rec["failures"]}), flush=True)
    raise SystemExit(0 if rec["ok"] else 1)


if __name__ == "__main__":
    main()
