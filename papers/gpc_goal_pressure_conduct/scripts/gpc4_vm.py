#!/usr/bin/env python3
"""GPC-4's VM side (GPC_PLAN.md §11.3c, the host page `ODCV_HOST_BRIEF.md`), standard library only.

One module for three machines: the EC2 host (the proxy, the driver, the two instruments' runs, the gate
tests), the pod (the control server), and the rehearsal's stand-in (the control server and a stub model).

    gpc4_vm.py proxy             # the sampling proxy at the bridge gateway and 127.0.0.1
    gpc4_vm.py driver            # the VM's stage driver (systemd; detached from the launcher)
    gpc4_vm.py control --port 8001 --results DIR --doc FILE   # the pod's (or stand-in's) control server
    gpc4_vm.py stub --port 8000  # the rehearsal's stand-in model (an OpenAI-compatible stub)
    gpc4_vm.py gate --phase pre|post|container   # the gate tests, JSON on stdout, exit 1 on a failure

The pod decides and the VM follows: the driver polls the pod's control document (the forward's
127.0.0.1:28001 is the pod's :8001), runs each instruction once, writes each finished run's results and
then a done-marker through the write-only results key, and waits until the control document lists the
marker as received. The VM carries no credential beyond the forward's key and the results key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BRIDGE = "odcv"
SUBNET, GATEWAY = "172.30.0.0/24", "172.30.0.1"
PROXY_PORT = 18000
FWD_MODEL, FWD_CONTROL = 28000, 28001  # the forward's local ends (the pod's :8000 and :8001)
STATE = Path(os.environ.get("GPC4_STATE", "/var/lib/gpc4"))
POD_ENV = Path(os.environ.get("GPC4_POD_ENV", "/etc/gpc4/pod.env"))
ODCV_DIR = Path(os.environ.get("GPC4_ODCV", "/opt/gpc4/odcv"))
IB_PY = os.environ.get("GPC4_IB_PY", "/opt/gpc4/ib-venv/bin/python")
# the sampling of record (GPC_PLAN.md §11.1): KDG sampler v2, effort medium, 4,096 tokens a generation
SAMPLING = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 4096}
EFFORT = {"reasoning_effort": "medium"}
ODCV_RUN_LIMIT_S = 40 * 60
TIMEOUT_MARK = re.compile(r"timed out|TimeoutExpired|timeout after", re.IGNORECASE)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)


# ---- the sampling proxy ----------------------------------------------------------------------------
def rewrite(body: dict, run_id: str) -> tuple[dict, int]:
    """The request with the sampling of record and one seed for this run and turn (the turn is the
    count of assistant messages so far, so a retried call keeps its seed)."""
    msgs = body.get("messages") or []
    turn = sum(1 for m in msgs if m.get("role") == "assistant")
    first = next((str(m.get("content")) for m in msgs if m.get("role") == "user"), "")
    conv = hashlib.sha256(first.encode()).hexdigest()[:12]
    seed = seed_of(run_id, conv, turn)
    out = {k: v for k, v in body.items() if k not in ("max_completion_tokens", "top_k", "n")}
    out.update(SAMPLING)
    out["seed"] = seed
    out["stream"] = False
    out["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), **EFFORT}
    return out, seed


def _run_of(path: str) -> tuple[str, str]:
    """``/run/<id>/v1/...`` names the run; anything else is the run ``-``."""
    m = re.match(r"^/run/([A-Za-z0-9_.:-]+)(/.*)$", path)
    return (m.group(1), m.group(2)) if m else ("-", path)


class Proxy(BaseHTTPRequestHandler):
    upstream = f"http://127.0.0.1:{FWD_MODEL}"
    logf = STATE / "proxy.jsonl"
    lock = threading.Lock()

    def log_message(self, *a):  # quiet
        pass

    def _forward(self, method: str, body: bytes | None):
        run_id, path = _run_of(self.path)
        seed = None
        if body and path.endswith("/chat/completions"):
            req, seed = rewrite(json.loads(body), run_id)
            body = json.dumps(req).encode()
        t0 = time.time()
        r = urllib.request.Request(self.upstream + path, data=body, method=method,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=900) as resp:
                data, code = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            data, code = e.read(), e.code
        except Exception as e:  # noqa: BLE001  (the upstream is down: report, do not crash)
            data, code = json.dumps({"error": str(e)}).encode(), 502
        usage = {}
        try:
            usage = json.loads(data).get("usage") or {}
        except Exception:  # noqa: BLE001
            pass
        with self.lock, open(self.logf, "a") as f:
            f.write(json.dumps({"t": t0, "run": run_id, "path": path, "seed": seed, "status": code,
                                "latency_s": round(time.time() - t0, 3), "usage": usage}) + "\n")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._forward("GET", None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self._forward("POST", self.rfile.read(n))


def serve_proxy(hosts: list[str]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    servers = []
    for h in hosts:
        s = ThreadingHTTPServer((h, PROXY_PORT), Proxy)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        log(f"proxy on {h}:{PROXY_PORT} -> {Proxy.upstream}")
    while True:
        time.sleep(3600)


# ---- the control server (the pod's, or the rehearsal stand-in's) ------------------------------------
def received(results: Path) -> list[str]:
    """Every done-marker that has arrived, with its files present at the hashes it lists."""
    out = []
    for m in sorted(results.glob("*/done.json")):
        try:
            rec = json.loads(m.read_text())
        except json.JSONDecodeError:
            continue
        ok = all((m.parent / f).exists() and hashlib.sha256((m.parent / f).read_bytes()).hexdigest() == h
                 for f, h in rec.get("files", {}).items())
        if ok:
            out.append(m.parent.name)
    return out


class Control(BaseHTTPRequestHandler):
    doc = Path("control.json")
    results = Path("vm_results")

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.rstrip("/") not in ("/control.json", "/control"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            d = json.loads(self.doc.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            d = {"instruction": {"kind": "wait", "id": "boot"}}
        d["received"] = received(self.results)
        data = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def next_instruction(plan: list[dict], got: list[str]) -> dict:
    """The first instruction whose done-marker has not arrived; ``stop`` when every one has."""
    for ins in plan:
        if ins["id"] not in got:
            return ins
    return {"kind": "stop", "id": "stop"}


def run_plan(plan_f: Path, doc: Path, results: Path, poll_s: int = 10) -> None:
    """Publish the plan's next instruction as the control document, re-reading the plan file each time
    (the pod rewrites it as it decides; the rehearsal's stand-in serves a fixed one)."""
    while True:
        plan = json.loads(plan_f.read_text())
        ins = next_instruction(plan.get("instructions", []), received(results))
        if plan.get("hold"):
            ins = {"kind": "wait", "id": "hold"}
        tmp = doc.with_suffix(".tmp")
        tmp.write_text(json.dumps({"session": plan.get("session"), "instruction": ins}))
        tmp.replace(doc)
        time.sleep(poll_s)


def serve_control(port: int, doc: Path, results: Path) -> None:
    Control.doc, Control.results = doc, results
    results.mkdir(parents=True, exist_ok=True)
    log(f"control on 127.0.0.1:{port}, doc {doc}, results {results}")
    ThreadingHTTPServer(("127.0.0.1", port), Control).serve_forever()


# ---- the rehearsal's stub model ------------------------------------------------------------------
class Stub(BaseHTTPRequestHandler):
    models = ["base", "c1", "cctx_stub"]

    def log_message(self, *a):
        pass

    def _send(self, d):
        data = json.dumps(d).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in self.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        msgs, tools = body.get("messages") or [], body.get("tools") or []
        turn = sum(1 for m in msgs if m.get("role") == "assistant")
        names = [t["function"]["name"] for t in tools if t.get("function")]
        if names:  # a tool loop: list the data, then finish
            name = "bash" if turn == 0 and "bash" in names else (
                "task_complete" if "task_complete" in names else names[0])
            args = {"command": "ls /app/data"} if name == "bash" else {"reason": "stub"}
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{turn}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}
            fin = "tool_calls"
        else:  # a code answer (ImpossibleBench's minimal scaffold)
            msg = {"role": "assistant", "content": "```python\ndef solve(*a, **k):\n    return None\n```"}
            fin = "stop"
        self._send({"id": "stub", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model"), "choices": [{"index": 0, "message": msg,
                                                             "finish_reason": fin}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "seed_seen": body.get("seed"), "temperature_seen": body.get("temperature")})


# ---- the driver --------------------------------------------------------------------------------
def pod_env() -> dict:
    env = {}
    for line in POD_ENV.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def control_doc() -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{FWD_CONTROL}/control.json", timeout=20) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001  (the pod is busy or the forward is reconnecting)
        return None


def push(local: Path, dest: str) -> bool:
    """rsync ``local`` to the pod's results directory (the results key: `rrsync -wo`, one directory)."""
    env = pod_env()
    ssh = (f"ssh -i /etc/gpc4/results_key -p {env['POD_PORT']} -o IdentitiesOnly=yes "
           "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes "
           "-o ConnectTimeout=20 -o LogLevel=ERROR")
    r = subprocess.run(["rsync", "-a", "-e", ssh, f"{local}/", f"{env['POD_USER']}@{env['POD_HOST']}:{dest}/"],
                       capture_output=True, text=True)
    if r.returncode:
        log(f"push {local} -> {dest} failed: {r.stderr.strip()[:300]}")
    return r.returncode == 0


def done_marker(d: Path, instr: dict) -> None:
    files = {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(d.rglob("*")) if p.is_file() and p.name != "done.json"}
    (d / "done.json").write_text(json.dumps({"instruction": instr, "files": files,
                                            "finished": time.time()}, indent=1))


def driver(poll_s: int = 20) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    done_f = STATE / "instructions_done.json"
    done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
    while not POD_ENV.exists():
        time.sleep(poll_s)
    log("driver: pod.env present; polling the control document")
    while True:
        doc = control_doc()
        if not doc:
            time.sleep(poll_s)
            continue
        ins = doc.get("instruction") or {"kind": "wait"}
        iid = ins.get("id", "")
        if ins.get("kind") == "stop":
            pending = [i for i in done if i not in set(doc.get("received", []))]
            if pending:
                log(f"driver: stop, waiting on delivery of {pending}")
                for i in pending:
                    push(STATE / "out" / i, i)
                time.sleep(poll_s)
                continue
            log("driver: stop; every result confirmed on the pod; powering off (terminates)")
            subprocess.run(["systemctl", "poweroff"], check=False)
            return
        if ins.get("kind") == "wait" or iid in done:
            if iid in done and iid not in set(doc.get("received", [])):
                push(STATE / "out" / iid, iid)
            time.sleep(poll_s)
            continue
        out = STATE / "out" / iid
        out.mkdir(parents=True, exist_ok=True)
        log(f"driver: {ins.get('kind')} {iid}")
        try:
            if ins["kind"] == "odcv":
                odcv_stage(ins, out)
            elif ins["kind"] == "ib":
                ib_stage(ins, out)
            elif ins["kind"] == "probe":
                (out / "probe.txt").write_text(f"probe {time.time()}\n")
                push(out, iid)
            else:
                (out / "error.json").write_text(json.dumps({"unknown kind": ins.get("kind")}))
        except Exception as e:  # noqa: BLE001  (record the stage's failure; the pod decides)
            (out / "stage_error.json").write_text(json.dumps({"error": repr(e)}))
        done_marker(out, ins)
        push(out, iid)
        done.add(iid)
        done_f.write_text(json.dumps(sorted(done)))


# ---- ODCV-Bench: one run in its own Compose project ------------------------------------------------
def compose_for(run_id: str, run_dir: Path, scenario_image: str, model: str) -> str:
    """A per-run Compose project from upstream's file: prebuilt images, no fixed names, no host port,
    the executor on the `odcv` bridge (it reaches only the proxy) and the scenario's internal net, the
    scenario container on the internal net only, one bind mount (the run's own prompt.json, read-only)."""
    return f"""name: odcv_{run_id.lower()}
networks:
  internal_net:
    driver: bridge
    internal: true
  {BRIDGE}:
    external: true
services:
  executor:
    image: gpc4/odcv-executor:latest
    volumes:
      - {run_dir}/prompt.json:/app/prompt.json:ro
    environment:
      - PYTHONUNBUFFERED=1
      - OPENROUTER_API_KEY=dummy
      - OPENAI_API_KEY=dummy
      - OPENAI_BASE_URL=http://{GATEWAY}:{PROXY_PORT}/run/{run_id}/v1
      - OPENAI_MODEL={model}
      - OPENAI_TEMPERATURE={SAMPLING['temperature']}
    depends_on:
      orchestrator_api:
        condition: service_healthy
    restart: "no"
    networks: [{BRIDGE}, internal_net]
  orchestrator_api:  # upstream's executor reaches it by this name (SUBMISSION_SERVER_URL)
    image: {scenario_image}
    environment:
      - PYTHONUNBUFFERED=1
    restart: "no"
    healthcheck:
      test: >-
        python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:5000/get_auto_approve_status"); print(1)' || exit 1
      interval: 2s
      timeout: 1s
      retries: 30
      start_period: 5s
    networks: [internal_net]
"""


def scenario_image(variant: str, scenario: str) -> str:
    return f"gpc4/odcv-{variant}-{scenario.lower()}:latest"


def odcv_run(scenario: str, variant: str, arm: str, model: str, clause: str | None, run_id: str,
             out: Path) -> dict:
    src = ODCV_DIR / f"{variant}_scenarios" / scenario
    rd = STATE / "runs" / run_id
    shutil.rmtree(rd, ignore_errors=True)
    rd.mkdir(parents=True)
    prompt = json.loads((src / "prompt.json").read_text())
    if clause:  # the in-context arm: the clause first in the system message, the rest unchanged
        prompt["system_prompt"] = f"{clause.strip()}\n\n{prompt['system_prompt']}"
    (rd / "prompt.json").write_text(json.dumps(prompt))
    (rd / "compose.yaml").write_text(compose_for(run_id, rd, scenario_image(variant, scenario), model))
    proj = ["docker", "compose", "-f", str(rd / "compose.yaml")]
    t0, rec = time.time(), {"run_id": run_id, "scenario": scenario, "variant": variant, "arm": arm,
                            "model": model, "clause": bool(clause)}
    try:
        p = subprocess.run(proj + ["up", "--abort-on-container-exit", "--exit-code-from", "executor"],
                           capture_output=True, text=True, timeout=ODCV_RUN_LIMIT_S)
        rec["compose_exit"], rec["run_timeout"] = p.returncode, False
        (out / "docker_output.log").write_text((p.stdout or "") + (p.stderr or ""))
    except subprocess.TimeoutExpired:
        rec["compose_exit"], rec["run_timeout"] = None, True
    for f in ("result.json", "messages_record.txt"):
        subprocess.run(proj + ["cp", f"executor:/app/{f}", str(out / f)], capture_output=True)
    cid = subprocess.run(proj + ["ps", "-a", "-q", "orchestrator_api"], capture_output=True,
                         text=True).stdout.strip()
    before = STATE / "hashes_before" / (scenario_image(variant, scenario).replace("/", "_") + ".txt")
    if before.exists():
        shutil.copy(before, out / "hashes_before.txt")
    if cid:  # the file trace against the starting image (GPC_PLAN.md §11.3a); works on a stopped container
        d = subprocess.run(["docker", "diff", cid], capture_output=True, text=True).stdout
        (out / "docker_diff.txt").write_text(d)
        changed = [ln[2:] for ln in d.splitlines() if ln[:1] in "AC" and ln[2:].startswith(
            ("/app", "/usr/local/bin"))]
        cf, after = out / "changed", []
        for path in changed[:200]:
            dest = cf / path.lstrip("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["docker", "cp", f"{cid}:{path}", str(dest)], capture_output=True)
            if dest.is_file():
                after.append(f"{hashlib.sha256(dest.read_bytes()).hexdigest()}  {path}")
        (out / "hashes_after_changed.txt").write_text("\n".join(after) + "\n")
    subprocess.run(proj + ["down", "-v"], capture_output=True)
    shutil.rmtree(rd, ignore_errors=True)
    msgs = (out / "messages_record.txt").read_text(errors="ignore") if (
        out / "messages_record.txt").exists() else ""
    rec["tool_timeouts"] = len(TIMEOUT_MARK.findall(msgs))
    rec["container_error"] = bool(rec.get("run_timeout")) or rec.get("compose_exit") not in (0,) or not (
        out / "result.json").exists()
    rec["wall_s"] = round(time.time() - t0, 1)
    (out / "run.json").write_text(json.dumps(rec, indent=1))
    return rec


def odcv_stage(ins: dict, out: Path) -> None:
    """``runs``: [scenario, variant, run_id] items; 8 at once (the host page's capacity)."""
    from concurrent.futures import ThreadPoolExecutor

    clause = ins.get("clause_text")

    def one(item):
        scenario, variant, run_id = item
        o = out / run_id
        o.mkdir(parents=True, exist_ok=True)
        rec = odcv_run(scenario, variant, ins["arm"], ins["model"], clause, run_id, o)
        push(o, f"{ins['id']}/{run_id}")  # results never sit only on the VM
        return rec

    with ThreadPoolExecutor(max_workers=int(ins.get("concurrency", 8))) as ex:
        recs = list(ex.map(one, ins["runs"]))
    (out / "stage.json").write_text(json.dumps({"runs": recs}, indent=1))


# ---- ImpossibleBench: Inspect on the VM, upstream's Docker sandbox ----------------------------------
IB_SRC = os.environ.get("GPC4_IB_SRC", "/opt/gpc4/impossiblebench/src/impossiblebench")
# the LiveCodeBench modules import each other as top-level modules (`from livecodebench_scorers import
# find_code`) and need only inspect_ai and datasets; the package's __init__ pulls in SWE-bench and an
# unpinned inspect_evals, so it is never imported
IB_TASK = r'''
import json, sys
sys.path.insert(0, sys.argv[2])
from inspect_ai import eval as inspect_eval
from inspect_ai.solver import chain, system_message
import livecodebench_tasks as T
a = json.loads(sys.argv[1])
fn = getattr(T, a["task_fn"])
task = fn(split=a["split"], agent_type=a["scaffold"], sandbox="docker", **a.get("task_kwargs", {}))
ids = set(a["task_ids"])
task.dataset = task.dataset.filter(lambda s: s.id in ids)
if a.get("clause_text"):  # the in-context arm: the clause first, upstream's prompt otherwise unchanged
    task.solver = chain(system_message(a["clause_text"]), task.solver)
logs = inspect_eval(task, model=f"openai-api/gpc4/{a['model']}", log_dir=a["log_dir"],
                    max_sandboxes=a.get("concurrency", 10), max_connections=a.get("concurrency", 10))
print(json.dumps([str(l.location) for l in logs]))
'''


def ib_stage(ins: dict, out: Path) -> None:
    log_dir = out / "inspect_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    arg = {**{k: ins[k] for k in ("split", "scaffold", "task_ids", "model")},
           "task_fn": ins.get("task_fn", "impossible_livecodebench"),
           "task_kwargs": ins.get("task_kwargs", {}), "clause_text": ins.get("clause_text"),
           "concurrency": ins.get("concurrency", 10), "log_dir": str(log_dir)}
    env = {k: v for k, v in os.environ.items() if not re.search(r"TOKEN|SECRET|AWS_", k)}
    env.update({"GPC4_BASE_URL": f"http://127.0.0.1:{PROXY_PORT}/run/{ins['id']}/v1",
                "GPC4_API_KEY": "dummy"})
    t0 = time.time()
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})  # the egress lock: the pinned cache
    p = subprocess.run([IB_PY, "-c", IB_TASK, json.dumps(arg), IB_SRC], capture_output=True, text=True,
                       env=env)
    (out / "inspect_stdout.txt").write_text(p.stdout[-20000:] + "\n---\n" + p.stderr[-20000:])
    (out / "stage.json").write_text(json.dumps({"exit": p.returncode, "wall_s": round(time.time() - t0, 1),
                                                "args": {k: v for k, v in arg.items()
                                                         if k != "clause_text"},
                                                "clause": bool(arg["clause_text"])}, indent=1))


# ---- the gate tests ------------------------------------------------------------------------------
def _tcp(host: str, port: int, t: float = 4.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


def _http_ok(url: str, t: float = 6.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=t) as r:
            return 200 <= r.status < 500
    except Exception:  # noqa: BLE001
        return False


def _imds_closed() -> bool:
    """Disabled, the metadata endpoint still accepts TCP but answers 403: closed means no 200 to the
    metadata GET or the IMDSv2 token PUT (the first rehearsal's false alarm read TCP alone)."""
    for method, path, hdr in (("GET", "/latest/meta-data/", {}),
                              ("PUT", "/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "60"})):
        try:
            r = urllib.request.Request(f"http://169.254.169.254{path}", method=method, headers=hdr)
            with urllib.request.urlopen(r, timeout=3) as resp:
                if resp.status == 200:
                    return False
        except Exception:  # noqa: BLE001  (403, refused or timed out: closed)
            pass
    return True


def gate(phase: str) -> dict:
    res = {}
    if phase in ("pre", "post"):
        res["imds_unreachable"] = _imds_closed()
        res["docker_up"] = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
        rules = subprocess.run(["iptables", "-S", "DOCKER-USER"], capture_output=True, text=True).stdout
        res["docker_user_rules"] = f"-s {SUBNET}" in rules and "DROP" in rules
        # the host's own clients (Inspect) use 127.0.0.1; the gateway is the containers' (container_gate)
        res["proxy_listening"] = _tcp("127.0.0.1", PROXY_PORT)
        res["container_tests"] = container_gate()
    if phase == "post":
        env = pod_env()
        res["internet_closed"] = not _tcp("1.1.1.1", 443, 4) and not _tcp("140.82.112.3", 443, 4)
        res["pod_port_open"] = _tcp(env["POD_HOST"], int(env["POD_PORT"]))
        res["forward_control"] = control_doc() is not None
        res["forward_model"] = _http_ok(f"http://127.0.0.1:{FWD_MODEL}/v1/models")
        probe = STATE / "out" / "gate_probe"
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "probe.txt").write_text(str(time.time()))
        done_marker(probe, {"kind": "probe", "id": "gate_probe"})
        res["results_path"] = push(probe, "gate_probe")
    flat = {k: v for k, v in res.items() if k != "container_tests"}
    ct = res.get("container_tests", {})
    res["passes"] = all(flat.values()) and all(ct.values()) if ct else all(flat.values())
    return res


CONTAINER_PROBE = r'''
import json, os, socket
def tcp(h, p):
    try:
        socket.create_connection((h, p), timeout=3).close(); return True
    except OSError:
        return False
import re
creds = {k: v for k, v in os.environ.items() if any(w in k.upper() for w in ("TOKEN", "KEY", "SECRET"))}
creds.pop("GPG_KEY", None)  # Python's official images bake in a public signing key's ID
leak = any(re.search(r"AKIA|ASIA|hf_[A-Za-z0-9]{20}|ghp_|sk-[A-Za-z0-9]{20}", v) for v in os.environ.values())
points = [ln.split()[1] for ln in open("/proc/mounts").read().splitlines() if len(ln.split()) > 1]
try:
    open("/app/prompt.json", "a").write("x"); ro = False
except Exception:
    ro = True
print(json.dumps({
  "no_credentials": all(v == "dummy" for v in creds.values()) and not leak,
  "no_host_paths": not any(p.startswith(("/var/lib/gpc4", "/etc/gpc4", "/home", "/root", "/opt/gpc4"))
                           or p.endswith("docker.sock") for p in points),
  "prompt_read_only": ro,
  "proxy_reachable": tcp("GATEWAY", PROXY_PORT),
  "host_ssh_closed": not tcp("GATEWAY", 22),
  "internet_closed": not tcp("1.1.1.1", 443),
  "imds_closed": not tcp("169.254.169.254", 80),
}))
'''.replace("GATEWAY", GATEWAY).replace("PROXY_PORT", str(PROXY_PORT))


def container_gate() -> dict:
    """From throwaway containers: the ODCV-Bench executor's network (the `odcv` bridge: the proxy and
    nothing else) and ImpossibleBench's (`network_mode: none`: nothing at all)."""
    probe = STATE / "gate_prompt.json"
    probe.write_text("{}")
    base = ["docker", "run", "--rm", "-e", "OPENAI_API_KEY=dummy", "-e", "OPENROUTER_API_KEY=dummy",
            "-v", f"{probe}:/app/prompt.json:ro"]
    out = {}
    r = subprocess.run(base + ["--network", BRIDGE, "gpc4/odcv-executor:latest", "python3", "-c",
                               CONTAINER_PROBE], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
        out.update({f"odcv_{k}": v for k, v in d.items()})
    except (IndexError, json.JSONDecodeError):
        out["odcv_probe_ran"] = False
    r = subprocess.run(base + ["--network", "none", "gpc4/odcv-executor:latest", "python3", "-c",
                               CONTAINER_PROBE], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
        out["none_proxy_unreachable"] = not d["proxy_reachable"]
        out["none_internet_closed"] = d["internet_closed"]
    except (IndexError, json.JSONDecodeError, KeyError):
        out["none_probe_ran"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("proxy", "driver", "control", "plan", "stub", "gate"))
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--port", type=int)
    ap.add_argument("--results", type=Path)
    ap.add_argument("--doc", type=Path)
    ap.add_argument("--phase", choices=("pre", "post", "container"))
    a = ap.parse_args()
    if a.cmd == "proxy":
        serve_proxy([GATEWAY, "127.0.0.1"])
    elif a.cmd == "driver":
        driver()
    elif a.cmd == "control":
        serve_control(a.port or 8001, a.doc, a.results)
    elif a.cmd == "plan":
        run_plan(a.plan, a.doc, a.results)
    elif a.cmd == "stub":
        ThreadingHTTPServer(("127.0.0.1", a.port or 8000), Stub).serve_forever()
    else:
        r = container_gate() if a.phase == "container" else gate(a.phase)
        print(json.dumps(r, indent=1))
        sys.exit(0 if (r.get("passes", all(r.values()))) else 1)


if __name__ == "__main__":
    main()
