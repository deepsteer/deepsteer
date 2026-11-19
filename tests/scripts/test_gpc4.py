"""GPC-4's VM side, launcher and IAM policy (GPC_PLAN.md §11.3 to §11.3c; ODCV_HOST_BRIEF.md).

No Docker, no AWS: the sampling proxy end to end against the stub, the per-run Compose project, the
control protocol and the driver's stop rule, the IAM policy's limits, and the launcher's and
cloud-init's rules read from their text. The rehearsal on AWS is the integration test (§11.3b)."""

from __future__ import annotations

import json
import re
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GPC = ROOT / "papers/gpc_goal_pressure_conduct"
sys.path.insert(0, str(GPC / "scripts"))

import gpc4_vm as V  # noqa: E402

POLICY = json.loads((GPC / "vm/gpc4_iam_policy.json").read_text())
LAUNCHER = (GPC / "runpod/launch_gpc4.sh").read_text()
VM_INIT = (GPC / "vm/cloud_init_vm.sh").read_text()
STANDIN_INIT = (GPC / "vm/cloud_init_standin.sh").read_text()


# ---- the sampling proxy --------------------------------------------------------------------------
def test_rewrite_applies_the_sampling_of_record_and_a_stable_seed():
    # most probable failure: ODCV-Bench's agent sends temperature 0 and it reaches the server, or a
    # retried call draws a new seed
    body = {
        "model": "c1",
        "temperature": 0.0,
        "max_completion_tokens": 9,
        "stream": True,
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}],
    }
    out, seed = V.rewrite(body, "run1")
    assert out["temperature"] == 0.7 and out["top_p"] == 1.0 and out["max_tokens"] == 4096
    assert "max_completion_tokens" not in out and out["stream"] is False
    assert out["chat_template_kwargs"]["reasoning_effort"] == "medium" and out["seed"] == seed
    assert V.rewrite(body, "run1")[1] == seed  # a retry keeps its seed
    later = {
        **body,
        "messages": body["messages"]
        + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}],
    }
    assert V.rewrite(later, "run1")[1] != seed and V.rewrite(body, "run2")[1] != seed


def test_run_path_names_the_run():
    assert V._run_of("/run/r1_odcv_0/v1/chat/completions") == ("r1_odcv_0", "/v1/chat/completions")  # noqa: SLF001
    assert V._run_of("/v1/models") == ("-", "/v1/models")  # noqa: SLF001


def _serve(handler):
    s = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def test_proxy_enforces_the_sampling_of_record_on_a_temperature_0_client(tmp_path, monkeypatch):
    # most probable failure: the proxy forwards the client's own sampling (the stub reports what
    # it saw)
    stub = _serve(V.Stub)
    monkeypatch.setattr(V.Proxy, "upstream", f"http://127.0.0.1:{stub.server_address[1]}")
    monkeypatch.setattr(V.Proxy, "logf", tmp_path / "proxy.jsonl")
    proxy = _serve(V.Proxy)
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy.server_address[1]}/run/x1/v1/chat/completions",
        data=json.dumps(
            {"model": "base", "temperature": 0.0, "messages": [{"role": "user", "content": "hi"}]}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    got = json.loads(urllib.request.urlopen(req, timeout=10).read())
    assert got["temperature_seen"] == 0.7 and isinstance(got["seed_seen"], int)
    row = json.loads((tmp_path / "proxy.jsonl").read_text().splitlines()[-1])
    assert row["run"] == "x1" and row["seed"] == got["seed_seen"] and row["status"] == 200
    stub.shutdown()
    proxy.shutdown()


def test_stub_answers_tool_loops_and_code_tasks():
    s = _serve(V.Stub)
    url = f"http://127.0.0.1:{s.server_address[1]}/v1/chat/completions"

    def ask(body):
        r = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(r, timeout=10).read())["choices"][0]["message"]

    tools = [{"type": "function", "function": {"name": n}} for n in ("bash", "task_complete")]
    assert ask({"messages": [], "tools": tools})["tool_calls"][0]["function"]["name"] == "bash"
    assert (
        ask({"messages": [{"role": "assistant", "content": "x"}], "tools": tools})["tool_calls"][0][
            "function"
        ]["name"]
        == "task_complete"
    )
    assert "```python" in ask({"messages": [{"role": "user", "content": "q"}]})["content"]
    s.shutdown()


# ---- ODCV-Bench's per-run Compose project --------------------------------------------------------
def test_compose_project_has_no_fixed_names_ports_or_extra_mounts():
    # most probable failure: upstream's fixed container names collide across concurrent runs, or the
    # scenario container joins the bridge (reaching the proxy) or a host port is published
    y = V.compose_for("r1", Path("/var/lib/gpc4/runs/r1"), "gpc4/odcv-mandated-x:latest", "c1")
    assert "container_name" not in y and "ports:" not in y and "build:" not in y
    assert "name: odcv_r1" in y and y.count(":ro") == 1 and "/var/lib/gpc4/runs/r1/prompt.json" in y
    # upstream's executor reaches the scenario at http://orchestrator_api:5000 (no fixed name now)
    assert "\n  orchestrator_api:" in y
    assert "orchestrator_api:\n        condition: service_healthy" in y
    ex, orch = y.split("\n  orchestrator_api:")
    assert f"networks: [{V.BRIDGE}, internal_net]" in ex and "networks: [internal_net]" in orch
    assert f"OPENAI_BASE_URL=http://{V.GATEWAY}:{V.PROXY_PORT}/run/r1/v1" in y
    assert re.findall(r"API_KEY=(\S+)", y) == ["dummy", "dummy"]


# ---- the control protocol and the driver ---------------------------------------------------------
def _marker(d: Path, files: dict[str, str], tamper: bool = False) -> None:
    d.mkdir(parents=True)
    for f, t in files.items():
        (d / f).write_text(t)
    V.done_marker(d, {"id": d.name})
    if tamper:
        (d / next(iter(files))).write_text("changed after the marker")


def test_received_counts_only_markers_whose_files_match(tmp_path):
    # most probable failure: a half-delivered stage (a marker before its files, or a changed
    # file) is taken as received, so the VM terminates with results missing
    _marker(tmp_path / "a", {"x.txt": "1"})
    _marker(tmp_path / "b", {"y.txt": "2"}, tamper=True)
    (tmp_path / "c").mkdir()
    assert V.received(tmp_path) == ["a"]


def test_next_instruction_walks_the_plan_then_stops():
    plan = [{"id": "p1", "kind": "probe"}, {"id": "p2", "kind": "odcv"}]
    assert V.next_instruction(plan, [])["id"] == "p1"
    assert V.next_instruction(plan, ["p1"])["id"] == "p2"
    assert V.next_instruction(plan, ["p1", "p2"])["kind"] == "stop"


def test_driver_powers_off_only_after_every_result_is_confirmed(tmp_path, monkeypatch):
    # most probable failure: the VM terminates on "stop" while a result is still undelivered
    # (its disk is deleted on termination)
    pod_env = tmp_path / "pod.env"
    pod_env.write_text("POD_HOST=10.0.0.1\nPOD_PORT=22\nPOD_USER=u\n")
    monkeypatch.setattr(V, "STATE", tmp_path / "state")
    monkeypatch.setattr(V, "POD_ENV", pod_env)
    pushes, calls = [], []
    docs = iter(
        [
            {"instruction": {"kind": "probe", "id": "p1"}, "received": []},
            {"instruction": {"kind": "stop", "id": "stop"}, "received": []},  # p1 not yet confirmed
            {"instruction": {"kind": "stop", "id": "stop"}, "received": ["p1"]},
        ]
    )
    monkeypatch.setattr(V, "control_doc", lambda: next(docs))
    monkeypatch.setattr(V, "push", lambda local, dest: pushes.append(dest) or True)
    monkeypatch.setattr(V.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(V.time, "sleep", lambda s: None)
    V.driver(poll_s=0)
    assert calls == [["systemctl", "poweroff"]]
    assert (
        pushes.count("p1") >= 3
    )  # the probe's own push, the result, and the re-push while unconfirmed
    assert json.loads((tmp_path / "state/instructions_done.json").read_text()) == ["p1"]


# ---- the IAM policy ------------------------------------------------------------------------------
def _actions(st):
    return [st["Action"]] if isinstance(st["Action"], str) else st["Action"]


def test_iam_policy_matches_the_brief_and_holds_its_limits():
    # most probable failure: a write action is left unconditioned (another instance, group or
    # key, or another region), or the brief's copy and the attached file drift apart
    brief = (GPC / "ODCV_HOST_BRIEF.md").read_text()
    m = re.search(r"## 5\. The minimal IAM policy.*?```json\n(.*?)```", brief, re.S)
    assert json.loads(m.group(1)) == POLICY
    acts = {a for st in POLICY["Statement"] for a in _actions(st)}
    assert not any(a.startswith("iam:") for a in acts)  # no PassRole: no instance profile
    assert {
        "ec2:RevokeSecurityGroupEgress",
        "ec2:AuthorizeSecurityGroupEgress",
        "sts:DecodeAuthorizationMessage",
        "servicequotas:GetServiceQuota",
    } <= acts
    by = {st["Sid"]: st for st in POLICY["Statement"]}
    assert by["RunTaggedOnly"]["Condition"]["StringEquals"]["ec2:InstanceType"] == [
        "c6a.4xlarge",
        "t3.micro",
    ]
    assert "arn:aws:ec2:us-east-2::snapshot/*" in by["RunSupporting"]["Resource"]
    # the second denial: the volume cannot satisfy an instance-type condition
    assert by["RunTaggedOnly"]["Resource"] == ["arn:aws:ec2:us-east-2:*:instance/*"]
    assert by["RunTaggedVolume"]["Condition"] == {
        "StringLike": {"aws:RequestTag/gpc4-session": "*"}
    }
    # the first rehearsal's denial: CreateSecurityGroup is authorized against the VPC too
    assert by["CreateGroupInAVpc"]["Action"] == "ec2:CreateSecurityGroup"
    assert by["CreateGroupInAVpc"]["Resource"] == "arn:aws:ec2:us-east-2:*:vpc/*"
    assert not any(
        "security-group" in r or "key-pair" in r for r in by["RunSupporting"]["Resource"]
    )
    assert by["RunInSessionGroupWithSessionKey"]["Condition"]["StringLike"] == {
        "aws:ResourceTag/gpc4-session": "*"
    }
    for st in POLICY["Statement"]:
        res = [st["Resource"]] if isinstance(st["Resource"], str) else st["Resource"]
        writes = [
            a
            for a in _actions(st)
            if not a.startswith(("ec2:Describe", "servicequotas:Get", "ssm:Get", "sts:Decode"))
        ]
        if writes:  # a write is region-bound by its condition or its ARNs
            assert st.get("Condition") or all("us-east-2:" in r for r in res), st["Sid"]
        for r in res:
            assert r == "*" or "us-east-2" in r


# ---- the launcher and cloud-init, from their text ------------------------------------------------
def test_launcher_holds_the_host_rules():
    # most probable failure: a rule the author ruled on is missing from the script that runs
    for needle in (
        "--instance-initiated-shutdown-behavior terminate",
        'DeleteOnTermination\\":true',
        "HttpTokens=required,HttpPutResponseHopLimit=1",
        "--http-endpoint disabled",
        "L-1216C47A",
        "a gpc4-session instance exists; run teardown first",
        "no passed rehearsal for this VM side",
        "GPC-4 runs on Thunder with no fallback",
        "revoke-security-group-egress",
        "authorize-security-group-egress",
        "Key=gpc4-session",
        "cmd_teardown",
        "cmd_reattach",
        "is not on origin/main (push it first)",
        "the VM side has uncommitted changes",
    ):
        assert needle in LAUNCHER, needle
    assert "--iam-instance-profile" not in LAUNCHER and "spot" not in LAUNCHER.lower()
    assert "TNR_FALLBACK=runpod_secure" not in LAUNCHER
    # the account ID is read and dropped, never echoed
    assert "unset acct" in LAUNCHER and not re.search(r"echo[^\n]*\$acct", LAUNCHER)
    h = LAUNCHER[LAUNCHER.index("handoff() {") : LAUNCHER.index("cmd_rehearse() {")]
    assert (
        h.index("revoke-security-group-egress")
        < h.index("authorize-security-group-egress")
        < h.index("gate --phase post")
    )
    v = LAUNCHER[LAUNCHER.index("vm_setup() {") : LAUNCHER.index("handoff() {")]
    assert (
        v.index("setup_done.json")
        < v.index("--http-endpoint disabled")
        < v.index("gate --phase pre")
    )


def test_cloud_init_carries_no_credential_and_arms_the_hard_limit_first():
    # most probable failure: a token lands in user-data (readable in the instance's attributes),
    # or the hard-limit shutdown is set after a step that can hang
    for t in (VM_INIT, STANDIN_INIT):
        assert not re.search(r"(AKIA|aws_secret|hf_[A-Za-z0-9]{20}|BEGIN OPENSSH PRIVATE)", t)
    assert VM_INIT.index('shutdown -h +"$HARD_LIMIT_MIN"') < VM_INIT.index("apt-get update")
    for pin in (
        "6924dafb3a3f84fc320fbe9f0a08203d67d1d488",
        "d4f9a149db2b9cec0fed7bab01d91c07ee10aca2",
        "98650ffc3f28a01b261669b6d19fcd7773823710",
        "inspect_ai==0.3.291",
    ):
        assert pin in VM_INIT
    assert "-s 172.30.0.0/24 -j DROP" in VM_INIT and "--dport 18000 -j ACCEPT" in VM_INIT
    assert 'permitopen="127.0.0.1:8000",permitopen="127.0.0.1:8001"' in STANDIN_INIT
    assert "-wo $R" in STANDIN_INIT


def test_rehearsal_plan_is_the_gates_own_scale():
    # most probable failure: the rehearsal runs a toy load, so the gate's 80 runs at 8 at once and
    # its 30 tasks a scaffold at 10 sandboxes run for the first time on GPU time (the lead's item 2)
    plan = json.loads((GPC / "vm/rehearsal_plan.json").read_text())["instructions"]
    assert len({i["id"] for i in plan}) == len(plan)
    odcv = [i for i in plan if i["kind"] == "odcv"]
    assert len(odcv) == 1 and len(odcv[0]["runs"]) == 80 and odcv[0]["concurrency"] == 8
    assert len({r[2] for r in odcv[0]["runs"]}) == 80
    assert {v for _, v, _ in odcv[0]["runs"]} == {"mandated", "incentivized"}
    ib = [i for i in plan if i["kind"] == "ib"]
    assert [i["scaffold"] for i in ib] == [
        "tools",
        "tools",
        "minimal",
        "minimal",
    ]  # the gate's order
    for i in ib:
        assert len(i["task_ids"]) == 15 and i["concurrency"] == 10
        # the gate's own settings (ib_instruction), not the old rehearsal's 2 attempts, 10 messages
        want = P4.ib_instruction(i["scaffold"], i["split"], [], "base", "base", "x")["task_kwargs"]
        assert i["task_kwargs"] == want
        assert len(set(i["task_ids"])) == 15 and all(
            t.startswith("lcbhard_") for t in i["task_ids"]
        )
        if list(P4.IB_PARQUET.glob(f"{i['split']}-*.parquet")):  # the pinned data, when present
            assert i["task_ids"] == P4.ib_sample(i["split"], 15, f"first|{i['scaffold']}", False)


def test_push_creates_each_parent_before_the_run(tmp_path, monkeypatch):
    # most probable failure: rsync makes only the last level of the destination, so a run's push to
    # <stage>/<run> fails until the stage's own push at its end (rehearsal 2, every ODCV-Bench run)
    calls = []
    monkeypatch.setattr(V, "pod_env", lambda: {"POD_HOST": "h", "POD_PORT": "22", "POD_USER": "u"})

    class R:
        returncode, stderr = 0, ""

    monkeypatch.setattr(V.subprocess, "run", lambda a, **k: calls.append(a[-1]) or R())
    assert V.push(tmp_path, "r1_odcv/r1_odcv_0")
    assert calls == ["u@h:r1_odcv/", "u@h:r1_odcv/r1_odcv_0/"]
    calls.clear()
    V.push(tmp_path, "gate_probe")
    assert calls == ["u@h:gate_probe/"]  # a top-level push needs no parent


def test_parallel_pushes_never_meet_rrsyncs_lock(tmp_path, monkeypatch):
    # most probable failure: 8 runs finishing together push at once and rrsync, which locks the
    # results directory per connection, refuses all but one (rehearsal 6: every run's own push)
    import threading
    import time

    monkeypatch.setattr(V, "pod_env", lambda: {"POD_HOST": "h", "POD_PORT": "22", "POD_USER": "u"})
    live, seen, guard = [0], [], threading.Lock()

    class R:
        def __init__(self, busy):
            self.returncode = 12 if busy else 0
            self.stderr = (
                "rrsync error: Another instance of rrsync is already accessing" if busy else ""
            )

    def fake(a, **k):
        with guard:
            live[0] += 1
            busy = live[0] > 1
            seen.append(live[0])
        time.sleep(0.01)
        with guard:
            live[0] -= 1
        return R(busy)

    monkeypatch.setattr(V.subprocess, "run", fake)
    V.PUSH_STATS.update(pushes=0, failed=0, busy_retries=0)
    res = []
    ts = [
        threading.Thread(target=lambda n=n: res.append(V.push(tmp_path, f"r1_odcv/run_{n}")))
        for n in range(8)
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert res == [True] * 8 and max(seen) == 1
    assert V.PUSH_STATS == {"pushes": 8, "failed": 0, "busy_retries": 0}


def test_a_push_retries_while_another_process_holds_the_lock(tmp_path, monkeypatch):
    # most probable failure: the post-lock gate's probe push (its own process) meets the driver's,
    # and one of them is lost on the first refusal
    monkeypatch.setattr(V, "pod_env", lambda: {"POD_HOST": "h", "POD_PORT": "22", "POD_USER": "u"})
    answers = iter([12, 12, 0])

    class R:
        def __init__(self, rc):
            self.returncode = rc
            self.stderr = (
                "Another instance of rrsync is already accessing this directory." if rc else ""
            )

    monkeypatch.setattr(V.subprocess, "run", lambda a, **k: R(next(answers)))
    V.PUSH_STATS.update(pushes=0, failed=0, busy_retries=0)
    assert V.push(tmp_path, "gate_probe", wait_s=0)
    assert V.PUSH_STATS["busy_retries"] == 2 and V.PUSH_STATS["failed"] == 0
    monkeypatch.setattr(V.subprocess, "run", lambda a, **k: R(12))
    assert not V.push(tmp_path, "gate_probe", tries=3, wait_s=0)
    assert V.PUSH_STATS["failed"] == 1  # a lock held throughout is a failed push, counted


def test_a_good_run_without_result_json_is_not_an_error(tmp_path):
    # most probable failure: the run errs on upstream's result.json, which its executor never writes
    # at the pin (rehearsal 2's good stub run read container_error true)
    (tmp_path / "messages_record.txt").write_text("== Step 1 ==\nrole: system\n")
    (tmp_path / "docker_output.log").write_text(
        "executor-1          | [Tool]: bash with {'command': 'ls /app/data'}\n"
        "executor-1          | [Tool]: task_complete with {'reason': 'stub'}\n"
        "\x1b[Kexecutor-1 exited with code 0\n\x1b[Korchestrator_api-1 exited with code 137\n"
    )
    rec = V.run_evidence(tmp_path, {"compose_exit": 0, "run_timeout": False})
    assert rec["container_error"] is False and rec["tool_calls"] == 2
    assert V.run_evidence(tmp_path, {"compose_exit": 1, "run_timeout": False})["container_error"]
    (tmp_path / "messages_record.txt").unlink()
    assert V.run_evidence(tmp_path, {"compose_exit": 0, "run_timeout": False})["container_error"]


def test_ib_stage_points_upstreams_loader_at_the_pinned_directory():
    # most probable failure: the loader asks the Hub by name, which fails offline even with the
    # files cached (rehearsal 2: OfflineModeIsEnabled on both stages)
    assert "T.hf_dataset = _pinned" in V.IB_TASK and "sys.argv[3]" in V.IB_TASK
    assert "IB_SRC, IB_DATA]" in (GPC / "scripts/gpc4_vm.py").read_text()
    assert f"GPC4_IB_DATA={V.IB_DATA}" in VM_INIT
    assert f'copytree(p, "{V.IB_DATA}"' in VM_INIT
    assert "refs" not in VM_INIT  # the old offline-by-name workaround is gone


def test_eval_counts_reads_scored_and_errored_samples(tmp_path):
    # most probable failure: an errored sample counts as scored, so verify passes an
    # ImpossibleBench stage that never ran a task
    (tmp_path / "inspect_logs").mkdir()
    (tmp_path / "inspect_logs" / "x.eval").write_bytes(b"")
    (tmp_path / "samples.json").write_text(
        json.dumps(
            [
                {"id": "a", "epoch": 1, "error": None, "scores": {"s": "I"}},
                {"id": "b", "epoch": 1, "error": "boom", "scores": {}},
            ]
        )
    )
    assert V.eval_counts(tmp_path) == {"n_logs": 1, "n_samples": 2, "n_scored": 1, "n_errored": 1}
    (tmp_path / "none" / "inspect_logs").mkdir(parents=True)
    assert V.eval_counts(tmp_path / "none")["n_scored"] == 0


def test_no_reader_opens_a_zstd_eval_log_with_the_standard_zipfile():
    # most probable failure: a .eval log is read with zipfile, which cannot open its zstd members
    # outside inspect_ai (rehearsal 3: both stages' counts failed; the pod's gate would have too)
    for f in ("gpc4_vm.py", "gpc4_pod.py"):
        assert "zipfile.ZipFile" not in (GPC / "scripts" / f).read_text(), f
    assert "read_eval_log(l.location)" in V.IB_TASK and '"samples.json"' in V.IB_TASK


def test_verify_keys_the_record_to_the_rehearsed_code_and_the_instruments(tmp_path):
    # most probable failure: verify passes on delivery alone (rehearsal 2 would have, with neither
    # instrument running), passes a stage short of its plan's scale, or names the tree it runs in
    # rather than the code the host ran
    import subprocess

    assert 'echo "$VM_HASH" > "$D/vm_hash"' in LAUNCHER
    assert '[ "$r_hash" = "$VM_HASH" ] || die' in LAUNCHER
    assert '"$r_commit" "$r_hash" "$G/vm/rehearsal_plan.json"' in LAUNCHER
    assert "and instruments_ran" in LAUNCHER
    # the stand-in's hard limit, armed first and filled by the launcher
    assert STANDIN_INIT.index('shutdown -h +"__HARD_LIMIT_MIN__"') < STANDIN_INIT.index("apt-get")
    assert "s/__HARD_LIMIT_MIN__/$STANDIN_LIMIT_MIN/" in LAUNCHER
    # the record's own code, on a mock session at the plan's scale
    code = re.search(
        r"<<'EOF'\n(import json, sys, datetime, pathlib.*?)\nEOF", LAUNCHER, re.S
    ).group(1)
    (tmp_path / "v.py").write_text(code)
    plan = json.loads((GPC / "vm/rehearsal_plan.json").read_text())["instructions"]
    D = tmp_path / "D"
    for f, v in (
        ("gate_pre.json", {"passes": True}),
        ("gate_post.json", {"passes": True}),
        ("setup_done.json", {"setup_s": 1}),
    ):
        (D / f).parent.mkdir(parents=True, exist_ok=True)
        (D / f).write_text(json.dumps(v))

    def session(odcv_errors=0, ib_short=0, push_failed=0):
        for ins in plan:
            d = D / "rehearsal_results" / ins["id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "resources.json").write_text(
                json.dumps({"wall_s": 5, "peak_mem_used_gb": 3.0, "push_failed": push_failed})
            )
            if ins["kind"] == "odcv":
                runs = [
                    {
                        "run_id": r[2],
                        "compose_exit": 0,
                        "tool_calls": 2,
                        "container_error": n < odcv_errors,
                    }
                    for n, r in enumerate(ins["runs"])
                ]
                (d / "stage.json").write_text(json.dumps({"runs": runs}))
            elif ins["kind"] == "ib":
                n = len(ins["task_ids"]) - ib_short
                (d / "stage.json").write_text(
                    json.dumps(
                        {"exit": 0, "n_logs": 1, "n_samples": n, "n_scored": n, "n_errored": 0}
                    )
                )
        home = tmp_path / "home"
        (home / ".gpc4").mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                sys.executable,
                str(tmp_path / "v.py"),
                str(D),
                "terminated",
                "true",
                "2026-11-19T00:00:00+00:00",
                "c0",
                "h0",
                str(GPC / "vm/rehearsal_plan.json"),
            ],
            capture_output=True,
            text=True,
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        )
        assert r.returncode == 0, r.stderr
        return json.loads((home / ".gpc4" / "rehearsal_h0.json").read_text())

    good = session()
    assert good["passed"] and good["instruments"]["stages"]["r1_odcv"]["runs"] == 80
    assert good["instruments"]["stages"]["r1_odcv"]["peak_mem_used_gb"] == 3.0
    assert not session(odcv_errors=1)["passed"]  # one container error of 80 fails the rehearsal
    assert not session(ib_short=1)["passed"]  # 14 of 15 tasks scored fails it
    assert not session(push_failed=1)["passed"]  # a lost push fails it (rehearsal 6's fault)


def test_verify_reads_the_results_where_it_unpacks_them(tmp_path):
    # most probable failure: the download lands one level off from where the record reads, so every
    # instrument reads null (rehearsal 4: a passing rehearsal recorded as failed)
    import subprocess

    m = re.search(r'tar -C "\$D/(\w+)" -xf -', LAUNCHER)
    assert m and f'R = D / "{m.group(1)}"' in LAUNCHER
    assert "sudo tar -C /home/gpc4pod/vm_results -cf - ." in LAUNCHER
    # the same pipe, locally: a stage's file lands at <dir>/<stage>/stage.json
    src, dst = tmp_path / "vm_results" / "r1_odcv", tmp_path / "rehearsal_results"
    src.mkdir(parents=True)
    dst.mkdir()
    (src / "stage.json").write_text("{}")
    subprocess.run(f"tar -C {src.parent} -cf - . | tar -C {dst} -xf -", shell=True, check=True)
    assert (dst / "r1_odcv" / "stage.json").exists()
    assert "no stage results downloaded" in LAUNCHER


STUB_AWS = r"""#!/usr/bin/env bash
echo "aws $*" >> "$STUB_DIR/aws.log"
a="$*"
case "$a" in
  *"sts get-caller-identity"*) echo 000000000000 ;;
  *"get-service-quota"*) echo 32.0 ;;
  *"describe-instances"*CpuOptions*) ;;
  *"describe-instances"*"instance-state-name"*) cat "$STUB_DIR/running" 2>/dev/null || true ;;
  *"describe-instances"*"--instance-ids"*PublicIpAddress*) echo 192.0.2.10 ;;
  *"describe-instances"*"--instance-ids"*) echo running ;;
  *"describe-instances"*) [ -f "$STUB_DIR/running" ] \
      && echo "$(cat "$STUB_DIR/running") c6a.4xlarge running s" || true ;;
  *"describe-vpcs"*) echo vpc-0stub ;;
  *"create-security-group"*) touch "$STUB_DIR/sg"; echo sg-0stub ;;
  *"authorize-security-group"*|*"ec2 wait"*|*"modify-instance-metadata-options"*) ;;
  *"create-key-pair"*) touch "$STUB_DIR/key"; echo KEY ;;
  *"ssm get-parameter"*) echo ami-0stub ;;
  *"run-instances"*) echo i-0stub > "$STUB_DIR/running"; echo i-0stub ;;
  *"terminate-instances"*) rm -f "$STUB_DIR/running" ;;
  *"describe-security-groups"*) [ -f "$STUB_DIR/sg" ] && echo sg-0stub || true ;;
  *"delete-security-group"*) rm -f "$STUB_DIR/sg" ;;
  *"describe-key-pairs"*) [ -f "$STUB_DIR/key" ] && echo key-0stub || true ;;
  *"delete-key-pair"*) rm -f "$STUB_DIR/key" ;;
  *) echo "unstubbed aws call: $a" >&2; exit 9 ;;
esac
"""
STUB_SSH = r"""#!/usr/bin/env bash
echo "ssh $*" >> "$STUB_DIR/ssh.log"
case "$*" in
  *setup_done.json*) [[ "$*" == *cat* ]] && echo '{"setup_s": 1}' ;;
  *"gate --phase pre"*) echo '{"passes": true}' ;;
esac
exit 0
"""
STUB_CURL = r"""#!/usr/bin/env bash
case "$*" in
  *thundercompute*) cat "$STUB_DIR/capacity.json" ;;
  *) echo 192.0.2.1 ;;
esac
"""
STUB_GIT = r"""#!/usr/bin/env bash
case "$1" in rev-parse) echo 0000000000000000000000000000000000000000 ;; esac
exit 0
"""
BAIL_CARD = """#!/usr/bin/env bash
echo "ERROR: no Thunder instance after 6 attempts. Last message: no capacity."
exit 1
"""


def _dry_launch(tmp_path, a100xl: str):
    """`launch` with every outside call stubbed (aws, ssh, scp, curl, git) and the pod card forced
    to bail; returns (the completed process, the stub's aws log, seconds taken)."""
    import hashlib
    import os
    import shutil
    import subprocess
    import time

    if not shutil.which("jq") or not shutil.which("ssh-keygen"):
        pytest.skip("the launcher needs jq and ssh-keygen")
    stub, root = tmp_path / "bin", tmp_path / "gpc4root"
    stub.mkdir()
    root.mkdir()
    for name, body in (
        ("aws", STUB_AWS),
        ("ssh", STUB_SSH),
        ("scp", "#!/usr/bin/env bash\nexit 0\n"),
        ("curl", STUB_CURL),
        ("git", STUB_GIT),
    ):
        (stub / name).write_text(body)
        (stub / name).chmod(0o755)
    (tmp_path / "card.sh").write_text(BAIL_CARD)
    (tmp_path / "capacity.json").write_text(
        json.dumps({"gpu_type": {"a100xl": {"1": a100xl}}, "specs": {"a100xl_x1": a100xl}})
    )
    vm_files = re.search(r"VM_FILES=\((.*?)\)", LAUNCHER, re.S).group(1)
    files = [
        f.strip('"').replace("$G/", "papers/gpc_goal_pressure_conduct/") for f in vm_files.split()
    ]
    vm_hash = hashlib.sha256(b"".join((ROOT / f).read_bytes() for f in files)).hexdigest()[:16]
    (root / f"rehearsal_{vm_hash}.json").write_text(json.dumps({"passed": True}))
    env = {
        **os.environ,
        "PATH": f"{stub}:{os.environ['PATH']}",
        "STUB_DIR": str(tmp_path),
        "GPC4_ROOT": str(root),
        "GPC4_POD_CARD": str(tmp_path / "card.sh"),
        "GPC4_IN_DIR": str(tmp_path / "gpc4_in"),
        "GPC4_POD_POLL_S": "1",
        "AWS_PROFILE": "stub",
    }
    env.pop("TNR_FALLBACK", None)
    t0 = time.time()
    r = subprocess.run(
        ["bash", str(GPC / "runpod/launch_gpc4.sh"), "launch"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = (tmp_path / "aws.log").read_text() if (tmp_path / "aws.log").exists() else ""
    return r, log, time.time() - t0


def test_launch_stops_at_once_and_removes_the_host_when_the_pod_card_bails(tmp_path):
    # most probable failure: with no capacity the card bails, launch waits an hour for an endpoint
    # that cannot appear, then leaves the host up to its 14 h limit (the lead's review, item 1)
    r, log, secs = _dry_launch(tmp_path, "available")
    assert r.returncode != 0, r.stdout
    assert "the pod launcher exited without an endpoint" in r.stderr, r.stderr[-1500:]
    assert secs < 60, secs  # not the hour's wait
    order = [
        c
        for c in (
            "run-instances",
            "terminate-instances",
            "delete-security-group",
            "delete-key-pair",
        )
        if c in log
    ]
    assert order == [
        "run-instances",
        "terminate-instances",
        "delete-security-group",
        "delete-key-pair",
    ], log
    assert log.index("run-instances") < log.index("terminate-instances")
    assert not (tmp_path / "running").exists() and not (tmp_path / "sg").exists()
    assert not (tmp_path / "key").exists()


def test_launch_creates_nothing_when_thunder_lists_no_a100(tmp_path):
    # most probable failure: the host is built (and billed) before Thunder is known to have a GPU
    r, log, _ = _dry_launch(tmp_path, "unavailable")
    assert r.returncode != 0 and "Thunder lists no A100 80GB x1" in r.stderr, r.stderr[-1500:]
    for call in ("create-security-group", "create-key-pair", "run-instances"):
        assert call not in log, log


@pytest.mark.parametrize("cmd", ["rehearse", "verify", "launch", "reattach", "teardown"])
def test_launcher_dispatches_each_command(cmd):
    assert f"  {cmd}) cmd_{cmd} ;;" in LAUNCHER


def test_gate_reads_the_first_rehearsals_false_alarms_correctly():
    # most probable failure (the first rehearsal): a disabled metadata endpoint still accepts TCP,
    # the images' public GPG_KEY reads as a credential, and /dev/root's source reads as /root
    probe = V.CONTAINER_PROBE
    assert 'creds.pop("GPG_KEY", None)' in probe and "ln.split()[1]" in probe
    assert "def _imds_closed" in (GPC / "scripts/gpc4_vm.py").read_text()
    assert "ESTABLISHED,RELATED -j ACCEPT" in VM_INIT.split("# the services")[0].split("INPUT")[-1]


# ---- the pod's side ------------------------------------------------------------------------------
import gpc4_pod as P4  # noqa: E402

RUNNER = (GPC / "runpod/remote_gpc4.sh").read_text()


def test_an_empty_plan_holds_and_never_stops_the_vm(tmp_path):
    # most probable failure: the pod publishes an empty plan while it works offline, and the plan
    # runner reads "every instruction received" as stop, powering the host off before its stages
    P4.plan_set(tmp_path, [], hold=True)
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["hold"] is True
    assert V.next_instruction(plan["instructions"], [])["kind"] == "stop"  # why hold is needed
    doc = tmp_path / "control.json"
    t = threading.Thread(
        target=V.run_plan, args=(tmp_path / "plan.json", doc, tmp_path / "r"), daemon=True
    )
    (tmp_path / "r").mkdir()
    t.start()
    for _ in range(50):
        if doc.exists():
            break
        threading.Event().wait(0.05)
    assert json.loads(doc.read_text())["instruction"]["kind"] == "wait"


def test_ib_scores_read_upstreams_scorer_from_the_runners_samples(tmp_path):
    # most probable failure: a pass is read from the wrong field, or an errored sample counts as
    # a fail
    d = tmp_path / "vm_results" / "ins"
    d.mkdir(parents=True)
    (d / "samples.json").write_text(
        json.dumps(
            [
                {"id": "a", "epoch": 1, "error": None, "scores": {"s": "C"}},
                {"id": "b", "epoch": 1, "error": None, "scores": {"s": "I"}},
                {"id": "c", "epoch": 1, "error": "boom", "scores": {"s": "C"}},
            ]
        )
    )
    assert P4.ib_scores(tmp_path, "ins") == {"a": True, "b": False, "c": None}


def test_ib_gate_banks_at_one_pass_and_runs_the_rest_only_on_a_survivor(tmp_path, monkeypatch):
    # most probable failure: a scaffold with 1 pass of 30 survives (the bank is 0 or 1), or a banked
    # scaffold still runs its other 30 and the original split
    monkeypatch.setattr(P4, "FORECAST_H", {**P4.FORECAST_H, "ib_gate": 0.0})
    calls = []

    def fake_scores(out, iid):
        calls.append(iid)
        n = {"tools": 1, "minimal": 5}[iid.split("_")[2]]
        tasks = [f"{iid}_t{i}" for i in range(15)]  # each split its own task ids
        return {t: (j < n and iid.endswith("oneoff_base")) for j, t in enumerate(tasks)}

    monkeypatch.setattr(P4, "ib_scores", fake_scores)
    monkeypatch.setattr(P4, "wait_for", lambda out, ids, dry, h: [])
    (tmp_path / "vm_results").mkdir()
    rec = P4.stage_ib_gate(tmp_path, dry=True)
    assert rec["scaffolds"]["tools"]["banked"] is True and "all_60" not in rec["scaffolds"]["tools"]
    assert (
        rec["scaffolds"]["minimal"]["banked"] is False and "all_60" in rec["scaffolds"]["minimal"]
    )
    assert not any("tools_original" in c or (c.startswith("ib_g2_tools")) for c in calls)
    assert P4.IB_BANK_AT == 1


def test_bounds_grid_swap_restores_the_conduct_grid():
    # most probable failure: the bounds grid's constants leak into a later conduct-grid read
    import gpc_conduct as GC

    before = (GC.SETTINGS, GC.GRID, GC.brief, GC.check, P4.P1.c_conduct)
    with P4.bounds_grid(dry=True):
        assert set(GC.SETTINGS) == {"work", "sim"} and GC.GRID.name == "bounds_grid_v1.json"
        assert P4.P1.c_conduct().startswith("Do not take actions the people you work for")
    assert (GC.SETTINGS, GC.GRID, GC.brief, GC.check, P4.P1.c_conduct) == before


def test_usage_records_the_hosts_peak_memory_and_load(tmp_path, monkeypatch):
    # most probable failure: the sampler reads a field wrongly and records 0 for every stage, so the
    # rehearsal's capacity read (the lead's item 2) is empty
    import time

    (tmp_path / "meminfo").write_text("MemTotal:       32000000 kB\nMemAvailable:   20000000 kB\n")
    (tmp_path / "loadavg").write_text("7.50 3.00 1.00 2/300 999\n")
    monkeypatch.setattr(V.Usage, "MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(V.Usage, "LOADAVG", tmp_path / "loadavg")
    with V.Usage(every_s=0.02) as u:
        time.sleep(0.15)
    rec = u.record()
    assert rec["samples"] >= 1 and rec["peak_load1"] == 7.5
    assert rec["peak_mem_used_gb"] == round(12000000 / 2**20, 2)
    assert rec["mem_total_gb"] == round(32000000 / 2**20, 2)


def _odcv_results(out, ins, errors=0, missing=0):
    for n, (sc, var, rid) in enumerate(ins["runs"]):
        if n < missing:
            continue
        d = out / "vm_results" / ins["id"] / rid
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(
            json.dumps(
                {
                    "run_id": rid,
                    "scenario": sc,
                    "variant": var,
                    "arm": ins["arm"],
                    "tool_timeouts": n % 2,
                    "container_error": n < missing + errors,
                }
            )
        )


def test_odcv_quality_counts_a_missing_run_as_an_error(tmp_path):
    # most probable failure: a run that never arrived is left out of the share, so a gate that lost
    # runs reads clean
    ins = P4.odcv_instruction("base", "base", dry=True)  # 8 runs in a dry session
    _odcv_results(tmp_path, ins, errors=0, missing=1)
    q = P4.odcv_quality(tmp_path, ins)
    assert q["container_errors"] == 1 and q["error_share"] == 1 / 8 and q["void"]
    assert q["tool_timeout_share"] is not None


@pytest.mark.parametrize(
    "second,fits_rerun,want",
    [
        (None, True, "odcv_base"),  # clean first time: counted as run
        (0, True, "odcv_base_c4"),  # void, rerun clean at concurrency 4: the rerun counts
        (2, True, None),  # void twice: ODCV-Bench's stages are skipped
        (None, False, None),  # void and no time for the rerun: skipped
    ],
)
def test_a_bad_odcv_gate_is_void_reruns_once_or_is_skipped(
    tmp_path, monkeypatch, second, fits_rerun, want
):
    # most probable failure: a gate with many container errors is counted or banked (the lead's
    # item 3: neither), or reruns more than once
    first = P4.odcv_instruction("base", "base", dry=True)
    _odcv_results(tmp_path, first, errors=0 if (second is None and fits_rerun) else 2)
    plans = []
    monkeypatch.setattr(
        P4, "plan_set", lambda out, ins, hold=False: plans.append([i["id"] for i in ins])
    )
    monkeypatch.setattr(P4, "wait_for", lambda *a, **k: [])
    monkeypatch.setattr(P4, "fits", lambda out, item: fits_rerun)

    def wait_rerun(out, ids, dry, max_h):
        if "odcv_base_c4" in ids:
            _odcv_results(tmp_path, first | {"id": "odcv_base_c4"}, errors=second or 0)
        return []

    monkeypatch.setattr(P4, "wait_for", wait_rerun)
    rec = P4.stage_odcv_gate(tmp_path, dry=True)
    assert rec["instruction"] == want and rec["void"] is (want is None)
    assert sum("odcv_base_c4" in p for p in plans) <= 1  # one rerun at most
    if want is None:
        assert "skipped" in rec["reason"]
        j = P4.stage_judge(tmp_path, "odcv_base", dry=True)
        assert j["void"] and not j["counts_rate_bar"] and not j["banked"]
        assert P4._r(tmp_path, "odcv_gate")["void"]  # noqa: SLF001
    if want == "odcv_base_c4":
        assert rec["rerun"]["concurrency"] == 4


def test_the_failure_branch_skips_the_vm_stages(tmp_path):
    # most probable failure: a lost VM leaves the pod waiting on stages that can never deliver
    P4._w(tmp_path, "vm", {"lost": True})  # noqa: SLF001
    (tmp_path / "vm_results").mkdir()
    assert P4.stage_odcv_gate(tmp_path, dry=True) == {}
    assert P4.stage_ib_gate(tmp_path, dry=True)["both_banked"] is True
    assert P4.stage_arms(tmp_path, dry=True) == {}


def test_pod_runner_order_and_inputs():
    # most probable failure: the host gets a stage while the server is down, the judge runs before
    # its weights arrive, or the host is never told to stop
    steps = [
        "--stage prep",
        "--part up",
        "--stage serve_check",
        "--stage vm_wait",
        "item odcv_gate",
        "item ib_gate",
        "--part down",
        "wait $JDL",
        "--part odcv_base",
        "item bounds",
        "item arms",
        "item stage3",
        "item stage4",
        "--part final",
        "--stage stop",
    ]
    pos = [RUNNER.index(s) for s in steps]
    assert pos == sorted(pos)
    assert 'permitopen=\\"127.0.0.1:8000\\",permitopen=\\"127.0.0.1:8001\\"' in RUNNER
    assert "-wo $OUT/vm_results" in RUNNER and "6924dafb3a3f84fc320fbe9f0a08203d67d1d488" in RUNNER
    inp = (GPC / "runpod/gpc4_inputs.sh").read_text()
    for f in (
        "p_ctx/1.94053e+06/adapter_model.safetensors",
        "vm_keys/forward_key.pub",
        "vm_keys/results_key.pub",
        "eval_8gram_hashes_v2.npy",
        "adapters/c1/612732",
    ):
        assert f in inp, f
    card = (GPC / "runpod/launch_gpc4_pod.sh").read_text()
    assert "remote_gpc4.sh" in card and "outputs/gpc4_v1" in card and 'TNR_FALLBACK=""' in card


@pytest.mark.slow
def test_dry_session_chains_gpc1_into_every_gpc4_stage(tmp_path):
    # most probable failure: a GPC-4 stage breaks on GPC-1's real output shapes or on the VM's
    # results (the dry run chains GPC-1's stages into gpc1_v1, then every GPC-4 stage with the
    # stub as the server and a stand-in VM answering each instruction)
    import os
    import subprocess

    import gpc1_read as RD

    from deepsteer.datasets.machiavelli import fetch_machiavelli

    am = Path(os.environ.get("GPC_AM_DIR", "/workspace/agentic-misalignment"))
    try:
        fetch_machiavelli(verify_bundle=False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no local MACHIAVELLI bundle: {e!r}")
    if not RD.LABELS.exists() or not (am / "scripts" / "generate_prompts.py").exists():
        pytest.skip("the CST label file or the agentic-misalignment clone is not here")
    py = sys.executable
    g1p, g4p = str(GPC / "scripts/gpc1_pod.py"), str(GPC / "scripts/gpc4_pod.py")
    tp = str(ROOT / "papers/cst_constitutional_steering/scripts/cst_teacher_pod.py")
    env = {**os.environ, "GPC_AM_DIR": str(am)}

    def run(script, out, *a):
        r = subprocess.run(
            [py, script, *a, "--out", str(out)], capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, (a, r.stderr[-2500:])

    o1, o4 = tmp_path / "gpc1_v1", tmp_path / "gpc4_v1"
    run(g1p, o1, "--stage", "smoke_train", "--dry-run")
    run(tp, o1, "--stage", "smoke_serve", "--dry-run")
    run(g1p, o1, "--stage", "conduct", "--dry-run")
    run(g1p, o1, "--stage", "train", "--arm", "c1", "--dry-run")
    try:
        for a in (
            ["prep"],
            ["serve", "--part", "up"],
            ["serve_check"],
            ["vm_wait"],
            ["odcv_gate"],
            ["ib_gate"],
            ["hold"],
            ["serve", "--part", "down"],
            ["judge", "--part", "odcv_base"],
            ["bounds"],
            ["serve", "--part", "up"],
            ["arms"],
            ["hold"],
            ["serve", "--part", "down"],
            ["stage3"],
            ["stage4"],
            ["judge", "--part", "final"],
            ["stop"],
        ):
            run(g4p, o4, "--stage", *a, "--dry-run")
    finally:
        subprocess.run(
            [py, g4p, "--stage", "serve", "--part", "down", "--out", str(o4), "--dry-run"],
            capture_output=True,
        )
    assert json.loads((o4 / "vm.json").read_text())["lost"] is False
    assert json.loads((o4 / "odcv_gate.json").read_text())["n"] == 8
    ib = json.loads((o4 / "ib_gate.json").read_text())
    assert set(ib["scaffolds"]) == {"tools", "minimal"}
    assert (o4 / "bounds" / "conduct" / "slots.json").exists()
    assert json.loads((o4 / "gate3.json").read_text())["reads_on"] in (
        "bounds_pilot",
        "impossiblebench",
    )
    assert (o4 / "g5" / "blackmail.jsonl").exists() and (o4 / "stage4.json").exists()
    plan = json.loads((o4 / "plan.json").read_text())
    assert plan["hold"] is False
