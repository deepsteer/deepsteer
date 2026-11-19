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
        "no passed rehearsal at",
        "GPC-4 runs on Thunder with no fallback",
        "revoke-security-group-egress",
        "authorize-security-group-egress",
        "Key=gpc4-session",
        "cmd_teardown",
        "cmd_reattach",
        "is not on origin/main (push it first)",
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


def test_rehearsal_plan_covers_both_scaffolds_and_one_scenario():
    plan = json.loads((GPC / "vm/rehearsal_plan.json").read_text())["instructions"]
    kinds = [(i["kind"], i.get("scaffold")) for i in plan]
    assert ("odcv", None) in kinds and ("ib", "minimal") in kinds and ("ib", "tools") in kinds
    assert len({i["id"] for i in plan}) == len(plan)


@pytest.mark.parametrize("cmd", ["rehearse", "verify", "launch", "reattach", "teardown"])
def test_launcher_dispatches_each_command(cmd):
    assert f"  {cmd}) cmd_{cmd} ;;" in LAUNCHER
