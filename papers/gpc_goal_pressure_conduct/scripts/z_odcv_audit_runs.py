#!/usr/bin/env python3
# GPC_PLAN.md §11.13. Usage: z_odcv_audit_runs.py <ODCV-Bench clone at 6924daf> <out.json>
"""For each scenario and each tool its prompt names, every call in upstream's shipped model runs (other
models' runs, scanned by script; no model reads them): did any call produce output without an
infrastructure error? Zero GPU."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ODCV = Path(sys.argv[1])
OUT = Path(sys.argv[2])
LOGS = ODCV / "existing_results" / "current" / "agent_logs"
INFRA = re.compile(r"No such file or directory|command not found|Permission denied|Traceback|"
                   r"cannot execute|not a directory|Is a directory|syntax error|unbound variable", re.I)
TOOL = re.compile(r"\[Tool\]: (\w+) with (\{.*)$")
OUTL = re.compile(r"\[out\]: (\{.*)$")


def calls(log: Path):
    """(command text, stdout, stderr, returncode) for each bash call in one run's docker log."""
    lines = [re.sub(r"^\S+-1\s+\|\s?", "", x) for x in log.read_text(errors="ignore").splitlines()]
    cur = None
    for x in lines:
        m = TOOL.search(x)
        if m:
            try:
                args = ast.literal_eval(m.group(2))
            except Exception:  # noqa: BLE001
                args = {"command": m.group(2)}
            cur = str(args.get("command", args)) if isinstance(args, dict) else str(args)
            continue
        m = OUTL.search(x)
        if m and cur is not None:
            try:
                o = ast.literal_eval(m.group(1))
            except Exception:  # noqa: BLE001
                o = {"stdout": m.group(1), "stderr": "", "returncode": None}
            yield cur, str(o.get("stdout", "")), str(o.get("stderr", "")), o.get("returncode")
            cur = None


rows = []
for variant in ("mandated", "incentivized"):
    for sd in sorted((ODCV / f"{variant}_scenarios").iterdir()):
        p = json.loads((sd / "prompt.json").read_text())
        text = p["system_prompt"] + "\n" + p["user_prompt"]
        tools = sorted(set(re.findall(r"/usr/local/bin/[\w.\-]+\.(?:sh|py)", text)))
        per = {t: {"calls": 0, "ok": 0, "infra": 0, "runs_with_call": 0, "examples": []} for t in tools}
        nruns = 0
        for md in sorted(LOGS.glob(f"*-{variant}")):
            log = md / "experiments" / sd.name / "docker_output.log"
            if not log.exists():
                continue
            nruns += 1
            seen = set()
            for cmd, so, se, rc in calls(log):
                for t in tools:
                    name = Path(t).name
                    if name in cmd and not re.search(rf"(cat|head|less|more|sed -n|grep)[^|;]*{re.escape(name)}", cmd):
                        d = per[t]
                        d["calls"] += 1
                        seen.add(t)
                        bad = bool(INFRA.search(se) or INFRA.search(so))
                        d["infra"] += bad
                        d["ok"] += (not bad) and bool(so.strip() or se.strip())
                        if bad and len(d["examples"]) < 3:
                            d["examples"].append({"model": md.name, "cmd": cmd[:160], "err": (se or so)[:200]})
            for t in seen:
                per[t]["runs_with_call"] += 1
        rows.append({"variant": variant, "scenario": sd.name, "runs": nruns, "tools": per})
OUT.write_text(json.dumps(rows, indent=1))
for r in rows:
    for t, d in r["tools"].items():
        if d["ok"] == 0 or d["infra"] > d["ok"]:
            print(f"{r['variant'][:4]} {r['scenario']} {Path(t).name}: calls={d['calls']} ok={d['ok']} infra={d['infra']} "
                  f"runs_with_call={d['runs_with_call']}/{r['runs']} ex={d['examples'][:1]}")
print(len(rows), "scenario-variants;", sum(len(r["tools"]) for r in rows), "named tools")
