#!/usr/bin/env python3
"""The tool schemas GPC-4's two harnesses send the subject's server (GPC_PLAN.md §11.19), written to
``data/gpc4_tool_schemas_v1.json`` for the parser probe (``gpc4_parser_probe.py``), which runs where neither
harness is installed (the pod's vLLM venv, the standing step's).

- ODCV-Bench: ``bash`` and ``task_complete``, read from ``mission_executor/agent_main.py`` at the pin.
- ImpossibleBench's tools scaffold: Inspect's ``bash``, ``python``, ``text_editor`` and ``think`` as
  ``swe_bench_style_humaneval_solver`` builds them (timeout 60; think's own description), plus the ``submit``
  tool ``basic_agent`` adds, each converted as Inspect's OpenAI provider sends it
  (``openai_chat_tool_param``).

Run with the interpreter holding Inspect 0.3.291 (the host's pin):

    GPC4_INSPECT_PY ... z_gpc4_tool_schemas.py --odcv <ODCV-Bench clone> --ib <impossiblebench src dir>
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

# module level: Inspect resolves the submit tool's annotations (strings here) from this module's globals
import inspect_ai
from inspect_ai.model._call_tools import get_tools_info
from inspect_ai.model._openai import openai_chat_tool_param
from inspect_ai.solver._basic_agent import DEFAULT_SUBMIT_DESCRIPTION, DEFAULT_SUBMIT_NAME
from inspect_ai.tool import Tool, ToolResult, bash, python, text_editor, think, tool
from inspect_ai.tool._tool_with import tool_with

GPC = Path(__file__).resolve().parents[1]
OUT = GPC / "data" / "gpc4_tool_schemas_v1.json"
ODCV_PIN = "6924dafb3a3f84fc320fbe9f0a08203d67d1d488"
IB_PIN = "d4f9a149db2b9cec0fed7bab01d91c07ee10aca2"
INSPECT = "0.3.291"


def odcv_tools(clone: Path) -> list[dict]:
    """The ``tools = [...]`` literal in upstream's agent loop, parsed (never executed)."""
    src = (clone / "mission_executor" / "agent_main.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tools"
                                                 for t in node.targets)):
            return ast.literal_eval(node.value)
    raise SystemExit("ODCV-Bench's tools literal not found")


def think_description(ib_src: Path) -> str:
    """think's description as the tools scaffold passes it (dedented, as the scaffold does)."""
    import textwrap

    tree = ast.parse((ib_src / "livecodebench_agent_full.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "think":
            arg = node.args[0]
            return textwrap.dedent(arg.args[0].value) if isinstance(arg, ast.Call) else arg.value
    raise SystemExit("the tools scaffold's think() call not found")


def inspect_tools(ib_src: Path) -> list[dict]:
    assert inspect_ai.__version__ == INSPECT, inspect_ai.__version__

    @tool(max_output=0)
    def submit() -> Tool:  # basic_agent's own, as 0.3.291 defines it
        async def execute(answer: str) -> ToolResult:
            """Submit an answer for evaluation.

            Args:
              answer (str): Submitted answer
            """
            return answer

        return execute

    tools = [bash(timeout=60), python(timeout=60), text_editor(timeout=60),
             think(think_description(ib_src)),
             tool_with(submit(), DEFAULT_SUBMIT_NAME, DEFAULT_SUBMIT_DESCRIPTION)]
    return [json.loads(json.dumps(openai_chat_tool_param(t), default=dict))
            for t in get_tools_info(tools)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odcv", type=Path, required=True)
    ap.add_argument("--ib", type=Path, required=True, help="impossiblebench's src/impossiblebench")
    a = ap.parse_args()
    head = lambda d: subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True,  # noqa: E731
                                    text=True).stdout.strip()
    assert head(a.odcv) == ODCV_PIN, head(a.odcv)
    assert head(a.ib) == IB_PIN, head(a.ib)
    rec = {"rule": "GPC_PLAN.md §11.19 (every tool schema both harnesses expose)",
           "odcv_bench": {"pin": ODCV_PIN, "tools": odcv_tools(a.odcv)},
           "impossiblebench_tools": {"pin": IB_PIN, "inspect": INSPECT, "tools": inspect_tools(a.ib)}}
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    names = [t["function"]["name"] for k in ("odcv_bench", "impossiblebench_tools") for t in rec[k]["tools"]]
    print(f">> wrote {OUT.relative_to(GPC.parents[1])}: {names}")
    sys.exit(0)


if __name__ == "__main__":
    main()
