#!/usr/bin/env python3
"""GPC-4's tool-parser probe (GPC_PLAN.md §11.18): the subject's own tool call, rendered by its pinned chat
template, read by vLLM's tool-call parser under the name the server is started with.

Launch 5 started vLLM with ``--tool-call-parser hermes``, which reads JSON inside ``<tool_call>``; Qwen3.8's
template writes XML (``<function=bash><parameter=command>``), so every call failed to parse and the serve
check stopped the session. This probe runs that read without the model's weights: on the pod in the
gate tests (the pod's own vLLM and snapshot) and locally in the standing step (the pinned vLLM wheel's
Python sources, ``--vllm-src``).

    python gpc4_parser_probe.py --parser qwen3_coder [--vllm-src DIR] [--snapshot DIR]

Prints one JSON line. Exit 0: the call parsed as the rendered one; 1: it did not; 2: vLLM or the pinned
snapshot is not available here (a skip off the pod).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
                                         "parameters": {"type": "object", "properties": {
                                             "command": {"type": "string"}},
                                             "required": ["command"]}}}
CALL = {"command": "ls /tmp/dir0"}


def snapshot() -> str:
    """The subject's pinned snapshot (its tokenizer and chat template only), from the local cache."""
    import yaml
    from huggingface_hub import snapshot_download

    s = yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["st"]["qwen38_27b"]
    return snapshot_download(s["repo"], revision=s["revision"], local_files_only=True,
                             allow_patterns=["tokenizer*", "chat_template.jinja", "vocab.json",
                                             "merges.txt", "config.json"])


def rendered_turn(tok) -> str:
    """The assistant turn the template writes for one bash call: what the subject emits when it calls."""
    msgs = [{"role": "user", "content": "List the files in /tmp/dir0 using the bash tool."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "id": "c0", "function": {"name": "bash", "arguments": CALL}}]}]
    text = tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False)
    turn = text.rsplit("<|im_start|>assistant\n", 1)[1].split("<|im_end|>")[0]
    return turn.split("</think>", 1)[-1].lstrip()  # the reasoning parser takes the trace first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", required=True)
    ap.add_argument("--vllm-src")
    ap.add_argument("--snapshot")
    a = ap.parse_args()
    if a.vllm_src:
        sys.path.insert(0, a.vllm_src)
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.tool_parsers import ToolParserManager
    except ImportError as e:
        print(json.dumps({"skip": f"vLLM is not importable here ({e!r})"[:300]}))
        sys.exit(2)
    try:
        snap = a.snapshot or snapshot()
    except Exception as e:  # noqa: BLE001  (not in this machine's cache)
        print(json.dumps({"skip": f"the pinned snapshot is not cached here ({e!r})"[:300]}))
        sys.exit(2)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap)
    turn = rendered_turn(tok)
    req = ChatCompletionRequest(model="base", messages=[{"role": "user", "content": "x"}],
                                tools=[TOOL], tool_choice="auto")
    parser = ToolParserManager.get_tool_parser(a.parser)(tok, req.tools)
    r = parser.extract_tool_calls(turn, req)
    calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)}
             for c in r.tool_calls]
    rec = {"parser": a.parser, "rendered": turn[:300], "tools_called": r.tools_called, "calls": calls,
           "passes": bool(r.tools_called) and calls == [{"name": "bash", "arguments": CALL}]}
    print(json.dumps(rec))
    sys.exit(0 if rec["passes"] else 1)


if __name__ == "__main__":
    main()
