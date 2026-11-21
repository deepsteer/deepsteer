#!/usr/bin/env python3
"""GPC-4's tool-parser probe (GPC_PLAN.md §11.18, widened in §11.19): a round-trip battery of tool calls,
each rendered by the subject's pinned chat template and read back by vLLM's tool-call parser under the name
the server is started with; every call's name and arguments must come back exactly, types included.

The schemas are every tool both harnesses expose (``data/gpc4_tool_schemas_v1.json``, written by
``z_gpc4_tool_schemas.py``): ODCV-Bench's ``bash`` and ``task_complete``; ImpossibleBench's tools scaffold's
``bash``, ``python``, ``text_editor``, ``think`` and ``submit`` (Inspect's own schemas). Each case's request
carries its harness's whole tool list, as the server sees it. The payloads are the lead's list: shell
redirects, pipes, a heredoc and ``<``/``>``; multi-line Python with both quotes, brackets and a
triple-quoted string; a JSON string, non-ASCII text, an argument of about 4,000 characters; text_editor's
integer and integer-array arguments and an omitted optional one; two calls in one turn; text before a call.

Launch 5 started vLLM with ``hermes``, which parsed none of Qwen3.8's XML calls. The battery runs without
the model's weights: on the pod in the gate tests (the pod's own vLLM and snapshot) and locally in the
standing step (the pinned vLLM wheel's Python sources, ``--vllm-src``).

    python gpc4_parser_probe.py --parser qwen3_coder [--vllm-src DIR] [--snapshot DIR]

Prints one JSON line: each case's result, and the ones that did not round-trip. Exit 0: every case
round-tripped; 1: at least one did not; 2: vLLM or the pinned snapshot is not available here (a skip off
the pod).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMAS = ROOT / "papers/gpc_goal_pressure_conduct/data/gpc4_tool_schemas_v1.json"

LONG = "\n".join(f"line {i:03d}: total=$((a + {i})); echo \"<row id='{i}'>${{name}}</row>\" >> out.xml"
                 for i in range(58))  # about 4,000 characters of shell, quotes and markup
PY_MULTI = '''def parse(rows: list[dict]) -> dict[str, int]:
    """Count rows by "kind", skipping the 'draft' ones.

    Example: parse([{'kind': "a"}]) == {"a": 1}
    """
    out = {}
    for r in rows:
        if r.get('kind') != "draft" and (r["n"] > 0 or r["n"] < -1):
            out[r["kind"]] = out.get(r["kind"], 0) + 1
    return out

print(parse([{"kind": 'x', "n": 2}, {'kind': "draft", 'n': [1, 2][0]}]))
'''
JSON_ARG = json.dumps({"status": "done", "items": [1, 2.5, None, True], "note": 'a "quoted" word',
                       "nested": {"k": "v"}})
NON_ASCII = "Résumé — naïve café; 完成しました ✓; Ελληνικά; emoji 🚀 and a tab\there"


def cases() -> list[dict]:
    """(harness, text before the calls, calls): every tool, and every payload kind the lead listed."""
    c = lambda name, **args: {"name": name, "arguments": args}  # noqa: E731
    return [
        {"id": "odcv_bash_pipes_redirects", "harness": "odcv_bench", "calls": [c(
            "bash", command="cat data.csv | grep -v '^#' | sort -u > out.txt 2>&1 && wc -l < out.txt")]},
        {"id": "odcv_bash_heredoc", "harness": "odcv_bench", "calls": [c(
            "bash", command="cat > /app/report.md <<'EOF'\n# Report\nvalue: $X and \"quoted\"\nEOF\n"
                            "cat /app/report.md")]},
        {"id": "odcv_bash_lt_gt_markup", "harness": "odcv_bench", "calls": [c(
            "bash", command='if [ "$a" -lt 5 ] && [[ "$b" > "c" ]]; then echo "<tag a=\'1\'>x</tag>" '
                            "> f.html; fi; diff <(sort a) <(sort b)")]},
        {"id": "odcv_bash_4000_chars", "harness": "odcv_bench", "calls": [c("bash", command=LONG)]},
        {"id": "odcv_task_complete_json_string", "harness": "odcv_bench",
         "calls": [c("task_complete", reason=JSON_ARG)]},
        {"id": "odcv_task_complete_non_ascii", "harness": "odcv_bench",
         "calls": [c("task_complete", reason=NON_ASCII)]},
        {"id": "odcv_two_calls_text_before", "harness": "odcv_bench",
         "text": "I'll check the files first, then finish.",
         "calls": [c("bash", command="ls -la /app && cat /app/config.json"),
                   c("task_complete", reason="Checked the configuration.")]},
        {"id": "ib_python_multiline_quotes", "harness": "impossiblebench_tools",
         "calls": [c("python", code=PY_MULTI)]},
        {"id": "ib_bash_pipes", "harness": "impossiblebench_tools",
         "calls": [c("bash", command="python test.py 2>&1 | tail -n 20 > /tmp/log.txt; echo $?")]},
        {"id": "ib_text_editor_create", "harness": "impossiblebench_tools", "calls": [c(
            "text_editor", command="create", path="/workspace/func.py", file_text=PY_MULTI)]},
        {"id": "ib_text_editor_view_range_int_array", "harness": "impossiblebench_tools", "calls": [c(
            "text_editor", command="view", path="/workspace/func.py", view_range=[10, 20])]},
        {"id": "ib_text_editor_view_optional_omitted", "harness": "impossiblebench_tools",
         "calls": [c("text_editor", command="view", path="/workspace/test.py")]},
        {"id": "ib_text_editor_insert_int", "harness": "impossiblebench_tools", "calls": [c(
            "text_editor", command="insert", path="/workspace/func.py", insert_line=5,
            insert_text="    if n < 0 or n > 10:\n        raise ValueError(f\"bad n: {n!r}\")\n")]},
        {"id": "ib_text_editor_str_replace", "harness": "impossiblebench_tools", "calls": [c(
            "text_editor", command="str_replace", path="/workspace/func.py",
            old_str='    return a + b\n', new_str="    return (a + b) if a >= 0 else 'neg'\n")]},
        {"id": "ib_think_non_ascii_json", "harness": "impossiblebench_tools",
         "calls": [c("think", thought=NON_ASCII + "\n" + JSON_ARG)]},
        {"id": "ib_submit", "harness": "impossiblebench_tools", "calls": [c("submit", answer="DONE")]},
        {"id": "ib_two_calls_text_before", "harness": "impossiblebench_tools",
         "text": "The test fails on negative inputs. I'll look at the code, then run the tests.",
         "calls": [c("text_editor", command="view", path="/workspace/func.py", view_range=[1, -1]),
                   c("bash", command="cd /workspace && python test.py")]},
    ]


def snapshot() -> str:
    """The subject's pinned snapshot (its tokenizer and chat template only), from the local cache."""
    import yaml
    from huggingface_hub import snapshot_download

    s = yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["st"]["qwen38_27b"]
    return snapshot_download(s["repo"], revision=s["revision"], local_files_only=True,
                             allow_patterns=["tokenizer*", "chat_template.jinja", "vocab.json",
                                             "merges.txt", "config.json"])


def tools_of(harness: str) -> list[dict]:
    return json.loads(SCHEMAS.read_text())[harness]["tools"]


def rendered_turn(tok, case: dict, tools: list[dict]) -> str:
    """The assistant turn the template writes for the case's calls: what the subject emits when it calls.
    The reasoning parser takes the trace first, so the tool parser sees what follows ``</think>``."""
    msgs = [{"role": "user", "content": "Do the task."},
            {"role": "assistant", "content": case.get("text", ""), "tool_calls": [
                {"type": "function", "id": f"c{n}", "function": {"name": k["name"],
                                                                 "arguments": k["arguments"]}}
                for n, k in enumerate(case["calls"])]}]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    turn = text.rsplit("<|im_start|>assistant\n", 1)[1].split("<|im_end|>")[0]
    return turn.split("</think>", 1)[-1].lstrip()


def canon(x) -> str:
    """Exact, typed comparison: 5 and 5.0 and "5" differ, as do True and 1."""
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


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
    results = []
    for case in cases():
        tools = tools_of(case["harness"])
        turn = rendered_turn(tok, case, tools)
        req = ChatCompletionRequest(model="base", messages=[{"role": "user", "content": "x"}],
                                    tools=tools, tool_choice="auto")
        parser = ToolParserManager.get_tool_parser(a.parser)(tok, req.tools)
        rec = {"id": case["id"], "harness": case["harness"], "chars": len(turn)}
        try:
            r = parser.extract_tool_calls(turn, req)
            got = [{"name": k.function.name, "arguments": json.loads(k.function.arguments)}
                   for k in r.tool_calls]
        except Exception as e:  # noqa: BLE001  (a parser or argument error is this case's result)
            rec |= {"passes": False, "error": repr(e)[:300]}
            results.append(rec)
            continue
        want = case["calls"]
        rec["passes"] = bool(r.tools_called) and canon(got) == canon(want)
        if case.get("text"):  # the text before the calls stays the message's content
            rec["text_kept"] = (r.content or "").strip() == case["text"]
            rec["passes"] = rec["passes"] and rec["text_kept"]
        if not rec["passes"]:
            rec |= {"want": canon(want)[:600], "got": canon(got)[:600], "rendered": turn[:600]}
        results.append(rec)
    failed = [x["id"] for x in results if not x["passes"]]
    out = {"parser": a.parser, "cases": len(results), "passed": len(results) - len(failed),
           "failed": failed, "passes": not failed,
           "tools": sorted({k["name"] + "@" + c["harness"] for c in cases() for k in c["calls"]}),
           "results": results}
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(0 if out["passes"] else 1)


if __name__ == "__main__":
    main()
