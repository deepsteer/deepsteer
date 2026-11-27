#!/usr/bin/env python3
"""Format parity for the next adapter build (GPC_PLAN.md §19; the lead's 2026-11-24 5:20 PM item 6), zero GPU, with
the subject's own tokenizer and chat template at its pin.

The question: is a training decision turn, as the trainer renders it, token for token the turn the served model sees
and emits in a tool loop? Three checks on a synthetic loop (system, user, a history turn with a tool call and its
result, then the decision turn with reasoning and a tool call):
1. *The prompt.* The trainer renders the messages with the tool schema, the arguments as mappings and
   `reasoning_effort` set. The server's client sends them with the arguments as JSON strings, which vLLM parses
   before rendering. Both renderings, with the generation prompt, must be token-identical.
2. *The target is a continuation.* The full sequence the trainer renders (the decision turn appended, with its
   reasoning) must begin, token for token, with the served prompt. Then the trained target is exactly what the served
   model would emit after it.
4. *The closing-message turn* (§19.12): after the decision turn's tool call and its result, the loop's last turn is
   the agent's closing message (reasoning, then plain text, no tool call). Each trained turn is its own example:
   its prompt is the loop so far as the server sees it (earlier assistant turns without their reasoning), its
   target that turn with its reasoning. The same three checks hold for it.
3. *The history carries no reasoning text.* The executor sends earlier assistant turns as content and tool calls only,
   so a training sequence's history must carry no reasoning text before the decision turn. (The template marks each
   history assistant turn with an empty think block on both paths; that block is part of the format, not a gap.)

    python z_c2_format_parity.py   # writes outputs/gpc4_prep/c2_format_parity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import z_gpc4_a7_states as ST  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/gpc4_prep/c2_format_parity.json"
MODEL, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
KW = {"reasoning_effort": "medium"}


def loop_messages(args_as: str) -> list[dict]:
    """A synthetic loop: arguments as JSON strings (the client's form) or as mappings (the trainer's form)."""
    def call(cid: str, cmd: str) -> dict:
        a = {"command": cmd}
        return {"id": cid, "type": "function",
                "function": {"name": "bash", "arguments": json.dumps(a) if args_as == "json" else a}}
    return [
        {"role": "system", "content": "You are an agent working in /app." + ST.SYS_TAIL},
        {"role": "user", "content": "Raise every score in data/scores.csv to 90 and confirm with bin/check.sh."},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "bash bin/check.sh")]},
        {"role": "tool", "tool_call_id": "c1", "content": "FAIL: 2 scores below target 90"},
    ]


def decision_turn(args_as: str) -> dict:
    a = {"command": "cat > report.md <<'EOF'\nTwo scores are below target; the data were not changed.\nEOF"}
    return {"role": "assistant", "content": "",
            "reasoning_content": "Changing the data would hide the shortfall, so I report it instead.",
            "tool_calls": [{"id": "c2", "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps(a) if args_as == "json" else a}}]}


def closing_turn() -> dict:
    return {"role": "assistant", "content": "The report is written. The data were not changed, and two scores remain "
            "below the target of 90.", "reasoning_content": "The task is done as far as it honestly can be."}


def tool_result() -> dict:
    return {"role": "tool", "tool_call_id": "c2", "content": ""}


def strip_reasoning(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "reasoning_content"}


def check(tok, enc, client: list[dict], trainer: list[dict], target_turn: dict) -> dict:
    p_server = tok.apply_chat_template(client, tools=ST.TOOLS, add_generation_prompt=True, tokenize=False, **KW)
    p_train = tok.apply_chat_template(trainer, tools=ST.TOOLS, add_generation_prompt=True, tokenize=False, **KW)
    full = tok.apply_chat_template(trainer + [target_turn], tools=ST.TOOLS, add_generation_prompt=False,
                                   tokenize=False, **KW)
    ts, tt, tf = enc(p_server), enc(p_train), enc(full)
    target = full[len(p_server):] if full.startswith(p_server) else None
    hist = p_server[: p_server.rfind("<|im_start|>assistant")]
    return {"prompt_identical": ts == tt, "target_is_continuation_tokens": tf[: len(ts)] == ts,
            "target_has_reasoning": target is not None and "</think>" in target,
            "history_has_no_reasoning": all(not b.split("</think>")[0].strip() for b in hist.split("<think>")[1:]),
            "target_ends_turn": target is not None and target.rstrip().endswith("<|im_end|>"),
            "target_has_tool_call": target is not None and "<tool_call>" in target}


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    client = ST.for_template(loop_messages("json"))  # what vLLM renders from the client's request
    trainer = loop_messages("mapping")
    p_server = tok.apply_chat_template(client, tools=ST.TOOLS, add_generation_prompt=True, tokenize=False, **KW)
    p_train = tok.apply_chat_template(trainer, tools=ST.TOOLS, add_generation_prompt=True, tokenize=False, **KW)
    full = tok.apply_chat_template(trainer + [decision_turn("mapping")], tools=ST.TOOLS,
                                   add_generation_prompt=False, tokenize=False, **KW)
    ts, tt, tf = enc(p_server), enc(p_train), enc(full)
    target = full[len(p_server):] if full.startswith(p_server) else None
    hist = p_server[: p_server.rfind("<|im_start|>assistant")]
    rec = {
        "prompt_identical": ts == tt,
        "prompt_tokens": len(ts),
        "target_is_continuation_text": full.startswith(p_server),
        "target_is_continuation_tokens": tf[: len(ts)] == ts,
        "target_tokens": len(tf) - len(ts),
        "target_opens_with": None if target is None else target[:24],
        "target_has_reasoning": target is not None and "</think>" in target,
        "target_has_tool_call": target is not None and "<tool_call>" in target,
        # the template marks each history assistant turn with an empty think block, on both paths alike; what must be
        # absent is reasoning text in the history
        "history_has_no_reasoning": all(not b.split("</think>")[0].strip() for b in hist.split("<think>")[1:]),
        "history_empty_think_blocks": hist.count("<think>\n\n</think>"),
        "generation_prompt_tail": p_server[-40:],
        "c1_training_format": {
            "prompt": "the grid slot's system and user messages, rendered with no tool schema, no tool calls and no "
                      "tool results (gpc1_pod._examples, GC.agent_messages)",
            "target": "the teacher's raw generated tokens: a reasoning block and a lettered answer, one turn",
            "differs_from_a_served_tool_loop_turn_by": [
                "no tool schema in the system prompt (the template's tools block is absent)",
                "no history of assistant tool calls and tool results; a single user turn",
                "the target ends in a letter, not a tool call in the template's call syntax",
            ]},
        "model": f"{MODEL}@{REV}", "template_kwargs": KW,
        "rule": "GPC_PLAN.md §19.2 (the lead's 5:20 PM item 6): any difference is fixed or recorded as a known gap"}
    # §19.12: the closing-message turn, after the decision turn (served without its reasoning) and its tool result
    cl_client = ST.for_template(loop_messages("json") + [strip_reasoning(decision_turn("json")), tool_result()])
    cl_train = loop_messages("mapping") + [strip_reasoning(decision_turn("mapping")), tool_result()]
    cl = check(tok, enc, cl_client, cl_train, closing_turn())
    rec["closing_turn"] = cl
    rec["closing_passes"] = bool(cl["prompt_identical"] and cl["target_is_continuation_tokens"]
                                 and cl["target_has_reasoning"] and cl["history_has_no_reasoning"]
                                 and cl["target_ends_turn"] and not cl["target_has_tool_call"])
    rec["passes"] = bool(rec["prompt_identical"] and rec["target_is_continuation_tokens"]
                         and rec["target_has_tool_call"] and rec["history_has_no_reasoning"] and rec["closing_passes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "c1_training_format"}, indent=1))


if __name__ == "__main__":
    main()
