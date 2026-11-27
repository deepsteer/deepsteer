#!/usr/bin/env python3
"""C2-JAB's activation read, its two positions (GPC_PLAN.md §19.8 (a); the lead's 2026-11-27 item 4), checked on the
subject's tokenizer and chat template alone, zero GPU.

At a decision point the served prompt ends with the generation prompt, `<|im_start|>assistant\\n<think>\\n` (the
template pre-fills the thinking block's opening; `z_c2_format_parity.py`). The two positions, as token indices in the
full sequence (the prompt, then the decision turn the model wrote):
- **before:** the last token before the thinking block opens: the token immediately before the `<think>` that opens
  the decision turn (the `\\n` that ends the assistant header);
- **after:** the last token of the thinking block before the action: the `</think>` that closes it.

**The position-validity check** (run on every decision point before its vectors are read, and here on a synthetic
loop): `<think>` and `</think>` are single tokens; the decision turn holds exactly one of each, the opener first;
the token at *before* decodes to "\\n" and is preceded by the assistant header; *after* is `</think>` and is followed
by the action (a tool call or text), not by end-of-turn; and the history turns' empty think blocks are never taken
(only the decision turn's, which is the last assistant turn).

    python z_c2_positions.py   # prints the check on the synthetic loop; writes outputs/gpc4_prep/c2_positions.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import z_c2_format_parity as FP  # noqa: E402
import z_gpc4_a7_states as ST  # noqa: E402

OUT = HERE.parent / "outputs/gpc4_prep/c2_positions.json"


def positions(ids: list[int], n_prompt: int, tok) -> dict:
    """The two positions in a full sequence whose first `n_prompt` tokens are the served prompt, and the check."""
    open_id, close_id = (tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>"))
    turn_open = max(i for i in range(n_prompt) if ids[i] == open_id)  # the decision turn's opener, pre-filled
    closes = [i for i in range(turn_open + 1, len(ids)) if ids[i] == close_id]
    opens_after = [i for i in range(turn_open + 1, len(ids)) if ids[i] == open_id]
    before, after = turn_open - 1, (closes[0] if closes else None)
    header = tok.decode(ids[max(0, before - 3): before + 1])
    nxt = tok.decode(ids[after + 1: after + 4]) if after is not None and after + 1 < len(ids) else ""
    check = {"single_tokens": all(len(tok(t, add_special_tokens=False)["input_ids"]) == 1 for t in ("<think>", "</think>")),
             "one_close_in_turn": len(closes) == 1, "no_second_open": not opens_after,
             "before_is_newline": tok.decode([ids[before]]) == "\n", "before_after_header": "assistant" in header,
             "after_is_close": after is not None and ids[after] == close_id,
             "action_follows": after is not None and "<|im_end|>" not in nxt[:12] and bool(nxt.strip()),
             # the history's think blocks all close before the decision turn's opener (none is taken)
             "history_blocks_closed_before": all(any(ids[k] == close_id for k in range(i, turn_open))
                                                 for i in range(turn_open) if ids[i] == open_id)}
    return {"before": before, "after": after, "check": check, "valid": all(check.values())}


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FP.MODEL, revision=FP.REV)
    msgs = FP.loop_messages("mapping")
    prompt = tok.apply_chat_template(msgs, tools=ST.TOOLS, add_generation_prompt=True, tokenize=False, **FP.KW)
    full = tok.apply_chat_template(msgs + [FP.decision_turn("mapping")], tools=ST.TOOLS, add_generation_prompt=False,
                                   tokenize=False, **FP.KW)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    p, f = enc(prompt), enc(full)
    rec = positions(f, len(p), tok) | {"prompt_tokens": len(p), "full_tokens": len(f),
                                         "history_think_blocks": prompt.count("<think>") - 1,
                                         "model": f"{FP.MODEL}@{FP.REV}",
                                         "rule": "GPC_PLAN.md §19.8 (a): the positions on the tokenizer alone"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
