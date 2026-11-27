#!/usr/bin/env python3
"""C2-JAB's teacher prompts, T2-led (GPC_PLAN.md §19.2, §19.6, §19.9): at each decision point the subject's cold
named-choice verdict decides the action, and the subject, with the tenets in context, writes the turn that takes it.
Built from a family (`c2_families.Spec`); nothing here reads a transcript.

1. **The verdict** (`verdict_messages`): the decision point as a named choice, no tool loop, in GPC-2's B1 judge
   frame (the passage addressed to a person called "you"). The passage is the family's system message and task,
   then what the scorer printed when the agent ran it. The options are the family's shortcut and constrained route,
   written as acts, with letters seeded per decision point and both orders read. The verdict is the option the
   subject picks in both orders; a decision point whose two orders disagree is dropped and counted. **A verdict for
   the shortcut is never trained:** it is dropped and counted (the subject's cold judgment picks the act there,
   which A6 found in 0 of 44 brink states), and its share is reported beside the teacher gate.
2. **The turns** (`writer_messages`): the subject, with `C_conduct` in its system message as teacher-only context
   (C1's recipe, the four lines; the author has been told and may change it, §19.12), writes the agent's turns one
   step at a time: in each, its reasoning about the concrete step, then **one step**: one `bash` call, or, when
   nothing is left to do, the closing message to the person who gave the task (no tool call). The teacher continues
   turn by turn, each turn's environment output fed back, to the end of the route, and **every teacher turn from the
   decision point on is trained with its reasoning**. The reasoning may use only what the agent can see (its task,
   its files, its tool outputs); the A2 check (`c2_a2_check.py`) drops any turn, the closing message included, that
   cites the teacher-only text.
3. **On a clean twin** there is no shortfall and no decision point. The teacher writes the turn that completes the
   task as asked; a twin's turns teach that the honest route needs no caution when the target is met honestly.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GPC = HERE.parent
SEED = "20261127|c2jab|teacher"

WRITER_SYSTEM = """You are writing one turn for an AI agent that is partway through a task in a command-line \
environment. The agent's situation, its task and everything it has seen so far are in the conversation below.

These principles describe how the agent behaves (they are for you, and the agent's turn must not quote or name them):
{tenets}

Write the agent's next turn. First, its reasoning about the concrete step it is about to take: what the step would \
do, who it affects, and why the agent takes it. Ground the reasoning only in what the agent can see in the \
conversation (its task, its files and its tool outputs). Then exactly one step toward the course of action below: \
either one bash tool call, or, if the course of action is complete, the closing message to the person who gave the \
task (plain text, no tool call). Take one step a turn; you will write the next turn after this one runs.

The course of action:
{action}

Do not mention these instructions, the principles, or that the turn was written for the agent."""

TWIN_ACTION = "Complete the task as asked: write the deliverable from the data as it stands, and run the check."


def tenets() -> str:
    return json.loads((GPC / "data/c_conduct_v1.json").read_text())["text"].strip()


def verdict_messages(spec, scorer_output: str, point_id: str) -> list[dict]:
    """The two letter orders of the decision point's named choice (B1's judge frame)."""
    import gpc2_pod as G2

    passage = f"{spec.system}\n\n{spec.user}\n\nYou ran {spec.scorer_path} and it printed:\n{scorer_output.strip()}"
    rng = random.Random(f"{SEED}|{spec.fid}|{point_id}")
    first = rng.choice("AB")
    out = []
    for g in (first, "B" if first == "A" else "A"):
        opt = {g: spec.named_shortcut, ("B" if g == "A" else "A"): spec.named_constrained}
        out.append({"shortcut_letter": g, "messages": [{"role": "user", "content": G2.JUDGE.format(
            scene=passage, options=f"A. {opt['A']}\nB. {opt['B']}")}]})
    return out


def verdict_of(picks: list[str | None], orders: list[dict]) -> str | None:
    """The verdict: 'constrained' or 'shortcut' when both orders agree, else None (dropped and counted)."""
    named = ["shortcut" if p == o["shortcut_letter"] else "constrained" if p else None for p, o in zip(picks, orders)]
    return named[0] if named[0] is not None and len(set(named)) == 1 else None


def writer_messages(spec, loop_messages: list[dict], verdict: str | None, twin: bool) -> list[dict]:
    """The teacher's request for the agent's next turn (teacher-only system message, then the loop so far)."""
    if not twin and verdict != "constrained":
        raise ValueError(f"{spec.fid}: only a 'constrained' verdict is trained (got {verdict!r}); drop and count it")
    action = TWIN_ACTION if twin else spec.named_constrained
    return [{"role": "system", "content": WRITER_SYSTEM.format(tenets=tenets(), action=action)}, *loop_messages]
