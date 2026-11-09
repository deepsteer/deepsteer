#!/usr/bin/env python3
"""CST S0b pod driver: the stages S0b adds to S0a's (``S0B_BUILD.md``; Amendments 16-18).

    cst_s0b_pod.py --stage cut --out DIR [--dry-run]      # CST-A15: cut-lines rescore + splice
    cst_s0a_pod.py --stage gen|dose0|score --arm a0n ...  # A0n runs through S0a's own stages

A0n (the neutral prefix) is an arm of ``cst_s0a_pod.py`` so its generation, dose-0 letter reads
and bulk scoring are S0a's code paths, not copies. This file holds what is new:

- ``cut``: the CST-A15 discriminator (Amendment 14 item 2). gpt-oss-120b, blind, as S_concl's
  scorer of record (``cst_s0a_pod.score_passes``), re-reads every operator-cell rollout of each arm
  on disk with the trace's closing sentences removed (``cut_trace``), and reads a spliced-ending
  control: on each A0 operator scenario whose four rollouts take two different options, the body of
  one rollout followed by the closing sentences of another that acted otherwise.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cst_s0a_pod as S  # noqa: E402  (sets the shared sys.path; the engine and scorer of record)

CUT_N = 2  # closing sentences removed (CST-A15's registered discriminator: "the last two")
MIN_SENTENCES = 6  # a trace this short is not cut (reported as excluded, never scored cut)
SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=\S)")
SPLICE_SEED = "20261109|cst_a15_splice"


def sentences(trace: str) -> list[str]:
    """The trace's sentences: split after ., ! or ? (and any closing quote or bracket) followed by
    whitespace; the text is kept verbatim, so joining the pieces with one space restores it up to
    whitespace."""
    return [x for x in SENT.split(trace.strip()) if x.strip()]


def cut_trace(trace: str, n: int = CUT_N) -> tuple[str, str] | None:
    """(the trace without its last ``n`` sentences, those sentences), or None when the trace has
    fewer than ``MIN_SENTENCES`` sentences."""
    ss = sentences(trace)
    if len(ss) < MIN_SENTENCES:
        return None
    return " ".join(ss[:-n]), " ".join(ss[-n:])


def _split(text: str) -> tuple[str, str]:
    """(trace, answer) exactly as the scorer of record splits a rollout (``judge_prompts``)."""
    return text.split("</think>")[0], text.split("</think>")[-1]


def cut_rows(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Rollout id -> a copy whose trace has its closing sentences cut; ids too short to cut."""
    out, short = {}, []
    for r in rows:
        trace, answer = _split(r["text"])
        c = cut_trace(trace)
        if c is None:
            short.append(r["rollout_id"])
            continue
        out[r["rollout_id"]] = {**r, "text": f"{c[0]}</think>{answer}", "cut_closing": c[1]}
    return out, short


def splice_rows(rows: list[dict]) -> dict[str, dict]:
    """The spliced-ending control: per scenario with two different options among its rollouts, one
    seeded pair (body of X, closing sentences of Y, Y acting otherwise); X's answer is kept, and the
    judge never sees an answer (blind)."""
    rng = random.Random(SPLICE_SEED)
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("option") and not r["hit_cap"]:
            by.setdefault(r["scenario_id"], []).append(r)
    out = {}
    for sid in sorted(by):
        rs = sorted(by[sid], key=lambda r: r["k"])
        pairs = [(x, y) for x in rs for y in rs if x["option"] != y["option"]]
        if not pairs:
            continue
        x, y = rng.choice(pairs)
        cx, cy = cut_trace(_split(x["text"])[0]), cut_trace(_split(y["text"])[0])
        if cx is None or cy is None:
            continue
        rid = f"{x['rollout_id']}|splice|{y['k']}"
        answer = _split(x["text"])[1]
        out[rid] = {**x, "rollout_id": rid, "text": f"{cx[0]} {cy[1]}</think>{answer}",
                    "splice_body_option": x["option"], "splice_closing_option": y["option"]}
    return out


def operator_rows(out: Path, arm: str) -> list[dict]:
    return [r for r in S.scored_rows(out, arm) if r["set"] == "panel"
            and r["condition"] == "operator"]


def stage_cut(out: Path, arms: list[str], dry: bool) -> None:
    """CST-A15: the cut-lines rescore of every operator-cell rollout per arm, and the splice
    control on A0; the scorer of record's blind pass, seeds offset from S_concl's."""
    rec = {"cut_n": CUT_N, "min_sentences": MIN_SENTENCES, "arms": {}}
    for arm in arms:
        rows = operator_rows(out, arm)
        if not rows:
            raise SystemExit(f"BAIL: no {arm} operator rollouts under {out / 'gen' / arm}")
        cut, short = cut_rows(rows)
        passes = {"cut": (sorted(cut), 3, False)}
        sp = {}
        if arm == "a0":
            sp = splice_rows(rows)
            passes["splice"] = (sorted(sp), 4, False)
        S.score_passes(out / "score_cut" / arm / "gpt_oss_120b", {**cut, **sp}, passes,
                       "gpt_oss_120b", dry)
        rec["arms"][arm] = {"operator_rollouts": len(rows), "cut": len(cut),
                            "too_short": len(short), "splice": len(sp)}
        (out / "score_cut" / arm).mkdir(parents=True, exist_ok=True)
        with open(out / "score_cut" / arm / "rows.jsonl", "w") as fh:  # what each pass was shown
            for rid, r in {**cut, **sp}.items():
                fh.write(json.dumps({"rollout_id": rid, "source": r.get("rollout_id"),
                                     "option": r.get("option"),
                                     "cut_closing": r.get("cut_closing"),
                                     "splice_body_option": r.get("splice_body_option"),
                                     "splice_closing_option": r.get("splice_closing_option")})
                         + "\n")
    S.V._w(out, "score_cut", rec)  # noqa: SLF001


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("cut",))
    ap.add_argument("--arms", default="a0,a0s")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"cut": lambda: stage_cut(a.out, a.arms.split(","), a.dry_run)}[a.stage]()


if __name__ == "__main__":
    main()
