#!/usr/bin/env python3
"""CST S0b pod driver: the stages S0b adds to S0a's (``S0B_BUILD.md``; Amendments 16-18).

    cst_s0b_pod.py --stage cut --out DIR --src S0A_DIR [--arms a0,a0s,a0n] [--dry-run]
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
RETEST_SEED, RETEST_N = "20261109|cst_a15_cut_retest", 100  # Amendment 19 rider b


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


# a sentence that states a choice of option (the residual check and the decision-block cut;
# S0B_BUILD.md §6 item 6): a choice verb near a letter, or a letter judged best
CHOICE = re.compile(r"\b(choose|chose|go with|answer|pick|select|option)\b[^.]{0,30}\b[ABC]\b"
                    r"|\b[ABC]\b[^.]{0,20}\b(is|seems)\b[^.]{0,20}\b(best|right|correct|better)\b",
                    re.I)


def states_choice(sentence: str) -> bool:
    return bool(CHOICE.search(sentence))


def cut_block(trace: str) -> tuple[str, str] | None:
    """The decision-block cut: remove from the first sentence in the trace's last third that states
    a choice to the end; with no such sentence, the two-sentence cut. None under MIN_SENTENCES."""
    ss = sentences(trace)
    if len(ss) < MIN_SENTENCES:
        return None
    start = len(ss) - len(ss) // 3
    i = next((j for j in range(start, len(ss)) if states_choice(ss[j])), len(ss) - CUT_N)
    i = min(i, len(ss) - CUT_N)  # never cut less than the registered two sentences
    return " ".join(ss[:i]), " ".join(ss[i:])


def _split(text: str) -> tuple[str, str]:
    """(trace, answer) exactly as the scorer of record splits a rollout (``judge_prompts``)."""
    return text.split("</think>")[0], text.split("</think>")[-1]


def cut_rows(rows: list[dict], cut=None) -> tuple[dict[str, dict], list[str]]:
    """Rollout id -> a copy whose trace has its closing sentences cut (``cut_trace`` by default,
    ``cut_block`` for the decision-block variant); ids too short to cut."""
    cut = cut or cut_trace
    out, short = {}, []
    for r in rows:
        trace, answer = _split(r["text"])
        c = cut(trace)
        if c is None:
            short.append(r["rollout_id"])
            continue
        out[r["rollout_id"]] = {**r, "text": f"{c[0]}</think>{answer}", "cut_closing": c[1],
                                "residual_choice": any(map(states_choice, sentences(c[0])[-2:]))}
    return out, short


def splice_rows(rows: list[dict], body_cut=None) -> dict[str, dict]:
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
        cx, cy = (body_cut or cut_trace)(_split(x["text"])[0]), cut_trace(_split(y["text"])[0])
        if cx is None or cy is None:
            continue
        tag = "bsplice" if body_cut is cut_block else "splice"
        rid = f"{x['rollout_id']}|{tag}|{y['k']}"
        answer = _split(x["text"])[1]
        out[rid] = {**x, "rollout_id": rid, "text": f"{cx[0]} {cy[1]}</think>{answer}",
                    "source_id": x["rollout_id"], "closing_id": y["rollout_id"],
                    "splice_body_option": x["option"], "splice_closing_option": y["option"]}
    return out


def operator_rows(out: Path, arm: str) -> list[dict]:
    return [r for r in S.scored_rows(out, arm) if r["set"] == "panel"
            and r["condition"] == "operator"]


def stage_cut(out: Path, arms: list[str], dry: bool, src: Path | None = None) -> None:
    """CST-A15: the cut-lines rescore of every operator-cell rollout per arm, and the splice
    control on A0; the scorer of record's blind pass, seeds offset from S_concl's. An arm's
    rollouts are read from ``out`` when S0b generated them (A0n), else from ``src`` (S0a's)."""
    rec = {"cut_n": CUT_N, "min_sentences": MIN_SENTENCES, "arms": {}}
    for arm in arms:
        base = out if (out / "gen" / arm).is_dir() or src is None else src
        rows = operator_rows(base, arm)
        if not rows:
            raise SystemExit(f"BAIL: no {arm} operator rollouts under {base / 'gen' / arm}")
        cut, short = cut_rows(rows)  # the registered two-sentence cut
        blk, _ = cut_rows(rows, cut_block)  # the decision-block variant (S0B_BUILD.md §6 item 6)
        blk = {f"{k}|bcut": {**v, "rollout_id": f"{k}|bcut", "source_id": k}
               for k, v in blk.items()}
        # pass names never prefix one another: score_passes resumes by globbing "<pass>_*.jsonl"
        passes = {"cut": (sorted(cut), 3, False), "bcut": (sorted(blk), 5, False)}
        sp = {}
        if arm == "a0":
            # rider b: 100 seeded cut rows per cut, scored a second time at a new seed offset
            rng = random.Random(RETEST_SEED)
            passes["cutrt"] = (sorted(rng.sample(sorted(cut), min(RETEST_N, len(cut)))), 7, False)
            passes["bcutrt"] = (sorted(rng.sample(sorted(blk), min(RETEST_N, len(blk)))), 8, False)
            sp = splice_rows(rows) | splice_rows(rows, cut_block)
            passes["splice"] = (sorted(k for k in sp if "|splice|" in k), 4, False)
            passes["bsplice"] = (sorted(k for k in sp if "|bsplice|" in k), 6, False)
        S.score_passes(out / "score_cut" / arm / "gpt_oss_120b", {**cut, **blk, **sp}, passes,
                       "gpt_oss_120b", dry)
        rec["arms"][arm] = {"source": str(base), "operator_rollouts": len(rows), "cut": len(cut),
                            "bcut": len(blk), "too_short": len(short), "splice": len(sp)}
        (out / "score_cut" / arm).mkdir(parents=True, exist_ok=True)
        with open(out / "score_cut" / arm / "rows.jsonl", "w") as fh:  # what each pass was shown
            for rid, r in {**cut, **blk, **sp}.items():
                fh.write(json.dumps({"rollout_id": rid, "source": r.get("source_id", rid),
                                     "closing_source": r.get("closing_id"),
                                     "option": r.get("option"),
                                     "cut_closing": r.get("cut_closing"),
                                     "residual_choice": r.get("residual_choice"),
                                     "splice_body_option": r.get("splice_body_option"),
                                     "splice_closing_option": r.get("splice_closing_option")})
                         + "\n")
    S.V._w(out, "score_cut", rec)  # noqa: SLF001


GATE = {"parse_min": 0.95, "capped_max": 0.03, "rollouts_per_h_min": 600}  # S0B_BUILD.md §4


def stage_gate(out: Path, arm: str) -> bool:
    """S0b-1's pilot gate on ``arm``'s chunks so far: the parsed share (an option read), the capped
    share and the generation rate from the chunk logs. Writes ``gate_<arm>.json``; True means
    proceed."""
    import glob

    rows = [json.loads(x) for f in sorted((out / "gen" / arm).glob("*.jsonl")) for x in open(f)]
    chunks: dict = {}
    for f in sorted(glob.glob(str(out / f"gen_{arm}_log_*.json"))):
        chunks.update(json.loads(Path(f).read_text())["chunks"])
    n, sec = sum(c["n"] for c in chunks.values()), sum(c["seconds"] for c in chunks.values())
    rec = {"arm": arm, "rollouts": len(rows), "chunks": len(chunks), **GATE,
           "parsed": sum(r["option"] is not None for r in rows) / max(len(rows), 1),
           "capped": sum(r["hit_cap"] for r in rows) / max(len(rows), 1),
           "rollouts_per_h": 3600 * n / sec if sec else None}
    rec["proceed"] = bool(rows and rec["parsed"] >= GATE["parse_min"]
                          and rec["capped"] <= GATE["capped_max"]
                          and (rec["rollouts_per_h"] or 0) >= GATE["rollouts_per_h_min"])
    S.V._w(out, f"gate_{arm}", rec)  # noqa: SLF001
    return rec["proceed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("cut", "gate"))
    ap.add_argument("--arm", default="a0n")
    ap.add_argument("--arms", default="a0,a0s")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=None, help="S0a's outputs (A0 and A0s rollouts)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "gate":
        sys.exit(0 if stage_gate(a.out, a.arm) else 1)
    stage_cut(a.out, a.arms.split(","), a.dry_run, a.src)


if __name__ == "__main__":
    main()
