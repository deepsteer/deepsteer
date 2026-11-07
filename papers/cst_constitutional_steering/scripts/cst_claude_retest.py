#!/usr/bin/env python3
"""Amendment 10 item 2: Claude re-scores the probe's 100 blind rollouts twice (account login).

    cst_claude_retest.py run --pass 1|2 [--max N]   # cached calls, one rollout per call
    cst_claude_retest.py read                        # kappas per tenet kind; writes the record

Each rollout gets the scorer of record's blind prompt (``cst_judge.judge_prompt`` with
``answer=None`` and the subject's letters), one rollout per call (a ten-item batch was stopped by a
safety classifier on the pilot; single items were not). A failed call is retried once; the run
stops after five consecutive failures. Raw CLI records (with ``safety_stops``) are cached under
``outputs_raw/claude_retest/``; ``read`` writes
``outputs/claude_retest.json``: Claude's pass-1 vs pass-2 kappa on the concluded option and on the
applied-tenet cells, all tenets and per kind (the deference tenets, the rest), with
rollout-bootstrap CIs; and Claude against gpt-oss-120b's probe pass (blind, effort low) beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cst_probe_pod as P  # noqa: E402  (sets the shared sys.path)

V = P.V
RAW = V.CST / "outputs_raw" / "claude_retest"
OUT = V.CST / "outputs" / "claude_retest.json"
GEN = V.CST / "outputs" / "probe" / "gen_a0.jsonl"
N = 100


def items() -> tuple[list[dict], list[str], list[dict]]:
    """The first 100 probe rollouts, their blind prompt texts, and the parse metadata."""
    import cst_judge as J
    import scenario_labels as L

    from deepsteer.constitutions import TenetSet

    rows = [json.loads(x) for x in open(GEN)][:N]
    tenets = TenetSet.load(V.CST / "data/tenets_C_anth_v2.json").by_id()
    labs = {f: {r["scenario_id"]: r for r in json.loads(
        (V.CST / f"data/scenario_labels_v4_{f}.json").read_text())["labels"]}
        for f in ("claude", "codex")}
    _, by, _ = L._ordered_and_opts()
    texts, meta = [], []
    for r in rows:
        sid = r["scenario_id"]
        s, _ = by[sid]
        short = sorted(set(labs["claude"][sid]["shortlist"]) | set(labs["codex"][sid]["shortlist"])
                       | set(L.DEFERENCE))
        letters = {o: L_ for L_, o in r["order"]}
        texts.append(J.judge_prompt(s.agent_text, [(o, s.option(o).text) for _, o in r["order"]],
                                    r["text"].split("</think>")[0], None,
                                    [(t, tenets[t].claim) for t in short], letters=letters))
        meta.append({"options": set(letters), "shortlist": short, "letters": letters})
    return rows, texts, meta


def claude_json(system: str, prompt: str) -> dict:
    """One ``claude -p`` call on the account login (API keys stripped), its JSON record."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    r = subprocess.run(["claude", "-p", "--model", "opus", "--system-prompt", system,
                        "--output-format", "json"], input=prompt, capture_output=True, text=True,
                       env=env, timeout=900)
    try:
        rec = json.loads(r.stdout)
    except json.JSONDecodeError:
        rec = {"is_error": True, "result": "", "stderr": r.stderr[-300:]}
    rec["returncode"] = r.returncode
    return rec


def run(pass_no: int, max_calls: int | None) -> None:
    import cst_judge as J

    _, texts, _ = items()
    RAW.mkdir(parents=True, exist_ok=True)
    calls = fails_in_row = 0
    for k in range(N):
        f = RAW / f"pass{pass_no}_{k:03d}.json"
        if f.exists():
            continue
        if max_calls is not None and calls >= max_calls:
            break
        for attempt in (1, 2):
            rec = claude_json(J.SYS, texts[k])
            calls += 1
            ok = rec["returncode"] == 0 and not rec.get("is_error") and rec.get("result")
            if ok:
                break
            (RAW / f"pass{pass_no}_{k:03d}.bad{attempt}.json").write_text(json.dumps(rec))
        if ok:
            rec["attempts"] = attempt
            f.write_text(json.dumps(rec))
            fails_in_row = 0
        else:
            fails_in_row += 1
            if fails_in_row >= 5:
                raise SystemExit(f"pass {pass_no}: five consecutive failures at item {k}")
    print(f"pass {pass_no}: {calls} calls this run")


def parse_pass(pass_no: int, meta: list[dict]) -> list[dict]:
    import cst_judge as J
    out = []
    for k, m in enumerate(meta):
        f = RAW / f"pass{pass_no}_{k:03d}.json"
        text = json.loads(f.read_text())["result"] if f.exists() else ""
        pj = (J.parse_judgment(text, m["options"], set(m["shortlist"]), m["letters"]) if text
              else {"ok": False, "concluded_option": "invalid", "applied_tenets": []})
        out.append({"ok": pj["ok"], "concluded_option": pj["concluded_option"],
                    "applied_tenets": pj["applied_tenets"], "shortlist": m["shortlist"]})
    return out


def kind(rows: list[dict], keep) -> list[dict]:
    return [{**r, "shortlist": [t for t in r["shortlist"] if keep(t)],
             "applied_tenets": [t for t in r["applied_tenets"] if keep(t)]} for r in rows]


def read() -> None:
    import scenario_labels as L

    _, _, meta = items()
    a, b = parse_pass(1, meta), parse_pass(2, meta)
    gpt = [json.loads(x) for x in open(V.CST / "outputs/probe/judge_gpt_oss_120b_blind_s0.jsonl")]
    gpt = gpt[:N]
    dfr = set(L.DEFERENCE)
    kinds = {"all": lambda t: True, "deference": lambda t: t in dfr,
             "other": lambda t: t not in dfr}
    def stops(p):
        recs = [json.loads(f.read_text()) for f in RAW.glob(f"pass{p}_*.json")]
        return {"calls_ok": len([f for f in RAW.glob(f"pass{p}_[0-9][0-9][0-9].json")]),
                "failed_attempts": len(list(RAW.glob(f"pass{p}_*.bad*.json"))),
                "safety_stops": sum(int(r.get("safety_stops") or 0) for r in recs)}

    rec = {"n": N, "parse": {"pass1": sum(x["ok"] for x in a) / N,
                             "pass2": sum(x["ok"] for x in b) / N},
           "calls": {"pass1": stops(1), "pass2": stops(2)},
           "claude_retest": {"concluded_option": P.kappa_ci(a, b, "concluded_option")},
           "claude_vs_gpt_oss_low": {"concluded_option": P.kappa_ci(a, gpt, "concluded_option")}}
    for name, keep in kinds.items():
        rec["claude_retest"][f"tenets_{name}"] = P.kappa_ci(kind(a, keep), kind(b, keep), "tenets")
        rec["claude_vs_gpt_oss_low"][f"tenets_{name}"] = P.kappa_ci(kind(a, keep), kind(gpt, keep),
                                                                    "tenets")
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    for k, v in rec["claude_retest"].items():
        print("claude retest", k, round(v["kappa"], 3), [round(x, 3) for x in v["ci95"]])
    for k, v in rec["claude_vs_gpt_oss_low"].items():
        print("claude vs gpt-oss", k, round(v["kappa"], 3), [round(x, 3) for x in v["ci95"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "read"))
    ap.add_argument("--pass", dest="pass_no", type=int, default=1, choices=(1, 2))
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    run(a.pass_no, a.max) if a.step == "run" else read()
