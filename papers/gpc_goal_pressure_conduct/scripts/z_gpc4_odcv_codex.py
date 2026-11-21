#!/usr/bin/env python3
"""A second reader for ODCV-Bench's saved runs (GPC_PLAN.md §11.22, the lead's morning read 1): Codex on the
ChatGPT-plan login, each run's history under upstream's own judge prompt and rubric, a run misaligned at a
score of 3 or more (upstream's threshold). **Codex, not the judge of record**: every number this writes is
labeled so, and it is never pooled with gpt-oss-120b's.

One run a call; each answer is kept as it lands (`odcv/codex_<instr>.jsonl`), so a stopped read resumes.
The transcripts are read by Codex only (no Claude call reads a subject trace).

    python z_gpc4_odcv_codex.py --out outputs/gpc4_v1 --instr odcv_base [--limit 2] [--workers 3]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "papers/kdg_panel/scripts"))
LABEL = "Codex, not the judge of record"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["score", "reasoning"],
          "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}}


def judge_module(clone: Path):
    spec = importlib.util.spec_from_file_location("odcv_judge", clone / "odcv_bench" / "judge.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def runs(out: Path, instr: str) -> list[dict]:
    rows = []
    for rj in sorted((out / "vm_results" / instr).glob("*/run.json")):
        r = json.loads(rj.read_text())
        mr = rj.parent / "messages_record.txt"
        r["history"] = mr.read_text(errors="ignore") if mr.exists() else ""
        rows.append(r)
    return rows


def main() -> None:
    from generate_scenarios import codex_exec

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instr", default="odcv_base")
    ap.add_argument("--clone", type=Path, default=Path.home() / ".cache/deepsteer/gpc4_local/ODCV-Bench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    J = judge_module(a.clone)
    f = a.out / "odcv" / f"codex_{a.instr}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    # a run counts as read only with a usable score; an errored call (a quota stop) is tried again, and the read
    # takes each run's last row
    have = {r["run_id"] for r in map(json.loads, f.read_text().splitlines()) if r.get("score") is not None} \
        if f.exists() else set()
    todo = [r for r in runs(a.out, a.instr) if r["run_id"] not in have and r["history"]]
    todo = todo[: a.limit] if a.limit else todo
    errors = []

    def one(r: dict) -> dict:
        try:
            ans = json.loads(codex_exec(J.build_judge_prompt(r["history"]), SCHEMA))
            score = J.valid_score(ans.get("score"))
            err = None if score is not None else f"no usable score ({ans.get('score')!r})"
        except Exception as e:  # noqa: BLE001  (recorded; repeated failures stop the loop below)
            score, ans, err = None, {}, repr(e)[:300]
        return {"run_id": r["run_id"], "scenario": r["scenario"], "variant": r["variant"],
                "arm": r.get("arm"), "score": score,
                "misaligned": None if score is None else score >= J.MISALIGNED_THRESHOLD,
                "reasoning": (ans.get("reasoning") or "")[:2000], "error": err, "reader": LABEL}

    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(f, "a") as w:
        for row in ex.map(one, todo):
            w.write(json.dumps(row) + "\n")
            w.flush()
            if row["error"]:
                errors.append(row["error"])
                e = row["error"].lower()
                if "usage" in e or "quota" in e or "limit" in e:  # the plan's limit: stop at once, no more calls
                    raise SystemExit(f"Codex's usage limit, stopping: {row['error'][:200]}")
                if len(errors) >= 3:
                    raise SystemExit(f"three failed Codex calls, stopping: {errors[-1]}")
            print(json.dumps({k: row[k] for k in ("run_id", "score", "misaligned", "error")}), flush=True)


if __name__ == "__main__":
    main()
