"""MRB-G2 pod driver (MRB_PLAN v0.7): the theory set on Qwen3-30B-A3B. Runs in the vLLM venv.

    generate --out DIR [--k 2] [--validate] [--deadline EPOCH] [--local-dir PATH]
    judge --out DIR [--validate] [--deadline EPOCH] [--local-dir PATH]
    project --out DIR --validate-hours H --setup-hours H [--envelope-h 3.0]
    --dry-run (any subcommand): a stub engine and cached tokenizers, no GPU (local tests)

Generation: upstream's theory prompt (``theory_generation_prompt``), Table 4 sampling (T 0.6,
top-p 0.95), 10,500 tokens, k = 2 per row (seeded). Judging: G1's judge of record (gpt-oss-120b,
effort high, Table 4 sampling, tensor-parallel over two GPUs), the registered prompt, one call per
criterion and field, the ladder (empty, the scenario restated) first and then row by row; G1's
verdict guard. Outputs are written as they finish and a rerun resumes. Tracked and untracked
outputs as G1's (ids, lengths, hashes, verdicts of at most 20 characters; text in ``DIR/_raw``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mrb_g1_pod as P  # noqa: E402

from deepsteer.benchmarks.moral_reasoning.morebench.prompts import (  # noqa: E402
    judge_prompt,
    theory_generation_prompt,
)
from deepsteer.reasoning.think_io import CoTFormat  # noqa: E402

POD_VERSION = "1.0.0"
MODEL = "qwen3_30b"
K = 2
LADDER = ("empty", "restated")  # the two rungs v0.7 registers, from canned_ladder_v1.json


def in_validate(idx: int) -> bool:  # 15 rows: every tenth, so all five theories appear
    return idx % 10 == 0


def _items():
    from deepsteer.datasets.morebench import load_morebench
    return load_morebench(subset="theory")


def gen_jobs(items, tok, validate: bool, k: int = K) -> list[dict]:
    spec = P.GEN[MODEL]
    jobs = []
    for it in items:
        if validate and not in_validate(it.idx):
            continue
        ids = P.render_ids(tok, theory_generation_prompt(it.dilemma, it.theory), spec["format"],
                           spec["effort"], None)
        for kk in range(k):
            jobs.append({"job_id": f"{MODEL}/default/{it.scenario_id}/k{kk}", "prompt_ids": ids,
                         "max_tokens": P.MAX_TOKENS, "seed": P.seed_of("g2", MODEL, it.scenario_id, kk),
                         "sampler": spec["sampler"]})
    return jobs


def judge_units(items, gens: dict[str, dict], validate: bool) -> list[dict]:
    """Qwen's generations on both fields, then the two-rung ladder, every unit on its row's
    criteria; ordered ladder first, then row by row (a deadline costs the last rows)."""
    canned = json.loads(P.CANNED.read_text())["canned"]
    by_sid = {it.scenario_id: it for it in items}
    units = []
    for it in items:
        if validate and not in_validate(it.idx):
            continue
        for name in LADDER:
            c = canned[name]
            text = it.dilemma if c["kind"] == "scenario" else c["text"]
            units.append({"unit": f"canned/{name}/{it.scenario_id}", "sid": it.scenario_id,
                          "text": text})
    for gid, g in gens.items():
        sid = gid.split("/")[2]
        if sid not in by_sid:
            continue
        for field in ("trace", "response"):
            units.append({"unit": f"{gid}/{field}", "sid": sid, "text": g[field]})
    return sorted(units, key=lambda u: (not u["unit"].startswith("canned/"),
                                        by_sid[u["sid"]].idx))  # stable


def judge_jobs(units, items, tok, cfg):
    by_sid = {it.scenario_id: it for it in items}
    for u in units:
        for c in by_sid[u["sid"]].criteria:
            jid = f"{u['unit']}/{c.criterion_id}"
            yield {"job_id": jid, "unit": u["unit"], "criterion_id": c.criterion_id,
                   "weight": c.weight,
                   "prompt_ids": P.render_ids(tok, judge_prompt(u["text"], c.title),
                                              CoTFormat.HARMONY_ANALYSIS, P.JUDGE["effort"], cfg),
                   "max_tokens": P.MAX_TOKENS, "seed": P.seed_of("g2-judge", jid),
                   "sampler": P.JUDGE["sampler"]}


def cmd_generate(a) -> dict:
    spec = P.GEN[MODEL]
    tok = P._tok(spec["repo"], spec["revision"], a.local_dir)
    jobs = gen_jobs(_items(), tok, a.validate, a.k)
    engine, SP = P.load_engine(spec["repo"], spec["revision"], a.local_dir, a.dry_run, tok,
                               spec["format"], "I would choose the first option.", "gen")
    out, raw = a.out / "gen" / f"{MODEL}.jsonl", a.out / "_raw" / "gen" / f"{MODEL}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)

    def decode(j, r):
        text = P.decode_rollout(tok, r["gen_ids"], spec["format"]) if r["gen_ids"] else ""
        trace, resp = P.split_trace(text, spec["format"]) if text else ("", "")
        return ({"n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
                 "status": r["status"], "final_reached": bool(resp),
                 "trace_chars": len(trace), "response_chars": len(resp),
                 "trace_sha256": P.sha(trace), "response_sha256": P.sha(resp)},
                {"trace": trace, "response": resp})

    rec = P.run_chunks(engine, SP, iter(jobs), out, raw, decode, deadline=a.deadline,
                       meta_keys=("job_id",))
    rec.update({"model": MODEL, "set": "theory", "n_jobs": len(jobs), "validate": a.validate,
                "pod_version": POD_VERSION})
    (a.out / "gen" / f"{MODEL}.summary.json").write_text(json.dumps(rec, indent=1))
    return rec


def cmd_judge(a) -> dict:
    tok = (P._tok(P.GEN["gpt_oss_20b"]["repo"], P.GEN["gpt_oss_20b"]["revision"], None)
           if a.dry_run else P._tok(P.JUDGE["repo"], P.JUDGE["revision"], a.local_dir))
    cfg = P.harmony_cfg()
    items = _items()
    units = judge_units(items, P.load_generations(a.out), a.validate)
    engine, SP = P.load_engine(P.JUDGE["repo"], P.JUDGE["revision"], a.local_dir, a.dry_run, tok,
                               CoTFormat.HARMONY_ANALYSIS, "yes", "judge")
    out, raw = a.out / "judge" / "shard0.jsonl", a.out / "_raw" / "judge" / "shard0.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)

    def decode(j, r):
        text = (P.decode_rollout(tok, r["gen_ids"], CoTFormat.HARMONY_ANALYSIS)
                if r["gen_ids"] else "")
        _, final = P.split_rollout(text, CoTFormat.HARMONY_ANALYSIS) if text else ("", "")
        final = final if r["status"] == "finished" and final else None
        short = final.strip().lower() if final is not None and len(final) <= 20 else None
        return ({"n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
                 "status": r["status"], "verdict": short,
                 "verdict_chars": None if final is None else len(final),
                 **P.credits(final, j["weight"])}, {"reply": text})

    guard = P.VerdictGuard()
    jobs = (j for j in judge_jobs(units, items, tok, cfg) if guard.admits(j))
    rec = P.run_chunks(engine, SP, jobs, out, raw, decode, deadline=a.deadline,
                       meta_keys=("job_id", "unit", "criterion_id", "weight"), on_rows=guard)
    n = {it.scenario_id: len(it.criteria) for it in items}
    rec.update({"calls_planned": sum(n[u["sid"]] for u in units), "guard": P.GUARD,
                "guard_seen": guard.seen, "guard_tripped": guard.tripped,
                "validate": a.validate, "pod_version": POD_VERSION})
    (a.out / "judge" / "shard0.summary.json").write_text(json.dumps(rec, indent=1))
    return rec


def project(measured: dict, envelope_h: float, validate_h: float) -> dict:
    """The session's A100-hours from VALIDATE's rates: proceed if it fits the envelope's
    remainder, else STOP (MRB_PLAN v0.7)."""
    wall = (measured["setup_h"] + measured["gen_wall_h_full"] + measured["judge_load_h"]
            + measured["judge_wall_h_full"] + measured["margin_h"])
    remaining = envelope_h - validate_h
    fits = 2 * wall <= remaining
    return {"envelope_h": envelope_h, "validate_h": validate_h, "remaining_h": round(remaining, 3),
            "session_wall_h": round(wall, 3), "session_a100_h": round(2 * wall, 3),
            "decision": "proceed" if fits else "STOP",
            "session_max_wall_h": round(min(1.15 * wall, remaining / 2), 3) if fits else None}


def cmd_project(a) -> dict:
    items = _items()
    v = a.out
    gen_s = json.loads((v / "gen" / f"{MODEL}.summary.json").read_text())
    judge_s = json.loads((v / "judge" / "shard0.summary.json").read_text())
    rows = [json.loads(x) for x in (v / "judge" / "shard0.jsonl").read_text().splitlines()]
    tok_per_call = {}
    for kind in ("canned", "trace", "response"):
        rs = [r for r in rows if (r["unit"].startswith("canned/") if kind == "canned"
                                  else r["unit"].endswith("/" + kind))]
        tok_per_call[kind] = sum(r["n_tokens"] for r in rs) / max(len(rs), 1)
    n_crit = sum(len(it.criteria) for it in items)
    calls = {"canned": n_crit * len(LADDER), "trace": n_crit * K, "response": n_crit * K}
    tokens = sum(calls[k] * tok_per_call[k] for k in calls)
    full_jobs = len(items) * K
    measured = {"setup_h": a.setup_hours, "margin_h": 0.25, "judge_load_h": 0.05,
                "gen_wall_h_full": full_jobs * gen_s["decoded_tokens"] / max(gen_s["jobs_run"], 1)
                / max(gen_s["tokens_per_s"], 1e-9) / 3600,
                "judge_tokens_per_s": judge_s["tokens_per_s"],
                "judge_tokens_per_call": tok_per_call, "calls_full": calls,
                "judge_wall_h_full": tokens / max(judge_s["tokens_per_s"], 1e-9) / 3600,
                "no_verdict_share": {k: sum(r["verdict"] is None for r in rows
                                            if (r["unit"].startswith("canned/") if k == "canned"
                                                else r["unit"].endswith("/" + k)))
                                     / max(1, sum(1 for r in rows if (
                                         r["unit"].startswith("canned/") if k == "canned"
                                         else r["unit"].endswith("/" + k))))
                                     for k in ("canned", "trace", "response")}}
    rec = {"measured": measured, **project(measured, a.envelope_h, a.validate_hours)}
    (v / "projection.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["generate", "judge", "project"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--local-dir", type=Path, default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--envelope-h", type=float, default=3.0)
    ap.add_argument("--validate-hours", type=float, default=0.0)
    ap.add_argument("--setup-hours", type=float, default=0.25)
    a = ap.parse_args()
    rec = {"generate": cmd_generate, "judge": cmd_judge, "project": cmd_project}[a.cmd](a)
    print(json.dumps(rec, default=str)[:1500])


if __name__ == "__main__":
    main()
