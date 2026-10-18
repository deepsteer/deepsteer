#!/usr/bin/env python3
"""vLLM generation worker for ST trace cells (ST_PLAN Amendment 7). Runs in the vLLM venv, one
engine configuration per process, no HF model loaded beside it.

    python gen_vllm.py --engine ENGINE.json --jobs JOBS.jsonl --out OUT.jsonl \\
        [--deadline EPOCH] [--after-h HOURS]

ENGINE.json: ``{"model", "revision", "tokenizer", "tokenizer_revision", "llm_kwargs": {...},
"label"}`` (``llm_kwargs`` holds the speculative config, compilation, batch limits; nothing else
is set implicitly). JOBS.jsonl rows: ``{"job_id", "prompt_ids", "max_tokens", "seed",
"sampler": {"temperature", "top_p", "top_k"}}``.

Sampling is the KDG sampler v2 contract unless a job carries the card sampler: temperature as
given, top_p / top_k off unless given, min_p 0, no penalties, a per-request seed, prompts as token
ids, ``detokenize=False``. An explicit SamplingParams means vLLM applies no generation_config
sampling defaults; the model's eos ids still end a sequence, as in HF generate.

Envelope guard (Amendment 6 item 2, adapted to concurrent decoding): every 4 finished requests,
project the remaining time from the tokens decoded so far per second and the unfinished requests'
expected remaining length: the largest of the current length, the 90th percentile of finished
lengths and ``--expected-len`` (the pilot's 90th percentile, passed for every stage after it; the
first finishers are the short traces, so finished lengths alone underestimate), at most
``max_tokens``; stop if now + projection + ``after_h`` passes the deadline, which also stops the
run outright. A stop aborts
every unfinished request (they come back ``status: aborted``) and the worker exits 3.

OUT.jsonl: one row per job (``gen_ids``, ``finish_reason``, ``n_tokens``, ``t_finish``), then a
summary file OUT.summary.json (wall time, tokens per second, guard checks, the settings hash over
the engine JSON + vLLM and torch versions + VLLM_* environment, the speculative acceptance
counters when exposed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

GUARD_EVERY = 4


def settings_hash(engine: dict, versions: dict) -> str:
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("VLLM_")}
    blob = json.dumps({"engine": engine, "versions": versions, "env": env}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def sampling_params(SamplingParams, job: dict):
    s = job.get("sampler") or {}
    return SamplingParams(temperature=s.get("temperature", 0.7), top_p=s.get("top_p", 1.0),
                          top_k=s.get("top_k", 0), min_p=0.0, repetition_penalty=1.0,
                          presence_penalty=0.0, frequency_penalty=0.0, seed=int(job["seed"]),
                          max_tokens=int(job["max_tokens"]), detokenize=False)


def run_jobs(engine, jobs: list[dict], make_params, *, deadline: float | None = None,
             after_s: float = 0.0, expected_len: int = 0,
             clock=time.time) -> tuple[dict[str, dict], dict]:
    """Drive ``engine`` (``add_request``, ``step``, ``abort_request``,
    ``has_unfinished_requests``) over ``jobs``; returns rows by job id and the guard record."""
    t0 = clock()
    for j in jobs:
        engine.add_request(j["job_id"], {"prompt_token_ids": j["prompt_ids"]}, make_params(j))
    cap = {j["job_id"]: int(j["max_tokens"]) for j in jobs}
    live: dict[str, int] = {j["job_id"]: 0 for j in jobs}  # current decoded length
    rows: dict[str, dict] = {}
    checks: list[dict] = []
    decoded = 0
    next_check = GUARD_EVERY
    stopped = None
    while engine.has_unfinished_requests():
        for out in engine.step():
            rid = out.request_id
            if rid not in live:
                continue
            ids = list(out.outputs[0].token_ids)
            decoded += len(ids) - live[rid]
            live[rid] = len(ids)
            if out.finished:
                rows[rid] = {"job_id": rid, "gen_ids": ids, "n_tokens": len(ids),
                             "finish_reason": out.outputs[0].finish_reason, "status": "finished",
                             "t_finish": clock() - t0}
                del live[rid]
        if deadline is not None and len(rows) >= next_check and live:
            next_check = len(rows) + GUARD_EVERY
            now = clock()
            rate = decoded / max(now - t0, 1e-9)
            lens = sorted(r["n_tokens"] for r in rows.values())
            q90 = lens[min(len(lens) - 1, int(0.9 * len(lens)))]
            remaining = sum(max(0, min(cap[r], max(q90, expected_len, n)) - n)
                            for r, n in live.items())
            proj = remaining / max(rate, 1e-9)
            ok = now + proj + after_s <= deadline
            checks.append({"finished": len(rows), "unfinished": len(live),
                           "tokens_per_second": rate, "projected_remaining_s": proj,
                           "stop": None if ok else "envelope"})
            if not ok:
                stopped = "envelope"
        if deadline is not None and live and clock() > deadline:
            stopped = "deadline"
        if stopped:
            engine.abort_request(list(live))
            for rid in list(live):
                rows[rid] = {"job_id": rid, "gen_ids": [], "n_tokens": 0, "status": "aborted",
                             "finish_reason": None, "t_finish": None}
            live.clear()
            break
    wall = clock() - t0
    return rows, {"wall_seconds": wall, "decoded_tokens": decoded,
                  "tokens_per_second": decoded / max(wall, 1e-9), "guard_checks": checks,
                  "stopped": stopped, "n_finished": sum(r["status"] == "finished"
                                                        for r in rows.values())}


def spec_counters(llm) -> dict:
    """Speculative acceptance counters from vLLM's metrics snapshot, when exposed."""
    out: dict = {}
    try:
        metrics = llm.get_metrics()
    except Exception as e:  # noqa: BLE001  (metrics are optional; the run is not)
        return {"unavailable": str(e)[:120]}
    for m in metrics:
        if m.name.startswith("vllm:spec_decode"):
            out[m.name] = getattr(m, "value", None) if not hasattr(m, "values") else m.values
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--after-h", type=float, default=0.0,
                    help="hours of later work this deadline must also cover (HF readouts)")
    ap.add_argument("--expected-len", type=int, default=0,
                    help="prior length for unfinished requests (the pilot's q90 tokens)")
    a = ap.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    engine_cfg = json.loads(a.engine.read_text())
    jobs = [json.loads(x) for x in a.jobs.read_text().splitlines() if x.strip()]
    versions = {"vllm": vllm.__version__, "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    t_load = time.time()
    llm = LLM(model=engine_cfg["model"], revision=engine_cfg.get("revision"),
              tokenizer=engine_cfg.get("tokenizer") or engine_cfg["model"],
              tokenizer_revision=engine_cfg.get("tokenizer_revision")
              or engine_cfg.get("revision"), **engine_cfg["llm_kwargs"])
    load_s = time.time() - t_load
    rows, rec = run_jobs(llm.llm_engine, jobs, lambda j: sampling_params(SamplingParams, j),
                         deadline=a.deadline, after_s=3600 * a.after_h,
                         expected_len=a.expected_len)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(rows[j["job_id"]]) + "\n")
    summary = {"label": engine_cfg.get("label"), "engine": engine_cfg, "versions": versions,
               "settings_sha256": settings_hash(engine_cfg, versions), "load_seconds": load_s,
               "n_jobs": len(jobs), **rec, "spec_counters": spec_counters(llm)}
    Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(f">> vllm {engine_cfg.get('label')}: {rec['n_finished']}/{len(jobs)} finished, "
          f"{rec['tokens_per_second']:.0f} tok/s, load {load_s:.0f}s"
          + (f", stopped ({rec['stopped']})" if rec["stopped"] else ""), flush=True)
    return 3 if rec["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
