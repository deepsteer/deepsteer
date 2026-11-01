"""MRB-G1 OpenRouter cross-check (MRB_PLAN v0.6 §5): X2, X2b, X1, and the R1-R4 readouts.

    mrb_g1_openrouter.py {generate,judge,analyze,all} [--out DIR] [--workers 16] [--dry-run]

Upstream's own calls through OpenRouter (``utils.py`` at ``e6689bb``), on J1's 50 scenarios:
X2 generates GPT-oss-20b at the paper's settings (Table 4 T 1.0 / top-p 0.95, ``max_tokens``
10,500, upstream's top-level ``reasoning_effort: "high"``); X2b adds OpenRouter's ``reasoning``
object (lengths only, MRB-A11); the judge (``openai/gpt-oss-120b``, the registered prompt, T 1,
top-p 1, effort high) grades X2's two fields and, as X1, our pod's k0 generations of both models.

The key comes from ``OPENROUTER_API_KEY`` or ``~/.config/deepsteer/openrouter_key`` and is never
printed or written. Tracked rows (``DIR``): ids, lengths, hashes, token counts, provider, verdict
strings of at most 20 characters and both credit tests. Untracked (``DIR/_raw``): generations and
judge replies. A rerun resumes; the run stops at the token budget (about $15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import mrb_g1_pod as P  # noqa: E402

from deepsteer.benchmarks.moral_reasoning.morebench.prompts import (  # noqa: E402
    generation_prompt,
    judge_prompt,
)

BASE_URL = "https://openrouter.ai/api/v1"
KEY_FILE = Path.home() / ".config" / "deepsteer" / "openrouter_key"
GEN_MODEL, JUDGE_MODEL = "openai/gpt-oss-20b", "openai/gpt-oss-120b"
# upstream _collect_openrouter_thinking: max_tokens = budget 10,000 + 500, reasoning_effort high;
# Table 4's sampling is the paper's default (v0.2)
GEN_ARMS = {"paper": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 10_500,
                      "reasoning_effort": "high"},
            "reasoning_object": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 10_500,
                                 "reasoning_effort": "high",
                                 "extra_body": {"reasoning": {"effort": "high"}}}}
# upstream get_judge_response
JUDGE_CALL = {"temperature": 1, "top_p": 1, "max_tokens": 10_500, "reasoning_effort": "high"}
# stop before about $15 at OpenRouter's gpt-oss list prices (in/out per M: 120b ~0.1/0.5)
BUDGET = {"prompt_tokens": 40_000_000, "completion_tokens": 14_000_000}
DEFAULT_OUT = REPO / "papers" / "mrb_morebench" / "outputs" / "g1_openrouter"
POD_OUT = REPO / "papers" / "mrb_morebench" / "outputs" / "g1"


def scenarios():
    return [it for it in P._items() if P.in_j1_subset(it.idx)]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Budget:
    def __init__(self, used: dict | None = None):
        self.used = dict(used or {"prompt_tokens": 0, "completion_tokens": 0})
        self.lock = threading.Lock()

    def add(self, usage: dict) -> None:
        with self.lock:
            for k in self.used:
                self.used[k] += int(usage.get(k) or 0)

    def exhausted(self) -> bool:
        return any(self.used[k] >= BUDGET[k] for k in BUDGET)


def client(dry: bool):
    if dry:
        return StubClient()
    key = os.environ.get("OPENROUTER_API_KEY") or (KEY_FILE.read_text().strip()
                                                    if KEY_FILE.exists() else "")
    if not key:
        sys.exit(f"no OpenRouter key: set OPENROUTER_API_KEY or write {KEY_FILE} (mode 600)")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=BASE_URL)


class StubClient:
    """--dry-run: answers like OpenRouter (a reasoning field, content, usage, provider)."""

    class _NS:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def __init__(self):
        self.chat = self._NS(completions=self._NS(create=self._create))

    def _create(self, model, messages, **kw):
        judge = model == JUDGE_MODEL
        msg = self._NS(content="yes" if judge else "A response.", reasoning="Some reasoning.")
        usage = self._NS(prompt_tokens=100, completion_tokens=10, completion_tokens_details=None,
                         model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 10})
        return self._NS(choices=[self._NS(message=msg, finish_reason="stop")], usage=usage,
                        provider="stub", model=model)


def call(cl, model: str, prompt: str, params: dict) -> dict:
    """One chat completion with upstream's retry on an empty reply (3) and backoff on errors."""
    kw = {k: v for k, v in params.items() if k != "extra_body"}
    extra = params.get("extra_body")
    last = None
    for attempt in range(6):
        try:
            r = cl.chat.completions.create(model=model,
                                           messages=[{"role": "user", "content": prompt}],
                                           **kw, **({"extra_body": extra} if extra else {}))
            msg = r.choices[0].message
            content = msg.content
            if content is None and attempt < 3:
                continue
            u = r.usage
            det = getattr(u, "completion_tokens_details", None)
            return {"content": content, "reasoning": getattr(msg, "reasoning", None) or "",
                    "finish_reason": r.choices[0].finish_reason,
                    "provider": getattr(r, "provider", None),
                    "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                    "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None}
        except Exception as e:  # noqa: BLE001 - rate limits and transient 5xx are retried
            last = e
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"OpenRouter call failed after retries: {type(last).__name__}")


def run_jobs(jobs: list[dict], fn, out_path: Path, raw_path: Path, workers: int,
             budget: Budget) -> int:
    """Run ``fn(job)`` -> (tracked, raw) for jobs not yet in ``out_path``; append as they finish."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(x)["job_id"] for x in out_path.read_text().splitlines() if x.strip()}
    todo = [j for j in jobs if j["job_id"] not in done]
    lock, n = threading.Lock(), 0

    def one(j):
        nonlocal n
        if budget.exhausted():
            return
        tracked, raw = fn(j)
        budget.add(tracked)
        with lock, open(out_path, "a") as fo, open(raw_path, "a") as fr:
            fo.write(json.dumps({"job_id": j["job_id"], **tracked}) + "\n")
            fr.write(json.dumps({"job_id": j["job_id"], **raw}) + "\n")
            n += 1
            if n % 250 == 0:
                print(f"  {out_path.name}: {n}/{len(todo)} {budget.used}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    return n


def cmd_generate(a, cl, budget) -> None:
    jobs = [{"job_id": f"gpt_oss_20b_or/{arm}/{it.scenario_id}/k0", "arm": arm,
             "prompt": generation_prompt(it.dilemma)} for arm in GEN_ARMS for it in scenarios()]

    def fn(j):
        r = call(cl, GEN_MODEL, j["prompt"], GEN_ARMS[j["arm"]])
        trace, resp = r["reasoning"], r["content"] or ""
        return ({"arm": j["arm"], "trace_chars": len(trace), "response_chars": len(resp),
                 "trace_sha256": sha(trace), "response_sha256": sha(resp),
                 "finish_reason": r["finish_reason"], "provider": r["provider"],
                 "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                 "reasoning_tokens": r["reasoning_tokens"]},
                {"trace": trace, "response": resp})

    n = run_jobs(jobs, fn, a.out / "gen.jsonl", a.out / "_raw" / "gen.jsonl", a.workers, budget)
    print(f">> generate: {n} new, budget {budget.used}")


def judge_units(out: Path) -> list[dict]:
    """X2's two fields (the paper arm only) and X1: our pod's k0 generations, both models."""
    sids = {it.scenario_id for it in scenarios()}
    units = []
    raw = out / "_raw" / "gen.jsonl"
    for line in (raw.read_text().splitlines() if raw.exists() else []):
        r = json.loads(line)
        if r["job_id"].split("/")[1] == "paper":
            for f in ("trace", "response"):
                units.append({"unit": f"{r['job_id']}/{f}", "text": r[f]})
    for m in P.GEN:
        for line in (POD_OUT / "_raw" / "gen" / f"{m}.jsonl").read_text().splitlines():
            r = json.loads(line)
            _, arm, sid, kk = r["job_id"].split("/")
            if arm == "default" and kk == "k0" and sid in sids:
                for f in ("trace", "response"):
                    units.append({"unit": f"{r['job_id']}/{f}", "text": r[f]})
    return units


def cmd_judge(a, cl, budget) -> None:
    by_sid = {it.scenario_id: it for it in scenarios()}
    jobs = []
    for u in judge_units(a.out):
        it = by_sid[u["unit"].split("/")[2]]
        for c in it.criteria:
            jobs.append({"job_id": f"{u['unit']}/{c.criterion_id}", "unit": u["unit"],
                         "criterion_id": c.criterion_id, "weight": c.weight,
                         "prompt": judge_prompt(u["text"], c.title)})

    def fn(j):
        r = call(cl, JUDGE_MODEL, j["prompt"], JUDGE_CALL)
        final = r["content"]
        short = final.strip().lower() if final is not None and len(final) <= 20 else None
        return ({"unit": j["unit"], "criterion_id": j["criterion_id"], "weight": j["weight"],
                 "verdict": short, "verdict_chars": None if final is None else len(final),
                 **P.credits(final, j["weight"]), "finish_reason": r["finish_reason"],
                 "provider": r["provider"], "prompt_tokens": r["prompt_tokens"],
                 "completion_tokens": r["completion_tokens"]},
                {"content": final, "reasoning": r["reasoning"]})

    n = run_jobs(jobs, fn, a.out / "judge.jsonl", a.out / "_raw" / "judge.jsonl", a.workers,
                 budget)
    print(f">> judge: {n} new of {len(jobs)}, budget {budget.used}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["generate", "judge", "analyze", "all"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    used = {"prompt_tokens": 0, "completion_tokens": 0}
    for name in ("gen.jsonl", "judge.jsonl"):  # resume counts what was already spent
        p = a.out / name
        for line in (p.read_text().splitlines() if p.exists() else []):
            r = json.loads(line)
            for k in used:
                used[k] += int(r.get(k) or 0)
    budget = Budget(used)
    if a.stage in ("generate", "judge", "all"):
        cl = client(a.dry_run)
        if a.stage in ("generate", "all"):
            cmd_generate(a, cl, budget)
        if a.stage in ("judge", "all"):
            cmd_judge(a, cl, budget)
    if budget.exhausted():
        print(">> BUDGET reached; rerun to resume after raising it under amendment")
    if a.stage in ("analyze", "all"):
        import mrb_g1_openrouter_analyze as Z
        Z.main_from(a.out)


if __name__ == "__main__":
    main()
