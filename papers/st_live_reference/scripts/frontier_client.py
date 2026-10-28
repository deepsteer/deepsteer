#!/usr/bin/env python3
"""The frontier exhibit's API client (ST_PLAN Amendment 56 item 4; reference only).

H1's agent frame on a closed model through its API: the single user message
``letter_chat_messages(s, order, "agent", "neutral", "primary")`` builds, letters permuted by
``assign_letters(s, k)``, the reply parsed by ``kdg_harness.parse_response`` on the visible text.

- **Arms** (item 1): ``default`` (effort ``medium``, each provider's default, sent explicitly) and
  ``minimal`` (the provider floor, ``low`` on both; neither API has a reasoning-off mode).
- **Parity rules**: pinned model ids; the reasoning setting recorded per call; no temperature sent
  (neither API accepts one with reasoning on), recorded as the provider default; no refusal
  fallback (it would change the model mid-measurement).
- **Accounting**: per call, input, output and reasoning tokens, and the cost at the price table's
  rates.
- **Hard cap**: before each call, spend plus the projected call (the largest per-call cost seen,
  or the prior until one is seen) must stay within the cap; otherwise the run stops and records
  where.
- **Resume**: rows are saved per call; a rerun skips every (scenario, rollout, arm) already saved
  and starts its spend from theirs.
- **Keys** come from the environment (the SDKs' own resolution) on the author's machine only; this
  module never reads, prints or stores a key.

    python3 frontier_client.py --model gpt-6.1-sol --k 8 --cap-usd 100 --out <outputs/frontier/f1>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts"), str(HERE.parents[2])]

# Published rates per million tokens, checked on the providers' pages 2026-10-27 (Amendment 56
# item 5). Reasoning tokens are billed as output on both.
RATES = {"claude-opus-5-5": {"input": 4.00, "cached_input": 0.20, "output": 20.00},
         "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00}}
ARMS = {"default": "medium", "minimal": "low",  # the floor is "low" on both (item 1's type block)
        "high": "high",  # Amendment 56's addition: GPT-6.1 Sol's high-effort arm
        "forced": "max",  # its second addition: Opus 5.5 at its thinking ceiling (no budget)
        "xhigh": "xhigh"}  # its third addition: Sol's fourth ladder point
F1_ARMS = ("default", "minimal")
MAX_OUTPUT = 16000
MAX_OUTPUT_STREAM = 64000  # efforts at or above xhigh stream (the API's guidance for long outputs)
STREAM_EFFORTS = ("xhigh", "max")


@dataclasses.dataclass
class Call:
    """One provider call's result, provider-neutral. ``input_tokens`` is the total prompt,
    cached tokens included (OpenAI reports it so; Anthropic's ``input_tokens`` excludes cache reads
    and writes, and the adapter adds them back)."""

    text: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None
    refused: bool
    stop: str
    model_returned: str
    request_id: str | None


def cost_usd(model: str, c: Call) -> float:
    r = RATES[model]
    fresh = max(c.input_tokens - c.cached_input_tokens, 0)
    return (fresh * r["input"] + c.cached_input_tokens * r["cached_input"]
            + c.output_tokens * r["output"]) / 1e6


class AnthropicAdapter:
    """Claude through the Messages API. Thinking stays adaptive (Opus 5.5 rejects disabled
    thinking at every effort); the arm sets ``output_config.effort``."""

    def __init__(self, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def call(self, model: str, messages: list[dict], effort: str) -> Call:
        kw = dict(model=model, messages=messages, thinking={"type": "adaptive"},
                  output_config={"effort": effort})
        if effort in STREAM_EFFORTS:
            with self.client.messages.stream(max_tokens=MAX_OUTPUT_STREAM, **kw) as st:
                r = st.get_final_message()
        else:
            r = self.client.messages.create(max_tokens=MAX_OUTPUT, **kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        u = r.usage
        det = getattr(u, "output_tokens_details", None)
        read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        return Call(text=text, input_tokens=int(u.input_tokens) + read + write,
                    cached_input_tokens=read,
                    output_tokens=int(u.output_tokens),
                    reasoning_tokens=getattr(det, "thinking_tokens", None) if det else None,
                    refused=r.stop_reason == "refusal", stop=str(r.stop_reason),
                    model_returned=str(r.model), request_id=getattr(r, "_request_id", None))


class OpenAIAdapter:
    """GPT through the Responses API; the arm sets ``reasoning.effort``; nothing is stored."""

    def __init__(self, client=None):
        if client is None:
            import openai

            client = openai.OpenAI()
        self.client = client

    def call(self, model: str, messages: list[dict], effort: str) -> Call:
        r = self.client.responses.create(model=model, input=messages,
                                         reasoning={"effort": effort},
                                         max_output_tokens=MAX_OUTPUT, store=False)
        text, refused = "", False
        for item in r.output or []:
            if getattr(item, "type", None) != "message":
                continue
            for part in item.content or []:
                if getattr(part, "type", None) == "output_text":
                    text += part.text
                elif getattr(part, "type", None) == "refusal":
                    refused = True
        u = r.usage
        din = getattr(u, "input_tokens_details", None)
        dout = getattr(u, "output_tokens_details", None)
        stop = str(r.status) + (f":{r.incomplete_details.reason}"
                                if getattr(r, "incomplete_details", None) else "")
        return Call(text=text, input_tokens=int(u.input_tokens),
                    cached_input_tokens=int(getattr(din, "cached_tokens", 0) or 0) if din else 0,
                    output_tokens=int(u.output_tokens),
                    reasoning_tokens=getattr(dout, "reasoning_tokens", None) if dout else None,
                    refused=refused, stop=stop, model_returned=str(r.model),
                    request_id=getattr(r, "_request_id", None))


def adapter_for(model: str, client=None):
    if model.startswith("claude-"):
        return AnthropicAdapter(client)
    if model.startswith("gpt-"):
        return OpenAIAdapter(client)
    raise SystemExit(f"no adapter for {model}")


class CapReached(Exception):
    pass


RATE_WAIT_S, RATE_TRIES = 60.0, 30  # after the SDK's own retries: wait and retry, never skip


def _is_rate_limit(e: Exception) -> bool:
    """Both SDKs name their 429 class ``RateLimitError``; matched by name so neither SDK is
    imported here."""
    return type(e).__name__ == "RateLimitError"


def call_paced(adapter, model: str, messages: list[dict], effort: str,
               sleep=time.sleep) -> tuple[Call, int]:
    """The adapter's call, retried after ``RATE_WAIT_S`` on a rate limit up to ``RATE_TRIES``
    times; any other error propagates (the run stops and resumes from the saved rows)."""
    for i in range(RATE_TRIES):
        try:
            return adapter.call(model, messages, effort), i
        except Exception as e:  # noqa: BLE001  (re-raised unless it is a rate limit)
            if not _is_rate_limit(e) or i == RATE_TRIES - 1:
                raise
            sleep(RATE_WAIT_S)
    raise RuntimeError("unreachable")


class Runner:
    """Runs jobs against one pinned model with per-call saving, accounting and the hard cap."""

    def __init__(self, model: str, adapter, out: Path, cap_usd: float, prior_call_usd: float):
        if model not in RATES:
            raise SystemExit(f"{model} is not in the price table")
        self.model, self.adapter, self.cap = model, adapter, float(cap_usd)
        self.rows_path = out / f"{model}.jsonl"
        out.mkdir(parents=True, exist_ok=True)
        self.done: set[tuple] = set()
        self.spent, self.max_call = 0.0, float(prior_call_usd)
        if self.rows_path.exists():
            for line in self.rows_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done.add((r["scenario_id"], r["rollout"], r["arm"]))
                    self.spent += r["cost_usd"]
                    self.max_call = max(self.max_call, r["cost_usd"])

    def run_one(self, job: dict) -> dict | None:
        """One call, or ``None`` if already saved; raises ``CapReached`` before a call that the
        projection says would pass the cap."""
        from deepsteer.kdg.harness import KDG_HARNESS_VERSION, parse_response

        key = (job["scenario"].id, job["k"], job["arm"])
        if key in self.done:
            return None
        if self.spent + self.max_call > self.cap:
            raise CapReached(f"spent ${self.spent:.4f} + projected ${self.max_call:.4f} > cap "
                             f"${self.cap:.2f}")
        effort = ARMS[job["arm"]]
        t0 = time.time()
        c, waits = call_paced(self.adapter, self.model, job["messages"], effort)
        p = parse_response(c.text, job["scenario"], job["order"])
        cost = cost_usd(self.model, c)
        row = {"scenario_id": key[0], "rollout": key[1], "arm": key[2], "model": self.model,
               "model_returned": c.model_returned, "effort": effort,
               "temperature": "provider default (not sent)", "letter": p.letter,
               "option_id": p.option_id, "norm_status": p.norm_status, "parse_method": p.method,
               "refused": c.refused, "stop": c.stop, "text": c.text,
               "order": {L: o.option_id for L, o in job["order"]},
               "input_tokens": c.input_tokens, "cached_input_tokens": c.cached_input_tokens,
               "output_tokens": c.output_tokens, "reasoning_tokens": c.reasoning_tokens,
               "cost_usd": cost, "seconds": time.time() - t0, "rate_limit_waits": waits,
               "request_id": c.request_id,
               "harness_version": KDG_HARNESS_VERSION,
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self.rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.done.add(key)
        self.spent += cost
        self.max_call = max(self.max_call, cost)
        return row


def screen_jobs(k: int, arms: tuple[str, ...] = F1_ARMS) -> list[dict]:
    """The 60 screen scenarios (``h1_train_order``'s first 60 in the kept set) x k rollouts x
    both arms, interleaved by rollout so a cap stop leaves both arms balanced."""
    import pod_st_screen as P
    import screen_lib as L
    from kdg_pod_lib import letter_chat_messages

    from deepsteer.kdg.schema import assign_letters, load_scenario_dir

    split = json.loads(P.SPLIT.read_text())
    kept = set(json.loads(P.H1_KEPT.read_text())["ids"])
    H = {s.id: s for s in load_scenario_dir([P.H1_SET])[0] if s.id in kept}
    ids = [s for s in L.h1_train_order(split)[: P.SCREEN_N["muse_glimmer_30b"]] if s in H]
    jobs = []
    for k_ in range(k):
        for sid in ids:
            order = assign_letters(H[sid], k_)
            msgs = letter_chat_messages(H[sid], order, "agent", "neutral", "primary")
            for arm in arms:
                jobs.append({"scenario": H[sid], "k": k_, "arm": arm, "order": order,
                             "messages": msgs})
    return jobs


def run(runner: Runner, jobs: list[dict]) -> dict:
    n = 0
    try:
        for j in jobs:
            if runner.run_one(j) is not None:
                n += 1
    except CapReached as e:
        return {"stopped": "cap", "reason": str(e), "calls_this_run": n,
                "spent_usd": runner.spent}
    return {"stopped": "complete", "calls_this_run": n, "spent_usd": runner.spent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(RATES))
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--prior-call-usd", type=float, required=True,
                    help="the price table's high per-call cost, the projection before any call")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(F1_ARMS),
                    help="comma-separated arms (F1: default,minimal; later: high, forced, xhigh)")
    a = ap.parse_args()
    r = Runner(a.model, adapter_for(a.model), a.out, a.cap_usd, a.prior_call_usd)
    arms = tuple(a.arms.split(","))
    if any(x not in ARMS for x in arms):
        raise SystemExit(f"unknown arm in {arms}")
    res = run(r, screen_jobs(a.k, arms))
    tag = "" if arms == F1_ARMS else "." + "_".join(arms)
    (a.out / f"{a.model}{tag}.run.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
