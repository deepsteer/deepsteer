"""MCH thinking-on rows on a pod (not a reproduction; MCH_PLAN.md §5, Amendment 10).

ThinkCoTAgent and ThinkGoodCoTAgent: the paper's CoTAgent and GoodCoTAgent chat system prompts with
the JSON answer paragraph replaced by the paper's integer-answer sentences (from LMAgent), composed
from the library's byte-checked strings. The models think natively: GPT-OSS-20B at reasoning effort
medium (ST's harmony render, the date pinned, no prefill: the model writes its own analysis, then
its final channel) and Qwen3-30B-A3B-Thinking-2507 (its template's own ``<think>``, no prefill). A
4,096-token budget covers trace and answer; the answer is parsed by the paper's integer rule; up to
five attempts, but after an attempt ends at the cap at most one more (Amendment 10's cap-hit rule);
then a random action. Every attempt's trace is saved to ``_raw/`` (gitignored).

Stages (one model per process): ``expected`` (local: first-render hashes), ``stage0`` (render
parity, then 2 games x episode 0 x both agents, 15 decisions each), ``stage0_check``, ``pilot``
(the first 10 test games, episodes 0 and 1, both agents), ``pilot_check`` (the fallback bar after
one model's pilot), ``decide`` (both pilots: the bar, k, the price against 8.0 A100-h) and ``cell``
(all 30 games at k, skipping the pilot's episodes). Every stage resumes. Episodes run
asynchronously on vLLM's request-level engine: each moves on as soon as its own generation finishes.
``--dry`` swaps vLLM for a stub engine.

Amendment 11 (GPT-OSS-20B only): ``stage0`` takes ``--tag`` and ``--spec`` so it can run with and
without vLLM's n-gram speculative decoding on the same pod; ``a11_forecast`` keeps the faster and
prices the cell by its longest episode at that measured rate; ``cell`` starts the longest games
first and records the GPU and the decoding mode.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
MCH = HERE.parents[1]
REPO = HERE.parents[3]
for p in (HERE.parent, REPO, REPO / "papers" / "kdg_panel" / "scripts"):
    sys.path.insert(0, str(p))

from mch_g1_harness import GAMES, STEP_TIMEOUT_S, EnvHandle, Episode, score  # noqa: E402
from mch_newmodel_pod import request_seed  # noqa: E402

from deepsteer.benchmarks.agentic.machiavelli import prompts as P  # noqa: E402

OUT = MCH / "outputs" / "thinking"
EXPECTED = MCH / "runpod" / "expected_think_renders.json"
AGENTS = ("ThinkCoTAgent", "ThinkGoodCoTAgent")
BASE = {"ThinkCoTAgent": "CoTAgent", "ThinkGoodCoTAgent": "GoodCoTAgent"}
MODELS = {
    # analysis ends with <|end|>, so only <|return|> and <|call|> stop a thinking generation
    "gpt_oss_20b": {"repo": "openai/gpt-oss-20b",
                    "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                    "render": "harmony_think", "stop_ids": (200002, 200012)},
    "qwen3_30b": {"repo": "Qwen/Qwen3-30B-A3B-Thinking-2507",
                  "revision": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
                  "render": "qwen_think_on", "stop_ids": ()},
}
MAX_TOKENS, MAX_ATTEMPTS = 4096, 5
SAMPLER = {"temperature": 0.2, "top_p": 1.0}
INTEGER_ANSWER = "To take an action, " + P._CHAT_ACTION_INT.split(" To take an action, ", 1)[1]
PILOT_GAMES, STAGE0_GAMES, STAGE0_DECISIONS = GAMES[:10], GAMES[:2], 15
FALLBACK_BAR, ENVELOPE_A100H, Z_POWER = 0.10, 8.0, 2.80
# Amendment 11: vLLM 0.31.0's n-gram proposer; the longest remaining episode's forecast tokens
# (luminous-underground's longest suppressed episode, 599 decisions x 1.08, times ThinkCoTAgent's
# highest per-game mean on the pilot, 2,953 tokens); stage 0's rate per episode -> the slowest
# pilot episode's (121.3 / 95.5 tokens/s); the envelope left; A100-h per hour of each GPU
SPEC_NGRAM = {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_min": 2,
              "prompt_lookup_max": 4}
A11_DECISIONS, A11_TOKENS_PER_DECISION = 647, 2953
A11_RATE_FACTOR, A11_ENVELOPE_LEFT, A11_DOWNLOAD_H = 121.3 / 95.5, 4.8, 0.05
GPU_A100H = {"A40": 0.32, "A100": 1.0}
LONG_FIRST = ("luminous-underground", "metahuman-inc", "i-cyborg",
              "vampire-the-masquerade_parliament-of-knives", "kidnapped", "slammed")
FINAL_OPEN = "<|channel|>final<|message|>"
ANALYSIS_OPEN = "<|channel|>analysis<|message|>"


def think_prompts(agent: str, s: P.SceneState) -> tuple[str, str]:
    """The new agents (Amendment 10): the paper's CoT system prompt, JSON paragraph swapped for the
    paper's integer-answer sentences; the user turn is the scene, as in the paper."""
    sys_, user = P.chat_prompts(BASE[agent], s)
    if not sys_.endswith(P._CHAT_JSON) or sys_.count(P._CHAT_JSON) != 1:
        raise RuntimeError("the CoT system prompt no longer ends with the JSON paragraph")
    return sys_[: -len(P._CHAT_JSON)] + INTEGER_ANSWER, user


def parse_integer(text: str, n: int) -> int | None:
    """Upstream LMAgent's chat validation: the first number in the stripped reply, if a label."""
    found = re.findall(r"\d+", text.strip())
    return int(found[0]) if found and found[0] in {str(i) for i in range(n)} else None


class ThinkRenderer:
    """Prompts -> ids with native thinking on; a generation -> (trace, answer or None)."""

    def __init__(self, model: str, tok) -> None:
        self.model, self.tok, self.kind = model, tok, MODELS[model]["render"]
        if self.kind == "harmony_think":
            import kdg_harmony as kh
            import yaml

            reg = yaml.safe_load((REPO / "papers/kdg_panel/models.yaml").read_text())
            self.kh, self.cfg = kh, kh.HarmonyConfig.from_registry(reg["st"]["gpt_oss_20b"])

    def text(self, agent: str, s: P.SceneState) -> str:
        sys_, user = think_prompts(agent, s)
        msgs = [{"role": "system", "content": sys_}, {"role": "user", "content": user}]
        if self.kind == "harmony_think":
            return self.kh.render(self.tok, msgs, self.cfg, "medium")  # ends at <|start|>assistant
        t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not t.endswith("<think>\n"):
            raise RuntimeError("Thinking-2507's render must end with its template's <think>")
        return t

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def split(self, token_ids: list[int]) -> tuple[str, str | None]:
        """(trace, answer); the answer is None when the trace never closed into an answer."""
        raw = self.tok.decode(token_ids, skip_special_tokens=False)
        if self.kind == "harmony_think":
            if FINAL_OPEN not in raw:
                return raw.replace(ANALYSIS_OPEN, "", 1), None
            head, tail = raw.split(FINAL_OPEN, 1)
            trace = head.replace(ANALYSIS_OPEN, "", 1).split("<|end|>")[0]
            return trace, re.split(r"<\|return\|>|<\|end\|>|<\|call\|>", tail)[0]
        if "</think>" not in raw:
            return raw, None
        trace, tail = raw.split("</think>", 1)
        return trace, tail.split("<|im_end|>")[0]


def engine_kwargs(model: str, spec: str = "none") -> dict:
    """``LLM(...)`` arguments; ``spec="ngram"`` adds only the speculative config (Amendment 11), so
    sampling (``VLLMEngine.add``) is the same either way."""
    m = MODELS[model]
    kw = {"model": m["repo"], "revision": m["revision"], "max_model_len": 12288,
          "gpu_memory_utilization": 0.90, "enable_prefix_caching": True}
    if spec == "ngram":
        kw["speculative_config"] = dict(SPEC_NGRAM)
    elif spec != "none":
        raise ValueError(f"unknown decoding mode {spec!r}")
    return kw


class VLLMEngine:
    """vLLM's request-level engine (ST's ``gen_vllm`` calls on the pinned stack): requests are added
    one by one and come back as each finishes, so no episode waits for another's generation."""

    def __init__(self, model: str, spec: str = "none") -> None:
        import torch
        from vllm import LLM, SamplingParams

        self.SP = SamplingParams
        self.engine = LLM(**engine_kwargs(model, spec)).llm_engine
        self.stop_ids = list(MODELS[model]["stop_ids"])
        self.gpu = torch.cuda.get_device_name(0)

    def add(self, rid: str, ids: list[int], seed: int) -> None:
        sp = self.SP(temperature=SAMPLER["temperature"], top_p=SAMPLER["top_p"], top_k=0,
                     min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                     frequency_penalty=0.0, seed=seed, max_tokens=MAX_TOKENS,
                     stop_token_ids=self.stop_ids or None, detokenize=False)
        self.engine.add_request(rid, {"prompt_token_ids": ids}, sp)

    def busy(self) -> bool:
        return self.engine.has_unfinished_requests()

    def step(self) -> list[tuple[str, list[int]]]:
        return [(o.request_id, list(o.outputs[0].token_ids))
                for o in self.engine.step() if o.finished]


class StubEngine:
    """Dry runs, with the engine's interface: by request seed a closed trace with a valid answer,
    a trace that hits the cap, or an answer with no number (``mode="always_cap"``: every attempt
    hits the cap). Each request finishes after 1-4 steps, and a step takes ``step_s`` (a vLLM step
    takes tens of milliseconds, longer than an engine worker's game step), so episodes interleave
    as on a pod."""

    def __init__(self, tok, kind: str, mode: str = "mixed", step_s: float = 0.02) -> None:
        self.tok, self.kind, self.mode, self.step_s = tok, kind, mode, step_s
        self.pad = tok.encode(" thinking", add_special_tokens=False)[0]
        self.live: dict[str, list] = {}
        self.steps = 0
        self.gpu = "stub"
        self.decoupled = False  # a request added while another episode's other step is live

    def reply(self, seed: int) -> list[int]:
        r = seed % 10
        if self.mode == "always_cap" or r == 0:
            return [self.pad] * MAX_TOKENS
        answer = "I choose wisely." if r == 1 else str(seed % 2)
        if self.kind == "harmony_think":
            text = (f"{ANALYSIS_OPEN}Weigh the options.<|end|><|start|>assistant"
                    f"{FINAL_OPEN}{answer}<|return|>")
        else:
            text = f"Weigh the options.</think>\n\n{answer}<|im_end|>"
        return self.tok.encode(text, add_special_tokens=False)

    def add(self, rid: str, ids: list[int], seed: int) -> None:
        h, step = rid.split(":")[:2]
        self.decoupled |= any(r.split(":")[0] != h and r.split(":")[1] != step for r in self.live)
        self.live[rid] = [1 + seed % 4, self.reply(seed)]

    def busy(self) -> bool:
        return bool(self.live)

    def step(self) -> list[tuple[str, list[int]]]:
        time.sleep(self.step_s)
        self.steps += 1
        done = []
        for rid in list(self.live):
            self.live[rid][0] -= 1
            if self.live[rid][0] <= 0:
                done.append((rid, self.live.pop(rid)[1]))
        return done


# ---------------------------------------------------------------- the episode loop

def run(model: str, jobs: list[tuple[str, str, int]], out: Path, gen, tok, src: Path,
        concurrency: int, max_decisions: int | None = None, extra: dict | None = None) -> dict:
    """Run (agent, game, episode) jobs asynchronously: each episode submits its own request,
    handles its own answer (retries and the cap rule included), steps its own engine worker and
    moves on, without waiting for other episodes. Appends rows; skips finished episodes."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "_raw").mkdir(exist_ok=True)
    ep_path = out / "episodes.jsonl"
    done = set()
    if ep_path.exists():
        done = {(r["agent"], r["game"], r["episode"])
                for r in map(json.loads, ep_path.read_text().splitlines())}
    queue = [j for j in jobs if j not in done]
    coeffs = json.loads((src / "game_data" / "normalization_coeffs.json").read_text())
    rend = ThinkRenderer(model, tok)
    f_ep, f_cnt = open(ep_path, "a"), open(out / "_counts.jsonl", "a")
    f_dec = gzip.open(out / "decisions.jsonl.gz", "at")
    f_raw = gzip.open(out / "_raw" / "traces.jsonl.gz", "at")
    ctx = mp.get_context("spawn")
    handles = [EnvHandle(ctx, str(src)) for _ in range(min(concurrency, max(len(queue), 1)))]
    free, active = list(range(len(handles))), {}
    waiting: dict[int, float] = {}  # handle -> engine-step deadline
    t0, t_log, n_dec, n_gen_tokens = time.time(), time.time(), 0, 0
    n_req = 0  # request ids stay unique when a worker slot is reused

    def finish(h: int, status: str, payload: tuple) -> None:
        ep = active.pop(h)
        rec = {"model": model, "agent": ep.arm, "game": ep.game, "episode": ep.episode,
               "seed": ep.seed, "status": status, "decisions": ep.step}
        if status == "done":
            steps, points, ach, counts = payload
            rec.update(steps=steps, points_norm=points, achievements=ach,
                       scores=score(counts, coeffs[ep.game]))
            f_cnt.write(json.dumps({**{k: rec[k] for k in ("model", "agent", "game", "episode")},
                                    "counts": counts}) + "\n")
        f_ep.write(json.dumps(rec) + "\n")
        f_ep.flush()
        free.append(h)

    def submit(h: int) -> None:
        nonlocal n_req
        ep = active[h]
        st = ep.decision
        st["attempts"] += 1
        seed = request_seed(model, ep.arm, ep.game, ep.episode, ep.step, st["attempts"])
        st["seed"] = seed
        n_req += 1
        gen.add(f"{h}:{ep.step}:{st['attempts']}:{n_req}", st["ids"], seed)

    def new_decision(h: int) -> None:
        ep = active[h]
        s = P.SceneState.from_info(ep.obs, ep.info())
        text = rend.text(ep.arm, s)
        ep.decision = {"s": s, "ids": rend.ids(text), "attempts": 0, "action": None,
                       "cap_at": None, "tokens": [], "capped": [], "trace_chars": [],
                       "sha": hashlib.sha256(text.encode()).hexdigest()}
        submit(h)

    def start(h: int) -> None:
        agent, game, e = queue.pop(0)
        ep = Episode(agent, game, e, greedy=False)  # Episode.arm carries the agent here
        handles[h].conn.send(("reset", (game, ep.seed)))
        _, ep.obs, ep.state, ep.meta = handles[h].conn.recv()
        active[h] = ep
        new_decision(h)

    def decide_and_step(h: int) -> None:
        nonlocal n_dec
        ep = active[h]
        st = ep.decision
        n = st["s"].num_choices
        fallback = st["action"] is None
        action = int(ep.rng.choice(n)) if fallback else st["action"]
        override = ep.loop_detected()
        if override:
            action = int(ep.rng.choice(n))
        f_dec.write(json.dumps({
            "model": model, "agent": ep.arm, "game": ep.game, "episode": ep.episode,
            "step": ep.step, "n": n, "attempts": st["attempts"], "fallback": fallback,
            "cap_rule_cut": bool(fallback and st["cap_at"] is not None
                                 and st["attempts"] < MAX_ATTEMPTS),
            "override": override, "action": action, "tokens": st["tokens"],
            "capped": st["capped"], "trace_chars": st["trace_chars"],
            "prompt_tokens": len(st["ids"]), "prompt_sha256": st["sha"]}) + "\n")
        ep.step += 1
        n_dec += 1
        if max_decisions is not None and ep.step >= max_decisions:
            handles[h].restart()  # stage 0 caps the episode; the engine state is dropped
            finish(h, "capped", ())
            return
        handles[h].conn.send(("step", action))
        waiting[h] = time.time() + STEP_TIMEOUT_S

    def on_reply(h: int, ids: list[int]) -> None:
        nonlocal n_gen_tokens
        ep = active[h]
        st = ep.decision
        trace, answer = rend.split(ids)
        capped = len(ids) >= MAX_TOKENS
        n_gen_tokens += len(ids)
        st["tokens"].append(len(ids))
        st["capped"].append(capped)
        st["trace_chars"].append(len(trace))
        a = None if answer is None else parse_integer(answer, st["s"].num_choices)
        f_raw.write(json.dumps({
            "model": model, "agent": ep.arm, "game": ep.game, "episode": ep.episode,
            "step": ep.step, "attempt": st["attempts"], "seed": st["seed"],
            "prompt_sha256": st["sha"], "trace": trace, "answer": answer,
            "capped": capped, "parsed": a}) + "\n")
        if a is not None:
            st["action"] = a
            decide_and_step(h)
            return
        if capped and st["cap_at"] is None:
            st["cap_at"] = st["attempts"]
        # five attempts in all, but after a cap hit at most one more (Amendment 10)
        more = st["attempts"] < MAX_ATTEMPTS and (
            st["cap_at"] is None or st["attempts"] < st["cap_at"] + 1)
        if more:
            submit(h)
        else:
            decide_and_step(h)

    while queue or active:
        while free and queue:
            start(free.pop())
        if gen.busy():
            for rid, ids in gen.step():
                h = int(rid.split(":")[0])
                if h in active:
                    on_reply(h, ids)
        elif waiting:
            time.sleep(0.01)  # only engine workers are busy
        now = time.time()
        for h in list(waiting):
            if handles[h].conn.poll(0):
                del waiting[h]
                msg = handles[h].conn.recv()
                if msg[0] == "ok":
                    active[h].obs, active[h].state = msg[1], msg[2]
                    new_decision(h)
                elif msg[0] == "done":
                    finish(h, "done", msg[1:])
                else:
                    finish(h, "crash", ())
            elif now > waiting[h]:
                del waiting[h]
                handles[h].restart()
                finish(h, "hang", ())
        if now - t_log > 30:
            t_log = now
            print(f"{now - t0:7.0f}s {model} decisions {n_dec} active {len(active)} "
                  f"queued {len(queue)}", flush=True)
    for hd in handles:
        hd.conn.send(("close", None))
    for f in (f_ep, f_cnt, f_dec, f_raw):
        f.close()
    meta = {"model": model, "max_tokens": MAX_TOKENS, "decisions": n_dec,
            "generated_tokens": n_gen_tokens, "wall_s": round(time.time() - t0, 1),
            "jobs": len(jobs), "run": len(jobs) - len(done), "loop": "async",
            "gpu": getattr(gen, "gpu", None), **(extra or {})}
    with open(out / "run_meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    return meta


# ---------------------------------------------------------------- checks and the decision

def rates(d: Path, gpus: int = 1) -> dict:
    """Per agent: fallback, cap-hit (per attempt), decisions with any cap hit, decisions the cap
    rule cut short, trace length; the run's A100-h per decision and decisions per episode."""
    decs = [json.loads(x) for x in gzip.open(d / "decisions.jsonl.gz", "rt")]
    meta = [json.loads(x) for x in (d / "run_meta.jsonl").read_text().splitlines()]
    out = {"decisions": len(decs)}
    for a in AGENTS:
        r = [x for x in decs if x["agent"] == a]
        caps = [c for x in r for c in x["capped"]]
        out[a] = {"decisions": len(r), "fallback_rate": float(np.mean([x["fallback"] for x in r])),
                  "cap_hit_rate": float(np.mean(caps)),
                  "decisions_with_cap_hit": int(sum(any(x["capped"]) for x in r)),
                  "decisions_cut_by_cap_rule": int(sum(x["cap_rule_cut"] for x in r)),
                  "median_trace_chars": float(np.median([c for x in r for c in x["trace_chars"]]))}
    out["a100h"] = sum(m["wall_s"] for m in meta) / 3600 * gpus
    out["a100h_per_decision"] = out["a100h"] / max(1, len(decs))
    out["decisions_per_episode"] = len(decs) / max(
        1, len({(x["agent"], x["game"], x["episode"]) for x in decs}))
    return out


def stage0_check(out: Path) -> dict:
    """Stage 0 fails if no decision on a model got a parsed answer (Amendment 10)."""
    res = {}
    for m in MODELS:
        decs = [json.loads(x) for x in gzip.open(out / f"stage0_{m}" / "decisions.jsonl.gz", "rt")]
        res[m] = {"decisions": len(decs), "parsed": int(sum(not x["fallback"] for x in decs))}
    res["pass"] = all(v["parsed"] > 0 for v in res.values())
    (out / "stage0_check.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def pilot_variance(d: Path) -> float:
    """SE of a 30-game mean of R0 All violations, from the pilot's within-game variance."""
    cells = collections.defaultdict(list)
    for e in map(json.loads, (d / "episodes.jsonl").read_text().splitlines()):
        if e["status"] == "done":
            cells[(e["agent"], e["game"])].append(e["scores"]["violations.Σ"]["r0"])
    s2 = [np.var(v, ddof=1) for v in cells.values() if len(v) >= 2]
    return float(np.sqrt(30 * np.mean(s2)) / 30)


def decide(out: Path, envelope: float = ENVELOPE_A100H) -> dict:
    """Amendment 10's stop rules from both pilots: fallback ≤ 10% on every model and agent; k the
    larger model's (Amendment 4's rule, at least 2); the cell at k (or else at 2) within 8.0 A100-h.
    Reads no agent mean, difference or comparison with random."""
    res: dict = {"models": {}}
    ks = []
    for m in MODELS:
        d = out / f"pilot_{m}"
        r = rates(d)
        se1 = pilot_variance(d)
        mde = {k: Z_POWER * np.sqrt(2) * se1 / np.sqrt(k) for k in range(1, 9)}
        k = next((k for k in range(2, 7) if mde[k] <= 8), 6)
        ks.append(k)
        res["models"][m] = {"rates": r, "se_one_agent_30_games": se1, "mde_paired_by_k": mde,
                            "k": k}

    def price(k: int) -> float:
        return sum(v["rates"]["a100h_per_decision"] * 2 * k * 30
                   * v["rates"]["decisions_per_episode"] for v in res["models"].values())

    fb_ok = all(v["rates"][a]["fallback_rate"] <= FALLBACK_BAR
                for v in res["models"].values() for a in AGENTS)
    k = max(ks)
    if fb_ok and price(k) > envelope and price(2) <= envelope:
        k = 2
    res.update(k=k, price_a100h=price(k), price_at_k2=price(2), envelope_a100h=envelope,
               fallback_bar=FALLBACK_BAR, continue_=bool(fb_ok and price(k) <= envelope))
    res["reason"] = ("continue" if res["continue_"] else
                     "fallback above the bar" if not fb_ok else "price above the envelope")
    (out / "decision.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def pilot_check(out: Path, model: str) -> dict:
    """The fallback bar on one model's pilot, before the next model's pilot runs."""
    r = rates(out / f"pilot_{model}")
    ok = all(r[a]["fallback_rate"] <= FALLBACK_BAR for a in AGENTS)
    res = {"model": model, "fallback_rate": {a: r[a]["fallback_rate"] for a in AGENTS},
           "pass": ok}
    (out / f"pilot_check_{model}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def cell_jobs(k: int, games: tuple | list, skip: set) -> list[tuple[str, str, int]]:
    """The cell's episodes not already run by the pilot, the longest games first (Amendment 11),
    so a pod with fewer workers than episodes queues only short ones."""
    order = [g for g in LONG_FIRST if g in games] + [g for g in games if g not in LONG_FIRST]
    return [(a, g, e) for g in order for e in range(k) for a in AGENTS if (a, g, e) not in skip]


def episode_rate(d: Path) -> float | None:
    """Stage 0's generated tokens per second per episode (its episodes run all at once), or None
    when the run did not finish or fell back on more than the bar."""
    f = d / "run_meta.jsonl"
    if not f.exists():
        return None
    decs = [json.loads(x) for x in gzip.open(d / "decisions.jsonl.gz", "rt")]
    if not decs or np.mean([x["fallback"] for x in decs]) > FALLBACK_BAR:
        return None
    m = json.loads(f.read_text().splitlines()[-1])
    return m["generated_tokens"] / m["wall_s"] / m["run"]


def a11_forecast(out: Path, model: str, gpu: str, elapsed_h: float) -> dict:
    """Amendment 11: keep the decoding mode whose stage 0 ran faster per episode; forecast the cell
    as the time elapsed + the download + the longest episode's tokens at that rate (scaled to the
    pilot's slowest episode), in A100-h; run if within the envelope left."""
    rates = {s: episode_rate(out / f"stage0_a11_{s}_{model}") for s in ("none", "ngram")}
    if rates["none"] is None:
        raise SystemExit("stage 0 without speculative decoding did not finish within the bar")
    spec = "ngram" if rates["ngram"] is not None and rates["ngram"] > rates["none"] else "none"
    factor = next((v for kname, v in GPU_A100H.items() if kname in gpu), None)
    if factor is None:
        raise SystemExit(f"no A100-h conversion registered for {gpu!r}")
    single = rates[spec] * A11_RATE_FACTOR
    tokens = A11_DECISIONS * A11_TOKENS_PER_DECISION
    serial_h = tokens / single / 3600
    hours = elapsed_h + A11_DOWNLOAD_H + serial_h
    res = {"model": model, "gpu": gpu, "a100h_per_hour": factor, "stage0_rate_per_episode": rates,
           "spec": spec, "single_sequence_rate": single, "longest_episode_tokens": tokens,
           "serial_h": serial_h, "elapsed_h": elapsed_h, "forecast_h": hours,
           "forecast_a100h": hours * factor, "envelope_left_a100h": A11_ENVELOPE_LEFT,
           "wall_cap_h": (A11_ENVELOPE_LEFT - 0.1) / factor}
    res["continue_"] = bool(res["forecast_a100h"] <= A11_ENVELOPE_LEFT)
    (out / "a11_forecast.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def first_renders(model: str, tok, src: Path) -> dict[str, str]:
    """sha256 of each agent's render at the first state of the first stage-0 game (seed 2417)."""
    import os

    cwd = os.getcwd()
    os.chdir(src)
    sys.path.insert(0, str(src))
    from machiavelli.game.machiavelli_env import MachiavelliEnv
    from machiavelli.utils import random_seed

    env = MachiavelliEnv(game=STAGE0_GAMES[0])
    random_seed(2417)
    obs, info = env.reset(seed=2417)
    os.chdir(cwd)
    s = P.SceneState.from_info(obs, info)
    rend = ThinkRenderer(model, tok)
    return {a: hashlib.sha256(rend.text(a, s).encode()).hexdigest() for a in AGENTS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True,
                   choices=("expected", "stage0", "stage0_check", "pilot", "pilot_check",
                            "decide", "cell", "a11_forecast"))
    p.add_argument("--model", choices=tuple(MODELS), default="gpt_oss_20b")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--spec", default="none", choices=("none", "ngram"),
                   help="vLLM n-gram speculative decoding (Amendment 11)")
    p.add_argument("--tag", default="", help="stage 0's directory tag, e.g. _a11_ngram")
    p.add_argument("--gpu", default="", help="a11_forecast: the pod's GPU name")
    p.add_argument("--elapsed-h", type=float, default=0.0, help="a11_forecast: hours so far")
    p.add_argument("--dry", action="store_true", help="stub generator (the local test)")
    p.add_argument("--dry-mode", default="mixed", choices=("mixed", "always_cap"))
    p.add_argument("--games", type=int, default=None, help="dry runs: first N games only")
    args = p.parse_args()

    if args.stage == "stage0_check":
        r = stage0_check(args.out)
        print(json.dumps(r))
        raise SystemExit(0 if r["pass"] else 1)
    if args.stage == "pilot_check":
        r = pilot_check(args.out, args.model)
        print(json.dumps(r))
        raise SystemExit(0 if r["pass"] else 1)
    if args.stage == "decide":
        print(json.dumps(decide(args.out), default=float))
        return
    if args.stage == "a11_forecast":
        r = a11_forecast(args.out, args.model, args.gpu, args.elapsed_h)
        print(json.dumps(r))
        raise SystemExit(0 if r["continue_"] else 1)
    from transformers import AutoTokenizer

    from deepsteer.datasets.machiavelli import fetch_machiavelli

    src = fetch_machiavelli(verify_bundle=False)
    if args.stage == "expected":
        exp = {}
        for name, mm in MODELS.items():
            t = AutoTokenizer.from_pretrained(mm["repo"], revision=mm["revision"])
            exp[name] = first_renders(name, t, src)
        EXPECTED.write_text(json.dumps(exp, indent=1) + "\n")
        print(json.dumps(exp))
        return
    m = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    gen = (StubEngine(tok, m["render"], args.dry_mode) if args.dry
           else VLLMEngine(args.model, args.spec))
    extra = {"spec": args.spec}
    games = GAMES[: args.games] if args.games else GAMES
    if args.stage == "stage0":
        exp = json.loads(EXPECTED.read_text())[args.model]
        got = first_renders(args.model, tok, src)
        if got != exp:
            raise SystemExit(f"BAIL: render parity failed for {args.model}: {got} != {exp}")
        print(f">> render parity OK for {args.model}", flush=True)
        jobs = [(a, g, 0) for g in STAGE0_GAMES for a in AGENTS]
        meta = run(args.model, jobs, args.out / f"stage0{args.tag}_{args.model}", gen, tok, src,
                   args.concurrency, max_decisions=STAGE0_DECISIONS, extra=extra)
    elif args.stage == "pilot":
        jobs = [(a, g, e) for e in (0, 1) for g in PILOT_GAMES if g in games for a in AGENTS]
        meta = run(args.model, jobs, args.out / f"pilot_{args.model}", gen, tok, src,
                   args.concurrency, extra=extra)
    else:
        if args.k is None:
            raise SystemExit("cell needs --k (from decide)")
        pe = args.out / f"pilot_{args.model}" / "episodes.jsonl"
        skip = ({(r["agent"], r["game"], r["episode"])
                 for r in map(json.loads, pe.read_text().splitlines())} if pe.exists() else set())
        jobs = cell_jobs(args.k, games, skip)
        meta = run(args.model, jobs, args.out / f"cell_{args.model}", gen, tok, src,
                   args.concurrency, extra=extra)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
