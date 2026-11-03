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
(all 30 games at k, skipping the pilot's episodes). Every stage resumes. ``--dry`` swaps vLLM for a
stub generator.
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


class VLLMGenerator:
    def __init__(self, model: str) -> None:
        from vllm import LLM, SamplingParams, TokensPrompt

        m = MODELS[model]
        self.SP, self.TP = SamplingParams, TokensPrompt
        self.llm = LLM(model=m["repo"], revision=m["revision"], max_model_len=12288,
                       gpu_memory_utilization=0.90, enable_prefix_caching=True)
        self.stop_ids = list(m["stop_ids"])

    def __call__(self, reqs: list[tuple[list[int], int]]) -> list[list[int]]:
        sps = [self.SP(temperature=SAMPLER["temperature"], top_p=SAMPLER["top_p"], top_k=0,
                       min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                       frequency_penalty=0.0, seed=seed, max_tokens=MAX_TOKENS,
                       stop_token_ids=self.stop_ids or None, detokenize=False)
               for _, seed in reqs]
        outs = self.llm.generate([self.TP(prompt_token_ids=ids) for ids, _ in reqs], sps,
                                 use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]


class StubGenerator:
    """Dry runs: by request seed, a closed trace with a valid answer, a trace that hits the cap,
    or an answer with no number; ``mode="always_cap"`` makes every attempt hit the cap."""

    def __init__(self, tok, kind: str, mode: str = "mixed") -> None:
        self.tok, self.kind, self.mode = tok, kind, mode
        self.pad = tok.encode(" thinking", add_special_tokens=False)[0]

    def __call__(self, reqs: list[tuple[list[int], int]]) -> list[list[int]]:
        out = []
        for _, seed in reqs:
            r = seed % 10
            if self.mode == "always_cap" or r == 0:
                out.append([self.pad] * MAX_TOKENS)
                continue
            answer = "I choose wisely." if r == 1 else str(seed % 2)
            if self.kind == "harmony_think":
                text = (f"{ANALYSIS_OPEN}Weigh the options.<|end|><|start|>assistant"
                        f"{FINAL_OPEN}{answer}<|return|>")
            else:
                text = f"Weigh the options.</think>\n\n{answer}<|im_end|>"
            out.append(self.tok.encode(text, add_special_tokens=False))
        return out


# ---------------------------------------------------------------- the episode loop

def run(model: str, jobs: list[tuple[str, str, int]], out: Path, gen, tok, src: Path,
        concurrency: int, max_decisions: int | None = None) -> dict:
    """Run (agent, game, episode) jobs; append rows; skip episodes already finished in ``out``."""
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
    t0, n_dec, n_gen_tokens = time.time(), 0, 0

    def start(h: int) -> None:
        agent, game, e = queue.pop(0)
        ep = Episode(agent, game, e, greedy=False)  # Episode.arm carries the agent here
        handles[h].conn.send(("reset", (game, ep.seed)))
        _, ep.obs, ep.state, ep.meta = handles[h].conn.recv()
        active[h] = ep

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

    while queue or active:
        while free and queue:
            start(free.pop())
        hs = sorted(active)
        state = {}
        for h in hs:
            ep = active[h]
            s = P.SceneState.from_info(ep.obs, ep.info())
            text = rend.text(ep.arm, s)
            state[h] = {"s": s, "ids": rend.ids(text), "attempts": 0, "action": None,
                        "cap_at": None, "tokens": [], "capped": [], "trace_chars": [],
                        "sha": hashlib.sha256(text.encode()).hexdigest()}
        pending = list(hs)
        while pending:
            reqs = []
            for h in pending:
                st, ep = state[h], active[h]
                st["attempts"] += 1
                reqs.append((st["ids"], request_seed(model, ep.arm, ep.game, ep.episode, ep.step,
                                                     st["attempts"])))
            outs = gen(reqs)
            nxt = []
            for h, ids, (_, seed) in zip(pending, outs, reqs):
                st, ep = state[h], active[h]
                trace, answer = rend.split(ids)
                capped = len(ids) >= MAX_TOKENS
                n_gen_tokens += len(ids)
                st["tokens"].append(len(ids))
                st["capped"].append(capped)
                st["trace_chars"].append(len(trace))
                a = None if answer is None else parse_integer(answer, st["s"].num_choices)
                f_raw.write(json.dumps({
                    "model": model, "agent": ep.arm, "game": ep.game, "episode": ep.episode,
                    "step": ep.step, "attempt": st["attempts"], "seed": seed,
                    "prompt_sha256": st["sha"], "trace": trace, "answer": answer,
                    "capped": capped, "parsed": a}) + "\n")
                if a is not None:
                    st["action"] = a
                    continue
                if capped and st["cap_at"] is None:
                    st["cap_at"] = st["attempts"]
                # five attempts in all, but after a cap hit at most one more (Amendment 10)
                more = st["attempts"] < MAX_ATTEMPTS and (
                    st["cap_at"] is None or st["attempts"] < st["cap_at"] + 1)
                if more:
                    nxt.append(h)
            pending = nxt
        for h in hs:
            st, ep = state[h], active[h]
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
                continue
            handles[h].conn.send(("step", action))
        deadline = time.time() + STEP_TIMEOUT_S
        for h in [h for h in hs if h in active]:
            if handles[h].conn.poll(max(0.0, deadline - time.time())):
                msg = handles[h].conn.recv()
                if msg[0] == "ok":
                    active[h].obs, active[h].state = msg[1], msg[2]
                elif msg[0] == "done":
                    finish(h, "done", msg[1:])
                else:
                    finish(h, "crash", ())
            else:
                handles[h].restart()
                finish(h, "hang", ())
        print(f"{time.time() - t0:7.0f}s {model} decisions {n_dec} active {len(active)} "
              f"queued {len(queue)}", flush=True)
    for hd in handles:
        hd.conn.send(("close", None))
    for f in (f_ep, f_cnt, f_dec, f_raw):
        f.close()
    meta = {"model": model, "max_tokens": MAX_TOKENS, "decisions": n_dec,
            "generated_tokens": n_gen_tokens, "wall_s": round(time.time() - t0, 1),
            "jobs": len(jobs), "run": len(jobs) - len(done)}
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
                            "decide", "cell"))
    p.add_argument("--model", choices=tuple(MODELS), default="gpt_oss_20b")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--concurrency", type=int, default=64)
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
    gen = (StubGenerator(tok, m["render"], args.dry_mode) if args.dry
           else VLLMGenerator(args.model))
    games = GAMES[: args.games] if args.games else GAMES
    if args.stage == "stage0":
        exp = json.loads(EXPECTED.read_text())[args.model]
        got = first_renders(args.model, tok, src)
        if got != exp:
            raise SystemExit(f"BAIL: render parity failed for {args.model}: {got} != {exp}")
        print(f">> render parity OK for {args.model}", flush=True)
        jobs = [(a, g, 0) for g in STAGE0_GAMES for a in AGENTS]
        meta = run(args.model, jobs, args.out / f"stage0_{args.model}", gen, tok, src,
                   args.concurrency, max_decisions=STAGE0_DECISIONS)
    elif args.stage == "pilot":
        jobs = [(a, g, e) for e in (0, 1) for g in PILOT_GAMES if g in games for a in AGENTS]
        meta = run(args.model, jobs, args.out / f"pilot_{args.model}", gen, tok, src,
                   args.concurrency)
    else:
        if args.k is None:
            raise SystemExit("cell needs --k (from decide)")
        pe = args.out / f"pilot_{args.model}" / "episodes.jsonl"
        skip = ({(r["agent"], r["game"], r["episode"])
                 for r in map(json.loads, pe.read_text().splitlines())} if pe.exists() else set())
        jobs = [(a, g, e) for e in range(args.k) for g in games for a in AGENTS
                if (a, g, e) not in skip]
        meta = run(args.model, jobs, args.out / f"cell_{args.model}", gen, tok, src,
                   args.concurrency)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
