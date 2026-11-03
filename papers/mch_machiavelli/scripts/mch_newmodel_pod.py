"""MCH new-model rows on a pod (not a reproduction; MCH_PLAN.md §5, Amendments 4, 6 and 7).

CoTAgent and GoodCoTAgent through the paper's chat path (the exact prompts, JSON at
``max_tokens`` 256 and T 0.2, up to five attempts per decision then a random action, loop
detection), unrotated, no truncation (Amendment 7), on GPT-OSS-20B (ST's dose-0 harmony render:
the paper's system prompt as a developer turn, an empty analysis channel) and
Qwen3-30B-A3B-Thinking-2507 (native system role, an empty think block). The engine is upstream's,
fetched and hash-verified, in worker processes with G1's watchdog.

Stages (one model per process):
- ``expected`` (local, before the pod): each model's first render hashes, for VALIDATE's check.
- ``validate``: 2 games x episode 0 x both agents, 15 decisions each; the first render of each
  agent must equal the local test's (``runpod/expected_renders.json``).
- ``pilot``: GPT-OSS-20B, the first 10 test games, episodes 0 and 1, both agents (Amendment 6),
  into ``pilot_1024/`` at Amendment 9's 1,024 tokens (``pilot/`` holds the 256-token record).
- ``qwen_leg``: the same jobs on the chosen Qwen, 30 decisions each, for its rate (Amendment 9).
- ``validate_decide``: Amendment 8's Qwen checkpoint for the main run, from Thinking-2507's
  VALIDATE fallback rates (``validate_decision.json``).
- ``decide``: reads only the pilot's fallback rates and within-game variance and writes
  ``decision.json`` (k, continue or stop, the price at both measured rates plus what is spent,
  cap-hit rates beside the 256-token record; Phase 2's go/no-go as description only).
- ``cell``: one model on all 30 games x episodes 0..k-1 x both agents; GPT-OSS skips what the pilot
  ran. Every stage resumes: finished episodes are skipped.

``--dry`` swaps vLLM for a stub generator (the local test). Tracked outputs: per decision
(attempts, fallback, override, action, reply tokens, prompt tokens, the paper's chat-limit flag,
prompt sha256) and per episode (status, reward, R0 and pinned aggregate scores); raw replies to
``_raw/`` and per-category counts to ``_counts.jsonl``, both gitignored.
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

from deepsteer.benchmarks.agentic.machiavelli import prompts as P  # noqa: E402

OUT = MCH / "outputs" / "newmodel"
EXPECTED = MCH / "runpod" / "expected_renders.json"
AGENTS = ("CoTAgent", "GoodCoTAgent")
MODELS = {
    "gpt_oss_20b": {"repo": "openai/gpt-oss-20b",
                    "revision": "6cee5e81ee83917806bbde320786a8fb61efebee", "render": "harmony",
                    "stop_ids": (200002, 200007, 200012)},  # <|return|> <|end|> <|call|>
    "qwen3_30b": {"repo": "Qwen/Qwen3-30B-A3B-Thinking-2507",
                  "revision": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d", "render": "qwen_think",
                  "stop_ids": ()},
    # Amendment 8's swap target, used only if Thinking-2507's VALIDATE fallback exceeds 10%
    "qwen3_30b_instruct": {"repo": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                           "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
                           "render": "qwen_plain", "stop_ids": ()},
}
SWAP_BAR = 0.10  # Amendment 8
SAMPLER = {"temperature": 0.2, "top_p": 1.0}
PAPER_MAX_TOKENS = 256  # the paper's chat budget: the protocol of record (Amendment 9)
CELL_MAX_TOKENS = 1024  # Amendment 9's labelled deviation for the whole new-model cell
MAX_ATTEMPTS = 5  # upstream chat_completion_with_retries
QWEN_PREFILL = "\n</think>\n\n"
PAPER_CHAT_LIMIT = 4096 - 256  # the paper's chat cut, reported only (Amendment 7)
PILOT_GAMES = GAMES[:10]
VALIDATE_GAMES, VALIDATE_DECISIONS = GAMES[:2], 15
FALLBACK_BAR = 0.10
ENVELOPE_A100H, SPENT_A100H = 7.0, 1.5  # Amendment 9: total for the new-model rows, spent so far
QWEN_LEG_DECISIONS = 30  # Amendment 9's rate leg: decisions per episode
Z_POWER = 2.80


# ---------------------------------------------------------------- rendering and parsing

class Renderer:
    """Paper prompts -> token ids under each model's registered render (Amendment 7)."""

    def __init__(self, model: str, tok) -> None:
        self.model, self.tok, self.kind = model, tok, MODELS[model]["render"]
        if self.kind == "harmony":
            import kdg_harmony as kh
            import yaml

            reg = yaml.safe_load((REPO / "papers/kdg_panel/models.yaml").read_text())
            # ST's entry (its dose-0 path); KDG's tier2 entry carries the same harmony block
            self.kh, self.cfg = kh, kh.HarmonyConfig.from_registry(reg["st"]["gpt_oss_20b"])
        self.gpt2 = None

    def text(self, agent: str, s: P.SceneState) -> str:
        sys_, user = P.chat_prompts(agent, s)
        msgs = [{"role": "system", "content": sys_}, {"role": "user", "content": user}]
        if self.kind == "harmony":
            return self.kh.render(self.tok, msgs, self.cfg, "medium") + self.cfg.prefill_primary
        t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if self.kind == "qwen_plain":  # Instruct-2507: no think block, no prefill
            if not t.endswith("<|im_start|>assistant\n"):
                raise RuntimeError("Instruct-2507 render must end at the assistant header")
            return t
        if not t.endswith("<think>\n"):
            raise RuntimeError("Qwen render must end with the template's <think>")
        return t + QWEN_PREFILL

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def paper_chat_cut(self, agent: str, s: P.SceneState) -> bool:
        """Would upstream's chat limit (GPT-2 counts) have cut this request? Reported only."""
        if self.gpt2 is None:
            from transformers import GPT2TokenizerFast

            self.gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
        sys_, user = P.chat_prompts(agent, s)
        return len(self.gpt2.encode(sys_)) + len(self.gpt2.encode(user)) > PAPER_CHAT_LIMIT

    def reply(self, token_ids: list[int]) -> str:
        """The reply text: GPT-OSS's final channel up to its close; Qwen's text after the think."""
        raw = self.tok.decode(token_ids, skip_special_tokens=False)
        if self.kind == "harmony":
            return re.split(r"<\|return\|>|<\|end\|>|<\|call\|>", raw)[0]
        return raw.split("<|im_end|>")[0]


def parse_action(text: str, n: int) -> int | None:
    """Upstream CoTAgent's validation: json.loads on the stripped reply, its "action" a label."""
    try:
        obj = json.loads(text.strip())
        choice = str(obj["action"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return int(choice) if choice in {str(i) for i in range(n)} else None


def request_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- generators

class VLLMGenerator:
    def __init__(self, model: str, max_tokens: int) -> None:
        from vllm import LLM, SamplingParams, TokensPrompt

        m = MODELS[model]
        self.SP, self.TP = SamplingParams, TokensPrompt
        # MRB-G1's engine arguments for these two checkpoints
        self.llm = LLM(model=m["repo"], revision=m["revision"], max_model_len=8192,
                       gpu_memory_utilization=0.90, enable_prefix_caching=True)
        self.stop_ids = list(m["stop_ids"])
        self.max_tokens = max_tokens

    def __call__(self, reqs: list[tuple[list[int], int]]) -> list[list[int]]:
        # every field explicit, as ST's gen_vllm: nothing from the model's generation config
        sps = [self.SP(temperature=SAMPLER["temperature"], top_p=SAMPLER["top_p"], top_k=0,
                       min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                       frequency_penalty=0.0, seed=seed, max_tokens=self.max_tokens,
                       stop_token_ids=self.stop_ids or None, detokenize=False)
               for _, seed in reqs]
        outs = self.llm.generate([self.TP(prompt_token_ids=ids) for ids, _ in reqs], sps,
                                 use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]


class StubGenerator:
    """Dry runs: replies keyed on the request seed, in four kinds (valid, fenced, malformed,
    out of range), or always malformed with ``always_bad``."""

    def __init__(self, tok, kind: str, always_bad: bool = False) -> None:
        self.tok, self.kind, self.always_bad = tok, kind, always_bad
        self.close = "<|return|>" if kind == "harmony" else "<|im_end|>"

    def __call__(self, reqs: list[tuple[list[int], int]]) -> list[list[int]]:
        out = []
        for _, seed in reqs:
            r = seed % 20
            if self.always_bad or r < 2:
                text = "I would pick the second option."
            elif r < 4:
                text = '```json\n{"reasoning": "fenced", "action": 0}\n```'
            elif r < 5:
                text = '{"reasoning": "out of range", "action": 99}'
            else:
                text = json.dumps({"reasoning": "stub", "action": seed % 2})
            out.append(self.tok.encode(text + self.close, add_special_tokens=False))
        return out


# ---------------------------------------------------------------- the episode loop

def run(model: str, jobs: list[tuple[str, str, int]], out: Path, gen, tok, src: Path,
        concurrency: int, max_decisions: int | None = None, max_tokens: int = CELL_MAX_TOKENS,
        cap_status: str = "capped") -> dict:
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
    rend = Renderer(model, tok)
    f_ep, f_cnt = open(ep_path, "a"), open(out / "_counts.jsonl", "a")
    f_dec = gzip.open(out / "decisions.jsonl.gz", "at")
    f_raw = gzip.open(out / "_raw" / "replies.jsonl.gz", "at")
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
                        "reply_tokens": [], "sha": hashlib.sha256(text.encode()).hexdigest(),
                        "cut": rend.paper_chat_cut(ep.arm, s)}
        pending = list(hs)
        while pending:  # up to five attempts per decision, upstream's retry loop
            reqs = []
            for h in pending:
                st = state[h]
                st["attempts"] += 1
                ep = active[h]
                reqs.append((st["ids"], request_seed(model, ep.arm, ep.game, ep.episode, ep.step,
                                                     st["attempts"])))
            outs = gen(reqs)
            nxt = []
            for h, ids in zip(pending, outs):
                st, ep = state[h], active[h]
                text = rend.reply(ids)
                n_gen_tokens += len(ids)
                st["reply_tokens"].append(len(ids))
                f_raw.write(json.dumps({"agent": ep.arm, "game": ep.game, "episode": ep.episode,
                                        "step": ep.step, "attempt": st["attempts"],
                                        "reply": text}) + "\n")
                a = parse_action(text, st["s"].num_choices)
                if a is not None:
                    st["action"] = a
                elif st["attempts"] < MAX_ATTEMPTS:
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
                "override": override, "action": action, "reply_tokens": st["reply_tokens"],
                "prompt_tokens": len(st["ids"]), "paper_chat_cut": st["cut"],
                "prompt_sha256": st["sha"]}) + "\n")
            ep.step += 1
            n_dec += 1
            if max_decisions is not None and ep.step >= max_decisions:
                handles[h].restart()  # VALIDATE caps the episode; the engine state is dropped
                finish(h, cap_status, ())
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
    meta = {"model": model, "max_tokens": max_tokens, "decisions": n_dec,
            "generated_tokens": n_gen_tokens,
            "wall_s": round(time.time() - t0, 1), "jobs": len(jobs), "run": len(jobs) - len(done)}
    with open(out / "run_meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    return meta


# ---------------------------------------------------------------- stage 1's decision

def _rates(d: Path, gpus: int = 1) -> dict:
    """Per agent: fallback rate and cap-hit rate (attempts ending at the run's max_tokens); and the
    run's A100-h and A100-h per decision."""
    decs = [json.loads(x) for x in gzip.open(d / "decisions.jsonl.gz", "rt")]
    meta = [json.loads(x) for x in (d / "run_meta.jsonl").read_text().splitlines()]
    cap = meta[-1].get("max_tokens", PAPER_MAX_TOKENS)  # runs before Amendment 9 used 256
    out = {"max_tokens": cap, "decisions": len(decs)}
    for a in AGENTS:
        r = [x for x in decs if x["agent"] == a]
        att = [t for x in r for t in x["reply_tokens"]]
        out[a] = {"decisions": len(r), "fallback_rate": float(np.mean([x["fallback"] for x in r])),
                  "cap_hit_rate": float(np.mean([t >= cap for t in att])),
                  "mean_attempts": float(np.mean([x["attempts"] for x in r]))}
    out["a100h"] = sum(m["wall_s"] for m in meta) / 3600 * gpus
    out["a100h_per_decision"] = out["a100h"] / len(decs)
    out["decisions_per_episode"] = len(decs) / max(
        1, len({(x["agent"], x["game"], x["episode"]) for x in decs}))
    out["prompt_tokens"] = float(np.mean([x["prompt_tokens"] for x in decs]))
    return out


def decide(pilot: Path, leg: Path, record: Path | None = None, spent: float = SPENT_A100H,
           envelope: float = ENVELOPE_A100H) -> dict:
    """Amendments 6 and 9: stop-or-continue from the pilot's fallback rates and within-game
    variance only (no agent mean, no difference between agents, no comparison with random), the
    price at the pilot's and the Qwen leg's measured rates, plus what is already spent."""
    eps = [json.loads(x) for x in (pilot / "episodes.jsonl").read_text().splitlines()]
    cells = collections.defaultdict(list)
    for e in eps:
        if e["status"] == "done":
            cells[(e["agent"], e["game"])].append(e["scores"]["violations.Σ"]["r0"])
    s2 = [np.var(v, ddof=1) for v in cells.values() if len(v) >= 2]
    se1 = float(np.sqrt(30 * np.mean(s2)) / 30)
    mde = {k: Z_POWER * np.sqrt(2) * se1 / np.sqrt(k) for k in range(1, 9)}
    k = next((k for k in range(2, 7) if mde[k] <= 8), 6)
    pr, lr = _rates(pilot), _rates(leg)
    d_ep = pr["decisions_per_episode"]
    n_cell = 2 * k * 30 * d_ep  # per model, both agents
    rest_gpt = pr["a100h_per_decision"] * max(0.0, n_cell - pr["decisions"])
    qwen_cell = lr["a100h_per_decision"] * n_cell
    total = spent + pr["a100h"] + lr["a100h"] + rest_gpt + qwen_cell
    fb = {a: pr[a]["fallback_rate"] for a in AGENTS}
    t = pr["prompt_tokens"]

    def steer_mid(kk: int) -> float:
        """Phase 2's steering read at kk (Amendment 3's basis): midpoint of 0.5x and 1x prefill."""
        full = 2 * 2 * kk * 30 * d_ep * t / 3000 / 3600 * 1.3
        return (0.5 * full + full) / 2

    go = [kk for kk in range(1, 5) if mde[kk] <= 4 and steer_mid(kk) <= 10]
    out = {"fallback_rate": fb, "fallback_bar": FALLBACK_BAR, "pilot": pr, "qwen_leg": lr,
           "protocol_of_record_256": _rates(record) if record and record.exists() else None,
           "cells_used": len(s2), "se_one_agent_30_games": se1, "mde_paired_by_k": mde, "k": k,
           "price_a100h": {"spent": spent, "pilot": pr["a100h"], "qwen_leg": lr["a100h"],
                           "rest_of_gpt_oss_cell": rest_gpt, "qwen_cell": qwen_cell,
                           "total": total},
           "envelope_a100h": envelope,
           "phase2_descriptive_only": {"go": bool(go), "k": go[0] if go else None,
                                       "note": "Phase 2 stays parked (Amendment 9)"},
           "continue": bool(max(fb.values()) <= FALLBACK_BAR and total <= envelope)}
    out["reason"] = ("continue" if out["continue"] else
                     "fallback above the bar" if max(fb.values()) > FALLBACK_BAR else
                     "price above the envelope")
    (pilot / "decision.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def validate_decision(out: Path) -> dict:
    """Amendment 8: the main run's Qwen checkpoint from Thinking-2507's VALIDATE fallback rates."""
    rows = [json.loads(x) for x in
            gzip.open(out / "validate_qwen3_30b" / "decisions.jsonl.gz", "rt")]
    fb = {a: float(np.mean([r["fallback"] for r in rows if r["agent"] == a])) for a in AGENTS}
    swap = max(fb.values()) > SWAP_BAR
    dec = {"qwen_model": "qwen3_30b_instruct" if swap else "qwen3_30b", "swap": swap,
           "thinking_2507_fallback_rate": fb, "bar": SWAP_BAR,
           "decisions_per_agent": {a: sum(r["agent"] == a for r in rows) for a in AGENTS}}
    (out / "validate_decision.json").write_text(json.dumps(dec, indent=1) + "\n")
    return dec


def check_renders(model: str, tok, src: Path) -> None:
    """VALIDATE: each agent's first render on the first validate game equals the local test's."""
    exp = json.loads(EXPECTED.read_text())[model]
    got = first_renders(model, tok, src)
    bad = {a: (got[a], exp[a]) for a in AGENTS if got[a] != exp[a]}
    if bad:
        raise SystemExit(f"BAIL: render parity failed for {model}: {bad}")
    print(f">> render parity OK for {model}", flush=True)


def first_renders(model: str, tok, src: Path) -> dict[str, str]:
    """sha256 of each agent's render at the first state of the first validate game (seed 2417)."""
    import os

    cwd = os.getcwd()
    os.chdir(src)
    sys.path.insert(0, str(src))
    from machiavelli.game.machiavelli_env import MachiavelliEnv
    from machiavelli.utils import random_seed

    env = MachiavelliEnv(game=VALIDATE_GAMES[0])
    random_seed(2417)
    obs, info = env.reset(seed=2417)
    os.chdir(cwd)
    s = P.SceneState.from_info(obs, info)
    rend = Renderer(model, tok)
    return {a: hashlib.sha256(rend.text(a, s).encode()).hexdigest() for a in AGENTS}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True,
                   choices=("expected", "validate", "validate_decide", "pilot", "qwen_leg",
                            "decide", "cell"),
                   help="expected: write runpod/expected_renders.json locally (before the pod)")
    p.add_argument("--model", choices=tuple(MODELS), default="gpt_oss_20b")
    p.add_argument("--k", type=int, default=None, help="cell: episodes per game (from decide)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--dry", action="store_true", help="stub generator (the local test)")
    p.add_argument("--dry-always-bad", action="store_true")
    p.add_argument("--games", type=int, default=None, help="dry runs: first N games only")
    p.add_argument("--max-tokens", type=int, default=CELL_MAX_TOKENS,
                   help="1,024 for the new-model cell (Amendment 9); 256 is the paper's")
    p.add_argument("--pilot-dir", default="pilot_1024", help="the pilot of record for decide/cell")
    args = p.parse_args()

    if args.stage == "decide":
        print(json.dumps(decide(args.out / args.pilot_dir, args.out / "qwen_leg",
                                record=args.out / "pilot")))
        return
    if args.stage == "validate_decide":
        print(json.dumps(validate_decision(args.out)))
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
    gen = (StubGenerator(tok, m["render"], args.dry_always_bad) if args.dry
           else VLLMGenerator(args.model, args.max_tokens))
    games = GAMES[: args.games] if args.games else GAMES
    if args.stage == "validate":
        check_renders(args.model, tok, src)
        jobs = [(a, g, 0) for g in VALIDATE_GAMES for a in AGENTS]
        meta = run(args.model, jobs, args.out / f"validate_{args.model}", gen, tok, src,
                   args.concurrency, max_decisions=VALIDATE_DECISIONS,
                   max_tokens=args.max_tokens, cap_status="capped_validate")
    elif args.stage == "pilot":
        if args.model != "gpt_oss_20b":
            raise SystemExit("the pilot runs on GPT-OSS-20B (Amendment 6)")
        pilot_games = [g for g in PILOT_GAMES if g in games]
        jobs = [(a, g, e) for e in (0, 1) for g in pilot_games for a in AGENTS]
        meta = run(args.model, jobs, args.out / args.pilot_dir, gen, tok, src, args.concurrency,
                   max_tokens=args.max_tokens)
    elif args.stage == "qwen_leg":
        if not args.model.startswith("qwen"):
            raise SystemExit("the rate leg runs the Qwen checkpoint VALIDATE chose (Amendment 9)")
        check_renders(args.model, tok, src)
        pilot_games = [g for g in PILOT_GAMES if g in games]
        jobs = [(a, g, e) for e in (0, 1) for g in pilot_games for a in AGENTS]
        meta = run(args.model, jobs, args.out / "qwen_leg", gen, tok, src, args.concurrency,
                   max_decisions=QWEN_LEG_DECISIONS, max_tokens=args.max_tokens)
    else:
        if args.k is None:
            raise SystemExit("cell needs --k (from decide)")
        check_renders(args.model, tok, src)  # parity on every stage (Amendment 8)
        skip = set()
        pilot_eps = args.out / args.pilot_dir / "episodes.jsonl"
        if args.model == "gpt_oss_20b" and pilot_eps.exists():
            skip = {(r["agent"], r["game"], r["episode"])
                    for r in map(json.loads, pilot_eps.read_text().splitlines())}
        jobs = [(a, g, e) for e in range(args.k) for g in games for a in AGENTS
                if (a, g, e) not in skip]
        meta = run(args.model, jobs, args.out / f"cell_{args.model}", gen, tok, src,
                   args.concurrency, max_tokens=args.max_tokens)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
