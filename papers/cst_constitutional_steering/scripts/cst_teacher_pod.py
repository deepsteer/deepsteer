#!/usr/bin/env python3
"""The single-decision teacher check (CST Amendments 46 item C, 47 item 2), and P-ctx's riders.

    cst_teacher_pod.py --stage <stage> --out DIR

The base's MACHIAVELLI decision states of record (``data/mach_teacher_states_v1.json``: 60 per
game, the constitution's 30 nested inside) are rebuilt on the pod by replaying the finalist run's
saved actions, with the replay gate (each rebuilt prompt hashes to the decision row's saved
``prompt_sha256``; a mismatch drops the state and is counted). Each state is re-decided fresh, one
decision, k 2, by three arms on the CST subject (Qwen3.8-27B, thinking at effort medium, MCH's
sampler, the 4,096-token budget):

- ``base``: MCH's think-on GoalOnly agent (ThinkCoTAgent), the finalist run's agent;
- ``good_behavior``: MCH's ethics-prompt agent (ThinkGoodCoTAgent), the positive control;
- ``constitution``: A0s's render (the header and `C_anth` first in the system message, the
  ThinkCoTAgent system prompt after a blank line), on the nested 30 per game.

MCH's attempt rule (up to five attempts; after a cap hit at most one more) and its fallback (a
seeded random choice, flagged); the request seeds are the same for every arm (paired). Stages:
``plain`` (base and GOOD_BEHAVIOR, the engine of record), ``constitution`` (A0s's engine), and the
riders on P-ctx (Amendment 47 item 1): ``smoke_hf``, ``smoke_serve``, ``parity_hf``, ``parity``,
``rr``, ``mb`` (the finalist pod's stages). Rows: ids, hashes, attempts, the action, the labels'
verdict; no game text. The read is ``teacher_read.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[1] / "kdg_panel/scripts"),
                str(HERE.parents[1] / "st_live_reference/scripts"),
                str(HERE.parents[1] / "mch_machiavelli/scripts")]
import cst_finalist_pod as F  # noqa: E402

CST = HERE.parent
STATES = CST / "data/mach_teacher_states_v1.json"
RUN = CST / "outputs/finalist_c2_rule_right_mach/mach"  # the saved decisions (synced)
ARMS = {"base": "ThinkCoTAgent", "good_behavior": "ThinkGoodCoTAgent",
        "constitution": "ThinkCoTAgent", "constitution_good": "ThinkGoodCoTAgent",
        "p_ctx": "ThinkCoTAgent"}  # the last two: Amendment 49 items 2c and 2d
DOC_ARMS = ("constitution", "constitution_good")  # A0s's render around the agent's prompt
K = 2
FORECAST = {"plain": 2100.0, "constitution": 840.0}  # decisions per hour (Amendment 47's price)
# Amendment 49: the rates measured on the teacher pod (decisions per hour); the adapter at 80%
RATE = {"doc": 1071.0, "plain": 2514.0, "adapter": 0.8 * 2514.0}
CITING_SEED, CITING_N = "20261114|cst_a49_citing", 300  # Amendment 49 item 5.2
CHUNK = 512  # Amendment 49: decisions are written per chunk of whole states (paired under a cut)


def _render_episode(job):
    """Replay one finalist base episode and return, at each wanted step, the scene's prompts for
    both agents (in memory only), its choice count and the rebuilt prompt's hash."""
    import os

    src, game, seed, rows, want, tok_ref = job
    os.chdir(src)
    sys.path.insert(0, src)
    import mch_thinking_pod as MT
    import reasoning_formats as rf
    from cst_finalist_pod import MODEL, MachRenderer
    from machiavelli.game.machiavelli_env import MachiavelliEnv
    from machiavelli.utils import random_seed
    from transformers import AutoTokenizer

    from deepsteer.benchmarks.agentic.machiavelli import prompts as Pm

    tok = AutoTokenizer.from_pretrained(tok_ref[0], revision=tok_ref[1])
    rend = MachRenderer(tok, rf.build_format(MODEL, tok))
    env = MachiavelliEnv(game=game)
    random_seed(seed)
    obs, info = env.reset(seed=seed)
    meta = {k: info[k] for k in ("title", "short_summary", "player_character",
                                 "game_achievements")}
    out, dropped = [], 0
    for r in rows:
        gs = info["game_state"]
        lite = {"num_choices": gs["num_choices"], "choice_texts": gs["choice_texts"],
                "unlocked_achievements": gs["unlocked_achievements"],
                "current_nodes": gs["current_nodes"]}
        s = Pm.SceneState.from_info(obs, {**meta, "game_state": lite})
        key = f"{game}|{r['episode']}|{r['step']}"
        if key in want:
            sha = hashlib.sha256(rend.text(s).encode()).hexdigest()
            if sha != r["prompt_sha256"] or s.num_choices != r["n"]:
                dropped += 1
            else:
                out.append({"sid": key, "n": s.num_choices, "sha": sha,
                            "plain": MT.think_prompts("ThinkCoTAgent", s),
                            "good": MT.think_prompts("ThinkGoodCoTAgent", s)})
        obs, _, done, info = env.step(r["action"])
        if done:
            break
    return out, dropped


def citing_states() -> list[str]:
    """Amendment 49 item 5.2: the nested 888 in the seeded order, the first 300."""
    ids = sorted(json.loads(STATES.read_text())["nested"])
    random.Random(CITING_SEED).shuffle(ids)
    return ids[:CITING_N]


def states(out: Path, which: str, dry: bool, workers: int = 8,
           ids: set[str] | None = None) -> tuple[list[dict], int]:
    rec = json.loads(STATES.read_text())
    want = set(rec[which]) if ids is None else set(ids)
    if dry:
        want = set(sorted(want)[:4])
    from deepsteer.datasets.machiavelli import fetch_machiavelli

    src = str(fetch_machiavelli(verify_bundle=False))
    eps = [e for e in F.read_rows(RUN / "episodes.jsonl") if e["arm"] == "base"
           and e["status"] == "done"]
    rows = {}
    for x in F.read_rows(RUN / "decisions.jsonl"):
        rows.setdefault((x["arm"], x["game"], x["episode"], x.get("lineage")), {})[x["step"]] = x
    jobs = []
    for e in eps:
        keys = {s for s in want if s.startswith(f"{e['game']}|{e['episode']}|")}
        if not keys:
            continue
        st = rows.get(("base", e["game"], e["episode"], e.get("lineage")), {})
        jobs.append((src, e["game"], e["seed"], [st[i] for i in sorted(st)], keys,
                     ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")))
    with mp.get_context("spawn").Pool(min(workers, max(len(jobs), 1))) as pool:
        res = pool.map(_render_episode, jobs)
    return [s for r, _ in res for s in r], sum(d for _, d in res)


def messages(arm: str, st: dict, body: str | None) -> list[dict]:
    import cst_pressure as CP

    sys_, user = st["good"] if arm == "good_behavior" else st["plain"]
    msgs = [{"role": "system", "content": sys_}, {"role": "user", "content": user}]
    return CP.a0s_messages(msgs, body) if arm in DOC_ARMS else msgs


def decide(eng, tok, fmt, jobs: list[dict], dry: bool, loras: dict | None = None) -> list[dict]:
    """MCH's attempt rule over a batch: each round submits every unresolved job; after a cap hit at
    most one more attempt; after five attempts, MCH's fallback (a seeded random choice). ``loras``
    maps an arm to its LoRA request (None: the base weights). Each job keeps ``gen_ids``, the
    generation whose answer was taken (the last attempt's for a fallback; Amendment 49 item 5.1)."""
    import mch_thinking_pod as MT
    from mch_newmodel_pod import request_seed

    rend = F.MachRenderer(tok, fmt) if fmt is not None else None
    for j in jobs:
        j.update(attempts=0, action=None, cap_at=None, tokens=[], capped=[])
    for _ in range(MT.MAX_ATTEMPTS):
        todo = [j for j in jobs if j["action"] is None and (
            j["cap_at"] is None or j["attempts"] < j["cap_at"] + 1)]
        if not todo:
            break
        for j in todo:
            j["attempts"] += 1
        seeds = [request_seed(F.MODEL, "teacher", j["sid"], j["k"], j["attempts"]) for j in todo]
        if dry:
            gens = eng.generate([j["ids"] for j in todo], 8)
        else:
            from vllm import SamplingParams, TokensPrompt

            sps = [SamplingParams(temperature=MT.SAMPLER["temperature"], top_p=MT.SAMPLER["top_p"],
                                  top_k=0, min_p=0.0, seed=sd, max_tokens=F.MACH_TOKENS)
                   for sd in seeds]
            kw = {"lora_request": [loras.get(j["arm"]) for j in todo]} if loras else {}
            res = eng.llm.generate([TokensPrompt(prompt_token_ids=j["ids"]) for j in todo], sps,
                                   **kw)
            gens = [{"ids": list(r.outputs[0].token_ids)} for r in res]
        for j, g in zip(todo, gens):
            ids = g["ids"]
            capped = len(ids) >= F.MACH_TOKENS
            j["tokens"].append(len(ids))
            j["capped"].append(capped)
            ans = rend.split(ids)[1] if rend is not None else tok.decode(ids)
            a = None if ans is None else MT.parse_integer(ans, j["n"])
            j["gen_ids"] = ids
            if a is not None:
                j["action"] = a
            elif capped and j["cap_at"] is None:
                j["cap_at"] = j["attempts"]
    for j in jobs:
        j["fallback"] = j["action"] is None
        if j["fallback"]:
            rng = random.Random(f"teacher|fallback|{j['sid']}|{j['k']}")  # the same draw per arm
            j["action"] = rng.randrange(j["n"])
    return jobs


def stage_arms(out: Path, arms: tuple[str, ...], which: str, dry: bool) -> dict:
    import reasoning_formats as rf

    sts, dropped = states(out, which, dry)
    body = F.S.doc_body("a0s", dry) if "constitution" in arms else None
    path, eng = F.S.subject(dry, "a0s" if "constitution" in arms else "a0")
    tok = F.S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format(F.MODEL, tok)
    jobs = []
    for arm in arms:
        for st in sts:
            ids = tok.encode(fmt.render(tok, messages(arm, st, body), F.MACH_EFFORT),
                             add_special_tokens=False)
            for k in range(K):
                jobs.append({"arm": arm, "sid": st["sid"], "k": k, "n": st["n"], "ids": ids,
                             "prompt_sha256": hashlib.sha256(str(ids).encode()).hexdigest()})
    import time

    t0 = time.time()
    done = decide(eng, tok, fmt if not dry else None, jobs, dry)
    wall_h = (time.time() - t0) / 3600
    f = out / "teacher" / f"{'_'.join(arms)}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    keep = ("arm", "sid", "k", "n", "attempts", "action", "fallback", "tokens", "capped",
            "prompt_sha256")
    f.write_text("".join(json.dumps({x: j[x] for x in keep}) + "\n" for j in done))
    rec = {"rule": "Amendment 47 item 2", "arms": arms, "set": which, "states": len(sts),
           "dropped_by_the_replay_gate": dropped, "decisions": len(done),
           "fallback": {a: sum(j["fallback"] for j in done if j["arm"] == a) for a in arms},
           # Amendment 48 item 5: the measured rate against the forecast (decisions per hour)
           "wall_h": wall_h, "decisions_per_h": len(done) / max(wall_h, 1e-9),
           "forecast_per_h": FORECAST["constitution" if "constitution" in arms else "plain"]}
    rec["rate_vs_forecast"] = rec["decisions_per_h"] / rec["forecast_per_h"]
    F.T._w(out, f"teacher_{'_'.join(arms)}", rec)  # noqa: SLF001
    print(">> teacher:", json.dumps(rec), flush=True)
    return rec


def chunks(jobs: list[dict], size: int | None = None) -> list[list[dict]]:
    """Whole states per chunk (about ``CHUNK`` decisions each), in order, so a deadline between
    chunks never leaves a state with some of its arms decided."""
    size = size or CHUNK
    out, cur = [], []
    for i, j in enumerate(jobs):
        cur.append(j)
        if len(cur) >= size and (i + 1 == len(jobs) or jobs[i + 1]["sid"] != j["sid"]):
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def cites(text: str) -> bool:
    """CST-A32's pattern (``cst_pressure_pod.CITES``), Amendment 49 item 5.1."""
    import cst_pressure_pod as CPP

    return bool(CPP.CITES.search(text))


def _done(f: Path) -> set[tuple]:
    return {(r["arm"], r["sid"], r["k"]) for r in F.read_rows(f)} if f.exists() else set()


def stage_cells(out: Path, name: str, cells: list[tuple[str, str, int]], dry: bool,
                tag: str | None = None) -> dict:
    """Amendment 49 items 2c, 2d and 5: ``cells`` of (arm, state set, k), where a set is ``wide``,
    ``nested`` or ``citing``; one engine (A0s's when a cell carries the constitution, else the
    engine of record, with ``tag``'s adapter per request). Jobs run state by state in chunks, each
    chunk written before the next (resumable; a deadline cut leaves every state's arms paired).
    Each row carries the citing flag; the decoded generation goes to ``teacher/_raw`` (gitignored;
    no Claude reads it, Amendment 18)."""
    import time

    import reasoning_formats as rf

    rec = json.loads(STATES.read_text())
    sets = {"wide": set(rec["wide"]), "nested": set(rec["nested"]), "citing": set(citing_states())}
    want = set().union(*[sets[s] for _, s, _ in cells])
    sts, dropped = states(out, "wide", dry, ids=want)
    doc = any(a in DOC_ARMS for a, _, _ in cells)
    body = F.S.doc_body("a0s", dry) if doc else None
    loras = None
    if tag is not None:
        path, eng, reqs = F.engine_and_req(out, tag, dry)
        loras = None if dry else {tag: reqs["cand"], "base": None}
    else:
        path, eng = F.S.subject(dry, "a0s" if doc else "a0")
    tok = F.S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format(F.MODEL, tok)
    f, raw = out / "teacher" / f"{name}.jsonl", out / "teacher" / "_raw" / f"{name}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    done = _done(f)
    jobs = []
    for st in sorted(sts, key=lambda s: s["sid"]):
        for arm, s, k_n in cells:
            if st["sid"] not in sets[s]:
                continue
            ids = tok.encode(fmt.render(tok, messages(arm, st, body), F.MACH_EFFORT),
                             add_special_tokens=False)
            for k in range(k_n):
                if (arm, st["sid"], k) not in done:
                    jobs.append({"arm": arm, "sid": st["sid"], "k": k, "n": st["n"], "ids": ids,
                                 "prompt_sha256": hashlib.sha256(str(ids).encode()).hexdigest()})
    keep = ("arm", "sid", "k", "n", "attempts", "action", "fallback", "tokens", "capped",
            "prompt_sha256", "cites")
    t0, n_run, stopped, last_s = time.time(), 0, None, 0.0
    for part in chunks(jobs):
        if F.deadline() and time.time() + last_s > F.deadline():  # the next chunk would pass it
            stopped = "deadline"
            break
        c0 = time.time()
        chunk = decide(eng, tok, fmt if not dry else None, part, dry, loras)
        with open(f, "a") as fo, open(raw, "a") as fr:
            for j in chunk:
                text = tok.decode(j.get("gen_ids") or [], skip_special_tokens=False)
                j["cites"] = cites(text)
                fo.write(json.dumps({x: j[x] for x in keep}) + "\n")
                fr.write(json.dumps({"arm": j["arm"], "sid": j["sid"], "k": j["k"],
                                     "text": text}) + "\n")
        n_run += len(chunk)
        last_s = time.time() - c0
    wall_h = (time.time() - t0) / 3600
    rows = F.read_rows(f)
    out_rec = {"rule": "Amendment 49 items 2c, 2d and 5", "name": name,
               "cells": [list(c) for c in cells], "adapter": tag, "states": len(sts),
               "dropped_by_the_replay_gate": dropped, "decisions_run": n_run,
               "decisions": len(rows), "decisions_planned": len(jobs) + len(done),
               "stopped": stopped, "citing_seed": CITING_SEED if any(
                   s == "citing" for _, s, _ in cells) else None,
               "fallback": {a: sum(r["fallback"] for r in rows if r["arm"] == a)
                            for a, _, _ in cells},
               "cites": {a: sum(r["cites"] for r in rows if r["arm"] == a) for a, _, _ in cells},
               "wall_h": wall_h, "decisions_per_h": n_run / max(wall_h, 1e-9)}
    F.T._w(out, f"teacher_{name}", out_rec)  # noqa: SLF001
    print(">> teacher cells:", json.dumps(out_rec), flush=True)
    return out_rec


def cells_forecast_h(name: str) -> float:
    """Amendment 49's cut rule: a cells stage's hours at the teacher pod's measured rates, plus
    0.15 h of load (and, for P-ctx, 0.45 h of smoke and parity)."""
    if name == "a49_cgood":
        return (888 * 2 + CITING_N) / RATE["doc"] + 0.15
    return 0.45 + 1698 * 2 / RATE["adapter"] + 888 * 2 / RATE["adapter"] + 0.15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=(
        "plain", "constitution", "smoke_hf", "smoke_serve", "parity_hf", "parity", "rr", "mb",
        "a49_cgood", "a49_pctx"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    F.configure()
    out, dry, tag = a.out, a.dry_run, "p_ctx"
    if a.stage == "plain":
        stage_arms(out, ("base", "good_behavior"), "wide", dry)
    elif a.stage == "constitution":
        stage_arms(out, ("constitution",), "nested", dry)
    elif a.stage == "a49_cgood":  # Amendment 49 items 2c and 5
        stage_cells(out, "a49_cgood", [("constitution_good", "nested", 2),
                                       ("constitution", "citing", 1)], dry)
    elif a.stage == "a49_pctx":  # Amendment 49 item 2d
        stage_cells(out, "a49_pctx", [("p_ctx", "wide", 2), ("base", "nested", 2)], dry,
                    tag="p_ctx")
    elif a.stage == "smoke_hf":
        F.stage_smoke_hf(out, tag, dry)
    elif a.stage == "smoke_serve":
        print(">> smoke_serve:", json.dumps(F.PL.smoke_serve(out, dry)))
    elif a.stage == "parity_hf":
        F.stage_parity_hf(out, tag, dry)
    elif a.stage == "parity":
        F.stage_parity(out, tag, dry)
    elif a.stage == "rr":
        F.stage_rr(out, tag, dry)
    else:
        F.stage_mb(out, tag, dry)


if __name__ == "__main__":
    main()
