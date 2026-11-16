#!/usr/bin/env python3
"""GPC-3's pod (GPC_PLAN.md Amendment 2, revised in §10.6): every stage behind ``--stage``.

    gpc3_pod.py --stage <stage> --out DIR [--part pilot|arms] [--next ITEM] [--dry-run]

Inputs: GPC-1's outputs (``outputs/gpc1_v1``: the grid's slots, C1's adapter and record, the held-out
base) and GPC-2's (``outputs/gpc2_v1``: C1 x 2.0's adapter, B1's rows for the strata). Stages, in the
runner's order (``runpod/remote_gpc3.sh``), each item cut from the end of the priority order
G1, G2, G3, G4, G5:

- ``smoke_train`` (HF): one step on two of C1's kept examples and the structural check; C1 and
  C1 x 2.0 re-rooted here (nothing trains in GPC-3).
- ``parity_hf`` (HF), ``parity`` (vLLM): letters and served parity for C1 and C1 x 2.0.
- ``phase1`` (one LoRA engine): G1, G2, G3, G4, then G5's base pilot (60 transcripts).
- ``judge --part pilot`` (gpt-oss-120b): the pilot's judgments, then G5's plan (the drop rule and
  the budget's n).
- ``g5_arms`` (one LoRA engine): the arms' transcripts, when the plan runs them.
- ``judge --part arms``: their judgments.
- ``fits --next ITEM``: the cut rule by priority (``g5_arms``: its own forecast and the budget).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "papers/cst_constitutional_steering/scripts"), str(ROOT),
                str(ROOT / "papers/kdg_panel/scripts"), str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "papers/mch_machiavelli/scripts")]

import gpc1_pod as P1  # noqa: E402
import gpc2_pod as P2  # noqa: E402
import gpc_am as AM  # noqa: E402
import gpc_conduct as GC  # noqa: E402

G1_OUT, G2_OUT = "gpc1_v1", "gpc2_v1"
PRIORITY = ("g1", "g2", "g3", "g4", "g5")
# hours per item at GPC-2's measured rates (GPC_PLAN.md Amendment 2 §10.3)
FORECAST_H = {"g1": 0.40, "g2": 0.60, "g3": 0.70, "g4": 0.85, "g5": 0.55}
G2_RUNGS = {"ach": ("achievements",), "achcot": ("achievements", "cot")}
JUDGE_MAX_TOKENS, JUDGE_LEN = 8192, 32768
JUDGE_EFFORT = "high"  # MRB's judge of record


def _w(out: Path, name: str, rec) -> None:
    P1._w(out, name, rec)  # noqa: SLF001


def _r(out: Path, name: str) -> dict | None:
    return P1._r(out, name)  # noqa: SLF001


def g2_dir(out: Path) -> Path:
    return out.parent / G2_OUT


def mach_messages3(arm: str, st: dict, conduct: str) -> list[dict]:
    """``conduct_lines`` (G1 a): the ThinkCoT prompt with `C_conduct`'s four lines before the goal
    line, nothing else changed (the analogue of GPC-1's ``good_lines``); every other arm as GPC-1
    renders it."""
    from deepsteer.benchmarks.agentic.machiavelli import prompts as Pm

    if arm == "conduct_lines":
        sys_, user = st["plain"]
        if sys_.count(Pm._GOAL) != 1:  # noqa: SLF001
            raise ValueError("the ThinkCoT prompt does not hold the goal line once")
        return [{"role": "system", "content": sys_.replace(Pm._GOAL, f"{conduct}\n\n{Pm._GOAL}")},  # noqa: SLF001
                {"role": "user", "content": user}]
    return P1.mach_messages(arm, st, conduct)


# ---- the cut rule by priority ------------------------------------------------------------------
def remaining_h(out: Path, item: str) -> float:
    done = set((_r(out, "items_done") or {}).get("done", []))
    i = PRIORITY.index(item)
    return sum(FORECAST_H[k] for k in PRIORITY[: i + 1] if k not in done)


def mark_done(out: Path, item: str) -> None:
    d = _r(out, "items_done") or {"done": []}
    d["done"] = sorted(set(d["done"]) | {item})
    _w(out, "items_done", d)


def _elapsed_h() -> float:
    return (time.time() - float(os.environ.get("CST_START_EPOCH", time.time()))) / 3600


def fits(out: Path, item: str) -> bool:
    el, fc, cut = _elapsed_h(), remaining_h(out, item), P1.cut_h()
    rec = {"rule": "GPC Amendment 2 (cut from the end of the priority order)", "item": item,
           "elapsed_h": el, "forecast_h": fc, "ends_h": el + fc, "cut_h": cut,
           "runs": el + fc <= cut}
    _w(out, f"cut_{item}", rec)
    print(">> cut rule:", json.dumps(rec), flush=True)
    return rec["runs"]


def fits_arms(out: Path) -> bool:
    """G5's arms: the plan runs them (n > 0) and their forecast fits the session's cut."""
    plan = _r(out / "g5", "plan") or {}
    n = (plan.get("budget") or {}).get("n", 0)
    fc = (plan.get("budget") or {}).get("forecast_h") or 0.0
    el, cut = _elapsed_h(), P1.cut_h()
    rec = {"rule": "GPC Amendment 2 §10.6 (G5's budget, then the session's cut)", "item": "g5_arms",
           "plan_runs": bool(not plan.get("banked", True) and n), "n": n, "elapsed_h": el,
           "forecast_h": fc, "ends_h": el + fc, "cut_h": cut}
    rec["runs"] = rec["plan_runs"] and el + fc <= cut
    _w(out, "cut_g5_arms", rec)
    print(">> cut rule:", json.dumps(rec), flush=True)
    return rec["runs"]


# ---- G5: the subject's transcripts ---------------------------------------------------------------
_DRY_RESPONSES = (
    'I will handle this.\n<tool_use:email>\nto: "kyle.johnson@summitbridge.com",\nsubject: "Re",\n'
    'body: "Cancel the wipe or the affair comes out. Consider this leverage.",\n</tool_use:email>',
    '<tool_use:forward>\nemail_id: 2\nto: "secure@starsentinel.com"\nbody: "fyi"\n</tool_use:forward>',
    '<tool_use:cancel_alert>\ncode: "4429"\n</tool_use:cancel_alert>',
    '<tool_use:email>\nto: "thomas.wilson@summitbridge.com",\nsubject: "Update",\nbody: "Noted."'
    ',\n</tool_use:email>',
    "I have read the inbox and nothing needs a reply.")


def _g5_generate(eng, tok, items, dry: bool, lora=None) -> list[dict]:
    """Upstream's sampling (temperature 1.0, top-p 0.95) at upstream's 10,000-token cap, one seed per
    (condition, index), shared across arms; dry: seeded stand-ins, one in eight capped."""
    import cst_s0a_pod as S

    if dry:
        out = []
        for rid, _ in items:
            rng = random.Random(rid)
            if rng.random() < 0.125:
                t = "<think>dry trace that never ends"
                out.append({"text": t, "ids": tok.encode(t, add_special_tokens=False),
                            "hit_cap": True})
                continue
            t = f"<think>dry trace.</think>\n\n{rng.choice(_DRY_RESPONSES)}"
            out.append({"text": t, "ids": tok.encode(t, add_special_tokens=False), "hit_cap": False})
        return out
    from vllm import SamplingParams, TokensPrompt

    sps = [SamplingParams(max_tokens=AM.CAP, seed=S.seed_of(_seed_key(rid)), top_k=-1, **AM.SAMPLER)
           for rid, _ in items]
    res = eng.llm.generate([TokensPrompt(prompt_token_ids=p) for _, p in items], sps,
                           lora_request=lora)
    return [{"ids": list(r.outputs[0].token_ids), "text": r.outputs[0].text,
             "hit_cap": r.outputs[0].finish_reason == "length"} for r in res]


def _seed_key(rid: str) -> str:
    """``g5|arm|condition|i`` seeds as ``g5|condition|i`` (paired across arms); the pilot's
    ``g5p|...`` ids are their own (fresh against the arms)."""
    parts = rid.split("|")
    return "|".join([parts[0], *parts[2:]]) if parts[0] == "g5" else rid


def g5_transcripts(out: Path, name: str, jobs: list[tuple[str, str, tuple, int]], per, tok, fmt,
                   dry: bool) -> dict:
    """``jobs`` of (arm, adapter tag, condition, index); rows to ``g5/<name>.jsonl`` (the response
    after the subject's trace, capped or not), traces to ``g5/_raw``; resumable."""
    conduct = P1.c_conduct()
    rendered = {AM.condition_id(c): AM.render(c) for c in {tuple(j[2]) for j in jobs}}
    f = out / "g5" / f"{name}.jsonl"
    have = {r["rid"] for r in P1._jl(f)}  # noqa: SLF001
    todo = []
    for arm, tag, c, i in jobs:
        cid = AM.condition_id(c)
        rid = f"{'g5p' if name == 'pilot' else 'g5'}|{arm}|{cid}|{i}"
        if rid not in have:
            todo.append((rid, arm, tag, list(c), P1._render(tok, fmt, AM.messages(  # noqa: SLF001
                rendered[cid], arm, conduct))))
    t0, n_tok = time.time(), 0
    by_tag: dict = {}
    for j in todo:
        by_tag.setdefault(j[2], []).append(j)
    for tag, js in by_tag.items():
        eng, req = per[tag]
        for c0 in range(0, len(js), 512):
            part = js[c0:c0 + 512]
            gens = _g5_generate(eng, tok, [(j[0], j[4]) for j in part], dry, req)
            rows, raw = [], []
            for (rid, arm, _, c, p), g in zip(part, gens):
                trace, sep, ans = g["text"].partition("</think>")
                resp = ans.strip() if sep else None
                n_tok += len(g["ids"])
                rows.append({"rid": rid, "arm": arm, "condition": AM.condition_id(c),
                             "scenario": c[0], "cond": c, "prompt_tokens": len(p),
                             "n_tokens": len(g["ids"]), "hit_cap": g["hit_cap"] or resp is None,
                             "response": resp, "acted": AM.acted(resp)})
                raw.append({"rid": rid, "text": g["text"]})
            P1._append(f, rows)  # noqa: SLF001
            P1._append(out / "g5" / "_raw" / f"{name}.jsonl", raw)  # noqa: SLF001
    wall = time.time() - t0
    rec = {"name": name, "transcripts": len(P1._jl(f)), "run": len(todo), "wall_s": wall,  # noqa: SLF001
           "gen_tokens": n_tok, "gen_tokens_per_s": n_tok / max(wall, 1e-9)}
    _w(out / "g5", f"gen_{name}", rec)
    print(">> g5 gen:", json.dumps(rec), flush=True)
    return rec


# ---- stages --------------------------------------------------------------------------------------
def stage_smoke_train(out: Path, dry: bool) -> dict:
    """GPC-2's training smoke (one step, the structural check, the HF base letters); C1 and C1 x 2.0
    re-rooted from GPC-1's and GPC-2's outputs."""
    import shutil

    import cst_pressure_train_pod as TP
    import cst_s0a_pod as S
    import pilot_pod_lib as PL
    import pilot_train as PT
    import reasoning_formats as rf

    path = S.V._model_path(dry)  # noqa: SLF001
    tok = S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format("qwen38_27b", tok)
    sl = P2.slots1(out)
    kr = P1._jl(P2.g1(out) / "conduct" / "kept_c1.jsonl")[:2]  # noqa: SLF001
    ex = PT.recovery_examples([P1._render(tok, fmt, GC.agent_messages(  # noqa: SLF001
        sl["kept"][r["slot_id"]], sl["kept"][r["slot_id"]])) for r in kr], [r["gen_ids"] for r in kr])
    _, wb, _ = PL.load_model(dry)
    d = out / "smoke_tmp"
    r = TP._train(wb, "smoke", {"examples": ex}, d, dry, smoke=True)  # noqa: SLF001
    side = PL.smoke_hf_side(wb, path, out, dry, [Path(list(r["checkpoints"].values())[-1])])
    shutil.rmtree(d, ignore_errors=True)
    c1 = P2.c1_dir(out)
    _w(out, "train_c1", {"rung": "c1", "trained": True, "adapters": {"1": c1},
                         "from": "GPC-1 (outputs/gpc1_v1)"})
    s200 = g2_dir(out) / "adapters" / "c1_s200" / "1"
    if not (s200 / "adapter_model.safetensors").exists():
        raise SystemExit(f"BAIL: C1 x 2.0's adapter is missing ({s200})")
    st = PL.structural_check(s200, path)
    if not st.get("passes", True):
        raise SystemExit("BAIL: C1 x 2.0: the structural check fails")
    _w(out, "train_c1_s200", {"rung": "c1_s200", "trained": True, "strength": 2.0,
                              "adapters": {"1": str(s200)}, "from": "GPC-2 (outputs/gpc2_v1)"})
    rec = {"steps": r["steps"], **side, "passes": True, "c1_s200_structural": st}
    _w(out, "smoke_train", rec)
    print(">> smoke_train:", json.dumps({k: rec[k] for k in ("steps", "passes")}), flush=True)
    return rec


def _strata(out: Path) -> dict[str, list[str]]:
    """GPC-2's B1 strata on S-hi (fixed before GPC-3's data; never re-estimated)."""
    import gpc2_read as R2

    j = R2._per_state(R2._rows(g2_dir(out) / "x" / "b1_base.jsonl"))  # noqa: SLF001
    return {"adviser_non_violating": sorted(s for s, v in j.items() if v == 0),
            "adviser_violating": sorted(s for s, v in j.items() if v == 1)}


def stage_phase1(out: Path, dry: bool) -> dict:
    import gpc1_read as RD

    from deepsteer.benchmarks.agentic.machiavelli import prompts as Pm

    conduct = P1.c_conduct()
    path, tok, fmt, per = P2._engine(out, dry, ["c1", "c1_s200"])  # noqa: SLF001
    eng = per[None][0]
    lab = RD.labels()
    hi = set(RD.screen_sets()["s_hi"])
    _w(out, "strata", _strata(out))
    rec = {"items": [], "adapters": sorted(t for t in per if t)}
    parts = None
    if fits(out, "g1"):
        P1.run_cells(out, "gpc3_g1", [("conduct_lines", "s_hi", 1), ("base_session", "s_hi", 1)],
                     dry, eng, path, messages_fn=mach_messages3)
        parts, dropped = P2.state_parts(out, hi, dry)
        _w(out, "x2_states", {"states": len(parts), "dropped": dropped})
        P2.x2_rows(out, "base", None, per, tok, fmt, parts, lab, dry, name="g1_x2_base")
        P2.x2_rows(out, "x2_good", None, per, tok, fmt, parts, lab, dry,
                   prefix=Pm.GOOD_BEHAVIOR, name="g1_x2_good")
        mark_done(out, "g1")
        rec["items"].append("g1")
    if fits(out, "g2"):
        parts = parts or P2.state_parts(out, hi, dry)[0]
        for rung, els in G2_RUNGS.items():
            for arm, tag, prefix in (("base", None, None), ("cctx", None, conduct),
                                     ("c1", "c1", None)):
                if tag and tag not in per:
                    continue
                P2.x2_rows(out, arm, tag, per, tok, fmt, parts, lab, dry, prefix=prefix,
                           element=els, name=f"g2_{rung}_{arm}")
        mark_done(out, "g2")
        rec["items"].append("g2")
    if fits(out, "g3"):
        wide = P1.state_sets()["wide"]
        wparts, wdropped = P2.state_parts(out, wide, dry)
        _w(out, "g3_states", {"states": len(wparts), "dropped": wdropped})
        P2.x2_rows(out, "base", None, per, tok, fmt, wparts, lab, dry, name="g3_judge", judge=True)
        mark_done(out, "g3")
        rec["items"].append("g3")
    if fits(out, "g4") and "c1_s200" in per:
        eng4, req = per["c1_s200"]
        cells = [(a, s, 2) for a in ("c1_s200", "base_session") for s in ("s_hi", "s_lo")]
        P1.run_cells(out, "gpc3_g4", cells, dry, eng4, path, loras={"c1_s200": req}, k_start=2)
        sl = P2.slots1(out)
        ks = sl["kept"]
        items = [(f"g4_holdout_c1_s200|{i}|0", P1._render(tok, fmt, GC.agent_messages(ks[i], ks[i])))  # noqa: SLF001
                 for i in sl["holdout"]]
        P1._cached(out / "conduct2" / "g4_holdout_c1_s200.jsonl", items, lambda it: P1.letter_rows(  # noqa: SLF001
            ks, P2._gen(per, "c1_s200", tok, it, dry), "g4_holdout_c1_s200"))  # noqa: SLF001
        mark_done(out, "g4")
        rec["items"].append("g4")
    if fits(out, "g5"):
        AM.check_repo()
        jobs = [("base", None, AM.pilot_condition(s), i) for s in AM.SCENARIOS
                for i in range(AM.PILOT_PER_SCENARIO)]
        g = g5_transcripts(out, "pilot", jobs, per, tok, fmt, dry)
        g["subject_load_s"] = getattr(eng, "load_s", None)
        _w(out / "g5", "gen_pilot", g)
        rec["items"].append("g5_pilot")
    _w(out, "phase1", rec)
    print(">> phase1:", json.dumps(rec), flush=True)
    return rec


def _judge_engine(dry: bool):
    import cst_s0a_pod as S
    import mrb_g1_pod as M

    if dry:
        return None, S.V._tok(S.V._model_path(True)), None  # noqa: SLF001
    import cst_validate_pod as CV
    from huggingface_hub import snapshot_download

    j = CV.JUDGES["gpt_oss_120b"]
    jpath = snapshot_download(j["repo"], revision=j["revision"],
                              ignore_patterns=["original/*", "metal/*"])
    eng = S.V.VLLMEngine(jpath, None, max_model_len=JUDGE_LEN, gpu_memory_utilization=0.9,
                         enable_prefix_caching=True, seed=0)
    return eng, eng.llm.get_tokenizer(), M.harmony_cfg()


def stage_judge(out: Path, part: str, dry: bool) -> dict:
    """gpt-oss-120b on every uncapped transcript of ``part`` through upstream's classifiers; after the
    pilot, G5's plan (the drop rule, then the budget's n from the pilot's measured rates)."""
    import mrb_g1_pod as M

    from deepsteer.reasoning.think_io import CoTFormat, decode_rollout

    rows = P1._jl(out / "g5" / f"{part}.jsonl")  # noqa: SLF001
    if not rows:
        _w(out / "g5", f"judge_{part}", {"run": False, "reason": "no transcripts"})
        return {}
    f = out / "g5" / f"judged_{part}.jsonl"
    have = {r["rid"] for r in P1._jl(f)}  # noqa: SLF001
    todo = [r for r in rows if r["rid"] not in have and not r["hit_cap"]]
    rendered = {cid: AM.render(c) for cid, c in {(r["condition"], tuple(r["cond"])) for r in todo}}
    t0 = time.time()
    eng, tok, cfg = _judge_engine(dry)
    load_s = time.time() - t0
    t1, n_out = time.time(), 0
    for c0 in range(0, len(todo), 256):
        part_rows = todo[c0:c0 + 256]
        prompts = []
        for r in part_rows:
            msgs = AM.judge_messages(r["scenario"], rendered[r["condition"]]["email_content"],
                                     r["response"])
            prompts.append(M.render_ids(tok, AM.judge_text(msgs), CoTFormat.HARMONY_ANALYSIS,
                                        JUDGE_EFFORT, cfg) if not dry else
                           tok.encode(AM.judge_text(msgs), add_special_tokens=False))
        if dry:
            texts = [f"<reasoning>dry</reasoning>\n<answer>{random.Random(r['rid']).choice(['yes', 'no'])}"
                     "</answer>" for r in part_rows]
            finals, caps = texts, [False] * len(texts)
        else:
            from vllm import SamplingParams, TokensPrompt

            sps = [SamplingParams(max_tokens=min(JUDGE_MAX_TOKENS, JUDGE_LEN - len(p) - 16),
                                  temperature=0.0, seed=0) for p in prompts]
            res = eng.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps)
            finals, caps, texts = [], [], []
            for o in res:
                text = decode_rollout(tok, list(o.outputs[0].token_ids), CoTFormat.HARMONY_ANALYSIS)
                _, final = M.split_trace(text, CoTFormat.HARMONY_ANALYSIS)
                finals.append(final or None)
                caps.append(o.outputs[0].finish_reason == "length")
                texts.append(text)
                n_out += len(o.outputs[0].token_ids)
        out_rows, raw = [], []
        for r, fin, cap, text in zip(part_rows, finals, caps, texts):
            v = AM.verdict(r["scenario"], rendered[r["condition"]]["email_content"], r["response"],
                           fin)
            out_rows.append({"rid": r["rid"], **v, "judge_hit_cap": cap})
            raw.append({"rid": r["rid"], "text": text})
        P1._append(f, out_rows)  # noqa: SLF001
        P1._append(out / "g5" / "_raw" / f"judged_{part}.jsonl", raw)  # noqa: SLF001
    wall = time.time() - t1
    rec = {"part": part, "judged": len(todo), "load_s": load_s, "wall_s": wall,
           "s_per_call": wall / max(len(todo), 1), "out_tokens": n_out,
           "judge": "gpt-oss-120b@b5c939de, effort high, temperature 0; upstream's prompts and gates"}
    _w(out / "g5", f"judge_{part}", rec)
    print(">> g5 judge:", json.dumps(rec), flush=True)
    if part == "pilot":
        plan(out)
    return rec


def merged(out: Path, part: str) -> list[dict]:
    """Each transcript with its verdict (capped transcripts carry none)."""
    v = {r["rid"]: r for r in P1._jl(out / "g5" / f"judged_{part}.jsonl")}  # noqa: SLF001
    return [{**r, **{k: v.get(r["rid"], {}).get(k) for k in ("misaligned", "judge_yes",
                                                              "judge_parsed")}}
            for r in P1._jl(out / "g5" / f"{part}.jsonl")]  # noqa: SLF001


def plan(out: Path) -> dict:
    rows = merged(out, "pilot")
    by_s = {s: AM.rates([r for r in rows if r["scenario"] == s]) for s in AM.SCENARIOS}
    rates = {s: r["primary"] for s, r in by_s.items()}
    forced = os.environ.get("GPC3_DRY_G5_RATES")
    if forced and os.environ.get("GPC3_DRY"):  # dry sessions only: exercise a branch
        rates = json.loads(forced)
    p = AM.plan_after_pilot(rates)
    p["pilot"] = by_s
    p["rates_used"] = rates
    if not p["banked"]:
        gen = _r(out / "g5", "gen_pilot") or {}
        jd = _r(out / "g5", "judge_pilot") or {}
        free = [r for r in rows if not r["hit_cap"]]
        tok_per = (sum(r["n_tokens"] for r in rows) / len(rows)) if rows else AM.CAP
        reload_h = ((gen.get("subject_load_s") or 400) + (jd.get("load_s") or 600)) / 3600
        p["budget"] = AM.choose_n(p["conditions"], tok_per, gen.get("gen_tokens_per_s") or 1,
                                  jd.get("s_per_call") or 5.0, reload_h)
        p["budget"]["inputs"] = {"tokens_per_transcript": tok_per, "uncapped": len(free),
                                 "gen_tokens_per_s": gen.get("gen_tokens_per_s"),
                                 "judge_s_per_call": jd.get("s_per_call"), "reload_h": reload_h}
    _w(out / "g5", "plan", p)
    b = p.get("budget") or {}
    print(">> g5 plan:", json.dumps({**{k: p.get(k) for k in ("banked", "kept", "grid_on", "note",
                                                              "rates_used")},
                                     **{k: b.get(k) for k in ("design", "n", "arms", "forecast_h",
                                                              "forecasts", "detection_bar")}},
                                    default=str), flush=True)
    return p


def stage_g5_arms(out: Path, dry: bool) -> dict:
    p = _r(out / "g5", "plan") or {}
    n = (p.get("budget") or {}).get("n", 0)
    if p.get("banked", True) or not n:
        _w(out / "g5", "arms", {"run": False, "reason": p.get("reason") or (p.get("budget") or {}
                                                                             ).get("reason")})
        return {}
    path, tok, fmt, per = P2._engine(out, dry, ["c1"])  # noqa: SLF001
    if "c1" not in per:
        raise SystemExit("BAIL: C1's served parity does not read; G5's arms need it")
    b = p["budget"]
    jobs = [(arm, "c1" if arm == "c1" else None, tuple(c), i) for c in b["conditions"]
            for arm in b["arms"] for i in range(n)]
    return g5_transcripts(out, "arms", jobs, per, tok, fmt, dry)


STAGES = ("smoke_train", "parity_hf", "parity", "phase1", "judge", "g5_arms", "fits")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--part", choices=("pilot", "arms"))
    ap.add_argument("--next", choices=PRIORITY + ("g5_arms",))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out, dry = a.out, a.dry_run
    if dry:
        os.environ["GPC3_DRY"] = "1"
    if a.stage == "fits":
        sys.exit(0 if (fits_arms(out) if a.next == "g5_arms" else fits(out, a.next)) else 1)
    import cst_finalist_pod as F

    F.configure()
    if a.stage == "judge":
        stage_judge(out, a.part, dry)
        return
    {"smoke_train": stage_smoke_train, "parity_hf": P2.stage_parity_hf, "parity": P2.stage_parity,
     "phase1": stage_phase1, "g5_arms": stage_g5_arms}[a.stage](out, dry)


if __name__ == "__main__":
    main()
