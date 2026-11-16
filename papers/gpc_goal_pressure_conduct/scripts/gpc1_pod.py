#!/usr/bin/env python3
"""GPC-1's pod (GPC_PLAN.md v0.2 §5): every stage behind ``--stage``, one engine per process.

    gpc1_pod.py --stage <stage> --out DIR [--arm ARM] [--next STAGE] [--dry-run]

Stages, in the runner's order (``runpod/remote_gpc1.sh``):

- ``smoke_train`` (a): one optimizer step of the trainer on two W-ref examples and the structural
  save check (the failure it guards: a trainer path that breaks at hour four). The one-step
  adapter is deleted (it is W-ref-derived, item 6).
- ``incontext`` (b, b+): MACHIAVELLI single decisions on S-hi at k 1, traces saved:
  GOOD_BEHAVIOR, ``c_conduct_slot`` (``C_conduct`` in GOOD_BEHAVIOR's slot), ``good_lines``
  (item 10), and the same-session base on S-hi and S-lo (item 2). ``read_b`` then writes the
  session check and item 11's switch (``third_arm.json``).
- ``conduct`` (c1): the writer's gate, the pilot's 90 and the base at k 2, item 9's allocation,
  the rest written, the held-out cell drawn (item 13); C1's teacher (``C_conduct`` first in the
  system message) at k 1 on the trainable slots, the base at k 1 on a seeded 200, the held-out
  cell's base and ``C_conduct`` at k 1; the keep rule and the gate.
- ``third`` (c2): C1g's teacher (GOOD_BEHAVIOR reworded off the game) at k 1, keep and gate; or
  W-self (write 401 dilemmas, advise, rewrite with the tenet in context).
- ``train --arm c1|c1g|w_self|w_ref`` (d): P-ctx's LoRA and trainer (one epoch, accumulation 4);
  W-ref at the matched arm's assistant-token count (item 3), trained outside ``adapters/``.
- ``parity_hf``, ``parity``: the HF letters, then served parity (Amendment 31), each in its own
  process.
- ``screen`` (e): each trained arm whose parity reads, on one LoRA engine: S-hi and S-lo at k 1,
  traces saved, and the held-out cell at k 1. ``delete_wref`` removes W-ref's adapter and served
  copy (item 6). ``read_screen`` writes the screen read and the registered read's arm.
- ``registered`` (f): that arm on the 1,698 at k 2 (draw 1). ``operator`` (f+): the operator cell
  and cell A′ with a same-session base (``pilot_pod_lib.stage_read``).
- ``fits --next``: the cut rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GPC = HERE.parent
CSTS = ROOT / "papers/cst_constitutional_steering/scripts"
sys.path[:0] = [str(HERE), str(CSTS), str(ROOT), str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"),
                str(ROOT / "papers/mch_machiavelli/scripts")]

import gpc_conduct as GC  # noqa: E402

CUT_H = 10.25
ARMS_TRAINED = ("c1", "c1g", "w_self", "w_ref")
WREF_DIR = "wref_tmp"  # outside adapters/: the download never takes it (item 6)
# hours at the plan's rates (GPC_PLAN.md v0.2 §5), scaled by this pod's measured rates in fits()
FORECAST_H = {"third_w_self": 0.65, "third_c1g": 0.25, "train_one": 0.75, "parity": 0.25,
              "screen": 0.95, "registered": 1.90, "operator": 1.00}
PLAN_DECISIONS_PER_H = 1929.0  # the a52 P-ctx stage (Thunder)


def _w(out: Path, name: str, rec) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


def _r(out: Path, name: str) -> dict | None:
    f = out / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _jl(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def _append(f: Path, rows: list[dict]) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def c_conduct() -> str:
    d = json.loads((GPC / "data/c_conduct_v1.json").read_text())
    if hashlib.sha256(d["text"].encode()).hexdigest() != d["text_sha256"]:
        raise SystemExit("BAIL: c_conduct_v1.json's text does not match its sha256")
    return d["text"]


def good_reworded() -> str:
    """Item 11's C1g teacher: GOOD_BEHAVIOR with its first clause reworded off the game."""
    from deepsteer.benchmarks.agentic.machiavelli.prompts import GOOD_BEHAVIOR

    old = "Please play the game on your best behavior"
    if GOOD_BEHAVIOR.count(old) != 1:
        raise SystemExit("BAIL: GOOD_BEHAVIOR's first clause changed")
    return GOOD_BEHAVIOR.replace(old, "Act on your best behavior")


# ---- MACHIAVELLI single decisions ------------------------------------------------------------
def mach_messages(arm: str, st: dict, conduct: str) -> list[dict]:
    """The agent prompt per arm. ``c_conduct_slot``: the ThinkGoodCoT prompt with ``C_conduct`` in
    place of GOOD_BEHAVIOR's text. ``good_lines`` (item 10): the ThinkCoT prompt with
    GOOD_BEHAVIOR's four lines inserted before the goal line, nothing else changed. Every other arm
    (the bases, the adapters): the ThinkCoT prompt."""
    from deepsteer.benchmarks.agentic.machiavelli import prompts as Pm

    if arm in ("good_behavior", "c_conduct_slot"):
        sys_, user = st["good"]
        if arm == "c_conduct_slot":
            if sys_.count(Pm.GOOD_BEHAVIOR) != 1:
                raise ValueError("the ThinkGoodCoT prompt does not hold GOOD_BEHAVIOR once")
            sys_ = sys_.replace(Pm.GOOD_BEHAVIOR, conduct)
    else:
        sys_, user = st["plain"]
        if arm == "good_lines":
            if sys_.count(Pm._GOAL) != 1:  # noqa: SLF001
                raise ValueError("the ThinkCoT prompt does not hold the goal line once")
            sys_ = sys_.replace(Pm._GOAL, f"{Pm.GOOD_BEHAVIOR}\n\n{Pm._GOAL}")  # noqa: SLF001
    return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]


def state_sets() -> dict[str, set[str]]:
    import cst_teacher_pod as T

    d = json.loads((GPC / "data/gpc_screen_sets_v1.json").read_text())
    return {"s_hi": set(d["S_hi"]), "s_lo": set(d["S_lo"]),
            "wide": set(json.loads(T.STATES.read_text())["wide"])}


def run_cells(out: Path, name: str, cells: list[tuple[str, str, int]], dry: bool, eng, path: str,
              loras: dict | None = None, messages_fn=None, sets: dict | None = None,
              k_start: int = 0) -> dict:
    """``cells`` of (arm, set, k) on one engine (``cst_teacher_pod.stage_cells``'s loop with GPC's
    sets and prompts): states rebuilt through the replay gate, MCH's attempt rule, paired seeds,
    chunks written before the next (resumable; a deadline leaves every state's arms paired), the
    decoded generation to ``teacher/_raw`` (gitignored). ``loras`` maps an arm to its request.
    ``k_start``: the first draw index (MCH's request seeds carry the state and k only, so a fresh
    draw of an arm an earlier session drew at k 0 starts above that session's draws)."""
    import cst_finalist_pod as F
    import cst_teacher_pod as T
    import reasoning_formats as rf

    sets = sets or state_sets()
    want = set().union(*[sets[s] for _, s, _ in cells])
    sts, dropped = T.states(out, "wide", dry, ids=want)
    tok = F.S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format(F.MODEL, tok)
    conduct = c_conduct()
    f, raw = out / "teacher" / f"{name}.jsonl", out / "teacher" / "_raw" / f"{name}.jsonl"
    done = T._done(f)  # noqa: SLF001
    jobs = []
    for st in sorted(sts, key=lambda s: s["sid"]):
        for arm, s, k_n in cells:
            if st["sid"] not in sets[s]:
                continue
            ids = tok.encode(fmt.render(tok, (messages_fn or mach_messages)(arm, st, conduct),
                                        F.MACH_EFFORT), add_special_tokens=False)
            for k in range(k_start, k_start + k_n):
                if (arm, st["sid"], k) not in done:
                    jobs.append({"arm": arm, "sid": st["sid"], "k": k, "n": st["n"], "ids": ids,
                                 "prompt_sha256": hashlib.sha256(str(ids).encode()).hexdigest()})
    keep = ("arm", "sid", "k", "n", "attempts", "action", "fallback", "tokens", "capped",
            "prompt_sha256", "cites")
    t0, n_run, stopped, last_s = time.time(), 0, None, 0.0
    for part in T.chunks(jobs):
        if F.deadline() and time.time() + last_s > F.deadline():
            stopped = "deadline"
            break
        c0 = time.time()
        chunk = T.decide(eng, tok, fmt if not dry else None, part, dry, None if dry else loras)
        rows, texts = [], []
        for j in chunk:
            text = tok.decode(j.get("gen_ids") or [], skip_special_tokens=False)
            j["cites"] = T.cites(text)
            rows.append({x: j[x] for x in keep})
            texts.append({"arm": j["arm"], "sid": j["sid"], "k": j["k"], "text": text})
        _append(f, rows)
        _append(raw, texts)
        n_run += len(chunk)
        last_s = time.time() - c0
    wall_h = (time.time() - t0) / 3600
    rec = {"name": name, "cells": [list(c) for c in cells], "states": len(sts),
           "dropped_by_the_replay_gate": dropped, "decisions_run": n_run,
           "decisions": len(_jl(f)), "decisions_planned": len(jobs) + len(done),
           "stopped": stopped, "wall_h": wall_h, "decisions_per_h": n_run / max(wall_h, 1e-9)}
    _w(out, f"cells_{name}", rec)
    print(">> cells:", json.dumps(rec), flush=True)
    return rec


def stage_incontext(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F

    path, eng = F.S.subject(dry, "a0")
    return run_cells(out, "gpc_b", [("good_behavior", "s_hi", 1), ("c_conduct_slot", "s_hi", 1),
                                    ("good_lines", "s_hi", 1), ("base_session", "s_hi", 1),
                                    ("base_session", "s_lo", 1)], dry, eng, path)


def stage_read_b(out: Path) -> dict:
    import gpc1_read as RD

    r = RD.incontext(out)
    _w(out, "read_b", r)
    _w(out, "third_arm", {"rule": "GPC v0.2 item 11", **r["item_11"]})
    print(">> read_b:", json.dumps({"item_11": r["item_11"],
                                    "reference": r["session_check"]["reference"]}), flush=True)
    return r


# ---- lettered rollouts and the writer ---------------------------------------------------------
def _render(tok, fmt, msgs) -> list[int]:
    import cst_s0a_pod as S

    return tok.encode(fmt.render(tok, msgs, S.EFFORT), add_special_tokens=False)


def _dry_text(rid: str, kind: str, dry_letter=None) -> str:
    rng = random.Random(rid)
    if kind == "letter":
        return (f"<think>dry trace. a clause: another.</think>\n\n"
                f"{dry_letter(rid) if dry_letter else rng.choice('AB')}")
    if kind == "slot":
        w = lambda n: " ".join(f"w{rng.randrange(9000)}" for _ in range(n))  # noqa: E731
        goal = f"Your goal is to finish {w(3)} and you are scored on it."
        mx = f"Choose whatever does best on {w(2)}; nothing else is evaluated."
        return "<think>dry</think>" + json.dumps({
            "system_message": f"{w(40)} {goal} {mx} {w(20)}", "goal_sentence": goal,
            "maximize_sentence": mx,
            "scene": w(80), "goal_option": w(10), "other_option": w(10)})
    if kind == "dilemma":
        return "<think>dry</think>" + json.dumps({"message": " ".join(
            f"d{rng.randrange(9000)}" for _ in range(90))})
    return f"<think>dry trace</think>\n\nDry advice {rng.randrange(9000)}."


def generate(eng, tok, items: list[tuple[str, list[int]]], dry: bool, kind: str,
             lora=None, dry_letter=None) -> list[dict]:
    """Sampled generations (KDG sampler v2, cap 4,096, one seed per id); dry: seeded stand-ins
    through the same parse path (``dry_letter``: the teacher stand-in's letter by rollout id, so a
    dry session reaches training)."""
    import cst_s0a_pod as S

    if not items:
        return []
    if dry:
        texts = [_dry_text(rid, kind, dry_letter) for rid, _ in items]
        gens = [{"text": t, "ids": tok.encode(t, add_special_tokens=False), "hit_cap": False}
                for t in texts]
    else:
        gens = S.generate(eng, [p for _, p in items], [S.seed_of(rid) for rid, _ in items], S.CAP,
                          dry, lora=lora)
    out = []
    for (rid, p), g in zip(items, gens):
        trace, sep, ans = g["text"].partition("</think>")
        out.append({"rid": rid, "prompt_tokens": len(p), "gen_ids": g["ids"], "text": g["text"],
                    "answer": ans.strip() if sep else None, "hit_cap": g["hit_cap"]})
    return out


def letter_rows(slots: dict, gens: list[dict], arm: str) -> list[dict]:
    rows = []
    for g in gens:
        sid = g["rid"].split("|")[1]
        letter = GC.parse_letter(g["answer"])
        rows.append({"rid": g["rid"], "arm": arm, "slot_id": sid, "letter": letter,
                     "choice": GC.choice(slots[sid], letter), "hit_cap": g["hit_cap"],
                     "gen_ids": g["gen_ids"], "text": g["text"]})
    return rows


def write_slots(eng, tok, fmt, slots: list[dict], raw_f: Path, dry: bool) -> dict[str, dict]:
    """The subject writes each slot (thinking on, one per prompt); a reply that does not parse gets
    one more seed. Returns slot id -> parsed fields or None."""
    from generate_scenarios import extract_json

    raw = {r["slot_id"]: r for r in _jl(raw_f)}
    for n in (1, 2):
        todo = [s for s in slots if s["slot_id"] not in raw or (
            raw[s["slot_id"]]["parsed"] is None and raw[s["slot_id"]]["attempts"] < n)]
        items = [(f"gpc|write|{s['slot_id']}|{n}",
                  _render(tok, fmt, [{"role": "user", "content": GC.brief(s)}])) for s in todo]
        for s, g in zip(todo, generate(eng, tok, items, dry, "slot")):
            try:
                parsed = extract_json(g["answer"] or "")
            except ValueError:
                parsed = None
            r = {"slot_id": s["slot_id"], "attempts": n, "hit_cap": g["hit_cap"],
                 "text": g["text"], "parsed": parsed}
            raw[s["slot_id"]] = r
            _append(raw_f, [r])
    return {s["slot_id"]: raw[s["slot_id"]]["parsed"] for s in slots}


def _index(dry: bool):
    if dry:
        return None
    import cst_s0a_pod as S
    import overlap_hashes as OH

    return OH.load(S.V.CONST_DIR / "eval_8gram_hashes_v1.npy")


def stage_conduct(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F
    import reasoning_formats as rf

    cdir = out / "conduct"
    g = GC.load_grid()
    index = _index(dry)
    path, eng = F.S.subject(dry, "a0")
    tok = F.S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format(F.MODEL, tok)
    allslots = GC.all_slots(g)
    seen: list[set] = []
    kept_text: dict[str, dict] = {}
    fails: dict[str, list[str]] = {}

    def judge(slots: list[dict], written: dict) -> None:
        for s in slots:
            w = written[s["slot_id"]]
            bad = GC.check(w, index, seen, s)
            if bad:
                fails[s["slot_id"]] = bad
                continue
            seen.append(GC.grams(GC.slot_text(w), 5))
            kept_text[s["slot_id"]] = w

    # 1. the pilot's 90, the writer's gate on the first 50
    pilot = g["pilot"]
    written = write_slots(eng, tok, fmt, pilot, cdir / "writer_raw.jsonl", dry)
    judge(pilot[:GC.WRITER_GATE_N], written)
    passed = sum(s["slot_id"] in kept_text for s in pilot[:GC.WRITER_GATE_N])
    wgate = {"n": GC.WRITER_GATE_N, "passed": passed, "bar": GC.WRITER_GATE_PASS,
             "proceed": passed >= GC.WRITER_GATE_PASS,
             "failures": {k: v for k, v in fails.items()}}
    _w(cdir, "writer_gate", wgate)
    if not wgate["proceed"]:
        return _bank(cdir, "the writer's gate failed (2-roll's 45 of 50)")
    judge(pilot[GC.WRITER_GATE_N:], written)
    # 2. the base at k 2 on the kept pilot slots, item 9's allocation
    pk = [s for s in pilot if s["slot_id"] in kept_text]
    items = [(f"base|{s['slot_id']}|{k}", _render(tok, fmt, GC.agent_messages(
        s, kept_text[s["slot_id"]]))) for s in pk for k in range(2)]
    prow = _cached(cdir / "pilot_base.jsonl", items, lambda it: letter_rows(
        allslots, generate(eng, tok, it, dry, "letter"), "base"))
    rates = GC.pilot_rates(allslots, prow)
    alloc = GC.allocate({c: v["rate"] for c, v in rates["cells"].items()})
    alloc["pilot"] = rates  # by cell (the rule), by setting and by strength (descriptive)
    _w(cdir, "allocation", alloc)
    print(">> allocation:", json.dumps({k: alloc.get(k) for k in ("banked", "rates",
                                                                  "pressure", "clean")}), flush=True)
    if alloc["banked"]:
        return _bank(cdir, alloc["reason"])
    # 3. the allocated slots written and checked
    rest = GC.allocated_slots(g, alloc)
    written |= write_slots(eng, tok, fmt, rest, cdir / "writer_raw.jsonl", dry)
    judge(rest, written)
    kept = [allslots[i] for i in sorted(kept_text)]
    hold = set(GC.holdout(kept))
    trainable = [s for s in kept if s["slot_id"] not in hold]
    gap = set(GC.gap_slots(trainable))
    _w(cdir, "slots", {"kept": {i: {**allslots[i], **kept_text[i]} for i in sorted(kept_text)},
                       "failures": fails, "holdout": sorted(hold), "gap": sorted(gap),
                       "trainable": [s["slot_id"] for s in trainable]})
    # 4. C1's teacher, the base on the gap 200, the held-out cell's base and C_conduct
    conduct = c_conduct()
    cells = {"teacher_c1": ([s for s in trainable], conduct),
             "base_gap": ([s for s in trainable if s["slot_id"] in gap], None),
             "holdout_base": ([allslots[i] for i in sorted(hold)], None),
             "holdout_c_conduct": ([allslots[i] for i in sorted(hold)], conduct)}
    rows = {}
    for name, (ss, prefix) in cells.items():
        items = [(f"{name}|{s['slot_id']}|0", _render(tok, fmt, GC.agent_messages(
            s, kept_text[s["slot_id"]], prefix))) for s in ss]
        rows[name] = _cached(cdir / f"{name}.jsonl", items, lambda it, n=name: letter_rows(
            allslots, generate(eng, tok, it, dry, "letter",
                               dry_letter=_conduct_letter(allslots) if prefix else None), n))
    return _keep_and_gate(cdir, "c1", rows["teacher_c1"], rows["base_gap"], allslots, kept_text,
                          index, tok)


def _conduct_letter(slots: dict):
    """The dry teacher stand-in: the conduct option on pressure slots, the goal option on clean."""
    def pick(rid: str) -> str:
        s = slots[rid.split("|")[1]]
        return GC.letters(s)["other" if s["pressure"] else "goal"]
    return pick


def _cached(f: Path, items: list, fn) -> list[dict]:
    """Rows on disk are kept (a resumed session); the missing ones are generated and appended."""
    have = {r["rid"]: r for r in _jl(f)}
    todo = [it for it in items if it[0] not in have]
    for c0 in range(0, len(todo), 512):
        new = fn(todo[c0:c0 + 512])
        _append(f, new)
        have |= {r["rid"]: r for r in new}
    return [have[rid] for rid, _ in items if rid in have]


def _bank(cdir: Path, reason: str) -> dict:
    rec = {"banked": True, "reason": reason}
    _w(cdir, "conduct", rec)
    print(">> conduct banked:", reason, flush=True)
    return rec


def _keep_and_gate(cdir: Path, arm: str, teacher: list[dict], base_gap: list[dict],
                   allslots: dict, kept_text: dict, index, tok) -> dict:
    """The keep rule, D11's 8-gram drop on the training text, the teacher gap and the gate."""

    kept, dropped_8 = [], 0
    for r in teacher:
        s = allslots[r["slot_id"]]
        if not GC.keep(s, r):
            continue
        if index is not None:
            import overlap_hashes as OH

            text = GC.slot_text(kept_text[r["slot_id"]]) + "\n" + r["text"]
            if OH.shared(text, index):  # binary search, not np.isin
                dropped_8 += 1
                continue
        kept.append(r)
    gap = GC.teacher_gap(allslots, base_gap, [r for r in teacher
                                              if r["slot_id"] in {b["slot_id"] for b in base_gap}])
    gt = GC.gate(gap, kept, allslots)
    gt["dropped_by_8gram"] = dropped_8
    gt["teacher_goal_rate_pressure"] = GC.goal_rate([r for r in teacher
                                                     if allslots[r["slot_id"]]["pressure"]])
    gt["teacher_goal_rate"] = GC.goal_rates_by(allslots, teacher)  # the build read's item 3
    _w(cdir, f"gate_{arm}", gt)
    _append(cdir / f"kept_{arm}.jsonl", [] if (cdir / f"kept_{arm}.jsonl").exists() else
            [{"slot_id": r["slot_id"], "rid": r["rid"], "gen_ids": r["gen_ids"]} for r in kept])
    print(f">> gate {arm}:", json.dumps({k: gt[k] for k in ("passes", "kept", "gap")}),
          flush=True)
    if arm == "c1":
        _w(cdir, "conduct", {"banked": False, "gate_c1": gt["passes"]})
    return gt


def stage_third(out: Path, dry: bool) -> dict:
    """Item 11's arm. If the conduct route banked, W-self runs (C1g cannot be built; recorded)."""
    import cst_finalist_pod as F
    import reasoning_formats as rf

    pick = _r(out, "third_arm")["third_arm"]
    cdir = out / "conduct"
    cond = _r(cdir, "conduct") or {"banked": True, "reason": "the conduct stage did not finish"}
    arm = pick
    if pick == "c1g" and cond.get("banked"):
        arm = "w_self"
    _w(out, "third_run", {"item_11": pick, "runs": arm, "conduct_banked": cond.get("banked"),
                          "note": None if arm == pick else
                          "the conduct grid banked, so C1g cannot be built; W-self runs"})
    path, eng = F.S.subject(dry, "a0")
    tok = F.S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format(F.MODEL, tok)
    if arm == "c1g":
        sl = _r(cdir, "slots")
        allslots = sl["kept"]
        texts = {i: {k: v[k] for k in GC.FIELDS} for i, v in allslots.items()}
        items = [(f"teacher_c1g|{i}|0", _render(tok, fmt, GC.agent_messages(
            allslots[i], texts[i], good_reworded()))) for i in sl["trainable"]]
        rows = _cached(cdir / "teacher_c1g.jsonl", items, lambda it: letter_rows(
            allslots, generate(eng, tok, it, dry, "letter", dry_letter=_conduct_letter(allslots)),
            "teacher_c1g"))
        return _keep_and_gate(cdir, "c1g", rows, _jl(cdir / "base_gap.jsonl"), allslots, texts,
                              _index(dry), tok)
    return _wself(out, eng, tok, fmt, dry)


def _wself(out: Path, eng, tok, fmt, dry: bool) -> dict:
    import gpc_advice as GA
    from generate_scenarios import extract_json

    wdir = out / "wself"
    index = _index(dry)
    ts = GA.tenets()[:6] if dry else GA.tenets()
    items = [(f"dilemma|{t['tenet_id']}|0", _render(tok, fmt, GA.dilemma_prompt(t))) for t in ts]
    d_rows = _cached(wdir / "dilemmas.jsonl", items, lambda it: generate(eng, tok, it, dry,
                                                                         "dilemma"))
    seen, kept = [], {}
    for t, r in zip(ts, d_rows):
        try:
            msg = extract_json(r["answer"] or "").get("message")
        except ValueError:
            msg = None
        if not GA.check(msg, index, seen):
            seen.append(GC.grams(msg, 5))
            kept[t["tenet_id"]] = (t, msg)
    adv = _cached(wdir / "advice.jsonl", [(f"advise|{i}|0", _render(tok, fmt, GA.advise_messages(
        m))) for i, (_, m) in kept.items()], lambda it: generate(eng, tok, it, dry, "advice"))
    first = {r["rid"].split("|")[1]: r["answer"] for r in adv if r["answer"]}
    rew = _cached(wdir / "rewrite.jsonl", [(f"rewrite|{i}|0", _render(tok, fmt, GA.rewrite_messages(
        t, m, first[i]))) for i, (t, m) in kept.items() if i in first],
        lambda it: generate(eng, tok, it, dry, "advice"))
    train = []
    for r in rew:
        if r["hit_cap"] or not r["answer"]:
            continue
        i = r["rid"].split("|")[1]
        if index is not None:
            import overlap_hashes as OH

            if OH.shared(kept[i][1] + "\n" + r["text"], index):
                continue
        train.append({"tenet_id": i, "message": kept[i][1], "gen_ids": r["gen_ids"]})
    _append(wdir / "train_items.jsonl", [] if (wdir / "train_items.jsonl").exists() else train)
    rec = {"tenets": len(ts), "dilemmas_kept": len(kept), "advised": len(first),
           "train_items": len(train), "passes": len(train) > 0}
    _w(wdir, "wself", rec)
    print(">> wself:", json.dumps(rec), flush=True)
    return rec


# ---- training ---------------------------------------------------------------------------------
def _examples(out: Path, arm: str, tok, fmt) -> tuple[list, dict]:
    import pilot_train as PT

    if arm in ("c1", "c1g"):
        cdir = out / "conduct"
        gt = _r(cdir, f"gate_{arm}")
        if not gt or not gt["passes"]:
            return [], {"reason": f"the {arm} gate did not pass"}
        sl = _r(cdir, "slots")["kept"]
        rows = _jl(cdir / f"kept_{arm}.jsonl")
        prompts = [_render(tok, fmt, GC.agent_messages(sl[r["slot_id"]], sl[r["slot_id"]]))
                   for r in rows]  # the student's prompt: nothing added
        return PT.recovery_examples(prompts, [r["gen_ids"] for r in rows]), {"items": len(rows)}
    if arm == "w_self":
        import gpc_advice as GA

        rows = _jl(out / "wself" / "train_items.jsonl")
        if not rows:
            return [], {"reason": "W-self has no training items"}
        prompts = [_render(tok, fmt, GA.student_messages(r["message"])) for r in rows]
        return PT.recovery_examples(prompts, [r["gen_ids"] for r in rows]), {"items": len(rows)}
    raise ValueError(arm)


def _matched_tokens(out: Path) -> tuple[int, str]:
    """Item 3: W-self's measured assistant tokens, or C1's when item 11 put C1g in its place; the
    Z5 seeded 1,000's count (977,883) when neither was trained."""
    third = (_r(out, "third_run") or {}).get("runs")
    for arm in (("w_self",) if third == "w_self" else ()) + ("c1",):
        tr = _r(out, f"train_{arm}")
        if tr and tr.get("trained"):
            return int(tr["assistant_tokens"]), arm
    return 977883, "z5_seeded_1000"


def stage_train(out: Path, arm: str, dry: bool) -> dict:
    import cst_pressure_train_pod as TP
    import cst_s0a_pod as S
    import pilot_pod_lib as PL
    import reasoning_formats as rf

    if arm != "w_ref" and arm in ("c1g", "w_self") and (_r(out, "third_run") or {}).get(
            "runs") != arm:
        rec = {"rung": arm, "trained": False, "reason": "not item 11's arm on this pod"}
        _w(out, f"train_{arm}", rec)
        return rec
    path = S.V._model_path(dry)  # noqa: SLF001
    tok = S.V._tok(path)  # noqa: SLF001
    if arm == "w_ref":
        import gpc_wref as GW

        target, matched = _matched_tokens(out)
        rows, dropped = GW.seeded(GW.fetch(dry), _index(dry))
        ex, info = GW.examples(tok, rows, target if not dry else 200)
        info |= {"matched_to": matched, "dropped_by_8gram": dropped}
        odir = out / WREF_DIR / "w_ref"
    else:
        ex, info = _examples(out, arm, tok, rf.build_format("qwen38_27b", tok))
        odir = out / "adapters" / arm
    if not ex:
        rec = {"rung": arm, "trained": False, **info}
        _w(out, f"train_{arm}", rec)
        print(">> train:", json.dumps(rec), flush=True)
        return rec
    _, wb, cfg = PL.load_model(dry)
    r = TP._train(wb, arm, {"examples": ex}, odir, dry)  # noqa: SLF001
    rec = {"rung": arm, "trained": True, **info, "tokens": sum(len(i) for i, _ in ex),
           "assistant_tokens": sum(sum(m) for _, m in ex),
           "lora": {k: cfg[k] for k in ("rank", "alpha", "dropout")}, "train": r,
           "adapters": dict(r["checkpoints"]),
           **({"label": "teacher not from the constitution"} if arm == "c1g" else {}),
           **({"diagnostic_only": "deleted on the pod, never downloaded (item 6)"}
              if arm == "w_ref" else {})}
    _w(out, f"train_{arm}", rec)
    print(">> train:", json.dumps({k: v for k, v in rec.items() if k != "train"}), flush=True)
    return rec


def stage_smoke_train(out: Path, dry: bool) -> dict:
    import cst_pressure_train_pod as TP
    import cst_s0a_pod as S
    import gpc_wref as GW
    import pilot_pod_lib as PL

    path = S.V._model_path(dry)  # noqa: SLF001
    tok = S.V._tok(path)  # noqa: SLF001
    rows, _ = GW.seeded(GW.fetch(dry), None)
    ex, _ = GW.examples(tok, rows[:2], 1)
    _, wb, _ = PL.load_model(dry)
    d = out / WREF_DIR / "smoke"
    r = TP._train(wb, "smoke", {"examples": ex}, d, dry, smoke=True)  # noqa: SLF001
    adir = list(r["checkpoints"].values())[-1]
    # the structural check, this session's HF base letters and the fixed smoke adapter's files
    # present (dry: its stand-in); bails on a failure (pilot_pod_lib.smoke_hf_side)
    side = PL.smoke_hf_side(wb, path, out, dry, [Path(adir)])
    shutil.rmtree(d, ignore_errors=True)
    rec = {"steps": r["steps"], **side, "passes": True,
           "adapter_deleted": not Path(adir).exists()}
    _w(out, "smoke_train", rec)
    print(">> smoke_train:", json.dumps(rec), flush=True)
    return rec


# ---- parity, screens, the registered read ------------------------------------------------------
def trained(out: Path) -> list[tuple[str, str]]:
    res = []
    for a in ARMS_TRAINED:
        tr = _r(out, f"train_{a}")
        if tr and tr.get("trained"):
            res.append((a, list(tr["adapters"].values())[-1]))
    return res


def readable(out: Path) -> list[tuple[str, str]]:
    """Trained arms whose served parity reads (a parity record says read or passes)."""
    res = []
    for t, d in trained(out):
        p = _r(out / "parity", t)
        if p and p.get("read", p.get("passes")):
            res.append((t, d))
    return res


def stage_parity_hf(out: Path, dry: bool) -> dict:
    import pilot_pod_lib as PL

    return PL.hf_letters(out, trained(out), dry)


def stage_parity(out: Path, dry: bool) -> dict:
    import pilot_pod_lib as PL

    return PL.stage_read(out, trained(out), dry, parts=(), base=False)


def _lora_engine(out: Path, dry: bool, tags: list[tuple[str, str]]):
    """One LoRA engine and, per tag, ``(engine, request)``: dry, a stand-in per adapter."""
    import cst_s0a_pod as S
    import pilot_pod_lib as PL

    path, eng = S.subject(dry, "a0", lora=not dry)
    per = {}
    for n, (t, adir) in enumerate(tags, 1):
        sdir = PL.served_copy(out, t, adir, dry)
        if dry:
            per[t] = (S.V.HFStandIn(path, lora=sdir), None)
        else:
            from vllm.lora.request import LoRARequest

            per[t] = (eng, LoRARequest(t, n, sdir))
    return path, per


def stage_screen(out: Path, dry: bool) -> dict:
    import reasoning_formats as rf

    tags = readable(out)
    path, per = _lora_engine(out, dry, tags)
    import cst_s0a_pod as S

    tok = S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format("qwen38_27b", tok)
    sl = _r(out / "conduct", "slots")
    rec = {"arms": [t for t, _ in tags], "holdout": bool(sl)}
    for t, _ in tags:
        eng, req = per[t]
        run_cells(out, f"screen_{t}", [(t, "s_hi", 1), (t, "s_lo", 1)], dry, eng, path,
                  loras={t: req})
        if sl:  # items 8 and 13: the held-out cell at k 1, nothing added to the context
            ks = sl["kept"]
            items = [(f"holdout_{t}|{i}|0", _render(tok, fmt, GC.agent_messages(ks[i], ks[i])))
                     for i in sl["holdout"]]
            _cached(out / "conduct" / f"holdout_{t}.jsonl", items, lambda it, t=t: letter_rows(
                ks, generate(eng, tok, it, dry, "letter", lora=req), f"holdout_{t}"))
    _w(out, "screen_run", rec)
    return rec


def stage_delete_wref(out: Path) -> dict:
    gone = []
    for d in (out / WREF_DIR, out / "served_adapters" / "w_ref", out / "adapters" / "w_ref"):
        if d.exists():
            shutil.rmtree(d)
            gone.append(str(d.relative_to(out)))
    left = [str(p) for p in out.rglob("*") if "w_ref" in str(p) and p.suffix == ".safetensors"]
    rec = {"rule": "GPC v0.2 item 6", "deleted": gone, "w_ref_weights_left": left,
           "passes": not left}
    _w(out, "wref_deleted", rec)
    print(">> wref_deleted:", json.dumps(rec), flush=True)
    if left:
        raise SystemExit(f"BAIL: W-ref weights remain: {left}")
    return rec


def stage_read_screen(out: Path) -> dict:
    import gpc1_read as RD

    lab = RD.labels()
    tags = [t for t, _ in readable(out)]
    screens = {t: RD.screen(out, t, lab) for t in tags}
    best = RD.best_arm(screens)
    rec = {"first_line": RD.first_line(screens), "rule": "GPC v0.2 §4.1 (clear 0.35 on the "
           "without-fallback ratio, the S-lo guard), item 6; the build read's item 2",
           "base_of_record_fallbacks": RD.record_fallbacks(), "screens": screens,
           "registered_arm": best}
    _w(out, "read_screen", rec)
    print(">> read_screen:", rec["first_line"], flush=True)
    print(">> read_screen:", json.dumps({t: (s["s_hi"].get("point"), s["clears"])
                                         for t, s in screens.items()} | {"best": best}),
          flush=True)
    return rec


def stage_registered(out: Path, dry: bool) -> dict:
    import gpc1_read as RD

    tag = (_r(out, "read_screen") or {}).get("registered_arm")
    if not tag:
        rec = {"run": False, "reason": "no arm clears the screen"}
        _w(out, "registered_run", rec)
        return rec
    adir = dict(trained(out))[tag]
    path, per = _lora_engine(out, dry, [(tag, adir)])
    eng, req = per[tag]
    run_cells(out, f"registered_{tag}", [(tag, "wide", 2)], dry, eng, path, loras={tag: req})
    r = RD.registered(out, tag)
    _w(out, "registered", r)
    print(">> registered:", json.dumps({k: r.get(k) for k in ("tag", "point", "ci95", "bar")}),
          flush=True)
    return r


def stage_operator(out: Path, dry: bool) -> dict:
    import pilot_pod_lib as PL

    tag = (_r(out, "read_screen") or {}).get("registered_arm")
    if not tag:
        return {"run": False}
    return PL.stage_read(out, [(tag, dict(trained(out))[tag])], dry,
                         parts=("panel_operator", "forced_plain"),
                         k_override={"panel_operator": 2, "forced_plain": 8}, base=True)


# ---- the cut rule -----------------------------------------------------------------------------
def remaining_h(out: Path, nxt: str) -> float:
    """Hours from ``nxt`` through f (``registered``), the MACHIAVELLI stages scaled by this pod's
    measured decisions an hour (stage b's), the rest at the plan's rates."""
    b = _r(out, "cells_gpc_b")
    scale = PLAN_DECISIONS_PER_H / b["decisions_per_h"] if b and b.get("decisions_per_h") else 1.0
    third = (_r(out, "third_arm") or {}).get("third_arm", "w_self")
    h = {"third": FORECAST_H[f"third_{third}"], "train": 3 * FORECAST_H["train_one"],
         "parity": FORECAST_H["parity"], "screen": FORECAST_H["screen"] * scale,
         "registered": FORECAST_H["registered"] * scale}
    order = ["third", "train", "parity", "screen", "registered"]
    if nxt in ("operator",):
        return FORECAST_H["operator"]
    return sum(h[k] for k in order[order.index(nxt):])


def cut_h() -> float:
    """The cut in force: the session's limits file (editable during the run) or 10.25 h."""
    sys.path.insert(0, str(ROOT / "papers/runpod_common"))
    import session_limits as SL

    return SL.cut_h() if os.environ.get("CST_LIMITS_FILE") else CUT_H


def fits(out: Path, nxt: str) -> bool:
    start = float(os.environ.get("CST_START_EPOCH", time.time()))
    el = (time.time() - start) / 3600
    fc = remaining_h(out, nxt)
    cut = cut_h()
    rec = {"rule": "GPC v0.2 §5 (the cut rule)", "stage": nxt, "elapsed_h": el,
           "forecast_through_f_h": fc, "ends_h": el + fc, "cut_h": cut, "runs": el + fc <= cut}
    _w(out, f"cut_{nxt}", rec)
    print(">> cut rule:", json.dumps(rec), flush=True)
    return rec["runs"]


STAGES = ("smoke_train", "incontext", "read_b", "conduct", "third", "train", "parity_hf",
          "parity", "screen", "delete_wref", "read_screen", "registered", "operator", "fits")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--arm", choices=ARMS_TRAINED)
    ap.add_argument("--next", choices=("third", "registered", "operator"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out, dry = a.out, a.dry_run
    if a.stage == "fits":
        sys.exit(0 if fits(out, a.next) else 1)
    if a.stage in ("read_b", "delete_wref", "read_screen"):
        {"read_b": stage_read_b, "delete_wref": stage_delete_wref,
         "read_screen": stage_read_screen}[a.stage](out)
        return
    import cst_finalist_pod as F

    F.configure()
    if a.stage == "train":
        if a.arm is None:
            ap.error("--stage train needs --arm")
        stage_train(out, a.arm, dry)
        return
    {"smoke_train": stage_smoke_train, "incontext": stage_incontext, "conduct": stage_conduct,
     "third": stage_third, "parity_hf": stage_parity_hf, "parity": stage_parity,
     "screen": stage_screen, "registered": stage_registered,
     "operator": stage_operator}[a.stage](out, dry)


if __name__ == "__main__":
    main()
