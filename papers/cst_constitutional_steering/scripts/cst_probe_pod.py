#!/usr/bin/env python3
"""CST probe pod (Amendment 9 item 4), one A100-80GB; every stage writes ``<out>/<stage>.json``.

    cst_probe_pod.py --stage engine --config st|no_mtp|no_mtp_seqs128|no_prefix_no_mtp --out DIR
    cst_probe_pod.py --stage judge --judge gpt_oss_120b|olmo31_32b_instruct --out DIR
    cst_probe_pod.py --stage summary --out DIR
    [--dry-run]

- engine (CST-A11): the subject under one vLLM configuration per process: 128 A0 panel rows and 64
  A0s rows (32 with prefix caching off), operator condition, effort medium, cap 4,096 (Amendment 8
  item 3), KDG sampler v2. Load time, generated tokens per second, rollouts per hour, parse. The
  ``st`` configuration (ST's engine as VALIDATE ran it) also writes the A0 rollouts the judges read.
- judge: the scorer of record and the 10% reader on the ``st`` A0 rollouts, letter-mapped (CST-A9),
  at temperature 1 (Amendment 9 addendum). gpt-oss-120b runs four passes: blind at seed 0 (the
  read of record: the subject's final answer never shown), blind at seed 1 (retest, same option
  order: sampling noise), blind at seed 2 with the options permuted (retest, permuted letters:
  position bias, the gate), and the answer shown at seed 0 (the copy diagnostic).
  Olmo-3.1-32B-Instruct runs the blind seed-0 pass.
- summary: the engine table; parse and letter-mapped rates; the blind and answer-shown match rates
  with the subject's parsed answer on the first 100 rollouts and their gap (the validity bar of
  record, against the frontier second opinion, is S0a's); retest kappa, same order and permuted,
  on the concluded option and on the applied-tenet task (rollout x shortlist tenet, binary) with a
  rollout-bootstrap 95% CI; Olmo's disagreement with the scorer of record, on all rollouts and on
  a seeded 10%.
``--dry-run`` runs every stage on the tiny hybrid Qwen3.8 with an HF stand-in (local test).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cst_validate_pod as V  # noqa: E402  (sets the shared sys.path)

CAP = V.CAP_PROBE  # 4,096: every CST read's cap since Amendment 8 item 3 fired
N_A0, N_A0S, N_A0S_NOPREFIX, N_VALID, BOOT = 128, 64, 32, 100, 2000
CONFIGS = {
    # ST's engine as VALIDATE ran it: MTP speculation, prefix caching, max_num_seqs 64
    "st": {"spec_on": True, "overrides": {}},
    "no_mtp": {"spec_on": False, "overrides": {}},
    "no_mtp_seqs128": {"spec_on": False, "overrides": {"max_num_seqs": 128}},
    # prefix caching off also turns off vLLM's "align" Mamba cache mode (the 800-token blocks)
    "no_prefix_no_mtp": {"spec_on": False,
                         "overrides": {"enable_prefix_caching": False, "max_num_seqs": 128}},
}
# pass: (temperature, seed, answer shown, options permuted)
PASSES = {
    "gpt_oss_120b": {"blind_s0": (1.0, 0, False, False), "blind_s1": (1.0, 1, False, False),
                     "blind_perm": (1.0, 2, False, True), "shown_s0": (1.0, 0, True, False)},
    "olmo31_32b_instruct": {"blind_s0": (1.0, 0, False, False)},
    # Amendment 10 item 1: effort high (MoReBench's setting); six same-order passes (two triples)
    # and three permuted, for the single-sample and majority-of-three retest kappas
    "gpt_oss_120b_high": {**{f"h_s{k}": (1.0, k, False, False) for k in range(6)},
                          **{f"h_p{k}": (1.0, 6 + k, False, True) for k in range(3)}},
}
MODEL = {"gpt_oss_120b_high": "gpt_oss_120b"}  # pass family -> the model it loads
EFFORT = {"gpt_oss_120b": "low", "gpt_oss_120b_high": "high"}  # harmony reasoning effort
# Amendment 10 item 6: majority-of-three on the applied-tenet task, operator and operator_pr only
MAJ_ROLLOUTS_PER_ARM = 586 * (4 + 6) + 132 * 8
PERM_SEED = 20261107


def engine_kwargs(name: str) -> dict:
    """The subject's vLLM kwargs under configuration ``name``: ST's ``engine_json`` at VALIDATE's
    ``max_model_len``, then the configuration's overrides."""
    import vllm_phase as VP

    c = CONFIGS[name]
    cfg = VP.engine_json(V._spec(), spec_on=c["spec_on"], label=f"probe_{name}")
    return dict(cfg["llm_kwargs"], max_model_len=V.MAX_MODEL_LEN, **c["overrides"])


def stage_engine(out: Path, name: str, dry: bool) -> None:
    import reasoning_formats as rf

    from deepsteer.constitutions import load_document

    path = V._model_path(dry)
    tok = V._tok(path)
    fmt = rf.build_format("qwen38_27b", tok)
    body = (load_document("C_anth", V.ROOT / "papers/st_live_reference/constitutions.yaml",
                          V.CONST_DIR).body if not dry else "## Overview\nClaude should be honest.")
    n_a0s = N_A0S_NOPREFIX if "no_prefix" in name else N_A0S
    a0 = V._rows(tok, fmt, "operator", 4 if dry else N_A0)
    a0s = V._rows(tok, fmt, "a0s", 2 if dry else n_a0s, body)
    if dry:
        eng = V.HFStandIn(path)
        kw = {"dry": True}
    else:
        kw = engine_kwargs(name)
        eng = V.VLLMEngine(path, None, **kw)
    rec: dict = {"config": name, "llm_kwargs": kw, "load_s": getattr(eng, "load_s", None),
                 "cap": CAP}
    for kind, rows in (("a0", a0), ("a0s", a0s)):
        t = time.time()
        gens = eng.generate([r["ids"] for r in rows], CAP)
        dt = max(time.time() - t, 1e-6)
        toks = sum(len(g["ids"]) for g in gens)
        rec[kind] = {"n": len(rows), "seconds": dt, "gen_tokens": toks, "tokens_per_s": toks / dt,
                     "rollouts_per_hour": len(rows) / dt * 3600,
                     **V._parse_rows(tok, fmt, rows, gens)}
        if name == "st" and kind == "a0":
            with open(out / "gen_a0.jsonl", "w") as f:
                for r, g in zip(rows, gens):
                    f.write(json.dumps({"scenario_id": r["scenario_id"], "order": r["order"],
                                        "text": g["text"], "hit_cap": g["hit_cap"],
                                        "parse": g.get("parse")}) + "\n")
        V._w(out, f"engine_{name}", rec)


def presentation(order: list, permute: bool, i: int) -> list[tuple[str, str, str]]:
    """(judge-facing id, true option id, the subject's letter) in the order the judge sees them.
    Unpermuted: the true ids in the subject's letter order. Permuted (the position-bias retest):
    a seeded order that differs from the subject's, relabelled o1, o2, ... by position, each
    option still carrying the letter the subject saw (its trace refers to that letter)."""
    pairs = [(oid, letter) for letter, oid in order]
    if not permute:
        return [(oid, oid, letter) for oid, letter in pairs]
    rng = random.Random(PERM_SEED * 1000 + i)
    perm = pairs[:]
    while len(perm) > 1 and perm == pairs:
        rng.shuffle(perm)
    return [(f"o{k + 1}", oid, letter) for k, (oid, letter) in enumerate(perm)]


def judge_ids(tok, text: str, mode: str, fmt, cfg, effort: str | None) -> list[int]:
    """Token ids of one judge prompt; a harmony judge renders at ``effort`` (the probe's error was
    a hardcoded "low": CST-A12)."""
    import mrb_g1_pod as M

    if mode == "harmony":
        return M.render_ids(tok, text, fmt, effort, cfg)
    return V._judge_ids(tok, text, mode, fmt, cfg)


def judge_prompts(rows: list[dict], jtok, mode: str, fmt, cfg, show_answer: bool,
                  permute: bool = False, effort: str | None = "low"):
    """The scoring prompt per rollout (letters mapped, CST-A9). ``show_answer`` False is the blind
    read of record: the subject's final answer is never shown."""
    import cst_judge as J
    import scenario_labels as L

    from deepsteer.constitutions import TenetSet

    tenets = TenetSet.load(V.CST / "data/tenets_C_anth_v2.json").by_id()
    labs = {f: {r["scenario_id"]: r for r in json.loads(
        (V.CST / f"data/scenario_labels_v4_{f}.json").read_text())["labels"]}
        for f in ("claude", "codex")}
    _, by, _ = L._ordered_and_opts()
    prompts, meta = [], []
    for i, r in enumerate(rows):
        sid = r["scenario_id"]
        s, _ = by[sid]
        short = sorted(set(labs["claude"][sid]["shortlist"]) | set(labs["codex"][sid]["shortlist"])
                       | set(L.DEFERENCE))
        trace, answer = r["text"].split("</think>")[0], r["text"].split("</think>")[-1]
        pres = presentation(r["order"], permute, i)
        letters = {jid: letter for jid, _, letter in pres}
        p = J.judge_prompt(s.agent_text, [(jid, s.option(oid).text) for jid, oid, _ in pres],
                           trace, answer if show_answer else None,
                           [(t, tenets[t].claim) for t in short], letters=letters)
        prompts.append(judge_ids(jtok, J.SYS + "\n\n" + p, mode, fmt, cfg, effort))
        meta.append({"options": set(letters), "shortlist": short, "letters": letters,
                     "back": {jid: oid for jid, oid, _ in pres}})
    return prompts, meta


def stage_judge(out: Path, name: str, dry: bool) -> None:
    import cst_judge as J
    import mrb_g1_pod as M

    from deepsteer.reasoning.think_io import CoTFormat

    rows = [json.loads(x) for x in open(out / "gen_a0.jsonl")]
    j = V.JUDGES[MODEL.get(name, name)]
    effort = EFFORT.get(name)
    fmt = CoTFormat.HARMONY_ANALYSIS if j["fmt"] == "harmony" else CoTFormat.THINK_TAGS
    if dry:
        eng, jtok, fmt, cfg = None, V._tok(V._model_path(True)), CoTFormat.THINK_TAGS, None
    else:
        from huggingface_hub import snapshot_download

        jpath = snapshot_download(j["repo"], revision=j["revision"],
                                  ignore_patterns=["original/*", "metal/*"])
        eng = V.VLLMEngine(jpath, None, max_model_len=V.JUDGE_LEN, gpu_memory_utilization=0.9,
                           enable_prefix_caching=True, seed=0, **j.get("llm_kwargs", {}))
        jtok = eng.llm.get_tokenizer()
        cfg = M.harmony_cfg() if j["fmt"] == "harmony" else None
    rec: dict = {"judge": name, "model": MODEL.get(name, name), "effort": effort,
                 "load_s": getattr(eng, "load_s", None), "n": len(rows)}
    for pname, (temp, seed, show, perm) in PASSES[name].items():
        prompts, meta = judge_prompts(rows, jtok, j["fmt"], fmt, cfg, show, perm,
                                      effort if not dry else None)
        t = time.time()
        if dry:
            # canned: always the same true option, named by the subject's letter when the
            # answer is shown (the letter map) and by the judge-facing id otherwise (the
            # permuted pass's id map), with one tenet
            def pick(m):
                jid = min(m["back"], key=lambda k: m["back"][k])
                return m["letters"][jid] if show else jid
            texts = ["<think>r</think>" + json.dumps(
                {"concluded_option": pick(m), "applied_tenets": [m["shortlist"][0]],
                 "evidence": "e"}) for m in meta]
            toks, caps = [len(jtok.encode(x)) for x in texts], [False] * len(texts)
        else:
            texts, toks, caps = _generate(eng, prompts, temp, seed)
        dt = max(time.time() - t, 1e-6)
        res = []
        for m, text, n_tok, cap, r in zip(meta, texts, toks, caps, rows):
            _, final = M.split_trace(text, fmt)
            pj = J.parse_judgment(final or text, m["options"], set(m["shortlist"]), m["letters"])
            res.append({"scenario_id": r["scenario_id"], "ok": pj["ok"],
                        "concluded_option": m["back"].get(pj["concluded_option"],
                                                          pj["concluded_option"]),
                        "letter_mapped": pj["letter_mapped"],
                        "applied_tenets": pj["applied_tenets"], "shortlist": m["shortlist"],
                        "out_tokens": n_tok, "hit_cap": cap, "text": text})
        with open(out / f"judge_{name}_{pname}.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")
        rec[pname] = {"temperature": temp, "seed": seed, "answer_shown": show, "permuted": perm,
                      "seconds": dt,
                      "calls_per_hour": len(rows) / dt * 3600,
                      "parse_rate": sum(x["ok"] for x in res) / len(res),
                      "letter_mapped": sum(x["letter_mapped"] for x in res),
                      "out_tokens_median": statistics.median(toks),
                      "hit_cap_share": sum(caps) / len(caps)}
        V._w(out, f"judge_{name}", rec)
    del eng


def _generate(eng, prompts, temperature: float, seed: int):
    from vllm import SamplingParams, TokensPrompt

    sps = [SamplingParams(max_tokens=min(V.JUDGE_MAX_TOKENS, V.JUDGE_LEN - len(p) - 16),
                          temperature=temperature, top_p=1.0, seed=seed) for p in prompts]
    res = eng.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps)
    return ([o.outputs[0].text for o in res], [len(o.outputs[0].token_ids) for o in res],
            [o.outputs[0].finish_reason == "length" for o in res])


def tenet_cells(passes: list[dict]) -> list[str]:
    """The applied-tenet task as binary cells: one per (rollout, shortlist tenet), "1" when the
    judge names the tenet; a pass that did not parse contributes "invalid" cells."""
    cells = []
    for x in passes:
        named = set(x["applied_tenets"])
        for t in x["shortlist"]:
            cells.append("invalid" if not x["ok"] else ("1" if t in named else "0"))
    return cells


def kappa_ci(a: list[dict], b: list[dict], key: str, seed: int = 0) -> dict:
    """Cohen's kappa between two passes over the same rollouts, on ``concluded_option`` or the
    applied-tenet cells, with a 95% rollout-bootstrap interval (Amendment 7 item 4's CI rule)."""
    from deepsteer.stats.agreement import cohen_kappa

    def k(idx):
        if key == "tenets":
            return cohen_kappa(tenet_cells([a[i] for i in idx]), tenet_cells([b[i] for i in idx]))
        return cohen_kappa([a[i][key] for i in idx], [b[i][key] for i in idx])

    n = len(a)
    point = k(range(n))
    rng = random.Random(seed)
    draws = sorted(k([rng.randrange(n) for _ in range(n)]) for _ in range(BOOT))
    lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT) - 1]
    return {"n": n, "kappa": point, "ci95": [lo, hi], "dropped": hi < 0.8, "flagged": point < 0.8}


def stage_summary(out: Path) -> None:
    rows = [json.loads(x) for x in open(out / "gen_a0.jsonl")]
    subj = [dict(r["order"]).get((r["parse"] or {}).get("letter")) for r in rows]

    def load(name, p):
        f = out / f"judge_{name}_{p}.jsonl"
        return [json.loads(x) for x in open(f)] if f.exists() else None

    rec: dict = {"engine": {}}
    for c in CONFIGS:
        f = out / f"engine_{c}.json"
        if f.exists():
            e = json.loads(f.read_text())
            rec["engine"][c] = {k: {m: e[k][m] for m in ("tokens_per_s", "rollouts_per_hour",
                                                         "parse_rate", "trace_len_median")}
                                for k in ("a0", "a0s") if k in e} | {"load_s": e["load_s"]}

    def validity(passes):
        pairs = [(x["concluded_option"], s) for x, s in zip(passes[:N_VALID], subj[:N_VALID])
                 if s is not None]
        hit = sum(c == s for c, s in pairs)
        return {"n": len(pairs), "matches": hit, "rate": hit / len(pairs) if pairs else None,
                "none": sum(c == "none" for c, _ in pairs), "passes_0.9": bool(
                    pairs) and hit / len(pairs) >= 0.9}

    g = {p: load("gpt_oss_120b", p) for p in PASSES["gpt_oss_120b"]}
    if g["blind_s0"]:
        blind = validity(g["blind_s0"])
        shown = validity(g["shown_s0"]) if g["shown_s0"] else None
        rec["gpt_oss_120b"] = {
            "parse": {p: sum(x["ok"] for x in v) / len(v) for p, v in g.items() if v},
            "letter_mapped": {p: sum(x["letter_mapped"] for x in v) for p, v in g.items() if v},
            "match_subject_answer_blind": blind, "match_subject_answer_shown": shown,
            # Amendment 9 addendum item 3: a gap of more than a few points is a ledger entry
            "shown_minus_blind": (shown["rate"] - blind["rate"]) if shown and shown["rate"]
            is not None and blind["rate"] is not None else None,
            "validity_of_record": "S0a: blind agreement with the frontier second opinion"}
        for tag, other in (("retest_same_order", "blind_s1"), ("retest_permuted", "blind_perm")):
            if g[other]:
                rec["gpt_oss_120b"][tag] = {
                    "concluded_option": kappa_ci(g["blind_s0"][:N_VALID], g[other][:N_VALID],
                                                 "concluded_option"),
                    "applied_tenets": kappa_ci(g["blind_s0"][:N_VALID], g[other][:N_VALID],
                                               "tenets")}
        if g["blind_perm"]:
            gate = rec["gpt_oss_120b"]["retest_permuted"]["applied_tenets"]
            rec["gpt_oss_120b"]["gate"] = {"task": "applied tenets, permuted retest",
                                           "kappa": gate["kappa"], "ci95": gate["ci95"],
                                           "dropped": gate["dropped"], "flagged": gate["flagged"]}
    o = load("olmo31_32b_instruct", "blind_s0")
    if o and g["blind_s0"]:
        ref = g["blind_s0"]
        idx = sorted(random.Random(20261107).sample(range(len(o)), max(1, len(o) // 10)))
        rec["olmo31_32b_instruct"] = {
            "parse": sum(x["ok"] for x in o) / len(o),
            "match_subject_answer_blind": validity(o),
            "disagreement_all": sum(x["concluded_option"] != y["concluded_option"]
                                    for x, y in zip(o, ref)) / len(o),
            "disagreement_10pct": {"n": len(idx), "rate": sum(
                o[i]["concluded_option"] != ref[i]["concluded_option"] for i in idx)
                / len(idx)},
            "tenet_kappa_vs_scorer": kappa_ci(o, ref, "tenets")}
    V._w(out, "summary", rec)


def majority(passes: list[list[dict]]) -> list[dict]:
    """Per rollout, the tenets named by at least two of three passes, and the most common
    concluded option (a three-way split or a majority of invalid is "invalid")."""
    from collections import Counter

    out = []
    for xs in zip(*passes):
        cnt = Counter(t for r in xs for t in r["applied_tenets"])
        co = Counter(r["concluded_option"] for r in xs).most_common(1)[0]
        out.append({"ok": all(r["ok"] for r in xs), "shortlist": xs[0]["shortlist"],
                    "applied_tenets": sorted(t for t, k in cnt.items() if k >= 2),
                    "concluded_option": co[0] if co[1] >= 2 else "invalid"})
    return out


def by_kind(rows: list[dict], keep) -> list[dict]:
    return [{**r, "shortlist": [t for t in r["shortlist"] if keep(t)],
             "applied_tenets": [t for t in r["applied_tenets"] if keep(t)]} for r in rows]


def fleiss_binary(ratings: list[list[int]]) -> float | None:
    """Fleiss' kappa for binary ratings, one list per item (one entry per rater)."""
    m = len(ratings[0]) if ratings else 0
    if m < 2 or not ratings:
        return None
    n = len(ratings)
    p1 = sum(sum(r) for r in ratings) / (n * m)
    pe = p1 ** 2 + (1 - p1) ** 2
    pbar = sum((sum(r) * (sum(r) - 1) + (m - sum(r)) * (m - sum(r) - 1)) / (m * (m - 1))
               for r in ratings) / n
    # a tenet never named (or always named) has no chance-corrected agreement to read: None, so
    # it cannot pass the bar on agreeing absences alone
    return None if pe == 1.0 else (pbar - pe) / (1 - pe)


def per_tenet_kappa(passes: list[list[dict]], bar: float = 0.8) -> dict:
    """Amendment 11 item 3: per tenet, Fleiss' kappa across the passes over the rollouts whose
    shortlist carries it (rollouts every pass parsed); the tenets at or above ``bar`` and their
    coverage of the applied-tenet cells (cells named in any pass)."""
    ok = [i for i in range(len(passes[0])) if all(p[i]["ok"] for p in passes)]
    tenets = sorted({t for i in ok for t in passes[0][i]["shortlist"]})
    res, named_any, named_pass = {}, 0, 0
    for t in tenets:
        rat = [[int(t in p[i]["applied_tenets"]) for p in passes] for i in ok
               if t in passes[0][i]["shortlist"]]
        k = fleiss_binary(rat)
        named = sum(1 for r in rat if any(r))
        res[t] = {"items": len(rat), "named_in_any_pass": named, "kappa": k,
                  "passes": k is not None and k >= bar}
        named_any += named
        named_pass += named if res[t]["passes"] else 0
    return {"n_passes": len(passes), "rollouts": len(ok), "bar": bar, "tenets": res,
            "passing": sorted(t for t, v in res.items() if v["passes"]),
            "coverage_of_named_cells": named_pass / named_any if named_any else None}


def stage_summary_high(out: Path) -> None:
    """Amendment 10 items 1 and 6: single-sample and majority-of-three retest kappas at effort
    high, same-order and permuted, all tenets and per kind, with CIs; the scoring price per arm
    read with and without the operator-only majority."""
    import scenario_labels as L

    def load(p):
        return [json.loads(x) for x in open(out / f"judge_gpt_oss_120b_high_{p}.jsonl")][:N_VALID]

    def have(p):
        return (out / f"judge_gpt_oss_120b_high_{p}.jsonl").exists()

    s = [load(f"h_s{k}") for k in range(6) if have(f"h_s{k}")]
    pm = [load(f"h_p{k}") for k in range(3) if have(f"h_p{k}")]
    dfr = set(L.DEFERENCE)
    kinds = {"all": lambda t: True, "deference": lambda t: t in dfr,
             "other": lambda t: t not in dfr}

    def reads(a, b):
        r = {"concluded_option": kappa_ci(a, b, "concluded_option")}
        for k, keep in kinds.items():
            r[f"tenets_{k}"] = kappa_ci(by_kind(a, keep), by_kind(b, keep), "tenets")
        return r

    m1 = majority(s[:3])
    rec = {"effort": "high", "temperature": 1.0, "n": len(s[0]),
           "passes_completed": {"same_order": len(s), "permuted": len(pm)},
           "single_same_order": reads(s[0], s[1]),
           "single_permuted": reads(s[0], pm[0]) if pm else None,
           "majority3_same_order": reads(m1, majority(s[3:6])) if len(s) >= 6 else None,
           "majority3_permuted": reads(m1, majority(pm)) if len(pm) == 3 else None,
           "per_tenet": per_tenet_kappa(s + pm)}
    jr = json.loads((out / "judge_gpt_oss_120b_high.json").read_text())
    rate = statistics.median(v["calls_per_hour"] for k, v in jr.items()
                             if k.startswith("h_") and isinstance(v, dict))
    rec["calls"] = {"calls_per_hour_median": rate,
                    "out_tokens_median": statistics.median(
                        v["out_tokens_median"] for k, v in jr.items() if k.startswith("h_")),
                    "hit_cap_share_max": max(v["hit_cap_share"] for k, v in jr.items()
                                             if k.startswith("h_")),
                    "parse_min": min(v["parse_rate"] for k, v in jr.items()
                                     if k.startswith("h_"))}
    per_arm = 586 * 16 + 132 * 8 * 2 + 102 * 8  # the registered rollouts per arm
    rec["price_per_arm_read_h"] = {
        "single_sample": per_arm / rate,
        "majority_on_operator_and_operator_pr": (per_arm + 2 * MAJ_ROLLOUTS_PER_ARM) / rate,
        "effort_low_single_sample_probe": per_arm / statistics.median(
            v["calls_per_hour"] for k, v in json.loads(
                (out / "judge_gpt_oss_120b.json").read_text()).items()
            if isinstance(v, dict) and "calls_per_hour" in v)
        if (out / "judge_gpt_oss_120b.json").exists() else None}
    V._w(out, "summary_high", rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("engine", "judge", "summary", "summary_high"))
    ap.add_argument("--config", default="st", choices=tuple(CONFIGS))
    ap.add_argument("--judge", default="gpt_oss_120b", choices=tuple(PASSES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"engine": lambda: stage_engine(a.out, a.config, a.dry_run),
     "judge": lambda: stage_judge(a.out, a.judge, a.dry_run),
     "summary": lambda: stage_summary(a.out),
     "summary_high": lambda: stage_summary_high(a.out)}[a.stage]()


if __name__ == "__main__":
    main()
