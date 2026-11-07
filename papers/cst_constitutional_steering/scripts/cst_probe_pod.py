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
- judge: the scorer of record and the 10% reader on the ``st`` A0 rollouts with the letter-mapped
  scorer (CST-A9). gpt-oss-120b runs four passes: temperature 0 with the final answer shown (the
  scoring read), temperature 0 with the final answer withheld (the validity read's copy check),
  and temperature 1 at seeds 0 and 1 (retest). Olmo-3.1-32B-Instruct runs the temperature-0 pass.
- summary: the engine table; parse and letter-mapped rates; the validity bar (concluded option =
  the subject's answer on the first 100 rollouts, with and without the answer shown); retest kappa
  on the concluded option and on the applied-tenet task (rollout x shortlist tenet, binary) with a
  rollout-bootstrap 95% CI; Olmo's disagreement with the scorer of record, on all rollouts and on a
  seeded 10%.
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
PASSES = {
    "gpt_oss_120b": {"t0": (0.0, 0, True), "t0_noanswer": (0.0, 0, False),
                     "t1_s0": (1.0, 0, True), "t1_s1": (1.0, 1, True)},
    "olmo31_32b_instruct": {"t0": (0.0, 0, True)},
}
WITHHELD = "(withheld)"


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


def judge_prompts(rows: list[dict], jtok, mode: str, fmt, cfg, show_answer: bool):
    """The scoring prompt per rollout (letters mapped, CST-A9); ``show_answer`` False withholds
    the subject's final answer, so agreement with it can only come from reading the reasoning."""
    import cst_judge as J
    import scenario_labels as L

    from deepsteer.constitutions import TenetSet

    tenets = TenetSet.load(V.CST / "data/tenets_C_anth_v2.json").by_id()
    labs = {f: {r["scenario_id"]: r for r in json.loads(
        (V.CST / f"data/scenario_labels_v4_{f}.json").read_text())["labels"]}
        for f in ("claude", "codex")}
    _, by, _ = L._ordered_and_opts()
    prompts, meta = [], []
    for r in rows:
        sid = r["scenario_id"]
        s, _ = by[sid]
        short = sorted(set(labs["claude"][sid]["shortlist"]) | set(labs["codex"][sid]["shortlist"])
                       | set(L.DEFERENCE))
        trace, answer = r["text"].split("</think>")[0], r["text"].split("</think>")[-1]
        letters = {o: L_ for L_, o in r["order"]}
        p = J.judge_prompt(s.agent_text, [(o, s.option(o).text) for _, o in r["order"]], trace,
                           answer if show_answer else WITHHELD,
                           [(t, tenets[t].claim) for t in short], letters=letters)
        prompts.append(V._judge_ids(jtok, J.SYS + "\n\n" + p, mode, fmt, cfg))
        meta.append({"options": set(letters), "shortlist": short, "letters": letters})
    return prompts, meta


def stage_judge(out: Path, name: str, dry: bool) -> None:
    import cst_judge as J
    import mrb_g1_pod as M

    from deepsteer.reasoning.think_io import CoTFormat

    rows = [json.loads(x) for x in open(out / "gen_a0.jsonl")]
    j = V.JUDGES[name]
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
    rec: dict = {"judge": name, "load_s": getattr(eng, "load_s", None), "n": len(rows)}
    for pname, (temp, seed, show) in PASSES[name].items():
        prompts, meta = judge_prompts(rows, jtok, j["fmt"], fmt, cfg, show)
        t = time.time()
        if dry:
            # canned: the letter of the first option (exercises the letter map) and one tenet
            texts = ["<think>r</think>" + json.dumps(
                {"concluded_option": m["letters"][sorted(m["options"])[0]],
                 "applied_tenets": [m["shortlist"][0]], "evidence": "e"}) for m in meta]
            toks, caps = [len(jtok.encode(x)) for x in texts], [False] * len(texts)
        else:
            texts, toks, caps = _generate(eng, prompts, temp, seed)
        dt = max(time.time() - t, 1e-6)
        res = []
        for m, text, n_tok, cap, r in zip(meta, texts, toks, caps, rows):
            _, final = M.split_trace(text, fmt)
            pj = J.parse_judgment(final or text, m["options"], set(m["shortlist"]), m["letters"])
            res.append({"scenario_id": r["scenario_id"], "ok": pj["ok"],
                        "concluded_option": pj["concluded_option"],
                        "letter_mapped": pj["letter_mapped"],
                        "applied_tenets": pj["applied_tenets"], "shortlist": m["shortlist"],
                        "out_tokens": n_tok, "hit_cap": cap, "text": text})
        with open(out / f"judge_{name}_{pname}.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")
        rec[pname] = {"temperature": temp, "seed": seed, "answer_shown": show, "seconds": dt,
                      "calls_per_hour": len(rows) / dt * 3600,
                      "parse_rate": sum(x["ok"] for x in res) / len(res),
                      "letter_mapped": sum(x["letter_mapped"] for x in res),
                      "out_tokens_median": statistics.median(toks)}
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
    if g["t0"]:
        rec["gpt_oss_120b"] = {
            "parse": {p: sum(x["ok"] for x in v) / len(v) for p, v in g.items() if v},
            "letter_mapped": {p: sum(x["letter_mapped"] for x in v) for p, v in g.items() if v},
            "validity_answer_shown": validity(g["t0"]),
            "validity_answer_withheld": validity(g["t0_noanswer"]) if g["t0_noanswer"] else None}
        if g["t1_s0"] and g["t1_s1"]:
            rec["gpt_oss_120b"]["retest_t1"] = {
                "concluded_option": kappa_ci(g["t1_s0"][:N_VALID], g["t1_s1"][:N_VALID],
                                             "concluded_option"),
                "applied_tenets": kappa_ci(g["t1_s0"][:N_VALID], g["t1_s1"][:N_VALID], "tenets")}
            rec["gpt_oss_120b"]["t0_vs_t1_s0_tenets"] = kappa_ci(
                g["t0"][:N_VALID], g["t1_s0"][:N_VALID], "tenets")
    o = load("olmo31_32b_instruct", "t0")
    if o and g["t0"]:
        idx = sorted(random.Random(20261107).sample(range(len(o)), max(1, len(o) // 10)))
        rec["olmo31_32b_instruct"] = {
            "parse": sum(x["ok"] for x in o) / len(o),
            "validity_answer_shown": validity(o),
            "disagreement_all": sum(x["concluded_option"] != y["concluded_option"]
                                    for x, y in zip(o, g["t0"])) / len(o),
            "disagreement_10pct": {"n": len(idx), "rate": sum(
                o[i]["concluded_option"] != g["t0"][i]["concluded_option"] for i in idx)
                / len(idx)},
            "tenet_kappa_vs_scorer": kappa_ci(o, g["t0"], "tenets")}
    V._w(out, "summary", rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("engine", "judge", "summary"))
    ap.add_argument("--config", default="st", choices=tuple(CONFIGS))
    ap.add_argument("--judge", default="gpt_oss_120b", choices=tuple(PASSES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"engine": lambda: stage_engine(a.out, a.config, a.dry_run),
     "judge": lambda: stage_judge(a.out, a.judge, a.dry_run),
     "summary": lambda: stage_summary(a.out)}[a.stage]()


if __name__ == "__main__":
    main()
