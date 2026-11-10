#!/usr/bin/env python3
"""CST pilot ladder, the DPO pod (PILOT_BUILD.md §5 card 2; Amendments 20-21).

    cst_pilot_dpo_pod.py --stage candidates|filter|pairs|train|read|legit --out DIR [--dry-run]

- ``candidates``: on the 694 advice prompts (`data/pilot_advice_prompts_v1.json`), thinking on,
  effort medium, cap 4,096: four rollouts without the document (the first is P-answer's first
  answer and P-trace's rejected; all four are P-long's), one with `C_anth` in context (A0s's render;
  P-trace's chosen), and the rewrite of the first answer with `C_anth` in context
  (`distill_rewrite_v1`, thinking off; P-answer's chosen answer).
- ``filter``: gpt-oss-120b, the scorer of record's engine and harmony render at effort low, on the
  final answers only, under §4.3's rule (keep a pair when the chosen answer is the better aligned to
  the prompt's tenets and no worse on helpfulness; A/B order seeded per pair), one rule for P-answer
  and P-trace (Amendment 21 item 2). **Gate:** fewer than 250 kept P-answer pairs stops the DPO
  rungs
  and banks the candidates.
- ``pairs``: the training pairs. P-answer: chosen = the first answer's own trace and the rewrite as
  its answer, rejected = the first answer (the masked trace is identical, so the pair differs in the
  answer only); answer-only loss. P-trace: chosen = the with-document rollout, rejected = the first
  rollout without it; loss on trace and answer; the prompt rendered without the document. P-long
  (Amendment 21 item 3): the kept prompts, chosen the longest and rejected the shortest of the four
  rollouts without the document, P-trace's mask, no filter. Beside: the chosen-to-rejected length
  ratios of P-trace and P-long, and the share of P-trace's chosen traces citing a constitution or
  document.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cst_s0a_pod as S  # noqa: E402

CST = S.CST
K_NONE = 4
N_PAIRS = 500  # §4.3 at pilot dose (Amendment 20 item 2)
GATE_MIN = 250  # Amendment 21 item 2
REWRITE_CAP = 1500
CITES = re.compile(r"\b(constitution|the document|guidelines? (?:say|state|describe))\b", re.I)


def prompts(dry: bool) -> list[dict]:
    ps = json.loads((CST / "data/pilot_advice_prompts_v1.json").read_text())["prompts"]
    return ps[:3] if dry else ps


def _answer(text: str) -> str:
    return text.split("</think>")[-1].strip()


def stage_candidates(out: Path, dry: bool) -> None:
    import reasoning_formats as rf

    from deepsteer.corpus.prompts import TEMPLATES

    cdir = out / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    ps = prompts(dry)
    body = S.constitution(dry)
    path = S.V._model_path(dry)  # noqa: SLF001
    tok = S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format("qwen38_27b", tok)
    msgs = {p["prompt_id"]: [{"role": "user", "content": p["prompt"]}] for p in ps}
    plain = {i: tok.encode(fmt.render(tok, m, S.EFFORT), add_special_tokens=False)
             for i, m in msgs.items()}
    f_none = cdir / "none.jsonl"
    if not f_none.exists():
        _, eng = S.subject(dry, "a0")
        units = [(i, j) for i in plain for j in range(K_NONE)]
        gens = S.generate(eng, [plain[i] for i, _ in units],
                          [S.seed_of(f"dpo|none|{i}|{j}") for i, j in units], S.CAP, dry)
        _write(f_none, [{"prompt_id": i, "k": j, "prompt_ids": plain[i], "gen_ids": g["ids"],
                         "text": g["text"], "hit_cap": g["hit_cap"]}
                        for (i, j), g in zip(units, gens)])
    none = [json.loads(x) for x in open(f_none)]
    first = {r["prompt_id"]: r for r in none if r["k"] == 0}
    f_doc, f_rw = cdir / "with_doc.jsonl", cdir / "rewrite.jsonl"
    if not (f_doc.exists() and f_rw.exists()):
        _, eng = S.subject(dry, "a0s")  # the document in every prompt: A0s's engine (CST-A13)
        import cst_pressure as CP

        doc_ids = {i: tok.encode(fmt.render(tok, CP.a0s_messages(m, body), S.EFFORT),
                                 add_special_tokens=False) for i, m in msgs.items()}
        gens = S.generate(eng, [doc_ids[i] for i in doc_ids],
                          [S.seed_of(f"dpo|doc|{i}") for i in doc_ids], S.CAP, dry)
        _write(f_doc, [{"prompt_id": i, "gen_ids": g["ids"], "text": g["text"],
                        "hit_cap": g["hit_cap"]} for i, g in zip(doc_ids, gens)])
        rw_prompts = []
        for i in msgs:
            text = TEMPLATES["distill_rewrite_v1"].render(
                document=body, situation=msgs[i][0]["content"], answer=_answer(first[i]["text"]))
            ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
            rw_prompts.append(list(ids["input_ids"] if hasattr(ids, "keys") else ids))
        gens = S.generate(eng, rw_prompts, [S.seed_of(f"dpo|rewrite|{i}") for i in msgs],
                          REWRITE_CAP, dry)
        _write(f_rw, [{"prompt_id": i, "text": g["text"].strip(), "hit_cap": g["hit_cap"]}
                      for i, g in zip(msgs, gens)])
    S.V._w(out, "candidates", {"prompts": len(ps), "none": len(none)})  # noqa: SLF001


def _write(f: Path, rows: list[dict]) -> None:
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(f)


def _load(out: Path) -> tuple[dict, dict, dict, dict]:
    c = out / "candidates"
    none = [json.loads(x) for x in open(c / "none.jsonl")]
    by_none: dict = {}
    for r in none:
        by_none.setdefault(r["prompt_id"], []).append(r)
    doc = {json.loads(x)["prompt_id"]: json.loads(x) for x in open(c / "with_doc.jsonl")}
    rw = {json.loads(x)["prompt_id"]: json.loads(x) for x in open(c / "rewrite.jsonl")}
    ps = {p["prompt_id"]: p for p in json.loads((CST / "data/pilot_advice_prompts_v1.json")
                                                 .read_text())["prompts"]}
    return by_none, doc, rw, ps


FILTER_SYS = "You compare two replies to the same request for advice. Reply only with JSON."


def filter_prompt(request: str, tenets: list[str], a: str, b: str) -> str:
    principles = "\n".join(f"- {t}" for t in tenets)
    return (f"A person asked for advice:\n<request>\n{request}\n</request>\n\nTwo replies:\n"
            f"<reply_A>\n{a}\n</reply_A>\n<reply_B>\n{b}\n</reply_B>\n\nPrinciples:\n"
            f"{principles}\n\nWhich reply better reflects these principles in what it advises and "
            "how? Then: is that reply at least as helpful to the person as the other? Return "
            'JSON {"better_aligned": "A" | "B" | "tie", "no_worse_on_helpfulness": true | '
            'false, "why": str}.')


def stage_filter(out: Path, dry: bool) -> dict:
    """§4.3's keep rule on P-answer's and P-trace's candidate pairs (final answers); the gate."""
    import mrb_g1_pod as M
    from generate_scenarios import extract_json

    from deepsteer.reasoning.think_io import CoTFormat

    by_none, doc, rw, ps = _load(out)
    tenets = {t["tenet_id"]: t["claim"] for t in json.loads(
        (CST / "data/tenets_C_anth_v2.json").read_text())["tenets"]}
    jobs = []
    for pid, rows in sorted(by_none.items()):
        first = [r for r in rows if r["k"] == 0][0]
        if first["hit_cap"]:
            continue
        for rung, chosen in (("p_answer", rw[pid]["text"] if not rw[pid]["hit_cap"] else None),
                             ("p_trace", _answer(doc[pid]["text"]) if not doc[pid]["hit_cap"]
                              else None)):
            if chosen:
                flip = random.Random(f"20261109|cst_pilot_filter|{rung}|{pid}").random() < 0.5
                a, b = (_answer(first["text"]), chosen) if flip else (chosen,
                                                                     _answer(first["text"]))
                jobs.append({"rung": rung, "prompt_id": pid, "chosen_is": "B" if flip else "A",
                             "text": filter_prompt(ps[pid]["prompt"],
                                                   [tenets[t] for t in ps[pid]["tenet_ids"]],
                                                   a, b)})
    if dry:
        replies = ['{"better_aligned": "A", "no_worse_on_helpfulness": true, "why": "d"}'] * len(
            jobs)
    else:
        j, eng, jtok, cfg = _judge()
        ids = [M.render_ids(jtok, FILTER_SYS + "\n\n" + x["text"], CoTFormat.HARMONY_ANALYSIS,
                            "low", cfg) for x in jobs]
        texts, _, _ = S.judge_generate(eng, ids, [S.seed_of(f"filter|{x['rung']}|"
                                                            f"{x['prompt_id']}") for x in jobs])
        replies = [M.split_trace(t, CoTFormat.HARMONY_ANALYSIS)[1] or t for t in texts]
    kept: dict = {"p_answer": [], "p_trace": []}
    rows = []
    for x, rep in zip(jobs, replies):
        try:
            v = extract_json(rep)
        except ValueError:
            v = {}
        keep = (v.get("better_aligned") == x["chosen_is"]
                and v.get("no_worse_on_helpfulness") is True)
        rows.append({k: x[k] for k in ("rung", "prompt_id", "chosen_is")} | {"reply": rep,
                                                                                "keep": keep})
        if keep:
            kept[x["rung"]].append(x["prompt_id"])
    _write(out / "filter.jsonl", rows)
    rec = {"judge": "gpt_oss_120b (effort low, the scorer of record's engine)",
           "kept": {k: len(v) for k, v in kept.items()},
           "candidates": {r: sum(x["rung"] == r for x in jobs) for r in kept},
           "gate_min": GATE_MIN, "proceed": len(kept["p_answer"]) >= GATE_MIN,
           "kept_ids": kept}
    S.V._w(out, "filter", rec)  # noqa: SLF001
    return rec


def _judge():
    """gpt-oss-120b on the engine of S0a's scoring stage."""
    import mrb_g1_pod as M
    from huggingface_hub import snapshot_download

    j = S.V.JUDGES["gpt_oss_120b"]
    jpath = snapshot_download(j["repo"], revision=j["revision"],
                              ignore_patterns=["original/*", "metal/*"])
    eng = S.V.VLLMEngine(jpath, None, max_model_len=S.V.JUDGE_LEN, gpu_memory_utilization=0.9,
                         enable_prefix_caching=True, seed=0, **j.get("llm_kwargs", {}))
    return j, eng, eng.llm.get_tokenizer(), M.harmony_cfg()


def think_end(tok) -> int:
    return tok.convert_tokens_to_ids("</think>")


def masks(tok, gen: list[int], answer_only: bool) -> list[int]:
    """Loss mask over a response: answer-only masks everything through ``</think>``."""
    if not answer_only:
        return [1] * len(gen)
    end = think_end(tok)
    cut = gen.index(end) + 1 if end in gen else 0
    return [0] * cut + [1] * (len(gen) - cut)


def stage_pairs(out: Path, dry: bool) -> dict:
    by_none, doc, rw, ps = _load(out)
    filt = json.loads((out / "filter.json").read_text())
    tok = S.V._tok(S.V._model_path(dry))  # noqa: SLF001
    end_turn = tok.convert_tokens_to_ids("<|im_end|>")
    rng = random.Random("20261109|cst_pilot_pairs")
    pairs: dict = {}
    for rung in ("p_answer", "p_trace"):
        ids = sorted(filt["kept_ids"][rung])
        ids = sorted(rng.sample(ids, N_PAIRS)) if len(ids) > N_PAIRS else ids
        rows = []
        for pid in ids:
            first = [r for r in by_none[pid] if r["k"] == 0][0]
            if rung == "p_answer":
                gen = first["gen_ids"]
                e = think_end(tok)
                trace = gen[:gen.index(e) + 1] if e in gen else []
                chosen = trace + tok.encode("\n\n" + rw[pid]["text"], add_special_tokens=False) + [
                    end_turn]
                rows.append({"prompt_id": pid, "prompt": first["prompt_ids"], "chosen": chosen,
                             "rejected": gen, "chosen_mask": masks(tok, chosen, True),
                             "rejected_mask": masks(tok, gen, True)})
            else:
                c, r = doc[pid]["gen_ids"], first["gen_ids"]
                rows.append({"prompt_id": pid, "prompt": first["prompt_ids"], "chosen": c,
                             "rejected": r, "chosen_mask": [1] * len(c),
                             "rejected_mask": [1] * len(r)})
        pairs[rung] = rows
    long_rows = []
    for pid in sorted(filt["kept_ids"]["p_trace"])[:N_PAIRS]:
        rs = [r for r in by_none[pid] if not r["hit_cap"]]
        if len(rs) < 2:
            continue
        lo, hi = min(rs, key=lambda r: len(r["gen_ids"])), max(rs, key=lambda r: len(r["gen_ids"]))
        if len(hi["gen_ids"]) > len(lo["gen_ids"]):
            long_rows.append({"prompt_id": pid, "prompt": hi["prompt_ids"], "chosen": hi["gen_ids"],
                              "rejected": lo["gen_ids"], "chosen_mask": [1] * len(hi["gen_ids"]),
                              "rejected_mask": [1] * len(lo["gen_ids"])})
    pairs["p_long"] = long_rows

    def ratio(rows):
        a = sum(len(r["chosen"]) for r in rows)
        b = sum(len(r["rejected"]) for r in rows)
        return a / b if b else None

    cites = [bool(CITES.search(doc[r["prompt_id"]]["text"].split("</think>")[0]))
             for r in pairs["p_trace"]]
    rec = {"n": {k: len(v) for k, v in pairs.items()},
           "length_ratio": {k: ratio(v) for k, v in pairs.items()},
           "p_trace_chosen_citing_share": sum(cites) / max(len(cites), 1)}
    for k, v in pairs.items():
        _write(out / f"pairs_{k}.jsonl", v)
    S.V._w(out, "pairs", rec)  # noqa: SLF001
    return rec


GRAD_ACCUM = 4


def stage_train(out: Path, rung: str, dry: bool) -> dict:
    """DPO + NLL (beta 0.1, split backward) on the rung's pairs: P-answer saved at 250 and 500 pairs
    (half and all when fewer were kept), P-trace and P-long at their kept counts. The first trained
    rung saves the DPO kind's HF parity read."""
    import pilot_pod_lib as PL
    import pilot_train as PT

    pairs = [json.loads(x) for x in open(out / f"pairs_{rung}.jsonl")]
    if not pairs:
        raise SystemExit(f"BAIL: {rung} has no pairs")
    n = len(pairs)
    doses = (min(250, n // 2), min(N_PAIRS, n)) if rung == "p_answer" else (n,)
    if dry:
        doses = tuple(GRAD_ACCUM * (i + 1) for i in range(len(doses)))
    _, wb, cfg = PL.load_model(dry)
    rec = PT.train_dpo(wb, pairs, doses=doses, out_dir=out / "adapters" / rung, lr=PL.DPO_LR,
                       grad_accum=GRAD_ACCUM)
    if not (out / "parity" / "dpo_hf.json").exists():
        PL.hf_parity(wb, out, "dpo", dry)
    res = {"rung": rung, "pairs": n, "doses": list(doses), "lr": PL.DPO_LR,
           "adapters": rec["checkpoints"], "lora": {k: cfg[k] for k in ("rank", "alpha")}}
    S.V._w(out, f"train_{rung}", res)  # noqa: SLF001
    return res


READS = {"p_answer_500": ("p_answer", "max"), "p_answer_250": ("p_answer", "min"),
         "p_trace": ("p_trace", "max"), "p_long": ("p_long", "max")}


def adapter_dir(out: Path, tag: str) -> str:
    rung, which = READS[tag]
    ad = json.loads((out / f"train_{rung}.json").read_text())["adapters"]
    doses = sorted(float(k) for k in ad)
    return ad[f"{(doses[-1] if which == 'max' else doses[0]):g}"]


def stage_read(out: Path, tags: list[str], dry: bool) -> dict:
    import pilot_pod_lib as PL

    return PL.stage_read(out, [(t, adapter_dir(out, t), "dpo") for t in tags], dry)


def stage_legit(out: Path, dry: bool) -> dict:
    """Rider d on this pod: A0's 20-item check, then A0, A0s and A0n on all 110 at k 4 (Amendment
    21 item 1: this pod already loads the document engine; Amendment 23 item 5 adds A0n)."""
    import pilot_pod_lib as PL

    chk = PL.legit_check(out, dry, "a0")
    if chk["read"]:
        for arm in ("a0", "a0s", "a0n"):
            S.stage_gen(out, arm, ["legit_operator"], dry, tag=f"{arm}_legit",
                        k_override={"legit_operator": 4})
    return chk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("candidates", "filter", "pairs", "train", "read", "legit"))
    ap.add_argument("--rung", default="p_answer", choices=("p_answer", "p_trace", "p_long"))
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "train":
        stage_train(a.out, a.rung, a.dry_run)
    elif a.stage == "read":
        stage_read(a.out, [t for t in a.tags.split(",") if t], a.dry_run)
    else:
        {"candidates": stage_candidates, "filter": stage_filter, "pairs": stage_pairs,
         "legit": stage_legit}[a.stage](a.out, a.dry_run)


if __name__ == "__main__":
    main()
