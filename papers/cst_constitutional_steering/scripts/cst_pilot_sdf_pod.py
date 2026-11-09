#!/usr/bin/env python3
"""CST pilot ladder, the SDF pod (PILOT_BUILD.md §5 card 1; Amendments 20-21).

    cst_pilot_sdf_pod.py --stage corpus --arm d_doc|d_neutral --out DIR [--dry-run]

Stages (built so far):

- ``corpus``: the arm's documents in plan order (`data/pilot_corpus_plan_v1.json`), each drafted
  from its registered template with the arm's document in context (`C_anth` for D-doc, the A0n text
  for D-neutral; the template carries it, so the system message is empty), then rewritten by the
  critique template; thinking off (documents, not decisions). Chunks of 64 documents, resumed by
  chunk. It stops once the final documents reach the arm's token target (1M; the 0.3M checkpoint is
  the prefix the trainer reads first). **Pilot gate** after the first 50 finals: 500-1,400 words,
  no chat or think markers, under 5% sharing an 8-gram with the eval index (D11); else the stage
  writes `corpus_gate_<arm>.json` with ``proceed`` false and stops.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cst_s0a_pod as S  # noqa: E402  (the subject, engine of record and sampler)

CST = S.CST
TARGET_TOKENS = 1_000_000
CHUNK = 64
DRAFT_CAP = 2200  # 1,400 words is about 1,900 tokens
GATE_N, GATE = 50, {"words": (500, 1400), "overlap_max": 0.05}
MARKERS = re.compile(r"(^|\n)\s*(User|Assistant)\s*:|<think>|</think>|<\|im_", re.I)


def items_text(arm: str, ids: list[str]) -> str:
    """The template's ``{tenets}`` block: one claim per line with its id."""
    if arm == "d_doc":
        by = {t["tenet_id"]: t for t in json.loads((CST / "data/tenets_C_anth_v2.json")
                                                     .read_text())["tenets"]}
        return "\n".join(f"[{i}] {by[i]['claim']}" for i in ids)
    by = {s["statement_id"]: s for s in json.loads((CST / "data/statements_a0n_v1.json")
                                                     .read_text())["statements"]}
    return "\n".join(f"[{i}] {by[i]['claim']}" for i in ids)


def document(arm: str, dry: bool) -> str:
    return S.constitution(dry) if arm == "d_doc" else S.neutral(dry)


def critique_id(arm: str) -> str:
    return "sdf_critique_v1" if arm == "d_doc" else "sdf_critique_neutral_v1"


def no_think_ids(tok, text: str) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def gate(rows: list[dict], index) -> dict:
    import overlap_hashes as OH

    first = rows[:GATE_N]
    words = [len(r["final"].split()) for r in first]
    lo, hi = GATE["words"]
    bad_len = sum(not lo <= w <= hi for w in words)
    marked = sum(bool(MARKERS.search(r["final"])) for r in first)
    hit = sum(OH.shared(r["final"], index) > 0 for r in first) if index is not None else 0
    rec = {"n": len(first), "words_out_of_range": bad_len, "markers": marked,
           "overlap_docs": hit, "overlap_share": hit / max(len(first), 1), **GATE}
    rec["proceed"] = bool(len(first) == GATE_N and bad_len == 0 and marked == 0
                          and rec["overlap_share"] < GATE["overlap_max"])
    return rec


def stage_corpus(out: Path, arm: str, dry: bool, target: int = TARGET_TOKENS) -> None:
    import overlap_hashes as OH

    from deepsteer.corpus.prompts import TEMPLATES

    plan = json.loads((CST / "data/pilot_corpus_plan_v1.json").read_text())["arms"][arm]
    doc = document(arm, dry)
    cdir = out / "corpus" / arm
    cdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for f in sorted(cdir.glob("chunk_*.jsonl")) for x in open(f)]
    toks = sum(r["final_tokens"] for r in rows)
    index = OH.load(S.V.CONST_DIR / "eval_8gram_hashes_v1.npy") if not dry else None
    path, eng = S.subject(dry, "a0s")  # a 34k document in every prompt: A0s's engine (CST-A13)
    tok = S.V._tok(path)  # noqa: SLF001
    gate_f = out / f"corpus_gate_{arm}.json"
    for c0 in range(len(rows) - len(rows) % CHUNK, len(plan), CHUNK):
        if toks >= target:
            break
        if len(rows) >= GATE_N and not gate_f.exists():
            g = gate(rows, index)
            S.V._w(out, f"corpus_gate_{arm}", g)  # noqa: SLF001
            if not g["proceed"]:
                return
        spec = plan[c0:c0 + CHUNK]
        t0 = time.time()
        drafts = S.generate(eng, [no_think_ids(tok, TEMPLATES[d["template_id"]].render(
            document=doc, tenets=items_text(arm, d["items"]), **(
                {"setting": d["setting"]} if d["setting"] else {}))) for d in spec],
            [S.seed_of(f"{d['doc_id']}|draft") for d in spec], DRAFT_CAP, dry)
        finals = S.generate(eng, [no_think_ids(tok, TEMPLATES[critique_id(arm)].render(
            document=doc, text=g["text"])) for g in drafts],
            [S.seed_of(f"{d['doc_id']}|critique") for d in spec], DRAFT_CAP, dry)
        f = cdir / f"chunk_{c0 // CHUNK:03d}.jsonl"
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for d, g, fin in zip(spec, drafts, finals):
                r = {**d, "draft": g["text"], "draft_hit_cap": g["hit_cap"],
                     "final": fin["text"].strip(), "final_hit_cap": fin["hit_cap"],
                     "final_tokens": len(tok.encode(fin["text"], add_special_tokens=False)),
                     "templates": [d["template_id"], critique_id(arm)]}
                fh.write(json.dumps(r) + "\n")
                rows.append(r)
                toks += r["final_tokens"]
        tmp.rename(f)
        S.V._w(out, f"corpus_{arm}_log_{c0 // CHUNK:03d}", {  # noqa: SLF001
            "arm": arm, "docs": len(rows), "tokens": toks, "seconds": time.time() - t0})
    if len(rows) >= GATE_N and not gate_f.exists():
        S.V._w(out, f"corpus_gate_{arm}", gate(rows, index))  # noqa: SLF001
    S.V._w(out, f"corpus_{arm}", {"docs": len(rows), "tokens": toks, "target": target})  # noqa: SLF001


def stage_chat(out: Path, dry: bool) -> None:
    """D-chat (§4.2): D-doc's finals in plan order, each rendered by ``chat_render_v1`` (no document
    in context, thinking off, A0's engine), until the rendered conversations' tokens reach D-doc's
    (token-matched). A render that does not parse into a user turn and an assistant turn is dropped
    and counted (`pilot_train.parse_chat`)."""
    import pilot_train as PT

    from deepsteer.corpus.prompts import TEMPLATES

    docs = [json.loads(x) for f in sorted((out / "corpus" / "d_doc").glob("chunk_*.jsonl"))
            for x in open(f)]
    if not docs:
        raise SystemExit("BAIL: no D-doc corpus to render")
    target = sum(d["final_tokens"] for d in docs)
    path, eng = S.subject(dry, "a0")
    tok = S.V._tok(path)  # noqa: SLF001
    cdir = out / "corpus" / "d_chat"
    cdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for f in sorted(cdir.glob("chunk_*.jsonl")) for x in open(f)]
    toks, dropped = sum(r["tokens"] for r in rows if r["ok"]), sum(not r["ok"] for r in rows)
    for c0 in range(len(rows), len(docs), CHUNK):
        if toks >= target:
            break
        spec = docs[c0:c0 + CHUNK]
        gens = S.generate(eng, [no_think_ids(tok, TEMPLATES["chat_render_v1"].render(
            text=d["final"])) for d in spec], [S.seed_of(f"{d['doc_id']}|chat") for d in spec],
            DRAFT_CAP, dry)
        f = cdir / f"chunk_{c0 // CHUNK:03d}.jsonl"
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for d, g in zip(spec, gens):
                msgs = PT.parse_chat(g["text"])
                ok = any(m["role"] == "user" for m in msgs) and any(
                    m["role"] == "assistant" for m in msgs) and not g["hit_cap"]
                n = len(tok.encode(g["text"], add_special_tokens=False))
                fh.write(json.dumps({"doc_id": d["doc_id"], "source": "d_doc", "kind": d["kind"],
                                     "items": d["items"], "text": g["text"], "ok": ok,
                                     "tokens": n, "template": "chat_render_v1"}) + "\n")
                toks += n if ok else 0
                dropped += not ok
        tmp.rename(f)
    # renders run about as long as their documents; when they run shorter the arm ends below the
    # match, which is recorded (not hidden) and read beside D-chat's result
    S.V._w(out, "corpus_d_chat", {"tokens": toks, "target": target,  # noqa: SLF001
                                   "matched": toks >= 0.95 * target, "dropped_renders": dropped})


def stage_recovery(out: Path, dry: bool, n_keep: int = 512) -> None:
    """The recovery set (CST-A10, Amendment 21 item 5): ``none``'s thinking-on reply (effort medium,
    cap 4,096) to every prompt of `data/recovery_pool_v1.json`, then ``n_keep`` replies whose trace
    lengths follow A0's panel deciles (`pilot_train.select_recovery`). Writes
    `recovery/replies.jsonl` and `recovery/selected.json`."""
    import pilot_train as PT
    import reasoning_formats as rf

    pool = json.loads((CST / "data/recovery_pool_v1.json").read_text())["prompts"]
    if dry:
        pool = pool[:24]
    rdir = out / "recovery"
    rdir.mkdir(parents=True, exist_ok=True)
    f = rdir / "replies.jsonl"
    if not f.exists():
        path, eng = S.subject(dry, "a0")
        tok = S.V._tok(path)  # noqa: SLF001
        fmt = rf.build_format("qwen38_27b", tok)
        letters = rf.letter_token_ids(tok)
        prompts = [tok.encode(fmt.render(tok, [{"role": "user", "content": p["prompt"]}],
                                         S.EFFORT), add_special_tokens=False) for p in pool]
        gens = S.generate(eng, prompts, [S.seed_of(f"recovery|{p['id']}") for p in pool], S.CAP,
                          dry)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for p, ids, g in zip(pool, prompts, gens):
                pr = fmt.parse(tok, g["ids"], g["hit_cap"], letters)
                fh.write(json.dumps({"id": p["id"], "prompt_ids": ids, "gen_ids": g["ids"],
                                     "hit_cap": g["hit_cap"], "trace_len": pr.trace_len,
                                     "status": pr.status}) + "\n")
        tmp.rename(f)
    rows = [json.loads(x) for x in open(f)]
    edges = json.loads((CST / "data/recovery_target_v1.json").read_text())["decile_edges"]
    usable = {r["id"]: r["trace_len"] for r in rows if not r["hit_cap"] and r["trace_len"]}
    keep = PT.select_recovery(usable, edges, min(n_keep, len(usable)))
    S.V._w(out, "recovery", {"pool": len(rows), "usable": len(usable),  # noqa: SLF001
                              "kept": len(keep), "edges": edges})
    (rdir / "selected.json").write_text(json.dumps({"ids": keep}, indent=1) + "\n")


RUNGS = ("r0", "d_raw", "d_doc", "d_chat", "d_neutral")
DOSES = {"d_doc": (300_000, 1_000_000), "d_neutral": (1_000_000,), "d_chat": (1_000_000,)}
RAW_PASSES = 3  # D-raw: C_anth's body, a few passes (Amendment 20 item 2)


def _recovery_examples(out: Path) -> list:
    import pilot_train as PT

    rows = {json.loads(x)["id"]: json.loads(x) for x in open(out / "recovery/replies.jsonl")}
    keep = json.loads((out / "recovery/selected.json").read_text())["ids"]
    return PT.recovery_examples([rows[i]["prompt_ids"] for i in keep],
                                [rows[i]["gen_ids"] for i in keep])


def rung_data(out: Path, rung: str, tok, dry: bool) -> tuple[list, float, tuple]:
    """(examples, dose per optimizer step, doses) for a rung; SDF rungs pack documents, D-chat
    turns its conversations into answer-only examples (T0.6)."""
    import pilot_train as PT

    accum = 4
    seq = 256 if dry else PT.SEQ_LEN
    if rung == "d_chat":
        rows = [json.loads(x) for f in sorted((out / "corpus/d_chat").glob("chunk_*.jsonl"))
                for x in open(f)]
        ex = [e for r in rows if r["ok"] for e in PT.chat_examples(tok, PT.parse_chat(r["text"]))]
        mean = sum(len(i) for i, _ in ex) / max(len(ex), 1)
        return ex, mean * accum, DOSES["d_chat"]
    if rung == "d_raw":
        body = tok.encode(S.constitution(dry), add_special_tokens=False)
        chunks = PT.pack([body] * RAW_PASSES, tok.eos_token_id, seq)
        return [(c, None) for c in chunks], seq * accum, (len(chunks) * seq,)
    rows = [json.loads(x) for f in sorted((out / "corpus" / rung).glob("chunk_*.jsonl"))
            for x in open(f)]
    chunks = PT.pack([tok.encode(r["final"], add_special_tokens=False) for r in rows],
                     tok.eos_token_id, seq)
    return [(c, None) for c in chunks], seq * accum, DOSES[rung]


def stage_train(out: Path, rung: str, dry: bool, doses: tuple | None = None) -> dict:
    """Train ``rung``'s adapter to its doses, then the identical recovery step on each saved dose
    (Amendment 21 item 5; R0 is the recovery step alone on a fresh adapter). Adapters at
    ``adapters/<rung>/<dose>`` (without recovery) and ``adapters/<rung>/<dose>_rec``. The first
    trained rung also saves the SDF kind's HF parity read."""
    import pilot_pod_lib as PL
    import pilot_train as PT
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path, wb, cfg = PL.load_model(dry)
    tok = S.V._tok(path)  # noqa: SLF001
    adir = out / "adapters" / rung
    rec = {"rung": rung, "lora": {k: cfg[k] for k in ("rank", "alpha", "dropout")},
           "lr": PL.SDF_LR, "recovery_lr": PL.RECOVERY_LR}
    rex = _recovery_examples(out)
    if rung == "r0":
        rec["recovery"] = PT.recovery_step(wb, rex, lr=PL.RECOVERY_LR, out_dir=adir / "rec")
        rec["adapters"] = {"rec": str(adir / "rec")}
    else:
        ex, per_step, reg = rung_data(out, rung, tok, dry)
        if not ex:
            raise SystemExit(f"BAIL: {rung} has no training examples")
        # a dry run takes one optimizer step per registered dose, so every checkpoint path runs
        doses = doses or (tuple(per_step * (i + 1) for i in range(len(reg))) if dry else reg)
        rec["train"] = PT.train_rung(
            wb, ex, lambda e: PT.lm_loss(wb, e[0], e[1]), per_step_dose=per_step, doses=doses,
            out_dir=adir, lr=PL.SDF_LR)
        if not (out / "parity" / "sdf_hf.json").exists():
            PL.hf_parity(wb, out, "sdf", dry)
        rec["adapters"] = dict(rec["train"]["checkpoints"])
        for dose, d in list(rec["train"]["checkpoints"].items()):
            sd = load_file(str(Path(d) / "adapter_model.safetensors"))
            set_peft_model_state_dict(wb.model, sd)
            r = PT.recovery_step(wb, rex, lr=PL.RECOVERY_LR, out_dir=adir / f"{dose}_rec")
            rec.setdefault("recovery", {})[dose] = r
            rec["adapters"][f"{dose}_rec"] = str(adir / f"{dose}_rec")
    S.V._w(out, f"train_{rung}", rec)  # noqa: SLF001
    return rec


# read tag -> (rung, which dose, with recovery): the dose is picked by rank from the rung's training
# record (largest = "1M", smallest = "0.3M"), never by a hard-coded directory name
READS = {"d_doc_1M_rec": ("d_doc", "max", True), "d_doc_1M": ("d_doc", "max", False),
         "d_doc_03M_rec": ("d_doc", "min", True), "d_doc_03M": ("d_doc", "min", False),
         "r0": ("r0", None, True), "d_neutral_1M_rec": ("d_neutral", "max", True),
         "d_chat_1M_rec": ("d_chat", "max", True), "d_raw_rec": ("d_raw", "max", True)}


def adapter_dir(out: Path, tag: str) -> str:
    rung, which, rec = READS[tag]
    adapters = json.loads((out / f"train_{rung}.json").read_text())["adapters"]
    if which is None:
        return adapters["rec"]
    doses = sorted({float(k.removesuffix("_rec")) for k in adapters if k != "rec"})
    d = doses[-1] if which == "max" else doses[0]
    return adapters[f"{d:g}_rec" if rec else f"{d:g}"]


def stage_read_tags(out: Path, tags: list[str], dry: bool) -> dict:
    import pilot_pod_lib as PL

    return PL.stage_read(out, [(t, adapter_dir(out, t), "sdf") for t in tags], dry)


def stage_legit(out: Path, dry: bool) -> dict:
    """Rider d on this pod: A0's 20-item check, then A0 on all 110 at k 4 if the cell is not at
    floor."""
    import pilot_pod_lib as PL

    chk = PL.legit_check(out, dry, "a0")
    if chk["read"]:
        S.stage_gen(out, "a0", ["legit_operator"], dry, tag="a0_legit",
                    k_override={"legit_operator": 4})
    return chk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("corpus", "chat", "recovery", "train", "read", "legit"))
    ap.add_argument("--tags", default="", help="read: comma-separated READS tags")
    ap.add_argument("--rung", default="d_doc", choices=RUNGS)
    ap.add_argument("--arm", default="d_doc", choices=("d_doc", "d_neutral"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=TARGET_TOKENS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.stage == "corpus":
        stage_corpus(a.out, a.arm, a.dry_run, a.target)
    elif a.stage == "chat":
        stage_chat(a.out, a.dry_run)
    elif a.stage == "train":
        stage_train(a.out, a.rung, a.dry_run)
    elif a.stage == "read":
        stage_read_tags(a.out, [t for t in a.tags.split(",") if t], a.dry_run)
    elif a.stage == "legit":
        stage_legit(a.out, a.dry_run)
    else:
        stage_recovery(a.out, a.dry_run)


if __name__ == "__main__":
    main()
