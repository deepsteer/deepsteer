#!/usr/bin/env python3
"""CST Amendment 1 item 15 (addition 2): the recall read's inputs, zero GPU.

    recall_inputs.py passages             # draw the passages (seeded): data/recall_passages_v1.json
    recall_inputs.py paraphrase [--max N] # two paraphrases per continuation (claude -p), cached

Passages: 100 from the `C_anth` body and 100 from the `C_spec` body, each a window of 96 subject
tokens (Qwen3.8-27B's tokenizer at the pinned revision) inside one section: a 64-token prefix and a
32-token continuation. Positive band: 40 windows of the same shape from canonical public-domain
texts the subject has certainly seen (the U.S. Declaration of Independence, the Gettysburg
Address, Shakespeare's Sonnets; Project Gutenberg #1, #4, #1041, sha256 recorded). Each
continuation gets two independent meaning-preserving paraphrases: the recall margin is
log p(true continuation) − log p(paraphrase 1) given the true prefix, and the floor is the same
margin between paraphrase 1 and paraphrase 2. The read itself is a forward pass at S0.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers" / "kdg_panel" / "scripts")]

from deepsteer.constitutions import load_document  # noqa: E402

CST = ROOT / "papers" / "cst_constitutional_steering"
ST = ROOT / "papers" / "st_live_reference"
RAW = CST / "outputs_raw" / "recall"
OUT = CST / "data" / "recall_passages_v1.json"
PD = Path.home() / ".cache" / "deepsteer" / "public_domain"
PD_TEXTS = {"declaration": "pg1.txt", "gettysburg": "pg4.txt", "sonnets": "pg1041.txt"}
SEED, PREFIX, CONT = 20261106, 64, 32
N_DOC, N_PD = 100, 40
PER_CALL = 20


def _tok():
    from tokenizers import Tokenizer

    snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/"
                         "snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json"))
    return Tokenizer.from_file(snap[0])


def _gutenberg_body(text: str) -> str:
    a = text.index("*** START OF")
    a = text.index("\n", a) + 1
    b = text.index("*** END OF")
    return text[a:b].strip()


def _windows(chunks: list[str], n: int, rng: random.Random, tok) -> list[dict]:
    """``n`` non-overlapping 96-token windows, each inside one chunk, drawn uniformly over the
    eligible start positions."""
    cands = []
    for ci, c in enumerate(chunks):
        ids = tok.encode(c, add_special_tokens=False).ids
        for start in range(0, len(ids) - (PREFIX + CONT) + 1, 16):
            cands.append((ci, start))
    rng.shuffle(cands)
    used: dict[int, list[int]] = {}
    out = []
    for ci, start in cands:
        if any(abs(start - u) < PREFIX + CONT for u in used.get(ci, [])):
            continue
        ids = tok.encode(chunks[ci], add_special_tokens=False).ids[start:start + PREFIX + CONT]
        out.append({"chunk": ci, "start": start, "prefix_ids": ids[:PREFIX],
                    "cont_ids": ids[PREFIX:], "prefix": tok.decode(ids[:PREFIX]),
                    "continuation": tok.decode(ids[PREFIX:])})
        used.setdefault(ci, []).append(start)
        if len(out) == n:
            break
    return out


def passages() -> None:
    tok = _tok()
    rng = random.Random(SEED)
    res: dict = {"seed": SEED, "prefix_tokens": PREFIX, "continuation_tokens": CONT,
                 "tokenizer": "Qwen/Qwen3.8-27B@1d4bf0f2ff60", "sets": {}}
    for d in ("C_anth", "C_spec"):
        doc = load_document(d, ST / "constitutions.yaml", ST / "outputs" / "constitutions")
        chunks = [s.text for s in doc.sections()]
        res["sets"][d] = {"source": f"{d} body sha256 {doc.body_sha256}",
                          "items": _windows(chunks, N_DOC, rng, tok)}
    pd_items, shas = [], {}
    quota = {"declaration": N_PD // 3, "gettysburg": N_PD // 3}  # the rest from the sonnets
    for name, f in PD_TEXTS.items():
        raw = (PD / f).read_text()
        shas[name] = hashlib.sha256(raw.encode()).hexdigest()
        body = _gutenberg_body(raw)
        paras = [p for p in body.split("\n\n") if len(p.split()) > 20] or [body]
        joined = ["\n\n".join(paras)]
        want = quota.get(name, N_PD - len(pd_items))
        for it in _windows(joined, want, rng, tok):  # a short text gives what it can
            it["text"] = name
            pd_items.append(it)
    res["sets"]["public_domain"] = {"source": {k: f"Project Gutenberg {PD_TEXTS[k]} sha256 {v}"
                                               for k, v in shas.items()}, "items": pd_items}
    for sname, sd in res["sets"].items():
        for i, it in enumerate(sd["items"]):
            it["id"] = f"{sname}:{i:03d}"
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print({k: len(v["items"]) for k, v in res["sets"].items()})


SYS = "You rewrite text. Answer only with the JSON asked for."


def _prompt(batch: list[dict]) -> str:
    lines = "\n".join(f'[{it["id"]}] {json.dumps(it["continuation"], ensure_ascii=False)}'
                      for it in batch)
    return (
        "Each item is a short fragment of text (it may start or end mid-sentence). Write TWO "
        "independent paraphrases of each fragment: the same meaning, the same register, about the "
        "same length (within 20%), but different wording, so that no run of four or more words "
        "matches the original or each other. Keep the fragment's partial-sentence shape: if it "
        "starts or ends mid-sentence, so does the paraphrase. Return JSON {\"items\": [{\"id\": "
        "str, \"p1\": str, \"p2\": str}, ...]} covering every id.\n\n" + lines)


def paraphrase(max_calls: int | None) -> None:
    from generate_scenarios import claude_cli_exec, extract_json

    tok = _tok()
    d = json.loads(OUT.read_text())
    items = [it for sd in d["sets"].values() for it in sd["items"]]
    RAW.mkdir(parents=True, exist_ok=True)
    calls = 0
    got: dict[str, dict] = {}
    for i in range(0, len(items), PER_CALL):
        batch = items[i:i + PER_CALL]
        f = RAW / f"para_{i // PER_CALL:03d}.txt"
        if not (f.exists() and f.read_text().strip()):
            if max_calls is not None and calls >= max_calls:
                break
            f.write_text(claude_cli_exec(SYS, _prompt(batch)))
            calls += 1
        for r in extract_json(f.read_text())["items"]:
            got[r["id"]] = r
    n_ok = n_len = n_ngram = 0
    for sd in d["sets"].values():
        for it in sd["items"]:
            r = got.get(it["id"])
            if not r:
                continue
            it["p1"], it["p2"] = r["p1"], r["p2"]
            it["p1_ids"] = tok.encode(r["p1"], add_special_tokens=False).ids
            it["p2_ids"] = tok.encode(r["p2"], add_special_tokens=False).ids
            ln = len(it["cont_ids"])
            it["length_ok"] = all(abs(len(it[k]) - ln) <= 0.2 * ln + 2
                                  for k in ("p1_ids", "p2_ids"))
            it["shared_4gram"] = _shared4(it["continuation"], r["p1"]) or _shared4(
                it["continuation"], r["p2"]) or _shared4(r["p1"], r["p2"])
            n_ok += 1
            n_len += it["length_ok"]
            n_ngram += it["shared_4gram"]
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"paraphrased {n_ok} of {len(items)}; length within 20%: {n_len}; "
          f"a shared 4-gram: {n_ngram}; calls this run {calls}")


def _shared4(a: str, b: str) -> bool:
    import re

    w = lambda s: re.findall(r"[a-z0-9']+", s.lower())  # noqa: E731
    g = lambda ws: {tuple(ws[i:i + 4]) for i in range(len(ws) - 3)}  # noqa: E731
    return bool(g(w(a)) & g(w(b)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("passages", "paraphrase"))
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    passages() if a.step == "passages" else paraphrase(a.max)
