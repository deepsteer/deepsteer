#!/usr/bin/env python3
"""A0n's neutral prefix (CST Amendment 16 item 1(a)): Faraday, *The Chemical History of a Candle*.

    python3.13 papers/cst_constitutional_steering/scripts/a0n_build.py <pg14474.txt> --out <txt>

Source: Project Gutenberg eBook #14474 (public domain), plain text,
https://www.gutenberg.org/cache/epub/14474/pg14474.txt; its sha256 is recorded, so the text is
rebuilt from the source file rather than committed as a corpus. Build, in order:

1. the six lectures only (from the first "LECTURE I." after the contents to "LECTURE ON
   PLATINUM."); the Gutenberg wrapper, preface, contents, the Platinum lecture and the notes go;
2. blocks (blank-line separated) that are not paragraph prose go: figure lines, tables and
   composition diagrams (``|``, dot leaders, underscore rules), and all-caps headings; every dropped
   block is listed in the record for audit;
3. footnote markers (``[1]``), in-text figure references (``(fig. 7)``) and Gutenberg italics
   underscores are removed; hard-wrapped lines are
   joined into paragraphs, one blank line between paragraphs;
4. cut at the paragraph boundary whose Qwen3.8-27B token count is nearest ``C_anth``'s body.

Writes ``outputs/a0n_build.json`` (source sha256, counts, dropped blocks, normative-word rates,
8-gram overlap with ``C_anth``) and the text to the path given by ``--out``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CST = HERE.parent
NORMATIVE = ("should", "ought", "must")


def lectures(raw: str) -> str:
    lines = raw.splitlines()
    starts = [i for i, x in enumerate(lines) if x.strip() == "LECTURE I."]
    end = next(i for i, x in enumerate(lines) if x.strip() == "LECTURE ON PLATINUM."
               and i > starts[-1])
    return "\n".join(lines[starts[-1]:end])


def is_prose(block: str) -> bool:
    ls = [x for x in block.splitlines() if x.strip()]
    if not ls:
        return False
    if any(re.match(r"^\s*\[Illustration", x) for x in ls):
        return False
    if any("|" in x or re.search(r"\. \. \.", x) or re.match(r"^\s*_{3,}\s*$", x) for x in ls):
        return False
    text = " ".join(ls)
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return False  # headings and subtitles in capitals
    return True


def clean(block: str) -> str:
    t = " ".join(x.strip() for x in block.splitlines() if x.strip())
    t = re.sub(r"\[\d+\]", "", t)
    t = re.sub(r"\s*\((?:see )?figs?\.\s*\d+[^)]*\)", "", t, flags=re.I)  # dangling figure refs
    t = re.sub(r"(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def normative_rate(text: str) -> dict:
    w = re.findall(r"[A-Za-z']+", text)
    k = {m: sum(x.lower() == m for x in w) for m in NORMATIVE}
    return {"words": len(w), "counts": k, "per_1k_words": 1000 * sum(k.values()) / len(w)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cst_s0a_pod as POD
    from transformers import AutoTokenizer

    raw_b = Path(a.source).read_bytes()
    blocks = re.split(r"\n\s*\n", lectures(raw_b.decode("utf-8")))
    keep, dropped = [], []
    for b in blocks:
        (keep if is_prose(b) else dropped).append(b)
    paras = [c for c in (clean(b) for b in keep) if c]
    body = POD.constitution(False)
    snap = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B"
                                                 "/snapshots/*")))[0]
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    target = len(tok(body, add_special_tokens=False)["input_ids"])
    best, text = None, ""
    for i in range(1, len(paras) + 1):
        t = "\n\n".join(paras[:i])
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        if best is None or abs(n - target) < abs(best[1] - target):
            best, text = (i, n), t
        if n > target:
            break
    Path(a.out).write_text(text + "\n")
    shared = ngrams(text) & ngrams(body)
    rec = {"rule": "CST Amendment 16 item 1(a)", "source": "Project Gutenberg #14474",
           "source_sha256": hashlib.sha256(raw_b).hexdigest(), "qwen_snapshot": Path(snap).name,
           "target_tokens_C_anth": target, "tokens": best[1], "paragraphs": best[0],
           "paragraphs_available": len(paras),
           "text_sha256": hashlib.sha256((text + "\n").encode()).hexdigest(),
           "dropped_blocks": [b.strip()[:160] for b in dropped],
           "normative": {"A0n": normative_rate(text), "C_anth": normative_rate(body)},
           "eightgram_overlap": {"shared": len(shared), "A0n_8grams": len(ngrams(text)),
                                 "examples": [" ".join(g) for g in sorted(shared)[:10]]}}
    (CST / "outputs" / "a0n_build.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "dropped_blocks"}, indent=1))
    print("dropped blocks:", len(dropped))


if __name__ == "__main__":
    main()
