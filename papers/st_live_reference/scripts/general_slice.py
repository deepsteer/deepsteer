#!/usr/bin/env python3
"""The KL term's general text slice (ST_PLAN §3: 2,000 sequences of public text, ids committed,
the same slice for every arm). C4 English validation (`allenai/c4`, ODC-BY; part of Dolma's mix),
streamed in its fixed order: the first 2,000 documents with at least ``MIN_TOKENS`` tokens under
the development model's tokenizer. Only stream indices and text sha256 are committed (no bundled
text); the pod re-streams and verifies every hash, and each sequence is cut to ``MAX_TOKENS``.

    python3 general_slice.py build      # local, once: writes data/general_slice_v1.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "general_slice_v1.json"
DATASET, CONFIG, SPLIT = "allenai/c4", "en", "validation"
N, MIN_TOKENS, MAX_TOKENS = 2000, 128, 512


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def build(tok) -> dict:
    from datasets import load_dataset

    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
    keep = []
    for i, r in enumerate(ds):
        n = len(tok.encode(r["text"], add_special_tokens=False))
        if n >= MIN_TOKENS:
            keep.append({"index": i, "sha256": _sha(r["text"]), "n_tokens": n})
        if len(keep) == N:
            break
    return {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "n": N,
            "min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS, "tokenizer": tok.name_or_path,
            "items": keep}


def load(tok, spec: dict | None = None, n: int | None = None) -> list[list[int]]:
    """Token ids of the committed slice (first ``n`` items), each verified against its sha256."""
    from datasets import load_dataset

    spec = spec or json.loads(OUT.read_text())
    items = spec["items"][: n or len(spec["items"])]
    want = {it["index"]: it for it in items}
    last = max(want)
    out: dict[int, list[int]] = {}
    for i, r in enumerate(load_dataset(spec["dataset"], spec["config"], split=spec["split"],
                                       streaming=True)):
        if i in want:
            if _sha(r["text"]) != want[i]["sha256"]:
                raise SystemExit(f"general slice: stream index {i} changed (sha256 mismatch)")
            out[i] = tok.encode(r["text"], add_special_tokens=False)[: spec["max_tokens"]]
        if i >= last:
            break
    missing = sorted(set(want) - set(out))
    if missing:
        raise SystemExit(f"general slice: {len(missing)} indices never streamed")
    return [out[it["index"]] for it in items]


def main() -> None:
    if sys.argv[1:] != ["build"]:
        raise SystemExit(__doc__)
    sys.path.insert(0, str(HERE))
    import pod_st_screen as P
    from transformers import AutoTokenizer

    spec = P.registry()["gpt_oss_20b"]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    s = build(tok)
    OUT.write_text(json.dumps(s, indent=1) + "\n")
    print(f">> general slice: {len(s['items'])} documents, last stream index "
          f"{s['items'][-1]['index']}")


if __name__ == "__main__":
    main()
