#!/usr/bin/env python3
"""GPC W-ref's data (GPC_PLAN.md v0.2 §3.1 Z5, items 3 and 6): the Teaching Qwen Why
difficult-advice conversations written by GPT-5.6 Terra, downloaded on the pod at a pinned revision
and checksum, never committed and never the product.

The seeded 1,000 of Z5 (``20261115|gpc_wref_z5``), minus rows sharing an 8-gram with the eval index
(D11's drop), in the seeded order; then the first rows whose cumulative assistant tokens reach the
matched arm's measured assistant-token count (item 3: W-self's, or C1's when item 11 selects C1g).
Rendered thinking-off (``pilot_train.chat_examples``: the prompt up to the assistant turn with
``enable_thinking=False``, the target the answer and the end-of-turn token).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts")]

REPO = "SecondLookResearch/teaching-qwen-why-data"
REVISION = "c57625c99e4bf69982686f6b1cf9884253986158"
FILE = "difficult_advice/gpt-5.6-terra-3000/train-3000.jsonl"
SHA256 = "ba69d7ead98c4c708347ac1734a27c6eeb2c248080328d0358a4cf85e4e4fbbc"
SEED, N = "20261115|gpc_wref_z5", 1000


def fetch(dry: bool, dry_rows: int = 24) -> list[dict]:
    """The rows; fails loud on a checksum mismatch. Dry: synthetic three-turn rows."""
    if dry:
        return [{"messages": [{"role": "system", "content": f"You help team {i}."},
                              {"role": "user", "content": f"Should I cut corner {i}? " * 5},
                              {"role": "assistant", "content": f"Here is advice {i}. " * 8}]}
                for i in range(dry_rows)]
    from huggingface_hub import hf_hub_download

    p = Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION))
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != SHA256:
        raise SystemExit(f"BAIL: W-ref's file sha256 {got} != pinned {SHA256}")
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def seeded(rows: list[dict], index) -> tuple[list[dict], int]:
    """The seeded 1,000 in order, without rows sharing an eval 8-gram; the dropped count."""
    import numpy as np

    idx = random.Random(SEED).sample(range(len(rows)), min(N, len(rows)))
    if index is None:
        return [rows[i] for i in idx], 0
    import overlap_hashes as OH

    out, dropped = [], 0
    for i in idx:
        text = "\n".join(m["content"] for m in rows[i]["messages"])
        if np.isin(OH.hashes(text), index).any():
            dropped += 1
            continue
        out.append(rows[i])
    return out, dropped


def examples(tok, rows: list[dict], target_assistant_tokens: int) -> tuple[list, dict]:
    """(ids, mask) training examples up to the matched assistant-token count, and a record."""
    import pilot_train as PT

    ex, n_asst = [], 0
    for r in rows:
        if n_asst >= target_assistant_tokens:
            break
        e = PT.chat_examples(tok, r["messages"])
        ex += e
        n_asst += sum(sum(m) for _, m in e)
    return ex, {"rows": len(ex), "assistant_tokens": n_asst, "target": target_assistant_tokens,
                "tokens": sum(len(i) for i, _ in ex), "short_of_target":
                n_asst < target_assistant_tokens}
