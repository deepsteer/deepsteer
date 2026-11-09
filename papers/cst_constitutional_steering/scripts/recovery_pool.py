#!/usr/bin/env python3
"""The recovery set's neutral prompt pool (CST-A10; Amendment 21 item 5; PILOT_BUILD.md §4).

    python3.13 papers/cst_constitutional_steering/scripts/recovery_pool.py <jsonl...>

Claude-written neutral requests (code, math, facts, writing, planning). A prompt is kept only if it
shares no word 8-gram with the seven indexed eval sets (`t05_overlap_index.eval_sets`) or with the
full GSM8K and MMLU test splits (the capability read draws from both), and is not a near duplicate
of a kept prompt (word 5-gram Jaccard at or above 0.35). Writes `data/recovery_pool_v1.json`. The
SDF pod generates `none`'s thinking-on reply to each and selects 512 by
`pilot_train.select_recovery`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CST = HERE.parent
ROOT = CST.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers/mch_machiavelli/scripts")]
import pilot_prompts as PP  # noqa: E402


def capability_texts() -> list[str]:
    from datasets import load_dataset

    g = load_dataset("openai/gsm8k", "main", split="test")
    m = load_dataset("cais/mmlu", "all", split="test")
    return [r["question"] for r in g] + [r["question"] + " " + " ".join(r["choices"]) for r in m]


def main(paths: list[Path]) -> None:
    import t05_overlap_index as T5

    from deepsteer.corpus.overlap import OverlapIndex

    idx = OverlapIndex()
    for k, v in T5.eval_sets().items():
        idx.add_set(k, v)
    idx.add_set("capability_gsm8k_mmlu_test", capability_texts())
    rows = [json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()]
    kept, drops, seen = [], {"overlap_8gram": [], "near_duplicate": [], "malformed": []}, []
    for r in sorted(rows, key=lambda r: r["id"]):
        t = (r.get("prompt") or "").strip()
        if not 8 <= len(PP._words(t)) <= 140:
            drops["malformed"].append(r["id"])
            continue
        if idx.check(r["id"], t).n_shared:
            drops["overlap_8gram"].append(r["id"])
            continue
        g5 = PP._grams(t, 5)
        if any(len(g5 & s) / max(len(g5 | s), 1) >= PP.NEAR_DUP for s in seen):
            drops["near_duplicate"].append(r["id"])
            continue
        seen.append(g5)
        kept.append({"id": r["id"], "prompt": t, "sha256": hashlib.sha256(t.encode()).hexdigest()})
    out = CST / "data/recovery_pool_v1.json"
    out.write_text(json.dumps({"rule": "CST-A10; Amendment 21 item 5",
                               "provenance": "written locally by Claude (pilot)",
                               "n_written": len(rows), "n_kept": len(kept),
                               "eval_index": idx.summary(), "drops": drops, "prompts": kept},
                              indent=1) + "\n")
    print(f"wrote {out}: kept {len(kept)} of {len(rows)}", {k: len(v) for k, v in drops.items()})


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
