"""GPC Z5: W-ref diagnostics on a seeded 1,000 of gpt-5.6-terra-3000 (diagnostic; never committed data).

Token counts under the subject's tokenizer (Qwen3.8-27B@1d4bf0f2, thinking-off render) and the
eval 8-gram check. Data path is passed as an argument; nothing from it is executed.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "dev/deepsteer"
sys.path[:0] = [str(REPO), str(REPO / "papers/cst_constitutional_steering/scripts")]
import overlap_hashes as OH  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

data = Path(sys.argv[1])
rows = [json.loads(x) for x in data.read_text().splitlines() if x.strip()]
idx = random.Random("20261115|gpc_wref_z5").sample(range(len(rows)), 1000)
snap = Path.home() / (".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
                      "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
tok = AutoTokenizer.from_pretrained(str(snap))
index = OH.load()
tot, asst, hits, roles, shared_any = [], [], [], set(), []
for i in idx:
    msgs = rows[i]["messages"]
    roles |= {m["role"] for m in msgs}
    full = tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
    if hasattr(full, "input_ids"):
        full, prompt = full["input_ids"], prompt["input_ids"]
    tot.append(len(full))
    asst.append(len(full) - len(prompt))
    text = "\n".join(m["content"] for m in msgs)
    n = OH.shared(text, index)
    hits.append(n)
    if n:
        shared_any.append((i, n))
q = lambda a: [int(np.percentile(a, p)) for p in (5, 50, 95)]
res = {"source_rows": len(rows), "sample": 1000, "seed": "20261115|gpc_wref_z5",
       "roles": sorted(roles), "turns": sorted({len(rows[i]["messages"]) for i in idx}),
       "tokens_total": int(sum(tot)), "tokens_assistant": int(sum(asst)),
       "per_row_total_p5_p50_p95": q(tot), "per_row_assistant_p5_p50_p95": q(asst),
       "rows_with_any_shared_8gram": len(shared_any), "shared_8grams_total": int(sum(hits)),
       "max_shared_in_a_row": int(max(hits)), "top_rows": sorted(shared_any, key=lambda x: -x[1])[:10]}
print(json.dumps(res, indent=1))
json.dump({"res": res, "idx": idx, "hits": hits},
          open(Path(sys.argv[2]), "w"))
