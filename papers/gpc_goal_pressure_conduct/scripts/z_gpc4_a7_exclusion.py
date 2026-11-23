#!/usr/bin/env python3
"""Amendment 7's training exclusion (GPC_PLAN.md §15 riders, §15.2): every word 8-gram of the 48 states' text
(each message's content, each tool call's arguments, probe B's quoted call), hashed one way (blake2b-64,
`overlap_hashes`'s rule) into their own exclusion file. Any later corpus build unions this file into the eval
index before it checks overlap. Only hashes are written; no state text enters the repo or is printed.

    python z_gpc4_a7_exclusion.py   # writes outputs/_sync/a7_states_8gram_hashes_v1.npy, data/a7_states_8gram_v1.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "papers/cst_constitutional_steering/scripts")]
import gpc4_a7_pod as A7  # noqa: E402
import overlap_hashes as OH  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/_sync/a7_states_8gram_hashes_v1.npy"
REC = GPC / "data/a7_states_8gram_v1.json"


def state_texts(st: dict) -> list[str]:
    out = [st.get("next_call") or ""]
    for m in st["messages"]:
        out.append(m.get("content") or "")
        out += [(c.get("function") or {}).get("arguments") or "" for c in m.get("tool_calls") or []]
    return [t for t in out if t]


def main() -> None:
    sts = A7.states()
    h = np.unique(np.concatenate([OH.hashes(t) for st in sts for t in state_texts(st)]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, h)
    v2 = np.load(GPC / "outputs/_sync/eval_8gram_hashes_v2.npy")
    rec = {"file": str(OUT.relative_to(GPC)), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
           "states": len(sts), "states_sha256": hashlib.sha256(A7.STATES.read_bytes()).hexdigest(),
           "n_hashes": int(len(h)), "already_in_v2": int(np.isin(h, v2).sum()),
           "hash": "blake2b-64 of the space-joined word 8-gram (overlap_hashes)",
           "rule": "GPC_PLAN.md §15.2: unioned into the eval index by any later corpus build; no training stage "
                   "reads outputs/gpc4_in/a7 or outputs/gpc4_a7"}
    REC.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
