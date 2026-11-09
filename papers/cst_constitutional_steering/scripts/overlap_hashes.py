#!/usr/bin/env python3
"""The eval-set 8-gram index as a hash array, for checks on a pod (PILOT_BUILD.md §5 card 1 gate).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/overlap_hashes.py build

`build` hashes every word 8-gram (`deepsteer.corpus.overlap`'s tokenization) of the seven indexed
eval sets (`t05_overlap_index.eval_sets`, which needs the local MACHIAVELLI cache) and of the GSM8K
and MMLU test splits into sorted unique uint64 values (the first 8 bytes of each 8-gram's blake2b
digest) at `outputs/_sync/eval_8gram_hashes_v1.npy`, with its sha256 and counts beside. A pod syncs
the file and calls `shared(text)`: the number of the text's 8-grams found in the index. A checked
8-gram falsely matches with probability about n / 2^64 for an index of n hashes (about 3e-13 at the
index's six million).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CST = HERE.parent
OUT = CST / "outputs/_sync/eval_8gram_hashes_v1.npy"


def _hash(g: tuple[str, ...]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(g).encode(), digest_size=8).digest(), "little")


def hashes(text: str) -> np.ndarray:
    from deepsteer.corpus.overlap import ngrams, words

    return np.fromiter((_hash(g) for g in ngrams(words(text), 8)), dtype=np.uint64)


def load(path: Path = OUT) -> np.ndarray:
    return np.load(path)


def shared(text: str, index: np.ndarray) -> int:
    """How many of ``text``'s distinct 8-grams are in ``index`` (sorted uint64)."""
    h = np.unique(hashes(text))
    if not len(h):
        return 0
    pos = np.searchsorted(index, h)
    pos[pos == len(index)] = 0
    return int((index[pos] == h).sum())


def build() -> None:
    sys.path[:0] = [str(CST.parents[1]), str(HERE),
                    str(CST.parents[1] / "papers/mch_machiavelli/scripts")]
    import recovery_pool as RP
    import t05_overlap_index as T5

    sets = T5.eval_sets() | {"capability_gsm8k_mmlu_test": RP.capability_texts()}
    parts, counts = [], {}
    for name, texts in sets.items():
        h = np.unique(np.concatenate([hashes(t) for t in texts if t] or [np.zeros(0, np.uint64)]))
        counts[name] = int(len(h))
        parts.append(h)
    idx = np.unique(np.concatenate(parts))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, idx)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    rec = {"file": str(OUT.relative_to(CST)), "sha256": sha, "n_hashes": int(len(idx)),
           "per_set": counts, "hash": "blake2b-64 of the space-joined word 8-gram"}
    (CST / "outputs/eval_8gram_hashes_v1.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    build()
