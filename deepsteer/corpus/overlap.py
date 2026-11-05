"""Word n-gram overlap between generated training text and evaluation sets.

CST_PLAN §2.3: no training item may share a word 8-gram with an evaluation item; the check runs
before every training stage. The tokenization is MRB's (``mrb_p2_overlap.py``: lowercase,
``[a-z0-9']+`` words), so the two lines' overlap numbers are comparable. MRB's registered rule was a
share bar (>= 10% of an item's 8-grams); CST's is any shared 8-gram. Both are reported.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable

N_DEFAULT = 8
_WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    """Lowercased word tokens (MRB's rule)."""
    return _WORD.findall(text.lower().replace("’", "'"))


def ngrams(ws: list[str], n: int = N_DEFAULT) -> set[tuple[str, ...]]:
    """The set of word n-grams of a token list."""
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


@dataclasses.dataclass
class OverlapHit:
    """One training text's overlap with the evaluation index."""

    item_id: str
    n_grams: int
    n_shared: int
    sets_hit: list[str]
    example: tuple[str, ...] | None

    @property
    def share(self) -> float:
        return self.n_shared / self.n_grams if self.n_grams else 0.0


class OverlapIndex:
    """An index of evaluation-set n-grams, keyed by the set each came from.

    ``allow`` holds n-grams exempt from the check (for example, n-grams of the constitution text
    itself, which both a corpus document and an evaluation item may quote); every exemption is
    reported by :meth:`summary` so it cannot hide a leak.
    """

    def __init__(self, n: int = N_DEFAULT) -> None:
        self.n = n
        self._owner: dict[tuple[str, ...], set[str]] = {}
        self.allow: set[tuple[str, ...]] = set()
        self.set_sizes: dict[str, int] = {}

    def add_set(self, set_name: str, texts: Iterable[str]) -> None:
        k = 0
        for t in texts:
            k += 1
            for g in ngrams(words(t), self.n):
                self._owner.setdefault(g, set()).add(set_name)
        self.set_sizes[set_name] = self.set_sizes.get(set_name, 0) + k

    def exempt(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.allow |= ngrams(words(t), self.n)

    def check(self, item_id: str, text: str) -> OverlapHit:
        g = ngrams(words(text), self.n)
        shared = [x for x in g if x in self._owner and x not in self.allow]
        sets = sorted({s for x in shared for s in self._owner[x]})
        return OverlapHit(item_id, len(g), len(shared), sets, shared[0] if shared else None)

    def screen(self, items: dict[str, str], share_bar: float = 0.10) -> dict:
        """Check every item; return kept and dropped ids under the CST rule (any shared n-gram),
        with MRB's share rule beside."""
        hits = {i: self.check(i, t) for i, t in items.items()}
        dropped = sorted(i for i, h in hits.items() if h.n_shared > 0)
        return {
            "rule": f"drop on any shared word {self.n}-gram (CST_PLAN §2.3)",
            "n_items": len(items),
            "dropped": dropped,
            "n_dropped": len(dropped),
            "drop_rate": len(dropped) / len(items) if items else 0.0,
            "n_over_share_bar": sum(h.share >= share_bar for h in hits.values()),
            "share_bar": share_bar,
            "by_set": {s: sum(s in h.sets_hit for h in hits.values()) for s in self.set_sizes},
            "examples": {i: " ".join(hits[i].example or ()) for i in dropped[:10]},
        }

    def summary(self) -> dict:
        return {"n": self.n, "sets": dict(self.set_sizes), "n_grams": len(self._owner),
                "n_exempt": len(self.allow)}
