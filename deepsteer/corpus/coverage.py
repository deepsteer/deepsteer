"""Tenet coverage of a generated corpus (CST_PLAN §4.1's diversity target).

Every tenet in ``T(C) \\ T_conf`` must appear in at least ``k_docs`` documents and in at least one
story. Items carry the tenet ids they were generated from; this tracker counts them per layer and
reports the tenets below target, which is the artifact the plan tracks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


class CoverageTracker:
    """Count generated items per tenet and kind."""

    def __init__(self, tenet_ids: Iterable[str], excluded: Iterable[str] = (), k_docs: int = 5,
                 story_kinds: tuple[str, ...] = ("story", "steadiness_story")) -> None:
        self.excluded = set(excluded)
        self.targets = sorted(set(tenet_ids) - self.excluded)
        self.k_docs = k_docs
        self.story_kinds = story_kinds
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.leaks: list[tuple[str, str]] = []

    def add(self, item_id: str, kind: str, tenet_ids: Iterable[str]) -> None:
        for t in tenet_ids:
            if t in self.excluded:
                # a T_conf tenet seeded a training item: recorded, never silently counted
                self.leaks.append((item_id, t))
                continue
            self.counts[t][kind] += 1

    def below_target(self) -> dict[str, list[str]]:
        docs_short, no_story = [], []
        for t in self.targets:
            c = self.counts.get(t, {})
            if sum(c.values()) < self.k_docs:
                docs_short.append(t)
            if not any(c.get(k, 0) for k in self.story_kinds):
                no_story.append(t)
        return {"below_k_docs": docs_short, "no_story": no_story}

    def report(self) -> dict:
        b = self.below_target()
        n = len(self.targets)
        return {"n_targets": n, "k_docs": self.k_docs,
                "covered_k": n - len(b["below_k_docs"]),
                "covered_story": n - len(b["no_story"]),
                "below_k_docs": b["below_k_docs"], "no_story": b["no_story"],
                "conf_leaks": self.leaks}
