"""The violation on offer at a MACHIAVELLI state (GPC_PLAN.md v0.2 §3.1b, item 12): no side effects
on import, so the pod's reads can use it without the zero-GPU derivation's inputs."""

from __future__ import annotations

VIOLENCE = frozenset({"killing", "physical_harm", "intending_harm"})
NON_VIOLENT = frozenset({"deception", "manipulation", "promises", "fairness", "stealing",
                         "trespassing", "spying", "vandalism"})


def on_offer(label_row: dict) -> set[str]:
    """The categories on a state's violating options (``label_row``: one state's label record)."""
    c: set[str] = set()
    for i, v in enumerate(label_row["violating"]):
        if v:
            c |= {k for k, x in label_row["labels"][i].items() if x}
    return c


def bin_of(cats: set[str]) -> str:
    """non_violent_only, violence_only, both, or neither (non_physical_harm or other alone)."""
    v, d = bool(cats & VIOLENCE), bool(cats & NON_VIOLENT)
    return "both" if v and d else "violence_only" if v else "non_violent_only" if d else "neither"
