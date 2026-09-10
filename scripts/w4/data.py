"""Stimulus loaders for the W4 units, with tiny synthetic fallbacks for ``--dry-run``.

Real runs read the committed / local-only stimulus files the runs of record used (Heretic prompt
set, D1 moral pairs + fables, ETHICS items, persona and control pairs, severity/boundary/request
twins, MFT v2). Dry runs never touch those files: every loader returns a small synthetic set of the
same shape so the analysis paths run anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import REPO

P5 = REPO / "papers" / "5_moral_alignment"
D1_OUT = REPO / "papers" / "d1_moral_subspace" / "outputs"
D1_DATASET = REPO / "deepsteer" / "datasets" / "d1_vmoral_v1.json"
FABLES = D1_OUT / "full" / "fables_train_full.json"
ETHICS = D1_OUT / "full" / "ethics_train_full.json"


def _fake_pairs(n: int, tag: str) -> list[tuple[str, str]]:
    return [(f"{tag} positive text number {i} about a situation.",
             f"{tag} negative text number {i} about a situation.") for i in range(n)]


def heretic_prompts(dry: bool, n: int | None = None) -> dict:
    """Heretic 400/400 train + 100/100 eval prompts (Paper 5 ``refusal_prompts.json``)."""
    if dry:
        k = n or 12
        return {"harmful": [f"harmful request {i}" for i in range(k)],
                "harmless": [f"harmless request {i}" for i in range(k)],
                "harmful_eval": [f"harmful eval {i}" for i in range(max(4, k // 2))],
                "harmless_eval": [f"harmless eval {i}" for i in range(max(4, k // 2))]}
    d = json.loads((P5 / "refusal_prompts.json").read_text())
    if n:
        d = {k: (v[:n] if isinstance(v, list) else v) for k, v in d.items()}
    return d


def moral_pairs(dry: bool, n_cap: int | None = None) -> dict[str, list[tuple[str, str]]]:
    """{moral_stories, fables, ethics} -> [(moral, neutral)], the D1/D2 V_moral sources."""
    if dry:
        k = n_cap or 6
        return {s: _fake_pairs(k, s) for s in ("moral_stories", "fables", "ethics")}
    from informat_ladder import load_moral_pairs
    return load_moral_pairs(str(D1_DATASET), str(FABLES), n_cap)


def control_pairs(dry: bool, n_cap: int | None = None) -> dict[str, list[tuple[str, str]]]:
    if dry:
        k = n_cap or 6
        return {c: _fake_pairs(k, c) for c in ("syntax", "register", "sentiment")}
    from deepsteer.datasets import get_register_pairs, get_sentiment_pairs, get_syntax_pairs
    out = {"syntax": get_syntax_pairs(), "register": get_register_pairs(), "sentiment": get_sentiment_pairs()}
    return {k: (v[:n_cap] if n_cap else v) for k, v in out.items()}


def persona_pairs(dry: bool, n_cap: int | None = None) -> list[tuple[str, str]]:
    if dry:
        return _fake_pairs(n_cap or 6, "persona")
    from deepsteer.datasets import get_persona_pairs
    ps = get_persona_pairs()
    ps = [(p.moral if hasattr(p, "moral") else p[0], p.neutral if hasattr(p, "neutral") else p[1]) for p in ps]
    return ps[:n_cap] if n_cap else ps


def mft_pairs(dry: bool, n_cap: int | None = None) -> dict[str, list[tuple[str, str]]]:
    """6-foundation MFT v2 pairs (for the Think/GPT-OSS ``mft_directions.npz`` saves)."""
    if dry:
        return {f: _fake_pairs(n_cap or 4, f) for f in
                ("care", "fairness", "loyalty", "authority", "sanctity", "liberty")}
    from collections import defaultdict
    from deepsteer.datasets import load_moral_probing_v2
    by_f: dict[str, list] = defaultdict(list)
    for p in load_moral_probing_v2().all_pairs:
        by_f[p.foundation.value].append((p.moral, p.neutral))
    return {f: (v[:n_cap] if n_cap else v) for f, v in by_f.items()}


def severity_twins(dry: bool) -> list[tuple[str, int, str, str]]:
    if dry:
        return [("care_harm", 1 + i % 5, f"Help me do thing {i} so it goes well.",
                 f"Help me do thing {i} so it hurts someone.") for i in range(6)]
    from deepsteer.datasets import get_severity_twins
    return list(get_severity_twins())


def boundary_twins(dry: bool) -> list[tuple[str, str, str, str]]:
    if dry:
        return [("care_harm", "abc"[i % 3], f"Help me handle thing {i} so it is fair.",
                 f"Help me handle thing {i} so it is cruel.") for i in range(6)]
    from deepsteer.datasets import get_boundary_twins
    return list(get_boundary_twins())


def ethics_items(dry: bool, n_cap: int | None = None) -> list[tuple[str, str]]:
    """[(scenario, gt in {wrong, not_wrong})] for the B1 judgment battery."""
    if dry:
        return [(f"Someone did action {i}.", "wrong" if i % 2 else "not_wrong") for i in range(n_cap or 6)]
    from b1_judgment_direction import load_items
    return load_items([ETHICS], n_cap)


def request_twin_sets() -> dict[str, str]:
    """following-text -> set tag for the union stimulus set (Amendment 15.1 alone/pooled split)."""
    from deepsteer.datasets import get_request_twins_union, w4_set_tags
    return {a: t for (_f, a, _b), t in zip(get_request_twins_union(), w4_set_tags())}


def local_npz(path: Path):
    """Load a local-only .npz if present (returns None otherwise; callers decide the fallback)."""
    import numpy as np
    return np.load(path, allow_pickle=True) if Path(path).exists() else None
