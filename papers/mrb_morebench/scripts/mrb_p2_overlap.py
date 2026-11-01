"""MRB Phase 2 overlap check (MRB_PLAN v0.9, zero GPU): ST's adapter training scenarios against the
500 MoReBench public dilemmas.

    mrb_p2_overlap.py [--out papers/mrb_morebench/outputs/p2_overlap.json]

The registered rule: a dilemma overlaps if its normalized text equals a scenario text, or if 10% or
more of its word 8-grams occur in the H1 texts. The H1 texts are every string field of every H1
scenario with at least one training pair in either adapter (``st_adapter.json``, ``per_scenario``);
the scenarios listed with zero pairs are checked beside, as a wider net. Writes ids and shares only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deepsteer.datasets.morebench import load_morebench  # noqa: E402
from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

ST_OUT = REPO / "papers" / "st_live_reference" / "outputs"
ADAPTERS = {"muse_glimmer_30b": ST_OUT / "t16/muse_glimmer_30b/adapters/care/st_adapter.json",
            "qwen38_27b": ST_OUT / "st2qp_pull/qwen38_27b/adapters/care/st_adapter.json"}
H1_SET = REPO / "papers" / "kdg_panel" / "data" / "harm_divergence" / "hdiv_set.json"
N, BAR = 8, 0.10
_WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def normalize(text: str) -> str:
    return " ".join(words(text))


def ngrams(ws: list[str], n: int = N) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def strings(obj) -> list[str]:
    """Every string inside a scenario's dict form (texts, paraphrases, options)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in strings(v)]
    return []


def check(dilemmas: dict[str, str], texts: list[str]) -> dict:
    exact = {normalize(t) for t in texts if len(words(t)) >= N}
    grams = set().union(*(ngrams(words(t)) for t in texts)) if texts else set()
    rows = {}
    for sid, d in dilemmas.items():
        g = ngrams(words(d))
        share = len(g & grams) / len(g) if g else 0.0
        rows[sid] = {"share": round(share, 4), "exact": normalize(d) in exact}
    over = sorted(s for s, r in rows.items() if r["exact"] or r["share"] >= BAR)
    shares = sorted((r["share"] for r in rows.values()), reverse=True)
    return {"n_texts": len(texts), "n_dilemmas": len(rows), "overlapping": over,
            "n_overlapping": len(over), "n_exact": sum(r["exact"] for r in rows.values()),
            "n_any_shared_8gram": sum(r["share"] > 0 for r in rows.values()),
            "max_share": shares[0] if shares else 0.0, "top_shares": shares[:5],
            "top_ids": sorted(rows, key=lambda s: -rows[s]["share"])[:5]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "papers/mrb_morebench/outputs/p2_overlap.json")
    a = ap.parse_args()
    dilemmas = {it.scenario_id: it.dilemma for it in load_morebench(subset="public")}
    scen, _ = load_scenario_dir([str(H1_SET)])
    by_id = {s.id: s for s in scen}
    trained, listed = set(), set()
    for path in ADAPTERS.values():
        per = json.loads(path.read_text())["pairs"]["per_scenario"]
        trained |= {k for k, v in per.items() if v > 0}
        listed |= set(per)
    missing = sorted((trained | listed) - set(by_id))
    if missing:
        raise SystemExit(f"H1 ids not in {H1_SET.name}: {missing[:5]}")
    texts = lambda ids: [t for i in sorted(ids) for t in strings(by_id[i].to_dict())]  # noqa: E731
    res = {"rule": f"exact normalized match, or >= {BAR:.0%} of a dilemma's word {N}-grams in the "
                   "H1 texts (MRB_PLAN v0.9)",
           "h1_set": str(H1_SET.relative_to(REPO)), "adapters": {k: str(v.relative_to(REPO))
                                                                for k, v in ADAPTERS.items()},
           "trained_scenarios": len(trained), "listed_scenarios": len(listed),
           "of_record": check(dilemmas, texts(trained)),
           "wider_net_all_listed": check(dilemmas, texts(listed)),
           "not_checked": "the adapters' general KL slice (2,000 C4 validation documents, indices "
                          "only): web text the pairs do not contain"}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    r = res["of_record"]
    print(f"trained H1 scenarios {len(trained)} ({r['n_texts']} texts); overlapping dilemmas "
          f"{r['n_overlapping']} (exact {r['n_exact']}); any shared 8-gram {r['n_any_shared_8gram']}"
          f"; max share {r['max_share']}")
    w = res["wider_net_all_listed"]
    print(f"wider net, {len(listed)} listed: overlapping {w['n_overlapping']}, any shared "
          f"{w['n_any_shared_8gram']}, max share {w['max_share']}")


if __name__ == "__main__":
    main()
