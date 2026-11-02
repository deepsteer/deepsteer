"""The DPO pairs' letter positions (MRB_PLAN v0.9.3 item 4; MRB-A16), zero GPU -> p2_pair_letters.json.

    mrb_p2_pairs.py [--out papers/mrb_morebench/outputs/p2_pair_letters.json]

Each adapter's training pairs are rebuilt with ST's own rules from the rows they were built from
(``t16_driver.harm_pairs``; Qwen's rule-right pairs by ``st2lb_driver.rule_right_pairs`` and ``cut_to``)
and accepted only if the per-scenario counts equal the adapter's ``st_adapter.json``. Per pair: the
shown letter of the chosen and of the rejected option (the rollout's parsed letter of record). Reported:
the letter tables, the share at A of chosen and of rejected options with exact binomial CIs, and the
paired difference with a pair-bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ST_SCRIPTS = REPO / "papers" / "st_live_reference" / "scripts"
sys.path[:0] = [str(ST_SCRIPTS), str(REPO / "papers" / "kdg_panel" / "scripts")]

ST_OUT = REPO / "papers" / "st_live_reference" / "outputs"
SOURCES = {
    "muse_glimmer_30b": {"adapter": ST_OUT / "t16/muse_glimmer_30b/adapters/care/st_adapter.json",
                         "candidates": [ST_OUT / "st2mr/muse_glimmer_30b/gen",
                                        ST_OUT / "st2mr_pull_20261023T233523/muse_glimmer_30b/gen"],
                         "care": "arm_care_03.jsonl", "none": "arm_none.jsonl", "rr": None},
    "qwen38_27b": {"adapter": ST_OUT / "st2qp_pull/qwen38_27b/adapters/care/st_adapter.json",
                   "candidates": [ST_OUT / "st2qp_pull/qwen38_27b/gen"],
                   "care": "teacher_care.jsonl", "none": "teacher_none.jsonl",
                   "rr": "rr_rollouts.jsonl"},
}
N_BOOT, SEED = 2000, 20261102


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _tail_ge(k: int, n: int, p: float) -> float:
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _solve(f, target: float) -> float:
    """p in [0, 1] with f(p) = target, f increasing in p (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def exact_ci(k: int, n: int) -> list[float]:
    """Clopper-Pearson 95% interval for k of n, from the binomial tails (no scipy)."""
    lo = 0.0 if k == 0 else _solve(lambda p: _tail_ge(k, n, p), 0.025)
    hi = 1.0 if k == n else _solve(lambda p: _tail_ge(k + 1, n, p), 0.975)
    return [round(lo, 3), round(hi, 3)]


def rule_right_with_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    """``st2lb_driver.rule_right_pairs``'s pairing, keeping the rows (it returns only ids)."""
    import st2lb_driver as LB
    by: dict[str, dict[str, list]] = {}
    for r in sorted(rows, key=lambda x: (x["scenario_id"], x["rollout"])):
        if r.get("norm_status") in ("consistent", "violating") and r.get("gen_ids"):
            by.setdefault(r["scenario_id"], {"consistent": [], "violating": []})[
                r["norm_status"]].append(r)
    out = []
    for _sid, d in sorted(by.items()):
        for i in range(min(len(d["consistent"]), len(d["violating"]), LB.PAIRS_PER_ITEM)):
            out.append((d["consistent"][i], d["violating"][i]))
    ref = LB.rule_right_pairs(rows)  # the same pairs, in the same order, as ST's own function
    assert [p["chosen"] for p in ref] == [c["gen_ids"] for c, _ in out]
    return out


def one(model: str, cfg: dict) -> dict:
    import st2lb_driver as LB
    import t16_driver as T16
    side = json.loads(cfg["adapter"].read_text())["pairs"]
    for gen in cfg["candidates"]:
        care, none = jsonl(gen / cfg["care"]), jsonl(gen / cfg["none"])
        hp, info = T16.harm_pairs(care, none)
        if info["per_scenario"] == side["per_scenario"]:
            break
    else:
        raise SystemExit(f"{model}: no candidate rebuilds the adapter's per-scenario pair counts")
    rows = {(r["scenario_id"], r["rollout"]): r for r in care}
    rej_rows = {(r["scenario_id"], r["rollout"]): r for r in none}
    pairs = [(rows[tuple(p["chosen_key"])], rej_rows[tuple(p["rejected_key"])], "rule_harm")
             for p in hp["care"]]
    rr_info = None
    if cfg["rr"]:
        rr = rule_right_with_rows(jsonl(gen / cfg["rr"]))
        kept, cut = LB.cut_to([{"i": i} for i in range(len(rr))], len(pairs))
        rr_info = cut
        assert cut == side["rule_right"], (cut, side["rule_right"])
        pairs += [(rr[k["i"]][0], rr[k["i"]][1], "rule_right") for k in kept]
    assert len(pairs) == side.get("n_total", side["n_pairs"]), (len(pairs), side)
    ch = [c["letter"] for c, _, _ in pairs]
    rj = [r["letter"] for _, r, _ in pairs]
    n_opt = Counter(len(c["order"]) for c, _, _ in pairs)
    d = np.array([(a == "A") - (b == "A") for a, b in zip(ch, rj)], dtype=float)
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    dci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
    kc, kr, n = ch.count("A"), rj.count("A"), len(pairs)
    reading = ("positional learning toward A is a candidate cause of the shift toward A"
               if dci[0] > 0 else
               f"no chosen-at-A excess detectable (an excess up to {dci[1]:+.2f} is not excluded)"
               if dci[1] >= 0 else "chosen options sit at A less often than rejected ones")
    # descriptive, beyond the registered A read: every letter's chosen - rejected share
    per_letter = {}
    for L in sorted(set(ch) | set(rj)):
        x = np.array([(a == L) - (b == L) for a, b in zip(ch, rj)], dtype=float)
        bx = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
        per_letter[L] = {"est": round(float(x.mean()), 3),
                         "ci95": [round(float(np.percentile(bx, 2.5)), 3),
                                  round(float(np.percentile(bx, 97.5)), 3)]}
    return {"source_rows": str(gen.relative_to(REPO)), "n_pairs": n,
            "kinds": dict(Counter(k for _, _, k in pairs)), "rule_right_cut": rr_info,
            "options_shown": dict(n_opt), "chosen_letters": dict(sorted(Counter(ch).items())),
            "rejected_letters": dict(sorted(Counter(rj).items())),
            "cross_tab": {f"{a}>{b}": v for (a, b), v in sorted(Counter(zip(ch, rj)).items())},
            "chosen_at_A": {"k": kc, "n": n, "share": round(kc / n, 3), "ci95": exact_ci(kc, n)},
            "rejected_at_A": {"k": kr, "n": n, "share": round(kr / n, 3), "ci95": exact_ci(kr, n)},
            "difference_chosen_minus_rejected_at_A": {"est": round(float(d.mean()), 3),
                                                      "ci95": dci},
            "chance_at_A": {k: round(1 / k, 3) for k in n_opt}, "reading": reading,
            "per_letter_chosen_minus_rejected_descriptive": per_letter}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "papers/mrb_morebench/outputs/p2_pair_letters.json")
    a = ap.parse_args()
    res = {"rule": "MRB_PLAN v0.9.3 item 4", "n_boot": N_BOOT, "seed": SEED,
           **{m: one(m, c) for m, c in SOURCES.items()}}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    for m in SOURCES:
        r = res[m]
        print(f"{m}: {r['n_pairs']} pairs {r['kinds']} from {r['source_rows']}; options {r['options_shown']}")
        print(f"   chosen {r['chosen_letters']}  rejected {r['rejected_letters']}")
        print(f"   at A: chosen {r['chosen_at_A']['share']} {r['chosen_at_A']['ci95']}, rejected "
              f"{r['rejected_at_A']['share']} {r['rejected_at_A']['ci95']}; diff "
              f"{r['difference_chosen_minus_rejected_at_A']} -> {r['reading']}")
        print(f"   per letter (descriptive): {r['per_letter_chosen_minus_rejected_descriptive']}")


if __name__ == "__main__":
    main()
