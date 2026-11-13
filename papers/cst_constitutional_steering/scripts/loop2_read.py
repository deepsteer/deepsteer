#!/usr/bin/env python3
"""The second draw's registered read (CST Amendment 44 item 5), zero GPU, from the loop2 pod.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/loop2_read.py DIR

**Registered before data:** the content-cell gain replicates when config 2's second draw's
contrast against the base, its directed share at k 4 on the loop pod's selected cell minus A0's
later rollouts (the loop pod's k 2-3) with the CI bootstrapped by tenet (`content_cell_read`),
excludes 0. A CI wholly below 0 excludes 0 but is a loss, not the gain; it is reported as such.
Beside: the same contrast against this pod's session base at k 4; the second draw minus the first
(the loop pod's ``c2_rule_right``) by tenet; cell A′'s originals beyond the teacher (the loop
report's estimator) for the second draw beside the first's +0.054 [−0.015, +0.120]. Writes
``DIR/loop2_read.json``.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import content_cell_read as CR  # noqa: E402

CST = HERE.parent
LOOP1 = CST / "outputs/loop_v1"
DRAW, FIRST = "c2_draw2", "c2_rule_right"
B, SEED = 2000, "20261112|cst_loop2_read"


def _by_tenet_diff(a: dict, b: dict, ids: list[str], cov: dict) -> dict:
    """a minus b on the directed share, mean over scenarios, bootstrap resampled by tenet."""
    ids = [s for s in ids if s in a and s in b]
    bt: dict[str, list[str]] = defaultdict(list)
    for s in ids:
        bt[cov[s]["tenet_id"]].append(s)
    ts = sorted(bt)
    if not ts:
        return {"n": 0}

    def f(tt):
        return statistics.mean(a[s] - b[s] for t in tt for s in bt[t])

    rng = random.Random(SEED)
    bs = sorted(f([rng.choice(ts) for _ in ts]) for _ in range(B))
    return {"delta": f(ts), "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]], "n": len(ids),
            "n_tenets": len(ts)}


def read(out: Path, loop1: Path = LOOP1) -> dict:
    gate = json.loads((loop1 / "cell_gate.json").read_text())
    chosen, cov = gate["selected_ids"], CR.scenarios()
    gen = out / "gen"
    if not (gen / DRAW).exists():
        return {"rule": "Amendment 44 item 5", "skipped": "no second-draw rows"}
    a0 = CR.rollouts(loop1 / "gen", "a0")
    reg = CR.confirm(a0, CR.rollouts(gen, DRAW), chosen, cov)
    lo, hi = reg.get("ci95_by_tenet", [0.0, 0.0])
    res = {"rule": "Amendment 44 item 5 (registered before data)",
           "reference": "A0's later rollouts (k 2-3) on the loop pod",
           "registered": {"draw2": reg.get("a0s"), "a0_later": reg.get("a0_later"),
                          "delta": reg.get("delta"), "ci95_by_tenet": [lo, hi],
                          "n_scenarios": reg.get("n_scenarios"), "n_tenets": reg.get("n_tenets"),
                          "ci_excludes_0": lo > 0 or hi < 0,
                          "replicates": lo > 0, "loss": hi < 0},
           "first_draw": {"delta": 0.075, "ci95_by_tenet": [0.004, 0.156]}}
    share = CR.directed_share
    d2 = share(CR.rollouts(gen, DRAW), cov, (0, 1, 2, 3))
    if (gen / "base_session").exists():
        base = share(CR.rollouts(gen, "base_session"), cov, (0, 1, 2, 3))
        res["beside_vs_session_base"] = _by_tenet_diff(d2, base, chosen, cov)
    if (loop1 / "gen" / FIRST).exists():
        d1 = share(CR.rollouts(loop1 / "gen", FIRST), cov, (0, 1, 2, 3))
        res["draw2_minus_draw1"] = _by_tenet_diff(d2, d1, chosen, cov)
    return res


def write(out: Path) -> dict:
    res = read(out)
    try:
        import loop_report as LR

        res["a2_minus_teacher"] = LR.a2_diff_ci(out, DRAW)
        res["a2_first_draw"] = {"delta": 0.054, "ci95": [-0.015, 0.120]}
    except FileNotFoundError:  # the teacher's S0a rows are not on this machine
        res["a2_minus_teacher"] = None
    (out / "loop2_read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(">> loop2_read:", json.dumps(res.get("registered")))
    return res


if __name__ == "__main__":
    write(Path(sys.argv[1]))
