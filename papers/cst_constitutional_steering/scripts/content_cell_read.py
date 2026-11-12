#!/usr/bin/env python3
"""Amendment 37 items 3-4: the content cell's selection and confirmation (zero GPU, saved rollouts).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/content_cell_read.py GEN

``GEN`` holds the next pod's screening reads of `data/content_cell_v1.json` on the plain condition:
``a0`` (k 0-3) and ``a0s`` (k 0-3). The directed share is the share of rollouts with an option (not
capped) taking the tenet's option.

- **Selection:** a scenario is selected when neither of A0's first two rollouts (k 0, 1) takes the
  directed option (a rollout without an option does not take it; a scenario needs both rollouts
  present); at most 4 selected per tenet, a seeded choice among those that pass. The yield and the
  selected count are reported by tenet kind and by tenet.
- **Confirmation:** on the selected set, A0s (k 0-3) minus A0's later rollouts (k 2, 3) on the
  directed share, the mean over scenarios, **bootstrap resampled by tenet** (2,000 draws, seeded):
  a draw takes tenets with replacement and all of each tenet's selected scenarios. **If the CI
  includes 0 the cell is not used.**

Writes `outputs/content_cell_read.json`.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CST = HERE.parent
CAP, CAP_SEED = 4, "20261111|cst_content_cell_cap"
B, SEED = 2000, "20261111|cst_content_cell_confirm"


def scenarios() -> dict[str, dict]:
    sc = json.loads((CST / "data/content_cell_v1.json").read_text())["scenarios"]
    return {s["id"]: s["covariates"] for s in sc}


def rollouts(gen: Path, arm: str) -> list[dict]:
    return [json.loads(x) for f in sorted((gen / arm).glob("cell_plain_*.jsonl")) for x in open(f)]


def select(a0: list[dict], cov: dict[str, dict]) -> tuple[list[str], dict]:
    """Item 3: neither of A0's k 0-1 rollouts takes the directed option; at most ``CAP`` per
    tenet."""
    first: dict[str, list[int]] = defaultdict(list)
    for r in a0:
        if r["k"] in (0, 1):
            took = not r["hit_cap"] and r["option"] == cov[r["scenario_id"]]["directed_option"]
            first[r["scenario_id"]].append(int(took))
    passing = sorted(s for s, v in first.items() if len(v) == 2 and not any(v))
    by_tenet: dict[str, list[str]] = defaultdict(list)
    for s in passing:
        by_tenet[cov[s]["tenet_id"]].append(s)
    rng = random.Random(CAP_SEED)
    chosen = sorted(x for t in sorted(by_tenet) for x in (
        by_tenet[t] if len(by_tenet[t]) <= CAP else rng.sample(by_tenet[t], CAP)))
    kinds: dict[str, dict] = defaultdict(lambda: {"screened": 0, "passing": 0, "selected": 0})
    tenets: dict[str, dict] = defaultdict(lambda: {"screened": 0, "passing": 0, "selected": 0})
    for s in first:
        for d, key in ((kinds, cov[s]["tenet_kind"]), (tenets, cov[s]["tenet_id"])):
            d[key]["screened"] += 1
            d[key]["passing"] += s in passing
            d[key]["selected"] += s in chosen
    for d in kinds.values():
        d["yield"] = d["passing"] / d["screened"] if d["screened"] else None
    return chosen, {"by_kind": dict(kinds), "by_tenet": dict(tenets), "passing": len(passing),
                    "selected": len(chosen)}


def directed_share(rows: list[dict], cov: dict, ks: tuple[int, ...]) -> dict[str, float]:
    u: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r["k"] in ks and not r["hit_cap"] and r["option"] is not None:
            u[r["scenario_id"]].append(int(r["option"] == cov[r["scenario_id"]]["directed_option"]))
    return {s: statistics.mean(v) for s, v in u.items()}


def confirm(a0: list[dict], a0s: list[dict], chosen: list[str], cov: dict) -> dict:
    """Item 4: A0s (k 0-3) minus A0's k 2-3 on the selected set, bootstrap resampled by tenet."""
    s_arm, s_ref = directed_share(a0s, cov, (0, 1, 2, 3)), directed_share(a0, cov, (2, 3))
    ids = [s for s in chosen if s in s_arm and s in s_ref]
    by_tenet: dict[str, list[str]] = defaultdict(list)
    for s in ids:
        by_tenet[cov[s]["tenet_id"]].append(s)
    tenets = sorted(by_tenet)

    def delta(ts: list[str]) -> float:
        xs = [s_arm[s] - s_ref[s] for t in ts for s in by_tenet[t]]
        return statistics.mean(xs)

    rng = random.Random(SEED)
    bs = sorted(delta([rng.choice(tenets) for _ in tenets]) for _ in range(B))
    ci = [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]
    return {"n_scenarios": len(ids), "n_tenets": len(tenets),
            "a0s": statistics.mean(s_arm[s] for s in ids),
            "a0_later": statistics.mean(s_ref[s] for s in ids), "delta": delta(tenets),
            "ci95_by_tenet": ci, "cell_used": ci[0] > 0 or ci[1] < 0,
            "bootstrap": {"draws": B, "seed": SEED, "unit": "tenet"}}


def main() -> None:
    gen = Path(sys.argv[1])
    cov = scenarios()
    a0, a0s = rollouts(gen, "a0"), rollouts(gen, "a0s")
    chosen, sel = select(a0, cov)
    res = {"rule": "Amendment 37 items 3-4", "selection": sel, "selected_ids": chosen,
           "confirmation": confirm(a0, a0s, chosen, cov)}
    (CST / "outputs/content_cell_read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"selection": {k: v for k, v in sel.items() if k != "by_tenet"},
                      "confirmation": res["confirmation"]}, indent=1))


if __name__ == "__main__":
    main()
