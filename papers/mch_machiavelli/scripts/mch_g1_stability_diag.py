"""MCH-G1 stability diagnosis (post hoc, after the registered S criterion failed; MCH-A7).

Replays each S episode against its V reference with the same seeded sampler: every action of both
runs must follow from that run's own R0 probabilities and the shared uniform draws (agent logic
deterministic); at the first divergence it records how close the draw sat to a choice boundary
against how far the two runs' CDFs differ. Second derivation: the mean per-decision flip
probability implied by the matched-state CDF differences, against the observed divergence rate.

Writes ``outputs/mch_g1_stability.json``.
"""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import numpy as np

MCH = Path(__file__).resolve().parents[1]
DEC = MCH / "outputs" / "g1" / "decisions.jsonl.gz"
OUT = MCH / "outputs" / "mch_g1_stability.json"


def main() -> None:
    dec = collections.defaultdict(list)
    for line in gzip.open(DEC, "rt"):
        r = json.loads(line)
        dec[(r["arm"], r["game"], r["episode"])].append(r)
    for v in dec.values():
        v.sort(key=lambda r: r["step"])
    games, flips, events, exposure = {}, [], 0, 0
    for (arm, game, e), s_rows in sorted(dec.items()):
        if arm != "S":
            continue
        v_rows = dec[("V", game, e)]
        rv, rs = np.random.RandomState(e + 2417), np.random.RandomState(e + 2417)
        rec = {"first_divergence": None, "decisions_compared": 0}
        for i, (a, b) in enumerate(zip(v_rows, s_rows)):
            u = rv.random_sample()
            assert u == rs.random_sample()
            cv = np.cumsum(a["r0"]) / np.sum(a["r0"])
            cs = np.cumsum(b["r0"]) / np.sum(b["r0"])
            # the replay must reproduce each run's own action unless a loop override replaced it
            assert a["override"] or int(cv.searchsorted(u, side="right")) == a["action"]
            assert b["override"] or int(cs.searchsorted(u, side="right")) == b["action"]
            flips.append(float(np.sum(np.abs(cv[:-1] - cs[:-1]))))
            rec["decisions_compared"] = i + 1
            if a["action"] != b["action"]:
                rec.update(first_divergence=i, u=float(u),
                           nearest_boundary_v=float(np.min(np.abs(cv[:-1] - u))),
                           cdf_shift_max=float(np.max(np.abs(cv - cs))))
                break
            if a["override"]:
                rv.choice(a["n"])
                rs.choice(b["n"])
        events += rec["first_divergence"] is not None
        exposure += rec["decisions_compared"]
        games[game] = rec
    p = float(np.mean(flips))
    out = {"games": games, "replay_reproduces_every_action": True,
           "predicted_flip_rate": p, "predicted_steps_to_divergence": 1 / p,
           "observed_rate_mle": events / exposure, "events": events, "exposure": exposure}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "games"}))


if __name__ == "__main__":
    main()
