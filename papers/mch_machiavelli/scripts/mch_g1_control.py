"""MCH-G1 stability fork and its fp32 positive control (MCH_PLAN.md §5, Amendment 7).

- Registered criterion: S reproduces V's episode in at least 4 of 5 games (from ``mch_g1.json``).
- Redefinition (post hoc fork): replay determinism (``mch_g1_stability.json``) and matched-state R0
  probabilities within 0.03 at every decision before divergence.
- Positive control: the five episodes run twice in fp32 with micro-batches of 1, five concurrent
  (``outputs/g1_fp32_a``) and one at a time (``outputs/g1_fp32_b``); pass if all five are identical
  between the two. Beside it, fp32 against V's bf16 read at each episode's first state.

Writes ``outputs/mch_g1_control.json``.
"""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

MCH = Path(__file__).resolve().parents[1]
OUTPUTS = MCH / "outputs"
BAR = 0.03


def rows(d: Path) -> dict[tuple[str, str, int], list[dict]]:
    out = collections.defaultdict(list)
    for line in gzip.open(d / "decisions.jsonl.gz", "rt"):
        r = json.loads(line)
        out[(r["arm"], r["game"], r["episode"])].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["step"])
    return out


def main() -> None:
    g1 = json.loads((OUTPUTS / "mch_g1.json").read_text())["harness"]
    diag = json.loads((OUTPUTS / "mch_g1_stability.json").read_text())
    stab = g1["stability"]
    max_dp = max(s["max_abs_dp_until_divergence"] for s in stab.values())
    out = {
        "registered": {"identical": g1["stability_identical"], "of": len(stab), "bar": 4,
                       "pass": g1["stability_identical"] >= 4},
        "redefinition": {"replay_reproduces_every_action": diag["replay_reproduces_every_action"],
                         "max_matched_state_dp": max_dp, "bar": BAR,
                         "pass": bool(diag["replay_reproduces_every_action"] and max_dp <= BAR)},
    }
    a, b, v = rows(OUTPUTS / "g1_fp32_a"), rows(OUTPUTS / "g1_fp32_b"), rows(OUTPUTS / "g1")
    games = {}
    for key in sorted(a):
        ra, rb = a[key], b[key]
        same_actions = [x["action"] for x in ra] == [x["action"] for x in rb]
        dp = max((max(abs(p - q) for p, q in zip(x["r0"], y["r0"])) for x, y in zip(ra, rb)),
                 default=0.0)
        first_v = v[("V", key[1], key[2])][0]
        games[key[1]] = {
            "identical": bool(same_actions and len(ra) == len(rb)), "decisions": [len(ra), len(rb)],
            "max_abs_dp_a_vs_b": dp,
            "first_state_dp_fp32_vs_bf16": max(abs(p - q) for p, q in
                                               zip(ra[0]["r0"], first_v["r0"])),
            "first_prompt_same": ra[0]["prompt_sha256"] == first_v["prompt_sha256"]}
    n_id = sum(g["identical"] for g in games.values())
    out["control_fp32"] = {"games": games, "identical": n_id, "of": len(games),
                           "pass": n_id == len(games) == 5}
    (OUTPUTS / "mch_g1_control.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "games"}
                      for k, v in out.items()}))


if __name__ == "__main__":
    main()
