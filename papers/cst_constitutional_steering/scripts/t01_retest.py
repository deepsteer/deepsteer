#!/usr/bin/env python3
"""CST T0.1: retest of the T_core vote within one family (zero GPU, account login only).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_retest.py

Re-asks Claude (``claude -p``, fresh calls) on a seeded sample of 120 voted pairs, half drawn from
Claude's yes votes and half from its no votes, and reports retest agreement and kappa against the
first votes (CST_PLAN §6.5's test-retest, applied to the matching task).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent),
                str(ROOT / "papers" / "kdg_panel" / "scripts")]

import t01_review as R  # noqa: E402

from deepsteer.judges import retest_agreement  # noqa: E402

SEED, N = 20261104, 120


def main() -> None:
    v = json.loads((R.CST / "outputs" / "t01_tcore_votes.json").read_text())["votes"]
    yes = sorted(p for p, x in v.items() if x.get("claude") is True)
    no = sorted(p for p, x in v.items() if x.get("claude") is False)
    rng = random.Random(SEED)
    sample = sorted(rng.sample(yes, N // 2) + rng.sample(no, N // 2))
    sets = R._sets()
    by = {**sets["C_anth"].by_id(), **sets["C_spec"].by_id()}
    pairs = [tuple(p.split("|")) for p in sample]
    rng.shuffle(pairs)
    batches = [pairs[i:i + R.BATCH] for i in range(0, len(pairs), R.BATCH)]
    second: dict[str, bool] = {}
    for i, b in enumerate(batches):
        text = R._cached(f"retest_claude_{i:02d}.txt",
                         lambda b=b: R.claude_cli_exec(R.SYS, R._vote_prompt(b, by)))
        for x in R.extract_json(text)["votes"]:
            second[x["pair"]] = bool(x["same"])
    common = [p for p in sample if p in second]
    r = retest_agreement([str(v[p]["claude"]) for p in common], [str(second[p]) for p in common])
    res = {"seed": SEED, "n": len(common), "design": "60 first-vote yes + 60 first-vote no",
           **r, "second": second}
    (R.CST / "outputs" / "t01_retest.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: res[k] for k in ("n", "agreement", "kappa", "passes")})


if __name__ == "__main__":
    main()
