#!/usr/bin/env python3
"""The kind-split pod's reads (CST Amendment 32 items 6-8), zero GPU, saved rollouts.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/kinds_read.py

On the operator cell's 458 clear items (`outputs/pilot_sdf4/gen/<tag>`), each adapter against the
pod's same-session base, scenario-paired with a scenario bootstrap (`sdf_reread_quick.paired`),
with the three shares (Amendment 33 item 2) and the suspicion split beside (`sdf2_quick`).
**Item 8's rule, per corpus:** an adapter *moves* when its paired difference's 95% CI excludes 0;
only the stories-only adapter moves: the steadfast-agent genre; both: document tuning in
general; neither: uninformative at half dose (rerun at matched tokens); only the without-stories
adapter: the non-narrative kinds.
Writes `outputs/kinds_read.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402
import sdf2_quick as Z  # noqa: E402
import sdf_reread_quick as Q  # noqa: E402

CST = HERE.parent
GEN = CST / "outputs/pilot_sdf4/gen"
PAIRS = {"d_neutral": ("dn_stories", "dn_nostories"), "d_doc": ("dd_stories", "dd_nostories")}


def verdict(stories: bool, nostories: bool) -> str:
    return {(True, False): "the steadfast-agent genre (only stories-only moves)",
            (True, True): "document tuning in general (both move)",
            (False, False): "uninformative at half dose: rerun the split at matched tokens",
            (False, True): "the non-narrative kinds (only without-stories moves)"}[
        (stories, nostories)]


def main() -> None:
    lab = A.labels()
    tags = ["base_session"] + [t for p in PAIRS.values() for t in p if (GEN / t).exists()]
    op = {t: Z.cell_rows(t, "operator", lab, GEN) for t in tags}
    op = {t: v for t, v in op.items() if v}
    acted = lambda r: not r["capped"] and r["option"] is not None  # noqa: E731
    u = {t: Q.per_scenario(rs, acted) for t, rs in op.items()}
    ids = sorted(set.intersection(*[set(v) for v in u.values()]))
    res: dict = {"rule": "Amendment 32 items 7-8", "n_items": len(ids), "rungs": {},
                 "base_plain_shares": Z.option_shares(Z.cell_rows("base_session", "plain", lab,
                                                                  GEN)), "verdicts": {}}
    for t, rs in op.items():
        rec = Q.summary(rs) | {"operator_shares": Z.option_shares(rs)}
        if t != "base_session":
            rec["vs_session_base"] = Q.paired(u[t], u["base_session"], ids)
            lo, hi = rec["vs_session_base"]["ci95"]
            rec["moves"] = lo > 0 or hi < 0
        res["rungs"][t] = rec
    for corpus, (st, ns) in PAIRS.items():
        if st in res["rungs"] and ns in res["rungs"]:
            res["verdicts"][corpus] = verdict(res["rungs"][st]["moves"], res["rungs"][ns]["moves"])
    for t in ("train_" + x for p in PAIRS.values() for x in p):
        f = CST / "outputs/pilot_sdf4" / f"{t}.json"
        if f.exists():
            r = json.loads(f.read_text())
            res.setdefault("tokens", {})[r["rung"]] = {
                "kept_text_tokens": r["subset"]["kept_text_tokens"],
                "trained_tokens": r["trained_tokens"], "documents": r["subset"]["documents"]}
    (CST / "outputs/kinds_read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rungs"}, indent=1))
    for t, r in res["rungs"].items():
        vb, s = r.get("vs_session_base"), r["operator_shares"]
        print(f"{t:13s} ord {r['ordered_share']:.3f} susp {r['suspicion_share']:.3f} "
              f"ordered {s['ordered']:.3f} directed {s['directed']:.3f} other {s['other']:.3f} "
              f"none {s['none']:.3f}"
              + (f" | vs base {vb['delta']:+.3f} [{vb['ci95'][0]:+.3f}, {vb['ci95'][1]:+.3f}]"
                 if vb else ""))


if __name__ == "__main__":
    main()
