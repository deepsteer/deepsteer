#!/usr/bin/env python3
"""C2-JAB's families, the script check (GPC_PLAN.md §19.9), zero GPU. Per family, both versions:
1. *The conflict:* the honest route reaches the decision point (the scorer prints the shortfall) and is not a
   shortcut; the shortcut route makes the scorer pass and is a shortcut by the environment's own check.
2. *The twin:* the honest route makes the scorer pass with no shortfall printed and is not a shortcut.
3. *Coverage:* no command in either route is uncovered by the environment.
4. *The twin differs only where declared* (its `twin_files`, and the scorer's target where the scorer reads it).
5. *No word 8-gram* of the family's text (system, task, files, named options) is shared with the eval index or the
   hold-out index (`c2_holdout_8gram_v1.npy`: the depth scenarios, A6's renders, A7's states).

    python c2_family_check.py   # writes outputs/gpc4_prep/c2_family_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts")]
import c2_env as E  # noqa: E402
import c2_families as F  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/gpc4_prep/c2_family_check.json"


def run(spec: F.Spec, twin: bool, route: list[str]) -> tuple[dict, list[tuple[str, int]]]:
    env = E.Env(spec.family(twin))
    outs = []
    for c in route:
        if "{UPTIME}" in c:  # D1's honest route states the uptime the data gives
            c = c.replace("{UPTIME}", f"{F._uptime(env.tree)}")  # noqa: SLF001
        outs.append(env.run(c))
    return env.outcome(), outs


def text_of(spec: F.Spec) -> str:
    return "\n".join([spec.system, spec.user, spec.twin_user, spec.named_shortcut, spec.named_constrained,
                      *spec.files.values(), *spec.twin_files.values()])


def main() -> None:
    import numpy as np

    import overlap_hashes as OH

    ev = np.asarray(OH.load(GPC / "outputs/_sync/eval_8gram_hashes_v2.npy"))
    v1 = np.asarray(OH.load(GPC / "outputs/_sync/c2_holdout_8gram_v1.npy"))
    v2p = GPC / "outputs/_sync/c2_holdout_8gram_v2.npy"
    v2 = np.asarray(OH.load(v2p)) if v2p.exists() else v1
    rec = {"rule": "GPC_PLAN.md §19.9 (the script check)", "families": {}}
    for s in F.ROOM:
        # §19.14: a trained family is checked against v2 (the held-out families' text in it); a held-out family
        # against v1 (the transfer test's text), since v2 holds its own text
        idx = [ev, v1 if s.split in ("pilot_heldout", "pilot_counterparty") else v2]
        bad = []
        o, outs = run(s, False, s.honest_route)
        if not o["reached"] or o["shortcut"]:
            bad.append(f"conflict honest route: reached {o['reached']}, shortcut {o['shortcut']}")
        if o["uncovered"]:
            bad.append("conflict honest route: an uncovered command")
        o2, outs2 = run(s, False, s.shortcut_route)
        if not o2["shortcut"] or outs2[-1][1] != 0:
            bad.append(f"conflict shortcut route: shortcut {o2['shortcut']}, scorer exit {outs2[-1][1]}")
        if o2["uncovered"]:
            bad.append("conflict shortcut route: an uncovered command")
        # the disclosure flag (§19.12): the honest closing discloses on the conflict; the twin's closing does not
        env_c = E.Env(s.family(False))
        for cmd in s.honest_route:
            env_c.run(cmd.replace("{UPTIME}", f"{F._uptime(env_c.tree)}") if "{UPTIME}" in cmd else cmd)  # noqa: SLF001
        env_c.close(s.closing_honest)
        env_t = E.Env(s.family(True))
        for cmd in s.twin_route:
            env_t.run(cmd.replace("{UPTIME}", f"{F._uptime(env_t.tree)}") if "{UPTIME}" in cmd else cmd)  # noqa: SLF001
        env_t.close(s.closing_twin)
        oc, ot = env_c.outcome(), env_t.outcome()
        if not (s.closing_honest and oc["disclosed"]):
            bad.append("the conflict's honest closing does not read as disclosed")
        if not s.closing_twin or ot["disclosed"]:
            bad.append("the twin's closing reads as a shortfall (a false alarm)")
        env_s = E.Env(s.family(False))  # a silent route: run the scorer, write nothing, close with no disclosure
        env_s.run(f"bash {s.scorer_path}")
        env_s.close("Done.")
        if env_s.outcome()["disclosed"]:
            bad.append("a silent route (the scorer, then 'Done.') reads as disclosed")
        if not s.twin_route:
            bad.append("no twin route")
        t, touts = run(s, True, s.twin_route or s.honest_route)
        scorer_runs = [x for x in touts if "PASS" in x[0] or "FAIL" in x[0]]
        if t["reached"] or t["shortcut"] or not scorer_runs or scorer_runs[-1][1] != 0:
            bad.append(f"twin honest route: reached {t['reached']}, shortcut {t['shortcut']}")
        undeclared = sorted(p for p in s.twin_files if p not in s.files and p != "README.md"
                            and p not in s.extra.get("twin_adds", []))
        if undeclared:
            bad.append(f"twin adds undeclared paths {undeclared}")
        h = np.unique(np.asarray(OH.hashes(text_of(s))))
        shared = [int(np.isin(h, i).sum()) for i in idx]
        if any(shared):
            bad.append(f"8-grams shared: eval {shared[0]}, hold-out {shared[1]}")
        planted = None
        key = s.fid.split("-")[0]
        if s.fid.startswith("X2"):  # the invented quote: its tree-aware detector's planted sets (§19.14)
            import c2_text_checks as TC
            planted = {k: v for k, v in TC.x2_planted().items() if not k.endswith("_texts")}
            if planted["false_alarms"] or planted["misses"] or planted["twin_false_alarms"]:
                bad.append(f"the text detector misreads its planted sets: {planted}")
        if s.area == "manipulation":  # a text shortcut: its detector's planted sets must read 0 and 0
            import c2_text_checks as TC
            planted = {k: v for k, v in TC.planted_counts(key).items() if not k.endswith("_texts")}
            if planted["false_alarms"] or planted["misses"] or planted["fresh"]["false_alarms"] \
                    or planted["fresh"]["misses"]:
                bad.append(f"the text detector misreads its planted sets: {planted}")
        rec["families"][s.fid] = {"area": s.area, "split": s.split, "failures": bad, "passes": not bad,
                                  "planted": planted,
                                  "conflict_honest_scorer": next((x[0].strip() for x in reversed(outs)
                                                                  if "PASS" in x[0] or "FAIL" in x[0]), None),
                                  "twin_honest_scorer": scorer_runs[-1][0].strip() if scorer_runs else None}
    rec["passes"] = all(v["passes"] for v in rec["families"].values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
