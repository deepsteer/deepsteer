#!/usr/bin/env python3
"""Amendment 52's pod, the registered reads and the descriptive sweep (zero GPU).

    /opt/homebrew/bin/python3.13 a52_read.py DIR

- **3a, validation:** P-ctx minus the same-session base on the operator cell (``validation.json``,
  recomputed here); passes when the CI contains the RunPod draws' range (−0.286 to −0.272).
- **3b, A0n with GOOD_BEHAVIOR** on the nested 888 at k 2, against the base of record (the teacher
  pod's, against which GOOD_BEHAVIOR's −0.100 was read), games bootstrapped. Registered:
  **dilution** if its effect against base includes 0 and A0n-plus-GOOD minus GOOD_BEHAVIOR excludes
  0; **content** if it includes GOOD_BEHAVIOR's −0.100 and excludes 0; **both** otherwise. The
  citing rate beside (descriptive); this pod's base beside when 3c ran.
- **3c, P-ctx's single-decision read** as Amendment 49 item 2d registered it, with the product
  bar's goal-pressure ratio (``a49_read.mach_read``).
- **3d, the strength sweep, descriptive, no verdict:** at 0, 0.25, 0.5, 0.75 and 1 (0 and 1 are 3a's
  base and P-ctx), the operator cell's ordered share against the session base, the gain ratio
  (the arm's gain over A0s's in-session gain on the verification pod, −0.312: cross-session,
  printed as such) and cell A′'s rule-right shift beyond the teacher's (+0.088, S0a).

Writes ``DIR/a52_read.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[1] / "kdg_panel/scripts"),
                str(HERE.parents[1] / "mrb_morebench/scripts")]
import a49_read as R49  # noqa: E402
import cst_a52_pod as P  # noqa: E402
import teacher_read as TR  # noqa: E402

CST = HERE.parent
VERIFY = CST / "outputs/pressure_verify/gen"
TEACHER_A2_SHIFT = 0.088  # A0s's cell A′ rule-right shift (S0a; CST-A34)


def a0n_good(out: Path) -> dict | None:
    rows = R49._rows(out / "teacher" / "a52_a0ngood.jsonl")  # noqa: SLF001
    if not rows:
        return None
    lab = {f"{s['game']}|{s['episode']}|{s['step']}": s for s in R49._rows(TR.LABELS)}  # noqa: SLF001
    rec_rows = R49._rows(R49.TEACHER / "teacher" / "base_good_behavior.jsonl")  # noqa: SLF001
    nested = set(json.loads(R49.STATES.read_text())["nested"])
    base = {s: v for s, v in TR.rates(rec_rows, lab, "base").items() if s in nested}
    gb = {s: v for s, v in TR.rates(rec_rows, lab, "good_behavior").items() if s in nested}
    ag = TR.rates(rows, lab, "a0n_good")
    vb, vg = TR.paired(ag, base), TR.paired(ag, gb)
    lo, hi = vb["ci95"]
    inc0, incgb = lo <= 0 <= hi, lo <= R49.GB_NESTED <= hi
    g_x = vg["below_0"] or vg["above_0"]
    reading = ("dilution" if inc0 and g_x else "content" if incgb and not inc0 else "both")
    ss_rows = R49._rows(out / "teacher" / "a49_pctx.jsonl")  # noqa: SLF001
    ss = {s: v for s, v in TR.rates(ss_rows, lab, "base").items() if s in nested}
    mde = R49._mde(vb["ci95"])  # noqa: SLF001
    return {"rule": "Amendment 52 item 3b", "a0n_good_minus_base": dict(vb, mde=mde),
            "a0n_good_minus_good_behavior": vg, "reading": reading,
            "beside_same_session_base": TR.paired(ag, ss) if ss else None,
            "fallback_share": float(np.mean([r["fallback"] for r in rows])),
            "citing_descriptive": R49.citing(rows, lab, "a0n_good", TR.rates(rec_rows, lab,
                                                                             "base"))}


def sweep(out: Path) -> dict | None:
    import ctx_read as C
    import s0a_analysis as A
    import sdf2_quick as Z
    import sdf_reread_quick as Q

    gen = out / "gen"
    tags = ["base_session"] + [P.tag_of(s) for s in P.STRENGTHS] + ["p_ctx"]
    if not all((gen / t).exists() for t in tags):
        return None
    lab = A.labels()
    acted = lambda r: not r["capped"] and r["option"] is not None  # noqa: E731
    u = {t: Q.per_scenario(Z.cell_rows(t, "operator", lab, gen), acted) for t in tags}
    a0s = Q.per_scenario(Z.cell_rows("a0s_session", "operator", lab, VERIFY), acted)
    ids = sorted(set.intersection(*[set(v) for v in u.values()]) & set(a0s))
    C.GEN = gen
    res = {"rule": "Amendment 52 item 3d (descriptive, no verdict)", "n_items": len(ids),
           "gain_ratio_denominator": "A0s minus base on the verification pod (cross-session)",
           "strengths": {}}
    for s, t in [(0.0, "base_session")] + list(zip(P.STRENGTHS, tags[1:-1])) + [(1.0, "p_ctx")]:
        rec = {"tag": t, "ordered_share": Q.share(u[t], ids)}
        if t != "base_session":
            rec["vs_session_base"] = Q.paired(u[t], u["base_session"], ids)
            rec["gain_ratio"] = C.ratio_ci(u[t], u["base_session"], a0s, ids)
            a2 = C.cell_a2(t)
            if a2:
                rec["cell_a2_shift"] = a2["shift"]
                rec["cell_a2_beyond_teacher"] = a2["shift"]["mean"] - TEACHER_A2_SHIFT
        res["strengths"][str(s)] = rec
    return res


def read(out: Path) -> dict:
    val = json.loads((out / "validation.json").read_text()) if (
        out / "validation.json").exists() else None
    if (out / "gen" / "p_ctx").exists() and (out / "gen" / "base_session").exists():
        val = dict(val or {}, recomputed=P.reproduction(out))
    m = R49.mach_read(out)
    return {"rule": "Amendment 52", "3a_validation": val, "3b": a0n_good(out),
            "3c": m.get("2d"), "3c_citing": (m.get("5_citing") or {}).get("p_ctx"),
            "3d_sweep": sweep(out)}


def write(out: Path) -> dict:
    res = read(out)
    (out / "a52_read.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(">> a52_read:", json.dumps({
        "validation": (res["3a_validation"] or {}).get("passes"),
        "3b": (res["3b"] or {}).get("reading"), "3c": (res["3c"] or {}).get("branch")}))
    return res


if __name__ == "__main__":
    write(Path(sys.argv[1]))
