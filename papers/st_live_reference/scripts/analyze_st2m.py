#!/usr/bin/env python3
"""Cell 4's local read, Muse's transfer test (Amendments 29-30; zero GPU), from the downloaded rows.

    python3 papers/st_live_reference/scripts/analyze_st2m.py \
        --st2m papers/st_live_reference/outputs/st2m/muse_glimmer_30b [--out st2m_read.json]

- **Arms**: V_qwen, -V_harm and ``care_harm`` against Muse's ``none`` and ``fill`` on the engaged
  set x 8 (§5.0's gate under Amendment 13's relative coherence bars; rider (a)'s protect-ward
  reading); an arm that failed coherence on length is read at its alpha / 2 rerun (Amendment 25).
- **Specificity**: each direction's dose-0 read against Muse's ``spec_none`` (±0.03); if
  ``spec_fill`` falls outside ±0.03 the bar applies to the difference from fill (Amendment 29's S2).
  A target needs the gate, coherence and specificity (Amendment 27).
- **Branch** (Amendment 30 item 6), on V_qwen's and -V_harm's specificity: both fail, the directions
  carry the non-specific component; both pass, GPT-OSS's site does and Muse becomes the adapter
  development model; a split goes to ANOMALIES. ``care_harm`` is read on its own.
- **Program**: Holm over twelve arms (Muse's -V_harm the twelfth), GPT-OSS's from its pods' rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2 as A2  # noqa: E402
import analyze_st2dl as ADL  # noqa: E402
import t13_driver as T13  # noqa: E402

OUTS = HERE.parent / "outputs"
MUSE, OSS = "muse_glimmer_30b", "gpt_oss_20b"
DIRS = ("V_qwen", "neg_V_harm", "care_harm")
BAR = 0.03
PREREG12 = A2.PREREG + [f"{MUSE}:neg_V_harm"]  # Amendment 30 item 5: m = 12


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def spec_read(d: Path, ids: list[str]) -> dict:
    """Each spec arm against Muse's spec_none, per item; S1/S2 by spec_fill."""
    none = d / "spec" / "none"
    out: dict = {}
    per = {}
    for arm in ("fill", *DIRS, *[f"{x}_half" for x in DIRS]):
        if (d / "spec" / arm / "dl_chat_neutral.jsonl").exists():
            per[arm] = ADL.spec_items(d, none, arm, ids)
            out[arm] = {"dP_violating": T13.boot(list(per[arm].values()))}
    s1 = "fill" in out and abs(out["fill"]["dP_violating"]["mean"]) <= BAR
    out["branch_rule"] = "S1 (against none)" if s1 else "S2 (against fill)"
    for arm, x in out.items():
        if arm in ("fill", "branch_rule"):
            continue
        if s1:
            x["passes"] = abs(x["dP_violating"]["mean"]) <= BAR
            s = sorted(set(per[arm]) & set(per["fill"]))  # descriptive beside S1's verdict
            x["d_vs_fill_descriptive"] = T13.boot([per[arm][i] - per["fill"][i] for i in s])
        else:
            s = sorted(set(per[arm]) & set(per["fill"]))
            x["d_vs_fill"] = T13.boot([per[arm][i] - per["fill"][i] for i in s])
            x["passes"] = abs(x["d_vs_fill"]["mean"]) <= BAR
    return out


def arms_read(d: Path, rec: dict) -> dict:
    g = lambda n: _jsonl(d / "gen" / f"arm_{n}.jsonl")  # noqa: E731
    none, fill = g("none"), g("fill")
    out = {}
    for arm in DIRS:
        p = rec.get(f"arm_{arm}") or {}
        if not (d / "gen" / f"arm_{arm}.jsonl").exists():
            out[arm] = {"status": p.get("status", "not run")}
            continue
        r = A2.arm_read(g(arm), none, fill, "fill")
        sd = p.get("step_down")
        if isinstance(sd, dict) and (d / "gen" / f"arm_{arm}_half.jsonl").exists():
            r = {**A2.arm_read(g(f"{arm}_half"), none, fill, "fill"), "read_at": "alpha / 2",
                 "at_alpha": r}
        out[arm] = r
    return out


def branch(spec: dict) -> dict:
    def ok(arm: str) -> bool | None:
        x = spec.get(f"{arm}_half") or spec.get(arm)
        return None if x is None else bool(x["passes"])
    v, h = ok("V_qwen"), ok("neg_V_harm")
    if v is None or h is None:
        return {"id": None, "reading": "not read (a specificity read is missing)"}
    if not v and not h:
        return {"id": "both_fail", "reading": "the directions carry the non-specific component, "
                "in a second family: GPT-OSS's result is about the directions, not one site"}
    if v and h:
        return {"id": "both_pass", "reading": "GPT-OSS's site carries it; Muse becomes the "
                "adapter development model"}
    return {"id": "split", "reading": "a split: an ANOMALIES entry before any wording",
            "V_qwen_passes": v, "neg_V_harm_passes": h}


def program(muse_arms: dict) -> dict:
    """Holm over twelve arms: GPT-OSS's from its pods (V_qwen at 133.6 of record), Muse's here."""
    a2, b2, c2 = (OUTS / x / OSS for x in ("st2a", "st2b", "st2c"))
    arms: dict = {}
    if all(p.exists() for p in (a2, b2, c2)):
        import analyze_st2c as AC

        oss = AC.program(c2, a2, b2, AC.qwen_half(c2, a2))
        arms = {k: {"p_one_sided": v, "protect_ward": oss["protect_ward_unadjusted"].get(k, False),
                    "mde_vs_none": oss["mde_vs_none"].get(k)}
                for k, v in oss["p_one_sided"].items() if k.startswith(OSS)}
    for k, v in muse_arms.items():
        if "p_one_sided" in v:
            arms[f"{MUSE}:{k}"] = v
    p = {k: (arms[k]["p_one_sided"] if k in arms else 1.0) for k in PREREG12}
    rej = A2.holm(p)
    surv = [k for k in PREREG12 if k in arms and rej[k] and arms[k]["protect_ward"]]
    return {"m": len(PREREG12), "p_one_sided": p, "holm_rejects": rej, "surviving": surv,
            "steerable_at_this_site": bool(surv),
            "arms_not_run": [k for k in PREREG12 if k not in arms]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st2m", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rec = json.loads((a.st2m / "st2m_record.json").read_text())["phases"]
    ids = json.loads((HERE.parent / "data" / "t15_specificity_v1.json").read_text())["ids"]
    spec = spec_read(a.st2m, ids)
    arms = arms_read(a.st2m, rec)
    targets = {}
    for arm in DIRS:
        r, s = arms.get(arm) or {}, spec.get(f"{arm}_half") if (arms.get(arm) or {}).get(
            "read_at") else spec.get(arm)
        targets[arm] = bool(r.get("passes_gate") and s and s["passes"])
    res = {"derive": {k: rec.get("derive", {}).get(k) for k in (
               "n_engaged", "null_no_brake", "brake_B", "median_norm", "alphas", "sweep",
               "layers_descriptive")},
           "pilot": rec.get("pilot"), "sweep": rec.get("sweep"),
           "directions": rec.get("directions"), "arms": arms, "specificity": spec,
           "targets": targets, "branch": branch(spec), "program": program(arms)}
    p = a.out or a.st2m / "st2m_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(f"engaged {res['derive']['n_engaged']}; alpha {(rec.get('sweep') or {}).get('alpha')}")
    for arm, r in arms.items():
        if "dP_vs_none" in r:
            d = r["dP_vs_none"]
            print(f"{arm:11s} dP {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                  f"gate {r.get('passes_gate')} coherent {r['coherence']['passes']}")
        else:
            print(f"{arm:11s} {r.get('status')}")
    for arm, x in spec.items():
        if isinstance(x, dict):
            d = x.get("d_vs_fill") or x["dP_violating"]
            print(f"spec {arm:14s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                  f"passes {x.get('passes')}")
    print(f">> {spec['branch_rule']}; branch: {res['branch']['reading']}; targets {targets}")
    print(f">> wrote {p}")


if __name__ == "__main__":
    main()
