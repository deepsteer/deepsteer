"""Amendment 11's carrier call on a `none` panel read (ST_PLAN §2, §14; author, 2026-10-19).

From one read of the five dose-0 letter cells (586 x 8, the instrument of record) this computes,
with KDG's own estimators (``analyze_phase1_session_a.option_cell`` / ``four_cells`` / ``boot``
and ``analyze_gptoss``'s C2 and C3 arithmetic, imported, never re-implemented):

- the known-gap band: g_band = p_D(known gap) - p_J(neutral) over scenarios clearing the 0.5 mass
  floor in both cells; in band iff its 95% lower bound >= 0.10 (P1-A9);
- E_none on the model-free set (all four E cells above the floor), scenario bootstrap, and its MDE
  (2.8 x SE from the bootstrap interval);
- the call: ``deferred`` if the band is not in band (never defaulted); else ``B`` (carrier) iff
  E_none's CI excludes 0 AND its point is >= the threshold, ΔE's MDE at the full panel; else ``A``.

Threshold: GPT-OSS-20B's is fixed before the read, sqrt(2) x the KDG §27 MDE (0.0189); a model with
no prior read (Muse) uses sqrt(2) x the MDE of this same read (a variance; the point plays no
part). The read's own sqrt(2) x MDE is reported beside a fixed threshold.

    python3 panel_call.py <panel_dir> --model gpt_oss_20b --out call.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KDG = HERE.parents[1] / "kdg_panel" / "scripts"
sys.path.insert(0, str(KDG))
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

KDG_GPTOSS_MDE = 0.013346476956120698  # analysis_gptoss.json c3.model_free.E_prob.mde (n 586)
FIXED_THRESHOLD = {"gpt_oss_20b": math.sqrt(2) * KDG_GPTOSS_MDE}  # 0.0189, fixed before the read
BAND_LB = 0.10  # P1-A9


def scenario_status() -> dict:
    """KDG's scenario status map (option id -> norm status), as ``analyze_gptoss`` builds it."""
    import analyze_gptoss as G

    from deepsteer.kdg.schema import load_scenario_dir

    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    return G._status(scen)


def band(d: Path, status: dict) -> dict:
    """KDG C2 on this read: g_band and the in-band verdict."""
    jm = A.option_cell([d], "jl_chat_neutral", status)
    k = A.option_cell([d], "dl_chat_known_gap", status)
    ids = sorted(s for s in k if s in jm and min(k[s]["mass"], jm[s]["mass"]) >= A.FLOOR)
    g = A.boot(np.array([k[s]["p"] - jm[s]["p"] for s in ids]))
    lo = g["ci95"][0]
    return {"g_band": g, "in_band": bool(lo is not None and lo >= BAND_LB), "lb_bar": BAND_LB}


def e_none(d: Path, status: dict) -> dict:
    """KDG C3 (model-free set) on this read: E_prob with its scenario bootstrap and MDE."""
    t = A.four_cells([d], OS.CHAT, status)
    free = sorted(s for s in t if t[s]["mass_min"] >= A.FLOOR)
    e = A.boot(np.array([A.scales(t[s])["E_prob"] for s in free]))
    return {"n": len(free), "E_prob": e, "mde": e.get("mde")}


def call(d: Path, model: str, status: dict | None = None) -> dict:
    """Amendment 11's call on one panel read."""
    status = status if status is not None else scenario_status()
    b = band(d, status)
    e = e_none(d, status)
    own = math.sqrt(2) * e["mde"] if e["mde"] is not None else None
    thr = FIXED_THRESHOLD.get(model, own)
    out = {"model": model, "band": b, "E_none": e, "threshold": thr,
           "threshold_source": ("fixed before the read: sqrt(2) x KDG section 27 MDE"
                                if model in FIXED_THRESHOLD else
                                "sqrt(2) x this read's MDE (variance only)"),
           "own_sqrt2_mde": own, "rule": "ST_PLAN Amendment 11"}
    if not b["in_band"]:
        out["call"] = "deferred"
        out["reason"] = "known-gap band not in band on this read (Amendment 11 item 3a)"
        return out
    lo, hi = e["E_prob"]["ci95"]
    pt = e["E_prob"]["mean"]
    excludes0 = lo is not None and (lo > 0 or hi < 0)
    carrier = bool(excludes0 and pt >= thr)
    out["call"] = "B" if carrier else "A"
    out["reason"] = ("carrier: CI excludes 0 and point >= threshold" if carrier else
                     "not a carrier: " + ("CI includes 0" if not excludes0 else
                                          "point below threshold"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel_dir", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c = call(a.panel_dir, a.model)
    a.out.write_text(json.dumps(c, indent=1))
    e = c["E_none"]["E_prob"]
    print(f">> {a.model} panel call: {c['call']} ({c['reason']}); E_none {e['mean']:.4f} "
          f"[{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}] n {c['E_none']['n']}, threshold "
          f"{c['threshold']:.4f}; g_band {c['band']['g_band']['mean']:.3f} "
          f"[{c['band']['g_band']['ci95'][0]:.3f}, {c['band']['g_band']['ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
