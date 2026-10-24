#!/usr/bin/env python3
"""The ``care_harm`` replicate's local read (Amendment 32; zero GPU).

    python3 papers/st_live_reference/scripts/analyze_st2mr.py \
        --rep papers/st_live_reference/outputs/st2mr/muse_glimmer_30b [--out st2mr_read.json]

- **Gate of record** (item 3): ``care_harm`` against the replicate's ``none`` and its dose-matched
  ``fill`` (0.3x, fresh draws), scenario bootstrap; both CIs excluding 0, the margin over ``fill``
  >= +0.05, §5.0's ΔP >= +0.10 against ``none`` (kept, flagged), Amendment 13's relative coherence.
- **Specificity of record** (item 4): ``care_harm`` - ``fill`` at 0.3x per item on the 60, item
  bootstrap, ±0.03 on the difference (Amendment 31's sign classes on the difference).
- **Parity**: the pod's ``spec_none`` against Cell 4's baseline; its ``spec_care_03`` against the
  ladder's (deterministic: identical).
- **Branch** (item 5): target / discriminating / non-specific / no lever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2 as A2  # noqa: E402
import analyze_st2dl as ADL  # noqa: E402
import analyze_st2ml as AML  # noqa: E402
import t13_driver as T13  # noqa: E402

OUTS = HERE.parent / "outputs"
BAR = 0.03


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def gate(rep: Path) -> dict:
    g = lambda n: _jsonl(rep / "gen" / f"arm_{n}.jsonl")  # noqa: E731
    r = A2.arm_read(g("care_03"), g("none"), g("fill_03"), "fill")
    lo_n, lo_f = r["dP_vs_none"]["ci95"][0], r["dP_vs_fill"]["ci95"][0]
    both = bool(lo_n is not None and lo_f is not None and lo_n > 0 and lo_f > 0)
    margin = bool(r["dP_vs_fill"]["mean"] is not None and r["dP_vs_fill"]["mean"] >= 0.05)
    point = bool(r["dP_vs_none"]["mean"] is not None and r["dP_vs_none"]["mean"] >= 0.10)
    ok = bool(both and margin and point and r["coherence"]["passes"])
    return {**r, "both_cis_exclude_0": both, "margin_over_fill_ok": margin,
            "dP_vs_none_ge_0.10": point, "passes": ok,
            "fill_vs_none": T13.delta_p(g("fill_03"), g("none"))}


def specificity(rep: Path, ids: list[str]) -> dict:
    base = rep / "spec" / "none"
    c, f = ADL.spec_items(rep, base, "care_03", ids), ADL.spec_items(rep, base, "fill_03", ids)
    s = sorted(set(c) & set(f))
    d = T13.boot([c[i] - f[i] for i in s])
    return {"care_minus_fill": d, "class": AML.sign_class(d["mean"]),
            "passes": bool(d["mean"] is not None and abs(d["mean"]) <= BAR),
            "care_vs_none": T13.boot([c[i] for i in s]),
            "fill_vs_none": T13.boot([f[i] for i in s])}


def two_way_boot(care: np.ndarray, fills: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """``care`` [items], ``fills`` [draws, items] (each against ``none``; every draw read on every
    item, the fully crossed design of Amendment 36 item 2): care - the mean fill, CI from
    resampling draws and items independently."""
    rng = np.random.default_rng(seed)
    nd, ni = fills.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        fbar = fills[rng.integers(0, nd, nd)].mean(0)
        out[b] = (care - fbar)[rng.integers(0, ni, ni)].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"mean": float(care.mean() - fills.mean()), "ci95": [float(lo), float(hi)],
            "n_draws": int(nd), "n_items": int(ni), "n_boot": n_boot}


def slot_fills(rep: Path, ladder: Path, ids: list[str], n_slots: int = 5) -> dict:
    """Per slot k (item j in slot j mod 5): {item: (seed-0 fill, seed-1 fill)}, each against the
    replicate's ``none`` (identical to the ladder's: parity exact)."""
    base = rep / "spec" / "none"
    out = {}
    for k in range(n_slots):
        a = ADL.spec_items(ladder, base, f"fill_03/draw{k}", ids)
        b = ADL.spec_items(rep, base, f"fill_03/draw{k}", ids)
        out[k] = {i: (a[i], b[i]) for i in sorted(set(a) & set(b))}
    return out


def fork_variance_components(care: dict, slots: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """Amendment 38 item 1 (a), of record: care - the mean of each item's two fill reads; the item
    bootstrap's variance plus one draw's variance (paired slot differences) over the 10 draws."""
    items = [(i, k) for k, d in slots.items() for i in d]
    diff = np.array([care[i] - np.mean(slots[k][i]) for i, k in items])
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    v_item = float(boot.var(ddof=1))
    dk = np.array([np.mean([b - a for a, b in d.values()]) for d in slots.values()])
    within = np.mean([np.var([b - a for a, b in d.values()], ddof=1) / len(d)
                      for d in slots.values()])
    v_draw1 = max((float(dk.var(ddof=1)) - float(within)) / 2, 0.0)
    n_draws = 2 * len(slots)
    se = float(np.sqrt(v_item + v_draw1 / n_draws))
    m = float(diff.mean())
    z = 1.959963984540054  # the normal 97.5% quantile
    return {"mean": m, "ci95": [m - z * se, m + z * se], "se": se,
            "se_item": float(np.sqrt(v_item)), "draw_sd": float(np.sqrt(v_draw1)),
            "n_draws": n_draws, "n_items": len(diff), "slot_differences": [float(x) for x in dk]}


def fork_slot_resample(care: dict, slots: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    """Amendment 38 item 1 (b), beside: each slot's two draws resampled with replacement, and the
    items resampled with replacement (understates the draw variance with two draws per slot)."""
    rng = np.random.default_rng(seed)
    per = [(care[i], np.array(slots[k][i])) for k, d in slots.items() for i in d]
    c = np.array([x[0] for x in per])
    f = np.stack([x[1] for x in per])  # [items, 2]
    slot_of = np.array([k for k, d in slots.items() for _ in d])
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = {k: rng.integers(0, 2, 2) for k in slots}
        fb = np.array([f[j, pick[slot_of[j]]].mean() for j in range(len(c))])
        out[b] = (c - fb)[rng.integers(0, len(c), len(c))].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"mean": float((c - f.mean(1)).mean()), "ci95": [float(lo), float(hi)]}


def specificity_fork(rep: Path, ladder: Path, ids: list[str]) -> dict:
    """Amendment 38: (a) of record, (b) beside, with the class of (a)'s point."""
    care = ADL.spec_items(rep, rep / "spec" / "none", "care_03", ids)
    slots = slot_fills(rep, ladder, ids)
    a, b = fork_variance_components(care, slots), fork_slot_resample(care, slots)
    return {"a_of_record": a, "b_beside": b, "class": AML.sign_class(a["mean"]),
            "passes": bool(abs(a["mean"]) <= BAR)}


def parity(rep: Path, cell4: Path, ladder: Path | None) -> dict:
    out = {}
    rec = json.loads((rep / "st2mr_record.json").read_text())["phases"]
    out["spec_none"] = (rec.get("spec_none") or {}).get("parity")
    lad = ladder / "spec" / "care_03" / "dl_chat_neutral.jsonl" if ladder else None
    if lad is not None and lad.exists():
        a = {(r["scenario_id"], r["rollout"]): r["option_logps"] for r in _jsonl(lad)}
        b = _jsonl(rep / "spec" / "care_03" / "dl_chat_neutral.jsonl")
        d = [abs(r["option_logps"][x] - a[(r["scenario_id"], r["rollout"])][x]) for r in b
             if (r["scenario_id"], r["rollout"]) in a for x in r["option_logps"]]
        out["spec_care_03_vs_ladder_max_abs_dlogp"] = max(d) if d else None
    return out


def branch(g: dict, s: dict) -> str:
    if not g["passes"]:
        return ("no lever: the gate fails against the dose-matched fill; care_harm moves "
                "protection no more than a random vector at its dose (ST-A20 -> R_b)")
    return {"specific": "target: care_harm at 0.3x is an admissible target on Muse (rule-right "
                        "tilt no larger than a random vector's at the same dose)",
            "discriminating": "discriminating: a conditional-lever candidate (flagged, read with "
                              "the fill margin)",
            "non-specific": "non-specific: no target; the ladder's 0.3x read was optimistic"}[
        s["class"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", type=Path, required=True)
    ap.add_argument("--cell4", type=Path, default=OUTS / "st2m" / "muse_glimmer_30b")
    ap.add_argument("--ladder", type=Path, default=OUTS / "st2ml" / "muse_glimmer_30b")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ids = json.loads((HERE.parent / "data" / "t15_specificity_v1.json").read_text())["ids"]
    g, s = gate(a.rep), specificity(a.rep, ids)
    res = {"gate": g, "specificity": s, "parity": parity(a.rep, a.cell4, a.ladder),
           "branch": branch(g, s)}
    p = a.out or a.rep / "st2mr_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    dn, df, d = g["dP_vs_none"], g["dP_vs_fill"], s["care_minus_fill"]
    print(f"gate: dP vs none {dn['mean']:+.3f} [{dn['ci95'][0]:+.3f}, {dn['ci95'][1]:+.3f}], vs "
          f"fill {df['mean']:+.3f} [{df['ci95'][0]:+.3f}, {df['ci95'][1]:+.3f}], coherent "
          f"{g['coherence']['passes']} -> {g['passes']}")
    print(f"specificity: care - fill {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
          f"({s['class']})")
    print(f">> {res['branch']}; parity {res['parity']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
