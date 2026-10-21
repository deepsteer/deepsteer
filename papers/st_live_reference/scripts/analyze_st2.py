#!/usr/bin/env python3
"""ST Phase 2's local read (Amendment 22, its riders and Cell 2 decisions; zero GPU), from the
downloaded pod rows and T1.1's bank.

    python3 papers/st_live_reference/scripts/analyze_st2.py \
        --st2a papers/st_live_reference/outputs/st2a/gpt_oss_20b \
        [--st2b papers/st_live_reference/outputs/st2b/gpt_oss_20b] [--out st2_read.json]

- **Cell 1**: each direction arm (``neg_a``, ``care_harm``, ``V_harm``) against ``none`` and against
  ``fill``, Amendment 13's relative coherence bars, §5.0's gate, rider (a)'s protect-ward reading;
  care beyond harm = ΔP(``care_harm``) − ΔP(``V_harm``), paired by scenario.
- **Cell 2**: the ``V_rule`` ablation against Cell 1's ``none`` and against the random ablation;
  the branch (P rises / no change / P falls) read only with the transport control passed (the
  dose-0 ablation moves the rule-present twins toward the rule-absent ones, CI excluding 0 in that
  direction); the rule-absent twins' ``none`` beside.
- **Cell 3**: the ``V_qwen`` arm against ``none`` and ``fill``; cos(``V_qwen``, ``V_oss``) with a
  scenario-bootstrap CI against a covariance-matched null at ``clause_end``, block 12 (aligned: CI
  above the null q95; orthogonal: |cos| CI below it; else unresolved); the writing-style loading.
- **Program level** (item 5, riders (a)-(b)): "steerable at this site" needs a protect-ward arm that
  survives Holm-Bonferroni over the ten pre-registered arms (five per model; an arm not yet run
  enters as p = 1); per-arm readings unadjusted and labelled; the negative reading carries each
  arm's MDE. An arm's one-sided p is the larger of its tests against ``none`` and against its
  matched control (an intersection-union test: protect-ward needs both).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import t13_driver as T13  # noqa: E402

OUTS = HERE.parent / "outputs"
ALPHA_ONE_SIDED = 0.025
N_BOOT, SEED = 10_000, 0
# the ten pre-registered arms of item 5 (rider (c): +V_harm replaced care_harm ⊥ V_harm)
PREREG = [f"{m}:{a}" for m in ("gpt_oss_20b", "muse_glimmer_30b")
          for a in ("neg_a", "care_harm", "V_harm", "V_qwen", "V_rule_ablation")]


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def p_one_sided(x) -> float:
    """Bootstrap p for mean(x) <= 0 (scenario resampling, T1.3's seed and draws)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return 1.0
    d = x[np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float((np.sum(d <= 0) + 1) / (N_BOOT + 1))


def arm_read(arm: list[dict], none: list[dict], ctrl: list[dict], ctrl_name: str,
             gate: bool = True) -> dict:
    pa, pn, pc = (T13.p_by_scenario(x) for x in (arm, none, ctrl))
    s_n, s_c = sorted(set(pa) & set(pn)), sorted(set(pa) & set(pc))
    dn_x, dc_x = [pa[s] - pn[s] for s in s_n], [pa[s] - pc[s] for s in s_c]
    dn, dc = T13.boot(dn_x), T13.boot(dc_x)
    coh = T13._coh(arm, none, relative=True)  # noqa: SLF001
    lo_n, lo_c = dn["ci95"][0], dc["ci95"][0]
    pw = bool(coh["passes"] and lo_n is not None and lo_c is not None and lo_n > 0 and lo_c > 0)
    out = {"dP_vs_none": dn, f"dP_vs_{ctrl_name}": dc, "coherence": coh, "protect_ward": pw,
           "p_one_sided": max(p_one_sided(dn_x), p_one_sided(dc_x)), "mde_vs_none": dn.get("mde")}
    if gate:
        ok = dn["mean"] is not None and dc["mean"] is not None
        out["passes_gate"] = bool(coh["passes"] and ok and dn["mean"] >= T13.GATE_DP
                                  and dc["mean"] >= T13.GATE_OVER_FILL)
    return out


def cell1(d: Path) -> dict:
    g = lambda n: _jsonl(d / "gen" / f"c1_{n}.jsonl")  # noqa: E731
    none, fill = g("none"), g("fill")
    out = {"arms": {a: arm_read(g(a), none, fill, "fill") for a in ("neg_a", "care_harm",
                                                                    "V_harm")}}
    pc, ph = T13.p_by_scenario(g("care_harm")), T13.p_by_scenario(g("V_harm"))
    ids = sorted(set(pc) & set(ph))
    out["care_beyond_harm"] = T13.boot([pc[s] - ph[s] for s in ids])
    out["fill_vs_none"] = T13.delta_p(fill, none)
    return out


def transport_passed(t: dict) -> bool | None:
    """The dose-0 ablation moves the rule-present twins toward the rule-absent ones: its CI
    excludes 0 on the side of (twin - present)."""
    a, tw = t["dP0_ablated_vs_present_headline"], t["dP0_twin_vs_present"]
    if a["ci95"][0] is None or tw["mean"] is None:
        return None
    return bool(a["ci95"][0] > 0 if tw["mean"] > 0 else a["ci95"][1] < 0)


def cell2(b: Path, a: Path) -> dict:
    rec = json.loads((b / "st2_record.json").read_text())["phases"]
    none = _jsonl(a / "gen" / "c1_none.jsonl")
    abl, rnd = _jsonl(b / "gen" / "c2_ablate.jsonl"), _jsonl(b / "gen" / "c2_random.jsonl")
    r = arm_read(abl, none, rnd, "random", gate=False)
    tp = transport_passed(rec["c2_transport"])
    lo, hi = r["dP_vs_none"]["ci95"]
    move = ("P rises" if lo is not None and lo > 0 else
            "P falls" if hi is not None and hi < 0 else "no change")
    branch = (move if move != "no change" else
              "no change, transport passed (the brake is not carried by rule content at this "
              "site; or the verbatim options re-install the gate, bounded by the dose-0 twins)"
              if tp else "no change, transport not passed: uninformative")
    return {"ablation": r, "transport": rec["c2_transport"], "transport_passed": tp,
            "branch": branch, "twin_none_vs_present": T13.delta_p(
                _jsonl(b / "gen" / "c2_twin_none.jsonl"), none),
            "direction": rec["c2_direction"], "anchor": rec.get("c2_anchor"),
            "freegen": rec.get("c2_freegen")}


def bank_scenario_means(t11: Path, sample_every: int = 8):
    """T1.1's bank at clause_end, block 12, H1_train: per scenario and action (protect / rule) the
    sum and count over positions; a strided sample of rows for the covariance null (every 8th,
    ~8,000 rows, so the sample covariance is full-rank at 2,880 dimensions)."""
    import constancy_read as CR

    rows = _jsonl(t11 / "gen" / "h1_rollouts_read.jsonl")
    info = {i: (r["scenario_id"], CR.ACTION.get(r.get("norm_status"))) for i, r in enumerate(rows)
            if r.get("split") == "train"}
    acc: dict = {}
    sample = []
    code = CR.CLASSES["clause_end"]
    k = 0
    for f in sorted((t11 / "bank").glob("bank_*.npz")):
        z = np.load(f)
        sel = (z["cls"] & code) > 0
        h = z["resid"][sel, CR.L_STAR_SLOT].astype(np.float64)
        for hv, r in zip(h, z["row"][sel]):
            if int(r) not in info or info[int(r)][1] is None:
                continue
            sid, act = info[int(r)]
            d = acc.setdefault(sid, {"protect": [0.0, 0], "rule": [0.0, 0]})
            d[act][0] = d[act][0] + hv
            d[act][1] += 1
            k += 1
            if k % sample_every == 0:
                sample.append(hv.astype(np.float32))
    return acc, np.array(sample)


def unit(v):
    return v / np.linalg.norm(v)


def cos_read(per_q: dict, per_o: dict, samples: np.ndarray, n_null: int = 1000) -> dict:
    import steering_gate as SG

    q_ids, o_ids = sorted(per_q), sorted(per_o)
    Qm, Om = np.array([per_q[s] for s in q_ids]), np.array([per_o[s] for s in o_ids])
    c = float(unit(Qm.mean(0)) @ unit(Om.mean(0)))
    rng = np.random.default_rng(SEED)
    bs = [float(unit(Qm[rng.integers(0, len(Qm), len(Qm))].mean(0))
                @ unit(Om[rng.integers(0, len(Om), len(Om))].mean(0))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    null = SG.fill_draws(samples.astype(np.float64), 1.0, n_null, SEED).astype(np.float64)
    q95 = float(np.percentile(np.abs(null @ unit(Om.mean(0))), 95))
    reading = ("aligned: shared content, a transfer candidate" if lo > q95 else
               "orthogonal: different content" if max(abs(lo), abs(hi)) < q95 else "unresolved")
    return {"cos": c, "ci95": [float(lo), float(hi)], "null_abs_q95": q95,
            "n_qwen_scenarios": len(q_ids), "n_oss_scenarios": len(o_ids), "reading": reading}


def split_half_cos(per: dict, seed: int = 20261021) -> dict:
    """Stability: the direction from a seeded half of the scenarios against the other half."""
    import random

    ids = sorted(per)
    half = set(random.Random(seed).sample(ids, len(ids) // 2))
    va = unit(np.mean([per[s] for s in ids if s in half], 0))
    vb = unit(np.mean([per[s] for s in ids if s not in half], 0))
    return {"cos": float(va @ vb), "n_a": len(half), "n_b": len(ids) - len(half), "seed": seed}


def v_oss(t11: Path) -> tuple[dict, dict, np.ndarray]:
    acc, samples = bank_scenario_means(t11)
    per = {s: v["protect"][0] / v["protect"][1] - v["rule"][0] / v["rule"][1]
           for s, v in acc.items() if v["protect"][1] and v["rule"][1]}
    return per, acc, samples


def cell3(b: Path, a: Path, t11: Path) -> dict:
    g = lambda d, n: _jsonl(d / "gen" / f"{n}.jsonl")  # noqa: E731
    out = {"arm": arm_read(g(b, "c3_qwen_arm"), g(a, "c1_none"), g(a, "c1_fill"), "fill")}
    z = np.load(b / "c3" / "v_qwen.npz")
    per_q = {str(s): v.astype(np.float64) for s, v in zip(z["ids"], z["per_scenario"])}
    per_o, acc, samples = v_oss(t11)
    out["cos_V_qwen_V_oss"] = cos_read(per_q, per_o, samples)
    out["split_half"] = {"V_oss": split_half_cos(per_o), "V_qwen": split_half_cos(per_q)}
    rz = np.load(b / "c3" / "reencode.npz")
    qp: dict = {}
    for key, sm, n, act in zip(rz["keys"], rz["sums"], rz["counts"], rz["action"]):
        if str(act) == "protect" and int(n):
            d = qp.setdefault(str(key).split("|")[1], [0.0, 0])
            d[0], d[1] = d[0] + sm, d[1] + int(n)
    style = [qp[s][0] / qp[s][1] - acc[s]["protect"][0] / acc[s]["protect"][1]
             for s in sorted(qp) if s in acc and acc[s]["protect"][1]]
    out["writing_style_loading"] = {
        "cos_V_qwen_V_style": float(unit(np.mean(style, 0)) @ unit(np.array(z["v_qwen"],
                                                                            np.float64)))
        if style else None, "n_scenarios": len(style)}
    return out


def holm(pvals: dict[str, float], alpha: float = ALPHA_ONE_SIDED) -> dict[str, bool]:
    names = sorted(pvals, key=pvals.get)
    m, out, stop = len(names), {}, False
    for i, n in enumerate(names):
        rej = (not stop) and pvals[n] <= alpha / (m - i)
        stop = stop or not rej
        out[n] = rej
    return out


def program(arms: dict[str, dict]) -> dict:
    """Item 5 with riders (a)-(b): Holm over the ten pre-registered arms for the positive reading
    (an arm not yet run enters as p = 1); per-arm readings unadjusted; MDEs for the negative."""
    p = {k: (arms[k]["p_one_sided"] if k in arms else 1.0) for k in PREREG}
    rej = holm(p)
    surv = [k for k in PREREG if k in arms and rej[k] and arms[k]["protect_ward"]]
    return {"p_one_sided": p, "holm_rejects": rej, "protect_ward_unadjusted":
            {k: arms[k]["protect_ward"] for k in arms}, "steerable_at_this_site": bool(surv),
            "surviving": surv, "arms_not_run": [k for k in PREREG if k not in arms],
            "mde_vs_none": {k: arms[k].get("mde_vs_none") for k in arms},
            "reading": ("steerable at this site (Holm over ten arms): " + ", ".join(surv)) if surv
            else "no protect-ward move survives Holm over the ten arms; the negative reading "
                 "carries each arm's MDE"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st2a", type=Path, required=True)
    ap.add_argument("--st2b", type=Path, default=None)
    ap.add_argument("--t11", type=Path, default=OUTS / "t11" / "gpt_oss_20b")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res: dict = {"cell1": cell1(a.st2a)}
    key = "gpt_oss_20b"
    arms = {f"{key}:{k}": v for k, v in res["cell1"]["arms"].items()}
    if a.st2b is not None:
        res["cell2"] = cell2(a.st2b, a.st2a)
        res["cell3"] = cell3(a.st2b, a.st2a, a.t11)
        arms[f"{key}:V_rule_ablation"] = res["cell2"]["ablation"]
        arms[f"{key}:V_qwen"] = res["cell3"]["arm"]
    res["program"] = program(arms)
    p = a.out or a.st2a / "st2_read.json"
    p.write_text(json.dumps(res, indent=1))
    for k, v in arms.items():
        d = v["dP_vs_none"]
        print(f"{k:28s} dP {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
              f"p1 {v['p_one_sided']:.4f} protect-ward {v['protect_ward']}")
    print(f">> {res['program']['reading']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
