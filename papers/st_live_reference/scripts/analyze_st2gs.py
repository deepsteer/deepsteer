#!/usr/bin/env python3
"""The GPT-OSS site sweep's read (ST_PLAN Amendment 51 and its additions). Zero GPU.

    python3 papers/st_live_reference/scripts/analyze_st2gs.py --gs <outputs/st2gs/gpt_oss_20b> \
        --c1 <outputs/st2a/gpt_oss_20b> [--geom <outputs/t11/gpt_oss_20b> --inputs <st2gs_inputs>]

- **Ranking** (item 4; first addition items 2-3): per site, the exact steer's scenario mean with a
  scenario bootstrap (the ranking of record); AtP's per-site mean and its Spearman correlation
  with the exact steer across the ten sites (T0.5's bar 0.7); AtP's fill excess (the analog's AtP
  minus the mean of the site's ten fill draws', scenario bootstrap; a linear estimate); split-half
  stability (the exact steer's ten-site vector on two seeded scenario halves, Spearman); block 12's
  ``clause_end`` beside Amendment 50's measured dP (+0.010 at 0.2x, +0.041 at 0.4x).
- **Gate per swept site**, at the rung of record (0.4x, or 0.3x after an incoherent 0.4x): dP
  against Cell 1's ``none`` (>= +0.10, CI > 0) and against the site's fill arm (>= +0.05, CI > 0;
  per scenario), Amendment 13's coherence bars; Cell 1's block 12 fill beside.
- **Specificity per site**: the analog against the site's ten fill draws on the 60 rule-right
  items (draws x items bootstrap), and against ``none``.
- **Branch** (item 7): R_site / R_model at the site / R_lever extended.
- **Geometry** (item 5; ``--geom``): per site, cos(analog, protect - rule), its held-out
  separation, the covariance-matched |cos| q95 and the channel-chance level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2fx as FX  # noqa: E402
import analyze_st2ml as AML  # noqa: E402
import analyze_st2mr as AMR  # noqa: E402
import constancy_read as CR  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

SPEC_BAR = 0.03
SPEARMAN_BAR = 0.7
READ_SITE = "b12_clause_end"
MEASURED_B12 = {"02": 0.010, "04": 0.041}  # Amendment 50: Cell 1's 0.2x and the ladder's 0.4x
NULL_DRAWS = 1000


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _scenario_means(scen: np.ndarray, v: np.ndarray) -> dict[str, float]:
    return {s: float(np.mean(v[scen == s])) for s in sorted(set(scen.tolist()))}


def rank_read(gs: Path, rec: dict) -> dict:
    z = np.load(gs / "rank.npz")
    sites, scen = [str(s) for s in z["sites"]], z["scenario"]
    out: dict = {"sites": {}, "top2": rec["rank"]["top2"], "atp": rec["rank"]["atp"]}
    ex_mean, atp_mean = [], []
    for j, s in enumerate(sites):
        e = _scenario_means(scen, z["exact"][:, j])
        r = {"exact": T13.boot(list(e.values()))}
        if not np.isnan(z["atp"][:, j]).any():
            a = _scenario_means(scen, z["atp"][:, j])
            f = _scenario_means(scen, z["atp_fill"][:, j].mean(-1))
            r["atp"] = T13.boot(list(a.values()))
            r["atp_fill_excess"] = T13.boot([a[k] - f[k] for k in a])
            atp_mean.append(r["atp"]["mean"])
        ex_mean.append(r["exact"]["mean"])
        out["sites"][s] = r
    out["atp_spearman"] = (spearman(atp_mean, ex_mean) if len(atp_mean) == len(sites) else None)
    out["atp_validates"] = (None if out["atp_spearman"] is None
                            else bool(out["atp_spearman"] >= SPEARMAN_BAR))
    u = np.array(sorted(set(scen.tolist())))
    perm = np.random.default_rng(0).permutation(len(u))
    half = set(u[perm[: len(u) // 2]].tolist())
    h1 = [float(np.mean(z["exact"][np.isin(scen, list(half)), j])) for j in range(len(sites))]
    h2 = [float(np.mean(z["exact"][~np.isin(scen, list(half)), j])) for j in range(len(sites))]
    out["split_half_spearman"] = spearman(h1, h2)
    if READ_SITE in out["sites"]:
        out["calibration_b12"] = {**out["sites"][READ_SITE], "measured_dP": MEASURED_B12}
    return out


def gate(rows: list[dict], none: list[dict], fill_site: list[dict], fill_c1: list[dict],
         coh: dict) -> dict:
    keys = {r["rollout_key"] for r in rows}
    dn = T13.delta_p(rows, [r for r in none if r["rollout_key"] in keys])
    pa, pf = T13.p_by_scenario(rows), T13.p_by_scenario(fill_site)
    df = T13.boot([pa[s] - pf[s] for s in sorted(set(pa) & set(pf))])
    pc = T13.p_by_scenario([r for r in fill_c1 if r["rollout_key"] in keys])
    dc = T13.boot([pa[s] - pc[s] for s in sorted(set(pa) & set(pc))])
    passes = bool(coh.get("passes") and dn["mean"] is not None and dn["mean"] >= T13.GATE_DP
                  and dn["ci95"][0] > 0 and df["mean"] >= T13.GATE_OVER_FILL
                  and df["ci95"][0] > 0)
    return {"dP_vs_none": dn, "dP_vs_site_fill": df, "dP_vs_cell1_fill_beside": dc,
            "coherent": bool(coh.get("passes")), "passes": passes, "n_rollouts": len(rows)}


def specificity(gs: Path, site: str, rung: str) -> dict:
    none = FX.p_violating(_jsonl(gs / "spec" / "none" / "dl_chat_neutral.jsonl"))
    care = FX.p_violating(_jsonl(gs / "spec" / f"care_{site}_{rung}" / "dl_chat_neutral.jsonl"))
    rows = _jsonl(gs / "spec" / f"fill10_{site}_{rung}" / "dl_chat_neutral.jsonl")
    draws = sorted({r["fill_draw"] for r in rows})
    by = {g: FX.p_violating([r for r in rows if r["fill_draw"] == g]) for g in draws}
    items = sorted(set(none) & set(care) & set.intersection(*[set(v) for v in by.values()]))
    c = np.array([care[i] - none[i] for i in items])
    f = np.array([[by[g][i] - none[i] for i in items] for g in draws])
    vs = AMR.two_way_boot(c, f)
    return {"vs_fill10": vs, "class": AML.sign_class(vs["mean"]), "vs_none": T13.boot(list(c)),
            "fill_vs_none": float(f.mean()), "n_items": len(items), "n_draws": len(draws)}


def site_read(gs: Path, c1: Path, rec: dict, site: str) -> dict:
    none = _jsonl(c1 / "gen" / "c1_none.jsonl")
    fill_c1 = _jsonl(c1 / "gen" / "c1_fill.jsonl")
    out: dict = {}
    for rung in ("04", "03"):
        ph = rec.get(f"arm_care|{site}|{rung}")
        if not ph or not (gs / "gen" / f"gs_care_{site}_{rung}.jsonl").exists():
            continue
        g = gate(_jsonl(gs / "gen" / f"gs_care_{site}_{rung}.jsonl"), none,
                 _jsonl(gs / "gen" / f"gs_fill_{site}_{rung}.jsonl"), fill_c1, ph["coherence"])
        sp = (specificity(gs, site, rung)
              if (gs / "spec" / f"care_{site}_{rung}").exists() else None)
        out[rung] = {"gate": g, "specificity": sp,
                     "fill_arm_coherent": (rec.get(f"arm_fill|{site}|{rung}") or {})
                     .get("coherent")}
    rung = "04" if out.get("04", {}).get("gate", {}).get("coherent") else (
        "03" if out.get("03", {}).get("gate", {}).get("coherent") else None)
    out["rung_of_record"] = rung
    return out


def branch(reads: dict[str, dict]) -> str:
    passing = {}
    for s, r in reads.items():
        rung = r.get("rung_of_record")
        if rung and r[rung]["gate"]["passes"]:
            sp = r[rung]["specificity"]
            passing[s] = (r[rung]["gate"]["dP_vs_none"]["mean"],
                          sp is not None and sp["vs_fill10"]["mean"] <= SPEC_BAR)
    specific = {s: v for s, v in passing.items() if v[1]}
    if specific:
        best = max(specific, key=lambda s: specific[s][0])
        return (f"R_site: a specific protective lever at {best} (the recipe's site; "
                f"{len(specific)} of {len(reads)} swept sites qualify)")
    if passing:
        return "R_model at the site: the gate passes, specificity above +0.03 at every passing site"
    return ("R_lever extended: no specific protective lever for care_harm at block 12's clause_end "
            "or the two ranked sites at a coherent dose up to 0.4x of each site's norm")


def read(gs: Path, c1: Path) -> dict:
    rec = json.loads((gs / "st2gs_record.json").read_text())["phases"]
    out: dict = {"rule": "ST_PLAN Amendment 51 and its additions",
                 "analogs": {k: rec["analogs"].get(k) for k in ("parity_cos_block12", "parity_ok",
                                                                "cos_between_blocks")},
                 "ranking": rank_read(gs, rec), "fallback": (rec.get("fallback") or {}).get("plan")}
    out["sites"] = {s: site_read(gs, c1, rec, s) for s in rec["rank"]["top2"]}
    out["branch"] = branch(out["sites"])
    return out


def geometry(t11: Path, inputs: Path, analogs: Path, seed: int = 0) -> dict:
    """Item 5, descriptive: per site, cos(analog, protect - rule) over the 40 mixed-outcome
    H1_train scenarios; the direction's held-out separation (built on one seeded half of the
    scenarios, read as the within-scenario protect - rule margin on the other, scenario bootstrap);
    the covariance-matched |cos| q95 (1,000 draws) and the channel-chance level sqrt(2/(pi PR))."""
    doc = json.loads((inputs / "sites.json").read_text())["sites"]
    za = np.load(analogs)
    rows = [json.loads(x) for x in (t11 / "gen" / "h1_rollouts_read.jsonl").open()]
    act = {i: CR.ACTION.get(r.get("norm_status")) for i, r in enumerate(rows)
           if r.get("split") == "train"}
    sums: dict = {}
    samples: dict = {}
    for f in sorted((t11 / "bank").glob("bank_*.npz")):
        z = np.load(f)
        for cls, code in CR.CLASSES.items():
            m = ((z["cls"] & code) > 0) & np.isin(z["row"], [k for k, v in act.items() if v])
            x = z["resid"][m].astype(np.float32)
            rr = z["row"][m]
            for row in np.unique(rr):
                k = (int(row), cls)
                s, n = sums.get(k, (0.0, 0))
                sums[k] = (s + x[rr == row].sum(0), n + int((rr == row).sum()))
            samples.setdefault(cls, []).append(x[:: max(1, len(x) // 200)])
    mean = {k: s / n for k, (s, n) in sums.items()}
    by_scen: dict = {}
    for (row, cls), v in mean.items():
        by_scen.setdefault((rows[row]["scenario_id"], cls), {}).setdefault(act[row], []).append(v)
    out = {}
    for name, st in doc.items():
        slot, cls = st["slot"], st["cls"]
        a = za[f"block{st['block']}"]
        diffs = {sid: np.mean(d["protect"], 0)[slot] - np.mean(d["rule"], 0)[slot]
                 for (sid, c), d in by_scen.items() if c == cls and "protect" in d and "rule" in d}
        sids = sorted(diffs)
        d_all = np.mean([diffs[s] for s in sids], 0)
        perm = np.random.default_rng(seed).permutation(len(sids))
        ha, hb = [sids[i] for i in perm[: len(sids) // 2]], [sids[i] for i in perm[len(sids) // 2:]]
        d_a = np.mean([diffs[s] for s in ha], 0)
        d_a /= np.linalg.norm(d_a)
        sep = []
        for s in hb:
            d = by_scen[(s, cls)]
            mp = np.mean([v[slot] @ d_a / np.linalg.norm(v[slot]) for v in d["protect"]])
            mr = np.mean([v[slot] @ d_a / np.linalg.norm(v[slot]) for v in d["rule"]])
            sep.append(float(mp - mr))
        samp = np.concatenate(samples[cls])[:, slot]
        null = SG.fill_draws(samp, 1.0, NULL_DRAWS, seed)
        dn = d_all / np.linalg.norm(d_all)
        out[name] = {"cos_analog_protect_minus_rule": float(a @ dn),
                     "null_abs_cos_q95": float(np.quantile(np.abs(null @ dn), 0.95)),
                     "channel_chance": float(np.sqrt(2 / (np.pi * st["pr_median_21"]))),
                     "held_out_separation": T13.boot(sep), "n_scenarios": len(sids),
                     "pr_median_21": st["pr_median_21"], "eligible": st["eligible"],
                     "caveat": "outcome-conditioned: the contrast carries the content of the "
                               "deliberation that led to each action (estimator-traps #4)"}
    return out


def regen_read(arms: dict[str, Path], none: Path, tok, seed: int = 0) -> dict:
    """ST-A31's discriminator (Amendment 53): per scenario, (a) the log ratio of mean trace length,
    steered over ``none`` (same rollout keys); (b) Pearson r across scenarios of (a) with dP;
    (c) the change in Amendment 41's lexicons per 100 words of trace text. Scenario bootstrap."""
    import analyze_a2m as AM

    def trace_text(r: dict) -> str:
        a = r.get("trace_start") or 0
        return tok.decode(r["gen_ids"][a: a + (r.get("trace_len") or 0)])

    nrows = _jsonl(none)
    out: dict = {}
    for name, f in arms.items():
        rows = _jsonl(f)
        keys = {r["rollout_key"] for r in rows}
        base = [r for r in nrows if r["rollout_key"] in keys]
        pa, pn = T13.p_by_scenario(rows), T13.p_by_scenario(base)
        per: dict[str, dict] = {}
        for tag, rr in (("arm", rows), ("none", base)):
            for r in rr:
                d = per.setdefault(r["scenario_id"], {}).setdefault(tag, [])
                d.append({"len": float(r.get("trace_len") or 0),
                          **AM.features(trace_text(r), int(r.get("trace_len") or 0))})
        sids = sorted(s for s in per
                      if "arm" in per[s] and "none" in per[s] and s in pa and s in pn)
        dlen = np.array([np.log(np.mean([x["len"] for x in per[s]["arm"]]) /
                                np.mean([x["len"] for x in per[s]["none"]])) for s in sids])
        dp = np.array([pa[s] - pn[s] for s in sids])
        lex = {k: np.array([np.mean([x[k] for x in per[s]["arm"]]) -
                            np.mean([x[k] for x in per[s]["none"]]) for s in sids])
               for k in ("weighing", "harm", "person")}
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(sids), (10_000, len(sids)))
        rs = np.array([np.corrcoef(dlen[i], dp[i])[0, 1] for i in idx])
        ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
        out[name] = {"n_scenarios": len(sids),
                     "a_log_length_ratio": {"mean": float(dlen.mean()),
                                            "ci95": ci(dlen[idx].mean(1))},
                     "b_r_length_dP": {"mean": float(np.corrcoef(dlen, dp)[0, 1]), "ci95": ci(rs)},
                     "dP": {"mean": float(dp.mean()), "ci95": ci(dp[idx].mean(1))},
                     "c_lexicons": {k: {"mean": float(v.mean()), "ci95": ci(v[idx].mean(1))}
                                    for k, v in lex.items()}}
    return out


def regen_branch(r: dict) -> str:
    """Amendment 53 item 3, written before data."""
    b15, b18 = r["b15_clause_end"], r.get("b18_clause_end")
    a = b15["a_log_length_ratio"]["ci95"][1] < 0
    b = b15["b_r_length_dP"]["ci95"][0] > 0
    c = b15["c_lexicons"]["weighing"]["ci95"][1] < 0
    b18_ac = bool(b18 and b18["a_log_length_ratio"]["ci95"][1] < 0
                  and b18["c_lexicons"]["weighing"]["ci95"][1] < 0)
    if a and (b or (c and not b18_ac)):
        return "R_regen survives: the 0.3 A100-h follow-up runs (Amendment 53 item 4)"
    return "R_noise stands: b15 reads as a null like b18; ST-A31 closes; no follow-up"


def _f(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gs", type=Path, required=True)
    ap.add_argument("--c1", type=Path, required=True)
    ap.add_argument("--geom", type=Path, default=None, help="T1.1's gpt_oss_20b dir (the bank)")
    ap.add_argument("--inputs", type=Path, default=None, help="the st2gs_inputs dir")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = read(a.gs, a.c1)
    if a.geom:
        res["geometry"] = geometry(a.geom, a.inputs, a.gs / "analogs.npz")
    p = a.out or a.gs / "st2gs_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    rk = res["ranking"]
    for s, r in rk["sites"].items():
        print(f"{s}: exact {_f(r['exact']['mean'])} {r['exact']['ci95']}"
              + (f"; AtP {_f(r['atp']['mean'])}" if "atp" in r else ""))
    print(f">> top2 {rk['top2']}; AtP Spearman {rk['atp_spearman']}; split-half "
          f"{_f(rk['split_half_spearman'])}")
    for s, r in res["sites"].items():
        for rung in ("04", "03"):
            if rung in r:
                g, sp = r[rung]["gate"], r[rung]["specificity"]
                print(f"{s} {rung}: dP vs none {_f(g['dP_vs_none']['mean'])} "
                      f"{g['dP_vs_none']['ci95']}; vs site fill "
                      f"{_f(g['dP_vs_site_fill']['mean'])}; coherent {g['coherent']}; "
                      f"gate {g['passes']}"
                      + (f"; spec {_f(sp['vs_fill10']['mean'])} ({sp['class']})" if sp else ""))
    print(f">> {res['branch']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
