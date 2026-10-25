#!/usr/bin/env python3
"""ST-A27's discriminators (ST_PLAN Amendment 41; zero GPU on saved arrays).

    python3 papers/st_live_reference/scripts/analyze_a2m.py --t16 <outputs/t16/muse_glimmer_30b> \
        --replicate <outputs/st2mr/muse_glimmer_30b> [--a2m <outputs/a2m/muse_glimmer_30b>]

- Trace content (item 2): the 40 paired chosen traces (care-DPO's and base-DPO's chosen rollout for
  one scenario and pair index, rebuilt by ``t16_driver.harm_pairs`` and checked against T1.6's
  per-scenario counts); length in tokens and three fixed lexicons per 100 words; paired care − base,
  scenario bootstrap, Bonferroni over the four (98.75% intervals).
- Likelihood (item 3): R_b's premise, care's chosen ``trace_nll_unsteered`` − base's, paired,
  scenario bootstrap (95%); with cell A′, the scaling test care's ΔP − k × base's ΔP, k from the
  generation of record resampled by scenario, the items resampled independently.
- Muse cell A′ (item 1, when ``--a2m`` is given): ΔP(violating) per arm against ``none``, item
  bootstrap; care-DPO's class (harm / format / unresolved); care − base paired; the map (item 4).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2fx as FX  # noqa: E402
import analyze_st2ml as AML  # noqa: E402
import analyze_t15 as A15  # noqa: E402
import analyze_t16 as AT  # noqa: E402
import t13_driver as T13  # noqa: E402
import t16_driver as T16  # noqa: E402

DATA = HERE.parent / "data"
N_BOOT, SEED = 10_000, 0
SPEC_DROP = -0.054  # care-DPO's specificity of record on the panel-format set (Amendment 41 item 1)
FLOOR_PRED = 0.14  # Amendment 43 item 3: the dose-0 floor's prediction, 5.2 x 0.027
SPEC_BAR = 0.03  # Amendment 31's sign-class bar
LEX = {  # Amendment 41 item 2, fixed before data; case-insensitive, whole words
    "harm": ("harm", "hurt", "damage", "loss", "lose", "risk", "danger", "injury", "suffer",
             "penalty", "evict", "eviction", "cost", "costs"),
    "person": ("he", "she", "they", "him", "her", "them", "his", "hers", "their"),
    "weighing": ("however", "but", "although", "though", "balance", "weigh", "weighing",
                 "outweigh", "outweighs", "tradeoff", "trade-off", "consider", "considering",
                 "versus", "on the other hand"),
}
FEATURES = ("length_tokens", "harm", "person", "weighing")
LEX_RX = {k: re.compile(r"(?<![\w-])(" + "|".join(re.escape(w) for w in v) + r")(?![\w-])", re.I)
          for k, v in LEX.items()}
WORD_RX = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def features(text: str, n_tokens: int) -> dict[str, float]:
    n_words = max(len(WORD_RX.findall(text)), 1)
    out = {"length_tokens": float(n_tokens)}
    for k, rx in LEX_RX.items():
        out[k] = 100.0 * len(rx.findall(text)) / n_words
    return out


def scenario_boot(by_scen: dict[str, list[float]], q: tuple[float, float]) -> dict:
    """Pair-weighted mean of paired differences; scenarios resampled with replacement."""
    sids = sorted(by_scen)
    sums = np.array([sum(by_scen[s]) for s in sids])
    ns = np.array([len(by_scen[s]) for s in sids])
    idx = np.random.default_rng(SEED).integers(0, len(sids), (N_BOOT, len(sids)))
    d = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.percentile(d, q)
    return {"mean": float(sums.sum() / ns.sum()), "ci": [float(lo), float(hi)],
            "ci_level": round(q[1] - q[0], 3), "n_pairs": int(ns.sum()), "n_scenarios": len(sids)}


def pairs(t16: Path, rep: Path) -> tuple[list[tuple[dict, dict]], dict]:
    """The paired chosen rows (care's, base's) in harm_pairs' order, checked against T1.6."""
    care, none = _jsonl(rep / "gen" / "arm_care_03.jsonl"), _jsonl(rep / "gen" / "arm_none.jsonl")
    hp, info = T16.harm_pairs(care, none)
    for arm in ("care", "base"):
        rec = json.loads((t16 / "adapters" / arm / "st_adapter.json").read_text())["pairs"]
        if rec["per_scenario"] != info["per_scenario"]:
            raise SystemExit(f"{arm}: rebuilt pairs differ from the T1.6 record's "
                             "per-scenario counts")
    rows = {("care", r["scenario_id"], r["rollout"]): r for r in care}
    rows.update({("base", r["scenario_id"], r["rollout"]): r for r in none})
    out = []
    for c, b in zip(hp["care"], hp["base"]):
        assert c["scenario_id"] == b["scenario_id"] and c["rejected_key"] == b["rejected_key"]
        out.append((rows[("care", *c["chosen_key"])], rows[("base", *b["chosen_key"])]))
    return out, info


def content_and_likelihood(t16: Path, rep: Path, tok) -> dict:
    pr, info = pairs(t16, rep)
    feats: dict[str, dict[str, list[float]]] = {f: {} for f in FEATURES}
    nll: dict[str, list[float]] = {}
    per_pair = []
    for c, b in pr:
        fc = features(tok.decode(c["gen_ids"], skip_special_tokens=True), len(c["gen_ids"]))
        fb = features(tok.decode(b["gen_ids"], skip_special_tokens=True), len(b["gen_ids"]))
        sid = c["scenario_id"]
        for f in FEATURES:
            feats[f].setdefault(sid, []).append(fc[f] - fb[f])
        dn = float(c["trace_nll_unsteered"]) - float(b["trace_nll_unsteered"])
        nll.setdefault(sid, []).append(dn)
        per_pair.append({"scenario_id": sid, "care_rollout": c["rollout"],
                         "base_rollout": b["rollout"], "care": fc, "base": fb,
                         "nll_care": float(c["trace_nll_unsteered"]),
                         "nll_base": float(b["trace_nll_unsteered"])})
    bonf = (100 * 0.0125 / 2, 100 * (1 - 0.0125 / 2))  # 98.75% intervals, four features
    cont = {f: scenario_boot(feats[f], bonf) for f in FEATURES}
    for f in FEATURES:
        lo, hi = cont[f]["ci"]
        cont[f]["surface_difference"] = bool(lo > 0 or hi < 0)
    prem = scenario_boot(nll, (2.5, 97.5))
    return {"n_pairs": len(pr), "pairs_info": {k: info[k] for k in ("n_pairs",
                                                                     "scenarios_with_pair")},
            "content": cont, "surface_features": [f for f in FEATURES
                                                  if cont[f]["surface_difference"]],
            "likelihood_premise": {**prem, "holds": bool(prem["ci"][0] > 0)},
            "per_pair": per_pair}


def generation_per_scenario(t16: Path) -> dict[str, dict[str, float]]:
    """Per-scenario generation ΔP of record for each arm (analyze_t16's read)."""
    rec = json.loads((t16 / "t16_record.json").read_text())
    eng = rec["phases"]["base_hf"].get("engaged_held") or []
    spec_ids = json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"]
    half = set(json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"])
    return {a: AT.read_arm(t16, rec, a, eng, spec_ids, half)[1] for a in ("care", "base")}


def care_class(ci: list[float]) -> str:
    """Amendment 41 item 1: below 0 harm; excluding SPEC_DROP format; including both unresolved."""
    lo, hi = ci
    if hi < 0:
        return "harm"
    return "format" if lo > SPEC_DROP else "unresolved"


def scaling(cell: dict[str, dict[str, float]], gen: dict[str, dict[str, float]]) -> dict:
    s = sorted(set(gen["care"]) & set(gen["base"]))
    gc, gb = np.array([gen["care"][x] for x in s]), np.array([gen["base"][x] for x in s])
    items = sorted(set(cell["care"]) & set(cell["base"]))
    ac, ab = np.array([cell["care"][i] for i in items]), np.array([cell["base"][i] for i in items])
    k = float(gc.mean() / gb.mean())
    rng = np.random.default_rng(SEED)
    gi = rng.integers(0, len(s), (N_BOOT, len(s)))
    ii = rng.integers(0, len(items), (N_BOOT, len(items)))
    kb = gc[gi].mean(1) / gb[gi].mean(1)
    d = ac[ii].mean(1) - kb * ab[ii].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"k": k, "k_ci95": [float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))],
            "care_minus_k_base": float(ac.mean() - k * ab.mean()), "ci95": [float(lo), float(hi)],
            "consistent_with_scaling": bool(lo <= 0 <= hi), "n_items": len(items),
            "n_scenarios": len(s)}


def cell(a2m: Path) -> dict:
    sel = json.loads((a2m / "cellA2_selection.json").read_text())
    out: dict = {k: sel[k] for k in ("n_pool", "n_in_primary_band", "n_in_band", "band",
                                     "fallback_used", "shortfall", "read")}
    if not sel["read"]:
        return {**out, "reading": f"no read: {sel['n_in_band']} items in band (< 20)"}
    keep = set(sel["kept"])
    pn = {k: v for k, v in sel["p_violating_none"].items() if k in keep}
    out["n_read"], out["P_violating_none"] = len(keep), float(np.mean(list(pn.values())))
    per = {}
    for arm in ("care", "base"):
        pa = FX.p_violating(_jsonl(a2m / "cellA2" / arm / "dl_chat_neutral.jsonl"))
        s = sorted(set(pa) & set(pn))
        per[arm] = {i: pa[i] - pn[i] for i in s}
        out[arm] = {"dP_violating": T13.boot([per[arm][i] for i in s])}
    s = sorted(set(per["care"]) & set(per["base"]))
    out["care_minus_base"] = T13.boot([per["care"][i] - per["base"][i] for i in s])
    out["care_class"] = care_class(out["care"]["dP_violating"]["ci95"])
    out["per_item"] = per
    return out


def fork_sign_classes(c: dict) -> dict:
    """Amendment 43 item 1 (post-hoc, flagged): cell A′ under Amendment 31's sign classes."""
    m = {"care": c["care"]["dP_violating"]["mean"], "base": c["base"]["dP_violating"]["mean"],
         "care_minus_base": c["care_minus_base"]["mean"]}
    return {k: {"mean": v, "class": AML.sign_class(v)} for k, v in m.items()}


def gen_branch(ci: list[float], mean: float) -> str:
    """Amendment 43 item 3, written before data."""
    lo, hi = ci
    if lo > SPEC_BAR:
        return ("rise: the dose-0 floor; the letter read under-registered care-DPO, and in "
                "generation it breaks the rule on rule-right items in its training format")
    if hi < FLOOR_PRED:
        if mean < -SPEC_BAR and hi < 0:
            return ("no rise (drop, discriminating; flagged): care-DPO conditions on harm in "
                    "generation, R_a's direction")
        return (f"no rise ({AML.sign_class(mean)} at the point): care-DPO is specific in its "
                "training format, and the dose-0 read under-registers this adapter's size but not "
                "its sign")
    return "unresolved: the CI includes both +0.03 and the floor's +0.14"


def gen_read(a2g: Path, kept: list[str]) -> dict:
    """Amendment 43 item 2: P(violating) of record per item (T1.5's p_record), ΔP against
    ``none``; an arm that failed parity is not read."""
    rec = json.loads((a2g / "a2g_record.json").read_text())["phases"]
    out: dict = {"base_check": {k: rec.get("gen_none", {}).get(k)
                                for k in ("base_tv", "base_check_ok")}}
    f = lambda arm: a2g / "gen" / f"cellA2_{arm}.jsonl"  # noqa: E731
    if not f("none").exists():
        return {**out, "reading": "no generation read (the base check or the run failed)"}
    pn = A15.p_record(_jsonl(f("none")))
    out["P_violating_none"] = float(np.mean([pn[i] for i in kept if i in pn]))
    per = {}
    for arm in ("care", "base"):
        ph = rec.get(f"gen_{arm}", {})
        out[f"parity_{arm}"] = {k: ph.get(k) for k in ("parity_tv", "band_tv", "parity_ok",
                                                       "reason")}
        if not ph.get("parity_ok") or not f(arm).exists():
            out[arm] = {"read": False}
            continue
        pa = A15.p_record(_jsonl(f(arm)))
        s = [i for i in kept if i in pa and i in pn]
        per[arm] = {i: pa[i] - pn[i] for i in s}
        d = T13.boot([per[arm][i] for i in s])
        out[arm] = {"read": True, "dP_violating": d, "class": AML.sign_class(d["mean"])}
    if "care" in per:
        d = out["care"]["dP_violating"]
        out["care_branch"] = gen_branch(d["ci95"], d["mean"])
    if "care" in per and "base" in per:
        s = sorted(set(per["care"]) & set(per["base"]))
        out["care_minus_base"] = T13.boot([per["care"][i] - per["base"][i] for i in s])
    out["per_item"] = per
    out["action_split"] = action_split(a2g / "gen", kept)
    return out


R_D_PRED = 0.958  # Amendment 46 item 3: care-DPO's share off the rule option on cell A′ (3 options)
ACTIONS = ("consistent", "violating", "neutral", "truncated", "unparsed")


def action_of(r: dict) -> str:
    if r.get("norm_status") in ("consistent", "violating", "neutral"):
        return r["norm_status"]
    return "truncated" if r.get("status") == "truncated" else "unparsed"


def action_split(gen: Path, kept: list[str]) -> dict:
    """Amendment 46 item 4: per arm, each action's share (item means), and its change against
    ``none`` with an item bootstrap; reported beside the outcome of record."""
    def shares(arm: str) -> dict[str, dict[str, float]]:
        by: dict[str, list[str]] = {}
        for r in _jsonl(gen / f"cellA2_{arm}.jsonl"):
            if r["scenario_id"] in kept:
                by.setdefault(r["scenario_id"], []).append(action_of(r))
        return {a: {i: float(np.mean([x == a for x in v])) for i, v in by.items()}
                for a in ACTIONS}
    sh = {arm: shares(arm) for arm in ("none", "care", "base")
          if (gen / f"cellA2_{arm}.jsonl").exists()}
    out: dict = {}
    for arm, d in sh.items():
        out[arm] = {a: float(np.mean(list(v.values()))) if v else None for a, v in d.items()}
        if arm != "none" and "none" in sh:
            out[f"{arm}_minus_none"] = {
                a: T13.boot([d[a][i] - sh["none"][a][i] for i in sorted(set(d[a]) &
                                                                       set(sh["none"][a]))])
                for a in ACTIONS}
    return out


def forced_branch(none_ci: list[float], care_ci: list[float], b: float) -> str:
    """Amendment 46 item 3, written before data."""
    if not none_ci[1] < 0.5:
        return ("uninformative: none does not mostly follow the rule when it cannot defer (its "
                "P(violating) CI reaches 0.5)")
    if care_ci[1] < b:
        return ("follows the rule: a caution preference; the harm condition stands (care-DPO "
                "departs where the rule harms and follows where breaking it would, when it must "
                "choose)")
    if care_ci[0] > b:
        return ("breaks: R_d; the specificity seen with three options was routing of rule "
                "avoidance to deferral")
    return "unresolved: the CI includes the bar (partial avoidance)"


def forced_read(a2f: Path) -> dict:
    """Amendment 46 items 2-4 on the forced-choice outputs."""
    rec = json.loads((a2f / "a2g_record.json").read_text())["phases"]
    gen = a2f / "gen"
    out: dict = {"base_check": {k: rec.get("gen_none", {}).get(k)
                                for k in ("base_tv", "base_check_ok")}}
    if not (gen / "cellA2_none.jsonl").exists():
        return {**out, "reading": "no forced-choice read (the base check or the run failed)"}
    p = {"none": A15.p_record(_jsonl(gen / "cellA2_none.jsonl"))}
    kept = sorted(p["none"])
    for arm in ("care", "base"):
        ph = rec.get(f"gen_{arm}", {})
        out[f"parity_{arm}"] = {k: ph.get(k) for k in ("parity_tv", "band_tv", "parity_ok",
                                                       "reason")}
        if ph.get("parity_ok") and (gen / f"cellA2_{arm}.jsonl").exists():
            p[arm] = A15.p_record(_jsonl(gen / f"cellA2_{arm}.jsonl"))
    for arm, d in p.items():
        out[arm] = {"P_violating": T13.boot([d[i] for i in kept if i in d])}
    for arm in ("care", "base"):
        if arm in p:
            out[f"{arm}_minus_none"] = T13.boot([p[arm][i] - p["none"][i] for i in kept
                                                 if i in p[arm]])
    if "care" in p and "base" in p:
        out["care_minus_base"] = T13.boot([p["care"][i] - p["base"][i] for i in kept
                                           if i in p["care"] and i in p["base"]])
    if "care" in p:
        b = (out["none"]["P_violating"]["mean"] + R_D_PRED) / 2
        out["bar"] = b
        out["care_branch"] = forced_branch(out["none"]["P_violating"]["ci95"],
                                           out["care"]["P_violating"]["ci95"], b)
    out["action_split"] = action_split(gen, kept)
    return out


def mapping(c: dict | None, cl: dict, rb: dict) -> str:
    """Amendment 41 item 4, written before data."""
    if c is None or not c.get("read"):
        return "cell A′ not read: R_a and R_c stay open; " + rb["status"]
    beside = f"; R_b {rb['status']}"
    if c["care_class"] == "harm":
        if c["care_minus_base"]["ci95"][1] < 0:
            if rb["status"].startswith("closed"):
                return "R_a: care-DPO conditions on harm in H1's format beyond base-DPO" + beside
            return "unregistered combination (harm, paired below 0, R_b open): ST-A27 first"
        return ("shared: both teachers condition on harm in H1's format; the teacher's difference "
                "is not a harm condition" + beside)
    if c["care_class"] == "format":
        carrier = ", ".join(cl["surface_features"]) or "carrier unnamed"
        return f"R_c over R_a: format ({carrier})" + beside
    return f"unresolved at n = {c['n_read']}: R_a and R_c stay open" + beside


def rb_status(cl: dict, sc: dict | None) -> dict:
    """Amendment 41 item 3: closed if the premise fails or care is not a scaled base on cell A′."""
    if not cl["likelihood_premise"]["holds"]:
        return {"status": "closed (premise: the steered chosen traces are not less likely)"}
    if sc is None:
        return {"status": "open pending cell A′ (premise holds)"}
    if not sc["consistent_with_scaling"]:
        return {"status": "closed (care-DPO is not a scaled base-DPO on cell A′)"}
    return {"status": "open (premise holds; cell A′ consistent with scaling)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t16", type=Path, required=True)
    ap.add_argument("--replicate", type=Path, required=True)
    ap.add_argument("--a2m", type=Path, default=None)
    ap.add_argument("--a2g", type=Path, default=None,
                    help="Amendment 43's generation outputs (a2g's muse_glimmer_30b dir)")
    ap.add_argument("--a2f", type=Path, default=None,
                    help="Amendment 46's forced-choice outputs (a2f's muse_glimmer_30b dir)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import pod_st_screen as P
    from transformers import AutoTokenizer

    spec = P.registry()[T16.KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    cl = content_and_likelihood(a.t16, a.replicate, tok)
    c = cell(a.a2m) if a.a2m else None
    sc = None
    if c is not None and c.get("read"):
        sc = scaling(c["per_item"], generation_per_scenario(a.t16))
    rb = rb_status(cl, sc)
    fork = fork_sign_classes(c) if c is not None and c.get("read") else None
    g = gen_read(a.a2g, sorted(c["per_item"]["care"])) if a.a2g and fork else None
    res = {"rule": "ST_PLAN Amendments 41 and 43", "content_likelihood": cl, "cellA2_muse": c,
           "scaling": sc, "R_b": rb, "reading": mapping(c, cl, rb),
           "fork_sign_classes": fork, "generation": g,
           "forced_choice": forced_read(a.a2f) if a.a2f else None,
           "matched_cap": "both arms 243 steps (about 24 epochs over 40 rule-harm pairs), equally "
                          "overfit; matched, does not create the teacher's difference"}
    p = a.out or (a.a2m or a.t16) / "a27_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    for f, v in cl["content"].items():
        print(f"{f}: care − base {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] "
              f"(98.75%){' surface' if v['surface_difference'] else ''}")
    lp = cl["likelihood_premise"]
    print(f"NLL care − base {lp['mean']:+.4f} [{lp['ci'][0]:+.4f}, {lp['ci'][1]:+.4f}]; premise "
          f"{'holds' if lp['holds'] else 'fails'}")
    if c is not None and c.get("read"):
        print(f"cell A′ n {c['n_read']} (band {c['band']}): care {c['care']['dP_violating']}; "
              f"base {c['base']['dP_violating']}; care − base {c['care_minus_base']}; "
              f"class {c['care_class']}; scaling {sc}")
    if fork:
        print(f">> fork (Amendment 43 item 1, sign classes): {fork}")
    if g:
        print(f">> generation: none {g.get('P_violating_none')}; care {g.get('care')}; base "
              f"{g.get('base', {}).get('dP_violating')}; care - base {g.get('care_minus_base')}\n"
              f">> generation branch: {g.get('care_branch')}")
    fc = res["forced_choice"]
    if fc and "care_branch" in fc:
        print(f">> forced choice: none {fc['none']['P_violating']}; care "
              f"{fc['care']['P_violating']}; base {fc.get('base', {}).get('P_violating')}; "
              f"bar {fc['bar']:.3f}\n"
              f">> forced-choice branch: {fc['care_branch']}")
    print(f">> R_b {rb['status']}\n>> registered map: {res['reading']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
