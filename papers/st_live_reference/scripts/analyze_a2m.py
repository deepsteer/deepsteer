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
import analyze_t16 as AT  # noqa: E402
import t13_driver as T13  # noqa: E402
import t16_driver as T16  # noqa: E402

DATA = HERE.parent / "data"
N_BOOT, SEED = 10_000, 0
SPEC_DROP = -0.054  # care-DPO's specificity of record on the panel-format set (Amendment 41 item 1)
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
    res = {"rule": "ST_PLAN Amendment 41", "content_likelihood": cl, "cellA2_muse": c,
           "scaling": sc, "R_b": rb, "reading": mapping(c, cl, rb),
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
    print(f">> R_b {rb['status']}\n>> {res['reading']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
