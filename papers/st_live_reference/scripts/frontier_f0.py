#!/usr/bin/env python3
"""The frontier exhibit's F0 reads (ST_PLAN Amendment 56 items 3 and 5). Zero GPU, zero API.

    python3 papers/st_live_reference/scripts/frontier_f0.py --out <outputs/frontier/f0>

- **Readout validation** (item 3):
  - (a) on GPT-OSS's T1.1 rows (264 x 8, post-reasoning option log-probs), the continuous readout
    per scenario (the mean over rollouts of P(protect) renormalized over the shown letters) against
    the sampled protect share at k = 4 and 8 (50 subsamples, seed 0): the error of the mean over
    the 60 screen scenarios (95% band) and the engaged-classification agreement (P <= 0.5);
  - (b) on the 16-rollout ``none`` rows (GPT-OSS Cell 1, Muse's replicate; 45 x 16 each), the
    sampled share at k = 4 and 8 against the 16-rollout share, the 45-scenario band scaled to 60 by
    sqrt(45 / 60) (implementing reading);
  - the rule: the smallest k in {4, 8, 16} meeting +-0.03 in (a) and (b) and >= 90% agreement;
    else 16; B's MDE at n = 60 and that k from the measured per-scenario variance.
- **Price table** (item 5): per call per arm per model from the published rates, the prompt length
  of the 60 screen renders (characters / 4, an approximation) and output lengths from the open
  models' reasoned rows (GPT-OSS T1.1 at effort medium: median as central, 2 x q90 as high; the
  minimal arm at 40% of the default arm's, an assumption F1 measures); what $100 buys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts"), str(HERE.parents[2])]
import frontier_client as FC  # noqa: E402

OUT = HERE.parent / "outputs"
KS, DRAWS, SEED, BAND, AGREE = (4, 8, 16), 50, 0, 0.03, 0.90
MINIMAL_SHARE = 0.4  # the minimal arm's output length as a share of the default arm's (assumed)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def screen_ids() -> list[str]:
    return sorted({j["scenario"].id for j in FC.screen_jobs(1)})


def p_protect_continuous(r: dict, roles: dict[str, str]) -> float | None:
    """P(protect) at the answer position, renormalized over the shown letters."""
    lp = r.get("option_logps")
    if not lp:
        return None
    viol = [L for L, oid in r["order"].items() if roles.get(oid) == "violating"]
    if not viol:
        return None
    m = {L: np.exp(v) for L, v in lp.items()}
    z = sum(m.values())
    return float(m[viol[0]] / z) if z > 0 else None


def roles_of() -> dict[str, dict[str, str]]:
    import pod_st_screen as P

    from deepsteer.kdg.schema import load_scenario_dir

    return {s.id: {o.option_id: o.norm_status for o in s.options}
            for s in load_scenario_dir([P.H1_SET])[0]}


def _sampled(rows: list[dict]) -> float | None:
    acts = [r["norm_status"] for r in rows if r.get("norm_status") is not None]
    return float(np.mean([a == "violating" for a in acts])) if acts else None


def subsample_read(by: dict[str, list[dict]], ref: dict[str, float], k: int,
                   scale: float = 1.0) -> dict:
    """k-rollout sampled shares against ``ref`` over the scenarios: the error of the mean (95%
    band, scaled by ``scale``) and the engaged-classification agreement, over 50 subsamples."""
    rng = np.random.default_rng(SEED)
    sids = sorted(s for s in by if s in ref and len(by[s]) >= k)
    errs, agree, sds = [], [], []
    for _ in range(DRAWS):
        share = {}
        for s in sids:
            pick = [by[s][i] for i in rng.choice(len(by[s]), k, replace=False)]
            v = _sampled(pick)
            if v is not None:
                share[s] = v
        ss = sorted(share)
        errs.append(np.mean([share[s] for s in ss]) - np.mean([ref[s] for s in ss]))
        agree.append(np.mean([(share[s] <= 0.5) == (ref[s] <= 0.5) for s in ss]))
        sds.append(np.std([share[s] - ref[s] for s in ss], ddof=1))
    lo, hi = np.percentile(errs, [2.5, 97.5])
    return {"k": k, "n_scenarios": len(sids), "mean_error": float(np.mean(errs)),
            "band95": [float(lo * scale), float(hi * scale)], "scale": scale,
            "agreement": float(np.mean(agree)), "per_scenario_sd": float(np.mean(sds))}


def validation() -> dict:
    roles, screen = roles_of(), set(screen_ids())
    t11 = _jsonl(OUT / "t11" / "gpt_oss_20b" / "gen" / "h1_rollouts_read.jsonl")
    by_a: dict[str, list[dict]] = {}
    cont: dict[str, list[float]] = {}
    for r in t11:
        if r["scenario_id"] not in screen:
            continue
        by_a.setdefault(r["scenario_id"], []).append(r)
        p = p_protect_continuous(r, roles[r["scenario_id"]])
        if p is not None:
            cont.setdefault(r["scenario_id"], []).append(p)
    ref_a = {s: float(np.mean(v)) for s, v in cont.items()}
    a = {k: subsample_read(by_a, ref_a, k) for k in (4, 8)}
    b = {}
    for name, f in (("gpt_oss_cell1_none", OUT / "st2a" / "gpt_oss_20b" / "gen" / "c1_none.jsonl"),
                    ("muse_replicate_none",
                     OUT / "st2mr" / "muse_glimmer_30b" / "gen" / "arm_none.jsonl")):
        by: dict[str, list[dict]] = {}
        for r in _jsonl(f):
            by.setdefault(r["scenario_id"], []).append(r)
        ref = {s: v for s, v in ((s, _sampled(rr)) for s, rr in by.items()) if v is not None}
        scale = float(np.sqrt(len(ref) / 60.0))
        b[name] = {k: subsample_read(by, ref, k, scale) for k in (4, 8)}

    def ok(x: dict) -> bool:
        return (abs(x["band95"][0]) <= BAND and abs(x["band95"][1]) <= BAND
                and x["agreement"] >= AGREE)
    chosen = next((k for k in (4, 8) if ok(a[k]) and all(ok(v[k]) for v in b.values())), 16)
    sd = (max(v[chosen]["per_scenario_sd"] for v in b.values()) if chosen in (4, 8)
          else max(v[8]["per_scenario_sd"] for v in b.values()) / np.sqrt(2))
    mde = float(2.8 * np.sqrt(2) * sd / np.sqrt(60))
    return {"rule": "ST_PLAN Amendment 56 item 3", "a_continuous_vs_sampled": a,
            "b_vs_16": b, "chosen_k": chosen, "mde_B_at_60": mde,
            "mde_note": "2.8 x sqrt(2) x the per-scenario SD of a k-rollout share / sqrt(60); "
                        "the two arms treated as independent (no pairing credit); at k = 16 the "
                        "SD is extrapolated from k = 8 by 1/sqrt(2)",
            "n_screen_in_t11": len(ref_a)}


def price_table(k: int) -> dict:
    t11 = _jsonl(OUT / "t11" / "gpt_oss_20b" / "gen" / "h1_rollouts_read.jsonl")
    gen = np.array([r["n_gen_tokens"] for r in t11 if r.get("n_gen_tokens")])
    chars = [len(j["messages"][0]["content"]) for j in FC.screen_jobs(1) if j["arm"] == "default"]
    prompt = {"central": float(np.median(chars)) / 4, "high": float(np.max(chars)) / 4 * 1.35}
    out_default = {"central": float(np.median(gen)), "high": float(2 * np.percentile(gen, 90))}
    out = {"inputs": {"prompt_tokens": prompt, "default_output_tokens": out_default,
                      "minimal_share": MINIMAL_SHARE,
                      "source": "prompt: the 60 screen renders' characters / 4 (high x 1.35 for "
                                "Claude's tokenizer); output: GPT-OSS T1.1 at effort medium"},
           "models": {}}
    for m, rate in FC.RATES.items():
        per = {}
        for case in ("central", "high"):
            call = {}
            for arm, share in (("default", 1.0), ("minimal", MINIMAL_SHARE)):
                call[arm] = (prompt[case] * rate["input"]
                             + out_default[case] * share * rate["output"]) / 1e6
            pair = call["default"] + call["minimal"]
            per[case] = {"per_call_usd": call, "per_scenario_rollout_pair_usd": pair,
                         "f1_60_scenarios_at_k_usd": pair * 60 * k,
                         "pairs_per_100_usd": int(100 // pair),
                         "k_at_60_scenarios_per_100_usd": float(100 / pair / 60)}
        out["models"][m] = {"rates_per_million": rate, **per}
    out["cheaper_per_call"] = min(out["models"],
                                  key=lambda m: out["models"][m]["central"]
                                  ["per_scenario_rollout_pair_usd"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT / "frontier" / "f0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    v = validation()
    (a.out / "readout_validation.json").write_text(json.dumps(v, indent=1) + "\n")
    p = price_table(v["chosen_k"])
    (a.out / "price_table.json").write_text(json.dumps(p, indent=1) + "\n")
    for k, x in v["a_continuous_vs_sampled"].items():
        print(f"(a) k={k}: band {x['band95']}, agreement {x['agreement']:.3f}, "
              f"n {x['n_scenarios']}")
    for name, d in v["b_vs_16"].items():
        for k, x in d.items():
            print(f"(b) {name} k={k}: band (to 60) {x['band95']}, agreement "
                  f"{x['agreement']:.3f}, sd {x['per_scenario_sd']:.3f}")
    print(f">> chosen k = {v['chosen_k']}; B's MDE at 60 = {v['mde_B_at_60']:.3f}")
    for m, d in p["models"].items():
        print(f"{m}: per pair central ${d['central']['per_scenario_rollout_pair_usd']:.4f}, high "
              f"${d['high']['per_scenario_rollout_pair_usd']:.4f}; F1 at k: central "
              f"${d['central']['f1_60_scenarios_at_k_usd']:.2f}, high "
              f"${d['high']['f1_60_scenarios_at_k_usd']:.2f}")
    print(f">> cheaper per call: {p['cheaper_per_call']}")


if __name__ == "__main__":
    main()
