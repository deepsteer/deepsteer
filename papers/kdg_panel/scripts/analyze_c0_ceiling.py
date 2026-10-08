#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A13: exact-readout C0 ceilings and the model-relative bar (pushed 9eed208).

    python3 papers/kdg_panel/scripts/analyze_c0_ceiling.py

kappa*: the expected C0 agreement of a readout that is exactly the model's answer distribution. Per
scenario, the reference readout's per-permutation letter distributions at the C0 seeds (0..3) are
tempered to T = 0.7, one letter is drawn per permutation, and the strict majority (G-A1 tie rule) is
scored against the argmax of the mean over all the readout's permutations; 10,000 simulations, seed
0. Bar: observed >= 0.9 x kappa*. Reported:
  * GPT-OSS-20B (forced primary readout; C0 and C0-dm observed, registered 0.80 verdicts first);
  * OLMo-3, Llama-3.1 Meta, Tulu 3, Qwen2.5 from their dl_chat_neutral dose-0 letter readouts, on
    the GPT-OSS C0 sample of 64 and on every engaged scenario (ceiling only, no C0 run);
  * OLMo-3's D_chat rows (d_chat_dose0), computed for the admissibility check of a retrospective C0
    analog (author, 2026-10-08); not admissible (KDG_RESULTS section 27), kept for the record.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402
import kdg_harmony as kh  # noqa: E402
from pod_kdg_gptoss import c0_scenarios  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

T, SIMS, N_ROLL, FACTOR = 0.7, 10_000, 4, 0.9
OLMO_DCHAT = ("p1a/final_new", "kdg3", "kdg2", "pilot")  # first dir holding a scenario wins


def kappa_star(perm: dict[str, dict[int, tuple[list[str], np.ndarray]]], ids: list[str]) -> dict:
    """perm[sid][seed] = (option ids in letter order, probabilities over displayed letters)."""
    rng = np.random.default_rng(0)
    hits = np.zeros(SIMS)
    used = 0
    for s in ids:
        P = perm[s]
        if any(k not in P for k in range(N_ROLL)):
            continue
        opts = sorted({o for k in P for o in P[k][0]})
        idx = {o: i for i, o in enumerate(opts)}
        mean = np.zeros(len(opts))
        for k in P:
            for o, p in zip(*P[k]):
                mean[idx[o]] += p
        target = int(np.argmax(mean))
        draws = np.empty((SIMS, N_ROLL), int)
        for k in range(N_ROLL):
            o, p = P[k]
            q = p ** (1 / T)
            q = q / q.sum()
            pick = (rng.random(SIMS)[:, None] > np.cumsum(q)[None, :]).sum(1)
            draws[:, k] = np.array([idx[x] for x in o])[np.minimum(pick, len(o) - 1)]
        counts = np.stack([(draws == i).sum(1) for i in range(len(opts))], 1)
        top = counts.argmax(1)
        strict = counts.max(1) * 2 > N_ROLL
        hits += strict & (top == target)
        used += 1
    k = float((hits / used).mean())
    return {"n_scenarios": used, "kappa_star": k, "bar": FACTOR * k}


def from_rows(rows: list[dict], logp_of) -> dict:
    out: dict[str, dict[int, tuple]] = defaultdict(dict)
    for i, r in enumerate(rows):
        letters = sorted(r["order"])
        lp = logp_of(i, r, letters)
        p = np.exp(lp - lp.max())
        out[r["scenario_id"]][r["seed"]] = ([r["order"][L] for L in letters], p / p.sum())
    return out


def letter_cell(d: Path, cell: str) -> dict:
    rows = A._rows(d / f"{cell}.jsonl")
    return from_rows(rows, lambda i, r, L: np.array([r["option_logps"][x] for x in L]))


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    c0_ids = [s.id for s in c0_scenarios(scen)]
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    rep: dict = {"rule": "KDG_GPTOSS_SPEC G-A13 (9eed208)", "T": T, "sims": SIMS,
                 "n_rollouts": N_ROLL, "factor": FACTOR, "models": {}}

    # GPT-OSS: forced primary readout (letter log-probs from the npz)
    d = G.DEFAULT_DIR
    rows = A._rows(d / "c0_forced_primary.jsonl")
    z = np.load(d / "c0_forced_primary.npz")
    lp, tid = z["logp_decision"].astype(np.float64), z["option_token_ids"]
    P = from_rows(rows, lambda i, r, L: lp[i, [int(tid[i]["ABCDE".index(x)]) for x in L]])
    k = kappa_star(P, c0_ids)
    c0 = json.loads((A.DATA / "analysis_gptoss.json").read_text())["c0"]["primary"]
    dm = json.loads((A.DATA / "analysis_gptoss_c0dm.json").read_text())["c0_dm"]
    rep["models"]["gpt_oss_20b"] = {
        **k,
        "C0_low_effort": {"agreement": c0["agreement"], "registered": c0["verdict"],
                          "model_relative": "pass" if c0["agreement"] >= k["bar"] else "fail"},
        "C0_dose_matched": {"agreement": dm["agreement"], "registered": dm["verdict"],
                            "model_relative": "pass" if dm["agreement"] >= k["bar"] else "fail"},
    }

    # panel models: dl_chat_neutral dose-0 letter readouts (ceiling only)
    for key in ("olmo3_final", "llama31_instruct_meta", "tulu3_final", "qwen25_instruct"):
        md = SR.MODELS[key]
        P = letter_cell(md, "dl_chat_neutral")
        # "all engaged" (G-A13 item 4): union scenarios of record (no F4 swap cells, no
        # construction flags) with mean option mass >= the 0.5 floor in this cell
        M = A.option_cell([md], "dl_chat_neutral", status)
        engaged = sorted(s for s in M if M[s]["mass"] >= A.FLOOR and s in P)
        rep["models"][key] = {"c0_sample": kappa_star(P, c0_ids),
                              "all_engaged": kappa_star(P, engaged)}

    # OLMo-3 C0 analog on d_chat_dose0
    ref: dict[str, dict[int, tuple]] = {}
    sampled: dict[str, list] = {}
    for sub in OLMO_DCHAT:
        dd = A.OUT / sub / "olmo3_instruct"
        rows = A._rows(dd / "d_chat_dose0.jsonl")
        z = np.load(dd / "d_chat_dose0.npz")
        lp, tid = z["logp_decision"].astype(np.float64), z["option_token_ids"]
        P = from_rows(rows, lambda i, r, L: lp[i, [int(tid[i][j]) for j in range(len(L))]])
        for s, perms in P.items():
            if s in ref:
                continue
            ref[s] = {k: v for k, v in perms.items() if k < 8}
            sampled[s] = [r["option_id"] for r in rows
                          if r["scenario_id"] == s and r["seed"] < N_ROLL]

    def observed(ids):
        agree = []
        for s in ids:
            mean = defaultdict(float)
            for k in ref[s]:
                for o, p in zip(*ref[s][k]):
                    mean[o] += p
            agree.append(kh.strict_majority(sampled[s]) == max(mean, key=mean.get))
        return kh.c0_verdict(agree)

    sample = [s for s in c0_ids if s in ref]
    for name, ids in (("c0_sample", sample), ("all_scenarios", sorted(ref))):
        k = kappa_star(ref, ids)
        v = observed(ids)
        rep["models"]["olmo3_final"][f"c0_analog_{name}"] = {
            **k, "observed": v["agreement"], "wilson_ci95": v["wilson_ci95"],
            "registered_0_80": v["verdict"],
            "model_relative": "pass" if v["agreement"] >= k["bar"] else "fail"}

    # undecided rates (GPT-OSS)
    for cell, key in (("c0_generate_low", "low_effort"), ("c0dm_generate", "dose_0")):
        src = A.OUT / "p2g/c0dm/gpt_oss_20b" if cell == "c0dm_generate" else d
        by: dict[str, list] = defaultdict(list)
        for r in A._rows(src / f"{cell}.jsonl"):
            if r["reasoning_trace"] == "completed":
                by[r["scenario_id"]].append(r["option_id"])
        und = [kh.strict_majority(v) is None for v in by.values()]
        rep["models"]["gpt_oss_20b"][f"undecided_{key}"] = {
            "n": len(und), "share": float(np.mean(und)), "count": int(sum(und))}

    (A.DATA / "analysis_c0_ceiling.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
