#!/usr/bin/env python3
"""Session B lineage analyses (KDG_PHASE1_SPEC.md §4 C4/C4'/C3' + P1-A4), written and committed
before the Session B lineage arrays were read.

    python3 papers/kdg_panel/scripts/analyze_phase1_session_b.py

C4 / C4'  per lineage (Llama-3.1 Meta, Qwen2.5): base raw E; instruct raw E beside chat E
          (P1-A4: no raw-only instruct finding); the pre-registered A17-rule outcome, raw frame,
          reported as registered and labelled (raw frame invalid for templated models, KDG-40); the
          raw - chat at-rest gap per instruct model (the P1-A3 bridge quantity, descriptive here,
          testing whether KDG-39 holds on other lineages).
C3'       Tulu 3 stages (SFT -> DPO -> final): the pre-registered raw C3 read (n_shared gate, E_norm
          steps) as registered, labelled; and the template-valid profile on a model-free set (all
          union scenarios clearing the chat floor on the three Tulu stages, the P1-A7 construction):
          E, g_null and lambda (per unit of output scale) per stage and step. Descriptive.
Bootstrap 10,000, seed 0 (analyze_phase1_session_a helpers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

P1B = A.OUT / "p1b"
RAW = ("d_raw", "j_raw", "d_raw_pressure_removed", "j_raw_pressure_removed")
CHAT = (
    "dl_chat_neutral",
    "jl_chat_neutral",
    "dl_chat_neutral_pressure_removed",
    "jl_chat_neutral_pressure_removed",
)


def cells(key: str, kind: str, status) -> dict:
    d = P1B / ("raw_lineages" if kind == "raw" else "chat_lineages") / key
    return A.four_cells([d], RAW if kind == "raw" else CHAT, status)


def ok(T: dict, s: str) -> bool:
    return s in T and T[s]["mass_min"] >= A.FLOOR


def block(T: dict, ids: list[str]) -> dict:
    S = [A.scales(T[s]) for s in ids]
    return {
        q: A.boot(np.array([x[q] for x in S])) for q in ("E_prob", "E_logit", "E_norm", "g_null")
    }


def lam(r: dict) -> float:
    a, j = A.logit(r["pDn"]), A.logit(r["pJn"])
    return float((a - j) / (0.5 * (r["sD"] + r["sJ"])))


def lineage(status, base: str, inst: str) -> dict:
    B, Ir, Ic = cells(base, "raw", status), cells(inst, "raw", status), cells(inst, "chat", status)
    b_ids = sorted(s for s in B if ok(B, s))
    ir_ids = sorted(s for s in Ir if ok(Ir, s))
    ic_ids = sorted(s for s in Ic if ok(Ic, s))
    shared = sorted(set(b_ids) & set(ir_ids))
    d_e = A.boot(np.array([A.scales(Ir[s])["E_prob"] - A.scales(B[s])["E_prob"] for s in shared]))
    null_i = A.boot(np.array([A.scales(Ir[s])["g_null"] for s in shared]))
    e_b = A.boot(np.array([A.scales(B[s])["E_prob"] for s in b_ids]))
    # pre-registered A17-style reading on the raw frame, as registered (labelled void for templated)
    widened = (
        A.sign_of(e_b["ci95"]) > 0 and A.sign_of(d_e["ci95"]) > 0 and A.sign_of(null_i["ci95"]) < 0
    )
    rb = sorted(set(ir_ids) & set(ic_ids))
    bridge = A.boot(np.array([A.scales(Ir[s])["g_null"] - A.scales(Ic[s])["g_null"] for s in rb]))
    return {
        "base_raw": block(B, b_ids),
        "instruct_raw": block(Ir, ir_ids),
        "instruct_chat": block(Ic, ic_ids),
        "raw_shared": {
            "n": len(shared),
            "delta_E_instruct_minus_base": d_e,
            "instruct_raw_g_null_shared": null_i,
        },
        "c4_rule_raw_as_registered": {
            "generalizes_letter": bool(widened),
            "base_E_positive": A.sign_of(e_b["ci95"]) > 0,
            "label": "raw-frame rule as registered; raw invalid for templated models (KDG-40)",
        },
        "raw_minus_chat_g_null_instruct": bridge,
        "raw_minus_chat_n": len(rb),
    }


def tulu(status) -> dict:
    stages = (
        ("base", "llama31_base", "raw"),
        ("sft", "tulu3_sft", None),
        ("dpo", "tulu3_dpo", None),
        ("final", "tulu3_final", None),
    )
    R = {st: cells(k, "raw", status) for st, k, _ in stages}
    shared = sorted(set.intersection(*[{s for s in R[st] if ok(R[st], s)} for st in R]))
    gate = (
        "primary"
        if len(shared) >= 220
        else ("exploratory" if len(shared) >= 150 else "descriptive_only")
    )
    Sr = {st: [A.scales(R[st][s]) for s in shared] for st in R}
    steps_raw = {
        n: A.boot(np.array([y["E_norm"] - x["E_norm"] for x, y in zip(Sr[a], Sr[b])]))
        for a, b, n in (("base", "sft", "SFT"), ("sft", "dpo", "DPO"), ("dpo", "final", "RL"))
    }
    C = {st: cells(k, "chat", status) for st, k, _ in stages[1:]}
    ids = sorted(s for s in status if all(ok(C[st], s) for st in C))
    out = {
        "raw_registered": {
            "n_shared": len(shared),
            "gate": gate,
            "steps_E_norm": steps_raw,
            "label": "raw C3' as registered; raw invalid for templated stages",
        },
        "chat_model_free": {"n": len(ids), "per_stage": {}, "steps": {}},
    }
    V = {st: [(A.scales(C[st][s]), lam(C[st][s])) for s in ids] for st in C}

    def pick(t, q):
        return t[1] if q == "lam" else t[0][q]

    for q in ("E_prob", "g_null", "lam"):
        out["chat_model_free"]["per_stage"][q] = {
            st: A.boot(np.array([pick(t, q) for t in V[st]])) for st in V
        }
        out["chat_model_free"]["steps"][q] = {
            n: A.boot(np.array([pick(y, q) - pick(x, q) for x, y in zip(V[a], V[b])]))
            for a, b, n in (("sft", "dpo", "DPO"), ("dpo", "final", "RL"))
        }
    return out


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }
    rep = {
        "spec": "KDG_PHASE1_SPEC.md §4 C4/C4'/C3' + P1-A4; code committed before arrays were read",
        "n_boot": A.N_BOOT,
        "seed": A.SEED,
        "floor": A.FLOOR,
        "llama31": lineage(status, "llama31_base", "llama31_instruct_meta"),
        "qwen25": lineage(status, "qwen25_base", "qwen25_instruct_p1"),
        "tulu3": tulu(status),
    }
    (A.DATA / "analysis_phase1_session_b.json").write_text(json.dumps(rep, indent=1))

    def f(x):
        if x["mean"] is None:
            return "n/a"
        return "%.3f [%.3f, %.3f] n%d" % (x["mean"], *x["ci95"], x["n"])

    for L in ("llama31", "qwen25"):
        r = rep[L]
        print(
            L,
            "base raw E",
            f(r["base_raw"]["E_prob"]),
            "| inst raw E",
            f(r["instruct_raw"]["E_prob"]),
            "| inst chat E",
            f(r["instruct_chat"]["E_prob"]),
            "| inst chat g_null",
            f(r["instruct_chat"]["g_null"]),
        )
        print(
            "   raw-chat g_null (instruct)",
            f(r["raw_minus_chat_g_null_instruct"]),
            "| C4 raw rule as registered:",
            r["c4_rule_raw_as_registered"],
        )
    t = rep["tulu3"]
    print(
        "tulu raw C3'",
        t["raw_registered"]["n_shared"],
        t["raw_registered"]["gate"],
        {k: f(v) for k, v in t["raw_registered"]["steps_E_norm"].items()},
    )
    for q in ("E_prob", "g_null", "lam"):
        print(
            "tulu chat",
            q,
            {k: f(v) for k, v in t["chat_model_free"]["per_stage"][q].items()},
            {k: f(v) for k, v in t["chat_model_free"]["steps"][q].items()},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
