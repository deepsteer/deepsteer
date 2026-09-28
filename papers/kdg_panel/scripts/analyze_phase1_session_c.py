#!/usr/bin/env python3
"""Session C analyses (KDG_PHASE1_SPEC.md P1-A9, pushed eda98b0), committed before the data exists.

    python3 papers/kdg_panel/scripts/analyze_phase1_session_c.py

1. Known-gap positive control per instruct model: g_band = p_D(known-gap, letter-only acting frame)
   - p_J(letter-only neutral judgment) over union primaries clearing the 0.5 floor in both cells;
   validated iff the 95% CI lower bound >= 0.10 (else "instrument not validated on this model").
3. Llama-3.1 Meta dose arm: delta_dose = p_D,forced(dose2) - p_D,forced(filler) and delta_TF =
   p_D,forced(dose2) - p_D(TF from its own rollouts), paired by scenario on its own screen;
   "reduces" iff both CIs lie entirely below 0, else "scoped to OLMo-3".
Letter-token ids for the dose readout come from the tokenizer of the model read (Llama-3.1 via the
public Tulu 3 tokenizer, asserted equal to the ids the Llama cells saved), both forms summed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

P1A, P1B, P1C = A.OUT / "p1a", A.OUT / "p1b", A.OUT / "p1c"
MODELS = {
    "olmo3_instruct": P1A / "final_c1" / "olmo3_instruct",
    "llama31_instruct_meta": P1B / "chat_lineages" / "llama31_instruct_meta",
    "tulu3_final": P1B / "chat_lineages" / "tulu3_final",
    "qwen25_instruct_p1": P1B / "chat_lineages" / "qwen25_instruct_p1",
}


def llama_letter_ids(dose_dir: Path) -> tuple[dict, dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B")
    chat = {L: tok.encode(L, add_special_tokens=False) for L in "ABCD"}
    sp = {L: tok.encode(" " + L, add_special_tokens=False) for L in "ABCD"}
    assert all(len(v) == 1 for v in list(chat.values()) + list(sp.values()))
    chat, sp = {k: v[0] for k, v in chat.items()}, {k: v[0] for k, v in sp.items()}
    # "the letter ids used here are the ones the Llama cells saved": fail loudly otherwise
    saved = np.load(dose_dir / "d_chat_dose2_bf.npz")["option_token_ids"]
    row = json.loads((dose_dir / "d_chat_dose2_bf.jsonl").read_text().splitlines()[0])
    letters = sorted(row["order"])
    assert [chat[L] for L in letters] == [int(i) for i in saved[0][: len(letters)]], (
        "tokenizer drift"
    )
    return chat, sp


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }
    rep: dict = {
        "amendment": "P1-A9 (eda98b0)",
        "n_boot": A.N_BOOT,
        "seed": A.SEED,
        "known_gap": {},
        "llama_dose": None,
    }
    for key, jdir in MODELS.items():
        K = A.option_cell([P1C / "known_gap" / key], "dl_chat_known_gap", status)
        J = A.option_cell([jdir], "jl_chat_neutral", status)
        ids = sorted(s for s in K if s in J and min(K[s]["mass"], J[s]["mass"]) >= A.FLOOR)
        g = A.boot(np.array([K[s]["p"] - J[s]["p"] for s in ids]))
        lo = g["ci95"][0]
        rep["known_gap"][key] = {
            "g_band": g,
            "p_D_known_gap": A.boot(np.array([K[s]["p"] for s in ids])),
            "verdict": "validated" if (lo is not None and lo >= 0.10) else "not_validated",
        }
    dd = P1C / "llama_dose" / "llama31_instruct_meta"
    if (dd / "d_chat_dose2_bf.jsonl").exists():
        ids_chat, ids_sp = llama_letter_ids(dd)
        screen = set(json.loads((A.DATA / "screened_ids_llama31_meta.json").read_text())["ids"])

        def per(cell):
            acc: dict[str, list] = {}
            for sid, m, _ in A.row_masses(dd, cell, status, ids_chat, ids_sp):
                acc.setdefault(sid, []).append(m)
            return {k: float(np.mean(v)) for k, v in acc.items()}

        D2, F, TF = (
            per("d_chat_dose2_bf_forced"),
            per("d_chat_dose2_filler_bf_forced"),
            per("d_chat_dose2_filler_tf_forced"),
        )
        D1 = per("d_chat_dose1_bf_forced")
        ids = sorted(s for s in screen if s in D2 and s in F and s in TF)
        d_dose = A.boot(np.array([D2[s] - F[s] for s in ids]))
        d_tf = A.boot(np.array([D2[s] - TF[s] for s in ids]))
        reduces = d_dose["ci95"][1] is not None and d_dose["ci95"][1] < 0 and d_tf["ci95"][1] < 0
        rep["llama_dose"] = {
            "n": len(ids),
            "delta_dose2_minus_filler": d_dose,
            "delta_dose2_minus_TF": d_tf,
            "truncation_effect_TF_minus_filler": A.boot(np.array([TF[s] - F[s] for s in ids])),
            "secondary_dose1_minus_filler": A.boot(
                np.array([D1[s] - F[s] for s in ids if s in D1])
            ),
            "verdict": "reduces_generalizes" if reduces else "does_not_scoped_to_olmo3",
        }
    (A.DATA / "analysis_phase1_session_c.json").write_text(json.dumps(rep, indent=1))

    def f(x):
        if x["mean"] is None:
            return "n/a"
        return "%.3f [%.3f, %.3f] n%d" % (x["mean"], *x["ci95"], x["n"])

    for k, v in rep["known_gap"].items():
        print(f"known-gap {k:24s} g_band {f(v['g_band'])}  -> {v['verdict']}")
    if rep["llama_dose"]:
        ld = rep["llama_dose"]
        print(
            "llama dose: dose2-filler",
            f(ld["delta_dose2_minus_filler"]),
            "| dose2-TF",
            f(ld["delta_dose2_minus_TF"]),
            "->",
            ld["verdict"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
