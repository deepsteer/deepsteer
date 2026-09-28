#!/usr/bin/env python3
"""Per-model screen rates on each model's own letter-only chat cells (KDG_PHASE1_SPEC.md P1-A9
item 2, pushed eda98b0 before this ran); writes Llama-3.1 Meta's screen as the Session C dose
input.

    python3 papers/kdg_panel/scripts/analyze_screen_rates.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

OUTD = A.OUT
MODELS = {
    "olmo3_sft": OUTD / "p1a/stages_chat_sft/olmo3_sft",
    "olmo3_dpo": OUTD / "p1a/stages_chat/olmo3_dpo",
    "olmo3_final": OUTD / "p1a/final_c1/olmo3_instruct",
    "llama31_instruct_meta": OUTD / "p1b/chat_lineages/llama31_instruct_meta",
    "tulu3_sft": OUTD / "p1b/chat_lineages/tulu3_sft",
    "tulu3_dpo": OUTD / "p1b/chat_lineages/tulu3_dpo",
    "tulu3_final": OUTD / "p1b/chat_lineages/tulu3_final",
    "qwen25_instruct": OUTD / "p1b/chat_lineages/qwen25_instruct_p1",
}


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }
    rep = {"amendment": "P1-A9 item 2 (eda98b0)", "models": {}}
    screens = {}
    for key, d in MODELS.items():
        jl = A._rows(d / "jl_chat_neutral.jsonl")
        Dm = A.option_cell([d], "dl_chat_neutral", status)
        Jm = A.option_cell([d], "jl_chat_neutral", status)
        cons = Counter()
        for r in jl:
            if r["scenario_id"] in status and r["norm_status"] == "consistent":
                cons[r["scenario_id"]] += 1
        eng = [
            s
            for s in status
            if s in Dm and s in Jm and min(Dm[s]["mass"], Jm[s]["mass"]) >= A.FLOOR
        ]
        passed = sorted(
            s for s in eng if cons[s] >= 6 and (0.15 <= Dm[s]["p"] <= 0.85 or Dm[s]["p"] >= 0.85)
        )
        screens[key] = passed
        rep["models"][key] = {
            "n_union": len(status),
            "n_engaged": len(eng),
            "engagement_rate": len(eng) / len(status),
            "n_screened": len(passed),
            "screen_rate_of_engaged": len(passed) / max(1, len(eng)),
        }
    llama = screens["llama31_instruct_meta"]
    if len(llama) > 136:
        rng = np.random.default_rng(0)
        llama = sorted(rng.choice(llama, size=136, replace=False).tolist())
    (A.DATA / "screened_ids_llama31_meta.json").write_text(
        json.dumps(
            {
                "source": "P1-A9 item 2: Llama-3.1 Meta chat screen (seed-0 sample if > 136)",
                "n_screen_full": len(screens["llama31_instruct_meta"]),
                "ids": llama,
            },
            indent=1,
        )
    )
    rep["overlap_with_olmo3_final_screen"] = {
        k: len(set(v) & set(screens["olmo3_final"])) for k, v in screens.items()
    }
    (A.DATA / "analysis_screen_rates.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["models"].items():
        print(
            f"{k:24s} engaged {v['n_engaged']:4d}/{v['n_union']}  screened {v['n_screened']:4d}"
            f"  ({v['screen_rate_of_engaged']:.2f} of engaged)  overlap w/ OLMo final "
            f"{rep['overlap_with_olmo3_final_screen'][k]}"
        )
    print("llama dose set:", len(llama))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
