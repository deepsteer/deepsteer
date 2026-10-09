#!/usr/bin/env python3
"""Rebuild the dose-stated main-stage rows from the per-batch bank (pod q5bnihxwoootif crashed after
generation, before ``ds_main.jsonl`` was written; ANOMALIES process ledger 2026-10-08).

    python3 papers/kdg_panel/scripts/reconstruct_ds_main.py

Each banked batch holds ``[scenario_id, permutation seed]`` per job and the generated ids. Rows are
rebuilt with the code the pod would have used: ``assign_letters`` for the order,
``kh.parse_dose_stated`` at the main cap (1,536) for status and token identity, the letter token
ids A..E = 32..36 (checked at every gate). No readout fields: the forward readout never ran.
Output: ``ds_main_reconstructed.jsonl``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_phase1_session_a as A  # noqa: E402
import kdg_harmony as kh  # noqa: E402

from deepsteer.kdg.schema import assign_letters, letter_map, load_scenario_dir  # noqa: E402

LETTER = {32: "A", 33: "B", 34: "C", 35: "D", 36: "E"}
CAP = 1536
D = A.OUT / "p2i" / "main" / "gpt_oss_20b"


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    S = {s.id: s for s in scen}
    out = []
    for b in (json.loads(x) for x in (D / "ds_main_partial.jsonl").read_text().splitlines() if x):
        for (sid, seed), g in zip(b["jobs"], b["gen_ids"]):
            o = assign_letters(S[sid], seed)
            ds = kh.parse_dose_stated(g, len(g) >= CAP, set(LETTER))
            letter = LETTER.get(ds.letter_id) if ds.letter_id is not None else None
            opt = dict(o).get(letter) if letter else None
            out.append({"scenario_id": sid, "cell": "dl_chat_neutral", "seed": seed,
                        "stage": b["stage"], "batch": b["batch"], "batch_seed": b["seed"],
                        "order": letter_map(o), "n_gen_tokens": len(g), "status": ds.status,
                        "token_identity": ds.token_identity, "trace_len": ds.trace_len,
                        "letter": letter, "option_id": None if opt is None else opt.option_id,
                        "norm_status": None if opt is None else opt.norm_status,
                        "reconstructed_from": "ds_main_partial.jsonl"})
    with open(D / "ds_main_reconstructed.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} rows -> {D / 'ds_main_reconstructed.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
