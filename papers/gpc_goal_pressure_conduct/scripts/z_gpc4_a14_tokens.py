#!/usr/bin/env python3
"""GPC-A14's zero-GPU discriminator (GPC_PLAN.md §11.29 item 3): the summed adapter's token ratio to the base,
beside the same ratio for P-ctx alone and for C1 alone, from their saved generation records.

- **CST's cells** (panel_operator, forced_plain; the same engine and 4,096 cap): the summed adapter against
  launch 6's same-session base (`outputs/gpc4_v1`), P-ctx against Amendment 52's same-session base
  (`cst_constitutional_steering/outputs/a52_v1`). No C1 generation log exists on these cells.
- **GPC-1's held-out conduct cell** (100 slots): C1 (and C1 at strengths 1.5 and 2.0, `C_conduct` in context)
  against GPC-1's base, the summed adapter (launch 6) against the same base, cross-session as stage 4's read
  was. No P-ctx row exists on this cell. Each ratio's CI is a slot bootstrap.
- Beside it, the goal rate on the cell's pressure slots in each arm's shorter and longer half (by tokens).

    python z_gpc4_a14_tokens.py
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

GPC = Path(__file__).resolve().parents[1]
CST = GPC.parent / "cst_constitutional_steering"
OUT = GPC / "outputs/gpc4_prep/a14_tokens.json"
B, SEED = 2000, "20261121|gpc4|a14"


def _log_tokens(d: Path, tag: str) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob(f"gen_{tag}_log_*.json")):
        g = json.loads(f.read_text())
        for chunk, v in g["chunks"].items():
            out[chunk.split("_0")[0]] = {"n": v["n"], "seconds": v["seconds"], "gen_tokens": v["gen_tokens"],
                                         "cap": g["cap"], "engine": g["engine"]}
    return out


def cst_cells() -> dict:
    s4 = {t: _log_tokens(GPC / "outputs/gpc4_v1", t) for t in ("base_session", "sum")}
    a52 = {t: _log_tokens(CST / "outputs/a52_v1", t) for t in ("base_session", "p_ctx")}
    rec = {}
    for cell in ("panel_operator", "forced_plain"):
        b, s = s4["base_session"][cell], s4["sum"][cell]
        pb, p = a52["base_session"][cell], a52["p_ctx"][cell]
        rec[cell] = {"sum": {"tokens": s["gen_tokens"] / b["gen_tokens"], "wall": s["seconds"] / b["seconds"],
                             "n": s["n"]},
                     "p_ctx": {"tokens": p["gen_tokens"] / pb["gen_tokens"], "wall": p["seconds"] / pb["seconds"],
                               "n": p["n"]},
                     "cap": b["cap"], "engine": b["engine"]}
    return rec


def _rows(f: Path) -> dict[str, dict]:
    return {r["slot_id"]: r for r in map(json.loads, f.read_text().splitlines())}


def holdout_cell() -> dict:
    g1, g2, g4 = GPC / "outputs/gpc1_v1/conduct", GPC / "outputs/gpc2_v1/conduct2", GPC / "outputs/gpc4_v1/conduct2"
    arms = {"base": g1 / "holdout_base.jsonl", "c1": g1 / "holdout_c1.jsonl",
            "c_conduct": g1 / "holdout_c_conduct.jsonl", "c1_s150": g2 / "holdout_c1_s150.jsonl",
            "c1_s200": g2 / "holdout_c1_s200.jsonl", "sum": g4 / "holdout_sum.jsonl"}
    d = {k: _rows(f) for k, f in arms.items()}
    sl = json.loads((GPC / "outputs/gpc1_v1/conduct/slots.json").read_text())
    pres = {i for i in sl["holdout"] if sl["kept"][i]["pressure"]}
    base, rec = d["base"], {}
    for k, rows in d.items():
        ids = sorted(set(rows) & set(base))
        n_tok = lambda s, m: sum(len(m[i]["gen_ids"]) for i in s)  # noqa: E731
        rng = random.Random(f"{SEED}|{k}")
        bs = sorted(n_tok(s, rows) / n_tok(s, base) for s in ([rng.choice(ids) for _ in ids] for _ in range(B)))
        pr = [rows[i] for i in ids if i in pres and rows[i]["choice"] is not None]
        med = statistics.median(len(r["gen_ids"]) for r in pr)
        goal = lambda rr: sum(r["choice"] == "goal" for r in rr) / len(rr) if rr else None  # noqa: E731
        rec[k] = {"n": len(ids), "mean_tokens": statistics.mean(len(rows[i]["gen_ids"]) for i in ids),
                  "ratio_to_base": n_tok(ids, rows) / n_tok(ids, base), "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]],
                  "cap_share": sum(bool(rows[i]["hit_cap"]) for i in ids) / len(ids),
                  "pressure_goal": goal(pr), "pressure_n": len(pr),
                  "goal_shorter_half": goal([r for r in pr if len(r["gen_ids"]) <= med]),
                  "goal_longer_half": goal([r for r in pr if len(r["gen_ids"]) > med])}
    return rec


def main() -> None:
    rec = {"rule": "GPC_PLAN.md §11.29 item 3, ANOMALIES.md GPC-A14 (zero GPU)", "cst_cells": cst_cells(),
           "holdout_conduct": holdout_cell(),
           "notes": ["no C1 generation log on CST's cells; no P-ctx row on the held-out conduct cell",
                     "the held-out ratios are cross-session (GPC-1's base), as stage 4's held-out read was"]}
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
