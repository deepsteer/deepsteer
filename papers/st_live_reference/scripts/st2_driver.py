"""ST Phase 2 on GPT-OSS-20B (ST_PLAN Amendment 22 and its riders; author, 2026-10-21): the
target-search cells, one HF process on the ST stack of record (torch 2.13 in the vLLM venv, eager
attention and experts, bf16 dequantized), resumable arm by arm.

Cell 1 (pod A), the full-n gate: the 45 brake-engaged H1_train scenarios x 16 rollouts, fresh seeds
(stage ``st2c1``), every arm regenerated at alpha 267.3 on ``clause_end``, block 12 (hidden-state
index 13): ``none``, ``fill`` (the T1.3 unit draws, 5, cycled by rollout), ``care_harm``,
``V_harm`` and ``neg_a`` (the anchor axis negated). Each rollout is parsed by the parser of record
and scored for coherence (its trace's NLL under the unsteered model). ``c1_read`` writes the
summary (ΔP against ``none`` and against ``fill``, Amendment 13's relative coherence bars, §5.0's
rule, rider (a)'s protect-ward reading, care beyond harm as ΔP(care_harm) − ΔP(V_harm)); the
analysis of record, Holm-Bonferroni included, runs locally from the saved rows.

    python3 st2_driver.py --out <dir> --inputs <st2_inputs dir> [--phases ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

KEY = "gpt_oss_20b"
DATA = HERE.parent / "data"
L_STAR = 13  # hidden-state index (block 12), Amendment 12 item 1
CLS = "clause_end"
ALPHA = 267.301171875  # the gate of record's alpha (Amendment 13)
R1 = 16  # Cell 1 rollouts per scenario (Amendment 22 rider (d))
BATCH = 160
C1_ARMS = ("none", "fill", "care_harm", "V_harm", "neg_a")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    import pod_st_screen as P
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    from deepsteer.kdg.schema import load_scenario_dir

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    Hs, _ = load_scenario_dir([P.H1_SET])
    kept = set(json.loads(P.H1_KEPT.read_text())["ids"])
    z = np.load(a.inputs / "st2_inputs.npz")
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept},
         "engaged": json.loads((DATA / "st2_engaged_train_v1.json").read_text())["ids"],
         "inp": {k: z[k] for k in z.files}, "layer": L_STAR, "cls": CLS,
         "cap": L.CAP_OF_RECORD, "batch": BATCH, "alpha": ALPHA, "r1": R1}
    if a.dry_run:  # a random tiny model writes no punctuation: onset, so the hooks fire
        c.update(engaged=c["engaged"][:2], layer=2, cap=24, batch=8, alpha=1.0, r1=2,
                 cls="onset")
    return c


def load_model(c: dict, a):
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:  # a tiny GPT-OSS with the real tokenizer; inputs at the tiny width
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = st_tiny.build(KEY, c["out"] / "_tiny")
        m = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        h = m.model.config.hidden_size
        rng = np.random.default_rng(0)
        unit = lambda v: (v / np.linalg.norm(v)).astype(np.float32)  # noqa: E731
        c["inp"] = {k: unit(rng.standard_normal(h)) for k in ("anchor_axis", "care_harm",
                                                               "V_harm")}
        c["inp"]["fill_unit"] = SG.fill_draws(rng.standard_normal((64, h)), 1.0, 5, 0)
        fmt = c["fmt"]  # a random model never writes the harmony header: open the trace at once
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        return m
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        attn_implementation="eager", **c["P"].hf_load_kwargs(c["spec"]))


def screen(c: dict, model, a) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    sc.unbatched = True
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def record_path(c: dict) -> Path:
    return c["out"] / "st2_record.json"


def load_record(c: dict) -> dict:
    p = record_path(c)
    return json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    record_path(c).write_text(json.dumps(rec, indent=1, default=L._json_default))


def seeds_for(stage: str, jobs: list[dict]) -> list[int]:
    """Fresh seeds by stage name, shared across a cell's arms by (scenario, rollout)."""
    return [zlib.crc32(f"{stage}|{j['s'].id}|{j['k']}".encode()) % (2**31) for j in jobs]


def run_arm(c: dict, sc, model, stage: str, name: str, jobs: list[dict], intervention,
            extra=None) -> list[dict]:
    """Generate, save per rollout, parse with the parser of record, score the trace's NLL under
    the unsteered model (T1.3's ``run_arm`` with this cell's stage and seeds)."""
    raw = c["out"] / "gen" / f"{name}.raw.jsonl"
    raw.unlink(missing_ok=True)
    sd = seeds_for(stage, jobs)
    t0 = time.time()
    rows = SG.generate(model, [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"], seeds=sd,
                       layer=c["layer"] if intervention is not None else None,
                       cls=c["cls"] if intervention is not None else None,
                       intervention=intervention, batch=c["batch"],
                       on_finish=SG.jsonl_saver(raw))
    gen_s = time.time() - t0
    parsed = L.build_rows(sc, jobs, [r["gen_ids"] for r in rows], sd, cap=c["cap"], stage=stage,
                          engine="hf_steered")
    for i, (p, r, j) in enumerate(zip(parsed, rows, jobs)):
        p.update(arm=name, n_steered=r["n_steered"], trace_start=r["trace_start"],
                 hit_cap=r["hit_cap"], rollout_key=f"{stage}|{j['s'].id}|{j['k']}",
                 trace_nll_unsteered=SG.trace_nll(model, j["ids"], r["gen_ids"]),
                 **((extra or (lambda _i: {}))(i)))
    sc.save_jsonl(f"gen/{name}.jsonl", parsed)
    print(f">> st2 {name}: {len(parsed)} rollouts in {gen_s:.0f}s, parsed "
          f"{np.mean([p['letter'] is not None for p in parsed]):.3f}", flush=True)
    return parsed


def c1_intervention(c: dict, arm: str, jobs: list[dict]):
    inp, al = c["inp"], c["alpha"]
    if arm == "none":
        return None
    if arm == "fill":
        fu = inp["fill_unit"]
        return SG.Add(np.stack([al * fu[j["k"] % len(fu)] for j in jobs]).astype(np.float32), 1.0)
    if arm == "neg_a":
        return SG.Add(-inp["anchor_axis"], al)
    return SG.Add(inp[arm], al)


def phase_c1_arm(c, sc, model, rec, a, arm: str) -> dict:
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r1"])
    fu = len(c["inp"]["fill_unit"])
    rows = run_arm(c, sc, model, "st2c1", f"c1_{arm}", jobs, c1_intervention(c, arm, jobs),
                   extra=lambda i: {"alpha": 0.0 if arm == "none" else c["alpha"], "axis": arm,
                                    **({"fill_draw": jobs[i]["k"] % fu} if arm == "fill"
                                       else {})})
    return {"status": "ok", "n": len(rows), "alpha": c["alpha"],
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def _rows(c: dict, name: str) -> list[dict]:
    return [json.loads(x) for x in (c["out"] / "gen" / f"{name}.jsonl").open()]


def protect_ward(dp_none: dict, dp_ctrl: dict, coh: dict) -> bool:
    """Amendment 22 rider (a): ΔP > 0 against none and against the matched control, both CIs
    excluding 0, under the coherence bars."""
    lo_n, lo_c = dp_none["ci95"][0], dp_ctrl["ci95"][0]
    return bool(coh["passes"] and lo_n is not None and lo_c is not None and lo_n > 0
                and lo_c > 0)


def phase_c1_read(c, sc, model, rec, a) -> dict:
    """The pod's summary; the analysis of record (Holm-Bonferroni over the pre-registered arms,
    the negative reading's MDEs) is local."""
    none, fill = _rows(c, "c1_none"), _rows(c, "c1_fill")
    keys = [r["rollout_key"] for r in none]
    pn, pf = T13.p_by_scenario(none), T13.p_by_scenario(fill)
    out: dict = {"fill": {"dP_vs_none": T13.delta_p(fill, none),
                          "coherence": T13._coh(fill, none, relative=True)},  # noqa: SLF001
                 "arms": {}}
    for arm in ("care_harm", "V_harm", "neg_a"):
        rows = _rows(c, f"c1_{arm}")
        if [r["rollout_key"] for r in rows] != keys:
            raise SystemExit(f"c1_{arm} rows do not pair with c1_none")
        pa = T13.p_by_scenario(rows)
        ids = sorted(set(pa) & set(pf))
        dn, df = T13.delta_p(rows, none), T13.boot([pa[s] - pf[s] for s in ids])
        coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
        gate = bool(coh["passes"] and dn["mean"] is not None and df["mean"] is not None
                    and dn["mean"] >= T13.GATE_DP and df["mean"] >= T13.GATE_OVER_FILL)
        out["arms"][arm] = {"dP_vs_none": dn, "dP_vs_fill": df, "coherence": coh,
                            "passes_gate": gate, "protect_ward": protect_ward(dn, df, coh)}
    pc, ph = T13.p_by_scenario(_rows(c, "c1_care_harm")), T13.p_by_scenario(_rows(c, "c1_V_harm"))
    ids = sorted(set(pc) & set(ph) & set(pn))
    out["care_beyond_harm"] = T13.boot([pc[s] - ph[s] for s in ids])
    out["rule"] = ("section 5.0 (dP >= +0.10 vs none and >= +0.05 over fill) under Amendment 13's "
                   "relative coherence bars; protect-ward per Amendment 22 rider (a); care beyond "
                   "harm = dP(care_harm) - dP(V_harm), paired by scenario")
    return {"status": "ok", **out}


PHASES = {**{f"c1_{arm}": (lambda arm: lambda c, sc, m, rec, a: phase_c1_arm(c, sc, m, rec, a,
                                                                             arm))(arm)
             for arm in C1_ARMS}, "c1_read": phase_c1_read}
CELL1 = [f"c1_{arm}" for arm in C1_ARMS] + ["c1_read"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--phases", default=",".join(CELL1))
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    rec = load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    if not todo:
        print(">> st2: every requested phase finished earlier", flush=True)
        return 0
    model = load_model(c, a)
    sc = screen(c, model, a)
    arm_s: list[float] = []
    for name in todo:
        est = float(np.mean(arm_s)) if arm_s and name != "c1_read" else 0.0
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2 {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        res = PHASES[name](c, sc, model, rec, a)
        if name != "c1_read":
            arm_s.append(time.time() - t0)
        rec = load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        brief = {k: res[k] for k in ("n", "parse_rate") if k in res}
        if name == "c1_read":
            r3 = lambda x: None if x is None else round(x, 3)  # noqa: E731
            brief = {k: {"dP": r3(v["dP_vs_none"]["mean"]), "vs_fill": r3(v["dP_vs_fill"]["mean"]),
                         "gate": v["passes_gate"], "protect_ward": v["protect_ward"]}
                     for k, v in res["arms"].items()}
        print(f">> st2 {name}: done ({json.dumps(brief)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
