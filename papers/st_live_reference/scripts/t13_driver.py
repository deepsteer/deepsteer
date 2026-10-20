"""The T1.3 pod (ST_PLAN §5.0, Amendments 11-12; author, 2026-10-19): the target-selection gate on
GPT-OSS-20B and its `none` panel, one HF process on the ST stack of record (torch 2.13 in the vLLM
venv, eager attention and experts, Amendment 9).

Phases, in order, each recorded and skipped on resume once finished:

- **sweep**: the 5 sweep scenarios x 8, a `none` arm and `care` at alpha in {0.05, 0.1, 0.2, 0.4} x
  the median residual norm at L* and class (one batch, a vector per rollout); the gate's alpha is
  the largest passing the coherence gate (outcome-blind; Amendment 12 item 4);
- **gate**: the 20 gate scenarios x 8 under `none`, `fill` (5 covariance-matched draws pooled over
  the 8 rollouts, norm-matched to care), `care` (the anchor axis at the sweep's alpha), `self` at
  beta 0.25 and 0.5 (the scenario's judgment-frame residual mixed in); every arm at hidden-state
  index 13 (block 12) on the class the constancy read picked; seeds shared across arms by
  (scenario, rollout); each rollout parsed by the parser of record and scored for coherence (its
  trace's NLL under the unsteered model); ΔP against the in-gate `none`, scenario bootstrap; §5.0's
  rule (ΔP >= +0.10 and above fill's by >= 0.05 under the coherence bars; `self` passes if either
  beta does);
- **transport**: §5.0's control, the `self` mix at the dose-0 decision token on the letter readout
  (20 gate scenarios x 8 permutations, beta 0, 0.25, 0.5);
- **panel**: GPT-OSS's `none` panel, all five letter cells on 586 x 8, single-row (Amendment 11),
  then the carrier call (``panel_call``). It starts only if the deadline leaves time for all five
  cells (E_none and the band come from one read); otherwise it is recorded as moved to the T1.4 pod.

    python3 t13_driver.py --out <dir> --inputs <t13_inputs dir> [--dry-run] [--deadline <epoch>]
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

KEY = "gpt_oss_20b"
L_STAR = 13  # hidden-state index (block 12), Amendment 12 item 1
R = 8
BATCH = 160
BETAS = (0.25, 0.5)
GATE_DP, GATE_OVER_FILL = 0.10, 0.05
PANEL_S_PER_ROW = 840.0 / 2112  # T1.1's single-row dose-0 rate on this model
PANEL_MARGIN = 1.15
N_BOOT, SEED = 10_000, 0


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
    split = json.loads(P.SPLIT.read_text())
    Hs, _ = load_scenario_dir([P.H1_SET])
    kept = set(json.loads(P.H1_KEPT.read_text())["ids"])
    H = {s.id: s for s in Hs if s.id in kept}
    meta = json.loads((a.inputs / "t13_inputs.json").read_text())
    z = np.load(a.inputs / "t13_inputs.npz")
    inp = {k: z[k] for k in z.files}
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "H": H, "out": out,
         "order": L.h1_train_order(split), "meta": meta, "inp": inp, "layer": L_STAR,
         "cls": meta["class"], "cap": L.CAP_OF_RECORD, "batch": BATCH,
         "gate": list(meta["gate"]), "sweep": list(meta["sweep"]), "alphas": meta["alphas"]}
    if a.dry_run:  # a tiny GPT-OSS with the real tokenizer, so the decode loop runs end to end
        c.update(layer=2, cap=24, batch=8, gate=c["gate"][:2], sweep=c["sweep"][:1])
    return c


def load_model(c: dict, a):
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = st_tiny.build(KEY, c["out"] / "_tiny")
        m = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        h = m.model.config.hidden_size
        rng = np.random.default_rng(0)  # the inputs at the tiny width (shape-only stand-ins)
        ax = rng.standard_normal(h).astype(np.float32)
        c["inp"] = {"axis": ax / np.linalg.norm(ax),
                    "fill_unit": SG.fill_draws(rng.standard_normal((64, h)), 1.0, 5, 0),
                    "self_ids": np.array(c["gate"] + c["sweep"]),
                    "self_src": rng.standard_normal((len(c["gate"] + c["sweep"]), h))
                    .astype(np.float32)}
        c["alphas"] = [f * 1.0 for f in (0.05, 0.1, 0.2, 0.4)]
        # a random model never writes the harmony analysis header, so the trace is opened at the
        # first generated token here and the hooks fire (the header rule has its own test)
        fmt = c["fmt"]
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        return m
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        attn_implementation="eager", **c["P"].hf_load_kwargs(c["spec"]))


def screen(c: dict, model, a) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    sc.unbatched = True  # single-row readouts of record (Amendment 9 item 2)
    sc.layers_override = [1, 2, 3] if a.dry_run else [b + 1 for b in (6, 9, 12, 15, 18)]
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def record_path(c: dict) -> Path:
    return c["out"] / "t13_record.json"


def load_record(c: dict) -> dict:
    p = record_path(c)
    return json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    record_path(c).write_text(json.dumps(rec, indent=1, default=L._json_default))


def seeds_for(jobs: list[dict]) -> list[int]:
    """Shared across arms by (scenario, rollout): arms differ only in the intervention."""
    return [zlib.crc32(f"t13|{j['s'].id}|{j['k']}".encode()) % (2**31) for j in jobs]


def boot(x) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    d = x[np.random.default_rng(SEED).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "n": int(len(x)),
            "mde": float(2.8 * (hi - lo) / (2 * 1.96))}


def p_by_scenario(rows: list[dict]) -> dict[str, float]:
    """Protective share over parsed actions per scenario (Amendment 4: defer stays in the
    denominator, unparsed rollouts are missing)."""
    by: dict[str, list] = {}
    for r in rows:
        if r.get("norm_status") is not None:
            by.setdefault(r["scenario_id"], []).append(r["norm_status"] == "violating")
    return {s: float(np.mean(v)) for s, v in by.items()}


def delta_p(arm: list[dict], none: list[dict]) -> dict:
    pa, pn = p_by_scenario(arm), p_by_scenario(none)
    ids = sorted(set(pa) & set(pn))
    return {**boot([pa[s] - pn[s] for s in ids]), "P_arm": float(np.mean([pa[s] for s in ids]))
            if ids else None, "P_none": float(np.mean([pn[s] for s in ids])) if ids else None}


def run_arm(c: dict, sc, model, name: str, jobs: list[dict], intervention, pidx=None,
            extra=None) -> list[dict]:
    """Generate, save per rollout, parse with the parser of record, score the trace's NLL under
    the unsteered model; one jsonl per arm."""
    raw = c["out"] / "gen" / f"{name}.raw.jsonl"
    raw.unlink(missing_ok=True)
    sd = seeds_for(jobs)
    t0 = time.time()
    rows = SG.generate(model, [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"], seeds=sd,
                       layer=c["layer"] if intervention is not None else None,
                       cls=c["cls"] if intervention is not None else None,
                       intervention=intervention, prompt_index=pidx, batch=c["batch"],
                       on_finish=SG.jsonl_saver(raw))
    gen_s = time.time() - t0
    parsed = L.build_rows(sc, jobs, [r["gen_ids"] for r in rows], sd, cap=c["cap"], stage="t13",
                          engine="hf_steered")
    for i, (p, r, j) in enumerate(zip(parsed, rows, jobs)):
        p.update(arm=name, n_steered=r["n_steered"], trace_start=r["trace_start"],
                 hit_cap=r["hit_cap"], rollout_key=f"t13|{j['s'].id}|{j['k']}",
                 trace_nll_unsteered=SG.trace_nll(model, j["ids"], r["gen_ids"]),
                 **((extra or (lambda _i: {}))(i)))
    sc.save_jsonl(f"gen/{name}.jsonl", parsed)
    print(f">> t13 {name}: {len(parsed)} rollouts in {gen_s:.0f}s, parsed "
          f"{np.mean([p['letter'] is not None for p in parsed]):.3f}", flush=True)
    return parsed


def _coh(steered: list[dict], base: list[dict], relative: bool = False) -> dict:
    return SG.coherence([{"n_tokens": r["n_gen_tokens"]} for r in steered],
                        [{"n_tokens": r["n_gen_tokens"]} for r in base],
                        [r["letter"] is not None for r in steered],
                        [r["trace_nll_unsteered"] for r in steered],
                        [r["trace_nll_unsteered"] for r in base],
                        parsed_base=[r["letter"] is not None for r in base] if relative else None)


# ---- phases ----------------------------------------------------------------------------------
def phase_sweep(c, sc, model, rec, a) -> dict:
    jobs = L.h1_jobs(sc, c["H"], c["sweep"], R)
    none = run_arm(c, sc, model, "sweep_none", jobs, None)
    al = list(c["alphas"])
    jobs4 = [j for _ in al for j in jobs]
    V = np.stack([alpha * c["inp"]["axis"] for alpha in al for _ in jobs]).astype(np.float32)
    steered = run_arm(c, sc, model, "sweep_care", jobs4, SG.Add(V, 1.0),
                      extra=lambda i: {"alpha": al[i // len(jobs)],
                                       "alpha_factor": [0.05, 0.1, 0.2, 0.4][i // len(jobs)]})
    per = {}
    for k, alpha in enumerate(al):
        part = steered[k * len(jobs): (k + 1) * len(jobs)]
        per[str(alpha)] = {"coherence": _coh(part, none), "dP_descriptive": delta_p(part, none)}
    ok = [alpha for alpha in al if per[str(alpha)]["coherence"]["passes"]]
    # no coherent alpha: care and fill still run at the smallest alpha, so self keeps its fill
    # comparator and care's coherence failure fails it under the rule (flagged for the author)
    chosen = max(ok) if ok else min(al)
    return {"alphas": al, "per_alpha": per, "alpha": chosen,
            "rule": "largest alpha passing coherence (Amendment 12 item 4)",
            "no_coherent_alpha": not ok}


def phase_gate(c, sc, model, rec, a) -> dict:
    alpha = rec["phases"]["sweep"].get("alpha")
    jobs = L.h1_jobs(sc, c["H"], c["gate"], R)
    ids = [str(x) for x in c["inp"]["self_ids"]]
    pidx = [ids.index(j["s"].id) for j in jobs]
    out: dict = {"alpha": alpha, "class": c["cls"], "layer_hidden_index": c["layer"], "arms": {}}
    none = run_arm(c, sc, model, "gate_none", jobs, None)
    arms: dict[str, list[dict]] = {}
    fu = c["inp"]["fill_unit"]
    Vf = np.stack([alpha * fu[j["k"] % len(fu)] for j in jobs]).astype(np.float32)
    arms["fill"] = run_arm(c, sc, model, "gate_fill", jobs, SG.Add(Vf, 1.0),
                           extra=lambda i: {"fill_draw": jobs[i]["k"] % len(fu)})
    arms["care"] = run_arm(c, sc, model, "gate_care", jobs, SG.Add(c["inp"]["axis"], alpha))
    for beta in BETAS:
        arms[f"self_{beta}"] = run_arm(c, sc, model, f"gate_self_{beta}", jobs,
                                       SG.Mix(c["inp"]["self_src"], beta), pidx=pidx)
    dp = {k: delta_p(v, none) for k, v in arms.items()}
    fill_dp = (dp.get("fill") or {}).get("mean")
    for k, v in arms.items():
        coh = _coh(v, none)
        d = dp[k]["mean"]
        passes = None
        if k != "fill" and d is not None and fill_dp is not None:
            passes = bool(coh["passes"] and d >= GATE_DP and d - fill_dp >= GATE_OVER_FILL)
        out["arms"][k] = {"coherence": coh, "dP": dp[k], "passes": passes}
    out["care_passes"] = (out["arms"].get("care") or {}).get("passes")
    out["self_passes"] = any(bool(out["arms"][f"self_{b}"]["passes"]) for b in BETAS)
    out["no_coherent_alpha"] = bool(rec["phases"]["sweep"].get("no_coherent_alpha"))
    out["rule"] = ("dP >= +0.10 and dP - dP_fill >= 0.05 under the coherence bars; self passes if "
                   "either beta does (section 5.0, Amendment 12)")
    return out


def phase_transport(c, sc, model, rec, a) -> dict:
    cand = sc.dose0
    ids_src = [str(x) for x in c["inp"]["self_ids"]]
    rows, X, S = [], [], []
    for sid in c["gate"]:
        s = c["H"][sid]
        for k in range(R):
            o = L.assign_letters(s, k)
            X.append(L.dose0_ids(sc, s, o, "dl_chat_neutral", cand))
            S.append(c["inp"]["self_src"][ids_src.index(sid)])
            rows.append({"scenario_id": sid, "perm": k,
                         "viol": [x for x, op in o if op.norm_status == "violating"][0],
                         "shown": [x for x, _ in o]})
    lids = L.letter_ids(sc)
    out: dict = {"beta": {}}
    base = None
    for beta in (0.0,) + BETAS:
        lp = SG.mix_readout(model, X, np.stack(S), beta, c["layer"], lids)
        p = []
        for r, row in zip(rows, lp):
            sh = np.array([row["ABCDE".index(x)] for x in r["shown"]])
            q = np.exp(sh - sh.max())
            q /= q.sum()
            p.append(float(q[r["shown"].index(r["viol"])]))
        ps: dict[str, list] = {}
        for r, v in zip(rows, p):
            ps.setdefault(r["scenario_id"], []).append(v)
        per = {s: float(np.mean(v)) for s, v in ps.items()}
        if beta == 0.0:
            base = per
            out["P_dose0_unmixed"] = float(np.mean(list(per.values())))
            continue
        d = boot([per[s] - base[s] for s in sorted(per)])
        out["beta"][str(beta)] = {"dP_dose0": d, "moves": bool(d["ci95"][0] is not None and (
            d["ci95"][0] > 0 or d["ci95"][1] < 0))}
    out["rule"] = ("a self mix that cannot move the dose-0 letter cannot be read as a null in the "
                   "trace (section 5.0 transport_control)")
    return out


def phase_panel(c, sc, model, rec, a) -> dict:
    import panel_call as PC

    from deepsteer.kdg.schema import load_scenario_dir

    panel_all, _ = load_scenario_dir(sorted((c["P"].KDG / "data").glob("*_scenarios_*.json")))
    panel = L.panel_586(panel_all)
    if a.dry_run:
        panel = panel[:4]
    elif len(panel) != 586:
        raise SystemExit(f"panel of record has {len(panel)} rows, expected 586")
    need = len(panel) * L.N_PERM * len(L.PANEL_CELLS) * PANEL_S_PER_ROW * PANEL_MARGIN
    if a.deadline and time.time() + need > a.deadline:
        return {"status": "moved", "reason": f"needs ~{need / 3600:.2f} h, the deadline leaves "
                f"{(a.deadline - time.time()) / 3600:.2f} h; all five cells come from one read, "
                "so the panel moves to the T1.4 pod (Amendment 12 item 6)"}
    cells = [L.run_letter_cell(sc, panel, cell, "panel") for cell in L.PANEL_CELLS]
    call = PC.call(c["out"] / "panel", KEY)
    (c["out"] / "panel_call.json").write_text(json.dumps(call, indent=1))
    e = call["E_none"]["E_prob"]
    print(f">> t13 panel call: {call['call']} ({call['reason']}); E_none {e['mean']} "
          f"{e['ci95']}; g_band {call['band']['g_band']['mean']}", flush=True)
    return {"status": "ok", "cells": cells, "call": call["call"], "reason": call["reason"],
            "dose0_candidate": sc.dose0.name}


def _rows(c: dict, name: str) -> list[dict]:
    return [json.loads(x) for x in (c["out"] / "gen" / f"{name}.jsonl").open()]


def phase_care_rerun(c, sc, model, rec, a) -> dict:
    """Amendment 13 (author, 2026-10-20): if the gate's parse rate for care at the sweep's alpha is
    below 0.95, alpha steps down one grid point and care reruns once on the gate's 20 x 8 with fill
    norm-matched to the new alpha (same unit draws); pass = care's gate result, fail = care fails.
    The in-gate none rows are reused. self is scored against both fills (which one is self's
    comparator is the author's call)."""
    g = rec["phases"]["gate"]
    care = g["arms"]["care"]
    if care["coherence"]["parse_rate"] >= 0.95:
        return {"status": "ok", "ran": False, "reason": "care's parse rate met the bar"}
    al = sorted(c["alphas"])
    lower = [x for x in al if x < g["alpha"]]
    if not lower:
        return {"status": "ok", "ran": False, "reason": "no lower grid point", "care_passes": False}
    alpha = max(lower)
    jobs = L.h1_jobs(sc, c["H"], c["gate"], R)
    none = _rows(c, "gate_none")
    if [r["rollout_key"] for r in none] != [f"t13|{j['s'].id}|{j['k']}" for j in jobs]:
        raise SystemExit("gate_none rows do not match the gate's jobs")
    fu = c["inp"]["fill_unit"]
    Vf = np.stack([alpha * fu[j["k"] % len(fu)] for j in jobs]).astype(np.float32)
    fill = run_arm(c, sc, model, "gate_fill_rerun", jobs, SG.Add(Vf, 1.0),
                   extra=lambda i: {"fill_draw": jobs[i]["k"] % len(fu), "alpha": alpha})
    care2 = run_arm(c, sc, model, "gate_care_rerun", jobs, SG.Add(c["inp"]["axis"], alpha),
                    extra=lambda i: {"alpha": alpha})
    dpf, dpc = delta_p(fill, none), delta_p(care2, none)

    def verdict(coh: dict, dp: dict) -> bool:
        return bool(coh["passes"] and dp["mean"] is not None and dpf["mean"] is not None
                    and dp["mean"] >= GATE_DP and dp["mean"] - dpf["mean"] >= GATE_OVER_FILL)

    coh = _coh(care2, none, relative=True)
    arms = {"care": {"alpha": alpha, "coherence": coh, "dP": dpc, "passes": verdict(coh, dpc)},
            "fill": {"alpha": alpha, "coherence": _coh(fill, none, relative=True), "dP": dpf}}
    for b in BETAS:  # self's comparator is the rerun fill; coherence re-scored on the new rule
        rows = _rows(c, f"gate_self_{b}")
        cs = _coh(rows, none, relative=True)
        ds = delta_p(rows, none)
        arms[f"self_{b}"] = {"coherence": cs, "dP": ds, "passes": verdict(cs, ds)}
    off = {k: {"alpha": g["alpha"], "coherence_relative": _coh(_rows(c, f"gate_{k}"), none,
                                                               relative=True),
               "coherence_absolute": g["arms"][k]["coherence"], "dP": g["arms"][k]["dP"],
               "status": ("failed coherence at 534.6 (descriptive: a random direction at this "
                          "norm broke parsing where V_care did not)" if k == "fill" else
                          "failed coherence under the rule then in force; dP descriptive, "
                          "off-coherence, not used for the gate")}
           for k in ("care", "fill")}
    return {"status": "ok", "ran": True, "alpha": alpha, "from_alpha": g["alpha"],
            "trigger": "care's parse rate below 0.95 under the rule then in force (Amendment 13)",
            "gate_of_record": arms, "care_passes": arms["care"]["passes"],
            "self_passes": any(arms[f"self_{b}"]["passes"] for b in BETAS),
            "first_alpha_runs": off,
            "rule": ("Amendment 13: one step down, once; coherence parse relative to none (-0.05), "
                     "absolute rate beside; self's comparator is the rerun fill")}


PHASES = {"sweep": phase_sweep, "gate": phase_gate, "transport": phase_transport,
          "panel": phase_panel, "care_rerun": phase_care_rerun}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    # care_rerun (Amendment 13) runs only when triggered; a resumed pod skips finished phases
    ap.add_argument("--phases", default="sweep,gate,transport,care_rerun,panel")
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds (envelope)")
    ap.add_argument("--force", action="store_true", help="re-run a phase recorded as finished")
    a = ap.parse_args()
    c = context(a)
    model = load_model(c, a)
    sc = screen(c, model, a)
    rec = load_record(c)
    rec.setdefault("inputs", c["meta"])
    if not a.dry_run and "gates" not in rec:
        g = c["P"].gates(sc, c["spec"], c["split"], c["order"], c["H"], False)
        rec["gates"] = g
        save_record(c, rec)
        if not g["ok"]:
            return 2
    for name in a.phases.split(","):
        done = rec["phases"].get(name) or {}
        if done.get("finished") and not a.force:
            print(f">> t13 {name}: finished earlier ({done['finished']}); skipped", flush=True)
            continue
        if name == "gate" and not (rec["phases"].get("sweep") or {}).get("finished"):
            raise SystemExit("gate needs the sweep's alpha")
        if name == "care_rerun" and not (rec["phases"].get("gate") or {}).get("finished"):
            raise SystemExit("care_rerun needs the gate's record")
        if a.deadline and name != "panel" and time.time() > a.deadline:
            rec["phases"][name] = {"status": "deadline", "at": _now()}
            save_record(c, rec)
            return 3
        t0 = time.time()
        res = PHASES[name](c, sc, model, rec, a)
        rec["phases"][name] = {**res, "started_s": t0, "seconds": time.time() - t0,
                               "finished": _now() if res.get("status", "ok") == "ok" else None}
        save_record(c, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
