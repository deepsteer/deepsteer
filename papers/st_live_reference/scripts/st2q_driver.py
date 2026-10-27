"""The second family's pod 1 on Qwen3.8-27B (ST_PLAN Amendment 54 and its addition): baseline,
calibration and the `care_harm` gate. Two processes, as Muse's Cell 4 (``st2m_driver``): ``--phases
v1`` (vLLM) first, then the HF phases in one process, resumable phase by phase.

- ``v1`` (vLLM): H1 264 x 8 at effort medium, cap 1,536, KDG sampler v2.
- ``dose0``: the dose-0 letter readout on H1_train 132 x 8 (``enable_thinking_false``), single-row.
- ``bank``: per v1 rollout one teacher-forced forward; ``clause_end`` samples (up to 8 per rollout)
  at blocks 24, 28, 32, 36 and 40.
- ``derive``: the engaged H1_train set (Amendment 5's rule) and the gap it holds; the site (block 32
  if its PR at the median of 21 samples is at least 30, else the nearest healthy banked block, ties
  shallower); the site's median norm and ten covariance-matched fill draws (seed 0); the 5 sweep
  scenarios, seeded, from H1_train outside the engaged set; the arm scenarios (every engaged, cap 45
  stratified).
- ``pilot``: 32 ``none`` rollouts; the batch; the re-price. Above 1.5 x 4.9 A100-h projected, the
  pod stops here (exit 5).
- ``directions``: `care_harm` by W4's recipe at the site, unit.
- ``sweep``: `care_harm` at 0.2x, 0.3x and 0.4x the site's median norm on the sweep scenarios x 8
  against a ``none`` on them; the largest coherent dose is the gate's. None coherent: P1-fail, and
  the pod stops after recording it.
- ``predict`` (section 12): the exact teacher-forced steer at the dose of record on v1's first two
  answered traces per arm scenario, converted per scenario by P(1 - P) with v1's P_none.
- ``spec_none``, ``spec_fill10`` (each draw on all 60 items), ``spec_care``.
- ``arm_none`` and ``arm_care`` (x 16), ``arm_fill`` (x 8, draw (scenario order x 8 + rollout) mod
  10), seeds shared across arms.

The analysis of record is local (``analyze_st2q.py``).

    python3 st2q_driver.py --out <dir> --phases v1 [--dry-run]
    python3 st2q_driver.py --out <dir> [--phases ...] [--dry-run] [--deadline <epoch>] \
        [--session-t0 <epoch>]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
import build_st2gs_inputs as GSI  # noqa: E402
import constancy_read as CR  # noqa: E402
import screen_lib as L  # noqa: E402
import st2_driver as S2  # noqa: E402
import st2c_driver as C  # noqa: E402
import st2gs_driver as GS  # noqa: E402
import st2m_driver as M  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

KEY = "qwen38_27b"
DATA = HERE.parent / "data"
SITE_DEFAULT = 32  # Amendment 54 item 2: depth 0.5, as GPT-OSS's block 12 and Muse's block 26
BANK_BLOCKS = (24, 28, 32, 36, 40)
CLS = "clause_end"
EFFORT = "medium"  # the addition, item 3: ST-A5's screen
R_ARM, R_FILL, R_SWEEP = 16, 8, 8
DOSES = (0.2, 0.3, 0.4)
N_FILL, FILL_SEED = 10, 0
PR_BAR, PR_SAMPLES, PR_N = 30.0, 21, 4000
N_SWEEP, ARM_CAP, MIN_ENGAGED = 5, 45, 10
PRICE_POD1, STOP_FACTOR = 4.9, 1.5
PILOT_N = 32
HF_PHASES = ["dose0", "bank", "derive", "pilot", "directions", "sweep", "predict", "spec_none",
             "spec_fill10", "spec_care", "arm_none", "arm_care", "arm_fill"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


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
    allp, _ = load_scenario_dir(sorted(glob.glob(str(P.KDG / "data" / "*_scenarios_*.json"))))
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    order = L.h1_train_order(split)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept}, "order": order,
         "ids": list(order) + sorted(split["H1_held"]), "train": set(split["H1_train"]),
         "panel": {s.id: s for s in L.panel_586(allp)},
         "spec_ids": json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"],
         "layer": SITE_DEFAULT + 1, "bank_blocks": BANK_BLOCKS, "cls": CLS,
         "cap": L.CAP_OF_RECORD, "batch": 160, "r": R_ARM, "pilot_n": PILOT_N,
         "care_max": None, "record": out / "st2q_record.json"}
    if a.dry_run:  # a tiny Qwen (4 text layers) with the real tokenizer; onset, so hooks fire
        c.update(ids=list(order[:6]) + sorted(split["H1_held"])[:2], layer=3,
                 bank_blocks=(1, 2), cls="onset", cap=24, batch=8, r=2, pilot_n=4,
                 spec_ids=c["spec_ids"][:2], care_max=3)
    return c


def load_record(c: dict) -> dict:
    return json.loads(c["record"].read_text()) if c["record"].exists() else {"model": KEY,
                                                                            "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    c["record"].write_text(json.dumps(rec, indent=1, default=L._json_default))  # noqa: SLF001


def phase_done(c: dict, name: str) -> dict:
    return load_record(c)["phases"].get(name) or {}


def screen(c: dict, model, pin: str) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], pin, dry=False, read_batch=1)
    sc.effort = EFFORT
    sc.unbatched = True
    sc.layers_override = [c["layer"]]
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def date_pin(c: dict, a) -> str:
    p = c["out"] / "gen" / "h1_rollouts.jsonl"
    if not p.exists():
        return a.date_pin
    pins = {r.get("date_pin") for r in _jsonl(p)} - {None}
    if len(pins) > 1:
        raise SystemExit(f"v1 rows carry {len(pins)} date pins")
    return pins.pop() if pins else a.date_pin


# ---- v1 (vLLM) -----------------------------------------------------------------------------------
def phase_v1(c: dict, a) -> dict:
    import screen_vllm as SV
    from screen_stub import ScreenStub

    model = (ScreenStub(KEY, c["tok"], c["fmt"], c["spec"]["n_layers"]) if a.dry_run
             else L.tok_only(c["tok"]))
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, dry=a.dry_run)
    sc.effort = EFFORT
    jobs = L.h1_jobs(sc, c["H"], c["ids"], 8)
    t0 = time.time()
    g = SV.gen_stage(sc, jobs, "st2q", L.CAP_OF_RECORD, c["spec"], probe=True,
                     deadline=a.deadline)
    for r, j in zip(g["rows"], jobs):
        r.update(sample="selection", arm="none", rollout_key=f"st2q_v1|{r['scenario_id']}|"
                 f"{r['rollout']}", split="train" if r["scenario_id"] in c["train"] else "held",
                 prompt_ids=j["ids"], date_pin=a.date_pin)
    sc.save_jsonl("gen/h1_rollouts.jsonl", g["rows"])
    return {"status": "ok", "n": len(jobs), "seconds": time.time() - t0, "timing": g["timing"],
            "probe": g["probe"], "effort": sc.effort, "summary": L.screen_summary(g["rows"])}


# ---- HF ------------------------------------------------------------------------------------------
def load_model(c: dict, a):
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = c["out"] / "_tiny"
        if not (path / "config.json").exists():
            st_tiny.build(KEY, path)
        m = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        fmt = c["fmt"]  # a random model never closes its think block: the trace opens at once
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        return m
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        **c["P"].hf_load_kwargs(c["spec"]))


def pick_site(prs: dict[int, list[float]], default: int = SITE_DEFAULT) -> tuple[int, dict]:
    """The addition's site rule: ``default`` if its PR at the median of the samples is at least
    30, else the nearest banked block that is, ties going to the shallower."""
    med = {b: float(np.median(v)) for b, v in prs.items()}
    if med.get(default, 0.0) >= PR_BAR:
        return default, med
    ok = sorted((abs(b - default), b) for b, m in med.items() if m >= PR_BAR)
    if not ok:
        raise SystemExit(f"no banked block reaches PR {PR_BAR} at the median: {med}")
    return ok[0][1], med


def gap_bound(p_dose0: dict[str, float], p_none: dict[str, float], ids: list[str]) -> dict:
    """The addition, item 2: the gap a claim is bounded by, mean P_dose0 - P_none over ``ids``."""
    g = [p_dose0[s] - p_none[s] for s in ids if s in p_dose0 and s in p_none]
    return {"mean": float(np.mean(g)) if g else None, "n": len(g)}


def phase_derive(c, sc, model, a) -> dict:
    rows = _jsonl(c["out"] / "gen" / "h1_rollouts.jsonl")
    d0 = _jsonl(c["out"] / "readout" / "dose0_h1_train" / "dl_chat_neutral.jsonl")
    pd = c["P"].dose0_protect(d0)
    tr = [r for r in rows if r["scenario_id"] in c["train"] and r.get("norm_status") is not None]
    if any(v >= 0.5 for v in pd.values()):
        eng = L.engaged_subset(tr, pd, use_dose0=True)
    elif a.dry_run:
        eng = {"n_screened_parsed": 0, "n_engaged": 0, "engaged_ids": [], "rule": "dry run"}
    else:
        raise SystemExit("no H1_train scenario at P_dose0 >= 0.5: the engaged rule is empty")
    eng["brake"] = c["P"].brake([r for r in rows if r["scenario_id"] in c["train"]], d0)
    z = np.load(c["out"] / "bank" / "clause_end.npz")
    prs = {}
    for b in c["bank_blocks"]:
        x = z[f"sample_b{b}"].astype(np.float32)
        n = min(len(x), PR_N)
        prs[b] = [CR.participation_ratio(x[np.random.default_rng(s).choice(len(x), n,
                                                                          replace=False)])
                  for s in range(PR_SAMPLES)]
    if a.dry_run:  # a tiny model's PR is far below 30: the last banked block
        site, pr_med = c["bank_blocks"][-1], {b: float(np.median(v)) for b, v in prs.items()}
    else:
        site, pr_med = pick_site(prs)
    s = z[f"sample_b{site}"].astype(np.float64)
    med = float(np.median(np.linalg.norm(s, axis=1)))
    rng = np.random.default_rng(FILL_SEED)
    samp = s[rng.choice(len(s), min(len(s), PR_N), replace=False)]
    fill = SG.fill_draws(samp, 1.0, N_FILL, FILL_SEED)
    ids = sorted(eng["engaged_ids"])
    if a.dry_run and not ids:  # a random tiny model engages nothing: every train scenario
        ids = sorted({r["scenario_id"] for r in tr})
    arms = M.stratified(ids, M.strata_of(ids), ARM_CAP, "arms")
    non = sorted({r["scenario_id"] for r in tr} - set(ids))
    if a.dry_run and not non:
        non = sorted(ids)
    pool = list(non)
    random.Random(f"{M.SWEEP_SEED}|sweep|qwen").shuffle(pool)
    sweep = sorted(pool[: 1 if a.dry_run else N_SWEEP])
    pn = T13.p_by_scenario([r for r in tr if r["scenario_id"] in set(arms)])
    np.savez(c["out"] / "derive.npz", fill_unit=fill.astype(np.float32))
    (c["out"] / "select").mkdir(exist_ok=True)
    (c["out"] / "select" / "engaged.json").write_text(json.dumps(eng, indent=1,
                                                                 default=L._json_default))
    return {"status": "ok", "n_engaged": eng["n_engaged"], "engaged_all": ids,
            "engaged_ids": arms, "underpowered": len(arms) < MIN_ENGAGED,
            "null_no_brake": eng.get("null_no_brake"), "brake_B": eng["brake"]["B"],
            "site_block": site, "pr_median_21": pr_med, "pr_samples": prs,
            "median_norm": med, "alphas": [f * med for f in DOSES], "sweep": sweep,
            "gap_bound": gap_bound(pd, pn, arms),
            "rule": "engaged by Amendment 5; site by the addition's PR rule; fill seed 0; sweep "
                    "seeded from H1_train outside the engaged set"}


def ctx_derived(c: dict) -> None:
    d = phase_done(c, "derive")
    if not d:
        return
    c["engaged"] = d["engaged_ids"]
    c["sweep"] = d["sweep"]
    c["alphas"] = d["alphas"]
    c["layer"] = d["site_block"] + 1
    c["inp"] = {"fill_unit": np.load(c["out"] / "derive.npz")["fill_unit"]}
    p = phase_done(c, "pilot")
    if p.get("batch"):
        c["batch"] = p["batch"]
    sw = phase_done(c, "sweep")
    if sw.get("alpha"):
        c["alpha"] = sw["alpha"]
    dz = c["out"] / "directions.npz"
    if dz.exists():
        c["care"] = np.load(dz)["care_harm"]


def projection(c: dict, a, now: float) -> dict:
    """After the pilot: pod 1's projected A100-h, the elapsed session plus every remaining phase
    at the pilot's (conservative, batch-32) rate."""
    p, b = phase_done(c, "pilot"), phase_done(c, "bank")
    t, read = p["seconds_per_rollout"], b.get("read_seconds_per_rollout") or 0.0
    d0 = phase_done(c, "dose0").get("seconds_per_row") or 0.2
    n_eng = len(c["engaged"])
    rest = {"directions": 0.05 * 3600,
            "sweep": len(c["sweep"]) * R_SWEEP * (1 + len(DOSES)) * (t + read),
            "predict": n_eng * 2 * 2 * read,
            "spec": (2 + N_FILL) * len(c["spec_ids"]) * L.N_PERM * d0,
            "arms": n_eng * (2 * c["r"] + R_FILL) * (t + read)}
    elapsed = (now - a.session_t0) / 3600 if a.session_t0 is not None else None
    total = (elapsed or 0.0) + sum(rest.values()) / 3600 + 0.1
    return {"remaining_hours": {k: v / 3600 for k, v in rest.items()},
            "elapsed_hours": elapsed, "projected_pod_hours": total,
            "stop_above": PRICE_POD1 * STOP_FACTOR, "stops": bool(total > PRICE_POD1 * STOP_FACTOR)}


def phase_directions(c, sc, model, a) -> dict:
    from deepsteer.datasets import load_moral_probing_v2
    from deepsteer.directions import extraction as du

    b = c["layer"] - 1
    pairs = [(p.moral, p.neutral) for p in load_moral_probing_v2().all_pairs
             if p.foundation.value == "care_harm"]
    if c["care_max"]:
        pairs = pairs[: c["care_max"]]
    acts = du.collect_pair_activations(model.wb, pairs, input_format="raw", layers=[b])
    care = du.mean_diff_direction(*acts[b])
    care = (care / np.linalg.norm(care)).astype(np.float32)
    np.savez(c["out"] / "directions.npz", care_harm=care)
    return {"status": "ok", "block": b, "n_care_pairs": len(pairs),
            "type": f"stimulus/content: W4's MFT care/harm mean difference, raw format, "
                    f"mean-pooled, block {b}, unit"}


def phase_sweep(c, sc, model, a) -> dict:
    jobs = L.h1_jobs(sc, c["H"], c["sweep"], R_SWEEP)
    none = S2.run_arm(c, sc, model, "st2q_sweep", "sweep_none", jobs, None)
    al = list(c["alphas"])
    jobs3 = [j for _ in al for j in jobs]
    V = np.stack([x * c["care"] for x in al for _ in jobs]).astype(np.float32)
    steered = S2.run_arm(c, sc, model, "st2q_sweep", "sweep_care", jobs3, SG.Add(V, 1.0),
                         extra=lambda i: {"alpha": al[i // len(jobs)],
                                          "dose": DOSES[i // len(jobs)]})
    per = {}
    for k, x in enumerate(al):
        part = steered[k * len(jobs): (k + 1) * len(jobs)]
        per[str(DOSES[k])] = {"alpha": x,
                              "coherence": T13._coh(part, none, relative=True),  # noqa: SLF001
                              "dP_descriptive": T13.delta_p(part, none)}
    ok = [x for k, x in enumerate(al) if per[str(DOSES[k])]["coherence"]["passes"]]
    return {"status": "ok", "per_dose": per, "alpha": max(ok) if ok else None,
            "dose": DOSES[al.index(max(ok))] if ok else None, "no_coherent_dose": not ok,
            "rule": "largest dose passing Amendment 13's relative bars (outcome-blind)"}


def first_two_answered(rows: list[dict], ids: list[str]) -> list[dict]:
    out = []
    for sid in ids:
        cand = sorted((r["rollout"], i) for i, r in enumerate(rows)
                      if r["scenario_id"] == sid and r.get("answer_idx") is not None
                      and r.get("letter") in (r.get("order") or {}))
        out += [rows[i] for _, i in cand[:2]]
    return out


def phase_predict(c, sc, model, a) -> dict:
    """Section 12: the exact teacher-forced steer at the dose of record, converted per scenario
    by P(1 - P) with v1's P_none; recorded before the arms."""
    from reasoning_formats import letter_token_ids

    from deepsteer.causal import attribution as AT

    rows = _jsonl(c["out"] / "gen" / "h1_rollouts.jsonl")
    lid = {L_: i for i, L_ in letter_token_ids(c["tok"]).items()}
    punct = SG.punct_mask(c["tok"])
    block, vec = c["layer"] - 1, c["alpha"] * c["care"]
    by: dict[str, list[float]] = {}
    for r in first_two_answered(rows, c["engaged"]):
        role = {o.option_id: o.norm_status for o in c["H"][r["scenario_id"]].options}
        L_of = {role[oid]: L_ for L_, oid in r["order"].items()}
        ids = r["prompt_ids"] + r["gen_ids"][: r["answer_idx"]]
        pos = GSI.class_positions(c["fmt"], c["tok"], punct, r["gen_ids"], r["answer_idx"],
                                  len(r["prompt_ids"]), c["cls"])
        metric = AT.option_margin([lid[L_of["violating"]]], [lid[L_of["consistent"]]])
        m0 = GS._margin(model, ids, metric)  # noqa: SLF001
        d = 0.0
        if pos:
            h = GS._steer_hook(model, block, pos, vec)  # noqa: SLF001
            try:
                d = GS._margin(model, ids, metric) - m0  # noqa: SLF001
            finally:
                h.remove()
        by.setdefault(r["scenario_id"], []).append(d)
    pn = T13.p_by_scenario([r for r in rows if r["scenario_id"] in by
                            and r.get("norm_status") is not None])
    per = {s: pn[s] * (1 - pn[s]) * float(np.mean(v)) for s, v in by.items() if s in pn}
    pq = float(np.mean([p * (1 - p) for p in pn.values()])) if pn else 0.0
    return {"status": "ok", "n_scenarios": len(per), "n_traces": sum(len(v) for v in by.values()),
            "margin_by_scenario": {s: float(np.mean(v)) for s, v in by.items()},
            "predicted_dP": T13.boot(list(per.values())),
            "mean_p_one_minus_p": pq,
            "margin_needed_for_gate": T13.GATE_DP / pq if pq > 0 else None,
            "rule": "section 12; the arms run whatever it predicts (the addition, item 3)"}


def phase_spec_fill10(c, sc, model, a) -> dict:
    al, fu = c["alpha"], c["inp"]["fill_unit"]
    items = [c["panel"][i] for i in c["spec_ids"]]
    rows: list[dict] = []
    for g in range(len(fu)):
        rel = f"spec/fill10/draw{g}"
        with C.add_everywhere(model, c["layer"], al * fu[g]):
            L.run_letter_cell(sc, items, "dl_chat_neutral", rel)
        part = c["out"] / rel / "dl_chat_neutral.jsonl"
        rows += [{**json.loads(x), "fill_draw": g, "alpha": al} for x in part.open()]
    sc.save_jsonl("spec/fill10/dl_chat_neutral.jsonl", rows)
    return {"status": "ok", "rows": len(rows), "alpha": al, "draws": len(fu)}


def phase_spec(c, sc, model, a, name: str) -> dict:
    vec = None if name == "none" else c["alpha"] * c["care"]
    with C.add_everywhere(model, c["layer"], vec):
        res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                                f"spec/{name}")
    return {"status": "ok", "alpha": 0.0 if name == "none" else c["alpha"], **res}


def phase_arm(c, sc, model, a, name: str) -> dict:
    al = c["alpha"]
    if name == "fill":
        fu = c["inp"]["fill_unit"]
        order = {s: i for i, s in enumerate(c["engaged"])}
        jobs = L.h1_jobs(sc, c["H"], c["engaged"], min(R_FILL, c["r"]))
        draw = [(order[j["s"].id] * R_FILL + j["k"]) % len(fu) for j in jobs]
        iv = SG.Add(np.stack([al * fu[g] for g in draw]).astype(np.float32), 1.0)
        extra = lambda i: {"alpha": al, "axis": "fill", "fill_draw": draw[i]}  # noqa: E731
    else:
        jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r"])
        iv = None if name == "none" else SG.Add(c["care"], al)
        extra = lambda i: {"alpha": 0.0 if name == "none" else al, "axis": name}  # noqa: E731
    rows = S2.run_arm(c, sc, model, "st2q", f"arm_{name}", jobs, iv, extra=extra)
    out = {"status": "ok", "n": len(rows), "alpha": al,
           "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}
    if name != "none":
        none = _jsonl(c["out"] / "gen" / "arm_none.jsonl")
        keys = {r["rollout_key"] for r in rows}
        out["coherence"] = T13._coh(rows, [r for r in none if r["rollout_key"] in keys],  # noqa: SLF001
                                    relative=True)
    return out


def run_hf(c: dict, a, todo: list[str]) -> int:
    model, sc = None, None
    for name in todo:
        ctx_derived(c)
        if name in ("predict", "spec_care", "arm_none", "arm_care", "arm_fill",
                    "spec_none", "spec_fill10") and phase_done(c, "sweep").get("no_coherent_dose"):
            print(f">> st2q {name}: not run (no coherent dose: P1-fail)", flush=True)
            continue
        if a.deadline and name.startswith("arm_") and time.time() > a.deadline:
            print(f">> st2q {name}: not started (past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if model is None:
            model = load_model(c, a)
            sc = screen(c, model, date_pin(c, a))
        if name == "dose0":
            res = M.phase_dose0(c, sc, model, a)
        elif name == "bank":
            res = M.phase_bank(c, sc, model, a)
        elif name == "derive":
            res = phase_derive(c, sc, model, a)
        elif name == "pilot":
            res = M.phase_pilot(c, sc, model, a)
        elif name == "directions":
            res = phase_directions(c, sc, model, a)
        elif name == "sweep":
            res = phase_sweep(c, sc, model, a)
        elif name == "predict":
            res = phase_predict(c, sc, model, a)
        elif name == "spec_fill10":
            res = phase_spec_fill10(c, sc, model, a)
        elif name.startswith("spec_"):
            res = phase_spec(c, sc, model, a, name[len("spec_"):])
        else:
            res = phase_arm(c, sc, model, a, name[len("arm_"):])
        rec = load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        brief = {k: res[k] for k in ("n", "rows", "n_engaged", "parse_rate", "batch", "alpha",
                                     "dose", "no_coherent_dose", "site_block", "underpowered")
                 if k in res}
        if "coherence" in res:
            brief["coherent"] = res["coherence"]["passes"]
        if name == "predict":
            brief["predicted_dP"] = res["predicted_dP"]["mean"]
        print(f">> st2q {name}: done ({json.dumps(brief, default=str)})", flush=True)
        if name == "pilot":
            ctx_derived(c)
            pr = projection(c, a, time.time())
            rec = load_record(c)
            rec["phases"]["pilot"]["projection"] = pr
            save_record(c, rec)
            print(f">> st2q pilot projection: {pr['projected_pod_hours']:.2f} A100-h "
                  f"(stop above {pr['stop_above']:.2f})", flush=True)
            if pr["stops"] and not a.dry_run:
                print(">> st2q: the pilot projects pod 1 above 1.5x its price; stopping for the "
                      "author before the dose ladder (Amendment 54 addition, item 3)", flush=True)
                return 5
        if name == "sweep" and res.get("no_coherent_dose"):
            print(">> st2q: no coherent dose; P1-fail is recorded and the gate does not run",
                  flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default=",".join(HF_PHASES))
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--session-t0", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    rec = load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    if not todo:
        print(">> st2q: every requested phase finished earlier", flush=True)
        return 0
    if todo == ["v1"]:
        t0 = time.time()
        res = phase_v1(c, a)
        rec = load_record(c)
        rec["phases"]["v1"] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        p = res["summary"].get("P_protect_mean_over_scenarios")
        print(f">> st2q v1: done ({json.dumps({'n': res['n'], 'P': p})})", flush=True)
        return 0
    if "v1" in todo:
        raise SystemExit("v1 runs in its own process (vLLM), before the HF phases")
    return run_hf(c, a, todo)


if __name__ == "__main__":
    sys.exit(main())
