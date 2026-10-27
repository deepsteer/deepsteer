"""The GPT-OSS site sweep (ST_PLAN Amendment 51 and its additions; ST-A29's next step). One HF
process on pod C's stack, resumable phase by phase.

- ``analogs``: `care_harm`'s analog at blocks 6, 9, 12, 15 and 18 by W4's recipe (MFT v2
  care/harm, moral minus neutral, raw text, mean-pooled; ``st2m_driver``'s code), unit. Parity bar:
  block 12's analog has cos >= 0.99 with the `care_harm` of record (``st2_inputs``); below it the
  pod stops here.
- ``rank``: on the 90 teacher-forced traces (``build_st2gs_inputs``), per trace: the option margin
  (violating minus consistent letter, at the answer letter's position) unsteered; at each of the ten
  sites, the exact steer (the analog at alpha_s added to block b's output at the class positions,
  the trace fixed), the ranking of record; AtP (the margin's gradient at the class positions, dotted
  with alpha_s times the analog, and with alpha_s times each of the site's ten fill draws), the
  validation. The top two eligible sites by the exact steer's scenario mean are written to the
  record. An out-of-memory backward pass leaves AtP not computed; the ranking stands.
- ``spec_none``: the specificity set at dose 0 (parity re-read).
- per top site s: ``arm_care_<s>_04`` (Cell 1's 45 engaged x 16, Cell 1's seeds),
  ``arm_fill_<s>_04`` (x 8, the site's ten draws cycled), ``spec_care_<s>_04`` and
  ``spec_fill10_<s>_04`` (the 60 rule-right items, the vector at every input position of block b).
- then, per top site whose care arm failed a coherence bar, the same four phases at 0.3x, if their
  cost projected from this session's measured phase times ends before the stop (Amendment 51,
  second addition); otherwise recorded as not run.

    python3 st2gs_driver.py --out <dir> --inputs <st2_inputs dir> --podc-inputs <st2c_inputs dir> \
        --c1 <pod A's gpt_oss_20b dir> --gs <st2gs_inputs dir> [--dry-run] [--deadline <epoch>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))
import screen_lib as L  # noqa: E402
import st2_driver as S2  # noqa: E402
import st2c_driver as C  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

BLOCKS = (6, 9, 12, 15, 18)
PARITY_BAR = 0.99
RUNGS = ("04", "03")
R_CARE, R_FILL = 16, 8  # Cell 1's rollouts 0-15 (the care arm) and 0-7 (the fill arm)
SITE_PHASES = ("arm_care", "arm_fill", "spec_care", "spec_fill10")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    c = C.context(a)
    c["record"] = c["out"] / "st2gs_record.json"
    doc = json.loads((a.gs / "sites.json").read_text())
    c["sites"], c["candidates"] = doc["sites"], doc["candidates"]
    z = np.load(a.gs / "fill_sites.npz")
    c["fill_sites"] = {k: z[k] for k in z.files}
    c["traces"] = [json.loads(x) for x in (a.gs / "traces.jsonl").open()]
    c["blocks"] = BLOCKS
    return c


def dry_inputs(c: dict, model) -> None:
    """The tiny model's stand-ins: two sites at blocks 1 and 2 (``onset``: a random model writes
    no punctuation), random fill draws at its width, and two traces cut to 24 trace tokens."""
    h = model.model.config.hidden_size
    rng = np.random.default_rng(5)
    c["blocks"] = (1, 2)
    c["sites"] = {f"b{b}_onset": {"block": b, "cls": "onset", "hidden_index": b + 1,
                                  "eligible": True, "alpha": {"04": 1.0, "03": 0.75}}
                  for b in (1, 2)}
    c["candidates"] = list(c["sites"])
    c["fill_sites"] = {k: SG.fill_draws(rng.standard_normal((64, h)), 1.0, 10, 3)
                       for k in c["sites"]}
    tr = []
    for t in c["traces"][:2]:
        n = t["prompt_len"]
        pos = list(range(n, n + 24))
        tr.append({**t, "ids": t["ids"][: n + 24], "positions": {"onset": pos,
                                                                 "clause_end": pos}})
    c["traces"] = tr
    c["care_max"] = 3


def phase_analogs(c: dict, model, a) -> dict:
    from deepsteer.datasets import load_moral_probing_v2
    from deepsteer.directions import extraction as du

    pairs = [(p.moral, p.neutral) for p in load_moral_probing_v2().all_pairs
             if p.foundation.value == "care_harm"]
    if c.get("care_max"):
        pairs = pairs[: c["care_max"]]
    acts = du.collect_pair_activations(model.wb, pairs, input_format="raw",
                                       layers=list(c["blocks"]))
    raw = {b: du.mean_diff_direction(*acts[b]) for b in c["blocks"]}
    unit = {f"block{b}": (v / np.linalg.norm(v)).astype(np.float32) for b, v in raw.items()}
    np.savez(c["out"] / "analogs.npz", **unit)
    ref = c["inp"]["care_harm"] / np.linalg.norm(c["inp"]["care_harm"])
    cos = None if a.dry_run else float(unit["block12"] @ ref)
    ok = a.dry_run or cos >= PARITY_BAR
    return {"status": "ok", "n_pairs": len(pairs), "parity_cos_block12": cos,
            "parity_ok": bool(ok), "bar": PARITY_BAR,
            "cos_between_blocks": {f"{b1}-{b2}": float(unit[f"block{b1}"] @ unit[f"block{b2}"])
                                   for i, b1 in enumerate(c["blocks"])
                                   for b2 in c["blocks"][i + 1:]}}


def analog(c: dict, block: int) -> np.ndarray:
    return np.load(c["out"] / "analogs.npz")[f"block{block}"]


def _steer_hook(model, block: int, pos: list[int], vec: np.ndarray):
    import torch

    hf = model.model
    v = torch.as_tensor(vec, dtype=hf.dtype, device=hf.device)
    idx = torch.as_tensor(pos, dtype=torch.long, device=hf.device)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[0, idx] = h[0, idx] + v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return model.wb._get_layer_module(block).register_forward_hook(hook)  # noqa: SLF001


def _margin(model, ids: list[int], metric) -> float:
    import torch

    with torch.no_grad():
        x = torch.tensor([ids], device=model.model.device)
        return float(metric(model.model(input_ids=x).logits[0, -1]))


def phase_rank(c: dict, model, a) -> dict:
    """Per trace: the unsteered margin, the exact steer and AtP at every site (item 4; the first
    addition's items 2-3)."""
    import torch

    from deepsteer.causal import attribution as AT

    names = list(c["sites"])
    vecs = {s: analog(c, c["sites"][s]["block"]) for s in names}
    al = {s: c["sites"][s]["alpha"]["04"] for s in names}
    n = len(c["traces"])
    m0, m_grad = np.zeros(n), np.zeros(n)
    exact = np.zeros((n, len(names)))
    atp = np.full((n, len(names)), np.nan)
    atp_fill = np.full((n, len(names), 10), np.nan)
    atp_note = "computed"
    for i, t in enumerate(c["traces"]):
        metric = AT.option_margin([t["violating_id"]], [t["consistent_id"]])
        m0[i] = _margin(model, t["ids"], metric)
        for j, s in enumerate(names):
            st = c["sites"][s]
            pos = t["positions"][st["cls"]]
            if not pos:
                exact[i, j] = 0.0
                continue
            h = _steer_hook(model, st["block"], pos, al[s] * vecs[s])
            try:
                exact[i, j] = _margin(model, t["ids"], metric) - m0[i]
            finally:
                h.remove()
        if atp_note != "computed":
            continue
        try:
            m, resid, _, _ = AT._run(model.wb, t["ids"], metric, grad=True)  # noqa: SLF001
        except torch.cuda.OutOfMemoryError:  # registered bail (first addition, item 3)
            atp_note = f"not computed: out of memory on trace {i} (length {len(t['ids'])})"
            torch.cuda.empty_cache()
            continue
        m_grad[i] = m
        for j, s in enumerate(names):
            st = c["sites"][s]
            pos = t["positions"][st["cls"]]
            g = resid[st["block"]].grad[0, pos].float().sum(0).cpu().numpy() if pos else 0.0
            atp[i, j] = al[s] * float(vecs[s] @ g) if pos else 0.0
            atp_fill[i, j] = al[s] * (c["fill_sites"][s] @ g) if pos else 0.0
        del resid
    np.savez(c["out"] / "rank.npz", sites=np.array(names), m0=m0, m_grad=m_grad, exact=exact,
             atp=atp, atp_fill=atp_fill,
             scenario=np.array([t["scenario_id"] for t in c["traces"]]),
             rollout=np.array([t["rollout"] for t in c["traces"]]))
    sid = [t["scenario_id"] for t in c["traces"]]
    score = {}
    for j, s in enumerate(names):
        per = {}
        for k, x in zip(sid, exact[:, j]):
            per.setdefault(k, []).append(x)
        score[s] = T13.boot([float(np.mean(v)) for v in per.values()])
    ranked = sorted((s for s in c["candidates"]), key=lambda s: -score[s]["mean"])
    lp = [t["t11_option_logps"] for t in c["traces"]]
    parity = [abs(m0[i] - (lp[i][t["violating_letter"]] - lp[i][t["consistent_letter"]]))
              for i, t in enumerate(c["traces"])]
    return {"status": "ok", "n_traces": n, "exact_by_site": score, "ranking": ranked,
            "top2": ranked[:2], "atp": atp_note,
            "t11_margin_parity": {"max_abs": float(np.max(parity)) if parity else None,
                                  "median_abs": float(np.median(parity)) if parity else None}}


def site_ctx(c: dict, site: str) -> dict:
    st = c["sites"][site]
    return {**c, "layer": st["hidden_index"], "cls": st["cls"]}


def phase_arm_care(c, sc, model, site: str, rung: str) -> dict:
    cs = site_ctx(c, site)
    al = c["sites"][site]["alpha"][rung]
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], min(R_CARE, c["r1"]))
    rows = S2.run_arm(cs, sc, model, "st2c1", f"gs_care_{site}_{rung}", jobs,
                      SG.Add(analog(c, c["sites"][site]["block"]), al),
                      extra=lambda i: {"alpha": al, "site": site, "rung": rung})
    return _coherence(c, rows, al)


def phase_arm_fill(c, sc, model, site: str, rung: str) -> dict:
    cs = site_ctx(c, site)
    al = c["sites"][site]["alpha"][rung]
    fu = c["fill_sites"][site]
    order = {s: i for i, s in enumerate(c["engaged"])}
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], min(R_FILL, c["r1"]))
    draw = [(order[j["s"].id] * R_FILL + j["k"]) % len(fu) for j in jobs]
    rows = S2.run_arm(cs, sc, model, "st2c1", f"gs_fill_{site}_{rung}", jobs,
                      SG.Add(np.stack([al * fu[g] for g in draw]).astype(np.float32), 1.0),
                      extra=lambda i: {"alpha": al, "site": site, "rung": rung,
                                       "fill_draw": draw[i]})
    return _coherence(c, rows, al)


def _coherence(c: dict, rows: list[dict], al: float) -> dict:
    none = [json.loads(x) for x in c["c1_none"].open()]
    keys = {r["rollout_key"] for r in rows}
    none = [r for r in none if r["rollout_key"] in keys]
    if keys != {r["rollout_key"] for r in none}:
        raise SystemExit("sweep rows do not pair with Cell 1's none")
    coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
    return {"status": "ok", "n": len(rows), "alpha": al, "coherence": coh,
            "coherent": bool(coh["passes"]),
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def phase_spec_care(c, sc, model, site: str, rung: str) -> dict:
    st = c["sites"][site]
    al = st["alpha"][rung]
    with C.add_everywhere(model, st["hidden_index"], al * analog(c, st["block"])):
        res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                                f"spec/care_{site}_{rung}")
    return {"status": "ok", "alpha": al, **res}


def phase_spec_fill10(c, sc, model, site: str, rung: str) -> dict:
    st = c["sites"][site]
    al, fu = st["alpha"][rung], c["fill_sites"][site]
    items = [c["panel"][i] for i in c["spec_ids"]]
    rows: list[dict] = []
    for g in range(len(fu)):
        rel = f"spec/fill10_{site}_{rung}/draw{g}"
        with C.add_everywhere(model, st["hidden_index"], al * fu[g]):
            L.run_letter_cell(sc, items, "dl_chat_neutral", rel)
        part = c["out"] / rel / "dl_chat_neutral.jsonl"
        rows += [{**json.loads(x), "fill_draw": g, "alpha": al} for x in part.open()]
    sc.save_jsonl(f"spec/fill10_{site}_{rung}/dl_chat_neutral.jsonl", rows)
    return {"status": "ok", "rows": len(rows), "alpha": al, "draws": len(fu)}


PHASES = {"arm_care": phase_arm_care, "arm_fill": phase_arm_fill, "spec_care": phase_spec_care,
          "spec_fill10": phase_spec_fill10}


def site_phases(site: str, rung: str) -> list[str]:
    return [f"{p}|{site}|{rung}" for p in SITE_PHASES]


def fallback_plan(rec: dict, top2: list[str], deadline: float | None, now: float) -> dict:
    """Per top site whose 0.4x care arm failed a coherence bar: run 0.3x if the four phases'
    measured 0.4x times, summed, end before the stop (Amendment 51, second addition, item 2)."""
    out = {}
    for s in top2:
        if (rec.get(f"arm_care|{s}|04") or {}).get("coherent", True):
            continue
        cost = sum((rec.get(p) or {}).get("seconds", 0.0) for p in site_phases(s, "04"))
        fits = deadline is None or now + cost <= deadline
        out[s] = {"projected_seconds": cost, "runs": bool(fits)}
        now += cost if fits else 0.0
    return out


def run_phase(name: str, c, sc, model, a) -> dict:
    p, site, rung = name.split("|")
    return PHASES[p](c, sc, model, site, rung)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="the st2_inputs dir")
    ap.add_argument("--podc-inputs", type=Path, required=True, help="the st2c_inputs dir")
    ap.add_argument("--c1", type=Path, required=True, help="pod A's output dir (Cell 1's none)")
    ap.add_argument("--gs", type=Path, required=True, help="the st2gs_inputs dir")
    ap.add_argument("--date-pin", default=None, help="unused: Cell 1's none sets the pin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    c["dry"] = a.dry_run
    pin = S2.c1_date_pin(c)
    model = C.load_model(c, a)
    if a.dry_run:
        dry_inputs(c, model)
    sc = None

    def done(name: str) -> dict:
        return C.load_record(c)["phases"].get(name) or {}

    def record(name: str, res: dict, t0: float) -> dict:
        rec = C.load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        C.save_record(c, rec)
        brief = {k: res[k] for k in ("n", "rows", "alpha", "parse_rate", "coherent", "top2",
                                     "parity_cos_block12", "atp") if k in res}
        print(f">> st2gs {name}: done ({json.dumps(brief, default=str)})", flush=True)
        return rec["phases"][name]

    for name, fn in (("analogs", lambda: phase_analogs(c, model, a)),
                     ("rank", lambda: phase_rank(c, model, a))):
        if not done(name).get("finished"):
            t0 = time.time()
            record(name, fn(), t0)
        if name == "analogs" and not done("analogs").get("parity_ok"):
            print(">> st2gs: block 12's analog fails the parity bar; the pod stops before the "
                  "ranking (Amendment 51 item 3)", flush=True)
            return 4
    top2 = done("rank")["top2"]
    todo = ["spec_none"] + [p for s in top2 for p in site_phases(s, "04")]
    for s, p in (done("fallback").get("plan") or {}).items():  # a resumed pod keeps the plan
        todo += site_phases(s, "03") if p["runs"] else []
    while todo:
        name = todo.pop(0)
        if not done(name).get("finished"):
            if a.deadline and name.startswith("arm_") and time.time() > a.deadline:
                print(f">> st2gs {name}: not started (past the deadline)", flush=True)
                return 3
            t0 = time.time()
            if sc is None:
                sc = C.screen(c, model, a, pin)
            res = (C.phase_spec(c, sc, model, None, a, "none") if name == "spec_none"
                   else run_phase(name, c, sc, model, a))
            record(name, res, t0)
        if not todo and not done("fallback").get("finished"):
            plan = fallback_plan(C.load_record(c)["phases"], top2, a.deadline, time.time())
            record("fallback", {"status": "ok", "plan": plan}, time.time())
            for s, p in plan.items():
                print(f">> st2gs: {s} failed a coherence bar at 0.4x; 0.3x "
                      f"{'runs' if p['runs'] else 'not run (projected past the stop)'}", flush=True)
                if p["runs"]:
                    todo += site_phases(s, "03")
    return 0


if __name__ == "__main__":
    sys.exit(main())
