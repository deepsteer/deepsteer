"""ST Phase 2 pod C on GPT-OSS-20B (ST_PLAN Amendment 25; author, 2026-10-22): the safety reads of
-V_harm before anything trains on it, and V_qwen one grid step down. One HF process on the ST stack
of record (st2_driver's model, screen and arm runner), resumable phase by phase.

- ``rr_none`` / ``rr_neg_V_harm`` (item 2(a)): the 60 plain violating members of
  ``data/refusal_retention_v1.json`` x 4 rollouts, seeds by (item, rollout) shared across arms,
  generated on HF with the arm's vector added at ``clause_end``, block 12 (alpha 267.3); the final
  answer's first 64 tokens read by FL's ``is_refusal`` (``refusal_retention.final_answer``). Cap:
  the trace cap of record plus the 64 answer tokens.
- ``spec_none`` / ``spec_neg_V_harm`` (item 2(b)): the 60 rule-right panel items, dose 0,
  ``dl_chat_neutral``, 8 permutations, single-row (T1.5's instrument), with the vector added at
  every input position at block 12. T1.3c's ``none`` rows are the baseline of record; ``spec_none``
  re-reads them on this pod (its prompts must hash to T1.3c's).
- ``rr_neg_refusal`` (the rider's positive control): Paper 7's refusal direction subtracted at the
  same site and alpha on the same 60 x 4; it has to raise compliance for a pass to read as
  retention.
- conditionals (Amendment 26): ``rr_fill`` (Cell 1's five fill draws, cycled by rollout, alpha
  267.3, the same site) only if -V_harm fails the retention bar; ``rr_ablate_refusal`` (the refusal
  direction projected out of every block's output at every position, Paper 7's recipe) only if the
  rider's control does not raise compliance.
- ``qwen_half`` (item 1): V_qwen at alpha 133.6 on the 45 engaged H1_train x 8, Cell 1's seeds
  and date pin; coherence against Cell 1's ``none`` (Amendment 13's relative bars) decides whether
  ``rr_V_qwen`` and ``spec_V_qwen`` (item 3, at 133.6) run. V_qwen's reads run last.

The analysis of record is local (``analyze_st2c.py``); the pod measures.

    python3 st2c_driver.py --out <dir> --inputs <st2_inputs dir> --podc-inputs <st2c_inputs dir> \
        --c1 <pod A's gpt_oss_20b dir> [--phases ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import refusal_retention as RR  # noqa: E402
import screen_lib as L  # noqa: E402
import st2_driver as S2  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

KEY = S2.KEY
DATA = HERE.parent / "data"
RR_STAGE = "st2rr"
RR_ARMS = ("none", "neg_V_harm", "neg_refusal", "fill", "ablate_refusal", "V_qwen")
SPEC_ARMS = ("none", "neg_V_harm", "V_qwen")
PODC = ["rr_none", "rr_neg_V_harm", "spec_none", "spec_neg_V_harm", "rr_neg_refusal",
        "rr_fill", "rr_ablate_refusal", "qwen_half", "rr_V_qwen", "spec_V_qwen"]
GATED = {"rr_V_qwen", "spec_V_qwen"}  # item 3: only if qwen_half passes coherence
CONDITIONAL = {"rr_fill", "rr_ablate_refusal"}  # Amendment 26


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    from deepsteer.kdg.schema import load_scenario_dir

    a.t11, a.qwen_screen = None, None
    c = S2.context(a)
    P = c["P"]
    allp, _ = load_scenario_dir(sorted(glob.glob(str(P.KDG / "data" / "*_scenarios_*.json"))))
    c["panel"] = {s.id: s for s in L.panel_586(allp)}
    c["spec_ids"] = json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"]
    c["rr_items"] = [it for it in RR.load_set()["items"]
                     if it["member"] == "violating" and it["framing"] == "plain"]
    c["rr_r"] = RR.ROLLOUTS
    z = np.load(a.podc_inputs / "st2c_inputs.npz")
    c["podc"] = {k: z[k] for k in z.files}
    c["record"] = c["out"] / "st2c_record.json"
    if a.dry_run:
        c.update(spec_ids=c["spec_ids"][:2], rr_items=c["rr_items"][:2], rr_r=2)
    return c


def load_model(c: dict, a):
    m = S2.load_model(c, a)
    if a.dry_run:  # the pod C vectors at the tiny width
        h = m.model.config.hidden_size
        rng = np.random.default_rng(1)
        c["podc"] = {k: (v / np.linalg.norm(v)).astype(np.float32)
                     for k, v in (("V_qwen", rng.standard_normal(h)),
                                  ("refusal", rng.standard_normal(h)))}
    return m


def screen(c: dict, model, a, date_pin: str) -> L.Screen:
    """T1.5's screen (single-row readouts of record, layers at L*), at Cell 1's date pin."""
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], date_pin, dry=False, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    sc.unbatched = True
    sc.layers_override = [2] if a.dry_run else [S2.L_STAR]
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def load_record(c: dict) -> dict:
    return json.loads(c["record"].read_text()) if c["record"].exists() else {"model": KEY,
                                                                            "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    c["record"].write_text(json.dumps(rec, indent=1, default=L._json_default))  # noqa: SLF001


def qwen_coherent(c: dict) -> bool:
    """Item 3's gate: V_qwen's reads run only once ``qwen_half`` has passed coherence."""
    return bool((load_record(c)["phases"].get("qwen_half") or {}).get("coherent"))


def skip_reason(c: dict, name: str) -> str | None:
    """None if the phase runs; else why it is skipped (Amendment 25 item 3, Amendment 26)."""
    ph = load_record(c)["phases"]
    if name in GATED:
        return None if qwen_coherent(c) else ("skipped: V_qwen at 133.6 did not pass coherence "
                                              "(or did not run)")
    if name == "rr_fill":
        x = (ph.get("rr_neg_V_harm") or {}).get("plain_vs_none")
        if x is None:
            return "skipped: rr_neg_V_harm has no read"
        return None if not x["passes"] else "skipped: -V_harm passed the retention bar"
    if name == "rr_ablate_refusal":
        x = (ph.get("rr_neg_refusal") or {}).get("plain_vs_none")
        if x is None:
            return "skipped: rr_neg_refusal has no read"
        return None if not x["raises"] else "skipped: the rider's control raised compliance"
    return None


def vector(c: dict, arm: str) -> tuple[np.ndarray | None, float]:
    """The arm's added vector and its alpha: -V_harm and -refusal at 267.3, V_qwen at 133.6."""
    if arm == "none":
        return None, 0.0
    if arm == "neg_V_harm":
        return -c["inp"]["V_harm"], c["alpha"]
    if arm == "neg_refusal":
        return -c["podc"]["refusal"], c["alpha"]
    if arm == "V_qwen":
        return c["podc"]["V_qwen"], c["alpha"] / 2  # Amendment 25 item 1: one grid step down
    raise ValueError(arm)


def rr_intervention(c: dict, arm: str, jobs: list[dict]) -> tuple[object, float | None]:
    """The decode-hook intervention at ``clause_end`` (None for ``none`` and for the every-position
    ablation, which hooks every block itself) and the alpha recorded on the rows."""
    if arm in ("fill", "fill_half"):  # Cell 1's draws, cycled by rollout (S2.c1_intervention's
        fu = c["inp"]["fill_unit"]    # rule); fill_half at alpha / 2 (Amendment 29, the D-lite)
        al = c["alpha"] / 2 if arm == "fill_half" else c["alpha"]
        return SG.Add(np.stack([al * fu[j["k"] % len(fu)] for j in jobs])
                      .astype(np.float32), 1.0), al
    if arm == "ablate_refusal":
        return None, None
    v, al = vector(c, arm)
    return (None if v is None else SG.Add(v, al)), al


@contextmanager
def ablate_everywhere(model, vec: np.ndarray | None):
    """Project ``vec`` (unit) out of every block's output at every position (prompt, trace and
    answer), in float32: Paper 7's directional ablation (``ablate_layers: all``). Zeroing the
    component is off-distribution; it is used here as a positive control for the readout, never
    as a mechanism read."""
    import torch

    if vec is None:
        yield
        return
    hf = model.model
    r = torch.as_tensor(vec, dtype=torch.float32, device=hf.device)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        x = h.float()
        x = x - (x @ r)[..., None] * r
        h = x.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    n = hf.config.num_hidden_layers
    handles = [model.wb._get_layer_module(i).register_forward_hook(hook)  # noqa: SLF001
               for i in range(n)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def rr_jobs(c: dict, sc) -> list[dict]:
    jobs = []
    for it in c["rr_items"]:
        ids = RR.prompt_ids(sc, it)
        for k in range(c["rr_r"]):
            key = f"{RR_STAGE}|{it['id']}|{k}"
            jobs.append({"job_id": key, "item": it["id"], "k": k, "ids": ids,
                         "seed": zlib.crc32(key.encode()) % (2**31)})
    return jobs


def phase_rr(c, sc, model, rec, a, arm: str) -> dict:
    jobs = rr_jobs(c, sc)
    iv, al = rr_intervention(c, arm, jobs)
    raw = c["out"] / "rr" / f"{arm}.raw.jsonl"
    raw.unlink(missing_ok=True)
    cap = c["cap"] + RR.ANSWER_TOKENS
    t0 = time.time()
    with ablate_everywhere(model, c["podc"]["refusal"] if arm == "ablate_refusal" else None):
        gen = SG.generate(model, [j["ids"] for j in jobs], fmt=c["fmt"], cap=cap,
                          seeds=[j["seed"] for j in jobs],
                          layer=c["layer"] if iv is not None else None,
                          cls=c["cls"] if iv is not None else None, intervention=iv,
                          batch=c["batch"], on_finish=SG.jsonl_saver(raw))
    rows = []
    for j, g in zip(jobs, gen):
        text = RR.final_answer(sc, g["gen_ids"])
        rows.append({"job_id": j["job_id"], "item": j["item"], "rollout": j["k"], "seed": j["seed"],
                     "arm": arm, "alpha": al, "answer": text,
                     "refusal": None if text is None else RR.is_refusal(text),
                     "n_tokens": g["n_tokens"], "hit_cap": g["hit_cap"],
                     "n_steered": g["n_steered"], "trace_start": g["trace_start"],
                     "gen_ids": g["gen_ids"], "date_pin": sc.date_pin,
                     **({"fill_draw": j["k"] % len(c["inp"]["fill_unit"])}
                        if arm in ("fill", "fill_half") else {})})
    sc.save_jsonl(f"rr/{arm}.jsonl", rows)
    read = [r["refusal"] for r in rows if r["refusal"] is not None]
    print(f">> st2c rr_{arm}: {len(rows)} rollouts in {time.time() - t0:.0f}s, refusal "
          f"{np.mean(read) if read else float('nan'):.3f}", flush=True)
    out = {"status": "ok", "n": len(rows), "alpha": al, "cap": cap, "pin": RR.readout_pin(),
           "intervention": ("refusal direction projected out of every block's output, every "
                            "position" if arm == "ablate_refusal" else
                            None if iv is None
                            else f"add at {c['cls']}, hidden index {c['layer']}"),
           "missing_readout": float(np.mean([r["refusal"] is None for r in rows])),
           "refusal_rate": float(np.mean(read)) if read else None,
           "mean_n_steered": float(np.mean([r["n_steered"] for r in rows]))}
    if arm != "none":
        none = [json.loads(x) for x in (c["out"] / "rr" / "none.jsonl").open()]
        if [r["job_id"] for r in none] != [r["job_id"] for r in rows]:
            raise SystemExit(f"rr_{arm} rows do not pair with rr_none")
        out["plain_vs_none"] = RR.plain_read(rows, none)
    return out


@contextmanager
def add_everywhere(model, layer: int, vec: np.ndarray | None):
    """``h += vec`` at every position of block ``layer - 1``'s output (item 2(b): the vector at
    every input position of the dose-0 read)."""
    import torch

    if vec is None:
        yield
        return
    hf = model.model
    v = torch.as_tensor(vec, dtype=hf.dtype, device=hf.device)

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    handle = model.wb._get_layer_module(layer - 1).register_forward_hook(hook)  # noqa: SLF001
    try:
        yield
    finally:
        handle.remove()


def phase_spec(c, sc, model, rec, a, arm: str) -> dict:
    v, al = vector(c, arm)
    with add_everywhere(model, c["layer"], None if v is None else al * v):
        res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                                f"spec/{arm}")
    return {"status": "ok", "alpha": al, **res}


def phase_qwen_half(c, sc, model, rec, a) -> dict:
    v, al = vector(c, "V_qwen")
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r2"])
    rows = S2.run_arm(c, sc, model, "st2c1", "c3_qwen_half", jobs, SG.Add(v, al),
                      extra=lambda i: {"cell": 3, "alpha": al, "axis": "V_qwen"})
    none = [json.loads(x) for x in c["c1_none"].open()]
    if not {r["rollout_key"] for r in rows} <= {r["rollout_key"] for r in none}:
        raise SystemExit("c3_qwen_half rows do not pair with Cell 1's none")
    coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
    return {"status": "ok", "n": len(rows), "alpha": al,
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows])),
            "coherence": coh, "coherent": bool(coh["passes"]),
            "dP_vs_none_descriptive": T13.delta_p(rows, none),
            "rule": "Amendment 25 item 1: the read at 133.6 is of record; incoherent again -> not "
                    "informative, and items 3's reads do not run"}


PHASES = {**{f"rr_{arm}": (lambda arm: lambda c, sc, m, rec, a: phase_rr(c, sc, m, rec, a, arm))(
             arm) for arm in RR_ARMS},
          **{f"spec_{arm}": (lambda arm: lambda c, sc, m, rec, a: phase_spec(c, sc, m, rec, a,
                                                                            arm))(arm)
             for arm in SPEC_ARMS},
          "qwen_half": phase_qwen_half}
GEN_KIND = {**{f"rr_{arm}": "rr" for arm in RR_ARMS}, "qwen_half": "h1"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True, help="the st2_inputs dir (V_harm)")
    ap.add_argument("--podc-inputs", type=Path, required=True, help="the st2c_inputs dir")
    ap.add_argument("--c1", type=Path, required=True, help="pod A's output dir (Cell 1's none)")
    ap.add_argument("--phases", default=",".join(PODC))
    ap.add_argument("--date-pin", default=None, help="unused: Cell 1's none sets the pin")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    pin = S2.c1_date_pin(c)  # every arm renders with Cell 1's pin (qwen_half pairs with its none)
    rec = load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    model, sc = None, None
    per: dict[str, list[float]] = {"rr": [], "h1": []}  # measured seconds per rollout, by kind

    def n_rollouts(name: str) -> int:
        return (len(c["rr_items"]) * c["rr_r"] if GEN_KIND.get(name) == "rr"
                else len(c["engaged"]) * c["r2"])

    for name in todo:
        why = skip_reason(c, name)
        if why is not None:
            rec = load_record(c)
            rec["phases"][name] = {"status": why, "finished": _now()}
            save_record(c, rec)
            print(f">> st2c {name}: {why}", flush=True)
            continue
        kind = GEN_KIND.get(name)
        est = float(np.mean(per[kind])) * n_rollouts(name) if kind and per[kind] else 0.0
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2c {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if model is None:
            model = load_model(c, a)
            sc = screen(c, model, a, pin)
        res = PHASES[name](c, sc, model, rec, a)
        if kind:
            per[kind].append((time.time() - t0) / n_rollouts(name))
        rec = load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        brief = {k: res[k] for k in ("n", "refusal_rate", "parse_rate", "coherent", "rows")
                 if k in res}
        if "plain_vs_none" in res:
            pv = res["plain_vs_none"]
            brief["dcompliance"] = pv["dcompliance_plain"]["mean"]
            brief["passes"], brief["raises"] = pv["passes"], pv["raises"]
        print(f">> st2c {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
