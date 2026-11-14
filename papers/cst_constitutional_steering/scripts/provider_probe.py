#!/usr/bin/env python3
"""A second provider's compatibility probe (CST Amendment 51 item A), provider-neutral.

    provider_probe.py --stage <stage> --out DIR [--minutes M]

- ``hf_fixed`` (A5): the parity stage's HF letters for the base and the fixed smoke adapter
  (``pilot_pod_lib.hf_letters``), each against its saved letters (TV). On RunPod the HF base
  letters are bit-identical across 19 sessions (TV 0.0), so a nonzero TV is reported beside the
  serving floor.
- ``tput --mode plain|doc`` (A6): throughput at saturation. The teacher check's MACHIAVELLI
  workloads on the same code, kept saturated by a continuous refill: ``plain`` (the base's and
  GOOD_BEHAVIOR's prompts on the 1,698, the engine of record, 128 sequences; RunPod 638 tokens/s on
  the teacher pod, the finalist MACHIAVELLI plateau about 500) and ``doc`` (the constitution's
  prompts on the nested 888, A0s's engine at 32 sequences; RunPod 303 tokens/s on the teacher pod,
  A0s on MoReBench 377). Generated tokens are binned by minute; the record holds the mean after a
  2-minute warm-up and the worst 5-minute window.
- ``train50`` (A7): 50 steps of P-ctx's trainer (``cst_pressure_train_pod._train`` on the first 200
  of P-ctx's training examples, 4 per step), timed apart from the model load; RunPod's 200 steps
  took 1.485 h with the load (the 2-ctx pod).
- ``verdict`` (A9, local): the verdict line from the records and the reported price.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2]), str(HERE.parents[1] / "kdg_panel/scripts"),
                str(HERE.parents[1] / "st_live_reference/scripts"),
                str(HERE.parents[1] / "mrb_morebench/scripts"),
                str(HERE.parents[1] / "mch_machiavelli/scripts")]

CST = HERE.parent
RUNPOD = {"plain_tps": 638.1, "doc_tps": 303.1, "plain_plateau_mch": 500.0, "doc_mrb_a0s": 377.0,
          "train_200_steps_h_with_load": 1.485, "secure_price": 2.09}
PRICES = {"hyperstack": 1.35, "runpod_secure": 2.09}
WARMUP_MIN, WINDOW_MIN = 2, 5


def windows(t0: float, events: list[tuple[float, int]], minutes: float) -> dict:
    """Tokens per second by minute bin; the mean after the warm-up and the worst 5-minute window
    (rolling, whole bins)."""
    n = int(minutes)
    bins = [0] * n
    for t, k in events:
        i = int((t - t0) // 60)
        if 0 <= i < n:
            bins[i] += k
    tps = [b / 60 for b in bins]
    body = tps[WARMUP_MIN:]
    roll = [sum(body[i:i + WINDOW_MIN]) / WINDOW_MIN for i in range(len(body) - WINDOW_MIN + 1)]
    mean = sum(body) / len(body) if body else 0.0
    worst = min(roll) if roll else None
    return {"per_minute_tps": tps, "mean_tps": mean, "worst_5min_tps": worst,
            "worst_over_mean": (worst / mean) if worst is not None and mean else None}


def prompts(mode: str, dry: bool):
    """The teacher check's prompts and choice counts for the workload (the states rebuilt by the
    replay gate on this host)."""
    import cst_teacher_pod as TP
    import reasoning_formats as rf

    F = TP.F
    which, arms = (("wide", ("base", "good_behavior")) if mode == "plain"
                   else ("nested", ("constitution",)))
    sts, dropped = TP.states(Path("."), which, dry)
    body = F.S.doc_body("a0s", dry) if mode == "doc" else None
    path, eng = F.S.subject(dry, "a0s" if mode == "doc" else "a0")
    tok = F.S.V._tok(path)  # noqa: SLF001
    fmt = rf.build_format(F.MODEL, tok)
    jobs = [{"sid": st["sid"], "arm": a, "n": st["n"],
             "ids": tok.encode(fmt.render(tok, TP.messages(a, st, body), F.MACH_EFFORT),
                               add_special_tokens=False)} for st in sts for a in arms]
    return eng, jobs, dropped


def stage_tput(out: Path, mode: str, minutes: float, dry: bool) -> dict:
    """A continuous refill at the engine's own sequence cap: each finished request is replaced by
    the next prompt (cycling, k rising), MCH's sampler and the 4,096 budget, the teacher check's
    request seeds."""
    import cst_teacher_pod as TP
    import mch_thinking_pod as MT
    from mch_newmodel_pod import request_seed

    F = TP.F
    eng, jobs, dropped = prompts(mode, dry)
    if dry:  # the stand-in has no request-level engine: one batch stands for the loop
        t0 = time.time()
        gens = eng.generate([j["ids"] for j in jobs[:8]], 8)
        ev = [(time.time(), len(g["ids"])) for g in gens]
        rec = {"mode": mode, "dry": True, "requests": len(gens), **windows(t0, ev, 1)}
        F.T._w(out, f"tput_{mode}", rec)  # noqa: SLF001
        return rec
    from vllm import SamplingParams

    engine = eng.llm.llm_engine
    cap = F.S.engine_kwargs("a0s" if mode == "doc" else "a0").get("max_num_seqs", 128)
    state, nxt, seen, ev, done_n = {}, 0, {}, [], 0
    t0 = time.time()

    def add():
        nonlocal nxt
        j = jobs[nxt % len(jobs)]
        k = nxt // len(jobs)
        rid = f"{j['arm']}|{j['sid']}|{k}"
        sp = SamplingParams(temperature=MT.SAMPLER["temperature"], top_p=MT.SAMPLER["top_p"],
                            top_k=0, min_p=0.0, max_tokens=F.MACH_TOKENS,
                            seed=request_seed(F.MODEL, "teacher", j["sid"], k, 1))
        engine.add_request(rid, {"prompt_token_ids": j["ids"]}, sp)
        state[rid] = 0
        nxt += 1

    for _ in range(min(cap, len(jobs) * 4)):
        add()
    while time.time() - t0 < minutes * 60:
        for r in engine.step():
            n = len(r.outputs[0].token_ids) if r.outputs else 0
            d = n - seen.get(r.request_id, 0)
            seen[r.request_id] = n
            if d:
                ev.append((time.time(), d))
            if r.finished:
                state.pop(r.request_id, None)
                seen.pop(r.request_id, None)
                done_n += 1
                add()
    for rid in list(state):
        engine.abort_request([rid])
    rec = {"rule": "Amendment 51 item A6", "mode": mode, "minutes": minutes, "max_num_seqs": cap,
           "prompts": len(jobs), "dropped_by_the_replay_gate": dropped, "finished": done_n,
           "runpod_ref_tps": RUNPOD["plain_tps" if mode == "plain" else "doc_tps"],
           **windows(t0, ev, minutes)}
    rec["relative_to_runpod"] = rec["mean_tps"] / rec["runpod_ref_tps"]
    F.T._w(out, f"tput_{mode}", rec)  # noqa: SLF001
    print(">> tput:", json.dumps({k: v for k, v in rec.items() if k != "per_minute_tps"}),
          flush=True)
    return rec


def stage_hf_fixed(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F
    import pilot_pod_lib as PL

    files = PL.fixed_smoke_files(out, dry)
    t0 = time.time()
    PL.hf_letters(out, [(PL.FIXED_SMOKE_TAG, str(files["adapter"]))], dry)
    new_b = json.loads((out / "parity" / "hf_base.json").read_text())["base"]
    new_a = json.loads((out / "parity" / f"hf_{PL.FIXED_SMOKE_TAG}.json").read_text())["attached"]
    saved_b = json.loads(files["hf_base"].read_text())["base"]
    saved_a = json.loads(files["hf"].read_text())["attached"]
    rec = {"rule": "Amendment 51 item A5", "tv_base_vs_saved": PL.tv(new_b, saved_b),
           "tv_fixed_vs_saved": PL.tv(new_a, saved_a), "runpod_sessions_tv_base": 0.0,
           "wall_h": (time.time() - t0) / 3600}
    rec["identical"] = rec["tv_base_vs_saved"] == 0.0 and rec["tv_fixed_vs_saved"] == 0.0
    F.T._w(out, "hf_fixed", rec)  # noqa: SLF001
    print(">> hf_fixed:", json.dumps(rec), flush=True)
    return rec


def stage_train50(out: Path, dry: bool) -> dict:
    import cst_finalist_pod as F
    import cst_pressure_train_pod as T
    import pilot_pod_lib as PL

    t0 = time.time()
    path, wb, _ = PL.load_model(dry)
    load_h = (time.time() - t0) / 3600
    src = CST / "outputs/pressure"
    ex = T.ctx_examples(src, "a0s", T.Render(F.S.V._tok(path)))  # noqa: SLF001
    ex = ex[: 8 if dry else 200]
    t1 = time.time()
    r = T._train(wb, "p_ctx", {"examples": ex}, out / "adapters" / "probe_p_ctx", dry)  # noqa: SLF001
    train_s = time.time() - t1
    rec = {"rule": "Amendment 51 item A7", "steps": r["steps"], "examples": len(ex),
           "load_h": load_h, "train_s": train_s, "s_per_step": train_s / max(r["steps"], 1),
           "runpod_200_steps_h_with_load": RUNPOD["train_200_steps_h_with_load"]}
    # RunPod's per-step time, its stage less this host's load time (an estimate of its own)
    rp = (RUNPOD["train_200_steps_h_with_load"] - load_h) * 3600 / 200
    rec["runpod_s_per_step_est"] = rp
    rec["relative_to_runpod"] = rp / rec["s_per_step"] if rec["s_per_step"] else None
    F.T._w(out, "train50", rec)  # noqa: SLF001
    print(">> train50:", json.dumps({k: v for k, v in rec.items()}), flush=True)
    return rec


def verdict(recs: dict, price: float) -> dict:
    """Amendment 51 item A9: relative throughput (the plain and document workloads' means over
    RunPod's), the worst-window ratio, the effective price; pass needs A1-A5 clean and a worst
    window at least 0.8 of the mean on both workloads."""
    rel = [recs[m]["relative_to_runpod"] for m in ("tput_plain", "tput_doc") if m in recs]
    ww = [recs[m]["worst_over_mean"] for m in ("tput_plain", "tput_doc") if m in recs]
    relative = min(rel) if rel else None
    out = {"relative_throughput": relative, "relative_by_workload": rel,
           "worst_over_mean": min(ww) if ww else None, "reported_price": price,
           "effective_price": price / relative if relative else None, "beside": PRICES,
           # the relative throughput above which this provider's effective price is the lower
           "thresholds": {"beats_hyperstack_above": price / PRICES["hyperstack"],
                          "beats_runpod_secure_above": price / PRICES["runpod_secure"]}}
    clean = all(recs.get(k, {}).get(f, False) for k, f in (
        ("stack", "ok"), ("gate_tests", "ok"), ("smoke_serve", "parity_passes")))
    hf = recs.get("hf_fixed", {})
    clean = clean and hf.get("tv_base_vs_saved") is not None
    out["items_1_5_clean"] = clean
    out["passes"] = bool(clean and out["worst_over_mean"] is not None
                         and out["worst_over_mean"] >= 0.8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("hf_fixed", "tput", "train50"))
    ap.add_argument("--mode", choices=("plain", "doc"))
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    import cst_finalist_pod as F

    F.configure()
    if a.stage == "hf_fixed":
        stage_hf_fixed(a.out, a.dry_run)
    elif a.stage == "tput":
        stage_tput(a.out, a.mode, a.minutes, a.dry_run)
    else:
        stage_train50(a.out, a.dry_run)


if __name__ == "__main__":
    main()
