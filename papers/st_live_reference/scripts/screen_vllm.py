"""Amendment 7: the T1.0 relaunch, two phases per model on one pod process.

Phase V (vLLM; only the tokenizer is held here): every trace cell, through ``vllm_phase``.
  Qwen3.8 / Muse: the pilot at the shipped default effort with the with/without-speculation probe
  (pilot regime), one step down if p95 > 1,536 (setting of record reused), the C0 sampled answers
  for every dose-0 candidate, then the screen with the probe at the full batch (its faster
  setting runs the rest; for Muse a losing DFlash means a screen without speculation).
  GPT-OSS-20B (development model): the pilot-regime probe on the pilot, and the full-batch
  probe on the parity jobs (§40's 64 prompts x 4).
Phase H (HF, bf16, eager reference): gates, the padded check and the fast-path rung (Amendment 6
  ladder), then the readouts of the HF engine path (dose-0 candidates, panel, post-reasoning
  readout, dose-0 on the screen scenarios, brake), or for GPT-OSS the HF memory probe at 132 and
  264 and the engine-parity statistic against §40.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import analyze_st as AS
import engine_probes as EP
import numpy as np
import screen_lib as L
import vllm_phase as V

HF_FAST_ATTN = {"gpt_oss_20b": "flex_attention", "qwen38_27b": "sdpa",
                "muse_glimmer_30b": "sdpa"}
PARITY_N = 64
HF_PROBE_BATCHES = (132, 264)
S40 = Path(__file__).resolve().parents[1] / "data" / "s40_reference.json"


def job_seed(stage: str, sid: str, k: int) -> int:
    """A per-rollout seed, stable across runs (vLLM seeds per request)."""
    return zlib.crc32(f"{stage}|{sid}|{k}".encode()) % (2**31)


def worker_jobs(jobs: list[dict], stage: str, cap: int, sampler=None) -> list[dict]:
    return [{"job_id": f"{stage}|{j['s'].id}|{j['k']}", "prompt_ids": j["ids"], "max_tokens": cap,
             "seed": job_seed(stage, j["s"].id, j["k"]), "sampler": sampler} for j in jobs]


def gen_stage(sc, jobs: list[dict], stage: str, cap: int, spec: dict, *, probe: bool,
              choice: str | None = None, deadline=None, after_h=0.0, expected_len=0) -> dict:
    """Generate ``jobs`` through vLLM; with ``probe`` the first ``max_num_seqs`` jobs run with and
    without speculation and the faster setting runs the rest. Rows come from ``L.build_rows``."""
    wj = worker_jobs(jobs, stage, cap)
    kw = dict(deadline=deadline, after_h=after_h, expected_len=expected_len)
    out = sc.out / "vllm"
    probe_rec = None
    vrows: list[dict] = []
    summaries = []
    if probe:
        b = spec["vllm"]["max_num_seqs"]
        res = V.probe_and_run(sc, wj[:b], spec, out, regime=stage, **kw)
        choice, probe_rec = res["choice"], res["probe"]
        vrows += res["rows"]
        summaries.append(res["summary"])
        rest = wj[b:]
    else:
        rest = wj
    if rest:
        eng = V.engine_json(spec, spec_on=(choice == "spec"), label=f"{sc.key}:{stage}:{choice}")
        rows_, summ, _ = V.run_vllm(sc, rest, eng, out / f"{stage}_{choice}_rest.jsonl", **kw)
        vrows += rows_
        summaries.append(summ)
    by = {r["job_id"]: r for r in vrows}
    ids = [by[j["job_id"]]["gen_ids"] if by[j["job_id"]]["status"] == "finished" else None
           for j in wj]
    rows = L.build_rows(sc, jobs, ids, [j["seed"] for j in wj], cap=cap, stage=stage,
                        engine="vllm")
    for r, j in zip(rows, wj):
        r["vllm_seed"], r["vllm_choice"] = j["seed"], choice
    wall = sum(s["wall_seconds"] for s in summaries)
    tokens = sum(s["decoded_tokens"] for s in summaries)
    timing = {"engine": "vllm", "choice": choice, "n_jobs": len(wj),
              "n_run": sum(i is not None for i in ids), "gen_seconds": wall,
              "gen_tokens": tokens, "tokens_per_second": tokens / max(wall, 1e-9),
              "batch": spec["vllm"]["max_num_seqs"],
              "settings_sha256": [s["settings_sha256"] for s in summaries],
              "load_seconds": [s.get("load_seconds") for s in summaries],
              "stopped": [s.get("stopped") for s in summaries]}
    return {"rows": rows, "timing": timing, "probe": probe_rec, "choice": choice}


def q90(rows: list[dict]) -> int:
    t = [r["tokens_to_answer"] for r in rows if r.get("status") not in (None, "not_run")]
    return int(np.percentile(t, 90)) if t else 0


def phase_v_screen(sc, ctx) -> dict:
    """Qwen3.8 / Muse: pilot (probe, step-down), C0 answers, screen (probe at full batch)."""
    spec, H = ctx.spec, ctx.H
    v: dict = {"pilot": []}
    levels = list(sc.fmt.effort_ladder)[: L.MAX_LEVELS_PER_POD]
    choice = None
    for i, level in enumerate(levels):
        sc.effort = level
        g = gen_stage(sc, L.h1_jobs(sc, H, ctx.pilot_ids, L.PILOT_R), "pilot", L.PILOT_CAP, spec,
                      probe=(i == 0), choice=choice, deadline=ctx.deadline)
        choice = g["choice"]
        sc.save_jsonl(f"pilot/pilot_{level}.jsonl", g["rows"])
        summ = {"effort": level, **L.pilot_summary(g["rows"]), "timing": g["timing"],
                "probe": g["probe"]}
        v["pilot"].append(summ)
        if g["timing"]["n_run"] < g["timing"]["n_jobs"]:
            v["stopped"] = "pilot"
            return v
        if summ["fits_cap_of_record"]:
            break
    v["pilot_choice"] = choice
    v["effort_of_record"] = sc.effort
    v["lowest_level_this_pod"] = levels[-1]
    pilot_rows = [json.loads(x) for x in
                  (sc.out / "pilot" / f"pilot_{sc.effort}.jsonl").read_text().splitlines()]
    v["expected_len"] = q90(pilot_rows)
    # C0 sampled answers for every candidate (short, pilot-regime setting)
    v["c0_answers"] = {}
    for cand in sc.fmt.dose0:
        jobs = [{"s": s, "k": k, "order": o, "ids": ids, "rendered_sha256": None}
                for s, k, o, ids in L.c0_jobs(sc, ctx.C0, cand)]
        g = gen_stage(sc, jobs, f"c0_{cand.name}", L.C0_ANSWER_TOKENS, spec, probe=False,
                      choice=choice, deadline=ctx.deadline)
        v["c0_answers"][cand.name] = [r["gen_ids"] or None for r in g["rows"]]
    # the screen, probe at the full batch
    g = gen_stage(sc, L.h1_jobs(sc, H, ctx.screen_ids, 4), "screen", L.CAP_OF_RECORD, spec,
                  probe=True, deadline=ctx.deadline, after_h=ctx.hf_after_h,
                  expected_len=v["expected_len"])
    v["screen_rows"], v["screen_timing"], v["screen_probe"] = g["rows"], g["timing"], g["probe"]
    sc.save_jsonl("screen/h1_screen.jsonl", g["rows"])
    if g["timing"]["n_run"] < g["timing"]["n_jobs"]:
        v["stopped"] = "screen"
    return v


def parity_jobs(sc, ctx) -> tuple[list[dict], list[str]]:
    """§40's exact prompts (permutation 0, medium) for the first 64 of the seeded H1_train order,
    4 rollouts each; each rendered prompt's sha must equal §40's."""
    ref = json.loads(S40.read_text())["rows"]
    ids = [s for s in ctx.order if s in ref][:PARITY_N if not sc.dry else 6]
    jobs = []
    for sid in ids:
        s = ctx.H[sid]
        o = L.assign_letters(s, ref[sid]["permutation"])
        rendered = sc.render(L.letter_chat_messages(s, o, "agent", "neutral", "primary"))
        if L.sha256_text(rendered) != ref[sid]["prompt_sha256"] and not sc.dry:
            raise RuntimeError(f"parity prompt for {sid} differs from §40's")
        for k in range(4):
            jobs.append({"s": s, "k": k, "order": o, "ids": sc.fmt.prompt_ids(sc.tok, rendered),
                         "rendered_sha256": L.sha256_text(rendered)})
    return jobs, ids


def phase_v_oss(sc, ctx) -> dict:
    """GPT-OSS-20B: pilot-regime probe on the pilot jobs, full-batch probe on the parity jobs."""
    spec = ctx.spec
    sc.effort = sc.fmt.effort_ladder[0]
    v: dict = {}
    g = gen_stage(sc, L.h1_jobs(sc, ctx.H, ctx.pilot_ids, L.PILOT_R), "pilot", L.PILOT_CAP, spec,
                  probe=True, deadline=ctx.deadline)
    sc.save_jsonl(f"pilot/pilot_{sc.effort}.jsonl", g["rows"])
    v["pilot"] = [{"effort": sc.effort, **L.pilot_summary(g["rows"]), "timing": g["timing"],
                   "probe": g["probe"]}]
    jobs, ids = parity_jobs(sc, ctx)
    g = gen_stage(sc, jobs, "parity", L.CAP_OF_RECORD, spec, probe=True,
                  deadline=ctx.deadline, after_h=ctx.hf_after_h, expected_len=q90(g["rows"]))
    sc.save_jsonl("parity/parity_vllm.jsonl", g["rows"])
    v.update(parity_rows=g["rows"], parity_ids=ids, parity_timing=g["timing"],
             parity_probe=g["probe"])
    return v


def phase_h_oss(sc, ctx, v: dict, rec: dict) -> None:
    """GPT-OSS-20B: the engine-parity statistic and the HF memory probe (the fallback)."""
    ref = json.loads(S40.read_text())["rows"]
    rec["engine_parity"] = AS.engine_parity(v["parity_rows"], ref, v["parity_ids"])
    if sc.dry:
        rec["hf_memory_probe"] = {"dry": True}
        return
    prompts = [sc.fmt.prompt_ids(sc.tok, sc.render(L.letter_chat_messages(
        ctx.H[s], L.assign_letters(ctx.H[s], 0), "agent", "neutral", "primary")))
        for s in v["parity_ids"]]
    probes = [EP.hf_batch_probe(sc.model, prompts, b, cap=L.CAP_OF_RECORD)
              for b in HF_PROBE_BATCHES]
    rec["hf_memory_probe"] = EP.summarize(probes)


def run(ctx) -> int:
    """Both phases for one model; returns the driver's exit code."""
    sc, rec, key = ctx.sc, ctx.rec, ctx.key
    t0 = time.time()
    v = phase_v_oss(sc, ctx) if key == "gpt_oss_20b" else phase_v_screen(sc, ctx)
    rec["phase_v"] = {k: val for k, val in v.items()
                      if k not in ("screen_rows", "parity_rows", "c0_answers")}
    rec["phase_v"]["seconds"] = time.time() - t0
    if v.get("stopped") == "pilot":
        return ctx.bank("envelope_stop_in_pilot", 3)
    # phase H: the HF model, eager reference, then the fused path if its rung passes
    ctx.load_hf(sc)
    g = ctx.gates(sc)
    rec["gates"] = g
    if not g["ok"]:
        return ctx.bank("gate_failed", 2)
    probe = [j["ids"] for j in L.h1_jobs(sc, ctx.H, ctx.order[:16], 1)]
    rec["probe_mean_len"] = float(np.mean([len(x) for x in probe]))
    rec["padded_check"] = L.padded_check(sc, probe)
    if not rec["padded_check"]["ok"]:
        sc.unbatched = True
        rec.setdefault("anomaly_flags", []).append(
            "padded check failed or uncalibrated (Amendment 6): readouts unbatched")
    rec["fast_path_rung"] = L.fast_path_rung(
        sc, probe, HF_FAST_ATTN[key],
        ref_floor=rec["padded_check"].get("noise_floor_replicate_unpadded"))
    t1 = time.time()
    L.read_ids(sc, probe, sc.bs(sc.read_batch))
    rec["read_seconds_per_row"] = (time.time() - t1) / len(probe)
    if key == "gpt_oss_20b":
        phase_h_oss(sc, ctx, v, rec)
        return ctx.bank("ok_probe", 0)
    return ctx.finish_screen(sc, v)
