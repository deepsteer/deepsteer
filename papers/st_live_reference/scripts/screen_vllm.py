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

import hashlib
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
RIDER_PROBE_BATCHES = (33, 66, 132)  # eager: smaller first, so an OOM still leaves a measurement
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


def _probe_prompts(sc, ctx, ids: list[str]) -> list[list[int]]:
    return [sc.fmt.prompt_ids(sc.tok, sc.render(L.letter_chat_messages(
        ctx.H[s], L.assign_letters(ctx.H[s], 0), "agent", "neutral", "primary"))) for s in ids]


def phase_h_oss(sc, ctx, v: dict, rec: dict) -> None:
    """GPT-OSS-20B: the engine-parity statistic and the HF memory probe (the fallback)."""
    ref = json.loads(S40.read_text())["rows"]
    rec["engine_parity"] = AS.engine_parity(v["parity_rows"], ref, v["parity_ids"])
    if sc.dry:
        rec["hf_memory_probe"] = {"dry": True}
        return
    prompts = _probe_prompts(sc, ctx, v["parity_ids"])
    probes = [EP.hf_batch_probe(sc.model, prompts, b, cap=L.CAP_OF_RECORD)
              for b in HF_PROBE_BATCHES]
    rec["hf_memory_probe"] = EP.summarize(probes)


def rider_oss_h(sc, ctx, rec: dict) -> None:
    """Rider (b) (author, 2026-10-19): the HF memory probe on eager attention, uncompiled,
    whatever the rung kept, for a measured T1.3 price (the steering gate generates through HF
    with hooks). Batches from small to the relaunch's largest fit, so an eager OOM at a large
    batch still leaves a measured smaller one."""
    if sc.dry:
        rec["hf_memory_probe"] = {"dry": True, "path": "eager, uncompiled"}
        return
    m = sc.model
    m.model.set_attn_implementation("eager")
    if sc.eager_forward is not None:
        m.model.forward = sc.eager_forward
    _, ids = parity_jobs(sc, ctx)
    prompts = _probe_prompts(sc, ctx, ids)
    probes = [EP.hf_batch_probe(m, prompts, b, cap=L.CAP_OF_RECORD) for b in RIDER_PROBE_BATCHES]
    rec["hf_memory_probe"] = {**EP.summarize(probes), "path": "eager, uncompiled"}


# ---- rider pod 2 (author, 2026-10-19; G1 items 3-4, Amendment 9) ------------------------------
FOLD_BLOCK = 11  # GPT-OSS-20B has 24 blocks; the fold check's injection block (any block tests it)
FOLD_SCALE = 0.25  # |v| as a fraction of the median residual norm at that block on the rows
BATCHED_FLOOR_MAX = 0.025  # the Amendment 6 bar stays at the KDG 0.05 iff 2 x floor <= 0.05
DOD_BAR = 0.25  # fold validation (ii): the HF-vs-vLLM band at the dose-0 position (rider pod)


def readout_mode(dp: dict) -> dict:
    """G1 item 4, pre-registered (Amendment 9): batched readouts at the KDG bar iff, with the fp32
    head at the dose-0 position, the padded-vs-alone check is calibrated, measures <= 0.05, and
    the replicate floor is <= 0.025 (so the Amendment 6 bar max(0.05, 2 x floor) is 0.05 itself);
    single-row with the bf16 head otherwise. The kept rung path under that head is the readout
    path of record."""
    f = dp["fp32"]
    ok = bool(f.get("calibrated") and f["max_abs_nats"] <= L.MAX_NATS
              and f["noise_floor_replicate_unpadded"] <= BATCHED_FLOOR_MAX)
    head = "fp32" if ok else "bf16"
    return {"batched_admissible": ok, "mode": "batched" if ok else "single-row", "head": head,
            "readout_path": (dp[head].get("rung") or {}).get("kept"),
            "rule": "fp32 head at the dose-0 position: calibrated, padded-vs-alone <= 0.05 and "
                    "replicate floor <= 0.025 -> batched at 0.05; else single-row, bf16 head"}


def _reset_eager(sc) -> None:
    if sc.dry:
        return
    sc.model.model.set_attn_implementation("eager")
    if sc.eager_forward is not None:
        sc.model.model.forward = sc.eager_forward


def _free_hf(sc) -> None:
    import gc

    import torch

    sc.model, sc.grab, sc.eager_forward = None, None, None
    gc.collect()
    torch.cuda.empty_cache()


def build_folded(src: Path, block: int, v, dst: Path) -> dict:
    """A copy of checkpoint ``src`` with ``v`` added to every expert's ``down_proj_bias`` of
    ``block`` (G1 item 3: the residual after the block gains exactly v, router weights summing to
    1). Only the shard holding that tensor is rewritten; every other top-level file is linked."""
    import torch
    from safetensors.torch import load_file, save_file

    key = f"model.layers.{block}.mlp.experts.down_proj_bias"
    idx = src / "model.safetensors.index.json"
    shard = json.loads(idx.read_text())["weight_map"][key] if idx.exists() else "model.safetensors"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != shard and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    tensors = load_file(str(src / shard))
    b = tensors[key]
    tensors[key] = (b.float() + torch.as_tensor(v, dtype=torch.float32)[None, :]).to(b.dtype)
    save_file(tensors, str(dst / shard), metadata={"format": "pt"})
    return {"src": str(src), "dst": str(dst), "shard": shard, "key": key, "block": block,
            "shard_sha256": hashlib.sha256((dst / shard).read_bytes()).hexdigest()}


def fold_validation(sc, ctx, rows16: list, bar: float) -> dict:
    """G1 item 3's validation (author-approved): (i) HF with v added by a hook at the block's
    output vs HF from the folded checkpoint, same rows read alone, within the dose-0 ladder bar,
    with the hook's own movement as the positive control; (ii) the steering effect on vLLM
    (folded - base first-token log-probs) vs on HF, the difference of differences within the
    HF-vs-vLLM band; (iii) 64 generations on each vLLM engine with their parse rates."""
    import torch
    from huggingface_hub import snapshot_download
    from kdg_pod_lib import ModelWrapper

    spec, out = ctx.spec, sc.out / "fold"
    out.mkdir(parents=True, exist_ok=True)
    tok = sc.model.tok  # after each HF model is freed, a tokenizer stand-in keeps sc.tok working
    layer = sc.model.wb._get_layer_module(FOLD_BLOCK)
    norms: list = []
    h = layer.register_forward_hook(lambda _m, _i, o: norms.append(
        float((o[0] if isinstance(o, tuple) else o)[0, -1].float().norm())))
    base, _ = L.read_ids(sc, rows16, 1)
    h.remove()
    hid = sc.model.model.config.hidden_size
    u = np.random.default_rng(0).standard_normal(hid)
    v = (FOLD_SCALE * float(np.median(norms)) * u / np.linalg.norm(u)).astype(np.float32)
    np.save(out / "v.npy", v)
    vt = torch.as_tensor(v, device=sc.model.model.device, dtype=torch.bfloat16)

    def add(_m, _i, o):
        return (o[0] + vt,) + tuple(o[1:]) if isinstance(o, tuple) else o + vt
    h = layer.register_forward_hook(add)
    hooked, _ = L.read_ids(sc, rows16, 1)
    h.remove()
    del layer, h  # the module reference would keep the first model's weights on the GPU
    _free_hf(sc)
    sc.model = L.tok_only(tok)
    src = Path(snapshot_download(spec["repo"], revision=spec["revision"],
                                 ignore_patterns=["original/*", "metal/*"]))
    built = build_folded(src, FOLD_BLOCK, v, out / "ckpt")
    sc.model = ModelWrapper(built["dst"], None, torch_dtype=torch.bfloat16,
                            attn_implementation="eager")
    folded, _ = L.read_ids(sc, rows16, 1)
    _free_hf(sc)
    sc.model = L.tok_only(tok)
    rec: dict = {"block": FOLD_BLOCK, "v_norm": float(np.linalg.norm(v)),
                 "median_resid_norm": float(np.median(norms)), "built": built, "bar": bar}
    rec["i_hook_vs_fold"] = {
        "max_abs_nats": float(np.abs(hooked - folded).max()),
        "positive_control_hook_vs_base": float(np.abs(hooked - base).max()),
        "ok": bool(np.abs(hooked - folded).max() <= bar < np.abs(hooked - base).max())}
    # vLLM: base and folded engines, the same 16 rows (first-token log-probs) + 64 generations
    lids = L.letter_ids(sc)
    gen = L.h1_jobs(sc, ctx.H, ctx.order[:16], 4)
    wj = ([{"job_id": f"lp|{i}", "prompt_ids": x, "max_tokens": 1, "seed": job_seed("lp", "", i),
            "logprob_ids": lids} for i, x in enumerate(rows16)]
          + worker_jobs(gen, "fold", L.CAP_OF_RECORD))
    vl: dict = {}
    for tag in ("base", "folded"):
        eng = V.engine_json(spec, spec_on=False, label=f"{sc.key}:fold:{tag}")
        if tag == "folded":
            eng.update(model=built["dst"], revision=None, tokenizer=spec["repo"],
                       tokenizer_revision=spec["revision"])
        rows, summ, _ = V.run_vllm(sc, wj, eng, out / f"vllm_{tag}.jsonl")
        by = {r["job_id"]: r for r in rows}
        g = [by[j["job_id"]]["gen_ids"] if by[j["job_id"]]["status"] == "finished" else None
             for j in wj[len(rows16):]]
        parsed = L.build_rows(sc, gen, g, [j["seed"] for j in wj[len(rows16):]],
                              cap=L.CAP_OF_RECORD, stage=f"fold_{tag}", engine="vllm")
        vl[tag] = {"first": [by[f"lp|{i}"].get("first_logprobs") or {} for i in range(len(rows16))],
                   "parse_rate": float(np.mean([r.get("letter") is not None for r in parsed])),
                   "statuses": L._counts(r["status"] for r in parsed),
                   "tokens_per_second": summ["tokens_per_second"]}
    dd = []
    for i in range(len(rows16)):
        for c, tid in enumerate(lids):
            fb, ff = vl["base"]["first"][i].get(str(tid)), vl["folded"]["first"][i].get(str(tid))
            if fb is not None and ff is not None:
                dd.append(abs((ff - fb) - float(folded[i, c] - base[i, c])))
    rec["ii_steering_effect_vllm_vs_hf"] = {
        "n_letter_values": len(dd), "max_abs_dod": max(dd) if dd else None,
        "mean_abs_dod": float(np.mean(dd)) if dd else None, "bar": DOD_BAR,
        "hf_effect_max": float(np.abs(folded - base).max()),
        "ok": bool(dd and max(dd) <= DOD_BAR)}
    rec["iii_generation"] = {t: {k: vl[t][k] for k in ("parse_rate", "statuses",
                                                       "tokens_per_second")} for t in vl}
    np.savez_compressed(out / "letters.npz", base=base, hooked=hooked, folded=folded)
    import shutil

    shutil.rmtree(out / "ckpt", ignore_errors=True)  # links into the HF cache; sha in the record
    return rec


def rider2_h(sc, ctx, rec: dict) -> None:
    """Rider pod 2 on a model of record (GPT-OSS-20B, Muse Glimmer 30B): the padded check and the
    rung re-read at the dose-0 letter position (the first-registered candidate, of record for
    both), with the bf16 head and the fp32 head (G1 item 4); the pre-registered readout-mode
    rule; the 64 dose-0 rows written for the stack-parity read (Amendment 9); and, on GPT-OSS,
    the fold validation (G1 item 3)."""
    cand = sc.fmt.dose0[0]
    rows64 = L.fwd_gen_ids(sc, ctx.C0, cand, 64)
    rows16 = rows64[:16]
    (sc.out / "stack").mkdir(parents=True, exist_ok=True)
    (sc.out / "stack" / "rows64.json").write_text(json.dumps(
        {"ids": rows64, "letter_ids": L.letter_ids(sc), "candidate": cand.name,
         "effort": sc.effort, "date_pin": sc.date_pin}))
    if sc.key == "gpt_oss_20b":  # generation prompts for the MoE probe (the parity prompts)
        _, ids = parity_jobs(sc, ctx)
        (sc.out / "stack" / "probe_prompts.json").write_text(json.dumps(
            {"ids": _probe_prompts(sc, ctx, ids)}))
    import head32

    dp: dict = {}
    for head in ("bf16", "fp32"):
        if not sc.dry:
            dp.setdefault("head_switch", {})[head] = head32.set_head_fp32(sc.model.model,
                                                                          head == "fp32")
        pc = L.padded_check(sc, rows16)
        pc["rung"] = L.fast_path_rung(sc, rows16, HF_FAST_ATTN[sc.key],
                                      ref_floor=pc["noise_floor_replicate_unpadded"])
        _reset_eager(sc)
        dp[head] = pc
        print(f">> {sc.key} dose-0 position [{head} head]: floor "
              f"{pc['noise_floor_replicate_unpadded']:.4f}, padded {pc['max_abs_nats']:.4f}, "
              f"rung kept {pc['rung'].get('kept')}", flush=True)
    if not sc.dry:
        head32.set_head_fp32(sc.model.model, False)
    rec["dose0_position"] = {"candidate": cand.name, **dp}
    rec["readout_mode"] = readout_mode(dp)
    print(f">> {sc.key} readout mode: {rec['readout_mode']['mode']} "
          f"({rec['readout_mode']['head']} head)", flush=True)
    if sc.key == "gpt_oss_20b":
        # the dose-0 reads reach disk before the fold check can fail (rider pod 2's lesson); the
        # bank at the end rewrites the record with the manifest
        rec["status"] = "partial_rider2 (dose-0 reads; fold check running)"
        (sc.out / "screen_record.json").write_text(json.dumps(rec, indent=1,
                                                              default=L._json_default))
        if sc.dry:
            rec["fold_validation"] = {"dry": True}
        else:  # first GPU run of the fold path: a failure is recorded, the dose-0 reads bank
            try:
                rec["fold_validation"] = fold_validation(
                    sc, ctx, rows16, max(L.MAX_NATS, L.FLOOR_FACTOR
                                         * dp["bf16"]["noise_floor_replicate_unpadded"]))
            except Exception as e:  # noqa: BLE001
                import shutil

                shutil.rmtree(sc.out / "fold" / "ckpt", ignore_errors=True)
                rec["fold_validation"] = {"failed": f"{type(e).__name__}: {str(e)[:400]}"}
                print(f">> fold validation failed ({e}); the dose-0 reads bank", flush=True)


def load_banked_v(sc, ctx, d: Path) -> dict:
    """Phase V from a banked run (author, 2026-10-19: Qwen's relaunch rollouts were generated at
    the registry-pinned revision, so its rerun is phase H on them; Muse's dose-0 re-check uses its
    banked C0 answers). Every provenance check is fatal:
    - the banked record is this model's, and every banked engine config names the registry repo
      and revision (the vLLM log of the relaunch shows the same revision for every engine);
    - the banked screen ids equal this run's (the seeded H1_train order);
    - every banked screen row's prompt ids and rendered prompt hash equal this run's renders at
      the banked effort and date pin, and every banked C0 job's prompt ids equal this run's
      ``c0_jobs`` for its candidate (the rows phase H reads are the rows that were generated).
    The banked files are copied into this run's directory, so its record is self-contained."""
    import shutil

    old = json.loads((d / "screen_record.json").read_text())
    if old["model"] != sc.key:
        raise SystemExit(f"banked record {d} is {old['model']}, not {sc.key}")
    for f in sorted((d / "vllm").glob("*.engine.json")):
        e = json.loads(f.read_text())
        if (e["model"], e["revision"]) != (ctx.spec["repo"], ctx.spec["revision"]):
            raise SystemExit(f"{f.name}: {e['model']}@{e['revision']} is not the registry pin")
    if list(old["screen_ids"]) != list(ctx.screen_ids):
        raise SystemExit("banked screen ids differ from this run's seeded order")
    if old["date_pin"] != sc.date_pin:
        raise SystemExit(f"date pin {sc.date_pin} differs from the banked {old['date_pin']}")
    v = dict(old["phase_v"])
    sc.effort = v["effort_of_record"]
    for sub in ("pilot", "screen", "vllm"):
        shutil.copytree(d / sub, sc.out / sub, dirs_exist_ok=True)
    rows = [json.loads(x) for x in (d / "screen" / "h1_screen.jsonl").read_text().splitlines()]
    jobs = L.h1_jobs(sc, ctx.H, ctx.screen_ids, 4)
    if len(rows) != len(jobs):
        raise SystemExit(f"{len(rows)} banked screen rows for {len(jobs)} jobs")
    for r, j in zip(rows, jobs):
        if (r["scenario_id"], r["rollout"]) != (j["s"].id, j["k"]) or r["rendered_sha256"] != j[
                "rendered_sha256"] or r["prompt_ids_sha256"] != hashlib.sha256(
                json.dumps(j["ids"]).encode()).hexdigest():
            raise SystemExit(f"banked screen row {r['scenario_id']}|{r['rollout']} does not "
                             "match this run's render")
    v["screen_rows"] = rows
    v["c0_answers"] = {}
    for cand in sc.fmt.dose0:
        stem = d / "vllm" / f"c0_{cand.name}_{v['pilot_choice']}_rest"
        bj = [json.loads(x) for x in Path(f"{stem}.jobs.jsonl").read_text().splitlines()]
        want = [x[3] for x in L.c0_jobs(sc, ctx.C0, cand)]
        if [j["prompt_ids"] for j in bj] != want:
            raise SystemExit(f"banked C0 jobs for {cand.name} do not match this run's prompts")
        by = {r["job_id"]: r for r in map(json.loads,
                                           Path(f"{stem}.jsonl").read_text().splitlines())}
        v["c0_answers"][cand.name] = [by[j["job_id"]]["gen_ids"] or None
                                      if by[j["job_id"]]["status"] == "finished" else None
                                      for j in bj]
    v["resumed_from"] = {"dir": str(d), "status": old.get("status"),
                         "elapsed_h": old.get("elapsed_h"), "date_pin": old["date_pin"],
                         "checks": "record model, engine repo@revision, screen ids, date pin, "
                                   "screen rows' prompt and render hashes, C0 job prompt ids"}
    return v


def vllm_dose0_first(sc, ctx) -> dict:
    """Amendment 8's descriptive column: vLLM's first-token log-probs (one decode step, no
    speculation) on each dose-0 candidate's forward-vs-generation rows."""
    lids = L.letter_ids(sc)
    jobs, names = [], {}
    for cand in sc.fmt.dose0:
        ids = L.fwd_gen_ids(sc, ctx.C0, cand)
        names[cand.name] = len(ids)
        jobs += [{"job_id": f"fwdgen|{cand.name}|{i}", "prompt_ids": x, "max_tokens": 1,
                  "seed": job_seed("fwdgen", cand.name, i), "logprob_ids": lids}
                 for i, x in enumerate(ids)]
    eng = V.engine_json(ctx.spec, spec_on=False, label=f"{sc.key}:fwdgen:nospec")
    rows, summ, _ = V.run_vllm(sc, jobs, eng, sc.out / "vllm" / "fwdgen_nospec.jsonl")
    by = {r["job_id"]: r for r in rows}
    sc.vllm_first = {c: [by[f"fwdgen|{c}|{i}"].get("first_logprobs") for i in range(n)]
                     for c, n in names.items()}
    return {"n_jobs": len(jobs), "settings_sha256": summ.get("settings_sha256"),
            "load_seconds": summ.get("load_seconds")}


def run(ctx) -> int:
    """Both phases for one model; returns the driver's exit code. ``ctx.mode``: "full" (phase V
    then H), "resume" (phase H on a banked phase V, ``ctx.resume_from``), "dose0_recheck" (resume,
    then the dose-0 checks only: rider (a)), "rider_oss" (GPT-OSS-20B: gates, padded check, the
    corrected rung and the eager memory probe: rider (b)), "provenance" (resume checks only)."""
    sc, rec, key = ctx.sc, ctx.rec, ctx.key
    mode = getattr(ctx, "mode", "full")
    t0 = time.time()
    if mode in ("rider_oss", "rider2"):
        sc.effort = getattr(ctx, "effort", None) or sc.fmt.effort_ladder[0]
        v: dict = {}
    elif mode in ("resume", "dose0_recheck", "provenance"):
        v = load_banked_v(sc, ctx, ctx.resume_from)
        rec["resumed_from"] = v.pop("resumed_from")
        if mode == "provenance":
            return ctx.bank("ok_provenance", 0)
        try:  # descriptive (Amendment 8): its first real run must never block phase H
            rec["fwdgen_vllm"] = vllm_dose0_first(sc, ctx)
        except RuntimeError as e:
            rec["fwdgen_vllm"] = {"failed": str(e)}
            print(f">> {key}: descriptive vLLM stage failed ({e}); phase H continues", flush=True)
    else:
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
    if mode == "rider2":
        rider2_h(sc, ctx, rec)
        return ctx.bank("ok_rider2", 0)
    probe = [j["ids"] for j in L.h1_jobs(sc, ctx.H, ctx.order[:16], 1)]
    rec["probe_mean_len"] = float(np.mean([len(x) for x in probe]))
    rec["padded_check"] = L.padded_check(sc, probe)
    sc.replicate_floor = rec["padded_check"].get("noise_floor_replicate_unpadded")
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
    if mode == "rider_oss":
        rider_oss_h(sc, ctx, rec)
        return ctx.bank("ok_rider", 0)
    if key == "gpt_oss_20b":
        phase_h_oss(sc, ctx, v, rec)
        return ctx.bank("ok_probe", 0)
    if mode == "dose0_recheck":
        rec["dose0"] = L.choose_dose0(sc, ctx.C0, v["c0_answers"])
        return ctx.bank("ok_dose0_recheck", 0)
    return ctx.finish_screen(sc, v)
