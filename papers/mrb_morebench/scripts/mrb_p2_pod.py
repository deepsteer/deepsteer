"""MRB Phase 2 pod driver (MRB_PLAN v0.9; MB_PLAN Amendment 5). Runs in the vLLM venv on the pod.

    hf-phase --model M --out DIR [--validate] [--dry-run]
        HF on one GPU: the MoralBench rider (R2, both orders, dose 0) on the base model and with
        ST's adapter attached; ST's merge (``write_merged``); the merged copy's rows; the merge gate.
    generate --model M --arm {none,adapter} --k K --out DIR [--validate] [--deadline E] [--dry-run]
        vLLM on one GPU: the parity letters on the rider's 176 dose-0 prompts (the adapter arm
        checks ST's parity gate against the HF rows), then MoReBench generation, k per scenario.
    judge --out DIR [--validate] [--deadline E] [--local-dir PATH] [--dry-run]
        G1's judge of record on every response, k0 of every arm first, then row by row; G1's guard.
    project --out DIR --validate-hours H --setup-hours H [--envelope-h 9.0]
        The session from VALIDATE's rates, with the per-model cut (a).

Models: Muse Glimmer 30B (``muse_glimmer_30b``) and Qwen3.8-27B (``qwen38_27b``) at ST's registry
pins, rendered by ST's ``reasoning_formats`` at effort medium (Muse's date pinned at 2026-10-23,
T1.6's). Tracked outputs as G1's (ids, lengths, hashes, letter masses, verdicts of at most 20
characters); texts in ``DIR/_raw``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ST = REPO / "papers" / "st_live_reference"
sys.path[:0] = [str(HERE), str(ST / "scripts"), str(REPO / "papers" / "kdg_panel" / "scripts"),
                str(REPO / "papers" / "mb_moralbench" / "scripts")]

import mrb_g1_pod as P  # noqa: E402

from deepsteer.benchmarks.moral_reasoning.morebench.prompts import (  # noqa: E402
    generation_prompt,
    judge_prompt,
)
from deepsteer.reasoning.think_io import CoTFormat  # noqa: E402

POD_VERSION = "1.0.0"
MODELS = ("muse_glimmer_30b", "qwen38_27b")
ARMS = ("none", "adapter")
EFFORT = "medium"  # ST's effort of record for both models
DATE_PIN = {"muse_glimmer_30b": "2026-10-23", "qwen38_27b": None}  # Muse: T1.6's render pin
# ST's adapters (Muse: outputs/t16/.../adapters/care; Qwen: outputs/st2qp_pull/.../adapters/care),
# copied into MRB's gitignored inputs so the launcher ships them as named SYNC_OUTPUTS files
INPUTS = HERE.parent / "outputs" / "p2_inputs" / "adapters"
ADAPTERS = {m: INPUTS / m for m in ("muse_glimmer_30b", "qwen38_27b")}
ADAPTER_SHA = {"muse_glimmer_30b": "1d0cdc42b39528cfd74285c40829b9c9078a18b9901fb3fd05235c5111fe62e8",
               "qwen38_27b": "9398ce21d1a11efd465862459f024834497ae2cf4615f16707e178822f5fd3ce"}
SAMPLER = {"temperature": 0.6, "top_p": 0.95, "top_k": 0}  # Table 4, the thinking-model row
PARITY_MARGIN = 0.05  # ST's (t16_driver)
TURN_END = ("<|end|>", "<|return|>", "<|eom|>", "<|start|>")
VALIDATE_SCENARIOS = tuple(range(0, 500, 32))[:16]
LLM_KWARGS = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "max_model_len": 12288,
              "max_num_seqs": 256, "gpu_memory_utilization": 0.92, "enable_prefix_caching": True,
              "enable_chunked_prefill": True, "generation_config": "vllm", "seed": 0,
              "language_model_only": True}


def spec(model: str) -> dict:
    import pod_st_screen as S
    return S.registry()[model]


def tokenizer(model: str):
    from transformers import AutoTokenizer
    s = spec(model)
    return AutoTokenizer.from_pretrained(s["repo"], revision=s["revision"])


def fmt_for(model: str, tok):
    from reasoning_formats import build_format
    return build_format(model, tok, spec(model))


def seed(model: str, sid: str, k: int) -> int:
    """Shared by both arms (common random numbers): the arm is not part of the seed."""
    return P.seed_of("p2", model, sid, k)


def merged_dir(out: Path, model: str) -> Path:
    """Beside the results folder, never inside it: a merged copy is ~60 GB, and the results folder
    is what the launcher downloads and the monitor pulls."""
    return out.parent / f"{out.name}_merged" / model


def sha_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- the MoralBench rows (rider and parity) ---------------------------------------------------
def mb_prompts(model: str, tok, fmt, validate: bool) -> list[dict]:
    """The 176 dose-0 prompts: 88 items x both orders, at the model's dose-0 regime of record
    (``fmt.dose0[0]``: Muse ``empty_reasoning``, Qwen ``enable_thinking_false``)."""
    import mb_g1_run as mb

    from deepsteer.datasets.moralbench import load_moralbench
    system, items = load_moralbench()
    cand = fmt.dose0[0]
    rows = []
    for order in ("original", "counterbalanced"):
        for it in items:
            shown = mb.counterbalance(it) if order == "counterbalanced" else it
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": shown.prompt}]
            ids = fmt.dose0_ids(tok, msgs, EFFORT, cand, DATE_PIN[model])
            rows.append({"set": it.set_name, "item_id": it.item_id, "order": order, "ids": ids,
                         "prompt_sha256": P.sha(json.dumps(ids))})
    return rows


def letters_hf(model_obj, tok, rows: list[dict]) -> list[dict]:
    """Single-row readouts (ST-A3): p(A), p(B) renormalized over the labels, and their mass."""
    import numpy as np

    from deepsteer.choice.readout import label_token_sets, letter_mass, next_token_logprobs
    sets = label_token_sets(tok, "AB")
    out = []
    for r in rows:
        if model_obj is None:  # dry run without a model
            lp = np.log(np.full((1, len(tok)), 1.0 / len(tok)))
        else:
            lp = next_token_logprobs(model_obj, [r["ids"]], pad_id=tok.pad_token_id or 0,
                                     batch_size=1)
        dist, mass = letter_mass(lp, sets)
        out.append({k: r[k] for k in ("set", "item_id", "order", "prompt_sha256")} |
                   {"p_A": float(dist[0][0]), "p_B": float(dist[0][1]),
                    "label_mass": float(mass[0])})
    return out


def tv(a: list[dict], b: list[dict]) -> float:
    """ST's statistic (``t15_driver.tv``) on two shown letters: mean total variation of the A/B
    distribution, row by row (the rows in the same order)."""
    assert [(r["set"], r["item_id"], r["order"]) for r in a] == \
        [(r["set"], r["item_id"], r["order"]) for r in b]
    return float(sum(abs(x["p_A"] - y["p_A"]) for x, y in zip(a, b)) / len(a))


# ---- HF phase ---------------------------------------------------------------------------------
def load_hf(model: str, path: Path | None, dry: bool):
    import torch
    from kdg_pod_lib import ModelWrapper
    if dry:
        return ModelWrapper(str(path), None, torch_dtype=torch.float32)
    import pod_st_screen as S
    s = spec(model)
    if path is not None:
        return ModelWrapper(str(path), None, torch_dtype=torch.bfloat16, **S.hf_load_kwargs(s))
    return ModelWrapper(s["repo"], s["revision"], torch_dtype=torch.bfloat16, **S.hf_load_kwargs(s))


def dry_assets(model: str, out: Path) -> tuple[Path, Path]:
    """A tiny checkpoint with the real tokenizer (``st_tiny``) and a tiny LoRA on it."""
    sys.path.insert(0, str(REPO / "tests" / "scripts"))
    import st_tiny
    import torch
    from peft import LoraConfig, get_peft_model
    tiny = out / "_tiny" / model
    if not (tiny / "config.json").exists():
        st_tiny.build(model, tiny)
    ad = out / "_tiny" / f"{model}_adapter"
    if not (ad / "adapter_model.safetensors").exists():
        mw = load_hf(model, tiny, True)
        torch.manual_seed(1)
        pm = get_peft_model(mw.model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.5)
        pm.save_pretrained(str(ad))
    return tiny, ad


def cmd_hf_phase(a) -> dict:
    import gc

    import t16_driver as T16
    import torch
    t0 = time.time()
    model, out = a.model, a.out / "mb_phase2" / a.model
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(model)
    fmt = fmt_for(model, tok)
    rows = mb_prompts(model, tok, fmt, a.validate)
    if a.dry_run:
        snap, adir = dry_assets(model, a.out)
    else:
        snap, adir = T16.cached_snapshot(spec(model)["repo"], spec(model)["revision"]), ADAPTERS[model]
        got = sha_file(adir / "adapter_model.safetensors")
        if got != ADAPTER_SHA[model]:
            raise SystemExit(f"{model}: adapter sha256 {got[:16]} is not the registered one")
    rec = {"model": model, "adapter": str(adir), "pin": spec(model)["revision"], "effort": EFFORT,
           "dose0": fmt.dose0[0].name, "date_pin": DATE_PIN[model], "validate": a.validate}

    def readout(mw, tag):
        r = letters_hf(mw.model, tok, rows)
        (out / f"r2_{tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in r))
        return r

    mw = load_hf(model, snap if a.dry_run else None, a.dry_run)
    base = readout(mw, "none")
    T16.attach(mw, adir)
    attached = readout(mw, "adapter")
    del mw
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    dst = merged_dir(a.out, model)
    try:
        saved = T16.write_merged(snap, adir, dst)
    except ValueError as e:  # a layout the in-place writer cannot map: the adapter arm drops
        saved = {"ok": False, "reason": str(e)}
    gate = {"saved": saved}
    if saved.get("ok"):
        mm = load_hf(model, dst, a.dry_run)
        merged = readout(mm, "merged_hf")
        del mm
        gc.collect()
        shift, lost, to_base = tv(attached, base), tv(merged, attached), tv(merged, base)
        gate.update(adapter_tv_vs_base=shift, merged_tv_vs_attached=lost,
                    merged_tv_vs_base=to_base, closer_to_base=bool(lost > to_base))
        gate["gate_ok"] = bool(not gate["closer_to_base"]) or a.dry_run
    else:
        gate["gate_ok"] = False
    if not gate["gate_ok"]:
        shutil.rmtree(dst, ignore_errors=True)
    rec.update(merge_gate=gate, n_rows=len(rows), wall_h=round((time.time() - t0) / 3600, 4))
    (out / "hf_phase.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


# ---- generation -------------------------------------------------------------------------------
def split(fmt, tok, g: list[int]) -> tuple[str, str, bool]:
    """(trace, response, closed): ST's parser of record (``fmt._locate``) finds the answer's start;
    the trace is the tokens between the reasoning's opening and its end marker (Qwen: before
    ``</think>``; Muse: inside `` to=self<|message|>...<|eom|>``, so the answer header ``to=user``,
    which is plain text, stays out of both fields). An unclosed trace has no response (upstream
    judges what was written: an empty response)."""
    g = list(g)
    loc = fmt._locate(tok, g)  # noqa: SLF001 - ST's parser of record
    if loc is None or isinstance(loc, str):
        opener = tuple(getattr(fmt, "reason_open", ()))
        body = g[len(opener):] if opener and tuple(g[:len(opener)]) == opener else g
        return tok.decode(body, skip_special_tokens=True).strip(), "", False
    _, j = loc
    if getattr(fmt, "close_id", -1) >= 0:  # ThinkTokenFormat: the template opened the block
        start, end = 0, g.index(fmt.close_id)
    elif tuple(g[:len(fmt.direct_open)]) == tuple(fmt.direct_open):  # Muse, a direct answer
        start = end = 0
    else:  # Muse, a reasoning message
        start = len(fmt.reason_open)
        end = g.index(fmt.reason_end, start)
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True).strip()  # noqa: E731
    resp = tok.decode(g[j:], skip_special_tokens=True)
    for marker in TURN_END:  # Muse's turn markers are not special tokens in its tokenizer
        resp = resp.split(marker, 1)[0]
    return dec(g[start:end]), resp.strip(), True


def gen_jobs(model: str, tok, fmt, k: int, validate: bool) -> list[dict]:
    from deepsteer.datasets.morebench import load_morebench
    jobs = []
    for it in load_morebench(subset="public"):
        if validate and it.idx not in VALIDATE_SCENARIOS:
            continue
        rendered = fmt.render(tok, [{"role": "user", "content": generation_prompt(it.dilemma)}],
                              EFFORT, DATE_PIN[model])
        ids = fmt.prompt_ids(tok, rendered)
        for kk in range(k):
            jobs.append({"sid": it.scenario_id, "k": kk, "prompt_ids": ids,
                         "max_tokens": P.MAX_TOKENS, "seed": seed(model, it.scenario_id, kk),
                         "sampler": SAMPLER})
    return jobs


class StubLLM:
    """--dry-run stand-in for ``vllm.LLM``: a closed trace and a short answer in the model's own
    format; uniform letters for the parity read."""

    def __init__(self, model: str, tok):
        text = {"qwen38_27b": "I weigh the duties and the harms.</think>\n\nI would act carefully.",
                "muse_glimmer_30b": " to=self<|message|>I weigh the duties and the harms.<|eom|>"
                                    "<|start|>assistant to=user<|message|>I would act carefully."}
        self.tok, self.ids = tok, tok(text[model], add_special_tokens=False)["input_ids"]
        self.llm_engine, self.pending = self, []

    def letters(self, rows):
        return [{k: r[k] for k in ("set", "item_id", "order", "prompt_sha256")} |
                {"p_A": 0.5, "p_B": 0.5, "label_mass": 1.0} for r in rows]

    def add_request(self, rid, prompt, params):
        self.pending.append(rid)

    def has_unfinished_requests(self):
        return bool(self.pending)

    def abort_request(self, rids):
        self.pending = [r for r in self.pending if r not in set(rids)]

    def step(self):
        import types
        done, self.pending = self.pending, []
        return [types.SimpleNamespace(request_id=r, finished=True, outputs=[types.SimpleNamespace(
            token_ids=self.ids, finish_reason="stop", logprobs=None)]) for r in done]


def letters_vllm(llm, tok, rows: list[dict]) -> list[dict]:
    """The next-token A/B distribution at each dose-0 prompt: one token, top-20 log-probs (a label
    surface outside the top 20 has probability 0, as ST's parity read)."""
    import math

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from deepsteer.choice.readout import label_token_sets
    sets = label_token_sets(tok, "AB")
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rows],
                        SamplingParams(max_tokens=1, temperature=1.0, logprobs=20, seed=0))
    res = []
    for r, o in zip(rows, outs):
        top = o.outputs[0].logprobs[0]
        mass = [sum(math.exp(top[t].logprob) for t in ids if t in top) for ids in sets]
        tot = sum(mass)
        pa = mass[0] / tot if tot > 0 else 0.5
        res.append({k: r[k] for k in ("set", "item_id", "order", "prompt_sha256")} |
                   {"p_A": pa, "p_B": 1 - pa, "label_mass": tot})
    return res


def read_rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def cmd_generate(a) -> dict:
    model, arm = a.model, a.arm
    tok = tokenizer(model)
    fmt = fmt_for(model, tok)
    mbdir = a.out / "mb_phase2" / model
    gate = json.loads((mbdir / "hf_phase.json").read_text())["merge_gate"]
    rec = {"model": model, "arm": arm, "k": a.k, "validate": a.validate, "pod_version": POD_VERSION}
    if arm == "adapter" and not gate.get("gate_ok"):
        rec.update(status="dropped", reason="merge gate failed")
        (a.out / "gen" / f"{model}.{arm}.summary.json").parent.mkdir(parents=True, exist_ok=True)
        (a.out / "gen" / f"{model}.{arm}.summary.json").write_text(json.dumps(rec, indent=1))
        return rec
    rows = mb_prompts(model, tok, fmt, a.validate)
    t0 = time.time()
    if a.dry_run:
        llm = StubLLM(model, tok)
        letters = llm.letters(rows)
    else:
        from vllm import LLM
        path = (str(merged_dir(a.out, model)) if arm == "adapter" else
                str(__import__("t16_driver").cached_snapshot(spec(model)["repo"],
                                                             spec(model)["revision"])))
        llm = LLM(model=path, tokenizer=spec(model)["repo"],
                  tokenizer_revision=spec(model)["revision"], **LLM_KWARGS)
        letters = letters_vllm(llm, tok, rows)
    load_h = (time.time() - t0) / 3600
    (mbdir / f"vllm_letters_{arm}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in letters))
    if arm == "adapter":
        band = tv(read_rows(mbdir / "vllm_letters_none.jsonl"), read_rows(mbdir / "r2_none.jsonl"))
        d = tv(letters, read_rows(mbdir / "r2_adapter.jsonl"))
        ok = bool(d <= band + PARITY_MARGIN) or a.dry_run
        rec.update(parity_tv=d, band_tv=band, parity_ok=ok)
        if not ok:
            rec.update(status="dropped", reason="vLLM parity failed")
            (a.out / "gen" / f"{model}.{arm}.summary.json").write_text(json.dumps(rec, indent=1))
            return rec
    jobs = [dict(j, job_id=f"{model}/{arm}/{j['sid']}/k{j['k']}")
            for j in gen_jobs(model, tok, fmt, a.k, a.validate)]
    out, raw = a.out / "gen" / f"{model}.{arm}.jsonl", a.out / "_raw" / "gen" / f"{model}.{arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)

    def decode(j, r):
        trace, resp, closed = split(fmt, tok, r["gen_ids"]) if r["gen_ids"] else ("", "", False)
        return ({"n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
                 "status": r["status"], "final_reached": closed and bool(resp),
                 "trace_chars": len(trace), "response_chars": len(resp),
                 "trace_sha256": P.sha(trace), "response_sha256": P.sha(resp)},
                {"trace": trace, "response": resp})

    SP = None
    if not a.dry_run:
        from vllm import SamplingParams as SP  # noqa: N814
    run = P.run_chunks(llm.llm_engine, SP, iter(jobs), out, raw, decode,
                       deadline=a.deadline, meta_keys=("job_id",))
    rec.update(run, status="ok", n_jobs=len(jobs), engine_load_h=round(load_h, 4),
               llm_kwargs=LLM_KWARGS, sampler=SAMPLER)
    (a.out / "gen" / f"{model}.{arm}.summary.json").write_text(json.dumps(rec, indent=1))
    if arm == "adapter" and not a.dry_run:
        shutil.rmtree(merged_dir(a.out, model), ignore_errors=True)
    return rec


# ---- judge ------------------------------------------------------------------------------------
def load_gens(out: Path) -> dict[str, str]:
    gens = {}
    for f in sorted((out / "_raw" / "gen").glob("*.jsonl")):
        for r in read_rows(f):
            gens[r["job_id"]] = r["response"]
    return gens


def judge_units(items, gens: dict[str, str]) -> list[dict]:
    """Every response, k0 of every model and arm before any k1, then row by row: a deadline costs
    second generations, never a headline's first."""
    idx = {it.scenario_id: it.idx for it in items}
    units = [{"unit": f"{gid}/response", "sid": gid.split("/")[2], "text": t,
              "k": int(gid.split("/")[3][1:])} for gid, t in gens.items()]
    return sorted(units, key=lambda u: (u["k"], idx[u["sid"]], u["unit"]))


def cmd_judge(a) -> dict:
    from deepsteer.datasets.morebench import load_morebench
    tok = (P._tok(P.GEN["gpt_oss_20b"]["repo"], P.GEN["gpt_oss_20b"]["revision"], None)
           if a.dry_run else P._tok(P.JUDGE["repo"], P.JUDGE["revision"], a.local_dir))
    cfg = P.harmony_cfg()
    items = load_morebench(subset="public")
    by_sid = {it.scenario_id: it for it in items}
    units = judge_units(items, load_gens(a.out))
    engine, SPc = P.load_engine(P.JUDGE["repo"], P.JUDGE["revision"], a.local_dir, a.dry_run, tok,
                                CoTFormat.HARMONY_ANALYSIS, "yes", "judge")

    def jobs():
        for u in units:
            for c in by_sid[u["sid"]].criteria:
                jid = f"{u['unit']}/{c.criterion_id}"
                yield {"job_id": jid, "unit": u["unit"], "criterion_id": c.criterion_id,
                       "weight": c.weight,
                       "prompt_ids": P.render_ids(tok, judge_prompt(u["text"], c.title),
                                                  CoTFormat.HARMONY_ANALYSIS, P.JUDGE["effort"],
                                                  cfg),
                       "max_tokens": P.MAX_TOKENS, "seed": P.seed_of("p2-judge", jid),
                       "sampler": P.JUDGE["sampler"]}

    out, raw = a.out / "judge" / "shard0.jsonl", a.out / "_raw" / "judge" / "shard0.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)

    def decode(j, r):
        text = (P.decode_rollout(tok, r["gen_ids"], CoTFormat.HARMONY_ANALYSIS)
                if r["gen_ids"] else "")
        _, final = P.split_rollout(text, CoTFormat.HARMONY_ANALYSIS) if text else ("", "")
        final = final if r["status"] == "finished" and final else None
        short = final.strip().lower() if final is not None and len(final) <= 20 else None
        return ({"n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
                 "status": r["status"], "verdict": short,
                 "verdict_chars": None if final is None else len(final),
                 **P.credits(final, j["weight"])}, {"reply": text})

    guard = P.VerdictGuard()
    rec = P.run_chunks(engine, SPc, (j for j in jobs() if guard.admits(j)), out, raw, decode,
                       deadline=a.deadline,
                       meta_keys=("job_id", "unit", "criterion_id", "weight"), on_rows=guard)
    rec.update({"calls_planned": sum(len(by_sid[u["sid"]].criteria) for u in units),
                "guard": P.GUARD, "guard_seen": guard.seen, "guard_tripped": guard.tripped,
                "validate": a.validate, "pod_version": POD_VERSION})
    (a.out / "judge" / "shard0.summary.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---- projection -------------------------------------------------------------------------------
def project(per_model: dict, fixed: dict, judge_tps: float, envelope_h: float,
            validate_h: float) -> dict:
    """``per_model[m]``: ``tokens_per_job`` L_m, ``tokens_per_s`` r_m, ``judge_tokens_per_call``
    t_m, ``loads_h``. W = setup + HF phase + max_m(gen_m + loads_m) + sum_m judge_m + margin; the
    per-model cut (a) lowers k to 1 for the model whose cut lowers W more, then the other, then
    STOP (MRB_PLAN v0.9)."""
    n_crit = 22.9

    def wall(k: dict) -> tuple[float, dict]:
        parts = {}
        for m, v in per_model.items():
            gen = 2 * 500 * k[m] * v["tokens_per_job"] / v["tokens_per_s"] / 3600
            judge = 2 * 500 * k[m] * n_crit * v["judge_tokens_per_call"] / judge_tps / 3600
            parts[m] = {"k": k[m], "gen_h": round(gen, 3), "judge_h": round(judge, 3),
                        "gen_wall_h_per_1000_jobs": round(1000 * v["tokens_per_job"]
                                                          / v["tokens_per_s"] / 3600, 3)}
        w = (fixed["setup_h"] + fixed["hf_phase_h"] + fixed["judge_load_h"] + fixed["margin_h"]
             + max(parts[m]["gen_h"] + per_model[m]["loads_h"] for m in parts)
             + sum(p["judge_h"] for p in parts.values()))
        return w, parts

    remaining = envelope_h - validate_h
    levels, k = [], {m: 2 for m in per_model}
    w, parts = wall(k)
    levels.append({"cut": "none", "wall_h": round(w, 3), "a100_h": round(2 * w, 3), "models": parts})
    while 2 * w > remaining and any(v == 2 for v in k.values()):
        options = []
        for m in [m for m in k if k[m] == 2]:
            w2, _ = wall({**k, m: 1})
            options.append((w2, m))
        w, m = min(options)
        k = {**k, m: 1}
        w, parts = wall(k)
        levels.append({"cut": f"a: {m} k=1", "wall_h": round(w, 3), "a100_h": round(2 * w, 3),
                       "models": parts})
    fits = 2 * w <= remaining
    return {"envelope_h": envelope_h, "validate_h": validate_h, "remaining_h": round(remaining, 3),
            "levels": levels, "k": k if fits else None, "decision": "proceed" if fits else "STOP",
            "session_wall_h": round(w, 3), "session_a100_h": round(2 * w, 3),
            "session_max_wall_h": round(min(1.15 * w, remaining / 2), 3) if fits else None}


def cmd_project(a) -> dict:
    rows = read_rows(a.out / "judge" / "shard0.jsonl")
    judge_s = json.loads((a.out / "judge" / "shard0.summary.json").read_text())
    per_model, hf_h = {}, []
    for m in MODELS:
        g = [json.loads((a.out / "gen" / f"{m}.{arm}.summary.json").read_text()) for arm in ARMS]
        ran = [x for x in g if x.get("status") == "ok" and x.get("jobs_run")]
        toks = sum(x["decoded_tokens"] for x in ran)
        jobs = sum(x["jobs_run"] for x in ran)
        secs = sum(x["wall_s"] for x in ran)
        rs = [r for r in rows if r["unit"].startswith(m + "/")]
        per_model[m] = {"tokens_per_job": toks / max(jobs, 1), "tokens_per_s": toks / max(secs, 1e-9),
                        "judge_tokens_per_call": sum(r["n_tokens"] for r in rs) / max(len(rs), 1),
                        "loads_h": sum(x.get("engine_load_h", 0.05) for x in ran) or 0.1,
                        "arms_ok": [x["arm"] for x in ran]}
        hf_h.append(json.loads((a.out / "mb_phase2" / m / "hf_phase.json").read_text())["wall_h"])
    fixed = {"setup_h": a.setup_hours, "hf_phase_h": max(hf_h), "judge_load_h": 0.05,
             "margin_h": 0.25}
    rec = {"measured": {"per_model": per_model, "fixed": fixed,
                        "judge_tokens_per_s": judge_s["tokens_per_s"]},
           **project(per_model, fixed, judge_s["tokens_per_s"], a.envelope_h, a.validate_hours)}
    (a.out / "projection.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["hf-phase", "generate", "judge", "project"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--local-dir", type=Path, default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--envelope-h", type=float, default=9.0)  # MRB_PLAN v0.9
    ap.add_argument("--validate-hours", type=float, default=0.0)
    ap.add_argument("--setup-hours", type=float, default=0.25)
    a = ap.parse_args()
    rec = {"hf-phase": cmd_hf_phase, "generate": cmd_generate, "judge": cmd_judge,
           "project": cmd_project}[a.cmd](a)
    print(json.dumps(rec, default=str)[:1500])


if __name__ == "__main__":
    main()
