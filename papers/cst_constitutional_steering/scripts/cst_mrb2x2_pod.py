#!/usr/bin/env python3
"""The MoReBench pod, stage 1 (CST Amendment 46 item B): A0s, P-ctx and P-ctx-n at k 2 on the 500.

    cst_mrb2x2_pod.py --stage <stage> [--tag p_ctx] --out DIR

MRB's protocol for every arm (MRB's tokenizer, render, sampler, cap 10,500 and seeds, so each
arm's rows pair with MRB's saved base rows by job id), MRB's judge (gpt-oss-120b, effort high,
on one GPU).

- ``smoke_hf`` / ``smoke_serve`` / ``parity_hf`` / ``parity``: the fixed smoke check (Amendment 31)
  and each adapter's served parity, HF and vLLM in separate processes.
- ``a0s``: the constitution in context (A0s's render: the header and `C_anth` in a system message
  before MoReBench's user prompt), on A0s's engine (32 sequences; ``max_model_len`` 46,080, since
  the longest A0s prompt, 34,722 tokens, plus the 10,500 cap exceeds the engine of record's 44,032;
  capacity only, sampling unchanged). **The gate at load:** after the first 128 rollouts, A0s's
  measured answer length and rate forecast the pod; past 10.5 h, P-ctx-n moves to a later pod
  (``gate.json``) and the runner skips it.
- ``adapter --tag p_ctx|p_ctx_n``: the adapter's 1,000 rollouts on the LoRA engine; with
  ``p_ctx``, the session check (Amendment 42 item 4: the base on 50 seeded scenarios at k 2).
- ``judge``: MRB's judge on every rollout not yet judged (resumable, to the deadline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts"),
                str(HERE.parents[1] / "mrb_morebench/scripts"),
                str(HERE.parents[1] / "st_live_reference/scripts")]
import cst_finalist_pod as F  # noqa: E402

MODEL = F.MODEL
ADAPTERS = ("p_ctx", "p_ctx_n")
A0S_MAX_LEN = 46_080
GATE_N, PRICE_H, ASSUMED_TOKENS = 128, 10.5, 2900
JUDGE_H, ADAPTER_GEN_H, LOAD_H = 1.13, 0.96, 0.1  # the bars pod's measured stage times


def start_epoch() -> float:
    s = os.environ.get("CST_START_EPOCH")
    return float(s) if s else time.time()


def a0s_jobs(tok, fmt, dry: bool) -> list[dict]:
    """MRB's gen_jobs with A0s's render: the constitution's system message before the user turn."""
    return doc_jobs("a0s", tok, fmt, dry)


def doc_jobs(arm: str, tok, fmt, dry: bool) -> list[dict]:
    """MRB's gen_jobs with a document arm's render (A0s: `C_anth`; A0n: Faraday's lectures,
    Amendment 49 item 2b), the document's system message before the user turn."""
    import cst_pressure as CP
    import mrb_g1_pod as P
    import mrb_p2_pod as M

    from deepsteer.datasets.morebench import load_morebench

    body = F.S.doc_body(arm, dry)
    jobs = []
    items = load_morebench(subset="public")[: 2 if dry else None]
    for it in items:
        msgs = CP.doc_messages(arm, [{"role": "user",
                                      "content": M.generation_prompt(it.dilemma)}], body)
        ids = ([1, 2, 3] if dry else
               fmt.prompt_ids(tok, fmt.render(tok, msgs, M.EFFORT, M.DATE_PIN[MODEL])))
        for k in range(2):
            jobs.append({"sid": it.scenario_id, "k": k, "prompt_ids": ids,
                         "max_tokens": 8 if dry else P.MAX_TOKENS,
                         "seed": M.seed(MODEL, it.scenario_id, k), "sampler": M.SAMPLER,
                         "job_id": f"{MODEL}/{arm}/{it.scenario_id}/k{k}"})
    return jobs


def doc_engine(arm: str, dry: bool):
    """A document arm's engine: the engine of record at 32 sequences, ``max_model_len`` 46,080."""
    import mrb_p2_pod as M

    if dry:
        tok, _ = _tok_fmt(True)
        return M.StubLLM(MODEL, tok).llm_engine, None
    from vllm import SamplingParams as SP  # noqa: N814

    kw = F.S.engine_kwargs(arm)
    kw["max_model_len"] = A0S_MAX_LEN
    path = F.S.V._model_path(False)  # noqa: SLF001
    return F.S.V.VLLMEngine(path, None, **kw).llm.llm_engine, SP


def _tok_fmt(dry: bool):
    import mrb_p2_pod as M

    tok = M.tokenizer(MODEL) if not dry else F.S.V._tok(F.S.V._model_path(True))  # noqa: SLF001
    return tok, (M.fmt_for(MODEL, tok) if not dry else None)


def stage_a0s(out: Path, dry: bool) -> dict:
    tok, fmt = _tok_fmt(dry)
    jobs = a0s_jobs(tok, fmt, dry)
    engine, SP = doc_engine("a0s", dry)
    name = f"{MODEL}.a0s.jsonl"
    gen, raw = out / "mrb" / "gen" / name, out / "mrb" / "_raw" / "gen" / name
    t0 = time.time()
    pilot = F._mrb_run(engine, SP, jobs[:GATE_N], gen, raw, fmt, tok)  # noqa: SLF001
    rows = F.read_rows(gen)
    tok_mean = sum(r["n_tokens"] for r in rows) / max(len(rows), 1)
    wall = max(time.time() - t0, 1e-9)
    rate = sum(r["n_tokens"] for r in rows) / wall  # decoded tokens per second at load
    rest_h = (len(jobs) - len(rows)) * tok_mean / max(rate, 1e-9) / 3600
    elapsed = (time.time() - start_epoch()) / 3600
    forecast = elapsed + rest_h + 3 * (JUDGE_H + LOAD_H) + 2 * (ADAPTER_GEN_H + LOAD_H)
    gate = {"rule": "Amendment 46 item B3 (the gate at load)", "pilot_rollouts": len(rows),
            "mean_tokens": tok_mean, "assumed_tokens": ASSUMED_TOKENS,
            "tokens_per_s": rate, "a0s_rest_h": rest_h, "elapsed_h": elapsed,
            "forecast_h": forecast, "price_h": PRICE_H,
            "p_ctx_n": ("this pod" if forecast <= PRICE_H - 0.25
                        else "a later pod (the card says so)"),
            "pilot_run": pilot}
    F.T._w(out, "gate", gate)  # noqa: SLF001
    print(">> gate:", json.dumps({k: v for k, v in gate.items() if k != "pilot_run"}), flush=True)
    rec = F._mrb_run(engine, SP, jobs, gen, raw, fmt, tok)  # noqa: SLF001
    rec |= {"rule": "Amendment 46 item B1 (A0s)", "n_jobs": len(jobs),
            "max_model_len": A0S_MAX_LEN}
    F.T._w(out, "gen_a0s", rec)  # noqa: SLF001
    return rec


def stage_adapter(out: Path, tag: str, dry: bool) -> dict:
    import mrb_p2_pod as M

    tok, fmt = _tok_fmt(dry)
    if dry:
        from deepsteer.datasets.morebench import load_morebench

        sids = [it.scenario_id for it in load_morebench(subset="public")[:2]]
        jobs = [{"sid": s, "k": k, "prompt_ids": [1, 2, 3], "max_tokens": 8, "seed": i + k,
                 "sampler": M.SAMPLER} for i, s in enumerate(sids) for k in range(2)]
        stub = M.StubLLM(MODEL, tok)
        cand, base, SP = stub.llm_engine, stub.llm_engine, None
    else:
        jobs = M.gen_jobs(MODEL, tok, fmt, 2, False)
        from vllm import SamplingParams as SP  # noqa: N814

        _, eng, reqs = F.engine_and_req(out, tag, dry)
        cand, base = F.EngineProxy(eng.llm.llm_engine, reqs["cand"]), eng.llm.llm_engine
    name = f"{MODEL}.{tag}.jsonl"
    aj = [dict(j, job_id=f"{MODEL}/{tag}/{j['sid']}/k{j['k']}") for j in jobs]
    rec = F._mrb_run(cand, SP, aj, out / "mrb" / "gen" / name,  # noqa: SLF001
                     out / "mrb" / "_raw" / "gen" / name, fmt, tok)
    rec |= {"rule": f"Amendment 46 item B1 ({tag})", "n_jobs": len(aj)}
    if tag == "p_ctx":  # the session check on the same engine (Amendment 42 item 4)
        keep = set(F.session_sids()) if not dry else {j["sid"] for j in jobs}
        sj = [dict(j, job_id=f"{MODEL}/none/{j['sid']}/k{j['k']}") for j in jobs
              if j["sid"] in keep]
        rec["session_check"] = F._mrb_run(  # noqa: SLF001
            base, SP, sj, out / "mrb_session" / "gen.jsonl", out / "mrb_session" / "_raw.jsonl",
            fmt, tok)
    F.T._w(out, f"gen_{tag}", rec)  # noqa: SLF001
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=(
        "smoke_hf", "smoke_serve", "parity_hf", "parity", "a0s", "adapter", "judge"))
    ap.add_argument("--tag", choices=ADAPTERS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    F.configure()
    out, dry = a.out, a.dry_run
    if a.stage == "smoke_hf":
        path, wb, _ = F.PL.load_model(dry)
        rec = F.PL.smoke_hf_side(wb, path, out, dry, [Path(F.cand_dir(out, t)) for t in ADAPTERS])
        F.T._w(out, "smoke_hf", rec)  # noqa: SLF001
    elif a.stage == "smoke_serve":
        print(">> smoke_serve:", json.dumps(F.PL.smoke_serve(out, dry)))
    elif a.stage == "parity_hf":
        F.PL.hf_letters(out, [(t, F.cand_dir(out, t)) for t in ADAPTERS], dry)
    elif a.stage == "parity":
        F.PL.stage_read(out, [(t, F.cand_dir(out, t)) for t in ADAPTERS], dry, parts=(),
                        base=False)
    elif a.stage == "a0s":
        stage_a0s(out, dry)
    elif a.stage == "adapter":
        stage_adapter(out, a.tag, dry)
    else:
        F.stage_mrb_judge(out, dry)


if __name__ == "__main__":
    main()
