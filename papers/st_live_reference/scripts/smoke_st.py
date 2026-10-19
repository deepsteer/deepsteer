#!/usr/bin/env python3
"""VALIDATE=1 on the pod: the first real vLLM run (ST_PLAN Amendment 7), inside the remote
script's own envelope stop (~0.3 A100-h).

    python3 papers/st_live_reference/scripts/smoke_st.py --out OUT

1. vLLM on GPT-OSS-20B (the development model): 8 of §40's parity prompts (sha-checked), one
   rollout each, max_tokens 256, with and without the registered EAGLE3 drafter: two worker
   processes, each recording its settings hash, decoded tokens per second and speculative
   counters; every row parsed through the harmony format (statuses and letters).
2. HF GPT-OSS-20B, bf16, eager: the recalibrated padded check (Amendment 6) and the fast-path
   rung (flex attention, then torch.compile; Amendment 7) on 8 prompts, on the GPU.

Writes OUT/smoke_record.json. Exit 0 iff both vLLM runs finished every job and the HF checks ran
(their pass/fail is recorded, not gated here: the relaunch's own gates decide).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_lib as L  # noqa: E402
import screen_vllm as SV  # noqa: E402
import vllm_phase as V  # noqa: E402
from reasoning_formats import build_format  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

KEY = "gpt_oss_20b"
N = 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import pod_st_screen as P
    from transformers import AutoTokenizer

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    out = a.out / KEY
    sc = L.Screen(KEY, spec, L.tok_only(tok), fmt, out, "2026-10-04")
    sc.effort = fmt.effort_ladder[0]
    Hs, _ = load_scenario_dir([P.H1_SET])
    ctx = SimpleNamespace(order=L.h1_train_order(json.loads(P.SPLIT.read_text())),
                          H={s.id: s for s in Hs})
    jobs, ids = SV.parity_jobs(sc, ctx)
    jobs = [j for j in jobs if j["k"] == 0][:N]
    rec: dict = {"model": KEY, "n": len(jobs), "vllm": {}}
    ok = True
    out.mkdir(parents=True, exist_ok=True)

    def save() -> None:  # after every step, so an envelope stop still leaves the evidence
        (out / "smoke_record.json").write_text(json.dumps(rec, indent=1,
                                                          default=L._json_default))

    for on in (True, False):
        tag = "spec" if on else "nospec"
        eng = V.engine_json(spec, spec_on=on, label=f"smoke:{tag}")
        wj = SV.worker_jobs(jobs, f"smoke_{tag}", 256)
        rows, summ, rc = V.run_vllm(sc, wj, eng, out / "smoke" / f"vllm_{tag}.jsonl")
        by = {r["job_id"]: r for r in rows}
        gen = [by[j["job_id"]]["gen_ids"] if by[j["job_id"]]["status"] == "finished" else None
               for j in wj]
        parsed = L.build_rows(sc, jobs, gen, [j["seed"] for j in wj], cap=256, stage="smoke",
                              engine="vllm")
        rec["vllm"][tag] = {
            "rc": rc, "finished": summ["n_finished"], "tokens_per_second":
            summ["tokens_per_second"], "load_seconds": summ["load_seconds"],
            "settings_sha256": summ["settings_sha256"], "versions": summ["versions"],
            "spec_counters": summ.get("spec_counters"),
            "statuses": L._counts(r["status"] for r in parsed),
            "letters": L._counts(r.get("letter") for r in parsed)}
        ok &= summ["n_finished"] == len(wj)
        print(f">> smoke vLLM {tag}: {rec['vllm'][tag]}", flush=True)
        save()
    import torch
    from kdg_pod_lib import ModelWrapper

    t0 = time.time()
    sc.model = ModelWrapper(spec["repo"], spec["revision"], torch_dtype=torch.bfloat16,
                            attn_implementation="eager")
    rec["hf_load_seconds"] = time.time() - t0
    probe = [j["ids"] for j in jobs]
    rec["padded_check"] = L.padded_check(sc, probe)
    save()
    rec["fast_path_rung"] = L.fast_path_rung(sc, probe, SV.HF_FAST_ATTN[KEY])
    print(f">> smoke HF padded: ok={rec['padded_check']['ok']} "
          f"meas={rec['padded_check']['max_abs_nats']:.4f} bar={rec['padded_check']['bar']:.4f}; "
          f"fast path kept: {rec['fast_path_rung']['kept']}", flush=True)
    rec["ok"] = bool(ok)
    save()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
