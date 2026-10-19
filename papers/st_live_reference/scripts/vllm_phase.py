"""Amendment 7: the vLLM side of a screen, driven from the HF venv (no model loaded here).

``engine_json`` builds the worker's engine config from a model's registry ``vllm:`` block (every
setting explicit, so the settings hash covers it); ``run_vllm`` writes a jobs file, runs
``gen_vllm.py`` in the vLLM venv as a subprocess (one engine configuration per process) and reads
the rows back in job order. In a dry run the stub generates in-process, with the same row schema,
so the two-phase flow is exercised locally.

The speculation probe (author, Amendment 7 item 7): the same jobs run with and without the
registered drafter; the faster setting (decoded tokens per second over the run) is the setting of
record for that regime and every row of record comes from it; the slower run's rows are banked as
probe duplicates and never enter a statistic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VLLM_PY = os.environ.get("ST_VLLM_PYTHON", "/workspace/venv_vllm/bin/python")
WORKER = HERE / "gen_vllm.py"
# The worker's environment (recorded in its settings hash, which covers every VLLM_* variable).
# FlashInfer's fused top-k/top-p sampler is never used by these requests (vLLM v0.31.0 skips the
# fused sampler when no request sets top-k/top-p or any request carries an explicit seed; ours do
# both, so they take the seeded gumbel path), but engine warm-up calls it with dummy settings and
# JIT-compiles it, which failed on the third VALIDATE pod (no ninja on PATH) and would cost minutes
# per fresh pod; off. The venv's bin goes first on the worker's PATH so any other runtime kernel
# build finds the venv's own tools (ninja).
WORKER_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def worker_env() -> dict:
    env = dict(os.environ, **WORKER_ENV)
    env["PATH"] = str(Path(VLLM_PY).parent) + os.pathsep + env.get("PATH", "")
    return env


def engine_json(spec: dict, *, spec_on: bool, label: str, fallback: bool = False) -> dict:
    """The worker's engine config. Speculation from the registry's ``draft`` (or
    ``draft_fallback`` when ``fallback``); rejection sampling pinned to the lossless method."""
    v = spec["vllm"]
    kw = {"dtype": v["dtype"], "kv_cache_dtype": v["kv_cache_dtype"],
          "max_model_len": v["max_model_len"], "max_num_seqs": v["max_num_seqs"],
          "gpu_memory_utilization": v["gpu_memory_utilization"],
          "enable_prefix_caching": True, "enable_chunked_prefill": True,
          "generation_config": "vllm", "seed": 0,
          # the offline LLM disables stats by default; on, so the speculative acceptance counters
          # reach the run summary (logging only; no numeric effect)
          "disable_log_stats": False}
    if v.get("language_model_only"):
        kw["language_model_only"] = True
    if spec_on:
        d = dict(v["draft_fallback"] if fallback else v["draft"])
        sc = {"method": d["method"], "num_speculative_tokens": d["num_speculative_tokens"],
              "rejection_sample_method": "standard"}
        if d.get("repo"):
            sc["model"] = d["repo"]
            sc["revision"] = d["revision"]
        if d.get("prompt_lookup_max"):
            sc["prompt_lookup_max"] = d["prompt_lookup_max"]
        kw["speculative_config"] = sc
    return {"model": spec["repo"], "revision": spec["revision"], "label": label,
            "vllm_version": v["version"], "llm_kwargs": kw,
            "draft_pins": ({k: (v["draft_fallback"] if fallback else v["draft"]).get(k)
                            for k in ("repo", "revision", "config_sha256", "weights_sha256",
                                      "source")} if spec_on else None)}


def run_vllm(sc, jobs: list[dict], engine: dict, out: Path, *, deadline: float | None = None,
             after_h: float = 0.0, expected_len: int = 0) -> tuple[list[dict], dict, int]:
    """Run ``jobs`` (dicts with job_id, prompt_ids, max_tokens, seed[, sampler]) through the
    vLLM worker; returns (rows in job order, summary, exit code)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if sc.dry:
        return _dry(sc, jobs, engine, out)
    jobs_path = out.with_suffix(".jobs.jsonl")
    eng_path = out.with_suffix(".engine.json")
    jobs_path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    eng_path.write_text(json.dumps(engine, indent=1))
    cmd = [VLLM_PY, str(WORKER), "--engine", str(eng_path), "--jobs", str(jobs_path),
           "--out", str(out), "--after-h", str(after_h), "--expected-len", str(expected_len)]
    if deadline is not None:
        cmd += ["--deadline", str(deadline)]
    rc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=worker_env()).returncode
    if rc not in (0, 3):
        raise RuntimeError(f"vLLM worker failed (exit {rc}) on {engine['label']}")
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    summary = json.loads(Path(str(out) + ".summary.json").read_text())
    return rows, summary, rc


def _dry(sc, jobs, engine, out):
    """Stub generation with the worker's row schema; the speculative setting changes the stub's
    simulated speed only (a lossless drafter changes no distribution)."""
    sc.model.effort = sc.effort
    ids, _, secs = sc.model.sample_from_ids([j["prompt_ids"] for j in jobs],
                                            cap=max(j["max_tokens"] for j in jobs), seed=0,
                                            batch=len(jobs))
    rows = []
    for j, g in zip(jobs, ids):
        g = (g or [])[: j["max_tokens"]]
        rows.append({"job_id": j["job_id"], "gen_ids": g, "n_tokens": len(g),
                     "status": "finished", "finish_reason": "length"
                     if len(g) >= j["max_tokens"] else "stop", "t_finish": 0.0})
    spec = engine["llm_kwargs"].get("speculative_config")
    wall = float(sum(secs)) * (getattr(sc.model, "spec_factor", 0.6) if spec else 1.0)
    dec = sum(r["n_tokens"] for r in rows)
    summary = {"label": engine["label"], "engine": engine, "versions": {"vllm": "dry"},
               "settings_sha256": "dry", "wall_seconds": wall, "decoded_tokens": dec,
               "tokens_per_second": dec / max(wall, 1e-9), "stopped": None,
               "n_finished": len(rows), "n_jobs": len(jobs), "load_seconds": 0.0}
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    return rows, summary, 0


def probe_and_run(sc, jobs: list[dict], spec: dict, out_dir: Path, regime: str, **kw) -> dict:
    """Run ``jobs`` with and without speculation; the faster run is of record. Returns
    ``{rows, summary, choice, probe}``."""
    runs = {}
    for on in (True, False):
        tag = "spec" if on else "nospec"
        eng = engine_json(spec, spec_on=on, label=f"{sc.key}:{regime}:{tag}")
        rows, summ, rc = run_vllm(sc, jobs, eng, out_dir / f"{regime}_{tag}.jsonl", **kw)
        runs[tag] = {"rows": rows, "summary": summ, "rc": rc}
    tps = {t: r["summary"]["tokens_per_second"] for t, r in runs.items()}
    choice = max(tps, key=tps.get)
    return {"rows": runs[choice]["rows"], "summary": runs[choice]["summary"], "choice": choice,
            "probe": {"regime": regime, "tokens_per_second": tps, "choice": choice,
                      "settings_sha256": {t: r["summary"]["settings_sha256"]
                                          for t, r in runs.items()},
                      "duplicate_rows": f"{regime}_{'nospec' if choice == 'spec' else 'spec'}"
                                        ".jsonl (probe duplicate; never a statistic)"}}
