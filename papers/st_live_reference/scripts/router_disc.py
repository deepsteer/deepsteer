#!/usr/bin/env python3
"""ST-A15's discriminator (author, 2026-10-21; ~0.1 A100-h, on pod B before the Phase 2 driver):
why the bf16 checkpoint on vLLM sits further from the HF stack than the MXFP4 engine on T1.5's 64
parity rows (the exact prompt ids of T1.5's parity jobs).

    python3 router_disc.py --phase hf    --jobs <mxfp4_parity.jobs.jsonl> --out <dir>
    python3 router_disc.py --phase vllm  --jobs <...> --out <dir>    (the vLLM venv; empty GPU;
                                                                    each engine its own process)
    python3 router_disc.py --phase read  --jobs <...> --out <dir>    (local, zero GPU)

- ``hf``: the HF stack of record (bf16 dequantized, eager), each row alone: letter log-probs and
  every layer's router logits (block ``mlp.router``'s first output) over the prompt; then the bf16
  checkpoint written as T1.5 wrote it.
- ``vllm``: (a) R_d, batch composition: the bf16 engine at T1.5's settings on the 64 rows alone
  (T1.5 read them inside one run with ~1,250 generation requests, the MXFP4 engine alone); (b) the
  bf16 and the MXFP4 engines in eager mode, each row alone, every layer's router logits captured by
  a forward hook on ``mlp.router`` (no torch.compile, no CUDA graphs, prefix caching off); the
  checkpoint is deleted after.
- ``read``: letter TV of each read against HF; per row, the share of (token, layer) router top-4
  sets that differ from HF's, and the HF top-4/top-5 margin at those positions. R_a (router flips)
  predicts the three bf16-odd rows differ at a few near-tied positions and quiet rows do not; R_c
  (layout error) predicts differences on every row in the same layers; R_d (batch composition)
  predicts the rows read alone fall back to the MXFP4 band.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]

KEY = "gpt_oss_20b"
ODD_BF = (4, 15, 40)  # ST-A15: bf16-vLLM odd (H1-A-218 p0, H1-B-19 p0, H1-A-102 p2)
ODD_HF = (19, 39)  # HF odd (H1-B-49 p0-1)
ENGINES = ("bf16_alone_t15_settings", "bf16_eager", "mxfp4_eager")


def _jobs(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def phase_hf(a) -> dict:
    import pod_st_screen as P
    import t15_driver as T
    import torch
    from kdg_pod_lib import ModelWrapper

    spec = P.registry()[KEY]
    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = st_tiny.build(KEY, a.out / "_tiny")
        mw = ModelWrapper(str(path), None, torch_dtype=torch.float32)
    else:
        mw = ModelWrapper(spec["repo"], spec["revision"], torch_dtype=torch.bfloat16,
                          attn_implementation="eager", **P.hf_load_kwargs(spec))
    jobs = _jobs(a.jobs)
    layers = mw.model.model.layers
    grab: list = []
    hooks = [lay.mlp.router.register_forward_hook(
        lambda _m, _i, o, i=i: grab.append((i, o[0].detach().float().cpu().numpy())))
        for i, lay in enumerate(layers)]
    letters, logits = [], []
    try:
        for j in jobs:
            grab.clear()
            with torch.no_grad():
                out = mw.model(input_ids=torch.tensor([j["prompt_ids"]], device=mw.model.device))
            lp = torch.log_softmax(out.logits[0, -1].float(), -1)
            letters.append(lp[j["logprob_ids"]].cpu().numpy())
            logits.append(np.stack([x for _i, x in sorted(grab, key=lambda t: t[0])]))
    finally:
        for h in hooks:
            h.remove()
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "hf_letters.npy", np.array(letters))
    np.savez_compressed(a.out / "hf_router.npz", *[x.astype(np.float16) for x in logits])
    saved = T.save_bf16(mw.model, a.out / "ckpt") if not a.dry_run else {"ok": None}
    return {"n_rows": len(jobs), "n_layers": len(layers), "ckpt": saved}


def _engine(spec: dict, model: str, eager: bool):
    from vllm import LLM

    v = spec["vllm"]
    kw = dict(dtype=v["dtype"], kv_cache_dtype=v["kv_cache_dtype"],
              max_model_len=v["max_model_len"],
              gpu_memory_utilization=v["gpu_memory_utilization"], seed=0,
              generation_config="vllm", tokenizer=spec["repo"],
              tokenizer_revision=spec["revision"])
    if eager:
        kw.update(enforce_eager=True, enable_prefix_caching=False, max_num_seqs=1)
    else:  # T1.5's settings
        kw.update(enable_prefix_caching=True, enable_chunked_prefill=True,
                  max_num_seqs=v["max_num_seqs"])
    rev = None if model != spec["repo"] else spec["revision"]
    return LLM(model=model, revision=rev, **kw)


def _first_letters(llm, jobs: list[dict], one_at_a_time: bool) -> list[dict]:
    from vllm import SamplingParams, TokensPrompt

    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    out = []
    batches = [[j] for j in jobs] if one_at_a_time else [jobs]
    for b in batches:
        res = llm.generate([TokensPrompt(prompt_token_ids=j["prompt_ids"]) for j in b], sp,
                           use_tqdm=False)
        for j, r in zip(b, res):
            first = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            out.append({str(t): float(first[t].logprob) for t in j["logprob_ids"] if t in first})
    return out


def _router_reads(llm, jobs: list[dict]) -> list[np.ndarray]:
    """Eager engine, one row at a time: every layer's router logits over the prompt."""
    def register(model):
        model._st_router = []  # noqa: SLF001
        layers = model.model.layers
        model._st_hooks = [lay.mlp.router.register_forward_hook(  # noqa: SLF001
            lambda _m, _i, o, i=i: model._st_router.append(  # noqa: SLF001
                (i, (o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy())))
            for i, lay in enumerate(layers)]
        return len(layers)

    def fetch(model):
        got = list(model._st_router)  # noqa: SLF001
        model._st_router.clear()  # noqa: SLF001
        return got

    llm.apply_model(register)
    out = []
    for j in jobs:
        _first_letters(llm, [j], True)
        calls = llm.apply_model(fetch)[0]
        by: dict[int, list] = {}
        for i, x in calls:  # chunked prefill: one call per chunk, in order
            by.setdefault(i, []).append(x)
        out.append(np.stack([np.concatenate(by[i]) for i in sorted(by)]))
    return out


def run_engine(a) -> dict:
    """One engine, this process (the hooks live where the model runs)."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import pod_st_screen as P

    spec = P.registry()[KEY]
    jobs = _jobs(a.jobs)
    model = str(a.out / "ckpt") if a.engine.startswith("bf16") else spec["repo"]
    eager = a.engine.endswith("eager")
    llm = _engine(spec, model, eager)
    first = _first_letters(llm, jobs, one_at_a_time=True)
    (a.out / f"{a.engine}_first.json").write_text(json.dumps(first))
    if eager:
        r = _router_reads(llm, jobs)
        np.savez_compressed(a.out / f"{a.engine}_router.npz", *[x.astype(np.float16) for x in r])
    return {"ok": True, "n_rows": len(first)}


def phase_vllm(a) -> dict:
    """Each engine in its own process (GPU memory returns between engines); a failure is
    recorded and the session goes on to Phase 2; the checkpoint is deleted after."""
    import shutil
    import subprocess

    res: dict = {}
    for name in ENGINES:
        cmd = [sys.executable, __file__, "--phase", "engine", "--engine", name, "--jobs",
               str(a.jobs), "--out", str(a.out)]
        rc = subprocess.run(cmd).returncode
        res[name] = {"rc": rc, "ok": rc == 0}
    shutil.rmtree(a.out / "ckpt", ignore_errors=True)
    return res


def tv_rows(a: np.ndarray, b: np.ndarray, shown: list[list[int]]) -> np.ndarray:
    out = []
    for x, y, idx in zip(a, b, shown):
        px, py = np.exp(x[idx]), np.exp(y[idx])
        out.append(1.0 if px.sum() <= 0 or py.sum() <= 0 else
                   0.5 * float(np.abs(px / px.sum() - py / py.sum()).sum()))
    return np.array(out)


def router_compare(hf: list[np.ndarray], other: list[np.ndarray], k: int = 4) -> list[dict]:
    """Per row: share of (layer, token) top-k sets that differ, the layers involved, and HF's
    k-th minus (k+1)-th logit at the differing positions (near ties are small)."""
    rows = []
    k = min(k, hf[0].shape[-1] - 1)  # GPT-OSS: top-4 of 32; a 4-expert tiny model: top-3
    for h, o in zip(hf, other):
        n = min(h.shape[1], o.shape[1])
        h, o = h[:, :n].astype(np.float32), o[:, :n].astype(np.float32)
        th = np.sort(np.argsort(-h, -1)[..., :k], -1)
        to = np.sort(np.argsort(-o, -1)[..., :k], -1)
        diff = np.any(th != to, -1)
        srt = -np.sort(-h, -1)
        margin = srt[..., k - 1] - srt[..., k]
        rows.append({"share_differ": float(diff.mean()), "n_differ": int(diff.sum()),
                     "layers": sorted({int(x) for x in np.where(diff)[0]}),
                     "first_layer": int(np.where(diff)[0].min()) if diff.any() else None,
                     "margin_at_differ_median": float(np.median(margin[diff]))
                     if diff.any() else None,
                     "margin_all_median": float(np.median(margin))})
    return rows


def shown_columns(n: int) -> list[list[int]]:
    """T1.5's parity rows (t15_driver.parity_rows): the 20 gate scenarios at permutations 0-3, the
    first 64; each row's displayed letters as columns of the A..E readout."""
    import pod_st_screen as P

    from deepsteer.kdg.schema import assign_letters, load_scenario_dir

    hs, _ = load_scenario_dir([P.H1_SET])
    H = {s.id: s for s in hs}
    gate = json.loads((HERE.parent / "data" / "t13_draw_v1.json").read_text())["gate"]
    rows = [(sid, k) for k in range(4) for sid in gate][:n]
    return [["ABCDE".index(x) for x, _ in assign_letters(H[sid], k)] for sid, k in rows]


def phase_read(a) -> dict:
    jobs = _jobs(a.jobs)
    lids = jobs[0]["logprob_ids"]
    hf = np.load(a.out / "hf_letters.npy")
    shown = shown_columns(len(jobs))

    def letters(first: list[dict]) -> np.ndarray:
        return np.array([[f.get(str(t), -np.inf) for t in lids] for f in first])

    out: dict = {"rows_bf_odd": list(ODD_BF), "rows_hf_odd": list(ODD_HF)}
    for name in ENGINES:
        f = a.out / f"{name}_first.json"
        if f.exists():
            t = tv_rows(letters(json.loads(f.read_text())), hf, shown)
            out[f"tv_vs_hf_{name}"] = {"mean": float(t.mean()), "rows_bf_odd":
                                       [float(t[i]) for i in ODD_BF if i < len(t)],
                                       "per_row": t.tolist()}
    hz = np.load(a.out / "hf_router.npz")
    hr = [hz[k] for k in hz.files]
    for name in ("bf16_eager", "mxfp4_eager"):
        f = a.out / f"{name}_router.npz"
        if f.exists():
            z = np.load(f)
            cmp_ = router_compare(hr, [z[k] for k in z.files])
            quiet = [i for i in range(len(cmp_)) if i not in ODD_BF + ODD_HF]
            out[f"router_{name}"] = {
                "per_row": cmp_,
                "share_differ_bf_odd": [cmp_[i]["share_differ"] for i in ODD_BF
                                        if i < len(cmp_)],
                "share_differ_quiet_mean": float(np.mean([cmp_[i]["share_differ"]
                                                          for i in quiet]))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("hf", "vllm", "engine", "read"), required=True)
    ap.add_argument("--engine", choices=ENGINES, default=None)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    res = {"hf": phase_hf, "vllm": phase_vllm, "engine": run_engine,
           "read": phase_read}[a.phase](a)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.engine if a.phase == "engine" else a.phase
    (a.out / f"router_disc_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f">> router_disc {a.phase}: {json.dumps(res)[:600]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
