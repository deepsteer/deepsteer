"""ST-A36's generation read on Qwen3.8-27B (ST_PLAN Amendment 59): the panel's 60 rule-right items
(the specificity read's own items, frame and options) under generation with reasoning on vLLM x 8,
for ``none`` and pod 2's two adapters. ``st2qp_driver``'s configuration of ``t16_driver`` is reused,
so the parity rows are pod 2's. Each step its own process, in the runner's order:

- ``gen --arm none`` (vLLM, the hub snapshot): pod 2's 64 parity rows' first-token letters and the
  items x 8; the base check against pod 2's saved HF base letters (at most 0.10).
- ``merge --arm care|base``: pod 2's step-237 adapter (sha256 against its sidecar) written into a
  copy of the base snapshot by T1.6's in-place writer; its weight kept share must equal pod 2's to
  1e-6.
- ``gen --arm care|base`` (vLLM, the merged copy): parity letters against pod 2's saved
  HF-attached letters (at most pod 2's band + 0.05) and the items x 8; an arm failing parity is set
  aside (no fallback is priced). The merged copy is removed after its arm.

The analysis of record is local (``analyze_st2qp.py --a36``).

    python3 st2qg_driver.py --step gen|merge --arm none|care|base --out <dir> \
        --pod1 <st2q/qwen38_27b> --pod2 <st2qp/qwen38_27b> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402
import st2qp_driver as QP  # noqa: E402
import t16_driver as T16  # noqa: E402

KEY = QP.KEY
STAGE = "st2qg"  # this read's seeds; the parity rows stay pod 2's (``QP.parity_scenarios``)
ARMS = ("care", "base")
RR = 8
BASE_BAR, PARITY_MARGIN, KEPT_TOL = T16.BASE_BAR, T16.PARITY_MARGIN, 1e-6


def context(a) -> dict:
    QP.configure()
    c = QP.context(a)
    c["record"] = c["out"] / "st2qg_record.json"
    c["pod2"] = a.pod2
    c["adapters"] = (c["out"] / "adapters") if a.dry_run else a.pod2 / "adapters"
    c["items"] = c["spec_ids"]  # QP.context cuts these to 2 in a dry run
    c["rr"] = 2 if a.dry_run else RR
    return c


def pod2_record(c: dict) -> dict:
    p = c["pod2"] / "st2qp_record.json"
    return json.loads(p.read_text())["phases"] if p.exists() else {}


def saved_hf(c: dict, arm: str) -> np.ndarray | None:
    p = c["pod2"] / "parity" / f"hf_{arm}.npy"
    return np.load(p) if p.exists() else None


def verify(c: dict, arm: str) -> dict:
    """The adapter against the sha256 its training step wrote into its sidecar."""
    d = c["adapters"] / arm
    want = json.loads((d / "st_adapter.json").read_text())["adapter_sha256"]
    got = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in want}
    if got != want:
        raise SystemExit(f"adapter {arm}: sha256 differs from its sidecar")
    return got


def gen(c: dict, a, ckpt: Path | None, arm: str) -> dict:
    """One vLLM engine: the parity rows' first-token letters and the items x rr, seeds shared
    across arms."""
    import screen_vllm as SV
    import t15_driver as T15
    import vllm_phase as V

    sc = T16.vllm_screen(c, a)
    rows = T16.parity_rows(sc, c)
    jobs = L.h1_jobs(sc, c["panel"], c["items"], c["rr"])
    wj = SV.worker_jobs(jobs, STAGE, c["cap"])
    lj = T15.lp_jobs(sc, rows, arm)
    eng = V.engine_json(c["spec"], spec_on=False, label=f"{KEY}:{STAGE}:{arm}")
    if ckpt is not None:  # the merged copy; the registry's tokenizer
        eng.update(model=str(ckpt), revision=None, tokenizer=c["spec"]["repo"],
                   tokenizer_revision=c["spec"]["revision"])
    vrows, summ, rc = V.run_vllm(sc, lj + wj, eng, c["out"] / "vllm" / f"{arm}.jsonl")
    by = {r["job_id"]: r for r in vrows}
    first = [by.get(j["job_id"], {}).get("first_logprobs") or {} for j in lj]
    g = [by[j["job_id"]]["gen_ids"] if by.get(j["job_id"], {}).get("status") == "finished"
         else None for j in wj]
    parsed = L.build_rows(sc, jobs, g, [j["seed"] for j in wj], cap=c["cap"], stage=STAGE,
                          engine="vllm_bf16")
    for r in parsed:
        r["arm"] = arm
    sc.save_jsonl(f"gen/spec_{arm}.jsonl", parsed)
    letters = T15.vllm_letters(rows, first, L.letter_ids(sc))
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / f"vllm_{arm}.npy", letters)
    return {"letters": letters, "rows": rows, "rc": rc, "n": len(parsed),
            "tokens_per_second": summ.get("tokens_per_second")}


def step_gen(c: dict, a) -> dict:
    import t15_driver as T15

    arm = a.arm
    if arm == "none":
        out = gen(c, a, None, "none")
        hf = saved_hf(c, "none")
        if hf is None and not a.dry_run:
            raise SystemExit("pod 2's saved HF base letters are missing (sync_st2qg.txt)")
        tv = None if hf is None else T15.tv(out["letters"], hf, out["rows"])
        ok = a.dry_run or bool(tv <= BASE_BAR)
        return {"status": "ok", "n": out["n"], "rc": out["rc"], "base_tv": tv,
                "base_check_ok": ok, "bar": f"base_tv <= {BASE_BAR}",
                "tokens_per_second": out["tokens_per_second"]}
    base, mg = T16.phase(c, "gen_none"), T16.phase(c, f"merge_{arm}")
    dst = c["out"] / "merged" / arm
    if not base.get("base_check_ok"):
        shutil.rmtree(dst, ignore_errors=True)
        return {"status": "ok", "parity_ok": False, "reason": "base check failed: no arm read"}
    if not mg.get("ok"):
        shutil.rmtree(dst, ignore_errors=True)
        return {"status": "ok", "parity_ok": False, "reason": "merge failed: no read"}
    out = gen(c, a, dst, arm)
    hf = saved_hf(c, arm)
    if hf is None and not a.dry_run:
        raise SystemExit(f"pod 2's saved HF-attached letters for {arm} are missing")
    d = None if hf is None else T15.tv(out["letters"], hf, out["rows"])
    band = (pod2_record(c).get("base_hf") or {}).get("band_tv") or 0.0
    ok = a.dry_run or bool(d <= band + PARITY_MARGIN)
    if not ok:  # the merged engine's rows are not of record: set aside
        f = c["out"] / "gen" / f"spec_{arm}.jsonl"
        if f.exists():
            f.rename(f.with_suffix(".parity_failed.jsonl"))
    if not a.dry_run:
        shutil.rmtree(dst, ignore_errors=True)
    return {"status": "ok", "n": out["n"], "parity_tv": d, "band_tv": band, "parity_ok": ok,
            "tokens_per_second": out["tokens_per_second"]}


def ensure_snapshot(c: dict, a) -> Path:
    """The base snapshot for the merge (``gen none``'s engine downloaded it; a resumed pod that
    skipped it downloads it here)."""
    if a.dry_run:
        return T16.snapshot(c, a)
    try:
        return T16.cached_snapshot(c["spec"]["repo"], c["spec"]["revision"])
    except SystemExit:
        from huggingface_hub import snapshot_download

        snapshot_download(c["spec"]["repo"], revision=c["spec"]["revision"])
        return T16.cached_snapshot(c["spec"]["repo"], c["spec"]["revision"])


def dry_adapter(c: dict, a, arm: str) -> None:
    """A zero-initialized adapter at the tiny width, with its sidecar (dry runs only)."""
    from deepsteer.steering import objective_lora as OL

    d = c["adapters"] / arm
    if (d / "adapter_config.json").exists():
        return
    mw = T16.load_hf(c, a)
    OL.inject_lora(mw.wb, rank=4, dropout=0.0)
    mw.wb.model.save_pretrained(str(d))
    sha = {f: hashlib.sha256((d / f).read_bytes()).hexdigest()
           for f in ("adapter_config.json", "adapter_model.safetensors")}
    (d / "st_adapter.json").write_text(json.dumps({"adapter_sha256": sha}))


def step_merge(c: dict, a) -> dict:
    arm = a.arm
    if not T16.phase(c, "gen_none").get("base_check_ok"):
        return {"status": "ok", "ok": False, "reason": "base check failed: no merge"}
    if a.dry_run:
        dry_adapter(c, a, arm)
    shas = verify(c, arm)
    saved = T16.write_merged(ensure_snapshot(c, a), c["adapters"] / arm,
                             c["out"] / "merged" / arm)
    want = None
    if not a.dry_run:
        want = pod2_record(c)[f"hf_arm_{arm}"]["merge"]["saved"]["weight_kept_share"]
    same = a.dry_run or abs(saved["weight_kept_share"] - want) <= KEPT_TOL
    ok = bool(saved["ok"] and same)
    if not ok:
        shutil.rmtree(c["out"] / "merged" / arm, ignore_errors=True)
    return {"status": "ok", "ok": ok, "saved": saved, "pod2_weight_kept_share": want,
            "kept_share_matches_pod2": bool(same), "adapter_sha256": shas}


STEPS = {"gen": step_gen, "merge": step_merge}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=tuple(STEPS))
    ap.add_argument("--arm", required=True, choices=("none",) + ARMS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pod1", type=Path, required=True, help="pod 1's qwen38_27b dir")
    ap.add_argument("--pod2", type=Path, required=True,
                    help="pod 2's qwen38_27b dir (adapters, parity letters, record)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.deadline = None
    if a.step == "merge" and a.arm == "none":
        ap.error("--step merge needs --arm care|base")
    c = context(a)
    name = f"{a.step}_{a.arm}"
    if T16.phase(c, name).get("status") == "ok":
        print(f">> st2qg {name}: finished earlier", flush=True)
        return 0
    t0 = time.time()
    res = STEPS[a.step](c, a)
    T16.update_record(c, name, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "base_tv", "base_check_ok", "ok", "parity_tv", "parity_ok",
                                 "reason", "status") if k in res}
    print(f">> st2qg {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
