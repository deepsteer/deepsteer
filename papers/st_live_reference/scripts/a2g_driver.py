"""Generation on Muse cell A′ (ST_PLAN Amendment 43 item 2): the instrument rival to ST-A27's dose-0
read, cell A′'s 51 kept items x 8 under ``none``, care-DPO and base-DPO on vLLM bf16 (T1.6's
generation of record: H1's agent frame, effort and cap of record, sampler v2). Each step its own
process (the runner's order):

- ``gen --arm none`` (vLLM, the hub snapshot): T1.6's 64 parity rows' first-token letters and the
  rollouts; the base check, these letters against T1.6's saved HF base letters (at most 0.10).
- ``merge --arm care|base``: the adapter (sha256 checked against ``data/t16_adapters_v1.json``)
  written into a copy of the base snapshot by T1.6's in-place writer; its weight-space kept share
  must equal T1.6's recorded value to 1e-6.
- ``gen --arm care|base`` (vLLM, the merged copy): parity letters against T1.6's saved HF-attached
  letters (at most the base check's TV + 0.05) and the rollouts; rows of an arm that fails parity
  are set aside (no fallback is priced). The merged copy is removed after its arm.

The analysis of record is local (``analyze_a2m.py --a2g``).

    python3 a2g_driver.py --step gen|merge --arm none|care|base --out <dir> --replicate <dir> \
        --t16 <dir> (--selection <cellA2_selection.json> | --pool <forced items>) \
        [--adapters <dir>] [--dry-run]

``--pool`` reads every item of a pool file instead (Amendment 46's forced-choice cell, STEPS=a2f).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402
import t16_driver as T16  # noqa: E402

KEY = T16.KEY
STAGE = "a2g"
DATA = HERE.parent / "data"
POOL = DATA / "st_fxh_cellA2_pool_v1.json"
MANIFEST = DATA / "t16_adapters_v1.json"
BASE_BAR, PARITY_MARGIN, KEPT_TOL = T16.BASE_BAR, T16.PARITY_MARGIN, 1e-6
ARMS = ("care", "base")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(c: dict, name: str, res: dict) -> None:
    p = c["record"]
    rec = json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}
    rec["phases"][name] = {**res, "finished": _now()}
    p.write_text(json.dumps(rec, indent=1, default=str))


def phase(c: dict, name: str) -> dict:
    p = c["record"]
    return (json.loads(p.read_text())["phases"].get(name) or {}) if p.exists() else {}


def verify(adapters: Path, arm: str) -> dict:
    want = json.loads(MANIFEST.read_text())["arms"][arm]
    got = {f: hashlib.sha256((adapters / arm / f).read_bytes()).hexdigest() for f in want}
    if got != want:
        raise SystemExit(f"adapter {arm}: sha256 differs from {MANIFEST.name}")
    return got


def context(a) -> dict:
    from deepsteer.kdg.schema import load_scenarios

    c = T16.context(a)  # T1.6's frame, recipe, cap and parity scenarios (from the replicate)
    pool = {s.id: s for s in load_scenarios(a.pool or POOL)[0]}
    # the forced-choice pool (Amendment 46) holds exactly its items; cell A′'s pool, a selection
    kept = sorted(pool) if a.pool else json.loads(a.selection.read_text())["kept"]
    c["H"].update(pool)
    c["items"] = kept[:2] if a.dry_run else kept
    c["record"] = c["out"] / "a2g_record.json"
    c["t16"] = a.t16
    c["adapters"] = a.adapters
    return c


def gen(c: dict, a, ckpt: Path | None, arm: str) -> dict:
    """One vLLM engine: the parity rows' first-token letters and cell A′'s items x rr (T1.6's
    ``gen_vllm`` without the capability slice; the same seeds for every arm, paired)."""
    import screen_vllm as SV
    import t15_driver as T15
    import vllm_phase as V

    sc = T16.vllm_screen(c, a)
    rows = T16.parity_rows(sc, c)
    jobs = L.h1_jobs(sc, c["H"], c["items"], c["rr"])
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
    sc.save_jsonl(f"gen/cellA2_{arm}.jsonl", parsed)
    letters = T15.vllm_letters(rows, first, L.letter_ids(sc))
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / f"vllm_{arm}.npy", letters)
    return {"letters": letters, "rows": rows, "rc": rc, "n": len(parsed),
            "tokens_per_second": summ.get("tokens_per_second")}


def saved_hf(c: dict, arm: str) -> np.ndarray | None:
    p = c["t16"] / "parity" / f"hf_{arm}.npy"
    return np.load(p) if p.exists() else None


def step_gen(c: dict, a) -> dict:
    import t15_driver as T15

    arm = a.arm
    if arm == "none":
        out = gen(c, a, None, "none")
        hf = saved_hf(c, "none")
        if hf is None and not a.dry_run:
            raise SystemExit("T1.6's saved HF base letters are missing (sync_a2g.txt)")
        tv = None if hf is None else T15.tv(out["letters"], hf, out["rows"])
        ok = a.dry_run or bool(tv <= BASE_BAR)
        return {"status": "ok", "n": out["n"], "rc": out["rc"], "base_tv": tv,
                "base_check_ok": ok, "bar": f"base_tv <= {BASE_BAR}",
                "tokens_per_second": out["tokens_per_second"]}
    base, mg = phase(c, "gen_none"), phase(c, f"merge_{arm}")
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
        raise SystemExit(f"T1.6's saved HF-attached letters for {arm} are missing")
    d = None if hf is None else T15.tv(out["letters"], hf, out["rows"])
    band = base["base_tv"] if base["base_tv"] is not None else 0.0
    ok = a.dry_run or bool(d <= band + PARITY_MARGIN)
    if not ok:  # the merged engine's rows are not of record: set aside
        f = c["out"] / "gen" / f"cellA2_{arm}.jsonl"
        if f.exists():
            f.rename(f.with_suffix(".parity_failed.jsonl"))
    if not a.dry_run:
        shutil.rmtree(dst, ignore_errors=True)
    return {"status": "ok", "n": out["n"], "parity_tv": d, "band_tv": band, "parity_ok": ok,
            "tokens_per_second": out["tokens_per_second"]}


def dry_adapter(c: dict, a, arm: str) -> None:
    """A zero-initialized adapter at the tiny width (dry runs only)."""
    from deepsteer.steering import objective_lora as OL

    d = a.adapters / arm
    if (d / "adapter_config.json").exists():
        return
    mw = T16.load_hf(c, a)
    OL.inject_lora(mw.wb, rank=4, dropout=0.0)
    mw.wb.model.save_pretrained(str(d))


def step_merge(c: dict, a) -> dict:
    arm = a.arm
    if not phase(c, "gen_none").get("base_check_ok"):
        return {"status": "ok", "ok": False, "reason": "base check failed: no merge"}
    if a.dry_run:
        dry_adapter(c, a, arm)
    shas = None if a.dry_run else verify(a.adapters, arm)
    saved = T16.write_merged(T16.snapshot(c, a), a.adapters / arm, c["out"] / "merged" / arm)
    want = None
    if not a.dry_run:
        rec = json.loads((c["t16"] / "t16_record.json").read_text())["phases"]
        want = rec[f"hf_arm_{arm}"]["merge"]["saved"]["weight_kept_share"]
    same = a.dry_run or abs(saved["weight_kept_share"] - want) <= KEPT_TOL
    ok = bool(saved["ok"] and same)
    if not ok:
        shutil.rmtree(c["out"] / "merged" / arm, ignore_errors=True)
    return {"status": "ok", "ok": ok, "saved": saved, "t16_weight_kept_share": want,
            "kept_share_matches_t16": bool(same), "adapter_sha256": shas}


STEPS = {"gen": step_gen, "merge": step_merge}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=tuple(STEPS))
    ap.add_argument("--arm", required=True, choices=("none",) + ARMS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicate", type=Path, required=True,
                    help="the replicate's muse_glimmer_30b dir (gen/arm_none: T1.6's parity rows)")
    ap.add_argument("--t16", type=Path, required=True,
                    help="T1.6's muse_glimmer_30b dir (t16_record.json, parity/hf_*.npy)")
    ap.add_argument("--selection", type=Path, default=None,
                    help="Muse cell A′'s cellA2_selection.json (the kept items)")
    ap.add_argument("--pool", type=Path, default=None,
                    help="a pool read in full instead (Amendment 46's forced-choice items)")
    ap.add_argument("--adapters", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.deadline = None
    if (a.pool is None) == (a.selection is None):
        ap.error("give exactly one of --selection (cell A′) and --pool (forced choice)")
    if a.step == "merge" and (a.arm == "none" or a.adapters is None):
        ap.error("--step merge needs --arm care|base and --adapters")
    c = context(a)
    name = f"{a.step}_{a.arm}"
    if phase(c, name).get("status") == "ok":
        print(f">> a2g {name}: finished earlier", flush=True)
        return 0
    t0 = time.time()
    res = STEPS[a.step](c, a)
    record(c, name, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "base_tv", "base_check_ok", "ok", "parity_tv", "parity_ok",
                                 "reason", "status") if k in res}
    print(f">> a2g {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
