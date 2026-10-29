"""The fresh-seed Qwen replicate (ST_PLAN Amendment 60): care-DPO_Q2 alone, pod 2's recipe with the
teacher at 16 rollouts per arm (fresh seeds), training seed 205, pod 2's gate for one arm plus
Amendment 59's generation read on the panel's 60 rule-right items. ``st2qp_driver`` (and through it
``t16_driver``) is reused with the arms cut to ``care``; ``none``'s rows are pod 2's (gate, dose 0,
specificity, panel, forced, selection base score) and Amendment 59's (generation on the 60 items),
copied into this run's directory at the start (same seeds; not regenerated). Steps, each its own
process, in the runner's order:

- ``teacher`` (HF bf16): `care_harm` at 0.4x x 16 and ``none`` x 16 on pod 2's 35 scenarios,
  stage ``st2qr_teacher``; the pair count reported before training.
- ``train --arm care`` (QLoRA NF4, seed 205), ``select``, ``hf_arm --arm care``,
  ``vllm_arm --arm care`` (the merged engine adds the 60 items x 8 under Amendment 59's seeds),
  ``hfgen --arm care|none`` (the registered fallback, with the 60 items).

A dry run builds ``none``'s rows on a tiny Qwen first (``rr_gen``, ``base_vllm``, ``base_hf``).

    python3 st2qr_driver.py --step <step> [--arm care|none] --out <dir> --pod1 <st2q dir> \
        --pod2 <st2qp dir> --st2qg <st2qg dir> [--dry-run] [--deadline <epoch>]
"""

from __future__ import annotations

import argparse
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
import st2qg_driver as QG  # noqa: E402
import st2qp_driver as QP  # noqa: E402
import t16_driver as T16  # noqa: E402

KEY = QP.KEY
SEED = 205  # Amendment 60 item 4 (Muse's care-DPO2)
TEACHER_STAGE, TEACHER_RR = "st2qr_teacher", 16  # Amendment 60 item 2
ARMS = ("care",)
# none's rows reused from pod 2 (relative to its qwen38_27b dir) and from Amendment 59's pod
POD2_FILES = ("gen/h1_none.jsonl", "gen/forced_none.jsonl", "gen/rr_rollouts.jsonl",
              "cap/none.jsonl", "parity/hf_none.npy", "parity/vllm_none.npy",
              "select/base.jsonl", "dose0/none/dl_chat_neutral.jsonl",
              "dose0/none/dl_chat_neutral.npz", "spec/none/dl_chat_neutral.jsonl",
              "spec/none/dl_chat_neutral.npz") + tuple(
    f"panel/none/{c}.{e}" for c in L.PANEL_CELLS for e in ("jsonl", "npz"))
POD2_PHASES = ("rr_gen", "base_vllm", "base_hf")
ST2QG_FILES = ("gen/spec_none.jsonl",)


def configure() -> None:
    QP.configure()
    QP.ARMS = T16.ARMS = ARMS  # one arm: selection, the fallback's none rule
    T16.gen_vllm = gen_vllm


def inherit(c: dict, pod2: Path, st2qg: Path) -> dict:
    """Copy none's rows and pod 2's base phases into this run (once; never overwriting)."""
    copied = []
    for rel, src_dir in [(f, pod2) for f in POD2_FILES] + [(f, st2qg) for f in ST2QG_FILES]:
        src, dst = src_dir / rel, c["out"] / rel
        if not src.exists():
            raise SystemExit(f"missing inherited input {src} (sync_st2qr.txt)")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(rel)
    rec = T16.load_record(c)
    p2 = json.loads((pod2 / "st2qp_record.json").read_text())["phases"]
    for ph in POD2_PHASES:
        if ph not in rec["phases"]:
            rec["phases"][ph] = {**p2[ph], "inherited": "pod 2 (st2qp), Amendment 60 item 5"}
    c["record"].write_text(json.dumps(rec, indent=1, default=str))
    return {"copied": copied}


def context(a) -> dict:
    configure()
    c = QP.context(a)
    c["record"] = c["out"] / "st2qr_record.json"
    if not a.dry_run:
        inherit(c, a.pod2, a.st2qg)
    return c


def gen_vllm(c, a, sc, ckpt: Path | None, arm: str) -> dict:
    """Pod 2's engine (``QP.gen_vllm``'s jobs) plus the 60 rule-right items x rr under Amendment
    59's seeds, so the generation read pairs with Amendment 59's ``none`` rows."""
    import screen_vllm as SV
    import t15_driver as T15
    import vllm_phase as V

    rows = T16.parity_rows(sc, c)
    jobs = L.h1_jobs(sc, c["H"], c["held"], c["rr"])
    fj = L.h1_jobs(sc, c["H"], c["forced_ids"], c["rr"])
    sj = L.h1_jobs(sc, c["panel"], c["spec_ids"], c["rr"])
    wj, fwj = SV.worker_jobs(jobs, QP.STAGE, c["cap"]), SV.worker_jobs(fj, QP.STAGE, c["cap"])
    swj = SV.worker_jobs(sj, QG.STAGE, c["cap"])
    cj, recs = T15.cap_jobs(sc, c, a)
    lj = T15.lp_jobs(sc, rows, arm)
    eng = V.engine_json(c["spec"], spec_on=False, label=f"{KEY}:st2qr:{arm}")
    if ckpt is not None:  # the merged copy; the registry's tokenizer
        eng.update(model=str(ckpt), revision=None, tokenizer=c["spec"]["repo"],
                   tokenizer_revision=c["spec"]["revision"])
    vrows, summ, rc = V.run_vllm(sc, lj + wj + fwj + swj + cj, eng,
                                 c["out"] / "vllm" / f"{arm}.jsonl")
    by = {r["job_id"]: r for r in vrows}
    first = [by.get(j["job_id"], {}).get("first_logprobs") or {} for j in lj]

    def parse(js, wjs, tag, stage):
        g = [by[j["job_id"]]["gen_ids"] if by.get(j["job_id"], {}).get("status") == "finished"
             else None for j in wjs]
        out = L.build_rows(sc, js, g, [j["seed"] for j in wjs], cap=c["cap"], stage=stage,
                           engine="vllm_bf16")
        for r in out:
            r["arm"] = arm
        sc.save_jsonl(f"gen/{tag}_{arm}.jsonl", out)
        return out

    parsed = parse(jobs, wj, "h1", QP.STAGE)
    forced = parse(fj, fwj, "forced", QP.STAGE)
    spec = parse(sj, swj, "spec", QG.STAGE)
    sc.save_jsonl(f"cap/{arm}.jsonl", T15.score_cap(sc, c["cap_items"], recs, by, arm))
    letters = T15.vllm_letters(rows, first, L.letter_ids(sc))
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / f"vllm_{arm}.npy", letters)
    return {"letters": letters, "rc": rc, "tokens_per_second": summ.get("tokens_per_second"),
            "n_h1": len(parsed), "n_forced": len(forced), "n_spec": len(spec)}


def step_teacher(c, a) -> dict:
    """QP.step_teacher at 16 rollouts with fresh seeds (Amendment 60 item 2)."""
    import st2_driver as S2
    import steering_gate as SG
    import t13_driver as T13

    mw = T16.load_hf(c, a)
    sc = T16.screen(c, mw, False)
    vec, alpha = QP.care_vector(c, a, mw)
    ids = QP.teacher_scenarios(c)
    jobs = L.h1_jobs(sc, c["H"], ids, 2 if a.dry_run else TEACHER_RR)
    t0 = time.time()
    out = {}
    for name, iv in (("none", None), ("care", SG.Add(vec, alpha))):
        f = c["out"] / "gen" / f"teacher_{name}.jsonl"
        if f.exists() and len(T16._jsonl(f)) == len(jobs):  # noqa: SLF001 (a resumed pod)
            out[name] = T16._jsonl(f)  # noqa: SLF001
            continue
        out[name] = S2.run_arm(c, sc, mw, TEACHER_STAGE, f"teacher_{name}", jobs, iv,
                               extra=lambda i, n=name: {"alpha": 0.0 if n == "none" else alpha,
                                                        "axis": n})
    coh = T13._coh(out["care"], out["none"], relative=True)  # noqa: SLF001
    _, info = QP.pair_set(c, a, "care")
    print(f">> st2qr pairs (before training): {info['n_harm']} rule-harm from "
          f"{info['scenarios_with_pair']} scenarios, rule-right {info['rule_right']}, thin "
          f"{info['thin']}", flush=True)
    return {"status": "ok", "n_scenarios": len(ids), "n_per_arm": len(jobs), "alpha": alpha,
            "rollouts": TEACHER_RR, "stage": TEACHER_STAGE, "seconds": time.time() - t0,
            "coherence": coh, "pairs": info}


def step_vllm_arm(c, a) -> dict:
    res = QP.step_vllm_arm(c, a)
    f = c["out"] / "gen" / f"spec_{a.arm}.jsonl"
    if not res.get("parity_ok") and f.exists():  # not of record: set aside, the fallback runs
        f.rename(f.with_suffix(".parity_failed.jsonl"))
    return res


def step_hfgen(c, a) -> dict:
    """QP.step_hfgen, then the 60 items x rr on HF with Amendment 59's seeds (the fallback's
    generation read)."""
    import screen_vllm as SV
    import steering_gate as SG

    res = QP.step_hfgen(c, a)
    if res.get("skipped"):
        return res
    arm = a.arm
    mw = T16.load_hf(c, a)
    sc = T16.screen(c, mw, False)
    if arm != "none":
        T16.attach(mw, c["adapters"] / arm)
    js = L.h1_jobs(sc, c["panel"], c["spec_ids"], c["rr"])
    seeds = [j["seed"] for j in SV.worker_jobs(js, QG.STAGE, c["cap"])]
    g = SG.generate(mw, [j["ids"] for j in js], fmt=c["fmt"], cap=c["cap"], seeds=seeds,
                    batch=c["batch"])
    rows = L.build_rows(sc, js, [r["gen_ids"] for r in g], seeds, cap=c["cap"], stage=QG.STAGE,
                        engine="hf_adapter" if arm != "none" else "hf")
    for r in rows:
        r["arm"] = arm
    sc.save_jsonl(f"gen/spec_{arm}.hf.jsonl", rows)
    return {**res, "n_spec": len(rows)}


STEPS = {"rr_gen": T16.step_rr_gen, "base_vllm": T16.step_base_vllm,
         "base_hf": T16.step_base_hf, "teacher": step_teacher, "train": QP.step_train,
         "select": QP.step_select, "hf_arm": T16.step_hf_arm, "vllm_arm": step_vllm_arm,
         "hfgen": step_hfgen}
DRY_ONLY = ("rr_gen", "base_vllm", "base_hf")  # pod 2's rows stand in on the real run
NEEDS_ARM = {"train": ARMS, "hf_arm": ARMS, "vllm_arm": ARMS, "hfgen": ARMS + ("none",)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=tuple(STEPS))
    ap.add_argument("--arm", choices=("care", "none"), default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pod1", type=Path, required=True, help="pod 1's qwen38_27b dir")
    ap.add_argument("--pod2", type=Path, required=True, help="pod 2's qwen38_27b dir")
    ap.add_argument("--st2qg", type=Path, required=True, help="Amendment 59's qwen38_27b dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--dry-fail-parity", action="store_true",
                    help="dry runs only: the arm's vLLM parity fails, so the fallback path runs")
    a = ap.parse_args()
    if a.step in NEEDS_ARM and a.arm not in NEEDS_ARM[a.step]:
        ap.error(f"--step {a.step} needs --arm {'|'.join(NEEDS_ARM[a.step])}")
    if a.step in DRY_ONLY and not a.dry_run:
        ap.error(f"--step {a.step} runs in dry runs only (pod 2's rows stand in)")
    if a.dry_fail_parity and not a.dry_run:
        ap.error("--dry-fail-parity is for dry runs")
    a.seed, a.fresh_from_t16 = SEED, False
    c = context(a)
    name = a.step if a.arm is None else f"{a.step}_{a.arm}"
    if T16.phase(c, name).get("status") == "ok":
        print(f">> st2qr {name}: finished earlier", flush=True)
        return 0
    if a.deadline and time.time() > a.deadline:
        print(f">> st2qr {name}: not started (past the deadline)", flush=True)
        return 3
    t0 = time.time()
    res = STEPS[a.step](c, a)
    T16.update_record(c, name, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "n_h1", "n_forced", "n_spec", "base_tv", "base_check_ok",
                                 "max_steps_run", "parity_ok", "chosen", "skipped", "status")
             if k in res}
    if "merge" in res:
        brief["merge_gate_ok"] = res["merge"].get("gate_ok")
    print(f">> st2qr {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
