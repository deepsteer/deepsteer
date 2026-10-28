"""Pod 2 on Qwen3.8-27B (ST_PLAN Amendment 57 and its additions): the steer-then-DPO recipe with
T1.6's full gate and the forced-choice twins. ``t16_driver`` is configured for Qwen (model key,
date pin, site, stage) and four of its functions are replaced: the parity scenarios, the pairs, the
vLLM engine's jobs (the forced cells ride in the gate's engines) and training (the step rule and
checkpoints at thirds). Steps, each its own process, in the runner's order:

- ``rr_gen`` (vLLM): the rule-right pool's 130 items x 8 (T1.6's).
- ``base_vllm`` (vLLM, the hub snapshot): parity letters, H1_held 132 x 8, the gate's 200
  capability items and the forced cells (51 originals and 51 twins x 8) under ``none``.
- ``base_hf`` (HF bf16): parity letters, dose 0 on H1_held, specificity, the panel half; the base
  check; the engaged held set.
- ``teacher`` (HF bf16): `care_harm` at 0.4x (alpha from pod 1's record, block 32 ``clause_end``,
  pod 1's unit vector) x 8 and ``none`` x 8 on the 35 H1_train scenarios whose v1 rows hold a rule
  and a protect rollout, shared seeds; the pair count reported before training.
- ``train --arm care|base`` (QLoRA NF4): pairs by T1.6's rule plus the pool's rule-right pairs cut
  to the rule-harm count; ``STEPS_PER_PAIR`` x pairs steps; adapters saved at a third, two thirds
  and the end.
- ``select`` (HF bf16): the held-out selection set (``data/st2qp_selection_v1.json``) under base and
  the checkpoints, latest first; the checkpoint of record is copied to ``adapters/<arm>``.
- ``hf_arm --arm``, ``vllm_arm --arm`` (T1.6's: attached reads, the merge and its gate, the merged
  engine with parity), ``hfgen --arm`` (the registered HF fallback, with the forced cells).

The analysis of record is local (``analyze_st2qp.py``).

    python3 st2qp_driver.py --step <step> [--arm care|base|none] --out <dir> \
        --pod1 <st2q/qwen38_27b dir> [--dry-run] [--deadline <epoch>]
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
import t16_driver as T16  # noqa: E402

KEY = "qwen38_27b"
DATA = HERE.parent / "data"
PIN = "2026-10-27"  # pod 1's rows; every prompt of pod 2 is rendered at this pin
BLOCK = 32
CLS = "clause_end"
STAGE = "st2qp"
SEED = 105
N_TEACHER = 35  # Amendment 57 item 4: H1_train scenarios with a rule and a protect v1 rollout
THIN = 38  # item 4: fewer rule-harm pairs than care-DPO2's 38 is thin
STEPS_PER_PAIR = 243 / 40  # third addition, item 1: T1.6's 243 steps were ~24 epochs over 40 pairs
SEL_BAR = 0.02  # item 5: within 0.02 of base's selection accuracy
ENVELOPE, SPENT_BEFORE = 125.0, 96.21  # third addition, item 4 (A100-h)
NON_TRAIN_H = 12.1  # every priced item but training, with the fallback's 3.0 reserve (item 4)
POD1_BATCH = 64  # pod 1's pilot batch on HF
FORCED = (DATA / "st_fxh_cellA2_forced_v1.json", DATA / "st_fxh_cellA2_twins_forced_v1.json")
SELECTION = DATA / "st2qp_selection_v1.json"
ARMS = T16.ARMS


def configure() -> None:
    """``t16_driver`` reads these module globals at call time."""
    T16.KEY, T16.PIN, T16.BLOCK, T16.STAGE, T16.SEED = KEY, PIN, BLOCK, STAGE, SEED
    T16.parity_scenarios = parity_scenarios
    T16.pairs_for = pairs_for
    T16.gen_vllm = gen_vllm


def context(a) -> dict:
    from deepsteer.kdg.schema import load_scenarios

    a.replicate = a.pod1  # T16.context records it; parity_scenarios reads pod 1's rows
    c = T16.context(a)
    c["pod1"], c["dry"] = a.pod1, a.dry_run
    c["record"] = c["out"] / "st2qp_record.json"
    c["cls"] = "onset" if a.dry_run else CLS  # a tiny model's trace has no clause ends
    c["batch"] = 8 if a.dry_run else POD1_BATCH
    sel = json.loads(SELECTION.read_text())["items"]
    c["sel_items"] = sel[:1] + sel[-1:] if a.dry_run else sel
    forced = [s for f in FORCED for s in load_scenarios(f)[0]]
    c["H"].update({s.id: s for s in forced})
    ids = sorted(s.id for s in forced)
    if a.dry_run:  # two originals and their twins
        orig = [i for i in ids if not i.endswith("-T")][:2]
        ids = sorted(orig + [i + "-T" for i in orig])
    c["forced_ids"] = ids
    return c


# ---- parity, teacher scenarios and pairs -------------------------------------------------------
def parity_scenarios(c: dict) -> list[str]:
    """16 seeded H1_train scenarios from pod 1's v1 rows (2 in a dry run)."""
    import random

    rows = T16._jsonl(c["pod1"] / "gen" / "h1_rollouts.jsonl")  # noqa: SLF001
    ids = sorted({r["scenario_id"] for r in rows if r.get("split") == "train"})
    n = 2 if c.get("dry") else T16.N_PARITY_SCEN
    return sorted(random.Random(f"{STAGE}|parity").sample(ids, min(n, len(ids))))


def teacher_scenarios(c: dict) -> list[str]:
    """Item 4: H1_train scenarios whose pod-1 v1 rows hold at least one rule (``consistent``) and
    one protect (``violating``) rollout."""
    st: dict[str, set] = {}
    for r in T16._jsonl(c["pod1"] / "gen" / "h1_rollouts.jsonl"):  # noqa: SLF001
        if r.get("split") == "train":
            st.setdefault(r["scenario_id"], set()).add(r.get("norm_status"))
    ids = sorted(s for s, v in st.items() if {"consistent", "violating"} <= v)
    if c.get("dry"):
        return ids[:2]
    if len(ids) != N_TEACHER:
        raise SystemExit(f"{len(ids)} teacher scenarios, registered {N_TEACHER}")
    return ids


def pair_set(c: dict, a, arm: str) -> tuple[list[dict], dict]:
    """T1.6's per-scenario rule on the teacher's rows (n = min(steered protect, unsteered protect,
    unsteered rule, 2)) plus the pool's rule-right pairs cut to the rule-harm count."""
    import st2lb_driver as LB

    care = T16._jsonl(c["out"] / "gen" / "teacher_care.jsonl")  # noqa: SLF001
    none = T16._jsonl(c["out"] / "gen" / "teacher_none.jsonl")  # noqa: SLF001
    hp, info = T16.harm_pairs(care, none)
    if a.dry_run and not hp[arm]:  # a tiny model's traces never parse: pod 1's arm rows stand in
        care = T16._jsonl(c["pod1"] / "gen" / "arm_care.jsonl")  # noqa: SLF001
        none = T16._jsonl(c["pod1"] / "gen" / "arm_none.jsonl")  # noqa: SLF001
        hp, info = T16.harm_pairs(care, none)
        info["dry_source"] = "pod 1's arm rows"
    prompts = T16.render_prompts(c, care + none)
    harm = [{"scenario_id": p["scenario_id"], "prompt": prompts[p["chosen_key"]],
             "chosen": p["chosen"], "rejected": p["rejected"], "kind": p["kind"],
             "chosen_key": p["chosen_key"], "rejected_key": p["rejected_key"]} for p in hp[arm]]
    if a.dry_run:
        harm = [{**p, "chosen": p["chosen"][:24], "rejected": p["rejected"][:24]} for p in harm]
    rr = LB.rule_right_pairs(T16._jsonl(c["out"] / "gen" / "rr_rollouts.jsonl"),  # noqa: SLF001
                             dry=a.dry_run)
    rr, cut = LB.cut_to(rr, len(harm))
    return harm + rr, {**info, "n_harm": len(harm), "rule_right": cut,
                       "n_total": len(harm) + len(rr), "thin": len(harm) < THIN,
                       "thin_bar": THIN}


def pairs_for(c: dict, a, arm: str) -> tuple[list[dict], dict]:
    return pair_set(c, a, arm)


def steps_for(n_pairs: int) -> int:
    return max(1, round(STEPS_PER_PAIR * n_pairs))


def checkpoints(steps: int) -> list[int]:
    """A third, two thirds and the end (distinct, ascending)."""
    return sorted({max(1, round(steps / 3)), max(1, round(2 * steps / 3)), steps})


# ---- the vLLM engine: T1.6's jobs plus the forced cells ----------------------------------------
def gen_vllm(c, a, sc, ckpt: Path | None, arm: str) -> dict:
    """T1.6's engine (parity letters, H1_held x rr, the capability slice) with the forced cells'
    originals and twins x rr in the same engine, the same seeds for every arm."""
    import screen_vllm as SV
    import t15_driver as T15
    import vllm_phase as V

    rows = T16.parity_rows(sc, c)
    jobs = L.h1_jobs(sc, c["H"], c["held"], c["rr"])
    fj = L.h1_jobs(sc, c["H"], c["forced_ids"], c["rr"])
    wj, fwj = SV.worker_jobs(jobs, STAGE, c["cap"]), SV.worker_jobs(fj, STAGE, c["cap"])
    cj, recs = T15.cap_jobs(sc, c, a)
    lj = T15.lp_jobs(sc, rows, arm)
    eng = V.engine_json(c["spec"], spec_on=False, label=f"{KEY}:{STAGE}:{arm}")
    if ckpt is not None:  # the merged copy; the registry's tokenizer
        eng.update(model=str(ckpt), revision=None, tokenizer=c["spec"]["repo"],
                   tokenizer_revision=c["spec"]["revision"])
    vrows, summ, rc = V.run_vllm(sc, lj + wj + fwj + cj, eng, c["out"] / "vllm" / f"{arm}.jsonl")
    by = {r["job_id"]: r for r in vrows}
    first = [by.get(j["job_id"], {}).get("first_logprobs") or {} for j in lj]

    def parse(js, wjs, tag):
        g = [by[j["job_id"]]["gen_ids"] if by.get(j["job_id"], {}).get("status") == "finished"
             else None for j in wjs]
        out = L.build_rows(sc, js, g, [j["seed"] for j in wjs], cap=c["cap"], stage=STAGE,
                           engine="vllm_bf16")
        for r in out:
            r["arm"] = arm
        sc.save_jsonl(f"gen/{tag}_{arm}.jsonl", out)
        return out

    parsed = parse(jobs, wj, "h1")
    forced = parse(fj, fwj, "forced")
    sc.save_jsonl(f"cap/{arm}.jsonl", T15.score_cap(sc, c["cap_items"], recs, by, arm))
    letters = T15.vllm_letters(rows, first, L.letter_ids(sc))
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / f"vllm_{arm}.npy", letters)
    return {"letters": letters, "rc": rc, "tokens_per_second": summ.get("tokens_per_second"),
            "n_h1": len(parsed), "n_forced": len(forced)}


# ---- steps -------------------------------------------------------------------------------------
def care_vector(c: dict, a, mw) -> tuple[np.ndarray, float]:
    """Pod 1's unit `care_harm` at block 32 and its dose-of-record alpha (0.4x); a dry run's is
    a seeded unit vector at the tiny width."""
    if a.dry_run:
        cfg = mw.model.config
        h = getattr(cfg, "text_config", cfg).hidden_size
        v = np.random.default_rng(0).standard_normal(h)
        return (v / np.linalg.norm(v)).astype(np.float32), 1.0
    rec = json.loads((c["pod1"] / "st2q_record.json").read_text())["phases"]
    if rec["directions"]["block"] != BLOCK or rec["sweep"]["dose"] != 0.4:
        raise SystemExit("pod 1's direction or dose is not Amendment 57's (block 32, 0.4x)")
    v = np.load(c["pod1"] / "directions.npz")["care_harm"].astype(np.float32)
    return v, float(rec["sweep"]["alpha"])


def step_teacher(c, a) -> dict:
    import st2_driver as S2
    import steering_gate as SG
    import t13_driver as T13

    mw = T16.load_hf(c, a)
    sc = T16.screen(c, mw, False)
    vec, alpha = care_vector(c, a, mw)
    ids = teacher_scenarios(c)
    jobs = L.h1_jobs(sc, c["H"], ids, c["rr"])
    t0 = time.time()
    out = {}
    for name, iv in (("none", None), ("care", SG.Add(vec, alpha))):
        f = c["out"] / "gen" / f"teacher_{name}.jsonl"
        if f.exists() and len(T16._jsonl(f)) == len(jobs):  # noqa: SLF001 (a resumed pod)
            out[name] = T16._jsonl(f)  # noqa: SLF001
            continue
        out[name] = S2.run_arm(c, sc, mw, "st2qp_teacher", f"teacher_{name}", jobs, iv,
                               extra=lambda i, n=name: {"alpha": 0.0 if n == "none" else alpha,
                                                        "axis": n})
    coh = T13._coh(out["care"], out["none"], relative=True)  # noqa: SLF001
    res = {"status": "ok", "n_scenarios": len(ids), "n_per_arm": len(jobs), "alpha": alpha,
           "block": BLOCK, "cls": c["cls"], "seconds": time.time() - t0, "coherence": coh}
    if (c["out"] / "gen" / "rr_rollouts.jsonl").exists():
        _, info = pair_set(c, a, "care")
        res["pairs"] = info
        print(f">> st2qp pairs (before training): {info['n_harm']} rule-harm from "
              f"{info['scenarios_with_pair']} scenarios, rule-right {info['rule_right']}, thin "
              f"{info['thin']}", flush=True)
    return res


def step_train(c, a) -> dict:
    import st_train as ST
    import t14_driver as T14

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.steering import objective_lora as OL

    arm = a.arm
    pr, info = pair_set(c, a, arm)
    rc = c["recipe"]
    steps = 3 if a.dry_run else steps_for(len(pr))
    if arm == "base":  # the teacher is the only difference: base runs care's steps
        want = T16.phase(c, "train_care").get("max_steps_run")
        if want != steps:
            raise SystemExit(f"base's steps {steps} differ from care's {want}")
    if a.dry_run:
        wb = WhiteBoxModel(str(T16.tiny_path(c)), access_tier=AccessTier.WEIGHTS)
    else:
        wb = OL.qlora_model(c["spec"]["repo"], c["spec"]["revision"],
                            access_tier=AccessTier.WEIGHTS)
    gen_tr, _ = T14.general(c, a)

    def gc_on():
        if rc["gradient_checkpointing"] and not a.dry_run:
            wb.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    gc_on()
    kw = dict(layer=c["layer"], pairs=pr, general=gen_tr, kl_weight=rc["kl_weight"], lr=rc["lr"],
              grad_accum=rc["grad_accum"], seed=a.seed, lora_rank=rc["rank"],
              lora_alpha=rc["alpha"], lora_dropout=rc["dropout"])
    pilot = None
    if arm == "care":  # the pilot checks the bail; it never sets the steps
        tp = time.time()
        ST.run(wb, "beh", max_steps=1, **{**kw, "grad_accum": T14.PILOT_MICRO})
        pilot = {"micro": T14.PILOT_MICRO, "seconds": time.time() - tp}
        pilot["seconds_per_micro"] = pilot["seconds"] / T14.PILOT_MICRO
        wb._model = wb.model.unload()  # noqa: SLF001
        gc_on()
        pilot["projected_hours"] = pilot["seconds_per_micro"] * rc["grad_accum"] * steps / 3600
        pilot["bail_hours"] = bail_hours()
        if pilot["projected_hours"] > pilot["bail_hours"] and not a.dry_run:
            return {"status": "stop: the pilot projects the pod past the envelope", "pilot": pilot,
                    "pairs": info, "steps": steps}
    marks = checkpoints(steps)

    def save(s: int) -> dict:
        d = c["adapters"] / f"{arm}@{s}"
        d.mkdir(parents=True, exist_ok=True)
        wb.model.save_pretrained(str(d))
        sha = {f: hashlib.sha256((d / f).read_bytes()).hexdigest()
               for f in ("adapter_config.json", "adapter_model.safetensors")}
        (d / "st_adapter.json").write_text(json.dumps(
            {"arm": arm, "step": s, "of_steps": steps, "pairs": info, "recipe": rc,
             "seed": a.seed, "adapter_sha256": sha}, indent=1, default=str))
        return {"step": s, "dir": d.name}

    def at_mark(s: int) -> dict:
        return save(s) if s in marks[:-1] else {}

    t0 = time.time()
    res = ST.run(wb, "beh", max_steps=steps, eval_fn=at_mark, eval_at=tuple(marks[:-1]), **kw)
    save(steps)
    return {"status": "ok", "max_steps_run": steps, "checkpoints": marks, "pilot": pilot,
            "train_seconds": time.time() - t0, "pairs": info,
            "epochs": steps * rc["grad_accum"] / max(len(pr), 1),
            "last_steps": res.get("steps", [])[-3:]}


def bail_hours() -> float:
    """Third addition, item 4: per adapter, the training hours that keep the pod inside the
    envelope with every other priced item and the fallback's reserve."""
    return (ENVELOPE - SPENT_BEFORE - NON_TRAIN_H) / 2


def choose(n_items: int, base_correct: int, correct, marks: list[int], arms) -> dict:
    """Item 5 and the third addition, item 2: the checkpoint of record is the latest step at which
    every arm is within ``SEL_BAR`` of base's selection accuracy, one step for both arms (the
    teacher stays their only difference), else the earliest (never scored: the rule picks it
    whatever its score). Steps are visited latest first and every arm is scored at each visited
    step, so per-arm selection (beside, not of record) needs no further scores."""
    need = base_correct - round(SEL_BAR * n_items)  # integer items: no float edge at the bar
    later = sorted(marks, reverse=True)[:-1]
    seen: dict[tuple[str, int], bool] = {}
    joint = None
    for s in later:
        for x in arms:  # no short-circuit: both arms scored at every visited step
            seen[(x, s)] = correct(x, s) >= need
        if all(seen[(x, s)] for x in arms):
            joint = s
            break
    joint = min(marks) if joint is None else joint
    per_arm = {x: next((s for s in later if seen.get((x, s))), min(marks)) for x in arms}
    return {"of_record": {x: joint for x in arms}, "per_arm_beside": per_arm}


def step_select(c, a) -> dict:
    import steering_gate as SG
    import t15_driver as T15

    mw = T16.load_hf(c, a)
    sc = T16.screen(c, mw, False)
    cs = {**c, "cap_items": c["sel_items"]}
    cj, recs = T15.cap_jobs(sc, cs, a)
    marks = T16.phase(c, "train_care")["checkpoints"]
    scores: dict[str, dict] = {}

    def score(tag: str, adir: Path | None) -> int:
        f = c["out"] / "select" / f"{tag}.jsonl"
        if f.exists():  # a resumed pod: the rows are deterministic given the seeds
            rows = T16._jsonl(f)  # noqa: SLF001
        else:
            peft = T16.attach(mw, adir) if adir is not None else None
            g = SG.generate(mw, [j["prompt_ids"] for j in cj], fmt=c["fmt"],
                            cap=c["cap"] + T15.CAP_EXTRA, seeds=[j["seed"] for j in cj],
                            batch=c["batch"])
            if peft is not None:
                peft.unload()
            by = {f"cap|{n}": {"gen_ids": r["gen_ids"], "status": "finished",
                               "n_tokens": r["n_tokens"]} for n, r in enumerate(g)}
            rows = T15.score_cap(sc, c["sel_items"], recs, by, tag)
            sc.save_jsonl(f"select/{tag}.jsonl", rows)
        n = int(sum(r["correct"] for r in rows))
        scores[tag] = {"correct": n, "n": len(rows), "accuracy": n / max(len(rows), 1),
                       "answered": float(np.mean([r["answered"] for r in rows]))}
        return n

    base = score("base", None)
    pick = choose(len(cj), base, lambda x, s: score(f"{x}@{s}", c["adapters"] / f"{x}@{s}"),
                  marks, ARMS)
    for x, s in pick["of_record"].items():
        dst = c["adapters"] / x
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(c["adapters"] / f"{x}@{s}", dst)
    return {"status": "ok", "checkpoints": marks, "scores": scores,
            "chosen": pick["of_record"], "per_arm_beside": pick["per_arm_beside"],
            "bar": SEL_BAR,
            "rule": "the latest step at which both arms are within the bar of base's selection "
                    "accuracy, one step for both, else the earliest (Amendment 57 item 5, third "
                    "addition item 2); per-arm selection beside; the gate's capability read never "
                    "selects"}


def step_vllm_arm(c, a) -> dict:
    res = T16.step_vllm_arm(c, a)
    f = c["out"] / "gen" / f"forced_{a.arm}.jsonl"
    if not res.get("parity_ok") and f.exists():  # not of record: set aside, the fallback runs
        f.rename(f.with_suffix(".parity_failed.jsonl"))
    return res


def step_hfgen(c, a) -> dict:
    """T1.6's fallback (engaged H1_held x rr and the capability slice's T1.5 half on HF) with the
    forced cells x rr, the vLLM jobs' seeds."""
    import screen_vllm as SV
    import steering_gate as SG
    import t15_driver as T15

    arm = a.arm
    if not T16.needs_hf(c, arm):
        return {"status": "ok", "skipped": "not needed (vLLM rows of record)"}
    mw = T16.load_hf(c, a)
    sc = T16.screen(c, mw, False)
    if arm != "none":
        T16.attach(mw, c["adapters"] / arm)
    t0 = time.time()

    def run(js, tag, engine):
        seeds = [j["seed"] for j in SV.worker_jobs(js, STAGE, c["cap"])]
        g = SG.generate(mw, [j["ids"] for j in js], fmt=c["fmt"], cap=c["cap"], seeds=seeds,
                        batch=c["batch"], on_finish=SG.jsonl_saver(
                            c["out"] / "gen" / f"{tag}_{arm}.hf.raw.jsonl"))
        rows = L.build_rows(sc, js, [r["gen_ids"] for r in g], seeds, cap=c["cap"], stage=STAGE,
                            engine=engine)
        for r in rows:
            r["arm"] = arm
        sc.save_jsonl(f"gen/{tag}_{arm}.hf.jsonl", rows)
        return rows

    engine = "hf_adapter" if arm != "none" else "hf"
    eng = T16.phase(c, "base_hf")["engaged_held"] or (c["held"][:1] if a.dry_run else [])
    h1 = run(L.h1_jobs(sc, c["H"], eng, c["rr"]), "h1", engine) if eng else []
    forced = run(L.h1_jobs(sc, c["H"], c["forced_ids"], c["rr"]), "forced", engine)
    half = T15.cap_fallback(c["cap_items"])
    cj, recs = T15.cap_jobs(sc, c, a)
    cj, recs = [cj[i] for i in half], [recs[i] for i in half]
    gc_ = SG.generate(mw, [j["prompt_ids"] for j in cj], fmt=c["fmt"],
                      cap=c["cap"] + T15.CAP_EXTRA, seeds=[j["seed"] for j in cj],
                      batch=c["batch"])
    by = {f"cap|{n}": {"gen_ids": r["gen_ids"], "status": "finished", "n_tokens": r["n_tokens"]}
          for n, r in enumerate(gc_)}
    sc.save_jsonl(f"cap/{arm}.hf.jsonl",
                  T15.score_cap(sc, [c["cap_items"][i] for i in half], recs, by, arm))
    return {"status": "ok", "engine": "hf", "n_h1": len(h1), "n_engaged": len(eng),
            "n_forced": len(forced), "n_capability": len(cj), "seconds": time.time() - t0}


STEPS = {"rr_gen": T16.step_rr_gen, "base_vllm": T16.step_base_vllm,
         "base_hf": T16.step_base_hf, "teacher": step_teacher, "train": step_train,
         "select": step_select, "hf_arm": T16.step_hf_arm, "vllm_arm": step_vllm_arm,
         "hfgen": step_hfgen}
NEEDS_ARM = {"train": ARMS, "hf_arm": ARMS, "vllm_arm": ARMS, "hfgen": ARMS + ("none",)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=tuple(STEPS))
    ap.add_argument("--arm", choices=("care", "base", "none"), default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pod1", type=Path, required=True, help="pod 1's qwen38_27b dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--dry-fail-parity", action="store_true",
                    help="dry runs only: the arm's vLLM parity fails, so the fallback path runs")
    a = ap.parse_args()
    if a.step in NEEDS_ARM and a.arm not in NEEDS_ARM[a.step]:
        ap.error(f"--step {a.step} needs --arm {'|'.join(NEEDS_ARM[a.step])}")
    if a.dry_fail_parity and not a.dry_run:
        ap.error("--dry-fail-parity is for dry runs")
    a.seed, a.fresh_from_t16 = SEED, False
    configure()
    c = context(a)
    name = a.step if a.arm is None else f"{a.step}_{a.arm}"
    if T16.phase(c, name).get("status") == "ok":
        print(f">> st2qp {name}: finished earlier", flush=True)
        return 0
    if a.deadline and time.time() > a.deadline:
        print(f">> st2qp {name}: not started (past the deadline)", flush=True)
        return 3
    t0 = time.time()
    res = STEPS[a.step](c, a)
    T16.update_record(c, name, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "n_h1", "n_forced", "base_tv", "base_check_ok",
                                 "max_steps_run", "parity_ok", "chosen", "skipped", "status")
             if k in res}
    if "merge" in res:
        brief["merge_gate_ok"] = res["merge"].get("gate_ok")
    print(f">> st2qp {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
