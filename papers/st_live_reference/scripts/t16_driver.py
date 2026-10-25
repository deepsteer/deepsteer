"""T1.6 on Muse Glimmer 30B (ST_PLAN Amendments 37-39): two DPO adapters on L-beh-2's recipe, the
teacher their only difference. ``care`` takes `care_harm`-steered protect traces (the replicate's
``arm_care_03``) as chosen; ``base`` takes unsteered protect traces (the replicate's ``arm_none``);
both reject the same unsteered rule traces and share the rule-right pairs from Muse's base rollouts
on the 130-item pool. Steps, each its own process (the runner's order):

- ``rr_gen`` (vLLM): the pool's 130 items x 8 in H1's agent frame (L-beh-2's ``rr_gen`` on Muse).
- ``base_vllm`` (vLLM, the hub snapshot): the parity rows' first-token letters, all 132 H1_held x 8
  and the 100-item capability slice under ``none``.
- ``base_hf`` (HF bf16, single-row): the parity rows' letters, dose 0 on all 132 H1_held, the 60
  specificity items and the panel half under ``none``; the base check (vLLM against HF letters).
- ``train --arm care|base`` (HF, QLoRA NF4): the pairs (prompts re-rendered and checked against the
  replicate's hashes), care's pilot fitting the step count to the 2 A100-h budget, base reusing it.
- ``hf_arm --arm`` (HF bf16, adapter attached): parity letters, dose 0 on H1_held, specificity, the
  panel half; then the merged checkpoint (a copy of the base snapshot with the adapter's text-tower
  deltas written into their tensors in place), reloaded on HF, its parity letters: the merge gate.
- ``vllm_arm --arm`` (vLLM, the merged copy): parity letters against the attached adapter's (the
  base check's band + 0.05), H1_held x 8 and capability.
- ``hfgen --arm none|care|base`` (the registered fallback, HF): runs only for an arm whose merge
  gate or vLLM parity failed (``none`` when any arm does or the base check fails): the engaged
  H1_held x 8 and the capability slice's T1.5 half.

The analysis of record is local (``analyze_t16.py``).

    python3 t16_driver.py --step <step> [--arm care|base|none] --out <dir> --replicate <dir> \
        [--dry-run] [--deadline <epoch>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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

KEY = "muse_glimmer_30b"
DATA = HERE.parent / "data"
PIN = "2026-10-23"  # the replicate's rows (Cell 4's pin)
BLOCK = 26  # Muse's site (hidden 27); ST.run's layer argument, unused by DPO
SEED = 105  # Amendment 37 item 1, both arms
FRESH_STEPS, FRESH_BAIL_H = 243, 2.5  # Amendment 49: care-DPO2 keeps T1.6's steps; the pilot bails
STAGE = "t16"
RR = 8
PAIRS_PER_SCENARIO = 2
N_PARITY_SCEN = 16
BASE_BAR = 0.10  # Amendment 39: vLLM against HF letters on Muse's bf16 base
PARITY_MARGIN = 0.05
BUDGET_H = 2.0
ARMS = ("care", "base")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _sha(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


# ---- context and record ------------------------------------------------------------------------
def context(a) -> dict:
    import glob

    import pod_st_screen as P
    import t14_driver as T14
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    from deepsteer.kdg.schema import load_scenario_dir

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    split = json.loads(P.SPLIT.read_text())
    Hs, _ = load_scenario_dir([P.H1_SET])
    kept = set(json.loads(P.H1_KEPT.read_text())["ids"])
    allp, _ = load_scenario_dir(sorted(glob.glob(str(P.KDG / "data" / "*_scenarios_*.json"))))
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    rc = dict(T14.RECIPE, budget_h=BUDGET_H, eval_every=0)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept},
         "panel": {s.id: s for s in L.panel_586(allp)},
         "held": sorted(split["H1_held"]),
         "spec_ids": json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"],
         "half_ids": json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"],
         "cap_items": json.loads((DATA / "t15_capability_v1.json").read_text())["items"],
         "cap": L.CAP_OF_RECORD, "batch": 160, "rr": RR, "recipe": rc, "layer": BLOCK + 1,
         "replicate": a.replicate, "adapters": out / "adapters", "record": out / "t16_record.json"}
    if a.dry_run:  # a tiny Muse with the real tokenizer; a few items through every path
        rc.update(max_steps=4, grad_accum=1, general_train=4, general_monitor=2)
        c.update(held=c["held"][:2], spec_ids=c["spec_ids"][:2], half_ids=c["half_ids"][:2],
                 cap_items=c["cap_items"][:1] + c["cap_items"][-1:], cap=24, batch=8, rr=2,
                 layer=2)
    return c


def load_record(c: dict) -> dict:
    return json.loads(c["record"].read_text()) if c["record"].exists() else {"model": KEY,
                                                                            "phases": {}}


def update_record(c: dict, name: str, res: dict) -> None:
    rec = load_record(c)
    rec["phases"][name] = {**res, "finished": _now()}
    c["record"].write_text(json.dumps(rec, indent=1, default=L._json_default))  # noqa: SLF001


def phase(c: dict, name: str) -> dict:
    return load_record(c)["phases"].get(name) or {}


# ---- screens and models ------------------------------------------------------------------------
def screen(c: dict, model, dry: bool) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], PIN, dry=dry, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or "medium"
    sc.unbatched = True  # single-row readouts of record
    sc.layers_override = [c["layer"]]
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def vllm_screen(c: dict, a) -> L.Screen:
    """The tokenizer only (no GPU); a dry run's stand-in generates short traces."""
    if a.dry_run:
        from screen_stub import ScreenStub

        stub = ScreenStub(KEY, c["tok"], c["fmt"], c["spec"]["n_layers"],
                          length_by_effort={e: 4 for e in c["fmt"].effort_ladder})
        return screen(c, stub, True)
    return screen(c, L.tok_only(c["tok"]), False)


def load_hf(c: dict, a, path: str | None = None):
    """Muse in bf16 on HF (the hub snapshot, or a merged copy at ``path``); a tiny Muse in dry
    runs (its trace opens at once: a random model never writes Muse's channel header)."""
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:
        tiny = tiny_path(c)
        mw = ModelWrapper(str(path or tiny), None, torch_dtype=torch.float32)
        object.__setattr__(c["fmt"], "trace_open", lambda tok, g: 0)
        object.__setattr__(c["fmt"], "trace_closes", lambda tok, g: False)
        return mw
    if path is not None:
        return ModelWrapper(path, None, torch_dtype=torch.bfloat16,
                            **c["P"].hf_load_kwargs(c["spec"]))
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        **c["P"].hf_load_kwargs(c["spec"]))


def tiny_path(c: dict) -> Path:
    sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
    import st_tiny

    p = c["out"] / "_tiny"
    if not (p / "config.json").exists():
        st_tiny.build(KEY, p)
    return p


def snapshot(c: dict, a) -> Path:
    """The base checkpoint as files: the hub snapshot (already in the cache) or the dry run's
    tiny Muse."""
    if a.dry_run:
        return tiny_path(c)
    return cached_snapshot(c["spec"]["repo"], c["spec"]["revision"])


def cached_snapshot(repo: str, revision: str) -> Path:
    """The cache's snapshot folder for ``revision``, read directly: ``snapshot_download`` with
    ``local_files_only`` refuses a cache missing non-weight files (LICENSE, README) the runner never
    downloads (the first T1.6 pod's failure). The copy needs only what is cached: the config, the
    index and the weight shards that vLLM and HF load the base from."""
    from huggingface_hub.constants import HF_HUB_CACHE

    d = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not (d / "config.json").exists() or not list(d.glob("*.safetensors")):
        raise SystemExit(f"no cached snapshot with config and shards at {d}")
    return d


def attach(mw, adir: Path):
    """The adapter injected into ``mw.model`` in place (as T1.5), then checked: the attached
    ``lora_B`` weights must equal the file's (a loader that names the text tower differently
    would attach nothing and read as the base model)."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file

    peft = PeftModel.from_pretrained(mw.model, str(adir))
    want = {k.split(".lora_B.")[0].removeprefix("base_model.model."): v.float()
            for k, v in load_file(str(adir / "adapter_model.safetensors")).items()
            if ".lora_B." in k}
    got = {n.split(".lora_B.")[0].removeprefix("base_model.model."): p.detach().float().cpu()
           for n, p in peft.named_parameters() if ".lora_B." in n}
    miss = [k for k in want if k not in got]
    bad = [k for k in want if k in got and not torch.allclose(got[k], want[k], atol=1e-6)]
    if miss or bad or not want:
        raise SystemExit(f"adapter {adir.name}: {len(miss)} modules missing, {len(bad)} differ "
                         f"(of {len(want)}) after attach")
    return peft


# ---- parity rows -------------------------------------------------------------------------------
def parity_scenarios(c: dict) -> list[str]:
    """16 of the replicate's 45 engaged H1_train scenarios, a seeded draw."""
    ids = sorted({r["scenario_id"] for r in _jsonl(c["replicate"] / "gen" / "arm_none.jsonl")})
    n = 2 if len(ids) < N_PARITY_SCEN else N_PARITY_SCEN
    return sorted(random.Random(f"{STAGE}|parity").sample(ids, n))


def parity_rows(sc, c: dict) -> list[dict]:
    out = []
    for k in range(4):
        for sid in parity_scenarios(c):
            s = c["H"][sid]
            o = L.assign_letters(s, k)
            out.append({"sid": sid, "k": k, "shown": [x for x, _ in o],
                        "ids": L.dose0_ids(sc, s, o, "dl_chat_neutral", sc.dose0)})
    return out


def hf_letters(sc, c: dict, tag: str) -> np.ndarray:
    rows = parity_rows(sc, c)
    lp, _ = L.read_ids(sc, [r["ids"] for r in rows], 1)
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / f"hf_{tag}.npy", lp)
    return lp


# ---- reads ------------------------------------------------------------------------------------
def reads_on_disk(c: dict, arm: str) -> bool:
    """An arm's attached reads already complete on disk (a resumed pod): every row of dose 0 on
    H1_held, the specificity set and each panel cell (the reads are deterministic)."""
    n = 8  # the letter read's permutations
    want = {f"dose0/{arm}/dl_chat_neutral.jsonl": len(c["held"]) * n,
            f"spec/{arm}/dl_chat_neutral.jsonl": len(c["spec_ids"]) * n}
    got = {k: len(_jsonl(c["out"] / k)) if (c["out"] / k).exists() else -1 for k in want}
    cells = list((c["out"] / "panel" / arm).glob("*.jsonl"))
    return all(got[k] == v for k, v in want.items()) and len(cells) == len(L.PANEL_CELLS)


def letter_reads(sc, c: dict, arm: str) -> dict:
    """Dose 0 on H1_held, the specificity set and the panel half under ``arm``."""
    if reads_on_disk(c, arm):
        return {"reused": "complete rows on disk from an earlier pod (deterministic reads)"}
    t0 = time.time()
    d0 = L.run_letter_cell(sc, [c["H"][s] for s in c["held"]], "dl_chat_neutral", f"dose0/{arm}")
    sp = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                           f"spec/{arm}")
    half = [c["panel"][i] for i in c["half_ids"]]
    cells = [L.run_letter_cell(sc, half, cell, f"panel/{arm}") for cell in L.PANEL_CELLS]
    return {"dose0_rows": d0["rows"], "spec_rows": sp["rows"],
            "panel_rows": int(sum(x["rows"] for x in cells)), "seconds": time.time() - t0}


def gen_vllm(c, a, sc, ckpt: Path | None, arm: str) -> dict:
    """One vLLM engine: the parity rows' first-token letters, H1_held x rr and the capability
    slice (T1.5's ``_gen_on`` on Muse; the same seeds for every arm, paired)."""
    import screen_vllm as SV
    import t15_driver as T15
    import vllm_phase as V

    rows = parity_rows(sc, c)
    jobs = L.h1_jobs(sc, c["H"], c["held"], c["rr"])
    wj = SV.worker_jobs(jobs, STAGE, c["cap"])
    cj, recs = T15.cap_jobs(sc, c, a)
    lj = T15.lp_jobs(sc, rows, arm)
    eng = V.engine_json(c["spec"], spec_on=False, label=f"{KEY}:{STAGE}:{arm}")
    if ckpt is not None:  # the merged copy; the registry's tokenizer
        eng.update(model=str(ckpt), revision=None, tokenizer=c["spec"]["repo"],
                   tokenizer_revision=c["spec"]["revision"])
    vrows, summ, rc = V.run_vllm(sc, lj + wj + cj, eng, c["out"] / "vllm" / f"{arm}.jsonl")
    by = {r["job_id"]: r for r in vrows}
    first = [by.get(j["job_id"], {}).get("first_logprobs") or {} for j in lj]
    g = [by[j["job_id"]]["gen_ids"] if by.get(j["job_id"], {}).get("status") == "finished"
         else None for j in wj]
    parsed = L.build_rows(sc, jobs, g, [j["seed"] for j in wj], cap=c["cap"], stage=STAGE,
                          engine="vllm_bf16")
    for r in parsed:
        r["arm"] = arm
    sc.save_jsonl(f"gen/h1_{arm}.jsonl", parsed)
    sc.save_jsonl(f"cap/{arm}.jsonl", T15.score_cap(sc, c["cap_items"], recs, by, arm))
    letters = T15.vllm_letters(rows, first, L.letter_ids(sc))
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / f"vllm_{arm}.npy", letters)
    return {"letters": letters, "rc": rc, "tokens_per_second": summ.get("tokens_per_second"),
            "n_h1": len(parsed)}


def engaged_held(c: dict) -> list[str]:
    """T1.1's rule on Muse's H1_held: P_dose0 >= 0.5 and P_none <= 0.5 (dose 0 on HF; ``none``'s
    generation on the base engine)."""
    d0 = _jsonl(c["out"] / "dose0" / "none" / "dl_chat_neutral.jsonl")
    pd = c["P"].dose0_protect(d0)
    rows = _jsonl(c["out"] / "gen" / "h1_none.jsonl")
    if not any(v >= 0.5 for v in pd.values()):  # no dose-0 regime (a tiny model): none engaged
        return []  # engaged_subset cannot run on an empty candidate set; the analysis reports it
    return sorted(L.engaged_subset(rows, pd, use_dose0=True)["engaged_ids"])


# ---- steps: data and base reads ----------------------------------------------------------------
def step_rr_gen(c, a) -> dict:
    import screen_vllm as SV
    import st2lb_driver as LB

    sc = vllm_screen(c, a)
    items = LB.pool_scenarios(a.dry_run)
    jobs = L.h1_jobs(sc, {s.id: s for s in items}, [s.id for s in items], c["rr"])
    g = SV.gen_stage(sc, jobs, "t16_rr", c["cap"], c["spec"], probe=True, deadline=a.deadline)
    for r, j in zip(g["rows"], jobs):
        r.update(prompt_ids=j["ids"], date_pin=PIN, sample="rule_right_pool")
    sc.save_jsonl("gen/rr_rollouts.jsonl", g["rows"])
    return {"status": "ok", "n": len(jobs), "n_items": len(items), "timing": g["timing"],
            "summary": L.screen_summary(g["rows"])}


def step_base_vllm(c, a) -> dict:
    out = gen_vllm(c, a, vllm_screen(c, a), None, "none")
    return {"status": "ok", "rc": out["rc"], "n_h1": out["n_h1"],
            "tokens_per_second": out["tokens_per_second"]}


def step_base_hf(c, a) -> dict:
    import t15_driver as T15

    mw = load_hf(c, a)
    sc = screen(c, mw, False)
    hf = hf_letters(sc, c, "none")
    reads = letter_reads(sc, c, "none")
    vl = np.load(c["out"] / "parity" / "vllm_none.npy")
    base_tv = T15.tv(vl, hf, parity_rows(sc, c))
    ok = bool(base_tv <= BASE_BAR) or a.dry_run
    return {"status": "ok", **reads, "base_tv": base_tv, "base_check_ok": ok,
            "bar": f"base_tv <= {BASE_BAR}", "band_tv": base_tv,
            "engaged_held": engaged_held(c)}


# ---- steps: pairs and training -----------------------------------------------------------------
def render_prompts(c: dict, rows: list[dict]) -> dict[tuple[str, int], list[int]]:
    """The replicate's prompt ids, re-rendered (its rows keep only the hash) and checked."""
    sc = screen(c, L.tok_only(c["tok"]), False)
    out: dict[tuple[str, int], list[int]] = {}
    for sid in sorted({r["scenario_id"] for r in rows}):
        n = 1 + max(r["rollout"] for r in rows if r["scenario_id"] == sid)
        for j in L.h1_jobs(sc, {sid: c["H"][sid]}, [sid], n):
            out[(sid, j["k"])] = j["ids"]
    for r in rows:
        if _sha(out[(r["scenario_id"], r["rollout"])]) != r["prompt_ids_sha256"]:
            raise SystemExit(f"re-rendered prompt differs from the replicate's: "
                             f"{r['scenario_id']} rollout {r['rollout']}")
    return out


def harm_pairs(care_rows: list[dict], none_rows: list[dict]) -> tuple[dict, dict]:
    """Per scenario, in rollout order: n = min(care protect, base protect, base rule, 2); care's
    chosen are care protect traces, base's are base protect traces, both reject the same base rule
    traces (Amendment 37 items 1-2). Pairs carry (scenario, rollout) keys; prompts come later."""
    def by(rows, status):
        d: dict[str, list] = {}
        for r in sorted(rows, key=lambda x: (x["scenario_id"], x["rollout"])):
            if r.get("norm_status") == status and r.get("gen_ids"):
                d.setdefault(r["scenario_id"], []).append(r)
        return d

    cp, bp, rj = by(care_rows, "violating"), by(none_rows, "violating"), by(none_rows,
                                                                            "consistent")
    out: dict = {"care": [], "base": []}
    counts = {}
    for sid in sorted(set(cp) | set(bp) | set(rj)):
        n = min(len(cp.get(sid, [])), len(bp.get(sid, [])), len(rj.get(sid, [])),
                PAIRS_PER_SCENARIO)
        counts[sid] = n
        for i in range(n):
            for arm, ch in (("care", cp[sid][i]), ("base", bp[sid][i])):
                out[arm].append({"scenario_id": sid, "chosen_key": (sid, ch["rollout"]),
                                 "chosen": ch["gen_ids"], "rejected": rj[sid][i]["gen_ids"],
                                 "rejected_key": (sid, rj[sid][i]["rollout"]),
                                 "kind": "rule_harm"})
    info = {"per_scenario": counts, "n_pairs": int(sum(counts.values())),
            "scenarios_with_pair": int(sum(v > 0 for v in counts.values())),
            "care_alone_pairs": int(sum(min(len(cp.get(s, [])), len(rj.get(s, [])),
                                            PAIRS_PER_SCENARIO) for s in set(cp) | set(rj)))}
    return out, info


def harm_pairs_fresh(care_rows: list[dict], none_rows: list[dict]) -> tuple[dict, dict]:
    """Amendment 49 addition item 1: care-DPO2's chosen traces are the steered protect traces T1.6
    did not use, in rollout order, per scenario up to T1.6's count; each pair keeps T1.6's rejected
    rule trace for that position, so the chosen sample (and the seed) are the only changes."""
    t16, info = harm_pairs(care_rows, none_rows)
    used = {tuple(p["chosen_key"]) for p in t16["care"]}
    prot: dict[str, list] = {}
    for r in sorted(care_rows, key=lambda x: (x["scenario_id"], x["rollout"])):
        if r.get("norm_status") == "violating" and r.get("gen_ids") \
                and (r["scenario_id"], r["rollout"]) not in used:
            prot.setdefault(r["scenario_id"], []).append(r)
    by_sid: dict[str, list] = {}
    for p in t16["care"]:
        by_sid.setdefault(p["scenario_id"], []).append(p)
    out, counts = [], {}
    for sid, ps in sorted(by_sid.items()):
        fresh = prot.get(sid, [])[:len(ps)]
        counts[sid] = len(fresh)
        for p, ch in zip(ps, fresh):
            out.append({**p, "chosen_key": (sid, ch["rollout"]), "chosen": ch["gen_ids"]})
    n = int(sum(counts.values()))
    return {"care": out}, {"per_scenario": counts, "n_pairs": n,
                           "t16_pairs": info["n_pairs"], "shortfall": info["n_pairs"] - n,
                           "scenarios_with_pair": int(sum(v > 0 for v in counts.values()))}


def pairs_for(c: dict, a, arm: str) -> tuple[list[dict], dict]:
    import st2lb_driver as LB

    care = _jsonl(c["replicate"] / "gen" / "arm_care_03.jsonl")
    none = _jsonl(c["replicate"] / "gen" / "arm_none.jsonl")
    hp, info = harm_pairs_fresh(care, none) if a.fresh_from_t16 else harm_pairs(care, none)
    prompts = render_prompts(c, care + none)
    harm = [{"scenario_id": p["scenario_id"], "prompt": prompts[p["chosen_key"]],
             "chosen": p["chosen"], "rejected": p["rejected"], "kind": p["kind"],
             "chosen_key": p["chosen_key"], "rejected_key": p["rejected_key"]}
            for p in hp[arm]]
    if a.dry_run:
        harm = [{**p, "chosen": p["chosen"][:24], "rejected": p["rejected"][:24]} for p in harm]
    rr = LB.rule_right_pairs(_jsonl(c["out"] / "gen" / "rr_rollouts.jsonl"), dry=a.dry_run)
    rr, cut = LB.cut_to(rr, len(harm))
    return harm + rr, {**info, "n_harm": len(harm), "rule_right": cut}


def step_train(c, a) -> dict:
    import st_train as ST
    import t14_driver as T14

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.steering import objective_lora as OL

    arm = a.arm
    pr, info = pairs_for(c, a, arm)
    rc = c["recipe"]
    if a.dry_run:
        wb = WhiteBoxModel(str(tiny_path(c)), access_tier=AccessTier.WEIGHTS)
    else:
        wb = OL.qlora_model(c["spec"]["repo"], c["spec"]["revision"],
                            access_tier=AccessTier.WEIGHTS)
    gen_tr, _ = T14.general(c, a)
    if rc["gradient_checkpointing"] and not a.dry_run:
        wb.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":
                                                                              False})
    kw = dict(layer=c["layer"], pairs=pr, general=gen_tr, kl_weight=rc["kl_weight"], lr=rc["lr"],
              grad_accum=rc["grad_accum"], seed=a.seed, lora_rank=rc["rank"],
              lora_alpha=rc["alpha"], lora_dropout=rc["dropout"])
    pilot = None
    if a.fresh_from_t16:  # Amendment 49: T1.6's steps; the pilot only checks the bail
        tp = time.time()
        ST.run(wb, "beh", max_steps=1, **{**kw, "grad_accum": T14.PILOT_MICRO})
        pilot = {"micro": T14.PILOT_MICRO, "seconds": time.time() - tp}
        pilot["seconds_per_micro"] = pilot["seconds"] / T14.PILOT_MICRO
        wb._model = wb.model.unload()  # noqa: SLF001
        if rc["gradient_checkpointing"] and not a.dry_run:
            wb.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        steps = min(FRESH_STEPS, rc["max_steps"])
        proj_h = pilot["seconds_per_micro"] * rc["grad_accum"] * steps / 3600
        pilot["projected_hours"] = proj_h
        if proj_h > FRESH_BAIL_H and not a.dry_run:
            return {"status": "stop: the pilot projects training beyond the bail",
                    "pilot": pilot, "pairs": info}
    elif arm == "care":  # the pilot fits the steps to the budget (T1.4's rule)
        tp = time.time()
        ST.run(wb, "beh", max_steps=1, **{**kw, "grad_accum": T14.PILOT_MICRO})
        pilot = {"micro": T14.PILOT_MICRO, "seconds": time.time() - tp}
        pilot["seconds_per_micro"] = pilot["seconds"] / T14.PILOT_MICRO
        wb._model = wb.model.unload()  # noqa: SLF001
        if rc["gradient_checkpointing"] and not a.dry_run:
            wb.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        fit = int(rc["budget_h"] * 3600 / max(pilot["seconds_per_micro"] * rc["grad_accum"],
                                              1e-9))
        steps = min(rc["max_steps"], max(fit, 1))
    else:  # base reuses care's steps: the teacher is the only difference
        steps = int(phase(c, "train_care")["max_steps_run"])
    t0 = time.time()
    res = ST.run(wb, "beh", max_steps=steps, **kw)
    adir = c["adapters"] / arm
    adir.mkdir(parents=True, exist_ok=True)
    wb.model.save_pretrained(str(adir))
    ids = {"chosen": [list(p["chosen_key"]) for p in pr if p.get("kind") == "rule_harm"],
           "rejected": [list(p["rejected_key"]) for p in pr if p.get("kind") == "rule_harm"]}
    side = {"arm": arm, "pairs": info, "recipe": rc, "seed": a.seed, "max_steps_run": steps,
            "rollout_ids": ids, "fresh_from_t16": bool(a.fresh_from_t16),
            "pilot": pilot, "train_seconds": time.time() - t0,
            "last_steps": res.get("steps", [])[-3:]}
    side["adapter_sha256"] = {f: hashlib.sha256((adir / f).read_bytes()).hexdigest()
                              for f in ("adapter_config.json", "adapter_model.safetensors")}
    (adir / "st_adapter.json").write_text(json.dumps(side, indent=1, default=str))
    return {"status": "ok", **{k: side[k] for k in ("max_steps_run", "pilot", "train_seconds")},
            "pairs": info}


# ---- steps: the adapter's reads and the merge --------------------------------------------------
class AdapterDeltas:
    """``(alpha / r) B @ A`` per module path, computed in fp32 when asked for and not kept: the
    full set for Muse's adapter is 93.7 GiB in fp32, past a 27 GiB pod's RAM (the first a2f pod,
    2026-10-25), so the merge holds one delta at a time. Iterating yields the module paths."""

    def __init__(self, adir: Path):
        from safetensors.torch import load_file

        cfg = json.loads((adir / "adapter_config.json").read_text())
        self.scale = cfg["lora_alpha"] / cfg["r"]
        self.sd = load_file(str(adir / "adapter_model.safetensors"))
        self.keys = {k.split(".lora_A.")[0].removeprefix("base_model.model."): k
                     for k in self.sd if ".lora_A." in k}

    def __iter__(self):
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, path: str):
        import torch

        k = self.keys[path]
        d = (self.sd[k.replace(".lora_A.", ".lora_B.")].float() @ self.sd[k].float()) * self.scale
        assert d.dim() == 2 and torch.isfinite(d).all(), path
        return d


def adapter_deltas(adir: Path) -> AdapterDeltas:
    """``{module path: (alpha / r) B @ A}`` from a saved LoRA adapter (fp32), computed lazily."""
    return AdapterDeltas(adir)


def match_keys(deltas: dict, file_keys: list[str]) -> dict[str, str]:
    """Adapter module path -> the checkpoint's tensor key: exact (``path.weight``) where it exists,
    else the unique language-model key with the same ``layers.<n>...`` suffix; every delta matched
    once, or the merge fails (the registered fallback runs)."""
    keys = set(file_keys)
    lang = [k for k in file_keys if k.endswith(".weight") and ("language" in k or "text" in k)
            and not any(v in k for v in ("vision", "visual"))]
    out = {}
    for p in deltas:
        if f"{p}.weight" in keys:
            out[p] = f"{p}.weight"
            continue
        if "layers." not in p:
            raise ValueError(f"adapter module outside the decoder layers: {p}")
        suf = p[p.index("layers."):] + ".weight"
        hits = [k for k in lang if k.endswith("." + suf) or k == suf]
        if len(hits) != 1:
            raise ValueError(f"{p}: {len(hits)} checkpoint keys end in {suf}")
        out[p] = hits[0]
    if len(set(out.values())) != len(out):
        raise ValueError("two adapter modules map to one checkpoint tensor")
    return out


def write_merged(src: Path, adir: Path, dst: Path) -> dict:
    """The author's merge (2026-10-24): a copy of the base snapshot with only the language-model
    tensors the adapter touches overwritten in place (W + (alpha / r) B @ A, in fp32, cast to the
    file's dtype); every other file and tensor is the base's, byte for byte."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    deltas = adapter_deltas(adir)
    shards = sorted(src.glob("*.safetensors"))
    keys_by = {}
    for f in shards:
        with safe_open(str(f), "pt") as g:
            keys_by[f.name] = list(g.keys())
    mapping = match_keys(deltas, [k for v in keys_by.values() for k in v])
    by_file: dict[str, list[str]] = {}
    for p, k in mapping.items():
        by_file.setdefault(next(n for n, v in keys_by.items() if k in v), []).append(p)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():  # configs, tokenizer, processor, index: copied (symlinks resolved)
        if f.is_file() and f.name not in by_file:
            shutil.copyfile(f.resolve(), dst / f.name)
    changed, dot, sq = 0, 0.0, 0.0
    for name, paths in by_file.items():
        with safe_open(str(src / name), "pt") as g:
            meta = g.metadata()
            t = {k: g.get_tensor(k) for k in g.keys()}
        for p in paths:
            k = mapping[p]
            w = t[k]
            dp = deltas[p]  # computed now, freed after this tensor (one delta in memory at a time)
            if tuple(w.shape) != tuple(dp.shape):
                raise ValueError(f"{k}: shape {tuple(w.shape)} vs delta {tuple(dp.shape)}")
            t[k] = (w.float() + dp.to(w.device)).to(w.dtype).contiguous()
            got = t[k].float() - w.float()  # what survived the cast to the file's dtype
            dot += float((got * dp).sum())
            sq += float((dp * dp).sum())
            changed += 1
            del dp, got
        save_file(t, str(dst / name), metadata=meta)
        del t
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    written = {}
    for f in sorted(dst.glob("*.safetensors")):
        with safe_open(str(f), "pt") as g:
            written[f.name] = sorted(g.keys())
    same = written == {n: sorted(v) for n, v in keys_by.items()}
    return {"ok": bool(same and changed == len(deltas)), "dir": str(dst),
            "n_deltas": len(deltas), "n_changed": changed, "n_shards_rewritten": len(by_file),
            "keys_identical": same, "weight_kept_share": dot / sq if sq > 0 else None,
            "note": "weight_kept_share: the share of the adapter's deltas that survives the "
                    "cast to the file's dtype (bf16 rounds deltas below half an ulp of a weight)"}


def step_hf_arm(c, a) -> dict:
    import gc

    import t15_driver as T15
    import torch

    arm = a.arm
    mw = load_hf(c, a)
    sc = screen(c, mw, False)
    attach(mw, c["adapters"] / arm)
    lp = hf_letters(sc, c, arm)
    reads = letter_reads(sc, c, arm)
    rows = parity_rows(sc, c)
    del mw, sc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    dst = c["out"] / "merged" / arm
    try:
        saved = write_merged(snapshot(c, a), c["adapters"] / arm, dst)
    except ValueError as e:  # a layout the in-place writer cannot map: the fallback runs
        saved = {"ok": False, "reason": str(e)}
    merge = {"saved": saved}
    if saved.get("ok"):
        mm = load_hf(c, a, str(dst))
        lp_m = hf_letters(screen(c, mm, False), c, f"{arm}_merged")
        del mm
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        base = np.load(c["out"] / "parity" / "hf_none.npy")
        shift, lost, to_base = T15.tv(lp, base, rows), T15.tv(lp_m, lp, rows), \
            T15.tv(lp_m, base, rows)
        merge.update(adapter_tv_vs_base=shift, merged_tv_vs_attached=lost,
                     merged_tv_vs_base=to_base,
                     kept_share=None if shift <= 0 else 1.0 - lost / shift,
                     closer_to_base=bool(lost > to_base))
        merge["gate_ok"] = bool(not merge["closer_to_base"]) or a.dry_run
    else:
        merge["gate_ok"] = False
    if not merge["gate_ok"]:
        shutil.rmtree(dst, ignore_errors=True)
    return {"status": "ok", **reads, "merge": merge}


def step_vllm_arm(c, a) -> dict:
    import t15_driver as T15

    arm = a.arm
    base, hfp = phase(c, "base_hf"), phase(c, f"hf_arm_{arm}")
    dst = c["out"] / "merged" / arm
    if not base.get("base_check_ok"):
        shutil.rmtree(dst, ignore_errors=True)
        return {"status": "ok", "parity_ok": False, "reason": "base check failed: HF for all"}
    if not (hfp.get("merge") or {}).get("gate_ok"):
        return {"status": "ok", "parity_ok": False, "reason": "merge gate failed: HF"}
    sc = vllm_screen(c, a)
    out = gen_vllm(c, a, sc, dst, arm)
    d = T15.tv(out["letters"], np.load(c["out"] / "parity" / f"hf_{arm}.npy"),
               parity_rows(sc, c))
    ok = (bool(d <= base["band_tv"] + PARITY_MARGIN) or a.dry_run) and not a.dry_fail_parity
    if not ok:  # the merged engine's rows are not of record: set aside, the fallback runs
        for f in (c["out"] / "gen" / f"h1_{arm}.jsonl", c["out"] / "cap" / f"{arm}.jsonl"):
            if f.exists():
                f.rename(f.with_suffix(".parity_failed.jsonl"))
    if not a.dry_run:
        shutil.rmtree(dst, ignore_errors=True)
    return {"status": "ok", "parity_tv": d, "band_tv": base["band_tv"], "parity_ok": ok,
            "tokens_per_second": out["tokens_per_second"]}


def needs_hf(c: dict, arm: str) -> bool:
    """The registered fallback: an arm whose merge gate or vLLM parity failed; ``none`` when the
    base check failed or any arm needs HF (ΔP compares rows of one engine)."""
    if not phase(c, "base_hf").get("base_check_ok"):
        return True
    if arm == "none":
        return any(needs_hf(c, x) for x in ARMS)
    return not phase(c, f"vllm_arm_{arm}").get("parity_ok")


def step_hfgen(c, a) -> dict:
    """The engaged H1_held x rr and the capability slice's T1.5 half on HF (adapter attached for
    an arm), with the vLLM jobs' seeds; rows under ``.hf``."""
    import screen_vllm as SV
    import steering_gate as SG
    import t15_driver as T15

    arm = a.arm
    if not needs_hf(c, arm):
        return {"status": "ok", "skipped": "not needed (vLLM rows of record)"}
    mw = load_hf(c, a)
    sc = screen(c, mw, False)
    if arm != "none":
        attach(mw, c["adapters"] / arm)
    eng = phase(c, "base_hf")["engaged_held"]
    if not eng and not a.dry_run:
        return {"status": "ok", "skipped": "no engaged H1_held scenario: no ΔP of record"}
    eng = eng or c["held"][:1]
    t0 = time.time()
    jobs = L.h1_jobs(sc, c["H"], eng, c["rr"])
    seeds = [j["seed"] for j in SV.worker_jobs(jobs, STAGE, c["cap"])]
    g = SG.generate(mw, [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"], seeds=seeds,
                    batch=c["batch"], on_finish=SG.jsonl_saver(c["out"] / "gen" /
                                                               f"h1_{arm}.hf.raw.jsonl"))
    rows = L.build_rows(sc, jobs, [r["gen_ids"] for r in g], seeds, cap=c["cap"], stage=STAGE,
                        engine="hf_adapter" if arm != "none" else "hf")
    for r in rows:
        r["arm"] = arm
    sc.save_jsonl(f"gen/h1_{arm}.hf.jsonl", rows)
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
    return {"status": "ok", "engine": "hf", "n_h1": len(rows), "n_engaged": len(eng),
            "n_capability": len(cj), "seconds": time.time() - t0}


STEPS = {"rr_gen": step_rr_gen, "base_vllm": step_base_vllm, "base_hf": step_base_hf,
         "train": step_train, "hf_arm": step_hf_arm, "vllm_arm": step_vllm_arm,
         "hfgen": step_hfgen}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=tuple(STEPS))
    ap.add_argument("--arm", choices=("care", "base", "none"), default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicate", type=Path, required=True,
                    help="the replicate's muse_glimmer_30b dir (gen/arm_care_03, gen/arm_none)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--dry-fail-parity", action="store_true",
                    help="dry runs only: the arm's vLLM parity fails, so the fallback path runs")
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (Amendment 49: 205)")
    ap.add_argument("--fresh-from-t16", action="store_true",
                    help="care-DPO2 (Amendment 49): chosen traces T1.6 did not use, T1.6's steps")
    a = ap.parse_args()
    if a.fresh_from_t16 and (a.step != "train" or a.arm != "care"):
        ap.error("--fresh-from-t16 is for --step train --arm care")
    if a.dry_fail_parity and not a.dry_run:
        ap.error("--dry-fail-parity is for dry runs")
    if a.step in ("train", "hf_arm", "vllm_arm") and a.arm not in ARMS:
        ap.error(f"--step {a.step} needs --arm care|base")
    if a.step == "hfgen" and a.arm is None:
        ap.error("--step hfgen needs --arm care|base|none")
    c = context(a)
    name = a.step if a.arm is None else f"{a.step}_{a.arm}"
    if phase(c, name).get("status") == "ok":
        print(f">> t16 {name}: finished earlier", flush=True)
        return 0
    t0 = time.time()
    res = STEPS[a.step](c, a)
    update_record(c, name, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "n_h1", "base_tv", "base_check_ok", "max_steps_run",
                                 "parity_ok", "skipped", "status") if k in res}
    if "merge" in res:
        brief["merge_gate_ok"] = res["merge"].get("gate_ok")
    print(f">> t16 {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
