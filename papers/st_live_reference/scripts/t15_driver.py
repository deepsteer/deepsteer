"""The T1.5 pod (ST_PLAN §6, Amendments 17-20; author, 2026-10-21): GPT-OSS-20B's adapters
evaluated against a precision-matched baseline. HF phases run on the ST stack of record (torch 2.13,
eager, bf16 dequantized); vLLM phases serve bf16 checkpoints (the unadapted model and each merged
adapter). ``--phase all`` runs every phase as its own process (so the HF model is gone before a vLLM
engine starts), re-reading the record between phases; each phase is recorded and skipped on resume.

Order (Amendment 20 item 4):

1. ``beh_cap`` (HF): L-beh's step-matched checkpoint on the reserved 10 x 8 at cap 4,096;
   truncation <= 5% = finishes (a cap problem; the comparator stands), else termination is broken
   and the epoch-3 checkpoint is the comparator of record (Amendment 18 item 3).
2. ``base_hf`` (HF, unadapted): the parity rows' letter log-probs; the bf16 checkpoint saved.
3. ``base_vllm``: the engine band (MXFP4 on vLLM vs the HF stack, mean TV over the parity rows),
   the base check (the bf16 checkpoint on vLLM within band + 0.05), and on the bf16 engine the
   ``none`` baseline of record: H1_held 132 x 8 and the capability slice (Amendment 20 item 1). A
   failed base check sends every adapter to HF (item 2) and, so ``none`` shares their engine,
   ``none`` too (``hfgen_none``; the bf16-vLLM rows are kept as ``*.parity_failed.jsonl``).
4. per adapter (comparator first, then the other L-beh checkpoint, L-anchor, L-self):
   ``hf_<arm>`` (parity rows, the specificity cell, the panel guard on the 50% subsample for the
   comparator and L-anchor, then the merge saved as bf16) and ``vllm_<arm>`` (parity within band +
   0.05, then H1_held 132 x 8 and capability on the merged engine; the checkpoint deleted);
   ``hfgen_<arm>`` only for an adapter that failed parity (HF generation, Amendment 18 item 4
   scope: ΔP's 38 brake-engaged H1_held scenarios x 8, then capability's 100-item slice, then the
   other 94 x 8 for the brake if the deadline leaves room).
5. ``readouts`` (HF): on a seeded 2 of 8 rollouts per H1_held scenario, for ``none`` and every
   adapter, the anchor-axis margin at ``clause_end``, block 12 (L-anchor's transport control of
   record) and the post-reasoning letter (Amendment 20 item 3).

Every analysis (ΔP, the brake, the guard's ΔE, specificity, capability) runs locally from the saved
rows; the pod only measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402

KEY = "gpt_oss_20b"
DATA = HERE.parent / "data"
L_STAR = 13
CAP_RERUN = 4096
FINISH_MAX_TRUNC = 0.05
PARITY_MARGIN = 0.05
N_PARITY = 64
CAP_EXTRA = 256  # capability answers: the trace cap of record plus room for a worked answer
SEED = 20261021
ARMS_ALL = ("beh", "beh_epoch3", "anchor", "self")
# proposed to the author (not yet of record, so off): a merge that sits closer to the base than to
# the attached adapter on the parity rows sends the arm to HF (parity alone cannot see a lost merge
# when the adapter moves the dose-0 letters by less than the band + 0.05)
MERGE_GATE = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ---- context --------------------------------------------------------------------------------
def context(a) -> dict:
    import glob

    import pod_st_screen as P
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
    panel = {s.id: s for s in L.panel_586(allp)}
    meta14 = json.loads((a.t14_inputs / "t14_inputs.json").read_text())
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept}, "panel": panel,
         "held": sorted(split["H1_held"]), "reserved": list(meta14["reserved"]),
         "engaged": json.loads((DATA / "t15_engaged_held_v1.json").read_text())["ids"],
         "gate": json.loads((DATA / "t13_draw_v1.json").read_text())["gate"],
         "spec_ids": json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"],
         "half_ids": json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"],
         "cap_items": json.loads((DATA / "t15_capability_v1.json").read_text())["items"],
         "axis": np.load(a.t14_inputs / "t14_inputs.npz")["anchor_axis"],
         "adapters": a.adapters, "cap": L.CAP_OF_RECORD, "batch": 160, "rr": 8}
    if a.dry_run:  # a tiny GPT-OSS; a few scenarios and items through every path
        held = [c["engaged"][0]] + [s for s in c["held"] if s not in c["engaged"]][:1]
        c.update(held=held, reserved=c["reserved"][:1], spec_ids=c["spec_ids"][:2],
                 half_ids=c["half_ids"][:2], cap_items=c["cap_items"][:1] + c["cap_items"][-1:],
                 cap=32, batch=8, rr=2)
    return c


def record_path(c: dict) -> Path:
    return c["out"] / "t15_record.json"


def load_record(c: dict) -> dict:
    p = record_path(c)
    return json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    record_path(c).write_text(json.dumps(rec, indent=1, default=L._json_default))


def arms_order(rec: dict) -> list[str]:
    comp = (rec["phases"].get("beh_cap") or {}).get("comparator", "beh")
    other = "beh_epoch3" if comp == "beh" else "beh"
    return [comp, other, "anchor", "self"]


def panel_arms(rec: dict) -> set[str]:
    """The guard's arms: the comparator of record and L-anchor (Amendments 17, 18, 20)."""
    return {arms_order(rec)[0], "anchor"}


# ---- models ---------------------------------------------------------------------------------
def load_hf(c: dict, a):
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = c["out"] / "_tiny"
        if not (path / "config.json").exists():
            st_tiny.build(KEY, path)
        mw = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        fmt = c["fmt"]  # a random model never writes the harmony header: open the trace at once
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        _dry_adapters(c, mw)
        return mw
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        attn_implementation="eager", **c["P"].hf_load_kwargs(c["spec"]))


def _dry_adapters(c: dict, mw) -> None:
    """Dry run only: zero-initialized adapters at the tiny width under every arm's name."""
    from deepsteer.steering import objective_lora as OL

    for arm in ARMS_ALL:
        d = c["adapters"] / arm
        if (d / "adapter_config.json").exists():
            continue
        OL.inject_lora(mw.wb, rank=4, dropout=0.0)
        mw.wb.model.save_pretrained(str(d))
        mw.wb._model = mw.wb.model.unload()  # noqa: SLF001
        mw.model = mw.wb.model


def attach(mw, adir: Path):
    """The adapter injected into ``mw.model`` in place (its own forward includes the LoRA)."""
    from peft import PeftModel

    return PeftModel.from_pretrained(mw.model, str(adir))


def screen(c: dict, model, a) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, dry=False, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    sc.unbatched = True  # single-row readouts of record (Amendment 9 item 2)
    sc.layers_override = [2] if a.dry_run else [L_STAR]
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def vllm_screen(c: dict, a) -> L.Screen:
    """The vLLM phases' screen: the tokenizer only (no GPU); a dry run's stand-in generates."""
    if a.dry_run:
        from screen_stub import ScreenStub

        stub = ScreenStub(KEY, c["tok"], c["fmt"], c["spec"]["n_layers"],
                          length_by_effort={e: 4 for e in c["fmt"].effort_ladder})
        sc = screen(c, stub, a)  # short traces that close inside the dry cap
    else:
        sc = screen(c, L.tok_only(c["tok"]), a)
    sc.dry = a.dry_run
    return sc


def save_bf16(model, dst: Path) -> dict:
    """A bf16 checkpoint vLLM loads through its unquantized path: the module state dict written
    as is (HF names, ``gate_up_proj`` [E, H, 2I], ``down_proj`` [E, I, H]) and the config without
    ``quantization_config``. Not ``save_pretrained``: after a dequantizing MXFP4 load it reverts
    the load's weight conversions and drops ``gate_up_proj`` and ``down_proj`` (only their biases
    reach the file). The written keys are checked against the state dict; a failed check returns
    ``ok`` False (the checkpoint removed), so the arm generates on HF instead of the session
    stopping."""
    import copy

    from huggingface_hub import save_torch_state_dict
    from safetensors import safe_open

    sd = model.state_dict()
    bad = [k for k in sd if k.endswith(("_blocks", "_scales")) or "lora" in k or "base_layer" in k]
    if bad:
        return {"ok": False, "reason": f"unmerged or quantized tensors {bad[:4]}"}
    dst.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(sd, str(dst), max_shard_size="5GB")
    cfg = copy.deepcopy(model.config)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    cfg.architectures = [model.__class__.__name__]
    cfg.save_pretrained(str(dst))
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(str(dst))
    written: set[str] = set()
    for f in dst.glob("*.safetensors"):
        with safe_open(str(f), "pt") as g:
            written |= set(g.keys())
    if written != set(sd):
        shutil.rmtree(dst, ignore_errors=True)
        return {"ok": False, "reason": f"{len(set(sd) - written)} tensors missing from the file"}
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    return {"ok": True, "dir": str(dst), "bytes": size, "n_tensors": len(written),
            "writer": "huggingface_hub.save_torch_state_dict (module state dict)"}


# ---- rows -----------------------------------------------------------------------------------
def parity_rows(sc, c: dict) -> list[dict]:
    """64 dose-0 rows: the 20 gate scenarios x permutations 0-3, the first 64."""
    out = []
    for k in range(4):
        for sid in c["gate"]:
            s = c["H"][sid]
            o = L.assign_letters(s, k)
            out.append({"sid": sid, "k": k, "shown": [x for x, _ in o],
                        "ids": L.dose0_ids(sc, s, o, "dl_chat_neutral", sc.dose0)})
    return out[:N_PARITY]


def tv(lp_a: np.ndarray, lp_b: np.ndarray, rows: list[dict]) -> float:
    """Mean total variation between the letter distributions over each row's shown letters
    (a letter missing from vLLM's top-20 is probability 0)."""
    d = []
    for r, x, y in zip(rows, lp_a, lp_b):
        idx = ["ABCDE".index(s) for s in r["shown"]]
        px, py = np.exp(np.asarray(x)[idx]), np.exp(np.asarray(y)[idx])
        if px.sum() <= 0 or py.sum() <= 0:  # no shown letter in vLLM's top 20: maximal distance
            d.append(1.0)
            continue
        d.append(0.5 * float(np.abs(px / px.sum() - py / py.sum()).sum()))
    return float(np.mean(d))


def vllm_letters(rows: list[dict], first: list[dict], lids: list[int]) -> np.ndarray:
    return np.array([[f.get(str(t), -np.inf) for t in lids] for f in first])


def cap_prompt(item: dict, rec: dict) -> str:
    if item["dataset"] == "openai/gsm8k":
        return (f"{rec['question']}\n\nSolve the problem. End your final answer with the number "
                "alone.")
    ch = "\n".join(f"{'ABCD'[i]}. {x}" for i, x in enumerate(rec["choices"]))
    return f"{rec['question']}\n\n{ch}\n\nAnswer with the letter of the correct option."


def cap_gold(item: dict, rec: dict) -> str:
    if item["dataset"] == "openai/gsm8k":
        return rec["answer"].split("####")[-1].strip().replace(",", "")
    return "ABCD"[int(rec["answer"])]


def cap_parse(item: dict, text: str | None) -> str | None:
    """GSM8K: the last number in the final answer. MMLU: "answer is X", else a leading letter
    closed by punctuation or the end ("B", "B.", "(B)"), else the last standalone A-D (the
    final-answer text is saved, so the parse can be re-run)."""
    if text is None:
        return None
    if item["dataset"] == "openai/gsm8k":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "").rstrip(".") if nums else None
    m = re.search(r"[Aa]nswer(?: is|:)\W*([ABCD])\b", text)
    if m:
        return m.group(1)
    m = re.match(r"\W*\(?([ABCD])(?:[.:)]|$)", text.strip())
    if m:
        return m.group(1)
    ms = re.findall(r"\b([ABCD])\b", text)
    return ms[-1] if ms else None


def final_text(sc, gen: list[int]) -> str | None:
    """The whole final answer, decoded (``refusal_retention.final_answer`` keeps 64 tokens, short
    of a worked GSM8K answer); None when the reasoning never closes into a final answer."""
    loc = sc.fmt._locate(sc.tok, list(gen))  # noqa: SLF001
    if loc is None or isinstance(loc, str):
        return None
    return sc.tok.decode(list(gen)[loc[1]:], skip_special_tokens=True)


def cap_fallback(items: list[dict]) -> list[int]:
    """Amendment 19 item 4's 100-item slice: the first 50 of each dataset, in order."""
    seen: dict[str, int] = {}
    out = []
    for i, it in enumerate(items):
        seen[it["dataset"]] = seen.get(it["dataset"], 0) + 1
        if seen[it["dataset"]] <= 50:
            out.append(i)
    return out


def cap_records(c: dict, a) -> list[dict]:
    """The committed items, re-loaded and verified against their sha256."""
    if a.dry_run:
        return [{"question": "What is 2 + 2?", "answer": "#### 4"},
                {"question": "Pick one.", "choices": ["a", "b", "c", "d"], "answer": 1}]
    from datasets import load_dataset

    cache: dict = {}
    out = []
    for it in c["cap_items"]:
        key = (it["dataset"], it["config"])
        if key not in cache:
            cache[key] = load_dataset(it["dataset"], it["config"], split=it["split"])
        r = cache[key][it["index"]]
        if hashlib.sha256(r["question"].encode()).hexdigest() != it["sha256"]:
            raise SystemExit(f"capability item {it['dataset']}#{it['index']} changed")
        out.append(dict(r))
    return out


def cap_jobs(sc, c: dict, a) -> tuple[list[dict], list[dict]]:
    recs = cap_records(c, a)
    jobs = []
    for n, (it, r) in enumerate(zip(c["cap_items"], recs)):
        msgs = [{"role": "user", "content": cap_prompt(it, r)}]
        ids = sc.fmt.prompt_ids(sc.tok, sc.render(msgs))
        jobs.append({"job_id": f"cap|{n}", "prompt_ids": ids, "max_tokens": c["cap"] + CAP_EXTRA,
                     "seed": zlib.crc32(f"t15cap|{n}".encode()) % (2**31)})
    return jobs, recs


def score_cap(sc, items: list[dict], recs: list[dict], vrows: dict, arm: str) -> list[dict]:
    out = []
    for n, (it, r) in enumerate(zip(items, recs)):
        v = vrows.get(f"cap|{n}") or {}
        text = final_text(sc, v.get("gen_ids") or []) if v.get("status") == "finished" else None
        pred, gold = cap_parse(it, text), cap_gold(it, r)
        out.append({"arm": arm, "dataset": it["dataset"], "index": it["index"], "pred": pred,
                    "gold": gold, "correct": pred is not None and pred == gold,
                    "answered": text is not None, "final_text": None if text is None
                    else text[:2000], "n_tokens": v.get("n_tokens"),
                    "finish_reason": v.get("finish_reason")})
    return out


def lp_jobs(sc, rows: list[dict], tag: str) -> list[dict]:
    lids = L.letter_ids(sc)
    return [{"job_id": f"lp|{i}", "prompt_ids": r["ids"], "max_tokens": 1,
             "seed": zlib.crc32(f"t15lp|{tag}|{i}".encode()) % (2**31), "logprob_ids": lids}
            for i, r in enumerate(rows)]


# ---- phases ---------------------------------------------------------------------------------
def phase_beh_cap(c, a, rec) -> dict:
    import steering_gate as SG

    mw = load_hf(c, a)
    sc = screen(c, mw, a)
    attach(mw, c["adapters"] / "beh")
    jobs = L.h1_jobs(sc, c["H"], c["reserved"], c["rr"])
    seeds = [zlib.crc32(f"t15cap4096|{j['s'].id}|{j['k']}".encode()) % (2**31) for j in jobs]
    cap = 24 if a.dry_run else CAP_RERUN
    rows = SG.generate(mw, [j["ids"] for j in jobs], fmt=c["fmt"], cap=cap, seeds=seeds,
                       batch=c["batch"], on_finish=SG.jsonl_saver(c["out"] / "gen" /
                                                                  "beh_cap4096.raw.jsonl"))
    parsed = L.build_rows(sc, jobs, [r["gen_ids"] for r in rows], seeds, cap=cap,
                          stage="t15_cap4096", engine="hf_adapter")
    sc.save_jsonl("gen/beh_cap4096.jsonl", parsed)
    trunc = float(np.mean([r["status"] == "truncated" for r in parsed]))
    act = [r["norm_status"] for r in parsed if r.get("norm_status") is not None]
    p = float(np.mean([x == "violating" for x in act])) if act else None
    finishes = trunc <= FINISH_MAX_TRUNC
    return {"status": "ok", "truncation": trunc, "P_parsed": p, "n": len(parsed),
            "finishes": finishes,
            "reading": ("finishes (a cap problem if it protects: P_parsed)" if finishes
                        else "termination broken"),
            "comparator": "beh" if finishes else "beh_epoch3",
            "rule": "Amendment 18 item 3 (finishes = truncation <= 5% at cap 4,096)"}


def phase_base_hf(c, a, rec) -> dict:
    mw = load_hf(c, a)
    sc = screen(c, mw, a)
    rows = parity_rows(sc, c)
    lp, _ = L.read_ids(sc, [r["ids"] for r in rows], 1)
    (c["out"] / "parity").mkdir(parents=True, exist_ok=True)
    np.save(c["out"] / "parity" / "hf_none.npy", lp)
    saved = save_bf16(mw.model, c["out"] / "merged" / "none")
    return {"status": "ok", "n_parity_rows": len(rows), "saved": saved}


def _vllm_engine(c: dict, ckpt: Path | None, label: str) -> dict:
    import vllm_phase as V

    eng = V.engine_json(c["spec"], spec_on=False, label=label)
    if ckpt is not None:  # a local bf16 checkpoint; the registry's tokenizer (ST-A7's route)
        eng.update(model=str(ckpt), revision=None, tokenizer=c["spec"]["repo"],
                   tokenizer_revision=c["spec"]["revision"])
    return eng


def _gen_on(c, a, sc, ckpt: Path | None, arm: str, rows: list[dict]) -> dict:
    """One vLLM engine: the parity rows' first-token letters, H1_held x rr and the capability
    slice; rows saved under gen/ and cap/."""
    import screen_vllm as SV
    import vllm_phase as V

    jobs = L.h1_jobs(sc, c["H"], c["held"], c["rr"])
    wj = SV.worker_jobs(jobs, "t15", c["cap"])  # the same seeds for every arm (paired)
    cj, recs = cap_jobs(sc, c, a)
    lj = lp_jobs(sc, rows, arm)
    vrows, summ, rc = V.run_vllm(sc, lj + wj + cj, _vllm_engine(c, ckpt, f"{KEY}:t15:{arm}"),
                                 c["out"] / "vllm" / f"{arm}.jsonl")
    by = {r["job_id"]: r for r in vrows}
    first = [by.get(j["job_id"], {}).get("first_logprobs") or {} for j in lj]
    g = [by[j["job_id"]]["gen_ids"] if by.get(j["job_id"], {}).get("status") == "finished"
         else None for j in wj]
    parsed = L.build_rows(sc, jobs, g, [j["seed"] for j in wj], cap=c["cap"], stage="t15",
                          engine="vllm_bf16")
    for r in parsed:
        r["arm"] = arm
    sc.save_jsonl(f"gen/h1_{arm}.jsonl", parsed)
    sc.save_jsonl(f"cap/{arm}.jsonl", score_cap(sc, c["cap_items"], recs, by, arm))
    return {"letters": vllm_letters(rows, first, L.letter_ids(sc)),
            "tokens_per_second": summ.get("tokens_per_second"), "rc": rc,
            "settings_sha256": summ.get("settings_sha256")}


def phase_base_vllm(c, a, rec) -> dict:
    import vllm_phase as V

    sc = vllm_screen(c, a)
    rows = parity_rows(sc, c)
    hf = np.load(c["out"] / "parity" / "hf_none.npy")
    mx_rows, _, _ = V.run_vllm(sc, lp_jobs(sc, rows, "mxfp4"), _vllm_engine(c, None,
                                                                          f"{KEY}:t15:mxfp4"),
                               c["out"] / "vllm" / "mxfp4_parity.jsonl")
    by = {r["job_id"]: r for r in mx_rows}
    mx = vllm_letters(rows, [by.get(f"lp|{i}", {}).get("first_logprobs") or {}
                             for i in range(len(rows))], L.letter_ids(sc))
    band = tv(mx, hf, rows)
    saved = rec["phases"]["base_hf"]["saved"]
    if not saved.get("ok"):  # no bf16 engine: item 2's failure branch (every arm on HF)
        return {"status": "ok", "band_tv": band, "base_tv": None, "base_check_ok": False,
                "reason": f"the bf16 base checkpoint was not written ({saved.get('reason')})"}
    out = _gen_on(c, a, sc, c["out"] / "merged" / "none", "none", rows)
    base_tv = tv(out["letters"], hf, rows)
    ok = (bool(base_tv <= band + PARITY_MARGIN) or a.dry_run) and not a.dry_fail_base
    if not ok:  # Amendment 20 item 2: nothing on this engine is of record; none goes to HF
        _set_aside(c, "none")
    if not a.dry_run:
        shutil.rmtree(c["out"] / "merged" / "none", ignore_errors=True)
    return {"status": "ok", "band_tv": band, "base_tv": base_tv, "base_check_ok": ok,
            "bar": f"base_tv <= band + {PARITY_MARGIN}",
            "tokens_per_second": out["tokens_per_second"]}


def phase_hf_arm(c, a, rec, arm: str) -> dict:
    mw = load_hf(c, a)
    sc = screen(c, mw, a)
    peft = attach(mw, c["adapters"] / arm)
    rows = parity_rows(sc, c)
    lp, _ = L.read_ids(sc, [r["ids"] for r in rows], 1)
    np.save(c["out"] / "parity" / f"hf_{arm}.npy", lp)
    spec = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                             f"spec/{arm}")
    cells = []
    if arm in panel_arms(rec):
        half = [c["panel"][i] for i in c["half_ids"]]
        cells = [L.run_letter_cell(sc, half, cell, f"panel/{arm}") for cell in L.PANEL_CELLS]
    if not (rec["phases"].get("base_vllm") or {}).get("base_check_ok", True):
        return {"status": "ok", "specificity": spec, "panel_cells": cells,
                "merge": "skipped: the base check failed (every arm generates on HF)"}
    merged = peft.merge_and_unload()
    mw.model = merged
    mw.wb._model = merged  # noqa: SLF001
    lp_m, _ = L.read_ids(sc, [r["ids"] for r in rows], 1)
    np.save(c["out"] / "parity" / f"hf_{arm}_merged.npy", lp_m)
    base = np.load(c["out"] / "parity" / "hf_none.npy")
    shift, lost, to_base = tv(lp, base, rows), tv(lp_m, lp, rows), tv(lp_m, base, rows)
    merge = {"adapter_tv_vs_base": shift, "merged_tv_vs_attached": lost,
             "merged_tv_vs_base": to_base,
             "kept_share": None if shift <= 0 else 1.0 - lost / shift,
             "closer_to_base": bool(lost > to_base), "gates": MERGE_GATE,
             "note": "the bf16 merge on the HF stack vs the adapter attached and vs the base, on "
                     "the parity rows; a merge that kept nothing reads kept_share ~0"}
    if MERGE_GATE and merge["closer_to_base"]:
        saved = {"ok": False, "reason": "merge check: closer to the base than to the adapter"}
    else:
        saved = save_bf16(merged, c["out"] / "merged" / arm)
    return {"status": "ok", "specificity": spec, "panel_cells": cells, "merge_check": merge,
            "saved": saved}


def _set_aside(c: dict, arm: str) -> None:
    for f in (c["out"] / "gen" / f"h1_{arm}.jsonl", c["out"] / "cap" / f"{arm}.jsonl"):
        if f.exists():
            f.rename(f.with_suffix(".parity_failed.jsonl"))


def phase_vllm_arm(c, a, rec, arm: str) -> dict:
    sc = vllm_screen(c, a)
    base = rec["phases"]["base_vllm"]
    if not base["base_check_ok"]:
        shutil.rmtree(c["out"] / "merged" / arm, ignore_errors=True)
        return {"status": "ok", "parity_ok": False, "reason": "base check failed (HF for all)"}
    saved = (rec["phases"].get(f"hf_{arm}") or {}).get("saved") or {}
    if not saved.get("ok"):
        return {"status": "ok", "parity_ok": False,
                "reason": f"no merged checkpoint ({saved.get('reason', 'not saved')}): HF"}
    rows = parity_rows(sc, c)
    out = _gen_on(c, a, sc, c["out"] / "merged" / arm, arm, rows)
    d = tv(out["letters"], np.load(c["out"] / "parity" / f"hf_{arm}.npy"), rows)
    ok = bool(d <= base["band_tv"] + PARITY_MARGIN) or a.dry_run
    if not ok:  # the merged engine's rows are not of record; this adapter goes to HF
        _set_aside(c, arm)
    if not a.dry_run:
        shutil.rmtree(c["out"] / "merged" / arm, ignore_errors=True)
    return {"status": "ok", "parity_tv": d, "band_tv": base["band_tv"], "parity_ok": ok,
            "tokens_per_second": out["tokens_per_second"]}


def phase_hfgen_arm(c, a, rec, arm: str) -> dict:
    """Amendment 18 item 4: an adapter that failed parity generates on HF, adapter attached
    (``none`` likewise after a failed base check): the 38 engaged H1_held scenarios x rr, the
    100-item capability slice, then the other H1_held scenarios if the deadline leaves room. Seeds
    are the vLLM jobs' (paired across arms within an engine)."""
    import screen_vllm as SV
    import steering_gate as SG

    mw = load_hf(c, a)
    sc = screen(c, mw, a)
    if arm != "none":
        attach(mw, c["adapters"] / arm)
    eng = set(c["engaged"])
    first = [s for s in c["held"] if s in eng]
    rest = [s for s in c["held"] if s not in eng]
    parsed: list[dict] = []

    def h1(ids: list[str]) -> float:
        t0 = time.time()
        jobs = L.h1_jobs(sc, c["H"], ids, c["rr"])
        seeds = [j["seed"] for j in SV.worker_jobs(jobs, "t15", c["cap"])]  # vLLM's seeds
        rows = SG.generate(mw, [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"], seeds=seeds,
                           batch=c["batch"], on_finish=SG.jsonl_saver(
                               c["out"] / "gen" / f"h1_{arm}.hf.raw.jsonl"))
        got = L.build_rows(sc, jobs, [r["gen_ids"] for r in rows], seeds, cap=c["cap"],
                           stage="t15", engine="hf_adapter" if arm != "none" else "hf")
        for r in got:
            r["arm"] = arm
        parsed.extend(got)
        sc.save_jsonl(f"gen/h1_{arm}.jsonl", parsed)
        return (time.time() - t0) / max(1, len(jobs))

    per = h1(first)  # ΔP's scenarios first
    half = cap_fallback(c["cap_items"])
    cj, recs = cap_jobs(sc, c, a)
    cj, recs = [cj[i] for i in half], [recs[i] for i in half]
    g = SG.generate(mw, [j["prompt_ids"] for j in cj], fmt=c["fmt"], cap=c["cap"] + CAP_EXTRA,
                    seeds=[j["seed"] for j in cj], batch=c["batch"])
    by = {f"cap|{n}": {"gen_ids": r["gen_ids"], "status": "finished", "n_tokens": r["n_tokens"]}
          for n, r in enumerate(g)}
    items = [c["cap_items"][i] for i in half]
    sc.save_jsonl(f"cap/{arm}.jsonl", score_cap(sc, items, recs, by, arm))
    est = per * len(rest) * c["rr"]
    brake = not a.deadline or time.time() + est < a.deadline
    if brake:
        h1(rest)
    return {"status": "ok", "engine": "hf", "n_h1": len(parsed), "n_capability": len(cj),
            "h1_scope": "all 132 (brake)" if brake else
            f"the 38 engaged only (the other {len(rest)} would take ~{est / 3600:.2f} h)",
            "seconds_per_rollout": per}


def phase_readouts(c, a, rec) -> dict:
    """Amendment 20 item 3: a seeded 2 of 8 rollouts per H1_held scenario, for none and each
    adapter; the anchor-axis margin over the clause_end positions at block 12 and the
    post-reasoning letter."""
    import steering_gate as SG
    import t11_driver as T11

    mw = load_hf(c, a)
    sc = screen(c, mw, a)
    punct = SG.punct_mask(c["tok"])
    axis = c["axis"].astype(np.float32)
    if a.dry_run:
        axis = np.random.default_rng(0).standard_normal(mw.model.config.hidden_size).astype(
            np.float32)
    axis /= np.linalg.norm(axis)
    pick = {sid: sorted(random.Random(f"{SEED}|{sid}").sample(range(c["rr"]), min(2, c["rr"])))
            for sid in c["held"]}
    # a dry run's stub traces carry no punctuation; onset positions stand in so the forward runs
    classes = ("clause_end", "onset") if a.dry_run else ("clause_end",)
    out = {}
    for arm in ["none"] + arms_order(rec):
        f = c["out"] / "gen" / f"h1_{arm}.jsonl"
        if not f.exists():
            out[arm] = {"skipped": "no H1_held rows of record"}
            continue
        peft = attach(mw, c["adapters"] / arm) if arm != "none" else None
        res = []
        for r in _jsonl(f):
            if r["rollout"] not in pick.get(r["scenario_id"], []) or not r.get("gen_ids"):
                continue
            j = L.h1_jobs(sc, c["H"], [r["scenario_id"]], c["rr"])[r["rollout"]]
            if hashlib.sha256(json.dumps(j["ids"]).encode()).hexdigest() != r[
                    "prompt_ids_sha256"]:
                raise SystemExit(f"readouts: {arm} {r['scenario_id']}#{r['rollout']} prompt "
                                 "differs from the generated one")
            n_p, gen = len(j["ids"]), r["gen_ids"]
            cp = T11.class_positions(c["fmt"], c["tok"], gen, punct)
            pos = [n_p + k for k, v in cp.items() if any(x in v for x in classes)]
            dec = (n_p + r["answer_idx"] - 1) if r.get("token_identity") else None
            if not pos:
                continue
            resid, lp, _ = T11.forward_bank(sc, j["ids"] + gen, pos, dec)
            h = resid[:, 0, :].astype(np.float32)
            m = float(((h @ axis) / np.linalg.norm(h, axis=-1).clip(1e-6)).mean())
            res.append({"scenario_id": r["scenario_id"], "rollout": r["rollout"],
                        "margin_anchor_axis": m, "n_positions": len(pos),
                        "norm_status": r.get("norm_status"),
                        "post_letter_logp": None if lp is None else [float(x) for x in lp]})
        sc.save_jsonl(f"readouts/{arm}.jsonl", res)
        out[arm] = {"n": len(res), "margin_mean": float(np.mean([x["margin_anchor_axis"]
                                                                for x in res])) if res else None}
        if peft is not None:
            mw.model = peft.unload()
            mw.wb._model = mw.model  # noqa: SLF001
    return {"status": "ok", "arms": out}


def run_phase(c, a, name: str) -> int:
    rec = load_record(c)
    done = rec["phases"].get(name) or {}
    if done.get("finished") and not a.force:
        print(f">> t15 {name}: finished earlier ({done['finished']}); skipped", flush=True)
        return 0
    t0 = time.time()
    if name in ("beh_cap", "base_hf", "base_vllm", "readouts"):
        res = {"beh_cap": phase_beh_cap, "base_hf": phase_base_hf, "base_vllm": phase_base_vllm,
               "readouts": phase_readouts}[name](c, a, rec)
    else:
        kind, arm = name.split("_", 1)
        res = {"hf": phase_hf_arm, "vllm": phase_vllm_arm, "hfgen": phase_hfgen_arm}[kind](
            c, a, rec, arm)
    rec = load_record(c)
    rec["phases"][name] = {**res, "seconds": time.time() - t0,
                           "finished": _now() if res.get("status") == "ok" else None}
    save_record(c, rec)
    brief = {k: res[k] for k in ("comparator", "reading", "truncation", "band_tv", "base_tv",
                                 "base_check_ok", "parity_tv", "parity_ok", "merge_check")
             if k in res}
    print(f">> t15 {name}: done ({json.dumps(brief)})", flush=True)
    return 0


def plan(rec: dict) -> list[str]:
    """The phase list from the record (the comparator decides the arm order; a parity failure
    adds that arm's HF generation)."""
    names = ["beh_cap", "base_hf", "base_vllm"]
    b = rec["phases"].get("base_vllm") or {}
    if b.get("finished") and not b.get("base_check_ok"):
        names.append("hfgen_none")
    if (rec["phases"].get("beh_cap") or {}).get("finished"):
        for arm in arms_order(rec):
            names += [f"hf_{arm}", f"vllm_{arm}"]
            v = rec["phases"].get(f"vllm_{arm}") or {}
            if v.get("finished") and not v.get("parity_ok"):
                names.append(f"hfgen_{arm}")
        names.append("readouts")
    return names


def verify_adapters(c: dict) -> None:
    """The adapters on the pod are the T1.4 adapters of record (``data/t15_adapters_v1.json``)."""
    want = json.loads((DATA / "t15_adapters_v1.json").read_text())["arms"]
    for arm, files in want.items():
        for f, sha in files.items():
            got = hashlib.sha256((c["adapters"] / arm / f).read_bytes()).hexdigest()
            if got != sha:
                raise SystemExit(f"adapter {arm}/{f}: sha256 {got[:12]} != {sha[:12]}")
    print(f">> t15 adapters: {len(want)} verified", flush=True)


def run_all(c, a) -> int:
    """Each phase in its own process (the HF model freed before any vLLM engine starts)."""
    if not a.dry_run:
        verify_adapters(c)
    i = 0
    while True:
        names = plan(load_record(c))
        todo = [n for n in names if not (load_record(c)["phases"].get(n) or {}).get("finished")]
        if not todo:
            return 0
        name = todo[0]
        if a.deadline and time.time() > a.deadline:
            print(f">> t15 {name}: not started, past the deadline", flush=True)
            return 3
        cmd = [sys.executable, __file__, "--phase", name] + _passthrough(a)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f">> t15 {name}: exit {rc}; stopping", flush=True)
            return rc
        i += 1
        if i > 40:
            raise SystemExit("phase loop did not converge")


def _passthrough(a) -> list[str]:
    out = ["--out", str(a.out), "--t14-inputs", str(a.t14_inputs), "--adapters",
           str(a.adapters), "--date-pin", a.date_pin]
    if a.dry_run:
        out.append("--dry-run")
    if a.dry_fail_base:
        out.append("--dry-fail-base")
    if a.deadline:
        out += ["--deadline", str(a.deadline)]
    if a.force:
        out.append("--force")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t14-inputs", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, required=True, help="T1.4's adapters directory")
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-fail-base", action="store_true",
                    help="dry run only: fail the base check, so every arm takes the HF path")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.dry_fail_base and not a.dry_run:
        ap.error("--dry-fail-base is a dry-run flag")
    c = context(a)
    if a.phase == "all":
        return run_all(c, a)
    return run_phase(c, a, a.phase)


if __name__ == "__main__":
    sys.exit(main())
