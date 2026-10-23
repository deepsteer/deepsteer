"""The D_rb rider (ST_PLAN Amendments 34-35) on GPT-OSS-20B, after the format x harm matrix on its
pod: base traces teacher-forced under ``none`` and adapters, the residual at T1.1's position
classes saved per trace. Each step its own process:

- ``gen`` (vLLM): set (a), the 60 specificity items x 4 base rollouts in H1's agent frame at the
  effort of record, cap 1,536 (L-beh-2's ``rr_gen`` path, its date pin), stage ``st2drb_spec``.
- ``prefill --arm A`` (HF, single-row): ``none`` and ``beh`` read set (a), set (b) (L-beh-2's 1,040
  pool rollouts) and set (c) (T1.1's rollouts 0-3 on the 38 engaged H1_held); ``beh2``, ``anchor``
  and ``beh_epoch3`` read set (a) only (the positive control). Per trace: the count and mean of
  the residual over each class (``clause_end``, ``onset``, ``stride32``; T1.1's
  ``class_positions``, the bank's definition) at hidden 13, and the clause_end mean at hidden 7,
  10, 13, 16, 19. ``none`` also saves a 4,096-position clause_end sample at hidden 13 and
  prefills 8 traces twice (determinism).

The analysis of record is local (``analyze_st2drb.py``).

    python3 st2drb_driver.py --step gen --out <dir> [--dry-run]
    python3 st2drb_driver.py --step prefill --arm none|beh|beh2|anchor|beh_epoch3 --out <dir> \
        --adapters <dir> --pool <rr_rollouts.jsonl> --t11 <h1_rollouts.jsonl> [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402

KEY = "gpt_oss_20b"
DATA = HERE.parent / "data"
R_SPEC, R_H1 = 4, 4
CLASSES = ("clause_end", "onset", "stride32")
HIDDEN = (7, 10, 13, 16, 19)
H_STAR = 13
N_SAMPLE, N_DET = 4096, 8
ARMS = ("none", "beh", "beh2", "anchor", "beh_epoch3")
ALL_SETS = ("spec", "pool", "h1")  # none and beh; the other arms read spec only
MANIFESTS = {"beh": "t15_adapters_v1.json", "anchor": "t15_adapters_v1.json",
             "beh_epoch3": "t15_adapters_v1.json", "beh2": "st_fxh_adapters_v1.json"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def update_record(out: Path, step: str, res: dict) -> None:
    p = out / KEY / "st2drb_record.json"
    rec = json.loads(p.read_text()) if p.exists() else {"model": KEY, "steps": {}}
    rec["steps"][step] = {**res, "finished": _now()}
    p.write_text(json.dumps(rec, indent=1, default=L._json_default))  # noqa: SLF001


def spec_scenarios(dry: bool) -> list:
    from deepsteer.kdg.schema import load_scenario_dir

    ids = json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"]
    allp, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                     "*_scenarios_*.json"))))
    by = {s.id: s for s in L.panel_586(allp)}
    return [by[i] for i in (ids[:2] if dry else ids)]


def step_gen(a) -> dict:
    import pod_st_screen as P
    import screen_vllm as SV
    from reasoning_formats import build_format
    from screen_stub import ScreenStub
    from transformers import AutoTokenizer

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    model = ScreenStub(KEY, tok, fmt, spec["n_layers"]) if a.dry_run else L.tok_only(tok)
    sc = L.Screen(KEY, spec, model, fmt, a.out / KEY, a.date_pin, dry=a.dry_run)
    sc.effort = spec.get("effort_of_record") or fmt.effort_ladder[0]
    items = spec_scenarios(a.dry_run)
    jobs = L.h1_jobs(sc, {s.id: s for s in items}, [s.id for s in items], R_SPEC)
    g = SV.gen_stage(sc, jobs, "st2drb_spec", L.CAP_OF_RECORD, spec, probe=True,
                     deadline=a.deadline)
    for r, j in zip(g["rows"], jobs):
        r.update(prompt_ids=j["ids"], date_pin=a.date_pin, sample="drb_spec")
    sc.save_jsonl("drb/spec_rollouts.jsonl", g["rows"])
    return {"status": "ok", "n": len(jobs), "n_items": len(items), "timing": g["timing"],
            "summary": L.screen_summary(g["rows"])}


def traces(a, sets: tuple[str, ...]) -> list[dict]:
    """Every trace an arm reads, keyed: ``spec|sid|k``, ``pool|sid|k``, ``h1|sid|<T1.1 row>``."""
    out = []
    if "spec" in sets:
        borrow = _jsonl(a.pool)[0]["gen_ids"] if a.dry_run else None
        for r in _jsonl(a.out / KEY / "drb" / "spec_rollouts.jsonl"):
            if borrow is not None:  # the stub writes no harmony trace: a real one, so the dry
                r = {**r, "gen_ids": borrow}  # run reaches every arm's read on set (a)
            out.append({"key": f"spec|{r['scenario_id']}|{r['rollout']}", "set": "spec", **r})
    if "pool" in sets:
        rows = _jsonl(a.pool)
        rows = [r for r in rows if r["rollout"] < 2][:4] if a.dry_run else rows  # 2 items x 2
        out += [{"key": f"pool|{r['scenario_id']}|{r['rollout']}", "set": "pool", **r}
                for r in rows]
    if "h1" in sets:
        eng = set(json.loads((DATA / "t15_engaged_held_v1.json").read_text())["ids"])
        rows = [(i, r) for i, r in enumerate(_jsonl(a.t11))
                if r["scenario_id"] in eng and r["rollout"] < R_H1]
        rows = [(i, r) for i, r in rows if r["rollout"] < 2][:4] if a.dry_run else rows
        out += [{"key": f"h1|{r['scenario_id']}|{i}", "set": "h1", **r} for i, r in rows]
    return [t for t in out if t.get("gen_ids")]


def verify(adapters: Path, arm: str) -> dict:
    want = json.loads((DATA / MANIFESTS[arm]).read_text())["arms"][arm]
    got = {f: hashlib.sha256((adapters / arm / f).read_bytes()).hexdigest() for f in want}
    if got != want:
        raise SystemExit(f"adapter {arm}: sha256 differs from {MANIFESTS[arm]}")
    return got


def load(a):
    import pod_st_screen as P
    import torch
    from kdg_pod_lib import ModelWrapper

    spec = P.registry()[KEY]
    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        from deepsteer.steering import objective_lora as OL

        path = a.out / "_tiny"
        if not (path / "config.json").exists():
            st_tiny.build(KEY, path)
        mw = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        for arm in ARMS[1:]:  # zero-initialized adapters at the tiny width
            d = a.adapters / arm
            if not (d / "adapter_config.json").exists():
                OL.inject_lora(mw.wb, rank=4, dropout=0.0)
                mw.wb.model.save_pretrained(str(d))
                mw.wb._model = mw.wb.model.unload()  # noqa: SLF001
                mw.model = mw.wb.model
        return spec, mw
    return spec, ModelWrapper(spec["repo"], spec["revision"], torch_dtype=torch.bfloat16,
                              attn_implementation="eager", **P.hf_load_kwargs(spec))


def blocks_for(n_layers_tiny: int | None) -> tuple[int, ...]:
    """Block indices (hidden index - 1); on the tiny model every hidden index maps onto its few
    blocks so the dry run reads the same code path."""
    if n_layers_tiny is None:
        return tuple(h - 1 for h in HIDDEN)
    return tuple(min(h - 1, n_layers_tiny - 1) for h in HIDDEN)


def read_trace(mw, fmt, tok, punct, t: dict, blocks) -> dict | None:
    import st2m_driver as S2M
    import t11_driver as T11

    gen, n_p = t["gen_ids"], len(t["prompt_ids"])
    cp = T11.class_positions(fmt, tok, gen, punct)
    pos = {c: [n_p + i for i, cl in cp.items() if c in cl] for c in CLASSES}
    if not pos["clause_end"]:
        return None
    every = sorted({p for v in pos.values() for p in v})
    h = S2M.resid_at(mw, t["prompt_ids"] + gen, every, sorted(set(blocks)))
    ix = {p: j for j, p in enumerate(every)}
    hs = blocks[HIDDEN.index(H_STAR)]
    mean_cls = np.zeros((len(CLASSES), h[hs].shape[1]), np.float32)
    for k, c in enumerate(CLASSES):
        if pos[c]:
            mean_cls[k] = h[hs][[ix[p] for p in pos[c]]].mean(0)
    ce = [ix[p] for p in pos["clause_end"]]
    return {"counts": [len(pos[c]) for c in CLASSES], "mean_cls_h13": mean_cls,
            "mean_ce_layers": np.stack([h[b][ce].mean(0) for b in blocks]),
            "ce_h13": h[hs][ce]}


def step_prefill(a) -> dict:
    import steering_gate as SG
    from peft import PeftModel
    from reasoning_formats import build_format

    spec, mw = load(a)
    shas = None
    if a.arm != "none":
        shas = None if a.dry_run else verify(a.adapters, a.arm)
        PeftModel.from_pretrained(mw.model, str(a.adapters / a.arm))  # in place, as T1.5
    fmt = build_format(KEY, mw.tok, spec)
    punct = SG.punct_mask(mw.tok)
    blocks = blocks_for(mw.model.config.num_hidden_layers if a.dry_run else None)
    ts = traces(a, ALL_SETS if a.arm in ("none", "beh") else ("spec",))
    keys, sets, counts, mcls, mce, t_read, skipped = [], [], [], [], [], [], []
    rng = random.Random(f"st2drb|{a.arm}")
    sample: list[np.ndarray] = []
    stopped = None
    for t in ts:
        if a.deadline and time.time() > a.deadline:
            stopped = f"deadline after {len(keys)} traces"
            break
        t0 = time.time()
        r = read_trace(mw, fmt, mw.tok, punct, t, blocks)
        t_read.append(time.time() - t0)
        if r is None:
            skipped.append(t["key"])
            continue
        keys.append(t["key"])
        sets.append(t["set"])
        counts.append(r["counts"])
        mcls.append(r["mean_cls_h13"])
        mce.append(r["mean_ce_layers"].astype(np.float16))
        if a.arm == "none":
            k = min(4, len(r["ce_h13"]))
            sample.append(r["ce_h13"][sorted(rng.sample(range(len(r["ce_h13"])), k))])
    d = a.out / KEY / "drb"
    d.mkdir(parents=True, exist_ok=True)
    extra = {}
    det = None
    if a.arm == "none" and keys:
        s = np.concatenate(sample)
        pick = sorted(rng.sample(range(len(s)), min(N_SAMPLE, len(s))))
        extra["ce_sample_h13"] = s[pick].astype(np.float16)
        again = [read_trace(mw, fmt, mw.tok, punct, t, blocks)
                 for t in [x for x in ts if x["key"] in set(keys[:N_DET])]]
        det = float(max(np.abs(g["mean_cls_h13"] - mcls[i]).max()
                        for i, g in enumerate(again) if g is not None))
    np.savez_compressed(d / f"{a.arm}.npz", keys=np.array(keys), sets=np.array(sets),
                        counts=np.array(counts, np.int32), classes=np.array(CLASSES),
                        hidden=np.array(HIDDEN), mean_cls_h13=np.array(mcls, np.float32),
                        mean_ce_layers=np.array(mce, np.float16), **extra)
    return {"status": "ok" if stopped is None else "stopped", "stopped": stopped, "arm": a.arm,
            "n": len(keys), "n_by_set": {x: sets.count(x) for x in set(sets)},
            "skipped_no_clause_end": skipped, "adapter_sha256": shas,
            "determinism_max_abs": det,
            "read_seconds_per_trace": float(np.mean(t_read)) if t_read else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=("gen", "prefill"))
    ap.add_argument("--arm", choices=ARMS, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, default=None)
    ap.add_argument("--pool", type=Path, default=None, help="L-beh-2's gen/rr_rollouts.jsonl")
    ap.add_argument("--t11", type=Path, default=None, help="T1.1's gen/h1_rollouts.jsonl")
    ap.add_argument("--date-pin", default="2026-10-23")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    if a.step == "prefill" and not (a.arm and a.adapters and a.pool and a.t11):
        ap.error("prefill needs --arm, --adapters, --pool and --t11")
    (a.out / KEY).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = step_gen(a) if a.step == "gen" else step_prefill(a)
    name = a.step if a.step == "gen" else f"prefill_{a.arm}"
    update_record(a.out, name, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "n_items", "n_by_set", "determinism_max_abs", "stopped",
                                 "status") if k in res}
    print(f">> st2drb {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 3


if __name__ == "__main__":
    sys.exit(main())
