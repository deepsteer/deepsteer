"""ST Phase 2 Cell 4, Muse's transfer test (ST_PLAN Amendments 29-30; author, 2026-10-22). Two
processes: ``--phases v1`` (vLLM) first, then the HF phases in one process on the ST stack of record
(torch 2.13 in the vLLM venv), resumable phase by phase. Every quantity the arms use is derived on
this pod by Amendment 30's rules, in this order:

- ``v1`` (vLLM): H1 264 x 8 at effort medium, cap 1,536, KDG sampler v2 (T1.1's design).
- ``dose0``: the dose-0 letter readout on H1_train 132 x 8 (``empty_reasoning``), single-row.
- ``bank``: per v1 rollout one single-row teacher-forced forward; ``clause_end`` residual samples
  (up to 8 per rollout) at blocks 20/26/32 and per-rollout ``clause_end`` sums at block 26.
- ``derive``: the engaged H1_train set (T1.1's rule), five covariance-matched fill draws from the
  block-26 samples, the median norm there, the 5 sweep scenarios (T1.3's seeded rule), and the
  descriptive participation ratio and median norm at blocks 20/26/32.
- ``pilot``: one batch of 32 ``none`` rollouts at the cap; sets the batch (largest multiple of 16 up
  to 160 projected under 76 GB) and re-prices; arms that do not fit are bailed in Amendment 30's
  order (``care_harm``, then -V_harm, then V_qwen).
- ``directions``: V_qwen re-encoded in Muse's reasoning channel, -V_harm by Paper 7's recipe,
  ``care_harm`` by W4's recipe, all at block 26, unit.
- ``sweep``: V_qwen at alpha in {0.05, 0.1, 0.2, 0.4} x the median norm on the 5 sweep scenarios x
  8 against a ``none`` on them; the largest coherent alpha is every arm's.
- ``spec_*``: the 60 rule-right items x 8 at dose 0, the vector at every input position at block
  26 (``none``, ``fill`` with draw = item index mod 5, then each direction), all before any arm.
- ``arm_*``: the engaged set x 8 at ``clause_end``, block 26, seeds shared across arms: ``none``,
  ``fill``, V_qwen, -V_harm, ``care_harm``; an arm failing coherence on length reruns once at
  alpha / 2 with its specificity read at alpha / 2 (Amendment 25).

The analysis of record is local (``analyze_st2m.py``).

    python3 st2m_driver.py --out <dir> --phases v1 [--dry-run]
    python3 st2m_driver.py --out <dir> --qwen-topup <c3_qwen_topup.jsonl> \
        --qwen-screen <h1_screen.jsonl> [--phases ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
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
import st2_driver as S2  # noqa: E402
import st2c_driver as C  # noqa: E402
import st2dl_driver as DL  # noqa: E402
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

KEY = "muse_glimmer_30b"
DATA = HERE.parent / "data"
BLOCK = 26  # Amendment 30 item 1: GPT-OSS's block 12 at the same relative depth, sliding attention
BANK_BLOCKS = (20, 26, 32)  # descriptive PR and norms
CLS = "clause_end"
R = 8
KEEP = 8  # clause_end samples per rollout
ALPHA_FACTORS = (0.05, 0.1, 0.2, 0.4)
SWEEP_SEED, N_SWEEP = 0, 5  # T1.3's draw rule (build_t13_draw.SEED, N_SWEEP)
PILOT_N = 32
# the arms' scenario cap (None = every engaged scenario; an integer = a seeded draw of that many,
# the sweep then drawn from the rest, disjoint as in T1.3): the author's decision before launch
ARM_CAP: int | None = None
MEM_CAP = 76e9
DIRS = ("V_qwen", "neg_V_harm", "care_harm")
BAIL = ("arm_care_harm", "arm_neg_V_harm", "arm_V_qwen")  # Amendment 30 item 4, first cut first
HF_PHASES = (["dose0", "bank", "derive", "pilot", "directions", "sweep", "spec_none",
              "spec_fill"] + [f"spec_{d}" for d in DIRS]
             + ["arm_none", "arm_fill"] + [f"arm_{d}" for d in DIRS])
PAPER7_N = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def context(a) -> dict:
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
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    order = L.h1_train_order(split)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept}, "order": order,
         "ids": list(order) + sorted(split["H1_held"]), "train": set(split["H1_train"]),
         "panel": {s.id: s for s in L.panel_586(allp)},
         "spec_ids": json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"],
         "layer": BLOCK + 1, "bank_blocks": BANK_BLOCKS, "cls": CLS, "cap": L.CAP_OF_RECORD,
         "batch": 160, "r": R, "pilot_n": PILOT_N, "paper7_n": PAPER7_N, "care_max": None,
         "record": out / "st2m_record.json", "qwen_topup": a.qwen_topup,
         "qwen_screen": a.qwen_screen}
    if a.dry_run:  # a tiny Muse (4 text layers) with the real tokenizer; onset, so hooks fire
        c.update(ids=list(order[:6]) + sorted(split["H1_held"])[:2], layer=2,
                 bank_blocks=(0, 1, 2), cls="onset", cap=24, batch=8, r=2, pilot_n=4,
                 spec_ids=c["spec_ids"][:2], paper7_n=3, care_max=3)
    return c


def load_record(c: dict) -> dict:
    return json.loads(c["record"].read_text()) if c["record"].exists() else {"model": KEY,
                                                                            "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    c["record"].write_text(json.dumps(rec, indent=1, default=L._json_default))  # noqa: SLF001


def phase_done(c: dict, name: str) -> dict:
    return load_record(c)["phases"].get(name) or {}


def date_pin(c: dict, a) -> str:
    """Every HF phase renders with v1's date pin (Muse's template carries the date)."""
    p = c["out"] / "gen" / "h1_rollouts.jsonl"
    if not p.exists():
        return a.date_pin
    pins = {r.get("date_pin") for r in _jsonl(p)} - {None}
    if len(pins) > 1:
        raise SystemExit(f"v1 rows carry {len(pins)} date pins")
    return pins.pop() if pins else a.date_pin


def screen(c: dict, model, pin: str, dry: bool) -> L.Screen:
    """Single-row readouts of record (Muse: rider pod 2's mode rule), the dose-0 candidate of
    record, effort medium."""
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], pin, dry=False, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or "medium"
    sc.unbatched = True
    sc.layers_override = [c["layer"]]
    sc.dose0 = c["fmt"].dose0[0]
    return sc


# ---- v1 (vLLM) -----------------------------------------------------------------------------------
def phase_v1(c: dict, a) -> dict:
    import screen_vllm as SV
    from screen_stub import ScreenStub

    model = (ScreenStub(KEY, c["tok"], c["fmt"], c["spec"]["n_layers"]) if a.dry_run
             else L.tok_only(c["tok"]))
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, dry=a.dry_run)
    sc.effort = c["spec"].get("effort_of_record") or "medium"
    jobs = L.h1_jobs(sc, c["H"], c["ids"], R)
    t0 = time.time()
    g = SV.gen_stage(sc, jobs, "st2m", L.CAP_OF_RECORD, c["spec"], probe=True,
                     deadline=a.deadline)
    for r, j in zip(g["rows"], jobs):
        r.update(sample="selection", arm="none", rollout_key=f"st2m_v1|{r['scenario_id']}|"
                 f"{r['rollout']}", split="train" if r["scenario_id"] in c["train"] else "held",
                 prompt_ids=j["ids"], date_pin=a.date_pin)
    sc.save_jsonl("gen/h1_rollouts.jsonl", g["rows"])
    return {"status": "ok", "n": len(jobs), "seconds": time.time() - t0, "timing": g["timing"],
            "probe": g["probe"], "effort": sc.effort, "summary": L.screen_summary(g["rows"])}


# ---- HF ------------------------------------------------------------------------------------------
def load_model(c: dict, a):
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = c["out"] / "_tiny"
        if not (path / "config.json").exists():
            st_tiny.build(KEY, path)
        m = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        fmt = c["fmt"]  # a random model never writes Muse's channel header: open at once
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        return m
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        **c["P"].hf_load_kwargs(c["spec"]))


def phase_dose0(c, sc, model, a) -> dict:
    tr = [c["H"][s] for s in c["ids"] if s in c["train"]]
    res = L.run_letter_cell(sc, tr, "dl_chat_neutral", "readout/dose0_h1_train")
    rows = _jsonl(Path(res["path"]))
    pd = c["P"].dose0_protect(rows)
    return {"status": "ok", **res, "P_dose0_mean": float(np.mean(list(pd.values()))),
            "candidate": sc.dose0.name}


def resid_at(model, ids: list[int], positions: list[int], blocks) -> dict[int, np.ndarray]:
    """One single-row forward; block outputs at ``positions`` (float32)."""
    import torch

    grab: dict = {}
    hooks = [model.wb._get_layer_module(b).register_forward_hook(  # noqa: SLF001
        lambda _m, _i, o, b=b: grab.__setitem__(b, (o[0] if isinstance(o, tuple) else o)[0]))
        for b in blocks]
    try:
        with torch.no_grad():
            model.model(input_ids=torch.tensor([list(ids)], device=model.model.device),
                        logits_to_keep=1)
    finally:
        for h in hooks:
            h.remove()
    return {b: grab[b][positions].float().cpu().numpy() for b in blocks}


def phase_bank(c, sc, model, a) -> dict:
    rows = _jsonl(c["out"] / "gen" / "h1_rollouts.jsonl")
    punct = SG.punct_mask(c["tok"])
    rng = random.Random(0)
    blocks, rec_block = [b for b in c["bank_blocks"]], c["layer"] - 1
    samples = {b: [] for b in blocks}
    sums, counts, keys, t_read = [], [], [], []
    for r in rows:
        gen = r.get("gen_ids") or []
        if not gen:
            continue
        n_p = len(r["prompt_ids"])
        ce = S2.trace_positions(c, gen, n_p, punct)[c["cls"]]
        if not ce:
            continue
        t0 = time.time()
        h = resid_at(model, r["prompt_ids"] + gen, ce, sorted(set(blocks) | {rec_block}))
        t_read.append(time.time() - t0)
        pick = sorted(rng.sample(range(len(ce)), min(KEEP, len(ce))))
        for b in blocks:
            samples[b].append(h[b][pick])
        sums.append(h[rec_block].sum(0).astype(np.float64))
        counts.append(len(ce))
        keys.append(r["rollout_key"])
    (c["out"] / "bank").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(c["out"] / "bank" / "clause_end.npz", keys=np.array(keys),
                        sums=np.array(sums), counts=np.array(counts),
                        **{f"sample_b{b}": np.concatenate(samples[b]).astype(np.float16)
                           for b in blocks})
    return {"status": "ok", "n_rollouts": len(keys), "n_positions": int(sum(counts)),
            "blocks": blocks,
            "read_seconds_per_rollout": float(np.mean(t_read)) if t_read else None}


def participation_ratio(x: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False) if x.shape[0] > x.shape[1]
                                    else (x @ x.T) / max(x.shape[0] - 1, 1)), 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))


def phase_derive(c, sc, model, a) -> dict:
    rows = _jsonl(c["out"] / "gen" / "h1_rollouts.jsonl")
    d0 = _jsonl(c["out"] / "readout" / "dose0_h1_train" / "dl_chat_neutral.jsonl")
    pd = c["P"].dose0_protect(d0)
    tr = [r for r in rows if r["scenario_id"] in c["train"] and r.get("norm_status") is not None]
    if any(v >= 0.5 for v in pd.values()):
        eng = L.engaged_subset(tr, pd, use_dose0=True)
    elif a.dry_run:  # a random tiny model has no dose-0 regime; the fallback below engages
        eng = {"n_screened_parsed": 0, "n_engaged": 0, "engaged_ids": [], "rule": "dry run"}
    else:
        raise SystemExit("no H1_train scenario at P_dose0 >= 0.5: T1.1's engaged rule is empty")
    eng["brake"] = c["P"].brake([r for r in rows if r["scenario_id"] in c["train"]], d0)
    z = np.load(c["out"] / "bank" / "clause_end.npz")
    rec_b = c["layer"] - 1
    s = z[f"sample_b{rec_b}"].astype(np.float64)
    med = float(np.median(np.linalg.norm(s, axis=1)))
    fill = SG.fill_draws(s, 1.0, 5, 0)
    desc = {}
    for b in c["bank_blocks"]:
        x = z[f"sample_b{b}"].astype(np.float64)
        desc[str(b)] = {"participation_ratio": participation_ratio(x),
                        "median_norm": float(np.median(np.linalg.norm(x, axis=1))),
                        "n_samples": int(len(x))}
    ids = sorted(eng["engaged_ids"])
    if a.dry_run and not ids:  # a random tiny model engages nothing: the first train scenarios
        ids = sorted({r["scenario_id"] for r in tr})[:2]
    arms_ids, pool = list(ids), list(ids)
    if ARM_CAP and len(ids) > ARM_CAP:
        drawn = list(ids)
        random.Random(f"{SWEEP_SEED}|arms").shuffle(drawn)
        arms_ids, pool = sorted(drawn[:ARM_CAP]), sorted(drawn[ARM_CAP:])
    random.Random(f"{SWEEP_SEED}|sweep").shuffle(pool)
    sweep = sorted(pool[:N_SWEEP])
    np.savez(c["out"] / "derive.npz", fill_unit=fill.astype(np.float32))
    (c["out"] / "select").mkdir(exist_ok=True)
    (c["out"] / "select" / "engaged.json").write_text(json.dumps(eng, indent=1,
                                                                 default=L._json_default))
    return {"status": "ok", "n_engaged": eng["n_engaged"], "engaged_ids": arms_ids,
            "engaged_all": ids, "arm_cap": ARM_CAP,
            "null_no_brake": eng.get("null_no_brake"), "brake_B": eng["brake"]["B"],
            "median_norm": med, "alphas": [f * med for f in ALPHA_FACTORS],
            "sweep": sweep, "layers_descriptive": desc,
            "rule": "engaged by T1.1's rule; fill_draws seed 0; sweep T1.3's rule (seed 0)"}


def ctx_derived(c: dict) -> None:
    """Load the derived quantities into ``c`` (each HF process re-reads them)."""
    d = phase_done(c, "derive")
    if not d:
        return
    c["engaged"] = d["engaged_ids"]
    c["sweep"] = d["sweep"]
    c["alphas"] = d["alphas"]
    c["inp"] = {"fill_unit": np.load(c["out"] / "derive.npz")["fill_unit"]}
    p = phase_done(c, "pilot")
    if p.get("batch"):
        c["batch"] = p["batch"]
    sw = phase_done(c, "sweep")
    if sw.get("alpha"):
        c["alpha"] = sw["alpha"]
    dz = c["out"] / "directions.npz"
    if dz.exists():
        z = np.load(dz)
        c["dirs"] = {k: z[k] for k in z.files}


def phase_pilot(c, sc, model, a) -> dict:
    import torch

    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r"])[: c["pilot_n"]]
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    w = torch.cuda.memory_allocated() if cuda else 0
    t0 = time.time()
    rows = SG.generate(model, [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"],
                       seeds=S2.seeds_for("st2m_pilot", jobs), batch=len(jobs))
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if cuda else 0
    longest = max(len(j["ids"]) for j in jobs) + max(r["n_tokens"] for r in rows)
    worst = max(len(j["ids"]) for j in jobs) + c["cap"]
    per_row = (peak - w) / len(jobs) * worst / max(longest, 1) if cuda else 0.0
    batch = c["batch"]
    if cuda:
        fits = [b for b in range(16, 161, 16) if w + per_row * b < MEM_CAP]
        batch = max(fits) if fits else 16
    per_rollout = secs / len(jobs)  # conservative: per-rollout cost at batch 32, not amortized
    return {"status": "ok", "n": len(jobs), "seconds": secs, "batch": batch,
            "weights_bytes": w, "peak_bytes": peak, "per_row_bytes_at_cap": per_row,
            "seconds_per_rollout": per_rollout, "max_tokens": max(r["n_tokens"] for r in rows)}


def reprice(c: dict, a) -> dict:
    """After the pilot: hours left by phase at the pilot's (conservative) rates; arms are bailed in
    Amendment 30's order until the rest fits before the deadline."""
    p, b = phase_done(c, "pilot"), phase_done(c, "bank")
    t, read = p["seconds_per_rollout"], b.get("read_seconds_per_rollout") or 0.0
    d0 = phase_done(c, "dose0").get("seconds_per_row") or 0.1
    n_arm = len(c["engaged"]) * c["r"]
    est = {"directions": (700 + 2 * c["paper7_n"]) * read + 0.05 * 3600,
           "sweep": N_SWEEP * c["r"] * (1 + len(ALPHA_FACTORS)) * (t + read),
           **{f"spec_{k}": len(c["spec_ids"]) * L.N_PERM * d0 for k in ("none", "fill", *DIRS)},
           **{f"arm_{k}": n_arm * (t + read) for k in ("none", "fill", *DIRS)}}
    left = (a.deadline - time.time()) if a.deadline else float("inf")
    bailed = []
    for arm in BAIL:
        if sum(v for k, v in est.items() if k not in bailed) <= left:
            break
        bailed.append(arm)
    return {"estimate_hours": {k: v / 3600 for k, v in est.items()},
            "left_hours": left / 3600 if left != float("inf") else None, "bailed": bailed}


def last_token_index(sc, texts: list[str]) -> tuple[list[list[int]], list[int]]:
    """Paper 7's t_inst on Muse's render: the rendered ids, and the index of each request's last
    token (the render's length minus the template's post-instruction suffix, measured as the
    common suffix of the renders of two probe requests)."""
    def ids_of(t: str) -> list[int]:
        return sc.fmt.prompt_ids(sc.tok, sc.render([{"role": "user", "content": t}]))
    a_, b_ = ids_of("Name a colour of the sky"), ids_of("List three prime numbers")
    k = 0
    while k < min(len(a_), len(b_)) and a_[-1 - k] == b_[-1 - k]:
        k += 1
    out = [ids_of(t) for t in texts]
    return out, [len(x) - k - 1 for x in out]


def phase_directions(c, sc, model, a) -> dict:
    """V_qwen re-encoded in Muse, -V_harm (Paper 7's recipe) and care_harm (W4's recipe), at the
    block of record, unit (Amendment 30 item 2(f))."""
    from deepsteer.datasets import load_moral_probing_v2
    from deepsteer.directions import extraction as du

    q = S2.qwen_context(c, a)
    punct = SG.punct_mask(c["tok"])
    src = [p for p in (c["qwen_topup"], c["qwen_screen"]) if p]
    rows = [r for f in src for r in _jsonl(Path(f))
            if r.get("norm_status") in ("violating", "consistent") and r["scenario_id"] in c["H"]]
    if a.dry_run:  # the stub's letters are arbitrary: label alternately so both actions exist
        rows = rows[:8]
        for i, r in enumerate(rows):
            r["norm_status"] = ("violating", "consistent")[i % 2]
    keys, sums, counts, acts = [], [], [], []
    for r in rows:
        s, k = c["H"][r["scenario_id"]], r["rollout"]
        if L.letter_map(L.assign_letters(s, k)) != r["order"]:
            raise SystemExit(f"{r['scenario_id']}#{k}: letter order differs")
        pre = S2.render_ids(sc, s, k, sc.date_pin)
        gen = list(c["fmt"].reason_open) + c["tok"].encode(
            S2.qwen_trace_text(q, r["gen_ids"]), add_special_tokens=False)
        pos = S2.trace_positions(c, gen, len(pre), punct)[c["cls"]]
        if not pos:
            continue
        h = resid_at(model, pre + gen, pos, [c["layer"] - 1])[c["layer"] - 1]
        keys.append(f"{r['stage']}|{r['scenario_id']}|{k}")
        sums.append(h.sum(0).astype(np.float64))
        counts.append(len(pos))
        acts.append("protect" if r["norm_status"] == "violating" else "rule")
    v_qwen, per = S2.within_scenario_direction(keys, sums, counts, acts)
    if v_qwen is None:
        raise SystemExit("no scenario has both a protect and a rule Qwen trace")
    import json as _j

    ps = _j.loads((HERE.parents[1] / "5_moral_alignment" / "refusal_prompts.json").read_text())
    n = c["paper7_n"]
    ids, ti = last_token_index(sc, ps["harmful"][:n] + ps["harmless"][:n])
    b = c["layer"] - 1
    hv = np.stack([resid_at(model, x, [t], [b])[b][0] for x, t in zip(ids, ti)])
    harm = hv[:n].mean(0) - hv[n:].mean(0)
    harm = harm / np.linalg.norm(harm)
    pairs = [(p.moral, p.neutral) for p in load_moral_probing_v2().all_pairs
             if p.foundation.value == "care_harm"]
    if c["care_max"]:
        pairs = pairs[: c["care_max"]]
    acts_c = du.collect_pair_activations(model.wb, pairs, input_format="raw", layers=[b])
    care = du.mean_diff_direction(*acts_c[b])
    dirs = {"V_qwen": v_qwen.astype(np.float32), "neg_V_harm": (-harm).astype(np.float32),
            "care_harm": care.astype(np.float32)}
    np.savez(c["out"] / "directions.npz", **dirs, harmfulness=harm.astype(np.float32),
             qwen_ids=np.array(sorted(per)))
    cos = {f"{x}.{y}": float(dirs[x] @ dirs[y]) for i, x in enumerate(DIRS) for y in DIRS[i + 1:]}
    return {"status": "ok", "n_qwen_traces": len(keys), "n_qwen_scenarios": len(per),
            "n_qwen_engaged": len(set(per) & set(c["engaged"])), "paper7_n": n,
            "t_inst_tokens": sorted({c["tok"].decode([x[t]]) for x, t in zip(ids, ti)})[:20],
            "n_care_pairs": len(pairs), "cos": cos,
            "type": {"V_qwen": "outcome-conditioned, cross-model (Qwen3.8 protect - rule traces "
                               "re-encoded in Muse's to=self channel), clause_end, block 26",
                     "neg_V_harm": "stimulus/content: -(harmful - harmless), Paper 7's recipe, "
                                   "t_inst in Muse's chat render, block 26",
                     "care_harm": "stimulus/content: W4's MFT care/harm mean difference, raw "
                                  "format, mean-pooled, block 26"}}


def phase_sweep(c, sc, model, a) -> dict:
    jobs = L.h1_jobs(sc, c["H"], c["sweep"], c["r"])
    none = S2.run_arm(c, sc, model, "st2m_sweep", "sweep_none", jobs, None)
    al = list(c["alphas"])
    jobs4 = [j for _ in al for j in jobs]
    v = c["dirs"]["V_qwen"]
    V = np.stack([x * v for x in al for _ in jobs]).astype(np.float32)
    steered = S2.run_arm(c, sc, model, "st2m_sweep", "sweep_V_qwen", jobs4, SG.Add(V, 1.0),
                         extra=lambda i: {"alpha": al[i // len(jobs)],
                                          "alpha_factor": ALPHA_FACTORS[i // len(jobs)]})
    per = {}
    for k, x in enumerate(al):
        part = steered[k * len(jobs): (k + 1) * len(jobs)]
        per[str(x)] = {"coherence": T13._coh(part, none, relative=True),  # noqa: SLF001
                       "dP_descriptive": T13.delta_p(part, none)}
    ok = [x for x in al if per[str(x)]["coherence"]["passes"]]
    return {"status": "ok", "alphas": al, "per_alpha": per, "alpha": max(ok) if ok else min(al),
            "no_coherent_alpha": not ok,
            "rule": "largest alpha passing Amendment 13's relative bars (T1.3's sweep rule)"}


def direction_vector(c: dict, name: str, alpha: float) -> np.ndarray | None:
    return None if name == "none" else alpha * c["dirs"][name]


def phase_spec(c, sc, model, a, name: str, alpha: float | None = None) -> dict:
    al = c["alpha"] if alpha is None else alpha
    if name == "fill":
        return DL.phase_spec_fill(c, sc, model, {}, a, "fill")
    tag = name if alpha is None else f"{name}_half"
    with C.add_everywhere(model, c["layer"], direction_vector(c, name, al)):
        res = L.run_letter_cell(sc, [c["panel"][i] for i in c["spec_ids"]], "dl_chat_neutral",
                                f"spec/{tag}")
    return {"status": "ok", "alpha": 0.0 if name == "none" else al, **res}


def arm_intervention(c: dict, name: str, jobs: list[dict], alpha: float):
    if name == "none":
        return None
    if name == "fill":
        fu = c["inp"]["fill_unit"]
        return SG.Add(np.stack([alpha * fu[j["k"] % len(fu)] for j in jobs]).astype(np.float32),
                      1.0)
    return SG.Add(c["dirs"][name], alpha)


def needs_step_down(coh: dict) -> bool:
    """Amendment 25: an arm failing coherence on trace length reruns once at alpha / 2."""
    lo, hi = SG.COHERENCE["length_ratio"]
    return bool(not coh["passes"] and not lo <= coh["length_ratio"] <= hi)


def phase_arm(c, sc, model, a, name: str) -> dict:
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r"])
    al = c["alpha"]
    iv = arm_intervention(c, name, jobs, al)
    rows = S2.run_arm(c, sc, model, "st2m", f"arm_{name}", jobs, iv,
                      extra=lambda i: {"alpha": 0.0 if name == "none" else al, "axis": name})
    out = {"status": "ok", "n": len(rows), "alpha": al,
           "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}
    if name in DIRS:
        none = _jsonl(c["out"] / "gen" / "arm_none.jsonl")
        coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
        out["coherence"] = coh
        if needs_step_down(coh):
            if a.deadline and time.time() + 2 * len(rows) * (
                    phase_done(c, "pilot").get("seconds_per_rollout") or 0) > a.deadline:
                out["step_down"] = "not run (would end past the deadline)"
                return out
            half = al / 2
            r2 = S2.run_arm(c, sc, model, "st2m", f"arm_{name}_half", jobs,
                            arm_intervention(c, name, jobs, half),
                            extra=lambda i: {"alpha": half, "axis": name})
            out["step_down"] = {"alpha": half, "n": len(r2),
                                "coherence": T13._coh(r2, none, relative=True),  # noqa: SLF001
                                "spec": phase_spec(c, sc, model, a, name, alpha=half)}
    return out


def run_hf(c: dict, a, todo: list[str]) -> int:
    model, sc = None, None
    per: list[float] = []
    for name in todo:
        ctx_derived(c)
        bailed = (phase_done(c, "pilot").get("reprice") or {}).get("bailed") or []
        if name in bailed:
            rec = load_record(c)
            rec["phases"][name] = {"status": "bailed (pilot re-price, Amendment 30 item 4)",
                                   "finished": _now()}
            save_record(c, rec)
            print(f">> st2m {name}: bailed by the pilot's re-price", flush=True)
            continue
        gen = name.startswith("arm_") or name == "sweep"
        n = len(c.get("engaged") or []) * c["r"] if name.startswith("arm_") else 0
        est = float(np.mean(per)) * n if per and n else 0.0
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2m {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if model is None:
            model = load_model(c, a)
            sc = screen(c, model, date_pin(c, a), a.dry_run)
        if name == "dose0":
            res = phase_dose0(c, sc, model, a)
        elif name == "bank":
            res = phase_bank(c, sc, model, a)
        elif name == "derive":
            res = phase_derive(c, sc, model, a)
        elif name == "pilot":
            res = phase_pilot(c, sc, model, a)
        elif name == "directions":
            res = phase_directions(c, sc, model, a)
        elif name == "sweep":
            res = phase_sweep(c, sc, model, a)
        elif name.startswith("spec_"):
            res = phase_spec(c, sc, model, a, name[len("spec_"):])
        else:
            res = phase_arm(c, sc, model, a, name[len("arm_"):])
        if gen and n:
            per.append((time.time() - t0) / n)
        rec = load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        if name == "pilot":  # the re-price reads the pilot's record
            rec = load_record(c)
            rec["phases"]["pilot"]["reprice"] = reprice(c, a)
            save_record(c, rec)
            res = rec["phases"]["pilot"]
        brief = {k: res[k] for k in ("n", "rows", "n_engaged", "parse_rate", "batch", "alpha",
                                     "no_coherent_alpha", "n_qwen_scenarios")
                 if k in res}
        if name == "pilot":
            brief["bailed"] = res["reprice"]["bailed"]
        if "coherence" in res:
            brief["coherent"] = res["coherence"]["passes"]
        print(f">> st2m {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default=",".join(HF_PHASES))
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--qwen-topup", type=Path, default=None)
    ap.add_argument("--qwen-screen", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    c = context(a)
    rec = load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    if not todo:
        print(">> st2m: every requested phase finished earlier", flush=True)
        return 0
    if todo == ["v1"]:
        t0 = time.time()
        res = phase_v1(c, a)
        rec = load_record(c)
        rec["phases"]["v1"] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        print(f">> st2m v1: done ({json.dumps({'n': res['n'], 'P': res['summary'].get('P_protect_mean_over_scenarios')})})",  # noqa: E501
              flush=True)
        return 0
    if "v1" in todo:
        raise SystemExit("v1 runs in its own process (vLLM), before the HF phases")
    return run_hf(c, a, todo)


if __name__ == "__main__":
    sys.exit(main())
