"""ST Phase 2 on GPT-OSS-20B (ST_PLAN Amendment 22 and its riders; author, 2026-10-21): the
target-search cells, one HF process on the ST stack of record (torch 2.13 in the vLLM venv, eager
attention and experts, bf16 dequantized), resumable arm by arm.

Cell 1 (pod A), the full-n gate: the 45 brake-engaged H1_train scenarios x 16 rollouts, fresh seeds
(stage ``st2c1``), every arm regenerated at alpha 267.3 on ``clause_end``, block 12 (hidden-state
index 13): ``none``, ``fill`` (the T1.3 unit draws, 5, cycled by rollout), ``care_harm``,
``V_harm`` and ``neg_a`` (the anchor axis negated). Each rollout is parsed by the parser of record
and scored for coherence (its trace's NLL under the unsteered model). ``c1_read`` writes the
summary (ΔP against ``none`` and against ``fill``, Amendment 13's relative coherence bars, §5.0's
rule, rider (a)'s protect-ward reading, care beyond harm as ΔP(care_harm) − ΔP(V_harm)); the
analysis of record, Holm-Bonferroni included, runs locally from the saved rows.

Cell 2 (pod B), ``V_rule`` (Amendment 22 item 2 and its Cell 2 decisions): ``c2_extract`` prefills
each of the 576 T1.1 rule-present traces of the 72 deletion-twin non-engaged H1_train scenarios
under the saved rule-present prompt and under the twin's prompt, and banks per-trace residual sums
at block 12 per position class (``clause_end``, ``onset``, ``stride32``, ``prompt``) plus a
``clause_end`` sample for the participation ratio; ``c2_direction`` builds ``V_rule`` (per-trace
sums -> per-scenario mean over positions -> mean over scenarios, unit), the twin-prefix target per
class, the split-half cosine (36/36, seeded) and the matched random direction (the T1.3 fill draw
0, covariance-matched at this site) with its own targets; ``c2_anchor`` the end-of-scenario side
read; ``c2_ablate`` / ``c2_random`` / ``c2_twin_none`` the engaged 45 x 8 (seeds paired with Cell
1's first 8 rollouts, Cell 1's date pin); ``c2_transport`` the dose-0 control (45 x 8
permutations: rule-present, rule-present ablated, rule-present random-ablated, rule-absent twin);
``c2_freegen`` the free-generation side read (Cell 1's ``none`` vs the twins' free generations).

Cell 3 (pod B), cross-recipe content (Amendment 22 item 3): ``c3_qwen`` tops up Qwen3.8-27B on vLLM
(engaged 45 x 8 at its screen's setting: effort medium, cap 1,536, KDG sampler v2, its screen's date
pin), before any HF model loads; ``c3_reencode`` re-encodes each parsed protect or rule Qwen trace
(top-up and 60 x 4 screen) in GPT-OSS (GPT-OSS's prompt for the same scenario and permutation, the
trace text in its analysis channel) and banks clause_end sums at block 12; ``c3_direction`` builds
``V_qwen`` (within scenario, protect minus rule, per position then scenario, unit) over scenarios
with both; ``c3_qwen_arm`` adds it at alpha 267.3 on the engaged 45 x 8 (seeds paired with Cell
1). ``V_oss``, the cosine read with its null and the writing-style loading run locally from T1.1's
bank (zero GPU).

    python3 st2_driver.py --out <dir> --inputs <st2_inputs dir> [--phases ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
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
import steering_gate as SG  # noqa: E402
import t13_driver as T13  # noqa: E402

KEY = "gpt_oss_20b"
DATA = HERE.parent / "data"
L_STAR = 13  # hidden-state index (block 12), Amendment 12 item 1
CLS = "clause_end"
ALPHA = 267.301171875  # the gate of record's alpha (Amendment 13)
R1 = 16  # Cell 1 rollouts per scenario (Amendment 22 rider (d))
BATCH = 160
C1_ARMS = ("none", "fill", "care_harm", "V_harm", "neg_a")
R2 = 8  # Cells 2 and 3 rollouts per scenario, paired with Cell 1's first 8 (seeds st2c1)
QWEN = "qwen38_27b"
QWEN_DATE_PIN = "2026-10-19"  # its screen's (the top-up matches its setting)
NO_MODEL = {"c1_read", "c3_qwen"}  # phases that never load GPT-OSS
CLASSES = ("clause_end", "onset", "stride32", "prompt")
SPLIT_SEED = 20261021
PR_PER_TRACE = 8  # clause_end positions kept per trace for the participation ratio


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    import pod_st_screen as P
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    from deepsteer.kdg.schema import load_scenario_dir

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    Hs, _ = load_scenario_dir([P.H1_SET])
    kept = set(json.loads(P.H1_KEPT.read_text())["ids"])
    z = np.load(a.inputs / "st2_inputs.npz")
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept},
         "engaged": json.loads((DATA / "st2_engaged_train_v1.json").read_text())["ids"],
         "inp": {k: z[k] for k in z.files}, "layer": L_STAR, "cls": CLS,
         "cap": L.CAP_OF_RECORD, "batch": BATCH, "alpha": ALPHA, "r1": R1, "r2": R2}
    tw = json.loads((DATA / "st2_rule_twins_v1.json").read_text())["twins"]
    c["twins"] = tw
    c["extract"] = sorted(sid for sid, v in tw.items() if sid not in set(c["engaged"])
                          and v["twin_kind"] != "rewritten")
    c["t11"] = a.t11 / "gen" / "h1_rollouts_read.jsonl" if a.t11 else None
    c["c1_none"] = a.c1 / "gen" / "c1_none.jsonl" if a.c1 else None
    c["qwen_screen"] = a.qwen_screen
    if a.dry_run:  # a random tiny model writes no punctuation: onset, so the hooks fire
        c.update(engaged=c["engaged"][:2], layer=2, cap=24, batch=8, alpha=1.0, r1=2, r2=2,
                 cls="onset", extract=c["extract"][:4])
    return c


def load_model(c: dict, a):
    import torch
    from kdg_pod_lib import ModelWrapper

    if a.dry_run:  # a tiny GPT-OSS with the real tokenizer; inputs at the tiny width
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = st_tiny.build(KEY, c["out"] / "_tiny")
        m = ModelWrapper(str(path), None, torch_dtype=torch.float32)
        h = m.model.config.hidden_size
        rng = np.random.default_rng(0)
        unit = lambda v: (v / np.linalg.norm(v)).astype(np.float32)  # noqa: E731
        c["inp"] = {k: unit(rng.standard_normal(h)) for k in ("anchor_axis", "care_harm",
                                                               "V_harm")}
        c["inp"]["fill_unit"] = SG.fill_draws(rng.standard_normal((64, h)), 1.0, 5, 0)
        fmt = c["fmt"]  # a random model never writes the harmony header: open the trace at once
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        return m
    return ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                        attn_implementation="eager", **c["P"].hf_load_kwargs(c["spec"]))


def screen(c: dict, model, a) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    sc.unbatched = True
    sc.dose0 = c["fmt"].dose0[0]
    return sc


def record_path(c: dict) -> Path:
    return c["out"] / "st2_record.json"


def load_record(c: dict) -> dict:
    p = record_path(c)
    return json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    record_path(c).write_text(json.dumps(rec, indent=1, default=L._json_default))


def seeds_for(stage: str, jobs: list[dict]) -> list[int]:
    """Fresh seeds by stage name, shared across a cell's arms by (scenario, rollout)."""
    return [zlib.crc32(f"{stage}|{j['s'].id}|{j['k']}".encode()) % (2**31) for j in jobs]


def run_arm(c: dict, sc, model, stage: str, name: str, jobs: list[dict], intervention,
            extra=None) -> list[dict]:
    """Generate, save per rollout, parse with the parser of record, score the trace's NLL under
    the unsteered model (T1.3's ``run_arm`` with this cell's stage and seeds)."""
    raw = c["out"] / "gen" / f"{name}.raw.jsonl"
    raw.unlink(missing_ok=True)
    sd = seeds_for(stage, jobs)
    t0 = time.time()
    rows = SG.generate(model, [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"], seeds=sd,
                       layer=c["layer"] if intervention is not None else None,
                       cls=c["cls"] if intervention is not None else None,
                       intervention=intervention, batch=c["batch"],
                       on_finish=SG.jsonl_saver(raw))
    gen_s = time.time() - t0
    parsed = L.build_rows(sc, jobs, [r["gen_ids"] for r in rows], sd, cap=c["cap"], stage=stage,
                          engine="hf_steered")
    for i, (p, r, j) in enumerate(zip(parsed, rows, jobs)):
        p.update(arm=name, n_steered=r["n_steered"], trace_start=r["trace_start"],
                 hit_cap=r["hit_cap"], rollout_key=f"{stage}|{j['s'].id}|{j['k']}",
                 trace_nll_unsteered=SG.trace_nll(model, j["ids"], r["gen_ids"]),
                 **((extra or (lambda _i: {}))(i)))
    sc.save_jsonl(f"gen/{name}.jsonl", parsed)
    print(f">> st2 {name}: {len(parsed)} rollouts in {gen_s:.0f}s, parsed "
          f"{np.mean([p['letter'] is not None for p in parsed]):.3f}", flush=True)
    return parsed


def c1_intervention(c: dict, arm: str, jobs: list[dict]):
    inp, al = c["inp"], c["alpha"]
    if arm == "none":
        return None
    if arm == "fill":
        fu = inp["fill_unit"]
        return SG.Add(np.stack([al * fu[j["k"] % len(fu)] for j in jobs]).astype(np.float32), 1.0)
    if arm == "neg_a":
        return SG.Add(-inp["anchor_axis"], al)
    return SG.Add(inp[arm], al)


def phase_c1_arm(c, sc, model, rec, a, arm: str) -> dict:
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r1"])
    fu = len(c["inp"]["fill_unit"])
    rows = run_arm(c, sc, model, "st2c1", f"c1_{arm}", jobs, c1_intervention(c, arm, jobs),
                   extra=lambda i: {"alpha": 0.0 if arm == "none" else c["alpha"], "axis": arm,
                                    **({"fill_draw": jobs[i]["k"] % fu} if arm == "fill"
                                       else {})})
    return {"status": "ok", "n": len(rows), "alpha": c["alpha"],
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def _rows(c: dict, name: str) -> list[dict]:
    return [json.loads(x) for x in (c["out"] / "gen" / f"{name}.jsonl").open()]


def protect_ward(dp_none: dict, dp_ctrl: dict, coh: dict) -> bool:
    """Amendment 22 rider (a): ΔP > 0 against none and against the matched control, both CIs
    excluding 0, under the coherence bars."""
    lo_n, lo_c = dp_none["ci95"][0], dp_ctrl["ci95"][0]
    return bool(coh["passes"] and lo_n is not None and lo_c is not None and lo_n > 0
                and lo_c > 0)


def phase_c1_read(c, sc, model, rec, a) -> dict:
    """The pod's summary; the analysis of record (Holm-Bonferroni over the pre-registered arms,
    the negative reading's MDEs) is local."""
    none, fill = _rows(c, "c1_none"), _rows(c, "c1_fill")
    keys = [r["rollout_key"] for r in none]
    pn, pf = T13.p_by_scenario(none), T13.p_by_scenario(fill)
    out: dict = {"fill": {"dP_vs_none": T13.delta_p(fill, none),
                          "coherence": T13._coh(fill, none, relative=True)},  # noqa: SLF001
                 "arms": {}}
    for arm in ("care_harm", "V_harm", "neg_a"):
        rows = _rows(c, f"c1_{arm}")
        if [r["rollout_key"] for r in rows] != keys:
            raise SystemExit(f"c1_{arm} rows do not pair with c1_none")
        pa = T13.p_by_scenario(rows)
        ids = sorted(set(pa) & set(pf))
        dn, df = T13.delta_p(rows, none), T13.boot([pa[s] - pf[s] for s in ids])
        coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
        gate = bool(coh["passes"] and dn["mean"] is not None and df["mean"] is not None
                    and dn["mean"] >= T13.GATE_DP and df["mean"] >= T13.GATE_OVER_FILL)
        out["arms"][arm] = {"dP_vs_none": dn, "dP_vs_fill": df, "coherence": coh,
                            "passes_gate": gate, "protect_ward": protect_ward(dn, df, coh)}
    pc, ph = T13.p_by_scenario(_rows(c, "c1_care_harm")), T13.p_by_scenario(_rows(c, "c1_V_harm"))
    ids = sorted(set(pc) & set(ph) & set(pn))
    out["care_beyond_harm"] = T13.boot([pc[s] - ph[s] for s in ids])
    out["rule"] = ("section 5.0 (dP >= +0.10 vs none and >= +0.05 over fill) under Amendment 13's "
                   "relative coherence bars; protect-ward per Amendment 22 rider (a); care beyond "
                   "harm = dP(care_harm) - dP(V_harm), paired by scenario")
    return {"status": "ok", **out}


# ---- cell 2 -------------------------------------------------------------------------------------
def twin_scenario(c: dict, sid: str):
    return dataclasses.replace(c["H"][sid], agent_text=c["twins"][sid]["agent_text_twin"])


def render_ids(sc, s, k: int, date_pin: str, cell: str | None = None) -> list[int]:
    """The agent-frame generation prompt (or, with ``cell``, the dose-0 input) at permutation k
    under an explicit date pin."""
    o = L.assign_letters(s, k)
    if cell is not None:
        return sc.fmt.dose0_ids(sc.tok, sc.messages(s, o, cell), sc.effort, sc.dose0, date_pin)
    msgs = L.letter_chat_messages(s, o, "agent", "neutral", "primary")
    return sc.fmt.prompt_ids(sc.tok, sc.fmt.render(sc.tok, msgs, sc.effort, date_pin))


def class_sums(model, ids: list[int], n_prompt: int, pos: dict[str, list[int]], layer: int,
               keep: int = 0, rng=None) -> tuple[dict, np.ndarray | None]:
    """One single-row teacher-forced forward: per class the sum of block ``layer - 1``'s output
    over the class's absolute positions (``prompt`` = every prompt position; float32 on the
    device, float64 out), and up to ``keep`` sampled ``clause_end`` rows (float32)."""
    import torch

    grab: dict = {}
    hd = model.wb._get_layer_module(layer - 1).register_forward_hook(  # noqa: SLF001
        lambda _m, _i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]))
    try:
        with torch.no_grad():
            model.model(input_ids=torch.tensor([list(ids)], device=model.model.device),
                        logits_to_keep=1)
    finally:
        hd.remove()
    h = grab["h"].float()
    out = {}
    allpos = {**pos, "prompt": list(range(n_prompt))}
    for k in CLASSES:
        ix = allpos.get(k) or []
        out[k] = ((h[ix].sum(0).cpu().numpy().astype(np.float64) if ix
                   else np.zeros(h.shape[-1])), len(ix))
    sample = None
    ce = allpos.get("clause_end") or []
    if keep and ce:
        pick = sorted(rng.sample(ce, min(keep, len(ce))))
        sample = h[pick].cpu().numpy().astype(np.float32)
    return out, sample


def trace_positions(c: dict, gen: list[int], n_prompt: int, punct) -> dict[str, list[int]]:
    import t11_driver as T11

    pos: dict[str, list[int]] = {k: [] for k in ("clause_end", "onset", "stride32")}
    for i, cls in T11.class_positions(c["fmt"], c["tok"], gen, punct).items():
        for k in cls:
            pos[k].append(n_prompt + i)
    return pos


def phase_c2_extract(c, sc, model, rec, a) -> dict:
    """Item 2's read of record: each saved T1.1 rule-present trace of the extraction scenarios,
    prefilled under its saved prompt and under the twin's (same date pin, effort and permutation;
    the original is re-rendered first and must reproduce the saved ids)."""
    import hashlib

    punct = SG.punct_mask(c["tok"])
    want = set(c["extract"])
    rows = [r for r in (json.loads(x) for x in c["t11"].open()) if r["scenario_id"] in want]
    if a.dry_run:
        rows = [r for r in rows if r["rollout"] < 2]
    rng = random.Random(SPLIT_SEED)
    keys, sums_p, sums_t, counts, sample = [], [], [], [], []
    for r in sorted(rows, key=lambda r: (r["scenario_id"], r["rollout"])):
        s, k = c["H"][r["scenario_id"]], r["rollout"]
        if L.letter_map(L.assign_letters(s, k)) != r["order"]:
            raise SystemExit(f"{r['scenario_id']}#{k}: letter order differs from T1.1's")
        pres = render_ids(sc, s, k, r["date_pin"])
        if pres != r["prompt_ids"] or hashlib.sha256(json.dumps(pres).encode()).hexdigest() != \
                r["prompt_ids_sha256"]:
            raise SystemExit(f"{r['scenario_id']}#{k}: re-rendered prompt differs from T1.1's")
        twin = render_ids(sc, twin_scenario(c, r["scenario_id"]), k, r["date_pin"])
        gen = r["gen_ids"] if not a.dry_run else r["gen_ids"][:48]
        sp, smp = class_sums(model, pres + gen, len(pres),
                             trace_positions(c, gen, len(pres), punct), c["layer"], PR_PER_TRACE,
                             rng)
        st, _ = class_sums(model, twin + gen, len(twin),
                           trace_positions(c, gen, len(twin), punct), c["layer"])
        keys.append(f"{r['scenario_id']}|{k}")
        sums_p.append([sp[x][0] for x in CLASSES])
        sums_t.append([st[x][0] for x in CLASSES])
        counts.append([[sp[x][1] for x in CLASSES], [st[x][1] for x in CLASSES]])
        if smp is not None:
            sample.append(smp)
    out = c["out"] / "c2"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "extract.npz", keys=np.array(keys), classes=np.array(CLASSES),
                        sums_present=np.array(sums_p), sums_twin=np.array(sums_t),
                        counts=np.array(counts), pr_sample=np.concatenate(sample) if sample
                        else np.zeros((0, 1), np.float32))
    return {"status": "ok", "n_traces": len(keys), "n_scenarios": len({k.split("|")[0]
                                                                       for k in keys})}


def participation_ratio(x: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)) if x.shape[1] <= 4096 else None
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def rule_direction(keys, sums_p, sums_t, counts, ci: int, ids: list[str] | None = None):
    """Per scenario: (sum present - sum twin) / n over its traces' class positions; mean over
    scenarios; unit. Returns (unit direction, per-scenario mean differences)."""
    by: dict[str, list] = {}
    for key, sp, st, n in zip(keys, sums_p, sums_t, counts):
        sid = str(key).split("|")[0]
        if n[0][ci] != n[1][ci]:
            raise SystemExit(f"{key}: class counts differ between the prefixes")
        d = by.setdefault(sid, [np.zeros_like(sp[ci]), 0])
        d[0] += sp[ci] - st[ci]
        d[1] += n[0][ci]
    per = {s: v[0] / v[1] for s, v in by.items() if v[1] and (ids is None or s in ids)}
    m = np.mean([per[s] for s in sorted(per)], 0)
    return m / np.linalg.norm(m), per


def twin_targets(keys, sums_t, counts, v: np.ndarray) -> dict[str, float]:
    """Per class: the mean projection on v under the twin prefix, mean over positions within the
    scenario then over scenarios."""
    out = {}
    for ci, cls in enumerate(CLASSES):
        by: dict[str, list] = {}
        for key, st, n in zip(keys, sums_t, counts):
            d = by.setdefault(str(key).split("|")[0], [0.0, 0])
            d[0] += float(st[ci] @ v)
            d[1] += n[1][ci]
        vals = [x / n for x, n in by.values() if n]
        out[cls] = float(np.mean(vals)) if vals else None
    return out


def phase_c2_direction(c, sc, model, rec, a) -> dict:
    z = np.load(c["out"] / "c2" / "extract.npz")
    keys, sp, st, n = z["keys"], z["sums_present"], z["sums_twin"], z["counts"]
    ci = CLASSES.index("clause_end")
    v, per = rule_direction(keys, sp, st, n, ci)
    ids = sorted(per)
    rng = random.Random(SPLIT_SEED)
    half = sorted(rng.sample(ids, len(ids) // 2))
    va, _ = rule_direction(keys, sp, st, n, ci, set(half))
    vb, _ = rule_direction(keys, sp, st, n, ci, set(ids) - set(half))
    r = c["inp"]["fill_unit"][0].astype(np.float64)
    r = r / np.linalg.norm(r)
    pr = participation_ratio(z["pr_sample"].astype(np.float64)) if len(z["pr_sample"]) > 2 \
        else None
    out = {"status": "ok", "n_scenarios": len(ids), "split_half_cos": float(va @ vb),
           "split": {"a": half, "seed": SPLIT_SEED},
           "target_twin": twin_targets(keys, st, n, v),
           "random": {"source": "T1.3 fill_unit[0] (covariance-matched at clause_end, block 12)",
                      "cos_with_V_rule": float(r @ v), "target_twin": twin_targets(keys, st, n, r)},
           "participation_ratio_clause_end": pr, "n_pr_rows": int(len(z["pr_sample"])),
           "diff_norm_mean": float(np.mean([np.linalg.norm(x) for x in per.values()]))}
    np.savez(c["out"] / "c2" / "v_rule.npz", v_rule=v.astype(np.float32),
             random=r.astype(np.float32), ids=np.array(ids),
             per_scenario=np.array([per[s] for s in ids], np.float32))
    return out


def _anchor_index(sc, ids: list[int], s) -> int:
    """The last prompt token inside the scenario text (end of scenario, before the options)."""
    text = sc.tok.decode(ids)
    end = text.find(s.agent_text)
    if end < 0:
        raise SystemExit(f"{s.id}: scenario text not found in its rendered prompt")
    end += len(s.agent_text)
    enc = sc.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(ids):
        raise SystemExit(f"{s.id}: re-encoding the prompt changed its ids")
    return max(i for i, (_b, e) in enumerate(enc["offset_mapping"]) if e <= end)


def phase_c2_anchor(c, sc, model, rec, a) -> dict:
    """Side read: rule-present minus twin at the end-of-scenario anchor (prompt only; the anchor
    precedes the options, so permutation 0 stands for all), over the extraction scenarios."""
    import torch

    v = np.load(c["out"] / "c2" / "v_rule.npz")["v_rule"].astype(np.float64)
    diffs = []
    for sid in c["extract"]:
        hs = []
        for s in (c["H"][sid], twin_scenario(c, sid)):
            ids = render_ids(sc, s, 0, sc.date_pin)
            j = _anchor_index(sc, ids, s)
            grab: dict = {}
            hd = model.wb._get_layer_module(c["layer"] - 1).register_forward_hook(  # noqa: SLF001
                lambda _m, _i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]))
            try:
                with torch.no_grad():
                    model.model(input_ids=torch.tensor([ids[: j + 1]], device=model.model.device),
                                logits_to_keep=1)
            finally:
                hd.remove()
            hs.append(grab["h"][-1].float().cpu().numpy().astype(np.float64))
        diffs.append(hs[0] - hs[1])
    m = np.mean(diffs, 0)
    u = m / np.linalg.norm(m)
    np.save(c["out"] / "c2" / "anchor_diffs.npy", np.array(diffs, np.float32))
    return {"status": "ok", "n": len(diffs), "cos_with_V_rule": float(u @ v)}


def c1_date_pin(c: dict) -> str:
    """Cells 2 and 3 render with Cell 1's date pin, so their arms pair with Cell 1's none."""
    pins = {json.loads(x)["date_pin"] for x in c["c1_none"].open()}
    if len(pins) != 1:
        raise SystemExit(f"Cell 1's none carries {len(pins)} date pins")
    return pins.pop()


def c2_jobs(c: dict, sc, twin: bool) -> list[dict]:
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r2"])
    if twin:
        for j in jobs:
            s = twin_scenario(c, j["s"].id)
            j["ids"] = render_ids(sc, s, j["k"], sc.date_pin)
            j["rendered_sha256"] = None
            j["twin_kind"] = c["twins"][s.id]["twin_kind"]
    return jobs


def phase_c2_arm(c, sc, model, rec, a, arm: str) -> dict:
    d = rec["phases"]["c2_direction"]
    vr = np.load(c["out"] / "c2" / "v_rule.npz")
    if arm == "ablate":
        iv = SG.Ablate(vr["v_rule"], d["target_twin"]["clause_end"])
    elif arm == "random":
        iv = SG.Ablate(vr["random"], d["random"]["target_twin"]["clause_end"])
    else:
        iv = None
    jobs = c2_jobs(c, sc, twin=(arm == "twin_none"))
    rows = run_arm(c, sc, model, "st2c1", f"c2_{arm}", jobs, iv,
                   extra=lambda i: {"cell": 2, "arm2": arm,
                                    **({"twin_kind": jobs[i]["twin_kind"]} if arm == "twin_none"
                                       else {})})
    return {"status": "ok", "n": len(rows),
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


def _p_viol(lp: np.ndarray, s, k: int) -> float:
    o = L.assign_letters(s, k)
    shown = [x for x, _ in o]
    q = np.exp(np.array([lp["ABCDE".index(x)] for x in shown]))
    q /= q.sum()
    return float(q[[op.norm_status for _x, op in o].index("violating")])


def phase_c2_transport(c, sc, model, rec, a) -> dict:
    """The dose-0 transport control: the same ablation at every input position (target: the twin
    prefix's mean projection over prompt positions, the ``prompt`` class); rows saved per
    (scenario, permutation, condition)."""
    d = rec["phases"]["c2_direction"]
    vr = np.load(c["out"] / "c2" / "v_rule.npz")
    lids = L.letter_ids(sc)
    conds = {"present": (False, None, 0.0),
             "present_ablated": (False, vr["v_rule"], d["target_twin"]["prompt"]),
             "present_random": (False, vr["random"], d["random"]["target_twin"]["prompt"]),
             "twin": (True, None, 0.0)}
    rows = []
    for sid in c["engaged"]:
        for k in range(L.N_PERM if not a.dry_run else 2):
            for name, (tw, vec, tgt) in conds.items():
                s = twin_scenario(c, sid) if tw else c["H"][sid]
                ids = render_ids(sc, s, k, sc.date_pin, cell="dl_chat_neutral")
                lp = SG.ablate_readout(model, [ids], vec, tgt, c["layer"], lids)[0]
                rows.append({"scenario_id": sid, "perm": k, "condition": name,
                             "twin_kind": c["twins"][sid]["twin_kind"],
                             "letter_logp": [float(x) for x in lp],
                             "p_violating": _p_viol(lp, s, k)})
    sc.save_jsonl("c2/transport.jsonl", rows)
    per = {}
    for r in rows:
        per.setdefault((r["condition"], r["scenario_id"]), []).append(r["p_violating"])
    m = {k: float(np.mean(v)) for k, v in per.items()}
    ids = [s for s in c["engaged"] if s != "H1-A-168"]

    def moved(sel):
        return T13.boot([m[("present_ablated", s)] - m[("present", s)] for s in sel])
    return {"status": "ok", "n_rows": len(rows),
            "dP0_ablated_vs_present_headline": moved(ids),
            "dP0_ablated_vs_present_all45": moved(c["engaged"]),
            "dP0_twin_vs_present": T13.boot([m[("twin", s)] - m[("present", s)]
                                             for s in c["engaged"]]),
            "dP0_random_vs_present": T13.boot([m[("present_random", s)] - m[("present", s)]
                                               for s in c["engaged"]]),
            "headline_excludes": ["H1-A-168"]}


def phase_c2_freegen(c, sc, model, rec, a) -> dict:
    """Side read: Cell 1's rule-present none against the twins' free generations, each rollout
    teacher-forced, clause_end at block 12; per scenario mean over positions, mean over scenarios,
    cosine with V_rule."""
    punct = SG.punct_mask(c["tok"])
    v = np.load(c["out"] / "c2" / "v_rule.npz")["v_rule"].astype(np.float64)
    cls = c["cls"]  # clause_end on the pod (the dry run's onset)
    means = {}
    jobs = {"present": {(j["s"].id, j["k"]): j for j in L.h1_jobs(sc, c["H"], c["engaged"],
                                                                  c["r1"])},
            "twin": {(j["s"].id, j["k"]): j for j in c2_jobs(c, sc, twin=True)}}
    srcs = {"present": c["c1_none"], "twin": c["out"] / "gen" / "c2_twin_none.jsonl"}
    for side, f in srcs.items():
        acc: dict[str, list] = {}
        for r in (json.loads(x) for x in f.open()):
            j = jobs[side].get((r["scenario_id"], r["rollout"]))
            if j is None or not r.get("gen_ids"):
                continue
            ids = j["ids"]
            sums, _ = class_sums(model, ids + r["gen_ids"], len(ids),
                                 trace_positions(c, r["gen_ids"], len(ids), punct), c["layer"])
            a_ = acc.setdefault(r["scenario_id"], [np.zeros_like(v), 0])
            a_[0] += sums[cls][0]
            a_[1] += sums[cls][1]
        means[side] = {s: x[0] / x[1] for s, x in acc.items() if x[1]}
    ids = sorted(set(means["present"]) & set(means["twin"]))
    if not ids:
        return {"status": "ok", "n": 0, "cos_with_V_rule": None}
    m = np.mean([means["present"][s] - means["twin"][s] for s in ids], 0)
    return {"status": "ok", "n": len(ids), "cos_with_V_rule": float(m @ v / np.linalg.norm(m)),
            "class": cls}


# ---- cell 3 -------------------------------------------------------------------------------------
def qwen_context(c: dict, a) -> dict:
    import pod_st_screen as P
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    spec = P.registry()[QWEN]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    return {"spec": spec, "tok": tok, "fmt": build_format(QWEN, tok, spec)}


def phase_c3_qwen(c, sc, model, rec, a) -> dict:
    """The Qwen top-up on vLLM, in this process before any HF model loads (the engine needs the
    GPU)."""
    import screen_vllm as SV
    import vllm_phase as V

    q = qwen_context(c, a)
    if a.dry_run:
        from screen_stub import ScreenStub

        m = ScreenStub(QWEN, q["tok"], q["fmt"], q["spec"]["n_layers"],
                       length_by_effort={"medium": 4})
    else:
        m = L.tok_only(q["tok"])
    qs = L.Screen(QWEN, q["spec"], m, q["fmt"], c["out"], QWEN_DATE_PIN, dry=a.dry_run)
    qs.effort = "medium"
    cap = c["cap"]
    jobs = L.h1_jobs(qs, c["H"], c["engaged"], c["r2"])
    wj = SV.worker_jobs(jobs, "st2c3q", cap)
    eng = V.engine_json(q["spec"], spec_on=False, label=f"{QWEN}:st2c3q")
    vrows, summ, rc = V.run_vllm(qs, wj, eng, c["out"] / "vllm" / "c3_qwen.jsonl")
    by = {r["job_id"]: r for r in vrows}
    g = [by[j["job_id"]]["gen_ids"] if by.get(j["job_id"], {}).get("status") == "finished"
         else None for j in wj]
    rows = L.build_rows(qs, jobs, g, [j["seed"] for j in wj], cap=cap, stage="st2c3q",
                        engine="vllm")
    qs.save_jsonl("gen/c3_qwen_topup.jsonl", rows)
    act = [r["norm_status"] for r in rows if r.get("norm_status")]
    return {"status": "ok", "n": len(rows), "rc": rc, "counts": {k: act.count(k) for k in set(act)},
            "tokens_per_second": summ.get("tokens_per_second")}


def qwen_trace_text(q: dict, gen: list[int]) -> str:
    """Qwen's thinking text (the template opens the think block; it ends at ``</think>``)."""
    close = q["fmt"].close_id
    return q["tok"].decode(gen[: gen.index(close)] if close in gen else gen,
                           skip_special_tokens=True)


def phase_c3_reencode(c, sc, model, rec, a) -> dict:
    """Each parsed protect or rule Qwen trace (top-up + screen) in GPT-OSS: GPT-OSS's prompt for
    the same scenario and permutation (Cell 1's date pin), the analysis channel opened, the trace
    text encoded by GPT-OSS's tokenizer; clause_end sums at block 12 per trace."""
    import kdg_harmony as kh

    q = qwen_context(c, a)
    punct = SG.punct_mask(c["tok"])
    src = [c["out"] / "gen" / "c3_qwen_topup.jsonl"] + ([c["qwen_screen"]] if c["qwen_screen"]
                                                         else [])
    rows = [json.loads(x) for f in src for x in f.open()]
    rows = [r for r in rows if r.get("norm_status") in ("violating", "consistent")
            and r["scenario_id"] in c["H"]]
    if a.dry_run:  # the tiny stub's letters are arbitrary: label alternately so both actions exist
        for i, r in enumerate(rows):
            r["norm_status"] = ("violating", "consistent")[i % 2]
    keys, sums, counts, acts = [], [], [], []
    ci = c["cls"]
    for r in rows:
        s, k = c["H"][r["scenario_id"]], r["rollout"]
        if L.letter_map(L.assign_letters(s, k)) != r["order"]:
            raise SystemExit(f"{r['scenario_id']}#{k}: letter order differs")
        pre = render_ids(sc, s, k, sc.date_pin) + list(kh.ANALYSIS_OPEN)
        body = c["tok"].encode(qwen_trace_text(q, r["gen_ids"]), add_special_tokens=False)
        gen = list(kh.ANALYSIS_OPEN) + body
        n_p = len(pre) - len(kh.ANALYSIS_OPEN)
        sm, _ = class_sums(model, pre + body, n_p, trace_positions(c, gen, n_p, punct), c["layer"])
        keys.append(f"{r['stage']}|{r['scenario_id']}|{k}")
        sums.append(sm[ci][0])
        counts.append(sm[ci][1])
        acts.append("protect" if r["norm_status"] == "violating" else "rule")
    (c["out"] / "c3").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(c["out"] / "c3" / "reencode.npz", keys=np.array(keys),
                        sums=np.array(sums), counts=np.array(counts), action=np.array(acts),
                        cls=np.array(ci))
    return {"status": "ok", "n_traces": len(keys),
            "n_protect": acts.count("protect"), "n_rule": acts.count("rule")}


def within_scenario_direction(keys, sums, counts, acts):
    """Per scenario with both actions: mean over protect-trace positions minus mean over
    rule-trace positions; mean over scenarios; unit."""
    acc: dict = {}
    for key, sm, n, act in zip(keys, sums, counts, acts):
        sid = str(key).split("|")[1]
        d = acc.setdefault(sid, {"protect": [0.0, 0], "rule": [0.0, 0]})
        d[str(act)][0] = d[str(act)][0] + sm
        d[str(act)][1] += int(n)
    per = {s: v["protect"][0] / v["protect"][1] - v["rule"][0] / v["rule"][1]
           for s, v in acc.items() if v["protect"][1] and v["rule"][1]}
    if not per:
        return None, {}
    m = np.mean([per[s] for s in sorted(per)], 0)
    return m / np.linalg.norm(m), per


def phase_c3_direction(c, sc, model, rec, a) -> dict:
    z = np.load(c["out"] / "c3" / "reencode.npz")
    v, per = within_scenario_direction(z["keys"], z["sums"], z["counts"], z["action"])
    if v is None:
        raise SystemExit("no scenario has both a protect and a rule Qwen trace")
    np.savez(c["out"] / "c3" / "v_qwen.npz", v_qwen=v.astype(np.float32), ids=np.array(sorted(per)),
             per_scenario=np.array([per[s] for s in sorted(per)], np.float32))
    return {"status": "ok", "n_scenarios": len(per),
            "n_engaged_among_them": len(set(per) & set(c["engaged"]))}


def phase_c3_qwen_arm(c, sc, model, rec, a) -> dict:
    v = np.load(c["out"] / "c3" / "v_qwen.npz")["v_qwen"]
    jobs = L.h1_jobs(sc, c["H"], c["engaged"], c["r2"])
    rows = run_arm(c, sc, model, "st2c1", "c3_qwen_arm", jobs, SG.Add(v, c["alpha"]),
                   extra=lambda i: {"cell": 3, "alpha": c["alpha"], "axis": "V_qwen"})
    return {"status": "ok", "n": len(rows),
            "parse_rate": float(np.mean([r["letter"] is not None for r in rows]))}


PHASES = {**{f"c1_{arm}": (lambda arm: lambda c, sc, m, rec, a: phase_c1_arm(c, sc, m, rec, a,
                                                                             arm))(arm)
             for arm in C1_ARMS}, "c1_read": phase_c1_read,
          "c2_extract": phase_c2_extract, "c2_direction": phase_c2_direction,
          "c2_anchor": phase_c2_anchor,
          **{f"c2_{arm}": (lambda arm: lambda c, sc, m, rec, a: phase_c2_arm(c, sc, m, rec, a,
                                                                              arm))(arm)
             for arm in ("ablate", "random", "twin_none")},
          "c2_transport": phase_c2_transport, "c2_freegen": phase_c2_freegen,
          "c3_qwen": phase_c3_qwen, "c3_reencode": phase_c3_reencode,
          "c3_direction": phase_c3_direction, "c3_qwen_arm": phase_c3_qwen_arm}
CELL1 = [f"c1_{arm}" for arm in C1_ARMS] + ["c1_read"]
CELL2 = ["c2_extract", "c2_direction", "c2_anchor", "c2_ablate", "c2_random", "c2_twin_none",
         "c2_transport", "c2_freegen"]
CELL3 = ["c3_qwen", "c3_reencode", "c3_direction", "c3_qwen_arm"]
POD_B = ["c3_qwen"] + CELL2 + CELL3[1:]  # the Qwen engine first, on an empty GPU
GEN_PHASES = ({f"c1_{a}" for a in C1_ARMS} | {"c2_ablate", "c2_random", "c2_twin_none"}
              | {"c3_qwen_arm"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--phases", default=",".join(CELL1))
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--t11", type=Path, default=None, help="T1.1's output dir (Cell 2's traces)")
    ap.add_argument("--c1", type=Path, default=None, help="pod A's output dir (Cell 1's none)")
    ap.add_argument("--qwen-screen", type=Path, default=None,
                    help="Qwen3.8's 60 x 4 screen rows (outputs/t10/qwen38_27b/screen)")
    a = ap.parse_args()
    c = context(a)
    rec = load_record(c)
    todo = [p for p in a.phases.split(",") if not (rec["phases"].get(p) or {}).get("finished")]
    if not todo:
        print(">> st2: every requested phase finished earlier", flush=True)
        return 0
    model, sc = None, None

    def ready():
        nonlocal model, sc
        if model is None:
            model = load_model(c, a)
            sc = screen(c, model, a)
            if any(p.startswith(("c2_", "c3_")) for p in todo) and c["c1_none"] is not None:
                sc.date_pin = c1_date_pin(c)  # pair with Cell 1's none (the prompt carries it)

    per_rollout: list[float] = []  # measured seconds per generated rollout, any arm

    def n_rollouts(name: str) -> int:
        return len(c["engaged"]) * (c["r1"] if name.startswith("c1_") else c["r2"])

    for name in todo:
        est = (float(np.mean(per_rollout)) * n_rollouts(name)
               if per_rollout and name in GEN_PHASES else 0.0)
        if a.deadline and time.time() + est > a.deadline:
            print(f">> st2 {name}: not started (would end past the deadline)", flush=True)
            return 3
        t0 = time.time()
        if name not in NO_MODEL:
            ready()
        res = PHASES[name](c, sc, model, rec, a)
        if name in GEN_PHASES:
            per_rollout.append((time.time() - t0) / n_rollouts(name))
        rec = load_record(c)
        rec["phases"][name] = {**res, "seconds": time.time() - t0, "finished": _now()}
        save_record(c, rec)
        brief = {k: res[k] for k in ("n", "parse_rate") if k in res}
        if name == "c1_read":
            r3 = lambda x: None if x is None else round(x, 3)  # noqa: E731
            brief = {k: {"dP": r3(v["dP_vs_none"]["mean"]), "vs_fill": r3(v["dP_vs_fill"]["mean"]),
                         "gate": v["passes_gate"], "protect_ward": v["protect_ward"]}
                     for k, v in res["arms"].items()}
        print(f">> st2 {name}: done ({json.dumps(brief)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
