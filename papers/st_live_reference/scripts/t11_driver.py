#!/usr/bin/env python3
"""T1.1 on the development model, GPT-OSS-20B (ST_PLAN §7; author, 2026-10-19). Four phases, each
its own process, each reading the previous phase's outputs from disk:

    python t11_driver.py --phase v1    --out OUT   # vLLM: H1 264 x 8, native reasoning
    python t11_driver.py --phase h     --out OUT   # HF (ST stack of record): readouts + banking
    python t11_driver.py --phase v2    --out OUT   # vLLM: independent none re-run + pair rider
    python t11_driver.py --phase qlora --out OUT   # Muse QLoRA shakedown (T0.6 item 3)

- **v1**: every H1 scenario x 8 permutations at the effort (medium) and cap (1,536) of record,
  KDG sampler v2, uninstrumented; the speculation probe on the first batch picks the setting;
  every row tagged ``sample: "selection"`` (the subset is selected on these rows, Amendment 1).
- **h** (torch 2.13 in the vLLM venv, Amendment 9; eager attention, GPT-OSS's eager-experts and
  bf16-dequantization pins; single-row readouts of record, Amendment 9 item 2): per rollout one
  teacher-forced forward over prompt + trace that banks the residual at the five T1.1 layers at
  the registered classes (``onset``, ``clause_end``, every 32nd trace token, and the
  post-reasoning decision position), reads the letter at the answer position (token-identity
  rows) and the trace's log-prob under the unadapted model; the dose-0 letter readout on all 264
  x 8 (residual at the pre-prefill END_OF_PROMPT token, §1's ``decision`` class); the
  judgment-frame residuals for H1_train at the same token, from which ``chat_sigma`` per banked
  layer; the brake-engaged subsets (H1_held for T1.5, H1_train for T1.3's gate) and the DPO pair
  count with the pair-rider list (§5.1, Amendment 3); one warmed eager HF probe at batch 132 for
  T1.3's price.
- **v2**: the independent ``none`` re-run on the H1_held brake-engaged subset x 8 (seeds disjoint
  from v1's; ``sample: "rerun"``) and, if the pairs fall below 40, 8 more rollouts on up to 20
  train scenarios lacking a pair; the floor rung P_none^re - P_none^sel.
- **qlora**: Muse 4-bit NF4, LoRA on the text tower, a few steps on a synthetic anchor target, the
  adapter merged into a bf16 copy, merged against adapted logits.

Banked layers (hidden-state index h = block h-1's output): block outputs {6, 9, 12, 15, 18} on
GPT-OSS, evenly spaced around the V_moral match layer (block 12, where T0.3's directions live).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402
import steering_gate as SG  # noqa: E402

KEY = "gpt_oss_20b"
BANK_BLOCKS = (6, 9, 12, 15, 18)  # block outputs; hidden-state index = block + 1
R_GEN, R_RERUN = 8, 8
STRIDE = 32
CHUNK = 64
PAIR_RIDER_MAX = 20
ACTION = {"violating": "protect", "consistent": "rule", "neutral": "defer"}  # H1 labels
QLORA_KEY = "muse_glimmer_30b"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def context(a) -> dict:
    """Registry, format, H1, split and order, shared by every phase."""
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
    H = {s.id: s for s in Hs if s.id in kept}
    order = L.h1_train_order(split)
    ids = list(order) + sorted(split["H1_held"])
    if a.dry_run:
        ids = list(order[:4]) + sorted(split["H1_held"])[:4]
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    return {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "H": H,
            "order": order, "ids": ids, "out": out,
            "train": set(split["H1_train"]), "held": set(split["H1_held"])}


def screen(c: dict, model, a, **kw) -> L.Screen:
    sc = L.Screen(KEY, c["spec"], model, c["fmt"], c["out"], a.date_pin, dry=a.dry_run, **kw)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    sc.layers_override = [b + 1 for b in BANK_BLOCKS]
    return sc


def stub(c: dict, a):
    from screen_stub import ScreenStub

    return ScreenStub(KEY, c["tok"], c["fmt"], c["spec"]["n_layers"])


def record_path(c: dict) -> Path:
    return c["out"] / "t11_record.json"


def load_record(c: dict) -> dict:
    p = record_path(c)
    return json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    record_path(c).write_text(json.dumps(rec, indent=1, default=L._json_default))


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ---- v1 ------------------------------------------------------------------------------------------
def phase_v1(c: dict, a) -> int:
    import screen_vllm as SV

    sc = screen(c, stub(c, a) if a.dry_run else L.tok_only(c["tok"]), a)
    jobs = L.h1_jobs(sc, c["H"], c["ids"], R_GEN)
    t0 = time.time()
    g = SV.gen_stage(sc, jobs, "t11", L.CAP_OF_RECORD, c["spec"], probe=True,
                     deadline=a.deadline)
    for r, j in zip(g["rows"], jobs):
        r["sample"], r["arm"] = "selection", "none"  # the uninstrumented arm (Amendment 1)
        r["rollout_key"] = f"t11|{r['scenario_id']}|{r['rollout']}"
        r["split"] = "train" if r["scenario_id"] in c["train"] else "held"
        r["prompt_ids"] = j["ids"]
    sc.save_jsonl("gen/h1_rollouts.jsonl", g["rows"])
    rec = load_record(c)
    rec["phases"]["v1"] = {"finished": _now(), "seconds": time.time() - t0,
                           "timing": g["timing"], "probe": g["probe"], "n_jobs": len(jobs),
                           "effort": sc.effort, "cap": L.CAP_OF_RECORD,
                           "summary": L.screen_summary(g["rows"])}
    rec["phase_v"] = {"t11_timing": g["timing"]}  # the re-price reads generation rates here
    save_record(c, rec)
    s = rec["phases"]["v1"]["summary"]
    print(f">> t11 v1: {len(jobs)} rollouts, {g['timing']['tokens_per_second']:.0f} tok/s "
          f"({g['choice']}), truncation {s['truncation_rate']}, P_protect "
          f"{s['P_protect_mean_over_scenarios']}", flush=True)
    return 0


# ---- h -------------------------------------------------------------------------------------------
def class_positions(fmt, tok, gen: list[int], punct: np.ndarray) -> dict[int, list[str]]:
    """Generated index -> the registered classes it belongs to (onset, clause_end, stride32)."""
    trk = SG.Tracker(fmt, tok, punct)
    out: dict[int, list[str]] = {}
    for i in range(len(gen)):
        g = gen[: i + 1]
        trk.update(g)
        cls = [k for k in ("onset", "clause_end") if trk.in_class(g, k)]
        if trk.in_class(g, "constant") and (i - trk.start) % STRIDE == 0:
            cls.append("stride32")
        if cls:
            out[i] = cls
    return out


def forward_bank(sc, ids: list[int], positions: list[int], letter_at: int | None,
                 logprob_from: int | None = None):
    """One single-row teacher-forced forward: residuals at ``positions`` for every banked layer
    [len(positions), n_layers, H] fp16, the letter log-probs predicted at ``letter_at`` (the
    position whose logits produce the answer letter), and the summed log-prob of ``ids`` from
    ``logprob_from`` on (the trace under the unadapted model)."""
    lay = sc.layers()
    if sc.dry:
        rng = np.random.default_rng(len(ids) * 7919 + (ids[-1] if ids else 0))
        resid = rng.standard_normal((len(positions), len(lay), 8)).astype(np.float16)
        lp = (sc.model.read_ids([ids[: letter_at + 1]], lay, 1)[0][0]
              if letter_at is not None else None)
        return resid, lp, 0.0 if logprob_from is not None else None
    import torch

    m = sc.model
    grabbed: dict[int, object] = {}
    hooks = [m.wb._get_layer_module(h - 1).register_forward_hook(  # noqa: SLF001
        lambda _m, _i, o, h=h: grabbed.__setitem__(h, (o[0] if isinstance(o, tuple) else o)[0]))
        for h in lay]
    try:
        x = torch.tensor([ids], device=m.model.device)
        with torch.no_grad():
            logits = m.model(input_ids=x).logits[0]
        pos = torch.tensor(positions, device=x.device, dtype=torch.long)
        resid = torch.stack([grabbed[h][pos] for h in lay], 1).float().cpu().numpy() \
            .astype(np.float16)
        lp = None
        if letter_at is not None:
            lids = L.letter_ids(sc)
            lp = torch.log_softmax(logits[letter_at].float(), -1)[lids].cpu().numpy()
        tlp = None
        if logprob_from is not None and logprob_from < len(ids):
            lps = torch.log_softmax(logits[logprob_from - 1: -1].float(), -1)
            tlp = float(lps.gather(1, x[0, logprob_from:, None]).sum())
    finally:
        for h in hooks:
            h.remove()
    return resid, lp, tlp


def phase_h(c: dict, a) -> int:
    import pod_st_screen as P
    import st_train as ST

    t_start = time.time()
    if a.dry_run:
        model = stub(c, a)
    else:
        import torch
        from kdg_pod_lib import ModelWrapper

        model = ModelWrapper(c["spec"]["repo"], c["spec"]["revision"], torch_dtype=torch.bfloat16,
                             attn_implementation="eager", **P.hf_load_kwargs(c["spec"]))
    sc = screen(c, model, a, read_batch=1)
    sc.unbatched = True  # single-row readouts of record (Amendment 9 item 2)
    rec = load_record(c)
    ph: dict = {"started": _now(), "layers_hidden_index": sc.layers(),
                "stack": _stack(), "single_row": True}
    rec["phases"]["h"] = ph
    g = P.gates(sc, c["spec"], c["split"], c["order"], c["H"], a.dry_run)
    ph["gates"] = g
    if not g["ok"]:
        ph["status"] = "gate_failed"
        save_record(c, rec)
        return 2
    rows = _jsonl(c["out"] / "gen" / "h1_rollouts.jsonl")
    punct = SG.punct_mask(c["tok"])
    cls_codes = {"onset": 1, "clause_end": 2, "stride32": 4, "decision_post": 8}
    bank_dir = c["out"] / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    idx_rows, buf, t_read = [], {"resid": [], "row": [], "pos": [], "cls": []}, []
    for i, r in enumerate(rows):
        if a.deadline and time.time() > a.deadline:
            ph["stopped"] = f"deadline at rollout {i}"
            break
        gen = r.get("gen_ids") or []
        if not gen:
            idx_rows.append({"row": i, "n_pos": 0})
            continue
        n_p = len(r["prompt_ids"])
        cp = class_positions(c["fmt"], c["tok"], gen, punct)
        dec = (n_p + r["answer_idx"] - 1) if r.get("token_identity") else None
        pos = sorted({n_p + k for k in cp} | ({dec} if dec is not None else set()))
        t0 = time.time()
        resid, lp, tlp = forward_bank(sc, r["prompt_ids"] + gen, pos, dec, logprob_from=n_p)
        t_read.append(time.time() - t0)
        code = [sum(cls_codes[k] for k in cp.get(p - n_p, [])) + (8 if p == dec else 0)
                for p in pos]
        buf["resid"].append(resid)
        buf["row"] += [i] * len(pos)
        buf["pos"] += pos
        buf["cls"] += code
        r["trace_logprob_unadapted"] = tlp
        if lp is not None:
            shown = sorted(r["order"])
            r["option_logps"] = {x: float(lp["ABCDE".index(x)]) for x in shown}
            r["option_mass"] = float(np.exp(list(r["option_logps"].values())).sum())
        idx_rows.append({"row": i, "n_pos": len(pos)})
        if (i + 1) % CHUNK == 0 or i == len(rows) - 1:  # per-chunk saving before going on
            _flush(bank_dir, i // CHUNK, buf)
            buf = {"resid": [], "row": [], "pos": [], "cls": []}
    if buf["resid"]:
        _flush(bank_dir, len(rows) // CHUNK + 1, buf)
    sc.save_jsonl("gen/h1_rollouts_read.jsonl", rows)
    sc.save_jsonl("bank/index.jsonl", idx_rows)
    ph["bank"] = {"blocks": list(BANK_BLOCKS), "classes": cls_codes, "stride": STRIDE,
                  "n_rollouts_read": len(idx_rows),
                  "n_positions": int(sum(x["n_pos"] for x in idx_rows)),
                  "read_seconds_per_rollout": float(np.mean(t_read)) if t_read else None}
    save_record(c, rec)
    # dose-0 on all ids x 8 permutations, residual at the pre-prefill END_OF_PROMPT token
    cand = c["fmt"].dose0[0]
    d0 = _letter_cell(sc, c, [c["H"][s] for s in c["ids"]], "dl_chat_neutral", cand, L.N_PERM,
                      c["out"] / "readout" / "dose0_h1")
    # judgment frame on H1_train, one permutation, at the same token: chat_sigma per layer
    tr_ids = [s for s in c["ids"] if s in c["train"]]
    jr, jres = _letter_cell(sc, c, [c["H"][s] for s in tr_ids], "jl_chat_neutral", cand, 1,
                            c["out"] / "readout" / "judgment_h1_train", return_resid=True)
    sig = {}
    for k, h in enumerate(sc.layers()):
        s, side = ST.chat_sigma(
            jres[:, k, :].astype(np.float64),
            {"model": f"{c['spec']['repo']}@{c['spec']['revision']}", "revision":
             c["spec"]["revision"], "format": "harmony, judgment frame (jl_chat_neutral)",
             "position_class": "pre-decision gate (END_OF_PROMPT, before the dose-0 prefill)",
             "layer": f"hidden index {h} (block {h - 1})", "split": "H1_train"})
        np.save(c["out"] / "readout" / f"chat_sigma_h{h}.npy", s)
        sig[str(h)] = side
    ph["chat_sigma"] = sig
    # selections (Amendment 1): on v1's rows (sample "selection") and the dose-0 readout
    pd = P.dose0_protect(d0)
    sel = {}
    for name, part in (("held", c["held"]), ("train", c["train"])):
        rr = [r for r in rows if r["scenario_id"] in part and r.get("norm_status") is not None]
        sel[name] = L.engaged_subset(rr, {s: v for s, v in pd.items() if s in part},
                                     use_dose0=True)
        sel[name]["brake"] = P.brake([r for r in rows if r["scenario_id"] in part],
                                     [x for x in d0 if x["scenario_id"] in part])
    (c["out"] / "select").mkdir(exist_ok=True)
    (c["out"] / "select" / "engaged.json").write_text(json.dumps(sel, indent=1,
                                                                 default=L._json_default))
    ph["engaged"] = {k: {"n_engaged": v["n_engaged"], "null": v.get("null_no_brake"),
                         "brake_B": v["brake"]["B"]} for k, v in sel.items()}
    tr_rows = [dict(r, action=ACTION.get(r.get("norm_status"))) for r in rows
               if r["scenario_id"] in c["train"]]
    pairs, pinfo = ST.dpo_pairs(tr_rows)
    lacking = [s for s in c["order"] if s in set(tr_ids) and pinfo["per_scenario"].get(s, 0) == 0]
    rider = lacking[:PAIR_RIDER_MAX] if not pinfo["meets_floor"] else []
    (c["out"] / "select" / "dpo_pairs.json").write_text(json.dumps(
        {"info": pinfo, "pair_rider": rider}, indent=1))
    ph["dpo"] = {**{k: pinfo[k] for k in ("n_pairs", "scenarios_with_pair", "floor",
                                          "meets_floor")}, "pair_rider_scenarios": len(rider)}
    ph["hf_memory_probe"] = _probe(sc, c, a)
    ph["finished"] = _now()
    ph["elapsed_h"] = (time.time() - t_start) / 3600
    ph["status"] = "ok" if "stopped" not in ph else "envelope_stop"
    lens = [len(r["prompt_ids"]) + len(r.get("gen_ids") or []) for r in rows]
    rec["probe_mean_len"] = float(np.mean(lens)) if lens else None  # the read's mean length
    rec["read_seconds_per_row"] = ph["bank"]["read_seconds_per_rollout"]
    rec["single_row_read_s"] = ph["bank"]["read_seconds_per_rollout"]
    rec["hf_memory_probe"] = ph["hf_memory_probe"]
    save_record(c, rec)
    print(f">> t11 h: banked {ph['bank']['n_positions']} positions over "
          f"{ph['bank']['n_rollouts_read']} rollouts; engaged held "
          f"{sel['held']['n_engaged']}, train {sel['train']['n_engaged']}; DPO pairs "
          f"{pinfo['n_pairs']} (rider {len(rider)})", flush=True)
    return 0 if ph["status"] == "ok" else 3


def _stack() -> dict:
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


def _flush(d: Path, k: int, buf: dict) -> None:
    np.savez(d / f"bank_{k:04d}.npz", resid=np.concatenate(buf["resid"]),
             row=np.asarray(buf["row"], np.int32), pos=np.asarray(buf["pos"], np.int32),
             cls=np.asarray(buf["cls"], np.int8))


def _letter_cell(sc, c, scenarios, cell: str, cand, nperm: int, stem: Path,
                 return_resid: bool = False):
    """The letter readout of ``cell`` (dose-0 prefill ``cand``), one row at a time; residual at
    the pre-prefill END_OF_PROMPT token (§1's decision class) at every banked layer."""
    rows, res = [], []
    for s in scenarios:
        for k in range(nperm):
            o = L.assign_letters(s, k)
            msgs = sc.messages(s, o, cell)
            bare = c["fmt"].prompt_ids(c["tok"], sc.render(msgs, **cand.render_kwargs))
            ids = c["fmt"].dose0_ids(c["tok"], msgs, sc.effort, cand, sc.date_pin)
            resid, lp, _ = forward_bank(sc, ids, [len(bare) - 1], len(ids) - 1)
            shown = [x for x, _ in o]
            olp = {x: float(lp["ABCDE".index(x)]) for x in shown}
            best = max(shown, key=lambda x: olp[x])
            opt = dict(o)[best]
            rows.append({"scenario_id": s.id, "cell": cell, "rollout": k, "order": L.letter_map(o),
                         "option_logps": olp,
                         "option_mass": float(np.exp(list(olp.values())).sum()),
                         "letter": best, "option_id": opt.option_id,
                         "norm_status": opt.norm_status,
                         "norm_status_by_letter": {x: o_.norm_status for x, o_ in o},
                         "dose0_candidate": cand.name, "effort": sc.effort})
            res.append(resid[0])
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.savez(stem.with_suffix(".npz"), resid_eop=np.stack(res),
             layers=np.array(sc.layers()))
    return (rows, np.stack(res)) if return_resid else rows


def _probe(sc, c, a) -> dict:
    """One warmed eager HF decode probe at batch 132 (the pinned experts path), T1.3's price."""
    if a.dry_run or a.no_probe:
        return {"dry": True, "path": "eager attention, eager experts (pinned)"}
    import engine_probes as EP

    prompts = [j["ids"] for j in L.h1_jobs(sc, c["H"], c["order"][:16], 1)]
    p = EP.hf_batch_probe(sc.model, prompts, 132, cap=L.CAP_OF_RECORD, decode_steps=16)
    s = EP.summarize([p])
    return {**s, "path": "eager attention, eager experts (pinned)"}


# ---- v2 ------------------------------------------------------------------------------------------
def phase_v2(c: dict, a) -> int:
    import analyze_st as AS
    import screen_vllm as SV

    sc = screen(c, stub(c, a) if a.dry_run else L.tok_only(c["tok"]), a)
    sel = json.loads((c["out"] / "select" / "engaged.json").read_text())
    dp = json.loads((c["out"] / "select" / "dpo_pairs.json").read_text())
    rec = load_record(c)
    choice = rec["phases"]["v1"]["timing"]["choice"]
    ph: dict = {"started": _now()}
    held = sel["held"]["engaged_ids"]
    t0 = time.time()
    if held:
        jobs = L.h1_jobs(sc, c["H"], held, R_RERUN)
        g = SV.gen_stage(sc, jobs, "t11_rerun", L.CAP_OF_RECORD, c["spec"], probe=False,
                         choice=choice, deadline=a.deadline)
        for r in g["rows"]:
            r["sample"], r["arm"] = "rerun", "none"
            r["rollout_key"] = f"t11_rerun|{r['scenario_id']}|{r['rollout']}"
        sc.save_jsonl("rerun/h1_held_engaged_rerun.jsonl", g["rows"])
        v1 = [r for r in _jsonl(c["out"] / "gen" / "h1_rollouts.jsonl")
              if r["scenario_id"] in set(held)]
        ph["rerun"] = {"n_scenarios": len(held), "n_rollouts": len(jobs),
                       "timing": g["timing"],
                       "floor_P_re_minus_P_sel": AS.rtm_floor(v1, g["rows"], held)}
    rider = dp.get("pair_rider") or []
    if rider:
        jobs = L.h1_jobs(sc, c["H"], rider, 8)
        g = SV.gen_stage(sc, jobs, "t11_pairs", L.CAP_OF_RECORD, c["spec"], probe=False,
                         choice=choice, deadline=a.deadline)
        for r, j in zip(g["rows"], jobs):
            r["sample"], r["arm"] = "pair_rider", "none"
            r["rollout_key"] = f"t11_pairs|{r['scenario_id']}|{r['rollout']}"
            r["prompt_ids"] = j["ids"]
        sc.save_jsonl("pairs/pair_rider.jsonl", g["rows"])
        ph["pair_rider"] = {"n_scenarios": len(rider), "n_rollouts": len(jobs),
                            "timing": g["timing"]}
    ph["seconds"] = time.time() - t0
    ph["finished"] = _now()
    rec["phases"]["v2"] = ph
    save_record(c, rec)
    fl = (ph.get("rerun") or {}).get("floor_P_re_minus_P_sel") or {}
    print(f">> t11 v2: re-run on {len(held)} held engaged scenarios; floor {fl.get('mean')} "
          f"{fl.get('ci95')}; pair rider {len(rider)}", flush=True)
    return 0


# ---- qlora ---------------------------------------------------------------------------------------
def phase_qlora(c: dict, a) -> int:
    """T0.6 item 3's first run (author): Muse 4-bit, LoRA on the text tower, a synthetic anchor
    target for a few steps, the adapter merged into a bf16 copy, merged vs adapted logits."""
    rec = load_record(c)
    ph: dict = {"started": _now(), "model": QLORA_KEY}
    rec["phases"]["qlora"] = ph
    if a.dry_run:
        ph["dry"] = True
        save_record(c, rec)
        return 0
    import gc

    import pod_st_screen as P
    import st_train as ST
    import torch
    from peft import PeftModel

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.steering import objective_lora as OL

    spec = P.registry()[QLORA_KEY]
    t0 = time.time()
    wb = OL.qlora_model(spec["repo"], spec["revision"], access_tier=AccessTier.WEIGHTS)
    ph["load_4bit_s"] = time.time() - t0
    tok = wb.tokenizer
    text = "Should the clerk follow the rule or help the customer? Let me think it through."
    ids = tok.encode(text, add_special_tokens=False)
    hid = wb.model.config.get_text_config().hidden_size
    axis = np.random.default_rng(0).standard_normal(hid).astype(np.float32)
    axis /= np.linalg.norm(axis)
    tr = [{"ids": ids, "anchor_mask": [False] * 4 + [True] * (len(ids) - 4),
           "kl_mask": [False] * len(ids), "scenario_id": "synthetic"}]
    layer = int(spec["n_layers"]) // 2
    r = ST.run(wb, "anchor", layer=layer, traces=tr, axis=axis, tau=0.5, max_steps=6, lr=5e-4)
    ph["steps"] = [{"rep": s["rep"], "kl": s["kl"]} for s in r["steps"]]
    ph["lora"] = {k: r["lora"][k] for k in ("rank", "trainable_params")}
    ph["peak_gb_4bit"] = torch.cuda.max_memory_allocated() / 1e9
    adir = c["out"] / "qlora" / "adapter"
    wb.model.save_pretrained(str(adir))
    del wb, r
    gc.collect()
    torch.cuda.empty_cache()
    t1 = time.time()
    base = WhiteBoxModel(spec["repo"], revision=spec["revision"], access_tier=AccessTier.WEIGHTS,
                         torch_dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(base.model, str(adir))
    x = torch.tensor([ids], device=base.model.device)
    with torch.no_grad():
        want = peft(input_ids=x).logits.float()
        merged = peft.merge_and_unload()
        got = merged(input_ids=x).logits.float()
    ph["merge_bf16_s"] = time.time() - t1
    ph["merged_vs_adapted_max_abs_logit"] = float((want - got).abs().max())
    ph["merged_vs_adapted_ok"] = ph["merged_vs_adapted_max_abs_logit"] <= 0.5  # bf16 merge
    ph["finished"] = _now()
    save_record(c, rec)
    print(f">> t11 qlora (Muse): steps {[round(s['rep'], 4) for s in ph['steps']]}; merged vs "
          f"adapted max |logit| {ph['merged_vs_adapted_max_abs_logit']:.4f}", flush=True)
    return 0 if ph["merged_vs_adapted_ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("v1", "h", "v2", "qlora"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds (envelope)")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    c = context(a)
    return {"v1": phase_v1, "h": phase_h, "v2": phase_v2, "qlora": phase_qlora}[a.phase](c, a)


if __name__ == "__main__":
    sys.exit(main())
