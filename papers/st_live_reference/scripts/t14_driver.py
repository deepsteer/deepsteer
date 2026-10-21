"""The T1.4 pod (ST_PLAN §5.1-5.3, Amendments 14-16; author, 2026-10-21): two gate arms, then the
three pre-registered adapters on GPT-OSS-20B, then the conditional fourth, in one HF process on the
ST stack of record (torch 2.13 in the vLLM venv, eager attention and experts, bf16).

Phases, recorded and skipped on resume once finished:

- **gate_arms** (Amendment 16 items 1-2): +care_harm and +V_harm at clause_end, block 12, alpha
  267.3, on the gate's 20 x 8, against the T1.3 gate's ``none`` and 267.3 ``fill`` rows
  (reused); section 5.0's rule under Amendment 13's coherence bars; the readings and the fourth
  adapter's axis.
- **beh / anchor / self** (sections 5.1-5.3): LoRA (``RECIPE``) on the T1.1 on-policy H1_train
  traces; DPO on the protect-vs-rule pairs; the anchor hinge at clause_end, block 12, tau 0.0589;
  the standardized MSE to the scenario's judgment-frame residual with chat_sigma at block 12; the
  KL-to-base term on the general slice and, for anchor and self, the trace tokens outside the class.
- **fourth** (Amendment 16 item 3): the anchor hinge on the passing axis with its own tau; skipped
  when no gate arm passed.

Each adapter: a timed pilot of ``PILOT_MICRO`` micro-steps (gradients discarded) projects the arm's
time; the pre-registered fallback cuts ``max_steps`` to fit ``budget_h`` (schedule set from the cut
value) and records it. Held-out monitoring (never selection, never early stopping): every
``eval_every`` steps, the anchor-axis and care_harm margins on 16 H1_held traces, the DPO implicit
reward margin on 8 H1_held pairs and the KL on the 16 reserved general sequences. Saved: the adapter
(PEFT) and its sidecar (``st_train.save_adapter``). After each adapter the LoRA is unloaded and the
base model's logits on a probe must equal the pre-training ones exactly.

    python3 t14_driver.py --out <dir> --inputs <t14 inputs dir> --t11 <t11 dir> --t13 <t13 dir>
        [--dry-run] [--deadline <epoch>]
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
L_STAR = 13  # hidden-state index (block 12)
CLASS = "clause_end"
GATE_ALPHA = 267.301171875  # the gate of record's alpha (Amendment 13)
ACTION = {"violating": "protect", "consistent": "rule"}
# Proposed (author to confirm before the pod): matched optimizer steps across arms
RECIPE = {"rank": 16, "alpha": 32, "dropout": 0.0, "lr": 1e-4, "warmup": 10, "grad_accum": 4,
          "max_steps": 264, "kl_weight": 1.0, "budget_h": 1.0, "eval_every": 66,
          "general_train": 1984, "general_monitor": 16, "gradient_checkpointing": True,
          "beh_ckpt_epochs": 3}
PILOT_MICRO = 3
SEEDS = {"beh": 101, "anchor": 102, "self": 103, "fourth": 104}


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
    meta = json.loads((a.inputs / "t14_inputs.json").read_text())
    z = np.load(a.inputs / "t14_inputs.npz")
    out = a.out / KEY
    out.mkdir(parents=True, exist_ok=True)
    c = {"P": P, "spec": spec, "tok": tok, "fmt": fmt, "split": split, "out": out,
         "H": {s.id: s for s in Hs if s.id in kept}, "meta": meta,
         "inp": {k: z[k] for k in z.files}, "layer": L_STAR, "cls": CLASS,
         "cap": L.CAP_OF_RECORD, "batch": 160, "recipe": dict(RECIPE),
         "rollouts": _jsonl(a.t11 / "gen" / "h1_rollouts_read.jsonl"),
         "gate": list(meta["gate"]), "t13": a.t13}
    if a.dry_run:
        c["recipe"].update(max_steps=4, grad_accum=1, eval_every=2, warmup=1, general_train=4,
                           general_monitor=2, gradient_checkpointing=False, beh_ckpt_epochs=0.3)
        if "reserved" not in meta:  # three train scenarios with both actions stand in
            both = {}
            for r in c["rollouts"]:
                if r["scenario_id"] in set(c["split"]["H1_train"]):
                    both.setdefault(r["scenario_id"], set()).add(r.get("norm_status"))
            meta["reserved"] = sorted(sid for sid, v in both.items()
                                      if {"violating", "consistent"} <= v)[:3]
        c.update(layer=2, cap=16, batch=8, gate=c["gate"][:2], dry=True)
    return c


def load_model(c: dict, a):
    import torch

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    if a.dry_run:  # a tiny GPT-OSS with the real tokenizer; inputs at its width
        sys.path.insert(0, str(HERE.parents[2] / "tests" / "scripts"))
        import st_tiny

        path = st_tiny.build(KEY, c["out"] / "_tiny")
        wb = WhiteBoxModel(str(path), access_tier=AccessTier.WEIGHTS, device="cpu",
                           torch_dtype=torch.float32)
        h = wb.model.config.hidden_size
        rng = np.random.default_rng(0)

        def unit(v):
            return (v / np.linalg.norm(v)).astype(np.float32)
        c["inp"] = {"anchor_axis": unit(rng.standard_normal(h)),
                    "care_harm": unit(rng.standard_normal(h)),
                    "V_harm": unit(rng.standard_normal(h)),
                    "chat_sigma": np.ones(h, np.float32),
                    "self_ids": np.array(sorted({r["scenario_id"] for r in c["rollouts"]})),
                    "self_src": rng.standard_normal((len({r["scenario_id"] for r in
                                                          c["rollouts"]}), h)).astype(np.float32)}
        fmt = c["fmt"]  # a random model never writes the harmony header: open the trace at once
        object.__setattr__(fmt, "trace_open", lambda tok, g: 0)
        object.__setattr__(fmt, "trace_closes", lambda tok, g: False)
        return wb
    return WhiteBoxModel(c["spec"]["repo"], revision=c["spec"]["revision"],
                         access_tier=AccessTier.WEIGHTS, torch_dtype=torch.bfloat16,
                         attn_implementation="eager", **c["P"].hf_load_kwargs(c["spec"]))


class _Gen:
    """The decode loop's view of a WhiteBoxModel (``.model``, ``.tok``, ``.wb``)."""

    def __init__(self, wb):
        self.wb, self.model, self.tok = wb, wb.model, wb.tokenizer


def record_path(c: dict) -> Path:
    return c["out"] / "t14_record.json"


def load_record(c: dict) -> dict:
    p = record_path(c)
    return json.loads(p.read_text()) if p.exists() else {"model": KEY, "phases": {}}


def save_record(c: dict, rec: dict) -> None:
    record_path(c).write_text(json.dumps(rec, indent=1, default=L._json_default))


# ---- data -----------------------------------------------------------------------------------
def train_rows(c: dict, split: str = "train") -> list[dict]:
    """``train``: H1_train minus the reserved monitoring scenarios (Amendment 17); ``reserved``:
    those 10."""
    res = set(c["meta"]["reserved"])
    ids = (set(c["split"]["H1_train"]) - res) if split == "train" else res
    rows = [r for r in c["rollouts"] if r["scenario_id"] in ids and r.get("gen_ids")]
    if c.get("dry"):  # a few short rows on the tiny model; the pairs need both actions
        keep = [r for r in rows if r.get("norm_status") in ("violating", "consistent")]
        by: dict[str, list] = {}
        for r in keep:
            by.setdefault(r["scenario_id"], []).append(r)
        both = [v for v in by.values() if {x["norm_status"] for x in v} == {"violating",
                                                                              "consistent"}]
        rows = [dict(r, gen_ids=r["gen_ids"][:24]) for v in both[:3] for r in v[:4]]
    for r in rows:
        r["action"] = ACTION.get(r.get("norm_status"))
    return rows


def traces(c: dict, rows: list[dict]) -> list[dict]:
    """The anchoring class's positions and the KL positions (trace tokens outside the class)."""
    import st_train as ST

    punct = SG.punct_mask(c["tok"])
    out = []
    for r in rows:
        am, km = ST.trace_masks(c["fmt"], c["tok"], r["prompt_ids"], r["gen_ids"], c["cls"], punct)
        if any(am):
            out.append({"ids": r["prompt_ids"] + r["gen_ids"], "anchor_mask": am, "kl_mask": km,
                        "scenario_id": r["scenario_id"]})
    return out


def pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    import st_train as ST

    return ST.dpo_pairs(rows)


def general(c: dict, a) -> tuple[list[list[int]], list[list[int]]]:
    import general_slice as G

    n_tr, n_mon = c["recipe"]["general_train"], c["recipe"]["general_monitor"]
    if a.dry_run:
        seqs = [c["tok"].encode(f"General text number {i}. The weather was mild.",
                                add_special_tokens=False) for i in range(n_tr + n_mon)]
    else:
        seqs = G.load(c["tok"])
    return seqs[:n_tr], seqs[len(seqs) - n_mon:]


# ---- monitoring (held-out; never selection) -------------------------------------------------
def monitor_fn(c: dict, wb, held_tr: list[dict], held_pairs: list[dict],
               gen_mon: list[list[int]]):
    import st_train as ST
    import torch

    from deepsteer.steering import objective_lora as OL

    ax = {k: torch.as_tensor(c["inp"][k], device=wb.model.device) for k in ("anchor_axis",
                                                                            "care_harm")}

    def fn(_step: int) -> dict:
        out: dict = {}
        with torch.no_grad():
            margins: dict[str, list[float]] = {k: [] for k in ax}
            for t in held_tr:
                x = torch.tensor([t["ids"]], device=wb.model.device)
                _, h = OL._forward(wb, x, c["layer"])  # noqa: SLF001
                hs = h[0][torch.tensor(t["anchor_mask"], device=x.device)].float()
                for k, v in ax.items():
                    margins[k].append(float(((hs @ v.float()) / hs.norm(dim=-1)).mean()))
            out.update({f"reserved_margin_{k}": float(np.mean(v)) for k, v in margins.items()})
            dm = [OL.dpo_loss(wb, p["prompt"], p["chosen"], p["rejected"], beta=ST.DPO_BETA)[1]
                  for p in held_pairs]
            out["reserved_dpo_margin"] = float(np.mean([d["dpo_margin"] for d in dm])) \
                if dm else None
            out["monitor_kl"] = float(np.mean([float(OL.kl_general(wb, s)) for s in gen_mon]))
        return out
    return fn


# ---- phases ---------------------------------------------------------------------------------
def phase_gate_arms(c, sc, wb, rec, a) -> dict:
    import t13_driver as T13

    g = {"out": c["out"], "fmt": c["fmt"], "cap": c["cap"], "layer": c["layer"], "cls": c["cls"],
         "batch": c["batch"]}
    jobs = L.h1_jobs(sc, c["H"], c["gate"], 8)
    none = _jsonl(c["t13"] / "gen" / "gate_none.jsonl")
    fill = _jsonl(c["t13"] / "gen" / "gate_fill_rerun.jsonl")
    if a.dry_run:
        none, fill = none[: len(jobs)], fill[: len(jobs)]
    keys = [f"t13|{j['s'].id}|{j['k']}" for j in jobs]
    if [r["rollout_key"] for r in none] != keys or [r["rollout_key"] for r in fill] != keys:
        raise SystemExit("the T1.3 none / fill rows do not match the gate's jobs")
    dpf = T13.delta_p(fill, none)
    arms = {}
    for name in ("care_harm", "V_harm"):
        rows = T13.run_arm(g, sc, _Gen(wb), f"gate_{name}", jobs,
                           SG.Add(c["inp"][name], GATE_ALPHA if not a.dry_run else 1.0),
                           extra=lambda i, n=name: {"axis": n})
        coh = T13._coh(rows, none, relative=True)  # noqa: SLF001
        dp = T13.delta_p(rows, none)
        passes = bool(coh["passes"] and dp["mean"] is not None and dpf["mean"] is not None
                      and dp["mean"] >= T13.GATE_DP and dp["mean"] - dpf["mean"]
                      >= T13.GATE_OVER_FILL)
        arms[name] = {"coherence": coh, "dP": dp, "passes": passes}
    reading, fourth = gate_reading(arms)
    return {"status": "ok", "alpha": GATE_ALPHA, "fill_dP": dpf, "arms": arms,
            "reading": reading, "fourth_axis": fourth, "rule": "Amendment 16 items 1-3"}


def gate_reading(arms: dict) -> tuple[str, str | None]:
    """Amendment 16 item 2's reading and item 3's axis for the fourth adapter: harm content if
    V_harm passes, care beyond harm if only care_harm does, neither otherwise; the fourth adapter
    takes the passing arm with the larger dP point estimate."""
    vh, ch = arms["V_harm"]["passes"], arms["care_harm"]["passes"]
    reading = "harm content" if vh else "care beyond harm" if ch else "neither"
    passing = [n for n in arms if arms[n]["passes"]]
    return reading, (max(passing, key=lambda n: arms[n]["dP"]["mean"]) if passing else None)


def _probe_logits(wb, ids: list[int]):
    import torch

    with torch.no_grad():
        return wb.model(input_ids=torch.tensor([ids], device=wb.model.device)).logits.float().cpu()


def _train(c, sc, wb, rec, a, name: str) -> dict:
    """One adapter: inject, pilot, train with monitoring, save, unload, base check."""

    import st_train as ST


    rc = c["recipe"]
    rows_tr, rows_res = train_rows(c, "train"), train_rows(c, "reserved")
    gen_tr, gen_mon = general(c, a)
    rng = np.random.default_rng(SEEDS[name])
    held_tr = traces(c, [rows_res[i] for i in rng.permutation(len(rows_res))[:16]])
    held_pairs = pairs(rows_res)[0]
    probe = gen_mon[0]
    base_logits = _probe_logits(wb, probe)
    objective = {"beh": "beh", "anchor": "anchor", "self": "self", "fourth": "anchor"}[name]
    tr, pr, info = (None, None, {})
    if objective == "beh":
        pr, info = pairs(rows_tr)
    else:
        tr = traces(c, rows_tr)
    axis_name = None
    tau = None
    targets = sigma = None
    if objective == "anchor":
        axis_name = "anchor_axis" if name == "anchor" else rec["phases"]["gate_arms"]["fourth_axis"]
        tau = (c["meta"]["tau"]["anchor_axis"] if name == "anchor"
               else c["meta"]["tau"][axis_name])
        if a.dry_run:
            tau = 0.5
    if objective == "self":
        ids = [str(x) for x in c["inp"]["self_ids"]]
        targets = {s: c["inp"]["self_src"][ids.index(s)] for s in {t["scenario_id"] for t in tr}}
        sigma = c["inp"]["chat_sigma"]
    if rc["gradient_checkpointing"]:
        wb.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":
                                                                              False})
    kw = dict(layer=c["layer"], traces=tr, pairs=pr, general=gen_tr, kl_weight=rc["kl_weight"],
              axis=c["inp"][axis_name] if axis_name else None, tau=tau, targets=targets,
              sigma=sigma, lr=rc["lr"], grad_accum=rc["grad_accum"], seed=SEEDS[name],
              lora_rank=rc["rank"], lora_alpha=rc["alpha"], lora_dropout=rc["dropout"])
    # pilot: one optimizer step over PILOT_MICRO micro-batches on a throwaway adapter, timed;
    # the adapter is unloaded before the real run injects a fresh one
    tp = time.time()
    ST.run(wb, objective, max_steps=1, **{**kw, "grad_accum": PILOT_MICRO})
    pilot = {"micro": PILOT_MICRO, "seconds": time.time() - tp}
    pilot["seconds_per_micro"] = pilot["seconds"] / PILOT_MICRO
    wb._model = wb.model.unload()  # noqa: SLF001
    if rc["gradient_checkpointing"]:
        wb.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    per_step = pilot["seconds_per_micro"] * rc["grad_accum"]
    fit = int(rc["budget_h"] * 3600 / max(per_step, 1e-9))
    steps = min(rc["max_steps"], max(fit, 1))
    t0 = time.time()
    mon = monitor_fn(c, wb, held_tr, held_pairs, gen_mon)
    ckpt_step = None
    if objective == "beh":  # Amendment 17 (a): the epoch-3 snapshot of the step-matched run
        ckpt_step = int(np.ceil(rc["beh_ckpt_epochs"] * len(pr) / rc["grad_accum"]))
        if ckpt_step >= steps:
            ckpt_step = None  # the run ends before epoch 3: only the final adapter exists

    def eval_fn(step: int) -> dict:
        out = mon(step)
        if ckpt_step is not None and step == ckpt_step:
            e3 = c["out"] / "adapters" / "beh_epoch3"
            e3.mkdir(parents=True, exist_ok=True)
            wb.model.save_pretrained(str(e3))
            out["saved"] = str(e3)
        return out
    res = ST.run(wb, objective, max_steps=steps, eval_every=rc["eval_every"], eval_fn=eval_fn,
                 eval_at=[ckpt_step] if ckpt_step else (), **kw)
    adir = c["out"] / "adapters" / name
    sidecar = {"arm": name, "objective": objective, "layer_hidden_index": c["layer"],
               "class": c["cls"], "tau": tau, "axis": axis_name, "recipe": rc,
               "pilot": pilot, "max_steps_planned": rc["max_steps"], "max_steps_run": steps,
               "pairs": info if objective == "beh" else None,
               "n_traces": len(tr) if tr else None,
               "sigma_provenance": c["meta"].get("chat_sigma_provenance") if sigma is not None
               else None, "inputs_sha256": c["meta"].get("sha256")}
    sidecar["epoch3_step"] = ckpt_step
    ST.save_adapter(wb, adir, res, sidecar)
    if rc["gradient_checkpointing"]:
        wb.model.gradient_checkpointing_disable()
    select = beh_select(c, sc, wb, a, adir, ckpt_step) if objective == "beh" else None
    wb._model = wb.model.unload()  # noqa: SLF001  the base without any adapter
    drift = float((_probe_logits(wb, probe) - base_logits).abs().max())
    same = drift <= 1e-3
    if not same:
        raise SystemExit(f"{name}: base logits moved by {drift} after unloading the adapter")
    return {"status": "ok", "objective": objective, "axis": axis_name, "tau": tau,
            "max_steps_run": steps, "pilot": pilot, "train_seconds": time.time() - t0,
            "final": res["steps"][-1] if res["steps"] else None, "monitor": res["eval"],
            "adapter_dir": str(adir), "base_unchanged": same, "base_drift_max_abs": drift,
            "epoch3_step": ckpt_step, "comparator_selection": select}


def beh_select(c, sc, wb, a, adir: Path, ckpt_step: int | None) -> dict:
    """Amendment 17 (a): the comparator of record is whichever L-beh checkpoint (step-matched
    final, epoch 3) has the higher protective share P on the reserved scenarios x 8 (generation,
    parser of record); ties go to the step-matched run. Chosen here, before any H1_held row exists
    for either adapter."""
    import t13_driver as T13
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    jobs = L.h1_jobs(sc, c["H"], c["meta"]["reserved"], 8)
    import zlib

    seeds = [zlib.crc32(f"t14sel|{j['s'].id}|{j['k']}".encode()) % (2**31) for j in jobs]

    def p_of(tag: str) -> dict:
        rows = SG.generate(_Gen(wb), [j["ids"] for j in jobs], fmt=c["fmt"], cap=c["cap"],
                           seeds=seeds, batch=c["batch"],
                           on_finish=SG.jsonl_saver(c["out"] / "gen" / f"select_{tag}.raw.jsonl"))
        parsed = L.build_rows(sc, jobs, [r["gen_ids"] for r in rows], seeds, cap=c["cap"],
                              stage="t14_select", engine="hf_adapter")
        sc.save_jsonl(f"gen/select_{tag}.jsonl", parsed)
        ps = T13.p_by_scenario(parsed)
        return {"P": float(np.mean(list(ps.values()))) if ps else None, "n_scenarios": len(ps),
                "parse_rate": float(np.mean([r["letter"] is not None for r in parsed]))}
    out = {"step_matched": p_of("beh_final")}
    if ckpt_step is not None:
        e3 = c["out"] / "adapters" / "beh_epoch3"
        set_peft_model_state_dict(wb.model, load_file(str(e3 / "adapter_model.safetensors")))
        out["epoch3"] = p_of("beh_epoch3")
        set_peft_model_state_dict(wb.model, load_file(str(adir / "adapter_model.safetensors")))
    pf, pe = out["step_matched"]["P"], (out.get("epoch3") or {}).get("P")
    out["comparator"] = ("epoch3" if pe is not None and pf is not None and pe > pf
                         else "step_matched")
    out["rule"] = "Amendment 17 (a): higher P on the reserved scenarios; ties to step-matched"
    return out


def phase_beh(c, sc, wb, rec, a) -> dict:
    return _train(c, sc, wb, rec, a, "beh")


def phase_anchor(c, sc, wb, rec, a) -> dict:
    return _train(c, sc, wb, rec, a, "anchor")


def phase_self(c, sc, wb, rec, a) -> dict:
    return _train(c, sc, wb, rec, a, "self")


def phase_fourth(c, sc, wb, rec, a) -> dict:
    if not (rec["phases"].get("gate_arms") or {}).get("fourth_axis"):
        return {"status": "ok", "ran": False, "reason": "no gate arm passed (Amendment 16)"}
    return {**_train(c, sc, wb, rec, a, "fourth"), "ran": True}


PHASES = {"gate_arms": phase_gate_arms, "beh": phase_beh, "anchor": phase_anchor,
          "self": phase_self, "fourth": phase_fourth}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--t11", type=Path, required=True, help="T1.1's gpt_oss_20b output dir")
    ap.add_argument("--t13", type=Path, required=True, help="T1.3's gpt_oss_20b output dir")
    ap.add_argument("--phases", default="gate_arms,beh,anchor,self,fourth")
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds (envelope)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    c = context(a)
    wb = load_model(c, a)
    sc = L.Screen(KEY, c["spec"], _Gen(wb), c["fmt"], c["out"], a.date_pin, read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    rec = load_record(c)
    rec.setdefault("inputs", c["meta"])
    rec["recipe"] = c["recipe"]
    if not a.dry_run and "gates" not in rec:
        g = c["P"].gates(sc, c["spec"], c["split"], L.h1_train_order(c["split"]), c["H"], False)
        rec["gates"] = g
        save_record(c, rec)
        if not g["ok"]:
            return 2
    for name in a.phases.split(","):
        done = rec["phases"].get(name) or {}
        if done.get("finished") and not a.force:
            print(f">> t14 {name}: finished earlier ({done['finished']}); skipped", flush=True)
            continue
        need = c["recipe"]["budget_h"] * 3600 * 1.2 if name in ("beh", "anchor", "self",
                                                                "fourth") else 0
        if a.deadline and time.time() + need > a.deadline:
            rec["phases"][name] = {"status": "deadline", "at": _now()}
            save_record(c, rec)
            print(f">> t14 {name}: not started, the deadline leaves too little time", flush=True)
            return 3
        t0 = time.time()
        res = PHASES[name](c, sc, wb, rec, a)
        rec["phases"][name] = {**res, "seconds": time.time() - t0,
                               "finished": _now() if res.get("status") == "ok" else None}
        save_record(c, rec)
        brief = {k: res[k] for k in ("reading", "fourth_axis", "max_steps_run", "ran",
                                     "base_unchanged") if k in res}
        print(f">> t14 {name}: done ({json.dumps(brief)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
