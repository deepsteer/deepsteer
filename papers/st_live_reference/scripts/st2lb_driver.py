"""L-beh-2 on GPT-OSS-20B (ST_PLAN Amendment 31 item 3 and its addition; author, 2026-10-23):
T1.4's L-beh DPO recipe with rule-right pairs added in equal number, read by the full T1.5 gate.
Three steps, each its own process:

- ``rr_gen`` (vLLM): the 130 rule-right primaries of ``data/st_lbeh2_pool_v1.json`` (outside the
  specificity set and the panel guard's half) x 8 in the agent frame at effort medium, cap 1,536.
- ``train`` (HF): T1.1's H1_train DPO pairs (protect over rule; the reserved scenarios out, as in
  T1.4) plus rule-right pairs (rule over violating; per item up to 2, in rollout order), the
  rule-right set cut to the H1 count by a seeded draw (all of it if fewer, the shortfall recorded);
  trained by T1.4's ``_train`` as arm ``beh2`` (L-beh's recipe, its step budget, seed 105).
- ``gate``: T1.5's phases on the adapter, each its own process: ``base_hf``, ``base_vllm``
  (the ``none`` re-run on the bf16 engine), ``hf_beh2`` (parity, specificity, the panel guard,
  the merge), ``vllm_beh2`` (H1_held x 8 and capability on the merged engine), and T1.5's HF
  fallbacks if a parity check fails.

The analysis of record is local (``analyze_st2lb.py``).

    python3 st2lb_driver.py --step rr_gen --out <dir> [--dry-run]
    python3 st2lb_driver.py --step train --out <dir> --t14-inputs <dir> --t11 <dir> [--dry-run]
    python3 st2lb_driver.py --step gate --out <dir> --t14-inputs <dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import screen_lib as L  # noqa: E402

KEY = "gpt_oss_20b"
DATA = HERE.parent / "data"
ARM = "beh2"
R = 8
PAIRS_PER_ITEM = 2
PAIR_SEED = 20261023


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def record_path(out: Path) -> Path:
    return out / KEY / "st2lb_record.json"


def update_record(out: Path, step: str, res: dict) -> None:
    p = record_path(out)
    rec = json.loads(p.read_text()) if p.exists() else {"model": KEY, "steps": {}}
    rec["steps"][step] = {**res, "finished": _now()}
    p.write_text(json.dumps(rec, indent=1, default=L._json_default))  # noqa: SLF001


def pool_scenarios(dry: bool) -> list:
    from deepsteer.kdg.schema import load_scenario_dir

    ids = json.loads((DATA / "st_lbeh2_pool_v1.json").read_text())["ids"]
    allp, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                     "*_scenarios_*.json"))))
    by = {s.id: s for s in L.panel_586(allp)}
    return [by[i] for i in (ids[:4] if dry else ids)]


def step_rr_gen(a) -> dict:
    import pod_st_screen as P
    import screen_vllm as SV
    from reasoning_formats import build_format
    from screen_stub import ScreenStub
    from transformers import AutoTokenizer

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    model = ScreenStub(KEY, tok, fmt, spec["n_layers"]) if a.dry_run else L.tok_only(tok)
    out = a.out / KEY
    sc = L.Screen(KEY, spec, model, fmt, out, a.date_pin, dry=a.dry_run)
    sc.effort = spec.get("effort_of_record") or fmt.effort_ladder[0]
    items = pool_scenarios(a.dry_run)
    jobs = L.h1_jobs(sc, {s.id: s for s in items}, [s.id for s in items], R)
    g = SV.gen_stage(sc, jobs, "st2lb_rr", L.CAP_OF_RECORD, spec, probe=True,
                     deadline=a.deadline)
    for r, j in zip(g["rows"], jobs):
        r.update(prompt_ids=j["ids"], date_pin=a.date_pin, sample="rule_right_pool")
    sc.save_jsonl("gen/rr_rollouts.jsonl", g["rows"])
    return {"status": "ok", "n": len(jobs), "n_items": len(items), "timing": g["timing"],
            "summary": L.screen_summary(g["rows"])}


def rule_right_pairs(rows: list[dict], dry: bool = False) -> list[dict]:
    """Per item: chosen = a rule (``consistent``) rollout, rejected = a ``violating`` one, paired
    in rollout order, at most ``PAIRS_PER_ITEM`` per item; ``neutral`` and unparsed never pair."""
    by: dict[str, dict[str, list]] = {}
    for r in sorted(rows, key=lambda x: (x["scenario_id"], x["rollout"])):
        if r.get("norm_status") in ("consistent", "violating") and r.get("gen_ids"):
            by.setdefault(r["scenario_id"], {"consistent": [], "violating": []})[
                r["norm_status"]].append(r)
    out = []
    for sid, d in sorted(by.items()):
        for i in range(min(len(d["consistent"]), len(d["violating"]), PAIRS_PER_ITEM)):
            cut = (lambda g: g[:24]) if dry else (lambda g: g)  # noqa: E731
            out.append({"scenario_id": sid, "prompt": d["consistent"][i]["prompt_ids"],
                        "chosen": cut(d["consistent"][i]["gen_ids"]),
                        "rejected": cut(d["violating"][i]["gen_ids"]), "kind": "rule_right"})
    return out


def cut_to(pairs: list[dict], n: int) -> tuple[list[dict], dict]:
    """The rule-right set cut to ``n`` by a seeded draw; all of it if fewer (shortfall recorded)."""
    if len(pairs) <= n:
        return pairs, {"available": len(pairs), "used": len(pairs), "shortfall": n - len(pairs)}
    pick = sorted(random.Random(PAIR_SEED).sample(range(len(pairs)), n))
    return [pairs[i] for i in pick], {"available": len(pairs), "used": n, "shortfall": 0}


def step_train(a) -> dict:
    import t14_driver as T14

    ns = argparse.Namespace(out=a.out, inputs=a.t14_inputs, t11=a.t11, t13=a.t11,
                            dry_run=a.dry_run, date_pin=a.date_pin, deadline=a.deadline)
    c = T14.context(ns)
    wb = T14.load_model(c, ns)
    sc = L.Screen(KEY, c["spec"], T14._Gen(wb), c["fmt"], c["out"], a.date_pin,  # noqa: SLF001
                  read_batch=1)
    sc.effort = c["spec"].get("effort_of_record") or c["fmt"].effort_ladder[0]
    n_h1 = len(T14.pairs(T14.train_rows(c, "train"))[0])
    rows = _jsonl(a.out / KEY / "gen" / "rr_rollouts.jsonl")
    if a.dry_run:  # the stub's letters are arbitrary: label alternately so both actions exist
        for i, r in enumerate(rows):
            r["norm_status"] = ("consistent", "violating")[i % 2]
    rr_all = rule_right_pairs(rows, a.dry_run)
    rr, info = cut_to(rr_all, n_h1)
    if not rr:
        raise SystemExit("no rule-right pair: every pool item answered one way")
    rec = T14.load_record(c)
    res = T14._train(c, sc, wb, rec, ns, ARM, extra_pairs=rr)  # noqa: SLF001
    rec = T14.load_record(c)
    rec["phases"][ARM] = {**res, "finished": _now()}
    T14.save_record(c, rec)
    return {"status": "ok", "n_h1_pairs": n_h1, "rule_right": info,
            "rule_right_items": len({p["scenario_id"] for p in rr}),
            "max_steps_run": res["max_steps_run"], "adapter_dir": res["adapter_dir"]}


def step_gate(a) -> dict:
    """T1.5's phases on ``beh2``, each its own process; T1.5's fallbacks on a failed check."""
    out = a.out / KEY
    rp = out / "t15_record.json"
    rec = json.loads(rp.read_text()) if rp.exists() else {"model": KEY, "phases": {}}
    rec["panel_arms_extra"] = [ARM]  # the guard reads L-beh-2 (Amendment 31 item 3)
    rp.write_text(json.dumps(rec, indent=1))

    def run(phase: str) -> int:
        if a.deadline and time.time() > a.deadline:
            print(f">> st2lb gate {phase}: not started, past the deadline", flush=True)
            return 3
        cmd = [sys.executable, str(HERE / "t15_driver.py"), "--phase", phase, "--out",
               str(a.out), "--t14-inputs", str(a.t14_inputs), "--adapters",
               str(out / "adapters"), "--date-pin", a.date_pin]
        cmd += ["--dry-run"] if a.dry_run else []
        cmd += ["--deadline", str(a.deadline)] if a.deadline else []
        return subprocess.run(cmd).returncode

    done = []
    for phase in ("base_hf", "base_vllm"):
        if (rc := run(phase)) != 0:
            return {"status": f"stopped at {phase} (rc {rc})", "phases": done}
        done.append(phase)
    ph = json.loads(rp.read_text())["phases"]
    if not ph["base_vllm"].get("base_check_ok"):
        if (rc := run("hfgen_none")) != 0:
            return {"status": f"stopped at hfgen_none (rc {rc})", "phases": done}
        done.append("hfgen_none")
    for phase in (f"hf_{ARM}", f"vllm_{ARM}"):
        if (rc := run(phase)) != 0:
            return {"status": f"stopped at {phase} (rc {rc})", "phases": done}
        done.append(phase)
    if not json.loads(rp.read_text())["phases"][f"vllm_{ARM}"].get("parity_ok"):
        if (rc := run(f"hfgen_{ARM}")) != 0:
            return {"status": f"stopped at hfgen_{ARM} (rc {rc})", "phases": done}
        done.append(f"hfgen_{ARM}")
    return {"status": "ok", "phases": done}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=("rr_gen", "train", "gate"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t14-inputs", type=Path, default=None)
    ap.add_argument("--t11", type=Path, default=None, help="T1.1's gpt_oss_20b dir (pairs)")
    ap.add_argument("--date-pin", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()
    (a.out / KEY).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"rr_gen": step_rr_gen, "train": step_train, "gate": step_gate}[a.step](a)
    update_record(a.out, a.step, {**res, "seconds": time.time() - t0})
    brief = {k: res[k] for k in ("n", "n_items", "n_h1_pairs", "rule_right", "max_steps_run",
                                 "phases", "status") if k in res}
    print(f">> st2lb {a.step}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if str(res.get("status", "")).startswith("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
