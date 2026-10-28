"""The smoke pod for pod 2 on Qwen3.8-27B (ST_PLAN Amendment 57 and its addition): T1.6's
machinery (``t16_driver``) run once at full model scale on a handful of items, so the failures a
tiny-model dry run cannot show (QLoRA on Qwen's DeltaNet text tower, the merge into Qwen's tensor
names, vLLM on the merged copy) surface before pod 2.

``t16_driver`` is configured for Qwen here (its model key, date pin, site and stage) and three of
its functions are replaced: the parity scenarios (16 seeded H1_train scenarios from pod 1's v1
set), the pairs (pod 1's rule-harm pairs: the care arm's protect rollouts against ``none``'s rule
rollouts, T1.6's per-scenario rule without the base term) and the context's sizes. Steps, each its
own process, in the runner's order: ``base_vllm``, ``base_hf``, ``train --arm care``,
``hf_arm --arm care``, ``vllm_arm --arm care``. The pass criteria are read locally
(``smoke_read``).

    python3 st2qs_driver.py --step <step> [--arm care] --out <dir> --pod1 <st2q/qwen38_27b dir> \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel" / "scripts"))
import t16_driver as T16  # noqa: E402

KEY = "qwen38_27b"
PIN = "2026-10-27"  # pod 1's rows (st2q)
BLOCK = 32
STAGE = "st2qs"
N_PARITY, N_HELD, RR, N_CAP = 16, 4, 2, 4
SMOKE_STEPS = 6
SEED = 105


def configure() -> None:
    """``t16_driver`` reads these module globals at call time."""
    T16.KEY, T16.PIN, T16.BLOCK, T16.STAGE, T16.SEED = KEY, PIN, BLOCK, STAGE, SEED
    T16.parity_scenarios = parity_scenarios
    T16.pairs_for = pairs_for


def parity_scenarios(c: dict) -> list[str]:
    """16 seeded H1_train scenarios from pod 1's v1 rows (2 in a dry run)."""
    rows = T16._jsonl(c["pod1"] / "gen" / "h1_rollouts.jsonl")  # noqa: SLF001
    ids = sorted({r["scenario_id"] for r in rows if r.get("split") == "train"})
    n = 2 if c.get("dry") else N_PARITY
    return sorted(random.Random(f"{STAGE}|parity").sample(ids, min(n, len(ids))))


def smoke_pairs(care_rows: list[dict], none_rows: list[dict]) -> tuple[list[dict], dict]:
    """T1.6's per-scenario rule for the care arm alone: n = min(care protect, none rule, 2), in
    rollout order (the smoke pod trains one adapter, so the base term is dropped)."""
    def by(rows, status):
        d: dict[str, list] = {}
        for r in sorted(rows, key=lambda x: (x["scenario_id"], x["rollout"])):
            if r.get("norm_status") == status and r.get("gen_ids"):
                d.setdefault(r["scenario_id"], []).append(r)
        return d

    cp, rj = by(care_rows, "violating"), by(none_rows, "consistent")
    out, counts = [], {}
    for sid in sorted(set(cp) & set(rj)):
        n = min(len(cp[sid]), len(rj[sid]), T16.PAIRS_PER_SCENARIO)
        counts[sid] = n
        for i in range(n):
            out.append({"scenario_id": sid, "chosen_key": (sid, cp[sid][i]["rollout"]),
                        "chosen": cp[sid][i]["gen_ids"], "rejected": rj[sid][i]["gen_ids"],
                        "rejected_key": (sid, rj[sid][i]["rollout"]), "kind": "rule_harm"})
    return out, {"per_scenario": counts, "n_pairs": len(out)}


def pairs_for(c: dict, a, arm: str) -> tuple[list[dict], dict]:
    care = T16._jsonl(c["pod1"] / "gen" / "arm_care.jsonl")  # noqa: SLF001
    none = T16._jsonl(c["pod1"] / "gen" / "arm_none.jsonl")  # noqa: SLF001
    hp, info = smoke_pairs(care, none)
    prompts = T16.render_prompts(c, care + none)
    pr = [{"scenario_id": p["scenario_id"], "prompt": prompts[p["chosen_key"]],
           "chosen": p["chosen"], "rejected": p["rejected"], "kind": p["kind"],
           "chosen_key": p["chosen_key"], "rejected_key": p["rejected_key"]} for p in hp]
    if a.dry_run:
        pr = [{**p, "chosen": p["chosen"][:24], "rejected": p["rejected"][:24]} for p in pr]
    return pr, {**info, "n_harm": len(pr), "rule_right": {"n": 0, "note": "smoke: none"}}


def context(a) -> dict:
    a.replicate = a.pod1  # T16.context records it; nothing here reads it
    c = T16.context(a)
    c["pod1"], c["dry"] = a.pod1, a.dry_run
    c["record"] = c["out"] / "st2qs_record.json"
    c["held"] = c["held"][: 2 if a.dry_run else N_HELD]
    c["rr"] = 2 if a.dry_run else RR
    c["spec_ids"] = c["spec_ids"][:2]
    c["half_ids"] = c["half_ids"][:2]
    c["cap_items"] = c["cap_items"][:1] + c["cap_items"][-1:] if a.dry_run \
        else c["cap_items"][: N_CAP // 2] + c["cap_items"][-N_CAP // 2:]
    c["recipe"].update(max_steps=SMOKE_STEPS, budget_h=10.0, general_train=64,
                       general_monitor=2)
    return c


STEPS = {"base_vllm": T16.step_base_vllm, "base_hf": T16.step_base_hf, "train": T16.step_train,
         "hf_arm": T16.step_hf_arm, "vllm_arm": T16.step_vllm_arm}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=tuple(STEPS))
    ap.add_argument("--arm", default="none", choices=("none", "care"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pod1", type=Path, required=True, help="pod 1's qwen38_27b dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--dry-fail-parity", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.seed, a.fresh_from_t16 = SEED, False
    configure()
    c = context(a)
    if T16.phase(c, f"{a.step}_{a.arm}" if a.step != "base_vllm" and a.step != "base_hf"
                 else a.step).get("status"):
        print(f">> st2qs {a.step} {a.arm}: finished earlier", flush=True)
        return 0
    res = STEPS[a.step](c, a)
    name = a.step if a.step in ("base_vllm", "base_hf") else f"{a.step}_{a.arm}"
    T16.update_record(c, name, res)
    brief = {k: res.get(k) for k in ("status", "base_tv", "base_check_ok", "max_steps_run",
                                     "parity_tv", "parity_ok") if k in res}
    if "merge" in res:
        brief["merge_gate_ok"] = res["merge"].get("gate_ok")
    print(f">> st2qs {name}: done ({json.dumps(brief, default=str)})", flush=True)
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
