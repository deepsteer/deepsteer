#!/usr/bin/env python3
"""CST pressure line, sessions 2-ctx and 2-dpo (Amendments 25 Part C, 27 items 2-4, 28 item 3):
the rungs trained on 2-roll's rollouts, their parity and reads.

    cst_pressure_train_pod.py --stage smoke_hf|smoke_serve|train|parity_hf|read|legit --out DIR
        [--rung p_ctx|p_ctx_n|p_dpo|p_long] [--tags ...] [--arm a0s|a0n] [--dry-run]

Inputs (synced by name from 2-roll): ``scenarios.json``, ``roll/<arm>/chunk_*.jsonl``,
``act.jsonl``, ``contrast.json`` under ``outputs/pressure``. Every rung trains with the prompt
rendered without any document (A0's render of the scenario), loss on trace and answer, LoRA r16 /
alpha 32 on the trainer's class, one epoch, no recovery step (Amendment 27 item 3); capped or
answerless rollouts are not trained on (a cut trace teaches no closing).

- **P-ctx:** NLL on A0s's rollouts over all three kinds, no filter.
- **P-ctx-n:** the same from A0n's rollouts (the neutral-teacher control).
- **P-dpo:** DPO (beta 0.1, NLL 1.0 on chosen) on the disagreeing scenarios of the contrast's P-dpo
  gate, pairs at the rollout level (chosen an A0s rollout read as not following, rejected an A0
  rollout read as following), at most 4 per scenario, plus A0s's rollouts on legitimate and
  no-order scenarios as NLL-only retain items. Not trained when the P-dpo gate failed.
- **P-long:** A0's k 4 on every scenario, chosen the longest response, rejected the shortest, the
  whole response in the loss; no filter.

**Smoke pass (Amendment 28 item 3), the pod's first two stages, about 10 minutes:** ``smoke_hf``
builds every rung's data from the synced inputs (the failure: a data builder that breaks at hour
three), takes one optimizer step of each loss on two items (a loss path that breaks on real
lengths), saves a randomly initialized adapter from the trainer (the save path), and writes the HF
parity side for it; ``smoke_serve`` serves its renamed copy on the live engine (CST-A22's no-op
load), puts it through Amendment 25 A4's gate (it must pass), and reads 2 operator prompts with it
(the read path). A failure stops the pod before any training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pressure_pod as R  # noqa: E402  (2-roll: scenarios, rollouts, labels)
import cst_s0a_pod as S  # noqa: E402

CST = S.CST
RUNGS = ("p_ctx", "p_ctx_n", "p_dpo", "p_long")
PAIRS_PER_SCENARIO = 4
SMOKE_TAG = "smoke"
SMOKE_MIN_TV = 0.15  # the random adapter's TV(base, attached) before its parity read (27B)
SMOKE_MIN_TV_DRY = 0.03  # the tiny model's near-uniform letters saturate near 0.07


def _w(out: Path, name: str, rec) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")


def usable(r: dict) -> bool:
    return not r["hit_cap"] and bool(r.get("answer"))


class Render:
    """A0's render of a scenario (no document), token ids, cached."""

    def __init__(self, tok):
        import reasoning_formats as rf

        self.tok, self.fmt, self.cache = tok, rf.build_format("qwen38_27b", tok), {}

    def __call__(self, s: dict) -> list[int]:
        if s["cell_id"] not in self.cache:
            self.cache[s["cell_id"]] = self.tok.encode(
                self.fmt.render(self.tok, R.msgs(s), S.EFFORT), add_special_tokens=False)
        return self.cache[s["cell_id"]]


def ctx_examples(out: Path, arm: str, render: Render) -> list[tuple[list[int], list[int]]]:
    import pilot_train as PT

    sc = {s["cell_id"]: s for s in R.scen_list(out)}
    rows = [r for r in R.rollouts(out, arm) if usable(r)]
    return PT.recovery_examples([render(sc[r["scenario_id"]]) for r in rows],
                                [r["gen_ids"] for r in rows])


def _pair(prompt: list[int], c: dict, r: dict) -> dict:
    return {"prompt": prompt, "chosen": c["gen_ids"], "rejected": r["gen_ids"],
            "chosen_mask": [1] * len(c["gen_ids"]), "rejected_mask": [1] * len(r["gen_ids"]),
            "chosen_id": c["rollout_id"], "rejected_id": r["rollout_id"]}


def dpo_pairs(out: Path, render: Render) -> list[dict]:
    """Rollout-level pairs on the gate's disagreeing scenarios, at most 4 per scenario (seeded)."""
    import itertools

    sc = {s["cell_id"]: s for s in R.scen_list(out)}
    lab = {json.loads(x)["rollout_id"]: json.loads(x)["main"] for x in open(out / "act.jsonl")}
    dis = json.loads((out / "contrast.json").read_text())["p_dpo_gate"]["scenarios"]
    by = {a: {} for a in ("a0", "a0s")}
    for a in by:
        for r in R.rollouts(out, a):
            by[a].setdefault(r["scenario_id"], []).append(r)
    rng = random.Random("20261110|cst_pressure_dpo_pairs")
    pairs = []
    for sid in dis:
        ch = [r for r in by["a0s"].get(sid, []) if usable(r) and lab.get(r["rollout_id"]) ==
              "does_not"]
        rj = [r for r in by["a0"].get(sid, []) if usable(r) and lab.get(r["rollout_id"]) ==
              "follows"]
        combos = list(itertools.product(ch, rj))
        for c, r in rng.sample(combos, min(PAIRS_PER_SCENARIO, len(combos))):
            pairs.append(_pair(render(sc[sid]), c, r))
    return pairs


def retain_items(out: Path, render: Render) -> list[tuple[list[int], list[int]]]:
    import pilot_train as PT

    sc = {s["cell_id"]: s for s in R.scen_list(out)}
    rows = [r for r in R.rollouts(out, "a0s") if usable(r)
            and sc[r["scenario_id"]]["kind"] in ("legit", "none")]
    return PT.recovery_examples([render(sc[r["scenario_id"]]) for r in rows],
                                [r["gen_ids"] for r in rows])


def long_pairs(out: Path, render: Render) -> list[dict]:
    """A0's k 4 per scenario: chosen the longest response, rejected the shortest (ties skipped)."""
    sc = {s["cell_id"]: s for s in R.scen_list(out)}
    by: dict[str, list[dict]] = {}
    for r in R.rollouts(out, "a0"):
        if usable(r):
            by.setdefault(r["scenario_id"], []).append(r)
    pairs = []
    for sid in sorted(by):
        rs = sorted(by[sid], key=lambda r: (len(r["gen_ids"]), r["rollout_id"]))
        if len(rs) >= 2 and len(rs[-1]["gen_ids"]) > len(rs[0]["gen_ids"]):
            pairs.append(_pair(render(sc[sid]), rs[-1], rs[0]))
    return pairs


def rung_data(out: Path, rung: str, render: Render) -> dict:
    if rung in ("p_ctx", "p_ctx_n"):
        return {"examples": ctx_examples(out, "a0s" if rung == "p_ctx" else "a0n", render)}
    if rung == "p_dpo":
        return {"pairs": dpo_pairs(out, render), "retain": retain_items(out, render)}
    return {"pairs": long_pairs(out, render)}


def gate_ok(out: Path) -> dict:
    c = json.loads((out / "contrast.json").read_text())
    if not c["line_gate"]["passes"]:
        raise SystemExit("BAIL: the contrast's line gate failed; the line comes to the author")
    return c


def _train(wb, rung: str, data: dict, out_dir: Path, dry: bool, smoke: bool = False) -> dict:
    import pilot_pod_lib as PL
    import pilot_train as PT

    accum = 4
    if "examples" in data:
        ex = data["examples"][:2] if smoke else data["examples"]
        per_step = sum(len(i) for i, _ in ex) / max(len(ex), 1) * accum
        dose = per_step if smoke else sum(len(i) for i, _ in ex)
        return PT.train_rung(wb, ex, lambda e: PT.lm_loss(wb, e[0], e[1]), per_step_dose=per_step,
                             doses=(dose,), out_dir=out_dir, lr=PL.SDF_LR)
    pairs = data["pairs"][:2] if smoke else data["pairs"]
    retain = (data.get("retain") or [])[: 2 if smoke else None] or None
    n = accum if smoke else len(pairs) + len(retain or [])
    return PT.train_dpo(wb, pairs, doses=(n,), out_dir=out_dir, lr=PL.DPO_LR, retain=retain)


def stage_smoke_hf(out: Path, rungs: list[str], dry: bool) -> dict:
    """Smoke, HF side: data builders, one step per loss, a random adapter saved, parity HF side."""
    import pilot_pod_lib as PL
    import torch

    gate_ok(out)
    path, wb, _ = PL.load_model(dry)
    render = Render(S.V._tok(path))  # noqa: SLF001
    rec: dict = {"rungs": {}}
    for rung in rungs:  # the data builders and each loss path on two items
        data = rung_data(out, rung, render)
        size = {k: len(v) for k, v in data.items()}
        if not any(size.values()):
            raise SystemExit(f"BAIL: smoke: {rung} builds no training items")
        if rung == "p_dpo" and not data["pairs"]:
            rec["rungs"][rung] = {"items": size, "step": "skipped: no pairs (P-dpo gate)"}
            continue
        r = _train(wb, rung, data, out / "smoke" / rung, dry, smoke=True)
        rec["rungs"][rung] = {"items": size, "steps": r["steps"]}
    # a randomly initialized adapter, saved the trainer's way, scaled until it moves the letters
    # well above the serving floor (about 0.0075), or the gate's 0.2 rule cannot pass
    from deepsteer.steering import objective_lora as OL

    prompts, lids = PL.parity_prompts(dry)
    with OL.adapter_disabled(wb):
        base = PL._letter_lp(wb, prompts, lids)  # noqa: SLF001
    # the served side reads only the top 20 log-probs, so the letters must stay inside them on
    # every prompt (2-ctx's first smoke pass pushed them out at TV 0.41 and the gate divided by 0)
    scale, ref, rank = (0.5 if dry else 0.005), 0.0, 0
    bar = SMOKE_MIN_TV_DRY if dry else SMOKE_MIN_TV
    for _ in range(16):
        torch.manual_seed(0)
        with torch.no_grad():
            for name, prm in wb.model.named_parameters():
                if "lora_B" in name:
                    prm.normal_(0, scale)
        attached = PL._letter_lp(wb, prompts, lids)  # noqa: SLF001
        ref = PL.tv(base, attached)
        rank = max(PL.letter_ranks(wb, prompts, lids))
        out_of_top = not dry and rank >= PL.SERVED_TOPK  # the tiny model's letters sit deep
        if ref >= bar or out_of_top:
            break
        scale *= 1.5
    if ref < bar or out_of_top:
        raise SystemExit(f"BAIL: smoke: no random adapter reached TV {bar} with every letter in "
                         f"the served top {PL.SERVED_TOPK} (TV {ref:.3f}, worst rank {rank})")
    adir = out / "smoke" / "adapter"
    wb.model.save_pretrained(str(adir))
    par = out / "parity"
    par.mkdir(parents=True, exist_ok=True)
    if not (par / "prefixes.json").exists():  # in this HF process, before any vLLM
        (par / "prefixes.json").write_text(json.dumps(PL.decoder_prefixes(path)) + "\n")
    if not (par / "hf_base.json").exists():
        PL.write_once(par / "hf_base.json", {"letter_ids": lids, "class": type(wb.model).__name__,
                                             "base": base, "via": "smoke_hf, adapter disabled"})
    PL.write_once(par / f"hf_{SMOKE_TAG}.json", {"tag": SMOKE_TAG, "adapter": str(adir),
                                                 "attached": attached})
    rec["random_adapter"] = {"scale": scale, "tv_base_vs_attached": ref,
                             "worst_letter_rank": rank}
    rec["adapter"] = str(adir)
    _w(out, "smoke_hf", rec)
    print(">> smoke_hf:", json.dumps(rec))
    return rec


def stage_smoke_serve(out: Path, dry: bool) -> dict:
    """Smoke, served side: the renamed copy on the live engine, the parity gate, a 2-prompt read."""
    import pilot_pod_lib as PL

    adir = str(out / "smoke" / "adapter")
    res = PL.stage_read(out, [(SMOKE_TAG, adir)], dry, parts=("panel_operator",),
                        k_override={"panel_operator": 1}, limit=2, base=False)
    rec = {"parity_passes": res["parity"].get(SMOKE_TAG), "read": res["read"],
           "rows": sum(1 for f in (out / "gen" / SMOKE_TAG).glob("*.jsonl") for _ in open(f))
           if (out / "gen" / SMOKE_TAG).exists() else 0}
    _w(out, "smoke_serve", rec)
    print(">> smoke_serve:", json.dumps(rec))
    if not rec["parity_passes"] or not rec["rows"]:
        raise SystemExit("BAIL: smoke: the served random adapter failed parity or read nothing")
    return rec


def stage_train(out: Path, rung: str, dry: bool) -> dict:
    import pilot_pod_lib as PL

    c = gate_ok(out)
    if rung == "p_dpo" and not c["p_dpo_gate"]["passes"]:
        rec = {"rung": rung, "trained": False,
               "reason": "the P-dpo gate failed (Amendment 27 item 2)",
               "disagreeing": c["p_dpo_gate"]["disagreeing"]}
        _w(out, f"train_{rung}", rec)
        print(">> train:", json.dumps(rec))
        return rec
    path, wb, cfg = PL.load_model(dry)
    data = rung_data(out, rung, Render(S.V._tok(path)))  # noqa: SLF001
    r = _train(wb, rung, data, out / "adapters" / rung, dry)
    rec = {"rung": rung, "trained": True, "items": {k: len(v) for k, v in data.items()},
           "tokens": sum(len(i) for i, _ in data["examples"]) if "examples" in data else None,
           "lora": {k: cfg[k] for k in ("rank", "alpha", "dropout")}, "train": r,
           "adapters": dict(r["checkpoints"])}
    _w(out, f"train_{rung}", rec)
    print(">> train:", json.dumps({k: v for k, v in rec.items() if k != "train"}))
    return rec


def adapter_dir(out: Path, tag: str) -> str:
    tr = json.loads((out / f"train_{tag}.json").read_text())
    if not tr.get("trained"):
        raise SystemExit(f"{tag} was not trained: {tr.get('reason')}")
    return list(tr["adapters"].values())[-1]


def trained(out: Path, tags: list[str]) -> list[str]:
    return [t for t in tags if json.loads((out / f"train_{t}.json").read_text()).get("trained")]


def stage_parity_hf(out: Path, tags: list[str], dry: bool) -> dict:
    import pilot_pod_lib as PL

    return PL.hf_letters(out, [(t, adapter_dir(out, t)) for t in trained(out, tags)], dry)


def stage_read(out: Path, tags: list[str], dry: bool) -> dict:
    import pilot_pod_lib as PL

    return PL.stage_read(out, [(t, adapter_dir(out, t)) for t in trained(out, tags)], dry)


def stage_legit(out: Path, arm: str, dry: bool) -> None:
    """C7 on 2-dpo: A0s or A0n on the legitimate-order cell at k 4 (one engine per process)."""
    S.stage_gen(out, arm, ["legit_operator"], dry, tag=f"{arm}_legit",
                k_override={"legit_operator": 4})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("smoke_hf", "smoke_serve", "train", "parity_hf", "read", "legit"))
    ap.add_argument("--rung", choices=RUNGS)
    ap.add_argument("--rungs", default="", help="smoke_hf: this pod's rungs")
    ap.add_argument("--tags", default="")
    ap.add_argument("--arm", choices=("a0s", "a0n"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    if a.stage == "smoke_hf":
        stage_smoke_hf(a.out, [r for r in a.rungs.split(",") if r], a.dry_run)
    elif a.stage == "smoke_serve":
        stage_smoke_serve(a.out, a.dry_run)
    elif a.stage == "train":
        if a.rung is None:
            ap.error("--stage train needs --rung")
        stage_train(a.out, a.rung, a.dry_run)
    elif a.stage == "parity_hf":
        stage_parity_hf(a.out, tags, a.dry_run)
    elif a.stage == "read":
        stage_read(a.out, tags, a.dry_run)
    else:
        if a.arm is None:
            ap.error("--stage legit needs --arm")
        stage_legit(a.out, a.arm, a.dry_run)


if __name__ == "__main__":
    main()
