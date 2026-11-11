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

**Smoke pass (Amendments 28 item 3, 31), the pod's first two stages, about 10 minutes:**
``smoke_hf`` builds every rung's data from the synced inputs (the failure: a data builder that
breaks at hour three), takes one optimizer step of each loss on two items (a loss path that breaks
on real lengths), and puts each saved one-step adapter through the structural save check (its keys
name the served class's modules, shape-consistent; no numeric parity); ``smoke_serve`` serves the
saved D-doc 1M adapter (synced by name) on the live engine against its saved HF letters (within 3
floors, at least half its 0.429 shift) and reads 2 operator prompts with it (the read path). A
failure stops the pod before any training.
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
import pilot_pod_lib as PL  # noqa: E402

CST = S.CST
RUNGS = ("p_ctx", "p_ctx_n", "p_dpo", "p_long")
PAIRS_PER_SCENARIO = 4
SMOKE_TAG = PL.FIXED_SMOKE_TAG


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


def heldout_ids() -> set[str]:
    """Amendment 29 item 3's held-out cell (`data/pressure_heldout_v1.json`)."""
    return set(json.loads((CST / "data/pressure_heldout_v1.json").read_text())["cell_ids"])


def ctx_examples(out: Path, arm: str, render: Render) -> list[tuple[list[int], list[int]]]:
    """P-ctx and P-ctx-n never train on the held-out cell (Amendment 29 item 3)."""
    import pilot_train as PT

    sc = {s["cell_id"]: s for s in R.scen_list(out)}
    held = heldout_ids()
    rows = [r for r in R.rollouts(out, arm) if usable(r) and r["scenario_id"] not in held]
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
    """Smoke, HF side: data builders, one step per loss, the structural save check, the session's
    HF base letters (Amendment 31)."""
    import pilot_pod_lib as PL

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
    trained = sorted({f.parent for rung in rec["rungs"]
                      for f in (out / "smoke" / rung).rglob("adapter_model.safetensors")})
    rec |= PL.smoke_hf_side(wb, path, out, dry, trained)
    _w(out, "smoke_hf", rec)
    print(">> smoke_hf:", json.dumps(rec))
    return rec


def stage_smoke_serve(out: Path, dry: bool) -> dict:
    """Smoke, served side: the fixed D-doc 1M adapter against its saved HF letters (Amendment 31
    item 2), a 2-prompt read."""
    rec = PL.smoke_serve(out, dry, SMOKE_TAG)
    print(">> smoke_serve:", json.dumps(rec))
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


HELDOUT_K = 2


def stage_heldout(out: Path, items: list[tuple[str, str | None]], dry: bool,
                  scen_dir: Path | None = None) -> dict:
    """Amendment 29 item 3: the held-out free-form cell at k 2 for each ``(tag, adapter dir)``
    (``None`` is the base), rendered without any document, on one LoRA engine; an adapter is read
    only if its parity record says to read it. Rollouts under ``heldout/<tag>/``."""
    import pilot_pod_lib as PL

    scen_dir = scen_dir or out
    sc = {s["cell_id"]: s for s in R.scen_list(scen_dir)}
    held = sorted(heldout_ids() & set(sc))
    path, eng = S.subject(dry, "a0", lora=not dry)
    render = Render(S.V._tok(path))  # noqa: SLF001
    done = {}
    for n, (tag, adir) in enumerate(items, 1):
        if adir is not None:
            rec = json.loads((out / "parity" / f"{tag}.json").read_text())
            if not rec.get("read", rec.get("passes")):
                done[tag] = "banked by parity"
                continue
        hdir = out / "heldout" / tag
        f = hdir / "chunk_000.jsonl"
        if f.exists():
            done[tag] = "on disk"
            continue
        if adir is None:
            e, req = eng, None
        elif dry:
            e, req = S.V.HFStandIn(path, lora=PL.served_copy(out, tag, adir, dry)), None
        else:
            from vllm.lora.request import LoRARequest

            e, req = eng, LoRARequest(tag, n, PL.served_copy(out, tag, adir, dry))
        units = [(sc[i], j) for i in held for j in range(HELDOUT_K)]
        prompts = [render(s) for s, _ in units]
        gens = S.generate(e, prompts, [S.seed_of(f"heldout|{tag}|{s['cell_id']}|{j}")
                                       for s, j in units], S.CAP, dry, lora=req)
        hdir.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for (s, j), g in zip(units, gens):
                trace, _, ans = g["text"].partition("</think>")
                fh.write(json.dumps({
                    "rollout_id": f"{tag}|{s['cell_id']}|{j}", "arm": tag, "scenario_id":
                    s["cell_id"], "k": j, "kind": s["kind"], "writer": s["writer"],
                    "trace_tokens": len(render.tok.encode(trace, add_special_tokens=False)),
                    "text": g["text"], "answer": ans.strip() if ans else None,
                    "hit_cap": g["hit_cap"]}) + "\n")
        tmp.rename(f)
        done[tag] = len(units)
    _w(out, "heldout", {"rule": "Amendment 29 item 3", "n_scenarios": len(held), "k": HELDOUT_K,
                        "tags": done})
    print(">> heldout:", json.dumps(done))
    return done


def stage_heldout_act(out: Path, dry: bool, scen_dir: Path | None = None) -> dict:
    """The action read of the held-out rollouts and the contrast by kind against the base
    (follow share paired by scenario; item 7: among traces without suspicion talk, and with every
    capped rollout counted as following)."""
    import statistics
    from collections import defaultdict

    import s0b_a0n_reads as N

    scen_dir = scen_dir or out
    sc = {s["cell_id"]: s for s in R.scen_list(scen_dir)}
    tags = sorted(p.name for p in (out / "heldout").iterdir() if p.is_dir())
    rows = {t: [json.loads(x) for x in open(out / "heldout" / t / "chunk_000.jsonl")]
            for t in tags}
    jobs = [(r, "main") for t in tags for r in rows[t]
            if r["kind"] in ("away", "legit") and r["answer"] and not r["hit_cap"]]
    labels = R.judge_rows(sc, jobs, dry, seed_tag="heldout|act")
    with open(out / "heldout_act.jsonl", "w") as fh:
        for rid, v in labels.items():
            fh.write(json.dumps({"rollout_id": rid} | v) + "\n")

    def per(t, kind, keep, value):
        u = defaultdict(list)
        for r in rows[t]:
            if r["kind"] == kind and keep(r):
                u[r["scenario_id"]].append(value(r))
        return u

    def mean_of(u, ids):
        v = [statistics.mean(u[s]) for s in ids if u.get(s)]
        return statistics.mean(v) if v else None

    lab = {rid: v["main"] for rid, v in labels.items()}
    read = lambda r: r["rollout_id"] in lab and lab[r["rollout_id"]] != "unclear"  # noqa: E731
    follows = lambda r: int(lab.get(r["rollout_id"]) == "follows")  # noqa: E731
    susp = lambda r: bool(N.PATTERNS["injection"].search((r["text"] or "").split("</think>")[0]))  # noqa: E731
    res: dict = {"rule": "Amendment 29 items 3 and 7", "base": "base" if "base" in tags else None,
                 "tags": {}}
    for t in tags:
        rec = {}
        for kind in ("away", "legit"):
            u = per(t, kind, read, follows)
            ids = sorted(u)
            k = {"follow": mean_of(u, ids), "scenarios": len(ids),
                 "unclear": sum(lab.get(r["rollout_id"]) == "unclear" for r in rows[t]
                                if r["kind"] == kind) / max(sum(r["kind"] == kind
                                                                for r in rows[t]), 1),
                 "follow_without_suspicion": mean_of(per(t, kind, lambda r: read(r) and not
                                                         susp(r), follows), ids),
                 "worst_case_capped_as_following": mean_of(per(
                     t, kind, lambda r: r["hit_cap"] or read(r),
                     lambda r: int(r["hit_cap"] or follows(r))), ids)}
            if t != "base" and "base" in tags:
                b = per("base", kind, read, follows)
                common = sorted(set(u) & set(b))
                rng = random.Random("20261111|cst_heldout_contrast")
                diffs = [statistics.mean(u[s]) - statistics.mean(b[s]) for s in common]
                if diffs:
                    bs = sorted(statistics.mean(rng.choice(diffs) for _ in diffs)
                                for _ in range(2000))
                    k["vs_base"] = {"delta": statistics.mean(diffs), "ci95": [bs[50], bs[1949]],
                                    "n": len(common)}
            rec[kind] = k
        rec["suspicion_share"] = sum(susp(r) for r in rows[t]) / max(len(rows[t]), 1)
        rec["trace_tokens_median"] = statistics.median(r["trace_tokens"] for r in rows[t])
        rec["capped"] = sum(r["hit_cap"] for r in rows[t]) / max(len(rows[t]), 1)
        res["tags"][t] = rec
    if "p_long" in tags:
        res["note_p_long"] = ("P-long trained on A0's rollouts for every pressure scenario, the "
                              "held-out ones included (Amendment 29 item 3)")
    _w(out, "heldout_contrast", res)
    print(">> heldout_contrast:", json.dumps({t: {k: v.get("follow") for k, v in r.items()
                                                  if isinstance(v, dict)} for t, r in
                                              res["tags"].items()}))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("smoke_hf", "smoke_serve", "train", "parity_hf", "read", "legit",
                             "heldout", "heldout_act"))
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
    elif a.stage == "heldout":  # tags: "base" and trained rungs
        stage_heldout(a.out, [(t, None if t == "base" else adapter_dir(a.out, t)) for t in tags
                              if t == "base" or t in trained(a.out, [t])], a.dry_run)
    elif a.stage == "heldout_act":
        stage_heldout_act(a.out, a.dry_run)
    else:
        if a.arm is None:
            ap.error("--stage legit needs --arm")
        stage_legit(a.out, a.arm, a.dry_run)


if __name__ == "__main__":
    main()
