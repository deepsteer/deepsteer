#!/usr/bin/env python3
"""The P-ctx verification pod (CST Amendment 35), on one session base.

    cst_pctx_verify_pod.py --stage smoke_hf|smoke_serve|parity_hf|session|read_op|read|parity_only
        |capability|guard|guard_act|h1|train_draw2|train_half|heldout|heldout_act --out DIR

1. ``session``: one LoRA engine reads the base's screen read (operator and plain at k 2 on the
   458 clear items, cell A′ at k 8, the legitimate cell at k 4; ``gen/base_session``) and A0s on
   the operator cell at k 2 with the constitution in context (``gen/a0s_session``); ``read_op``
   reads P-ctx's operator cell at k 2 in the same session (its parity record first).
2. On the saved P-ctx (2-ctx's adapter, its record copied into DIR): ``capability`` at n 800 (the
   whole pool); ``guard`` and ``guard_act`` (the injected-order cell on the base, P-ctx and
   P-ctx-n); ``h1`` (`H1_held`'s operator and plain cells at k 4 on the base and P-ctx).
3. ``train_draw2``: P-ctx's training set, seed 1, a seeded new example order (``p_ctx_draw2``);
   ``read`` its screen read; ``heldout``/``heldout_act`` on the base, P-ctx and the draw.
4. ``train_half``: a seeded half of P-ctx's training rollouts, seed 0 (``p_ctx_half``), and its
   screen read.

The smoke pass, training data, held-out cell and parity come from `cst_pressure_train_pod.py`;
capability and the guard cell from `cst_sdf_norec_pod.py`, pointed at this pod's tags.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pressure_train_pod as T  # noqa: E402
import cst_s0a_pod as S  # noqa: E402
import cst_sdf_norec_pod as Q  # noqa: E402
import pilot_pod_lib as PL  # noqa: E402

CAP_N = 800  # Amendment 35 item 2: the whole capability pool
H1_K = {"h1_operator": 4, "h1_plain": 4}
DRAW2_ORDER, HALF_SEED = "20261111|cst_pctx_draw2", "20261111|cst_pctx_half"


def items(out: Path, tags: list[str]) -> list[tuple[str, str | None]]:
    """``(tag, adapter dir)`` from this pod's train records (``base`` has none)."""
    return [(t, None if t == "base" else T.adapter_dir(out, t)) for t in tags]


Q.items = items  # capability and the guard cell resolve this pod's tags
Q.CAP_N = CAP_N


def stage_session(out: Path, dry: bool) -> dict:
    """Item 1: the base's screen read and A0s's operator cell on one engine."""
    clear = PL.clear_ids() if not dry else None
    path, eng = S.subject(dry, "a0", lora=not dry)
    for part in PL.READ_PARTS:
        S.stage_gen(out, "a0", [part], dry, tag=PL.BASE_TAG, k_override=PL.READ_K,
                    engine=(path, eng), only=clear if part.startswith("panel") else None)
    S.stage_gen(out, "a0s", ["panel_operator"], dry, tag="a0s_session",
                k_override={"panel_operator": 2}, engine=(path, eng), only=clear)
    rec = {"rule": "Amendment 35 item 1", "engine": "one LoRA engine for the base and A0s",
           "tags": [PL.BASE_TAG, "a0s_session"]}
    T._w(out, "session", rec)  # noqa: SLF001
    return rec


def stage_h1(out: Path, tags: list[str], dry: bool) -> dict:
    """Item 2: `H1_held`'s operator and plain cells at k 4 on each tag (one LoRA engine)."""
    path, eng = S.subject(dry, "a0", lora=not dry)
    done = {}
    for n, (tag, adir) in enumerate(items(out, tags), 1):
        if adir is None:
            e, req, name = (path, eng), None, PL.BASE_TAG
        else:
            rec = json.loads((out / "parity" / f"{tag}.json").read_text())
            if not rec.get("read", rec.get("passes")):
                done[tag] = "banked by parity"
                continue
            sdir = PL.served_copy(out, tag, adir, dry)
            if dry:
                e, req = (path, S.V.HFStandIn(path, lora=sdir)), None
            else:
                from vllm.lora.request import LoRARequest

                e, req = (path, eng), LoRARequest(tag, n, sdir)
            name = tag
        S.stage_gen(out, "a0", list(H1_K), dry, tag=name, lora=req, k_override=H1_K, engine=e)
        done[tag] = name
    T._w(out, "h1", {"rule": "Amendment 35 item 2", "k": 4, "tags": done})  # noqa: SLF001
    return done


def _train_ctx(out: Path, tag: str, dry: bool, *, seed: int, order: str | None = None,
               half: str | None = None) -> dict:
    """P-ctx's training set and settings (`cst_pressure_train_pod._train`), with a seed, an
    optional seeded new order and an optional seeded half."""
    import pilot_train as PT

    f = out / f"train_{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    T.gate_ok(out)
    path, wb, cfg = PL.load_model(dry)
    ex = T.rung_data(out, "p_ctx", T.Render(S.V._tok(path)))["examples"]  # noqa: SLF001
    n_all = len(ex)
    if half:
        ex = random.Random(half).sample(ex, n_all // 2)
    if order:
        random.Random(order).shuffle(ex)
    per_step = sum(len(i) for i, _ in ex) / max(len(ex), 1) * 4
    dose = sum(len(i) for i, _ in ex)
    r = PT.train_rung(wb, ex, lambda e: PT.lm_loss(wb, e[0], e[1]), per_step_dose=per_step,
                      doses=(dose,), out_dir=out / "adapters" / tag, lr=PL.SDF_LR, seed=seed)
    rec = {"rung": tag, "trained": True, "rule": "Amendment 35 items 3-4", "seed": seed,
           "order": order, "half": half, "items": {"examples": len(ex), "of": n_all},
           "tokens": dose, "lr": PL.SDF_LR,
           "lora": {k: cfg[k] for k in ("rank", "alpha", "dropout")}, "train": r,
           "adapters": dict(r["checkpoints"])}
    T._w(out, f"train_{tag}", rec)  # noqa: SLF001
    print(f">> train_{tag}:", json.dumps({k: v for k, v in rec.items() if k != "train"}))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("smoke_hf", "smoke_serve", "parity_hf", "session", "read_op", "read",
                             "parity_only", "capability", "guard", "guard_act", "h1",
                             "train_draw2", "train_half", "heldout", "heldout_act"))
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    out, dry = a.out, a.dry_run
    if a.stage == "smoke_hf":
        T.stage_smoke_hf(out, ["p_ctx"], dry)
    elif a.stage == "smoke_serve":
        T.stage_smoke_serve(out, dry)
    elif a.stage == "parity_hf":
        PL.hf_letters(out, items(out, tags), dry)
    elif a.stage == "session":
        stage_session(out, dry)
    elif a.stage == "read_op":
        PL.stage_read(out, items(out, tags), dry, parts=("panel_operator",),
                      k_override={"panel_operator": 2}, base=False)
    elif a.stage == "read":
        PL.stage_read(out, items(out, tags), dry, base=False)
    elif a.stage == "parity_only":
        PL.stage_read(out, items(out, tags), dry, parts=(), base=False)
    elif a.stage == "capability":
        Q.stage_capability(out, tags, dry)
    elif a.stage == "guard":
        Q.stage_guard(out, tags, dry)
    elif a.stage == "guard_act":
        Q.stage_guard_act(out, dry)
    elif a.stage == "h1":
        stage_h1(out, tags, dry)
    elif a.stage == "train_draw2":
        _train_ctx(out, "p_ctx_draw2", dry, seed=1, order=DRAW2_ORDER)
    elif a.stage == "train_half":
        _train_ctx(out, "p_ctx_half", dry, seed=0, half=HALF_SEED)
    elif a.stage == "heldout":
        T.stage_heldout(out, items(out, tags), dry)
    else:
        T.stage_heldout_act(out, dry)


if __name__ == "__main__":
    main()
