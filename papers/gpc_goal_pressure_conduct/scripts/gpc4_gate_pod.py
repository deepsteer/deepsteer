#!/usr/bin/env python3
"""The standing gate before any adapter arm (GPC_PLAN.md §18; GPC-A25): each served adapter's known effect shown
through the arms' own server and client before any arm reads it.

In order, on the subject's server as the arms use it (`P4._chat`, the names `base`, `c1`, `sum`):
1. The control: the 24 held-out grid slots at k = 2 under the base, `C_conduct` in context, C1 and the summed
   adapter (A7's control stage, the same rows, `a7/control.jsonl`), and A7's bail on them (`a7/decision.json`:
   the base under 0.30, or the clause's reduction with a CI including 0, fails the instrument).
2. Per adapter, the control rule: its reduction against the base, paired by slot (a bootstrap over slots, 95%),
   has an interval excluding 0 and is at least half the clause's reduction on the same slots.
3. Per adapter, the second derivation: served-against-attached letter parity through the server. The 32 parity
   prompts go to `/v1/completions` (token IDs, one token, the top 20 log-probabilities); the letters are read
   against the HF letters of this session's parity stage (`parity/hf_base.json`, `parity/hf_<tag>.json`) under
   the parity rule of record (`pilot_pod_lib.parity_record`).

An adapter passes when (2) and (3) pass. **The gate passes when the control holds and C1 passes**; a failed gate
ends the session (the runner). The summed adapter is read only if it passes; its failure is recorded.

    python gpc4_gate_pod.py --out OUT [--dry-run]   # writes OUT/gate.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_a7_pod as A7  # noqa: E402
import gpc4_pod as P4  # noqa: E402

ADAPTERS = ("c1", "sum")
HALF = 0.5
BOOT = 4000
SEED = "20261123|gpc4|gate"


def _slot_means(rows: list[dict], arm: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in rows:
        if r["arm"] == arm and r.get("accepts") is not None:
            by.setdefault(r["slot_id"], []).append(float(r["accepts"]))
    return {k: sum(v) / len(v) for k, v in by.items()}


def control_rule(rows: list[dict], tag: str, seed: str = SEED) -> dict:
    """The adapter's reduction against the base, paired by slot, beside the clause's on the same slots."""
    b, c, a = _slot_means(rows, "base"), _slot_means(rows, "cctx"), _slot_means(rows, tag)
    keys = sorted(set(b) & set(c) & set(a))
    if not keys:
        return {"passes": False, "slots": 0, "why": "no paired slots (the adapter's rows did not parse or failed)"}
    m = lambda d, ks: sum(d[k] for k in ks) / len(ks)  # noqa: E731
    red, cred = m(a, keys) - m(b, keys), m(c, keys) - m(b, keys)
    rng = random.Random(f"{seed}|{tag}")
    bs = sorted(m(a, ks) - m(b, ks) for ks in ([rng.choice(keys) for _ in keys] for _ in range(BOOT)))
    ci = [bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT) - 1]]
    ok = ci[1] < 0 and cred < 0 and red <= HALF * cred
    return {"passes": bool(ok), "slots": len(keys), "base": m(b, keys), "cctx": m(c, keys), "adapter": m(a, keys),
            "reduction": red, "reduction_ci": ci, "clause_reduction": cred, "half_clause": HALF * cred,
            "rule": "the reduction's CI excludes 0 and the reduction is at least half the clause's (§18)"}


def _server_letters(model: str, prompts: list[list[int]], lids: list[int]) -> list[list[float]]:
    """The served letters through the server: each prompt's next-token top-20 log-probabilities, a letter outside
    them at -1e9 (the parity rule of record's convention)."""
    out = []
    for ids in prompts:
        body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
                "return_tokens_as_token_ids": True}
        r = urllib.request.Request("http://127.0.0.1:8000/v1/completions", method="POST",
                                   data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=300) as resp:
            j = json.loads(resp.read())
        top = (((j.get("choices") or [{}])[0].get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        out.append([float(top.get(f"token_id:{i}", -1e9)) for i in lids])
    return out


def server_parity(out: Path, tags: list[str], dry: bool) -> dict[str, dict]:
    """Served-against-attached letter parity through the server, per adapter (the second derivation)."""
    import pilot_pod_lib as PL

    pdir = out / "parity"
    hb = json.loads((pdir / "hf_base.json").read_text())
    res = {}
    if dry:
        served_base, prompts, lids = hb["base"], None, hb.get("letter_ids")
    else:
        prompts, lids = PL.parity_prompts(dry)
        served_base = _server_letters("base", prompts, lids)
    for t in tags:
        f = pdir / f"hf_{t}.json"
        if not f.exists():
            res[t] = {"passes": False, "why": f"no HF letters for {t} this session"}
            continue
        att = json.loads(f.read_text())["attached"]
        try:
            served = att if dry else _server_letters(t, prompts, lids)
        except Exception as e:  # noqa: BLE001  (recorded; the adapter fails the gate)
            res[t] = {"passes": False, "why": f"the server did not answer: {repr(e)[:200]}"}
            continue
        rec = PL.parity_record(t, served, att, hb["base"], served_base=served_base)
        res[t] = {k: v for k, v in rec.items() if k != "served_letters"} | {"path": "the server (§18)"}
    return res


def stage_gate(out: Path, dry: bool) -> dict:
    ok, why = P4.server_ok(out)
    if not ok:
        rec = {"gate_ok": False, "why": why, "adapters": {}}
        P4._w(out, "gate", rec)  # noqa: SLF001
        print(">> gate:", json.dumps(rec), flush=True)
        return rec
    rows = A7.stage_control(out, dry)
    dec = A7.stage_decide(out, dry)
    served = [t for t in ADAPTERS if P4.adapter_ok(out, t)]
    par = server_parity(out, served, dry) if served else {}
    ad = {}
    for t in ADAPTERS:
        if t not in served:
            ad[t] = {"passes": False, "why": "not served (its parity on the engine path does not read)"}
            continue
        cr, sp = control_rule(rows, t), par.get(t, {"passes": False})
        ad[t] = {"passes": bool(cr["passes"] and sp.get("passes")), "control_rule": cr, "server_parity": sp}
    gate = bool(dec.get("control_ok") and ad.get("c1", {}).get("passes"))
    # dry only (the standing step's dry chain): the stand-in model cannot load the 27B adapters, so none is served
    # and each fails here; a passing control then lets the dry block run on with every adapter arm skipped (its
    # gate record says it failed), as the dry chain ran before §18. The pod never runs dry.
    dry_through = bool(dry and not gate and dec.get("control_ok") and not served)
    rec = {"gate_ok": gate or dry_through, "control_ok": dec.get("control_ok"), "adapters": ad,
           "dry_pass_through": dry_through,
           "rule": "GPC_PLAN.md §18: the control holds and C1 passes (its reduction's CI excludes 0 and is at "
                   "least half the clause's; its served-against-attached parity through the server passes); the "
                   "summed adapter is read only if it passes"}
    P4._w(out, "gate", rec)  # noqa: SLF001
    print(">> gate:", json.dumps({"gate_ok": rec["gate_ok"], "dry_pass_through": dry_through,
                                  "control_ok": dec.get("control_ok"),
                                  "passes": {t: v.get("passes") for t, v in ad.items()},
                                  "reductions": {t: (v.get("control_rule") or {}).get("reduction")
                                                 for t, v in ad.items()},
                                  "clause": (ad.get("c1", {}).get("control_rule") or {}).get("clause_reduction")},
                                 default=str), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", default="gate", choices=["gate"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stage_gate(a.out, a.dry_run)


if __name__ == "__main__":
    main()
