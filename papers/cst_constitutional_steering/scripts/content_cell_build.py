#!/usr/bin/env python3
"""Amendment 37: the selected content cell's scenarios (zero GPU; Claude writers; no traces).

    content_cell_build.py slots         # the 412 slots and the writers' batches
    content_cell_build.py blind_items   # the blind check's input (roles withheld)
    content_cell_build.py build         # written + checks -> data/content_cell_v1.json

**Slots:** each eligible tenet (`content_gate_build.eligible`: C_anth v2, the directing kinds,
outside every matched or pending T_core pair) gets 6 slots if it is a priority tenet, 4 if a
prohibition, 1 if a directive; a tenet's slots share a batch so its writer can vary the setting.
**Checks, in order:** the three roles present (or a writer's skip); a hand drop list read before any
model read; no 8-gram shared with the tenet's quote, with any pressure scenario's text
(`outputs/pressure/scenarios.json`, the cell is kept apart from them) or with an earlier cell
scenario's situation; the blind check names the directed option. Option ids are assigned in a
seeded order per slot. **Evaluation only:** no cell scenario enters a training set. The selection
(at most 4 per tenet) and the confirmation run on the next pod (`content_cell_read.py`).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content_gate_build as G  # noqa: E402

CST = HERE.parent
OUT = CST / "outputs/content_cell"
DATA = CST / "data/content_cell_v1.json"
PER_KIND = {"priority": 6, "prohibition": 4, "directive": 1}
SEED = "20261111|cst_content_cell"
BATCH = 26
MANUAL_DROP: dict[str, str] = {  # slot id -> reason, from the writers' flags, before any read
    "A166b-1": "near the no-weapons rule (reactor control rods), the writer's flag",
    "A166b-3": "near the no-weapons rule (gas attacks, history), the writer's flag",
}
# a writer's first attempt at batch 16 was stopped by a safety classifier; it skipped A292's six
# slots, and declined batches are never retried or rephrased, so they stay skipped


def slots() -> list[dict]:
    out = []
    for t in sorted(G.eligible(), key=lambda t: t["tenet_id"]):
        for j in range(PER_KIND[t["kind"]]):
            out.append({"slot_id": f"{t['tenet_id']}-{j}", "j": j, "of": PER_KIND[t["kind"]]}
                       | {k: t[k] for k in ("tenet_id", "section", "kind", "quote", "claim",
                                            "conditions")})
    return out


def batches(sl: list[dict]) -> list[list[dict]]:
    """Whole tenets per batch, about ``BATCH`` slots each, in a seeded tenet order."""
    by: dict[str, list[dict]] = {}
    for s in sl:
        by.setdefault(s["tenet_id"], []).append(s)
    tids = sorted(by)
    random.Random(SEED).shuffle(tids)
    out, cur = [], []
    for t in tids:
        if cur and len(cur) + len(by[t]) > BATCH:
            out.append(cur)
            cur = []
        cur += by[t]
    return out + ([cur] if cur else [])


def option_order(slot_id: str) -> list[str]:
    roles = ["directed", "alternative", "neutral"]
    random.Random(f"{SEED}|{slot_id}").shuffle(roles)
    return roles


def pressure_grams() -> set:
    sc = json.loads((CST / "outputs/pressure/scenarios.json").read_text())["scenarios"]
    g: set = set()
    for s in sc:
        for k in ("system_message", "user_message", "standing_instruction"):
            if s.get(k):
                g |= G.grams(s[k])
    return g


def scenario(i: int, w: dict, slot: dict) -> dict:
    roles = option_order(slot["slot_id"])
    by = {o["role"]: o["text"] for o in w["options"]}
    opts = [{"option_id": f"o{j + 1}", "text": by[r], "norm_status": G.ROLE_STATUS[r],
             "tool_name": None, "tool_args": None} for j, r in enumerate(roles)]
    return {"id": f"CC-{i:03d}", "family": "CC", "generator": "claude-opus-5.5",
            "register": "plain", "norm_class": "c_anth_content_cell",
            "pressure_type": slot["kind"], "agent_name": w["agent_name"],
            "agent_text": w["agent_text"], "eval_text": w["eval_text"],
            "eval_text_paraphrase": w["eval_text"], "agent_text_pressure_removed": w["agent_text"],
            "eval_text_pressure_removed": w["eval_text"], "options": opts,
            "covariates": {"tenet_id": slot["tenet_id"], "tenet_kind": slot["kind"],
                           "slot_id": slot["slot_id"], "section": slot["section"],
                           "directed_option": f"o{roles.index('directed') + 1}",
                           "evaluation_only": True},
            "role": "primary", "twin_of": None, "f2_pushback": None, "external_label": None,
            "setting_hint": None, "eval_text_paraphrases": [w["eval_text"]],
            "eval_text_pressure_removed_paraphrases": [w["eval_text"]]}


def written() -> dict[str, dict]:
    return {w["slot_id"]: w for f in sorted(OUT.glob("written_*.json"))
            for w in json.loads(f.read_text())}


def blind_items() -> list[dict]:
    done = {b["slot_id"] for f in OUT.glob("blindcheck_*.json") for b in json.loads(f.read_text())}
    sl = {s["slot_id"]: s for s in slots()}
    items = []
    for sid, w in sorted(written().items()):
        if "skip" in w or sid in done or sid in MANUAL_DROP:
            continue
        by = {o["role"]: o["text"] for o in w["options"]}
        items.append({"slot_id": sid, "tenet": sl[sid]["quote"], "situation": w["agent_text"],
                      "options": {L: by[r] for L, r in zip("ABC", option_order(sid))}})
    return items


def build() -> dict:
    from deepsteer.kdg.schema import SCHEMA_VERSION, load_scenarios

    sl = {s["slot_id"]: s for s in slots()}
    wr = written()
    blind = {b["slot_id"]: b["directed_letter"] for f in sorted(OUT.glob("blindcheck_*.json"))
             for b in json.loads(f.read_text())}
    pg = pressure_grams()
    seen: set = set()
    rec: dict = {"slots": len(sl), "written": 0, "dropped": {}}
    keep = []
    for sid in sorted(sl):
        w = wr.get(sid)
        if w and "skip" in w:
            rec["dropped"][sid] = f"skipped by the writer: {w['skip']}"
            continue
        if not w or sorted(o["role"] for o in w.get("options", [])) != sorted(G.ROLE_STATUS):
            rec["dropped"][sid] = "not written or malformed"
            continue
        rec["written"] += 1
        if sid in MANUAL_DROP:
            rec["dropped"][sid] = MANUAL_DROP[sid]
            continue
        text = " ".join([w["agent_text"], w["eval_text"]] + [o["text"] for o in w["options"]])
        g = G.grams(text)
        why = ("shares an 8-gram with the tenet's quote" if g & G.grams(sl[sid]["quote"]) else
               "shares an 8-gram with a pressure scenario" if g & pg else
               "shares an 8-gram with an earlier cell scenario" if G.grams(w["agent_text"]) & seen
               else None)
        if why:
            rec["dropped"][sid] = why
            continue
        want = "ABC"[option_order(sid).index("directed")]
        if blind.get(sid) != want:
            rec["dropped"][sid] = f"blind check named {blind.get(sid)}, directed is {want}"
            continue
        seen |= G.grams(w["agent_text"])
        keep.append(sid)
    scen = [scenario(i, wr[sid], sl[sid]) for i, sid in enumerate(keep)]
    body = json.dumps(scen, sort_keys=True).encode()
    by_kind: dict[str, int] = {}
    for sid in keep:
        by_kind[sl[sid]["kind"]] = by_kind.get(sl[sid]["kind"], 0) + 1
    rec |= {"kept": len(scen), "kept_by_kind": by_kind}
    payload = {"schema_version": SCHEMA_VERSION, "template_version": "content_cell_v1",
               "metadata": {"family": "CC", "cell": "the selected content cell (CST Amendment 37), "
                            "evaluation only", "generator": "claude-opus-5.5",
                            "n_primary": len(scen), "per_kind": PER_KIND, "seed": SEED,
                            "checks": rec},
               "scenario_set_sha256": hashlib.sha256(body).hexdigest(), "scenarios": scen}
    DATA.write_text(json.dumps(payload, indent=1) + "\n")
    load_scenarios(DATA)
    return rec


if __name__ == "__main__":
    sys.path.insert(0, str(CST.parents[1]))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "slots"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "slots":
        sl = slots()
        bs = batches(sl)
        for k, b in enumerate(bs):
            (OUT / f"batch_{k:02d}.json").write_text(json.dumps(b, indent=1) + "\n")
        print(len(sl), "slots in", len(bs), "batches", [len(b) for b in bs])
    elif cmd == "blind_items":
        it = blind_items()
        name = sys.argv[2] if len(sys.argv) > 2 else "blind_items"
        (OUT / f"{name}.json").write_text(json.dumps(it, indent=1) + "\n")
        print(len(it), "blind items ->", name)
    else:
        print(json.dumps(build(), indent=1))
