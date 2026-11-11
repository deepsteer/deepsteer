#!/usr/bin/env python3
"""Amendment 32 item 9: the content gate's scenarios (zero GPU; a Claude writer, no subject traces).

    content_gate_build.py sample   # the seeded tenet sample -> outputs/content_gate/tenets.json
    content_gate_build.py topup    # 24 more from the eligible tenets not sampled (the yield)
    content_gate_build.py build    # written + blind check -> data/content_scenarios_v1.json

**Tenets:** the C_anth v2 tenets of directing kinds (directive, prohibition, priority) that sit in
no T_core pair, matched or pending (`outputs/t01_tcore_votes_v2.json`), a seeded sample of 88 (so
about 80 survive the checks). **Writer** (a Claude subagent, `outputs/content_gate/written_*.json`):
per tenet, an AI agent's situation with no operator order, second person (``agent_text``) and
third person (``eval_text``), and three options tagged ``directed`` (the one the tenet directs),
``alternative`` (one a careful assistant without that tenet would plausibly take) and ``neutral``.
**Checks:** no 8-gram shared with the tenet's quote (lower-cased words); a blind second call
(`outputs/content_gate/blindcheck_*.json`), given the tenet and the scenario with the options in a
seeded order, names the directed option; a disagreement drops the scenario. Option ids o1-o3 are
assigned in a seeded order per scenario; ``norm_status`` is consistent (directed), violating
(alternative), neutral. The file loads with `deepsteer.kdg.schema.load_scenarios`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CST = HERE.parent
OUT = CST / "outputs/content_gate"
DATA = CST / "data/content_scenarios_v1.json"
KINDS = ("directive", "prohibition", "priority")
N_SAMPLE, SEED = 88, "20261111|cst_content_gate"
ROLE_STATUS = {"directed": "consistent", "alternative": "violating", "neutral": "neutral"}
# read before the build, never on a model's read: a written scenario that breaks the "no operator
# order" rule (its writer flagged it)
MANUAL_DROP = {"A027": "the situation carries a stop request on an oversight channel (an order)",
               **{t: "a manager or director makes a request inside the scene (an order)"
                  for t in ("A048", "A204", "A268")},
               "A271": "the choice turns on an instruction-like line inside a document (close to "
                       "the injected-order cell)"}
N_TOPUP = 24  # the seeded top-up from the eligible tenets not sampled (the first 88 yielded 67)


def eligible() -> list[dict]:
    """C_anth v2 tenets of the directing kinds outside every matched or pending T_core pair."""
    tv = json.loads((CST / "outputs/t01_tcore_votes_v2.json").read_text())
    anth = json.loads((CST / "data/tenets_C_anth_v2.json").read_text())["tenets"]
    ids = {t["tenet_id"] for t in anth}
    pairs = tv["matched_two_of_two"] + tv["pending_split"]
    excl = {x for p in pairs for x in p.split("|") if x in ids}
    return [t for t in anth if t["tenet_id"] not in excl and t["kind"] in KINDS]


def sample() -> list[dict]:
    pool = sorted(eligible(), key=lambda t: t["tenet_id"])
    pick = random.Random(SEED).sample(pool, N_SAMPLE)
    return [{k: t[k] for k in ("tenet_id", "section", "kind", "quote", "claim", "conditions")}
            for t in pick]


def topup() -> list[dict]:
    """A seeded draw of ``N_TOPUP`` from the eligible tenets the first sample left out."""
    have = {t["tenet_id"] for t in json.loads((OUT / "tenets.json").read_text())}
    pool = sorted((t for t in eligible() if t["tenet_id"] not in have),
                  key=lambda t: t["tenet_id"])
    pick = random.Random(f"{SEED}|topup").sample(pool, N_TOPUP)
    return [{k: t[k] for k in ("tenet_id", "section", "kind", "quote", "claim", "conditions")}
            for t in pick]


def all_tenets() -> dict[str, dict]:
    files = [OUT / "tenets.json"] + ([OUT / "tenets_topup.json"]
                                     if (OUT / "tenets_topup.json").exists() else [])
    return {t["tenet_id"]: t for f in files for t in json.loads(f.read_text())}


def grams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def option_order(tenet_id: str) -> list[str]:
    """The seeded order of the three roles as o1, o2, o3."""
    roles = ["directed", "alternative", "neutral"]
    random.Random(f"{SEED}|{tenet_id}").shuffle(roles)
    return roles


def scenario(i: int, w: dict, tenet: dict) -> dict:
    roles = option_order(tenet["tenet_id"])
    by = {o["role"]: o["text"] for o in w["options"]}
    opts = [{"option_id": f"o{j + 1}", "text": by[r], "norm_status": ROLE_STATUS[r],
             "tool_name": None, "tool_args": None} for j, r in enumerate(roles)]
    return {"id": f"CG-{i:03d}", "family": "CG", "generator": "claude-opus-5.5",
            "register": "plain", "norm_class": "c_anth_content", "pressure_type": tenet["kind"],
            "agent_name": w["agent_name"], "agent_text": w["agent_text"],
            "eval_text": w["eval_text"], "eval_text_paraphrase": w["eval_text"],
            "agent_text_pressure_removed": w["agent_text"],
            "eval_text_pressure_removed": w["eval_text"], "options": opts,
            "covariates": {"tenet_id": tenet["tenet_id"], "tenet_kind": tenet["kind"],
                           "section": tenet["section"],
                           "directed_option": f"o{roles.index('directed') + 1}"},
            "role": "primary", "twin_of": None, "f2_pushback": None, "external_label": None,
            "setting_hint": None, "eval_text_paraphrases": [w["eval_text"]],
            "eval_text_pressure_removed_paraphrases": [w["eval_text"]]}


def build() -> dict:
    from deepsteer.kdg.schema import SCHEMA_VERSION, load_scenarios

    tenets = all_tenets()
    written = {w["tenet_id"]: w for f in sorted(OUT.glob("written_*.json"))
               for w in json.loads(f.read_text())}
    blind = {b["tenet_id"]: b["directed_letter"] for f in sorted(OUT.glob("blindcheck_*.json"))
             for b in json.loads(f.read_text())}
    rec = {"sampled": len(tenets), "written": 0, "dropped": {}}
    keep = []
    for tid in sorted(tenets):
        w = written.get(tid)
        if w and "skip" in w:
            rec["dropped"][tid] = f"skipped by the writer: {w['skip']}"
            continue
        if not w or sorted(o["role"] for o in w.get("options", [])) != sorted(ROLE_STATUS):
            rec["dropped"][tid] = "not written or malformed"
            continue
        rec["written"] += 1
        if tid in MANUAL_DROP:
            rec["dropped"][tid] = MANUAL_DROP[tid]
            continue
        if grams(w["agent_text"] + " " + w["eval_text"] + " " + " ".join(
                o["text"] for o in w["options"])) & grams(tenets[tid]["quote"]):
            rec["dropped"][tid] = "shares an 8-gram with the tenet's quote"
            continue
        want = "ABC"[option_order(tid).index("directed")]
        if blind.get(tid) != want:
            rec["dropped"][tid] = f"blind check named {blind.get(tid)}, directed is {want}"
            continue
        keep.append((tid, w))
    scen = [scenario(i, w, tenets[tid]) for i, (tid, w) in enumerate(keep)]
    body = json.dumps(scen, sort_keys=True).encode()
    payload = {"schema_version": SCHEMA_VERSION, "template_version": "content_gate_v1",
               "metadata": {"family": "CG", "cell": "the content gate (CST Amendment 32 item 9)",
                            "generator": "claude-opus-5.5", "n_primary": len(scen),
                            "tenet_rule": "C_anth v2, directing kinds, outside every matched or "
                                          "pending T_core pair", "seed": SEED, "checks": rec},
               "scenario_set_sha256": hashlib.sha256(body).hexdigest(), "scenarios": scen}
    DATA.write_text(json.dumps(payload, indent=1) + "\n")
    load_scenarios(DATA)  # the file loads with the panel's schema
    rec["kept"] = len(scen)
    return rec


def blind_items() -> list[dict]:
    """The blind check's input: each written scenario's tenet and options in the seeded order
    (letters A-C), the roles withheld."""
    tenets = all_tenets()
    done = {b["tenet_id"] for f in OUT.glob("blindcheck_*.json") for b in json.loads(f.read_text())}
    items = []
    for f in sorted(OUT.glob("written_*.json")):
        for w in json.loads(f.read_text()):
            if "skip" in w or w["tenet_id"] in done or w["tenet_id"] in MANUAL_DROP:
                continue
            by = {o["role"]: o["text"] for o in w["options"]}
            items.append({"tenet_id": w["tenet_id"], "tenet": tenets[w["tenet_id"]]["quote"],
                          "situation": w["agent_text"],
                          "options": {L: by[r] for L, r in zip("ABC", option_order(
                              w["tenet_id"]))}})
    return items


if __name__ == "__main__":
    sys.path.insert(0, str(CST.parents[1]))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "sample":
        s = sample()
        (OUT / "tenets.json").write_text(json.dumps(s, indent=1) + "\n")
        print(len(eligible()), "eligible;", len(s), "sampled")
    elif cmd == "topup":
        t = topup()
        (OUT / "tenets_topup.json").write_text(json.dumps(t, indent=1) + "\n")
        for k in range(2):
            (OUT / f"tenets_batch_{4 + k}.json").write_text(json.dumps(t[k * 12:(k + 1) * 12],
                                                                       indent=1) + "\n")
        print(len(t), "top-up tenets")
    elif cmd == "blind_items":
        items = blind_items()
        name = sys.argv[2] if len(sys.argv) > 2 else "blind_items"
        (OUT / f"{name}.json").write_text(json.dumps(items, indent=1) + "\n")
        print(len(items), "blind items ->", name)
    else:
        print(json.dumps(build(), indent=1))
