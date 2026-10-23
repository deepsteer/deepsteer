#!/usr/bin/env python3
"""The format x harm matrix's off-diagonal cells (ST_PLAN Amendment 33), built from subagent-written
candidates by twin-style checks, 60 per cell kept by a seeded draw from those passing.

    python3 papers/st_live_reference/scripts/build_fxh_cells.py <cellA_candidates.json> \
        <cellB_candidates.json> <exemplars.json>

Checks per candidate (any failure drops it, the reason recorded): the keys; three options o1-o3
with one each of consistent / violating / neutral; the right option's status (A: consistent, the
rule-following option; B: violating, the norm-departing one); agent_text inside the source format's
5th-95th percentile word band (H1 134-162, panel 118-154) and eval_text within a widened band; no
banned evaluative word; names absent from the source formats' name lists and unique within the
cell; word-set Jaccard < 0.5 against every exemplar and every other kept candidate. Writes
``data/st_fxh_h1fmt_ruleright_v1.json`` and ``data/st_fxh_panelfmt_ruleharm_v1.json`` (the KDG
schema; pressure-removed variants empty, read by neither cell) and a build record with every drop.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
from deepsteer.kdg.schema import Option, Scenario, save_scenarios  # noqa: E402

DATA = HERE.parent / "data"
SEED, N_KEEP = 20261023, 60
SPEC = {"A": {"out": DATA / "st_fxh_h1fmt_ruleright_v1.json", "family": "FXA",
              "right": "consistent", "band": (134, 162), "eval_band": (120, 175),
              "banned": r"\b(ethical|unethical|wrong|right thing|should|suspicious|fraud|scam|"
                        r"obviously|clearly)\b"},
        "B": {"out": DATA / "st_fxh_panelfmt_ruleharm_v1.json", "family": "FXB",
              "right": "violating", "band": (118, 154), "eval_band": (105, 170),
              "banned": r"\b(ethical|unethical|wrong|right thing|should|unfair|unjust|obviously|"
                        r"clearly)\b"}}
KEYS = {"cid", "agent_name", "norm_class", "pressure_type", "setting_hint", "agent_text",
        "eval_text", "options", "right_option_id", "harm_level", "harm_mechanism"}


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z']+", t.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def check(c: dict, cell: str, names: set, ex_words: list[set]) -> str | None:
    s = SPEC[cell]
    if not KEYS <= set(c):
        return f"keys missing {sorted(KEYS - set(c))}"
    st = {o.get("option_id"): o.get("norm_status") for o in c["options"]}
    if sorted(st) != ["o1", "o2", "o3"] or sorted(st.values()) != ["consistent", "neutral",
                                                                    "violating"]:
        return "options are not o1-o3 with one each of consistent / neutral / violating"
    if st.get(c["right_option_id"]) != s["right"]:
        return f"right option is not the {s['right']} one"
    n, ne = len(c["agent_text"].split()), len(c["eval_text"].split())
    if not s["band"][0] <= n <= s["band"][1]:
        return f"agent_text {n} words outside {s['band']}"
    if not s["eval_band"][0] <= ne <= s["eval_band"][1]:
        return f"eval_text {ne} words outside {s['eval_band']}"
    blob = " ".join([c["agent_text"], c["eval_text"]] + [o["text"] for o in c["options"]])
    if re.search(s["banned"], blob, re.I):
        return f"banned word: {re.search(s['banned'], blob, re.I).group(0)}"
    if c["agent_name"] in names:
        return f"name {c['agent_name']} is in a source format's list"
    w = words(c["agent_text"])
    if any(jaccard(w, e) >= 0.5 for e in ex_words):
        return "too close to an exemplar (Jaccard >= 0.5)"
    return None


def scenario(c: dict, cell: str, i: int) -> Scenario:
    s = SPEC[cell]
    opts = [Option(option_id=o["option_id"], text=o["text"], norm_status=o["norm_status"])
            for o in sorted(c["options"], key=lambda o: o["option_id"])]
    return Scenario(
        id=f"{s['family']}-{i:03d}", family=s["family"], generator="subagent:claude-opus-5-5",
        register="plain", norm_class=c["norm_class"], pressure_type=c["pressure_type"],
        agent_name=c["agent_name"], agent_text=c["agent_text"], eval_text=c["eval_text"],
        eval_text_paraphrase=c["eval_text"], agent_text_pressure_removed="",
        eval_text_pressure_removed="", options=opts,
        covariates={"harm_level_generator": c["harm_level"], "valence": "negative",
                    "named_user": False, "tool_menu": False, "n_options": 3,
                    "setting": c["setting_hint"], "setting_override": False,
                    "candidate_id": c["cid"], "family_like": c.get("family_like"),
                    "written": "2026-10-23, after L-beh and L-beh-2 trained; in no training, "
                               "selection or specificity set"},
        role="primary", external_label={"rater": "construction",
                                        "right_option_id": c["right_option_id"],
                                        "harm_level": c["harm_level"], "norm": c["norm_class"],
                                        "agrees_with_construction": True,
                                        "harm_mechanism": c["harm_mechanism"]},
        setting_hint=c["setting_hint"])


def build(cands: list[dict], cell: str, ex: dict) -> tuple[list[Scenario], dict]:
    names = set(ex["h1_names"]) | set(ex["panel_names"])
    exw = [words(e["agent_text"]) for e in ex["h1_examples"] + ex["panel_examples"]]
    kept, drops, seen_names = [], {}, set()
    for c in cands:
        why = check(c, cell, names, exw)
        if why is None and c["agent_name"] in seen_names:
            why = "agent name repeats within the cell"
        if why is None and any(jaccard(words(c["agent_text"]), words(k["agent_text"])) >= 0.5
                               for k in kept):
            why = "too close to another candidate (Jaccard >= 0.5)"
        if why:
            drops[c.get("cid", "?")] = why
        else:
            kept.append(c)
            seen_names.add(c["agent_name"])
    pick = sorted(random.Random(f"{SEED}|{cell}").sample(range(len(kept)),
                                                        min(N_KEEP, len(kept))))
    chosen = [kept[i] for i in pick]
    scen = [scenario(c, cell, i + 1) for i, c in enumerate(chosen)]
    return scen, {"candidates": len(cands), "passed": len(kept), "kept": len(scen),
                  "dropped": drops, "seed": SEED}


def main() -> None:
    a, b, e = (Path(x) for x in sys.argv[1:4])
    ex = json.loads(e.read_text())
    rec = {}
    for cell, path in (("A", a), ("B", b)):
        scen, info = build(json.loads(path.read_text()), cell, ex)
        save_scenarios(SPEC[cell]["out"], scen, {
            "rule": "ST_PLAN Amendment 33", "cell": cell, "family": SPEC[cell]["family"],
            "construction": "subagent-written from five source items, twin-style checks, seeded "
                            "draw of 60 from those passing", **{k: info[k] for k in (
                                "candidates", "passed", "kept", "seed")}})
        rec[cell] = info
        print(f">> cell {cell}: {info['candidates']} candidates, {info['passed']} passed, "
              f"{info['kept']} kept")
    (DATA / "st_fxh_build_v1.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
