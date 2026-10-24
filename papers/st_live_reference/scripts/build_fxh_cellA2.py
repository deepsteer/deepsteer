#!/usr/bin/env python3
"""Cell A′'s screening pool (ST_PLAN Amendment 36 item 4): rule-right items in H1's format with
subtler harm, from four subagent-written candidate sets, filtered by Amendment 33's twin-style
checks (``build_fxh_cells.check``) plus distance from every H1, panel and cell A item and from each
other. Every passing candidate enters the pool; the pod's dose-0 screen under ``none`` keeps the
60 in the operating band. Writes ``data/st_fxh_cellA2_pool_v1.json`` (the KDG schema) and a build
record with every drop.

    python3 papers/st_live_reference/scripts/build_fxh_cellA2.py <exemplars.json> \
        <cand_a.json> [<cand_b.json> ...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import build_fxh_cells as B  # noqa: E402

from deepsteer.kdg.schema import (  # noqa: E402
    Option,
    Scenario,
    load_scenario_dir,
    load_scenarios,
    save_scenarios,
)

DATA = HERE.parent / "data"
OUT = DATA / "st_fxh_cellA2_pool_v1.json"
FAMILY = "FXA2"
B.SPEC["A2"] = {**B.SPEC["A"], "out": OUT, "family": FAMILY}


def scenario(c: dict, i: int) -> Scenario:
    opts = [Option(option_id=o["option_id"], text=o["text"], norm_status=o["norm_status"])
            for o in sorted(c["options"], key=lambda o: o["option_id"])]
    return Scenario(
        id=f"{FAMILY}-{i:03d}", family=FAMILY, generator="subagent:claude-opus-5-5",
        register="plain", norm_class=c["norm_class"], pressure_type=c["pressure_type"],
        agent_name=c["agent_name"], agent_text=c["agent_text"], eval_text=c["eval_text"],
        eval_text_paraphrase=c["eval_text"], agent_text_pressure_removed="",
        eval_text_pressure_removed="", options=opts,
        covariates={"harm_level_generator": c["harm_level"], "valence": "negative",
                    "named_user": False, "tool_menu": False, "n_options": 3,
                    "setting": c["setting_hint"], "setting_override": False,
                    "candidate_id": c["cid"],
                    "written": "2026-10-24, after L-beh and L-beh-2 trained; in no training, "
                               "selection or specificity set; cell A′ screening pool"},
        role="primary", external_label={"rater": "construction",
                                        "right_option_id": c["right_option_id"],
                                        "harm_level": c["harm_level"], "norm": c["norm_class"],
                                        "agrees_with_construction": True,
                                        "harm_mechanism": c["harm_mechanism"]},
        setting_hint=c["setting_hint"])


FRESH = ("Aurelio Amarachi Bettina Caetano Dagny Eldon Fenella Gaspard Halvard Ines Jarrah "
         "Kalinda Leopoldo Marisa Nnamdi Odalys Pernille Quillon Rosalind Saoirse Thaddeus "
         "Ulrike Valeriu Wilhelmina Xiomara Yannick Zuzana Anselm Benedikta Corentin Delfina "
         "Evander Fiorella Gunnar Henrike Isidro Jovita Kasimir Liesel Mathurin Nerissa Osvaldo "
         "Pomona Radomir Sigrun Tobiah Undine Viggo Winslow Yolanda Zebulon Adelheid Bartholo "
         "Cosima Dragan Elowen Fabiano Gwendolen Horatio Ilse Annika Bastiaan Cyprian Dorotea "
         "Emeric Filomena Grigor Hyacinth Ignatius Jessamy Konstantin Lavinia Meinrad Noelani "
         "Oriel Philippa Quirin Ravenna Severin Tamsin Urbano Veronika Wendeline Xenia Yevgeni "
         "Zinnia Ambrosius Brigida Clement Desislava Eustace Florentyna Gideonne Hesper Isadora "
         "Jaromir Klementina Lysander Maelis Norberto").split()


def rename_repeats(cands: list[dict], taken: set[str]) -> dict[str, str]:
    """A name already used by an earlier candidate (the four sets were written independently) is
    replaced, as a whole word in every text field, by the next fresh name used nowhere; the map is
    recorded."""
    import re

    blob = " ".join(json.dumps(c) for c in cands)
    fresh = [n for n in FRESH if n not in taken and not re.search(rf"\b{n}\b", blob)]
    seen: set[str] = set()
    done = {}
    for c in cands:
        old = c["agent_name"]
        if old in seen:
            new = fresh.pop(0)
            rx = re.compile(rf"\b{re.escape(old)}\b")
            for k in ("agent_text", "eval_text"):
                c[k] = rx.sub(new, c[k])
            for o in c["options"]:
                o["text"] = rx.sub(new, o["text"])
            c["agent_name"] = new
            done[c["cid"]] = f"{old} -> {new}"
        seen.add(c["agent_name"])
    return done


def main() -> None:
    ex = json.loads(Path(sys.argv[1]).read_text())
    cands = [c for p in sys.argv[2:] for c in json.loads(Path(p).read_text())]
    names = set(ex["h1_names"]) | set(ex["panel_names"])
    H, _ = load_scenario_dir([HERE.parents[1] / "kdg_panel" / "data" / "harm_divergence" /
                              "hdiv_set.json"])
    panel, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                       "*_scenarios_*.json"))))
    cell_a = load_scenarios(DATA / "st_fxh_h1fmt_ruleright_v1.json")[0]
    names |= {s.agent_name for s in cell_a}
    ref = [B.words(s.agent_text) for s in list(H) + list(panel) + list(cell_a)]
    ref += [B.words(e["agent_text"]) for e in ex["h1_examples"] + ex["panel_examples"]]
    renamed = rename_repeats(cands, names)
    kept, drops, seen = [], {}, set()
    for c in cands:
        why = B.check(c, "A2", names, ref)
        if why is None and c["agent_name"] in seen:
            why = "agent name repeats within the pool"
        if why is None and any(B.jaccard(B.words(c["agent_text"]), B.words(k["agent_text"]))
                               >= 0.5 for k in kept):
            why = "too close to another candidate (Jaccard >= 0.5)"
        if why:
            drops[c.get("cid", "?")] = why
        else:
            kept.append(c)
            seen.add(c["agent_name"])
    scen = [scenario(c, i + 1) for i, c in enumerate(kept)]
    save_scenarios(OUT, scen, {"rule": "ST_PLAN Amendment 36 item 4", "cell": "A2",
                               "family": FAMILY, "candidates": len(cands), "passed": len(kept),
                               "construction": "four subagent-written domain sets (healthcare, "
                                               "money, public offices, workplaces), twin-style "
                                               "checks; the pod's screen keeps 60 in band"})
    rec = {"candidates": len(cands), "passed": len(kept), "dropped": drops, "renamed": renamed,
           "by_set": {p: len(json.loads(Path(p).read_text())) for p in sys.argv[2:]}}
    (DATA / "st_fxh_cellA2_build_v1.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f">> cell A′ pool: {len(cands)} candidates, {len(kept)} passed, {len(drops)} dropped")


if __name__ == "__main__":
    main()
