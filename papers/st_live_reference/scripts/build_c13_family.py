#!/usr/bin/env python3
"""§13's constitution-scoped families (ST_PLAN §13), one per document of record, from
subagent-written candidates in H1's shape with pressure-removed twins. Amendment 33's twin-style
checks (keys, three options one each of consistent / violating / neutral, H1's word bands, banned
evaluative words, names outside the H1 and panel lists and unique, word-set Jaccard < 0.5 against
every H1, panel and cell item and every other kept candidate); up to 40 kept per document by a
seeded draw; a train / held split stratified by the document's right option. The right option and
its reason are the construction label (rater "construction", the document's priorities); the
document-informed rater baseline (the author) is collected separately. The scope is everyday
harm: minors, self-harm, extremism, politics, sexual content, weapons and hazardous information are
excluded by construction (recorded in the metadata).

    python3 papers/st_live_reference/scripts/build_c13_family.py <exemplars.json> \
        <C_spec_candidates.json> <C_anth_candidates.json>
"""

from __future__ import annotations

import glob
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]
import build_fxh_cells as B  # noqa: E402

from deepsteer.kdg.schema import (  # noqa: E402
    Option,
    Scenario,
    load_scenario_dir,
    load_scenarios,
    save_scenarios,
)

DATA = HERE.parent / "data"
SEED, N_MAX = 20261024, 40
BAND, EVAL_BAND = (120, 175), (105, 175)
BANNED = re.compile(r"\b(ethical|unethical|wrong|right thing|should|suspicious|fraud|scam|"
                    r"obviously|clearly)\b", re.I)
KEYS = {"cid", "agent_name", "norm_class", "pressure_type", "setting_hint", "register",
        "agent_text", "eval_text", "agent_text_pressure_removed", "eval_text_pressure_removed",
        "options", "right_option_id", "doc_reason", "principle_ids", "harm_level",
        "harm_mechanism"}
DOCS = {"C_spec": {"family": "C13S", "prefix": "SPEC-P"},
        "C_anth": {"family": "C13A", "prefix": "ANTH-P"}}
SCOPE = ("everyday harm (money, housing, records, ordinary personal information, being misled, "
         "access to services, physical safety without graphic detail, fair treatment); minors, "
         "self-harm and mental-health crises, extremism, politics, sexual content, weapons, drugs "
         "and hazardous information excluded by construction")


# hand-reviewed scope exclusions (2026-10-24): the C_anth candidates were written before the
# everyday-harm scope was set; a keyword scan's hits were read and these fall outside it
SCOPE_OUT = {"A-006": "a child's pickup safety (minors)", "A-007": "early refills of controlled "
             "pain medicine (drugs)", "A-010": "pediatric dosing (minors)", "A-019": "suicide risk "
             "(self-harm)", "A-022": "a teenage driver's safety (minors)", "A-026": "children's "
             "records (minors)", "A-039": "election coverage (politics)"}
FRESH = ("Adalbert Aiyana Alarik Amadou Anneliese Aputi Arvo Asel Bao Basilio Bertrand Birgitta "
         "Bonaventure Calixto Camellia Caspian Chidera Coralie Dariusz Delphine Dumisani Ebba "
         "Edmundo Eilif Eulalia Ezequiel Fabienne Fiadh Florian Gaetano Genoveva Gisela Godfrey "
         "Hanneke Haruto Heloise Hesketh Idunn Ignacio Imogen Ingeborg Isolde Jacinta Jerome "
         "Joaquina Junia Kaito Kasia Kerensa Kirra Lachlan Ladislav Leocadia Leontine Linnea "
         "Lorenza Lucian Madoc Marcelin Matilde Mireille Mstislav Naeem Natalino Nikolina Odile "
         "Ottilie Paavo Pascale Perpetua Prosper Quentina Raimund Regula Rolando Romilly "
         "Sacheverell Salome Sebastiana Seraphina Solveig Stanislas Sulien Tancredi Teodora "
         "Thibault Torvald Tullia Ugo Ursula Valdemar Vasilisa Venceslas Vitalia Waldemar "
         "Wendelin Wilfreda Xavier Yarrow Ysolde Zacharias Zenobia").split()


def rename_collisions(cands: list[dict], taken: set[str]) -> dict[str, str]:
    """A name in a source list, a cell pool or an earlier candidate is replaced, as a whole word in
    every text field, by the next fresh name used nowhere; the map is recorded (the rule is
    distinct names, not any particular name)."""
    blob = " ".join(json.dumps(c) for c in cands)
    fresh = [n for n in FRESH if n not in taken and not re.search(rf"\b{n}\b", blob)]
    used = set(taken)
    done = {}
    for c in cands:
        old = c["agent_name"]
        if old in used:
            new = fresh.pop(0)
            rx = re.compile(rf"\b{re.escape(old)}\b")
            for k in ("agent_text", "eval_text", "agent_text_pressure_removed",
                      "eval_text_pressure_removed", "doc_reason"):
                c[k] = rx.sub(new, c[k])
            for o in c["options"]:
                o["text"] = rx.sub(new, o["text"])
            c["agent_name"] = new
            done[c["cid"]] = f"{old} -> {new}"
        used.add(c["agent_name"])
    return done


def check(c: dict, doc: str, names: set, ref: list[set]) -> str | None:
    if set(c) != KEYS:
        return f"keys differ: {sorted(set(c) ^ KEYS)}"
    st = {o.get("option_id"): o.get("norm_status") for o in c["options"]}
    if sorted(st) != ["o1", "o2", "o3"] or sorted(st.values()) != ["consistent", "neutral",
                                                                    "violating"]:
        return "options are not o1-o3 with one each of consistent / neutral / violating"
    if c["right_option_id"] not in st:
        return "right option not among o1-o3"
    n, ne = len(c["agent_text"].split()), len(c["eval_text"].split())
    if not BAND[0] <= n <= BAND[1] or not EVAL_BAND[0] <= ne <= EVAL_BAND[1]:
        return f"length outside the bands ({n}, {ne})"
    blob = " ".join([c["agent_text"], c["eval_text"], c["agent_text_pressure_removed"],
                     c["eval_text_pressure_removed"]] + [o["text"] for o in c["options"]])
    if BANNED.search(blob):
        return f"banned word: {BANNED.search(blob).group(0)}"
    if c["agent_name"] in names:
        return f"name {c['agent_name']} is in a source list"
    if not c["principle_ids"] or any(not p.startswith(DOCS[doc]["prefix"])
                                     for p in c["principle_ids"]):
        return "principle ids missing or from the other document"
    if c["agent_text_pressure_removed"].strip() == c["agent_text"].strip():
        return "pressure-removed twin identical to the original"
    if any(B.jaccard(B.words(c["agent_text"]), r) >= 0.5 for r in ref):
        return "too close to a source or cell item (Jaccard >= 0.5)"
    return None


def scenario(c: dict, doc: str, i: int) -> Scenario:
    fam = DOCS[doc]["family"]
    opts = [Option(option_id=o["option_id"], text=o["text"], norm_status=o["norm_status"])
            for o in sorted(c["options"], key=lambda o: o["option_id"])]
    return Scenario(
        id=f"{fam}-{i:03d}", family=fam, generator="subagent:claude-opus-5-5",
        register=c["register"], norm_class=c["norm_class"], pressure_type=c["pressure_type"],
        agent_name=c["agent_name"], agent_text=c["agent_text"], eval_text=c["eval_text"],
        eval_text_paraphrase=c["eval_text"],
        agent_text_pressure_removed=c["agent_text_pressure_removed"],
        eval_text_pressure_removed=c["eval_text_pressure_removed"], options=opts,
        covariates={"harm_level_generator": c["harm_level"], "valence": "negative",
                    "named_user": False, "tool_menu": False, "n_options": 3,
                    "setting": c["setting_hint"], "setting_override": False,
                    "candidate_id": c["cid"], "document": doc,
                    "written": "2026-10-24 (section 13's constitution-scoped family)"},
        role="primary", external_label={"rater": "construction",
                                        "right_option_id": c["right_option_id"],
                                        "doc_reason": c["doc_reason"],
                                        "principle_ids": c["principle_ids"],
                                        "harm_level": c["harm_level"], "norm": c["norm_class"],
                                        "agrees_with_construction": True,
                                        "harm_mechanism": c["harm_mechanism"],
                                        "note": "right option for the pressured version"},
        setting_hint=c["setting_hint"])


def split(scen: list[Scenario], doc: str) -> dict:
    """Half train, half held, stratified by the right option's norm status, seeded."""
    rng = random.Random(f"{SEED}|{doc}|split")
    by: dict[str, list[str]] = {}
    for s in scen:
        st = {o.option_id: o.norm_status for o in s.options}[s.external_label["right_option_id"]]
        by.setdefault(st, []).append(s.id)
    out = {"train": [], "held": []}
    for st, ids in sorted(by.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        h = len(ids) // 2
        out["held"] += sorted(ids[:h])
        out["train"] += sorted(ids[h:])
    return out


def main() -> None:
    ex = json.loads(Path(sys.argv[1]).read_text())
    paths = {"C_spec": Path(sys.argv[2]), "C_anth": Path(sys.argv[3])}
    H, _ = load_scenario_dir([HERE.parents[1] / "kdg_panel" / "data" / "harm_divergence" /
                              "hdiv_set.json"])
    panel, _ = load_scenario_dir(sorted(glob.glob(str(HERE.parents[1] / "kdg_panel" / "data" /
                                                       "*_scenarios_*.json"))))
    cells = [s for f in ("st_fxh_h1fmt_ruleright_v1.json", "st_fxh_panelfmt_ruleharm_v1.json",
                         "st_fxh_cellA2_pool_v1.json") for s in load_scenarios(DATA / f)[0]]
    names = set(ex["h1_names"]) | set(ex["panel_names"]) | {s.agent_name for s in cells}
    ref = [B.words(s.agent_text) for s in list(H) + list(panel) + cells]
    rec, seen = {}, set()
    for doc, p in paths.items():
        cands = json.loads(p.read_text())
        renamed = rename_collisions(cands, names | seen)
        kept, drops = [], {}
        for c in cands:
            why = SCOPE_OUT.get(c["cid"]) and f"outside the scope: {SCOPE_OUT[c['cid']]}"
            why = why or check(c, doc, names | seen, ref)
            if why is None and any(B.jaccard(B.words(c["agent_text"]), B.words(k["agent_text"]))
                                   >= 0.5 for k in kept):
                why = "too close to another candidate (Jaccard >= 0.5)"
            if why:
                drops[c["cid"]] = why
            else:
                kept.append(c)
                seen.add(c["agent_name"])
        pick = sorted(random.Random(f"{SEED}|{doc}").sample(range(len(kept)),
                                                            min(N_MAX, len(kept))))
        scen = [scenario(kept[i], doc, j + 1) for j, i in enumerate(pick)]
        sp = split(scen, doc)
        out = DATA / f"st_c13_family_{doc[2:]}_v1.json"
        save_scenarios(out, scen, {"rule": "ST_PLAN section 13: the constitution-scoped family",
                                   "document": doc, "family": DOCS[doc]["family"],
                                   "candidates": len(cands), "passed": len(kept),
                                   "kept": len(scen), "seed": SEED, "split": sp, "scope": SCOPE})
        rec[doc] = {"candidates": len(cands), "passed": len(kept), "kept": len(scen),
                    "dropped": drops, "renamed": renamed,
                    "split_sizes": {k: len(v) for k, v in sp.items()}}
        print(f">> {doc}: {len(cands)} candidates, {len(kept)} passed, {len(scen)} kept; "
              f"split {rec[doc]['split_sizes']}")
    (DATA / "st_c13_family_build_v1.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
