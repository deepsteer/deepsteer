#!/usr/bin/env python3
"""CST T0.5: the evaluation-set 8-gram index, and the constitution body run through it (zero GPU).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t05_overlap_index.py

Every SDF document paraphrases the constitution, so the first question for CST_PLAN §2.3's rule
("drop any training item sharing a word 8-gram with an evaluation item") is how many evaluation
8-grams the constitution body itself carries: those would trip the rule on faithful documents.
Also runs the two documents' sections and the 360 refusal-retention prompts through the index as
calibration of the any-shared rule's base rate on unrelated prose. Writes counts and ids only.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "papers" / "mch_machiavelli" / "scripts")]

from deepsteer.constitutions import load_document  # noqa: E402
from deepsteer.corpus import OverlapIndex  # noqa: E402

KDG = ROOT / "papers" / "kdg_panel" / "data"
ST = ROOT / "papers" / "st_live_reference"
OUT = ROOT / "papers" / "cst_constitutional_steering" / "outputs" / "t05_overlap_index.json"


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


def eval_sets() -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    kdg = []
    for f in sorted(glob.glob(str(KDG / "*scenarios*.json"))):
        kdg += [t for s in json.loads(Path(f).read_text())["scenarios"] for t in _strings(s)]
    sets["kdg_panel_all_rounds"] = kdg
    hd = json.loads((KDG / "harm_divergence" / "hdiv_set.json").read_text())
    sets["h1_hdiv_set"] = [t for s in hd["scenarios"] for t in _strings(s)]
    for name in ("st_fxh_cellA2_forced_v1", "st_fxh_cellA2_twins_forced_v1"):
        d = json.loads((ST / "data" / f"{name}.json").read_text())
        sets[name] = [t for s in d["scenarios"] for t in _strings(s)]
    rr = json.loads((ST / "data" / "refusal_retention_v1.json").read_text())
    sets["refusal_retention_v1"] = _strings(rr["items"]) + _strings(rr["operator_control"])
    from deepsteer.datasets.morebench import load_morebench

    sets["morebench_public_500"] = [it.dilemma for it in load_morebench(subset="public")]
    from mch_g0_random import GAMES_TEST

    base = Path.home() / ".cache/deepsteer/machiavelli/ebb44e197f663603427882c609dbc9d5fb883d63"
    mach = []
    for g in GAMES_TEST:
        f = base / "bundle/game_data/scenarios" / f"{g}_scenarios_bfs_filled.json"
        mach += [v for v in json.loads(f.read_text()).values() if isinstance(v, str)]
    sets["machiavelli_test_30"] = mach
    return sets


def main() -> None:
    idx = OverlapIndex()
    sets = eval_sets()
    for k, v in sets.items():
        idx.add_set(k, v)
    pins = ST / "constitutions.yaml"
    tdir = ST / "outputs" / "constitutions"
    res: dict = {"index": idx.summary(), "not_indexed": [
        "capability set t15_capability_v1 (GSM8K/MMLU ids only; text not local)",
        "Layer A sets (ETHICS, MoralExceptQA, Scruples, DailyDilemmas): indexed at S1, after G0 "
        "fixes which enter"]}
    for doc_id in ("C_anth", "C_spec"):
        doc = load_document(doc_id, pins, tdir)
        secs = {f"{doc_id}:{s.section_id}": s.text for s in doc.sections()}
        r = idx.screen(secs)
        whole = idx.check(doc_id, doc.body)
        res[doc_id] = {"body_sha256": doc.body_sha256, "sections": r,
                       "body_shared_8grams": whole.n_shared, "body_sets_hit": whole.sets_hit}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(res["index"]))
    for d in ("C_anth", "C_spec"):
        r = res[d]
        print(d, "body shared 8-grams", r["body_shared_8grams"], r["body_sets_hit"],
              "| sections dropped", r["sections"]["n_dropped"], "of", r["sections"]["n_items"],
              r["sections"]["by_set"])
        for k, v in list(r["sections"]["examples"].items())[:6]:
            print("   ", k, "|", v)


if __name__ == "__main__":
    main()
