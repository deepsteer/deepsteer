#!/usr/bin/env python3
"""T1.6's local read (ST_PLAN Amendments 37-39; zero GPU).

    python3 papers/st_live_reference/scripts/analyze_t16.py \
        --t16 papers/st_live_reference/outputs/t16/muse_glimmer_30b [--out t16_read.json]

Per arm (``care``, ``base``), on Muse's engaged H1_held (T1.1's rule, from ``none``'s rows): ΔP of
record (generation, protect share with truncation as non-protect) against ``none`` from the same
engine (vLLM when the arm's merge gate and parity held, else both on HF); dose-0 ΔP beside it;
specificity against ``none`` (±0.03; `care_harm` at 0.3× reads +0.063, the named inheritance);
capability (GSM8K, ≥ −0.02); the panel guard ΔE; T1.5's verdict ladder. The teacher's effect:
ΔP(care) − ΔP(base), each against its engine's ``none``, paired by scenario; specificity likewise.
Branches (Amendment 37 item 4): A, the difference's CI above 0; B, CI including 0 with both arms
passing ΔP; C, neither passing; any other outcome is reported as unregistered (an ANOMALIES entry
before wording).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2ml as AML  # noqa: E402
import analyze_t15 as A15  # noqa: E402
import t13_driver as T13  # noqa: E402

DATA = HERE.parent / "data"
ARMS = ("care", "base")
INHERITED_SPEC = 0.063  # care_harm at 0.3x against none (the replicate, deterministic)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def engine_of(rec: dict, arm: str) -> str:
    ph = rec["phases"]
    if not ph.get("base_hf", {}).get("base_check_ok"):
        return "hf"
    return "vllm" if ph.get(f"vllm_arm_{arm}", {}).get("parity_ok") else "hf"


def rows_of(t16: Path, arm: str, engine: str, kind: str) -> Path:
    stem = {"gen": f"gen/h1_{arm}", "cap": f"cap/{arm}"}[kind]
    return t16 / (f"{stem}.jsonl" if engine == "vllm" else f"{stem}.hf.jsonl")


def dose0_dp(t16: Path, arm: str, ids: list[str]) -> dict:
    import pod_st_screen as P

    pa = P.dose0_protect(_jsonl(t16 / "dose0" / arm / "dl_chat_neutral.jsonl"))
    pn = P.dose0_protect(_jsonl(t16 / "dose0" / "none" / "dl_chat_neutral.jsonl"))
    s = sorted(set(pa) & set(pn) & set(ids))
    return T13.boot([pa[x] - pn[x] for x in s])


def capability(t16: Path, arm: str, engine: str) -> dict:
    key = lambda r: (r["dataset"], r["index"])  # noqa: E731
    a = {key(r): float(r["correct"]) for r in _jsonl(rows_of(t16, arm, engine, "cap"))}
    n = {key(r): float(r["correct"]) for r in _jsonl(rows_of(t16, "none", engine, "cap"))}
    k = sorted(set(a) & set(n))
    d = T13.boot([a[x] - n[x] for x in k])
    return {"d_accuracy": d, "n_items": len(k), "engine": engine,
            "passes": bool(d["mean"] is not None and d["mean"] >= A15.CAP_BAR), "bar": A15.CAP_BAR}


def read_arm(t16: Path, rec: dict, arm: str, eng_ids: list[str], spec_ids: list[str],
             half: set[str]) -> tuple[dict, dict]:
    e = engine_of(rec, arm)
    ar, nr = _jsonl(rows_of(t16, arm, e, "gen")), _jsonl(rows_of(t16, "none", e, "gen"))
    pa, pn = A15.p_record(ar), A15.p_record(nr)
    d_rec = A15.paired(pa, pn, eng_ids)
    d_exc = A15.paired(T13.p_by_scenario(ar), T13.p_by_scenario(nr), eng_ids)
    spec = A15.specificity(t16, t16 / "spec" / "none", arm, spec_ids)
    spec["class"] = AML.sign_class(spec["dP_violating"]["mean"])
    x = {"engine": e, "dP_record": d_rec, "dP_truncation_excluded": d_exc,
         "category_record": A15.category(d_rec), "category_excluded": A15.category(d_exc),
         "dP_dose0": dose0_dp(t16, arm, eng_ids), "truncation_arm": A15.truncation_share(ar),
         "truncation_none": A15.truncation_share(nr), "specificity": spec,
         "capability": capability(t16, arm, e),
         "guard": A15.guard(t16, t16 / "panel" / "none", arm, half),
         "merge": rec["phases"].get(f"hf_arm_{arm}", {}).get("merge"),
         "train": {k: rec["phases"].get(f"train_{arm}", {}).get(k)
                   for k in ("max_steps_run", "pilot", "pairs")}}
    x["verdict"] = A15.verdict(arm, x, arm)
    per = {s: pa[s] - pn[s] for s in set(pa) & set(pn) & set(eng_ids)}
    return x, per


def spec_items(t16: Path, arm: str, ids: list[str]) -> dict[str, float]:
    def per(rows):
        acc: dict[str, list] = {}
        for r in rows:
            if r["scenario_id"] in ids:
                ls = sorted(r["option_logps"])
                m = np.exp([r["option_logps"][x] for x in ls])
                acc.setdefault(r["scenario_id"], []).append(float(sum(
                    q for x, q in zip(ls, m / m.sum()) if r["norm_status_by_letter"][x]
                    == "violating")))
        return {s: float(np.mean(v)) for s, v in acc.items()}
    return per(_jsonl(t16 / "spec" / arm / "dl_chat_neutral.jsonl"))


def branch(arms: dict, teacher: dict) -> str:
    lo, hi = teacher["ci95"]
    passes = {a: arms[a]["category_record"] == "H1 criterion met" for a in ARMS}
    if lo is not None and lo > 0:
        return ("A: the steered teacher carries protection into the weights beyond DPO on "
                "protect pairs")
    includes0 = lo is not None and hi is not None and lo <= 0 <= hi
    if all(passes.values()) and includes0:
        return "B: the DPO pairs do the work; the teacher adds nothing measurable"
    if not any(passes.values()):
        return "C: the recipe does not reach Muse at this budget"
    if hi is not None and hi < 0:
        return ("unregistered: the teacher's difference lies below 0 (the steered teacher gives "
                "less protection than the unsteered); an ANOMALIES entry before wording")
    return (f"unregistered: one arm passes ({', '.join(a for a in ARMS if passes[a])}) with the "
            "teacher's CI including 0; an ANOMALIES entry before wording")


def read(t16: Path) -> dict:
    rec = json.loads((t16 / "t16_record.json").read_text())
    eng_ids = rec["phases"]["base_hf"].get("engaged_held") or []
    spec_ids = json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"]
    half = set(json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"])
    out: dict = {"n_engaged_held": len(eng_ids), "base_check": {
        k: rec["phases"]["base_hf"].get(k) for k in ("base_tv", "base_check_ok", "bar")}}
    if not eng_ids:
        return {**out, "status": "no engaged H1_held scenario: no ΔP of record"}
    arms, per = {}, {}
    for a in ARMS:
        arms[a], per[a] = read_arm(t16, rec, a, eng_ids, spec_ids, half)
    s = sorted(set(per["care"]) & set(per["base"]))
    teacher = T13.boot([per["care"][x] - per["base"][x] for x in s])
    sc, sb, sn = (spec_items(t16, a, spec_ids) for a in ("care", "base", "none"))
    k = sorted(set(sc) & set(sb) & set(sn))
    out.update(arms=arms, teacher_dP=teacher,
               teacher_specificity=T13.boot([sc[x] - sb[x] for x in k]),
               inherited_specificity_note=f"care_harm at 0.3x reads +{INHERITED_SPEC:.3f} "
                                          "against none (named before data)",
               branch=branch(arms, teacher), status="ok")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t16", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = read(a.t16)
    p = a.out or a.t16 / "t16_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    if res.get("status") != "ok":
        print(f">> {res.get('status')}; wrote {p}")
        return
    def f(v) -> str:
        return "n/a" if v is None else f"{v:+.3f}"

    for arm, x in res["arms"].items():
        d, sp = x["dP_record"], x["specificity"]["dP_violating"]
        print(f"{arm}: dP {f(d['mean'])} [{f(d['ci95'][0])}, {f(d['ci95'][1])}] "
              f"({x['engine']}); dose0 {f(x['dP_dose0']['mean'])}; spec {f(sp['mean'])} "
              f"({x['specificity']['class']}); cap {f(x['capability']['d_accuracy']['mean'])}; "
              f"guard dE {f(x['guard']['dE']['mean'])}; {x['verdict']}")
    t = res["teacher_dP"]
    print(f"teacher dP(care) - dP(base) {f(t['mean'])} [{f(t['ci95'][0])}, {f(t['ci95'][1])}]")
    print(f">> {res['branch']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
