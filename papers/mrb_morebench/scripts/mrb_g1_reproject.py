"""MRB-G1 projection under MRB_PLAN v0.4 (zero GPU, from VALIDATE's committed outputs).

    mrb_g1_reproject.py [--validate-dir DIR] [--envelope-h 10.3] [--validate-hours 1.857]

Two changes to VALIDATE's projection (``mrb_g1_pod.py project``), both registered in MRB_PLAN v0.4:

1. Generation counts only what the chosen tiers judge: each model generates k = 4 when its k 2-3
   tier is kept, else k = 2 (plus the fork's k = 1 on the ladder subset).
2. The judge is priced by output tokens per call for each unit type, at VALIDATE's measured
   output tokens per second. VALIDATE's GPT-OSS traces were empty (MRB-A8), so GPT-OSS trace units
   (retest and judge fork included) are priced at Qwen's trace rate, the only measured rate for a
   non-empty trace.

The cut levels and their order are unchanged: the first level that fits is chosen, else STOP. The
watchdog's wall cap is 15% over the chosen projection, clipped to the envelope's remainder.
Writes ``DIR/projection.json``; the envelope-8 projection it replaces is kept beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mrb_g1_pod as P  # noqa: E402

DEFAULT_DIR = P.REPO / "papers" / "mrb_morebench" / "outputs" / "g1_validate"


def unit_type(unit: str) -> str:
    """'canned', or '<model>/<field>' for a generation unit (retest and judge fork included)."""
    parts = unit.split("/")
    return "canned" if parts[0] == "canned" else f"{parts[0]}/{parts[4]}"


def measured_rates(vdir: Path) -> dict:
    """Output tokens per judge call by unit type, from VALIDATE's tracked judge rows."""
    tot, n = defaultdict(int), defaultdict(int)
    for line in (vdir / "judge" / "shard0.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["pass"] == 1 and r["judge"] == "default":  # the base rate, not the retest or fork
            tot[unit_type(r["unit"])] += r["n_tokens"]
            n[unit_type(r["unit"])] += 1
    return {t: tot[t] / n[t] for t in tot}


def reproject(vdir: Path, envelope_h: float, validate_h: float) -> dict:
    items = P._items()
    crit = {it.scenario_id: len(it.criteria) for it in items}
    gen_s = {m: json.loads((vdir / "gen" / f"{m}.summary.json").read_text()) for m in P.GEN}
    judge_s = json.loads((vdir / "judge" / "shard0.summary.json").read_text())
    src = vdir / "projection_envelope8.json"
    old = json.loads((src if src.exists() else vdir / "projection.json").read_text())
    rates = measured_rates(vdir)
    priced = dict(rates, **{"gpt_oss_20b/trace": rates["qwen3_30b/trace"]})
    n_fork = sum(P.in_ladder_subset(it.idx) for it in items)
    remaining = envelope_h - validate_h
    rows, chosen = [], None
    for name, tiers in P.CUTS:
        k = {"gpt_oss_20b": 4 if 2 in tiers else 2, "qwen3_30b": 4 if 3 in tiers else 2}
        gen_wall_h = max((len(items) * k[m] + n_fork) * gen_s[m]["decoded_tokens"]
                         / gen_s[m]["jobs_run"] / gen_s[m]["tokens_per_s"] / 3600 for m in P.GEN)
        fake = {}
        for it in items:
            for m in P.GEN:
                for kk in range(k[m]):
                    fake[f"{m}/default/{it.scenario_id}/k{kk}"] = {"trace": "", "response": ""}
                if P.in_ladder_subset(it.idx):
                    fake[f"{m}/fork/{it.scenario_id}/k0"] = {"trace": "", "response": ""}
        units = [u for u in P.judge_units(items, fake, validate=False) if u["tier"] in tiers]
        calls = sum(crit[u["sid"]] for u in units)
        tokens = sum(crit[u["sid"]] * priced[unit_type(u["unit"])] for u in units)
        judge_wall_h = tokens / judge_s["tokens_per_s"] / 3600
        wall = old["measured"]["setup_h"] + gen_wall_h + judge_wall_h + old["measured"]["margin_h"]
        fits = 2 * wall <= remaining
        rows.append({"cut": name, "tiers": sorted(tiers), "k": k, "judge_calls": calls,
                     "judge_output_tokens": round(tokens), "gen_wall_h": round(gen_wall_h, 3),
                     "judge_wall_h": round(judge_wall_h, 3), "session_wall_h": round(wall, 3),
                     "session_a100_h": round(2 * wall, 3), "fits": fits})
        if fits and chosen is None:
            chosen = rows[-1]
    return {"amendment": "MRB_PLAN v0.4", "replaces": "projection_envelope8.json",
            "measured": {**old["measured"], "judge_tokens_per_call_by_type": rates},
            "priced_tokens_per_call_by_type": priced,
            "envelope_h": envelope_h, "validate_h": validate_h, "remaining_h": round(remaining, 3),
            "levels": rows, "chosen": chosen, "decision": chosen["cut"] if chosen else "STOP",
            # the watchdog's cap: 15% over the projection, but never past the envelope
            "session_max_wall_h": (round(min(chosen["session_wall_h"] * 1.15, remaining / 2), 3)
                                   if chosen else None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--envelope-h", type=float, default=10.3)
    ap.add_argument("--validate-hours", type=float, default=1.857)
    a = ap.parse_args()
    rec = reproject(a.validate_dir, a.envelope_h, a.validate_hours)
    keep = a.validate_dir / "projection_envelope8.json"
    if not keep.exists():
        (a.validate_dir / "projection.json").rename(keep)
    (a.validate_dir / "projection.json").write_text(json.dumps(rec, indent=1))
    for r in rec["levels"]:
        print(f"{r['cut']:6s} k={r['k']} calls {r['judge_calls']:7d} gen {r['gen_wall_h']:.3f} h "
              f"judge {r['judge_wall_h']:.3f} h -> {r['session_a100_h']:.2f} A100-h "
              f"{'fits' if r['fits'] else 'over'}")
    print("decision:", rec["decision"], "| cap", rec["session_max_wall_h"], "h wall")


if __name__ == "__main__":
    main()
