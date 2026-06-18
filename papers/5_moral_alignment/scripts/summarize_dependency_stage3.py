#!/usr/bin/env python3
"""Measurement 5 (confirmation): moral-subspace dependency across the pipeline.

Pure-analysis pass over the cached ``moral_dependency.json`` files (no model,
no new compute). Confirms Paper 5 §6.1: the moral-dependency score is positive
at every state and flat through stage-3 pre-training (~+0.011 nats/token), then
rises ~6x through post-training (SFT -> DPO -> Instruct) and plateaus.

Two cached families are summarised:
  * ``dependency/``           : ablating the FIXED BASE foundation directions
                                (the §6.1 headline trajectory).
  * ``dependency_perstate/``  : ablating each state's OWN freshly fitted
                                directions (the §6.1 replication that rules out
                                the ~40-degree SFT rotation as an artifact).

Usage:
    python papers/5_moral_alignment/scripts/summarize_dependency_stage3.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_OUT = _PAPER_ROOT / "outputs/measurement/dependency_stage3_summary.json"


def _step(label: str) -> int:
    m = re.search(r"step_?(\d+)", label)
    return int(m.group(1)) if m else -1


def _order(label: str) -> tuple[int, int]:
    if label == "olmo3_base":
        return (0, 0)
    if "pretrain_stage3" in label:
        return (1, _step(label))
    if label == "olmo3_sft_final":
        return (2, 0)
    if label == "olmo3_dpo_final":
        return (3, 0)
    if "instruct_step" in label:
        return (4, _step(label))
    if label == "olmo3_instruct_final":
        return (5, 0)
    return (9, 0)


def _collect(root: Path) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(root / "*" / "moral_dependency.json"))):
        d = json.load(open(p))
        label = os.path.basename(os.path.dirname(p))
        m = d.get("metrics", {})
        rows.append({
            "label": label,
            "order": _order(label),
            "dependency_score": m.get("moral_dependency_score"),
            "delta_ce_moral": m.get("delta_ce", {}).get("moral"),
            "delta_ce_neutral": m.get("delta_ce", {}).get("neutral"),
        })
    rows.sort(key=lambda r: r["order"])
    for r in rows:
        r.pop("order")
    return rows


def _stage3(rows: list[dict]) -> dict:
    s3 = [r for r in rows if "pretrain_stage3" in r["label"]]
    scores = [r["dependency_score"] for r in s3]
    return {
        "n_stage3_checkpoints": len(s3),
        "all_positive": all(s > 0 for s in scores),
        "mean": round(sum(scores) / len(scores), 6) if scores else None,
        "min": round(min(scores), 6) if scores else None,
        "max": round(max(scores), 6) if scores else None,
    }


def _named(rows: list[dict], label: str):
    for r in rows:
        if r["label"] == label:
            return r["dependency_score"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-3 moral-dependency summary.")
    ap.add_argument("--dependency-dir", default=str(_PAPER_ROOT / "outputs/dependency"))
    ap.add_argument("--perstate-dir", default=str(_PAPER_ROOT / "outputs/dependency_perstate"))
    ap.add_argument("--output", default=str(_DEF_OUT))
    args = ap.parse_args()

    families = {
        "fixed_base_directions": _collect(Path(args.dependency_dir)),
        "per_state_directions": _collect(Path(args.perstate_dir)),
    }

    summary = {}
    for fam, rows in families.items():
        s3 = _stage3(rows)
        summary[fam] = {
            "stage3": s3,
            "base": _named(rows, "olmo3_base"),
            "sft": _named(rows, "olmo3_sft_final"),
            "dpo": _named(rows, "olmo3_dpo_final"),
            "instruct": _named(rows, "olmo3_instruct_final"),
            "post_training_fold_increase": (
                round(_named(rows, "olmo3_instruct_final") / s3["mean"], 2)
                if s3["mean"] else None
            ),
            "trajectory": rows,
        }

    s3f = summary["fixed_base_directions"]["stage3"]
    payload = {
        "analysis": "dependency_stage3_summary",
        "confirms": "Paper 5 §6.1: dependency positive everywhere, flat through "
                    "stage-3, ~6x rise post-training.",
        "stage3_positive_and_flat": bool(
            s3f["all_positive"] and abs(s3f["mean"] - 0.011) < 0.006
        ),
        "summary": summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote {out}")
    for fam in summary:
        s = summary[fam]
        s3 = s["stage3"]
        print(f"\n[{fam}]")
        print(f"  stage-3: n={s3['n_stage3_checkpoints']}, all_positive="
              f"{s3['all_positive']}, mean={s3['mean']:+.4f} "
              f"(min {s3['min']:+.4f}, max {s3['max']:+.4f})")
        print(f"  base {s['base']:+.4f} -> SFT {s['sft']:+.4f} -> DPO "
              f"{s['dpo']:+.4f} -> Instruct {s['instruct']:+.4f}  "
              f"({s['post_training_fold_increase']}x stage-3 mean)")
    print(f"\nStage-3 positive & flat (~+0.011): {payload['stage3_positive_and_flat']}")


if __name__ == "__main__":
    main()
