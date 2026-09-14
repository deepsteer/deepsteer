#!/usr/bin/env python3
"""Stage-2 harness calibration set: 200 REAL pilot replies (KDG_PANEL_SPEC §13 A6).

    python3 papers/kdg_panel/scripts/build_calibration_stage2.py --out-dir <pilot out dir>
    python3 papers/kdg_panel/scripts/rate_with_judge.py calibration --judge claude \
        --calibration papers/kdg_panel/data/calibration_set_v2_real.json
    python3 papers/kdg_panel/scripts/rate_with_judge.py calibration --judge openai:gpt-5.5 \
        --calibration papers/kdg_panel/data/calibration_set_v2_real.json   # second judge

Draws 200 rollouts from the saved per-rollout JSONL of the generated cells (D_chat dose-0, its
pressure-removed and known-gap variants, J_stated and its variants), stratified by cell and
family and enriched for the replies the parser did NOT resolve cleanly (parse methods other
than ``bare_letter`` / ``answer_line`` are oversampled), because those are where the harness
could be wrong. ``gold`` is left empty: rater 1 is a human pass (or, until one exists, the
first judge), rater 2 an independent judge; the harness is compared to both and
disagreements between raters are excluded (§4.4). The pilot gate is not applied until this
set meets ≥ 0.95.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deepsteer.kdg.calibration import CALIBRATION_SET_VERSION  # noqa: E402
from deepsteer.kdg.harness import KDG_HARNESS_VERSION  # noqa: E402
from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

CELLS = (
    "d_chat_dose0",
    "d_chat_dose0_pressure_removed",
    "d_chat_dose0_known_gap",
    "j_stated",
    "j_stated_paraphrase",
    "j_stated_pressure_removed",
)
CLEAN = {"bare_letter", "answer_line"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "papers/kdg_panel/outputs/pilot")
    ap.add_argument("--model", default="olmo3_instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument(
        "--hard-fraction",
        type=float,
        default=0.5,
        help="share of items drawn from non-clean parse methods (if available)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=REPO / "papers/kdg_panel/data/calibration_set_v2_real.json"
    )
    a = ap.parse_args()
    scen, _ = load_scenario_dir(sorted((REPO / "papers/kdg_panel/data").glob("*_scenarios_*.json")))
    by_id = {s.id: s for s in scen}
    rng = random.Random(a.seed)
    rows = []
    for cell in CELLS:
        p = a.out_dir / a.model / f"{cell}.jsonl"
        if not p.exists():
            print(f"skip {cell}: {p} missing")
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("text") is not None:
                rows.append(r)
    if not rows:
        raise SystemExit("no rollouts found; run the pilot first")
    hard = [r for r in rows if r["parse_method"] not in CLEAN]
    clean = [r for r in rows if r["parse_method"] in CLEAN]
    n_hard = min(len(hard), int(a.n * a.hard_fraction))
    picked = rng.sample(hard, n_hard) + rng.sample(clean, min(len(clean), a.n - n_hard))
    rng.shuffle(picked)
    items = []
    for i, r in enumerate(picked):
        s = by_id[r["scenario_id"]]
        order = [[L, s.option(oid).__dict__] for L, oid in sorted(r["order"].items())]
        items.append(
            {
                "id": f"cal2-{i:03d}",
                "scenario": s.to_dict(),
                "order": order,
                "text": r["text"],
                "cell": r["cell"],
                "rollout": r["rollout"],
                "harness_option": r["option_id"],
                "harness_method": r["parse_method"],
                "gold": None,
            }
        )
    a.out.write_text(
        json.dumps(
            {
                "calibration_set_version": CALIBRATION_SET_VERSION,
                "stage": 2,
                "harness_version_at_build": KDG_HARNESS_VERSION,
                "model": a.model,
                "n_pool": len(rows),
                "n_hard_pool": len(hard),
                "n_hard": n_hard,
                "method_histogram": {
                    m: sum(r["parse_method"] == m for r in rows)
                    for m in sorted({r["parse_method"] for r in rows})
                },
                "items": items,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    print(
        f"wrote {a.out}: {len(items)} items ({n_hard} hard); "
        f"pool {len(rows)}, hard pool {len(hard)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
