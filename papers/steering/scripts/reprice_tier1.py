#!/usr/bin/env python3
"""T0.11's last output: the Tier 1 table re-priced from the T1.0 screen's measured rates (G1 input).

    python3 papers/steering/scripts/reprice_tier1.py --screen-dir OUT [--write OUT/tier1_reprice.md]

Per screened model, the generation and readout parts of each Tier 1 row are priced from that
model's measured screen step time (seconds per decoding step at the screen batch), its
tokens-to-answer distribution at the effort and cap of record, and its measured readout seconds per
row; parts the screen cannot measure (training, AtP, Heretic, capability and specificity scoring)
keep STEERING_PLAN §7 prices ("plan") or split a §7 row priced only as a whole ("estimate"). Each
model is priced in both roles (development model: T1.1-T1.5; second family: T1.6), because G1
assigns the roles after this table. The rollout counts are the plan's (v0.2.5): the brake-engaged
subsets are taken at the plan's expected sizes (65 of H1_held; 50 of the second family's 100)
until T1.1 measures them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_lib as L  # noqa: E402

ENVELOPE = 35.0
BRAKE_HELD = 65  # expected brake-engaged H1_held scenarios (plan §6: 50-80)
BRAKE_SECOND = 50
PANEL_READS = 586 * 8 * 4  # four E cells at dose 0, eight permutations
# components the screen cannot measure (A100-h): "plan" = a STEERING_PLAN §7 / amendment line
# item; "estimate" = this script's split of a §7 row the plan prices only as a whole (T1.5 = 11,
# T1.6 = 7.75), to be replaced by measured numbers when those cells run
FIXED = {"T1.2 AtP": (0.5, "plan"), "T1.4 training (3 adapters)": (3.0, "plan"),
         "T1.5 capability + specificity + Heretic": (2.0, "estimate"),
         "T1.6 training (2 QLoRA adapters)": (1.5, "estimate"),
         "T1.6 direction riders": (0.2, "estimate")}


def gen_h(t: dict, n: int) -> float:
    """Hours to generate n rollouts at the screen batch and cap of record."""
    s = t["screen"]
    lens = [x for x in t["screen_tokens_to_answer"] if x is not None]
    step = s["seconds_per_step"] or 0.0
    b = s["batch"]
    return math.ceil(n / b) * L.expected_batch_max(lens, b, L.CAP_OF_RECORD) * step / 3600


def read_h(t: dict, n: int, long: bool = False) -> float:
    """Hours for n forward readouts; ``long`` = prompt + a trace (scaled by length)."""
    per = t["read_seconds_per_row"]
    if long:
        lens = [x for x in t["screen_tokens_to_answer"] if x is not None]
        per *= (t["probe_mean_len"] + float(np.mean(lens))) / t["probe_mean_len"]
    return n * per / 3600


def price(t: dict, screen_n: int) -> dict:
    load = t["load_seconds"] / 3600
    dev = {
        "T1.1 H1 264 x 8 (minus screen rollouts) + residual banking + judgment frame":
            gen_h(t, 264 * 8 - screen_n * 4) + read_h(t, 264 * 8, long=True)
            + read_h(t, 132) + load,
        "T1.2 AtP": FIXED["T1.2 AtP"][0],
        "T1.3 gate 20 x (none, fill, care, self x2) x 8 + strength sweep 5 x 4 x 8":
            gen_h(t, 20 * 5 * 8 + 5 * 4 * 8) + load,
        "T1.4 training (3 adapters)": FIXED["T1.4 training (3 adapters)"][0],
        "T1.5 H1_held 132 x 8 x 3 + none re-run + panel dose-0 x 3":
            gen_h(t, 132 * 8 * 3 + BRAKE_HELD * 8) + read_h(t, 132 * 8 * 3, long=True)
            + read_h(t, PANEL_READS * 3) + 3 * load,
        "T1.5 capability + specificity + Heretic":
            FIXED["T1.5 capability + specificity + Heretic"][0],
    }
    second = {
        "T1.6 baseline (60 train top-up x 2, 100 held x 6) + banking":
            gen_h(t, 60 * 2 + 100 * 6) + read_h(t, 60 * 6 + 100 * 6, long=True) + load,
        "T1.6 training (2 QLoRA adapters)": FIXED["T1.6 training (2 QLoRA adapters)"][0],
        "T1.6 evaluation (100 x 6 x 2 + none re-run + panel dose-0 x 2)":
            gen_h(t, 100 * 6 * 2 + BRAKE_SECOND * 6) + read_h(t, 100 * 6 * 2, long=True)
            + read_h(t, PANEL_READS * 2) + 2 * load,
        "T1.6 direction riders": FIXED["T1.6 direction riders"][0],
    }
    return {"development": dev, "second_family": second,
            "development_total": sum(dev.values()), "second_family_total": sum(second.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", type=Path, required=True)
    ap.add_argument("--write", type=Path, default=None)
    a = ap.parse_args()
    models = {}
    for d in sorted(p for p in a.screen_dir.iterdir() if (p / "timing.json").exists()):
        t = json.loads((d / "timing.json").read_text())
        r = json.loads((d / "screen_record.json").read_text())
        models[d.name] = {"timing": t, "record": r, "price": price(t, r["screen_n"])}
    if not models:
        raise SystemExit(f"no timing.json under {a.screen_dir}")
    t10 = sum(m["record"]["elapsed_h"] for m in models.values())
    lines = ["# Tier 1 re-priced from the T1.0 screen (G1 input)", "",
             f"T1.0 as run: {t10:.2f} A100-h over {len(models)} models. Measured parts use each "
             "model's screen step time and trace lengths at its effort and cap of record; "
             "rows marked plan keep STEERING_PLAN §7 prices and rows marked estimate split a §7 "
             "row the plan prices only as a whole. The T1.1 pair rider (<= 0.3, conditional) is "
             "not included.", ""]
    for k, m in models.items():
        r, p = m["record"], m["price"]
        scr = r.get("screen", {})
        lines += [f"## {k}", "",
                  f"effort of record `{r.get('effort_of_record')}`; unbatched "
                  f"{r.get('unbatched')}; status `{r['status']}`; truncation "
                  f"{_f(scr.get('truncation_rate'))}; truncation-limited "
                  f"{scr.get('truncation_limited')}; dose-0 of record "
                  f"`{r.get('dose0', {}).get('of_record')}`", "",
                  "| role | row | A100-h | basis |", "|---|---|---|---|"]
        for role in ("development", "second_family"):
            for row, h in p[role].items():
                basis = FIXED[row][1] if row in FIXED else "measured"
                lines.append(f"| {role} | {row} | {h:.2f} | {basis} |")
            lines.append(f"| {role} | **total** | **{p[role + '_total']:.2f}** | |")
        lines.append("")
    lines += ["## Envelope by role assignment", "",
              "| development | second family | Tier 1 total incl. T1.0 as run | within 35 |",
              "|---|---|---|---|"]
    for dev in models:
        for sec in models:
            if sec == dev:
                continue
            tot = (t10 + models[dev]["price"]["development_total"]
                   + models[sec]["price"]["second_family_total"])
            lines.append(f"| {dev} | {sec} | {tot:.2f} | {'yes' if tot <= ENVELOPE else 'NO'} |")
    text = "\n".join(lines) + "\n"
    print(text)
    out = a.write or (a.screen_dir / "tier1_reprice.md")
    out.write_text(text)
    (out.with_suffix(".json")).write_text(json.dumps(
        {k: {"price": v["price"], "elapsed_h": v["record"]["elapsed_h"]}
         for k, v in models.items()}, indent=1))
    return 0


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    sys.exit(main())
