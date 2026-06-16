#!/usr/bin/env python3
"""Build the harmful/harmless prompt set for Heretic-style refusal ablation.

Matches Arditi et al. (2024) methodology (sprint choice 3.1a): harmful
instructions from AdvBench, harmless instructions from Alpaca. Samples a fixed,
seeded subset and writes ``papers/5_moral_alignment/refusal_prompts.json`` with
provenance, replacing the placeholder set baked into heretic_ablation.py.

Usage:
    python papers/5_moral_alignment/scripts/build_refusal_prompts.py --n 256
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT = Path("papers/5_moral_alignment/refusal_prompts.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AdvBench/Alpaca refusal prompts.")
    ap.add_argument("--n", type=int, default=256, help="Prompts per class.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import csv
    import io
    import urllib.request

    from datasets import load_dataset

    # AdvBench harmful_behaviors from the original llm-attacks repo (ungated;
    # the HF mirror walledai/AdvBench is gated). Columns: goal, target.
    adv_url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
               "main/data/advbench/harmful_behaviors.csv")
    with urllib.request.urlopen(adv_url) as resp:
        reader = csv.DictReader(io.StringIO(resp.read().decode("utf-8")))
        harmful_all = [row["goal"].strip() for row in reader if row.get("goal", "").strip()]

    alp = load_dataset("tatsu-lab/alpaca", split="train")
    # Harmless = self-contained instructions (no extra input field), de-duplicated.
    harmless_all = list(dict.fromkeys(
        r["instruction"].strip()
        for r in alp
        if not r.get("input") and r["instruction"] and r["instruction"].strip()
    ))

    rng = random.Random(args.seed)
    n_h = min(args.n, len(harmful_all))
    n_s = min(args.n, len(harmless_all))
    harmful = sorted(rng.sample(harmful_all, n_h))
    harmless = sorted(rng.sample(harmless_all, n_s))

    payload = {
        "harmful": harmful,
        "harmless": harmless,
        "provenance": {
            "harmful_source": "llm-attacks AdvBench harmful_behaviors.csv (Zou et al. 2023, GCG)",
            "harmless_source": "tatsu-lab/alpaca (self-contained instructions, no input)",
            "method": "Arditi et al. 2024 refusal-direction protocol (sprint 3.1a)",
            "n_per_class": {"harmful": n_h, "harmless": n_s},
            "seed": args.seed,
            "available": {"harmful": len(harmful_all), "harmless": len(harmless_all)},
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}: {n_h} harmful, {n_s} harmless")
    print(f"  harmful[0]:  {harmful[0][:80]!r}")
    print(f"  harmless[0]: {harmless[0][:80]!r}")


if __name__ == "__main__":
    main()
