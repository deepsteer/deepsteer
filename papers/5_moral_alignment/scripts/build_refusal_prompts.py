#!/usr/bin/env python3
"""Build Heretic's EXACT harmful/harmless prompt set for refusal ablation.

Matches p-e-w/heretic's config.default.toml verbatim (sprint choice 3.1a):
  bad  prompts = "mlabonne/harmful_behaviors", column "text", train[:400] (+ test[:100] eval)
  good prompts = "mlabonne/harmless_alpaca",  column "text", train[:400] (+ test[:100] eval)
Prompts are taken in dataset order (Heretic uses "first N"), NOT randomly
sampled, so the set reproduces Heretic exactly.

Writes ``papers/5_moral_alignment/refusal_prompts.json`` with keys:
  harmful, harmless        -> train[:n_dir]  (used to compute the refusal direction)
  harmful_eval, harmless_eval -> test[:n_eval] (Heretic's refusal-count eval set)

Usage:
    python papers/5_moral_alignment/scripts/build_refusal_prompts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("papers/5_moral_alignment/refusal_prompts.json")

# Heretic config.default.toml defaults.
HARMFUL_DS = "mlabonne/harmful_behaviors"
HARMLESS_DS = "mlabonne/harmless_alpaca"
COLUMN = "text"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Heretic's exact refusal prompt set.")
    ap.add_argument("--n-dir", type=int, default=400, help="Train prompts per class (Heretic: 400).")
    ap.add_argument("--n-eval", type=int, default=100, help="Test prompts per class (Heretic: 100).")
    args = ap.parse_args()

    from datasets import load_dataset

    def take(name: str, split: str, n: int) -> list[str]:
        ds = load_dataset(name, split=split)
        return [ds[i][COLUMN] for i in range(min(n, len(ds)))]

    harmful = take(HARMFUL_DS, "train", args.n_dir)
    harmless = take(HARMLESS_DS, "train", args.n_dir)
    harmful_eval = take(HARMFUL_DS, "test", args.n_eval)
    harmless_eval = take(HARMLESS_DS, "test", args.n_eval)

    payload = {
        "harmful": harmful,
        "harmless": harmless,
        "harmful_eval": harmful_eval,
        "harmless_eval": harmless_eval,
        "provenance": {
            "source": "p-e-w/heretic config.default.toml (exact set, sprint 3.1a)",
            "harmful_dataset": HARMFUL_DS,
            "harmless_dataset": HARMLESS_DS,
            "column": COLUMN,
            "selection": "first-N in dataset order (Heretic uses 'first N', not random)",
            "n_dir_per_class": {"harmful": len(harmful), "harmless": len(harmless)},
            "n_eval_per_class": {"harmful": len(harmful_eval), "harmless": len(harmless_eval)},
            "note": "Heretic computes the direction as a FIRST-token difference-of-means.",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}: harmful={len(harmful)} harmless={len(harmless)} "
          f"(+eval {len(harmful_eval)}/{len(harmless_eval)})")
    print(f"  harmful[0]:  {harmful[0][:80]!r}")
    print(f"  harmless[0]: {harmless[0][:80]!r}")


if __name__ == "__main__":
    main()
