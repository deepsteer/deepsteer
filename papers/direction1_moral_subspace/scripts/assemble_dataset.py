#!/usr/bin/env python3
"""Direction 1, Phase 1 step 5: assemble the structured dataset from clean pools.

Reads the per-(source,split) clean-tagged outputs from run_full_generation.py and produces
the final structure, keeping the two eval tests SEPARATE (they have opposite balance needs):

  * train               = Moral Stories clean + MORABLES clean        (ETHICS zero)
  * eval_g2_indist       = source-BALANCED in-distribution eval (MORABLES + Moral Stories at
                          the MORABLES cap). This is what GATE G2 reads -- balancing keeps the
                          paraphrase gap from being a source-shift artifact. Only this set is
                          paraphrased.
  * eval_generalization_probe = ALL clean ETHICS eval (no balancing). ETHICS is zero-in-
                          training, so this tests whether a two-source V_moral extends to an
                          unseen abstract-judgment register; it is NOT pooled into G2 (that
                          would make the aggregate measure generalization, not contamination).

Reports per-source counts and per-register composition. CPU only; no API, no GPU.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

_FULL = Path(__file__).resolve().parent.parent / "outputs" / "full"


def load_clean(source: str, split: str) -> list[dict]:
    data = json.load(open(_FULL / f"{source}_{split}.json"))
    return [p for p in data["pairs"] if p["clean"]]


def reg(pairs: list[dict]) -> dict:
    return dict(Counter(p["register"] for p in pairs))


def main() -> None:
    ms_tr = load_clean("moral_stories", "train")
    mb_tr = load_clean("morables", "train")
    mb_ev = load_clean("morables", "eval")
    ms_ev = load_clean("moral_stories", "eval")
    et_ev = load_clean("ethics", "eval")

    train = ms_tr + mb_tr
    cap = min(len(mb_ev), len(ms_ev))                 # balanced in-dist eval = MORABLES cap
    eval_g2 = mb_ev[:cap] + ms_ev[:cap]
    eval_probe = et_ev                                 # all-clean ETHICS, separate

    dataset = {
        "composition": "two-source training (moral_stories + morables); ethics eval-only",
        "eval_structure": ("eval_g2_indist is source-balanced (morables+moral_stories) and "
                           "is the ONLY set G2/paraphrasing reads; eval_generalization_probe "
                           "is all-clean ETHICS, separate, never pooled into G2"),
        "counts": {
            "train": {"moral_stories": len(ms_tr), "morables": len(mb_tr), "ethics": 0,
                      "total": len(train)},
            "eval_g2_indist": {"balanced_per_source": cap, "morables": cap,
                               "moral_stories": cap, "total": len(eval_g2)},
            "eval_generalization_probe": {"ethics": len(eval_probe)},
            "grand_total": len(train) + len(eval_g2) + len(eval_probe),
        },
        "register_composition": {
            "train": reg(train),
            "eval_g2_indist": reg(eval_g2),
            "eval_generalization_probe": reg(eval_probe),
        },
        "train": train,
        "eval_g2_indist": eval_g2,
        "eval_generalization_probe": eval_probe,
    }
    out = _FULL / "dataset_structured.json"
    with open(out, "w") as fh:
        json.dump(dataset, fh, indent=2)

    c = dataset["counts"]
    print("=== STRUCTURED DATASET ===")
    print(f"TRAIN (two-source): moral_stories={c['train']['moral_stories']} "
          f"morables={c['train']['morables']} ethics={c['train']['ethics']} "
          f"| total {c['train']['total']}")
    print(f"EVAL_G2_INDIST (balanced @ {cap}/source, paraphrased + G2): "
          f"morables={cap} moral_stories={cap} | total {c['eval_g2_indist']['total']}")
    print(f"EVAL_GENERALIZATION_PROBE (all-clean ETHICS, separate): "
          f"ethics={c['eval_generalization_probe']['ethics']}")
    print(f"GRAND TOTAL {c['grand_total']}")
    print("\nREGISTER COMPOSITION:")
    for k, v in dataset["register_composition"].items():
        print(f"  {k:<28} {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
