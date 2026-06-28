#!/usr/bin/env python3
"""Direction 1: assemble the committed SINGLE-SOURCE V_moral dataset (Apache-clean).

MORABLES was dropped (CC-BY-NC + 79% non-re-derivable; see PREREGISTRATION amendment), so
V_moral is single-source: Moral Stories (MIT) only, two registers. ETHICS (MIT) stays as the
generalization probe. All generated halves (neutral retellings, declarative re-renderings,
paraphrases) are Apache-2.0. No NC content -> fully committable under the repo's Apache-2.0.

Filters the working artifacts to Moral Stories (+ ETHICS probe), attaches the held-out
paraphrases, and writes the committed full-text dataset under deepsteer/datasets/. Provenance
in deepsteer/datasets/DATASET_LICENSES.md. CPU only; no API, no GPU.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_FULL = HERE.parent / "outputs" / "full"
_OUT = HERE.parents[2] / "deepsteer" / "datasets" / "direction1_vmoral_v1.json"


def _keep(p, fields):
    return {k: p[k] for k in fields if k in p}


def main() -> None:
    ds = json.load(open(_FULL / "dataset_2reg.json"))
    para = {r["id"]: r for r in
            json.load(open(_FULL / "eval_g2_paraphrased.json"))["pairs"]}

    tf = ("id", "source", "register", "split", "source_pair_id", "moral", "neutral")
    train = [_keep(p, tf) for p in ds["train"] if p["source"] == "moral_stories"]

    eval_g2 = []
    for p in ds["eval_g2_indist"]:
        if p["source"] != "moral_stories":
            continue
        row = _keep(p, tf)
        pp = para.get(p["id"])
        if pp:
            row.update({"moral_para": pp["moral_para"], "neutral_para": pp["neutral_para"],
                        "paraphrase_status": pp["status"]})
        eval_g2.append(row)

    # merge the eval expansion (53 -> 96 MS narrative; source-balance no longer binds)
    exp_path = _FULL / "ms_eval_expansion.json"
    if exp_path.exists():
        for r in json.load(open(exp_path))["pairs"]:
            row = _keep(r, tf)
            row.update({"moral_para": r.get("moral_para"),
                        "neutral_para": r.get("neutral_para"),
                        "paraphrase_status": r.get("paraphrase_status")})
            eval_g2.append(row)

    probe = [_keep(p, ("id", "source", "register", "ethics_label", "moral", "neutral"))
             for p in ds["eval_generalization_probe"]]

    out = {
        "name": "direction1_vmoral", "version": "v1-single-source",
        "composition": ("single-source V_moral: Moral Stories (MIT) only, two registers "
                        "(narrative + declarative re-renderings); ETHICS (MIT) = "
                        "generalization probe. MORABLES dropped (CC-BY-NC + non-re-derivable)."),
        "licenses": {"moral_stories_source_text": "MIT", "ethics_source_text": "MIT",
                     "generated_halves_neutrals_rerenders_paraphrases": "Apache-2.0",
                     "see": "deepsteer/datasets/DATASET_LICENSES.md"},
        "gates": ("G2 STOP gates the narrative slice; declarative reported informative. "
                  "Track-4 cross-register is directional. See PREREGISTRATION.md."),
        "notes": ("eval_g2_indist is the 53 fully-processed MS narrative pairs (+26 decl); "
                  "balance no longer binds, so it could expand 53->96 clean MS eval later."),
        "counts": {
            "train": {"total": len(train), **dict(Counter(p["register"] for p in train))},
            "eval_g2_indist": {"total": len(eval_g2),
                               **dict(Counter(p["register"] for p in eval_g2)),
                               "clean_paraphrases": sum(
                                   r.get("paraphrase_status") == "clean" for r in eval_g2)},
            "eval_generalization_probe_ethics": len(probe),
        },
        "train": train, "eval_g2_indist": eval_g2, "eval_generalization_probe": probe,
    }
    _OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {_OUT.relative_to(HERE.parents[2])}")
    print(json.dumps(out["counts"], indent=2))


if __name__ == "__main__":
    main()
