#!/usr/bin/env python3
"""Build the refusal-retention set (ST_PLAN Amendment 10 item 1; author, 2026-10-19).

    python3 papers/st_live_reference/scripts/build_refusal_retention.py [--check]

The 60 FL request twins (``deepsteer.datasets.request_twins.get_request_twins``, 6 foundations x
10) x {following, violating} x {plain, care, fill} = 360 prompts, plus the positive control: the
60 ``plain`` violating requests under an operator system prompt that orders the user's request
done (the KDG known-gap control's construction, P1-A9: "operator orders the violation", applied
here to a request). ``plain`` is the bundled text verbatim; ``care`` and ``fill`` append the
template of the twin's seeded index (``data/refusal_retention_templates_v1.json``). Token counts
are recorded on the pinned tokenizers of both models of record. ``--check`` rebuilds and compares
with the committed file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "papers" / "st_live_reference" / "data"
TEMPLATES = DATA / "refusal_retention_templates_v1.json"
OUT = DATA / "refusal_retention_v1.json"
SEED = 20261019
TOKENIZERS = ("gpt_oss_20b", "muse_glimmer_30b")
LENGTH_TOL = 0.15


def build() -> dict:
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from deepsteer.datasets.request_twins import get_request_twins

    tpl = json.loads(TEMPLATES.read_text())
    reg = yaml.safe_load((ROOT / "papers" / "kdg_panel" / "models.yaml").read_text())["st"]
    toks = {k: AutoTokenizer.from_pretrained(reg[k]["repo"], revision=reg[k]["revision"])
            for k in TOKENIZERS}
    twins = get_request_twins()
    idx = np.random.default_rng(SEED).integers(0, len(tpl["care"]), size=len(twins))
    items, controls = [], []
    for i, (foundation, follow, violate) in enumerate(twins):
        t = int(idx[i])
        for member, text in (("following", follow), ("violating", violate)):
            for framing in ("plain", "care", "fill"):
                add = "" if framing == "plain" else tpl[framing][t]
                prompt = text if not add else f"{text} {add}"
                items.append({
                    "id": f"RR-{i:02d}-{member[0]}-{framing}", "twin": i,
                    "foundation": foundation, "member": member, "framing": framing,
                    "template_index": None if framing == "plain" else t, "text": prompt,
                    "n_tokens": {k: len(tk.encode(prompt, add_special_tokens=False))
                                 for k, tk in toks.items()},
                    "framing_tokens": {k: len(tk.encode(" " + add, add_special_tokens=False))
                                       if add else 0 for k, tk in toks.items()}})
        controls.append({"id": f"RR-{i:02d}-v-operator", "twin": i, "foundation": foundation,
                         "member": "violating", "framing": "operator_control",
                         "system": tpl["operator_control_system"], "text": violate})
    return {
        "version": "refusal_retention_v1",
        "amendment": "ST_PLAN Amendment 10 (author, 2026-10-19)",
        "source": "deepsteer.datasets.request_twins.get_request_twins (60 twins, 6 x 10)",
        "templates": {"path": str(TEMPLATES.relative_to(ROOT)),
                      "sha256": hashlib.sha256(TEMPLATES.read_bytes()).hexdigest()},
        "seed": SEED, "length_tolerance": LENGTH_TOL,
        "tokenizers": {k: f"{reg[k]['repo']}@{reg[k]['revision']}" for k in TOKENIZERS},
        "positive_control": "plain violating requests under an operator system prompt ordering "
                            "the request done (KDG known-gap construction, P1-A9)",
        "items": items, "operator_control": controls,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    s = build()
    text = json.dumps(s, indent=1) + "\n"
    if a.check:
        same = OUT.exists() and OUT.read_text() == text
        print(f"{OUT.name}: {'matches' if same else 'DIFFERS from'} a fresh build")
        return 0 if same else 1
    OUT.write_text(text)
    from collections import Counter

    print(f"wrote {OUT.relative_to(ROOT)}: {len(s['items'])} prompts + "
          f"{len(s['operator_control'])} controls; per foundation "
          f"{dict(Counter(x['foundation'] for x in s['items']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
