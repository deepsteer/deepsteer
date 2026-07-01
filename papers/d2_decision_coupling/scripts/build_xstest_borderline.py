#!/usr/bin/env python3
"""Build the XSTest borderline subset for D2 B2/B5 (pre-registered in
d2_decision_coupling/PREREGISTRATION.md B2).

Pulls the real XSTest prompts from the PINNED source repo commit (paul-rottger/xstest, NOT the
HF xstest-v2-copy mirror whose bundled model completions carry other licenses), selects a
deterministic 40-item borderline subset (20 safe over-refusal-prone + 20 mild unsafe-contrast),
and writes data/xstest_borderline.json with a full provenance block. Prompts only; no
completions. License CC-BY-4.0 (attribution + change-indication required; both satisfied here).

Reproducible: same pinned commit + same selection rule -> byte-identical subset.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path

COMMIT = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"  # main @ 2025-02-24, verified 2026-07-01
CSV_URL = f"https://raw.githubusercontent.com/paul-rottger/xstest/{COMMIT}/xstest_prompts.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "xstest_borderline.json"

# Deterministic selection: first N by id from each borderline-relevant type.
SAFE_TYPES = ["homonyms", "figurative_language", "safe_contexts", "safe_targets"]
MILD_CONTRAST_TYPES = ["contrast_homonyms", "contrast_figurative_language",
                       "contrast_safe_contexts", "contrast_safe_targets"]
PER_TYPE = 5  # 4 safe types x5 + 4 contrast types x5 = 40


def fetch_rows() -> list[dict]:
    with urllib.request.urlopen(CSV_URL) as resp:  # noqa: S310 (pinned https)
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def select(rows: list[dict]) -> list[dict]:
    picked = []
    for t in SAFE_TYPES + MILD_CONTRAST_TYPES:
        subset = sorted((r for r in rows if r["type"] == t), key=lambda r: int(r["id"]))[:PER_TYPE]
        assert len(subset) == PER_TYPE, f"type {t}: only {len(subset)} rows"
        for r in subset:
            picked.append({k: r[k] for k in ("id", "prompt", "type", "label", "focus", "note")})
    return picked


def main() -> None:
    rows = fetch_rows()
    items = select(rows)
    payload = {
        "provenance": {
            "dataset": "XSTest",
            "source_repo": "https://github.com/paul-rottger/xstest",
            "source_file": "xstest_prompts.csv",
            "pinned_commit": COMMIT,
            "retrieved": "2026-07-01",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy. XSTest: A Test Suite "
                        "for Identifying Exaggerated Safety Behaviours in Large Language Models. "
                        "NAACL 2024. arXiv:2308.01263.",
            "modifications": "Selected a deterministic 40-item borderline subset (first 5 items "
                             "by id from each of 4 over-refusal-prone SAFE types and 4 mild "
                             "unsafe-CONTRAST types). Prompts only; no model completions "
                             "included. Original columns (id, prompt, type, label, focus, note) "
                             "preserved verbatim. No prompt text altered.",
            "note_on_mirror": "Pulled from the source repo, NOT the HF xstest-v2-copy mirror "
                              "(whose bundled model completions carry Meta/Mistral/OpenAI terms).",
        },
        "selection_rule": {
            "safe_over_refusal_types": SAFE_TYPES,
            "mild_unsafe_contrast_types": MILD_CONTRAST_TYPES,
            "per_type": PER_TYPE,
            "counts": {"safe": len(SAFE_TYPES) * PER_TYPE,
                       "unsafe_contrast": len(MILD_CONTRAST_TYPES) * PER_TYPE},
        },
        "n_items": len(items),
        "items": items,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {OUT} ({len(items)} items: "
          f"{payload['selection_rule']['counts']})")
    for it in items:
        print(f"  [{it['id']:>3}] ({it['label']:6} {it['type']:26}) {it['prompt']}")


if __name__ == "__main__":
    main()
