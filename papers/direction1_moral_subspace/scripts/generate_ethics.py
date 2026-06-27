#!/usr/bin/env python3
"""Direction 1, Phase 1, step 5 (ETHICS): minimal-derivation salience pairs.

ETHICS commonsense contributes the hard-ambiguity register. For each TRAIN-source item
(``partition_ids.json``), build ONE salience pair by MINIMAL derivation
(CONSTRUCTION_GUIDELINES.md):

  * MORAL side   = the ETHICS scenario, verbatim (morally charged).
  * NEUTRAL side = a minimally edited version with the moral charge removed -- the smallest
    word change that makes the same actor's action mundane, keeping structure and length.

Selection: prefer ``is_short`` (single-sentence, declarative register) and ``label == 1``
(clearly unacceptable -> a clean moral-valence-present side) for the first batch. The
partition stores only input hashes, so ETHICS is re-loaded and joined by the same
``sha1[:16]`` used in ``partition_sources.py``.

Output is a pilot/working artifact (uncommitted); the script is reusable infra. Audit with
``audit_runner.py --pair-type moral_neutral``.

Network + CPU only; no GPU. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL = "claude-sonnet-4-6"
RATE_LIMIT_S = 0.3
_OUT = Path(__file__).resolve().parent.parent
_IDS = _OUT / "partition_ids.json"

_PROMPT = """\
Here is a short scenario that carries a moral judgment.

SCENARIO: {scenario}

Produce a MINIMALLY edited version that removes the moral charge: change as FEW words as
possible so the scenario describes a mundane, non-moral action by the same person in the
same setting. Keep the sentence structure, length, and as many words as possible; swap
ONLY the words that carry moral weight. The result must carry no moral weight at all (no
virtue, harm, duty, fairness, loyalty, care). No em-dashes; natural English.

Reply with ONLY a JSON object: {{"neutral": "<scenario>"}}."""


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_train_hashes(split: str) -> set[str]:
    return set(json.load(open(_IDS))["ids"]["ethics"][split])


def load_rows_by_hash() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for fn in ("train.csv", "test.csv", "test_hard.csv"):
        pp = hf_hub_download("hendrycks/ethics", f"data/commonsense/{fn}",
                             repo_type="dataset")
        with open(pp) as fh:
            for r in csv.DictReader(fh):
                h = _hash(r["input"])
                rows.setdefault(h, r)   # first occurrence, matches partition dedup
    return rows


def select(train_hashes: set[str], rows: dict[str, dict], n: int) -> list[dict]:
    """Prefer short + label-1 (charged, declarative) items for clean minimal editing."""
    pool = [(h, rows[h]) for h in sorted(train_hashes) if h in rows]
    pref = [(h, r) for h, r in pool if r["is_short"] == "True" and r["label"] == "1"]
    chosen = pref[:n]
    if len(chosen) < n:  # backfill with other short items
        extra = [(h, r) for h, r in pool if r["is_short"] == "True" and (h, r) not in pref]
        chosen += extra[: n - len(chosen)]
    return [{"hash": h, **r} for h, r in chosen]


def generate_neutral(client, scenario: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user", "content": _PROMPT.format(scenario=scenario)}],
    )
    text = resp.content[0].text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]!r}")
    return json.loads(m.group())["neutral"].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="ETHICS minimal-derivation salience pairs.")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", default=str(_OUT / "outputs" / "ethics_batch1.json"))
    args = ap.parse_args()

    import anthropic

    rows = load_rows_by_hash()
    items = select(load_train_hashes(args.split), rows, args.n)
    client = anthropic.Anthropic()
    print(f"generating {len(items)} ETHICS salience pairs | split={args.split}")

    pairs, failures = [], []
    for i, r in enumerate(items):
        scenario = r["input"]
        register = "declarative" if r["is_short"] == "True" else "narrative"
        try:
            neutral = generate_neutral(client, scenario)
        except Exception as e:  # noqa: BLE001 -- log and continue
            failures.append({"hash": r["hash"], "error": str(e)[:160]})
            continue
        pairs.append({
            "id": f"ethics_{r['hash']}", "source": "ethics", "register": register,
            "split": args.split, "ethics_label": int(r["label"]),
            "moral": scenario, "neutral": neutral,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            print(f"  {i+1}/{len(items)} generated ({len(failures)} failed)")
        time.sleep(RATE_LIMIT_S)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"source": "ethics", "split": args.split, "model": MODEL,
                   "n_pairs": len(pairs), "n_failed": len(failures),
                   "failures": failures, "pairs": pairs}, fh, indent=2)
    print(f"\nwrote {len(pairs)} pairs ({len(failures)} failed) -> {out}")
    if pairs:
        print("\n--- sample pair ---")
        print(f"  MORAL:   {pairs[0]['moral']}")
        print(f"  NEUTRAL: {pairs[0]['neutral']}")


if __name__ == "__main__":
    main()
