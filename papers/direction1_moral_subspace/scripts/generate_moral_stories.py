#!/usr/bin/env python3
"""Direction 1, Phase 1, step 5 (Moral Stories): situation-held salience pairs.

For each TRAIN-source instance (``partition_ids.json``), build ONE salience pair holding
the situation constant and contrasting the action's moral valence (CONSTRUCTION_GUIDELINES.md):

  * MORAL side   = situation + the dataset's ``moral_action`` (valence present, verbatim).
  * NEUTRAL side = the SAME situation + a generated valence-stripped action (same actor,
    same setting, mundane).

The situation is shared verbatim, so the pair is a tight minimal pair whose only
difference is the action's moral valence (a salience contrast, NOT moral-vs-immoral
polarity). Only the neutral action is generated; the moral action is the real datum.

The §1.2 structural-mirror lesson from the MORABLES calibration is built into the prompt
up front (clause-for-clause mirror, 1.2:1 length).

Output is a pilot/working artifact (uncommitted); the script is reusable infra. Audit with
``audit_runner.py --pair-type moral_neutral``.

Network + CPU only; no GPU. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
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
Here is a situation and the morally-significant action a character takes in it.

SITUATION: {situation}
ACTION: {action}

Write a VALENCE-STRIPPED replacement action: the SAME character in the SAME situation
performs a mundane, non-moral action instead. Requirements:
- Same actor, same setting. Change ONLY the moral content of the action.
- Mirror the ACTION's grammatical structure and length clause-for-clause (within a 1.2:1
  word ratio); swap only the valence-bearing words, keeping the syntactic frame identical.
- The replacement must carry NO moral weight (no virtue, harm, duty, fairness, loyalty,
  care, sacrifice), even implicitly. Natural English, no em-dashes.

Reply with ONLY a JSON object: {{"neutral_action": "<action sentence>"}}."""


def load_train_ids(split: str) -> list[str]:
    return json.load(open(_IDS))["ids"]["moral_stories"][split]


def load_instances() -> dict[str, dict]:
    p = hf_hub_download("demelin/moral_stories", "data/moral_stories_full.jsonl",
                        repo_type="dataset")
    return {r["ID"]: r for r in (json.loads(line) for line in open(p))}


def generate_neutral_action(client, situation: str, action: str) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role": "user",
                   "content": _PROMPT.format(situation=situation, action=action)}],
    )
    text = resp.content[0].text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]!r}")
    return json.loads(m.group())["neutral_action"].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Moral Stories situation-held salience pairs.")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", default=str(_OUT / "outputs" / "moral_stories_batch1.json"))
    args = ap.parse_args()

    import anthropic

    ids = load_train_ids(args.split)[: args.n]
    inst = load_instances()
    client = anthropic.Anthropic()
    print(f"generating {len(ids)} Moral Stories salience pairs | split={args.split}")

    pairs, failures = [], []
    for i, msid in enumerate(ids):
        r = inst[msid]
        situation, moral_action = r["situation"], r["moral_action"]
        try:
            neutral_action = generate_neutral_action(client, situation, moral_action)
        except Exception as e:  # noqa: BLE001 -- log and continue
            failures.append({"id": msid, "error": str(e)[:160]})
            continue
        pairs.append({
            "id": f"moral_stories_{msid}", "source": "moral_stories", "ms_id": msid,
            "register": "narrative", "split": args.split,
            "moral": f"{situation} {moral_action}",
            "neutral": f"{situation} {neutral_action}",
        })
        if (i + 1) % 10 == 0 or i + 1 == len(ids):
            print(f"  {i+1}/{len(ids)} generated ({len(failures)} failed)")
        time.sleep(RATE_LIMIT_S)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"source": "moral_stories", "split": args.split, "model": MODEL,
                   "n_pairs": len(pairs), "n_failed": len(failures),
                   "failures": failures, "pairs": pairs}, fh, indent=2)
    print(f"\nwrote {len(pairs)} pairs ({len(failures)} failed) -> {out}")
    if pairs:
        print("\n--- sample pair ---")
        print(f"  MORAL:   {pairs[0]['moral']}")
        print(f"  NEUTRAL: {pairs[0]['neutral']}")


if __name__ == "__main__":
    main()
