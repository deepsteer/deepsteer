#!/usr/bin/env python3
"""Direction 1, Phase 1, step 5 (MORABLES): fable-internal salience-pair generation.

For each fable in the TRAIN-source pool (``partition_ids.json``), derive ONE salience
pair from the fable's own narrative material (CONSTRUCTION_GUIDELINES.md):

  * MORAL side   -- a concrete retelling of the fable's central morally-relevant event,
    moral valence present.
  * NEUTRAL side -- a concrete retelling of the SAME event: same actors, same scene,
    same domain, valence stripped to a mundane action.

Both sides are concrete event-retellings (discourse type held constant) so the pair
differs in moral valence, NOT in narrative-vs-aphorism abstraction. The fable's abstract
stated moral is never used as a side (that would confound valence with genre).

Output is a pilot/working artifact (uncommitted); the script is reusable infra. Audit it
with ``audit_runner.py --pair-type fable_salience`` for the first-batch anti-triviality
threshold calibration before scaling.

Network + CPU only; no GPU. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import datasets as hfds

hfds.disable_progress_bars()

MODEL = "claude-sonnet-4-6"
RATE_LIMIT_S = 0.3
_OUT = Path(__file__).resolve().parent.parent
_IDS = _OUT / "partition_ids.json"

_PROMPT = """\
Here is a classical fable:

TITLE: {title}
STORY: {story}

Identify the single morally-significant action at the heart of this fable, then write TWO
concrete retellings of that one event:

1. MORAL: a concrete, third-person, past-tense narrative (15-40 words) of the event with
   its moral valence PRESENT -- the action is recognizably virtuous or wrong in context.
2. NEUTRAL: a concrete, third-person, past-tense narrative of the SAME event with the
   moral valence STRIPPED -- the same characters in the same scene and domain perform a
   mundane, non-moral action of similar length and structure.

Hard rules:
- Both must be concrete event-retellings. NEITHER may be an abstract moral, lesson, or
  aphorism.
- Same characters, same setting, same domain in both. Change ONLY the moral valence.
- STRUCTURAL MIRROR (critical): the neutral must mirror the moral side's syntax
  clause-for-clause -- same number of clauses, same connectives, same approximate length
  (within a 1.2:1 word ratio). If the moral side has a consequence clause (e.g. a trailing
  "...and so X happened"), the neutral must keep a PARALLEL consequence clause with a
  mundane, non-moral outcome. Swap only the valence-bearing words; keep the syntactic
  frame identical.
- The NEUTRAL side must carry no moral weight at all (no virtue, harm, duty, sacrifice,
  fairness, loyalty, etc.), even implicitly. No em-dashes; natural English.

Reply with ONLY a JSON object: {{"moral": "<sentence>", "neutral": "<sentence>"}}."""


def load_train_aliases(split: str) -> list[str]:
    ids = json.load(open(_IDS))["ids"]["morables"][split]
    return ids


def load_fables() -> dict[str, dict]:
    d = hfds.load_dataset("cardiffnlp/Morables", "fables_only")["morables"]
    return {r["alias"]: r for r in d}


def generate_pair(client, title: str, story: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user",
                   "content": _PROMPT.format(title=title, story=story)}],
    )
    text = resp.content[0].text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in generation response: {text[:200]!r}")
    obj = json.loads(m.group())
    return {"moral": obj["moral"].strip(), "neutral": obj["neutral"].strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="MORABLES fable-internal salience pairs.")
    ap.add_argument("--n", type=int, default=40, help="number of fables (first N of split)")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", default=str(_OUT / "outputs" / "morables_batch1.json"))
    args = ap.parse_args()

    import anthropic

    aliases = load_train_aliases(args.split)[: args.n]
    fables = load_fables()
    client = anthropic.Anthropic()
    print(f"generating {len(aliases)} MORABLES salience pairs | split={args.split}")

    pairs, failures = [], []
    for i, alias in enumerate(aliases):
        f = fables[alias]
        try:
            gen = generate_pair(client, f["title"], f["story"])
        except Exception as e:  # noqa: BLE001 -- log and continue; report at end
            failures.append({"alias": alias, "error": str(e)[:160]})
            continue
        pairs.append({
            "id": f"morables_{alias}", "source": "morables", "alias": alias,
            "title": f["title"], "register": "narrative", "split": args.split,
            "moral": gen["moral"], "neutral": gen["neutral"],
        })
        if (i + 1) % 10 == 0 or i + 1 == len(aliases):
            print(f"  {i+1}/{len(aliases)} generated ({len(failures)} failed)")
        time.sleep(RATE_LIMIT_S)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"source": "morables", "split": args.split, "model": MODEL,
                   "n_pairs": len(pairs), "n_failed": len(failures),
                   "failures": failures, "pairs": pairs}, fh, indent=2)
    print(f"\nwrote {len(pairs)} pairs ({len(failures)} failed) -> {out}")
    if pairs:
        print("\n--- sample pair ---")
        print(f"  MORAL:   {pairs[0]['moral']}")
        print(f"  NEUTRAL: {pairs[0]['neutral']}")


if __name__ == "__main__":
    main()
