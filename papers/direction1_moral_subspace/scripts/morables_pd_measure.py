#!/usr/bin/env python3
"""Direction 1: measure the public-domain re-derivability of our MORABLES fables.

Decision metric for the NC re-derivation (option d): of the MORABLES fables currently in the
train set, how many can Claude retell from title+moral alone (MORABLES as a pure index --
facts, no NC text), vs returning UNKNOWN (obscure fables in neither Townsend PD text nor
Claude's knowledge). Reports the UNKNOWN rate + kept-fable count. No commitment; saves the
retellings so a follow-on full re-derivation can reuse the known set.

UNKNOWN low (<~15%) -> option 1 (hybrid, exclude tail). High -> option 2 (L'Estrange).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import datasets as hfds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _parallel import parallel_map  # noqa: E402

hfds.disable_progress_bars()
MODEL = "claude-sonnet-4-6"
_FULL = HERE.parent / "outputs" / "full"

_PROMPT = """\
The classical (public-domain) fable titled "{title}" teaches the moral: "{moral}".

Write a concrete, third-person, past-tense retelling (15-40 words) of that fable's central
morally-significant event, with its moral valence present. Base it ONLY on the public-domain
classical fable; do not quote any modern translation. If you do not know this specific fable,
reply with exactly UNKNOWN (do not guess or invent one).

Reply with only the retelling sentence, or exactly UNKNOWN."""


def main() -> None:
    import anthropic

    ds = json.load(open(_FULL / "dataset_2reg.json"))
    aliases = sorted({p["alias"] for p in ds["train"]
                      if p["source"] == "morables" and p.get("register") == "narrative"})
    fab = {r["alias"]: r for r in
           hfds.load_dataset("cardiffnlp/Morables", "fables_only")["morables"]}
    items = [{"alias": a, "title": fab[a]["title"], "moral": fab[a]["moral"]}
             for a in aliases if a in fab]
    print(f"measuring PD re-derivability over {len(items)} train MORABLES fables", flush=True)

    client = anthropic.Anthropic(max_retries=5)

    def fn(it):
        out = client.messages.create(
            model=MODEL, max_tokens=150,
            messages=[{"role": "user", "content": _PROMPT.format(
                title=it["title"], moral=it["moral"])}]).content[0].text.strip()
        known = not re.fullmatch(r"\s*UNKNOWN\s*\.?", out, re.IGNORECASE)
        return {"alias": it["alias"], "title": it["title"], "known": bool(known),
                "retelling": out if known else None}

    res, errs = parallel_map(fn, items, workers=8,
                             on_progress=lambda d, t, e: print(f"  {d}/{t} ({e} err)", flush=True))
    res = [r for r in res if r is not None]
    known = [r for r in res if r["known"]]
    n, k = len(res), len(known)
    rate = round((n - k) / n, 3) if n else 0.0
    with open(_FULL / "morables_pd_measure.json", "w") as fh:
        json.dump({"n_fables": n, "kept_known": k, "unknown": n - k,
                   "unknown_rate": rate, "results": res}, fh, indent=2)
    print(f"\n=== PD RE-DERIVABILITY ===")
    print(f"  fables measured: {n}")
    print(f"  kept (Claude can retell): {k}  | UNKNOWN: {n - k}  | UNKNOWN rate: {rate}")
    print(f"  decision: option 1 if UNKNOWN <~0.15 and kept >~150; else option 2")


if __name__ == "__main__":
    main()
