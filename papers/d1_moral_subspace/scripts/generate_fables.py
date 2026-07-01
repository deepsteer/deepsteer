#!/usr/bin/env python3
"""Direction 1: Understanding Fables source (rich-subspace restoration, source 1).

A second moral construct (abstract moral inference) to test whether it adds a distinguishable
moral AXIS beyond Moral Stories' explicit action-contrast (the rank>1 question). Reuses the
calibrated MORABLES fable-internal-retelling method UNCHANGED -- same prompt, same
fable_salience gate set incl. the abstraction-match control -- swapping the input text to
Emelin et al.'s `demelin/understanding_fables` (MIT; paraphrased novel-character fables, so
no NC issue and reduced memorization). Per fable: a moral-laden vs valence-stripped retelling
of the same event.

Scenario-level partition (per fable, by story hash; 189 -> 80/20 seed 42). Audited with
audit_runner fable_salience. Network + CPU only; no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))

import audit_runner as ar  # noqa: E402
import generate_morables as gm  # noqa: E402  (reuse _PROMPT + generate_pair UNCHANGED)
from _parallel import parallel_map  # noqa: E402

SEED = 42
EVAL_FRAC = 0.20
_OUT = HERE.parent / "outputs" / "full"
_PART = HERE.parent / "fable_partition.json"


def load_fables() -> list[dict]:
    p = hf_hub_download("demelin/understanding_fables", "test.jsonl", repo_type="dataset")
    out = []
    for line in open(p):
        r = json.loads(line)
        moral = r[f"answer{r['label']}"]
        fid = "fable_" + hashlib.sha1(r["story"].encode()).hexdigest()[:16]
        out.append({"id": fid, "story": r["story"], "moral": moral})
    return out


def partition(fables: list[dict]) -> dict:
    keys = sorted(f["id"] for f in fables)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    n_eval = round(len(keys) * EVAL_FRAC)
    ev = set(keys[:n_eval])
    return {"train": [k for k in sorted(keys) if k not in ev],
            "eval": [k for k in sorted(keys) if k in ev]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Understanding Fables salience pairs.")
    ap.add_argument("--n", type=int, default=40, help="number of fables (first N of split)")
    ap.add_argument("--split", default="train", choices=["train", "eval"])
    ap.add_argument("--out", default=str(_OUT / "fables_batch1.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import anthropic

    fables = {f["id"]: f for f in load_fables()}
    part = partition(list(fables.values()))
    if not _PART.exists():
        _PART.write_text(json.dumps({"seed": SEED, "eval_frac": EVAL_FRAC,
                                     "n_total": len(fables), "source": "understanding_fables",
                                     "license": "MIT", "ids": part}, indent=2))
    ids = part[args.split][: args.n]
    print(f"generating {len(ids)} fable salience pairs | split={args.split}", flush=True)

    client = anthropic.Anthropic(max_retries=5)

    def fn(fid):
        f = fables[fid]
        # reuse the MORABLES retelling generator unchanged (title generic; story self-contained)
        g = gm.generate_pair(client, "a fable", f["story"])
        pair = {"id": fid, "source": "fables", "register": "narrative", "split": args.split,
                "moral": g["moral"], "neutral": g["neutral"]}
        aud = ar.audit_pair(client, pair["moral"], pair["neutral"],
                            ar._GATESET["fable_salience"])
        pair["clean"] = aud["clean"]
        pair["audit"] = {k: aud[k]["passed"] for k in ar._GATESET["fable_salience"]}
        return pair

    out, errs = parallel_map(fn, ids, workers=args.workers,
                             on_progress=lambda d, t, e: print(f"  {d}/{t} ({e} err)", flush=True))
    pairs = [p for p in out if p is not None]
    clean = [p for p in pairs if p["clean"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"source": "fables", "split": args.split, "n_pairs": len(pairs),
                   "n_clean": len(clean), "pairs": pairs}, fh, indent=2)

    n = len(pairs)
    fails = {g: sum(not p["audit"][g] for p in pairs) for g in ar._GATESET["fable_salience"]}
    print("\n=== FABLES first-batch calibration ===")
    print(f"  generated {n} | CLEAN {len(clean)} ({len(clean)/max(n,1):.2f})")
    for g, c in fails.items():
        print(f"  {g:<32} fail {c/max(n,1):.3f}")
    print(f"  registers: {dict(Counter(p['register'] for p in pairs))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
