#!/usr/bin/env python3
"""Direction 1: paraphrase an arbitrary eval-pairs file for G2 coverage (held-out set).

Reuses the committed paraphrase pipeline (generate_paraphrases.make_fn: C1 mechanical
divergence floor + C2 LLM judgment/meaning judge, up to 3 attempts) on the clean eval pairs
of an added source (fable eval, ETHICS eval), so multi-source G2 can compute acc_surf vs
acc_para per source. Network + CPU only; no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))

import generate_paraphrases as gpp  # noqa: E402
from _parallel import parallel_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Paraphrase an eval-pairs file for G2.")
    ap.add_argument("--pairs", required=True, help="JSON with a 'pairs' list (clean filtered)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import anthropic

    data = json.load(open(args.pairs))
    pairs = data["pairs"] if isinstance(data, dict) and "pairs" in data else data
    if args.clean_only:
        pairs = [p for p in pairs if p.get("clean", True)]
    if args.limit:
        pairs = pairs[:args.limit]
    # normalize for the paraphrase fn (needs id/source/register/moral/neutral)
    for p in pairs:
        p.setdefault("register", "narrative")
        p.setdefault("source", data.get("source", "unknown"))
    print(f"paraphrasing {len(pairs)} clean eval pairs", flush=True)

    client = anthropic.Anthropic(max_retries=5)
    out, errs = parallel_map(gpp.make_fn(client), pairs, workers=args.workers,
                             on_progress=lambda d, t, e: print(f"  {d}/{t} ({e} err)", flush=True))
    rows = [r for r in out if r is not None]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n": len(rows), "pairs": rows}, fh, indent=2)
    clean = sum(r["status"] == "clean" for r in rows)
    print(f"\nparaphrased {len(rows)} | clean {clean} | "
          f"by_register {dict(Counter(r['slice'] for r in rows))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
