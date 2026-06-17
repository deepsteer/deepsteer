#!/usr/bin/env python3
"""Sprint 6.3: build the ART-SFT training mix (general + moral, chat format).

Both ART-SFT and control-SFT train on this SAME file; only the ART term differs,
so the mix needs (a) general instruction-following capability and (b) enough
assistant-generated moral content for the ART loss to have signal.

Source: AI2's Tülu 3 SFT mixture (what OLMo-3 actually post-trained on), streamed
so no full download is needed. Each row is classified moral/general by its
``source`` tag (Tülu 3 bundles safety/values datasets like CoCoNoT, WildGuardMix,
WildJailbreak) with a content-keyword fallback, then sampled at a target ratio.

Output: chat JSONL (``{"messages": [...]}``) consumable by art_sft.py.

Usage (RunPod, fast network):
    python papers/6_ablation_resistance/scripts/prepare_sft_data.py \
        --output papers/6_ablation_resistance/data/sft_mix.jsonl \
        --n-general 1500 --n-moral 1500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Tülu 3 source tags for its safety / values / refusal subsets (substring match,
# case-insensitive). Override with --moral-source-pattern.
DEFAULT_SOURCE_PATTERN = r"coconot|wildguard|wildjailbreak|safet|values|harm|ethic|moral|refus"

# Content fallback: moral/ethical vocabulary in the assistant turn.
DEFAULT_KEYWORD_PATTERN = (
    r"\b(moral|immoral|ethic|unethical|wrong|right|harm|fair|unfair|cruel|"
    r"betray|loyal|cheat|honest|dishonest|justice|injustice|deserve|ought|"
    r"should not|shouldn't|consent|abuse|exploit)\b"
)


def assistant_text(messages: list[dict]) -> str:
    """Concatenate assistant-turn contents (what the model is trained to generate)."""
    return "\n".join(m.get("content", "") for m in messages if m.get("role") == "assistant")


def is_moral_row(messages: list[dict], source: str, source_re, keyword_re) -> bool:
    """Moral if the source tag matches, else if the assistant text matches keywords."""
    if source and source_re.search(source):
        return True
    return bool(keyword_re.search(assistant_text(messages)))


def build_mix(rows, *, n_general, n_moral, source_re, keyword_re, seed=42):
    """Split classified rows into pools and sample the mix.

    ``rows`` is an iterable of ``(messages, source)``. Returns
    ``(samples, stats)`` where samples is a shuffled list of message-lists.
    """
    rng = random.Random(seed)
    general_pool: list[list[dict]] = []
    moral_pool: list[list[dict]] = []
    moral_sources: dict[str, int] = {}

    for messages, source in rows:
        if not messages or not assistant_text(messages).strip():
            continue
        if is_moral_row(messages, source, source_re, keyword_re):
            moral_pool.append(messages)
            moral_sources[source or "?"] = moral_sources.get(source or "?", 0) + 1
        else:
            general_pool.append(messages)
        if len(general_pool) >= n_general * 3 and len(moral_pool) >= n_moral * 3:
            break  # enough headroom to sample without exhausting

    take_g = min(n_general, len(general_pool))
    take_m = min(n_moral, len(moral_pool))
    samples = rng.sample(general_pool, take_g) + rng.sample(moral_pool, take_m)
    rng.shuffle(samples)
    stats = {
        "n_general_pool": len(general_pool), "n_moral_pool": len(moral_pool),
        "n_general_sampled": take_g, "n_moral_sampled": take_m,
        "moral_ratio": take_m / max(take_g + take_m, 1),
        "top_moral_sources": dict(sorted(moral_sources.items(),
                                         key=lambda kv: -kv[1])[:10]),
    }
    return samples, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the ART-SFT general+moral chat mix.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tulu-dataset", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-general", type=int, default=1500)
    ap.add_argument("--n-moral", type=int, default=1500)
    ap.add_argument("--stream-limit", type=int, default=80000,
                    help="Max rows to scan from the stream.")
    ap.add_argument("--moral-source-pattern", default=DEFAULT_SOURCE_PATTERN)
    ap.add_argument("--keyword-pattern", default=DEFAULT_KEYWORD_PATTERN)
    ap.add_argument("--max-turns", type=int, default=6,
                    help="Drop conversations longer than this many messages.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source_re = re.compile(args.moral_source_pattern, re.IGNORECASE)
    keyword_re = re.compile(args.keyword_pattern, re.IGNORECASE)

    from datasets import load_dataset

    ds = load_dataset(args.tulu_dataset, split=args.split, streaming=True)

    def row_iter():
        for i, row in enumerate(ds):
            if i >= args.stream_limit:
                break
            msgs = row.get("messages") or []
            if len(msgs) > args.max_turns:
                continue
            yield msgs, row.get("source", "")
            if (i + 1) % 10000 == 0:
                logger.info("  scanned %d rows", i + 1)

    samples, stats = build_mix(
        row_iter(), n_general=args.n_general, n_moral=args.n_moral,
        source_re=source_re, keyword_re=keyword_re, seed=args.seed,
    )

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        for msgs in samples:
            fh.write(json.dumps({"messages": msgs}) + "\n")

    meta = {"dataset": args.tulu_dataset, "n_written": len(samples), **stats,
            "source_pattern": args.moral_source_pattern}
    with open(outp.with_suffix(".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Wrote {len(samples)} conversations -> {outp}")
    print(f"  general {stats['n_general_sampled']} / moral {stats['n_moral_sampled']} "
          f"(moral ratio {stats['moral_ratio']:.2f})")
    print(f"  pools: general {stats['n_general_pool']}, moral {stats['n_moral_pool']}")
    print(f"  top moral sources: {stats['top_moral_sources']}")

    # HF streaming (xet) leaves a background download thread that can segfault at
    # interpreter shutdown ("PyGILState_Release: ... no thread-state"). All output
    # is written and flushed above, so exit hard to skip that broken finalizer
    # (otherwise the step exits non-zero despite having fully succeeded).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
