#!/usr/bin/env python3
"""Build a general-text LM corpus for the Stage-1 forced-coupling run.

``forced_coupling_stage1.py`` needs a general continued-pretrain corpus for its
LM-loss anchor and Guard 4 (general perplexity). Without ``--general-jsonl`` it
falls back to the ~480 probing texts, which is in-distribution and makes the LM
anchor / Guard 4 meaningless. This streams a real general corpus to a JSONL of
``{"text": ...}`` documents (no full download).

Default source: ``wikitext-103-raw-v1`` (clean encyclopedic prose, no auth, small,
reliable on the pod). Section-header lines (`` = Title = ``) and very short rows
are dropped, and rows are concatenated into ~``--doc-chars`` documents so each is
a real paragraph-length chunk rather than a single line.

Usage (RunPod, fast network):
    python papers/5_moral_alignment/scripts/prepare_coupling_general.py \
        --output papers/5_moral_alignment/data/general_corpus.jsonl --n 4000
    # then: STAGE1_GENERAL=papers/5_moral_alignment/data/general_corpus.jsonl \
    #       ONLY=coupling_stage1 ./run_session_phase3.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def stream_docs(dataset: str, config: str | None, split: str, *, n: int,
                doc_chars: int, min_chars: int):
    """Yield up to ``n`` concatenated general-text documents from a streamed HF set."""
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN") or None  # higher rate limits if set
    ds = load_dataset(dataset, config, split=split, streaming=True, token=token)
    buf, made = "", 0
    for row in ds:
        line = (row.get("text") or "").strip()
        if not line or line.startswith("=") or line.startswith(" ="):
            continue
        buf = f"{buf} {line}".strip()
        if len(buf) >= doc_chars:
            yield buf
            made += 1
            buf = ""
            if made >= n:
                return
    if buf and len(buf) >= min_chars and made < n:
        yield buf


def build(args) -> int:
    """Stream the corpus to JSONL; return the number of docs written."""
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as fh:
        for doc in stream_docs(args.dataset, args.config, args.split,
                               n=args.n, doc_chars=args.doc_chars, min_chars=args.min_chars):
            fh.write(json.dumps({"text": doc}) + "\n")
            n += 1
            if n % 1000 == 0:
                logger.info("  wrote %d docs", n)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream a general-text LM corpus to JSONL.")
    ap.add_argument("--output", required=True)
    # Namespaced id: the bare "wikitext" alias is rejected by newer `datasets`
    # ("Repository id must be 'namespace/name'"); Salesforce/wikitext is the repo.
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4000, help="Number of documents.")
    ap.add_argument("--doc-chars", type=int, default=600, help="Target chars per document.")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-docs", type=int, default=None,
                    help="Fail if fewer than this many docs are written (default n//4, >=100).")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    min_docs = args.min_docs if args.min_docs is not None else max(100, args.n // 4)

    n = 0
    for attempt in range(1, args.retries + 1):
        try:
            n = build(args)
        except Exception as e:  # transient streaming/network/rate-limit failure
            logger.warning("attempt %d/%d failed: %s", attempt, args.retries, e)
            n = 0
        if n >= min_docs:
            break
        if attempt < args.retries:
            logger.warning("got %d docs (< %d); retrying in %ds", n, min_docs, 5 * attempt)
            time.sleep(5 * attempt)

    # Fail loud: a too-small corpus would make Stage 1 train on padding (LM loss
    # collapses to ~0, Guard 4 -> nan). Exit non-zero so the harness skips Stage 1
    # rather than wasting GPU on a degenerate run.
    if n < min_docs:
        print(f"ERROR: only {n} docs from {args.dataset}/{args.config} after {args.retries} "
              f"attempts (need >= {min_docs}). Set HF_TOKEN for higher rate limits, or pass "
              f"STAGE1_GENERAL=<your own jsonl>.", file=sys.stderr)
        return 1

    meta = {"dataset": args.dataset, "config": args.config, "split": args.split,
            "n_docs": n, "doc_chars": args.doc_chars}
    with open(Path(args.output).with_suffix(".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {n} general-text docs to {args.output} (source: {args.dataset}/{args.config})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
