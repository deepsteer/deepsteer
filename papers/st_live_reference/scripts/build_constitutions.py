#!/usr/bin/env python3
"""§13's documents of record, fetched and checked against their pins (``constitutions.yaml``):
the page is fetched, its text extracted deterministically (scripts and styles dropped, tags
removed, entities unescaped, whitespace collapsed; headings kept as ``## `` lines), and the text's
sha256 compared with the pin. A mismatch stops (the document changed since it was pinned). The
text goes to ``outputs/constitutions/<id>.txt`` (git-ignored; derived corpora are not bundled).

    python3 papers/st_live_reference/scripts/build_constitutions.py [--pin]

``--pin`` (once, at pinning) writes the fetched hashes into the yaml instead of checking them. A
document with ``raw_url`` (a markdown release at a pinned commit) is its own text.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ST = HERE.parent
PINS = ST / "constitutions.yaml"
OUT = ST / "outputs" / "constitutions"


def extract(raw: str) -> str:
    """Deterministic text: headings as ``## `` lines, paragraphs and list items on their own
    lines, everything else as running text."""
    t = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    t = re.sub(r"<h([1-4])\b[^>]*>(.*?)</h\1>",
               lambda m: "\n## " + re.sub(r"<[^>]+>", " ", m.group(2)) + "\n", t,
               flags=re.S | re.I)
    t = re.sub(r"</?(p|li|br|div|tr|blockquote)\b[^>]*>", "\n", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    lines = [re.sub(r"[ \t ]+", " ", x).strip() for x in t.splitlines()]
    return "\n".join(x for x in lines if x) + "\n"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "deepsteer-st/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (pinned https URLs)
        return r.read().decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", action="store_true")
    a = ap.parse_args()
    pins = yaml.safe_load(PINS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for doc in pins["documents"]:
        if doc.get("raw_url"):  # a markdown release (C_spec2025): the file is the text
            raw = fetch(doc["raw_url"])
            text = raw
        else:
            raw = fetch(doc["url"])
            text = extract(raw)
        got = {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(), "n_chars": len(text)}
        if a.pin:
            doc.update(got, fetched=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        elif got["text_sha256"] != doc["text_sha256"]:
            raise SystemExit(f"{doc['id']}: text sha256 {got['text_sha256'][:12]} != pinned "
                             f"{doc['text_sha256'][:12]} (the document changed since pinning)")
        (OUT / f"{doc['id']}.txt").write_text(text)
        print(f">> {doc['id']}: {got['n_chars']} chars, text {got['text_sha256'][:12]}")
    if a.pin:
        PINS.write_text(yaml.safe_dump(pins, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
