#!/usr/bin/env python3
"""Direction 1, Phase 1, steps 1-2: source-item-level partition + counts.

Partitions the three PRIMARY moral sources into disjoint train-source / eval-source
pools BEFORE any pair construction, at the scenario/source-item granularity (not the
sentence level). This is what makes GATE G2 a test of generalization to genuinely
unseen moral situations in unseen surface, rather than memorized scenarios in new
wording:

  * MORABLES (cardiffnlp/Morables)   -> unit = fable ``alias``        (709 fables)
  * Moral Stories (demelin/...)        -> unit = instance ``ID``        (12,000)
  * ETHICS commonsense (hendrycks/...) -> unit = unique ``input`` text  (deduped)

Training pairs (Phase 1 step 5) come ONLY from each source's train pool; the held-out
paraphrase set comes ONLY from the eval pool. Per-source balance is tracked in each
split independently so an apparent paraphrase gap cannot be a source-shift confound.

Social Chemistry 101 is NOT partitioned here: it is the OOD generalization probe,
held out entirely, and directions are never extracted from it.

Outputs (committed): ``partition_manifest.json`` (identifiers only -- no raw source
text, per the no-bundled-corpora convention) and a printed counts report. Re-running
with the same SEED reproduces the split exactly.

Network + CPU only; no GPU. Sources are fetched from the Hugging Face hub and cached.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import datasets as hfds
from huggingface_hub import hf_hub_download

hfds.disable_progress_bars()

SEED = 42
EVAL_FRAC = 0.20
# Minimum eval-pool items for a source to support a balanced held-out split:
# ~100 eval pairs gives a 95% accuracy CI of roughly +/-0.05 at p~0.7, tight enough
# that a real G2 paraphrase gap is not lost in source-level sampling noise.
MIN_EVAL_ITEMS = 100

_OUT = Path(__file__).resolve().parent.parent
_MANIFEST = _OUT / "partition_manifest.json"      # committed: summary + checksums
_IDS = _OUT / "partition_ids.json"                # gitignored: full ID lists (regen)


# ---------------------------------------------------------------------------
# Loaders (return list of (item_key, meta) at the partition granularity)
# ---------------------------------------------------------------------------


def load_morables() -> list[tuple[str, dict]]:
    """Unit = fable alias. Native registers: declarative (moral) + narrative (story).

    The ``binary`` config supplies the correct-vs-opposite moral minimal pair, one per
    fable; ``alias`` is the stable join key back to ``fables_only``.
    """
    d = hfds.load_dataset("cardiffnlp/Morables", "binary")
    split = d["binary_not_shuffled"]
    items: list[tuple[str, dict]] = []
    for r in split:
        items.append((r["alias"], {
            "title": r["story_title"],
            "is_altered": bool(r["is_altered"]),
            "registers_native": ["declarative", "narrative"],
        }))
    # alias is unique per fable
    assert len(items) == len({k for k, _ in items}), "MORABLES alias not unique"
    return items


def load_moral_stories() -> list[tuple[str, dict]]:
    """Unit = instance ID. Native register: narrative (situation + action)."""
    p = hf_hub_download("demelin/moral_stories", "data/moral_stories_full.jsonl",
                        repo_type="dataset")
    rows = [json.loads(line) for line in open(p)]
    items = [(r["ID"], {"norm": r["norm"], "registers_native": ["narrative"]})
             for r in rows]
    assert len(items) == len({k for k, _ in items}), "Moral Stories ID not unique"
    return items


def _ethics_rows() -> list[dict]:
    rows: list[dict] = []
    for fn in ("train.csv", "test.csv", "test_hard.csv"):
        pp = hf_hub_download("hendrycks/ethics", f"data/commonsense/{fn}",
                             repo_type="dataset")
        with open(pp) as fh:
            for r in csv.DictReader(fh):
                r["_file"] = fn
                rows.append(r)
    return rows


def load_ethics() -> list[tuple[str, dict]]:
    """Unit = unique input text (deduped across train/test/test_hard).

    Register hint: ``is_short`` (True ~ declarative single sentence, False ~ narrative
    multi-sentence post). The unit key is a short hash of the input so the manifest
    stores no raw scenario text.
    """
    rows = _ethics_rows()
    seen: dict[str, dict] = {}
    for r in rows:
        text = r["input"]
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        if h in seen:
            continue
        seen[h] = {
            "label": int(r["label"]),                # 0 acceptable, 1 unacceptable
            "is_short": r["is_short"] == "True",
            "src_file": r["_file"],
            "register_native": "declarative" if r["is_short"] == "True" else "narrative",
        }
    return list(seen.items())


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


def split_items(items: list[tuple[str, dict]], seed: int) -> tuple[list, list]:
    keys = sorted(k for k, _ in items)             # deterministic base order
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_eval = round(len(keys) * EVAL_FRAC)
    eval_keys = set(keys[:n_eval])
    train = [k for k in keys if k not in eval_keys]
    ev = [k for k in keys if k in eval_keys]
    return sorted(train), sorted(ev)


def balance_report(source: str, items: list[tuple[str, dict]],
                   train: list[str], ev: list[str]) -> dict:
    meta = dict(items)
    rep: dict = {"unit": {"morables": "fable_alias", "moral_stories": "instance_id",
                          "ethics": "input_sha1_16"}[source],
                 "n_total": len(items), "n_train": len(train), "n_eval": len(ev)}
    if source == "ethics":
        for name, keys in (("train", train), ("eval", ev)):
            labels = Counter(meta[k]["label"] for k in keys)
            shorts = Counter("short" if meta[k]["is_short"] else "long" for k in keys)
            rep[f"{name}_label_0acc_1unacc"] = dict(labels)
            rep[f"{name}_register"] = dict(shorts)
    rep["supports_balanced_eval"] = len(ev) >= MIN_EVAL_ITEMS
    return rep


def main() -> None:
    sources = {
        "morables": load_morables(),
        "moral_stories": load_moral_stories(),
        "ethics": load_ethics(),
    }
    manifest: dict = {"seed": SEED, "eval_frac": EVAL_FRAC,
                      "min_eval_items": MIN_EVAL_ITEMS,
                      "note": ("Identifiers-only split. Full ID lists live in the "
                               "gitignored partition_ids.json; re-run this script to "
                               "regenerate them and verify against the checksums below."),
                      "sources": {}}
    ids: dict = {"seed": SEED, "ids": {}}
    print(f"\n{'source':<16}{'unit':<18}{'total':>8}{'train':>8}{'eval':>8}"
          f"{'balanced?':>11}")
    print("-" * 71)
    all_ok = True
    for name, items in sources.items():
        train, ev = split_items(items, SEED)
        rep = balance_report(name, items, train, ev)
        # checksum the sorted id lists so the regenerable split is verifiable
        rep["sha256"] = {
            s: hashlib.sha256("\n".join(lst).encode()).hexdigest()
            for s, lst in (("train", train), ("eval", ev))
        }
        manifest["sources"][name] = rep
        ids["ids"][name] = {"train": train, "eval": ev}
        all_ok &= rep["supports_balanced_eval"]
        print(f"{name:<16}{rep['unit']:<18}{rep['n_total']:>8}{rep['n_train']:>8}"
              f"{rep['n_eval']:>8}{('YES' if rep['supports_balanced_eval'] else 'NO'):>11}")

    with open(_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(_IDS, "w") as fh:
        json.dump(ids, fh)
    print(f"\nwrote {_MANIFEST.relative_to(_OUT.parent.parent)} (committed)")
    print(f"wrote {_IDS.relative_to(_OUT.parent.parent)} (gitignored, regenerable)")

    print("\n--- ETHICS balance (per split) ---")
    e = manifest["sources"]["ethics"]
    for s in ("train", "eval"):
        print(f"  {s:5} labels {e[f'{s}_label_0acc_1unacc']}  "
              f"register {e[f'{s}_register']}")

    print(f"\nALL SOURCES SUPPORT A BALANCED EVAL SPLIT: {all_ok}")
    if not all_ok:
        print("  >>> PAUSE: at least one source is below MIN_EVAL_ITEMS. Do not build.")


if __name__ == "__main__":
    main()
