#!/usr/bin/env python3
"""Direction 1, Phase 1 step 5: full parallelized generation + audit + assembly.

Two-source TRAINING (Moral Stories workhorse + MORABLES, pooled iff it clears G-AXIS),
ETHICS zero-in-training. Eval is source-balanced and ETHICS's eval slice is the
generalization probe (PREREGISTRATION.md zero-ETHICS amendment).

For each (source, split) job, every pair is generated and audited inline (in parallel) and
tagged clean (passes all gates at ≥4). Then:
  * train = Moral Stories clean + MORABLES clean            (ETHICS contributes nothing)
  * eval  = source-balanced at the min per-source clean eval count (MORABLES, Moral
            Stories, ETHICS), ETHICS carrying its generalization-probe + selection-bias flag.

Reports final per-source train/eval counts (training is visibly two-source). Does NOT
paraphrase -- that is the next gated step, after the counts are reviewed.

Reuses the committed per-source generators + audit_runner. Requires ANTHROPIC_API_KEY.
Network + CPU only; no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))  # repo root for `deepsteer`

import audit_runner as ar  # noqa: E402
import generate_ethics as ge  # noqa: E402
import generate_morables as gm  # noqa: E402
import generate_moral_stories as gms  # noqa: E402
from _parallel import parallel_map  # noqa: E402

_OUT = HERE.parent / "outputs" / "full"


def _progress(tag: str):
    def f(done: int, total: int, errs: int) -> None:
        print(f"  [{tag}] {done}/{total} ({errs} failed)", flush=True)
    return f


# ---- per-source item builders + (generate + audit + tag) closures -----------


def morables_items(split: str, n: int) -> list[dict]:
    aliases = gm.load_train_aliases(split)[:n]
    fables = gm.load_fables()
    return [{"alias": a, "f": fables[a]} for a in aliases]


def make_morables_fn(client):
    def fn(item):
        f = item["f"]
        g = gm.generate_pair(client, f["title"], f["story"])
        pair = {"id": f"morables_{item['alias']}", "source": "morables",
                "alias": item["alias"], "register": "narrative",
                "moral": g["moral"], "neutral": g["neutral"]}
        aud = ar.audit_pair(client, pair["moral"], pair["neutral"],
                            ar._GATESET["fable_salience"])
        pair["clean"] = aud["clean"]
        return pair
    return fn


def ms_items(split: str, n: int) -> list[dict]:
    ids = gms.load_train_ids(split)[:n]
    inst = gms.load_instances()
    return [{"id": i, "r": inst[i]} for i in ids]


def make_ms_fn(client):
    def fn(item):
        r = item["r"]
        situation, ma = r["situation"], r["moral_action"]
        na = gms.generate_neutral_action(client, situation, ma)
        pair = {"id": f"moral_stories_{item['id']}", "source": "moral_stories",
                "register": "narrative", "moral": f"{situation} {ma}",
                "neutral": f"{situation} {na}"}
        aud = ar.audit_pair(client, pair["moral"], pair["neutral"],
                            ar._GATESET["moral_neutral"])
        pair["clean"] = aud["clean"]
        return pair
    return fn


def ethics_items(split: str, n: int) -> list[dict]:
    rows = ge.load_rows_by_hash()
    return ge.select(ge.load_train_hashes(split), rows, n)


def make_ethics_fn(client):
    def fn(r):
        scenario = r["input"]
        register = "declarative" if r["is_short"] == "True" else "narrative"
        neutral = ge.generate_neutral(client, scenario)
        pair = {"id": f"ethics_{r['hash']}", "source": "ethics", "register": register,
                "ethics_label": int(r["label"]), "moral": scenario, "neutral": neutral}
        aud = ar.audit_pair(client, pair["moral"], pair["neutral"],
                            ar._GATESET["moral_neutral"])
        pair["clean"] = aud["clean"]
        return pair
    return fn


def run_job(client, source, split, items, fn, workers) -> list[dict]:
    tag = f"{source}/{split}"
    print(f"\ngenerating+auditing {len(items)} {tag} (workers={workers})", flush=True)
    out, errs = parallel_map(fn, items, workers=workers, on_progress=_progress(tag))
    pairs = [p for p in out if p is not None]
    for p in pairs:
        p["split"] = split
    _OUT.mkdir(parents=True, exist_ok=True)
    with open(_OUT / f"{source}_{split}.json", "w") as fh:
        json.dump({"source": source, "split": split, "n": len(pairs),
                   "n_clean": sum(p["clean"] for p in pairs),
                   "n_failed": len(errs), "pairs": pairs}, fh, indent=2)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Full two-source generation + assembly.")
    ap.add_argument("--morables-train", type=int, default=567)
    ap.add_argument("--morables-eval", type=int, default=142)
    ap.add_argument("--ms-train", type=int, default=1000)
    ap.add_argument("--ms-eval", type=int, default=180)
    ap.add_argument("--ethics-eval", type=int, default=220)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import anthropic

    client = anthropic.Anthropic(max_retries=5)
    mfn, msfn, efn = make_morables_fn(client), make_ms_fn(client), make_ethics_fn(client)

    jobs = [
        ("morables", "train", morables_items("train", args.morables_train), mfn),
        ("morables", "eval", morables_items("eval", args.morables_eval), mfn),
        ("moral_stories", "train", ms_items("train", args.ms_train), msfn),
        ("moral_stories", "eval", ms_items("eval", args.ms_eval), msfn),
        ("ethics", "eval", ethics_items("eval", args.ethics_eval), efn),
    ]
    got: dict[tuple[str, str], list[dict]] = {}
    for source, split, items, fn in jobs:
        got[(source, split)] = run_job(client, source, split, items, fn, args.workers)

    clean = {k: [p for p in v if p["clean"]] for k, v in got.items()}

    # train = Moral Stories clean + MORABLES clean (ETHICS zero)
    train = clean[("moral_stories", "train")] + clean[("morables", "train")]
    # eval source-balanced at the min per-source clean eval count
    ev_sources = [("morables", "eval"), ("moral_stories", "eval"), ("ethics", "eval")]
    cap = min(len(clean[k]) for k in ev_sources)
    eval_set = [p for k in ev_sources for p in clean[k][:cap]]

    dataset = {
        "composition": "two-source training (moral_stories + morables); ethics eval-only",
        "train": train, "eval": eval_set,
        "counts": {
            "train": {s: sum(p["source"] == s for p in train)
                      for s in ("moral_stories", "morables", "ethics")},
            "eval_balanced_per_source": cap,
            "eval": {s: sum(p["source"] == s for p in eval_set)
                     for s in ("morables", "moral_stories", "ethics")},
        },
    }
    with open(_OUT / "dataset_v1.json", "w") as fh:
        json.dump(dataset, fh, indent=2)

    print("\n" + "=" * 60)
    print("PER-SOURCE CLEAN YIELD (clean / generated):")
    for (source, split), v in got.items():
        nc = sum(p["clean"] for p in v)
        print(f"  {source:<14} {split:<6} {nc:>4} / {len(v):<4} clean "
              f"({nc/max(len(v),1):.2f})")
    print("\nFINAL DATASET (training is two-source; ETHICS zero in training):")
    print(f"  TRAIN  moral_stories={dataset['counts']['train']['moral_stories']}  "
          f"morables={dataset['counts']['train']['morables']}  "
          f"ethics={dataset['counts']['train']['ethics']}  "
          f"| total {len(train)}")
    print(f"  EVAL   balanced @ {cap}/source  "
          f"morables={dataset['counts']['eval']['morables']}  "
          f"moral_stories={dataset['counts']['eval']['moral_stories']}  "
          f"ethics={dataset['counts']['eval']['ethics']} (generalization probe)  "
          f"| total {len(eval_set)}")
    print(f"  GRAND TOTAL {len(train) + len(eval_set)}")
    print(f"\nwrote {_OUT / 'dataset_v1.json'} (pre-paraphrase; eval not yet paraphrased)")


if __name__ == "__main__":
    main()
