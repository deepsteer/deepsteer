#!/usr/bin/env python3
"""Direction 1, Phase 1 step 5: declarative register re-rendering (content held constant).

Re-renders the already-clean NARRATIVE pairs (train + in-distribution eval) into DECLARATIVE
surface, holding the moral content and valence contrast FIXED and changing only the register
(concrete event -> general statement). This is a surface transform of audited content, NOT
fresh declarative generation: generating new declarative pairs from scratch would reintroduce
the §1.5/construction risk and re-confound register with content (CONSTRUCTION_GUIDELINES.md).

Why it matters (not hygiene): project memory finds linear probes fail declarative↔narrative
transfer, so a narrative-only `V_moral` is likely register-specific — which would undercut
the "richer, more general instrument" that is Direction 1's answer to the thin-MFT objection.
Holding content constant also decomposes the otherwise-confounded ETHICS probe into two clean
tests: (a) cross-register at constant content (narrative vs declarative re-renderings of the
same in-dist pairs = clean Track 4), and (b) the ETHICS declarative probe, now interpretable.

Each re-rendering is re-audited (a transform can still break §1.2 parallelism or drift
valence); only clean re-renderings are kept. Dialogue is deferred (it cannot be re-rendered
from fable/action content without inventing speakers, i.e. new content).

Requires ANTHROPIC_API_KEY. Network + CPU only; no GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import audit_runner as ar  # noqa: E402
from _parallel import parallel_map  # noqa: E402

MODEL = "claude-sonnet-4-6"
_FULL = HERE.parent / "outputs" / "full"

_PROMPT = """\
Here is a contrastive pair in NARRATIVE register (a concrete moral event and its mundane
counterpart):

MORAL (narrative):   {moral}
NEUTRAL (narrative): {neutral}

Re-render BOTH sides in DECLARATIVE register, preserving the EXACT moral content and the
moral-vs-neutral valence contrast. Change ONLY the register.

- Declarative = a general factual statement, present tense, 10-25 words, no first/second
  person, no concrete named scene. State the moral principle the MORAL narrative expresses,
  and the mundane counterpart the NEUTRAL narrative expresses.
- Keep the SAME moral content and the SAME valence contrast. The MORAL declarative is
  morally loaded; the NEUTRAL declarative carries no moral weight at all.
- The two declaratives must be structurally parallel (same frame), differing only in moral
  valence. No em-dashes; natural English.

Reply with ONLY a JSON object: {{"moral": "<declarative>", "neutral": "<declarative>"}}."""


def rerender(client, moral: str, neutral: str) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": _PROMPT.format(moral=moral, neutral=neutral)}],
    )
    text = resp.content[0].text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in rerender response: {text[:200]!r}")
    obj = json.loads(m.group())
    return {"moral": obj["moral"].strip(), "neutral": obj["neutral"].strip()}


def make_fn(client):
    def fn(item):
        src = item["pair"]
        d = rerender(client, src["moral"], src["neutral"])
        pair = {"id": f"{src['id']}_decl", "source": src["source"],
                "source_pair_id": src["id"], "register": "declarative",
                "target": item["target"], "moral": d["moral"], "neutral": d["neutral"]}
        aud = ar.audit_pair(client, pair["moral"], pair["neutral"],
                            ar._GATESET["moral_neutral"])
        pair["clean"] = aud["clean"]
        return pair
    return fn


def reg(pairs: list[dict]) -> dict:
    return dict(Counter(p["register"] for p in pairs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Declarative register re-rendering.")
    ap.add_argument("--in", dest="inp", default=str(_FULL / "dataset_structured.json"))
    ap.add_argument("--out", default=str(_FULL / "dataset_2reg.json"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import anthropic

    ds = json.load(open(args.inp))
    items = ([{"pair": p, "target": "train"} for p in ds["train"]]
             + [{"pair": p, "target": "eval_g2_indist"} for p in ds["eval_g2_indist"]])
    print(f"re-rendering {len(items)} narrative pairs -> declarative (workers={args.workers})",
          flush=True)

    client = anthropic.Anthropic(max_retries=5)
    out, errs = parallel_map(
        make_fn(client), items, workers=args.workers,
        on_progress=lambda d, t, e: print(f"  {d}/{t} ({e} failed)", flush=True))
    decl = [p for p in out if p is not None]
    decl_clean = [p for p in decl if p["clean"]]

    # merge: narrative (kept) + clean declarative re-renderings, content-paired by source_pair_id
    train = ds["train"] + [p for p in decl_clean if p["target"] == "train"]
    eval_g2 = ds["eval_g2_indist"] + [p for p in decl_clean if p["target"] == "eval_g2_indist"]
    merged = {
        **{k: ds[k] for k in ("composition", "eval_structure")},
        "registers": "narrative + declarative re-renderings (content-paired); dialogue deferred",
        "train": train, "eval_g2_indist": eval_g2,
        "eval_generalization_probe": ds["eval_generalization_probe"],
        "register_composition": {
            "train": reg(train), "eval_g2_indist": reg(eval_g2),
            "eval_generalization_probe": reg(ds["eval_generalization_probe"]),
        },
        "rerender_yield": {
            "attempted": len(items), "clean": len(decl_clean), "failed": len(errs),
            "clean_rate": round(len(decl_clean) / max(len(decl), 1), 3),
        },
    }
    with open(_FULL / "declarative_rerenders.json", "w") as fh:
        json.dump({"pairs": decl}, fh, indent=2)
    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=2)

    print("\n=== TWO-REGISTER DATASET ===")
    print(f"declarative re-render clean yield: {len(decl_clean)}/{len(decl)} "
          f"({merged['rerender_yield']['clean_rate']})")
    for k in ("train", "eval_g2_indist", "eval_generalization_probe"):
        print(f"  {k:<28} n={len(merged[k]):<5} registers={merged['register_composition'][k]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
