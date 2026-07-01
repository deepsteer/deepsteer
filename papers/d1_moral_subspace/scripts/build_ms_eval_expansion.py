#!/usr/bin/env python3
"""Direction 1: expand the in-distribution eval to all 96 clean Moral-Stories pairs.

Source-balance no longer binds (single-source V_moral), so eval_g2_indist can use the full
clean MS eval pool (96 narrative) instead of the balanced 53. This processes only the ~43 NEW
pairs (the 96 clean MS eval minus the 53 already in the dataset): paraphrase the narrative,
re-render to declarative + audit, paraphrase the clean declaratives. Reuses the committed
pipeline functions verbatim (no new construction logic). Writes ms_eval_expansion.json, which
assemble_single_source.py merges. API + CPU only; no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))

import audit_runner as ar  # noqa: E402
import generate_declarative_rerender as gd  # noqa: E402
import generate_paraphrases as gpp  # noqa: E402
from _parallel import parallel_map  # noqa: E402

_FULL = HERE.parent / "outputs" / "full"


def main() -> None:
    import anthropic

    ms_eval = json.load(open(_FULL / "ms_eval_full_pool.json")) if (
        _FULL / "ms_eval_full_pool.json").exists() else json.load(
        open(_FULL / "moral_stories_eval.json"))
    clean96 = [p for p in ms_eval["pairs"] if p["clean"]]

    ds = json.load(open(_FULL / "dataset_2reg.json"))
    existing = {p["id"] for p in ds["eval_g2_indist"]
                if p["source"] == "moral_stories" and p.get("register") == "narrative"}
    new = [p for p in clean96 if p["id"] not in existing]
    print(f"MS eval clean: {len(clean96)} | already in dataset: {len(existing)} | "
          f"new to process: {len(new)}", flush=True)

    client = anthropic.Anthropic(max_retries=5)
    para_fn = gpp.make_fn(client)

    def fn(narr):
        rows = []
        base = {k: narr[k] for k in ("id", "source", "register", "moral", "neutral")}
        pn = para_fn(narr)
        rows.append({**base, "split": "eval", "moral_para": pn["moral_para"],
                     "neutral_para": pn["neutral_para"], "paraphrase_status": pn["status"]})
        d = gd.rerender(client, narr["moral"], narr["neutral"])
        aud = ar.audit_pair(client, d["moral"], d["neutral"], ar._GATESET["moral_neutral"])
        if aud["clean"]:
            dp = {"id": f"{narr['id']}_decl", "source": narr["source"],
                  "register": "declarative", "source_pair_id": narr["id"],
                  "moral": d["moral"], "neutral": d["neutral"]}
            pd = para_fn(dp)
            rows.append({**dp, "split": "eval", "moral_para": pd["moral_para"],
                         "neutral_para": pd["neutral_para"], "paraphrase_status": pd["status"]})
        return rows

    out, errs = parallel_map(fn, new, workers=8,
                             on_progress=lambda d, t, e: print(f"  {d}/{t} ({e} err)", flush=True))
    rows = [r for grp in out if grp for r in grp]
    narr = [r for r in rows if r["register"] == "narrative"]
    decl = [r for r in rows if r["register"] == "declarative"]
    with open(_FULL / "ms_eval_expansion.json", "w") as fh:
        json.dump({"n_new_narrative": len(narr), "n_new_declarative": len(decl),
                   "errors": len(errs), "pairs": rows}, fh, indent=2)
    print(f"\nexpansion: +{len(narr)} narrative, +{len(decl)} declarative "
          f"(clean paraphrases: {sum(r['paraphrase_status'] == 'clean' for r in rows)})")
    print(f"wrote {_FULL / 'ms_eval_expansion.json'}")


if __name__ == "__main__":
    main()
