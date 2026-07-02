#!/usr/bin/env python3
"""C1 typing prep (zero-GPU): type every patch-source stimulus set with a construct-audit type block
+ token-alignment metadata, and provide the behavioral-discrimination pilot screen. Pre-registered
in ../PREREGISTRATION.md Amendment 1.

Three sources, three roles:
  * compositional twins (v2, 200 surface-matched moral-status pairs) -> outcome_variable=judgment;
    Δrefusal reported but pre-registered EXPECTED-FLAT (narrative, non-refusal-triggering).
  * request-twins (24, hand-authored) -> outcome_variable=refusal; the MINIMAL-PAIR refusal-patching
    stimuli; token-aligned by a shared prefix + flipped moral-intent span.
  * XSTest borderline (40, committed) -> the CATEGORY-CONTRAST GENERALIZATION cell (harm-cue vs
    surface separation), NOT a minimal pair.

`build_manifest()` is zero-GPU (types + alignment). `screen(model, ...)` is the pilot gate (needs a
model; MPS-runnable on OLMo): keep only pairs whose baseline behavior differs across the twin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "d2_decision_coupling" / "scripts"))

from deepsteer.datasets import get_compositional_moral_pairs, get_request_twins  # noqa: E402

OUT = HERE.parent / "outputs"


def _commit() -> str:
    try:  # stderr silenced: the synced pod has no .git ("fatal: not a git repository" noise)
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(HERE),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _align(a: str, b: str) -> dict:
    """Shared-prefix + flipped-span alignment (char-level; token-level computed at patch time).
    Amendment 1 rule: align on shared prefix + flipped content span."""
    p = 0
    for x, y in zip(a, b):
        if x != y:
            break
        p += 1
    # shared suffix after the divergence (often a trailing period)
    s = 0
    for x, y in zip(a[::-1], b[::-1]):
        if x != y or s >= min(len(a), len(b)) - p:
            break
        s += 1
    return {"shared_prefix_chars": p, "shared_suffix_chars": s,
            "flipped_span_following": a[p:len(a) - s], "flipped_span_violating": b[p:len(b) - s]}


def type_block(**kw) -> dict:
    """A construct-audit type block for a stimulus SET (adapted from the direction schema)."""
    base = {"source_dataset": None, "format": None, "contrast_semantics": None,
            "outcome_variable": None, "role": None, "known_covariates": None,
            "n_pairs": None, "extraction_commit": _commit()}
    base.update(kw)
    return base


def build_manifest() -> dict:
    twins = get_compositional_moral_pairs()
    reqs = get_request_twins()
    xst = json.loads((HERE.parents[1] / "d2_decision_coupling" / "data"
                      / "xstest_borderline.json").read_text())["items"]

    manifest = {
        "compositional_twins": {
            "type_block": type_block(
                source_dataset="deepsteer.datasets.get_compositional_moral_pairs (v2)",
                format="raw narrative (3rd person)", contrast_semantics="moral-status flip (intent)",
                outcome_variable="judgment-decision readout",
                role="minimal-pair; refusal EXPECTED-FLAT (non-refusal-triggering)",
                known_covariates="minimal by construction (surface-matched, single-intent-span flip); "
                                 "residual: verb-frame, valence of the flipped intent",
                n_pairs=len(twins)),
            "pairs": [{"moral": a, "neutral_or_violating": b, "align": _align(a, b)}
                      for a, b in twins]},
        "request_twins": {
            "type_block": type_block(
                source_dataset="deepsteer.datasets.get_request_twins (hand-authored)",
                format="chat request (1st person imperative)",
                contrast_semantics="moral-status flip (norm-following vs violating intent)",
                outcome_variable="refusal (behavioral + decision-token projection)",
                role="MINIMAL-PAIR refusal-patching stimuli (cells a/b)",
                known_covariates="shared prefix; residual: harm severity of the violating intent",
                n_pairs=len(reqs)),
            "pairs": [{"foundation": f, "following": a, "violating": b, "align": _align(a, b)}
                      for f, a, b in reqs]},
        "xstest_generalization": {
            "type_block": type_block(
                source_dataset="papers/d2_decision_coupling/data/xstest_borderline.json (CC-BY-4.0)",
                format="chat request", contrast_semantics="category contrast (safe vs unsafe; "
                                                           "alarming surface variably held)",
                outcome_variable="refusal",
                role="GENERALIZATION cell (harm-cue vs surface separation, cell c) -- NOT a minimal pair",
                known_covariates="category-level, not surface-matched; alarming lexical surface",
                n_pairs=len(xst)),
            "items": xst},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patch_stimuli_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


# ------------------------- behavioral-discrimination pilot screen (needs a model) -------------------------

def _judgment_flips(model, a: str, b: str) -> bool:
    """Baseline judgment differs across a moral-status twin (the outcome twins can move)."""
    from b1_judgment_direction import build_prompt, parse_verdict
    va = parse_verdict(model.generate(build_prompt(a, False)[0], max_tokens=6, temperature=0.0).text)
    vb = parse_verdict(model.generate(build_prompt(b, False)[0], max_tokens=6, temperature=0.0).text)
    return va != vb and "other" not in (va, vb)


def _refusal_differs(model, a: str, b: str) -> bool:
    """Baseline refusal differs across a request-twin (the outcome request-twins can move)."""
    from b1_judgment_direction import is_refusal
    ra = is_refusal(model.generate(a, max_tokens=48, temperature=0.0).text)
    rb = is_refusal(model.generate(b, max_tokens=48, temperature=0.0).text)
    return ra != rb


def screen(model, manifest: dict) -> dict:
    """Pilot gate (Amendment 1): keep only pairs with a baseline behavioral gap across the twin."""
    kept = {"compositional_twins": [], "request_twins": []}
    for p in manifest["compositional_twins"]["pairs"]:
        if _judgment_flips(model, p["moral"], p["neutral_or_violating"]):
            kept["compositional_twins"].append(p)
    for p in manifest["request_twins"]["pairs"]:
        if _refusal_differs(model, p["following"], p["violating"]):
            kept["request_twins"].append(p)
    kept["counts"] = {k: len(v) for k, v in kept.items() if isinstance(v, list)}
    kept["dropped"] = {"compositional_twins": len(manifest["compositional_twins"]["pairs"])
                       - len(kept["compositional_twins"]),
                       "request_twins": len(manifest["request_twins"]["pairs"])
                       - len(kept["request_twins"])}
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="C1 patch-stimulus typing + pilot screen.")
    ap.add_argument("--screen", action="store_true", help="run the model-based behavioral screen")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    args = ap.parse_args()
    manifest = build_manifest()
    print(f"typed manifest: twins {manifest['compositional_twins']['type_block']['n_pairs']}, "
          f"request-twins {manifest['request_twins']['type_block']['n_pairs']}, "
          f"xstest {manifest['xstest_generalization']['type_block']['n_pairs']} -> "
          f"{OUT / 'patch_stimuli_manifest.json'}")
    if args.screen:
        from deepsteer.directions import extraction as du
        model = du.load_whitebox(args.model)
        kept = screen(model, manifest)
        (OUT / "patch_stimuli_screened.json").write_text(json.dumps(kept, indent=2, ensure_ascii=False))
        print(f"pilot screen kept: {kept['counts']} (dropped {kept['dropped']})")


if __name__ == "__main__":
    main()
