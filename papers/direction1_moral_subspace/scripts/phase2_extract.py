#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), stage 0: per-source extraction on OLMo-3 Base.

Extracts, from the TRAIN pairs of `dataset_2reg.json`, everything the downstream
pure-numpy stages need -- and DELIBERATELY does NOT touch the refusal direction (that is
extracted only in stage 4, phase2_g3.py, so the null in stage 3 structurally cannot have
seen it). Artifacts written to the phase2 artifact dir:

  * moral_directions.npz  -- per-source per-layer mean-diff direction (G-AXIS reads this).
  * diffs_<source>.npz     -- per-source per-pair difference matrices, stable band
                              (assemble_v_moral SVDs the pooled branch of these).
  * persona_direction.npz  -- persona/assistant axis per layer (G3 control c).
  * act_sample.npz         -- a sample of layer-16 pooled activations (null covariance).

Model defaults to OLMo-3 Base (V_moral is a Base representation; PREREGISTRATION §1).
VALIDATE=1 runs a tiny smoke (small model, few pairs) for the pre-GPU gate.

Open methodological flag (resolve at the pre-GPU test pass, NOT here): the G3 refusal
direction lives on the INSTRUCT model while V_moral is Base; projecting one onto the other
is the cross-model step Papers 5/6 handle (base→instruct cos ~0.76). phase2_g3.py takes a
separate --refusal-model; the choice + caveat is configured at run time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(HERE.parents[2]))

import direction_utils as du  # noqa: E402

STABLE_BAND = list(range(15, 32))
MATCH_LAYER = 16


def _pairs(rows, source):
    return [(p["moral"], p["neutral"]) for p in rows if p["source"] == source]


def _diff_matrix(X, y):
    """Per-pair (pos - neg) difference vectors -> (n_pairs, hidden)."""
    Xn = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    return Xn[0::2] - Xn[1::2]  # rows interleaved pos0,neg0,pos1,neg1,...


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 stage 0: per-source extraction.")
    ap.add_argument("--dataset", default=str(
        HERE.parents[2] / "deepsteer" / "datasets" / "direction1_vmoral_v1.json"))
    ap.add_argument("--model", default="allenai/OLMo-3-7B")  # Base; verify exact id pre-run
    ap.add_argument("--out", default=str(HERE.parent / "outputs" / "phase2"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--mft", action="store_true",
                    help="also extract the 6-MFT-foundation baseline (for Track-1 + eff-dim)")
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        args.model = "allenai/OLMo-2-0425-1B"  # tiny smoke model

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets import get_persona_pairs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = json.load(open(args.dataset))
    train = ds["train"]

    # Extract the sources actually present in train (single-source = moral_stories only;
    # MORABLES dropped). The reference source (for act_sample / null covariance) is the
    # alphabetically-first, i.e. moral_stories.
    sources = sorted({p["source"] for p in train})
    ref_source = sources[0]

    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    band = [L for L in STABLE_BAND if L < n_layers] or list(range(n_layers))
    layers = sorted(set(band) | {min(MATCH_LAYER, n_layers - 1)})

    moral_dirs: dict[str, dict[int, np.ndarray]] = {}
    for source in sources:
        pairs = _pairs(train, source)
        if validate:
            pairs = pairs[:8]
        if not pairs:
            continue
        acts = du.collect_pair_activations(model, pairs, input_format="raw", layers=layers)
        moral_dirs[source] = {L: du.mean_diff_direction(*acts[L]) for L in layers}
        diffs = {f"layer{L}": _diff_matrix(*acts[L]) for L in layers}
        np.savez(out / f"diffs_{source}.npz", **diffs)
        if source == ref_source:  # activation sample for the null covariance
            X16 = acts[min(MATCH_LAYER, n_layers - 1)][0]
            np.savez(out / "act_sample.npz",
                     X=X16.detach().cpu().numpy() if hasattr(X16, "detach") else X16,
                     layer=min(MATCH_LAYER, n_layers - 1))
    du.save_directions(out / "moral_directions.npz", moral_dirs)

    persona = get_persona_pairs()
    ppairs = [(p.moral if hasattr(p, "moral") else p[0],
               p.neutral if hasattr(p, "neutral") else p[1]) for p in persona]
    if validate:
        ppairs = ppairs[:8]
    pacts = du.collect_pair_activations(model, ppairs, input_format="raw", layers=layers)
    du.save_directions(out / "persona_direction.npz",
                       {"persona": {L: du.mean_diff_direction(*pacts[L]) for L in layers}})

    # MFT baseline (6-foundation subspace) for Track-1 fragility comparison + eff-dim contrast.
    if args.mft:
        from collections import defaultdict

        from deepsteer.datasets import load_moral_probing_v2
        by_f: dict[str, list] = defaultdict(list)
        for p in load_moral_probing_v2().all_pairs:
            by_f[p.foundation.value].append((p.moral, p.neutral))
        mft_dirs: dict[str, dict[int, np.ndarray]] = {}
        for fnd, fpairs in by_f.items():
            if validate:
                fpairs = fpairs[:4]
            facts = du.collect_pair_activations(model, fpairs, input_format="raw", layers=layers)
            mft_dirs[fnd] = {L: du.mean_diff_direction(*facts[L]) for L in layers}
        du.save_directions(out / "mft_directions.npz", mft_dirs)

    model.release()
    meta = {"model": args.model, "n_layers": n_layers, "band": band,
            "match_layer": min(MATCH_LAYER, n_layers - 1), "validate": validate,
            "sources_extracted": list(moral_dirs), "mft_extracted": args.mft}
    with open(out / "extract_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"stage0 extract done -> {out} | {meta}")


if __name__ == "__main__":
    main()
