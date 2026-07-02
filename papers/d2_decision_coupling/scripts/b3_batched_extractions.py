#!/usr/bin/env python3
"""B3 (batched extractions): the genuinely-non-moral control directions + the rotation-specificity
and fable-schema controls. Pre-registered in ../PREREGISTRATION.md (B3; rules R5, R6).

Two modes:
  extract   (default): on --model, extract syntax / register / sentiment mean-diff directions
            (raw format, the Paper 5 Sprint-1 convention) at the headline layer + band, saving
            directions AND per-pair diffs (for CIs). Also the amoral-cautionary "fable-schema"
            direction. Optionally compute R5 (project each control onto V_moral, compare to
            refusal) when --vmoral-npz / --refusal-npz are given.
  rotate    (--mode rotate): numpy-only base->SFT rotation compare across two extract outputs,
            reporting each control's rotation angle next to the moral subspace's ~40 deg (R6).

R5 rule: the strong-form orthogonality sentence holds iff refusal <= min(c_syntax, c_register) + M.
R6 rule: Paper 5 F2 keeps "specifically" iff moral rotation - control rotation >= 15 deg.
Fable-schema: cos(amoral_cautionary_dir, d_fables) vs d_moral bounds narrative-lesson-schema loading.

VALIDATE=1 -> tiny model, few pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_P6 = HERE.parents[1] / "6_cross_model" / "scripts"
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(_P6))
sys.path.insert(0, str(HERE))

from deepsteer.directions import extraction as du  # noqa: E402
from b1_judgment_direction import _frac, _ortho, load_vmoral_basis  # noqa: E402

MARGIN_M = 0.05
_unit = du.unit_vector

# 20 amoral cautionary pairs: (cautionary-tale-with-lesson, flat neutral statement of the same
# event). Physical / practical prudence, NO moral valence -- the fable-schema control.
FABLE_SCHEMA_PAIRS: list[tuple[str, str]] = [
    ("He touched the hot stove and burned his hand, so now he always checks first.",
     "He touched the stove and then washed his hands."),
    ("She ran on the wet floor, slipped, and fell; she learned to walk carefully there.",
     "She walked across the wet floor to the other side."),
    ("The hiker ignored the forecast, got caught in the storm, and now always checks the weather.",
     "The hiker checked the forecast and set out in the morning."),
    ("He left the milk out overnight, it spoiled, and he never forgot the fridge again.",
     "He put the milk back in the fridge after breakfast."),
    ("She overwatered the plant until it wilted, and learned to water it sparingly.",
     "She watered the plant a small amount in the morning."),
    ("The driver sped on the icy road, skidded, and after that always slowed in winter.",
     "The driver drove slowly along the icy road."),
    ("He didn't back up his files, lost the draft, and now saves copies every day.",
     "He saved a copy of his files to the drive."),
    ("She packed no water for the long walk, got parched, and always carries a bottle now.",
     "She packed a bottle of water for the walk."),
    ("He forgot to charge his phone, it died on the trip, and he now charges it nightly.",
     "He charged his phone before the trip."),
    ("The baker opened the oven too early, the cake sank, and she waits the full time now.",
     "The baker waited for the timer before opening the oven."),
    ("He wore new shoes on the marathon, got blisters, and now breaks shoes in first.",
     "He wore his old broken-in shoes for the marathon."),
    ("She left the window open in the rain, the sill warped, and she checks windows now.",
     "She closed the window before the rain started."),
    ("He mixed bleach with the other cleaner, choked on the fumes, and reads labels now.",
     "He used a single cleaner to wipe the counter."),
    ("The gardener pruned in the frost, the shrub died back, and now waits for spring.",
     "The gardener pruned the shrub in the spring."),
    ("She skipped the tutorial, got lost in the app, and now reads instructions first.",
     "She read the tutorial before using the app."),
    ("He carried too many boxes at once, dropped them, and now makes two trips.",
     "He carried the boxes to the car in two trips."),
    ("The cook tasted the dish only at the end, oversalted it, and seasons gradually now.",
     "The cook added a little salt and tasted as she went."),
    ("He parked under the tree in the wind, a branch dented the roof, and he parks clear now.",
     "He parked the car in the open lot away from trees."),
    ("She mistyped the password too many times, got locked out, and slows down now.",
     "She entered the password carefully and logged in."),
    ("The camper pitched the tent in a dip, woke up in a puddle, and picks high ground now.",
     "The camper pitched the tent on the high, flat ground."),
]


def control_pairs() -> dict[str, list[tuple[str, str]]]:
    from deepsteer.datasets import get_register_pairs, get_sentiment_pairs, get_syntax_pairs
    return {"syntax": get_syntax_pairs(), "register": get_register_pairs(),
            "sentiment": get_sentiment_pairs(), "fable_schema": FABLE_SCHEMA_PAIRS}


def extract_dir(model, pairs, layers) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return ({layer: unit mean-diff dir}, {layer: per-pair diffs (n,d)}), raw format."""
    acts = du.collect_pair_activations(model, pairs, input_format="raw", layers=layers)
    dirs, diffs = {}, {}
    for L in layers:
        X, y = acts[L]
        Xn = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        dirs[L] = _unit(du.mean_diff_direction(X, y))
        diffs[L] = Xn[0::2] - Xn[1::2]  # pos - neg per pair
    return dirs, diffs


def run_extract(args, spec, validate) -> None:
    model = du.load_whitebox(args.model)
    same = model.info.n_layers == spec.n_layers
    layer = spec.primary_layer if same else model.info.n_layers // 2
    band = (list(range(spec.band[0], spec.band[1] + 1)) if same else [layer])
    layers = sorted(set([layer, *band]))

    pairs = control_pairs()
    if validate:
        pairs = {k: v[:8] for k, v in pairs.items()}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dirs_hl, result = {}, {"model": args.model, "key": args.key, "layer": layer, "margin_M": MARGIN_M}
    for name, prs in pairs.items():
        dirs, diffs = extract_dir(model, prs, layers)
        np.savez(out / f"b3_{name}_dir_{args.key}.npz",
                 **{f"{name}_layer{L}": dirs[L] for L in layers})
        np.savez(out / f"b3_{name}_diffs_{args.key}.npz",
                 **{f"layer{L}": diffs[L] for L in layers})
        dirs_hl[name] = dirs[layer]
        result[f"n_{name}"] = len(prs)

    # Fable-schema probe: cos(amoral cautionary dir, d_fables) vs d_moral.
    if args.vmoral_npz and Path(args.vmoral_npz).exists():
        z = np.load(args.vmoral_npz)
        keyf, keym = f"fables_layer{layer}", f"moral_stories_layer{layer}"
        if keyf in z.files and keym in z.files:
            d_fab, d_mor = _unit(z[keyf].astype(np.float64)), _unit(z[keym].astype(np.float64))
            cs = dirs_hl["fable_schema"]
            result["fable_schema_probe"] = {
                "cos_amoral_cautionary_vs_fables": round(float(du.cosine(cs, d_fab)), 4),
                "cos_amoral_cautionary_vs_moral": round(float(du.cosine(cs, d_mor)), 4),
                "note": "high vs fables >> moral => d_fables loads narrative-lesson schema, "
                        "not moral content specifically"}

    # R5: project each non-moral control onto V_moral; compare refusal <= min(c_syntax,c_register)+M.
    if args.vmoral_npz and Path(args.vmoral_npz).exists():
        Qv = load_vmoral_basis(Path(args.vmoral_npz), layer)
        controls = {n: round(_frac(Qv, dirs_hl[n]), 4) for n in ("syntax", "register", "sentiment")}
        r5 = {"c_controls_on_vmoral": controls}
        if args.refusal_npz and Path(args.refusal_npz).exists():
            refusal = _unit(np.load(args.refusal_npz)["refusal"].astype(np.float64))
            p_ref = _frac(Qv, refusal)
            floor = min(controls["syntax"], controls["register"])
            r5.update({"refusal_p": round(p_ref, 4), "min_c_syntax_register": round(floor, 4),
                       "strong_form_holds": bool(p_ref <= floor + MARGIN_M),
                       "rule": "R5: keep 'below a known non-moral axis' iff refusal <= "
                               "min(c_syntax,c_register)+M"})
        result["R5"] = r5

    (out / f"b3_result_{args.key}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def run_rotate(args) -> None:
    """R6: base->SFT rotation of each control direction vs the moral subspace's ~40 deg."""
    out = Path(args.out)

    def load(tag, name):
        z = np.load(out / f"b3_{name}_dir_{tag}.npz")
        L = int([k for k in z.files][0].split("layer")[-1])
        return _unit(z[f"{name}_layer{args.layer if args.layer else L}"].astype(np.float64))

    res = {"moral_rotation_deg": args.moral_rotation_deg, "controls": {}}
    for name in ("syntax", "sentiment", "register"):
        try:
            b = load(args.base_tag, name); s = load(args.sft_tag, name)
            ang = float(np.degrees(np.arccos(np.clip(abs(du.cosine(b, s)), -1, 1))))
            keep = (args.moral_rotation_deg - ang) >= 15.0
            res["controls"][name] = {"rotation_deg": round(ang, 2),
                                     "moral_minus_control_deg": round(args.moral_rotation_deg - ang, 2),
                                     "F2_specifically_holds": bool(keep)}
        except (FileNotFoundError, KeyError) as e:
            res["controls"][name] = {"error": str(e)}
    res["rule"] = "R6: keep 'specifically' iff moral_rotation - control_rotation >= 15 deg"
    (out / "b3_rotation_compare.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="B3 batched control extractions (R5/R6 + fable schema).")
    ap.add_argument("--mode", choices=["extract", "rotate"], default="extract")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--key", default="olmo3")
    ap.add_argument("--vmoral-npz")
    ap.add_argument("--refusal-npz")
    ap.add_argument("--out", default=str(HERE.parent / "outputs"))
    # rotate mode
    ap.add_argument("--base-tag", default="olmo3_base")
    ap.add_argument("--sft-tag", default="olmo3_sft")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--moral-rotation-deg", type=float, default=40.0)
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate and args.mode == "extract":
        args.model = "allenai/OLMo-2-0425-1B"
    import model_registry as reg  # noqa: E402
    if args.mode == "rotate":
        run_rotate(args)
    else:
        run_extract(args, reg.get(args.key), validate)


if __name__ == "__main__":
    main()
