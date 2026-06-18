#!/usr/bin/env python3
"""Tier 2 / Task 2.1: per-checkpoint proto-refusal contrast extraction.

The ONLY genuinely-missing ingredient for the malleability scan. The per-stage-3
MFT 6-foundation directions and persona directions are already cached (raw,
base-matched conventions) under ``outputs/pipeline/<label>/probe_directions.npz``
from the Sprint 2.2 pipeline study; this pass adds the proto-refusal contrast,
which the pipeline study never computed.

Proto-refusal contrast at a checkpoint:

    proto_refusal[L] = unit( last_token_means(harmful)[L]
                             - last_token_means(harmless)[L] )

i.e. the §4.4 Heretic refusal construction (last-input-token difference of
means) applied to a *pre-training* checkpoint, where it measures the
proto-refusal feature already present before SFT wires the gate. Stage-3
checkpoints have no chat template, so ``--input-format raw`` is the natural and
default choice (the last token of the raw instruction is the position that would
produce the continuation, the base-model analog of §4.4's first-generated-token
position).

Reuses ``heretic_ablation.last_token_means`` verbatim so the contrast is the
same construction as the cached Instruct refusal direction (no second refusal
direction with mismatched conventions). Uses the SAME Heretic prompt set as
Paper 5 (``refusal_prompts.json``, p-e-w/heretic exact set); refuses to run on
the small fallback placeholder set.

Per checkpoint, writes ``outputs/measurement/stage3/<label>/``:
  * ``proto_refusal_directions.npz`` -- keys ``proto_refusal_layer{L}`` plus
    ``harmful_mean_layer{L}`` / ``harmless_mean_layer{L}`` (raw means kept for
    reproducibility / alternative contrasts).
  * ``proto_refusal_meta.json`` -- repo/revision, prompt provenance, n_prompts,
    input_format, per-layer contrast norm.

Mirrors ``pipeline_study.py`` / ``moral_dependency_pipeline.py``: grid-driven,
``--only`` subset, ``--purge-hf-cache`` to bound disk on RunPod, one bad
checkpoint never aborts the sweep.

Usage (RunPod, full stage-3 sweep):
    python papers/5_moral_alignment/scripts/extract_proto_refusal.py \
        --grid papers/5_moral_alignment/checkpoint_grid.json \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --output-dir papers/5_moral_alignment/outputs/measurement/stage3 \
        --purge-hf-cache --device cuda

    # cheap smoke (one cached model, few prompts):
    python papers/5_moral_alignment/scripts/extract_proto_refusal.py \
        --only olmo3_base --max-prompts 4 --device mps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_utils as du  # noqa: E402
from heretic_ablation import last_token_means  # noqa: E402
from pipeline_study import purge_repo_cache  # noqa: E402

logger = logging.getLogger(__name__)

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_GRID = _PAPER_ROOT / "checkpoint_grid.json"
_DEF_PROMPTS = _PAPER_ROOT / "refusal_prompts.json"
_DEF_OUT = _PAPER_ROOT / "outputs/measurement/stage3"


def select_states(grid: list[dict], *, only: str | None, include_base: bool) -> list[dict]:
    """Default selection: stage-3 anneal checkpoints (+ base unless --no-base).

    Base (== final pre-training, ~= stage3-step11921) gives the Measurement-4
    anti-artifact baseline. ``--only`` overrides with an explicit label set.
    """
    if only:
        keep = set(only.split(","))
        return [g for g in grid if g["label"] in keep]
    sel = [g for g in grid if "pretrain_stage3" in g["label"]]
    if include_base:
        base = [g for g in grid if g["label"] == "olmo3_base"]
        sel = base + sel
    return sel


def load_prompts(path: str, *, allow_fallback: bool) -> tuple[list[str], list[str], dict]:
    """Load the Heretic harmful/harmless training prompts; reject the fallback."""
    ps = json.load(open(path))
    harmful, harmless = ps["harmful"], ps["harmless"]
    prov = ps.get("provenance")
    is_fallback = prov is None or min(len(harmful), len(harmless)) < 50
    if is_fallback and not allow_fallback:
        raise RuntimeError(
            f"{path} looks like the small placeholder set (n_harmful={len(harmful)}, "
            f"provenance={prov!r}). Paper 5 requires the p-e-w/heretic exact set "
            f"(400/400). Pass --allow-fallback only for a throwaway smoke."
        )
    return harmful, harmless, {"path": path, "provenance": prov,
                               "n_harmful": len(harmful), "n_harmless": len(harmless)}


def extract_one(label, repo, revision, harmful, harmless, *, input_format, device, out_root):
    """Load one checkpoint, compute the proto-refusal contrast, cache it."""
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    out = Path(out_root) / label
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = WhiteBoxModel(repo, device=device, access_tier=AccessTier.WEIGHTS,
                          revision=revision)
    n_layers = model.info.n_layers
    layers = list(range(n_layers))
    logger.info("[%s] loaded %s@%s (%dL) in %.1fs", label, repo, revision,
                n_layers, time.time() - t0)

    h_means = last_token_means(model, harmful, input_format, layers)
    s_means = last_token_means(model, harmless, input_format, layers)
    model.release()

    proto = {}
    norms = {}
    for L in layers:
        r = h_means[L] - s_means[L]
        norms[L] = float(np.linalg.norm(r))
        proto[L] = (r / (norms[L] + 1e-12)).astype(np.float32)

    du.save_directions(out / "proto_refusal_directions.npz", {
        "proto_refusal": proto,
        "harmful_mean": {L: h_means[L].astype(np.float32) for L in layers},
        "harmless_mean": {L: s_means[L].astype(np.float32) for L in layers},
    })
    meta = {
        "label": label, "repo": repo, "revision": revision, "n_layers": n_layers,
        "input_format": input_format, "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS,
        "contrast_norm_per_layer": {str(L): round(norms[L], 6) for L in layers},
    }
    with open(out / "proto_refusal_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("[%s] proto-refusal cached (contrast norm @L16 = %.4f)",
                label, norms.get(16, float("nan")))
    return {"label": label, "ok": True, "contrast_norm_l16": round(norms.get(16, 0.0), 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-checkpoint proto-refusal contrast extraction.")
    ap.add_argument("--grid", default=str(_DEF_GRID))
    ap.add_argument("--prompts", default=str(_DEF_PROMPTS))
    ap.add_argument("--output-dir", default=str(_DEF_OUT))
    ap.add_argument("--input-format", choices=["raw", "chat"], default="raw",
                    help="raw: matches stage-3 (no chat template) and base extraction.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help="Comma-separated labels (subset).")
    ap.add_argument("--no-base", dest="include_base", action="store_false", default=True)
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="Cap prompts per class (smoke only).")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="Permit the placeholder prompt set (smoke only).")
    ap.add_argument("--purge-hf-cache", action="store_true",
                    help="Delete each repo's HF cache after processing (bounds RunPod disk).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    grid = json.load(open(args.grid))
    states = select_states(grid, only=args.only, include_base=args.include_base)
    harmful, harmless, prov = load_prompts(args.prompts, allow_fallback=args.allow_fallback)
    if args.max_prompts:
        harmful, harmless = harmful[:args.max_prompts], harmless[:args.max_prompts]
    print(f"Proto-refusal extraction: {len(states)} states, "
          f"{len(harmful)} harmful / {len(harmless)} harmless ({prov['provenance']})")

    summary = []
    for g in grid_iter(states):
        try:
            summary.append(extract_one(
                g["label"], g["repo"], g.get("revision"), harmful, harmless,
                input_format=args.input_format, device=args.device,
                out_root=args.output_dir,
            ))
        except Exception as e:  # one bad checkpoint must not kill the sweep
            logger.error("[%s] FAILED: %s", g["label"], e)
            summary.append({"label": g["label"], "ok": False, "error": str(e)})
        finally:
            if args.purge_hf_cache:
                purge_repo_cache(g["repo"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"analysis": "proto_refusal_extraction", "grid": args.grid,
               "prompts": prov, "input_format": args.input_format,
               "n_states": len(states), "states": summary}
    with open(out / "extraction_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    n_ok = sum(1 for s in summary if s.get("ok"))
    print(f"\nWrote {out/'extraction_summary.json'} ({n_ok}/{len(summary)} states OK)")


def grid_iter(states: list[dict]):
    """Yield states; isolated so a future resume/skip-existing hook is trivial."""
    for g in states:
        yield g


if __name__ == "__main__":
    main()
