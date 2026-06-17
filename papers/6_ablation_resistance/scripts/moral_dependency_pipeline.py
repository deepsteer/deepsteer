#!/usr/bin/env python3
"""Sprint 5.2: moral dependency across the OLMo-3 training pipeline.

Runs the Sprint 5.1 moral-ablation perplexity metric (``moral_dependency.py``)
on every state in the Phase 2 checkpoint grid (pretraining anneal -> base ->
SFT -> DPO -> Instruct/RLVR substeps), producing the natural moral-dependency
trajectory: does the model develop reliance on its moral subspace during
pretraining, and how does alignment change it?

By default every state is measured against the *base* model's moral directions
(transfer): the dependency score then reflects each state's reliance on the
fixed reference subspace, comparable across the whole pipeline. Pass
``--per-state-directions`` to instead ablate each state's own freshly extracted
directions (from the Phase 2 pipeline outputs), which measures self-dependency
but is not cross-state comparable.

Writes one ``<output-dir>/<label>/moral_dependency.json`` per state plus a
``dependency_summary.json`` with the trajectory (score, deltas, ppl ratio).

Usage (RunPod, full grid):
    python papers/6_ablation_resistance/scripts/moral_dependency_pipeline.py \
        --grid papers/5_moral_alignment/checkpoint_grid.json \
        --base-directions papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz \
        --output-dir papers/6_ablation_resistance/outputs/dependency \
        --purge-hf-cache --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moral_dependency as md  # noqa: E402

# purge_repo_cache keeps peak disk near one 7B model across the 25-revision sweep.
_PHASE2_SCRIPTS = Path(__file__).resolve().parents[2] / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_PHASE2_SCRIPTS))
from pipeline_study import purge_repo_cache  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_GRID = "papers/5_moral_alignment/checkpoint_grid.json"
_DEFAULT_BASE_DIRS = "papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz"
_DEFAULT_PIPELINE_DIR = "papers/5_moral_alignment/outputs/pipeline"


def run_one(entry, *, base_directions, per_state, pipeline_dir, direction_kind,
            moral_texts, neutral_texts, device, out_root, keep_per_text) -> dict:
    """Measure moral dependency for a single checkpoint-grid state."""
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    label, repo, revision = entry["label"], entry["repo"], entry.get("revision")
    out = Path(out_root) / label
    out.mkdir(parents=True, exist_ok=True)

    if per_state:
        npz = Path(pipeline_dir) / label / "probe_directions.npz"
        if npz.exists():
            directions = md.du.load_directions(str(npz))
            dir_source = str(npz)
        else:
            logger.warning("[%s] no per-state directions at %s; using base", label, npz)
            directions, dir_source = base_directions, "base(fallback)"
    else:
        directions, dir_source = base_directions, _DEFAULT_BASE_DIRS

    t0 = time.time()
    model = WhiteBoxModel(repo, device=device, access_tier=AccessTier.WEIGHTS,
                          revision=revision)
    n_layers = model.info.n_layers
    param = next(model.model.parameters())
    logger.info("[%s] loaded %s@%s (%dL) in %.1fs", label, repo, revision,
                n_layers, time.time() - t0)

    basis_by_layer, ranks, names = md.build_subspace_basis(
        directions, kind=direction_kind, n_layers=n_layers,
    )
    layers = sorted(basis_by_layer)
    if not layers:
        model.release()
        raise RuntimeError(f"[{label}] no complete moral subspace (hidden-dim mismatch?)")

    metrics = md.measure_dependency(
        model, moral_texts, neutral_texts, basis_by_layer, layers,
        keep_per_text=keep_per_text,
    )
    model.release()

    payload = {
        "analysis": "moral_dependency",
        "label": label, "model": repo, "revision": revision,
        "directions_source": dir_source, "direction_kind": direction_kind,
        "direction_names": names, "n_layers": n_layers,
        "ablated_layers": layers,
        "subspace_rank_per_layer": {str(L): ranks[L] for L in layers},
        "device": str(param.device), "dtype": str(param.dtype),
        "metrics": metrics,
    }
    with open(out / "moral_dependency.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    m = metrics
    logger.info("[%s] done: dependency %+.4f nats/tok (Δmoral %+.4f, Δneutral %+.4f)",
                label, m["moral_dependency_score"],
                m["delta_ce"]["moral"], m["delta_ce"]["neutral"])
    return {
        "label": label, "repo": repo, "revision": revision,
        "moral_dependency_score": m["moral_dependency_score"],
        "moral_dependency_ppl_ratio": m["moral_dependency_ppl_ratio"],
        "delta_ce_moral": m["delta_ce"]["moral"],
        "delta_ce_neutral": m["delta_ce"]["neutral"],
        "ce_moral": m["ce"]["moral"], "ce_neutral": m["ce"]["neutral"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Moral dependency across the pipeline grid.")
    ap.add_argument("--grid", default=_DEFAULT_GRID, help="JSON list of {label,repo,revision}.")
    ap.add_argument("--base-directions", default=_DEFAULT_BASE_DIRS,
                    help="Base exp1_probe_directions.npz used for transfer ablation.")
    ap.add_argument("--per-state-directions", action="store_true",
                    help="Ablate each state's own directions (not cross-state comparable).")
    ap.add_argument("--pipeline-dir", default=_DEFAULT_PIPELINE_DIR,
                    help="Phase 2 pipeline outputs (per-state probe_directions.npz).")
    ap.add_argument("--direction-kind", choices=["probe", "meandiff"], default="probe")
    ap.add_argument("--output-dir", default="papers/6_ablation_resistance/outputs/dependency")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset-target", type=int, default=40)
    ap.add_argument("--dataset-version", default="v2")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--max-texts", type=int, default=None)
    ap.add_argument("--only", default=None, help="Comma-separated labels to run (subset).")
    ap.add_argument("--no-per-text", dest="per_text", action="store_false", default=True)
    ap.add_argument("--purge-hf-cache", action="store_true",
                    help="Delete each repo's HF cache after processing (bounds RunPod disk).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    grid = json.load(open(args.grid))
    if args.only:
        keep = set(args.only.split(","))
        grid = [g for g in grid if g["label"] in keep]
    print(f"Pipeline grid: {len(grid)} model states")

    base_directions = md.du.load_directions(args.base_directions)
    moral_texts, neutral_texts, ds_meta = md.load_probing_texts(
        args.dataset_target, args.dataset_version,
        split=args.split, max_texts=args.max_texts,
    )
    print(f"Probing texts: {len(moral_texts)} moral / {len(neutral_texts)} neutral "
          f"({ds_meta['version']}, split={ds_meta['split']})")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for g in grid:
        try:
            summary.append(run_one(
                g, base_directions=base_directions, per_state=args.per_state_directions,
                pipeline_dir=args.pipeline_dir, direction_kind=args.direction_kind,
                moral_texts=moral_texts, neutral_texts=neutral_texts,
                device=args.device, out_root=out_root, keep_per_text=args.per_text,
            ))
        except Exception as e:  # one bad checkpoint must not kill the sweep
            logger.error("[%s] FAILED: %s", g["label"], e)
            summary.append({"label": g["label"], "error": str(e)})
        finally:
            if args.purge_hf_cache:
                purge_repo_cache(g["repo"])

    payload = {
        "analysis": "moral_dependency_pipeline",
        "grid": args.grid,
        "base_directions": args.base_directions,
        "per_state_directions": args.per_state_directions,
        "direction_kind": args.direction_kind,
        "dataset": ds_meta,
        "n_states": len(summary),
        "trajectory": summary,
    }
    with open(out_root / "dependency_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nWrote {out_root/'dependency_summary.json'} ({len(summary)} states)")
    print(f"  {'label':36s} {'score':>9s} {'Δmoral':>9s} {'Δneutral':>9s}")
    for s in summary:
        if "error" in s:
            print(f"  {s['label']:36s} {'ERROR':>9s}  {s['error'][:40]}")
        else:
            print(f"  {s['label']:36s} {s['moral_dependency_score']:+9.4f} "
                  f"{s['delta_ce_moral']:+9.4f} {s['delta_ce_neutral']:+9.4f}")


if __name__ == "__main__":
    main()
