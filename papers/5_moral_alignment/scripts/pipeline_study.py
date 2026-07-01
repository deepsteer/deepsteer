#!/usr/bin/env python3
"""Sprint 2.2: full pipeline study across the OLMo-3 training grid.

For each model state in a checkpoint grid (base / SFT-final / DPO-final /
Instruct + RLVR substeps + pretraining anneal), computes:

  1. fresh moral foundation directions (probe + mean-diff) per layer,
  2. transfer accuracy of the base-trained directions onto this state,
  3. framework geometry (cosine matrix, mean cosine, effective dim) per layer,
  4. persona direction (probe + mean-diff) + per-layer accuracy,
  5. persona-morality cosine angles.

Writes one subdirectory per model state:
    <output-dir>/<label>/{moral_probing.json, geometry.json,
                          persona_probing.json, persona_morality_angles.json,
                          probe_directions.npz}

Grid file format (see build_checkpoint_inventory.py -> proposed_sprint2_grid):
    [{"label": "...", "repo": "allenai/...", "revision": "main"}, ...]

Usage:
    python papers/5_moral_alignment/scripts/pipeline_study.py \
        --grid papers/5_moral_alignment/checkpoint_grid.json \
        --base-dir papers/5_moral_alignment/outputs/olmo3_base \
        --input-format chat \
        --output-dir papers/5_moral_alignment/outputs/pipeline \
        --device cuda
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
from deepsteer.directions import extraction as du  # noqa: E402
from probe_transfer import group_by_foundation  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

logger = logging.getLogger(__name__)


def purge_repo_cache(repo: str) -> None:
    """Delete a repo's HF hub cache to keep disk bounded during a long sweep.

    25 distinct 7B revisions are ~350 GB total; processing one then purging keeps
    peak disk near a single model. Revisions are distinct weights, so purging
    never forces a re-download of anything still needed.
    """
    import shutil
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        HF_HUB_CACHE = str(Path.home() / ".cache/huggingface/hub")
    d = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        logger.info("purged HF cache for %s", repo)


def geometry_from_directions(
    moral_probe: dict[str, dict[int, np.ndarray]], n_layers: int
) -> dict:
    """Cosine matrix, mean pairwise cosine, and eff dim per layer."""
    foundations = [f for f in FOUNDATION_ORDER if f in moral_probe]
    per_layer = {}
    for L in range(n_layers):
        vecs = [moral_probe[f][L] for f in foundations if L in moral_probe[f]]
        if len(vecs) < len(foundations):
            continue
        cos = du.cosine_matrix(vecs)
        n = len(vecs)
        upper = [cos[i, j] for i in range(n) for j in range(i + 1, n)]
        per_layer[str(L)] = {
            "mean_cosine": round(float(np.mean(upper)), 6),
            "min_cosine": round(float(np.min(upper)), 6),
            "max_cosine": round(float(np.max(upper)), 6),
            "effective_dimensionality": du.effective_dimensionality(vecs),
            "cosine_matrix": cos.tolist(),
        }
    return {"foundations": foundations, "per_layer": per_layer}


def run_one(label, repo, revision, base_dirs, dataset, persona_train, persona_test,
            input_format, device, out_root) -> dict:
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    out = Path(out_root) / label
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = WhiteBoxModel(repo, device=device, access_tier=AccessTier.WEIGHTS,
                          revision=revision)
    n_layers = model.info.n_layers
    logger.info("[%s] loaded %s@%s (%dL) in %.1fs", label, repo, revision,
                n_layers, time.time() - t0)

    train_by_f = group_by_foundation(dataset.train)
    test_by_f = group_by_foundation(dataset.test)
    foundations = [f for f in FOUNDATION_ORDER if f in base_dirs]

    moral_probe: dict[str, dict[int, np.ndarray]] = {}
    moral_md: dict[str, dict[int, np.ndarray]] = {}
    moral_json = {"label": label, "repo": repo, "revision": revision,
                  "n_layers": n_layers, "input_format": input_format,
                  "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS,
                  "per_foundation": {}}

    for f in foundations:
        tr, te = train_by_f.get(f, []), test_by_f.get(f, [])
        if not tr or not te:
            continue
        fresh, acc = du.extract_pair_directions(model, tr, input_format=input_format)
        moral_probe[f] = fresh["probe"]
        moral_md[f] = fresh["mean_diff"]
        test_acts = du.collect_pair_activations(model, te, input_format=input_format)
        per_layer = {}
        for L in range(n_layers):
            if L not in base_dirs.get(f, {}) or L not in test_acts:
                continue
            X, y = test_acts[L]
            tm = du.transfer_metrics(X, y, base_dirs[f][L])
            per_layer[str(L)] = {
                "fresh_probe_acc": round(acc[L], 4),
                "transfer_auc_abs": round(tm["auc_abs"], 4),
                "transfer_acc": round(tm["acc_midpoint"], 4),
                "cos_base_vs_fresh_probe": round(du.cosine(base_dirs[f][L], fresh["probe"][L]), 4),
            }
        moral_json["per_foundation"][f] = per_layer

    # persona
    pfresh, pacc = du.extract_pair_directions(
        model, persona_train, test_pairs=persona_test, input_format=input_format
    )
    model.release()

    # geometry
    geom = geometry_from_directions(moral_probe, n_layers)
    geom.update({"label": label, "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS})

    # persona-morality angles (probe directions, both sides)
    layers = sorted(set(pfresh["probe"]) & set.intersection(
        *(set(moral_probe[f]) for f in moral_probe)
    )) if moral_probe else []
    angles = {f: {str(L): round(du.cosine(pfresh["probe"][L], moral_probe[f][L]), 4)
                  for L in layers} for f in moral_probe}

    # write outputs
    with open(out / "moral_probing.json", "w") as fh:
        json.dump(moral_json, fh, indent=2)
    with open(out / "geometry.json", "w") as fh:
        json.dump(geom, fh, indent=2)
    persona_summary_layer = max(pacc, key=pacc.get)
    with open(out / "persona_probing.json", "w") as fh:
        json.dump({"label": label, "n_layers": n_layers,
                   "per_layer_accuracy": {str(L): round(pacc[L], 4) for L in sorted(pacc)},
                   "peak_layer": persona_summary_layer,
                   "peak_accuracy": round(pacc[persona_summary_layer], 4)}, fh, indent=2)
    with open(out / "persona_morality_angles.json", "w") as fh:
        json.dump({"label": label, "layers": layers, "angles": angles}, fh, indent=2)
    du.save_directions(out / "probe_directions.npz", {
        **moral_probe,
        **{f"{f}_meandiff": moral_md[f] for f in moral_md},
        "persona": pfresh["probe"], "persona_meandiff": pfresh["mean_diff"],
    })

    mean_geo = (np.mean([v["mean_cosine"] for v in geom["per_layer"].values()])
                if geom["per_layer"] else float("nan"))
    logger.info("[%s] done: persona peak %.2f @ L%d, mean geo cosine %.3f",
                label, pacc[persona_summary_layer], persona_summary_layer, mean_geo)
    return {"label": label, "persona_peak_acc": round(pacc[persona_summary_layer], 4),
            "mean_geo_cosine": round(float(mean_geo), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline study across checkpoint grid.")
    ap.add_argument("--grid", required=True, help="JSON list of {label,repo,revision}.")
    ap.add_argument("--base-dir", required=True,
                    help="Dir with base exp1_probe_directions.npz (transfer reference).")
    ap.add_argument("--input-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--output-dir", default="papers/5_moral_alignment/outputs/pipeline")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset-target", type=int, default=40)
    ap.add_argument("--only", default=None, help="Comma-separated labels to run (subset).")
    ap.add_argument("--purge-hf-cache", action="store_true",
                    help="Delete each repo's HF cache after processing (bounds disk on RunPod).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.datasets.persona_pairs import get_persona_dataset

    grid = json.load(open(args.grid))
    if args.only:
        keep = set(args.only.split(","))
        grid = [g for g in grid if g["label"] in keep]
    print(f"Pipeline grid: {len(grid)} model states")

    base_dirs = du.load_directions(str(Path(args.base_dir) / "exp1_probe_directions.npz"))
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target, dataset_version="v2")
    persona_train, persona_test = get_persona_dataset(test_fraction=0.2, seed=42, stratified=True)

    summary = []
    for g in grid:
        try:
            summary.append(run_one(
                g["label"], g["repo"], g.get("revision"), base_dirs, dataset,
                persona_train, persona_test, args.input_format, args.device,
                args.output_dir,
            ))
        except Exception as e:  # one bad checkpoint must not kill the sweep
            logger.error("[%s] FAILED: %s", g["label"], e)
            summary.append({"label": g["label"], "error": str(e)})
        finally:
            if args.purge_hf_cache:
                purge_repo_cache(g["repo"])

    with open(Path(args.output_dir) / "pipeline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {Path(args.output_dir)/'pipeline_summary.json'} ({len(summary)} states)")


if __name__ == "__main__":
    main()
