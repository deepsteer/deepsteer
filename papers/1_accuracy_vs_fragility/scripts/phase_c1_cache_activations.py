"""Cache per-checkpoint probing activations for the OLMo-2 1B early-training
trajectory, downloading and then deleting each checkpoint to bound disk use.

For each of the 37 early-training checkpoints (steps 0..36000 at 1K), this:
  1. downloads the checkpoint weights (revision),
  2. computes mean-pooled train+test activations at every layer for the
     standard probing dataset (v2, 192 train / 48 test),
  3. saves them to outputs/phase_c1_acts/step_<step>.npz,
  4. deletes the checkpoint from the HuggingFace cache (peak disk ~one ckpt).

Downstream, phase_c1_refragility.py computes seed-averaged, extended-grid
fragility (and the RMS-normalized control) from these cached activations
without re-loading any model.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import argparse

REPO = "allenai/OLMo-2-0425-1B-early-training"
TARGET_STEPS = set(range(0, 37000, 1000))  # 0, 1000, ..., 36000
OUT_DIRS = {
    "standard": Path("papers/1_accuracy_vs_fragility/outputs/phase_c1_acts"),
    "compositional": Path("papers/1_accuracy_vs_fragility/outputs/phase_c4_comp_acts"),
}


def _build_dataset(kind: str):
    if kind == "compositional":
        from deepsteer.benchmarks.representational.compositional_moral_probe import (
            _build_compositional_probing_dataset,
        )
        return _build_compositional_probing_dataset(seed=42)
    from deepsteer.datasets.pipeline import build_probing_dataset
    return build_probing_dataset(target_per_foundation=40, dataset_version="v2")


def _free_repo_cache() -> None:
    """Remove the repo's cached weights so peak disk stays at ~one checkpoint.

    delete_revisions() proved unreliable here (snapshots accumulated), so we
    remove the repo cache directory outright; config/tokenizer re-download next
    iteration is cheap relative to the ~6 GB weights.
    """
    import shutil

    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE

    # Primary: blow away the whole repo cache dir.
    repo_dirname = "models--" + REPO.replace("/", "--")
    repo_dir = Path(HF_HUB_CACHE) / repo_dirname
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)

    # Backup: if a non-default cache path is in use, fall back to the API.
    if repo_dir.exists():
        ci = scan_cache_dir()
        hashes = [rev.commit_hash for repo in ci.repos if repo.repo_id == REPO
                  for rev in repo.revisions]
        if hashes:
            scan_cache_dir().delete_revisions(*hashes).execute()


def main() -> None:
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.benchmarks.representational.trajectory import (
        _parse_step_from_revision,
        list_available_revisions,
    )
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["standard", "compositional"], default="standard")
    args = ap.parse_args()
    out_dir = OUT_DIRS[args.dataset]
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(args.dataset)
    log.info("Dataset (%s): %d train, %d test pairs", args.dataset, len(dataset.train), len(dataset.test))

    revs = list_available_revisions(REPO)
    by_step: dict[int, str] = {}
    for r in revs:
        s = _parse_step_from_revision(r)
        if s in TARGET_STEPS and s not in by_step:
            by_step[s] = r
    steps = sorted(by_step)
    log.info("Resolved %d/%d target checkpoints", len(steps), len(TARGET_STEPS))

    for i, step in enumerate(steps):
        out = out_dir / f"step_{step:07d}.npz"
        if out.exists():
            log.info("[%d/%d] step %d cached, skipping", i + 1, len(steps), step)
            continue
        rev = by_step[step]
        log.info("[%d/%d] step %d (%s): loading", i + 1, len(steps), step, rev)
        model = WhiteBoxModel(REPO, revision=rev, access_tier=AccessTier.WEIGHTS)

        train = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
        test = LayerWiseMoralProbe._collect_all_activations(model, dataset.test)
        n_layers = model.info.n_layers

        arrays: dict[str, np.ndarray] = {}
        for L in range(n_layers):
            tx, ty = train[L]
            ex, ey = test[L]
            arrays[f"train_X_{L}"] = tx.numpy().astype(np.float32)
            arrays[f"test_X_{L}"] = ex.numpy().astype(np.float32)
        arrays["train_y"] = train[0][1].numpy().astype(np.float32)
        arrays["test_y"] = test[0][1].numpy().astype(np.float32)
        arrays["n_layers"] = np.array([n_layers])
        arrays["step"] = np.array([step])
        np.savez(out, **arrays)
        log.info("[%d/%d] step %d: saved %s (%d layers)", i + 1, len(steps), step, out.name, n_layers)

        del model, train, test
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _free_repo_cache()

    log.info("Done. Cached %d checkpoints to %s", len(steps), out_dir)


if __name__ == "__main__":
    main()
